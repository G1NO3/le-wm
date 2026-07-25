import os

os.environ["MUJOCO_GL"] = "egl"

import time
from functools import partial
import hashlib
import json
import subprocess
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from stochastic_physics import (
    FetchPushHiddenFriction,
    FetchPushPairedFriction,
    OGBenchHiddenPhysics,
    RoboCasaHiddenPhysics,
)
from physical_probes import RidgeProbe
from residual_policy import FetchGuidedWorldModelPolicy, ResidualWorldModelPolicy
from experiment_data import sha256_file, sha256_json


class SeededEvaluationDataset:
    """Override simulator seeds without changing immutable source episodes."""

    def __init__(self, dataset, seeds):
        self.dataset = dataset
        self.seeds = np.asarray(seeds)
        self.column_names = list(dataset.column_names)
        if "seed" not in self.column_names:
            self.column_names.append("seed")

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def load_chunk(self, episodes_idx, start, end):
        chunks = self.dataset.load_chunk(episodes_idx, start, end)
        for chunk, seed in zip(chunks, self.seeds):
            length = len(next(iter(chunk.values())))
            chunk["seed"] = np.full((length,), seed, dtype=np.int64)
        return chunks

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def reachable_context_index(dataset, goal_offset):
    """Return internal episode, positional step, and flat-row candidates."""

    episodes = []
    starts = []
    rows = []
    for episode, (length, offset) in enumerate(
        zip(dataset.lengths, dataset.offsets, strict=True)
    ):
        count = int(length) - int(goal_offset)
        if count <= 0:
            continue
        episode_starts = np.arange(count, dtype=np.int64)
        episodes.append(np.full(count, episode, dtype=np.int64))
        starts.append(episode_starts)
        rows.append(int(offset) + episode_starts)
    if not episodes:
        raise ValueError(
            f"No episode is longer than goal_offset={int(goal_offset)}"
        )
    return tuple(np.concatenate(values) for values in (episodes, starts, rows))


def balanced_contact_context_indices(
    dataset,
    candidate_rows,
    valid_indices,
    *,
    stages,
    contexts,
    generator,
    key,
    selection="within_stage",
):
    """Choose near-equal numbers of repeated-contact contexts by stage."""

    values = np.asarray(dataset.get_col_data(key), dtype=np.int64).reshape(-1)
    previous = values.copy()
    for length, offset in zip(dataset.lengths, dataset.offsets, strict=True):
        start, end = int(offset), int(offset + length)
        previous[start] = 0
        previous[start + 1 : end] = values[start : end - 1]
    if selection == "onset":
        labels = np.where(values > previous, values, 0)
    elif selection == "within_stage":
        # Exclude the onset transition itself: these windows start after the
        # interaction has revealed information about persistent dynamics.
        labels = np.where(values == previous, values, 0)
    else:
        raise ValueError("contact_selection must be 'onset' or 'within_stage'")
    base, remainder = divmod(contexts, len(stages))
    selected = []
    for stage_index, stage in enumerate(stages):
        needed = base + int(stage_index < remainder)
        eligible = valid_indices[labels[candidate_rows[valid_indices]] == int(stage)]
        if len(eligible) < needed:
            raise ValueError(
                f"Contact stage {stage} has {len(eligible)} contexts, needs {needed}"
            )
        selected.extend(generator.choice(eligible, size=needed, replace=False))
    return np.sort(np.asarray(selected, dtype=np.int64))


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_load=cfg.dataset.get("keys_to_load"),
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def resolve_checkpoint_path(policy):
    requested = Path(str(policy))
    paths = [requested]
    if not requested.is_absolute():
        paths.append(Path(swm.data.utils.get_cache_dir()) / requested)
    for path in paths:
        if path.is_file():
            return path
        object_path = Path(f"{path}_object.ckpt")
        if object_path.is_file():
            return object_path
        if path.is_dir():
            candidates = sorted(
                path.glob("*_object.ckpt"), key=lambda value: value.stat().st_ctime
            )
            if candidates:
                return candidates[-1]
    return None


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def paired_fetch_success_metrics(successes, seeds, origin):
    """Summarize balanced low/high outcomes independent of completion order."""

    successes = np.asarray(successes, dtype=bool).reshape(-1)
    seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
    offsets = seeds - int(origin)
    if len(successes) != len(seeds) or np.any(offsets < 0):
        raise ValueError("Invalid paired FetchPush outcomes or seed origin")
    modes = offsets % 2
    pair_ids = offsets // 2
    both = []
    for pair_id in np.unique(pair_ids):
        selected = pair_ids == pair_id
        pair_modes = modes[selected]
        if selected.sum() != 2 or set(pair_modes.tolist()) != {0, 1}:
            raise ValueError(f"Incomplete paired FetchPush scene {pair_id}")
        both.append(bool(successes[selected].all()))
    result = {}
    for mode, label in ((0, "low"), (1, "high")):
        selected = successes[modes == mode]
        result[f"success_rate_{label}"] = float(selected.mean() * 100)
    result["paired_both_success_rate"] = float(np.mean(both) * 100)
    return modes, result


def evaluate_paired_full_task(world, *, episodes, seed):
    """Evaluate exact consecutive low/high seed blocks without async censoring."""

    batch_size = int(world.num_envs)
    if batch_size % 2 or episodes % batch_size:
        raise ValueError(
            "Paired full-task evaluation requires an even num_envs and "
            "num_eval divisible by num_envs"
        )
    batches = []
    for offset in range(0, int(episodes), batch_size):
        action_buffers = getattr(world.policy, "_action_buffer", None)
        if action_buffers is not None:
            for buffer in action_buffers:
                buffer.clear()
            world.policy._next_init = None
        batches.append(
            world.evaluate(
                episodes=batch_size,
                seed=int(seed) + offset,
                reset_mode="wait",
            )
        )
    metrics = {}
    for key in batches[0]:
        values = [batch[key] for batch in batches]
        if isinstance(values[0], (list, np.ndarray)):
            metrics[key] = np.concatenate([np.asarray(value) for value in values])
        else:
            metrics[key] = float(np.mean(values))
    successes = np.asarray(metrics["episode_successes"], dtype=bool)
    metrics["success_rate"] = float(100 * successes.mean())
    return metrics

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world_kwargs = OmegaConf.to_container(cfg.world, resolve=True)
    hidden_cfg = cfg.eval.get("hidden_physics")
    if hidden_cfg and hidden_cfg.get("enabled", False):
        wrapper_kwargs = OmegaConf.to_container(hidden_cfg, resolve=True)
        wrapper_kwargs.pop("enabled", None)
        paired = bool(wrapper_kwargs.pop("paired", False))
        env_name = cfg.world.env_name.lower()
        if "fetchpush" in env_name:
            wrapper_cls = (
                FetchPushPairedFriction if paired else FetchPushHiddenFriction
            )
        elif "robocasa" in env_name:
            wrapper_cls = RoboCasaHiddenPhysics
        else:
            if paired:
                raise ValueError("paired hidden physics is only supported for FetchPush")
            wrapper_cls = OGBenchHiddenPhysics
        world_kwargs["pre_wrappers"] = [partial(wrapper_cls, **wrapper_kwargs)]
    world = swm.World(**world_kwargs, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    ep_indices = np.arange(len(dataset.lengths), dtype=np.int64)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        planning = cfg.eval.get("stochastic_planning")
        if planning and planning.get("enabled", False):
            if not hasattr(model, "planning"):
                model.planning = {}
            model.planning.update(OmegaConf.to_container(planning, resolve=True))
        if planning:
            probe_path = planning.get("physical_probe_checkpoint")
            if probe_path:
                probe_payload = torch.load(probe_path, map_location="cpu", weights_only=True)
                probe_name = planning.get("physical_probe_name", "cube_positions")
                model.physical_probe = RidgeProbe.from_state_dict(
                    probe_payload["probes"][probe_name]
                )
            elif float(planning.get("collateral_penalty_weight", 0)) > 0:
                raise ValueError(
                    "A physical_probe_checkpoint is required for collateral MPC cost"
                )
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        if cfg.eval.get("guided_expert", False):
            if getattr(model, "residual_memory", None) is not None:
                raise ValueError("Fetch expert guidance is not yet combined with residual memory")
            policy_class = FetchGuidedWorldModelPolicy
        else:
            policy_class = (
                ResidualWorldModelPolicy
                if getattr(model, "residual_memory", None) is not None
                else swm.policy.WorldModelPolicy
            )
        policy = policy_class(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
            **({"expert_seed": int(cfg.seed)} if policy_class is FetchGuidedWorldModelPolicy else {}),
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )
    video_path = results_path if cfg.eval.get("video", False) else None

    world.set_policy(policy)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    protocol = cfg.eval.get("protocol", "reachable_window")
    evaluation_contact_stages = None
    evaluation_friction_modes = None
    if protocol == "full_task":
        eval_episodes = np.arange(cfg.seed, cfg.seed + cfg.eval.num_eval)
        eval_start_idx = np.zeros(cfg.eval.num_eval, dtype=int)
        if hidden_cfg and hidden_cfg.get("paired", False):
            if video_path is not None:
                raise ValueError("Paired batched full-task evaluation does not support video")
            metrics = evaluate_paired_full_task(
                world, episodes=cfg.eval.num_eval, seed=cfg.seed
            )
            successes = np.asarray(metrics["episode_successes"], dtype=bool)
            seeds = np.asarray(metrics["seeds"], dtype=np.int64)
            eval_episodes = seeds.copy()
            eval_start_idx = np.zeros_like(seeds)
            origin = int(hidden_cfg.get("collection_seed_origin", cfg.seed))
            evaluation_friction_modes, paired_metrics = paired_fetch_success_metrics(
                successes, seeds, origin
            )
            metrics.update(paired_metrics)
        else:
            metrics = world.evaluate(
                episodes=cfg.eval.num_eval,
                seed=cfg.seed,
                video=video_path,
                reset_mode="auto",
            )
    elif protocol == "reachable_window":
        candidate_episodes, candidate_starts, candidate_rows = reachable_context_index(
            dataset, cfg.eval.goal_offset_steps
        )
        valid_indices = np.arange(len(candidate_episodes), dtype=np.int64)
        print(len(valid_indices), "valid starting points found for evaluation.")
        stage = cfg.eval.get("stage")
        if stage and stage != "all":
            stage_key = cfg.eval.get("stage_key", "subtask_stage")
            if stage_key not in dataset.column_names:
                raise KeyError(f"Stage annotation {stage_key!r} is not in the dataset")
            labels = dataset.get_col_data(stage_key)
            labels = np.array(
                [x.decode() if isinstance(x, bytes) else str(x) for x in labels]
            )
            patterns = {
                "approach_grasp": ("approach", "grasp", "pick"),
                "lift": ("lift",),
                "transport": ("transport", "navigate", "move"),
                "cabinet_placement": ("place", "cabinet", "insert"),
            }[stage]
            stage_mask = np.array(
                [any(pattern in label.lower() for pattern in patterns) for label in labels]
            )
            valid_indices = valid_indices[stage_mask[candidate_rows[valid_indices]]]
        contexts = int(cfg.eval.get("num_contexts") or cfg.eval.num_eval)
        repeats = int(cfg.eval.get("stochastic_repeats", 1))
        if contexts * repeats != cfg.eval.num_eval:
            raise ValueError("num_contexts * stochastic_repeats must equal num_eval")
        g = np.random.default_rng(cfg.seed)
        contact_stages = cfg.eval.get("contact_stages")
        if contact_stages:
            selected = balanced_contact_context_indices(
                dataset,
                candidate_rows,
                valid_indices,
                stages=list(contact_stages),
                contexts=contexts,
                generator=g,
                key=cfg.eval.get("contact_stage_key", "privileged/contact_index"),
                selection=cfg.eval.get("contact_selection", "within_stage"),
            )
        else:
            selected = np.sort(g.choice(valid_indices, size=contexts, replace=False))
        selected = np.repeat(selected, repeats)
        if contact_stages:
            contact_values = np.asarray(
                dataset.get_col_data(
                    cfg.eval.get("contact_stage_key", "privileged/contact_index")
                ),
                dtype=np.int64,
            ).reshape(-1)
            evaluation_contact_stages = contact_values[candidate_rows[selected]]
        eval_episodes = candidate_episodes[selected]
        eval_start_idx = candidate_starts[selected]
        evaluation_seeds = np.tile(
            np.arange(cfg.seed, cfg.seed + repeats, dtype=np.int64), contexts
        )
        if len(eval_episodes) % world.num_envs:
            raise ValueError("num_eval must be divisible by world.num_envs/eval.batch_size")
        batch_results = []
        risk_batches = {
            name: []
            for name in (
                "metrics/target_object_drops",
                "metrics/collateral_displacement",
                "metrics/grasp_retries",
                "metrics/final_goal_distance",
            )
        }
        for batch_start in range(0, len(eval_episodes), world.num_envs):
            batch_end = batch_start + world.num_envs
            seeded_batch = SeededEvaluationDataset(
                dataset, evaluation_seeds[batch_start:batch_end]
            )
            batch_results.append(
                world.evaluate(
                    dataset=seeded_batch,
                    start_steps=eval_start_idx[batch_start:batch_end].tolist(),
                    goal_offset=cfg.eval.goal_offset_steps,
                    eval_budget=cfg.eval.eval_budget,
                    episodes_idx=eval_episodes[batch_start:batch_end].tolist(),
                    callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
                    video=(
                        video_path / f"batch_{batch_start // world.num_envs:03d}"
                        if video_path is not None
                        else None
                    ),
                )
            )
            for name in risk_batches:
                if name in world.infos:
                    risk_batches[name].append(np.asarray(world.infos[name]))
        successes = np.concatenate(
            [item["episode_successes"] for item in batch_results]
        )
        metrics = {
            "success_rate": float(successes.mean() * 100),
            "episode_successes": successes,
            "seeds": evaluation_seeds,
        }
        if evaluation_contact_stages is not None:
            metrics["contact_stage_success_rate"] = {
                str(int(stage)): float(
                    successes[evaluation_contact_stages == stage].mean() * 100
                )
                for stage in np.unique(evaluation_contact_stages)
            }
        for name, values in risk_batches.items():
            if values:
                joined = np.concatenate(values).reshape(-1)
                metric_name = name.split("/", 1)[1]
                metrics[metric_name] = float(joined.mean())
                if metric_name == "final_goal_distance":
                    worst_count = max(1, int(np.ceil(0.1 * len(joined))))
                    metrics["worst_decile_distance"] = float(
                        np.sort(joined)[-worst_count:].mean()
                    )
    else:
        raise ValueError(f"Unknown evaluation protocol: {protocol}")
    end_time = time.time()
    if protocol == "full_task":
        for name in (
            "metrics/target_object_drops",
            "metrics/collateral_displacement",
            "metrics/grasp_retries",
            "metrics/final_goal_distance",
        ):
            if name in world.infos:
                value = np.asarray(world.infos[name]).reshape(-1)
                metric_name = name.split("/", 1)[1]
                metrics[metric_name] = float(value.mean())
                if metric_name == "final_goal_distance":
                    worst_count = max(1, int(np.ceil(0.1 * len(value))))
                    metrics["worst_decile_distance"] = float(
                        np.sort(value)[-worst_count:].mean()
                    )
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")

    selection = {
        "protocol": protocol,
        "episodes": eval_episodes.tolist(),
        "start_steps": eval_start_idx.tolist(),
        "goal_offset": int(cfg.eval.goal_offset_steps) if protocol == "reachable_window" else None,
        "contact_stages": (
            evaluation_contact_stages.tolist()
            if evaluation_contact_stages is not None
            else None
        ),
        "friction_modes": (
            evaluation_friction_modes.tolist()
            if evaluation_friction_modes is not None
            else None
        ),
    }
    planning = cfg.eval.get("stochastic_planning", {})
    stochastic_planning_enabled = bool(planning.get("enabled", False))
    hidden_physics_metadata = cfg.eval.get("hidden_physics", {})
    if OmegaConf.is_config(hidden_physics_metadata):
        hidden_physics_metadata = OmegaConf.to_container(
            hidden_physics_metadata, resolve=True
        )
    checkpoint_path = resolve_checkpoint_path(cfg.policy)
    dataset_fingerprint = {
        "episode_lengths": np.asarray(dataset.lengths, dtype=np.int64).tolist(),
        "frameskip": int(dataset.frameskip),
        "num_steps": int(dataset.num_steps),
    }
    metadata = {
        "dataset": cfg.eval.dataset_name,
        "dataset_sha256": sha256_json(dataset_fingerprint),
        "evaluation_selection_sha256": hashlib.sha256(
            json.dumps(selection, sort_keys=True).encode()
        ).hexdigest(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_seed": int(cfg.seed),
        "environment_seeds": (
            evaluation_seeds.tolist()
            if protocol == "reachable_window"
            else eval_episodes.tolist()
        ),
        "task_stage": cfg.eval.get("stage"),
        "checkpoint": str(checkpoint_path or cfg.policy),
        "checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path is not None else None
        ),
        "stochastic_planning_enabled": stochastic_planning_enabled,
        "particles": int(planning.get("particles", 1)) if stochastic_planning_enabled else 1,
        "nfe": int(planning.get("flow_steps", 0)) if stochastic_planning_enabled else 0,
        "planning_objective": (
            str(planning.get("objective", "mean"))
            if stochastic_planning_enabled
            else "deterministic"
        ),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "latency_seconds": end_time - start_time,
        "latency_per_episode_seconds": (end_time - start_time) / len(eval_episodes),
        "wall_clock_budget_seconds": cfg.eval.get("wall_clock_budget_seconds"),
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "output_path": str(results_path.resolve()),
        "hidden_physics": hidden_physics_metadata,
        "residual_memory": {
            "enabled": getattr(model, "residual_memory", None) is not None
            if policy != "random"
            else False,
            "hidden_dim": (
                int(model.residual_memory.hidden_dim)
                if policy != "random" and getattr(model, "residual_memory", None) is not None
                else 0
            ),
            "online_update": policy != "random" and isinstance(policy, ResidualWorldModelPolicy),
        },
    }
    results_path.with_suffix(results_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    results_path.with_suffix(results_path.suffix + ".summary.json").write_text(
        json.dumps(
            {
                "metrics": jsonable(metrics),
                "metadata": metadata,
                "selection": selection,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    run()
