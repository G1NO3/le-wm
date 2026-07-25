import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from residual_flow import ResidualFlow
from residual_memory import ResidualMemory
from residual_kernels import build_residual_kernel
from experiment_data import episode_disjoint_split, sha256_file
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def residual_kernel_loss(
    model,
    pred_emb,
    tgt_emb,
    ctx_emb,
    ctx_act,
    cfg,
    *,
    observation_history=None,
    update_statistics,
):
    """Train one uncertainty head on the deployment-time final transition."""

    residual_cfg = cfg.loss.residual_flow
    # Rollouts consume only the final history-conditioned prediction. Training
    # every predictor token would optimize a different conditional kernel.
    residual = tgt_emb - pred_emb
    final_residual = residual[:, -1:]
    if update_statistics:
        model.update_residual_statistics(
            final_residual,
            decay=residual_cfg.ema_decay,
            eps=residual_cfg.scale_eps,
        )

    normalized_residual = model.normalize_residual(
        residual, eps=residual_cfg.scale_eps
    )
    target_residual = normalized_residual[:, -1:]
    if residual_cfg.detach_residual_target:
        target_residual = target_residual.detach()

    eps = torch.randn_like(target_residual)
    tau = torch.rand(
        *target_residual.shape[:-1],
        1,
        device=target_residual.device,
        dtype=target_residual.dtype,
    )
    z_tau = (1.0 - tau) * eps + tau * target_residual
    target_velocity = target_residual - eps

    base_condition = model.residual_condition(ctx_emb, ctx_act, pred_emb)
    if residual_cfg.detach_condition:
        base_condition = base_condition.detach()

    memory = None
    if getattr(model, "residual_memory", None) is not None:
        memory = model.init_residual_memory(
            (pred_emb.size(0),), device=pred_emb.device, dtype=pred_emb.dtype
        )
        # Teacher-force only residuals that precede the deployment-time target.
        # The final target can never enter its own conditioning state.
        memory_cfg = residual_cfg.get("memory", {})
        max_updates = int(memory_cfg.get("max_history_updates", 5))
        sampled_probability = float(
            memory_cfg.get("sampled_history_probability", 0.0)
        )
        first_index = max(0, pred_emb.size(1) - 1 - max_updates)
        for index in range(first_index, max(pred_emb.size(1) - 1, 0)):
            memory_residual = normalized_residual[:, index].detach()
            if sampled_probability > 0:
                with torch.no_grad():
                    sampled = model.sample_residual(
                        pred_emb[:, index],
                        base_condition[:, index],
                        steps=int(memory_cfg.get("sampling_steps", 4)),
                        memory=memory,
                    )
                    sampled = model.normalize_residual(sampled).detach()
                use_sample = torch.rand(
                    pred_emb.size(0), 1, device=pred_emb.device
                ) < sampled_probability
                memory_residual = torch.where(
                    use_sample, sampled, memory_residual
                )
            memory = model.update_residual_memory(
                memory,
                base_condition[:, index].detach(),
                memory_residual,
                (
                    None
                    if observation_history is None
                    else (
                        observation_history[:, index + 1]
                        - observation_history[:, index]
                    ).detach()
                ),
            )

    condition = model.condition_residual(base_condition[:, -1], memory).unsqueeze(1)

    if model.residual_kernel is not None:
        return model.residual_kernel.loss(target_residual, condition)

    pred_velocity = model.residual_flow(tau, z_tau, condition)
    return (pred_velocity - target_velocity).pow(2).mean()


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    residual_cfg = cfg.loss.get("residual_flow")
    observation_key = (
        residual_cfg.get("memory", {}).get("observation_key")
        if residual_cfg is not None
        else None
    )
    if observation_key and observation_key in batch:
        # Constant state coordinates have zero empirical variance and therefore
        # become NaN under the dataset's standard-score transform.
        batch[observation_key] = torch.nan_to_num(batch[observation_key], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    if (
        residual_cfg is not None
        and residual_cfg.get("enabled", False)
        and (
            self.model.residual_flow is not None
            or self.model.residual_kernel is not None
        )
    ):
        output["residual_kernel_loss"] = residual_kernel_loss(
            self.model,
            pred_emb,
            tgt_emb,
            ctx_emb,
            ctx_act,
            cfg,
            observation_history=(
                batch.get(observation_key) if observation_key else None
            ),
            update_statistics=stage == "fit" and self.model.training,
        )
        output["loss"] = (
            output["loss"]
            + residual_cfg.weight * output["residual_kernel_loss"]
        )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    split_path = cfg.data.get("split_manifest") or run_dir / "split_manifest.json"
    split_fractions = tuple(cfg.data.get("split_fractions", [0.8, 0.1, 0.1]))
    subsets, split_manifest = episode_disjoint_split(
        dataset,
        split_path,
        seed=cfg.seed,
        fractions=split_fractions,
    )
    train_set, val_set = subsets["train"], subsets["val"]
    rnd_gen = torch.Generator().manual_seed(cfg.seed)

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    residual_flow = None
    residual_kernel = None
    residual_memory = None
    residual_mean = None
    residual_scale = None
    residual_cfg = cfg.loss.get("residual_flow")
    if residual_cfg is not None and residual_cfg.get("enabled", False):
        kernel_type = residual_cfg.get("kernel_type", "flow")
        memory_cfg = residual_cfg.get("memory", {})
        memory_enabled = bool(memory_cfg.get("enabled", False))
        condition_dim = 3 * embed_dim
        if memory_enabled:
            observation_key = memory_cfg.get("observation_key")
            observation_dim = (
                int(getattr(cfg.wm, f"{observation_key}_dim"))
                if observation_key
                else 0
            )
            residual_memory = ResidualMemory(
                condition_dim=condition_dim,
                residual_dim=embed_dim,
                hidden_dim=int(memory_cfg.get("hidden_dim", 128)),
                observation_dim=observation_dim,
            )
            condition_dim += residual_memory.hidden_dim
        if kernel_type == "flow":
            residual_flow = ResidualFlow(
                residual_dim=embed_dim,
                condition_dim=condition_dim,
                **OmegaConf.to_container(residual_cfg.kwargs, resolve=True),
            )
        else:
            residual_kernel = build_residual_kernel(
                kernel_type,
                residual_dim=embed_dim,
                condition_dim=condition_dim,
                **OmegaConf.to_container(
                    residual_cfg.get("kernel_kwargs", {}), resolve=True
                ),
            )
        residual_mean = torch.zeros(embed_dim)
        residual_scale = torch.ones(embed_dim)

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        residual_flow=residual_flow,
        residual_kernel=residual_kernel,
        residual_memory=residual_memory,
        residual_mean=residual_mean,
        residual_scale=residual_scale,
        residual_conditioning=(
            residual_cfg.get("conditioning", "conditional") if residual_cfg else "conditional"
        ),
    )

    nominal_checkpoint = residual_cfg.get("nominal_checkpoint") if residual_cfg else None
    if residual_cfg and residual_cfg.get("freeze_nominal", False) and not nominal_checkpoint:
        raise ValueError("freeze_nominal=true requires loss.residual_flow.nominal_checkpoint")
    if nominal_checkpoint:
        nominal = torch.load(nominal_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(nominal, JEPA):
            candidates = [module for module in nominal.modules() if isinstance(module, JEPA)]
            if len(candidates) != 1:
                raise RuntimeError("Nominal checkpoint must contain exactly one JEPA model")
            nominal = candidates[0]
        incompatible = world_model.load_state_dict(nominal.state_dict(), strict=False)
        residual_names = (
            "residual_flow.",
            "residual_kernel.",
            "residual_memory.",
            "residual_mean",
            "residual_scale",
        )
        bad_missing = [x for x in incompatible.missing_keys if not x.startswith(residual_names)]
        bad_unexpected = [x for x in incompatible.unexpected_keys if not x.startswith(residual_names)]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"Nominal checkpoint mismatch; missing={bad_missing}, unexpected={bad_unexpected}"
            )

    if residual_cfg and residual_cfg.get("freeze_nominal", False):
        world_model.freeze_nominal()

    residual_scale_path = residual_cfg.get("residual_scale_path") if residual_cfg else None
    scale_payload = None
    if residual_scale_path:
        scale_payload = torch.load(residual_scale_path, map_location="cpu", weights_only=True)
        scale_metadata = scale_payload if isinstance(scale_payload, dict) else {}
        expected_checkpoint_hash = scale_metadata.get("checkpoint_sha256")
        if expected_checkpoint_hash and nominal_checkpoint:
            actual_checkpoint_hash = sha256_file(nominal_checkpoint)
            if actual_checkpoint_hash != expected_checkpoint_hash:
                raise ValueError(
                    "Residual statistics were computed from a different nominal checkpoint"
                )
        expected_split_hash = scale_metadata.get("split_sha256")
        if expected_split_hash and expected_split_hash != split_manifest["sha256"]:
            raise ValueError("Residual statistics were computed from a different split manifest")
        expected_dataset_hash = scale_metadata.get("dataset_sha256")
        if expected_dataset_hash and expected_dataset_hash != split_manifest["dataset_sha256"]:
            raise ValueError("Residual statistics were computed from a different dataset")
        scale = scale_metadata.get("residual_scale", scale_payload)
        mean = scale_metadata.get("residual_mean", torch.zeros_like(scale))
        if tuple(scale.shape) != tuple(world_model.residual_scale.shape):
            raise ValueError(
                f"Residual scale shape {tuple(scale.shape)} does not match "
                f"{tuple(world_model.residual_scale.shape)}"
            )
        if tuple(mean.shape) != tuple(world_model.residual_mean.shape):
            raise ValueError(
                f"Residual mean shape {tuple(mean.shape)} does not match "
                f"{tuple(world_model.residual_mean.shape)}"
            )
        world_model.residual_mean.copy_(mean)
        world_model.residual_scale.copy_(scale)
        world_model.residual_scale_initialized.fill_(True)
        world_model.freeze_residual_scale()

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    logger = False
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    with open(run_dir / "run_metadata.json", "w") as f:
        import json
        import platform
        import subprocess

        hidden_physics = cfg.data.get("hidden_physics", {})
        if OmegaConf.is_config(hidden_physics):
            hidden_physics = OmegaConf.to_container(hidden_physics, resolve=True)
        metadata = {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "dataset": cfg.data.dataset.name,
            "dataset_sha256": split_manifest["dataset_sha256"],
            "split_sha256": split_manifest["sha256"],
            "model_seed": int(cfg.seed),
            "checkpoint": str(nominal_checkpoint) if nominal_checkpoint else None,
            "checkpoint_sha256": (
                sha256_file(nominal_checkpoint) if nominal_checkpoint else None
            ),
            "kernel": (
                residual_cfg.get("kernel_type", "none")
                if residual_cfg and residual_cfg.get("enabled", False)
                else "none"
            ),
            "residual_memory": (
                OmegaConf.to_container(
                    residual_cfg.get("memory", {}), resolve=True
                )
                if residual_cfg
                else {"enabled": False}
            ),
            "residual_statistics": (
                {
                    "path": str(residual_scale_path),
                    "version": scale_payload.get("version"),
                    "mean_sha256": scale_payload.get("mean_sha256"),
                    "scale_sha256": scale_payload.get("scale_sha256"),
                    "checkpoint_sha256": scale_payload.get("checkpoint_sha256"),
                    "split_sha256": scale_payload.get("split_sha256"),
                    "dataset_sha256": scale_payload.get("dataset_sha256"),
                }
                if isinstance(scale_payload, dict)
                else None
            ),
            "host": platform.node(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "hidden_physics": hidden_physics,
            "output_path": str(run_dir),
        }
        json.dump(metadata, f, indent=2, sort_keys=True)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    weights_checkpoint = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=cfg.seed,
        # stable-pretraining>=0.1.8 treats any supplied path as an explicit
        # resume request and rejects nonexistent files.
        ckpt_path=weights_checkpoint if weights_checkpoint.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
