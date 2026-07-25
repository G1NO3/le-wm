"""Immutable episode split manifests and run provenance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def sha256_file(path, chunk_size=1024 * 1024) -> str:
    """Hash an immutable dataset/checkpoint artifact without loading it at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_episode_ids(count, *, seed, fractions):
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Episode split fractions must sum to one")
    episodes = np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_train = int(count * fractions[0])
    n_val = int(count * fractions[1])
    return {
        "train": episodes[:n_train].tolist(),
        "val": episodes[n_train : n_train + n_val].tolist(),
        "test": episodes[n_train + n_val :].tolist(),
    }


def build_collection_manifest(
    episode_lengths, dataset_sha256, *, seed, fractions=(0.8, 0.1, 0.1)
):
    """Build an immutable episode-only manifest before model dataset loading."""

    lengths = np.asarray(episode_lengths, dtype=np.int64)
    manifest = {
        "version": 2,
        "seed": int(seed),
        "fractions": list(fractions),
        "dataset_sha256": str(dataset_sha256),
        "episode_lengths_sha256": sha256_json(lengths.tolist()),
        "episodes": _split_episode_ids(len(lengths), seed=seed, fractions=fractions),
    }
    manifest["sha256"] = sha256_json(manifest)
    return manifest


def build_grouped_collection_manifest(
    episode_lengths,
    dataset_sha256,
    group_ids,
    *,
    seed,
    fractions=(0.8, 0.1, 0.1),
    group_key="group_id",
):
    """Split whole paired/counterfactual groups, never individual episodes."""

    lengths = np.asarray(episode_lengths, dtype=np.int64)
    group_ids = np.asarray(group_ids, dtype=np.int64)
    if len(lengths) != len(group_ids):
        raise ValueError("Each episode must have exactly one group ID")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Episode split fractions must sum to one")
    groups = np.unique(group_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_train = int(len(groups) * fractions[0])
    n_val = int(len(groups) * fractions[1])
    split_groups = {
        "train": groups[:n_train],
        "val": groups[n_train : n_train + n_val],
        "test": groups[n_train + n_val :],
    }
    episodes = {
        name: np.flatnonzero(np.isin(group_ids, values)).tolist()
        for name, values in split_groups.items()
    }
    manifest = {
        "version": 3,
        "seed": int(seed),
        "fractions": list(fractions),
        "dataset_sha256": str(dataset_sha256),
        "episode_lengths_sha256": sha256_json(lengths.tolist()),
        "group_key": str(group_key),
        "group_ids_sha256": sha256_json(group_ids.tolist()),
        "groups": {name: values.tolist() for name, values in split_groups.items()},
        "episodes": episodes,
    }
    manifest["sha256"] = sha256_json(manifest)
    return manifest


def build_episode_manifest(dataset, *, seed: int, fractions=(0.8, 0.1, 0.1)):
    """Split episode IDs and map the dataset's clip indices to each split."""

    if not hasattr(dataset, "clip_indices"):
        raise TypeError("Episode-disjoint splitting requires dataset.clip_indices")

    groups = _split_episode_ids(len(dataset.lengths), seed=seed, fractions=fractions)
    dataset_fingerprint = {
        "episode_lengths": np.asarray(dataset.lengths, dtype=np.int64).tolist(),
        "frameskip": int(dataset.frameskip),
        "num_steps": int(dataset.num_steps),
    }
    dataset_sha256 = sha256_json(dataset_fingerprint)
    hash_source = None
    h5_path = getattr(dataset, "h5_path", None)
    if h5_path is not None:
        validation_path = Path(h5_path).with_suffix(".validation.json")
        if validation_path.exists():
            dataset_sha256 = json.loads(validation_path.read_text())["dataset_sha256"]
            hash_source = "validated_hdf5"
    manifest = {
        "version": 1,
        "seed": int(seed),
        "fractions": list(fractions),
        "dataset_sha256": dataset_sha256,
        "episodes": groups,
    }
    if hash_source is not None:
        manifest["dataset_hash_source"] = hash_source
    manifest["sha256"] = sha256_json(manifest)
    return manifest


def save_or_verify_manifest(path, manifest):
    """Create a manifest once; reject attempts to silently change it."""

    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != manifest:
            raise RuntimeError(f"Immutable split manifest differs from requested split: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def subsets_from_manifest(dataset, manifest):
    episode_sets = {key: set(value) for key, value in manifest["episodes"].items()}
    indices = {key: [] for key in episode_sets}
    for clip_index, (episode, _) in enumerate(dataset.clip_indices):
        memberships = [key for key, episodes in episode_sets.items() if episode in episodes]
        if len(memberships) != 1:
            raise RuntimeError(f"Episode {episode} belongs to {len(memberships)} splits")
        indices[memberships[0]].append(clip_index)
    return {key: torch.utils.data.Subset(dataset, value) for key, value in indices.items()}


def episode_disjoint_split(dataset, path, *, seed, fractions=(0.8, 0.1, 0.1)):
    path = Path(path)
    if path.exists():
        # A split is a property of the dataset, not of the model seed. Reuse
        # and validate an immutable precomputed manifest rather than trying to
        # regenerate it with each training run's seed. This also accepts the
        # version-2 collection manifests created before model dataset loading.
        manifest = json.loads(path.read_text())
        recorded_sha = manifest.get("sha256")
        unsigned = {key: value for key, value in manifest.items() if key != "sha256"}
        if recorded_sha != sha256_json(unsigned):
            raise RuntimeError(f"Immutable split manifest hash is invalid: {path}")

        expected = build_episode_manifest(
            dataset,
            seed=int(manifest["seed"]),
            fractions=tuple(manifest["fractions"]),
        )
        if manifest.get("dataset_sha256") != expected["dataset_sha256"]:
            raise RuntimeError(f"Split manifest references a different dataset: {path}")
        if not np.allclose(manifest.get("fractions", ()), fractions):
            raise RuntimeError(f"Immutable split manifest fractions differ: {path}")
        if manifest.get("version") in (2, 3):
            expected_lengths = sha256_json(
                np.asarray(dataset.lengths, dtype=np.int64).tolist()
            )
            if manifest.get("episode_lengths_sha256") != expected_lengths:
                raise RuntimeError(
                    f"Split manifest episode lengths differ from the dataset: {path}"
                )
        groups = manifest.get("episodes", {})
        if set(groups) != {"train", "val", "test"}:
            raise RuntimeError(f"Split manifest has invalid groups: {path}")
        assigned = [episode for values in groups.values() for episode in values]
        if sorted(assigned) != list(range(len(dataset.lengths))):
            raise RuntimeError(
                f"Split manifest must assign each dataset episode exactly once: {path}"
            )
    else:
        manifest = build_episode_manifest(dataset, seed=seed, fractions=fractions)
        manifest = save_or_verify_manifest(path, manifest)
    return subsets_from_manifest(dataset, manifest), manifest
