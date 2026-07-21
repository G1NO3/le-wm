---
license: mit
task_categories:
  - robotics
tags:
  - world-models
  - stochastic-dynamics
  - ogbench
  - robocasa
---

# LeWM stochastic manipulation artifacts

This private dataset repository stores the large, immutable artifacts for the
Flow-Matched Residual Kernel study. Source code, schemas, and compact run
records remain in the `le-wm` Git repository.

## Layout

Artifacts use paths of the form:

```text
ogbench/<task>/<physics-profile>/<episode-count>-seed<seed>/
  trajectories.h5
  trajectories.metadata.json
  trajectories.validation.json
  h5_forks.pt
  h5_forks.json
  split_manifest.json
```

The HDF5 data contains model-facing observations and commanded actions.
Hidden physics modes and transient-slip events are privileged evaluation
fields and must never be added to model input keys. Fork evidence contains
compact physical states only; it is cryptographically tied to its source HDF5
file through `dataset_sha256`.

## Integrity and publication policy

- Collection files are immutable: reruns use a new seed or path.
- Every HDF5 upload must include its validation manifest and matching SHA-256.
- Dataset/split hashes and source commit are recorded in every training run.
- Smoke artifacts live under `smoke/` and are not scientific results.
- No post-pilot array is published until its preceding preregistered gate passes.

Use `scripts/data/publish_hf_dataset.py` for checked uploads and `hf download`
with a pinned Hub revision for cluster consumption.
