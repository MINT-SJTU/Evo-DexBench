# Dataset validation, quality, replay, and export

Canonical HDF5 is the source format. Validate schema after collection, run
quality checks over a completed session, replay when reproducibility matters,
and export to LeRobot only after those checks pass.

## Environment boundary

Run validation, quality checks, and replay in `evodex-sim`, which uses NumPy
1.26.4. LeRobot 0.4.4 requires the separate NumPy 2 environment defined by
`environment_lerobot.yml`. Do not install LeRobot into the simulator
environment.

The converter uses LeRobot 0.4.4 with NumPy 2. KineDex uses a separate, pinned
LeRobot 0.4.0 dataset stack with NumPy 1.26.4. They exchange LeRobot v3 dataset
files; their installation instructions are not interchangeable. The VR
environment can also run canonical validation, quality checks, and replay.

## Validate

```bash
dexbench data validate path/to/episode.h5 --verify-checksums

dexbench data validate datasets/canonical/pilot_v1/m01_v1/train \
  --manifest datasets/canonical/pilot_v1/m01_v1/train/collection_manifest.json \
  --verify-checksums \
  --required-camera context \
  --required-camera interaction \
  --required-tactile-role right
```

Validation checks the schema/version, required groups, shape and dtype,
contiguous steps, action/observation association, RGB/depth alignment, tactile
roles, and optional checksums. It answers whether a file satisfies the
canonical contract, not whether the motion is useful.

## Quality checks

```bash
dexbench data quality \
  datasets/canonical/pilot_v1/m01_v1/train \
  datasets/quality/pilot_v1/m01_v1/train \
  --control-hz 30 --image-hz 30 \
  --required-camera context \
  --required-camera interaction \
  --required-tactile-role right
```

The command writes `quality_report.json` without moving or modifying source
HDF5. Review invalid reasons, cadence, repeated-image runs, non-finite or abrupt
actions, and missing sensor roles. Inspect samples before treating a threshold
failure as unusable data.

## Replay

```bash
dexbench data replay \
  datasets/canonical/pilot_v1/m01_v1/train/m01_v1_00000.h5 \
  --sim-backend cpu --render-backend gpu --atol 1e-5
```

Replay reads task, layout, reset seed, control mode, and generalization metadata
from the episode. CLI overrides are diagnostic. Missing reset or generalization
metadata prevents a strong reproducibility claim. Contact nondeterminism,
changed assets/controllers, an incorrect simulation backend, or incomplete CPU
copies before recording can all cause divergence.

## Export to LeRobot

Create the converter environment once from the repository root:

```bash
conda env create -f environment_lerobot.yml
conda activate evodex-lerobot
```

Do not install the full project in this environment: the simulator pins
`numpy<2`, while the converter needs NumPy 2, h5py, and LeRobot. Invoke the
lightweight module from the repository root and convert only episodes marked
valid by the quality report:

```bash
PYTHONPATH="$PWD" python -m dex_benchmark.commands.data.convert_lerobot \
  datasets/canonical/pilot_v1/m01_v1/train \
  datasets/lerobot/pilot_v1-m01 \
  --repo-id local/pilot_v1-m01 \
  --fps 30 \
  --manifest datasets/canonical/pilot_v1/m01_v1/train/collection_manifest.json \
  --verify-checksums \
  --quality-report datasets/quality/pilot_v1/m01_v1/train/quality_report.json \
  --image-writer-processes 0 \
  --image-writer-threads 4
```

LeRobot is a derived representation, not the benchmark source of truth. Keep
canonical HDF5 and manifests, keep quality and export outputs in different
directories, and return to `evodex-sim` for replay or evaluation. Test one
episode before converting a large dataset.
