# Quest 3 data collection

`dexbench data collect` manages multi-episode Quest 3 collection. It owns the
session and attempt lifecycle, layout/generalization assignment, canonical HDF5
recording, manifests, retries, and episode validation. The Quest runner still
owns hand retargeting, action construction, and safety filtering.

## Before collecting

1. Complete the [Quest 3 teleoperation checklist](quest3-teleoperation.md).
2. Review the current task in the [task catalog](../task-catalog.md).
3. Export `QUEST3_LAN_IP`, `QUEST3_CERT`, and `QUEST3_KEY`.
4. Check disk space, output permissions, cameras, and tactile observations.
5. Run the collector with `--dry-run` before a long session.

```bash
eval "$(tools/quest3/create_lan_certificate.sh --export)"

dexbench data collect \
  --task-id m01_v1 \
  --hand-side right \
  --dataset-root datasets/canonical \
  --dataset-id pilot_v1 \
  --split train \
  --episodes 20 \
  --max-attempts 60 \
  --seed 42 --seed-mode increment \
  --generalization-profile collection \
  --generalization-axis balanced \
  --operator-id operator_01 \
  --session-id pilot-session-01 \
  --host 0.0.0.0 --port 8013 \
  --advertise-host "$QUEST3_LAN_IP" \
  --cert "$QUEST3_CERT" --key "$QUEST3_KEY" \
  --control-hz 30 --image-hz 30 --sim-freq 120 \
  --render-mode human --quest-view --validate-after-episode \
  --dry-run
```

Remove `--dry-run` only after reviewing the printed configuration.

## Dataset layout

The collector isolates every dataset, task, and split:

```text
datasets/canonical/
└── pilot_v1/
    └── m01_v1/
        └── train/
            ├── collection_manifest.json
            ├── m01_v1_00000.h5
            └── ...
```

Dataset IDs, episode prefixes, and explicit episode IDs must be safe single
path components. A dataset may contain multiple tasks, but every task/split
pair has an independent manifest.

## Episodes, attempts, and operator keys

A logical episode may require several attempts. The manifest records the seed,
layout, sampled generalization, exit reason, output path, and validation result
for every attempt. Successful episodes are finalized; operator resets, physics
safety stops, and time limits discard the current attempt and retry it.
`operator_quit` and interruption stop the session while keeping the manifest
resumable. They do not promote an incomplete attempt into a valid episode.
`--max-attempts` limits all attempts in the current process, not the attempts
per episode.

| Key | Action |
| --- | --- |
| `R` | discard the current attempt and retry the same logical episode |
| `E` | skip the inter-episode wait and continue |
| `Q` | close the recorder, update the manifest, and stop safely |

Use `Q` instead of killing the process whenever possible.

On a host without a desktop viewer, keep the Quest operator view and disable
the local window:

```bash
dexbench data collect \
  --task-id m01_v1 --hand-side right \
  --dataset-root datasets/canonical --dataset-id pilot_v1 \
  --episodes 20 --advertise-host "$QUEST3_LAN_IP" \
  --cert "$QUEST3_CERT" --key "$QUEST3_KEY" \
  --control-hz 30 --image-hz 30 --sim-freq 120 \
  --render-mode none --quest-view
```

With `--quest-view --render-mode none`, the collector automatically enables
Quest-only viewing. Unlike `dexbench teleop quest3`, this command does not need
a separate `--quest-view-only` flag. A headless host still needs a working
GPU renderer for the operator view and recorded camera images.

By default, recorder chunks use a temporary directory and are removed after
each attempt. Add `--spool` only when failed or interrupted chunks must be kept
for diagnostics. Persistent chunks are written below
`<dataset-root>/<dataset-id>/.spool/`; they are not validated training data.

## Seeds, layouts, and collection generalization

`--seed-mode increment` creates a deterministic sequence; `fixed` reuses one
seed; and `random` creates a deterministic pseudorandom sequence from the base
seed. Each attempt deterministically selects one of the task's three supported
Studio or Kitchen layouts.

The `collection` profile supports the reduced-range `spatial` and `physical`
axes. `balanced` alternates their assignment across logical episodes in a
seeded order:

```bash
--generalization-profile collection --generalization-axis balanced
```

Collection generalization is intentionally narrower than the standard
evaluation protocol. Episode reset seeds and generalization seeds are stored as
separate metadata fields.

## Sensors and timing

The collector derives required camera and tactile roles from the task
embodiment. Additional requirements can be enforced with repeated
`--required-camera` and `--required-tactile-role` options. Do not request a role
that does not exist in the selected embodiment.

The standard collection command uses 120 Hz simulation, 30 Hz control, and
30 Hz images. The writer aligns held images through observation IDs and
timestamps. `--canonical-queue-capacity` and `--canonical-chunk-length` affect
I/O buffering only; they do not change episode semantics.

After collection, use the quality command below to verify measured timing.
Configured frequencies describe the requested schedule; trajectories that run
substantially slower in wall-clock time require a lower-overhead rendering
setup or a faster host before they are accepted as benchmark data.

## Resume and validation

Resume with the same semantic configuration and add `--resume`:

Reuse the original command, including any custom timing, retargeting, sensor,
or generalization settings, and append `--resume`. The example below matches
the initial collection example; it is not a substitute for a customized
session command.

```bash
dexbench data collect \
  --task-id m01_v1 \
  --dataset-root datasets/canonical \
  --dataset-id pilot_v1 --split train --episodes 20 \
  --hand-side right --max-attempts 60 \
  --seed 42 --seed-mode increment \
  --generalization-profile collection \
  --generalization-axis balanced \
  --operator-id operator_01 --session-id pilot-session-01 \
  --host 0.0.0.0 --port 8013 --advertise-host "$QUEST3_LAN_IP" \
  --cert "$QUEST3_CERT" --key "$QUEST3_KEY" \
  --control-hz 30 --image-hz 30 --sim-freq 120 \
  --render-mode human --quest-view --validate-after-episode \
  --resume
```

Resume rejects changes that alter dataset semantics and revalidates completed
episodes. `--overwrite` intentionally replaces a matching session; do not use
it as an automatic fallback for a failed resume.

After collection, validate schema and checksums, then run quality checks:

```bash
dexbench data validate datasets/canonical/pilot_v1/m01_v1/train \
  --verify-checksums

dexbench data quality \
  datasets/canonical/pilot_v1/m01_v1/train \
  datasets/quality/pilot_v1/m01_v1/train \
  --control-hz 30 --image-hz 30
```

Schema validity does not imply trajectory quality. See the
[dataset tools guide](dataset-tools.md) for quality, replay, and export.
