# KineDex integration

This integration directly uses the official
[KineDex](https://github.com/DinoMini00/KineDex_code) implementation at commit
`9246f77d7b969b181b0f1b600601b9a9c8083011`. The official ResNet-18 multi-view
encoder, shared per-finger 1D CNN, conditional 1D U-Net, DDIM scheduler, EMA,
and training workspace remain in use. Evo-DexBench adds only its LeRobot v3
data view, Shadow Hand sensor/action contract, bounded force controller, and
the common WebSocket policy boundary.

## One-command setup

Clone the pinned official source, apply the Evo-DexBench adapter patches, create
the isolated conda environment, install the validated CUDA wheels and policy
dependencies, and validate the result with:

```bash
tools/setup_kinedex.sh --name evodex-kinedex
export KINEDEX_ROOT="$PWD/.third_party/KineDex-official"
```

The script is idempotent and refuses to replace local changes in an existing
official checkout. `tools/setup_environment.sh --profile policy-kinedex` is a
short alias for the same setup. Use `--torch-index-url` or `--pip-index-url`
when a trusted package mirror is required.

## Runtime and resources

The tracked patch parameterizes the official tactile encoder's input channels
and finger count. It is required because the original XHand has five
`120x3` tactile arrays while Evo-DexBench has one scalar normal force per
taxel and supports one or two hands. No diffusion or visual architecture is
reimplemented.

The model environment uses Python 3.10, PyTorch 2.5.1+cu124,
torchvision 0.20.1+cu124, NumPy 1.26.4, Diffusers 0.33.1, and LeRobot 0.4.0.
Use the setup helper rather than installing the full LeRobot dependency stack:
the model only needs its dataset API, not robot drivers or visualization tools.
The LeRobot export environment is separate and uses LeRobot 0.4.4 with NumPy 2.

Official GitHub and package indexes are the defaults. Select a trusted
`--upstream-url`, `--pip-index-url`, or `--torch-index-url` explicitly if needed.
The setup helper isolates user-site packages; reactivate the model environment
after rerunning setup.

Training uses PyAV CPU video decoding. Keep data on fast local storage and
leave RAM available for checkpoint saving. On memory-constrained hosts, append
these overrides after `--` in the training command:

```bash
dataloader.num_workers=2 val_dataloader.num_workers=1 \
cpu_data_pipeline.prefetch_factor=1 cpu_data_pipeline.video_reader_cache_size=2
```

These settings reduce memory use but do not guarantee that training or
four-worker evaluation fits alongside other applications. Increasing GPU
memory does not remove host RAM or CPU decoding requirements.

## Evo-DexBench contract

Training reads the original LeRobot v3 dataset at 30 Hz. No KineDex-specific
dataset copy is produced. The official temporal sizes are retained: horizon
16, two observation frames, and 15 available predicted actions; runtime
executes eight before replanning by default.

| embodiment | state | model action | tactile | RGB |
| --- | ---: | ---: | ---: | --- |
| single | 31 | 35 | `5x29x1` | context, interaction, wrist_right |
| dual | 62 | 70 | `10x29x1` | context, wrist_left, wrist_right |

Each role's model action is 30 benchmark motion values followed by five finger
force targets. Force supervision is the sum of the same-timestamp taxels for
each anatomical finger. The five finger regions contain 29, 21, 18, 18, and
14 taxels and are zero-padded to 29; the 23 palm taxels are deliberately not
passed to the official per-finger encoder. Single and dual checkpoints are
separate.

At deployment, the session compares each predicted force target with current
per-finger force and adds a bounded residual to that finger's flexion joints.
Defaults are `0.005 rad/N` and at most `0.05 rad` total per finger per tick.
Predicted forces are clipped to the checkpoint's training range and joint
targets to the environment limits. Override these using `force_gain_rad_per_n`
and `max_force_residual_rad` policy kwargs.

## Training

The benchmark entry point accepts one or more LeRobot v3 datasets and an
explicit sampling ratio for every source. Ratios must be positive and sum to
one; the weighted sampler and normalization statistics use the same mixture:

```bash
conda run -n evodex-kinedex env KINEDEX_ROOT="$KINEDEX_ROOT" \
  python examples/policies/kinedex/train_benchmark.py \
  --embodiment single \
  --dataset /data/m01 local/evodex-m01 0.50 \
  --dataset /data/m02 local/evodex-m02 0.30 \
  --dataset /data/m03 local/evodex-m03 0.20 \
  -- dataloader.batch_size=32 training.num_epochs=300
```

This is the recommended concentrated-sampling interface. The resolved source
weights, per-source sample counts, repository IDs and content signatures are
stored in checkpoint metadata.

```bash
conda activate evodex-kinedex
export KINEDEX_ROOT=/absolute/path/to/KineDex-official
python examples/policies/kinedex/train.py \
  embodiment=single \
  task.dataset.dataset_path=/absolute/path/to/lerobot-v3-dataset \
  task.dataset.repo_id=local/evodex-m02 \
  dataloader.batch_size=32 training.num_epochs=300
```

Use `embodiment=dual` for dual-arm data. Hydra overrides control learning rate,
workers, validation/checkpoint intervals, and output location. The official
workspace logs train/validation loss, uses cosine scheduling and EMA, and
saves its resolved config plus Evo-DexBench sensor/action metadata. Resume is
the official `latest.ckpt` behavior in the same Hydra output directory.

### Complete tactile-stack ablation

Use the capacity-matched NoTouch configuration to train the benchmark tactile
ablation. It retains the tactile encoder and supplies an identity-normalized
zero placeholder at training and inference, but uses only the 30D per-role
motion action. It neither reads tactile dataset columns nor creates force
targets, force bounds, or runtime force control:

```bash
python examples/policies/kinedex/train.py \
  embodiment=single_no_touch \
  task.dataset.dataset_path=/absolute/path/to/lerobot-v3-dataset \
  task.dataset.repo_id=local/evodex-m02 \
  dataloader.batch_size=32 training.num_epochs=300
```

Use `embodiment=dual_no_touch` for dual-arm data with the context view, or
`embodiment=dual_interaction_no_touch` to match a dual-arm baseline that uses
the interaction view. Keep the dataset split,
optimizer steps, RGB/state inputs, motion targets, and all other training
overrides identical to the corresponding `single` or `dual` full-tactile run.
The total validation losses are not directly comparable because the full model
also predicts force; prefer a fixed final training step for checkpoint
selection. Train the ablation from scratch rather than importing a full-tactile
checkpoint whose action head has a different shape.

### Cross-dataset continuation

Use `training.resume_checkpoint` only for an identical dataset manifest and
sampling configuration. Use `training.init_checkpoint` to import EMA (or raw)
policy weights into a new-data run; KineDex recomputes state, action, and
tactile-force normalization and starts a fresh optimizer and schedule.
Configure primary sources with `task.dataset.dataset_paths/repo_ids` and old
sources with `task.dataset.replay_dataset_paths/replay_repo_ids`. With m05 as
the only primary and `task.dataset.replay_ratio=0.5`, m05 receives 50% and each
of m01--m04 receives 12.5%. Multiple primaries without replay are uniform.
Mixed single- and dual-arm sources are rejected.

KineDex defaults to one deduplicated Arrow query for all tabular horizons and
video timestamps in a batch and the shared batched CPU video path. The query
selects only policy-required Parquet columns before LeRobot tensor conversion,
with locality-aware 128-frame block shuffling, persistent workers, and
prefetch factor 2. Use global frame shuffling with
`cpu_data_pipeline.sampling_mode=random`; both modes preserve
the configured dataset sampling ratios and samples per epoch. Reader contexts use a
bounded per-worker LRU and do not cache decoded frames. On hosts with limited
`/dev/shm`, reduce `dataloader.num_workers` or set
`cpu_data_pipeline.prefetch_factor=1`.

A bounded 100-step installation validation run can be launched as follows:

```bash
python examples/policies/kinedex/train.py \
  embodiment=single \
  task.dataset.dataset_path=/absolute/path/to/lerobot-v3-dataset \
  task.dataset.repo_id=local/evodex-m02 \
  dataloader.batch_size=32 training.num_epochs=1 \
  training.max_train_steps=100 training.max_val_steps=1 \
  training.lr_warmup_steps=10 training.resume=false \
  optimizer.lr=1.0e-4 \
  hydra.run.dir=/absolute/path/to/kinedex-smoke
```

The official workspace reports training loss and learning rate every optimizer
step. `training.max_train_steps` is a per-epoch cap, so the smoke command keeps
`training.num_epochs=1` to produce exactly 100 optimizer steps.

At each sampling interval, `train_action_mse_error` reports the full physical
action MSE. The component metrics slice the final action dimension independently
for every manipulator role: `arm_pose_mse_error` covers the 6D end-effector
delta, `hand_mse_error` covers the 24D absolute hand-joint target, and
`force_mse_error` covers the five predicted finger-force targets. These sampled
action metrics, rather than diffusion noise-prediction `train_loss` alone, should
be used to judge deployment readiness.

Training instrumentation uses the standard W&B API. Select the W&B mode and
credentials through its environment variables rather than adding a
backend-specific dependency to the policy:

```bash
export WANDB_API_KEY=...
export WANDB_MODE=online
python examples/policies/kinedex/train_benchmark.py \
  --embodiment single \
  --dataset /data/m02 local/evodex-m02 1.0 \
  -- dataloader.batch_size=32 training.num_epochs=50 \
  optimizer.lr=1.0e-4 training.min_lr=1.0e-5 \
  hydra.run.dir=/data/outputs/kinedex-m02-50ep
```

W&B-compatible tracking services should be connected by an external launcher
or environment configuration. The benchmark training package does not import
or depend on a specific tracking backend.

## Deployment

Start the policy server from the repository root in the model environment:

```bash
conda activate evodex-kinedex
export KINEDEX_ROOT="$PWD/.third_party/KineDex-official"
python -m kinedex_policy.server \
  --checkpoint /absolute/path/to/latest.ckpt \
  --upstream-root "$KINEDEX_ROOT" \
  --device cuda:0 --host 127.0.0.1 --port 8400
```

In a second terminal, use the `sim` or `vr` environment created with
`--with-kinedex-client`. This installs only the lightweight policy client,
not the training dependencies.

### Standard evaluation

```bash
conda activate evodex-sim
source assets/activate_evodex_assets.sh
dexbench evaluate \
  --policy kinedex --task-id m02_v1 \
  --policy-kwargs '{"host":"127.0.0.1","port":8400,"embodiment":"single","execute_steps":8,"tactile_mode":"full"}' \
  --control-mode arm_pd_ee_delta_pose_hand_pd_joint_pos --obs-mode rgb \
  --log results/kinedex_m02.jsonl
```

Use `evodex-vr` instead if it is your simulator environment. A dual-arm
checkpoint requires `"embodiment":"dual"` and a bimanual task. The standard
protocol uses 100 episodes, four workers, 30 Hz control, 120 Hz simulation,
and the task's configured horizon. See the
[evaluation guide](../../../docs/guides/evaluation.md).

### Diagnostic rollout

Add `--episodes 1 --num-envs 1 --max-steps 300` and use a new log path for a
short integration check. Optional video recording uses
`--video log/videos/kinedex.mp4 --video-fps 30`. Diagnostic overrides produce
`standard_protocol=false`; they are not formal benchmark results.

For a NoTouch checkpoint, use the same evaluation command with
`"tactile_mode":"none"` instead of `"full"`. The server rejects nonzero
tactile placeholders and the session bypasses force control.

The session validates role order, cameras, 30 Hz, tactile layout/version/hash,
joint layout, checkpoint dimensions, and force bounds before applying actions.
It keeps independent histories and force-control state per environment while
the server performs one native GPU batch.

The default policy session uses 30 Hz control and 30 Hz RGB history, matching
the documented collection settings and checkpoint metadata. Unused intermediate
camera renders may still be skipped between action-chunk requests; the frames
actually included in a request retain their training-time spacing.

Runtime control and image frequencies must match the checkpoint metadata.
`control_hz` must also equal `checkpoint_fps`: the session executes native action
chunks without temporal resampling, so changing their playback rate is invalid.
