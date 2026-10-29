# DP-C / DP-T

This example uses the unmodified models and training workspaces from
[Stanford Diffusion Policy](https://github.com/real-stanford/diffusion_policy),
pinned to `5ba07ac6661db573af695b419a7947ecb704690f`. DP-C uses the official
convolutional U-Net; DP-T uses the official Transformer. The benchmark supplies
the LeRobot v3 dataset view, training configuration, and a lightweight
WebSocket + MessagePack policy plugin. Simulation and model dependencies remain
in separate environments.

## Setup

Run from the benchmark repository root:

```bash
tools/setup_environment.sh --profile policy-dp --name evodex-dp
conda activate evodex-dp
```

The profile delegates to `tools/setup_diffusion_policy.sh`, which can also be
called directly. It clones the pinned official source without patching it and installs
the model environment. Use `--pip-index-url` and `--torch-index-url` to select
package mirrors. To reuse an existing pinned checkout, call
`tools/setup_diffusion_policy.sh --name evodex-dp --upstream-root /path/to/checkout`
directly; `--upstream-root` is not an option of `tools/setup_environment.sh`.
The selected checkout is saved as `DIFFUSION_POLICY_ROOT` in the conda
environment; reactivate the environment after rerunning setup. Installation
validation imports both official model classes
and LeRobot's dataset API.

In the existing **simulator** environment, install only the client plugin:

```bash
conda activate evodex-sim
python -m pip install -e examples/policies/diffusion_policy
```

This plugin does not install the official model source or training dependencies
in the simulator. DP-C and DP-T do not consume tactile observations.
For a new simulator environment, install the client during setup instead:

```bash
tools/setup_environment.sh --profile sim --name evodex-sim --with-dp-client
```

## Model and data contract

| Embodiment | State dimension | Action dimension | RGB views |
| --- | ---: | ---: | --- |
| `single` | 31 | 30 | context, interaction, wrist_right |
| `dual` | 62 | 60 | context, wrist_left, wrist_right |

Each arm contributes EE pose (7) and hand joint positions (24) to state, and
EE delta pose (6) and hand joint targets (24) to actions. Use LeRobot v3 data
with `meta/dex_benchmark.json`, collected at 30 Hz. Single- and dual-arm data
train separate checkpoints; no zero padding mixes their embodiments.
`embodiment=dual_interaction` selects interaction, wrist_left, wrist_right
for training when that view is required.

Both variants expose prediction horizon 30, observation horizon 2, and 29
available actions. Receding-horizon execution defaults to 8 actions per
request and can be set from 1 to 29 without retraining. DP-C's internal
training window is 32 because the official U-Net requires divisibility by 8;
DP-T's is 30. Both use the original 100-step DDPM scheduler and three RGB
views. These model settings are unchanged from the benchmark's original
implementations.

## Training

In `evodex-dp`, train DP-C:

```bash
python examples/policies/diffusion_policy/train.py \
  variant=dp_c embodiment=single \
  task.dataset.dataset_path=/absolute/path/to/m02 \
  task.dataset.repo_id=local/evodex-m02 \
  training.num_epochs=100 variant.optimizer.lr=1e-4 \
  hydra.run.dir=/absolute/path/to/dp-c-output
```

For DP-T, use `variant=dp_t` and `variant.optimizer.learning_rate=1e-4` instead of
`variant.optimizer.lr`. For dual-arm data, select `embodiment=dual`. Standard Hydra
overrides configure batch size, epochs, data-loader workers and W&B logging.
Model architecture overrides use `variant.policy.*`.
Set `logging.mode=disabled` for local training without uploading logs.

A complete DP-T command is:

```bash
python examples/policies/diffusion_policy/train.py \
  variant=dp_t embodiment=single \
  task.dataset.dataset_path=/absolute/path/to/m02 \
  task.dataset.repo_id=local/evodex-m02 \
  training.num_epochs=100 variant.optimizer.learning_rate=1e-4 \
  logging.mode=disabled hydra.run.dir=/absolute/path/to/dp-t-output
```

The official cosine schedule decays to zero by default. Optionally set
`training.min_lr=1e-5` to retain the warmup/cosine shape with a nonzero floor.
For limited host RAM or `/dev/shm`, reduce `dataloader.num_workers` and
`val_dataloader.num_workers`; when using zero workers, also set
`dataloader.persistent_workers=false`. The default CPU pipeline uses batched
PyAV decoding and locality-aware block shuffling. Use
`cpu_data_pipeline.sampling_mode=random` for global frame shuffling.
The default validation split holds out 10% of episodes. Set
`task.dataset.val_ratio=0` to use every demonstration; validation-ranked top-k
checkpointing is then disabled automatically, while periodic and final
checkpoints are still saved.

### Multi-dataset training and continuation

Configure primary datasets with `task.dataset.dataset_paths` and
`task.dataset.repo_ids`. Primary sources are task-balanced. For Core-only
training, list all four Core datasets with these two parameters and leave
replay settings at their defaults:

```bash
python examples/policies/diffusion_policy/train.py \
  variant=dp_c embodiment=single \
  'task.dataset.dataset_paths=[/data/m01,/data/m02,/data/m03,/data/m04]' \
  'task.dataset.repo_ids=[local/m01,local/m02,local/m03,local/m04]' \
  task.dataset.val_ratio=0 training.num_epochs=600 training.min_lr=1e-6 \
  logging.mode=disabled hydra.run.dir=/absolute/path/to/dp-c-core-output
```

Do not mix singular and plural primary parameters in one command. To give m05 50%
of the samples and replay m01–m04 at 12.5% each, use:

```bash
python examples/policies/diffusion_policy/train.py \
  variant=dp_c embodiment=single \
  'task.dataset.dataset_paths=[/data/m05]' \
  'task.dataset.repo_ids=[local/m05]' \
  'task.dataset.replay_dataset_paths=[/data/m01,/data/m02,/data/m03,/data/m04]' \
  'task.dataset.replay_repo_ids=[local/m01,local/m02,local/m03,local/m04]' \
  task.dataset.replay_ratio=0.5 \
  task.dataset.val_ratio=0 training.num_epochs=600 training.min_lr=1e-6 \
  logging.mode=disabled \
  hydra.run.dir=/absolute/path/to/dp-c-replay-output
```

Use `training.resume_checkpoint` only with the identical dataset manifest and
policy contract; it restores full workspace state. For a new-data run, use
`training.init_checkpoint`: EMA weights (or raw weights when EMA is absent)
are imported, while normalization, optimizer, schedule and tracking start
fresh. These two options are mutually exclusive.

`checkpoint.save_last_ckpt=true` saves a synchronous final `latest.ckpt` under
the run's `checkpoints/` directory. `training.checkpoint_epochs` optionally
selects completed epochs for checkpoint saving.

## Pretrained checkpoints

Use the [released examples](https://huggingface.co/MINT-SJTU/Evo-DexBench_model_access_example)
without training. First complete the model/client setup above and the
[simulator asset setup](../../../README.md#download-assets). Run all commands
from the benchmark repository root; authenticate with `hf auth login` if needed.

Download the DP-C Single2 checkpoint in the simulator environment:

```bash
conda activate evodex-sim
hf download MINT-SJTU/Evo-DexBench_model_access_example \
  DP-C/single2-core.ckpt manifest.json \
  --repo-type model --local-dir checkpoints/model-access-example
```

Terminal 1 — model environment:

```bash
conda activate evodex-dp
python -m dp_policy.server \
  --checkpoint checkpoints/model-access-example/DP-C/single2-core.ckpt \
  --upstream-root "$PWD/.third_party/DiffusionPolicy-official" \
  --host 127.0.0.1 --port 8100 --device cuda:0
```

Terminal 2 — simulator environment, ten diagnostic episodes:

```bash
conda activate evodex-sim
source assets/activate_evodex_assets.sh
dexbench evaluate --policy dp_c --task-id m09_v1 \
  --episodes 10 --num-envs 1 \
  --control-mode arm_pd_ee_delta_pose_hand_pd_joint_pos --obs-mode rgb \
  --policy-kwargs '{"host":"127.0.0.1","port":8100,"embodiment":"single","execute_steps":8}' \
  --log results/dp_c_example_m09.jsonl
```

For **DP-T**, download and serve `DP-T/single2-core.ckpt`, then change the client
to `--policy dp_t`. Both Single2 checkpoints support `m06_v1`–`m09_v1`.

For **Double3**, download and serve the selected policy's `double3-core.ckpt`,
choose `b11_v1`–`b14_v1`, and replace the client kwargs with:

```json
{"host":"127.0.0.1","port":8100,"embodiment":"dual","execute_steps":8,"camera_roles":["interaction","wrist_left","wrist_right"]}
```

Double3 examples require the **interaction** view, not the default dual-arm
context view. All examples retain 30 Hz control and 120 Hz simulation. Omit
`--episodes` and `--num-envs` for the standard 100-episode, four-worker protocol;
the ten-episode command is only an integration check. Choose a new log filename
for each run.

These exports retain inference weights, normalizers and policy metadata but
omit optimizer and training-progress state. Do not use them with
`training.resume_checkpoint`. Only load trusted `.ckpt` files: they contain
pickled configuration/state and require the supplied server, not a generic
`from_pretrained` loader.

## Evaluation

Terminal 1, in the model environment:

```bash
conda activate evodex-dp
python -m dp_policy.server \
  --checkpoint /absolute/path/to/dp-c-output/checkpoints/latest.ckpt \
  --host 127.0.0.1 --port 8100 --device cuda:0
```

Terminal 2, in the simulator environment:

```bash
conda activate evodex-sim
source /absolute/path/to/evodex-assets/activate_evodex_assets.sh
dexbench evaluate --policy dp_c --task-id m02_v1 --obs-mode rgb \
  --policy-kwargs '{"host":"127.0.0.1","port":8100,"embodiment":"single","execute_steps":8}' \
  --log results/dp_c_m02.jsonl
```

For a DP-T checkpoint, serve that checkpoint and select `--policy dp_t`.
Dual-arm checkpoints use a dual-arm task and `embodiment=dual` in the client.
For a checkpoint trained with `embodiment=dual_interaction`, keep the client
embodiment `dual` and additionally pass
`"camera_roles":["interaction","wrist_left","wrist_right"]` inside
`--policy-kwargs`. This selects the training camera contract without changing
the dual-arm state or action definition. The server checks policy identity,
observation keys and output shape; the client additionally checks checkpoint
dimensions and horizons before using its actions. Only load trusted checkpoints:
official workspace checkpoints contain pickled training state.

The evaluator uses the unchanged standard protocol: simulation 120 Hz,
control 30 Hz, SR and StageScore. The session supports native batched inference
for parallel environments. During action-chunk execution it requests only
the consecutive observation frames needed by the next model request, preserving
the existing render-on-demand optimization.
