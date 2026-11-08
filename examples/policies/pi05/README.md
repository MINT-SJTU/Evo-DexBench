# π0.5

This example fine-tunes and serves
[π0.5](https://github.com/Physical-Intelligence/openpi) with the official
OpenPI **JAX** implementation, pinned to
`15a9616a00943ada6c20a0f158e3adb39df2ccac`. Training is full-parameter
fine-tuning of the released π0.5 base (PaliGemma 2B + 300M action expert)
with OpenPI's model, loss, AdamW optimizer, EMA, FSDP sharding and Orbax
checkpoints. The benchmark supplies a small overlay for the OpenPI checkout
(LeRobot v3 data, Evo-DexBench state/action/camera mapping and
task-balanced sampling) and a lightweight simulator client. The model server
and the simulator run in separate environments and communicate through
OpenPI's WebSocket protocol.

## Setup

### Model environment

OpenPI manages its own Python 3.11 environment with
[uv](https://docs.astral.sh/uv/getting-started/installation/). From the
benchmark repository root:

```bash
tools/setup_environment.sh --profile policy-pi05
export OPENPI_ROOT="$PWD/.third_party/openpi"
export PI05_PYTHON="$OPENPI_ROOT/.venv/bin/python"
```

The profile runs `tools/setup_pi05.sh`, which clones the pinned official
checkout, installs the overlay with `install_overlay.py`, runs
`uv sync --frozen` from OpenPI's lockfile, installs LeRobot 0.4.0's dataset
API for LeRobot v3.0 data, keeps a single headless OpenCV distribution, and
finally runs `validate_install.py`. Use `--pip-index-url` for a package mirror
and `--upstream-url` for a trusted OpenPI mirror. To reuse a checkout
elsewhere, call `tools/setup_pi05.sh --upstream-root /path/to/openpi`
directly.

The overlay adds four modules to the checkout and edits three upstream files
at fixed anchors:

| Change | Purpose |
| --- | --- |
| `training/evodex_config.py` | `evodex_pi05_single_jax` and `evodex_pi05_dual_jax` configs |
| `policies/evodex_policy.py` | Selects the policy state from the canonical LeRobot state; maps cameras and actions |
| `training/evodex_lerobot.py`, `evodex_lerobot_cpu.py` | Multi-dataset LeRobot v3 reader (PyAV) and task-balanced samplers shared with the other benchmark policies |
| `training/config.py` | Registers the configs |
| `training/data_loader.py` | LeRobot v3 import, batched reads and the benchmark samplers |
| `models/tokenizer.py` | Optional error instead of silent prompt truncation (enabled by `train.py` and `serve.py`) |

### Simulator client

In the simulator environment, install only the client plugin. It depends on
the lightweight `openpi-client` package from the same OpenPI commit, not on
JAX or the training stack:

```bash
conda activate evodex-sim
python -m pip install -e examples/policies/pi05
```

For a new simulator environment, use
`tools/setup_environment.sh --profile sim --name evodex-sim --with-pi05-client`.

## Model and data contract

| Embodiment | Policy state | Physical action | Model state / action | RGB views (base, left, right slot) | Prompt tokens |
| --- | ---: | ---: | ---: | --- | ---: |
| `single` | 31 | 30 | 32 / 32 | context, interaction, wrist_right | 200 |
| `dual` | 62 | 60 | 64 / 64 | context, wrist_left, wrist_right | 384 |

Each arm contributes its EE pose (7) and hand joint positions (24) to the
policy state, selected from the canonical 111D-per-arm `observation.state`
of the LeRobot dataset. Actions are the EE delta pose (6) and absolute hand
joint targets (24) per arm, left arm first. State and action are zero-padded
at the end to the model dimension; the client discards the padded action
entries. π0.5 discretizes the normalized state into the prompt, so dual-arm
prompts need 384 tokens; prompts are never truncated.

The action horizon is 50 at 30 Hz. The client executes `execute_steps`
actions (default 10) per request. Use LeRobot v3 data with `meta/info.json`
at 30 Hz, such as the
[Evo-DexBench dataset](https://huggingface.co/datasets/MINT-SJTU/Evo-DexBench-dataset).
Single- and dual-arm data train separate checkpoints.

## Training

`train.py` runs in the model environment. One Python process drives all local
GPUs with FSDP; do not wrap it in `torchrun`. Full-parameter training needs
80 GB GPUs; adding GPUs shards the model and optimizer state across them.
The global batch size, not the number of GPUs, defines the recipe.

Download the four Single2 Core tasks (M06–M09) and train jointly:

```bash
hf download MINT-SJTU/Evo-DexBench-dataset --repo-type dataset \
  --include "lerobot/Rotational manipulation/m0[6-9]/*" --local-dir datasets/evodex
D="$PWD/datasets/evodex/lerobot/Rotational manipulation"

"$PI05_PYTHON" examples/policies/pi05/train.py \
  --openpi-root "$OPENPI_ROOT" --embodiment single \
  --dataset "$D/m06" --dataset "$D/m07" --dataset "$D/m08" --dataset "$D/m09" \
  --dataset-id single2-core --output-dir /absolute/path/to/pi05-single2-core
```

The script first computes normalization statistics with OpenPI's
`scripts/compute_norm_stats.py` and then trains from the official base
`gs://openpi-assets/checkpoints/pi05_base/params`. The defaults are the
benchmark recipe: global batch 32, 25 epochs, cosine learning rate
5e-6 → 5e-7 with 1000 warmup updates, EMA 0.99, gradient clipping 1.0 and
seed 42. Every dataset passed with `--dataset` receives an equal share of each
epoch, and one epoch draws `datasets × largest dataset frames` samples, so the
update budget is `epochs × floor(samples per epoch / batch)`. Override it with
`--num-train-steps`, and the schedule with `--warmup-steps`, `--peak-lr` and
`--min-lr`.

Dual-arm training starts from a 64D copy of the base. Prepare it once:

```bash
"$PI05_PYTHON" examples/policies/pi05/extend_action_head.py \
  gs://openpi-assets/checkpoints/pi05_base/params \
  /absolute/path/to/pi05_base_dual64/params
```

Only the action input/output projections are widened: the released 32
coordinates are kept, new kernel entries use LeCun-normal initialization
(seed 42) and new bias entries are zero. The script reloads the result and
checks that every other tensor is unchanged. It holds the float32 parameters
in host memory while writing them and needs more than 32 GB of RAM, so run
it on the training node. Then train the Double2 Core tasks
(B06–B09):

```bash
hf download MINT-SJTU/Evo-DexBench-dataset --repo-type dataset \
  --include "lerobot/Assembly/b0[6-9]/*" --local-dir datasets/evodex
D="$PWD/datasets/evodex/lerobot/Assembly"

"$PI05_PYTHON" examples/policies/pi05/train.py \
  --openpi-root "$OPENPI_ROOT" --embodiment dual \
  --base-params /absolute/path/to/pi05_base_dual64/params \
  --dataset "$D/b06" --dataset "$D/b07" --dataset "$D/b08" --dataset "$D/b09" \
  --sampling exact \
  --dataset-id double2-core --output-dir /absolute/path/to/pi05-double2-core
```

`--sampling locality` (default) draws each dataset's share over the epoch
from shuffled temporal blocks, which keeps video decoding local.
`--sampling exact` additionally fixes the share inside every batch (8 frames
of each task in a batch of 32). `--sampling random` shuffles individual frames.
The released checkpoints used these settings:

| Checkpoint | Data | Sampling | Updates | GPUs |
| --- | --- | --- | ---: | ---: |
| `single2-core` | M06–M09, 50 demonstrations each | `locality` | 136,925 | 4 |
| `double2-core` | B06–B09, 50 demonstrations each | `exact` | 83,525 | 2 |

Outputs are written under `--output-dir`:

```text
run_config.json                      # resolved data, budget and schedule
metrics.jsonl                        # loss, grad/param norms, learning rate
assets/<config>/<dataset-id>/norm_stats.json
checkpoints/<config>/<dataset-id>/<step>/{params,assets,train_state}
```

Checkpoints are saved every `--save-interval` updates (default 10000) and at
the end; directory names are zero-based update indices, so the final
checkpoint of a 136,925-update run is `136924/`. OpenPI keeps only the most
recent checkpoint. `params/` holds the EMA weights used for inference;
`train_state/` holds the raw weights and optimizer state and is needed only
for `--resume`. Serving reads `params/` and `assets/`.

To continue an interrupted run, repeat the identical command with
`--resume`; the script refuses a different data or schedule configuration
for an existing output directory. As in OpenPI, resuming restores the
training state but not the data-loader position. For a new data mixture, use
a new output directory and pass a previous checkpoint's `params/` directory as
`--base-params`. To replay older tasks while training a new one, add
`--replay-dataset` (repeatable) and `--replay-ratio`; for example, a ratio of
0.5 gives the replay datasets half of every epoch. `--norm-stats` reuses an
existing `norm_stats.json`, and `--prepare-only` stops after preparation.
Add `--wandb-project NAME` to log metrics to Weights & Biases.

## Pretrained checkpoints

The [released examples](https://huggingface.co/MINT-SJTU/Evo-DexBench_model_access_example)
include π0.5 `single2-core` (M06–M09) and `double2-core` (B06–B09)
checkpoints. Complete the model and client setup above and the
[simulator asset setup](../../../README.md#download-assets), then download one
checkpoint with its license, notice, and manifest from the repository root:

```bash
hf download MINT-SJTU/Evo-DexBench_model_access_example \
  --repo-type model --local-dir checkpoints/model-access-example \
  --include "pi05/single2-core/*" --include "pi05/LICENSE_GEMMA.txt" \
  --include "pi05/NOTICE" --include "manifest.json"
```

Terminal 1 — model environment:

```bash
"$PI05_PYTHON" examples/policies/pi05/serve.py \
  --openpi-root "$OPENPI_ROOT" --embodiment single \
  --checkpoint checkpoints/model-access-example/pi05/single2-core --port 8000
```

The server is ready once it logs `server listening on 0.0.0.0:8000`.
Terminal 2 — simulator environment, ten diagnostic episodes:

```bash
conda activate evodex-sim
source assets/activate_evodex_assets.sh
dexbench evaluate --policy pi05 --task-id m09_v1 \
  --episodes 10 --num-envs 1 \
  --control-mode arm_pd_ee_delta_pose_hand_pd_joint_pos --obs-mode rgb \
  --policy-kwargs '{"host":"127.0.0.1","port":8000,"embodiment":"single","execute_steps":10}' \
  --log results/pi05_example_m09.jsonl
```

The Single2 checkpoint supports `m06_v1`–`m09_v1`. For **Double2**, replace
`pi05/single2-core/*` with `pi05/double2-core/*` in the download command and
keep the other includes. Serve it with `--embodiment dual`, choose
`b06_v1`–`b09_v1` and use
`{"host":"127.0.0.1","port":8000,"embodiment":"dual","execute_steps":10}`
as the client kwargs. Omit `--episodes` and `--num-envs` for the standard
100-episode, four-worker protocol; the ten-episode command is only an
integration check. Choose a new log filename for each run.

These checkpoints contain the EMA inference weights (`params/`, float32
Orbax, unchanged from training) and the normalization statistics used in
training (`assets/`). The training state (raw weights and optimizer) is not
included, so they cannot be continued with `--resume`; their `params/`
directory can initialize new training through `--base-params`.

## Evaluation

Serve a trained checkpoint directory, which contains `params/` and `assets/`:

```bash
"$PI05_PYTHON" examples/policies/pi05/serve.py \
  --openpi-root "$OPENPI_ROOT" --embodiment single \
  --checkpoint /absolute/path/to/pi05-single2-core/checkpoints/evodex_pi05_single_jax/single2-core/136924 \
  --port 8000
```

`serve.py` checks the pinned checkout, the Orbax parameters and the
normalization statistics, infers the dataset id when the checkpoint holds a
single one (otherwise pass `--dataset-id`), and then runs OpenPI's unmodified
`scripts/serve_policy.py`. `--dry-run` prints the command without starting the
server. JAX loads the parameters in bfloat16 for inference; the server and
four simulator workers can share one GPU.

In the simulator environment:

```bash
dexbench evaluate --policy pi05 --task-id m06_v1 --obs-mode rgb \
  --policy-kwargs '{"host":"127.0.0.1","port":8000,"embodiment":"single","execute_steps":10}' \
  --log results/pi05_m06.jsonl
```

Use `"embodiment":"dual"` with a dual-arm checkpoint and task. The evaluator
uses the standard protocol: simulation 120 Hz, control 30 Hz, SR and
StageScore. The client sends the current RGB views, state and task
instruction once per action chunk, so simulator workers skip rendering while
buffered actions are executed. It validates the returned chunk shape and
rejects tasks whose manipulators do not match the selected embodiment. If the
server runs on another host, prefer an SSH tunnel to exposing the
unauthenticated port.

## Licenses

OpenPI is released under the Apache License 2.0. The π0.5 base weights are
built on PaliGemma; checkpoints fine-tuned from them, including the released
examples, are Gemma model derivatives subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
Gemma is provided under and subject to the Gemma Terms of Use found at
ai.google.dev/gemma/terms.
