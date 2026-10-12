# Architecture

The public project separates its runtime, interfaces, data, and command layer:

```text
dex_benchmark/
  sim/                    tasks, environments, scenes, robots, sensors, rendering
  evaluation/             fixed protocol, parallel execution, metrics, video
  policy_api/             versioned model contract and plugin registry
  teleoperation/          Quest 3, keyboard/mouse, shared hand retargeting
  data/                    canonical dataset schema and conversion
  commands/               installed `dexbench` command implementations
  resources/              packaged task and protocol YAML
examples/policies/        external policy plugins (minimal, KineDex)
tools/                    environment, Vulkan, KineDex and Quest TLS setup
```

## Dependency direction

`sim` never imports a concrete policy. Policies consume the versioned
`policy_api` contract and may run in another process or environment. The
evaluation runner owns the connection between a `PolicySession` and a task.
Quest 3 produces the same semantic actions and canonical observations used by
the policy and dataset boundaries.

```text
policy plugin ── PolicySession ─┐
Quest 3 ─────── SemanticAction ─┼─> sim task ─> evaluation/data
zero reference policy ──────────┘
```

Task metadata is loaded from packaged resources without importing SAPIEN.
`make_task_env()` lazily imports only the selected task and fixed robot.
External asset lookup uses `EVODEX_ASSET_DIR`; package resources use
`importlib.resources`, so a wheel works outside the source checkout.

## Rendering and tactile capability

The evaluator uses `--obs-mode` to select visual inputs and the policy's
capture methods to skip unused intermediate observations. Static
`ObservationSpec` and `PolicyCapabilities` declarations alone do not configure
the simulator. State-only execution can
use `render_backend="none"`; camera sensors and RGB rendering are not created
or validated in that mode. Tactile observation work can likewise be disabled
for policies that do not request it without changing task contact predicates.

## Stable public entry points

The only installed executable is `dexbench`. Formal workflows are `run`,
`teleop`, `data`, `evaluate`, `assets`, `tasks`, and `doctor`. Repository-path
scripts provide reproducible setup for the supported public workflows.

The fixed evaluation schedule and SR/StageScore semantics remain in
`evaluation/` and packaged resources.

## Runtime setup

Run setup scripts from the repository root. The `sim` profile supports
simulation, evaluation, and canonical dataset tools. The `vr` profile includes
all of these plus Quest 3 dependencies; use one environment, not both.
KineDex training/inference and LeRobot export use separate environments.

The setup helpers isolate user-site packages and install PyTorch 2.5.1 with
CUDA 12.4. For a trusted package mirror, append
`--pip-index-url https://pypi.tuna.tsinghua.edu.cn/simple` to the setup command.
`--torch-index-url` separately selects the CUDA wheel index. Reactivate an
already active environment after rerunning setup.

For off-screen RGB/depth rendering on a headless NVIDIA Ubuntu host:

```bash
sudo tools/setup_headless_vulkan.sh
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/evodex_nvidia_egl_icd.json
```

Omit `sudo` when running as root. Export the ICD path in each simulator shell.
The helper validates NVIDIA Vulkan enumeration without starting an X server.
State-only evaluation can instead use
`--obs-mode state_dict --env-kwargs '{"render_backend":"none"}'`.
Keyboard teleoperation still requires an interactive display; Quest-only
teleoperation needs off-screen rendering but no desktop viewer.

## Asset setup

`dexbench assets download` installs the ordinary bundle and required official
PartNet archives into `--asset-dir`, then generates local collision overlays.
Source `<asset-dir>/activate_evodex_assets.sh` in every simulator shell.
The default bundle repository is `MINT-SJTU/Evo-DexBench_Assets`.
Use `--bundle-repo` or `EVODEX_ASSET_REPO` to select another bundle repository.

The downloader tries the official Hugging Face endpoint first and falls back
to `https://hf-mirror.com` for connectivity failures. PartNet is gated:
forwarding its token to a third-party mirror requires explicit consent:

```bash
dexbench assets download --asset-dir "$PWD/assets" --allow-token-to-mirror
```

This option sends credentials to the mirror. Only use it if you trust that
service. `--no-hf-mirror`, `--hf-endpoint`, and `--hf-mirror-endpoint` control
endpoint selection. If authentication itself requires a trusted mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com hf auth login
```

This login also sends credentials to that third party. Resume interrupted
downloads by rerunning the same command; do not delete the asset directory or
start competing downloads. HTTP 429 responses may require waiting before
retrying.

PartNet downloads remain unchanged. Generated overlays preserve benchmark
collision settings and repair model 25144's missing `new-24.obj` reference
using `original-24.obj` for visuals and collisions. See the
[PartNet notice](../assets/PARTNET_NOTICE.md) and
[third-party asset licenses](../THIRD_PARTY_ASSETS.md).
