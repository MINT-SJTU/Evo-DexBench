# Evo-DexBench

[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2610.21180)
[![Project Page](https://img.shields.io/badge/🌐_Project_Page-coming_soon-lightgrey)](https://mint-sjtu.github.io/Evo-DexBench.io)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-Hugging_Face-yellow)](https://huggingface.co/datasets/MINT-SJTU/Evo-DexBench-dataset)
[![Assets](https://img.shields.io/badge/🤗_Assets-Hugging_Face-yellow)](https://huggingface.co/datasets/MINT-SJTU/Evo-DexBench_Assets)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)

**Evo-DexBench** is a dexterous manipulation benchmark built on ManiSkill 3,
with visual, proprioceptive, and tactile observations for single-hand and
bimanual manipulation.

## ✨ Overview

- **30 tasks** across six affordance families, with three layouts per task.
- **Standard evaluation** with spatial, camera, lighting, and physical
  generalization, reporting Success Rate (SR) and StageScore.
- **Keyboard and Quest 3 teleoperation**, canonical data collection, and
  LeRobot export.
- **Model-independent interfaces**, with a tactile-capable KineDex training
  and evaluation example. Model and simulator runtimes remain separate.

Explore the [task catalog](docs/task-catalog.md) and
[documentation](docs/README.md).

## 🚀 Quick Start

Run these commands from the repository root. Conda is required; the setup
helpers create Python 3.10 environments. Graphical teleoperation requires an
interactive display.

### Install

For simulation and model evaluation:

```bash
tools/setup_environment.sh --profile sim --name evodex-sim --with-kinedex-client
conda activate evodex-sim
```

For simulation **and** Quest 3 teleoperation, use this environment instead:

```bash
tools/setup_environment.sh --profile vr --name evodex-vr --with-kinedex-client
conda activate evodex-vr
```

The VR profile includes all simulation dependencies. You do not need both
environments. See [setup options](docs/architecture.md#runtime-setup) for
package mirrors and headless rendering.

### Download assets

Accept the terms and obtain access to the official
[PartNet-Mobility repository](https://huggingface.co/datasets/sapien-sim/PartNetMobility)
before downloading:

```bash
hf auth login
dexbench assets download --asset-dir "$PWD/assets"
source assets/activate_evodex_assets.sh
dexbench assets verify --asset-dir "$PWD/assets"
```

Ordinary assets come from the benchmark bundle; PartNet models are downloaded
directly from the official gated repository. Robot descriptions ship with the
package. See [asset setup](docs/architecture.md#asset-setup) for mirrors and
third-party notices.

### Run a task

```bash
dexbench run --task-id m01_v1 --layout-id studio_l1 --control-hz 30 --sim-freq 120
```

See the [keyboard guide](docs/guides/keyboard-teleoperation.md) for controls.

## 🥽 Quest 3 Teleoperation

In `evodex-vr`, prepare the LAN certificate and start a task:

```bash
source assets/activate_evodex_assets.sh
eval "$(tools/quest3/create_lan_certificate.sh --export)"
dexbench teleop quest3 \
  --mode single --hand-side right \
  --task-id m01_v1 --layout-id studio_l1 --seed 42 \
  --host 0.0.0.0 --advertise-host "$QUEST3_LAN_IP" --port 8013 \
  --cert "$QUEST3_CERT" --key "$QUEST3_KEY" \
  --control-hz 30 --webxr-fps 30 --sim-freq 120 \
  --render-mode human --quest-view
```

In Quest Browser, open `https://<QUEST3_LAN_IP>:8013` to accept the certificate,
then open the printed Vuer URL and enable immersive hand tracking. See the
[Quest 3 guide](docs/guides/quest3-teleoperation.md) for dual-hand control,
headless viewing, and safety settings.

## 📦 Data Collection & Conversion

Using the same VR environment and certificate:

```bash
dexbench data collect \
  --task-id m02_v1 --hand-side right \
  --dataset-root datasets/canonical --dataset-id demo_m02 --split train \
  --episodes 20 --seed 42 --seed-mode increment \
  --generalization-profile collection --generalization-axis balanced \
  --advertise-host "$QUEST3_LAN_IP" \
  --cert "$QUEST3_CERT" --key "$QUEST3_KEY" \
  --control-hz 30 --image-hz 30 --sim-freq 120 \
  --render-mode human --quest-view --validate-after-episode
```

To export collected HDF5 episodes to LeRobot, use the separate converter
environment:

```bash
conda env create -f environment_lerobot.yml
conda activate evodex-lerobot
PYTHONPATH="$PWD" python -m dex_benchmark.commands.data.convert_lerobot \
  datasets/canonical/demo_m02/m02_v1/train datasets/lerobot/demo_m02 \
  --repo-id local/demo_m02 --fps 30
```

See [data collection](docs/guides/data-collection.md) and
[validation, quality, replay, and export](docs/guides/dataset-tools.md).

## 🧠 KineDex Training

Create the independent model environment and clone the pinned official source:

```bash
tools/setup_environment.sh --profile policy-kinedex --name evodex-kinedex
export KINEDEX_ROOT="$PWD/.third_party/KineDex-official"
```

Train from LeRobot v3 datasets with explicit sampling ratios that sum to one:

```bash
conda run -n evodex-kinedex env KINEDEX_ROOT="$KINEDEX_ROOT" \
  python examples/policies/kinedex/train_benchmark.py \
  --embodiment single \
  --dataset /absolute/path/to/m01 local/evodex-m01 0.7 \
  --dataset /absolute/path/to/m02 local/evodex-m02 0.3 \
  -- dataloader.batch_size=32 training.num_epochs=300 \
  optimizer.lr=1.0e-4 training.min_lr=1.0e-5 \
  hydra.run.dir=/absolute/path/to/output
```

Logging uses the standard W&B API. See the
[KineDex guide](examples/policies/kinedex/README.md) for logging, resource
settings, ablations, and checkpoint continuation.

## 📊 Evaluation

**Terminal 1 — model server:**

```bash
export KINEDEX_ROOT="$PWD/.third_party/KineDex-official"
conda run -n evodex-kinedex env KINEDEX_ROOT="$KINEDEX_ROOT" \
  python -m kinedex_policy.server \
  --checkpoint /absolute/path/to/latest.ckpt --upstream-root "$KINEDEX_ROOT" \
  --device cuda:0 --host 127.0.0.1 --port 8400
```

**Terminal 2 — simulator client** (use `evodex-vr` instead if installed):

```bash
conda activate evodex-sim
source assets/activate_evodex_assets.sh
dexbench evaluate \
  --policy kinedex --task-id m02_v1 \
  --control-mode arm_pd_ee_delta_pose_hand_pd_joint_pos --obs-mode rgb \
  --policy-kwargs '{"host":"127.0.0.1","port":8400,"embodiment":"single","execute_steps":8,"tactile_mode":"full"}' \
  --log results/kinedex_m02.jsonl
```

Standard evaluation runs 100 episodes at **30 Hz control / 120 Hz simulation**
and reports **SR** and **StageScore**. See the
[evaluation guide](docs/guides/evaluation.md) for the protocol and diagnostic
overrides.

## 🔌 Integrate Your Model

Implement `PolicySession` and return `SemanticAction` through
`dex_benchmark.policy_api`. Run inference locally or behind a separate server;
WebSocket is used by KineDex but is not required by the benchmark.

```bash
dexbench evaluate --policy my_policy:create_policy --task-id m01_v1 \
  --policy-kwargs '{"checkpoint":"/absolute/path/to/checkpoint"}' \
  --log results/my_model_m01.jsonl
```

See the [model integration guide](docs/model-integration.md) and
[minimal adapter](examples/policies/minimal/README.md).

## 📚 Documentation

- [Task catalog](docs/task-catalog.md)
- [Architecture and setup](docs/architecture.md)
- [Keyboard teleoperation](docs/guides/keyboard-teleoperation.md)
- [Quest 3 teleoperation](docs/guides/quest3-teleoperation.md)
- [Data collection](docs/guides/data-collection.md)
- [Dataset tools](docs/guides/dataset-tools.md)
- [Evaluation](docs/guides/evaluation.md)
- [Model integration](docs/model-integration.md)

## 🙏 Acknowledgement

We thank the developers and contributors of
[ManiSkill](https://github.com/haosulab/ManiSkill),
[SAPIEN](https://github.com/haosulab/SAPIEN),
[RoboCasa](https://github.com/robocasa/robocasa),
[KineDex](https://github.com/DinoMini00/KineDex_code),
[Vuer](https://github.com/vuer-ai/vuer), and
[dex-retargeting](https://github.com/dexsuite/dex-retargeting), as well as the
asset creators listed in [third-party notices](THIRD_PARTY_ASSETS.md).

## 📄 Citation

If you find Evo-DexBench useful in your research, please cite:

```bibtex
@misc{cheng2026evodexbenchbenchmarkinggeneralizationrelated,
      title={Evo-DexBench: Benchmarking Generalization across Related Tasks for Dexterous Manipulation},
      author={Bing Cheng and Yiran Mao and Zewei Ye and Yifan Jin and Xinyi Chen and Ruifeng Tu and Shuhuai Wang and Tao Lin and Zhimu Zhou and Yuxin Du and Fangwei Zhong and Junchi Yan and Bo Zhao},
      year={2026},
      eprint={2610.21180},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2610.21180},
}
```

## 📬 Contact

If you have any questions or encounter any issues, please open an issue in this repository.

You can also scan the QR codes below to contact Zewei Ye or join the WeChat discussion group:

<p align="center">
  <img src="contact/wechat_group.jpg" alt="Evo-DexBench WeChat discussion group" width="200">
  <img src="contact/zeweiYe.jpg" alt="Zewei Ye on WeChat" width="200">
</p>

## ⚖️ License

Evo-DexBench code and documentation are licensed under
[Apache License 2.0](LICENSE). Copyright © 2026 MINT-SJTU.
Third-party and separately licensed asset files retain their own licenses;
see [THIRD_PARTY_ASSETS.md](THIRD_PARTY_ASSETS.md).
