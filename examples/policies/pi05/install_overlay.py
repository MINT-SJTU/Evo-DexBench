"""Install the Evo-DexBench π0.5 overlay into a clean pinned OpenPI checkout.

The overlay adds three modules and makes narrow, anchored edits to upstream:

* register the ``evodex_pi05_*_jax`` configs in ``training/config.py``;
* read LeRobot v3.0 datasets with the PyAV backend and the batched benchmark
  reader, using task-balanced samplers, in ``training/data_loader.py``;
* optionally refuse prompt truncation in ``models/tokenizer.py`` (enabled by
  ``train.py`` and ``serve.py``), so a long dual-arm state is never cut off.

Every anchor is checked before any file is written.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PINNED_COMMIT = (HERE / "pi05_policy/UPSTREAM_COMMIT").read_text().split()[-1]

CONFIG_IMPORT_ANCHOR = "# Use `get_config` if you need to get a config by name in your code.\n"
CONFIG_IMPORT = "from openpi.training import evodex_config as _evodex_config\n\n"
CONFIG_LIST_ANCHOR = "    *roboarena_config.get_roboarena_configs(),\n"
CONFIG_LIST = "    *_evodex_config.get_evodex_configs(),\n"

LEROBOT_V2_IMPORT = "import lerobot.common.datasets.lerobot_dataset as lerobot_dataset"
LEROBOT_V3_IMPORT = "import lerobot.datasets.lerobot_dataset as lerobot_dataset"
CREATE_DATASET_ANCHOR = """    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )
"""
CREATE_DATASET_REPLACEMENT = """    from openpi.training import evodex_lerobot

    dataset, dataset_meta = evodex_lerobot.create_dataset(data_config, action_horizon)
"""
TRANSFORM_GETITEM_ANCHOR = """    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
"""
TRANSFORM_GETITEM_REPLACEMENT = """    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __getitems__(self, indices) -> list[T_co]:
        getter = getattr(self._dataset, "__getitems__", None)
        items = getter(indices) if getter is not None else [self._dataset[index] for index in indices]
        return [self._transform(item) for item in items]

    @property
    def episode_blocks(self):
        return getattr(self._dataset, "episode_blocks", None)

    @property
    def source_episode_blocks(self):
        return self._dataset.source_episode_blocks

    @property
    def source_weights(self):
        return self._dataset.source_weights

    @property
    def source_sample_counts(self):
        return self._dataset.source_sample_counts

    def __len__(self) -> int:
"""
TORCH_LOADER_ANCHOR = """        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
"""
TORCH_LOADER_REPLACEMENT = """        if sampler is None:
            from openpi.training import evodex_lerobot

            loader_kwargs = dict(
                batch_size=local_batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                multiprocessing_context=mp_context,
                persistent_workers=num_workers > 0,
                collate_fn=_collate_fn,
                worker_init_fn=_worker_init_fn,
                drop_last=True,
            )
            if num_workers > 0:
                loader_kwargs["prefetch_factor"] = int(os.environ.get("EVODEX_PREFETCH_FACTOR", "2"))
            sampling_mode = os.environ.get("EVODEX_SAMPLING_MODE", "locality")
            if getattr(dataset, "episode_blocks", None) is None:
                sampling_mode = "random"
            self._data_loader = evodex_lerobot.make_dataloader(
                typing.cast(torch.utils.data.Dataset, dataset),
                sampling_mode=sampling_mode,
                temporal_block_size=int(os.environ.get("EVODEX_TEMPORAL_BLOCK_SIZE", "128")),
                seed=seed,
                **loader_kwargs,
            )
            return
        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
"""
TOKENIZER_ANCHOR = """            if len(tokens) > self._max_len:
                logging.warning(
"""
TOKENIZER_REPLACEMENT = """            if len(tokens) > self._max_len:
                if os.environ.get("EVODEX_REJECT_PROMPT_TRUNCATION") == "1":
                    raise ValueError(
                        f"Refusing to truncate {len(tokens)} prompt tokens to {self._max_len}; "
                        "the complete discretized state is required."
                    )
                logging.warning(
"""
OVERLAY_FILES = {
    "src/openpi/policies/evodex_policy.py": HERE / "openpi_overlay/evodex_policy.py",
    "src/openpi/training/evodex_config.py": HERE / "openpi_overlay/evodex_config.py",
    "src/openpi/training/evodex_lerobot.py": HERE / "openpi_overlay/evodex_lerobot.py",
    "src/openpi/training/evodex_lerobot_cpu.py": REPO_ROOT / "dex_benchmark/data/lerobot_cpu.py",
}


def _replace_once(text: str, anchor: str, replacement: str, name: str) -> str:
    if text.count(anchor) < 1:
        raise RuntimeError(f"Pinned OpenPI anchor was not found: {name}.")
    return text.replace(anchor, replacement, 1)


def _patched_config(text: str) -> str:
    # Import only after TrainConfig/DataConfigFactory are defined; importing the
    # overlay among the top-level imports would see a partial config module.
    text = _replace_once(text, CONFIG_IMPORT_ANCHOR, CONFIG_IMPORT + CONFIG_IMPORT_ANCHOR, "config import")
    return _replace_once(text, CONFIG_LIST_ANCHOR, CONFIG_LIST + CONFIG_LIST_ANCHOR, "config list")


def _patched_data_loader(text: str) -> str:
    text = _replace_once(text, LEROBOT_V2_IMPORT, LEROBOT_V3_IMPORT, "LeRobot import")
    text = _replace_once(text, CREATE_DATASET_ANCHOR, CREATE_DATASET_REPLACEMENT, "dataset construction")
    text = _replace_once(text, TRANSFORM_GETITEM_ANCHOR, TRANSFORM_GETITEM_REPLACEMENT, "TransformedDataset")
    if text.count(TORCH_LOADER_ANCHOR) != 1:
        raise RuntimeError("Pinned OpenPI TorchDataLoader anchor was not found exactly once.")
    return text.replace(TORCH_LOADER_ANCHOR, TORCH_LOADER_REPLACEMENT, 1)


def _patched_tokenizer(text: str) -> str:
    # Patch only PaligemmaTokenizer, the tokenizer used by π0.5.
    first = text.find(TOKENIZER_ANCHOR)
    if first < 0 or first > text.find("class FASTTokenizer"):
        raise RuntimeError("Pinned OpenPI PaligemmaTokenizer anchor was not found.")
    return text.replace(TOKENIZER_ANCHOR, TOKENIZER_REPLACEMENT, 1)


def install(openpi_root: Path) -> None:
    root = openpi_root.expanduser().resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected OpenPI {PINNED_COMMIT}, got {commit}.")
    targets = [root / relative for relative in OVERLAY_FILES]
    if any(path.exists() for path in targets):
        raise FileExistsError("The overlay is already installed; use a clean pinned OpenPI checkout.")

    edits = {
        "src/openpi/training/config.py": _patched_config,
        "src/openpi/training/data_loader.py": _patched_data_loader,
        "src/openpi/models/tokenizer.py": _patched_tokenizer,
    }
    patched = {
        relative: patch((root / relative).read_text(encoding="utf-8"))
        for relative, patch in edits.items()
    }
    for relative, source in OVERLAY_FILES.items():
        shutil.copy2(source, root / relative)
    for relative, text in patched.items():
        (root / relative).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("openpi_root", type=Path)
    args = parser.parse_args()
    install(args.openpi_root)
    print(f"Installed the Evo-DexBench overlay into {args.openpi_root}.")


if __name__ == "__main__":
    main()
