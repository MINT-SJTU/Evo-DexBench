"""Prepare the 64D dual-arm π0.5 base from the released 32D JAX base.

The released π0.5 base projects 32D actions. Dual-arm Evo-DexBench actions are
60D and are padded to 64D, so only ``action_in_proj`` and ``action_out_proj``
are widened: the original 32 coordinates are kept, new kernel coordinates use
Flax Linear's default LeCun-normal initialization and new bias entries are
zero. Every other tensor is copied unchanged, which is checked after an Orbax
read-back. An audit JSON is written next to the output.

Run once in the OpenPI environment, then point ``train.py --base-params`` at
the output directory::

    python examples/policies/pi05/extend_action_head.py \
        gs://openpi-assets/checkpoints/pi05_base/params \
        /absolute/path/to/pi05_base_dual64/params
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECTIONS = ("action_in_proj/kernel", "action_out_proj/kernel", "action_out_proj/bias")


def tensor_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def expand_flat_params(params: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp

    missing = set(PROJECTIONS) - params.keys()
    if missing:
        raise ValueError(f"Missing action projections: {sorted(missing)}")
    source_in = params["action_in_proj/kernel"]
    source_out = params["action_out_proj/kernel"]
    source_bias = params["action_out_proj/bias"]
    width = source_in.shape[1]
    if source_in.shape != (32, width) or source_out.shape != (width, 32) or source_bias.shape != (32,):
        raise ValueError("Expected the released 32D π0.5 action projections.")
    key_in, key_out = jax.random.split(jax.random.key(seed))
    # [32, W] -> [64, W] and [W, 32] -> [W, 64]; keep the original coordinates.
    initializer = jax.nn.initializers.lecun_normal()
    new_in = np.asarray(initializer(key_in, (64, width), jnp.float32)).astype(source_in.dtype)
    new_out = np.asarray(initializer(key_out, (width, 64), jnp.float32)).astype(source_out.dtype)
    new_bias = np.zeros((64,), dtype=source_bias.dtype)
    new_in[:32] = source_in
    new_out[:, :32] = source_out
    new_bias[:32] = source_bias
    result = dict(params)
    result.update(
        {
            "action_in_proj/kernel": new_in,
            "action_out_proj/kernel": new_out,
            "action_out_proj/bias": new_bias,
        }
    )
    return result


def convert(source: str, output: Path, seed: int) -> dict[str, Any]:
    import gc

    import flax.traverse_util
    import orbax.checkpoint as ocp
    from openpi.models.model import restore_params
    from openpi.shared import download

    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source_path = download.maybe_download(source)
    flat = flax.traverse_util.flatten_dict(restore_params(source_path, restore_type=np.ndarray), sep="/")
    source_hashes = {name: tensor_hash(value) for name, value in flat.items()}
    originals = {name: flat[name].copy() for name in PROJECTIONS}
    expanded = expand_flat_params(flat, seed)
    expected_hashes = {name: tensor_hash(value) for name, value in expanded.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(output, {"params": flax.traverse_util.unflatten_dict(expanded, sep="/")})
    # Release the in-memory copy before reading the result back, so the peak
    # memory stays near one copy of the parameters.
    del flat, expanded
    gc.collect()

    restored = flax.traverse_util.flatten_dict(restore_params(output, restore_type=np.ndarray), sep="/")
    if restored.keys() != expected_hashes.keys():
        raise AssertionError("Orbax read-back changed the tensor set.")
    for name, tensor in restored.items():
        if tensor_hash(tensor) != expected_hashes[name]:
            raise AssertionError(f"Orbax read-back mismatch: {name}")
        if name not in PROJECTIONS and expected_hashes[name] != source_hashes[name]:
            raise AssertionError(f"A non-projection tensor changed: {name}")
    assert np.array_equal(restored["action_in_proj/kernel"][:32], originals["action_in_proj/kernel"])
    assert np.array_equal(restored["action_out_proj/kernel"][:, :32], originals["action_out_proj/kernel"])
    assert np.array_equal(restored["action_out_proj/bias"][:32], originals["action_out_proj/bias"])

    audit = {
        "state": "verified",
        "source": str(source),
        "output": str(output),
        "seed": seed,
        "source_action_dim": 32,
        "target_action_dim": 64,
        "initialization": "preserve first 32; new kernel LeCun-normal, new bias zero",
        "unchanged_hashes": {name: value for name, value in source_hashes.items() if name not in PROJECTIONS},
        "changed_tensors": {
            name: {
                "source_shape": list(originals[name].shape),
                "output_shape": list(restored[name].shape),
                "source_sha256": source_hashes[name],
                "output_sha256": expected_hashes[name],
            }
            for name in PROJECTIONS
        },
    }
    output.with_name(output.name + ".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="Released 32D params directory (local path or gs:// URL).")
    parser.add_argument("output", type=Path, help="New params directory to create.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    audit = convert(args.source, args.output, args.seed)
    print(json.dumps({key: audit[key] for key in ("state", "source", "output", "seed")}, indent=2))


if __name__ == "__main__":
    main()
