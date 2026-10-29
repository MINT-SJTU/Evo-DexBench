from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_policy(checkpoint: Path, device: str):
    from dp_policy.upstream import configure_upstream

    configure_upstream()

    import dill
    import hydra
    import torch

    from omegaconf import OmegaConf

    payload = torch.load(
        checkpoint.open("rb"),
        pickle_module=dill,
        map_location="cpu",
        weights_only=False,
    )
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if policy is None:
        raise RuntimeError("Checkpoint requested EMA but contains no EMA model.")
    policy.eval().to(device)
    metadata = OmegaConf.to_container(cfg.evodex, resolve=True)
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint evodex metadata must resolve to a mapping.")
    return policy, metadata


class DPInferenceServer:
    def __init__(self, checkpoint: Path, device: str) -> None:
        import msgpack
        import msgpack_numpy
        import torch

        msgpack_numpy.patch()
        self.msgpack = msgpack
        self.torch = torch
        self.device = torch.device(device)
        self.policy, self.metadata = _load_policy(checkpoint, device)

    def _predict(self, request: Mapping[str, Any]) -> dict[str, Any]:
        for key in ("variant", "embodiment"):
            if request.get(key) != self.metadata[key]:
                raise ValueError(
                    f"Request {key}={request.get(key)!r} does not match "
                    f"checkpoint {self.metadata[key]!r}."
                )
        raw_obs = request.get("observations")
        if not isinstance(raw_obs, Mapping):
            raise TypeError("observations must be a mapping.")
        expected_keys = set(self.metadata["observation_keys"])
        if set(raw_obs) != expected_keys:
            raise ValueError(
                f"Observation keys {set(raw_obs)} do not match {expected_keys}."
            )
        obs = {
            key: self.torch.from_numpy(np.array(value, copy=True)).to(
                self.device, dtype=self.torch.float32
            )
            for key, value in raw_obs.items()
        }
        with self.torch.inference_mode():
            result = self.policy.predict_action(obs)
        actions = result["action"].detach().cpu().numpy().astype(np.float32)
        expected = (
            next(iter(obs.values())).shape[0],
            int(self.metadata["available_action_steps"]),
            int(self.metadata["action_dim"]),
        )
        if actions.shape != expected:
            raise ValueError(f"Checkpoint returned {actions.shape}, expected {expected}.")
        return {"actions": actions, "metadata": self.metadata}

    async def handler(self, websocket) -> None:
        async for payload in websocket:
            try:
                request = self.msgpack.unpackb(payload, raw=False)
                response = self._predict(request)
            except Exception as exc:  # return a useful remote boundary error
                response = {"error": f"{type(exc).__name__}: {exc}"}
            await websocket.send(self.msgpack.packb(response, use_bin_type=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve an Evo-DexBench DP checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--upstream-root", type=Path, default=None)
    args = parser.parse_args()
    from dp_policy.upstream import configure_upstream

    configure_upstream(args.upstream_root)
    server = DPInferenceServer(args.checkpoint, args.device)

    async def run() -> None:
        from websockets.asyncio.server import serve

        async with serve(
            server.handler,
            args.host,
            args.port,
            max_size=None,
            compression=None,
        ):
            print(json.dumps(server.metadata, indent=2))
            await asyncio.Future()

    asyncio.run(run())


if __name__ == "__main__":
    main()
