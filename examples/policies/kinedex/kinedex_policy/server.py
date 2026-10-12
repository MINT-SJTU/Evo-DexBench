from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_kinedex_checkpoint(
    checkpoint: Path,
    *,
    device: str,
    upstream_root: Path | None,
    action_steps: int | None = None,
):
    from kinedex_policy.upstream import configure_upstream

    configure_upstream(upstream_root)
    import dill
    import hydra
    import torch
    from omegaconf import OmegaConf

    payload = torch.load(
        checkpoint.open("rb"), pickle_module=dill, map_location="cpu", weights_only=False
    )
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if policy is None:
        raise RuntimeError("KineDex checkpoint requested a missing EMA policy.")
    policy.eval().to(device)
    metadata = OmegaConf.to_container(cfg.evodex, resolve=True)
    if not isinstance(metadata, dict) or metadata.get("policy_type") != "kinedex":
        raise ValueError("Checkpoint is not an Evo-DexBench KineDex policy.")
    metadata.setdefault("tactile_mode", "full")
    from kinedex_policy.upstream import EXPECTED_COMMIT

    if metadata.get("upstream_commit") != EXPECTED_COMMIT:
        raise ValueError("KineDex checkpoint upstream commit mismatch.")
    if action_steps is not None:
        max_action_steps = int(metadata["prediction_horizon"]) - int(
            metadata["observation_horizon"]
        ) + 1
        if not 1 <= action_steps <= max_action_steps:
            raise ValueError(
                f"action_steps must be in [1, {max_action_steps}], got {action_steps}."
            )
        policy.n_action_steps = action_steps
        metadata["available_action_steps"] = action_steps
    return policy, metadata


class KineDexInferenceServer:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        *,
        upstream_root: Path | None = None,
        action_steps: int | None = None,
    ) -> None:
        import msgpack
        import msgpack_numpy
        import torch

        msgpack_numpy.patch()
        self.msgpack = msgpack
        self.torch = torch
        self.device = torch.device(device)
        self.policy, self.metadata = load_kinedex_checkpoint(
            checkpoint,
            device=device,
            upstream_root=upstream_root,
            action_steps=action_steps,
        )

    def _predict(self, request: Mapping[str, Any]) -> dict[str, Any]:
        for key in ("policy_type", "embodiment", "tactile_mode"):
            if request.get(key) != self.metadata[key]:
                raise ValueError(f"KineDex request {key} does not match checkpoint.")
        raw = request.get("observations")
        if not isinstance(raw, Mapping):
            raise TypeError("KineDex observations must be a mapping.")
        expected_keys = set(self.metadata["observation_keys"])
        if set(raw) != expected_keys:
            raise ValueError(f"KineDex observation keys must be {expected_keys}.")
        tactile_mode = str(self.metadata.get("tactile_mode", "full"))
        if tactile_mode not in ("full", "none"):
            raise ValueError(f"Unsupported KineDex tactile mode {tactile_mode!r}.")
        if tactile_mode == "none":
            placeholder = np.asarray(raw.get("finger_raw_pressure"))
            if placeholder.size == 0 or not np.isfinite(placeholder).all():
                raise ValueError("NoTouch KineDex requires a finite tactile placeholder.")
            if np.count_nonzero(placeholder) != 0:
                raise ValueError("NoTouch KineDex tactile placeholder must be all zeros.")
        obs = {
            key: self.torch.from_numpy(np.ascontiguousarray(value).copy()).to(
                self.device, dtype=self.torch.float32, non_blocking=True
            )
            for key, value in raw.items()
        }
        with self.torch.inference_mode():
            result = self.policy.predict_action(obs)
        action = result["action"].detach().cpu().numpy().astype(np.float32)
        expected = (
            next(iter(obs.values())).shape[0],
            int(self.metadata["available_action_steps"]),
            int(self.metadata["model_action_dim"]),
        )
        if action.shape != expected or not np.isfinite(action).all():
            raise ValueError(f"KineDex returned invalid action {action.shape}.")
        if tactile_mode == "none":
            return {"motions": action, "metadata": self.metadata}
        from kinedex_policy.tactile import split_motion_force

        motion, force = split_motion_force(action, len(self.metadata["roles"]))
        return {"motions": motion, "force_targets": force, "metadata": self.metadata}

    async def handler(self, websocket: Any) -> None:
        async for payload in websocket:
            try:
                response = self._predict(self.msgpack.unpackb(payload, raw=False))
            except Exception as exc:
                response = {"error": f"{type(exc).__name__}: {exc}"}
            await websocket.send(self.msgpack.packb(response, use_bin_type=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a KineDex checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--action-steps",
        type=int,
        help="Override the number of predicted actions returned per request.",
    )
    args = parser.parse_args()
    server = KineDexInferenceServer(
        args.checkpoint.expanduser().resolve(),
        args.device,
        upstream_root=(
            args.upstream_root.expanduser().resolve()
            if args.upstream_root is not None
            else None
        ),
        action_steps=args.action_steps,
    )

    async def run() -> None:
        from websockets.asyncio.server import serve

        async with serve(server.handler, args.host, args.port, max_size=None, compression=None):
            print(json.dumps(server.metadata, indent=2))
            await asyncio.Future()

    asyncio.run(run())


if __name__ == "__main__":
    main()
