"""Offline training runner used by the public KineDex example."""

from __future__ import annotations

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class OfflineImageRunner(BaseImageRunner):
    """Disable upstream rollouts; Evo-DexBench evaluates checkpoints separately."""

    def __init__(self, output_dir: str | None = None) -> None:
        self.output_dir = output_dir

    def run(self, policy) -> dict:
        del policy
        return {}
