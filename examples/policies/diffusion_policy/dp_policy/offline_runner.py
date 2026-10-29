from __future__ import annotations

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class OfflineImageRunner(BaseImageRunner):
    """Keep official workspaces offline; Evo-DexBench evaluates separately."""

    def __init__(self, output_dir: str | None = None) -> None:
        self.output_dir = output_dir

    def run(self, policy) -> dict:
        return {}
