"""Exact replay of recorded canonical episodes with a strict physics check.

"Exact" means: from the stored seed and layout, replaying the recorded native
actions (plus the arm re-anchors the teleoperation runner logged) reproduces
every recorded robot joint position bit for bit, at every step, for the whole
episode. Recorded ``success``/``stage_score`` labels are compared and reported
separately: they were computed by the scoring definitions active at collection
time, which may differ from the current task definitions.

Per-episode replay settings ("recipes") are looked up by the SHA-256 of the
HDF5 file. Recipes only describe how the episode was simulated at collection
time (CPU vendor numerics, settling semantics, inferred re-anchors); they never
modify recorded data or comparison thresholds.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from dex_benchmark.data.canonical.hdf5 import _require_h5py
from dex_benchmark.data.canonical.replay import (
    replay_canonical_episode,
    resolve_env_callable,
)

CpuNumerics = Literal["native", "amd"]
RECORDED_PINOCCHIO_VERSION = "2.7.0"
DEFAULT_RECIPE_MANIFESTS = tuple(
    sorted((Path(__file__).resolve().parents[2] / "resources" / "replay").glob("*.json"))
)


@dataclass(frozen=True)
class ReplayRecipe:
    """How a recorded episode must be simulated to be reproduced exactly."""

    cpu_numerics: CpuNumerics = "native"
    # CPU vendor whose numerics the recording reproduces ("intel", "amd" or "unknown").
    reference_cpu: str = "unknown"
    # False for episodes captured before settling zeroed residual velocities.
    settle_resets_velocity: bool = True
    # 1-based step -> roles re-anchored before that step. These events are
    # inferred (absent from the episode's own log) and are always reported.
    inferred_resets: tuple[tuple[int, tuple[str, ...]], ...] = ()
    expected: str = "unknown"
    note: str = ""
    source: str = "default"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], source: str) -> "ReplayRecipe":
        cpu_numerics = str(values.get("cpu_numerics", "native"))
        if cpu_numerics not in ("native", "amd"):
            raise ValueError(f"Unsupported cpu_numerics {cpu_numerics!r} in {source}.")
        settle = str(values.get("settle", "reset_velocity"))
        if settle not in ("reset_velocity", "keep_velocity"):
            raise ValueError(f"Unsupported settle mode {settle!r} in {source}.")
        resets = tuple(
            (int(item["step"]), tuple(str(role) for role in item["roles"]))
            for item in values.get("inferred_resets", ())
        )
        return cls(
            cpu_numerics=cpu_numerics,  # type: ignore[arg-type]
            reference_cpu=str(values.get("reference_cpu", "unknown")),
            settle_resets_velocity=settle == "reset_velocity",
            inferred_resets=resets,
            expected=str(values.get("expected", "unknown")),
            note=str(values.get("note", "")),
            source=source,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cpu_numerics": self.cpu_numerics,
            "reference_cpu": self.reference_cpu,
            "settle": "reset_velocity" if self.settle_resets_velocity else "keep_velocity",
            "inferred_resets": [{"step": step, "roles": list(roles)} for step, roles in self.inferred_resets],
            "expected": self.expected,
            "note": self.note,
            "source": self.source,
        }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 23), b""):
            digest.update(block)
    return digest.hexdigest()


def find_replay_recipe(sha256: str, manifests: tuple[Path, ...] = DEFAULT_RECIPE_MANIFESTS) -> ReplayRecipe | None:
    """Return the recipe registered for an HDF5 SHA-256, or ``None``."""

    for manifest in manifests:
        data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        entry = data.get("episodes", {}).get(sha256)
        if entry is not None:
            return ReplayRecipe.from_mapping(entry, f"{Path(manifest).name}:{sha256[:12]}")
    return None


def _meta(handle: Any, name: str) -> Any:
    key = f"meta/{name}.json"
    if key not in handle:
        return None
    value = handle[key][()]
    return json.loads(value.decode("utf-8") if isinstance(value, bytes) else str(value))


def _first(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1)[0]


def replay_episode_exactly(
    episode_path: str | Path,
    *,
    recipe: ReplayRecipe | None = None,
    cpu_numerics: Literal["auto", "native", "amd"] = "auto",
    render_backend: str = "gpu",
    sim_freq: int = 120,
) -> dict[str, Any]:
    """Replay one episode and return a JSON-serializable verification report."""

    path = Path(episode_path).expanduser().resolve()
    stat_before = path.stat()
    sha256 = file_sha256(path)
    if recipe is None:
        recipe = find_replay_recipe(sha256) or ReplayRecipe()
    numerics: str = recipe.cpu_numerics if cpu_numerics == "auto" else cpu_numerics
    report: dict[str, Any] = {
        "episode": str(path),
        "sha256": sha256,
        "recipe": recipe.as_dict(),
        "cpu_numerics": numerics,
        "physics_exact": False,
        "labels": {},
    }

    from dex_benchmark.sim.determinism import host_cpu_vendor

    host_vendor = host_cpu_vendor()
    report["host_cpu_vendor"] = host_vendor
    if numerics == "native" and recipe.reference_cpu == "intel" and host_vendor != "GenuineIntel":
        report["warning"] = "This episode reproduces Intel CPU numerics; exact replay requires an Intel host."
    ik_backend = _ik_backend()
    report["ik_backend"] = ik_backend
    if ik_backend != f"pinocchio {RECORDED_PINOCCHIO_VERSION}":
        report["ik_backend_warning"] = (
            f"Episodes were recorded with SAPIEN IK backed by pinocchio {RECORDED_PINOCCHIO_VERSION} "
            f"(pip package pin=={RECORDED_PINOCCHIO_VERSION}); this process uses {ik_backend}, so the "
            "arm controller targets cannot match the recording."
        )
    numerics_report: dict[str, Any] | None = None
    if numerics == "amd":
        from dex_benchmark.sim.determinism import install_cpu_numerics, prepare_cpu_numerics

        prepare_cpu_numerics("amd")
        numerics_report = install_cpu_numerics("amd").as_dict()
    report["cpu_numerics_install"] = numerics_report

    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        reset_info = _meta(handle, "reset_info") or {}
        budget = _meta(handle, "episode_budget") or {}
        seed = _meta(handle, "seed")
        layout_id = _meta(handle, "layout_id")
        native_roles = list(handle["actions/native"].keys())
        roles = list(reset_info.get("manipulator_roles", [])) if native_roles == ["value"] else native_roles
        if native_roles == ["value"] and len(roles) != 1:
            raise ValueError("Scalar native actions require exactly one manipulator role.")
        recorded_qpos = {role: np.asarray(handle[f"observations/state/{role}.joint_position"]) for role in roles}
        observation_ids = np.asarray(handle["observations/observation_id"])
        next_ids = np.asarray(handle["transitions/observation_id_tp1"])
        control_hz = int(handle.attrs.get("control_hz", reset_info.get("control_frequency_hz", 30)))
        action_count = int(next_ids.shape[0])
    index_of = {int(value): index for index, value in enumerate(observation_ids)}
    if seed is None:
        raise ValueError("Exact replay needs the reset seed stored in meta/seed.json.")

    from dex_benchmark.sim.scenes.layouts.builder import settle_velocity_reset
    from dex_benchmark.sim.tasks import make_task_env
    from dex_benchmark.sim.tasks.runtime import ManipulatorRole

    env_kwargs: dict[str, Any] = {
        "layout_id": layout_id,
        "obs_mode": "state_dict",
        "sim_backend": "cpu",
        "render_backend": render_backend,
        "control_mode": reset_info["control_mode"],
        "max_episode_steps": max(int(budget.get("max_steps", action_count)), action_count),
        "sim_config": {"sim_freq": sim_freq, "control_freq": control_hz},
    }
    for key in ("generalization_axis", "generalization_seed", "generalization_profile",
                "generalization_attempt", "generalization_sample_index"):
        if reset_info.get(key) is not None:
            env_kwargs[key] = reset_info[key]

    first_difference: int | None = None
    max_error = 0.0
    success_mismatch: list[int] = []
    stage_mismatch: list[int] = []
    first_replay_success: int | None = None

    def check_step(env: Any, step: int, recorded_info: Mapping[str, Any] | None, info: Mapping[str, Any]) -> None:
        nonlocal first_difference, max_error, first_replay_success
        agent_for_role = resolve_env_callable(env, "agent_for_role")
        error = 0.0
        for role in roles:
            live = agent_for_role(ManipulatorRole(role)).robot.get_qpos()
            live_array = live.detach().cpu().numpy() if hasattr(live, "detach") else np.asarray(live)
            expected = recorded_qpos[role][index_of[int(next_ids[step - 1])]]
            # [1, J] live vs [J] recorded -> exact element-wise comparison.
            error = max(error, float(np.max(np.abs(live_array.reshape(-1) - expected.reshape(-1)))))
        if error != 0.0 and first_difference is None:
            first_difference = step
        max_error = max(max_error, error)
        if recorded_info is not None:
            replay_success = bool(_first(info.get("success", False)))
            if replay_success and first_replay_success is None:
                first_replay_success = step
            if replay_success != bool(_first(recorded_info.get("success", False))):
                success_mismatch.append(step)
            replay_stage = float(_first(info.get("stage_score", 0.0)))
            if abs(replay_stage - float(_first(recorded_info.get("stage_score", 0.0)))) > 1e-7:
                stage_mismatch.append(step)

    settle_context = settle_velocity_reset(False) if not recipe.settle_resets_velocity else nullcontext()
    env = None
    try:
        with settle_context:
            env = make_task_env(str(reset_info["task_id"]), **env_kwargs)
            result = replay_canonical_episode(
                env,
                path,
                seed=int(seed),
                apply_recorded_recovery=True,
                counterfactual_resets=dict(recipe.inferred_resets),
                stop_on_done=False,
                pass_layout_to_reset=False,
                keep_outputs=False,
                on_step=check_step,
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    steps_executed = len(result.terminated)
    stat_after = path.stat()
    report.update(
        steps_recorded=action_count,
        scene_physics={
            "layout_version": result.reset_info.get("layout_physics_version"),
            "settle_resets_velocity": recipe.settle_resets_velocity
            and result.reset_info.get("layout_settle_resets_velocity", True),
        },
        steps_executed=steps_executed,
        first_qpos_difference_step=first_difference,
        max_joint_error_rad=max_error,
        physics_exact=bool(steps_executed == action_count and first_difference is None),
        source_unchanged=(stat_before.st_size, stat_before.st_mtime_ns) == (stat_after.st_size, stat_after.st_mtime_ns),
        labels={
            "success_mismatch_steps": len(success_mismatch),
            "stage_mismatch_steps": len(stage_mismatch),
            "first_success_mismatch": success_mismatch[:3],
            "first_stage_mismatch": stage_mismatch[:3],
            "first_replay_success_step": first_replay_success,
            "match": not success_mismatch and not stage_mismatch,
            "note": "Recorded labels follow the scoring definitions active at collection time.",
        },
    )
    if numerics_report is not None and numerics_report.get("patched"):
        from dex_benchmark.sim.determinism import emulation_hit_counts

        report["cpu_numerics_emulated_instructions"] = emulation_hit_counts()
    report["runtime"] = _runtime_versions()
    return report


def _ik_backend() -> str:
    """Return the solver behind SAPIEN's PinocchioModel (used by CPU arm controllers)."""

    from sapien.wrapper import pinocchio_model

    external = getattr(pinocchio_model, "pinocchio", None)
    if external is None:
        return "sapien built-in"
    return f"pinocchio {getattr(external, '__version__', 'unknown')}"


def _runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"mkl_cbwr": os.environ.get("MKL_CBWR")}
    for module_name in ("torch", "sapien", "mani_skill", "numpy", "pinocchio"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        versions[module_name] = getattr(module, "__version__", "unknown")
    return versions
