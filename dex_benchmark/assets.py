from __future__ import annotations

import os
import json
from importlib.resources import files
from pathlib import Path

import yaml


EVODEX_ASSET_DIR_ENV_VAR = "EVODEX_ASSET_DIR"
MS_ASSET_DIR_ENV_VAR = "MS_ASSET_DIR"
PARTNET_MOBILITY_DATASET_DIR_ENV_VAR = "PARTNET_MOBILITY_DATASET"
PARTNET_MOBILITY_MODIFIED_DATASET_DIR_ENV_VAR = (
    "PARTNET_MOBILITY_MODIFIED_DATASET"
)
YCB_DATASET_DIR_ENV_VAR = "YCB_DATASET"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_assets_dir() -> Path:
    return repo_root() / "assets"


def evodex_asset_dir() -> Path:
    configured_asset_dir = os.environ.get(EVODEX_ASSET_DIR_ENV_VAR)
    if configured_asset_dir:
        return Path(configured_asset_dir).expanduser().resolve()
    return repo_assets_dir().resolve()


def default_maniskill_data_dir() -> Path:
    """Return ManiSkill's data directory without mutating process environment."""
    configured_asset_root = os.environ.get(MS_ASSET_DIR_ENV_VAR)
    if configured_asset_root:
        return (
            Path(configured_asset_root).expanduser().resolve() / "data"
        )
    return (Path.home() / ".maniskill" / "data").resolve()


def default_partnet_mobility_dataset_dir() -> Path:
    configured_dataset_dir = os.environ.get(
        PARTNET_MOBILITY_DATASET_DIR_ENV_VAR
    )
    if configured_dataset_dir:
        return Path(configured_dataset_dir).expanduser().resolve()
    return evodex_asset_dir() / "partnet_mobility"


def default_partnet_mobility_modified_dataset_dir() -> Path:
    configured_dataset_dir = os.environ.get(
        PARTNET_MOBILITY_MODIFIED_DATASET_DIR_ENV_VAR
    )
    if configured_dataset_dir:
        return Path(configured_dataset_dir).expanduser().resolve()
    return evodex_asset_dir() / "partnet_mobility_modified"


def default_ycb_dataset_dir() -> Path:
    configured_dataset_dir = os.environ.get(YCB_DATASET_DIR_ENV_VAR)
    if configured_dataset_dir:
        return Path(configured_dataset_dir).expanduser().resolve()
    return evodex_asset_dir() / "ycb"


def default_robocasa_asset_dir() -> Path:
    return (
        default_maniskill_data_dir()
        / "scene_datasets"
        / "robocasa_dataset"
        / "assets"
    )


def default_ai2thor_asset_dir() -> Path:
    return default_maniskill_data_dir() / "scene_datasets" / "ai2thor"


def default_replicacad_asset_dir() -> Path:
    return (
        default_maniskill_data_dir()
        / "scene_datasets"
        / "replica_cad_dataset"
    )


def robocasa_asset_path(relative_path: str | Path) -> Path:
    return default_robocasa_asset_dir() / relative_path


def robocasa_texture_path(relative_path: str | Path) -> Path:
    return robocasa_asset_path(Path("textures") / relative_path)


def robocasa_asset_resolution_hint(relative_path: str | Path) -> str:
    checked_asset_path = robocasa_asset_path(relative_path)
    return (
        "Download RoboCasa with `python -m mani_skill.utils.download_asset "
        "RoboCasa`, "
        f"set {MS_ASSET_DIR_ENV_VAR} to the ManiSkill asset root, or package "
        "the selected asset in a project asset bundle. "
        f"Checked asset: {checked_asset_path}."
    )


def resolve_partnet_mobility_dataset_dir() -> Path:
    return default_partnet_mobility_dataset_dir().resolve()


def partnet_mobility_model_urdf_path(
    model_id: str,
    urdf_name: str = "mobility.urdf",
) -> Path:
    return resolve_partnet_mobility_dataset_dir() / str(model_id) / urdf_name


def partnet_mobility_model_textured_obj_path(
    model_id: str,
    obj_name: str = "textured.obj",
) -> Path:
    return (
        resolve_partnet_mobility_dataset_dir()
        / str(model_id)
        / "textured_objs"
        / obj_name
    )


def partnet_mobility_resolution_hint(
    model_id: str,
    urdf_name: str = "mobility.urdf",
) -> str:
    checked_urdf_path = partnet_mobility_model_urdf_path(
        model_id=model_id,
        urdf_name=urdf_name,
    )
    return (
        "Place the PartNet-Mobility dataset under "
        f"{default_partnet_mobility_dataset_dir()}, set "
        f"{PARTNET_MOBILITY_DATASET_DIR_ENV_VAR} or "
        f"{EVODEX_ASSET_DIR_ENV_VAR}, or pass a full URDF path to the task. "
        f"Checked URDF: {checked_urdf_path}."
    )


def resolve_partnet_mobility_modified_dataset_dir() -> Path:
    return default_partnet_mobility_modified_dataset_dir().resolve()


def partnet_mobility_modified_model_urdf_path(
    model_id: str,
    urdf_name: str = "mobility_collision_override.urdf",
) -> Path:
    return (
        resolve_partnet_mobility_modified_dataset_dir()
        / str(model_id)
        / urdf_name
    )


def partnet_mobility_modified_resolution_hint(
    model_id: str,
    urdf_name: str = "mobility_collision_override.urdf",
) -> str:
    checked_urdf_path = partnet_mobility_modified_model_urdf_path(
        model_id=model_id,
        urdf_name=urdf_name,
    )
    return (
        "Place the modified PartNet-Mobility dataset under "
        f"{default_partnet_mobility_modified_dataset_dir()}, set "
        f"{PARTNET_MOBILITY_MODIFIED_DATASET_DIR_ENV_VAR} or "
        f"{EVODEX_ASSET_DIR_ENV_VAR}. Checked URDF: {checked_urdf_path}."
    )


def resolve_ycb_dataset_dir() -> Path:
    return default_ycb_dataset_dir().resolve()


def ycb_model_obj_path(
    model_name: str,
    obj_name: str = "textured.obj",
) -> Path:
    return (
        resolve_ycb_dataset_dir()
        / str(model_name)
        / "google_16k"
        / obj_name
    )


def ycb_resolution_hint(
    model_name: str,
    obj_name: str = "textured.obj",
) -> str:
    checked_obj_path = ycb_model_obj_path(
        model_name=model_name,
        obj_name=obj_name,
    )
    return (
        "Place the repository YCB dataset under "
        f"{default_ycb_dataset_dir()}, set {YCB_DATASET_DIR_ENV_VAR} or "
        f"{EVODEX_ASSET_DIR_ENV_VAR}. Checked OBJ: {checked_obj_path}."
    )


def asset_manifest_path() -> Path:
    """Return the installed public asset manifest."""

    return Path(
        str(files("dex_benchmark.resources").joinpath("assets/manifest.yaml"))
    )


def load_asset_manifest() -> dict:
    with asset_manifest_path().open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict) or "collections" not in manifest:
        raise ValueError("The installed asset manifest is invalid.")
    return manifest


def verify_assets(asset_dir: str | Path | None = None) -> dict:
    """Check required public assets without downloading or mutating anything."""

    root = (
        Path(asset_dir).expanduser().resolve()
        if asset_dir is not None
        else evodex_asset_dir()
    )
    records: list[dict] = []
    for collection in load_asset_manifest()["collections"]:
        path = str(collection["path"])
        status = str(collection["status"])
        if path.startswith("package://"):
            package, _, relative_path = path.removeprefix("package://").partition("/")
            resolved_path = Path(str(files(package).joinpath(relative_path)))
            resolved = path
        else:
            resolved_path = root / path
            resolved = str(resolved_path)
        if collection.get("kind", "directory") == "file":
            present = resolved_path.is_file()
        else:
            # Explicit file requirements also work with symlinked model directories,
            # which Path.rglob intentionally does not descend into.
            has_file_requirements = bool(collection.get("required_files")) or bool(
                collection.get("required_models")
                and collection.get("required_model_files")
            )
            present = resolved_path.is_dir() and (
                has_file_requirements
                or any(item.is_file() for item in resolved_path.rglob("*"))
            )
        required_models = collection.get(
            "required_models", collection.get("missing_models", [])
        )
        missing_models = [
            model_id
            for model_id in required_models
            if not (resolved_path / model_id).is_dir()
            or not all(
                (resolved_path / model_id / filename).is_file()
                for filename in collection.get("required_model_files", [])
            )
        ]
        missing_files = [
            filename
            for filename in collection.get("required_files", [])
            if not (resolved_path / filename).is_file()
        ]
        if missing_models or missing_files:
            present = False
        records.append(
            {
                "id": collection["id"],
                "status": status,
                "present": present,
                "path": resolved,
                "missing_models": missing_models,
                "missing_files": missing_files,
            }
        )
    missing_required = [
        record["id"]
        for record in records
        if record["status"] in {
            "redistributable", "embedded", "generated", "pending"
        }
        and not record["present"]
    ]
    restricted_missing = [
        record["id"]
        for record in records
        if record["status"] == "restricted" and not record["present"]
    ]
    return {
        "asset_root": str(root),
        "ok": not missing_required,
        "missing_required": missing_required,
        "restricted_missing": restricted_missing,
        "collections": records,
    }


def format_asset_report(report: dict) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)
