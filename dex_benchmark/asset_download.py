"""Explicit, license-aware asset downloads for the public benchmark."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import tempfile
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

from dex_benchmark.assets import load_asset_manifest, verify_assets
from dex_benchmark.partnet_overrides import generate_partnet_overrides


OFFICIAL_PARTNET_REPO = "sapien-sim/PartNetMobility"
DEFAULT_ASSET_REPO = "MINT-SJTU/Evo-DexBench_Assets"
ASSET_REPO_ENV_VAR = "EVODEX_ASSET_REPO"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
HF_MIRROR_ENDPOINT_ENV_VAR = "EVODEX_HF_MIRROR_ENDPOINT"
HF_ENDPOINT_PROBE_TIMEOUT_SECONDS = 10.0


def required_partnet_model_ids() -> tuple[str, ...]:
    for collection in load_asset_manifest()["collections"]:
        if collection["id"] == "partnet_mobility_originals":
            return tuple(str(value) for value in collection["required_models"])
    raise ValueError("Asset manifest has no PartNet-Mobility collection.")


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise ValueError(
                    f"Unsafe path {member.filename!r} in {archive.name}."
                )
        handle.extractall(destination)


def _official_model_root(staging: Path, model_id: str) -> Path:
    """Return the model root for either supported official archive layout."""

    candidates = (staging, staging / model_id)
    for candidate in candidates:
        if (candidate / "mobility.urdf").is_file():
            return candidate
    raise ValueError(
        f"Official archive {model_id}.zip has no mobility.urdf at its root "
        f"or under {model_id}/."
    )


def _probe_dataset_endpoint(endpoint: str, repo_id: str) -> None:
    url = f"{endpoint}/api/datasets/{quote(repo_id, safe='/')}"
    request = Request(url, headers={"User-Agent": "evodexbench-asset-probe"})
    with urlopen(request, timeout=HF_ENDPOINT_PROBE_TIMEOUT_SECONDS):
        return


def resolve_hf_endpoint(
    *,
    probe_repo: str,
    primary_endpoint: str | None = None,
    mirror_endpoint: str | None = None,
    allow_mirror: bool = True,
) -> tuple[str, bool]:
    """Select a reachable Hub endpoint without hiding repository HTTP errors."""

    primary = (
        primary_endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    ).rstrip("/")
    mirror = (
        mirror_endpoint
        or os.environ.get(HF_MIRROR_ENDPOINT_ENV_VAR)
        or DEFAULT_HF_MIRROR_ENDPOINT
    ).rstrip("/")

    try:
        _probe_dataset_endpoint(primary, probe_repo)
        return primary, False
    except HTTPError:
        # Authentication, permission and missing-repository failures must stay
        # visible instead of being misreported as a connectivity fallback.
        raise
    except Exception as primary_error:
        if not allow_mirror or mirror == primary:
            raise RuntimeError(
                f"Cannot reach Hugging Face endpoint {primary}: {primary_error}"
            ) from primary_error
        try:
            _probe_dataset_endpoint(mirror, probe_repo)
        except Exception as mirror_error:
            raise RuntimeError(
                "Neither the Hugging Face primary endpoint nor its configured "
                f"mirror is reachable. Primary: {primary_error}; "
                f"mirror {mirror}: {mirror_error}"
            ) from mirror_error
        return mirror, True


def download_public_bundle(
    asset_root: Path,
    *,
    repo_id: str,
    revision: str | None = None,
    endpoint: str | None = None,
) -> str:
    """Download the project-owned, non-PartNet bundle into its runtime root."""

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=asset_root,
        endpoint=endpoint,
    )


def download_official_partnet(
    asset_root: Path,
    *,
    repo_id: str = OFFICIAL_PARTNET_REPO,
    force: bool = False,
    endpoint: str | None = None,
) -> tuple[str, ...]:
    """Download only benchmark-required models from the user's gated access."""

    from huggingface_hub import hf_hub_download

    official_root = asset_root / "partnet_mobility"
    downloaded: list[str] = []
    with tempfile.TemporaryDirectory(prefix="evodex-partnet-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for model_id in required_partnet_model_ids():
            destination = official_root / model_id
            if not force and (destination / "mobility.urdf").is_file():
                continue
            archive = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=f"{model_id}.zip",
                    token=True,
                    local_dir=temporary_root,
                    force_download=force,
                    endpoint=endpoint,
                )
            )
            staging = temporary_root / f"extract-{model_id}"
            _safe_extract_zip(archive, staging)
            model_root = _official_model_root(staging, model_id)
            if destination.exists():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(model_root), str(destination))
            downloaded.append(model_id)
    generate_partnet_overrides(
        official_root,
        asset_root / "partnet_mobility_modified",
    )
    return tuple(downloaded)


def write_asset_environment(asset_root: Path) -> Path:
    """Write an optional shell fragment with every runtime asset path."""

    output = asset_root / "activate_evodex_assets.sh"
    values = {
        "EVODEX_ASSET_DIR": asset_root,
        "MS_ASSET_DIR": asset_root / "maniskill",
        "PARTNET_MOBILITY_DATASET": asset_root / "partnet_mobility",
        "PARTNET_MOBILITY_MODIFIED_DATASET": asset_root
        / "partnet_mobility_modified",
        "YCB_DATASET": asset_root / "ycb",
    }
    output.write_text(
        "# Source this file before running Evo-DexBench.\n"
        + "\n".join(
            f"export {name}={shlex.quote(str(path))}"
            for name, path in values.items()
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def download_assets(
    asset_dir: str | Path,
    *,
    bundle_repo: str | None,
    bundle_revision: str | None = None,
    include_bundle: bool = True,
    include_partnet: bool = True,
    force_partnet: bool = False,
    hf_endpoint: str | None = None,
    hf_mirror_endpoint: str | None = None,
    allow_hf_mirror: bool = True,
    allow_token_to_mirror: bool = False,
) -> dict[str, Any]:
    """Install both asset sources and return a verification report."""

    asset_root = Path(asset_dir).expanduser().resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    resolved_bundle_repo = (
        bundle_repo or os.environ.get(ASSET_REPO_ENV_VAR) or DEFAULT_ASSET_REPO
    )
    probe_repo = resolved_bundle_repo if include_bundle else OFFICIAL_PARTNET_REPO
    endpoint, fallback_used = resolve_hf_endpoint(
        probe_repo=probe_repo,
        primary_endpoint=hf_endpoint,
        mirror_endpoint=hf_mirror_endpoint,
        allow_mirror=allow_hf_mirror,
    )
    if fallback_used and include_partnet and not allow_token_to_mirror:
        raise RuntimeError(
            "Hugging Face is unreachable and the configured mirror would receive "
            "your gated PartNet token. Re-run with --allow-token-to-mirror after "
            "reviewing and trusting that mirror, or configure a trusted endpoint."
        )
    if include_bundle:
        download_public_bundle(
            asset_root,
            repo_id=resolved_bundle_repo,
            revision=bundle_revision,
            endpoint=endpoint,
        )
    downloaded_partnet = ()
    if include_partnet:
        downloaded_partnet = download_official_partnet(
            asset_root,
            force=force_partnet,
            endpoint=endpoint,
        )
    environment_file = write_asset_environment(asset_root)
    return {
        "hf_endpoint": endpoint,
        "hf_mirror_fallback_used": fallback_used,
        "downloaded_partnet_models": list(downloaded_partnet),
        "environment_file": str(environment_file),
        "verification": verify_assets(asset_root),
    }
