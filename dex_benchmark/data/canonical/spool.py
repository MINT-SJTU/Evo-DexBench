"""Crash-tolerant local chunk spool for canonical HDF5 episodes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from dex_benchmark.data.canonical.hdf5 import (
    assemble_episode_hdf5,
    write_chunk_hdf5,
)
from dex_benchmark.data.canonical.schema import (
    CANONICAL_SCHEMA_VERSION,
    CanonicalTransition,
    jsonable,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as file_handle:
        os.fsync(file_handle.fileno())


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(jsonable(value), ensure_ascii=True, allow_nan=False, indent=2)
    with temporary.open("w", encoding="utf-8") as file_handle:
        file_handle.write(encoded)
        file_handle.write("\n")
        file_handle.flush()
        os.fsync(file_handle.fileno())
    temporary.replace(path)


class ChunkSpool:
    """Persist immutable chunk files and a resumable JSON manifest."""

    def __init__(
        self,
        root: str | Path,
        episode_id: str,
        *,
        chunk_length: int = 256,
        root_is_episode_dir: bool = False,
    ) -> None:
        """Create a spool directory.

        Legacy callers pass a collection root and rely on the historical
        ``root / episode_id`` layout.  Attempt-scoped callers pass the final
        attempt directory and set ``root_is_episode_dir=True`` so the episode
        identifier is not appended a second time.
        """
        if chunk_length <= 0:
            raise ValueError("chunk_length must be positive.")
        base_root = Path(root).expanduser().resolve()
        self.root = (
            base_root
            if root_is_episode_dir
            else base_root / str(episode_id)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.episode_id = str(episode_id)
        self.chunk_length = int(chunk_length)
        self.manifest_path = self.root / "manifest.json"
        self._manifest = self._load_manifest()
        self._reconcile_orphan_temps()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            manifest = {
                "schema_version": CANONICAL_SCHEMA_VERSION,
                "episode_id": self.episode_id,
                "chunk_length": self.chunk_length,
                "status": "open",
                "created_time_s": time.time(),
                "updated_time_s": time.time(),
                "chunks": [],
            }
            _atomic_json_write(self.manifest_path, manifest)
            return manifest
        with self.manifest_path.open("r", encoding="utf-8") as file_handle:
            manifest = json.load(file_handle)
        if manifest.get("episode_id") != self.episode_id:
            raise ValueError("Spool manifest episode_id does not match requested episode.")
        if int(manifest.get("schema_version", -1)) != CANONICAL_SCHEMA_VERSION:
            raise ValueError("Unsupported canonical spool schema version.")
        return manifest

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._manifest))

    @property
    def chunk_paths(self) -> list[Path]:
        return [self.root / str(item["path"]) for item in self._manifest["chunks"]]

    @property
    def checksums(self) -> list[str]:
        return [str(item["sha256"]) for item in self._manifest["chunks"]]

    def _reconcile_orphan_temps(self) -> None:
        """Promote a fully written but unmanifested chunk after a crash."""

        temporary_paths = sorted(self.root.glob("chunk-*.h5.tmp"))
        if not temporary_paths:
            return
        try:
            import h5py
        except ImportError:
            return
        changed = False
        known_paths = {str(item["path"]) for item in self._manifest["chunks"]}
        for temporary in temporary_paths:
            final_name = temporary.name.removesuffix(".tmp")
            if final_name in known_paths:
                continue
            try:
                with h5py.File(temporary, "r") as handle:
                    if "transitions/transition_id" not in handle:
                        continue
                    transition_ids = handle["transitions/transition_id"][:]
                    if len(transition_ids) == 0:
                        continue
                    chunk_index = int(handle.attrs["chunk_index"])
            except Exception:
                continue
            final_path = self.root / final_name
            if final_path.exists():
                continue
            _fsync_file(temporary)
            temporary.replace(final_path)
            self._manifest["chunks"].append(
                {
                    "index": chunk_index,
                    "path": final_name,
                    "sha256": sha256_file(final_path),
                    "bytes": final_path.stat().st_size,
                    "first_transition_id": int(transition_ids[0]),
                    "last_transition_id": int(transition_ids[-1]),
                    "committed_time_s": time.time(),
                    "recovered": True,
                }
            )
            changed = True
        if changed:
            self._manifest["chunks"].sort(key=lambda item: int(item["index"]))
            self._manifest["updated_time_s"] = time.time()
            _atomic_json_write(self.manifest_path, self._manifest)

    def recover_references(self) -> tuple[set[int], set[int]]:
        """Scan committed chunks so a restarted writer can deduplicate refs."""

        observation_ids: set[int] = set()
        quest_sequences: set[int] = set()
        for chunk_path in self.chunk_paths:
            if not chunk_path.exists():
                raise FileNotFoundError(f"Spool manifest references missing chunk {chunk_path}.")
            expected = next(
                item["sha256"]
                for item in self._manifest["chunks"]
                if self.root / str(item["path"]) == chunk_path
            )
            actual = sha256_file(chunk_path)
            if actual != expected:
                raise ValueError(
                    f"Checksum mismatch for spool chunk {chunk_path}: "
                    f"expected {expected}, got {actual}."
                )
            try:
                import h5py
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Chunk recovery requires h5py.") from exc
            with h5py.File(chunk_path, "r") as handle:
                if "observations/observation_id" in handle:
                    observation_ids.update(
                        int(value) for value in handle["observations/observation_id"][...]
                    )
                if "diagnostics/quest/sequence" in handle:
                    quest_sequences.update(
                        int(value) for value in handle["diagnostics/quest/sequence"][...]
                    )
        return observation_ids, quest_sequences

    def append_chunk(
        self,
        transitions: list[CanonicalTransition],
        *,
        known_observation_ids: set[int],
        known_quest_sequences: set[int],
    ) -> tuple[set[int], set[int], str, int]:
        if not transitions:
            raise ValueError("Cannot spool an empty chunk.")
        chunk_index = len(self._manifest["chunks"])
        filename = f"chunk-{chunk_index:06d}.h5"
        temporary = self.root / (filename + ".tmp")
        final_path = self.root / filename
        if temporary.exists():
            temporary.unlink()
        if final_path.exists():
            raise FileExistsError(f"Spool chunk already exists: {final_path}")
        new_observations, new_quest = write_chunk_hdf5(
            temporary,
            transitions,
            chunk_index=chunk_index,
            known_observation_ids=known_observation_ids,
            known_quest_sequences=known_quest_sequences,
        )
        # The worker knows the actual commit boundary; stamp it before fsync so
        # the transition timestamp survives crashes and checksum validation.
        try:
            import h5py
            with h5py.File(temporary, "r+") as handle:
                if "transitions/timestamps/disk_commit_s" in handle:
                    handle["transitions/timestamps/disk_commit_s"][:] = time.time()
        except ImportError as exc:  # pragma: no cover - h5py is required here.
            raise ImportError("Canonical spool writing requires h5py.") from exc
        _fsync_file(temporary)
        temporary.replace(final_path)
        checksum = sha256_file(final_path)
        record = {
            "index": chunk_index,
            "path": filename,
            "sha256": checksum,
            "bytes": final_path.stat().st_size,
            "first_transition_id": transitions[0].transition_id,
            "last_transition_id": transitions[-1].transition_id,
            "committed_time_s": time.time(),
        }
        self._manifest["chunks"].append(record)
        self._manifest["updated_time_s"] = time.time()
        _atomic_json_write(self.manifest_path, self._manifest)
        return new_observations, new_quest, checksum, int(record["bytes"])

    def finalize(
        self,
        output_path: str | Path,
        *,
        metadata: Mapping[str, Any],
        status: str = "completed",
    ) -> Path | None:
        self._manifest["status"] = str(status)
        self._manifest["updated_time_s"] = time.time()
        self._manifest["metadata"] = dict(metadata)
        _atomic_json_write(self.manifest_path, self._manifest)
        if not self.chunk_paths:
            return None
        final_metadata = dict(metadata)
        final_metadata["episode_status"] = str(status)
        output = assemble_episode_hdf5(
            output_path,
            self.chunk_paths,
            metadata=final_metadata,
            checksums=self.checksums,
        )
        self._manifest["output_path"] = str(output)
        self._manifest["output_sha256"] = sha256_file(output)
        self._manifest["updated_time_s"] = time.time()
        _atomic_json_write(self.manifest_path, self._manifest)
        return output

    def mark_status(self, status: str, **fields: Any) -> None:
        self._manifest["status"] = str(status)
        self._manifest.update(fields)
        self._manifest["updated_time_s"] = time.time()
        _atomic_json_write(self.manifest_path, self._manifest)
