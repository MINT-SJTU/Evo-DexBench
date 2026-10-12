"""Idempotent chunk upload hooks for canonical recorder backends."""

from __future__ import annotations

from collections.abc import Callable
import shutil
from pathlib import Path
import time
from typing import Any

from dex_benchmark.data.canonical.spool import ChunkSpool, sha256_file


UploadFunction = Callable[[Path, str], None]


class FilesystemChunkUploader:
    """Reference uploader that atomically mirrors chunks to another filesystem."""

    def __init__(self, destination: str | Path) -> None:
        self.destination = Path(destination).expanduser().resolve()

    def __call__(self, chunk_path: Path, checksum: str) -> None:
        target = self.destination / chunk_path.name
        self.destination.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256_file(target) == checksum:
            return
        temporary = target.with_name(target.name + ".tmp")
        shutil.copyfile(chunk_path, temporary)
        if sha256_file(temporary) != checksum:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Uploaded chunk checksum mismatch for {chunk_path}.")
        temporary.replace(target)


class ChunkUploader:
    """Retry an upload function while keeping progress in spool manifest."""

    def __init__(
        self,
        upload: UploadFunction,
        *,
        max_retries: int = 5,
        retry_backoff_s: float = 0.5,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if retry_backoff_s < 0.0:
            raise ValueError("retry_backoff_s must be non-negative.")
        self.upload = upload
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)

    def upload_spool(self, spool: ChunkSpool) -> dict[str, Any]:
        manifest = spool.manifest
        uploaded = {
            int(index)
            for index in manifest.get("uploaded_chunks", [])
        }
        retry_count = 0
        for record, path in zip(manifest["chunks"], spool.chunk_paths, strict=True):
            index = int(record["index"])
            if index in uploaded:
                continue
            checksum = str(record["sha256"])
            last_error: BaseException | None = None
            for attempt in range(self.max_retries + 1):
                started = time.perf_counter()
                try:
                    if sha256_file(path) != checksum:
                        raise ValueError(f"Local chunk checksum mismatch: {path}.")
                    self.upload(path, checksum)
                    uploaded.add(index)
                    break
                except BaseException as exc:
                    last_error = exc
                    retry_count += 1
                    if attempt >= self.max_retries:
                        raise RuntimeError(
                            f"Failed to upload canonical chunk {path} after "
                            f"{self.max_retries + 1} attempts."
                        ) from exc
                    time.sleep(self.retry_backoff_s * (2**attempt))
                finally:
                    del started
            if last_error is not None and index not in uploaded:
                raise RuntimeError(f"Chunk {index} was not uploaded.") from last_error
            spool.mark_status(
                "uploaded",
                uploaded_chunks=sorted(uploaded),
                upload_retries=retry_count,
                last_upload_commit_time_s=time.time(),
            )
        return {
            "uploaded_chunks": sorted(uploaded),
            "upload_retries": retry_count,
        }
