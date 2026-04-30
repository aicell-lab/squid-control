"""
Snapshot management utilities for microscope image capture.

Snapshots are uploaded via the Artifact Manager HTTP API (stateless PUT),
not via RPC/WebSocket.  A single dataset per microscope (no date suffix)
is created once at startup via RPC; all subsequent uploads bypass RPC.

Images carry metadata embedded as PNG tEXt chunks (Pillow PngInfo).
"""

import asyncio
import json
import logging
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

import httpx
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .artifact_manager.artifact_manager import SquidArtifactManager

logger = logging.getLogger(__name__)

_UPLOAD_TIMEOUT_S = 10.0
_MAX_RETRIES = 2
_RETRY_BACKOFF_S = 2.0


def _make_filename(timestamp: str) -> str:
    """Build a filename from timestamp and a short random suffix.

    Example: 20260430_213556_a1b2c3.png
    """
    short_id = uuid.uuid4().hex[:6]
    return f"{timestamp}_{short_id}.png"


def _embed_png_metadata(png_bytes: bytes, metadata: Dict) -> bytes:
    """Embed metadata key-value pairs into PNG tEXt chunks via Pillow."""
    img = Image.open(BytesIO(png_bytes))
    png_info = PngInfo()
    for key, value in metadata.items():
        png_info.add_text(key, str(value))
    buf = BytesIO()
    img.save(buf, format="PNG", pnginfo=png_info)
    return buf.getvalue()


class SnapshotManager:
    """Manages microscope snapshot storage via the Artifact Manager HTTP API.

    On startup, ensures a shared gallery and a per-microscope dataset exist
    (via RPC), then disconnects from RPC.  All uploads use the stateless
    ``PUT /{workspace}/artifacts/{alias}/files/{path}`` HTTP endpoint.
    """

    def __init__(
        self,
        artifact_manager: SquidArtifactManager,
        buffer_dir: Optional[str] = None,
    ):
        self.artifact_manager = artifact_manager
        self.workspace = "reef-imaging"
        self._server_url = "https://hypha.aicell.io"

        self._dataset_alias: Optional[str] = None
        self._initialized = False

        self.buffer_dir = Path(buffer_dir or "/tmp/snapshot-buffer")
        self.buffer_dir.mkdir(parents=True, exist_ok=True)

        self._flush_task: Optional[asyncio.Task] = None
        self._stopping = False

    # ------------------------------------------------------------------
    # Startup -- RPC only, then disconnect
    # ------------------------------------------------------------------

    async def initialize(self, microscope_service_id: str) -> None:
        """Ensure gallery + per-microscope dataset exist via RPC."""
        if self._initialized:
            return

        await self._ensure_gallery()
        await self._ensure_microscope_dataset(microscope_service_id)

        self._initialized = True
        logger.info(
            "SnapshotManager initialized (dataset=%s, HTTP file upload + RPC metadata)",
            self._dataset_alias,
        )

    async def _ensure_gallery(self) -> None:
        svc = self.artifact_manager._svc
        gallery_alias = f"{self.workspace}/microscope-snapshots"
        try:
            await svc.read(artifact_id=gallery_alias)
            logger.info("Found existing snapshots gallery")
            return
        except Exception as e:
            err = str(e).lower()
            if not any(
                phrase in err
                for phrase in (
                    "not found",
                    "does not exist",
                    "keyerror",
                    "artifact with id",
                )
            ):
                raise

        logger.info("Creating snapshots gallery: microscope-snapshots")
        await svc.create(
            alias=gallery_alias,
            type="collection",
            manifest={
                "name": "Microscope Snapshots Gallery",
                "description": "Shared gallery for per-microscope snapshot datasets",
                "created_by": "squid-control-system",
                "type": "snapshot-gallery",
            },
            config={
                "permissions": {"*": "r", "@": "r+"},
            },
        )

    async def _ensure_microscope_dataset(self, microscope_service_id: str) -> None:
        svc = self.artifact_manager._svc
        alias = self._sanitize_service_id(microscope_service_id)
        self._dataset_alias = f"snapshots-{alias}"
        full_id = f"{self.workspace}/{self._dataset_alias}"

        try:
            await svc.read(artifact_id=full_id)
            logger.info("Found existing dataset: %s", self._dataset_alias)
            return
        except Exception as e:
            err = str(e).lower()
            if not any(
                phrase in err
                for phrase in (
                    "not found",
                    "does not exist",
                    "keyerror",
                    "artifact with id",
                )
            ):
                raise

        logger.info("Creating snapshot dataset: %s", self._dataset_alias)
        gallery_full = f"{self.workspace}/microscope-snapshots"
        await svc.create(
            parent_id=gallery_full,
            alias=full_id,
            manifest={
                "name": self._dataset_alias,
                "description": f"Snapshots for microscope {microscope_service_id}",
                "microscope_service_id": microscope_service_id,
                "created_by": "squid-control-system",
            },
            stage=False,
        )

    @staticmethod
    def _sanitize_service_id(service_id: str) -> str:
        """Lowercase and replace underscores so the alias is URL-safe."""
        return service_id.lower().replace("_", "-")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_snapshot(
        self,
        microscope_service_id: str,
        image_bytes: bytes,
        metadata: Dict,
    ) -> str:
        """Save a snapshot: embed metadata, write locally, upload via HTTP.

        Returns a public artifact-manager file URL on success.
        Raises only if all retries are exhausted.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        filename = _make_filename(timestamp)

        # Embed metadata into the PNG itself
        image_bytes = _embed_png_metadata(image_bytes, metadata)

        # 1) Persist to local disk immediately
        local_path = self.buffer_dir / filename
        local_path.write_bytes(image_bytes)
        meta_path = self.buffer_dir / f"{filename}.meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "microscope_service_id": microscope_service_id,
                    "filename": filename,
                    "metadata": metadata,
                    "buffered_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.gmtime()
                    ),
                }
            )
        )
        logger.info("Snapshot saved locally: %s", local_path)

        # 2) Retry upload
        last_exc = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                url = await asyncio.wait_for(
                    self._upload_via_http(filename, image_bytes),
                    timeout=_UPLOAD_TIMEOUT_S,
                )
                local_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return url
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Upload attempt %d/%d failed: %s (%s)",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    type(exc).__name__,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_BACKOFF_S)

        raise Exception(
            f"Failed to upload snapshot after {_MAX_RETRIES} attempts: "
            f"{type(last_exc).__name__}: {last_exc}. "
            f"Image preserved at {local_path}"
        )

    @property
    def buffer_size(self) -> int:
        """Number of snapshots waiting in the local buffer."""
        return len(list(self.buffer_dir.glob("*.meta.json")))

    # ------------------------------------------------------------------
    # HTTP upload
    # ------------------------------------------------------------------

    async def _upload_via_http(self, filename: str, image_bytes: bytes) -> str:
        """PUT the file directly to the artifact manager HTTP endpoint."""
        if not self._dataset_alias:
            raise RuntimeError(
                "SnapshotManager not initialized -- call initialize() first"
            )

        token = self.artifact_manager.token
        if not token:
            raise RuntimeError("No artifact manager token available")

        url = (
            f"{self._server_url}/{self.workspace}/artifacts/"
            f"{self._dataset_alias}/files/{filename}"
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.put(
                url,
                content=image_bytes,
                params={"download_weight": 0.1},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "image/png",
                },
            )
            response.raise_for_status()

        logger.info(
            "Snapshot uploaded via HTTP: %s (%.2f KB)",
            filename,
            len(image_bytes) / 1024,
        )
        return url

    # ------------------------------------------------------------------
    # Crash recovery: flush orphaned local files
    # ------------------------------------------------------------------

    def start_flush_loop(self) -> None:
        """Start background task to re-upload files left from a crash."""
        if self._flush_task is not None and not self._flush_task.done():
            return
        self._stopping = False
        self._flush_task = asyncio.ensure_future(self._flush_loop())

    async def stop_flush_loop(self) -> None:
        self._stopping = True
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

    async def flush_buffer(self) -> int:
        """Upload orphaned local files left from a previous crash."""
        meta_files = sorted(self.buffer_dir.glob("*.meta.json"))
        if not meta_files:
            return 0

        remaining = 0
        for meta_path in meta_files:
            try:
                info = json.loads(meta_path.read_text())
            except Exception:
                logger.warning("Corrupt metadata file, removing: %s", meta_path)
                meta_path.unlink(missing_ok=True)
                continue

            image_path = self.buffer_dir / info["filename"]
            if not image_path.exists():
                meta_path.unlink(missing_ok=True)
                continue

            try:
                url = await asyncio.wait_for(
                    self._upload_via_http(
                        info["filename"], image_path.read_bytes()
                    ),
                    timeout=15.0,
                )
                logger.info("Flushed orphaned snapshot -> %s", url)
                image_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
            except Exception as exc:
                now = time.time()
                if (
                    not hasattr(self, "_last_flush_warning")
                    or now - self._last_flush_warning > 60
                ):
                    logger.warning(
                        "Flush failed for %s: %s (%s)",
                        info["filename"],
                        exc,
                        type(exc).__name__,
                    )
                    self._last_flush_warning = now
                remaining += 1

        return remaining

    async def _flush_loop(self) -> None:
        while not self._stopping:
            try:
                remaining = await self.flush_buffer()
                await asyncio.sleep(30 if remaining == 0 else 10)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("Flush loop error: %s", exc)
                await asyncio.sleep(10)

    async def cleanup_test_datasets(self):
        """Clean up test datasets from the snapshots gallery."""
        svc = self.artifact_manager._svc
        if svc is None:
            logger.warning(
                "Cannot clean up test datasets: no RPC connection"
            )
            return
        try:
            gallery = await svc.read(
                artifact_id=f"{self.workspace}/microscope-snapshots"
            )
            datasets = await svc.list(gallery["id"])
            test_datasets = [
                d
                for d in datasets
                if d.get("alias", "").startswith("snapshots-test")
            ]
            logger.info("Found %d test datasets to clean up", len(test_datasets))
            for dataset in test_datasets:
                try:
                    await svc.delete(
                        artifact_id=dataset["id"],
                        delete_files=True,
                        recursive=False,
                    )
                    logger.info("Deleted test dataset: %s", dataset.get("alias"))
                except Exception as e:
                    logger.warning(
                        "Failed to delete %s: %s", dataset.get("alias"), e
                    )
        except Exception as e:
            logger.warning("Error during test dataset cleanup: %s", e)
