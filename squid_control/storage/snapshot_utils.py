"""
Snapshot management utilities for microscope image capture using Artifact Manager.

This module provides the SnapshotManager class for storing microscope snapshots
in daily datasets with public URL access through the Hypha Artifact Manager.
Snapshots are first persisted to local disk as a safety net, then uploaded
to the artifact manager with automatic retries until the upload succeeds.
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import httpx

from .artifact_manager.artifact_manager import SquidArtifactManager

logger = logging.getLogger(__name__)


class SnapshotManager:
    """
    Manages microscope snapshot storage using Artifact Manager.

    Snapshots are organized into:
    - Gallery: "microscope-snapshots" (shared by all microscopes)
    - Datasets: "snapshots-{service_id}-{YYYY-MM-DD}" (one per day per microscope)
    - Files: "{uuid}.png" (UUID-based naming for uniqueness)

    All snapshots have public read access through the artifact manager.

    Images are first written to a local buffer directory, then uploaded with
    automatic retries until the artifact manager confirms the upload.
    """

    def __init__(
        self,
        artifact_manager: SquidArtifactManager,
        buffer_dir: Optional[str] = None,
    ):
        """
        Initialize the snapshot manager.

        Args:
            artifact_manager: An initialized SquidArtifactManager instance
            buffer_dir: Local directory for temporary image storage.
                        Defaults to /tmp/snapshot-buffer
        """
        self.artifact_manager = artifact_manager
        self.workspace = "reef-imaging"

        # Local directory for safety-net persistence before upload
        self.buffer_dir = Path(buffer_dir or "/tmp/snapshot-buffer")
        self.buffer_dir.mkdir(parents=True, exist_ok=True)

        # Background flush for crash recovery (orphaned local files)
        self._flush_task: Optional[asyncio.Task] = None
        self._stopping = False

    @property
    def _svc(self):
        """Always read the current artifact-manager proxy.

        Never caches a stale reference -- when artifact_manager.refresh_service()
        is called after a connection error, this property automatically picks up
        the fresh proxy.
        """
        return self.artifact_manager._svc

    async def get_or_create_snapshots_gallery(self, microscope_service_id: str) -> dict:
        """
        Get or create the shared snapshots gallery for all microscopes.
        
        Args:
            microscope_service_id: The hypha service ID of the microscope (used for logging)
            
        Returns:
            dict: The gallery artifact information
        """
        # Use a single shared gallery for all microscopes
        gallery_alias = "microscope-snapshots"

        try:
            # Try to get existing gallery
            gallery = await self._svc.read(artifact_id=f"{self.workspace}/{gallery_alias}")
            print(f"Found existing snapshots gallery: {gallery_alias}")
            return gallery
        except Exception as e:
            # Handle various error patterns for "not found"
            error_str = str(e).lower()
            if ("not found" in error_str or
                "does not exist" in error_str or
                "keyerror" in error_str or
                "artifact with id" in error_str):
                # Gallery doesn't exist, create it
                print(f"Creating new snapshots gallery: {gallery_alias}")

                gallery_manifest = {
                    "name": "Microscope Snapshots Gallery",
                    "description": "Shared gallery for daily snapshot collections from all microscopes",
                    "created_by": "squid-control-system",
                    "type": "snapshot-gallery"
                }

                gallery_config = {
                    "permissions": {"*": "r", "@": "r+"},  # Public read, authenticated can contribute
                    "collection_schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "record_type": {"type": "string", "enum": ["snapshot-dataset"]},
                            "microscope_service_id": {"type": "string"},
                            "date": {"type": "string"},
                            "snapshot_count": {"type": "integer"}
                        },
                        "required": ["name", "description", "record_type", "date"]
                    }
                }

                gallery = await self._svc.create(
                    alias=f"{self.workspace}/{gallery_alias}",
                    type="collection",
                    manifest=gallery_manifest,
                    config=gallery_config
                )
                print(f"Created snapshots gallery: {gallery_alias}")
                return gallery
            else:
                raise e

    async def get_or_create_daily_dataset(
        self,
        microscope_service_id: str,
        date_str: Optional[str] = None
    ) -> dict:
        """
        Get or create the snapshot dataset for a specific day.
        
        Args:
            microscope_service_id: The hypha service ID of the microscope
            date_str: Date string in YYYY-MM-DD format. Defaults to today's UTC date.
            
        Returns:
            dict: The dataset artifact information
        """
        # Generate date string if not provided
        if date_str is None:
            date_str = time.strftime("%Y-%m-%d", time.gmtime())

        dataset_name = f"snapshots-{microscope_service_id}-{date_str}"
        dataset_alias = f"{self.workspace}/{dataset_name}"

        try:
            # Try to get existing dataset
            dataset = await self._svc.read(artifact_id=dataset_alias)
            print(f"Found existing snapshot dataset: {dataset_name}")
            return dataset
        except Exception as e:
            # Handle various error patterns for "not found"
            error_str = str(e).lower()
            if ("not found" in error_str or
                "does not exist" in error_str or
                "keyerror" in error_str or
                "artifact with id" in error_str):
                # Dataset doesn't exist, create it
                print(f"Creating new snapshot dataset: {dataset_name}")

                # Ensure gallery exists
                gallery = await self.get_or_create_snapshots_gallery(microscope_service_id)

                # Create dataset manifest
                dataset_manifest = {
                    "name": dataset_name,
                    "description": f"Snapshots for {microscope_service_id} on {date_str}",
                    "record_type": "snapshot-dataset",
                    "microscope_service_id": microscope_service_id,
                    "date": date_str,
                    "snapshot_count": 0,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                }

                # Create dataset (not staged - we'll add files and edit/commit as needed)
                dataset = await self._svc.create(
                    parent_id=gallery["id"],
                    alias=dataset_alias,
                    manifest=dataset_manifest,
                    stage=False  # Create committed dataset that we can edit later
                )
                print(f"Created snapshot dataset: {dataset_name}")
                return dataset
            else:
                raise e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # RPC method timeout is ~30s; leave headroom for capture + encode.
    _UPLOAD_TIMEOUT_S = 10.0
    _MAX_RETRIES = 2
    _RETRY_BACKOFF_S = 2.0

    async def save_snapshot(
        self,
        microscope_service_id: str,
        image_bytes: bytes,
        metadata: Dict,
    ) -> str:
        """
        Save a snapshot: write locally first, then retry upload.

        Returns a real artifact manager URL on success.  Raises only if all
        retries are exhausted within the RPC call window (~25 s).  The image
        is always safe on disk and the background flush loop will upload it
        later if this call is interrupted by an RPC timeout.
        """
        filename = f"{uuid.uuid4()}.png"

        # 1) Persist to local disk immediately as a safety net
        local_path = self.buffer_dir / filename
        local_path.write_bytes(image_bytes)
        meta_path = self.buffer_dir / f"{filename}.meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "microscope_service_id": microscope_service_id,
                    "filename": filename,
                    "metadata": metadata,
                    "buffered_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                }
            )
        )
        logger.info("Snapshot saved locally: %s", local_path)

        # 2) Retry upload (bounded — must fit within RPC timeout window)
        last_exc = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                url = await asyncio.wait_for(
                    self._upload_to_artifact_manager(
                        microscope_service_id, filename, image_bytes, metadata
                    ),
                    timeout=self._UPLOAD_TIMEOUT_S,
                )
                # Success — clean up local copy
                local_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return url
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Upload attempt %d/%d failed: %s (%s)",
                    attempt,
                    self._MAX_RETRIES,
                    exc,
                    type(exc).__name__,
                )
                if attempt < self._MAX_RETRIES:
                    await asyncio.sleep(self._RETRY_BACKOFF_S)

        # All retries exhausted — the background flush loop will pick this up.
        raise Exception(
            f"Failed to upload snapshot after {self._MAX_RETRIES} attempts: "
            f"{type(last_exc).__name__}: {last_exc}. "
            f"Image is preserved locally at {local_path}"
        )

    @property
    def buffer_size(self) -> int:
        """Number of snapshots waiting in the local buffer."""
        return len(list(self.buffer_dir.glob("*.meta.json")))

    # ------------------------------------------------------------------
    # Remote upload
    # ------------------------------------------------------------------

    async def _upload_to_artifact_manager(
        self,
        microscope_service_id: str,
        filename: str,
        image_bytes: bytes,
        metadata: Dict,
        _is_retry: bool = False,
    ) -> str:
        """Upload a single snapshot via the artifact manager RPC."""
        dataset = await self.get_or_create_daily_dataset(microscope_service_id)

        await self._svc.edit(artifact_id=dataset["id"], stage=True)

        try:
            put_url = await self._svc.put_file(
                artifact_id=dataset["id"],
                file_path=filename,
                download_weight=0.1,
            )

            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                response = await client.put(put_url, content=image_bytes)
                response.raise_for_status()

            logger.info(
                "Uploaded snapshot: %s (%.2f KB)", filename, len(image_bytes) / 1024
            )

            current_manifest = dataset.get("manifest", {})
            snapshot_count = current_manifest.get("snapshot_count", 0) + 1
            if "snapshots" not in current_manifest:
                current_manifest["snapshots"] = {}
            current_manifest["snapshots"][filename] = metadata
            current_manifest["snapshot_count"] = snapshot_count
            current_manifest["last_updated"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime()
            )

            await self._svc.edit(
                artifact_id=dataset["id"], manifest=current_manifest, stage=True
            )
            await self._svc.commit(artifact_id=dataset["id"])

            dataset_alias = dataset.get("alias", "").replace(
                f"{self.workspace}/", ""
            )
            file_url = (
                f"https://hypha.aicell.io/{self.workspace}"
                f"/artifacts/{dataset_alias}/files/{filename}"
            )
            logger.info("Snapshot saved remotely: %s", file_url)
            return file_url

        except Exception as exc:
            try:
                await self._svc.discard(artifact_id=dataset["id"])
            except Exception:
                pass

            if not _is_retry and isinstance(
                exc, (asyncio.TimeoutError, ConnectionError, OSError)
            ):
                logger.warning(
                    "Artifact manager upload failed (%s), refreshing proxy and retrying...",
                    type(exc).__name__,
                )
                try:
                    await self.artifact_manager.refresh_service()
                    return await self._upload_to_artifact_manager(
                        microscope_service_id,
                        filename,
                        image_bytes,
                        metadata,
                        _is_retry=True,
                    )
                except Exception:
                    pass
            raise

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
                    self._upload_to_artifact_manager(
                        info["microscope_service_id"],
                        info["filename"],
                        image_path.read_bytes(),
                        info["metadata"],
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
        """
        Clean up test datasets from the snapshots gallery.
        Deletes any datasets that start with 'snapshots-test'.
        """
        try:
            # Get the shared snapshots gallery
            gallery = await self.get_or_create_snapshots_gallery("microscope")

            # List all datasets in the gallery
            datasets = await self._svc.list(gallery["id"])

            # Find test datasets (those starting with 'snapshots-test')
            test_datasets = [d for d in datasets if d.get("alias", "").startswith("snapshots-test")]

            print(f"Found {len(test_datasets)} test datasets to clean up")

            # Delete each test dataset
            for dataset in test_datasets:
                try:
                    dataset_id = dataset["id"]
                    dataset_alias = dataset.get("alias", "unknown")
                    print(f"Deleting test dataset: {dataset_alias}")

                    # Delete the dataset and all its files
                    await self._svc.delete(
                        artifact_id=dataset_id,
                        delete_files=True,
                        recursive=False
                    )
                    print(f"✅ Deleted test dataset: {dataset_alias}")

                except Exception as e:
                    print(f"⚠️ Failed to delete test dataset {dataset.get('alias', 'unknown')}: {e}")

            print("✅ Test dataset cleanup completed")

        except Exception as e:
            print(f"⚠️ Error during test dataset cleanup: {e}")

