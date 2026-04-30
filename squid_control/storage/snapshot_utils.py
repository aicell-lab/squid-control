"""
Snapshot management utilities for microscope image capture using Artifact Manager.

Uploads snapshots directly via Artifact Manager RPC — no local buffering.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict

import httpx

from hypha_rpc.rpc import RemoteException

from .artifact_manager.artifact_manager import SquidArtifactManager

logger = logging.getLogger(__name__)

_UPLOAD_TIMEOUT_S = 10.0
_MAX_RETRIES = 2
_RETRY_BACKOFF_S = 2.0


class SnapshotManager:
    """Manages microscope snapshot storage using Artifact Manager RPC.

    Snapshots are uploaded directly — no local disk buffering.
    """

    def __init__(self, artifact_manager: SquidArtifactManager):
        self.artifact_manager = artifact_manager
        self.workspace = "reef-imaging"

    @property
    def _svc(self):
        return self.artifact_manager._svc

    async def get_or_create_snapshots_gallery(self, microscope_service_id: str) -> dict:
        gallery_alias = "microscope-snapshots"
        try:
            gallery = await self._svc.read(
                artifact_id=f"{self.workspace}/{gallery_alias}"
            )
            logger.info("Found existing snapshots gallery: %s", gallery_alias)
            return gallery
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

        logger.info("Creating snapshots gallery: %s", gallery_alias)
        gallery = await self._svc.create(
            alias=f"{self.workspace}/{gallery_alias}",
            type="collection",
            manifest={
                "name": "Microscope Snapshots Gallery",
                "description": "Shared gallery for daily snapshot collections",
                "created_by": "squid-control-system",
                "type": "snapshot-gallery",
            },
            config={"permissions": {"*": "r", "@": "r+"}},
        )
        return gallery

    async def get_or_create_daily_dataset(
        self, microscope_service_id: str
    ) -> dict:
        date_str = time.strftime("%Y-%m-%d", time.gmtime())
        dataset_name = f"snapshots-{microscope_service_id}-{date_str}"
        dataset_alias = f"{self.workspace}/{dataset_name}"

        try:
            dataset = await self._svc.read(artifact_id=dataset_alias)
            logger.info("Found existing snapshot dataset: %s", dataset_name)
            return dataset
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

        logger.info("Creating snapshot dataset: %s", dataset_name)
        gallery = await self.get_or_create_snapshots_gallery(microscope_service_id)
        dataset = await self._svc.create(
            parent_id=gallery["id"],
            alias=dataset_alias,
            manifest={
                "name": dataset_name,
                "description": f"Snapshots for {microscope_service_id} on {date_str}",
                "record_type": "snapshot-dataset",
                "microscope_service_id": microscope_service_id,
                "date": date_str,
                "snapshot_count": 0,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            },
            stage=False,
        )
        return dataset

    async def save_snapshot(
        self,
        microscope_service_id: str,
        image_bytes: bytes,
        metadata: Dict,
    ) -> str:
        """Upload a snapshot via Artifact Manager RPC with retries.

        Returns a public file URL on success.  Raises if all retries fail.
        """
        filename = f"{uuid.uuid4()}.png"

        last_exc = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self._upload_one(filename, image_bytes, metadata, microscope_service_id),
                    timeout=_UPLOAD_TIMEOUT_S,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Upload attempt %d/%d failed: %s (%s)",
                    attempt, _MAX_RETRIES, exc, type(exc).__name__,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_BACKOFF_S)

        raise Exception(
            f"Failed to upload snapshot after {_MAX_RETRIES} attempts: "
            f"{type(last_exc).__name__}: {last_exc}"
        )

    async def _upload_one(
        self,
        filename: str,
        image_bytes: bytes,
        metadata: Dict,
        microscope_service_id: str,
        _is_retry: bool = False,
    ) -> str:
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

            logger.info("Uploaded snapshot: %s (%.2f KB)", filename, len(image_bytes) / 1024)

            current_manifest = dataset.get("manifest", {})
            current_manifest["snapshot_count"] = current_manifest.get("snapshot_count", 0) + 1
            current_manifest["last_updated"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime()
            )

            await self._svc.edit(
                artifact_id=dataset["id"], manifest=current_manifest, stage=True
            )
            await self._svc.commit(artifact_id=dataset["id"])

            dataset_alias = dataset.get("alias", "").replace(f"{self.workspace}/", "")
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
                exc, (asyncio.TimeoutError, ConnectionError, OSError, RemoteException)
            ):
                logger.warning(
                    "Artifact manager upload failed (%s), disconnecting and reconnecting...",
                    type(exc).__name__,
                )
                try:
                    await self.artifact_manager._reset_and_reconnect()
                    return await self._upload_one(
                        filename, image_bytes, metadata, microscope_service_id,
                        _is_retry=True,
                    )
                except Exception:
                    pass
            raise

    async def cleanup_test_datasets(self):
        """Clean up test datasets from the snapshots gallery."""
        try:
            gallery = await self.get_or_create_snapshots_gallery("microscope")
            datasets = await self._svc.list(gallery["id"])
            test_datasets = [
                d
                for d in datasets
                if d.get("alias", "").startswith("snapshots-test")
            ]
            logger.info("Found %d test datasets to clean up", len(test_datasets))
            for dataset in test_datasets:
                try:
                    await self._svc.delete(
                        artifact_id=dataset["id"],
                        delete_files=True,
                        recursive=False,
                    )
                    logger.info("Deleted test dataset: %s", dataset.get("alias"))
                except Exception as e:
                    logger.warning("Failed to delete %s: %s", dataset.get("alias"), e)
        except Exception as e:
            logger.warning("Error during test dataset cleanup: %s", e)
