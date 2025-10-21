"""
Snapshot management utilities for microscope image capture using Artifact Manager.

This module provides the SnapshotManager class for storing microscope snapshots
in daily datasets with public URL access through the Hypha Artifact Manager.
"""

import time
import uuid
import httpx
from typing import Optional, Dict
from .artifact_manager.artifact_manager import SquidArtifactManager


class SnapshotManager:
    """
    Manages microscope snapshot storage using Artifact Manager.
    
    Snapshots are organized into:
    - Gallery: "microscope-snapshots" (shared by all microscopes)
    - Datasets: "snapshots-{service_id}-{YYYY-MM-DD}" (one per day per microscope)
    - Files: "{uuid}.png" (UUID-based naming for uniqueness)
    
    All snapshots have public read access through the artifact manager.
    """
    
    def __init__(self, artifact_manager: SquidArtifactManager):
        """
        Initialize the snapshot manager.
        
        Args:
            artifact_manager: An initialized SquidArtifactManager instance
        """
        self.artifact_manager = artifact_manager
        self._svc = artifact_manager._svc
        self.workspace = "agent-lens"
        
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
    
    async def save_snapshot(
        self, 
        microscope_service_id: str, 
        image_bytes: bytes,
        metadata: Dict
    ) -> str:
        """
        Save a snapshot image to the daily dataset and return public URL.
        
        Args:
            microscope_service_id: The hypha service ID of the microscope
            image_bytes: PNG image data as bytes
            metadata: Dictionary containing snapshot metadata (timestamp, channel, position, etc.)
            
        Returns:
            str: Public URL to access the snapshot image
        """
        # Get or create today's dataset
        dataset = await self.get_or_create_daily_dataset(microscope_service_id)
        
        # Generate unique filename using UUID
        filename = f"{uuid.uuid4()}.png"
        
        # Stage the dataset for editing
        await self._svc.edit(artifact_id=dataset["id"], stage=True)
        
        try:
            # Upload the snapshot file
            put_url = await self._svc.put_file(
                artifact_id=dataset["id"],
                file_path=filename,
                download_weight=0.1  # Low weight for snapshots
            )
            
            # Upload file content using HTTP PUT
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                response = await client.put(put_url, content=image_bytes)
                response.raise_for_status()
            
            print(f"Uploaded snapshot: {filename} ({len(image_bytes) / 1024:.2f} KB)")
            
            # Update dataset manifest with snapshot metadata
            current_manifest = dataset.get("manifest", {})
            snapshot_count = current_manifest.get("snapshot_count", 0) + 1
            
            # Store metadata for this snapshot
            if "snapshots" not in current_manifest:
                current_manifest["snapshots"] = {}
            
            current_manifest["snapshots"][filename] = metadata
            current_manifest["snapshot_count"] = snapshot_count
            current_manifest["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            
            # Update the manifest
            await self._svc.edit(
                artifact_id=dataset["id"],
                manifest=current_manifest,
                stage=True
            )
            
            # Commit the changes
            await self._svc.commit(artifact_id=dataset["id"])
            
            # Get public URL for the snapshot
            file_url = await self._svc.get_file(
                artifact_id=dataset["id"],
                file_path=filename
            )
            
            print(f"Snapshot saved successfully: {filename}")
            print(f"Public URL: {file_url}")
            
            return file_url
            
        except Exception as e:
            # If upload fails, try to discard staged changes
            try:
                await self._svc.discard(artifact_id=dataset["id"])
            except:
                pass
            raise Exception(f"Failed to save snapshot: {e}")

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
            
            print(f"✅ Test dataset cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error during test dataset cleanup: {e}")

