"""
This module provides artifact management for the application.

Classes:
- SquidArtifactManager: Manages snapshots, vector collections, and basic file operations.

For zarr dataset uploads, use the hypha-artifact package (AsyncHyphaArtifact) instead.
For zarr visualization, use the vizarr package (GPU-accelerated, client-side viewer).
"""

import re

import dotenv
import httpx
from hypha_rpc.rpc import RemoteException

dotenv.load_dotenv()
ENV_FILE = dotenv.find_dotenv()
if ENV_FILE:
    dotenv.load_dotenv(ENV_FILE)

class SquidArtifactManager:
    """
    Manages artifacts for the application.
    Used for snapshot management, vector collections, and basic file operations.
    For zarr dataset uploads, use the hypha-artifact package (AsyncHyphaArtifact) instead.
    """

    def __init__(self):
        self._svc = None
        self.server = None

    def _sanitize_dataset_name(self, name: str) -> str:
        """
        Sanitize dataset name to comply with naming requirements:
        - Lowercase letters, numbers, hyphens, and colons only
        - Must start and end with alphanumeric character
        
        Args:
            name: Original dataset name
            
        Returns:
            str: Sanitized dataset name
        """
        # Convert to lowercase
        sanitized = name.lower()
        
        # Replace underscores with hyphens
        sanitized = sanitized.replace('_', '-')
        
        # Remove any characters that are not lowercase letters, numbers, hyphens, or colons
        sanitized = re.sub(r'[^a-z0-9\-:]', '', sanitized)
        
        # Remove leading/trailing hyphens and colons (must start/end with alphanumeric)
        sanitized = sanitized.strip('-:')
        
        # If empty after sanitization, use a default name
        if not sanitized:
            sanitized = 'dataset'
        
        # Ensure it starts with alphanumeric
        if not sanitized[0].isalnum():
            sanitized = 'dataset-' + sanitized
        
        # Ensure it ends with alphanumeric
        if not sanitized[-1].isalnum():
            sanitized = sanitized + '-data'
        
        return sanitized

    def _remove_timestamp_from_experiment_id(self, experiment_id: str) -> str:
        """Remove timestamp suffix (e.g., '_20251029-658166') and scan type ('_normal-scan', '_quick-scan') from experiment ID for gallery naming."""
        if not experiment_id:
            return experiment_id
        # Remove scan type and timestamp patterns from the end
        return re.sub(r'_(normal|quick)-scan(_\d{8}-\d+)?$', '', experiment_id)

    async def connect_server(self, server):
        """
        Connect to the server.

        Args:
            server (Server): The server instance.
        """
        self.server = server
        self._svc = await server.get_service("public/artifact-manager")


    def _artifact_id(self, workspace, name):
        """
        Generate the artifact ID.

        Args:
            workspace (str): The workspace.
            name (str): The artifact name.

        Returns:
            str: The artifact ID.
        """
        return f"{workspace}/{name}"

    async def create_vector_collection(
        self, workspace, name, manifest, config, overwrite=False, exists_ok=False
    ):
        """
        Create a vector collection.

        Args:
            workspace (str): The workspace.
            name (str): The collection name.
            manifest (dict): The collection manifest.
            config (dict): The collection configuration.
            overwrite (bool, optional): Whether to overwrite the existing collection.
        """
        art_id = self._artifact_id(workspace, name)
        try:
            await self._svc.create(
                alias=art_id,
                type="vector-collection",
                manifest=manifest,
                config=config,
                overwrite=overwrite,
            )
        except RemoteException as e:
            if not exists_ok:
                raise e

    async def add_vectors(self, workspace, coll_name, vectors):
        """
        Add vectors to the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
            vectors (list): The vectors to add.
        """
        art_id = self._artifact_id(workspace, coll_name)
        await self._svc.add_vectors(artifact_id=art_id, vectors=vectors)

    async def search_vectors(self, workspace, coll_name, vector, top_k=None):
        """
        Search for vectors in the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
            vector (ndarray): The query vector.
            top_k (int, optional): The number of top results to return.

        Returns:
            list: The search results.
        """
        art_id = self._artifact_id(workspace, coll_name)
        return await self._svc.search_vectors(
            artifact_id=art_id, query={"cell_image_vector": vector}, limit=top_k
        )

    async def add_file(self, workspace, coll_name, file_content, file_path):
        """
        Add a file to the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
            file_content (bytes): The file content.
            file_path (str): The file path.
        """
        art_id = self._artifact_id(workspace, coll_name)
        await self._svc.edit(artifact_id=art_id, version="stage")
        put_url = await self._svc.put_file(art_id, file_path, download_weight=1.0)
        async with httpx.AsyncClient() as client:
            response = await client.put(put_url, data=file_content, timeout=500)
        response.raise_for_status()
        await self._svc.commit(art_id)

    async def get_file(self, workspace, coll_name, file_path):
        """
        Retrieve a file from the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
            file_path (str): The file path.

        Returns:
            bytes: The file content.
        """
        art_id = self._artifact_id(workspace, coll_name)
        get_url = await self._svc.get_file(art_id, file_path)

        async with httpx.AsyncClient() as client:
            response = await client.get(get_url, timeout=500)
        response.raise_for_status()

        return response.content

    async def remove_vectors(self, workspace, coll_name, vector_ids=None):
        """
        Clear the vectors in the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
        """
        art_id = self._artifact_id(workspace, coll_name)
        if vector_ids is None:
            all_vectors = await self._svc.list_vectors(art_id)
            while len(all_vectors) > 0:
                vector_ids = [vector["id"] for vector in all_vectors]
                await self._svc.remove_vectors(art_id, vector_ids)
                all_vectors = await self._svc.list_vectors(art_id)
        else:
            await self._svc.remove_vectors(art_id, vector_ids)

    async def list_files_in_dataset(self, dataset_id):
        """
        List all files in a dataset.

        Args:
            dataset_id (str): The ID of the dataset.

        Returns:
            list: A list of files in the dataset.
        """
        files = await self._svc.list_files(dataset_id)
        return files

    async def navigate_collections(self, parent_id=None):
        """
        Navigate through collections and datasets.

        Args:
            parent_id (str, optional): The ID of the parent collection. Defaults to None for top-level collections.

        Returns:
            list: A list of collections and datasets under the specified parent.
        """
        collections = await self._svc.list(artifact_id=parent_id)
        return collections

    async def get_file_details(self, dataset_id, file_path):
        """
        Get details of a specific file in a dataset.

        Args:
            dataset_id (str): The ID of the dataset.
            file_path (str): The path to the file in the dataset.

        Returns:
            dict: Details of the file, including size, type, and last modified date.
        """
        files = await self._svc.list_files(dataset_id)
        for file in files:
            if file['name'] == file_path:
                return file
        return None

    async def download_file(self, dataset_id, file_path, local_path):
        """
        Download a file from a dataset.

        Args:
            dataset_id (str): The ID of the dataset.
            file_path (str): The path to the file in the dataset.
            local_path (str): The local path to save the downloaded file.
        """
        get_url = await self._svc.get_file(dataset_id, file_path)
        async with httpx.AsyncClient() as client:
            response = await client.get(get_url)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)

    async def search_datasets(self, keywords=None, filters=None):
        """
        Search and filter datasets based on keywords and filters.

        Args:
            keywords (list, optional): A list of keywords for searching datasets.
            filters (dict, optional): A dictionary of filters to apply.

        Returns:
            list: A list of datasets matching the search criteria.
        """
        datasets = await self._svc.list(keywords=keywords, filters=filters)
        return datasets

    async def list_subfolders(self, dataset_id, dir_path=None):
        """
        List all subfolders in a specified directory within a dataset.

        Args:
            dataset_id (str): The ID of the dataset.
            dir_path (str, optional): The directory path within the dataset to list subfolders. Defaults to None for the root directory.

        Returns:
            list: A list of subfolders in the specified directory.
        """
        try:
            print(f"Listing files for dataset_id={dataset_id}, dir_path={dir_path}")
            files = await self._svc.list_files(dataset_id, dir_path=dir_path)
            print(f"Files received, length: {len(files)}")
            subfolders = [file for file in files if file.get('type') == 'directory']
            print(f"Subfolders filtered, length: {len(subfolders)}")
            return subfolders
        except Exception as e:
            print(f"Error listing subfolders for {dataset_id}: {e}")
            import traceback
            print(traceback.format_exc())
            return []

    async def create_or_get_microscope_gallery(self, microscope_service_id, experiment_id=None):
        """
        Create or get a gallery for a specific microscope in the agent-lens workspace.
        
        Args:
            microscope_service_id (str): The hypha service ID of the microscope
            experiment_id (str, optional): The experiment ID for gallery naming. Required if microscope_service_id ends with a number
            
        Returns:
            dict: The gallery artifact information
        """
        workspace = "agent-lens"

        # Determine gallery naming based on microscope service ID
        # Check if microscope service ID ends with a number (e.g., '-1', '-2', etc.)
        import re
        number_match = re.search(r'-(\d+)$', microscope_service_id)

        if number_match:
            # Special case: microscope ID ends with a number, use experiment-based gallery
            if experiment_id is None:
                raise ValueError("experiment_id is required when microscope_service_id ends with a number")
            gallery_number = number_match.group(1)
            # Remove timestamp suffix from experiment_id for gallery naming
            # This ensures all datasets from the same experiment go into the same gallery
            # even if they have different timestamps
            experiment_id_for_gallery = self._remove_timestamp_from_experiment_id(experiment_id)
            gallery_alias = f"{gallery_number}-{experiment_id_for_gallery}"
        else:
            # Standard case: use microscope-based gallery
            gallery_alias = f"microscope-gallery-{microscope_service_id}"

        try:
            # Try to get existing gallery
            gallery = await self._svc.read(artifact_id=f"{workspace}/{gallery_alias}")
            print(f"Found existing gallery: {gallery_alias}")
            return gallery
        except Exception as e:
            # Handle both RemoteException and RemoteError, and check for various error patterns
            error_str = str(e).lower()
            if ("not found" in error_str or
                "does not exist" in error_str or
                "keyerror" in error_str or
                "artifact with id" in error_str):
                # Gallery doesn't exist, create it
                print(f"Creating new gallery: {gallery_alias}")

                # Determine gallery name and description based on type
                if number_match:
                    # Use cleaned experiment ID (without timestamp) for gallery name
                    experiment_id_for_gallery = self._remove_timestamp_from_experiment_id(experiment_id)
                    gallery_name = f"Experiment Gallery - {experiment_id_for_gallery}"
                    gallery_description = f"Dataset collection for experiment {experiment_id_for_gallery}"
                    gallery_type = "experiment-gallery"
                else:
                    gallery_name = f"Microscope Gallery - {microscope_service_id}"
                    gallery_description = f"Dataset collection for microscope service {microscope_service_id}"
                    gallery_type = "microscope-gallery"

                gallery_manifest = {
                    "name": gallery_name,
                    "description": gallery_description,
                    "microscope_service_id": microscope_service_id,
                    "experiment_id": experiment_id,
                    "created_by": "squid-control-system",
                    "type": gallery_type
                }

                gallery_config = {
                    "permissions": {"*": "r", "@": "r+"},
                    "collection_schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "record_type": {"type": "string", "enum": ["zarr-dataset"]},
                            "microscope_service_id": {"type": "string"},
                            "experiment_id": {"type": "string"},
                            "acquisition_settings": {"type": "object"},
                            "timestamp": {"type": "string"}
                        },
                        "required": ["name", "description", "record_type"]
                    }
                }

                gallery = await self._svc.create(
                    alias=f"{workspace}/{gallery_alias}",
                    type="collection",
                    manifest=gallery_manifest,
                    config=gallery_config
                )
                print(f"Created gallery: {gallery_alias}")
                return gallery
            else:
                raise e