"""
This module provides the ArtifactManager class, which manages artifacts for the application.
It includes methods for creating vector collections, adding vectors, searching vectors,
and handling file uploads and downloads.
"""

import httpx
from hypha_rpc.rpc import RemoteException
import asyncio
import os
import io
import dotenv
from hypha_rpc import connect_to_server
from PIL import Image
import numpy as np
import base64
import numcodecs
import blosc
import aiohttp
from collections import deque
import zarr
from zarr.storage import LRUStoreCache, FSStore
import fsspec

dotenv.load_dotenv()  
ENV_FILE = dotenv.find_dotenv()  
if ENV_FILE:  
    dotenv.load_dotenv(ENV_FILE)  

class SquidArtifactManager:
    """
    Manages artifacts for the application.
    """

    def __init__(self):
        self._svc = None
        self.server = None

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

    async def get_zarr_group(
        self,
        workspace: str,
        artifact_alias: str,
        timestamp: str,
        channel: str,
        cache_max_size=2**28 # 256 MB LRU cache
    ):
        """
        Access a Zarr group stored within a zip file in an artifact.

        Args:
            workspace (str): The workspace containing the artifact.
            artifact_alias (str): The alias of the artifact (e.g., 'image-map-20250429-treatment-zip').
            timestamp (str): The timestamp folder name.
            channel (str): The channel name (used for the zip filename).
            cache_max_size (int, optional): Max size for LRU cache in bytes. Defaults to 2**28.

        Returns:
            zarr.Group: The root Zarr group object.
        """
        if self._svc is None:
            raise ConnectionError("Artifact Manager service not connected. Call connect_server first.")

        art_id = self._artifact_id(workspace, artifact_alias)
        zip_file_path = f"{timestamp}/{channel}.zip"

        try:
            print(f"Accessing artifact via zip-files endpoint: {art_id}/{zip_file_path}")
            
            # Use the AGENT_LENS_WORKSPACE_TOKEN for authentication
            token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
            if not token:
                raise ValueError("AGENT_LENS_WORKSPACE_TOKEN environment variable is not set")
            
            # Base URL for accessing the contents of the zip file
            base_url = f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path}/~"
            
            # Create HTTP headers with authorization
            headers = {"Authorization": f"Bearer {token}"}
            
            # Define the synchronous function to open the Zarr store and group
            def _open_zarr_sync(url, headers, cache_size):
                print(f"Opening Zarr store using zip-files endpoint: {url}")
                
                # Create an HTTP file system with custom headers
                http_fs = fsspec.filesystem("http", headers=headers)
                
                # Create a mapper for accessing the zip contents
                store = zarr.storage.FSStore(url, mode="r", fs=http_fs)
                
                if cache_size and cache_size > 0:
                    print(f"Using LRU cache with size: {cache_size} bytes")
                    store = LRUStoreCache(store, max_size=cache_size)
                
                # Create a root group from the store
                root_group = zarr.open_group(store, mode="r")
                print(f"Zarr group opened successfully")
                
                return root_group

            # Run the synchronous Zarr operations in a thread pool
            print("Running Zarr open in thread executor...")
            zarr_group = await asyncio.to_thread(_open_zarr_sync, base_url, headers, cache_max_size)
            return zarr_group

        except RemoteException as e:
            print(f"Error accessing zip file via zip-files endpoint: {e}")
            raise FileNotFoundError(f"Could not find or access zip file {zip_file_path} in artifact {art_id}") from e
        except Exception as e:
            print(f"An error occurred while accessing the Zarr group: {e}")
            import traceback
            print(traceback.format_exc())
            raise

# Constants
SERVER_URL = "https://hypha.aicell.io"
AGENT_LENS_WORKSPACE_TOKEN = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
ARTIFACT_ALIAS = "image-map-20250429-treatment-zip"
DEFAULT_CHANNEL = "BF_LED_matrix_full"

# New class to replace TileManager using Zarr for efficient access
class ZarrImageManager:
    def __init__(self):
        self.artifact_manager = None
        self.artifact_manager_server = None
        self.workspace = "agent-lens"  # Default workspace
        self.chunk_size = 256  # Default chunk size for Zarr
        self.channels = [
            "BF_LED_matrix_full",
            "Fluorescence_405_nm_Ex",
            "Fluorescence_488_nm_Ex",
            "Fluorescence_561_nm_Ex",
            "Fluorescence_638_nm_Ex"
        ]
        self.zarr_groups_cache = {}  # Cache for open Zarr groups
        self.is_running = True
        self.session = None
        self.default_timestamp = "2025-04-29_16-38-27"  # Set a default timestamp

    async def connect(self, workspace_token=None, server_url="https://hypha.aicell.io"):
        """Connect to the Artifact Manager service"""
        try:
            token = workspace_token or os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
            if not token:
                raise ValueError("Workspace token not provided")
            
            self.artifact_manager_server = await connect_to_server({
                "name": "zarr-image-client",
                "server_url": server_url,
                "token": token,
            })
            
            self.artifact_manager = SquidArtifactManager()
            await self.artifact_manager.connect_server(self.artifact_manager_server)
            
            # Initialize aiohttp session for any HTTP requests
            self.session = aiohttp.ClientSession()
            
            print("ZarrImageManager connected successfully")
            return True
        except Exception as e:
            print(f"Error connecting to artifact manager: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    async def close(self):
        """Close the image manager and cleanup resources"""
        self.is_running = False
        
        # Close the cached Zarr groups
        self.zarr_groups_cache.clear()
        
        # Close the aiohttp session
        if self.session:
            await self.session.close()
            self.session = None
        
        # Disconnect from the server
        if self.artifact_manager_server:
            await self.artifact_manager_server.disconnect()
            self.artifact_manager_server = None
            self.artifact_manager = None

    async def get_zarr_group(self, dataset_id, timestamp, channel):
        """Get (or reuse from cache) a Zarr group for a specific dataset"""
        cache_key = f"{dataset_id}:{timestamp}:{channel}"
        
        if cache_key in self.zarr_groups_cache:
            #print(f"Using cached Zarr group for {cache_key}")
            return self.zarr_groups_cache[cache_key]
        
        try:
            # Parse the dataset_id to extract workspace and artifact_alias
            # dataset_id format is expected to be "workspace/artifact_alias"
            parts = dataset_id.split('/', 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid dataset_id format: {dataset_id}. Expected 'workspace/artifact_alias'")
            
            workspace, artifact_alias = parts
            
            # We no longer need to parse the dataset_id into workspace and artifact_alias
            # Just use the dataset_id directly since it's already the full path
            print(f"Accessing artifact via zip-files endpoint: {dataset_id}/{timestamp}/{channel}.zip")
            
            # Construct the zip file path
            zip_file_path = f"{timestamp}/{channel}.zip"
            
            # Instead of getting a download URL that expires, use the direct zip-files endpoint with tilde notation
            token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
            if not token:
                raise ValueError("AGENT_LENS_WORKSPACE_TOKEN environment variable is not set")
            
            # Base URL for accessing the contents of the zip file with tilde notation
            base_url = f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path}/~"
            
            # Create HTTP headers with authorization
            headers = {"Authorization": f"Bearer {token}"}
            
            # Define the synchronous function to open the Zarr store and group
            def _open_zarr_sync(url, headers, cache_size):
                print(f"Opening Zarr store using zip-files endpoint: {url}")
                
                # Create an HTTP file system with custom headers
                http_fs = fsspec.filesystem("http", headers=headers)
                
                # Create a mapper for accessing the zip contents
                store = zarr.storage.FSStore(url, mode="r", fs=http_fs)
                
                if cache_size and cache_size > 0:
                    store = LRUStoreCache(store, max_size=cache_size)
                
                # Create a root group from the store
                root_group = zarr.open_group(store, mode="r")
                print(f"Zarr group opened successfully")
                
                return root_group
                
            # Run the synchronous Zarr operations in a thread pool
            print("Running Zarr open in thread executor...")
            zarr_group = await asyncio.to_thread(_open_zarr_sync, base_url, headers, 2**28)  # Using default cache size
            
            # Cache the Zarr group for future use
            self.zarr_groups_cache[cache_key] = zarr_group
            
            return zarr_group
        except Exception as e:
            print(f"Error accessing zarr group via zip-files endpoint: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    async def get_region_np_data(self, dataset_id, timestamp, channel, scale, x, y):
        """
        Get a region as numpy array using Zarr for efficient access
        
        Args:
            dataset_id (str): The dataset ID (workspace/artifact_alias)
            timestamp (str): The timestamp folder 
            channel (str): Channel name
            scale (int): Scale level
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            np.ndarray: Region data as numpy array
        """
        try:
            # Use default timestamp if none provided
            timestamp = timestamp or self.default_timestamp
            
            # Get or create the zarr group
            zarr_group = await self.get_zarr_group(dataset_id, timestamp, channel)
            if not zarr_group:
                print(f"No Zarr group found for {dataset_id}/{timestamp}/{channel}")
                return np.zeros((self.chunk_size, self.chunk_size), dtype=np.uint8)
            
            # Navigate to the right array in the Zarr hierarchy
            try:
                # Debug: Print available keys in the zarr group
                print(f"Available keys in Zarr group: {list(zarr_group.keys())}")
                
                # Dynamically determine the correct scale key
                scale_key = None
                
                # Try different naming conventions for scale levels
                potential_keys = [
                    f'scale{scale}',  # Standard format: scale0, scale1, etc.
                    str(scale),       # Just the number: 0, 1, etc.
                    f's{scale}',      # Alternative prefix: s0, s1, etc.
                    f'level{scale}'   # Another common format: level0, level1, etc.
                ]
                
                for key in potential_keys:
                    if key in zarr_group:
                        scale_key = key
                        print(f"Found matching scale key: {scale_key}")
                        break
                
                # If no matching key, use the first available key
                if scale_key is None and len(list(zarr_group.keys())) > 0:
                    scale_key = list(zarr_group.keys())[0]
                    print(f"No matching scale key found, using first available key: {scale_key}")
                
                if scale_key is None:
                    raise KeyError("No usable keys found in Zarr group")
                
                # Get the scale array
                scale_array = zarr_group[scale_key]
                print(f"Scale array shape: {scale_array.shape}, dtype: {scale_array.dtype}")
                
                # Ensure chunk coordinates are valid
                array_shape = scale_array.shape
                if len(array_shape) < 2:
                    raise ValueError(f"Scale array has unexpected shape: {array_shape}")
                
                # Calculate the bounds
                y_start = y * self.chunk_size
                x_start = x * self.chunk_size
                
                # Ensure coordinates are within bounds
                if y_start >= array_shape[0] or x_start >= array_shape[1]:
                    print(f"Coordinates out of bounds: y={y_start}, x={x_start}, shape={array_shape}")
                    return np.zeros((self.chunk_size, self.chunk_size), dtype=np.uint8)
                
                # Get the specific chunk/region - adjust slicing as needed
                y_end = min(y_start + self.chunk_size, array_shape[0])
                x_end = min(x_start + self.chunk_size, array_shape[1])
                
                print(f"Reading region from y={y_start} to y={y_end}, x={x_start} to x={x_end}")
                region_data = scale_array[y_start:y_end, x_start:x_end]
                
                # Debug info about retrieved data
                print(f"Region data shape: {region_data.shape}, dtype: {region_data.dtype}, " 
                      f"min: {region_data.min() if region_data.size > 0 else 'N/A'}, "
                      f"max: {region_data.max() if region_data.size > 0 else 'N/A'}")
                
                # Make sure we have a properly shaped array
                if region_data.shape != (self.chunk_size, self.chunk_size):
                    # Resize or pad if necessary
                    print(f"Padding region from {region_data.shape} to {(self.chunk_size, self.chunk_size)}")
                    result = np.zeros((self.chunk_size, self.chunk_size), dtype=region_data.dtype or np.uint8)
                    h, w = region_data.shape
                    result[:min(h, self.chunk_size), :min(w, self.chunk_size)] = region_data[:min(h, self.chunk_size), :min(w, self.chunk_size)]
                    return result
                
                # Ensure data is in the right format (uint8)
                if region_data.dtype != np.uint8:
                    print(f"Converting data from {region_data.dtype} to uint8")
                    if region_data.dtype == np.float32 or region_data.dtype == np.float64:
                        # Normalize floating point data
                        if region_data.max() > 0:
                            normalized = (region_data / region_data.max() * 255).astype(np.uint8)
                        else:
                            normalized = np.zeros(region_data.shape, dtype=np.uint8)
                        return normalized
                    else:
                        # For other integer types, scale appropriately
                        return region_data.astype(np.uint8)
                
                return region_data
                
            except KeyError as e:
                print(f"Error accessing Zarr array path: {e}")
                # If specific key approach failed, try to explore the structure and find data
                try:
                    # Simple approach: try to traverse into any sub-groups until we find an array
                    def find_array(group, depth=0, max_depth=3):
                        if depth >= max_depth:
                            return None
                        
                        for key in group.keys():
                            item = group[key]
                            
                            # Check if it's an array with at least 2 dimensions
                            if hasattr(item, 'shape') and len(item.shape) >= 2:
                                print(f"Found array at path: {key}, shape: {item.shape}")
                                return item
                            
                            # If it's a group, recursively check inside
                            elif hasattr(item, 'keys'):
                                result = find_array(item, depth+1, max_depth)
                                if result is not None:
                                    return result
                        
                        return None
                    
                    array = find_array(zarr_group)
                    if array is not None:
                        # Calculate coordinates based on the found array's dimensions
                        y_start = min(y * self.chunk_size, array.shape[0] - 1)
                        x_start = min(x * self.chunk_size, array.shape[1] - 1)
                        y_end = min(y_start + self.chunk_size, array.shape[0])
                        x_end = min(x_start + self.chunk_size, array.shape[1])
                        
                        region_data = array[y_start:y_end, x_start:x_end]
                        
                        # Pad if necessary
                        if region_data.shape != (self.chunk_size, self.chunk_size):
                            result = np.zeros((self.chunk_size, self.chunk_size), dtype=region_data.dtype or np.uint8)
                            h, w = region_data.shape
                            result[:min(h, self.chunk_size), :min(w, self.chunk_size)] = region_data[:min(h, self.chunk_size), :min(w, self.chunk_size)]
                            return result
                        
                        return region_data
                
                except Exception as nested_e:
                    print(f"Alternative array search failed: {nested_e}")
                
                # If all else fails, return an empty array
                return np.zeros((self.chunk_size, self.chunk_size), dtype=np.uint8)
                
        except Exception as e:
            print(f"Error getting region data: {e}")
            import traceback
            print(traceback.format_exc())
            return np.zeros((self.chunk_size, self.chunk_size), dtype=np.uint8)

    async def get_region_bytes(self, dataset_id, timestamp, channel, scale, x, y):
        """Serve a region as PNG bytes"""
        try:
            # Use default timestamp if none provided
            timestamp = timestamp or self.default_timestamp
            
            # Get region data as numpy array
            region_data = await self.get_region_np_data(dataset_id, timestamp, channel, scale, x, y)
            
            # Convert to PNG bytes
            image = Image.fromarray(region_data)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as e:
            print(f"Error in get_region_bytes: {str(e)}")
            blank_image = Image.new("L", (self.chunk_size, self.chunk_size), color=0)
            buffer = io.BytesIO()
            blank_image.save(buffer, format="PNG")
            return buffer.getvalue()

    async def get_region_base64(self, dataset_id, timestamp, channel, scale, x, y):
        """Serve a region as base64 string"""
        # Use default timestamp if none provided
        timestamp = timestamp or self.default_timestamp
        
        region_bytes = await self.get_region_bytes(dataset_id, timestamp, channel, scale, x, y)
        return base64.b64encode(region_bytes).decode('utf-8')
