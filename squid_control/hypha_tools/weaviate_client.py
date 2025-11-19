"""
Weaviate client module for similarity search integration.

This module provides async functions for interacting with the Weaviate vector database
through direct Hypha RPC service calls (not HTTP endpoints).
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from hypha_rpc import connect_to_server

logger = logging.getLogger(__name__)

# Weaviate service configuration
WEAVIATE_SERVER_URL = "https://hypha.aicell.io"
WEAVIATE_WORKSPACE = "hypha-agents"
WEAVIATE_SERVICE_NAME = "hypha-agents/weaviate"
WEAVIATE_COLLECTION_NAME = "Agentlens"

# Global service connection (lazy initialization)
_weaviate_server = None
_weaviate_service = None


async def _get_weaviate_service():
    """
    Get or create connection to Weaviate service via Hypha RPC.
    
    Returns:
        weaviate_service: Connected Weaviate service object
    
    Raises:
        RuntimeError: If connection fails or service not available
    """
    global _weaviate_server, _weaviate_service
    
    if _weaviate_service is not None:
        return _weaviate_service
    
    try:
        token = os.getenv("HYPHA_AGENTS_TOKEN")
        if not token:
            raise RuntimeError("HYPHA_AGENTS_TOKEN not set in environment")
        
        _weaviate_server = await connect_to_server({
            "server_url": WEAVIATE_SERVER_URL,
            "workspace": WEAVIATE_WORKSPACE,
            "token": token
        })
        
        _weaviate_service = await _weaviate_server.get_service(WEAVIATE_SERVICE_NAME, mode="first")
        logger.info(f"Connected to Weaviate service: {WEAVIATE_SERVICE_NAME}")
        return _weaviate_service
        
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate service: {e}")
        raise RuntimeError(f"Could not connect to Weaviate service: {e}")


async def delete_weaviate_application(
    application_id: str,
    collection_name: str = WEAVIATE_COLLECTION_NAME
) -> Dict[str, Any]:
    """
    Delete existing Weaviate application if it exists.
    
    Args:
        application_id: Unique identifier for the application (e.g., experiment name)
        collection_name: Weaviate collection name (default: "Agentlens")
    
    Returns:
        Dict with 'success' (bool) and optional 'deleted_count' (int) or 'error' (str)
    """
    try:
        weaviate_service = await _get_weaviate_service()
        
        # Delete application using Hypha RPC service
        result = await weaviate_service.applications.delete(
            collection_name=collection_name,
            application_id=application_id
        )
        
        # Extract deleted count from result
        deleted_count = 0
        if isinstance(result, dict):
            deleted_count = result.get('successful', 0) or result.get('deleted', 0)
        
        logger.info(f"Deleted Weaviate application '{application_id}' ({deleted_count} objects removed)")
        return {'success': True, 'deleted_count': deleted_count}
        
    except Exception as e:
        error_msg = str(e)
        # Check if error indicates application doesn't exist
        if 'does not exist' in error_msg.lower() or 'not found' in error_msg.lower():
            logger.info(f"Weaviate application '{application_id}' does not exist (will create new one)")
            return {'success': True, 'deleted_count': 0}
        else:
            logger.error(f"Error deleting Weaviate application: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}


async def create_weaviate_application(
    application_id: str,
    description: str = "",
    collection_name: str = WEAVIATE_COLLECTION_NAME
) -> Dict[str, Any]:
    """
    Create new Weaviate application.
    
    Args:
        application_id: Unique identifier for the application
        description: Optional description for the application
        collection_name: Weaviate collection name (default: "Agentlens")
    
    Returns:
        Dict with 'success' (bool) and optional 'error' (str)
    """
    try:
        weaviate_service = await _get_weaviate_service()
        
        # Create application using Hypha RPC service
        await weaviate_service.applications.create(
            collection_name=collection_name,
            application_id=application_id,
            description=description
        )
        
        logger.info(f"Created Weaviate application '{application_id}'")
        return {'success': True}
        
    except Exception as e:
        logger.error(f"Error creating Weaviate application: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


async def set_current_weaviate_application(
    application_id: str
) -> Dict[str, Any]:
    """
    Set current Weaviate application.
    
    Note: This is a non-critical operation - failures are logged but don't stop the workflow.
    
    Args:
        application_id: Application ID to set as current
    
    Returns:
        Dict with 'success' (bool) and optional 'error' (str)
    """
    try:
        weaviate_service = await _get_weaviate_service()
        
        # Set current application (if the service supports this method)
        # Some Weaviate services may not have this method, so we handle gracefully
        if hasattr(weaviate_service.applications, 'set_current'):
            await weaviate_service.applications.set_current(application_id)
            logger.info(f"Set current Weaviate application to '{application_id}'")
            return {'success': True}
        else:
            # Method not available, but this is non-critical
            logger.debug(f"set_current method not available, skipping")
            return {'success': True}
        
    except Exception as e:
        logger.warning(f"Error setting current application (non-critical): {e}")
        return {'success': False, 'error': str(e)}


async def batch_upload_to_weaviate(
    objects: List[Dict[str, Any]],
    application_id: str,
    collection_name: str = WEAVIATE_COLLECTION_NAME,
    retry_attempts: int = 2
) -> Dict[str, Any]:
    """
    Upload batch of objects to Weaviate using insert_many.
    
    Args:
        objects: List of object dictionaries with fields:
            Required fields:
            - image_id: Unique identifier
            - description: Text description
            - metadata: Dict with annotation info
            - dataset_id: Dataset identifier
            - vector: Embedding vector (list of floats)
            
            Additional fields (may be None if extraction fails):
            - preview_image: Base64 PNG string (50x50 preview)
            
            Morphological features (in µm or unitless, None if extraction fails):
            - area: Cell area in µm²
            - perimeter: Cell perimeter in µm
            - equivalent_diameter: Diameter of circle with same area in µm
            - bbox_width: Bounding box width in µm
            - bbox_height: Bounding box height in µm
            - aspect_ratio: Major axis / minor axis (elongation, unitless)
            - circularity: 4π×area/perimeter² (roundness, unitless)
            - eccentricity: 0 = circle, → 1 = elongated (unitless)
            - solidity: Area / convex hull area (unitless)
            - convexity: Convex hull perimeter / perimeter (unitless)
            
        application_id: Application ID for the upload
        collection_name: Weaviate collection name (default: "Agentlens")
        retry_attempts: Number of retry attempts on failure (default: 2)
    
    Returns:
        Dict with 'success' (bool), 'uploaded_count' (int), and optional 'error' (str)
    """
    if not objects:
        return {'success': True, 'uploaded_count': 0}
    
    # Validate objects
    for idx, obj in enumerate(objects):
        # Core required fields for Weaviate
        required_fields = ['image_id', 'description', 'metadata', 'dataset_id', 'vector']
        
        # Morphological features that are always included (may have None values)
        expected_feature_fields = [
            'area', 'perimeter', 'equivalent_diameter', 'bbox_width', 'bbox_height',
            'aspect_ratio', 'circularity', 'eccentricity', 'solidity', 'convexity'
        ]
        
        # Check for missing required fields
        missing_fields = [f for f in required_fields if f not in obj]
        if missing_fields:
            logger.error(f"Object {idx} missing required fields: {missing_fields}")
            return {'success': False, 'error': f"Object {idx} missing fields: {missing_fields}"}
        
        # Check for missing feature fields (warn but don't fail)
        missing_features = [f for f in expected_feature_fields if f not in obj]
        if missing_features:
            logger.warning(f"Object {idx} missing feature fields: {missing_features}")
    
    # Prepare objects for insert_many
    prepared_objects = []
    for obj in objects:
        # Extract vector from object
        vector = obj.pop('vector')
        
        # Convert metadata dict to JSON string if needed
        if 'metadata' in obj and isinstance(obj['metadata'], dict):
            obj['metadata'] = json.dumps(obj['metadata'])
        
        # Create prepared object with properties and vector
        prepared_obj = {
            **obj,  # All properties
            'vector': vector
        }
        prepared_objects.append(prepared_obj)
    
    # Retry logic
    last_error = None
    for attempt in range(retry_attempts + 1):
        try:
            weaviate_service = await _get_weaviate_service()
            
            # Upload using insert_many
            result = await weaviate_service.data.insert_many(
                collection_name=collection_name,
                application_id=application_id,
                objects=prepared_objects
            )
            
            logger.info(f"Successfully uploaded {len(objects)} objects to Weaviate")
            return {'success': True, 'uploaded_count': len(objects)}
            
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Upload attempt {attempt + 1} failed with exception: {e}")
            
            # Don't retry on validation errors
            if 'validation' in last_error.lower() or 'invalid' in last_error.lower():
                break
    
    logger.error(f"Failed to upload batch after {retry_attempts + 1} attempts: {last_error}")
    return {'success': False, 'error': last_error, 'uploaded_count': 0}


async def setup_weaviate_application(
    application_id: str,
    description: str = "",
    collection_name: str = WEAVIATE_COLLECTION_NAME,
    reset_application: bool = True
) -> Dict[str, Any]:
    """
    Complete setup workflow: optionally delete existing, create new, and set current application.
    
    Args:
        application_id: Unique identifier for the application
        description: Optional description
        collection_name: Weaviate collection name
        reset_application: If True, delete existing application before creating (default: True)
    
    Returns:
        Dict with 'success' (bool) and optional 'error' (str)
    """
    # Step 1: Delete existing application if reset requested
    if reset_application:
        delete_result = await delete_weaviate_application(application_id, collection_name)
        if not delete_result['success']:
            logger.warning(f"Delete failed but continuing: {delete_result.get('error')}")
    
    # Step 2: Create new application
    create_result = await create_weaviate_application(application_id, description, collection_name)
    if not create_result['success']:
        return create_result
    
    # Step 3: Set as current application (non-critical)
    await set_current_weaviate_application(application_id)
    
    return {'success': True}


async def search_similar_by_uuid(
    object_uuid: str,
    collection_name: str = WEAVIATE_COLLECTION_NAME,
    application_id: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search for similar images by UUID.
    
    This function fetches the object by UUID to get its vector, then performs similarity search.
    
    Args:
        object_uuid: UUID of the object to search for similar items
        collection_name: Weaviate collection name (default: "Agentlens")
        application_id: Application ID (required for fetching the object)
        limit: Maximum number of results to return (default: 10)
    
    Returns:
        Dict with 'success' (bool), 'results' (list), 'count' (int), and optional 'error' (str)
    """
    try:
        if not application_id:
            raise ValueError("application_id is required for UUID search")
        
        weaviate_service = await _get_weaviate_service()
        
        # Step 1: Fetch the object by UUID to get its vector
        try:
            query_object = await weaviate_service.data.get(
                collection_name=collection_name,
                application_id=application_id,
                uuid=object_uuid,
                include_vector=True
            )
        except Exception as e:
            # If direct get fails, try fetching all and filtering (fallback)
            logger.debug(f"Direct UUID fetch failed, trying fallback: {e}")
            objects = await weaviate_service.query.fetch_objects(
                collection_name=collection_name,
                application_id=application_id,
                limit=10000,
                return_properties=["image_id"],
                include_vector=True
            )
            
            # Find the object with matching UUID
            query_object = None
            actual_objects = objects.objects if hasattr(objects, 'objects') else objects
            for obj in actual_objects:
                obj_uuid = getattr(obj, 'uuid', None) or getattr(obj, 'id', None)
                if str(obj_uuid) == str(object_uuid):
                    query_object = obj
                    break
            
            if query_object is None:
                raise ValueError(f"Object with UUID '{object_uuid}' not found")
        
        # Step 2: Extract vector from the object
        query_vector = None
        if hasattr(query_object, 'vector'):
            query_vector = query_object.vector
        elif isinstance(query_object, dict) and 'vector' in query_object:
            query_vector = query_object['vector']
        
        if query_vector is None:
            raise ValueError(f"Could not extract vector from object with UUID '{object_uuid}'")
        
        # Step 3: Perform similarity search using the extracted vector
        results = await weaviate_service.query.near_vector(
            collection_name=collection_name,
            application_id=application_id,
            near_vector=query_vector,
            limit=limit + 1,  # Fetch one extra to account for the query object itself
            include_vector=False,
            certainty=0.9,
            return_properties=[
                "image_id", "description", "metadata", "dataset_id", "file_path",
                "preview_image", "tag", "area", "perimeter", "equivalent_diameter",
                "bbox_width", "bbox_height", "aspect_ratio", "circularity",
                "eccentricity", "solidity", "convexity"
            ]
        )
        
        # Extract results and filter out the query object itself
        actual_results = results.objects if hasattr(results, 'objects') else results
        filtered_results = []
        for result in actual_results:
            result_uuid = getattr(result, 'uuid', None) or getattr(result, 'id', None)
            if str(result_uuid) != str(object_uuid):
                filtered_results.append(result)
            if len(filtered_results) >= limit:
                break
        
        logger.info(f"UUID search successful: {len(filtered_results)} results")
        return {
            'success': True,
            'results': filtered_results,
            'count': len(filtered_results),
            'query_type': 'uuid',
            'uuid': object_uuid
        }
        
    except Exception as e:
        logger.error(f"Error in UUID search: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'results': [],
            'count': 0
        }

