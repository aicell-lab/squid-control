"""
Weaviate client module for similarity search integration.

This module provides async functions for interacting with the Weaviate vector database
through the agent-lens service API endpoints.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Agent-lens service configuration
AGENT_LENS_BASE_URL = "https://hypha.aicell.io/agent-lens/apps/agent-lens"
WEAVIATE_COLLECTION_NAME = "Agentlens"


async def delete_weaviate_application(
    application_id: str,
    collection_name: str = WEAVIATE_COLLECTION_NAME,
    base_url: str = AGENT_LENS_BASE_URL
) -> Dict[str, Any]:
    """
    Delete existing Weaviate application if it exists.
    
    Args:
        application_id: Unique identifier for the application (e.g., experiment name)
        collection_name: Weaviate collection name (default: "Agentlens")
        base_url: Base URL for agent-lens service
    
    Returns:
        Dict with 'success' (bool) and optional 'deleted_count' (int) or 'error' (str)
    """
    url = f"{base_url}/similarity/applications/delete"
    params = {
        "collection_name": collection_name,
        "application_id": application_id
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    result = await response.json()
                    deleted_count = result.get('result', {}).get('successful', 0)
                    logger.info(f"Deleted Weaviate application '{application_id}' ({deleted_count} objects removed)")
                    return {'success': True, 'deleted_count': deleted_count}
                elif response.status == 404:
                    # Application doesn't exist - this is OK
                    logger.info(f"Weaviate application '{application_id}' does not exist (will create new one)")
                    return {'success': True, 'deleted_count': 0}
                else:
                    error_text = await response.text()
                    # Check if error message indicates application doesn't exist
                    if 'does not exist' in error_text.lower():
                        logger.info(f"Weaviate application '{application_id}' does not exist (will create new one)")
                        return {'success': True, 'deleted_count': 0}
                    else:
                        logger.warning(f"Failed to delete Weaviate application: {response.status} - {error_text}")
                        return {'success': False, 'error': f"HTTP {response.status}: {error_text}"}
    except Exception as e:
        logger.error(f"Error deleting Weaviate application: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


async def create_weaviate_application(
    application_id: str,
    description: str = "",
    collection_name: str = WEAVIATE_COLLECTION_NAME,
    base_url: str = AGENT_LENS_BASE_URL
) -> Dict[str, Any]:
    """
    Create new Weaviate application.
    
    Args:
        application_id: Unique identifier for the application
        description: Optional description for the application
        collection_name: Weaviate collection name (default: "Agentlens")
        base_url: Base URL for agent-lens service
    
    Returns:
        Dict with 'success' (bool) and optional 'error' (str)
    """
    url = f"{base_url}/similarity/collections"
    
    # Prepare query parameters
    params = {
        "collection_name": collection_name,
        "application_id": application_id
    }
    if description:
        params["description"] = description
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status in (200, 201):
                    logger.info(f"Created Weaviate application '{application_id}'")
                    return {'success': True}
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to create Weaviate application: {response.status} - {error_text}")
                    return {'success': False, 'error': f"HTTP {response.status}: {error_text}"}
    except Exception as e:
        logger.error(f"Error creating Weaviate application: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


async def set_current_weaviate_application(
    application_id: str,
    base_url: str = AGENT_LENS_BASE_URL
) -> Dict[str, Any]:
    """
    Set current Weaviate application.
    
    Note: This is a non-critical operation - failures are logged but don't stop the workflow.
    
    Args:
        application_id: Application ID to set as current
        base_url: Base URL for agent-lens service
    
    Returns:
        Dict with 'success' (bool) and optional 'error' (str)
    """
    url = f"{base_url}/similarity/current-application"
    params = {"application_id": application_id}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    logger.info(f"Set current Weaviate application to '{application_id}'")
                    return {'success': True}
                else:
                    error_text = await response.text()
                    logger.warning(f"Failed to set current application (non-critical): {response.status} - {error_text}")
                    return {'success': False, 'error': f"HTTP {response.status}: {error_text}"}
    except Exception as e:
        logger.warning(f"Error setting current application (non-critical): {e}")
        return {'success': False, 'error': str(e)}


async def batch_upload_to_weaviate(
    objects: List[Dict[str, Any]],
    application_id: str,
    collection_name: str = WEAVIATE_COLLECTION_NAME,
    base_url: str = AGENT_LENS_BASE_URL,
    retry_attempts: int = 2
) -> Dict[str, Any]:
    """
    Upload batch of objects to Weaviate using insert-many endpoint.
    
    Args:
        objects: List of object dictionaries with required fields:
            - image_id: Unique identifier
            - description: Text description
            - metadata: Dict with annotation info
            - dataset_id: Dataset identifier
            - vector: Embedding vector (list of floats)
            - preview_image: Optional base64 PNG string
        application_id: Application ID for the upload
        collection_name: Weaviate collection name (default: "Agentlens")
        base_url: Base URL for agent-lens service
        retry_attempts: Number of retry attempts on failure (default: 2)
    
    Returns:
        Dict with 'success' (bool), 'uploaded_count' (int), and optional 'error' (str)
    """
    if not objects:
        return {'success': True, 'uploaded_count': 0}
    
    url = f"{base_url}/similarity/insert-many"
    
    # Validate objects
    for idx, obj in enumerate(objects):
        required_fields = ['image_id', 'description', 'metadata', 'dataset_id', 'vector']
        missing_fields = [f for f in required_fields if f not in obj]
        if missing_fields:
            logger.error(f"Object {idx} missing required fields: {missing_fields}")
            return {'success': False, 'error': f"Object {idx} missing fields: {missing_fields}"}
    
    # Prepare query parameters
    params = {
        'collection_name': collection_name,
        'application_id': application_id
    }
    
    # Prepare form data (only objects_json)
    form_data = aiohttp.FormData()
    form_data.add_field('objects_json', json.dumps(objects), content_type='application/json')
    
    # Retry logic
    last_error = None
    for attempt in range(retry_attempts + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    params=params,
                    data=form_data, 
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully uploaded {len(objects)} objects to Weaviate")
                        return {'success': True, 'uploaded_count': len(objects)}
                    else:
                        error_text = await response.text()
                        last_error = f"HTTP {response.status}: {error_text}"
                        logger.warning(f"Upload attempt {attempt + 1} failed: {last_error}")
                        
                        # Don't retry on 4xx errors (client errors)
                        if 400 <= response.status < 500:
                            break
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Upload attempt {attempt + 1} failed with exception: {e}")
    
    logger.error(f"Failed to upload batch after {retry_attempts + 1} attempts: {last_error}")
    return {'success': False, 'error': last_error, 'uploaded_count': 0}


async def setup_weaviate_application(
    application_id: str,
    description: str = "",
    collection_name: str = WEAVIATE_COLLECTION_NAME,
    base_url: str = AGENT_LENS_BASE_URL,
    reset_application: bool = True
) -> Dict[str, Any]:
    """
    Complete setup workflow: optionally delete existing, create new, and set current application.
    
    Args:
        application_id: Unique identifier for the application
        description: Optional description
        collection_name: Weaviate collection name
        base_url: Base URL for agent-lens service
        reset_application: If True, delete existing application before creating (default: True)
    
    Returns:
        Dict with 'success' (bool) and optional 'error' (str)
    """
    # Step 1: Delete existing application if reset requested
    if reset_application:
        delete_result = await delete_weaviate_application(application_id, collection_name, base_url)
        if not delete_result['success']:
            logger.warning(f"Delete failed but continuing: {delete_result.get('error')}")
    
    # Step 2: Create new application
    create_result = await create_weaviate_application(application_id, description, collection_name, base_url)
    if not create_result['success']:
        return create_result
    
    # Step 3: Set as current application (non-critical)
    await set_current_weaviate_application(application_id, base_url)
    
    return {'success': True}


async def search_similar_by_uuid(
    object_uuid: str,
    application_id: Optional[str] = None,
    limit: int = 10,
    base_url: str = AGENT_LENS_BASE_URL
) -> Dict[str, Any]:
    """
    Search for similar images by UUID.
    
    Args:
        object_uuid: UUID of the object to search for similar items
        application_id: Application ID (optional, uses current active if not provided)
        limit: Maximum number of results to return (default: 10)
        base_url: Base URL for agent-lens service
    
    Returns:
        Dict with 'success' (bool), 'results' (list), 'count' (int), and optional 'error' (str)
    """
    url = f"{base_url}/similarity/search/text"
    
    # Prepare query with uuid: prefix (no space after colon)
    query_text = f"uuid:{object_uuid}"
    
    # Prepare query parameters (FastAPI POST with query params)
    params = {
        "query_text": query_text,
        "limit": limit
    }
    if application_id:
        params["application_id"] = application_id
    
    logger.info(f"Searching for UUID: {object_uuid} with query: {query_text}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"UUID search successful: {result.get('count', 0)} results")
                    return {
                        'success': True,
                        'results': result.get('results', []),
                        'count': result.get('count', 0),
                        'query_type': 'uuid',
                        'uuid': object_uuid
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"UUID search failed: {response.status} - {error_text}")
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}",
                        'results': [],
                        'count': 0
                    }
    except Exception as e:
        logger.error(f"Error in UUID search: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'results': [],
            'count': 0
        }

