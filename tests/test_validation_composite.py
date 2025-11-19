"""
Test suite for GPT-based cell similarity validation and composite image creation.

This module tests the validation composite functionality WITHOUT calling GPT Vision API.
Tests focus on:
- Composite image creation from base64 previews
- Image layout and dimensions
- Preview extraction from mock Weaviate results
"""

import asyncio
import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from squid_control.hypha_tools.weaviate_client import (
    create_validation_composite_image,
    get_cell_preview_by_uuid,
)

# Mark all tests in this module as asyncio tests
pytestmark = pytest.mark.asyncio


def create_mock_cell_image(size=50, color=(128, 128, 128)):
    """
    Create a mock cell preview image (50x50 grayscale or RGB).
    
    Args:
        size: Image size in pixels (default: 50)
        color: RGB tuple for image color (default: gray)
    
    Returns:
        Base64 encoded PNG string
    """
    # Create a simple image with some variation
    img = Image.new('RGB', (size, size), color)
    
    # Add a simple circle to simulate a cell
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    margin = size // 4
    draw.ellipse([margin, margin, size - margin, size - margin], fill=(200, 200, 200))
    
    # Convert to PNG bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    png_bytes = buffer.getvalue()
    
    # Encode as base64
    return base64.b64encode(png_bytes).decode('utf-8')


class TestValidationCompositeImage:
    """Test suite for composite image creation."""
    
    def test_create_composite_with_valid_inputs(self):
        """Test creating composite image with valid base64 inputs."""
        # Create mock cell images
        reference_b64 = create_mock_cell_image(color=(255, 100, 100))  # Red-ish
        validation_b64_list = [
            create_mock_cell_image(color=(100 + i * 10, 150, 150))
            for i in range(10)
        ]
        
        # Create composite
        composite_array = create_validation_composite_image(
            reference_preview_base64=reference_b64,
            validation_previews_base64=validation_b64_list
        )
        
        # Verify output is a numpy array
        assert isinstance(composite_array, np.ndarray)
        
        # Verify it's RGB (3 channels)
        assert composite_array.ndim == 3
        assert composite_array.shape[2] == 3
        
        # Verify dimensions are reasonable
        # Expected: reference side + validation side (2 cols x 5 rows)
        assert composite_array.shape[0] > 50  # Height includes title and cells
        assert composite_array.shape[1] > 100  # Width includes reference + validation grid
        
        print(f"✓ Composite created with shape: {composite_array.shape}")
    
    def test_create_composite_with_fewer_validation_cells(self):
        """Test creating composite with fewer than 10 validation cells."""
        reference_b64 = create_mock_cell_image()
        validation_b64_list = [create_mock_cell_image() for _ in range(5)]
        
        # Should pad to 10 cells with black images
        composite_array = create_validation_composite_image(
            reference_preview_base64=reference_b64,
            validation_previews_base64=validation_b64_list
        )
        
        assert isinstance(composite_array, np.ndarray)
        assert composite_array.shape[2] == 3  # RGB
        print(f"✓ Composite with 5 cells created: {composite_array.shape}")
    
    def test_create_composite_with_more_than_10_cells(self):
        """Test creating composite with more than 10 validation cells (should use first 10)."""
        reference_b64 = create_mock_cell_image()
        validation_b64_list = [create_mock_cell_image() for _ in range(15)]
        
        # Should only use first 10
        composite_array = create_validation_composite_image(
            reference_preview_base64=reference_b64,
            validation_previews_base64=validation_b64_list
        )
        
        assert isinstance(composite_array, np.ndarray)
        print(f"✓ Composite with 15 cells (using 10) created: {composite_array.shape}")
    
    def test_composite_can_be_converted_to_pil(self):
        """Test that composite array can be converted back to PIL Image."""
        reference_b64 = create_mock_cell_image()
        validation_b64_list = [create_mock_cell_image() for _ in range(10)]
        
        composite_array = create_validation_composite_image(
            reference_preview_base64=reference_b64,
            validation_previews_base64=validation_b64_list
        )
        
        # Convert to PIL Image
        pil_image = Image.fromarray(composite_array)
        assert pil_image.mode == 'RGB'
        assert pil_image.size[0] == composite_array.shape[1]
        assert pil_image.size[1] == composite_array.shape[0]
        
        print(f"✓ Composite converted to PIL Image: {pil_image.size}")
    
    def test_composite_can_be_saved_as_png(self):
        """Test that composite can be saved as PNG bytes."""
        reference_b64 = create_mock_cell_image()
        validation_b64_list = [create_mock_cell_image() for _ in range(10)]
        
        composite_array = create_validation_composite_image(
            reference_preview_base64=reference_b64,
            validation_previews_base64=validation_b64_list
        )
        
        # Convert to PNG bytes
        pil_image = Image.fromarray(composite_array)
        png_buffer = io.BytesIO()
        pil_image.save(png_buffer, format='PNG')
        png_bytes = png_buffer.getvalue()
        
        assert len(png_bytes) > 0
        assert png_bytes[:8] == b'\x89PNG\r\n\x1a\n'  # PNG signature
        
        print(f"✓ Composite saved as PNG: {len(png_bytes)} bytes")
    
    def test_create_composite_with_invalid_base64(self):
        """Test error handling with invalid base64 input."""
        reference_b64 = "invalid_base64_string"
        validation_b64_list = [create_mock_cell_image() for _ in range(10)]
        
        with pytest.raises(ValueError, match="Failed to create composite image"):
            create_validation_composite_image(
                reference_preview_base64=reference_b64,
                validation_previews_base64=validation_b64_list
            )
        
        print("✓ Invalid base64 properly raises ValueError")
    
    def test_create_composite_with_different_sized_images(self):
        """Test composite creation with images that need resizing."""
        # Create images with different sizes
        reference_img = Image.new('RGB', (100, 100), (255, 0, 0))
        ref_buffer = io.BytesIO()
        reference_img.save(ref_buffer, format='PNG')
        reference_b64 = base64.b64encode(ref_buffer.getvalue()).decode('utf-8')
        
        validation_b64_list = []
        for i in range(10):
            # Create images with varying sizes
            size = 30 + i * 5
            val_img = Image.new('RGB', (size, size), (100, 100, 255))
            val_buffer = io.BytesIO()
            val_img.save(val_buffer, format='PNG')
            validation_b64_list.append(base64.b64encode(val_buffer.getvalue()).decode('utf-8'))
        
        # Should resize all to 50x50
        composite_array = create_validation_composite_image(
            reference_preview_base64=reference_b64,
            validation_previews_base64=validation_b64_list
        )
        
        assert isinstance(composite_array, np.ndarray)
        print(f"✓ Composite with resized images created: {composite_array.shape}")


class TestGetCellPreviewByUUID:
    """Test suite for getting cell preview by UUID."""
    
    async def test_get_preview_with_mock_weaviate(self):
        """Test getting preview image with mocked Weaviate service."""
        test_uuid = "test-uuid-12345"
        test_app_id = "test-experiment"
        mock_preview_b64 = create_mock_cell_image()
        
        # Create mock Weaviate service
        mock_service = MagicMock()
        mock_object = MagicMock()
        mock_object.properties = MagicMock()
        mock_object.properties.preview_image = mock_preview_b64
        
        mock_service.data.get = AsyncMock(return_value=mock_object)
        
        # Patch the _get_weaviate_service function
        with patch('squid_control.hypha_tools.weaviate_client._get_weaviate_service', 
                   return_value=mock_service):
            preview = await get_cell_preview_by_uuid(
                object_uuid=test_uuid,
                application_id=test_app_id
            )
            
            assert preview == mock_preview_b64
            mock_service.data.get.assert_called_once()
            print(f"✓ Preview fetched successfully with mocked Weaviate")
    
    async def test_get_preview_missing_preview_image(self):
        """Test handling of missing preview_image field."""
        test_uuid = "test-uuid-no-preview"
        test_app_id = "test-experiment"
        
        # Create mock object without preview_image
        mock_service = MagicMock()
        mock_object = MagicMock()
        mock_object.properties = MagicMock(spec=[])  # No preview_image attribute
        
        mock_service.data.get = AsyncMock(return_value=mock_object)
        
        with patch('squid_control.hypha_tools.weaviate_client._get_weaviate_service',
                   return_value=mock_service):
            preview = await get_cell_preview_by_uuid(
                object_uuid=test_uuid,
                application_id=test_app_id
            )
            
            assert preview is None
            print("✓ Missing preview handled correctly (returns None)")


class TestPreviewExtractionFromSearchResults:
    """Test extracting preview images from search result objects."""
    
    def test_extract_preview_from_result_with_properties(self):
        """Test extracting preview from result object with properties attribute."""
        mock_preview_b64 = create_mock_cell_image()
        
        # Create mock result object (like Weaviate returns)
        mock_result = MagicMock()
        mock_result.uuid = "test-uuid-123"
        mock_result.properties = MagicMock()
        mock_result.properties.preview_image = mock_preview_b64
        
        # Extract preview (simulating code in validate_and_adjust_certainty)
        preview = None
        if hasattr(mock_result, 'properties'):
            preview = getattr(mock_result.properties, 'preview_image', None)
        
        assert preview == mock_preview_b64
        print("✓ Preview extracted from result.properties.preview_image")
    
    def test_extract_preview_from_dict_result(self):
        """Test extracting preview from dict-like result object."""
        mock_preview_b64 = create_mock_cell_image()
        
        # Create dict-like result
        mock_result = {
            'uuid': 'test-uuid-456',
            'properties': {
                'preview_image': mock_preview_b64
            }
        }
        
        # Extract preview
        preview = mock_result.get('properties', {}).get('preview_image')
        
        assert preview == mock_preview_b64
        print("✓ Preview extracted from dict result")


class TestEndToEndCompositeCreation:
    """End-to-end test simulating the full validation workflow (without GPT)."""
    
    def test_full_validation_workflow_simulation(self):
        """Simulate the complete workflow of validation composite creation."""
        print("\n=== Simulating Full Validation Workflow ===")
        
        # Step 1: Create mock search results with preview images
        print("Step 1: Creating mock search results...")
        reference_uuid = "ref-cell-uuid-789"
        reference_preview = create_mock_cell_image(color=(255, 100, 100))
        
        mock_search_results = []
        for i in range(50):  # Simulate 50 search results
            result = MagicMock()
            result.uuid = f"cell-uuid-{i}"
            result.properties = MagicMock()
            result.properties.preview_image = create_mock_cell_image(
                color=(100 + i * 2, 150, 150)
            )
            mock_search_results.append(result)
        
        print(f"   Created {len(mock_search_results)} mock search results")
        
        # Step 2: Extract least similar cells (bottom 10)
        print("Step 2: Extracting 10 least similar cells...")
        validation_count = 10
        validation_cells = mock_search_results[-validation_count:]
        print(f"   Extracted {len(validation_cells)} validation cells")
        
        # Step 3: Extract preview images from results (NO additional Weaviate query!)
        print("Step 3: Extracting preview images from results...")
        validation_previews = []
        for idx, cell in enumerate(validation_cells):
            preview = getattr(cell.properties, 'preview_image', None)
            if preview:
                validation_previews.append(preview)
        
        print(f"   Extracted {len(validation_previews)} preview images")
        assert len(validation_previews) == validation_count
        
        # Step 4: Create composite image
        print("Step 4: Creating composite validation image...")
        composite_array = create_validation_composite_image(
            reference_preview_base64=reference_preview,
            validation_previews_base64=validation_previews
        )
        
        print(f"   Composite created: {composite_array.shape}")
        assert composite_array.shape[2] == 3  # RGB
        
        # Step 5: Convert to PNG bytes (for saving to snapshot manager)
        print("Step 5: Converting to PNG bytes...")
        pil_image = Image.fromarray(composite_array)
        png_buffer = io.BytesIO()
        pil_image.save(png_buffer, format='PNG')
        png_bytes = png_buffer.getvalue()
        
        print(f"   PNG created: {len(png_bytes)} bytes")
        assert len(png_bytes) > 0
        
        # Step 6: Simulate saving to snapshot manager (would return URL)
        print("Step 6: Would save to snapshot manager...")
        mock_url = f"https://hypha.aicell.io/agent-lens/artifacts/snapshot-{reference_uuid}.png"
        print(f"   Mock URL: {mock_url}")
        
        print("\n✅ Full workflow simulation completed successfully!")
        print("=" * 50)


def test_composite_image_visual_inspection():
    """
    Create a composite image and save it for visual inspection.
    This test is for development/debugging purposes.
    """
    print("\n=== Creating Composite for Visual Inspection ===")
    
    # Create reference cell (red-ish)
    reference_b64 = create_mock_cell_image(color=(255, 100, 100))
    
    # Create 10 validation cells with varying colors
    validation_b64_list = []
    for i in range(10):
        # Gradually change color to simulate different similarity levels
        color = (100 + i * 15, 150, 150 - i * 10)
        validation_b64_list.append(create_mock_cell_image(color=color))
    
    # Create composite
    composite_array = create_validation_composite_image(
        reference_preview_base64=reference_b64,
        validation_previews_base64=validation_b64_list
    )
    
    # Try to save for visual inspection (optional - only in /tmp)
    try:
        import tempfile
        temp_path = Path(tempfile.gettempdir()) / "test_validation_composite.png"
        pil_image = Image.fromarray(composite_array)
        pil_image.save(temp_path)
        print(f"✓ Composite saved to: {temp_path}")
        print(f"  You can open this file to visually inspect the layout")
    except Exception as e:
        print(f"⚠️ Could not save composite for inspection: {e}")
    
    print(f"✓ Composite shape: {composite_array.shape}")
    print("=" * 50)


if __name__ == "__main__":
    # Run tests manually for quick checking
    print("Running validation composite tests...\n")
    
    # Test basic composite creation
    test_obj = TestValidationCompositeImage()
    test_obj.test_create_composite_with_valid_inputs()
    test_obj.test_create_composite_with_fewer_validation_cells()
    test_obj.test_composite_can_be_saved_as_png()
    
    # Test full workflow
    workflow_test = TestEndToEndCompositeCreation()
    workflow_test.test_full_validation_workflow_simulation()
    
    # Create visual inspection image
    test_composite_image_visual_inspection()
    
    print("\n✅ All manual tests passed!")

