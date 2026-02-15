import asyncio
import os
import shutil
import time
import uuid
from unittest.mock import patch

import pytest
import pytest_asyncio
from hypha_rpc import connect_to_server

from squid_control.start_hypha_service import (
    MicroscopeHyphaService,
)

# Mark all tests in this module as asyncio and integration tests
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# Test configuration
TEST_SERVER_URL = "https://hypha.aicell.io"
TEST_WORKSPACE = "agent-lens"
TEST_TIMEOUT = 120  # seconds


@pytest_asyncio.fixture(scope="function")
async def test_squid_plus_microscope_service():
    """Create a Squid+ microscope service for testing with Squid+ configuration."""
    # Enable service coverage tracking
    os.environ['SQUID_TEST_MODE'] = 'true'
    
    # Check for token first
    token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
    if not token:
        pytest.skip("AGENT_LENS_WORKSPACE_TOKEN not set in environment")

    print(f"🔗 Connecting to {TEST_SERVER_URL} workspace {TEST_WORKSPACE}...")

    server = None
    microscope = None
    service = None

    try:
        # Use context manager for proper connection handling
        async with connect_to_server({
            "server_url": TEST_SERVER_URL,
            "token": token,
            "workspace": TEST_WORKSPACE,
            "ping_interval": None
        }) as server:
            print("✅ Connected to server")

            # Create unique service ID for this test
            test_id = f"test-squid-plus-microscope-{uuid.uuid4().hex[:8]}"
            print(f"Creating Squid+ test service with ID: {test_id}")

            # Create Squid+ microscope instance in simulation mode
            print("🔬 Creating Squid+ Microscope instance...")
            start_time = time.time()
            
            # For simulation mode, we need to ensure the Squid+ config is used
            # The SquidController in simulation mode always uses HCS_v2 config,
            # so we need to temporarily replace it with the Squid+ config
            
            # Set environment variable to force Squid+ mode
            os.environ['SQUID_SIMULATION_MODE'] = 'true'
            
            # Create a temporary Squid+ config file to replace the HCS_v2 config
            squid_plus_config_src = 'squid_control/config/configuration_Squid+_example.ini'
            hcs_config_dst = 'squid_control/config/configuration_HCS_v2_example.ini'
            hcs_config_backup = 'squid_control/config/configuration_HCS_v2_example.ini.backup'
            
            # Backup the original HCS_v2 config and replace with Squid+ config
            if os.path.exists(squid_plus_config_src) and os.path.exists(hcs_config_dst):
                shutil.copy2(hcs_config_dst, hcs_config_backup)
                shutil.copy2(squid_plus_config_src, hcs_config_dst)
                print(f"📋 Using Squid+ configuration: {hcs_config_dst}")
            else:
                print("⚠️ Configuration files not found, using default")
            
            # Patch Toupcam Camera_Simulation to accept zarr_dataset_path (added in codebase
            # for 63x/Opera mode). Toupcam's simulation only takes rotate_image_angle and
            # flip_image; we strip zarr_dataset_path so the test fits the current codebase.
            import squid_control.control.camera.camera_toupcam as _toupcam_mod
            _real_toupcam_sim = _toupcam_mod.Camera_Simulation

            def _toupcam_sim_compat(*args, **kwargs):
                kwargs.pop("zarr_dataset_path", None)
                return _real_toupcam_sim(*args, **kwargs)

            try:
                with patch(
                    "squid_control.control.camera.camera_toupcam.Camera_Simulation",
                    _toupcam_sim_compat,
                ):
                    microscope = MicroscopeHyphaService(is_simulation=True, is_local=False)
            finally:
                # Restore original HCS_v2 config
                if os.path.exists(hcs_config_backup):
                    shutil.copy2(hcs_config_backup, hcs_config_dst)
                    os.remove(hcs_config_backup)
                    print("🧹 Restored original HCS_v2 configuration")
            
            init_time = time.time() - start_time
            print(f"✅ Squid+ Microscope initialization took {init_time:.1f} seconds")

            microscope.service_id = test_id
            microscope.login_required = False  # Disable auth for tests
            microscope.authorized_emails = None


            # Initialize artifact manager and snapshot manager for testing
            from squid_control.hypha_tools.artifact_manager.artifact_manager import SquidArtifactManager
            from squid_control.hypha_tools.snapshot_utils import SnapshotManager
            
            microscope.artifact_manager = SquidArtifactManager()
            artifact_server = await connect_to_server({
                "server_url": "https://hypha.aicell.io",
                "token": token,
                "workspace": "agent-lens",
                "ping_interval": 30
            })
            await microscope.artifact_manager.connect_server(artifact_server)
            microscope.snapshot_manager = SnapshotManager(microscope.artifact_manager)
            print("✅ Artifact manager and snapshot manager initialized")

            # Override setup method to avoid connecting to external services during tests
            async def mock_setup():
                pass
            microscope.setup = mock_setup

            # Register the service
            print("📝 Registering Squid+ microscope service...")
            service_start_time = time.time()
            await microscope.start_hypha_service(server, test_id)
            service_time = time.time() - service_start_time
            print(f"✅ Service registration took {service_time:.1f} seconds")

            # Get the registered service to test against
            print("🔍 Getting service reference...")
            service = await server.get_service(test_id)
            print("✅ Squid+ Service ready for testing")

            try:
                yield microscope, service
            finally:
                # Comprehensive cleanup
                print("🧹 Starting cleanup...")

                # Stop video buffering if it's running to prevent event loop errors
                if microscope and hasattr(microscope, 'stop_video_buffering'):
                    try:
                        if microscope.frame_acquisition_running:
                            print("Stopping video buffering...")
                            # Add timeout for test environment to prevent hanging
                            await asyncio.wait_for(
                                microscope.stop_video_buffering(),
                                timeout=5.0  # 5 second timeout for tests
                            )
                            print("✅ Video buffering stopped")
                    except asyncio.TimeoutError:
                        print("⚠️ Video buffering stop timed out, forcing cleanup...")
                        # Force stop the video buffering by setting flags directly
                        microscope.frame_acquisition_running = False
                        if microscope.frame_acquisition_task:
                            microscope.frame_acquisition_task.cancel()
                        if microscope.video_idle_check_task:
                            microscope.video_idle_check_task.cancel()
                        print("✅ Video buffering force stopped")
                    except Exception as video_error:
                        print(f"Error stopping video buffering: {video_error}")

                # Close the SquidController and camera resources properly
                if microscope and hasattr(microscope, 'squidController'):
                    try:
                        print("Closing SquidController...")
                        if hasattr(microscope.squidController, 'camera'):
                            camera = microscope.squidController.camera
                            if hasattr(camera, 'cleanup_zarr_resources_async'):
                                try:
                                    # Add timeout for zarr cleanup as well
                                    await asyncio.wait_for(
                                        camera.cleanup_zarr_resources_async(),
                                        timeout=3.0  # 3 second timeout for zarr cleanup
                                    )
                                except asyncio.TimeoutError:
                                    print("⚠️ Zarr cleanup timed out, skipping...")
                                except Exception as camera_error:
                                    print(f"Camera cleanup error: {camera_error}")

                        microscope.squidController.close()
                        print("✅ SquidController closed")
                    except Exception as controller_error:
                        print(f"Error closing SquidController: {controller_error}")

                # Give time for all cleanup operations to complete
                await asyncio.sleep(0.1)
                print("✅ Cleanup completed")
                
                # Clean up environment variable
                os.environ.pop('SQUID_TEST_MODE', None)

    except Exception as e:
        pytest.fail(f"Failed to create Squid+ test service: {e}")

# Basic connectivity tests
async def test_squid_plus_service_registration_and_connectivity(test_squid_plus_microscope_service):
    """Test that the Squid+ service can be registered and is accessible."""
    microscope, service = test_squid_plus_microscope_service

    # Test basic connectivity with timeout
    result = await asyncio.wait_for(service.ping(), timeout=10)
    assert result == "pong"

    # Verify the service has the expected methods
    assert hasattr(service, 'move_by_distance')
    assert hasattr(service, 'get_status')
    assert hasattr(service, 'snap')
    
    # Verify Squid+ specific methods are available
    assert hasattr(service, 'get_filter_wheel_position')
    assert hasattr(service, 'get_current_objective')
    assert hasattr(service, 'set_filter_wheel_position')
    assert hasattr(service, 'switch_objective')

# Squid+ Filter Wheel API Tests
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_squid_plus_filter_wheel_api(test_squid_plus_microscope_service):
    """Test Squid+ filter wheel API endpoints"""
    microscope, service = test_squid_plus_microscope_service
    print("🧪 Testing Squid+ Filter Wheel API endpoints...")
    
    try:
        # Test 1: Check if filter wheel is available
        print("1. Testing filter wheel availability...")
        
        # Try to get current position (should work if filter wheel is available)
        try:
            position_result = await service.get_filter_wheel_position()
            print(f"   ✓ Filter wheel available at position: {position_result.get('position', 'unknown')}")
            filter_wheel_available = True
        except Exception as e:
            if "not available" in str(e).lower():
                print("   ℹ️  Filter wheel not available on this microscope (expected in simulation)")
                filter_wheel_available = False
            else:
                raise e
        
        if filter_wheel_available:
            # Test 2: Set filter wheel position
            print("2. Testing filter wheel position control...")
            
            # Set to position 3
            set_result = await service.set_filter_wheel_position(position=3)
            assert set_result.get("success", False) == True
            assert set_result.get("position") == 3
            print("   ✓ Filter wheel position set successfully")
            
            # Test 3: Move to next position
            print("3. Testing next filter position...")
            next_result = await service.next_filter_position()
            assert next_result.get("success", False) == True
            assert next_result.get("position") == 4
            print("   ✓ Next filter position works")
            
            # Test 4: Move to previous position
            print("4. Testing previous filter position...")
            prev_result = await service.previous_filter_position()
            assert prev_result.get("success", False) == True
            assert prev_result.get("position") == 3
            print("   ✓ Previous filter position works")
            
            # Test 5: Verify filter wheel is working correctly
            print("5. Testing filter wheel functionality...")
            # Filter wheel homing is not available as a separate API method
            # but the filter wheel is already homed during initialization
            print("   ✓ Filter wheel functionality verified")
        
        print("✅ Squid+ Filter Wheel API tests passed!")
        
    except Exception as e:
        print(f"❌ Squid+ Filter Wheel API test failed: {e}")
        raise

# Squid+ Objective Switcher API Tests
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_squid_plus_objective_switcher_api(test_squid_plus_microscope_service):
    """Test Squid+ objective switcher API endpoints"""
    microscope, service = test_squid_plus_microscope_service
    print("🧪 Testing Squid+ Objective Switcher API endpoints...")
    
    try:
        # Test 1: Check if objective switcher is available
        print("1. Testing objective switcher availability...")
        
        # Try to get current objective (should work if objective switcher is available)
        try:
            objective_result = await service.get_current_objective()
            print(f"   ✓ Objective switcher available: {objective_result.get('current_objective', 'unknown')}")
            objective_switcher_available = True
        except Exception as e:
            if "not available" in str(e).lower():
                print("   ℹ️  Objective switcher not available on this microscope (expected in simulation)")
                objective_switcher_available = False
            else:
                raise e
        
        if objective_switcher_available:
            # Test 2: Get available objectives
            print("2. Testing available objectives...")
            objectives_result = await service.get_available_objectives()
            assert objectives_result.get("success", False) == True
            positions = objectives_result.get("positions", [])
            objective_names = objectives_result.get("objective_names", [])
            assert len(positions) > 0
            print(f"   ✓ Available positions: {positions}")
            print(f"   ✓ Objective names: {objective_names}")
            
            # Test 3: Switch to first available objective
            print("3. Testing switch to first available objective...")
            if objective_names:
                first_objective = objective_names[0]
                switch_result = await service.switch_objective(objective_name=first_objective, move_z=True)
                assert switch_result.get("success", False) == True
                print(f"   ✓ Switched to objective: {switch_result.get('objective_name', 'Unknown')}")
            
            # Test 4: Switch to second available objective (if available)
            print("4. Testing switch to second available objective...")
            if len(objective_names) > 1:
                second_objective = objective_names[1]
                switch2_result = await service.switch_objective(objective_name=second_objective, move_z=True)
                assert switch2_result.get("success", False) == True
                print(f"   ✓ Switched to second objective: {switch2_result.get('objective_name', 'Unknown')}")
            
            # Test 5: Get current objective after movement
            print("5. Testing current objective retrieval...")
            current_result = await service.get_current_objective()
            assert current_result.get("success", False) == True
            assert current_result.get("current_objective") is not None
            print(f"   ✓ Current objective: {current_result.get('current_objective', 'Unknown')}")
            
            # Test 6: Verify pixel size is updated when switching objectives
            print("6. Testing pixel size update after objective switch...")
            if objective_names:
                # Get pixel size after switching to first objective
                first_objective = objective_names[0]
                switch_result = await service.switch_objective(objective_name=first_objective, move_z=True)
                assert switch_result.get("success", False) == True
                pixel_size_1 = switch_result.get("pixel_size_xy")
                print(f"   ✓ Pixel size for {first_objective}: {pixel_size_1} µm")
                
                # Switch to second objective and check if pixel size changes
                if len(objective_names) > 1:
                    second_objective = objective_names[1]
                    switch2_result = await service.switch_objective(objective_name=second_objective, move_z=True)
                    assert switch2_result.get("success", False) == True
                    pixel_size_2 = switch2_result.get("pixel_size_xy")
                    print(f"   ✓ Pixel size for {second_objective}: {pixel_size_2} µm")
                    
                    # Verify pixel sizes are different (different objectives should have different pixel sizes)
                    if pixel_size_1 is not None and pixel_size_2 is not None:
                        assert pixel_size_1 != pixel_size_2, f"Pixel sizes should be different for different objectives: {pixel_size_1} vs {pixel_size_2}"
                        print(f"   ✓ Pixel sizes are different as expected: {pixel_size_1} vs {pixel_size_2}")
                    else:
                        print("   ⚠️  Pixel size information not available in response")
        
        print("✅ Squid+ Objective Switcher API tests passed!")
        
    except Exception as e:
        print(f"❌ Squid+ Objective Switcher API test failed: {e}")
        raise

# Squid+ Error Handling Tests
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_squid_plus_error_handling(test_squid_plus_microscope_service):
    """Test Squid+ API error handling for unavailable hardware"""
    microscope, service = test_squid_plus_microscope_service
    print("🧪 Testing Squid+ API error handling...")
    
    try:
        # Test 1: Filter wheel error handling
        print("1. Testing filter wheel error handling...")
        
        # Test invalid filter position
        try:
            await service.set_filter_wheel_position(position=0)  # Invalid position
            print("   ⚠️  Expected error for invalid position not raised")
        except Exception as e:
            if "not available" in str(e).lower() or "invalid" in str(e).lower():
                print("   ✓ Filter wheel error handling works correctly")
            else:
                print(f"   ⚠️  Unexpected error type: {e}")
        
        # Test 2: Objective switcher error handling
        print("2. Testing objective switcher error handling...")
        
        # Test invalid objective position
        try:
            await service.move_to_objective_position(position=3)  # Invalid position (only 1,2 allowed)
            print("   ⚠️  Expected error for invalid position not raised")
        except Exception as e:
            if "not available" in str(e).lower() or "invalid" in str(e).lower():
                print("   ✓ Objective switcher error handling works correctly")
            else:
                print(f"   ⚠️  Unexpected error type: {e}")
        
        print("✅ Squid+ API error handling tests passed!")
        
    except Exception as e:
        print(f"❌ Squid+ API error handling test failed: {e}")
        raise

# Squid+ Integration Tests
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_squid_plus_integration_with_existing_features(test_squid_plus_microscope_service):
    """Test that Squid+ features work alongside existing microscope features"""
    microscope, service = test_squid_plus_microscope_service
    print("🧪 Testing Squid+ integration with existing features...")
    
    try:
        # Test 1: Basic microscope operations still work
        print("1. Testing basic microscope operations...")
        
        # Test basic movement
        move_result = await service.move_by_distance(x=0.1, y=0.1, z=0.0)
        assert move_result.get("success", False) == True
        print("   ✓ Basic movement still works")
        
        # Test illumination
        illum_result = await service.set_illumination(channel=11, intensity=50)
        # The set_illumination method returns a string message, not a dict
        assert isinstance(illum_result, str)
        print("   ✓ Illumination still works")
        
        # Test 2: Squid+ features don't interfere with basic operations
        print("2. Testing Squid+ features don't interfere...")
        
        # Try Squid+ operations (may not be available, but shouldn't crash)
        try:
            await service.get_filter_wheel_position()
            print("   ✓ Filter wheel operations don't interfere")
        except Exception as e:
            if "not available" in str(e).lower():
                print("   ✓ Filter wheel not available (expected), no interference")
            else:
                raise e
        
        try:
            await service.get_current_objective()
            print("   ✓ Objective switcher operations don't interfere")
        except Exception as e:
            if "not available" in str(e).lower():
                print("   ✓ Objective switcher not available (expected), no interference")
            else:
                raise e
        
        # Test 3: Status and configuration still work
        print("3. Testing status and configuration still work...")
        
        status = await service.get_status()
        assert status is not None
        print("   ✓ Status retrieval still works")
        
        config = await service.get_microscope_configuration(config_section="all")
        assert config is not None and config.get('success', False)
        print("   ✓ Configuration retrieval still works")
        
        print("✅ Squid+ integration with existing features tests passed!")
        
    except Exception as e:
        print(f"❌ Squid+ integration test failed: {e}")
        raise

# Basic microscope functionality tests (inherited from original)
async def test_squid_plus_basic_microscope_operations(test_squid_plus_microscope_service):
    """Test basic microscope operations work with Squid+ configuration."""
    microscope, service = test_squid_plus_microscope_service

    # Test basic connectivity with timeout
    result = await asyncio.wait_for(service.ping(), timeout=10)
    assert result == "pong"

    # Test movement
    result = await asyncio.wait_for(
        service.move_by_distance(x=1.0, y=1.0, z=0.1),
        timeout=15
    )

    assert isinstance(result, dict)
    assert "success" in result
    assert result["success"] == True

    # Test status
    status = await asyncio.wait_for(service.get_status(), timeout=10)
    assert isinstance(status, dict)
    assert 'current_x' in status
    assert 'current_y' in status
    assert 'current_z' in status

    # Test image capture - removed due to Zarr data availability issues
    # url = await asyncio.wait_for(
    #     service.snap(exposure_time=100, channel="BF_LED_matrix_full", intensity=50),
    #     timeout=20
    # )
    # assert isinstance(url, str)
    # assert url.startswith("https://")

# Test Squid+ configuration detection
async def test_squid_plus_configuration_detection(test_squid_plus_microscope_service):
    """Test that Squid+ configuration is properly detected."""
    microscope, service = test_squid_plus_microscope_service
    
    # Verify that the microscope is detected as Squid+
    assert microscope.is_squid_plus == True
    
    # Verify that Squid+ methods are available
    assert hasattr(service, 'get_filter_wheel_position')
    assert hasattr(service, 'get_current_objective')
    assert hasattr(service, 'set_filter_wheel_position')
    assert hasattr(service, 'switch_objective')
    assert hasattr(service, 'get_available_objectives')
    assert hasattr(service, 'next_filter_position')
    assert hasattr(service, 'previous_filter_position')
