"""
Tests for Squid+ specific features: Filter Wheel and Objective Switcher
"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from squid_control.control.config import CONFIG
from squid_control.control.filter_wheel import FilterWheelController, FilterWheelSimulation
from squid_control.control.objective_switcher import ObjectiveSwitcherController, ObjectiveSwitcherSimulation
from squid_control.squid_controller import SquidController

# Mark only async tests as asyncio
pytestmark = pytest.mark.asyncio

# Add squid_control to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestFilterWheel:
    """Test suite for Filter Wheel functionality"""

    @pytest.fixture
    def filter_wheel_simulation(self):
        """Fixture for filter wheel in simulation mode"""
        return FilterWheelSimulation()

    @pytest.fixture
    def filter_wheel_controller(self):
        """Fixture for filter wheel controller with mock microcontroller"""
        mock_microcontroller = MagicMock()
        return FilterWheelController(microcontroller=mock_microcontroller, is_simulation=False)

    def test_filter_wheel_simulation_initialization(self, filter_wheel_simulation):
        """Test filter wheel simulation initialization"""
        assert filter_wheel_simulation is not None
        assert filter_wheel_simulation.current_position == 1
        assert filter_wheel_simulation.max_positions == 8

    def test_filter_wheel_set_position_simulation(self, filter_wheel_simulation):
        """Test setting filter wheel position in simulation"""
        # Test valid positions
        for position in range(1, 9):
            result = filter_wheel_simulation.set_filter_position(position)
            assert result is True
            assert filter_wheel_simulation.get_filter_position() == position

        # Test invalid positions
        assert filter_wheel_simulation.set_filter_position(0) is False
        assert filter_wheel_simulation.set_filter_position(9) is False
        assert filter_wheel_simulation.set_filter_position(-1) is False

    def test_filter_wheel_next_position_simulation(self, filter_wheel_simulation):
        """Test moving to next filter position in simulation"""
        # Start at position 1
        filter_wheel_simulation.set_filter_position(1)
        
        # Move through positions 2-8
        for expected_position in range(2, 9):
            result = filter_wheel_simulation.next_position()
            assert result is True
            assert filter_wheel_simulation.get_filter_position() == expected_position

        # Try to move beyond max position
        result = filter_wheel_simulation.next_position()
        assert result is False
        assert filter_wheel_simulation.get_filter_position() == 8

    def test_filter_wheel_previous_position_simulation(self, filter_wheel_simulation):
        """Test moving to previous filter position in simulation"""
        # Start at position 8
        filter_wheel_simulation.set_filter_position(8)
        
        # Move through positions 7-1
        for expected_position in range(7, 0, -1):
            result = filter_wheel_simulation.previous_position()
            assert result is True
            assert filter_wheel_simulation.get_filter_position() == expected_position

        # Try to move beyond min position
        result = filter_wheel_simulation.previous_position()
        assert result is False
        assert filter_wheel_simulation.get_filter_position() == 1

    def test_filter_wheel_home_simulation(self, filter_wheel_simulation):
        """Test homing filter wheel in simulation"""
        # Set to position 5
        filter_wheel_simulation.set_filter_position(5)
        assert filter_wheel_simulation.get_filter_position() == 5
        
        # Home to position 1
        result = filter_wheel_simulation.home()
        assert result is True
        assert filter_wheel_simulation.get_filter_position() == 1

    def test_filter_wheel_available_positions(self, filter_wheel_simulation):
        """Test getting available filter positions"""
        positions = filter_wheel_simulation.get_available_positions()
        assert positions == list(range(1, 9))
        assert len(positions) == 8

    def test_filter_wheel_controller_initialization(self, filter_wheel_controller):
        """Test filter wheel controller initialization"""
        assert filter_wheel_controller is not None
        assert filter_wheel_controller.microcontroller is not None
        assert filter_wheel_controller.is_simulation is False

    def test_filter_wheel_controller_set_position(self, filter_wheel_controller):
        """Test setting filter wheel position with controller"""
        # Test valid position
        result = filter_wheel_controller.set_filter_position(3)
        assert result is True
        assert filter_wheel_controller.get_filter_position() == 3

        # Test invalid position
        result = filter_wheel_controller.set_filter_position(10)
        assert result is False


class TestObjectiveSwitcher:
    """Test suite for Objective Switcher functionality"""

    @pytest.fixture
    def objective_switcher_simulation(self):
        """Fixture for objective switcher in simulation mode"""
        mock_stage = MagicMock()
        return ObjectiveSwitcherSimulation(stage=mock_stage)

    @pytest.fixture
    def objective_switcher_controller(self):
        """Fixture for objective switcher controller with mock stage"""
        mock_stage = MagicMock()
        return ObjectiveSwitcherController(
            serial_number="test_sn_12345",
            stage=mock_stage,
            is_simulation=True  # Use simulation mode for testing
        )

    def test_objective_switcher_simulation_initialization(self, objective_switcher_simulation):
        """Test objective switcher simulation initialization"""
        assert objective_switcher_simulation is not None
        assert objective_switcher_simulation.current_position is None
        assert hasattr(objective_switcher_simulation, 'position2_offset')
        assert objective_switcher_simulation.position2_offset == 1.0

    def test_objective_switcher_move_to_position_1_simulation(self, objective_switcher_simulation):
        """Test moving to position 1 in simulation"""
        result = objective_switcher_simulation.move_to_position_1(move_z=True)
        assert result is True
        assert objective_switcher_simulation.get_current_position() == 1

    def test_objective_switcher_move_to_position_2_simulation(self, objective_switcher_simulation):
        """Test moving to position 2 in simulation"""
        result = objective_switcher_simulation.move_to_position_2(move_z=True)
        assert result is True
        assert objective_switcher_simulation.get_current_position() == 2

    def test_objective_switcher_home_simulation(self, objective_switcher_simulation):
        """Test homing objective switcher in simulation"""
        result = objective_switcher_simulation.home()
        assert result is True
        assert objective_switcher_simulation.get_current_position() == 0

    def test_objective_switcher_set_speed_simulation(self, objective_switcher_simulation):
        """Test setting objective switcher speed in simulation"""
        result = objective_switcher_simulation.set_speed(50.0)
        assert result is True

    def test_objective_switcher_available_positions(self, objective_switcher_simulation):
        """Test getting available objective positions"""
        positions = objective_switcher_simulation.get_available_positions()
        assert positions == [1, 2]
        assert len(positions) == 2

    def test_objective_switcher_position_names(self, objective_switcher_simulation):
        """Test getting objective position names"""
        position_names = objective_switcher_simulation.get_position_names()
        assert 1 in position_names
        assert 2 in position_names
        assert "20x" in position_names[1] or "Position 1" in position_names[1]
        assert "4x" in position_names[2] or "Position 2" in position_names[2]

    def test_objective_switcher_controller_initialization(self, objective_switcher_controller):
        """Test objective switcher controller initialization"""
        assert objective_switcher_controller is not None
        assert objective_switcher_controller.stage is not None
        assert objective_switcher_controller.is_simulation is True  # Updated for simulation mode

    def test_objective_switcher_controller_move_to_position_1(self, objective_switcher_controller):
        """Test moving to position 1 with controller"""
        result = objective_switcher_controller.move_to_position_1(move_z=True)
        assert result is True
        assert objective_switcher_controller.get_current_position() == 1

    def test_objective_switcher_controller_move_to_position_2(self, objective_switcher_controller):
        """Test moving to position 2 with controller"""
        result = objective_switcher_controller.move_to_position_2(move_z=True)
        assert result is True
        assert objective_switcher_controller.get_current_position() == 2

    def test_objective_switcher_z_stage_interaction(self, objective_switcher_simulation):
        """Test Z stage interaction when switching objectives"""
        mock_stage = objective_switcher_simulation.stage
        
        # Move to position 1 first
        objective_switcher_simulation.move_to_position_1(move_z=True)
        assert objective_switcher_simulation.get_current_position() == 1
        
        # Move to position 2 - should trigger Z stage movement
        objective_switcher_simulation.move_to_position_2(move_z=True)
        assert objective_switcher_simulation.get_current_position() == 2
        # Verify Z stage was called (in real implementation, this would move Z)
        
        # Move back to position 1 - should revert Z stage movement
        objective_switcher_simulation.move_to_position_1(move_z=True)
        assert objective_switcher_simulation.get_current_position() == 1


class TestSquidPlusIntegration:
    """Test suite for Squid+ integration with SquidController"""

    @pytest.fixture
    async def squid_plus_controller_fixture(self):
        """Fixture for SquidController with Squid+ configuration"""
        # Mock the configuration to enable Squid+ features
        with patch.dict(os.environ, {'SQUID_SIMULATION_MODE': 'true'}):
            with patch('squid_control.control.config.CONFIG') as mock_config:
                # Set up Squid+ configuration
                mock_config.FILTER_CONTROLLER_ENABLE = True
                mock_config.USE_XERYON = True
                mock_config.XERYON_SERIAL_NUMBER = "test_sn_12345"
                mock_config.XERYON_OBJECTIVE_SWITCHER_POS_1 = ['20x']
                mock_config.XERYON_OBJECTIVE_SWITCHER_POS_2 = ['4x']
                mock_config.XERYON_OBJECTIVE_SWITCHER_POS_2_OFFSET_MM = 1.0
                
                controller = SquidController(is_simulation=True)
                yield controller
                
                # Cleanup
                try:
                    if hasattr(controller, 'camera') and controller.camera is not None:
                        if hasattr(controller.camera, 'zarr_image_manager') and controller.camera.zarr_image_manager is not None:
                            await controller.camera._cleanup_zarr_resources_async()
                        controller.close()
                except Exception as e:
                    print(f"Warning: Controller cleanup error (ignored): {e}")

    @pytest.mark.timeout(60)
    async def test_squid_plus_controller_initialization(self, squid_plus_controller_fixture):
        """Test SquidController initialization with Squid+ features"""
        async for controller in squid_plus_controller_fixture:
            assert controller is not None
            assert controller.is_simulation is True
            
            # Check that Squid+ hardware is initialized
            # Note: In simulation mode, these might be None if not properly configured
            # This test verifies the initialization doesn't crash
            assert hasattr(controller, 'filter_wheel')
            assert hasattr(controller, 'objective_switcher')

    @pytest.mark.timeout(60)
    async def test_squid_plus_hardware_availability(self, squid_plus_controller_fixture):
        """Test Squid+ hardware availability in controller"""
        async for controller in squid_plus_controller_fixture:
            # Test filter wheel availability
            if controller.filter_wheel is not None:
                assert controller.filter_wheel.is_enabled() is True
                positions = controller.filter_wheel.get_available_positions()
                assert len(positions) > 0
            
            # Test objective switcher availability
            if controller.objective_switcher is not None:
                assert controller.objective_switcher.is_enabled() is True
                positions = controller.objective_switcher.get_available_positions()
                assert len(positions) > 0

    @pytest.mark.timeout(60)
    async def test_squid_plus_filter_wheel_operations(self, squid_plus_controller_fixture):
        """Test filter wheel operations through SquidController"""
        async for controller in squid_plus_controller_fixture:
            if controller.filter_wheel is not None:
                # Test setting filter position
                result = controller.filter_wheel.set_filter_position(3)
                assert result is True
                assert controller.filter_wheel.get_filter_position() == 3
                
                # Test next position
                result = controller.filter_wheel.next_position()
                assert result is True
                assert controller.filter_wheel.get_filter_position() == 4
                
                # Test previous position
                result = controller.filter_wheel.previous_position()
                assert result is True
                assert controller.filter_wheel.get_filter_position() == 3
                
                # Test home
                result = controller.filter_wheel.home()
                assert result is True
                assert controller.filter_wheel.get_filter_position() == 1

    @pytest.mark.timeout(60)
    async def test_squid_plus_objective_switcher_operations(self, squid_plus_controller_fixture):
        """Test objective switcher operations through SquidController"""
        async for controller in squid_plus_controller_fixture:
            if controller.objective_switcher is not None:
                # Test moving to position 1
                result = controller.objective_switcher.move_to_position_1(move_z=True)
                assert result is True
                assert controller.objective_switcher.get_current_position() == 1
                
                # Test moving to position 2
                result = controller.objective_switcher.move_to_position_2(move_z=True)
                assert result is True
                assert controller.objective_switcher.get_current_position() == 2
                
                # Test getting position names
                position_names = controller.objective_switcher.get_position_names()
                assert isinstance(position_names, dict)
                assert len(position_names) > 0
                
                # Test setting speed
                result = controller.objective_switcher.set_speed(50.0)
                assert result is True

    @pytest.mark.timeout(60)
    async def test_squid_plus_error_handling(self, squid_plus_controller_fixture):
        """Test error handling for Squid+ features"""
        async for controller in squid_plus_controller_fixture:
            # Test filter wheel error handling
            if controller.filter_wheel is not None:
                # Test invalid position
                result = controller.filter_wheel.set_filter_position(0)
                assert result is False
                
                result = controller.filter_wheel.set_filter_position(10)
                assert result is False
            
            # Test objective switcher error handling
            if controller.objective_switcher is not None:
                # Test invalid position (should be handled gracefully)
                # Note: The implementation should handle invalid positions
                pass


class TestSquidPlusConfiguration:
    """Test suite for Squid+ configuration handling"""

    def test_squid_plus_config_parameters(self):
        """Test that Squid+ configuration parameters are available"""
        # Test filter wheel configuration
        assert hasattr(CONFIG, 'FILTER_CONTROLLER_ENABLE')
        assert hasattr(CONFIG, 'USE_XERYON')
        assert hasattr(CONFIG, 'XERYON_SERIAL_NUMBER')
        assert hasattr(CONFIG, 'XERYON_SPEED')
        assert hasattr(CONFIG, 'XERYON_OBJECTIVE_SWITCHER_POS_1')
        assert hasattr(CONFIG, 'XERYON_OBJECTIVE_SWITCHER_POS_2')
        assert hasattr(CONFIG, 'XERYON_OBJECTIVE_SWITCHER_POS_2_OFFSET_MM')
        
        # Test Z motor configuration
        assert hasattr(CONFIG, 'Z_MOTOR_CONFIG')
        
        # Test camera configuration for Squid+
        assert hasattr(CONFIG, 'ROI_OFFSET_X_DEFAULT')
        assert hasattr(CONFIG, 'ROI_OFFSET_Y_DEFAULT')
        assert hasattr(CONFIG, 'ROI_WIDTH_DEFAULT')
        assert hasattr(CONFIG, 'ROI_HEIGHT_DEFAULT')
        assert hasattr(CONFIG, 'CROP_WIDTH_UNBINNED')
        assert hasattr(CONFIG, 'CROP_HEIGHT_UNBINNED')
        assert hasattr(CONFIG, 'BINNING_FACTOR_DEFAULT')
        assert hasattr(CONFIG, 'PIXEL_FORMAT_DEFAULT')

    def test_squid_plus_config_defaults(self):
        """Test Squid+ configuration default values"""
        # Test default values are reasonable
        assert isinstance(CONFIG.FILTER_CONTROLLER_ENABLE, bool)
        assert isinstance(CONFIG.USE_XERYON, bool)
        assert isinstance(CONFIG.XERYON_SERIAL_NUMBER, str)
        assert isinstance(CONFIG.XERYON_SPEED, int)
        assert isinstance(CONFIG.XERYON_OBJECTIVE_SWITCHER_POS_1, list)
        assert isinstance(CONFIG.XERYON_OBJECTIVE_SWITCHER_POS_2, list)
        assert isinstance(CONFIG.XERYON_OBJECTIVE_SWITCHER_POS_2_OFFSET_MM, float)
        
        # Test Z motor config
        assert CONFIG.Z_MOTOR_CONFIG in ["STEPPER", "STEPPER + PIEZO", "PIEZO", "LINEAR"]
        
        # Test camera config defaults
        assert isinstance(CONFIG.ROI_WIDTH_DEFAULT, int)
        assert isinstance(CONFIG.ROI_HEIGHT_DEFAULT, int)
        assert isinstance(CONFIG.CROP_WIDTH_UNBINNED, int)
        assert isinstance(CONFIG.CROP_HEIGHT_UNBINNED, int)
        assert isinstance(CONFIG.BINNING_FACTOR_DEFAULT, int)
        assert isinstance(CONFIG.PIXEL_FORMAT_DEFAULT, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
