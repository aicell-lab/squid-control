"""
Filter Wheel Control for Squid+ Microscope
Based on the official Squid software filter wheel implementation

The filter wheel is controlled via the W axis (4th motor axis) on the microcontroller.
It uses a stepper motor to rotate between filter positions (1-8).
"""

import time
import logging
from typing import Optional
from squid_control.control.config import CONFIG

logger = logging.getLogger(__name__)


class FilterWheelController:
    """
    Controller for filter wheel hardware in Squid+ microscope.
    
    This controls the filter wheel using the W axis of the microcontroller.
    The wheel has 8 positions (1-8) and uses a stepper motor for positioning.
    
    Based on official software: control/filterwheel.py (SquidFilterWheelWrapper)
    """
    
    def __init__(self, microcontroller, is_simulation=False):
        """
        Initialize the filter wheel controller
        
        Args:
            microcontroller: Microcontroller instance for hardware communication
            is_simulation: Whether to run in simulation mode
        """
        if microcontroller is None and not is_simulation:
            raise Exception("Error: microcontroller is needed by the FilterWheelController")
        
        self.microcontroller = microcontroller
        self.is_simulation = is_simulation
        
        # Current filter wheel position (1-8)
        self.current_position = getattr(CONFIG, 'SQUID_FILTERWHEEL_MIN_INDEX', 1)
        self.min_position = getattr(CONFIG, 'SQUID_FILTERWHEEL_MIN_INDEX', 1)
        self.max_position = getattr(CONFIG, 'SQUID_FILTERWHEEL_MAX_INDEX', 8)
        
        # Configuration parameters
        self.screw_pitch_mm = getattr(CONFIG, 'SCREW_PITCH_W_MM', 1.0)
        self.microstepping = getattr(CONFIG, 'MICROSTEPPING_DEFAULT_W', 64)
        self.fullsteps_per_rev = getattr(CONFIG, 'FULLSTEPS_PER_REV_W', 200)
        self.stage_movement_sign = getattr(CONFIG, 'STAGE_MOVEMENT_SIGN_W', 1)
        self.filter_wheel_offset = getattr(CONFIG, 'SQUID_FILTERWHEEL_OFFSET', 0.008)
        
        # PID settings (if encoder is used)
        self.has_encoder = getattr(CONFIG, 'HAS_ENCODER_W', False)
        self.motor_slot_index = getattr(CONFIG, 'SQUID_FILTERWHEEL_MOTORSLOTINDEX', 3)
        self.transitions_per_rev = getattr(CONFIG, 'SQUID_FILTERWHEEL_TRANSITIONS_PER_REVOLUTION', 4000)
        
        if is_simulation:
            logger.info("Filter wheel controller initialized in simulation mode")
        else:
            logger.info("Filter wheel controller initialized with hardware")
            
            # Configure PID if encoder is enabled
            if self.has_encoder:
                pid_p = getattr(CONFIG, 'PID_P_W', 0)
                pid_i = getattr(CONFIG, 'PID_I_W', 0)
                pid_d = getattr(CONFIG, 'PID_D_W', 0)
                encoder_flip = getattr(CONFIG, 'ENCODER_FLIP_DIR_W', False)
                
                self.microcontroller.set_pid_arguments(self.motor_slot_index, pid_p, pid_i, pid_d)
                self.microcontroller.configure_stage_pid(
                    self.motor_slot_index, 
                    self.transitions_per_rev, 
                    encoder_flip
                )
                self.microcontroller.turn_on_stage_pid(self.motor_slot_index)
                logger.info("Filter wheel PID control configured")
    
    def move_w(self, delta_mm: float):
        """
        Move the W axis (filter wheel) by a relative distance in millimeters
        
        Args:
            delta_mm: Distance to move in mm (positive or negative)
        """
        if self.is_simulation:
            logger.debug(f"Simulation: Moving W axis by {delta_mm} mm")
            return
        
        # Convert mm to microsteps
        usteps = int(
            self.stage_movement_sign * delta_mm / 
            (self.screw_pitch_mm / (self.microstepping * self.fullsteps_per_rev))
        )
        
        logger.debug(f"Moving W axis by {delta_mm} mm ({usteps} microsteps)")
        self.microcontroller.move_w_usteps(usteps)
    
    def set_filter_position(self, position: int) -> bool:
        """
        Set the filter wheel to a specific position (1-8)
        
        This is the main method for changing filter positions.
        Matches official software's set_emission() method.
        
        Args:
            position (int): Filter position (1-8)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.min_position <= position <= self.max_position:
            logger.error(f"Invalid filter position: {position}. Must be {self.min_position}-{self.max_position}")
            return False
        
        try:
            if self.is_simulation:
                logger.info(f"Simulation: Setting filter wheel to position {position}")
                self.current_position = position
                return True
            else:
                # Check if we're already at the requested position
                if position == self.current_position:
                    logger.debug(f"Already at filter position {position}")
                    return True
                
                # Calculate movement distance
                position_delta = position - self.current_position
                distance_mm = (position_delta * self.screw_pitch_mm / 
                              (self.max_position - self.min_position + 1))
                
                logger.info(f"Moving filter wheel from position {self.current_position} to {position} ({distance_mm:.3f} mm)")
                
                # Execute the movement
                self.move_w(distance_mm)
                self.microcontroller.wait_till_operation_is_completed()
                
                self.current_position = position
                logger.info(f"✓ Filter wheel moved to position {position}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to set filter position {position}: {e}")
            return False
    
    def get_filter_position(self) -> int:
        """
        Get the current filter wheel position
        
        Returns:
            int: Current filter position (1-8)
        """
        return self.current_position
    
    def next_position(self) -> bool:
        """
        Move to the next filter position
        Matches official software's next_position() method.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.current_position < self.max_position:
            try:
                # Calculate distance for one position step
                distance_mm = self.screw_pitch_mm / (self.max_position - self.min_position + 1)
                
                if self.is_simulation:
                    logger.info(f"Simulation: Moving to next filter position ({self.current_position + 1})")
                    self.current_position += 1
                    return True
                
                logger.info(f"Moving filter wheel to next position ({self.current_position + 1})")
                self.move_w(distance_mm)
                self.microcontroller.wait_till_operation_is_completed()
                self.current_position += 1
                logger.info(f"✓ Filter wheel moved to position {self.current_position}")
                return True
            except Exception as e:
                logger.error(f"Failed to move to next position: {e}")
                return False
        else:
            logger.warning(f"Already at maximum filter position ({self.max_position})")
            return False
    
    def previous_position(self) -> bool:
        """
        Move to the previous filter position
        Matches official software's previous_position() method.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.current_position > self.min_position:
            try:
                # Calculate distance for one position step (negative)
                distance_mm = -(self.screw_pitch_mm / (self.max_position - self.min_position + 1))
                
                if self.is_simulation:
                    logger.info(f"Simulation: Moving to previous filter position ({self.current_position - 1})")
                    self.current_position -= 1
                    return True
                
                logger.info(f"Moving filter wheel to previous position ({self.current_position - 1})")
                self.move_w(distance_mm)
                self.microcontroller.wait_till_operation_is_completed()
                self.current_position -= 1
                logger.info(f"✓ Filter wheel moved to position {self.current_position}")
                return True
            except Exception as e:
                logger.error(f"Failed to move to previous position: {e}")
                return False
        else:
            logger.warning(f"Already at minimum filter position ({self.min_position})")
            return False
    
    def home(self) -> bool:
        """
        Home the filter wheel using the W axis home switch.
        Matches official software's homing() method.
        
        This performs a hardware homing sequence, then applies an offset
        and sets the position to the minimum index (position 1).
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.is_simulation:
                logger.info("Simulation: Homing filter wheel")
                self.current_position = self.min_position
                return True
            
            logger.info("Homing filter wheel using W axis home switch...")
            
            # Execute hardware homing on W axis
            self.microcontroller.home_w()
            
            # Homing needs more timeout time (15 seconds as in official software)
            self.microcontroller.wait_till_operation_is_completed(15)
            
            # Apply offset after homing (small adjustment from home switch position)
            if self.filter_wheel_offset != 0:
                logger.debug(f"Applying filter wheel offset: {self.filter_wheel_offset} mm")
                self.move_w(self.filter_wheel_offset)
                self.microcontroller.wait_till_operation_is_completed()
            
            # Set position to minimum index after homing
            self.current_position = self.min_position
            
            logger.info(f"✓ Filter wheel homed successfully (position {self.current_position})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to home filter wheel: {e}")
            return False
    
    def is_enabled(self) -> bool:
        """
        Check if filter wheel is enabled in configuration
        
        Returns:
            bool: True if filter wheel is enabled
        """
        return getattr(CONFIG, 'FILTER_CONTROLLER_ENABLE', False)
    
    def get_available_positions(self) -> list:
        """
        Get list of available filter positions
        
        Returns:
            list: List of available positions
        """
        return list(range(1, self.max_positions + 1))


class FilterWheelSimulation:
    """
    Simulation mode for filter wheel controller
    """
    
    def __init__(self):
        self.current_position = 1
        self.max_positions = 8
        logger.info("Filter wheel simulation initialized")
    
    def set_filter_position(self, position: int) -> bool:
        """Simulation version of set_filter_position"""
        if not 1 <= position <= self.max_positions:
            return False
        
        logger.info(f"Simulation: Filter wheel moving to position {position}")
        time.sleep(0.1)  # Simulate movement time
        self.current_position = position
        return True
    
    def get_filter_position(self) -> int:
        """Simulation version of get_filter_position"""
        return self.current_position
    
    def next_position(self) -> bool:
        """Simulation version of next_position"""
        if self.current_position < self.max_positions:
            return self.set_filter_position(self.current_position + 1)
        return False
    
    def previous_position(self) -> bool:
        """Simulation version of previous_position"""
        if self.current_position > 1:
            return self.set_filter_position(self.current_position - 1)
        return False
    
    def home(self) -> bool:
        """Simulation version of home"""
        return self.set_filter_position(1)
    
    def is_enabled(self) -> bool:
        """Simulation version of is_enabled"""
        return True  # Always enabled in simulation
    
    def get_available_positions(self) -> list:
        """Simulation version of get_available_positions"""
        return list(range(1, self.max_positions + 1))
