"""
Filter Wheel Control for Squid+ Microscope
Based on the official Squid software filter wheel implementation
"""

import time
import logging
from typing import Optional
from squid_control.control.config import CONFIG

logger = logging.getLogger(__name__)


class FilterWheelController:
    """
    Controller for filter wheel hardware in Squid+ microscope
    Supports both real hardware and simulation modes
    """
    
    def __init__(self, microcontroller=None, is_simulation=False):
        self.microcontroller = microcontroller
        self.is_simulation = is_simulation
        self.current_position = 1
        self.max_positions = 8  # Default to 8 filter positions
        
        if is_simulation:
            logger.info("Filter wheel controller initialized in simulation mode")
        else:
            logger.info("Filter wheel controller initialized with hardware")
    
    def set_filter_position(self, position: int) -> bool:
        """
        Set the filter wheel to a specific position
        
        Args:
            position (int): Filter position (1-8)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not 1 <= position <= self.max_positions:
            logger.error(f"Invalid filter position: {position}. Must be 1-{self.max_positions}")
            return False
        
        try:
            if self.is_simulation:
                logger.info(f"Simulation: Setting filter wheel to position {position}")
                self.current_position = position
                return True
            else:
                # Real hardware control would go here
                # This would interface with the microcontroller
                logger.info(f"Setting filter wheel to position {position}")
                self.current_position = position
                return True
                
        except Exception as e:
            logger.error(f"Failed to set filter position {position}: {e}")
            return False
    
    def get_filter_position(self) -> int:
        """
        Get the current filter wheel position
        
        Returns:
            int: Current filter position
        """
        if self.is_simulation:
            return self.current_position
        else:
            # Real hardware query would go here
            return self.current_position
    
    def next_position(self) -> bool:
        """
        Move to the next filter position
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.current_position < self.max_positions:
            return self.set_filter_position(self.current_position + 1)
        else:
            logger.warning("Already at maximum filter position")
            return False
    
    def previous_position(self) -> bool:
        """
        Move to the previous filter position
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.current_position > 1:
            return self.set_filter_position(self.current_position - 1)
        else:
            logger.warning("Already at minimum filter position")
            return False
    
    def home(self) -> bool:
        """
        Home the filter wheel to position 1
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Homing filter wheel to position 1")
        return self.set_filter_position(1)
    
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
