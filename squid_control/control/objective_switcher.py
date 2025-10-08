"""
Objective Switcher Control for Squid+ Microscope
Based on the official Squid software Xeryon objective switcher implementation
"""

import time
import logging
import serial
import serial.tools.list_ports
from typing import Optional
from squid_control.control.config import CONFIG

logger = logging.getLogger(__name__)


class ObjectiveSwitcherController:
    """
    Controller for Xeryon objective switcher in Squid+ microscope
    Supports both real hardware and simulation modes
    
    This class interfaces with the Xeryon linear actuator to switch between objectives.
    """
    
    def __init__(self, serial_number: str = None, stage=None, is_simulation=False):
        self.serial_number = serial_number
        self.stage = stage
        self.is_simulation = is_simulation
        
        # Position settings (in mm) - these are the DPOS values sent to Xeryon
        self.position1 = -19  # Position 1 (e.g., 20x objective)
        self.position2 = 19    # Position 2 (e.g., 4x objective)
        self.current_position = None
        self.retracted = False
        
        # Get offset from configuration
        self.position2_offset = getattr(CONFIG, 'XERYON_OBJECTIVE_SWITCHER_POS_2_OFFSET_MM', 2.0)
        
        # Xeryon controller and axis (only for real hardware)
        self.controller = None
        self.axisX = None
        
        if is_simulation:
            logger.info("Objective switcher controller initialized in simulation mode")
        else:
            if serial_number:
                self._initialize_hardware(serial_number)
            else:
                logger.warning("No serial number provided for objective switcher")
    
    def _initialize_hardware(self, serial_number: str):
        """Initialize hardware connection to Xeryon objective switcher"""
        try:
            # Import Xeryon module
            from squid_control.control.Xeryon import Xeryon, Stage
            
            # Find the port with the matching serial number
            ports = serial.tools.list_ports.comports()
            port = None
            for p in ports:
                if p.serial_number == serial_number:
                    port = p.device
                    break
            
            if port is None:
                raise Exception(f"Xeryon objective switcher with SN {serial_number} not found")
            
            # Initialize Xeryon controller - exact same as official software
            logger.info(f"Connecting to Xeryon objective switcher on port {port}")
            self.controller = Xeryon(port, 115200)
            self.axisX = self.controller.addAxis(Stage.XLA_1250_3N, "Z")
            self.controller.start()
            self.controller.reset()
            
            logger.info("Xeryon objective switcher initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Xeryon objective switcher: {e}")
            raise
    
    def home(self) -> bool:
        """
        Home the objective switcher
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.is_simulation:
                logger.info("Simulation: Homing objective switcher")
                self.current_position = 0
                return True
            else:
                # Real hardware homing - use Xeryon's findIndex method
                logger.info("Homing objective switcher")
                self.axisX.stopScan()
                self.axisX.findIndex()
                self.current_position = 0
                return True
        except Exception as e:
            logger.error(f"Failed to home objective switcher: {e}")
            return False
    
    def move_to_position_1(self, move_z: bool = True) -> bool:
        """
        Move to position 1 (e.g., 20x objective)
        
        Args:
            move_z (bool): Whether to move Z stage to compensate for objective change
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.is_simulation:
                logger.info("Simulation: Moving to objective position 1")
                if self.stage and self.current_position == 2 and self.retracted:
                    # Revert Z retraction
                    if move_z:
                        self.stage.move_z(self.position2_offset)
                    self.retracted = False
                self.current_position = 1
                return True
            else:
                # Real hardware movement - use Xeryon's setDPOS method
                logger.info(f"Moving to objective position 1 (DPOS={self.position1}mm)")
                
                # Move the Xeryon axis to position 1
                self.axisX.setDPOS(self.position1)
                
                # Handle Z stage compensation if switching from position 2
                if self.stage and self.current_position == 2 and self.retracted:
                    if move_z:
                        logger.info(f"Reverting Z retraction: moving Z by +{self.position2_offset}mm")
                        self.stage.move_z(self.position2_offset)
                    self.retracted = False
                
                self.current_position = 1
                logger.info("Successfully moved to objective position 1")
                return True
        except Exception as e:
            logger.error(f"Failed to move to position 1: {e}")
            return False
    
    def move_to_position_2(self, move_z: bool = True) -> bool:
        """
        Move to position 2 (e.g., 4x objective)
        
        Args:
            move_z (bool): Whether to move Z stage to compensate for objective change
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.is_simulation:
                logger.info("Simulation: Moving to objective position 2")
                if self.stage and self.current_position == 1:
                    # Retract Z for position 2
                    if move_z:
                        self.stage.move_z(-self.position2_offset)
                    self.retracted = True
                self.current_position = 2
                return True
            else:
                # Real hardware movement - use Xeryon's setDPOS method
                logger.info(f"Moving to objective position 2 (DPOS={self.position2}mm)")
                
                # Handle Z stage compensation BEFORE moving the objective
                # This is important to avoid collision
                if self.stage and self.current_position == 1:
                    if move_z:
                        logger.info(f"Retracting Z stage: moving Z by -{self.position2_offset}mm")
                        self.stage.move_z(-self.position2_offset)
                    self.retracted = True
                
                # Move the Xeryon axis to position 2
                self.axisX.setDPOS(self.position2)
                
                self.current_position = 2
                logger.info("Successfully moved to objective position 2")
                return True
        except Exception as e:
            logger.error(f"Failed to move to position 2: {e}")
            return False
    
    def get_current_position(self) -> Optional[int]:
        """
        Get the current objective position
        
        Returns:
            Optional[int]: Current position (1, 2, or None if unknown)
        """
        return self.current_position
    
    def set_speed(self, speed: float) -> bool:
        """
        Set the movement speed of the objective switcher
        
        Args:
            speed (float): Speed value
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.is_simulation:
                logger.info(f"Simulation: Setting objective switcher speed to {speed}")
                return True
            else:
                # Real hardware speed setting - use Xeryon's setSpeed method
                logger.info(f"Setting objective switcher speed to {speed}")
                self.axisX.setSpeed(speed)
                return True
        except Exception as e:
            logger.error(f"Failed to set speed: {e}")
            return False
    
    def is_enabled(self) -> bool:
        """
        Check if objective switcher is enabled in configuration
        
        Returns:
            bool: True if objective switcher is enabled
        """
        return getattr(CONFIG, 'USE_XERYON', False)
    
    def get_available_positions(self) -> list:
        """
        Get list of available objective positions
        
        Returns:
            list: List of available positions
        """
        return [1, 2]  # Two-position switcher
    
    def get_position_names(self) -> dict:
        """
        Get mapping of position numbers to objective names
        
        Returns:
            dict: Mapping of position to objective name
        """
        pos1_objectives = getattr(CONFIG, 'XERYON_OBJECTIVE_SWITCHER_POS_1', ['20x'])
        pos2_objectives = getattr(CONFIG, 'XERYON_OBJECTIVE_SWITCHER_POS_2', ['4x'])
        
        return {
            1: pos1_objectives[0] if pos1_objectives else "Position 1",
            2: pos2_objectives[0] if pos2_objectives else "Position 2"
        }


class ObjectiveSwitcherSimulation:
    """
    Simulation mode for objective switcher controller
    """
    
    def __init__(self, stage=None):
        self.stage = stage
        self.current_position = None
        self.retracted = False
        self.position2_offset = 2.0  # Default offset
        logger.info("Objective switcher simulation initialized")
    
    def home(self) -> bool:
        """Simulation version of home"""
        logger.info("Simulation: Homing objective switcher")
        self.current_position = 0
        return True
    
    def move_to_position_1(self, move_z: bool = True) -> bool:
        """Simulation version of move_to_position_1"""
        logger.info("Simulation: Moving to objective position 1")
        if self.stage and self.current_position == 2 and self.retracted:
            if move_z:
                self.stage.move_z(self.position2_offset)
            self.retracted = False
        self.current_position = 1
        return True
    
    def move_to_position_2(self, move_z: bool = True) -> bool:
        """Simulation version of move_to_position_2"""
        logger.info("Simulation: Moving to objective position 2")
        if self.stage and self.current_position == 1:
            if move_z:
                self.stage.move_z(-self.position2_offset)
            self.retracted = True
        self.current_position = 2
        return True
    
    def get_current_position(self) -> Optional[int]:
        """Simulation version of get_current_position"""
        return self.current_position
    
    def set_speed(self, speed: float) -> bool:
        """Simulation version of set_speed"""
        logger.info(f"Simulation: Setting objective switcher speed to {speed}")
        return True
    
    def is_enabled(self) -> bool:
        """Simulation version of is_enabled"""
        return True  # Always enabled in simulation
    
    def get_available_positions(self) -> list:
        """Simulation version of get_available_positions"""
        return [1, 2]
    
    def get_position_names(self) -> dict:
        """Simulation version of get_position_names"""
        return {
            1: "20x (Simulation)",
            2: "4x (Simulation)"
        }
