"""
Objective Switcher Control for Squid+ Microscope
Based on the official Squid software Xeryon objective switcher implementation
"""

import logging
import serial
import serial.tools.list_ports
from typing import Optional
from squid_control.hardware.config import CONFIG

logger = logging.getLogger(__name__)

DEFAULT_POSITION_1_MM = -19.0
DEFAULT_POSITION_2_MM = 19.0
DEFAULT_RUNTIME_PTOL = 40
DEFAULT_RUNTIME_PTO2 = 40
DEFAULT_RUNTIME_TOU3 = 10
XERYON_USB_IDS = {(0x04D8, 0xEF4A)}


class ObjectiveSwitcherController:
    """
    Controller for Xeryon objective switcher in Squid+ microscope
    Matches the official ObjectiveChanger2PosController implementation
    
    CRITICAL: You MUST call home() before using any movement commands!
    """
    
    def __init__(self, serial_number: str = None, stage=None, is_simulation=False):
        super().__init__()
        self.serial_number = serial_number
        self.stage = stage
        self.is_simulation = is_simulation
        
        # Position settings (in mm) - these are the DPOS values sent to Xeryon
        # These MUST match the positions in Xeryon_settings.txt LLIM and HLIM
        self.position1 = DEFAULT_POSITION_1_MM  # Position 1 (e.g., 20x objective)
        self.position2 = DEFAULT_POSITION_2_MM  # Position 2 (e.g., 4x objective)
        self.current_position = None
        self.retracted = False  # Track if Z was retracted for position 2
        self.runtime_ptol = getattr(CONFIG, 'XERYON_PTOL', DEFAULT_RUNTIME_PTOL)
        self.runtime_pto2 = getattr(CONFIG, 'XERYON_PTO2', DEFAULT_RUNTIME_PTO2)
        self.runtime_tou3 = getattr(CONFIG, 'XERYON_TOU3', DEFAULT_RUNTIME_TOU3)
        
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
            from squid_control.hardware.drivers.Xeryon import Xeryon, Stage
            
            # Find the port with the matching serial number.
            ports = serial.tools.list_ports.comports()
            xeryon_ports = [
                p for p in ports if (p.vid, p.pid) in XERYON_USB_IDS
            ]
            port = None
            for p in xeryon_ports:
                if p.serial_number == serial_number:
                    port = p.device
                    break
            
            if port is None:
                if len(xeryon_ports) == 1:
                    detected = xeryon_ports[0]
                    port = detected.device
                    logger.warning(
                        "Configured Xeryon SN %s not found; falling back to the only connected Xeryon device %s on %s",
                        serial_number,
                        detected.serial_number,
                        detected.device,
                    )
                else:
                    available_serials = [p.serial_number or "<unknown>" for p in xeryon_ports]
                    raise Exception(
                        f"Xeryon objective switcher with SN {serial_number} not found. "
                        f"Available Xeryon serials: {available_serials or 'none'}"
                    )
            
            # Initialize the Xeryon controller with the validated switcher profile.
            logger.info(f"Connecting to Xeryon objective switcher on port {port}")
            self.controller = Xeryon(port, 115200)
            self.axisX = self.controller.addAxis(Stage.XLS_1250_3N, "X")
            self.controller.start()
            self.controller.reset()
            self._apply_runtime_tuning()
            
            logger.info("Xeryon objective switcher initialized successfully")
            logger.warning("⚠️  You MUST call home() before using movement commands!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Xeryon objective switcher: {e}")
            raise

    def _apply_runtime_tuning(self):
        """
        Apply the same tolerance/timeouts that worked on the validated switcher.
        This keeps runtime behavior aligned even if the controller has stale values.
        """
        self.axisX.setSetting("PTOL", str(self.runtime_ptol))
        self.axisX.setSetting("PTO2", str(self.runtime_pto2))
        self.axisX.setSetting("TOU3", str(self.runtime_tou3))
        logger.info(
            "Applied Xeryon runtime tuning: PTOL=%s PTO2=%s TOU3=%s",
            self.runtime_ptol,
            self.runtime_pto2,
            self.runtime_tou3,
        )

    def _move_axis(self, target_mm: float, label: str) -> bool:
        logger.info("Moving objective switcher to %s (DPOS=%s mm)...", label, target_mm)
        success = self.axisX.setDPOS(target_mm, forceWaiting=True)
        if not success:
            try:
                epos = self.axisX.getEPOS()
            except Exception:
                epos = "unknown"
            logger.error(
                "Xeryon move failed for %s: target=%s mm current=%s mm",
                label,
                target_mm,
                epos,
            )
            return False
        return True
    
    def home(self) -> bool:
        """
        Home the objective switcher - CRITICAL step before any movements!
        This finds the index position and establishes the reference point.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.is_simulation:
                logger.info("Simulation: Homing objective switcher")
                self.current_position = 0
                return True
            else:
                # Stop any scan state before indexing so homing starts from a clean state.
                logger.info("Homing objective switcher...")
                self.axisX.stopScan()
                success = self.axisX.findIndex(forceWaiting=True)
                if not success:
                    logger.error("Objective switcher homing failed")
                    return False
                self.current_position = 0
                logger.info("✓ Objective switcher homed successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to home objective switcher: {e}")
            return False
    
    def move_to_zero(self) -> bool:
        """
        Move to the zero position (home position)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.is_simulation:
                logger.info("Simulation: Moving to zero position")
                self.current_position = 0
                return True
            else:
                if not self._move_axis(0, "zero position"):
                    return False
                self.current_position = 0
                return True
        except Exception as e:
            logger.error(f"Failed to move to zero: {e}")
            return False
    
    def move_to_position_1(self, move_z: bool = True) -> bool:
        """
        Move to position 1 (e.g., 20x objective)
        
        CRITICAL ORDER:
        1. Move Xeryon axis first
        2. THEN restore Z if coming from position 2
        
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
                # Real hardware movement: move the Xeryon axis first, then compensate Z.
                # STEP 1: Move the Xeryon axis FIRST
                if not self._move_axis(self.position1, "position 1"):
                    return False
                
                # STEP 2: Then handle Z stage compensation if switching from position 2
                if self.stage is not None and self.current_position == 2 and self.retracted:
                    # Revert retracting Z by position2_offset
                    if move_z:
                        logger.info(f"Reverting Z retraction: moving Z by +{self.position2_offset}mm")
                        self.stage.move_z(self.position2_offset)
                    self.retracted = False
                
                self.current_position = 1
                logger.info("✓ Successfully moved to objective position 1")
                return True
        except Exception as e:
            logger.error(f"Failed to move to position 1: {e}")
            return False
    
    def move_to_position_2(self, move_z: bool = True) -> bool:
        """
        Move to position 2 (e.g., 4x objective)
        
        CRITICAL ORDER:
        1. Move Xeryon axis first
        2. Then retract Z (if needed)
        
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
                # Real hardware movement: move the Xeryon axis first, then compensate Z.
                # STEP 1: Move the Xeryon axis FIRST
                if not self._move_axis(self.position2, "position 2"):
                    return False
                
                # STEP 2: Then handle Z stage compensation if switching from position 1
                if self.stage is not None and self.current_position == 1:
                    # Retract Z by position2_offset to avoid collision
                    if move_z:
                        logger.info(f"Retracting Z stage: moving Z by -{self.position2_offset}mm")
                        self.stage.move_z(-self.position2_offset)
                    self.retracted = True
                
                self.current_position = 2
                logger.info("✓ Successfully moved to objective position 2")
                return True
        except Exception as e:
            logger.error(f"Failed to move to position 2: {e}")
            return False
    
    def get_current_position(self) -> Optional[int]:
        """
        Get the current objective position
        
        Returns:
            Optional[int]: Current position (0=home, 1=position1, 2=position2, or None if unknown)
        """
        return self.current_position
    
    def set_speed(self, speed: float) -> bool:
        """
        Set the movement speed of the objective switcher
        
        Args:
            speed (float): Speed value (typically 80-100)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.is_simulation:
                logger.info(f"Simulation: Setting objective switcher speed to {speed}")
                return True
            else:
                # Real hardware speed setting - EXACT same as official software
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
    Matches the official ObjectiveChanger2PosController_Simulation implementation
    """
    
    def __init__(self, stage=None):
        super().__init__()
        self.stage = stage
        self.current_position = None
        self.retracted = False
        
        # Position settings
        self.position1 = -19
        self.position2 = 19
        self.position2_offset = getattr(CONFIG, 'XERYON_OBJECTIVE_SWITCHER_POS_2_OFFSET_MM', 2.0)
        
        logger.info("Objective switcher simulation initialized")
    
    def home(self) -> bool:
        """Simulation version of home"""
        logger.info("Simulation: Homing objective switcher")
        self.current_position = 0
        return True
    
    def move_to_zero(self) -> bool:
        """Simulation version of move_to_zero"""
        logger.info("Simulation: Moving to zero position")
        self.current_position = 0
        return True
    
    def move_to_position_1(self, move_z: bool = True) -> bool:
        """Simulation version of move_to_position_1"""
        logger.info("Simulation: Moving to objective position 1")
        if self.stage is not None and self.current_position == 2 and self.retracted:
            # Revert retracting Z by position2_offset
            if move_z:
                self.stage.move_z(self.position2_offset)
            self.retracted = False
        self.current_position = 1
        return True
    
    def move_to_position_2(self, move_z: bool = True) -> bool:
        """Simulation version of move_to_position_2"""
        logger.info("Simulation: Moving to objective position 2")
        if self.stage is not None and self.current_position == 1:
            # Retract Z by position2_offset
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
        # Get the actual objective names from configuration to match objectives_dict
        pos1_objectives = getattr(CONFIG, 'XERYON_OBJECTIVE_SWITCHER_POS_1', ['20x'])
        pos2_objectives = getattr(CONFIG, 'XERYON_OBJECTIVE_SWITCHER_POS_2', ['4x'])
        
        return {
            1: pos1_objectives[0] if pos1_objectives else "20x",
            2: pos2_objectives[0] if pos2_objectives else "4x"
        }
