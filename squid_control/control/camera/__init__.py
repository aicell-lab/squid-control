from enum import Enum


class TriggerModeSetting(Enum):
    SOFTWARE = "Software Trigger"
    HARDWARE = "Hardware Trigger"
    CONTINUOUS = "Continuous Acqusition"


def get_camera(camera_type, focus_camera_type=None):
    """
    Get camera modules for main and focus cameras.
    
    Args:
        camera_type: Type of main camera ("Toupcam", "FLIR", "Default")
        focus_camera_type: Type of focus camera (optional, defaults to same as main camera)
    
    Returns:
        Tuple of (main_camera_module, focus_camera_module)
    """
    # If focus camera type not specified, use same as main camera
    if focus_camera_type is None:
        focus_camera_type = camera_type
    
    # Load main camera module
    if camera_type == "Toupcam":
        try:
            import squid_control.control.camera.camera_toupcam as camera
        except:
            print("Problem importing Toupcam, defaulting to default camera")
            import squid_control.control.camera.camera_default as camera
    elif camera_type == "FLIR":
        try:
            import squid_control.control.camera.camera_flir as camera
        except:
            print("Problem importing FLIR camera, defaulting to default camera")
            import squid_control.control.camera.camera_default as camera
    else:
        import squid_control.control.camera.camera_default as camera
    
    # Load focus camera module (separate from main camera)
    if focus_camera_type == "Toupcam":
        try:
            import squid_control.control.camera.camera_toupcam as camera_fc
        except:
            print("Problem importing Toupcam for focus, defaulting to default camera")
            import squid_control.control.camera.camera_default as camera_fc
    elif focus_camera_type == "FLIR":
        try:
            import squid_control.control.camera.camera_flir as camera_fc
        except:
            print("Problem importing FLIR camera for focus, defaulting to default camera")
            import squid_control.control.camera.camera_default as camera_fc
    else:
        import squid_control.control.camera.camera_default as camera_fc
    
    return camera, camera_fc
