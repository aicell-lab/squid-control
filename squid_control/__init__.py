"""
Squid Microscope Control System

A Python-based control system for the Squid microscope (by Cephla Inc.), featuring:
- Real-time microscope hardware control and automation
- Web-based API service using Hypha RPC
- Camera integration with multiple vendors (ToupCam, FLIR, TIS)
- Well plate scanning and image acquisition
- WebRTC video streaming for remote microscope viewing
- AI-powered chatbot integration for natural language microscope control
- Simulation mode with Zarr-based virtual samples
- Multi-channel fluorescence imaging capabilities

Usage:
    # Run as module
    python -m squid_control [--simulation] [--local] [--verbose]
    
    # Import in Python code
    from squid_control.start_hypha_service import Microscope
    from squid_control.squid_controller import SquidController
"""

__version__ = "1.0.0"
__author__ = "Cephla Inc."

# Import main classes for easy access
from .start_hypha_service import Microscope
from .squid_controller import SquidController

__all__ = [
    "Microscope",
    "SquidController",
]
