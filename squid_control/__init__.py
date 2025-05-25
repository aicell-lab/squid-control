"""
Squid Microscope Control Software

A comprehensive control system for squid microscopes with support for
various cameras, stages, and imaging configurations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components for easy access
try:
    from .squid_controller import SquidController
except ImportError:
    # Handle optional dependencies gracefully
    pass

__all__ = [
    "__version__",
    "SquidController",
]
