import argparse
import asyncio
import logging
import signal
import sys
import traceback

from squid_control.service.microscope_service import MicroscopeHyphaService  # noqa: F401
from squid_control.service.video_stream import MicroscopeVideoTrack  # noqa: F401
from squid_control.utils.logging_utils import setup_logging

__all__ = ["MicroscopeHyphaService", "MicroscopeVideoTrack", "main"]

logger = setup_logging("squid_control_service.log")

_microscope_instance = None


def signal_handler(sig, frame):
    global _microscope_instance
    logger.info("Signal received, shutting down gracefully...")

    if _microscope_instance and hasattr(_microscope_instance, "frame_acquisition_running") and _microscope_instance.frame_acquisition_running:
        logger.info("Stopping video buffering...")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_microscope_instance.stop_video_buffering())
            else:
                loop.run_until_complete(_microscope_instance.stop_video_buffering())
                loop.close()
        except Exception as e:
            logger.error(f"Error stopping video buffering: {e}")

    if _microscope_instance and hasattr(_microscope_instance, "squidController"):
        _microscope_instance.squidController.close()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point for the microscope service."""
    global _microscope_instance

    parser = argparse.ArgumentParser(description="Squid microscope control services for Hypha.")
    parser.add_argument("--simulation", dest="simulation", action="store_true", default=False,
                        help="Run in simulation mode (default: False)")
    parser.add_argument("--local", dest="local", action="store_true", default=False,
                        help="Run with local server URL (default: False)")
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--config", type=str, default=None,
                        help="Configuration name, e.g. 'HCS_v2', 'HCS_v2_63x', 'Squid+'")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    microscope = MicroscopeHyphaService(is_simulation=args.simulation, is_local=args.local, config_name=args.config)
    _microscope_instance = microscope

    loop = asyncio.get_event_loop()

    async def async_main():
        try:
            microscope.setup_task = asyncio.create_task(microscope.setup())
            await microscope.setup_task
        except Exception:
            traceback.print_exc()

    loop.create_task(async_main())
    loop.run_forever()


if __name__ == "__main__":
    main()
