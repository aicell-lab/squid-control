#!/usr/bin/env python3
"""
Main entry point for the squid_control module.
This allows users to run: python -m squid_control [options]
"""

from .start_hypha_service import main

if __name__ == "__main__":
    main()
