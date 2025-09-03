import asyncio
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

# Configure matplotlib environment BEFORE any matplotlib import
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Configure matplotlib to use non-interactive backend for testing
try:
    # Import matplotlib with explicit backend configuration
    import matplotlib
    matplotlib.use('Agg', force=True)  # Use non-interactive backend
    matplotlib.interactive(False)

    # Additional configuration to prevent type registration issues
    matplotlib.rcParams['backend'] = 'Agg'
    matplotlib.rcParams['interactive'] = False

    # Clear any existing type registrations that might cause conflicts
    try:
        import matplotlib.colors
        # Force re-registration of types to prevent conflicts
        matplotlib.colors._colors_full_map.clear()
    except Exception:
        pass

except Exception as e:
    print(f"Warning: Could not configure matplotlib backend: {e}")
    # If matplotlib fails to import, we'll continue without it for testing

def pytest_collection_modifyitems(config, items):
    """Modify test collection to ensure matplotlib is configured."""
    # No need to reconfigure matplotlib here since it's already done above
    pass

# Configure asyncio policy for better event loop management
def pytest_configure(config):
    """Configure pytest with asyncio settings."""
    # Register custom marks
    config.addinivalue_line("markers", "integration: mark test as integration test requiring external services")
    config.addinivalue_line("markers", "asyncio: mark test as asyncio-based test")

    # Suppress deprecation warnings from websockets and other libraries
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Suppress matplotlib warnings that commonly occur in CI
    warnings.filterwarnings("ignore", message=".*matplotlib.*")
    warnings.filterwarnings("ignore", message=".*_InterpolationType.*")

    # Matplotlib is already configured at module level, no need to reconfigure here

    # Set asyncio policy for consistent event loop handling
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    # Create a new event loop for the test session
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Configure the loop with proper executor
    loop.set_default_executor(ThreadPoolExecutor(max_workers=4))

    yield loop

    # Cleanup
    try:
        # Cancel all remaining tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()

        # Run until all tasks are cancelled
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        # Close the loop
        loop.close()
    except Exception as e:
        print(f"Error during event loop cleanup: {e}")

@pytest.fixture(autouse=True, scope="function")
def cleanup_tasks():
    """Auto-cleanup fixture to ensure tasks are cleaned up after each test."""
    yield

    # Clean up any remaining tasks after each test
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
            if tasks:
                for task in tasks:
                    if not task.cancelled():
                        task.cancel()
    except RuntimeError:
        # No event loop available, nothing to clean up
        pass

def pytest_runtest_setup(item):
    """Setup for each test item."""
    # Skip integration tests if environment variable not set
    if "integration" in item.keywords:
        import os
        if not os.environ.get("AGENT_LENS_WORKSPACE_TOKEN"):
            pytest.skip("AGENT_LENS_WORKSPACE_TOKEN not set - skipping integration test")
