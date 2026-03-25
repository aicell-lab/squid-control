# AGENTS.md

This file is the repository-level guide for coding agents working in `squid-control`.

## Project Overview

`squid-control` is a Python 3.11+ control system for the Cephla Squid microscope. It provides a complete software stack for microscope hardware control, remote access via the Hypha platform, real-time video streaming, automated well plate scanning, and multi-channel fluorescence imaging.

### Key Capabilities

| Capability | Description |
|-----------|-------------|
| **Hardware control** | Stage X/Y/Z, multi-channel LED illumination, camera triggering via Teensy 4.1 microcontroller |
| **Remote API** | Full microscope control over the internet via Hypha RPC (WebSocket + WebRTC) |
| **Live video** | WebRTC streaming of the microscope camera feed to any web browser |
| **Well plate scanning** | Automated multi-well, multi-channel, multi-Z acquisition for 6–384 well plates |
| **Image stitching** | Real-time OME-Zarr tile assembly with multi-scale pyramid output |
| **Simulation mode** | Full software simulation using real Zarr sample data — no hardware required |
| **Mirror service** | Cloud proxy that gives public remote access to a microscope behind a firewall |

## Technology Stack

- **Language**: Python 3.11+
- **Build System**: setuptools (defined in `pyproject.toml`)
- **Key Dependencies**:
  - Scientific computing: numpy, scipy, pandas
  - Image processing: opencv-python, scikit-image, Pillow, tifffile
  - Data storage: zarr, blosc, fsspec
  - Hardware communication: pyserial, crc
  - Web services: hypha-rpc, hypha-artifact, flask, aiohttp
  - Video streaming: av, aiortc (WebRTC)
  - Data validation: pydantic
  - ML/AI: openai, jax

## Project Structure

```
squid_control/
├── __init__.py              # Public API — imports SquidController, MicroscopeHyphaService
├── __main__.py              # CLI entry point: python -m squid_control microscope|mirror
│
├── controller/              # High-level microscope orchestration
│   ├── squid_controller.py  # SquidController — central application object
│   ├── scanning.py          # ScanningMixin — plate_scan(), flexible_position_scan()
│   └── acquisition.py       # AcquisitionMixin — snap_image(), autofocus, camera frames
│
├── hardware/                # Hardware abstraction layer
│   ├── core.py              # Low-level controllers (NavigationController, LiveController, etc.)
│   ├── microcontroller.py   # Serial protocol for Teensy 4.1 (USB, 2Mbaud)
│   ├── config.py            # Configuration system (Pydantic models + INI file loader)
│   ├── filter_wheel.py      # Filter wheel hardware driver (Squid+)
│   ├── objective_switcher.py# Motorised objective switcher (Squid+)
│   ├── serial_peripherals.py# Additional serial-connected peripherals
│   ├── processing_handler.py# Image processing pipeline handler
│   ├── camera/              # Camera drivers
│   │   ├── __init__.py      # get_camera() factory, TriggerModeSetting enum
│   │   ├── camera_default.py# GeniCam / simulation camera
│   │   └── camera_toupcam.py# ToupCam vendor driver
│   └── drivers/             # Third-party SDK wrappers
│       ├── gxipy/           # Daheng Imaging GxiPy SDK
│       └── Xeryon.py        # Xeryon piezo stage driver
│
├── service/                 # Hypha RPC service layer
│   ├── microscope_service.py# MicroscopeHyphaService — all RPC endpoints (~100 methods)
│   └── video_stream.py      # MicroscopeVideoTrack — WebRTC video frame producer
│
├── mirror/                  # Cloud-to-local proxy service
│   ├── mirror_service.py    # MirrorMicroscopeService — dynamic method mirroring
│   ├── video_track.py       # WebRTC video track for mirror service
│   └── cli.py               # CLI for the mirror subcommand
│
├── stitching/               # Multi-well image stitching
│   └── zarr_canvas.py       # ZarrCanvas, ExperimentManager — OME-Zarr builder
│
├── storage/                 # Data storage and upload utilities
│   ├── artifact_manager/    # Hypha Artifact Manager client
│   └── snapshot_utils.py    # Snapshot capture and upload
│
├── simulation/              # Virtual sample data for simulation mode
│   └── samples.py           # SIMULATION_SAMPLES, SAMPLE_ALIASES — Zarr dataset registry
│
├── utils/                   # Shared utilities
│   ├── logging_utils.py     # setup_logging() — rotating file + console handler
│   ├── video_utils.py       # VideoBuffer, VideoFrameProcessor
│   ├── image_processing.py  # rotate_and_flip_image() and other transforms
│   └── illumination_calibration/  # LED intensity calibration tools
│
└── config/                  # Configuration files
    ├── configuration_HCS_v2_example.ini
    ├── configuration_HCS_v2_63x_example.ini
    ├── configuration_Squid+_example.ini
    ├── u2os_fucci_illumination_configurations.xml
    └── focus_camera_configurations.xml
```

## Build and Installation

### Development Setup

```bash
# Clone and setup
git clone https://github.com/aicell-lab/squid-control.git
cd squid-control

# Create conda environment (recommended)
conda create -n squid python=3.11
conda activate squid

# Install in editable mode with dev dependencies
pip install -e .[dev]
```

### Production Installation

```bash
pip install .
```

### Docker Build

```bash
docker build -f docker/Dockerfile -t squid-control .
docker run -it squid-control
```

## Running the Application

### Simulation Mode (No Hardware Required)

```bash
python -m squid_control microscope --simulation
```

### Hardware Mode

```bash
python -m squid_control microscope
```

### Local Mode (Private Server)

```bash
python -m squid_control microscope --local
```

### Mirror Service

```bash
python -m squid_control mirror \
  --local-service-id "microscope-control-squid-1" \
  --cloud-service-id "microscope-control-squid-1"
```

## Testing

### Quick Tests (Simulation Only)

```bash
pytest -m "not integration"
```

### All Tests (Requires Tokens)

```bash
pytest
```

### Specific Test Files

```bash
pytest tests/test_squid_controller.py   # Controller and simulation
pytest tests/test_hypha_service.py      # RPC API endpoints
pytest tests/test_mirror_service.py     # Mirror service
pytest tests/test_webrtc_e2e.py         # End-to-end video streaming
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `integration` | Requires network access and Hypha tokens |
| `simulation` | Uses simulation mode |
| `hardware` | Requires real hardware (skipped by default) |
| `local` | Requires local setup |
| `slow` | Long-running tests |
| `unit` | Unit tests |

## Code Style Guidelines

### Linting and Formatting

The project uses **ruff** for linting and formatting:

```bash
# Check code
ruff check .

# Fix issues
ruff check . --fix

# Format code
ruff format .
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

### Style Configuration

- **Line length**: 88 characters (Black-compatible)
- **Quote style**: Double quotes
- **Import sorting**: Enabled via ruff

### Type Hints

- Use type hints on new or changed functions where practical
- Use `async`/`await` for I/O-facing flows

### Logging

- Use `logger = logging.getLogger(__name__)` in each module
- Never use `print()` for logging
- Log hardware and service failures with enough context

## Configuration System

### Configuration Files

Hardware parameters are stored in INI files. Available profiles:

| Profile | Description |
|---------|-------------|
| `HCS_v2` | Standard setup — 20× objective, HCS well plate stage |
| `HCS_v2_63x` | High-magnification — 63× oil immersion objective |
| `Squid+` | Extended — motorised filter wheel and objective switcher |

### Loading Configuration

```python
from squid_control.hardware.config import CONFIG, load_config

# Load specific config
load_config("HCS_v2_63x")

# Access values
print(CONFIG.CAMERA_TYPE)
print(CONFIG.ACQUISITION.CROP_WIDTH)
```

### Custom Config Path

```bash
export SQUID_CONFIG_PATH=/path/to/custom_config.ini
```

## Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `REEF_WORKSPACE_TOKEN` | Hypha `reef-imaging` workspace token | Cloud integration |
| `REEF_LOCAL_TOKEN` | Local Hypha server token | Local mode |
| `REEF_LOCAL_WORKSPACE` | Local Hypha workspace name | Local mode |
| `SQUID_CONFIG_PATH` | Custom INI config file path | Custom hardware config |
| `MICROSCOPE_SERVICE_ID` | Service ID for Hypha registration | Service identification |
| `ZARR_PATH` | Base path for Zarr canvas storage | Image stitching |
| `AUTHORIZED_USERS` | Comma-separated list of authorized emails | Access control |

## Hardware Communication

The PC communicates with a **Teensy 4.1 microcontroller** over USB serial at 2,000,000 baud.

### Protocol

- Command (PC → Teensy): 8 bytes `[cmd_id, cmd_type, param×5, crc8]`
- Response (Teensy → PC): 24 bytes `[cmd_id, status, x×4, y×4, z×4, theta×4, switches, reserved×4, crc8]`

See `hardware/microcontroller.py` for the full command set.

## Key Classes Reference

### SquidController

Central application object that owns all hardware controllers.

```python
from squid_control.controller import SquidController

ctrl = SquidController(is_simulation=True)
await ctrl.snap_image(channel=0, intensity=30, exposure_time=50)
ctrl.move_to_well('C', 3)
```

**Inherits from:**
- `ScanningMixin` — plate scanning methods
- `AcquisitionMixin` — image acquisition and autofocus

### MicroscopeHyphaService

Exposes `SquidController` as a Hypha RPC service.

```python
from squid_control.service import MicroscopeHyphaService

svc = MicroscopeHyphaService(is_simulation=True)
await svc.setup()
```

When adding new RPC endpoints:
1. Add method to `MicroscopeHyphaService` in `service/microscope_service.py`
2. Decorate with `@schema_function(skip_self=True)`
3. Use `Field(...)` for parameter descriptions
4. Check permissions via `self.check_permission(context.get("user", {}))`

### MirrorMicroscopeService

Connects to a local microscope service and re-registers all its methods on the cloud Hypha server.

```python
from squid_control.mirror import MirrorMicroscopeService

mirror = MirrorMicroscopeService(
    cloud_server_url="https://hypha.aicell.io",
    cloud_workspace="reef-imaging",
    local_server_url="http://localhost:9527",
    local_service_id="microscope-control-squid-1",
    cloud_service_id="microscope-control-squid-1",
)
await mirror.run()
```

## Simulation Mode

Simulation mode replaces all hardware with software equivalents:

| Real hardware | Simulation replacement |
|--------------|----------------------|
| Physical camera | `Camera_Simulation` — serves crops from Zarr archives |
| Teensy microcontroller | `Microcontroller_Simulation` — tracks position in memory |
| Stage movement | Instant coordinate update |
| Illumination | No-op with state tracking |

The virtual sample data is fetched from the Hypha Artifact Manager.

## Working Agreements

### Prefer Simulation First

- Start with simulation mode unless the task is explicitly hardware-only
- Default validation path: `python -m squid_control microscope --simulation`
- Prefer tests marked for simulation or non-integration flows

### Keep Hardware-Safe Behavior

- Do not assume physical hardware is connected
- Be careful around stage motion, illumination, and camera control code
- Preserve cleanup paths around hardware access, especially `try`/`finally` blocks

### Error Handling

- Raise specific exceptions instead of returning error payloads
- Avoid bare `except:`
- Log failures with enough context to debug

### Image Processing

- Crop before resize
- Use `CONFIG.ACQUISITION.CROP_HEIGHT` and `CONFIG.ACQUISITION.CROP_WIDTH`
- Preserve source bit depth through the processing path
- Bounds-check crop coordinates

## Security Considerations

- Never commit tokens or credentials to the repository
- Use `.env` file for local environment variables (it's in `.gitignore`)
- The `AUTHORIZED_USERS` environment variable controls access to write operations
- Always validate user permissions in RPC methods that modify hardware state

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the correct conda environment and installed with `pip install -e .`
2. **Permission denied on /dev/ttyUSB0**: Add user to `dialout` group or run with appropriate permissions
3. **Camera not found**: Check camera connections and drivers
4. **Hypha connection failed**: Verify `REEF_WORKSPACE_TOKEN` is set and valid

### Debug Logging

```bash
python -m squid_control microscope --simulation --verbose
```

## Additional Documentation

- `README.md` — Project overview and quick start
- `squid_control/README.md` — Detailed developer reference with class descriptions
- `tests/README.md` — Testing documentation
- `scripts/README_OPERA_IMPORT.md` — Opera data import instructions
