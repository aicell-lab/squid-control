# squid_control — Package Reference

This document describes the internal structure of the `squid_control` Python package for developers working on the codebase.

## Package Structure

```
squid_control/
├── __init__.py              # Public API — imports SquidController, MicroscopeHyphaService, MirrorMicroscopeService
├── __main__.py              # CLI entry point: `python -m squid_control microscope|mirror`
│
├── controller/              # High-level microscope orchestration
│   ├── squid_controller.py  # SquidController — the central application object
│   ├── scanning.py          # ScanningMixin — plate_scan(), flexible_position_scan()
│   └── acquisition.py       # AcquisitionMixin — snap_image(), autofocus, camera frames
│
├── hardware/                # Hardware abstraction layer
│   ├── core.py              # Low-level controllers: NavigationController, LiveController,
│   │                        #   AutoFocusController, MultiPointController, etc.
│   ├── microcontroller.py   # Serial protocol for Teensy 4.1 (USB, 2Mbaud)
│   ├── config.py            # Configuration system (Pydantic models + INI file loader)
│   ├── filter_wheel.py      # Filter wheel hardware driver
│   ├── objective_switcher.py# Motorised objective switcher (Squid+)
│   ├── serial_peripherals.py# Additional serial-connected peripherals
│   ├── processing_handler.py# Image processing pipeline handler
│   ├── camera/              # Camera drivers
│   │   ├── __init__.py      # get_camera() factory, TriggerModeSetting enum
│   │   ├── camera_default.py# GeniCam / simulation camera (Camera_Simulation class)
│   │   └── camera_toupcam.py# ToupCam vendor driver
│   └── drivers/             # Third-party SDK wrappers (do not modify)
│       ├── gxipy/           # Daheng Imaging GxiPy SDK
│       └── Xeryon.py        # Xeryon piezo stage driver
│
├── service/                 # Hypha RPC service layer (remote control API)
│   ├── __init__.py          # main(), signal handler, re-exports
│   ├── microscope_service.py# MicroscopeHyphaService — all RPC endpoints (~100 methods)
│   └── video_stream.py      # MicroscopeVideoTrack — WebRTC video frame producer
│
├── mirror/                  # Cloud-to-local proxy service
│   ├── __init__.py          # Re-exports MirrorMicroscopeService, MicroscopeVideoTrack
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
└── utils/                   # Shared utilities
    ├── logging_utils.py     # setup_logging() — rotating file + console handler
    ├── video_utils.py       # VideoBuffer, VideoFrameProcessor
    ├── image_processing.py  # rotate_and_flip_image() and other transforms
    └── illumination_calibration/  # LED intensity calibration tools
```

---

## Key Classes

### `SquidController` — `controller/squid_controller.py`

The central application object. Created once and shared between the service layer and any direct users. Owns all hardware controller instances.

```python
from squid_control.controller import SquidController

ctrl = SquidController(is_simulation=True)
await ctrl.snap_image(channel=0, intensity=30, exposure_time=50)
ctrl.move_to_well('C', 3)
```

**Inherits from:**
- `ScanningMixin` (`controller/scanning.py`) — plate scanning methods
- `AcquisitionMixin` (`controller/acquisition.py`) — image acquisition and autofocus

**Key attributes set in `__init__`:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `camera` | Camera | Main imaging camera |
| `microcontroller` | Microcontroller | Teensy serial interface |
| `navigationController` | NavigationController | Stage X/Y/Z movement |
| `liveController` | LiveController | Continuous acquisition mode |
| `autofocusController` | AutoFocusController | Contrast-based autofocus |
| `multipointController` | MultiPointController | Well plate scan orchestration |
| `laserAutofocusController` | LaserAutofocusController | Reflection-based autofocus |

---

### `MicroscopeHyphaService` — `service/microscope_service.py`

Exposes `SquidController` as a Hypha RPC service. All public methods decorated with `@schema_function` are automatically discoverable and callable by remote clients.

```python
from squid_control.service import MicroscopeHyphaService

svc = MicroscopeHyphaService(is_simulation=True)
await svc.setup()   # connects to Hypha, registers service, starts WebRTC
```

**Constructor parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `is_simulation` | `False` | Use simulated camera and stage |
| `is_local` | `False` | Connect to local Hypha server |
| `config_name` | `None` | INI config name (e.g. `"HCS_v2_63x"`) |

**Method categories:**

| Category | Example methods |
|----------|----------------|
| Stage movement | `move_to_position`, `move_by_distance`, `home_stage`, `navigate_to_well` |
| Illumination | `turn_on_illumination`, `turn_off_illumination`, `set_illumination` |
| Imaging | `snap_image` (via controller), `get_video_frame`, `adjust_video_frame` |
| Scanning | `scan_start`, `scan_get_status`, `scan_cancel`, `scan_plate_save_raw_images` |
| Autofocus | `contrast_autofocus`, `reflection_autofocus` |
| Zarr/Stitching | `quick_scan_region_to_zarr`, `get_stitched_region`, `upload_zarr_dataset` |
| Experiments | `create_experiment`, `list_experiments`, `set_active_experiment` |
| Hardware (Squid+) | `switch_objective`, `set_filter_wheel_position` |
| Service | `ping`, `is_service_healthy`, `get_status` |

---

### `MirrorMicroscopeService` — `mirror/mirror_service.py`

Connects to a local microscope service and re-registers all its methods on the cloud Hypha server. Used to provide public remote access to a microscope behind a firewall.

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

---

### Configuration — `hardware/config.py`

Global `CONFIG` object loaded from an INI file. Accessed throughout the codebase.

```python
from squid_control.hardware.config import CONFIG, load_config

# Load a specific config
load_config("HCS_v2_63x")

# Access values
print(CONFIG.CAMERA_TYPE)            # e.g. "Default"
print(CONFIG.ACQUISITION.CROP_WIDTH) # e.g. 3000
print(CONFIG.SOFTWARE_POS_LIMIT.X_POSITIVE)
```

Available configurations (INI files in the repo root):
- `HCS_v2` — standard 20x objective setup
- `HCS_v2_63x` — 63x oil immersion objective
- `Squid+` — extended hardware with filter wheel and objective switcher

---

### Simulation Mode

Simulation mode replaces all hardware with software equivalents:

| Real hardware | Simulation replacement |
|--------------|----------------------|
| Physical camera | `Camera_Simulation` — serves crops from Zarr archives |
| Teensy microcontroller | `Microcontroller_Simulation` — tracks position in memory |
| Stage movement | Instant coordinate update |
| Illumination | No-op with state tracking |

The virtual sample data is fetched from the Hypha Artifact Manager:
- Dataset: `agent-lens/20250824-example-data-*`
- Format: OME-Zarr ZIP archives, one per channel
- Resolution used: scale1 (1/4 of full resolution)

Channel mapping in simulation:

| Channel ID | Name |
|-----------|------|
| 0 | BF_LED_matrix_full |
| 11 | Fluorescence_405_nm_Ex |
| 12 | Fluorescence_488_nm_Ex |
| 14 | Fluorescence_561_nm_Ex |
| 13 | Fluorescence_638_nm_Ex |

---

## Hardware Communication

The PC communicates with a **Teensy 4.1 microcontroller** over USB serial at 2,000,000 baud. Each command is 8 bytes; each response is 24 bytes. Both use CRC8-CCITT checksums.

```
Command  (PC → Teensy):  [cmd_id, cmd_type, param×5, crc8]
Response (Teensy → PC):  [cmd_id, status, x×4, y×4, z×4, theta×4, switches, reserved×4, crc8]
```

See `hardware/microcontroller.py` for the full command set (`CMD_SET` enum).

---

## Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `REEF_WORKSPACE_TOKEN` | `service/`, `mirror/`, `storage/` | Hypha `reef-imaging` workspace token |
| `REEF_LOCAL_TOKEN` | `service/` | Local Hypha server token |
| `REEF_LOCAL_WORKSPACE` | `service/` | Local Hypha workspace name |
| `SQUID_CONFIG_PATH` | `controller/` | Override path to INI config file |

---

## Adding a New RPC Endpoint

1. Add a method to `MicroscopeHyphaService` in `service/microscope_service.py`
2. Decorate it with `@schema_function(skip_self=True)`
3. Use `Field(...)` for parameter descriptions (Pydantic)
4. Add the method name to the `info` dict in `start_hypha_service()`
5. Add a test in `tests/test_hypha_service.py`

```python
@schema_function(skip_self=True)
async def my_new_endpoint(
    self,
    value: float = Field(..., description="Some value in micrometers"),
    context=None,
) -> str:
    """Short description shown in API docs."""
    try:
        if context and not self.check_permission(context.get("user", {})):
            raise Exception("User not authorized")
        result = self.squidController.do_something(value)
        return f"Done: {result}"
    except Exception as e:
        logger.error(f"my_new_endpoint failed: {e}")
        raise
```

---

## Running Tests

```bash
# Simulation tests only (no network, no hardware)
pytest -m "not integration" -v

# Integration tests (requires REEF_WORKSPACE_TOKEN)
pytest -m integration -v

# Specific test file
pytest tests/test_squid_controller.py -v
```
