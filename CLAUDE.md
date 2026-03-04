# Squid Microscope Control System — AI Assistant Rules

## Project Overview

Python control system for the [Squid microscope](https://www.cephla.com/) (Cephla Inc.), featuring:
- Real-time hardware control via Teensy 4.1 microcontroller
- Remote API via [Hypha RPC](https://hypha.aicell.io/) (WebSocket + WebRTC)
- Simulation mode with Zarr-based virtual samples (no hardware needed)
- Multi-well plate scanning and OME-Zarr image stitching
- Mirror service for cloud-to-local proxy control

## Technology Stack

| Category | Libraries |
|----------|-----------|
| Core | Python 3.11+, asyncio, NumPy |
| Hardware | PySerial (Teensy USB), OpenCV (headless) |
| Image processing | SciPy, scikit-image, PIL, TiffFile, Zarr |
| Remote API | Hypha RPC (`hypha-rpc`), aiortc (WebRTC) |
| Testing | pytest, pytest-asyncio, pytest-cov |
| Linting | ruff (format + lint), pre-commit |

Qt dependencies have been removed; the system runs headless.

---

## Package Structure

```
squid_control/
├── controller/          SquidController + ScanningMixin + AcquisitionMixin
├── hardware/            Hardware drivers — camera, microcontroller, stage, config
│   └── camera/          Camera factory + vendor drivers (default, toupcam)
├── service/             Hypha RPC service + WebRTC video streaming
├── mirror/              Cloud-to-local proxy service
├── stitching/           OME-Zarr image stitching (ZarrCanvas, ExperimentManager)
├── storage/             Artifact upload + snapshot utilities
├── simulation/          Virtual sample registry (Zarr datasets)
└── utils/               Logging, video buffer, image processing
```

**Key imports:**
```python
from squid_control.controller import SquidController
from squid_control.service import MicroscopeHyphaService
from squid_control.mirror import MirrorMicroscopeService
from squid_control.hardware.config import CONFIG, load_config
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `REEF_WORKSPACE_TOKEN` | Auth token for the `reef-imaging` Hypha workspace |
| `REEF_LOCAL_TOKEN` | Token for a private local Hypha server |
| `REEF_LOCAL_WORKSPACE` | Workspace name on the local server |
| `SQUID_CONFIG_PATH` | Absolute path to a custom INI config file |

Copy `.env.example` to `.env` before running.

---

## Entry Points

```bash
# Microscope service (simulation)
python -m squid_control microscope --simulation

# Microscope service (hardware)
python -m squid_control microscope [--local] [--config HCS_v2_63x]

# Mirror service (cloud proxy)
python -m squid_control mirror \
  --local-service-id "microscope-control-squid-1" \
  --cloud-service-id "microscope-control-squid-1"
```

**Classes:**
- `MicroscopeHyphaService` — `service/microscope_service.py` — all RPC endpoints
- `SquidController` — `controller/squid_controller.py` — hardware orchestration
- `MirrorMicroscopeService` — `mirror/mirror_service.py` — cloud proxy

---

## Coding Standards

### Python style
- PEP 8, 88-character line length (ruff-enforced)
- Type hints on all new functions and methods
- `async`/`await` for I/O and hardware communication
- Descriptive names: `exposure_time`, not `exp`

### Error handling
- Use specific exceptions (`except OSError:`, `except ValueError:`), never bare `except:`
- Log errors at the right level — hardware ops at INFO, hardware errors at ERROR with `exc_info=True`
- Use `raise` for API errors, never return error JSON
- Wrap hardware operations in try/finally for cleanup

### Logging
- Use `logger = logging.getLogger(__name__)` — never `print()`
- `setup_logging()` is in `squid_control/utils/logging_utils.py`
- Hardware operations → INFO; verbose state changes → DEBUG; zarr I/O → DEBUG

### Configuration
- Import: `from squid_control.hardware.config import CONFIG, load_config`
- Access values: `CONFIG.ACQUISITION.CROP_WIDTH`, `CONFIG.CAMERA_TYPE`
- INI profiles: `HCS_v2` (default), `HCS_v2_63x`, `Squid+`

### API endpoints
- Decorate with `@schema_function(skip_self=True)`
- Use `Field(...)` for all parameter descriptions (Pydantic)
- Always check `self.check_permission(context.get("user", {}))` for write operations
- Add method name to the `info` dict in `start_hypha_service()`

---

## Image Processing Standards

**Always crop before resize.** Use `CONFIG.ACQUISITION.CROP_HEIGHT/CROP_WIDTH`.

```python
crop_h = CONFIG.ACQUISITION.CROP_HEIGHT
crop_w = CONFIG.ACQUISITION.CROP_WIDTH
h, w = image.shape[:2]
y0 = max(0, h // 2 - crop_h // 2)
x0 = max(0, w // 2 - crop_w // 2)
y1 = min(h, y0 + crop_h)
x1 = min(w, x0 + crop_w)
cropped = image[y0:y1, x0:x1]
```

- Preserve original bit depth (MONO8/MONO12/MONO16) through the pipeline
- Always bounds-check crop coordinates

---

## Simulation Mode

Simulation replaces all hardware with software equivalents:

| Real | Simulated |
|------|-----------|
| Camera | `Camera_Simulation` — crops from Zarr archives |
| Teensy | `Microcontroller_Simulation` — tracks position in memory |
| Stage | Instant coordinate update |
| Illumination | No-op with state tracking |

**Start:** `python -m squid_control microscope --simulation`

**Virtual sample data:** fetched from `reef-imaging/20250824-example-data-*` on Hypha.

**Channel mapping:**
```python
{0: 'BF_LED_matrix_full', 11: 'Fluorescence_405_nm_Ex',
 12: 'Fluorescence_488_nm_Ex', 14: 'Fluorescence_561_nm_Ex',
 13: 'Fluorescence_638_nm_Ex'}
```

**Image effects simulated:**
- Exposure: `factor = max(0.1, exposure_time / 100)`
- Intensity: `factor = max(0.1, intensity / 60)`
- Z-blur: `gaussian_filter(image, sigma=abs(dz) * 6)`

> **Always test in simulation mode first.** Hardware testing requires additional setup.

---

## Video Buffering System (`service/video_stream.py`)

- Background task (`_frame_buffer_acquisition_loop`) pre-fetches frames at configurable FPS
- Frames stored in a thread-safe `deque`; WebRTC pulls the latest frame without waiting
- **Lazy start**: buffering begins on first `get_video_frame()` call
- **Auto-stop**: shuts down after 5 s with no active viewers (`video_idle_timeout`)
- **Tests**: buffering is disabled during pytest — tests use direct frame acquisition

---

## Mirror Service (`mirror/`)

Proxies all local microscope methods to the cloud Hypha server, enabling remote access through firewalls.

```
[Browser / remote client]
        │  WebRTC + RPC
        ▼
[hypha.aicell.io — cloud Hypha]
        │  RPC forwarding
        ▼
[MirrorMicroscopeService — workstation]
        │  local RPC
        ▼
[Local Hypha server — stable hardware control]
```

**Key behaviours:**
- Dynamically mirrors all callable methods from the local service to cloud
- WebRTC video streaming with metadata via data channels
- Automatic health checks with exponential-backoff reconnection
- Illumination managed automatically on WebRTC connect/disconnect

---

## Hardware Communication (Teensy 4.1)

USB serial at 2,000,000 baud. 8-byte commands, 24-byte responses, CRC8-CCITT checksums.

```
Command  (PC → Teensy): [cmd_id, cmd_type, param×5, crc8]
Response (Teensy → PC): [cmd_id, status, x×4, y×4, z×4, reserved×5, switches, reserved×4, crc8]
```

**Units:** microsteps internally; mm at the Python API level.
**Conversion:** `usteps = mm / (screw_pitch_mm / (microstepping × steps_per_rev))`

**Classes:** `Microcontroller` (real), `Microcontroller_Simulation` (testing), `Microcontroller2` (aux).

**Safety:** Software barriers defined in JSON; every movement validated against concave-hull boundaries.

---

## Zarr Canvas & Image Stitching (`stitching/zarr_canvas.py`)

### Classes
- **`ZarrCanvas`** — per-well OME-Zarr builder with coordinate conversion
- **`ExperimentManager`** — multi-well experiment lifecycle management

### Standards
- Format: OME-Zarr 0.4, axes T/C/Z/Y/X, 4× downsampling between scales
- Chunk size: 256×256 px; canvas dimensions must be divisible by chunk size
- Always use `zarr_lock` for multi-threaded access

### Critical: always validate bounds before writing
```python
# NEVER write zero-size slices — zarr will silently corrupt the array
if y_end > y_start and x_end > x_start:
    zarr_array[t, c, z, y_start:y_end, x_start:x_end] = data[...]
```

### ZIP export
```python
# Always use ZIP64 for large archives
with zipfile.ZipFile(buf, 'w', allowZip64=True, compression=zipfile.ZIP_STORED) as zf:
    ...

# Use forward slashes in ZIP paths (cross-platform)
arcname = "data.zarr/" + "/".join(relative_path.parts)  # correct
arcname = str(Path("data.zarr") / relative_path)         # wrong on Windows
```

---

## Conventions

### Coordinate system
- Stage: millimetres; Z positive = toward sample
- Camera: pixels
- Software barriers prevent out-of-bounds moves

### Channel IDs
| ID | Channel |
|----|---------|
| 0 | BF_LED_matrix_full |
| 11 | Fluorescence 405 nm |
| 12 | Fluorescence 488 nm |
| 13 | Fluorescence 638 nm |
| 14 | Fluorescence 561 nm |

### Well plate support
- Formats: 6, 12, 24, 96, 384 wells
- Rows: A–H; Columns: 1–12 (96-well)
- Navigate with `move_to_well(row, col)`

---

## Testing

```bash
# Simulation tests only (no network, no hardware)
pytest -m "not integration" -v

# Integration tests (needs REEF_WORKSPACE_TOKEN)
pytest -m integration -v

# Specific suite
pytest tests/test_squid_controller.py -v
```

Test files: `test_squid_controller.py`, `test_hypha_service.py`, `test_webrtc_e2e.py`, `test_upload_and_endpoints.py`, `test_mirror_service.py`

---

## Security
- Check `self.check_permission(context.get("user", {}))` before any write operation
- Validate all inputs with Pydantic `Field(...)`
- Never log tokens or credentials

---

## Working on this codebase

1. Test in simulation mode first — always
2. Never edit `hardware/core.py` without explicit instruction
3. Use `raise` for errors in RPC endpoints, not return values
4. Use specific except clauses — no bare `except:`
5. Use `logger` — no `print()`
6. Validate zarr bounds before every write
