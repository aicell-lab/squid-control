# AGENTS.md

This file is the repository-level guide for coding agents working in `squid-control`.

## Project Summary

`squid-control` is a headless Python 3.11+ control system for the Cephla Squid microscope. It includes:

- Real hardware control through a Teensy 4.1 microcontroller
- Remote control over Hypha RPC and WebRTC
- A simulation mode backed by Zarr sample data
- Plate scanning and OME-Zarr stitching
- A mirror service for cloud-to-local proxying

## Repo Map

```text
squid_control/
├── controller/    High-level microscope orchestration
├── hardware/      Camera, stage, microcontroller, config, drivers
├── service/       Hypha RPC service and WebRTC streaming
├── mirror/        Cloud-to-local proxy service
├── stitching/     OME-Zarr canvas and stitching logic
├── storage/       Artifact upload and snapshot helpers
├── simulation/    Virtual sample data and simulation utilities
└── utils/         Logging, image processing, calibration, video helpers
```

Important entry points:

- `python -m squid_control microscope --simulation`
- `python -m squid_control microscope`
- `python -m squid_control microscope --local`
- `python -m squid_control mirror --local-service-id <id> --cloud-service-id <id>`

## Environment And Tooling

- Python: `>=3.11`
- Install for development: `pip install -e .[dev]`
- Lint and format with `ruff`
- Tests run with `pytest`

Useful commands:

```bash
ruff check .
ruff format .
pytest -m "not integration"
pytest tests/test_squid_controller.py
pytest tests/test_hypha_service.py
pytest tests/test_mirror_service.py
pytest tests/test_webrtc_e2e.py
```

## Working Agreements

### Prefer simulation first

- Start with simulation mode unless the task is explicitly hardware-only.
- Default validation path: `python -m squid_control microscope --simulation`
- Prefer tests marked for simulation or non-integration flows when making routine changes.

### Keep hardware-safe behavior

- Do not assume physical hardware is connected.
- Be careful around stage motion, illumination, and camera control code.
- Preserve cleanup paths around hardware access, especially `try`/`finally` blocks.

### Follow existing Python conventions

- Use type hints on new or changed functions where practical.
- Keep line length aligned with repo tooling (`ruff`, 88 columns).
- Use `async`/`await` for I/O-facing flows that are already asynchronous.
- Prefer descriptive names over abbreviations.
- Use `logger = logging.getLogger(__name__)`; avoid `print()`.

### Error handling

- Raise specific exceptions instead of returning error payloads from Python internals.
- Avoid bare `except:`.
- Log hardware and service failures with enough context to debug them.

## Project-Specific Conventions

### Configuration

- Use `from squid_control.hardware.config import CONFIG, load_config`
- Read configuration from `CONFIG` instead of hardcoding values.
- Supported config profiles mentioned in the repo include `HCS_v2`, `HCS_v2_63x`, and `Squid+`.

### Hypha service methods

When editing RPC-facing service methods:

- Use `@schema_function(skip_self=True)` where the module already follows that pattern.
- Use `Field(...)` descriptions for Pydantic-exposed parameters.
- Check permissions for write operations via `self.check_permission(context.get("user", {}))`.
- Keep service metadata registrations in sync with new methods.

### Image processing

- Crop before resize.
- Use `CONFIG.ACQUISITION.CROP_HEIGHT` and `CONFIG.ACQUISITION.CROP_WIDTH`.
- Preserve source bit depth through the processing path where possible.
- Bounds-check crop coordinates.

### Video streaming

- `squid_control/service/video_stream.py` uses buffered frame acquisition.
- Buffering is lazy-started and auto-stops after idle time.
- Tests may bypass buffering behavior, so keep test expectations aligned with that design.

## Testing Guidance

- For quick validation, start with `pytest -m "not integration"`.
- Integration tests may require tokens or network access.
- Hardware tests should stay opt-in.
- When changing service, controller, or streaming behavior, run the narrowest relevant test file first, then broaden if needed.

Common test markers in the repo include:

- `simulation`
- `integration`
- `slow`
- `local`
- `hardware`
- `unit`

## Environment Variables

Common variables used by the project:

- `REEF_WORKSPACE_TOKEN`
- `REEF_LOCAL_TOKEN`
- `REEF_LOCAL_WORKSPACE`
- `SQUID_CONFIG_PATH`

Some older test docs also mention environment variables for integration and local test control. Check `tests/README.md` and `tests/conftest.py` if a specific test flow needs extra setup.

## Agent Notes

- Keep changes narrow and consistent with the surrounding module.
- Prefer small, verifiable edits over broad refactors unless the task asks for one.
- Update nearby docs when behavior or developer workflow changes.
- If you add a new operational workflow, include the command that future agents should run to verify it.
