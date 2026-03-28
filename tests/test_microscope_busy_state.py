import asyncio
import contextvars
import importlib.util
import logging
import sys
import threading
import types
from pathlib import Path

import pytest


def _ensure_package(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


def _install_service_stubs() -> None:
    for package_name in (
        "squid_control",
        "squid_control.hardware",
        "squid_control.storage",
        "squid_control.controller",
        "squid_control.service",
        "squid_control.utils",
        "squid_control.simulation",
        "hypha_rpc",
        "hypha_rpc.utils",
    ):
        _ensure_package(package_name)

    aiortc_module = types.ModuleType("aiortc")
    aiortc_module.MediaStreamTrack = type("MediaStreamTrack", (), {})
    sys.modules["aiortc"] = aiortc_module

    hypha_rpc_module = sys.modules["hypha_rpc"]
    hypha_rpc_module.connect_to_server = lambda *args, **kwargs: None
    hypha_rpc_module.login = lambda *args, **kwargs: None
    hypha_rpc_module.register_rtc_service = lambda *args, **kwargs: None

    schema_module = types.ModuleType("hypha_rpc.utils.schema")

    def schema_function(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    schema_module.schema_function = schema_function
    sys.modules["hypha_rpc.utils.schema"] = schema_module

    camera_module = types.ModuleType("squid_control.hardware.camera")
    camera_module.TriggerModeSetting = type("TriggerModeSetting", (), {})
    sys.modules["squid_control.hardware.camera"] = camera_module

    config_module = types.ModuleType("squid_control.hardware.config")
    config_module.CONFIG = types.SimpleNamespace(
        WELLPLATE_OFFSET_X_MM=0.0,
        WELLPLATE_OFFSET_Y_MM=0.0,
        CACHE_CONFIG_FILE_PATH="",
    )

    # Add SIMULATED_CAMERA class to prevent import errors in other tests
    class SIMULATED_CAMERA:
        ORIN_X = 20
        ORIN_Y = 20
        ORIN_Z = 4
        MAGNIFICATION_FACTOR = 80

    config_module.SIMULATED_CAMERA = SIMULATED_CAMERA

    # Add wellplate format classes
    class WELLPLATE_FORMAT_96:
        NUMBER_OF_SKIP = 0
        WELL_SIZE_MM = 6.21
        WELL_SPACING_MM = 9
        A1_X_MM = 14.3
        A1_Y_MM = 11.36

    config_module.WELLPLATE_FORMAT_96 = WELLPLATE_FORMAT_96

    class ChannelMapper:
        @staticmethod
        def get_id_to_param_map():
            return {
                0: "BF_LED_matrix_full",
                11: "Fluorescence_405_nm_Ex",
                12: "Fluorescence_488_nm_Ex",
                13: "Fluorescence_638_nm_Ex",
                14: "Fluorescence_561_nm_Ex",
                15: "Fluorescence_730_nm_Ex",
            }

    config_module.ChannelMapper = ChannelMapper
    config_module.load_config = lambda *args, **kwargs: None
    sys.modules["squid_control.hardware.config"] = config_module

    snapshot_module = types.ModuleType("squid_control.storage.snapshot_utils")
    snapshot_module.SnapshotManager = type("SnapshotManager", (), {})
    sys.modules["squid_control.storage.snapshot_utils"] = snapshot_module

    controller_module = types.ModuleType("squid_control.controller.squid_controller")
    controller_module.SquidController = type("SquidController", (), {})
    sys.modules["squid_control.controller.squid_controller"] = controller_module

    video_stream_module = types.ModuleType("squid_control.service.video_stream")
    video_stream_module.MicroscopeVideoTrack = type("MicroscopeVideoTrack", (), {})
    sys.modules["squid_control.service.video_stream"] = video_stream_module

    logging_utils_module = types.ModuleType("squid_control.utils.logging_utils")
    logging_utils_module.setup_logging = logging.getLogger
    sys.modules["squid_control.utils.logging_utils"] = logging_utils_module

    video_utils_module = types.ModuleType("squid_control.utils.video_utils")
    video_utils_module.VideoBuffer = type("VideoBuffer", (), {})
    video_utils_module.VideoFrameProcessor = type("VideoFrameProcessor", (), {})
    sys.modules["squid_control.utils.video_utils"] = video_utils_module

    simulation_module = types.ModuleType("squid_control.simulation.samples")
    simulation_module.SAMPLE_ALIASES = {}
    simulation_module.SIMULATION_SAMPLES = {}
    sys.modules["squid_control.simulation.samples"] = simulation_module


def _load_service_module():
    _install_service_stubs()
    module_name = "busy_state_test_service_module"
    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        return existing_module

    module_path = (
        Path(__file__).resolve().parents[1]
        / "squid_control"
        / "service"
        / "microscope_service.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


service_module = _load_service_module()
MicroscopeBusyError = service_module.MicroscopeBusyError
MicroscopeHyphaService = service_module.MicroscopeHyphaService

pytestmark = pytest.mark.asyncio


class FakeNavigationController:
    def update_pos(self, microcontroller=None):
        return (1.0, 2.0, 3.0, 0.0)


class FakeLiveController:
    illumination_on = False

    def stop_live(self):
        return None


class FakeController:
    def __init__(self):
        self.navigationController = FakeNavigationController()
        self.liveController = FakeLiveController()
        self.microcontroller = object()
        self.current_channel = 0
        self.pixel_size_xy = 0.5
        self.scan_stop_requested = False
        self.stop_scan_calls = 0

    def move_by_distance_limited(self, x, y, z):
        return True, 0.0, 0.0, 0.0, x, y, z

    def get_well_from_position(self, well_plate_type):
        return {"well": "A1", "well_plate_type": well_plate_type}

    def create_experiment(self, experiment_name):
        return {"success": True, "experiment_name": experiment_name}

    def stop_scan_and_stitching(self):
        self.stop_scan_calls += 1


@pytest.fixture
def microscope_service():
    service = MicroscopeHyphaService.__new__(MicroscopeHyphaService)
    service.authorized_emails = None
    service.is_simulation = True
    service.is_local = True
    service.config_name = None
    service.current_x = 0
    service.current_y = 0
    service.current_z = 0
    service.current_theta = 0
    service.current_illumination_channel = None
    service.current_intensity = None
    service.is_illumination_on = False
    service.dx = 1
    service.dy = 1
    service.dz = 1
    service.BF_intensity_exposure = [30, 10]
    service.F405_intensity_exposure = [50, 100]
    service.F488_intensity_exposure = [50, 100]
    service.F561_intensity_exposure = [50, 100]
    service.F638_intensity_exposure = [50, 100]
    service.F730_intensity_exposure = [50, 100]
    service.channel_param_map = {
        0: "BF_LED_matrix_full",
        11: "Fluorescence_405_nm_Ex",
        12: "Fluorescence_488_nm_Ex",
        13: "Fluorescence_638_nm_Ex",
        14: "Fluorescence_561_nm_Ex",
        15: "Fluorescence_730_nm_Ex",
    }
    service.parameters = {}
    service.webrtc_service_id = None
    service.mjpeg_url = None
    service.frame_acquisition_running = False
    service.buffer_fps = 5
    service.scanning_in_progress = False
    service.scan_state = {
        "state": "idle",
        "error_message": None,
        "scan_task": None,
        "saved_data_type": None,
    }
    service._operation_state_lock = threading.RLock()
    service._scope_owners = {"hardware": None, "processing": None}
    service._operations_by_token = {}
    service._operation_counter = 0
    service._operation_context_token = contextvars.ContextVar(
        "test_operation_token",
        default=None,
    )
    service.squidController = FakeController()
    service.check_permission = lambda user: True
    return service


async def test_get_busy_status_initial_state(microscope_service):
    status = microscope_service.get_busy_status()

    assert status["busy"] is False
    assert status["hardware_busy"] is False
    assert status["processing_busy"] is False
    assert status["active_operations"] == []


async def test_get_status_includes_busy_state(microscope_service):
    status = microscope_service.get_status()

    assert "busy_status" in status
    assert status["busy_status"]["busy"] is False
    assert status["hardware_busy"] is False
    assert status["processing_busy"] is False


async def test_hardware_busy_rejects_move(microscope_service):
    token = microscope_service._acquire_operation("manual_hardware_busy", "hardware")
    try:
        with pytest.raises(MicroscopeBusyError, match="MICROSCOPE_BUSY"):
            microscope_service.move_by_distance(x=0.1, y=0.0, z=0.0)
    finally:
        microscope_service._release_operation(token)


async def test_processing_busy_rejects_experiment_mutation(microscope_service):
    token = microscope_service._acquire_operation(
        "manual_processing_busy",
        "processing",
    )
    try:
        with pytest.raises(MicroscopeBusyError, match="MICROSCOPE_BUSY"):
            await microscope_service.create_experiment("busy-test-experiment")
    finally:
        microscope_service._release_operation(token)


async def test_scan_start_claims_busy_and_blocks_other_hardware_calls(
    microscope_service,
    monkeypatch,
):
    release_scan = asyncio.Event()

    async def mock_scan(*args, **kwargs):
        await release_scan.wait()
        return "done"

    monkeypatch.setattr(microscope_service, "scan_plate_save_raw_images", mock_scan)

    result = await microscope_service.scan_start(
        config={
            "saved_data_type": "raw_images_well_plate",
            "action_ID": "busy-lock-test",
        }
    )

    assert result["success"] is True

    busy_status = microscope_service.get_busy_status()
    assert busy_status["hardware_busy"] is True
    assert busy_status["processing_busy"] is True
    assert busy_status["active_operations"][0]["name"] == "scan_start"

    with pytest.raises(MicroscopeBusyError, match="MICROSCOPE_BUSY"):
        microscope_service.move_by_distance(x=0.1, y=0.0, z=0.0)

    release_scan.set()
    await microscope_service.scan_state["scan_task"]

    final_busy_status = microscope_service.get_busy_status()
    assert final_busy_status["busy"] is False
    assert microscope_service.scan_state["state"] == "completed"


async def test_scan_cancel_keeps_busy_until_scan_exits(
    microscope_service,
    monkeypatch,
):
    release_scan = asyncio.Event()

    async def mock_scan(*args, **kwargs):
        while not microscope_service.squidController.scan_stop_requested:
            await asyncio.sleep(0.01)
        await release_scan.wait()
        return "stopped"

    monkeypatch.setattr(microscope_service, "scan_plate_save_raw_images", mock_scan)

    await microscope_service.scan_start(
        config={
            "saved_data_type": "raw_images_well_plate",
            "action_ID": "cancel-lock-test",
        }
    )

    cancel_result = await microscope_service.scan_cancel()
    assert cancel_result["success"] is True
    assert microscope_service.squidController.stop_scan_calls == 1

    busy_status_during_cancel = microscope_service.get_busy_status()
    assert busy_status_during_cancel["busy"] is True

    with pytest.raises(MicroscopeBusyError, match="MICROSCOPE_BUSY"):
        microscope_service.move_by_distance(x=0.1, y=0.0, z=0.0)

    release_scan.set()
    await microscope_service.scan_state["scan_task"]

    final_busy_status = microscope_service.get_busy_status()
    assert final_busy_status["busy"] is False
    assert microscope_service.scan_state["state"] == "failed"
    assert "cancel" in microscope_service.scan_state["error_message"].lower()
