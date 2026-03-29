import json
import logging
import os
from pathlib import Path

import pytest

from squid_control.service.microscope_service import MicroscopeHyphaService
from squid_control.service.simulation_zarr_asgi import create_simulation_zarr_asgi_app


LOGGER = logging.getLogger(__name__)

# Skip all tests in this file when running on GitHub Actions (no simulated data available)
pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Simulated data not available in GitHub Actions environment"
)


def _write_zarr_dataset(
    dataset_path: Path,
    *,
    chunk_bytes: bytes,
    include_zgroup: bool = True,
):
    dataset_path.mkdir(parents=True, exist_ok=True)
    if include_zgroup:
        (dataset_path / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")
    (dataset_path / ".zattrs").write_text(
        json.dumps(
            {
                "squid_canvas": {
                    "channel_mapping": {"BF_LED_matrix_full": 0},
                    "canvas_width_mm": 120.0,
                    "canvas_height_mm": 80.0,
                    "pixel_size_xy_um": 0.5,
                },
                "omero": {
                    "channels": [
                        {
                            "label": "BF_LED_matrix_full",
                            "color": "FFFFFF",
                            "activate": True,
                            "window": {"start": 0, "end": 255},
                        }
                    ]
                },
                "multiscales": [{"datasets": [{"path": "0"}]}],
            }
        ),
        encoding="utf-8",
    )
    scale_dir = dataset_path / "0"
    scale_dir.mkdir(exist_ok=True)
    (scale_dir / ".zarray").write_text(
        json.dumps(
            {
                "chunks": [1, 1, 1, 16, 16],
                "compressor": None,
                "dtype": "|u1",
                "fill_value": 0,
                "filters": None,
                "order": "C",
                "shape": [1, 1, 1, 16, 16],
                "zarr_format": 2,
            }
        ),
        encoding="utf-8",
    )
    (scale_dir / "0.0.0.0.0").write_bytes(chunk_bytes)


async def _call_asgi_app(app, path: str, *, method: str = "GET", headers=None):
    sent_messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent_messages.append(message)

    encoded_headers = []
    for key, value in (headers or {}).items():
        encoded_headers.append([key.lower().encode("latin-1"), value.encode("latin-1")])

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": encoded_headers,
    }

    await app({"scope": scope, "receive": receive, "send": send})

    start = next(
        message for message in sent_messages if message["type"] == "http.response.start"
    )
    body = b"".join(
        message.get("body", b"")
        for message in sent_messages
        if message["type"] == "http.response.body"
    )
    headers_dict = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in start["headers"]
    }
    return start["status"], headers_dict, body


def _prepare_service(
    monkeypatch,
    sample_map,
    *,
    initial_sample: str,
):
    import squid_control.service.microscope_service as microscope_service_module
    import squid_control.hardware.config as config_module

    monkeypatch.setattr(
        microscope_service_module,
        "SIMULATION_SAMPLES",
        sample_map,
    )
    monkeypatch.setattr(
        microscope_service_module,
        "_SAMPLE_ALIASES",
        {},
    )
    monkeypatch.setattr(config_module, "load_config", lambda *args, **kwargs: None)

    service = MicroscopeHyphaService(is_simulation=True, is_local=True)
    service.service_id = "test-simulated-microscope"
    service.simulation_zarr_url = (
        "https://example.test/reef-imaging/apps/test-simulated-microscope-zarr"
    )
    service.squidController.camera.ZARR_DATASET_PATH = sample_map[initial_sample][
        "zarr_dataset_path"
    ]
    service._sync_active_simulation_sample(sample_key=initial_sample)

    service.squidController.navigationController.update_pos = (
        lambda microcontroller=None: (0.0, 0.0, 0.0, 0.0)
    )
    service.squidController.get_well_from_position = lambda well_plate_type="96": {
        "well": "A1"
    }
    service.squidController.liveController.illumination_on = False
    service.squidController.current_channel = 0
    service.squidController.pixel_size_xy = 0.5

    return service


@pytest.mark.asyncio
async def test_simulation_zarr_asgi_follows_active_sample_and_switch(
    monkeypatch, tmp_path
):
    sample_a = tmp_path / "sample_a.zarr"
    sample_b = tmp_path / "sample_b.zarr"
    _write_zarr_dataset(sample_a, chunk_bytes=b"AAAA")
    _write_zarr_dataset(sample_b, chunk_bytes=b"BBBB")

    sample_map = {
        "SAMPLE_A": {
            "config_name": "HCS_v2",
            "zarr_dataset_path": str(sample_a),
            "description": "Sample A",
            "cell_line": "Cells A",
            "staining": "BF",
            "objective": "20x",
            "channels": ["BF_LED_matrix_full"],
        },
        "SAMPLE_B": {
            "config_name": "HCS_v2_63x",
            "zarr_dataset_path": str(sample_b),
            "description": "Sample B",
            "cell_line": "Cells B",
            "staining": "BF",
            "objective": "63x",
            "channels": ["BF_LED_matrix_full"],
        },
    }
    service = _prepare_service(monkeypatch, sample_map, initial_sample="SAMPLE_A")
    app = create_simulation_zarr_asgi_app(
        service._get_active_simulation_zarr_path, LOGGER
    )

    status_code, headers, body = await _call_asgi_app(app, "/0/0.0.0.0.0")
    assert status_code == 200
    assert headers["cache-control"] == "no-store"
    assert body == b"AAAA"

    range_status, range_headers, range_body = await _call_asgi_app(
        app,
        "/0/0.0.0.0.0",
        headers={"Range": "bytes=1-2"},
    )
    assert range_status == 206
    assert range_headers["content-range"] == "bytes 1-2/4"
    assert range_body == b"AA"

    switch_result = service.switch_sample("SAMPLE_B")
    assert switch_result["active_sample"] == "SAMPLE_B"
    assert switch_result["simulation_active_sample"] == "SAMPLE_B"
    assert (
        switch_result["simulation_zarr_service_id"] == "test-simulated-microscope-zarr"
    )
    assert (
        switch_result["simulation_zarr_url"]
        == "https://example.test/reef-imaging/apps/test-simulated-microscope-zarr"
    )

    status = service.get_status()
    assert status["simulation_active_sample"] == "SAMPLE_B"
    assert status["simulation_sample_info"]["name"] == "SAMPLE_B"
    assert status["simulation_sample_info"]["objective"] == "63x"
    assert status["simulation_zarr_service_id"] == "test-simulated-microscope-zarr"
    assert (
        status["simulation_zarr_url"]
        == "https://example.test/reef-imaging/apps/test-simulated-microscope-zarr"
    )

    switched_status_code, _, switched_body = await _call_asgi_app(app, "/0/0.0.0.0.0")
    assert switched_status_code == 200
    assert switched_body == b"BBBB"


@pytest.mark.asyncio
async def test_simulation_zarr_asgi_rejects_path_traversal_and_synthesizes_root_metadata(
    tmp_path,
):
    dataset_path = tmp_path / "sample_missing_zgroup.zarr"
    _write_zarr_dataset(
        dataset_path,
        chunk_bytes=b"CCCC",
        include_zgroup=False,
    )
    app = create_simulation_zarr_asgi_app(lambda: str(dataset_path), LOGGER)

    status_code, headers, body = await _call_asgi_app(app, "/")
    assert status_code == 200
    assert headers["content-type"] == "application/json"
    assert headers["cache-control"] == "no-store"
    assert json.loads(body.decode("utf-8")) == {"zarr_format": 2}

    forbidden_status, _, forbidden_body = await _call_asgi_app(app, "/../secret")
    assert forbidden_status == 403
    assert "outside zarr dataset directory" in forbidden_body.decode("utf-8").lower()
