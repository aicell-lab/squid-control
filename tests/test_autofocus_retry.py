import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from squid_control.hardware.config import CONFIG


def _load_acquisition_mixin():
    acquisition_path = (
        Path(__file__).resolve().parents[1]
        / "squid_control"
        / "controller"
        / "acquisition.py"
    )
    spec = importlib.util.spec_from_file_location(
        "squid_control_acquisition_test_module",
        acquisition_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.AcquisitionMixin


AcquisitionMixin = _load_acquisition_mixin()


class _ReflectionAutofocusTestController(AcquisitionMixin):
    def __init__(self):
        self.is_simulation = False
        self.autofocusController = SimpleNamespace(use_focus_map=False)
        self.navigationController = Mock()
        self.microcontroller = Mock()
        self.laserAutofocusController = Mock()


@pytest.mark.asyncio
async def test_reflection_autofocus_retries_from_default_z():
    controller = _ReflectionAutofocusTestController()
    controller.laserAutofocusController.move_to_target.side_effect = [
        RuntimeError("spot not found"),
        None,
    ]
    controller.microcontroller.is_busy.side_effect = [True, False]

    await controller.reflection_autofocus()

    assert controller.laserAutofocusController.move_to_target.call_count == 2
    controller.navigationController.move_z_to.assert_called_once_with(
        CONFIG.DEFAULT_Z_POS_MM
    )
