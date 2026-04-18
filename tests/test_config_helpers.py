import os
import tempfile

from squid_control.hardware.config import update_ini_option


def test_update_ini_option_updates_default_z_pos_in_general_section():
    ini_contents = (
        "[GENERAL]\n"
        "default_z_pos_mm = 3.943  ; keep average focus height\n"
        "wellplate_format = 96\n"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "configuration_test.ini")
        with open(config_path, "w") as config_file:
            config_file.write(ini_contents)

        update_ini_option(config_path, "GENERAL", "default_z_pos_mm", "4.125")

        with open(config_path) as config_file:
            updated_contents = config_file.read()

    assert "default_z_pos_mm = 4.125  ; keep average focus height" in updated_contents
