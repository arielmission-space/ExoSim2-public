import logging
import os

import pytest

from exosim.log import setLogLevel
from exosim.tasks.load.loadOptions import LoadOptions

setLogLevel(logging.DEBUG)


@pytest.fixture
def create_payload_file(example_dir, test_data_dir):
    """Fixture per creare un file payload temporaneo."""
    payload_file_path = os.path.join(example_dir, "main_example.xml")
    tmp_file_path = os.path.join(test_data_dir, "payload_test.xml")

    def _create_payload_file(source):
        new_config_path = f"    <ConfigPath> {source}\n"
        try:
            os.remove(tmp_file_path)
        except OSError:
            pass

        with open(tmp_file_path, "w+") as new_file:
            with open(payload_file_path) as old_file:
                for line in old_file:
                    if "<ConfigPath>" in line:
                        new_file.write(new_config_path)
                    else:
                        new_file.write(line)
        return tmp_file_path

    yield _create_payload_file

    # Cleanup
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)


def test_load_options(create_payload_file, example_dir):
    loadOption = LoadOptions()
    payload_file_path = create_payload_file(source=example_dir)

    config = loadOption(filename=payload_file_path)
    assert config is not None
