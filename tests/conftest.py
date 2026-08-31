import logging
import os

# HDF5 file locking makes re-opening a just-written file flaky when several
# tests touch the same product in one session (BlockingIOError). The pipeline
# products are only ever read back within the test process, so disabling the
# lock is safe here. Must be set before h5py is imported anywhere.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import pathlib
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path

import pytest

# Disable numba debug logging that spams output
logging.getLogger("numba").setLevel(logging.WARNING)


@pytest.fixture
def regression_data_dir(project_root):
    """Path to regression test data directory."""
    path = os.path.join(project_root, "tests", "regression_data")
    os.makedirs(path, exist_ok=True)
    # Copy example files
    example_file = os.path.join(
        project_root, "tests", "test_data", "main_example_single.xml"
    )
    dest_file = os.path.join(path, "main_example_single.xml")
    if not os.path.exists(dest_file):
        shutil.copy2(example_file, dest_file)
    return path


# test_suite/conftest.py
# Configuration for the new organized test structure


@pytest.fixture(scope="session", autouse=True)
def ensure_temp_dir():
    tmp_base = Path(tempfile.gettempdir())
    tmp_base.mkdir(parents=True, exist_ok=True)
    fallback = tmp_base / "pytest-of-unknown"
    fallback.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def test_data_dir():
    """Fixture providing path to test data directory."""
    path = pathlib.Path(__file__).parent.absolute()
    return path / "test_data"


@pytest.fixture(scope="session")
def example_dir(project_root):
    """Fixture providing path to examples directory."""
    return project_root / "examples"


@pytest.fixture
def phoenix_stellar_model(test_data_dir, project_root):
    """Fixture providing path to Phoenix stellar model data."""
    # Check if data exists in test_data_dir first
    local_path = test_data_dir / "sed"
    if local_path.exists() and any(local_path.iterdir()):
        return str(local_path)

    # Fallback to project test_data if available
    source_path = project_root / "tests" / "test_data" / "sed"
    if source_path.exists():
        return str(source_path)

    # If no data found, tests will be skipped
    return None


@pytest.fixture
def phoenix_file(test_data_dir, project_root):
    """Fixture providing path to specific Phoenix stellar model file."""
    filename = "lte030.0-5.0-0.0a+0.0.BT-Settl.spec.fits.gz"

    # Check local test_data first
    local_file = test_data_dir / "sed" / filename
    if local_file.exists():
        return str(local_file)

    # Fallback to project test_data
    source_file = project_root / "tests" / "test_data" / "sed" / filename
    if source_file.exists():
        return str(source_file)

    return None


@pytest.fixture
def payload_file(test_data_dir):
    """Fixture providing path to payload configuration file."""
    import os

    payload_path = os.path.join(test_data_dir, "main_example.xml")
    if os.path.exists(payload_path):
        return payload_path
    return None


@pytest.fixture
def prepare_inputs_fixture(project_root):
    """Prepare inputs fixture that loads real XML configuration from examples."""
    import os

    from exosim.tasks.load import LoadOptions

    def _prepare_inputs(filename="main_example.xml", single=False):
        """Load configuration from example XML files."""
        # Use the example XML files with real configuration
        example_xml = os.path.join(project_root, "examples", filename)

        # Set config_path to examples directory (ABSOLUTE PATH) so __ConfigPath__ resolves correctly
        examples_dir = os.path.abspath(os.path.join(project_root, "examples"))

        # Load configuration using LoadOptions with correct config_path
        load_option = LoadOptions()
        main_config = load_option(filename=example_xml, config_path=examples_dir)

        if single and isinstance(main_config["payload"]["channel"], OrderedDict):
            # For single channel tests, keep only the first channel
            first_channel_name = next(iter(main_config["payload"]["channel"].keys()))
            main_config["payload"]["channel"] = OrderedDict(
                {
                    first_channel_name: main_config["payload"]["channel"][
                        first_channel_name
                    ]
                }
            )

        return main_config

    return _prepare_inputs
