# test_suite/conftest.py

import contextlib
import os
import pathlib
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path

import astropy.units as u
import pytest

import exosim.tasks.load as load
from exosim.utils import RunConfig


# Existing fixture (kept for robustness, though basetemp in pytest.ini handles it)
@pytest.fixture(scope="session", autouse=True)
def ensure_temp_dir():
    tmp_base = Path(tempfile.gettempdir())
    tmp_base.mkdir(parents=True, exist_ok=True)
    fallback = tmp_base / "pytest-of-unknown"
    fallback.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def seed():
    return 42


@pytest.fixture(scope="session", autouse=True)
def set_random_seed(seed):
    RunConfig.random_seed = seed


@pytest.fixture(scope="session")
def project_root(request):
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def main_example_config_name():
    return "main_example.xml"


@pytest.fixture(scope="session")
def tools_example_config_name():
    return "tools_input_example.xml"


@pytest.fixture(scope="session")
def main_path(project_root):
    return project_root / "src" / "exosim"


@pytest.fixture(scope="session")
def example_dir(project_root):
    return project_root / "examples"


@pytest.fixture
def test_data_dir():
    # This fixture provides the test data directory in the new structure
    path = pathlib.Path(__file__).parent.absolute()
    return path / "data"


@pytest.fixture
def regression_data_dir():
    path = pathlib.Path(__file__).parent.absolute()
    return path / "regression"


@pytest.fixture
def payload_file(example_dir, test_data_dir, main_example_config_name):
    source_file = example_dir / main_example_config_name
    destination_file = test_data_dir / main_example_config_name

    test_data_dir.mkdir(parents=True, exist_ok=True)  # Ensure destination exists

    # Copy and modify content
    content = source_file.read_text()
    modified_content = ""
    new_config_path = f"    <ConfigPath> {example_dir}\n"  # Original source path
    for line in content.splitlines(keepends=True):
        if "<ConfigPath>" in line:
            modified_content += new_config_path
        else:
            modified_content += line
    destination_file.write_text(modified_content)
    return str(destination_file)


@pytest.fixture
def tools_file(example_dir, test_data_dir, tools_example_config_name):
    source_file = example_dir / tools_example_config_name
    destination_file = test_data_dir / tools_example_config_name

    test_data_dir.mkdir(parents=True, exist_ok=True)  # Ensure destination exists

    # Copy and modify content
    content = source_file.read_text()
    modified_content = ""
    new_config_path = f"    <ConfigPath> {example_dir}\n"  # Original source path
    for line in content.splitlines(keepends=True):
        if "<ConfigPath>" in line:
            modified_content += new_config_path
        else:
            modified_content += line
    destination_file.write_text(modified_content)
    return str(destination_file)


@pytest.fixture
def prepare_inputs_fixture(regression_data_dir, example_dir):
    """
    Fixture factory per creare la configurazione principale e assicurare che
    tutti i file XML richiesti siano presenti e coerenti.
    """

    def set_payload_file(
        source=example_dir,
        destination=regression_data_dir,
        name="main_example.xml",
        source_name="main_example.xml",
    ):
        """
        Copia un file di configurazione XML e aggiorna il <ConfigPath> al path corretto.
        """
        payload_config_file = os.path.join(source, source_name)
        new_config_path = f"    <ConfigPath> {destination}\n"
        target_file = os.path.join(destination, name)

        with contextlib.suppress(OSError):
            os.remove(target_file)

        with open(target_file, "w+") as new_file, open(payload_config_file) as old_file:
            for line in old_file:
                if "<ConfigPath>" in line:
                    new_file.write(new_config_path)
                else:
                    new_file.write(line)

        return target_file

    def create_payload_example_single(destination):
        """
        Copia il file payload_example_single.xml se serve e non esiste.
        """
        src = os.path.join(example_dir, "payload_example_single.xml")
        dst = os.path.join(destination, "payload_example_single.xml")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    def _prepare_inputs(filename="main_example.xml", single=False):
        """
        Prepara il file XML finale e carica la configurazione principale.
        """
        # Se richiedi 'single', crea comunque il file a partire da 'main_example.xml'
        source_name = "main_example.xml" if single else filename
        name = filename  # target filename comunque rimane filename

        config_path = set_payload_file(
            source=example_dir,
            destination=regression_data_dir,
            name=name,
            source_name=source_name,
        )

        if single:
            # cambia a mano la stringa nel file già copiato
            with open(config_path) as f:
                content = f.read()

            content = content.replace("payload_example", "payload_example_single")

            with open(config_path, "w") as f:
                f.write(content)

            create_payload_example_single(regression_data_dir)

        # Carica la configurazione con ExoSim
        load_option = load.LoadOptions()
        main_config = load_option(filename=str(config_path))

        # Override parametri di test
        main_config["sky"]["source"]["value"] = "HD 209458"
        main_config["sky"]["source"]["source_type"] = "planck"
        main_config["sky"]["source"]["R"] = 1.17967 * u.R_sun
        main_config["sky"]["source"]["D"] = 47.4567 * u.pc

        # Oversampling a 1
        if isinstance(main_config["payload"]["channel"], OrderedDict):
            for ch in main_config["payload"]["channel"]:
                main_config["payload"]["channel"][ch]["detector"]["oversampling"] = 1
        else:
            main_config["payload"]["channel"]["detector"]["oversampling"] = 1

        return main_config

    return _prepare_inputs
