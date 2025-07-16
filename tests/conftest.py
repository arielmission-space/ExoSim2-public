# tests/conftest.py

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
def test_data_dir(tmp_path):
    # This fixture provides a temporary directory for test data
    path = pathlib.Path(__file__).parent.absolute()
    return path / "test_data"

@pytest.fixture
def regression_data_dir(tmp_path):
    path = pathlib.Path(__file__).parent.absolute()
    return path / "regression_data"

@pytest.fixture
def payload_file(example_dir, test_data_dir, main_example_config_name):
    source_file = example_dir / main_example_config_name
    destination_file = test_data_dir / main_example_config_name

    test_data_dir.mkdir(parents=True, exist_ok=True) # Ensure destination exists

    # Copy and modify content
    content = source_file.read_text()
    modified_content = ""
    new_config_path = f"    <ConfigPath> {example_dir}\n" # Original source path
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

    test_data_dir.mkdir(parents=True, exist_ok=True) # Ensure destination exists

    # Copy and modify content
    content = source_file.read_text()
    modified_content = ""
    new_config_path = f"    <ConfigPath> {example_dir}\n" # Original source path
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

        try:
            os.remove(target_file)
        except OSError:
            pass

        with open(target_file, "w+") as new_file:
            with open(payload_config_file) as old_file:
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


@pytest.fixture(scope="function")
def phoenix_stellar_model(test_data_dir, project_root):
    # Copia i file dalla cartella sorgente alla cartella temporanea sed (senza shutil)
    source_path = project_root / "tests" / "test_data" / "sed"
    destination_path = test_data_dir / "sed"
    destination_path.mkdir(parents=True, exist_ok=True)

    for item in source_path.iterdir():
        if item.is_file():
            dest_file = destination_path / item.name
            if not dest_file.exists():
                with open(item, "rb") as src, open(dest_file, "wb") as dst:
                    dst.write(src.read())

    return str(destination_path)


@pytest.fixture(scope="function")
def phoenix_file(test_data_dir, project_root):
    source_file = project_root / "tests" / "test_data" / "sed" / "lte030.0-5.0-0.0a+0.0.BT-Settl.spec.fits.gz"
    destination_file = test_data_dir / "sed" / source_file.name
    destination_file.parent.mkdir(parents=True, exist_ok=True)

    if not destination_file.exists():
        with open(source_file, "rb") as src, open(destination_file, "wb") as dst:
            dst.write(src.read())

    return str(destination_file)

@pytest.fixture(scope="session")
def arielrad_data(test_data_dir, project_root):
    source_file = project_root / "tests" / "test_data" / "out.h5"
    destination_file = test_data_dir / "out.h5"
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_file, destination_file)
    return str(destination_file)

@pytest.fixture(scope="session")
def fast_test():
    return True

@pytest.fixture(scope="session")
def skip_plot():
    return True

@pytest.fixture(scope="session")
def missing_package():
    def _missing_package(package_name):
        import importlib
        try:
            importlib.import_module(package_name)
            return False
        except ImportError:
            return True
    return _missing_package
