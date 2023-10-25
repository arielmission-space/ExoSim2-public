import os
import pathlib
from collections import OrderedDict

import astropy.units as u

import exosim.tasks.load as load
from exosim.utils import RunConfig

seed = 42
RunConfig.random_seed = seed

# Automatic relative directories and files
path = pathlib.Path(__file__).parent.absolute()
main_path = os.path.join(path.parent, "exosim")
example_dir = os.path.join(path.parent.absolute(), "examples")
test_dir = os.path.join(path, "test_data")
regression_dir = os.path.join(path, "regression_data")
main_example_config = "main_example.xml"
tools_example_config = "tools_input_example.xml"


def set_payload_file(
    source=example_dir,
    destination=test_dir,
    name=main_example_config,
    source_name=main_example_config,
):
    payload_config_file = os.path.join(example_dir, source_name)
    new_config_path = "    <ConfigPath> {}\n".format(source)
    tmp = os.path.join(destination, name)
    try:
        os.remove(tmp)
    except OSError:
        pass
    with open(tmp, "w+") as new_file:
        with open(payload_config_file) as old_file:
            for line in old_file:
                if "<ConfigPath>" in line:
                    new_file.write(new_config_path)
                else:
                    new_file.write(line)
    return tmp


payload_file = set_payload_file()


def set_tools_file(
    source=example_dir, destination=test_dir, name=tools_example_config
):
    tool_config_file = os.path.join(example_dir, tools_example_config)
    new_config_path = "    <ConfigPath> {}\n".format(source)
    tmp = os.path.join(destination, name)
    try:
        os.remove(tmp)
    except OSError:
        pass
    with open(tmp, "w+") as new_file:
        with open(tool_config_file) as old_file:
            for line in old_file:
                if "<ConfigPath>" in line:
                    new_file.write(new_config_path)
                else:
                    new_file.write(line)
    return tmp


tools_file = set_tools_file()


def prepare_inputs(filename=main_example_config, single=False):
    """
    It loads the main configuration file from the regression data directory

    Returns
    -------
    dict
        parsed configuration files
    """
    payload_file_name = set_payload_file(
        source=regression_dir, destination=regression_dir, name=filename
    )
    if single:
        change_payload_file(filename)
    load_option = load.LoadOptions()
    main_config = load_option(filename=payload_file_name)
    main_config["sky"]["source"]["value"] = "HD 209458"
    main_config["sky"]["source"]["source_type"] = "planck"
    main_config["sky"]["source"]["R"] = 1.17967 * u.R_sun
    main_config["sky"]["source"]["D"] = 47.4567 * u.pc

    # remove oversampling factor
    if isinstance(main_config["payload"]["channel"], OrderedDict):
        for ch in main_config["payload"]["channel"].keys():
            main_config["payload"]["channel"][ch]["detector"][
                "oversampling"
            ] = 1
    else:
        main_config["payload"]["channel"]["detector"]["oversampling"] = 1
    return main_config


def change_payload_file(filename):
    original_file = filename
    temp_file = "temp.xml"

    with open(original_file) as input:
        with open(temp_file, "w") as output:
            for line in input:
                for word in ["payload_example"]:
                    line = line.replace(word, "payload_example_single")
                output.write(line)
    # replace file with original name
    os.replace("temp.xml", filename)


# user custom directories and files
phoenix_stellar_model = os.path.join(test_dir, "sed")
phoenix_file = os.path.join(
    test_dir, "sed/lte030.0-5.0-0.0a+0.0.BT-Settl.spec.fits.gz"
)
arielrad_data = os.path.join(test_dir, "out.h5")

# test options
fast_test = False
skip_plot = True


def missing_package(package_name):
    import importlib

    try:
        importlib.import_module(package_name)
        return False
    except ImportError:
        return True
