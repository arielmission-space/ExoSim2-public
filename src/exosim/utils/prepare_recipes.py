import os.path
import shutil

import exosim.tasks.load as load
from exosim.log import with_logger
from exosim.utils.run_config import RunConfig


def load_options(options_file):
    """
    It loads the configuration files into dictionaries.

    Parameters
    ----------
    options_file: str, dict, or None
        configuration data to load

    Returns
    -------
    dict:
        main configuration dictionary
    dict:
        payload configuration dictionary

    """
    if isinstance(options_file, str):
        loadOption = load.LoadOptions()
        mainConfig = loadOption(filename=options_file)
    elif isinstance(options_file, dict):
        mainConfig = options_file
    elif options_file is None:
        raise ValueError("options_file cannot be None")
    else:
        raise TypeError(f"options_file must be str or dict, got {type(options_file)}")

    payloadConfig = mainConfig["payload"]
    return mainConfig, payloadConfig


@with_logger
def copy_input_files(output_dir, logger=None):
    """
    It copied the input configuration xml file to the output folder, if they are not there already.

    Parameters
    ----------
    output_dir: str
        output folder

    """
    for fname in RunConfig.config_file_list:
        try:
            shutil.copy(fname, output_dir)
            logger.debug(f"{os.path.basename(fname)} copied in the destination folder")
        except shutil.SameFileError:
            logger.debug(f"{os.path.basename(fname)} already in the destination folder")
            continue


@with_logger
def clean_config_files(logger=None):
    """
    It clean the list of configuration files
    """
    RunConfig.config_file_list = []
    logger.debug(f"Configuration files list cleaned: {RunConfig.config_file_list}")
