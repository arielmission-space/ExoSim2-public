import logging
import os.path
import shutil

import exosim.tasks.load as load
from exosim.log import generate_logger_name
from exosim.utils.runConfig import RunConfig


def load_options(options_file):
    """
    It loads the configuration files into dictionaries.

    Parameters
    ----------
    options_file: str or dict
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
    payloadConfig = mainConfig["payload"]
    return mainConfig, payloadConfig


def copy_input_files(output_dir):
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
            logger.debug(
                "{} copied in the destination folder".format(
                    os.path.basename(fname)
                )
            )
        except shutil.SameFileError:
            logger.debug(
                "{} already in the destination folder".format(
                    os.path.basename(fname)
                )
            )
            continue


logger = logging.getLogger(generate_logger_name(copy_input_files))
