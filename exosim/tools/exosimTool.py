from collections import OrderedDict
from typing import List
from typing import Union

import exosim.tasks.load as load
from exosim.log import Logger


class ExoSimTool(Logger):
    """
    *Abstract class*

    Base class for exosim tools.

    Attributes
    ----------------
    ch_param: dict
        dictionary for the channels' configurations
    options: dict
        configurations dictionary
    results: dict
        dictionary for the results output
    """

    def __init__(self, options_file: Union[str, dict]) -> None:
        super().__init__()
        if isinstance(options_file, str):
            loadOption = load.LoadOptions()
            self.options = loadOption(filename=options_file)
        elif isinstance(options_file, dict):
            self.options = options_file

        self.ch_param = {}
        if isinstance(self.options["channel"], OrderedDict):
            for key, value in self.options["channel"].items():
                self.ch_param[key] = value
        else:
            self.ch_param[self.options["channel"]["value"]] = self.options[
                "channel"
            ]

        self.results = {}

    @property
    def ch_list(self) -> List[str]:
        """list of channel names"""
        if isinstance(self.options["channel"], OrderedDict):
            ch_list = list(self.options["channel"])
            ch_list.sort()
        else:
            ch_list = [self.options["channel"]["value"]]
        return ch_list
