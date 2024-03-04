from collections import OrderedDict

from exosim.tasks.astrosignal.estimateAstronomicalSignal import (
    EstimateAstronomicalSignal,
)
from exosim.tasks.task import Task
from exosim.utils.klass_factory import find_task


class FindAstronomicalSignals(Task):
    """
    This tasks find astronomical signals in the sky parameters dictionary.
    The signals are identified by the presence of the key "signal_task" in the dictionary.
    """

    def __init__(self) -> None:
        """
        Parameters
        ------------
        sky_parameters: dict
            sky parameters dictionary
        """
        self.add_task_param("sky_parameters", "sky parameters dictionary")

    def execute(self) -> None:
        self.sky_parameters = self.get_task_param("sky_parameters")
        self.signals_dict = {}
        if isinstance(self.sky_parameters["source"], OrderedDict):
            for source in self.sky_parameters["source"].keys():
                for effect in self.sky_parameters["source"][source].keys():
                    if isinstance(
                        self.sky_parameters["source"][source][effect], dict
                    ):
                        self.find_signals(
                            self.sky_parameters["source"][source][effect],
                            source,
                            effect,
                        )
        else:
            source = self.sky_parameters["source"]["value"]
            for effect in self.sky_parameters["source"].keys():
                if isinstance(self.sky_parameters["source"][effect], dict):
                    self.find_signals(
                        self.sky_parameters["source"][effect], source, effect
                    )
        if len(self.signals_dict) > 1:
            self.error(
                "Astronomical signals found in more than one star. "
                "The current version of ExoSim applies all the astronomical signals found in the target star. "
                "Please, check your sky parameters dictionary."
            )
            raise ValueError()

        self.set_output(self.signals_dict)

    def find_signals(self, input_dict: dict, source: str, effect: str) -> None:
        """
        Finds and stores the signal estimation task and parameters for a given source and effect.

        Parameters
        ----------
        input_dict : dict
            The dictionary containing the configuration parameters, including the optional key 'signal_task' which specifies the task responsible for signal estimation.
        source : str
            The name of the source for which the signals are to be found.
        effect : str
            The name of the effect to be considered in the signal estimation.

        """
        if "signal_task" in input_dict.keys():
            signal_task = find_task(
                input_dict["signal_task"], EstimateAstronomicalSignal
            )
            signal_dict = {
                "task": signal_task,
                "parsed_parameters": (
                    self.sky_parameters["source"][source]
                    if source in self.sky_parameters["source"].keys()
                    else self.sky_parameters["source"]
                ),
            }
            if source not in self.signals_dict.keys():
                self.signals_dict.update({source: {effect: signal_dict}})
            else:
                self.signals_dict[source].update({effect: signal_dict})
