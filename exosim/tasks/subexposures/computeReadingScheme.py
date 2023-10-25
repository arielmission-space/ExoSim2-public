import astropy.units as u
import numpy as np

import exosim.output as output
from exosim.tasks.task import Task
from exosim.utils.checks import check_units


class ComputeReadingScheme(Task):
    """
    It computes the reading scheme for the sub-exposures given the instrument parameters and the focal planes.

    Returns
    -------
    :class:`astropy.units.Quantity`
        simulation frequency.
    :class:`numpy.ndarray`
        state machine for the reading operation on the ramp.
    :class:`numpy.ndarray`
        state machine of the group ends sampling the ramp.
    :class:`numpy.ndarray`
        state machine of the group starts sampling the ramp.
    :class:`numpy.ndarray`
        full list of simulation stapes for each steps on the ramp repeated by the number of ramps.
    int
        number of exposures needed to sample the full observation using ramps of the exposure time size.

    """

    def __init__(self):
        """
        Parameters
        ----------
        parameters: dict
            channel parameters dict
        main_parameters: dict
            main parameters dict
        output_file: :class:`~exosim.output.output.Output` (optional)
            output file

        """
        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param("main_parameters", "main parameters dict")
        self.add_task_param(
            "readout_oversampling", "readout oversampling factor", 1
        )
        self.add_task_param("output_file", "output file")

    def execute(self):
        parameters = self.get_task_param("parameters")
        main_parameters = self.get_task_param("main_parameters")
        readout_oversampling = self.get_task_param("readout_oversampling")
        output_file = self.get_task_param("output_file")

        # load reading scheme parameters and convert to cycle clocks
        total_observing_time = main_parameters["time_grid"]["end_time"]
        self.info("total observing time: {}".format(total_observing_time))

        clock_freq = check_units(
            parameters["readout"]["readout_frequency"], u.Hz
        )
        clock_freq *= readout_oversampling
        clock = check_units(clock_freq, u.s)

        n_NRDs_per_group = parameters["readout"]["n_NRDs_per_group"]
        n_GRPs = parameters["readout"]["n_groups"]
        n_clk_GND = (
            parameters["readout"]["n_sim_clocks_Ground"] * readout_oversampling
        )
        n_clk_NDR0 = (
            parameters["readout"]["n_sim_clocks_first_NDR"]
            * readout_oversampling
        )
        n_clk_NDR = (
            parameters["readout"]["n_sim_clocks_NDR"]
            if "n_sim_clocks_NDR" in parameters["readout"].keys()
            else 1
        )
        n_clk_NDR *= readout_oversampling
        n_clk_RST = (
            parameters["readout"]["n_sim_clocks_Reset"] * readout_oversampling
        )
        n_clk_GRP = (
            parameters["readout"]["n_sim_clocks_groups"] * readout_oversampling
        )

        # define exposures time
        exposure_time = (
            n_clk_GND
            + n_clk_NDR0
            + n_clk_NDR * (n_NRDs_per_group - 1)
            + (n_clk_GRP + n_clk_NDR * (n_NRDs_per_group - 1)) * (n_GRPs - 1)
            + n_clk_RST
        ) * clock

        self.info("exposure time: {}".format(exposure_time))

        # define number of exposures
        if "n_exposures" in parameters["readout"].keys():
            number_of_exposures = parameters["readout"]["n_exposures"]
            self.debug(
                "number of exposures manually set to {}".format(
                    number_of_exposures
                )
            )
        else:
            number_of_exposures = np.ceil(
                total_observing_time.to(u.s) / exposure_time
            ).astype(int)
            self.debug(
                "number of exposures automatically set to {}".format(
                    number_of_exposures
                )
            )
        self.info("number of exposures: {}".format(number_of_exposures))

        # prepare scheme output
        read_dict = {
            "n_NRDs_per_group": n_NRDs_per_group,
            "n_GRPs": n_GRPs,
            "n_clk_GND": n_clk_GND,
            "n_clk_RST": n_clk_RST,
            "n_clk_ndr": n_clk_NDR,
            "n_clk_GRP": n_clk_GRP,
            "exposure_time": exposure_time,
            "number_of_exposures": number_of_exposures,
        }

        if issubclass(output_file.__class__, output.Output):
            output_file.store_dictionary(read_dict, "reading_scheme_params")

        # build state machines
        base = (
            [n_clk_GND]
            + [n_clk_NDR0]
            + [n_clk_NDR] * (n_NRDs_per_group - 1)
            + ([n_clk_GRP] + [n_clk_NDR] * (n_NRDs_per_group - 1))
            * (n_GRPs - 1)
            + [n_clk_RST]
        )
        base_mask = np.array([0] + [1] * (n_NRDs_per_group) * (n_GRPs) + [0])

        self.info(
            "number of NDRs: {}".format(number_of_exposures * base_mask.sum())
        )

        frame_sequence = np.tile(base, number_of_exposures)
        self.set_output(
            [clock, base_mask, frame_sequence, number_of_exposures]
        )
