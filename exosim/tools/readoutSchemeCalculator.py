import os

import astropy.units as u
import h5py
import numpy as np

import exosim.tasks.instrument as instrument
from .exosimTool import ExoSimTool
from exosim.output.hdf5.utils import load_signal
from exosim.utils import check_units


class ReadoutSchemeCalculator(ExoSimTool):
    """
    This tool helps in the definition of the channels' readout schemes.
    Starting from the desired readout format, this tool estimates
    the best parameters to set in the channel description to best fit the ramp.

    Attributes
    ----------
    input: str
        input file
    options: dict
        This is parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`

    Examples
    --------
    >>> import exosim.tools as tools
    >>>
    >>> tools.ReadoutSchemeCalculator(options_file = 'tools_input_example.xml',
    >>>                               input_file='input_file.h5')
    """

    def __init__(self, options_file, input_file):
        super().__init__(options_file)
        self.input = input_file

        for ch in self.ch_list:
            focal_plane, frg_focal_plane = self.load_focal_plane(ch)
            self.info("Suggested readout scheme for {}".format(ch))
            (
                exposure_time,
                n_clk_GND,
                n_clk_ndr,
                n_NRDs_per_group,
                n_clk_GRP,
                n_GRPs,
                n_clk_RST,
            ) = self._compute_scheme(
                focal_plane, frg_focal_plane, self.ch_param[ch]
            )
            self._print_results(
                exposure_time,
                n_clk_GND,
                n_clk_ndr,
                n_NRDs_per_group,
                n_clk_GRP,
                n_GRPs,
                n_clk_RST,
            )
            # prepare scheme output
            read_dict = {
                "n_NRDs_per_group": n_NRDs_per_group,
                "n_groups": n_GRPs,
                "n_sim_clocks_Ground": n_clk_GND,
                "n_sim_clocks_Reset": n_clk_RST,
                "n_sim_clocks_first_NDR": n_clk_ndr,
                "n_sim_clocks_groups": n_clk_GRP,
                "exposure_time": exposure_time,
            }
            self.results.update({ch: read_dict})

    def _compute_scheme(self, focal_plane, frg_focal_plane, parameters):
        # compute saturation time
        computeSaturation = instrument.ComputeSaturation()
        (
            saturation_time,
            integration_time,
            max_signal,
            min_signal,
        ) = computeSaturation(
            well_depth=parameters["detector"]["well_depth"],
            f_well_depth=parameters["detector"]["f_well_depth"],
            focal_plane=focal_plane,
            frg_focal_plane=frg_focal_plane,
        )

        # load reading scheme parameters and convert to cycle clocks
        clock = check_units(parameters["readout"]["readout_frequency"], u.Hz)

        n_NRDs_per_group = parameters["readout"]["n_NRDs_per_group"]
        n_GRPs = parameters["readout"]["n_groups"]
        GND_time = parameters["readout"]["Ground_time"]
        RST_time = parameters["readout"]["Reset_time"]
        ndr_freq = check_units(
            parameters["readout"]["readout_frequency"], u.Hz
        )

        GND_time = check_units(GND_time, "s")
        n_clk_GND = int(np.ceil(GND_time * clock))
        RST_time = check_units(RST_time, "s")
        n_clk_RST = int(np.ceil(RST_time * clock))

        n_clk_ndr = int(np.ceil(clock / ndr_freq))

        if "exposure_time" in parameters["readout"].keys():
            exposure_time = parameters["readout"]["exposure_time"]
            exposure_time = check_units(exposure_time, "s")
        else:
            exposure_time = integration_time

        self.debug(
            "number of simulation clock in integration time: {}".format(
                exposure_time * clock
            )
        )
        available_clocks = int(
            np.floor(exposure_time * clock) - n_clk_RST - n_clk_GND - n_clk_ndr
        )

        # check for the number of groups
        clocks_per_group = (
            (n_NRDs_per_group - 1) * n_clk_ndr if n_NRDs_per_group > 1 else 1
        )
        n_clk_GRP = np.floor(
            available_clocks / (clocks_per_group * (n_GRPs - 1))
        ).astype(int)
        if n_clk_GRP < n_clk_ndr:
            while n_clk_GRP < n_clk_ndr:
                self.debug("reducing the number of groups")
                clocks_per_group = (
                    (n_NRDs_per_group - 1) * n_clk_ndr
                    if n_NRDs_per_group > 1
                    else 1
                )
                n_clk_GRP = np.floor(
                    available_clocks / (clocks_per_group * (n_GRPs - 1))
                ).astype(int)
                n_GRPs -= 1
                if n_GRPs <= 1:
                    self.warning(
                        "impossible to fit the desired number of groups inside the saturation time. Number of groups set to 1"
                    )
                    n_GRPs = 1
                    n_clk_GRP = 0
                    break

        # re-define exposures time
        exposure_time = self._get_exposure_time(
            n_clk_GND,
            n_clk_ndr,
            n_NRDs_per_group,
            n_clk_GRP,
            n_GRPs,
            n_clk_RST,
            clock,
        )
        if exposure_time > integration_time:
            self.warning(
                "Saturation problem: exposure time ({}) is {:.1f}% greater than saturation time ({}).".format(
                    exposure_time,
                    exposure_time / integration_time * 100,
                    integration_time,
                )
            )
            if "exposure_time" not in parameters["readout"].keys():
                while exposure_time > integration_time:
                    self.debug("reducing the number of NDRs")
                    exposure_time = self._get_exposure_time(
                        n_clk_GND,
                        n_clk_ndr,
                        n_NRDs_per_group,
                        n_clk_GRP,
                        n_GRPs,
                        n_clk_RST,
                        clock,
                    )
                    if n_NRDs_per_group == 1:
                        self.warning(
                            "impossible to fit the desired number of NDRs per group inside the saturation time. "
                            "Number of NDRs per group set to 1"
                        )
                        break
                    else:
                        n_NRDs_per_group -= 1
        return (
            exposure_time,
            n_clk_GND,
            n_clk_ndr,
            n_NRDs_per_group,
            n_clk_GRP,
            n_GRPs,
            n_clk_RST,
        )

    def _print_results(
        self,
        exposure_time,
        n_clk_GND,
        n_clk_ndr,
        n_NRDs_per_group,
        n_clk_GRP,
        n_GRPs,
        n_clk_RST,
    ):
        self.info("exposure time: {}".format(exposure_time))

        self.info("------------------------------------")
        self.info("n_NRDs_per_group:         {}".format(n_NRDs_per_group))
        self.info("n_GRPs:                   {}".format(n_GRPs))
        self.info("n_sim_clocks_Ground:      {}".format(n_clk_GND))
        self.info("n_sim_clocks_first_NDR:   {}".format(n_clk_ndr))
        self.info("n_sim_clocks_Reset:       {}".format(n_clk_RST))
        self.info("n_sim_clocks_groups:      {}".format(n_clk_GRP))
        self.info("------------------------------------")

    def _get_exposure_time(
        self,
        n_clk_GND,
        n_clk_ndr,
        n_NRDs_per_group,
        n_clk_GRP,
        n_GRPs,
        n_clk_RST,
        clock,
    ):
        return (
            n_clk_GND
            + n_clk_ndr * (n_NRDs_per_group)
            + (n_clk_GRP + n_clk_ndr * (n_NRDs_per_group - 1)) * (n_GRPs - 1)
            + n_clk_RST
        ) / clock.to(1 / u.s)

    def load_focal_plane(self, ch):
        """
        It loads the channel focal plane from the input file:

        Parameters
        ----------
        ch: str
            channel name

        Returns
        -------
        :class:`~exosim.models.signal.CountsPerSecond`
            focal plane
        :class:`~exosim.models.signal.CountsPerSecond`
            foreground focal plane
        """
        with h5py.File(self.input, "r") as f:
            file_path = os.path.join("channels", ch)
            fp = load_signal(f[os.path.join(file_path, "focal_plane")])
            frg_fp = load_signal(f[os.path.join(file_path, "frg_focal_plane")])

        self.debug("focal planes loaded from {}".format(ch))

        return fp, frg_fp
