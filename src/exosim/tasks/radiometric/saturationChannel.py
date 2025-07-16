import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

import exosim.tasks.instrument as instrument
from exosim.tasks.task import Task


class SaturationChannel(Task):
    """
    It computes and adds the saturation time to the radiometric table using :class:`~exosim.tasks.instrument.computeSaturation.ComputeSaturation`.

    Returns
    -------
    astropy.units.Quantity:
        saturation time
    astropy.units.Quantity:
        maximum signal in focal plane
    astropy.units.Quantity:
        minimum signal in focal plane
    """

    def __init__(self):
        """
        Parameters
        --------------
        table: :class:`astropy.table.QTable`
            wavelength table with bin edge
        description: dict (optional)
            channel description
        input_file: :class:`~exosim.output.output.Output`
            input HDF5 file
        """
        self.add_task_param("table", "wavelength table with bin edges")
        self.add_task_param("description", "channel description", None)
        self.add_task_param(
            "input_file", "fraction of detector well depth to use"
        )

    def execute(self):
        input_file = self.get_task_param("input_file")
        description = self.get_task_param("description")
        table = self.get_task_param("table")

        ch = description["value"]
        computeSaturationTime = instrument.ComputeSaturation()
        with input_file.open() as f:
            f = f["channels"]
            focal_plane_units = u.Unit(f[ch]["focal_plane"]["data_units"][()])
            osf = f[ch]["focal_plane"]["metadata"]["oversampling"][()]
            focal_plane = (
                f[ch]["focal_plane"]["data"][
                    0, osf // 2 :: osf, osf // 2 :: osf
                ]
                * focal_plane_units
            )
            frg_focal_plane = (
                f[ch]["frg_focal_plane"]["data"][
                    0, osf // 2 :: osf, osf // 2 :: osf
                ]
                * focal_plane_units
            )

            sat_, int_, max_sig, min_sig = computeSaturationTime(
                well_depth=description["detector"]["well_depth"],
                f_well_depth=description["detector"]["f_well_depth"],
                focal_plane=focal_plane,
                frg_focal_plane=frg_focal_plane,
            )
            sat = [sat_] * len(table[table["ch_name"] == ch])
            integration_time = [int_] * len(table[table["ch_name"] == ch])

            # compute maximum signal in bin
            if description["type"].lower() == "spectrometer":
                wl_grid = f[ch]["focal_plane"]["spectral"][
                    osf // 2 :: osf
                ] * u.Unit(f[ch]["focal_plane"]["spectral_units"][()])
                wl_sol = interp1d(
                    wl_grid,
                    np.arange(0, focal_plane.shape[1]),
                    fill_value="extrapolate",
                )

                f = focal_plane + frg_focal_plane
                max_signal_in_bin, min_signal_in_bin = [], []
                for wl_l, wl_r in zip(
                    table["left_bin_edge"][table["ch_name"] == ch],
                    table["right_bin_edge"][table["ch_name"] == ch],
                ):
                    wld = int(wl_sol(wl_l))
                    wlu = int(wl_sol(wl_r))

                    if wld > wlu:
                        wld, wlu = wlu, wld
                    if wld < 0:
                        wld = 0
                    if wlu > f.shape[1]:
                        wlu = f.shape[1] - 1
                    max_signal_in_bin += [np.max(f[:, wld : wlu + 1])]
                    min_signal_in_bin += [np.min(f[:, wld : wlu + 1])]

            elif description["type"].lower() == "photometer":
                max_signal_in_bin = [np.max(focal_plane + frg_focal_plane)]
                min_signal_in_bin = [np.min(focal_plane + frg_focal_plane)]

        self.set_output(
            [sat, integration_time, max_signal_in_bin, min_signal_in_bin]
        )
