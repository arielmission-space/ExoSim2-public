from copy import deepcopy

import astropy.units as u
import numpy as np

from exosim.tasks.task import Task


class ComputeTotalNoise(Task):
    """
    It computes the total noise from a radiometric table

    Returns
    -------
    table: :class:`astropy.table.QTable`
        total noise column
    """

    def __init__(self):
        """
        Parameters
        --------------
        table: :class:`astropy.table.QTable`
            wavelength table with bin edge
        """
        self.add_task_param("table", "wavelength table with bin edges")

    def execute(self):
        table = self.get_task_param("table")

        total_variance = np.zeros(table["Wavelength"].size) * u.hr

        noise_k = [key for key in table.keys() if "noise" in key]
        for key in noise_k:
            # first iter on photon noise
            if "_photon_noise" in key and (
                "source" not in key and "foreground" not in key
            ):
                continue
            if table[key].unit == u.ct / u.s:
                contrib = deepcopy(
                    table[key].filled(0.0)
                    if hasattr(table[key], "filled")
                    else table[key]
                )
                contrib /= table["source_signal_in_aperture"]
                total_variance += (contrib**2 / u.hr.to(u.s)) * u.hr

        table["total_noise"] = (
            total_variance**0.5
        )  # / table['source_signal_in_aperture']

        total_variance = table["total_noise"] ** 2
        for key in noise_k:
            # second iter on relative noise
            if table[key].unit == u.hr**0.5 or table[key].unit is None:
                contrib = (
                    table[key].filled(0.0)
                    if hasattr(table[key], "filled")
                    else table[key]
                )
                total_variance += contrib**2 * u.hr
        table["total_noise"] = total_variance**0.5

        self.set_output(table["total_noise"])
