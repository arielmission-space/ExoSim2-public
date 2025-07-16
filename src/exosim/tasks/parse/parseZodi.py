import numpy as np

import exosim.models.signal as signal
import exosim.output as output
import exosim.tasks.foregrounds as foregrounds
from exosim.tasks.task import Task


class ParseZodi(Task):
    """
    This tasks parses the zodiacal foreground.

    Returns
    -------
    dict
        dictionary of :class:`~exosim.models.signal.Radiance` and :class:`~exosim.models.signal.Dimensionless`,
        representing the radiance and efficiency of the optical element.
    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            dictionary contained the sources parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        wavelength: :class:`~astropy.units.Quantity`
            wavelength grid.
        time: :class:`~astropy.units.Quantity`
            time grid.
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        """
        self.add_task_param("parameters", "optical element parameters dict")
        self.add_task_param("wavelength", "wavelength grid")
        self.add_task_param("time", "time grid")
        self.add_task_param("output", "output file", None)

    def execute(self):
        parameters = self.get_task_param("parameters")
        self.debug("parsing: {}".format(parameters["value"]))

        wl = self.get_task_param("wavelength")
        tt = self.get_task_param("time")

        estimateZodi = foregrounds.EstimateZodi()
        # try to use zodiacal factor
        try:
            radiance = estimateZodi(
                wavelength=wl, zodiacal_factor=parameters["zodiacal_factor"]
            )
        # else, it try to use the map
        except KeyError:
            if "zodi_map" in parameters.keys():
                radiance = estimateZodi(
                    wavelength=wl,
                    coordinates=parameters["coordinates"],
                    zodi_map=parameters["zodi_map"],
                )
            else:
                radiance = estimateZodi(
                    wavelength=wl, coordinates=parameters["coordinates"]
                )
        efficiency = signal.Dimensionless(spectral=wl, data=np.ones(wl.size))
        efficiency.temporal_rebin(tt)

        output_file = self.get_task_param("output")
        if output_file:
            if issubclass(output_file.__class__, output.Output):
                og_ = output_file.create_group(parameters["value"])
                radiance.write(og_, "radiance")
                efficiency.write(og_, "efficiency")

        out = {"radiance": radiance, "efficiency": efficiency}

        self.set_output(out)
