import exosim.output as output
import exosim.utils.klass_factory as klass_factory
from exosim.tasks.load import LoadOpticalElement
from exosim.tasks.task import Task


class ParseOpticalElement(Task):
    """
    Given the element parameters, it parses the optical element and returns a dictionary.
    It also applyes the time variation if provided.

    Returns
    -------
    dict
        dictionary of :class:`~exosim.models.signal.Radiance` and :class:`~exosim.models.signal.Dimensionless`,
        represeting the radiance and efficiency of the optical element.
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

        element_task = (
            klass_factory.find_task(
                parameters["task_model"], LoadOpticalElement
            )
            if "task_model" in parameters.keys()
            else LoadOpticalElement
        )
        task_instance = element_task()
        radiance, efficiency = task_instance(
            parameters=parameters, wavelength=wl, time=tt
        )

        output_file = self.get_task_param("output")
        if output_file:
            if issubclass(output_file.__class__, output.Output):
                og_ = output_file.create_group(parameters["value"])
                radiance.write(og_, "radiance")
                efficiency.write(og_, "efficiency")

        out = {"radiance": radiance, "efficiency": efficiency}

        self.set_output(out)
