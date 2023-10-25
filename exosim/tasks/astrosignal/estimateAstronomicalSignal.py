from exosim.output import output
from exosim.tasks.task import Task
from exosim.utils.types import ArrayType


class EstimateAstronomicalSignal(Task):
    """
    It is a base class for all astronomical signal estimation tasks.
    If an output file is provided, the signal model is stored in the output file.

    Returns
    -------
    ArrayType
        returns the planetary signal in a 2D array (wavelength, time)
    """

    def __init__(self) -> None:
        """
        Parameters
        ----------
        timeline : :class:`astropy.units.Quantity`
            timeline to compute the signal
        wl_grid : :class:`astropy.units.Quantity`
            wavelength grid
        ch_parameters : dict
            channel parameters, by default {}
        source_parameters : dict
            sourec parameters, by default {}
        output : :class:`exosim.output.output.Output` (optional)
            output file, by default None
        """

        self.add_task_param("timeline", "timeline")
        self.add_task_param("source_parameters", "")
        self.add_task_param("ch_parameters", "")
        self.add_task_param("wl_grid", "")
        self.add_task_param("output", "", None)

    def execute(self):
        timeline = self.get_task_param("timeline")
        source_parameters = self.get_task_param("source_parameters")
        ch_parameters = self.get_task_param("ch_parameters")
        wl_grid = self.get_task_param("wl_grid")

        new_timeline, model = self.model(
            timeline, wl_grid, ch_parameters, source_parameters
        )

        # store the signal model in the output file
        output_file = self.get_task_param("output")
        if output_file and issubclass(output_file.__class__, output.Output):
            output_file.write_array(
                "signal_model",
                model,
                metadata={
                    "wavelength_axis": 0,
                    "time_axis": 1,
                },
            )
            output_file.write_quantity("model_timeline", new_timeline)
            output_file.write_quantity("model_wavelength", wl_grid)
            output_file.store_dictionary(source_parameters, "parameters")

        self.set_output([new_timeline, model])

    def model(
        self,
        timeline: ArrayType,
        wl_grid: ArrayType,
        ch_parameters: dict = {},
        source_parameters: dict = {},
    ) -> ArrayType:
        """Astronomical signal model to implement.

        Parameters
        ----------
        timeline : :class:`astropy.units.Quantity`
            timeline to compute the signal
        wl_grid : :class:`astropy.units.Quantity`
            wavelength grid
        ch_parameters : dict, optional
            channel parameters, by default {}
        source_parameters : dict
            source parameters, by default {}

        Returns
        -------
        ArrayType
            returns the planetary signal in a 2D array (wavelength, time)

        Raises
        ------
        NotImplementedError
            if the model is not implemented
        """
        raise NotImplementedError
