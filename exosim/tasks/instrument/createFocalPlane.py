import gc

import exosim.models.signal as signal
import exosim.tasks.instrument as instrument
from exosim.tasks.task import Task


class CreateFocalPlane(Task):
    """
    It produces the empty focal plane

    Returns
    -------
    :class:`~exosim.models.signal.Signal`
        focal plane array (with time evolution)
    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            channel parameter dictionary. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        time: :class:`~astropy.units.Quantity`
            time grid.
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        group_name: str (optional)
            group name in output
        """

        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param("efficiency", "chennel efficiency")
        self.add_task_param("time", "time grid")
        self.add_task_param("output", "output file")
        self.add_task_param(
            "group_name", "group name in output", "focal plane"
        )

    def execute(self):
        self.info("creating focal plane")
        parameters = self.get_task_param("parameters")
        efficiency = self.get_task_param("efficiency")
        tt = self.get_task_param("time")
        output = self.get_task_param("output")
        group_name = self.get_task_param("group_name")

        createFocalPlaneArray = instrument.CreateFocalPlaneArray()
        focal_plane_array = createFocalPlaneArray(
            parameters=parameters, efficiency=efficiency
        )
        focal_plane_array.temporal_rebin(tt)

        focal_plane = signal.Signal(
            spectral=focal_plane_array.spectral,
            spatial=focal_plane_array.spatial,
            data=focal_plane_array.data,
            time=tt,
            cached=False,
            output=output,
            dataset_name=group_name,
            metadata=focal_plane_array.metadata,
        )

        self.set_output(focal_plane)

        del focal_plane
        gc.collect()
