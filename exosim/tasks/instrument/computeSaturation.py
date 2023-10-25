from copy import copy

import astropy.units as u
import numpy as np

from exosim.models.signal import CountsPerSecond
from exosim.models.signal import Signal
from exosim.tasks.task import Task
from exosim.utils.checks import check_units


class ComputeSaturation(Task):
    """
    It computes the saturation time given the focal plane and the detector parameters.

    Returns
    --------
    :class:`~astropy.units.Quantity`
        saturation time
    :class:`~astropy.units.Quantity`
        frame time
    :class:`~astropy.units.Quantity`
        maximum signal
    :class:`~astropy.units.Quantity`
        minimum signal
    """

    def __init__(self):
        """
        Parameters
        __________
        well_depth: :class:`~astropy.units.Quantity`
            detector well depth
        f_well_depth: float
            fraction of detector well depth to use
        focal_plane: :class:`~astropy.units.Quantity`
            focal plane array (with time evolution)
        frg_focal_plane: :class:`~astropy.units.Quantity`
            foreground focal plane array (with time evolution)

        """
        self.add_task_param("well_depth", "detector well depth")
        self.add_task_param(
            "f_well_depth", "fraction of detector well depth to use", None
        )
        self.add_task_param("focal_plane", "focal plane")
        self.add_task_param("frg_focal_plane", "foreground focal plane", None)

    def execute(self):
        well_depth = self.get_task_param("well_depth")
        f_well_depth = self.get_task_param("f_well_depth")
        focal_plane = self.get_task_param("focal_plane")
        frg_focal_plane = self.get_task_param("frg_focal_plane")

        well_depth = check_units(
            well_depth, desired_units=u.ct, calling_class=self
        )

        if isinstance(focal_plane, (Signal, CountsPerSecond)):
            focal_plane = focal_plane.data * focal_plane.data_units
        else:
            focal_plane = check_units(
                focal_plane, desired_units=u.ct / u.s, calling_class=self
            )

        if frg_focal_plane is not None:
            if isinstance(frg_focal_plane, (Signal, CountsPerSecond)):
                frg_focal_plane = (
                    frg_focal_plane.data * frg_focal_plane.data_units
                )
            else:
                frg_focal_plane = check_units(
                    frg_focal_plane,
                    desired_units=u.ct / u.s,
                    calling_class=self,
                )
            focal_plane += frg_focal_plane

        max_signal = np.max(focal_plane)
        min_signal = np.min(focal_plane)

        saturation_time = well_depth / max_signal
        self.debug("saturation time : {}".format(saturation_time))

        integration_time = copy(saturation_time)
        if f_well_depth is not None:
            integration_time *= f_well_depth
        self.debug("integration_time time : {}".format(integration_time))

        self.set_output(
            [saturation_time, integration_time, max_signal, min_signal]
        )
