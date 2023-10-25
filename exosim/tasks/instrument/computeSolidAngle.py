import astropy.units as u
import mpmath
import numpy as np

from exosim.tasks.task import Task
from exosim.utils.checks import check_units


class ComputeSolidAngle(Task):
    r"""
    It computes the solid angle given the system parameters.

    Returns
    --------
    :class:`~astropy.units.Quantity`
        solid angle expressed as :math:`sr \cdot m^2`
    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        other_parameters: dict
            additional dict accounting for signal metadata
        """
        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param(
            "other_parameters",
            "additional dict accounting for signal metadata",
            None,
        )

    def execute(self):
        parameters = self.get_task_param("parameters")
        other_parameters = self.get_task_param("other_parameters")

        # all solid angles validated against ArielRad2 on 07 Sept 2021. Exact match.
        if other_parameters and "solid_angle" in other_parameters.keys():
            if other_parameters["solid_angle"] == "pi":
                omega_pix = np.pi * u.sr
            elif other_parameters["solid_angle"] == "pi-omega_pix":
                f_numx = parameters["Fnum_x"]
                if "Fnum_y" in parameters.keys():
                    f_numy = parameters["Fnum_y"]
                else:
                    f_numy = None
                omega_pix = np.pi * u.sr - self._omega_pix(f_numx, f_numy)
            elif isinstance(other_parameters["solid_angle"], u.Quantity):
                omega_pix = check_units(
                    other_parameters["solid_angle"], u.sr, self
                )
        else:
            f_numx = parameters["Fnum_x"]
            if "Fnum_y" in parameters.keys():
                f_numy = parameters["Fnum_y"]
            else:
                f_numy = None
            omega_pix = self._omega_pix(f_numx, f_numy)

        self.debug("omega pix: {}".format(omega_pix))

        area_pix = (parameters["detector"]["delta_pix"] ** 2).to(u.m**2)
        self.debug("pixel area: {}".format(area_pix))

        solid_angle = area_pix * omega_pix
        self.debug("solid angle: {}".format(solid_angle))

        self.set_output(solid_angle)

    def _omega_pix(self, Fnum_x, Fnum_y=None):
        """
        Calculate the solid angle subtended by an elliptical aperture on-axis.
        Algorithm from "John T. Conway. Nuclear Instruments and Methods in
        Physics Research Section A: Accelerators, Spectrometers, Detectors and
        Associated Equipment, 614(1):17 ? 27, 2010.
        https://doi.org/10.1016/j.nima.2009.11.075
        Equation n. 56

        Parameters
        ----------
        Fnum_x : scalar
                 Input F-number along dispersion direction
        Fnum_y : scalar
                 Optional, input F-number along cross-dispersion direction

        Returns
        -------
         Omega : scalar
                 The solid angle subtended by an elliptical aperture in units u.sr

        """

        if not Fnum_y:
            Fnum_y = Fnum_x

        if Fnum_x > Fnum_y:
            a = 1.0 / (2 * Fnum_y)
            b = 1.0 / (2 * Fnum_x)
        else:
            a = 1.0 / (2 * Fnum_x)
            b = 1.0 / (2 * Fnum_y)

        h = 1.0

        A = 4.0 * h * b / (a * np.sqrt(h * h + a * a))
        k = np.sqrt((a * a - b * b) / (h * h + a * a))
        alpha = np.sqrt(1 - (b / a) * (b / a))

        Omega = 2.0 * np.pi - A * mpmath.ellippi(alpha * alpha, k * k)

        return Omega * u.sr
