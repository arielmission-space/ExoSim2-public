import operator
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

from .pixelsNonLinearity import PixelsNonLinearity


class PixelsNonLinearityFromCorrection(PixelsNonLinearity):
    """
    This tools helps the user to find the pixel non-linearity coefficients to as inputs for ExoSim,
    starting from the measurable pixel non-linearity correction.

    In fact, the detector non linearity model, is usually written as polynomial such as

    .. math::
        Q_{det} = Q \\bigtriangleup (1 + \\sum_i a_i \\cdot Q^i)

    where :math:`Q_{det}` is the charge read by the detector, and :math:`Q` is the ideal count,
    as :math:`Q = \\phi_t`, with :math:`\\phi` being the number of electrons generated and :math:`t` being the elapsed time.
    In the equation above, :math:`\\bigtriangleup` is the operator used to defined the relation between :math:`Q_{det}` and :math:`Q`,
    which depends on the definition of the coefficients :math:`a_i` (see also equation below).

    However, it is usually the inverse operation that is known, as it's coefficients are measurable empirically:

    .. math::
        Q ={Q_{det}}\\bigtriangledown ( b_1 + \\sum_{i=2} b_i \\cdot Q_{det}^i)

    Where :math:`\\bigtriangledown` is the inverse operator of :math:`\\bigtriangleup`.
    Depending on the way the non linearity is estimated, the operator can either be a division (:math:`\\div`)
    or a multiplication (:math:`\\times`). If not specified, a division is assumed.

    The :math:`b_i` correction coefficients should be listed in the configuration file using the `pnl_coeff` keyword
    in increasing alphabetical order: `pnl_coeff_a` for :math:`b_1`,
    `pnl_coeff_b` for :math:`b_2`, `pnl_coeff_c` for :math:`b_3`,
    `pnl_coeff_d` for :math:`b_4`, `pnl_coeff_e` for :math:`b_5` and so on.
    The user can list any number of correction coefficients, and they will be automatically parsed.
    Please, note that using this notation, :math:`b_1` is not forced to be the unity.

    This class will restrieve the :math:`a_i` coefficients, starting from the the indicated :math:`b_i`.
    The results are the coefficients for a 4-th order polynomial:

    .. math::
        Q_{det} = Q \\cdot (a_1 + a_2 \\cdot Q + a_3 \\cdot Q^2 + a_4 \\cdot Q^3 + a_5 \\cdot Q^4)

    However, each pixel is different, and therefore, this class also produces a map of the coefficient for each pixel.
    Each coefficient is normally distributed around the mean value, with a standard deviation indicated in the configuration.
    If no standard deviation is indicated, the coefficients are assumed to be constant.

    The code output is a map of :math:`a_i` coefficients for each pixel, which can be injected into  :class:`~exosim.tasks.detector.applyPixelsNonLinearity.ApplyPixelsNonLinearity`.

    Attributes
    -----------
    operator_dict: dict
        dictionary of operators to use to estimates :math:`Q_{det}`
    Npt: int
        number of points used to estimate the coefficients

    Examples
    ----------

    >>> import exosim.tools as tools
    >>>
    >>> results = tools.PixelsNonLinearityFromCorrection(options_file='tools_input_example.xml',
    >>>                                                 output='output_pnl_map.h5')
    """

    operator_dict = {
        "/": operator.truediv,
        "*": operator.mul,
    }
    Npt = 2**10

    def compute_coefficients(
        self, parameters: dict, show_results: bool = True
    ) -> Tuple[List[float], float, float]:
        """
        It computes the non linearity coefficients.

        Parameters
        ----------
        parameters: dict
            dictionary contained the sources parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        show_results: bool
            it tells the code if showing the results in a plot. Default is `True`.
        """

        # retrieve correction coefficients
        keys = [
            k
            for k in parameters["detector"].keys()
            if "pnl_coeff" in k and "std" not in k
        ]
        if not keys:
            self.error("pixel non linear coefficients missing")
            raise KeyError("pixel non linear coefficients missing")

        keys.sort(reverse=True)
        coeff = [parameters["detector"][k] for k in keys]
        poli = np.poly1d(coeff)
        self.debug("correction coefficients: {}".format(coeff[::-1]))

        def poli_model(q, b, c, d, e):
            return 1 + b * q + c * q**2 + d * q**3 + e * q**4

        # build adu grid
        f = 1.0
        idx = [1023]
        while idx == [1023]:
            Qmax = parameters["detector"]["well_depth"]
            f += 0.1

            Qdet = np.linspace(
                1, f * Qmax.value, self.Npt
            )  # measured detector pixel counts in adu

            oper = (
                parameters["detector"]["pnl_correction_operator"]
                if "pnl_correction_operator" in parameters["detector"].keys()
                else "/"
            )
            Q = self.operator_dict[oper](
                Qdet, poli(Qdet)
            )  # linear detector pixel count in adu

            pnl_coeff, pcov = curve_fit(poli_model, Q, Qdet / Q)
            idx = np.argmin(np.abs(poli_model(Q, *pnl_coeff) - 0.95))
            self.debug("fitted coefficients: {}".format(pnl_coeff))
            self.debug("Well depth: {:.0f} (adu)".format(Q[idx]))

        if show_results:
            self._print_results(
                pnl_coeff,
                Q[idx],
            )
            self._plot(poli, Q[idx], pnl_coeff, poli_model, oper)

        return np.insert(pnl_coeff, 0, 1), Q[idx]

    def _plot(self, poli, Qmax, pnl_coeff, poli_model, oper) -> None:
        Qdet = np.linspace(
            1, Qmax * 1.2, self.Npt
        )  # detector pixel counts in adu
        Q = self.operator_dict[oper](Qdet, poli(Qdet))
        idx = np.argmin(np.abs(poli_model(Q, *pnl_coeff) - 0.95))

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle("pixel linearity model")
        ax.plot(Q, Q, "k", ls=":", label="Linear pixel count")
        ax.plot(Q, Qdet, "g", label="Detector pixel count")
        ax.plot(
            Q,
            Q * poli_model(Q, *pnl_coeff),
            "r",
            label="Non-linearity model",
            ls="--",
        )
        ax.axvline(
            Qmax,
            c="b",
            label="nominal saturation: {}".format(int(np.round(Qmax))),
            ls="--",
            alpha=0.5,
        )
        ax.axvline(
            Q[idx],
            c="k",
            label="real saturation (5% from linear): {}".format(
                int(np.round(Q[idx]))
            ),
            ls=":",
            alpha=0.5,
        )

        ax.set_xlabel("$Q$ [adu]")
        ax.set_ylabel("$Q_{det}$ [adu]")
        ax.legend()
        ax.grid()
        plt.show()
        # plt.savefig('wfc3_multiply.png')
