from math import factorial
from typing import List
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .exosimTool import ExoSimTool
from exosim.output import SetOutput
from exosim.utils import RunConfig


class PixelsNonLinearity(ExoSimTool):
    r"""
    This tools helps the user to find the pixel non-linearity coefficients to as inputs for ExoSim,
    starting from the an estimate of pixel non-linearity correction.

    This class will retrieve the :math:`a_i` coefficients, starting from physical assumptions.

    The detector non linearity model, is written as polynomial such as

    .. math::
        Q_{det} = Q \cdot (1 + \sum_i a_i \cdot Q^i)

    where :math:`Q_{det}` is the charge read by the detector, and :math:`Q` is the ideal count,
    as :math:`Q = \phi_t`, with :math:`\phi` being the number of electrons generated and :math:`t` being the elapsed time.

    Considering the detector as a capacitor, the charge :math:`Q_{det}` is given by

    .. math::
        Q_{det} = \phi \tau \cdot \left(1 - e^{-Q/\phi \tau}\right)

    where :math:`\phi` is the charge generated in the detector pixel, and :math:`\tau` is the capacitor time constant.
    In fact the product :math:`\phi \tau` is constant
    :math:`Q` is the response of a linear detectror is given by :math:`Q = \phi t`

    The detector is considered saturated when the charge :math:`Q_{det}` at the well depth :math:`Q_{det, \, wd}`
    differs from the ideal well depth :math:`Q_{wd}` by 5%.

    .. math::
        Q_{det} = (1-5\%)Q_{wd}

    Then

    .. math::
        \phi \tau \cdot \left(1 - e^{-Q_{wd}/\phi \tau}\right) = (1-5\%)Q_{wd}

    This equation can be solved numerically and gives

    .. math::
        \frac{Q_{wd}}{\phi \tau} \sim 0.103479

    Therefore the detector collected charge is given by

    .. math::
            Q_{det} = \frac{Q_{wd}}{0.103479} \cdot \left(1 - e^\frac{- 0.103479 \, Q}{Q_{wd}}\right)

    Which can be approximated by a polynomial of order 4 as

    .. math::

            Q_{det} = Q\left[ 1- \frac{1}{2!}\frac{0.103479}{Q_{wd}} Q

            + \frac{1}{3!}\left(\frac{0.103479}{Q_{wd}}\right)^2 Q^2

            - \frac{1}{4!}\left(\frac{0.103479}{Q_{wd}}\right)^3 Q^3

            + \frac{1}{5!}\left(\frac{0.103479}{Q_{wd}}\right)^4 Q^4 \right]

    The results are the coefficients for a 4-th order polynomial:

    .. math::
        Q_{det} = Q \cdot (a_1 + a_2 \cdot Q + a_3 \cdot Q^2 + a_4 \cdot Q^3 + a_5 \cdot Q^4)

    However, each pixel is different, and therefore, this class also produces a map of the coefficient for each pixel.
    Each coefficient is normally distributed around the mean value, with a standard deviation indicated in the configuration.
    If no standard deviation is indicated, the coefficients are assumed to be constant.

    The code output is a map of :math:`a_i` coefficients for each pixel, which can be injected into  :class:`~exosim.tasks.detector.applyPixelsNonLinearity.ApplyPixelsNonLinearity`.


    Examples
    ----------

    >>> import exosim.tools as tools
    >>>
    >>> results = tools.PixelsNonLinearity(options_file='tools_input_example.xml',
    >>>                                    output='output_pnl_map.h5')
    """

    def __init__(
        self,
        options_file: Union[str, dict],
        output: str = None,
        show_results: bool = True,
    ) -> None:
        """

        Parameters
        ----------
        parameters: str or dict
            dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        output: str (optional)
            output file
        """
        super().__init__(options_file)
        if output is not None:
            out = SetOutput(output, replace=True)

        for ch in self.ch_list:
            out_dict = {}
            self.info(
                "computing pixel no linearity coefficients for {}".format(ch)
            )
            pnl_coeff, sat = self.compute_coefficients(
                self.ch_param[ch], show_results
            )
            out_dict["coeff"] = pnl_coeff
            out_dict["saturation"] = sat
            map = self.create_map(self.ch_param[ch], out_dict, show_results)
            out_dict["map"] = map
            self.results.update({ch: out_dict})

            if output is not None:
                with out.use(append=True) as o:
                    o.store_dictionary(out_dict, ch)

    def compute_coefficients(
        self, parameters: dict, show_results: bool = True
    ) -> Tuple[List[float], float]:
        """
        It computes the non linearity coefficients.

        Parameters
        ----------
        parameters: dict
            dictionary contained the sources parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        show_results: bool
            it tells the code if showing the results in a plot. Default is `True`.
        """

        Q_wd = parameters["detector"]["well_depth"].value
        phi_t = 0.103479  # stimated solution of (1-exp(Qdet/phiT)) = (1-0.05)Qdet/phiT
        constant = phi_t / Q_wd
        # compute the coefficients
        pnl_coeff = np.array(
            [
                (-1) ** i * 1 / factorial(i + 1) * constant**i
                for i in range(1, 5)
            ]
        )

        if show_results:
            self._print_results(
                pnl_coeff,
                Q_wd,
            )
            self._plot(Q_wd, pnl_coeff)

        return np.insert(pnl_coeff, 0, 1), Q_wd

    def _plot(self, saturation, pnl_coeff) -> None:
        Q = np.linspace(
            1, saturation * 1.2, 2**10
        )  # detector pixel counts in adu

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle("pixel linearity model")
        ax.axvline(
            saturation,
            c="k",
            label="real saturation (5% from linear): {}".format(
                int(np.ceil(saturation))
            ),
            ls="-",
        )

        p = np.polynomial.Polynomial(np.insert(pnl_coeff, 0, 1))
        plt.plot(Q, Q * p(Q), "g", label="Detector pixel count")

        ax.plot(Q, Q, "k", ls=":", label="Linear pixel count")

        ax.set_xlabel("$Q$ [adu]")
        ax.set_ylabel("$Q_{det}$ [adu]")
        ax.legend()
        ax.grid()
        plt.show()

    def _print_results(self, coeff: List[float], sat: float) -> None:
        """It prints the estimated coefficients and saturation level to screen.

        Parameters
        ----------
        coeff : list
            list of fitted coefficents
        sat : float
            saturation value
        """

        self.info("saturation (5% from linear): {} (counts)".format(int(sat)))

        self.info("------------------------------------")
        self.info("pnl_coeff_a:     1")
        self.info("pnl_coeff_b:     {}".format(coeff[0]))
        self.info("pnl_coeff_c:     {}".format(coeff[1]))
        self.info("pnl_coeff_d:     {}".format(coeff[2]))
        self.info("pnl_coeff_e:     {}".format(coeff[3]))
        self.info("------------------------------------")

    def create_map(
        self, parameters: dict, input_dict: dict, show_results: bool = True
    ) -> np.ndarray:
        """
        Create a map of the pixel non-linearity correction coefficients.
        To create a non linearity map of the detector, we randomize the coefficients.
        If not specified, the standard deviation is set to 0 and the coefficients are assumed to be standard.
        of the mean value of the polynomial coefficients

        Parameters
        ----------
        parameters: dict
            dictionary contained the sources parameters.
            This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`

        input_dict: dict
            dictionary produced by `compute_coefficients`.
            It contains the coefficients (`coeff`), the saturation ('saturation') and the estimated well depth (`well_depth`)

        show_results: bool
            it tells the code if showing the results in a plot. Default is `True`.


        Returns
        -------
        np.ndarray
            map of the coefficients. The shape is (4, nrow, ncol)
            The first axes refers to the coefficients (a, b, c, d, e)

        """

        map = np.ones(
            (
                input_dict["coeff"].size,
                parameters["detector"]["spatial_pix"],
                parameters["detector"]["spectral_pix"],
            )
        )

        if "pnl_coeff_std" in parameters["detector"]:
            std = parameters["detector"]["pnl_coeff_std"]
        else:
            std = 0.0

        for i in range(input_dict["coeff"].size):
            map[i] *= RunConfig.random_generator.normal(
                input_dict["coeff"][i],
                std * np.abs(input_dict["coeff"][i]),
                size=map.shape[1:],
            )

        if show_results:
            self._plot_map(input_dict, map)
        return map

    def _plot_map(self, input_dict, map):
        Q = np.linspace(
            1, input_dict["saturation"] * 1.2, 2**10
        )  # detector pixel counts in adu

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle("detector linearity model")
        ax.axvline(
            input_dict["saturation"],
            c="k",
            label="real saturation (5% from linear): {}".format(
                int(np.ceil(input_dict["saturation"]))
            ),
            ls="-",
        )

        coeffs = map.T.reshape(map.shape[1] * map.shape[2], map.shape[0])
        for cs in tqdm(coeffs, total=coeffs.shape[0], desc="preparing plot"):
            p = np.polynomial.Polynomial(cs)
            plt.plot(Q, Q * p(Q), c="g", lw=0.5, alpha=0.1)

        ax.plot(Q, Q, "k", ls=":", label="Linear pixel count")

        ax.set_xlabel("$Q$ [adu]")
        ax.set_ylabel("$Q_{det}$ [adu]")
        ax.legend()
        ax.grid()
        plt.show()
