from exosim.tasks.task import Task


class ComputePhotonNoise(Task):
    r"""
    Computes the photon noise.
    Given the incoming signal :math:`S` the resulting photon noise variance is :math:`Var[S]=S`.
    If photon gain factor :math:`gain_{phot}` is given, then  :math:`Var[S]= gain_{phot} \cdot Var[S]`.
    If photon noise margin :math:`\chi` is found in the description, then  :math:`Var[S]= (1+\chi) \cdot Var[S]`.
    The noise returned is :math:`\sigma = \sqrt{Var[S]}`

    Returns
    -------
    astropy.table.QTable:
       photon noise
    """

    def __init__(self):
        """
        Parameters
        ----------
        signal: :class:`astropy.units.Quantity`
            signal
        description: dic (optional)
            channel description
        multiaccum_gain: :class:`numpy.ndarray` (optional)
            multiaccum gain factor for shotnoise
        """
        self.add_task_param("signal", "signal array")
        self.add_task_param("description", "channel description", None)
        self.add_task_param("multiaccum_gain", "channel description", None)

    def execute(self):
        self.debug("compute photon noise")
        signal = self.get_task_param("signal")
        description = self.get_task_param("description")
        multiaccum_gain = self.get_task_param("multiaccum_gain")

        variance = signal * signal.unit
        if description:
            if "photon_margin" in description["radiometric"].keys():
                self.debug(
                    "photon margin found:",
                    description["radiometric"]["photon_margin"],
                )
                variance *= 1 + description["radiometric"]["photon_margin"]
        if multiaccum_gain is not None:
            variance *= multiaccum_gain

        self.set_output(variance**0.5)
