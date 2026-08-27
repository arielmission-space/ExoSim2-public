import contextlib

import astropy.units as u
from joblib import Parallel, delayed

from exosim.tasks.task import Task
from exosim.utils import RunConfig


class PropagateSources(Task):
    """
    it propagates the sources though the channel.
    It multiplies the stellar SED by the effective telescope area, the channel efficiency and the channel responsivity:

    .. math::
        S_{\\nu} = S_{\\nu}^{*} \\times A_{tel} \\times \\eta \\times R_{\\nu}

    The result is in units of :math:`ct/s/\\mu m`

    Returns
    -------
    dict
        dictionary containing :class:`~exosim.models.signal.Signal`
    """

    def __init__(self):
        """
        Parameters
        __________
        sources:  dict
            dictionary containing :class:`~exosim.models.signal.Sed`
        Atel:  :class:`~astropy.units.Quantity`
            effective telescope Area
        efficiency:  :class:`~exosim.models.signal.Dimensionless`
            channel efficiency, defaults to 1
        responsivity:  :class:`~exosim.models.signal.Signal`
            channel responsivity, defaults to 1
        """

        self.add_task_param("sources", "sources dictionary")
        self.add_task_param("Atel", "effective telescope Area", None)
        self.add_task_param("efficiency", "channel efficiency", None)
        self.add_task_param("responsivity", "channel responsivity", None)

    def execute(self):
        sources = self.get_task_param("sources")
        Atel = self.get_task_param("Atel")
        efficiency = self.get_task_param("efficiency")
        responsivity = self.get_task_param("responsivity")

        if Atel is None:
            Atel = u.Quantity(1.0, u.dimensionless_unscaled)
        if efficiency is None:
            efficiency = u.Quantity(1.0, u.dimensionless_unscaled)
        if responsivity is None:
            responsivity = u.Quantity(1.0, u.dimensionless_unscaled)

        Parallel(n_jobs=RunConfig.n_job, require="sharedmem")(
            delayed(self._propagate)(source, Atel, efficiency, responsivity, sources)
            for source in sources
        )

        self.set_output(sources)

    def _propagate(self, source, Atel, efficiency, responsivity, sources):
        sources[source] *= Atel * efficiency * responsivity
        with contextlib.suppress(u.UnitConversionError):
            sources[source].unit = u.ct / u.s / u.um
