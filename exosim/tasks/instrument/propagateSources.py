import astropy.units as u
from joblib import delayed
from joblib import Parallel

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
            channel efficiency
        responsivity:  :class:`~exosim.models.signal.Signal`
            channel responsivity
        """

        self.add_task_param("sources", "sources dictionary")
        self.add_task_param("Atel", "effective telescope Area")
        self.add_task_param("efficiency", "channel efficiency")
        self.add_task_param("responsivity", "channel responsivity")

    def execute(self):
        sources = self.get_task_param("sources")
        Atel = self.get_task_param("Atel")
        efficiency = self.get_task_param("efficiency")
        responsivity = self.get_task_param("responsivity")

        Parallel(n_jobs=RunConfig.n_job, require="sharedmem")(
            delayed(self._propagate)(
                source, Atel, efficiency, responsivity, sources
            )
            for source in sources.keys()
        )
        # for source in sources.keys():
        #     sources[source] *= Atel * efficiency * responsivity
        #     sources[source].to(u.ct / u.s / u.um)

        self.set_output(sources)

    def _propagate(self, source, Atel, efficiency, responsivity, sources):
        sources[source] *= Atel * efficiency * responsivity
        sources[source].to(u.ct / u.s / u.um)
