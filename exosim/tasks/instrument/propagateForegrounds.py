import astropy.units as u
from joblib import delayed
from joblib import Parallel

from exosim.tasks.instrument import ComputeSolidAngle
from exosim.tasks.task import Task
from exosim.utils import RunConfig


class PropagateForegrounds(Task):
    """
    it propagates the foreground though the channel.

    Returns
    -------
    `~collections.OrderedDict`
        dictionary of :class:`~exosim.models.signal.Radiance` and :class:`~exosim.models.signal.Dimensionless`,
        represeting the radiance and efficiency of the path.
    """

    def __init__(self):
        """
        Parameters
        __________
        light_path: `~collections.OrderedDict`
            dictionary of :class:`~exosim.models.signal.Radiance` and :class:`~exosim.models.signal.Dimensionless`,
            represeting the radiance and efficiency of the path.
        responsivity:  :class:`~exosim.models.signal.Signal`
            channel responsivity
        parameters: dict
            channel parameter dictionary. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        """

        self.add_task_param("light_path", "sources dictionary")
        self.add_task_param("responsivity", "channel responsivity")
        self.add_task_param("parameters", "channel parameters dict")

    def execute(self):
        light_path = self.get_task_param("light_path")
        parameters = self.get_task_param("parameters")
        responsivity = self.get_task_param("responsivity")

        # for rad in [k for k in light_path.keys() if 'radiance' in k]:
        #     computeSolidAngle = ComputeSolidAngle()
        #     solid_angle = computeSolidAngle(parameters=parameters,
        #                                     other_parameters=light_path[
        #                                         rad].metadata)
        #     light_path[rad] *= solid_angle * responsivity
        #     light_path[rad].to(u.ct / u.s / u.um)
        computeSolidAngle = ComputeSolidAngle()
        Parallel(n_jobs=RunConfig.n_job, require="sharedmem")(
            delayed(self._computation_model)(
                rad,
                computeSolidAngle,
                parameters,
                light_path[rad].metadata,
                responsivity,
                light_path,
            )
            for rad in [k for k in light_path.keys() if "radiance" in k]
        )

        self.set_output(light_path)

    def _computation_model(
        self,
        rad,
        computeSolidAngle,
        parameters,
        other_parameters,
        responsivity,
        light_path,
    ):
        solid_angle = computeSolidAngle(
            parameters=parameters, other_parameters=other_parameters
        )
        light_path[rad] *= solid_angle * responsivity
        light_path[rad].to(u.ct / u.s / u.um)
