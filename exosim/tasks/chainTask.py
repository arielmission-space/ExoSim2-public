import exosim.models.signal as signal
from .task import Task


class ChainTask(Task):
    """
    Abstract class to operate on a :class:`~exosim.models.signal.Signal` and return a :class:`~exosim.models.signal.Signal`.

    Returns
    --------
    :class:`~exosim.models.signal.Radiance`
        optical element radiance
    """

    def __init__(self):
        """
        Parameters
        __________
        signal: :class:`~exosim.models.signal.Signal`
            input signal
        parameters: dict
            dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        wavelength: :class:`~astropy.units.Quantity`
            wavelength grid.
        time: :class:`~astropy.units.Quantity`
            time grid
        """

        self.add_task_param("signal", "input signal")
        self.add_task_param("wavelength", "wavelength grid")
        self.add_task_param("time", "time grid")
        self.add_task_param("parameters", "optical element parameters dict")

    def execute(self):
        input_signal = self.get_task_param("signal")
        parameters = self.get_task_param("parameters")
        wl = self.get_task_param("wavelength")
        tt = self.get_task_param("time")

        output_signal = self.model(
            signal=input_signal, parameters=parameters, wavelength=wl, time=tt
        )
        if issubclass(output_signal, signal.Signal):
            self.set_output(output_signal)
        else:
            self.error("output is not a Signal")
            raise TypeError("output is not a Signal")

    def model(self, parameters, wavelength, time):
        """
        Parameters
        ----------
        signal: :class:`~exosim.models.signal.Signal`
            input signal
        parameters: dict
            dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        wavelength: :class:`~astropy.units.Quantity`
            wavelength grid.
        time: :class:`~astropy.units.Quantity`
            time grid.

        Returns
        --------
        :class:`~exosim.models.signal.Signal`
            output signal c
        """
        raise NotImplementedError
