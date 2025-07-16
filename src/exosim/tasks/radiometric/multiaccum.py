from exosim.tasks.task import Task


class Multiaccum(Task):
    """
    It computes the MULTIACCUM gain factor from Rauscher and Fox 2007 (http://iopscience.iop.org/article/10.1086/520887/pdf)
    with the correction from Robberto 2009, also reported in Batalha 2017 (https://doi.org/10.1088/1538-3873/aa65b0)

    Returns
    ---------
    float:
        read noise gain factor
    float:
        shot noise gain factor
    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            dictionary containing the MULTIACCUM channel parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        """
        self.add_task_param("parameters", "multiaccum parameters dict")

    def execute(self):
        self.debug("computing MULTIACCUM")
        parameters = self.get_task_param("parameters")

        n = parameters["n"]
        m = parameters["m"]
        tf = parameters["tf"]
        tg = parameters["tg"]

        if n < 2:
            n = 2.0  # Force to CDS in nRead < 2

        read_gain = 12.0 * (n - 1.0) / (m * (n**2 + n))
        shot_gain = (
            6.0 / 5.0 * (n**2 + 1.0) / (n**2 + n) * (n - 1.0) * tg
            + (m**2 - 1.0) * (n - 1.0) / (m * (n**2 + n)) * tf
        )

        self.set_output([read_gain, shot_gain])
