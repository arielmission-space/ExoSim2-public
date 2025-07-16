import exosim.output as output
from exosim.tasks.task import Task


class LoadPsf(Task):
    """
    It loads the PSFs from files.

    Returns
    -------
    :class:`~numpy.ndarray`
        cube of psfs. axis=0 is time, axis=1 is wavelength, axis=2 is spatial direction, axis=3 is spectral direction.
    :class:`~numpy.ndarray`
        cube normalization factors. It contains the volume of each psf. axis=0 is time, axis=1 is wavelength.

    """

    def __init__(self):
        """

        Parameters
        __________
        parameters: dict
            dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        wavelength: :class:`~astropy.units.Quantity`
            wavelength grid.
        time: :class:`~astropy.units.Quantity`
           time grid.
        filename: str
            PSF input file
        output: :class:`~exosim.output.output.Output` (optional)
           output file
        """

        self.add_task_param("wavelength", "wavelength grid")
        self.add_task_param("time", "temporal grid")
        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param("filename", "PSF input file")
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("loading psf")
        wavelength = self.get_task_param("wavelength")
        time = self.get_task_param("time")
        filename = self.get_task_param("filename")
        parameters = self.get_task_param("parameters")

        psf_cube = self.model(filename, parameters, wavelength, time)
        norms = psf_cube.sum(axis=-1).sum(axis=-1)

        output_file = self.get_task_param("output")
        if output_file and issubclass(output_file.__class__, output.Output):
            output_group = output_file.create_group("psf")
            output_group.write_array(
                "psf_cube",
                psf_cube,
                metadata={
                    "spatial fp_axis": 2,
                    "spectral_fp_axis": 3,
                    "wavelength_axis": 1,
                    "time_axis": 0,
                },
            )
            output_group.write_array("norm", norms)
            output_group.write_quantity("wavelength", wavelength)

        self.set_output([psf_cube, norms])

    def model(self, filename, parameters, wavelength, time):
        """

        Parameters
        ----------
        filename: str
             PSF input file
        parameters: dict
             dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        wavelength: :class:`~astropy.units.Quantity`
             wavelength grid.

        Returns
        -------
        :class:`~numpy.ndarray`
            cube of psfs.  axis=0 is time, axis=1 is wavelength, axis=2 is spatial direction, axis=3 is spectral direction.

        """
        raise NotImplementedError
