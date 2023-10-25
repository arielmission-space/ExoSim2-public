import photutils

from exosim.tasks.task import Task


class AperturePhotometry(Task):
    """
    It performs the aperture photometry using :func:`photutils.aperture.aperture_photometry`.

    The details of the aperture strongly depends on the configurations set by the user.

    """

    def __init__(self):
        """
        Parameters
        --------------
        table: :class:`astropy.table.QTable`
            wavelength table with bin edge
        focal_plane:
            focal plane
        """
        self.add_task_param(
            "table", "wavelength table with bin edges and aperture sizes"
        )
        self.add_task_param("focal_plane", "focal plane")

    def execute(self):
        self.debug("performing aperture photometry")
        table = self.get_task_param("table")
        focal_plane = self.get_task_param("focal_plane")

        aperture_shapes = {
            "rectangular": photutils.aperture.RectangularAperture,
            "elliptical": photutils.aperture.EllipticalAperture,
        }

        photometry = []
        for i in range(table["Wavelength"].size):
            # wavelength position estimated from the focal plane wavelength solution
            center_spectral = table["spectral_center"][i]
            spectral_size = table["spectral_size"][i]
            spatial_size = table["spatial_size"][i]
            center_spatial = table["spatial_center"][i]
            shape = table["aperture_shape"][i]

            aper = aperture_shapes[shape](
                (center_spectral, center_spatial),
                spectral_size,
                spatial_size,
            )

            phot = photutils.aperture.aperture_photometry(focal_plane, aper)
            photometry.append(phot["aperture_sum"].data[0])

        self.set_output(photometry)
