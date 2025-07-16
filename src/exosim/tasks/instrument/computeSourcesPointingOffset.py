import astropy.units as u
from astropy.coordinates import SkyCoord

from exosim.tasks.task import Task


class ComputeSourcesPointingOffset(Task):
    """
    It computes the source offset on the focal plane respect to the pointing direction.
    The offset is in units of subpixels.

    Returns
    -------
     int
       offset in the spatial direction
     int
       offset in the spectral direction

    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            channel parameter dictionary. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        source:  dict
            dictionary containing :class:`~exosim.models.signal.Sed` metadata
        pointing: tuple
            telescope pointing direction, expressed as a tuple of RA and DEC. Default is ``None``

        """

        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param("source", "source source description dictionary")
        self.add_task_param("pointing", "telescope pointing")

    def execute(self):
        parameters = self.get_task_param("parameters")
        source = self.get_task_param("source")
        pointing = self.get_task_param("pointing")

        compute = True if pointing else False
        if "ra" in source["parsed_parameters"].keys():
            self.debug("RA found in source description")
            compute *= True
        else:
            compute *= False
        if "dec" in source["parsed_parameters"].keys():
            self.debug("DEC found in source description")
            compute *= True
        else:
            compute *= False
        if "plate_scale" in parameters["detector"].keys():
            self.debug("plate scale found in source description")
            compute *= True
        else:
            compute *= False

        if compute:
            aov_spatial, aov_spectral = angle_of_view(
                parameters["detector"]["plate_scale"],
                parameters["detector"]["delta_pix"],
                parameters["detector"]["oversampling"],
            )
            self.debug(
                "Angle of View estimated: {}, {}".format(
                    aov_spatial, aov_spectral
                )
            )
            c_tel = SkyCoord(pointing[0], pointing[1], frame="icrs")
            c_source = SkyCoord(
                source["parsed_parameters"]["ra"],
                source["parsed_parameters"]["dec"],
                frame="icrs",
            )
            offset_spatial = (
                c_tel.ra.deg - c_source.ra.deg
            ) / aov_spatial.value
            offset_spectral = (
                c_tel.dec.deg - c_source.dec.deg
            ) / aov_spectral.value
            self.debug(
                "offset estimated:{} {}".format(
                    offset_spectral, offset_spatial
                )
            )

        else:
            self.debug(
                "Angle of View computation skipped: missing information"
            )
            offset_spectral = offset_spatial = 0

        self.set_output([int(offset_spatial), int(offset_spectral)])


def angle_of_view(plate_scale, delta_pix, ovs):
    """
    Computes the Angle of View for a single pixel

    Parameters
    ----------
    plate_scale: :class:`astropy.units.Quantity`
        plate scale
    delta_pix: :class:`astropy.units.Quantity`
        size of a pixel
    Returns
    -------
    :class:`astropy.units.Quantity`
        angle of view in deg of each subpixel in the spatial direction
    :class:`astropy.units.Quantity`
        angle of view in deg of each subpixel in the spectral direction
    """

    def _compute_angle(plate_scale):
        if isinstance(plate_scale, u.Quantity):
            try:
                plate_scale.to(u.deg / u.micron)
                angle = plate_scale * delta_pix / ovs
            except u.UnitConversionError:
                try:
                    plate_scale.to(u.arcsec / u.pixel)
                    angle = plate_scale * u.pixel / ovs
                except u.UnitConversionError:
                    raise u.UnitConversionError(
                        "wrong plate scale units: {}".format(plate_scale.unit)
                    )

        else:
            raise OSError("missing plate scale units")

        return angle.to(u.deg)

    if isinstance(plate_scale, dict):
        spatial_angle = _compute_angle(plate_scale["spatial"])
        spectral_angle = _compute_angle(plate_scale["spectral"])
    else:
        spatial_angle = spectral_angle = _compute_angle(plate_scale)

    return spatial_angle, spectral_angle
