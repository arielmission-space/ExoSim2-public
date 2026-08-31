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
        offset to add to the spectral pixel index
    int
        offset to add to the spatial pixel index

    Notes
    -----
    The two returned values are consumed by
    :class:`~exosim.tasks.instrument.populate_focal_plane.PopulateFocalPlane`
    and :class:`~exosim.tasks.astrosignal.apply_astronomical_signal.ApplyAstronomicalSignal`
    as ``offset_spectral, offset_spatial = compute_offset(...)``: the first value
    shifts the source along the detector spectral axis, the second along the
    spatial axis. Which celestial coordinate (RA or Dec) corresponds to which
    detector axis depends on the instrument orientation on the sky; this task
    keeps the historical assignment (RA drives the first value, Dec the second).
    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            channel parameter dictionary. This is usually parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`
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

        compute = bool(pointing)
        if "ra" in source["parsed_parameters"]:
            self.debug("RA found in source description")
            compute *= True
        else:
            compute *= False
        if "dec" in source["parsed_parameters"]:
            self.debug("DEC found in source description")
            compute *= True
        else:
            compute *= False
        if "plate_scale" in parameters["detector"]:
            self.debug("plate scale found in source description")
            compute *= True
        else:
            compute *= False

        if compute:
            # tolerate missing oversampling by using a sensible default (1)
            ovs = parameters["detector"].get("oversampling", 1)
            if "oversampling" not in parameters["detector"]:
                self.debug("'oversampling' missing in detector params, using default 1")

            aov_spatial, aov_spectral = angle_of_view(
                parameters["detector"]["plate_scale"],
                parameters["detector"]["delta_pix"],
                ovs,
            )
            self.debug(f"Angle of View estimated: {aov_spatial}, {aov_spectral}")
            c_tel = SkyCoord(pointing[0], pointing[1], frame="icrs")
            c_source = SkyCoord(
                source["parsed_parameters"]["ra"],
                source["parsed_parameters"]["dec"],
                frame="icrs",
            )
            # true angular offsets from the source to the pointing direction:
            # this handles the RA 0/360 wrap and the cos(dec) foreshortening,
            # unlike a plain difference of the RA/Dec values.
            d_ra, d_dec = c_source.spherical_offsets_to(c_tel)
            # historical assignment: the RA offset drives the first returned
            # value (used as the spectral-axis shift), the Dec offset the second
            # (spatial-axis shift). See the class docstring.
            offset_along_spectral_axis = d_ra.to(u.deg).value / aov_spatial.value
            offset_along_spatial_axis = d_dec.to(u.deg).value / aov_spectral.value
            self.debug(
                f"offset estimated: {offset_along_spectral_axis} "
                f"{offset_along_spatial_axis}"
            )

        else:
            self.debug("Angle of View computation skipped: missing information")
            offset_along_spectral_axis = offset_along_spatial_axis = 0

        # the offset is a whole number of sub-pixels: round to the nearest one
        self.set_output(
            [round(offset_along_spectral_axis), round(offset_along_spatial_axis)]
        )


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
                        f"wrong plate scale units: {plate_scale.unit}"
                    ) from None

        else:
            raise OSError("missing plate scale units")

        return angle.to(u.deg)

    if isinstance(plate_scale, dict):
        spatial_angle = _compute_angle(plate_scale["spatial"])
        spectral_angle = _compute_angle(plate_scale["spectral"])
    else:
        spatial_angle = spectral_angle = _compute_angle(plate_scale)

    return spatial_angle, spectral_angle
