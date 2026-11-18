import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

import exosim.tasks.instrument as instrument
from exosim.tasks.task import Task
from exosim.utils.klass_factory import find_task
from exosim.utils.prf import (
    DEFAULT_PRF_NODES,
    DEFAULT_PRF_SAMPLING,
    build_prf_tables,
    cached_tables,
    deposit_prf,
    file_signature,
    fine_pixel_response,
    node_weights,
    physical_axis_edges,
    prf_cache_enabled,
    prf_cache_key,
    wavelength_coverage,
    wavelength_to_subpixel,
)
from exosim.utils.psf import create_psf, scale_psf_image


class PopulateFocalPlanePrf(Task):
    """
    It populates the empty focal plane with the pixel response function (PRF).

    Instead of adding point-sampled monochromatic PSFs and convolving the
    plane with the intra-pixel response (the IRF), the PRF
    :math:`\\int PSF(r - r_s) P(r - r_p) d^2 r` is tabulated once on a fine
    grid, for a set of wavelength nodes, and deposited for every sample of the
    source spectrum, at the exact position :math:`r_s(\\lambda)` of the
    wavelength (continuous wavelength solution for the spectrometers, fixed
    position for the photometers, exact -not rounded- pointing offset). The
    source spectrum is used on its own wavelength grid, multiplied by its bin
    width: it must *not* be rescaled onto the focal plane columns.

    Every sub-pixel of the oversampled plane holds the signal that a physical
    pixel *centred on that sub-pixel* collects (the content of the plane after
    the IRF in the sampled population), hence the decimation
    ``[osf // 2 :: osf]`` gives the physical pixels, the result does not depend
    on the oversampling factor (``osf = 1`` is enough, a larger factor is only
    needed to jitter the plane) and the IRF must not be applied afterwards.
    The plane is flagged with ``metadata["pixel_response_applied"] = True``.

    The pixel response comes from the detector keys used by the IRF
    (``diffusion_length``, ``intra_pix_distance``, ``delta_pix``); a custom
    ``irf_task`` cannot be folded into the analytic response and is refused.
    The PSF is the analytic one (``<shape>``) or the one of
    :class:`~exosim.tasks.instrument.load_psf_paos.LoadPsfPaos`.
    Time dependent PSFs are not supported.

    The configuration is read from the ``psf`` section of the channel:

    - ``population``: ``prf`` (default) enables this task, ``sampled`` selects the
      point-sampled PSFs followed by the IRF. If the key is absent and the
      configuration is not supported (see
      :func:`~exosim.utils.prf.prf_unsupported_reason`) ExoSim falls back to
      ``sampled`` with a warning;
    - ``prf_sampling``: fine samples per physical pixel of the PRF tables
      (default 10);
    - ``prf_nodes``: number of wavelength nodes of the analytic PSF tables
      (default 24, the PAOS wavelengths are used for the PAOS PSFs);
    - ``prf_cache``: the tables depend only on the PSF configuration of the
      channel (not on the source, the target or the time), hence they are kept
      in a process-level cache (see :func:`~exosim.utils.prf.cached_tables`,
      the last 4 channels) and built once when several targets or time steps
      are populated with the same channel. ``False`` disables it (default
      ``True``). Every worker process has its own cache.

    Returns
    -------
    :class:`~exosim.models.signal.Signal`
        focal plane array populated
    :class:`~exosim.utils.prf.PrfTables`
        PRF tables
    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            channel parameter dictionary. This is usually parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`
        focal_plane: :class:`~exosim.models.signal.Signal`
            focal plane array (with time evolution)
        sources:  dict
            dictionary containing :class:`~exosim.models.signal.Sed` (propagated, on their own wavelength grid in ct/s/um)
        pointing: (:class:`astropy.units.Quantity`, :class:`astropy.units.Quantity`) (optional)
            telescope pointing direction, expressed ad a tuple of RA and DEC in degrees. Default is ``None``
        prf: :class:`~exosim.utils.prf.PrfTables` (optional)
            PRF tables. Default is ``None``, in which case they are built.
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        """

        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param("focal_plane", "focal plane")
        self.add_task_param("sources", "sources")
        self.add_task_param("prf", "PRF tables", None)
        self.add_task_param("output", "output file", None)
        self.add_task_param("pointing", "telescope pointing", None)

    def execute(self):
        parameters = self.get_task_param("parameters")
        focal_plane = self.get_task_param("focal_plane")
        sources = self.get_task_param("sources")
        output = self.get_task_param("output")
        prf = self.get_task_param("prf")
        pointing = self.get_task_param("pointing")

        if prf is None:
            prf = self.build_tables(parameters, focal_plane, output)
            # first call for the channel: store the PSF cube
            self.write_psf(parameters, focal_plane, output)

        if sources:
            for source in sources:
                self.info(f"populating focal plane with {source} (PRF)")
                self.deposit_source(
                    parameters, focal_plane, sources[source], prf, pointing
                )
                focal_plane.data_units = sources[source].data_units * u.um

        # a new dictionary: the metadata are shared with the copies of the plane
        # (foreground and background planes), which are not populated here
        focal_plane.metadata = {
            **focal_plane.metadata,
            "pixel_response_applied": True,
            "psf_population": "prf",
            "prf_sampling": int(prf.sampling),
        }
        self.set_output([focal_plane, prf])

    @staticmethod
    def write_psf(parameters, focal_plane, output):
        """
        Stores in the output file the PSF cube (and its norm and wavelengths)
        that the sampled population stores, in the ``psf`` group: the
        point-sampled PSF on the oversampled grid, at the wavelengths of the
        focal plane. The PRF does not use it, but the instrument line shapes
        of the sub-exposures are built from it
        (:class:`~exosim.tasks.subexposures.load_ils.LoadILS`), which applies
        the pixel response later. The cube is built, with
        :meth:`~exosim.tasks.instrument.populate_focal_plane.PopulateFocalPlane.load_psf`,
        only if there is an output file, and it is not kept in memory.
        """
        if output is None:
            return
        instrument.PopulateFocalPlane.load_psf(parameters, focal_plane, output)

    # ------------------------------------------------------------------ tables
    def build_tables(self, parameters, focal_plane, output=None):
        """
        Builds the PRF tables of a channel.

        Parameters
        ----------
        parameters: dict
            channel parameters
        focal_plane: :class:`~exosim.models.signal.Signal`
            (empty) focal plane
        output: :class:`~exosim.output.output.Output` (optional)
            output file

        Returns
        -------
        :class:`~exosim.utils.prf.PrfTables`
        """
        self.check_configuration(parameters)
        psf_pars = parameters["psf"]
        sampling = int(psf_pars.get("prf_sampling", DEFAULT_PRF_SAMPLING))
        if sampling < 1:
            raise ValueError("<psf><prf_sampling> must be a positive integer")
        enabled = prf_cache_enabled(parameters)

        def build():
            detector = parameters["detector"]
            nodes, fine_psf = self.load_fine_psf(parameters, focal_plane, sampling)
            response = fine_pixel_response(
                detector["delta_pix"],
                detector.get("diffusion_length", 0.0 * u.um),
                detector.get("intra_pix_distance", 0.0 * u.um),
                sampling,
            )
            tables = build_prf_tables(fine_psf, nodes, response, sampling)
            self.info(
                f"PRF tables: {nodes.size} wavelengths, {sampling} samples per pixel, "
                f"size {tables.tables.shape[1:]} ({tables.nbytes / 1e6:.1f} MB)"
            )
            return tables

        key = self.cache_key(parameters, focal_plane) if enabled else None
        prf = cached_tables(key, build, enabled)
        nodes = prf.nodes
        if output is not None:
            group = output.create_group("prf")
            try:
                group.write_quantity("wavelength", nodes * u.um)
                group.write_array("pixel_response", prf.response)
                group.write_array("energy", prf.energy)
            except ValueError as e:  # the datasets already exist
                if "name already exists" not in str(e):
                    raise
        return prf

    @staticmethod
    def cache_key(parameters, focal_plane):
        """
        Key of the process-level cache of the tables (see
        :func:`~exosim.utils.prf.cached_tables`): a hash of everything the
        tables depend on, that is the PSF configuration (the PAOS file with its
        size and modification time, or the analytic shape, window and
        wavelength nodes), the F-numbers, the pixel size, the diffusion length
        and the inter-pixel gap and the sampling. It does
        *not* depend on the source, the target, the time or the channel name.
        """
        psf_pars = parameters["psf"]
        detector = parameters["detector"]
        psf = {
            k: v for k, v in psf_pars.items() if k not in ("population", "prf_cache")
        }
        if "filename" in psf:
            psf["filename"] = file_signature(psf["filename"])
        psf.setdefault("prf_sampling", DEFAULT_PRF_SAMPLING)
        osf = int(focal_plane.metadata.get("oversampling", 1))
        config = {
            "psf": psf,
            "plane": (
                focal_plane.data.shape[1] // osf,
                focal_plane.data.shape[2] // osf,
            ),
            "Fnum_x": parameters.get("Fnum_x"),
            "Fnum_y": parameters.get("Fnum_y"),
            "detector": {
                k: detector.get(k)
                for k in (
                    "delta_pix",
                    "diffusion_length",
                    "intra_pix_distance",
                    "spatial_pix",
                    "spectral_pix",
                )
            },
        }
        # the PAOS tables are extended beyond the PAOS wavelengths up to the band
        config["band"] = PopulateFocalPlanePrf.band_edges(parameters, focal_plane)
        if "psf_task" not in psf_pars:
            psf.setdefault("nzero", 4)
            psf.setdefault("prf_nodes", DEFAULT_PRF_NODES)
            if "size" in psf_pars:
                # the size of the window is given in sub-pixels
                config["oversampling"] = osf
        return prf_cache_key(config)

    @staticmethod
    def check_configuration(parameters):
        """
        Refuses the configurations the PRF population cannot reproduce (see
        :func:`~exosim.utils.prf.prf_unsupported_reason`).
        """
        detector = parameters["detector"]
        if "irf_task" in detector:
            klass = find_task(
                detector["irf_task"], instrument.CreateIntrapixelResponseFunction
            )
            if klass is not instrument.CreateIntrapixelResponseFunction:
                raise ValueError(
                    "<psf><population>prf</population> cannot be used with a custom "
                    f"irf_task ({detector['irf_task']}): the pixel response is "
                    "computed analytically from diffusion_length, intra_pix_distance "
                    "and delta_pix. Remove the irf_task or use the sampled population."
                )
        if "delta_pix" not in detector:
            raise ValueError("delta_pix is required to build the PRF tables")

    @staticmethod
    def load_fine_psf(parameters, focal_plane, sampling):
        """
        Fine-grid PSFs (energy per cell, cells of size ``delta_pix / sampling``)
        for a set of wavelength nodes.

        Returns
        -------
        :class:`numpy.ndarray`
            wavelengths of the nodes, in um, ascending
        :class:`numpy.ndarray`
            ``(n_nodes, n_spatial, n_spectral)`` cube
        """
        psf_pars = parameters["psf"]
        if "psf_task" in psf_pars:
            psf_task = find_task(psf_pars["psf_task"], instrument.LoadPsf)
            if issubclass(psf_task, instrument.LoadPsfPaosTimeInterp):
                raise NotImplementedError(
                    "<psf><population>prf</population> does not support time "
                    f"dependent PSFs ({psf_pars['psf_task']}): use the sampled population"
                )
            if not issubclass(psf_task, instrument.LoadPsfPaos):
                raise NotImplementedError(
                    "<psf><population>prf</population> supports the analytic PSFs and "
                    f"LoadPsfPaos, not {psf_pars['psf_task']}: its output is on the "
                    "oversampled grid and cannot be used to tabulate the PRF"
                )
            instance = psf_task()
            nodes, cube = instance.load_fine_psf(
                psf_pars["filename"], parameters, sampling
            )
            lo, hi = PopulateFocalPlanePrf.band_edges(parameters, focal_plane)
            return extend_prf_nodes(nodes, cube, lo, hi)

        osf = int(focal_plane.metadata.get("oversampling", 1))
        n_nodes = int(psf_pars.get("prf_nodes", DEFAULT_PRF_NODES))
        lo, hi = PopulateFocalPlanePrf.band_edges(parameters, focal_plane)
        nodes = np.linspace(lo, hi, n_nodes)
        if n_nodes < 2 or lo == hi:
            nodes = np.array([hi])
        shape_rows, shape_cols = focal_plane.data[0].shape
        n_rows, n_cols = shape_rows // osf, shape_cols // osf
        if "size_x" in psf_pars or "size_y" in psf_pars:
            # explicit window, in focal plane sub-pixels
            array_size = [
                _fine_size(psf_pars["size_y"], sampling, osf),
                _fine_size(psf_pars["size_x"], sampling, osf),
            ]
        elif "nzero" in psf_pars:
            # explicit number of dark rings
            array_size = None
        else:
            # default, as for the sampled population: a square window as large
            # as the shorter side of the detector, odd and centred, so that only
            # the light that misses the detector is lost
            side = min(n_rows, n_cols) * sampling
            side = side if side % 2 == 1 else side - 1
            array_size = [side, side]
        psf = create_psf(
            nodes * u.um,
            (parameters["Fnum_x"], parameters["Fnum_y"]),
            parameters["detector"]["delta_pix"] / sampling,
            shape=psf_pars["shape"].lower(),
            nzero=psf_pars.get("nzero", 4),
            max_array_size=(n_rows * sampling, n_cols * sampling),
            array_size=array_size,
            name=parameters.get("value"),
        )
        return nodes, psf

    @staticmethod
    def band_edges(parameters, focal_plane):
        """
        Wavelength limits of the focal plane, in um, defined on the physical
        pixels, hence independent of the oversampling: the wavelengths of the
        outer edges of the physical pixels for a spectrometer
        (:func:`~exosim.utils.prf.physical_axis_edges`; the first and last
        sub-pixel centres depend on the oversampling), the range of the
        wavelength axis for a photometer (the band of its efficiency, whose
        axis has ``32 * osf`` samples). They set the wavelength nodes of the
        tables (and, through them, the size of the PSF window, which is
        proportional to the longest wavelength) and the part of the source
        spectrum that falls on the focal plane.

        Returns
        -------
        float
            lower limit
        float
            upper limit
        """
        osf = int(focal_plane.metadata.get("oversampling", 1))
        axis = np.asarray(
            (focal_plane.spectral * focal_plane.spectral_units).to(u.um).value, float
        )
        positive = axis[axis > 0.0]
        if (
            str(parameters.get("type", "")).lower() == "photometer"
            or positive.size < 2
            or positive.min() == positive.max()
        ):
            return float(positive.min()), float(positive.max())
        return physical_axis_edges(axis, osf)

    # ---------------------------------------------------------------- deposit
    def deposit_source(self, parameters, focal_plane, source, prf, pointing=None):
        """
        Adds the PRF of a source to the focal plane.

        Parameters
        ----------
        parameters: dict
            channel parameters
        focal_plane: :class:`~exosim.models.signal.Signal`
            focal plane array
        source: :class:`~exosim.models.signal.Sed`
            source, propagated and *not* rescaled to the focal plane columns
        prf: :class:`~exosim.utils.prf.PrfTables`
            PRF tables
        pointing: tuple (optional)
            telescope pointing
        """
        osf = int(focal_plane.metadata.get("oversampling", 1))
        shape = focal_plane.data.shape
        n_rows, n_cols = shape[1] // osf, shape[2] // osf
        ch_type = parameters["type"].lower()
        if ch_type not in ("spectrometer", "photometer"):
            raise ValueError(
                f"unsupported channel type '{parameters['type']}'; "
                "expected 'spectrometer' or 'photometer'"
            )

        # exact pointing offsets, in sub-pixels -> physical pixels
        compute_offset = instrument.ComputeSourcesPointingOffset()
        off_spectral, off_spatial = compute_offset(
            source=source.metadata,
            pointing=pointing,
            parameters=parameters,
            exact=True,
        )
        off_spectral, off_spatial = off_spectral / osf, off_spatial / osf

        wl = np.asarray(source.spectral, float)
        d_wl = np.abs(np.gradient(wl)) if wl.size > 1 else np.ones_like(wl)
        # only the part of every spectral bin that falls on the wavelength axis
        # of the focal plane contributes, as in the sampled population where the
        # source is rebinned onto it
        edges = self.band_edges(parameters, focal_plane)
        if edges[0] == edges[1]:  # a single wavelength covers everything
            coverage = np.ones(wl.shape)
        else:
            coverage = wavelength_coverage(
                wl, d_wl, focal_plane.spectral, osf=osf, edges=edges
            )
        flux = np.nan_to_num(
            np.asarray(source.data)[:, 0, :] * (d_wl * coverage)[None, :]
        )
        keep = np.any(flux != 0.0, axis=0) & (wl > 0.0)

        if ch_type == "spectrometer":
            pos, _ = wavelength_to_subpixel(wl, focal_plane.spectral)
            # the wavelength of the sub-pixel j is the one of its centre, as in the
            # sampled population, and the centre of the physical pixel m is the
            # sub-pixel osf * m + osf // 2, hence q = (j - osf // 2) / osf
            qx = (pos - osf // 2) / osf + off_spectral
            if np.all(np.asarray(focal_plane.spatial) == 0):
                qy = np.full(wl.shape, float(n_rows // 2)) + off_spatial
            else:
                spatial_wl_sol = interp1d(
                    focal_plane.spatial,
                    np.arange(0, shape[1]),
                    fill_value="extrapolate",
                )
                qy = (spatial_wl_sol(wl) - osf // 2) / osf + off_spatial
            node0, w0 = node_weights(wl, prf.nodes)
            sel = np.where(keep)[0]
            deposit_prf(
                focal_plane.data,
                prf,
                node0[sel],
                w0[sel],
                qy[sel],
                qx[sel],
                flux[:, sel],
                osf,
            )
        else:
            # photometer: one position, the response sums linearly over the
            # wavelengths, hence the flux is first distributed over the nodes
            sel = np.where(keep)[0]
            node0, w0 = node_weights(wl[sel], prf.nodes)
            n_nodes = prf.nodes.size
            node_flux = np.zeros((flux.shape[0], n_nodes))
            for t in range(flux.shape[0]):
                node_flux[t] = (
                    np.bincount(node0, weights=flux[t, sel] * w0, minlength=n_nodes)[
                        :n_nodes
                    ]
                    + np.bincount(
                        node0 + 1,
                        weights=flux[t, sel] * (1.0 - w0),
                        minlength=n_nodes + 1,
                    )[:n_nodes]
                )
            used = np.where(np.any(node_flux != 0.0, axis=0))[0]
            ones = np.ones(used.size)
            deposit_prf(
                focal_plane.data,
                prf,
                used,
                ones,
                # pixel N // 2, where the sampled population places it and the
                # apertures are centred
                np.full(used.size, float(n_rows // 2) + off_spatial),
                np.full(used.size, float(n_cols // 2) + off_spectral),
                node_flux[:, used],
                osf,
            )


def extend_prf_nodes(nodes, cube, lo, hi, ratio=1.05):
    """
    Extends the wavelength nodes of fine PSFs to cover ``[lo, hi]``.

    Beyond the first and last node, extra nodes are added on a geometric grid
    (ratio between consecutive nodes not larger than ``ratio``), the last one
    at ``hi`` (the first one at ``lo``). Each is the fine PSF of the edge node
    scaled in size proportionally to the wavelength
    (:func:`~exosim.utils.psf.scale_psf_image`), as done by
    :meth:`~exosim.tasks.instrument.load_psf_paos.LoadPsfPaos.wl_interpolate`
    for the sampled population.

    Returns
    -------
    :class:`numpy.ndarray`
        wavelengths, ascending
    :class:`numpy.ndarray`
        cube of fine PSFs
    """
    nodes = np.asarray(nodes, float)
    if nodes.size < 1:
        return nodes, cube
    below, above = [], []
    if hi > nodes[-1]:
        n = int(np.ceil(np.log(hi / nodes[-1]) / np.log(ratio)))
        above = nodes[-1] * (hi / nodes[-1]) ** (np.arange(1, n + 1) / n)
    if lo < nodes[0]:
        n = int(np.ceil(np.log(nodes[0] / lo) / np.log(ratio)))
        below = nodes[0] * (lo / nodes[0]) ** (np.arange(n, 0, -1) / n)
    if len(below) == 0 and len(above) == 0:
        return nodes, cube
    low_imgs = [scale_psf_image(cube[0], w / nodes[0]) for w in below]
    high_imgs = [scale_psf_image(cube[-1], w / nodes[-1]) for w in above]
    new_nodes = np.concatenate([below, nodes, above]).astype(float)
    new_cube = np.concatenate(
        [
            np.reshape(low_imgs, (-1, *cube.shape[1:])),
            cube,
            np.reshape(high_imgs, (-1, *cube.shape[1:])),
        ]
    )
    return new_nodes, new_cube


def _fine_size(size, sampling, osf):
    """Converts a PSF size in focal plane sub-pixels into fine samples."""
    if isinstance(size, str):
        return size
    return round(size * sampling / osf)
