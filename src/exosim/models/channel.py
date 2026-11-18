from collections import OrderedDict

import astropy.units as u
import numpy as np

import exosim.log as log
import exosim.models.signal as signal
import exosim.tasks.instrument as instrument
import exosim.tasks.parse as parse
from exosim.utils.klass_factory import find_and_run_task
from exosim.utils.types import ArrayType, OutputType, ValueType


class Channel(log.Logger):
    """
    It handles the channel gven the description

    Attributes
    ----------
    ch_name: str
        channel name
    path: dict
        dictionary of :class:`~exosim.models.signal.Radiance` and :class:`~exosim.models.signal.Dimensionless`,
        representing the radiance and efficiency of the path.
    responsivity: :class:`~exosim.models.signal.Signal`
        channel responsivity
    sources: dict
        dictionary containing :class:`~exosim.models.signal.Signal`
    time: :class:`~astropy.units.Quantity`
        time grid.
    parameters: dict
        dictionary contained the optical element parameters. This is usually parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`
    wavelength: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity`
        wavelength grid. If no units are attached is considered as expressed in `um`.
    focal_plane: :class:`~exosim.models.signal.Signal`
        source focal plane
    frg_focal_plane: :class:`~exosim.models.signal.Signal`
        foreground focal plane
    frg_sub_focal_planes: dict
        dictionary of :class:`~exosim.models.signal.Signal`. It contains the sub focal planes produced by the radiances.
        This dictionary is produced only if at least one optical surface has ``isolate=True``.
        The sum of the sub focal planes returns the frg_focal_plane. If not surface has ``isolate=True``, the dictionary is empty.
    output: :class:`~exosim.output.output.Output`
        output file
    target_source: str
        name of the target source
    """

    def __init__(
        self,
        parameters: dict,
        wavelength: ArrayType,
        time: ArrayType,
        output: OutputType = None,
    ) -> None:
        """
        Parameters
        __________
        parameters: dict
            dictionary contained the optical element parameters. This is usually parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`
        wavelength: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity`
            wavelength grid. If no units are attached is considered as expressed in `um`.
        time: :class:`~astropy.units.Quantity`
            time grid.
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        """
        self.set_log_name()

        self.parameters = parameters
        self.wavelength = wavelength
        self.time = time
        self.ch_name = parameters["value"]

        self.output = output.create_group(self.ch_name) if output else None

        # init to None
        self.path, self.responsivity, self.sources, self.psf = (
            None,
            None,
            None,
            None,
        )
        (
            self.focal_plane,
            self.bkg_focal_plane,
            self.frg_focal_plane,
            self.frg_sub_focal_planes,
        ) = (
            None,
            None,
            None,
            None,
        )

    def parse_path(self, light_path: OrderedDict) -> dict:
        """
        It applies :class:`~exosim.tasks.parse.parsePath.ParsePath`

        Parameters
        ----------
        light_path: `~collections.OrderedDict` (optional)
            dictionary of contributes

        Returns
        -------
        dict
            dictionary of :class:`~exosim.models.signal.Radiance` and :class:`~exosim.models.signal.Dimensionless`,
            representing the radiance and efficiency of the path.

        Note
        ----
        The resulting information is also stored in the class under `path` attribute.
        """

        parsePath = parse.ParsePath()
        self.path = parsePath(
            parameters=self.parameters["optical_path"],
            wavelength=self.wavelength,
            time=self.time,
            output=self.output,
            light_path=light_path,
            group_name="path",
        )
        self.path["efficiency"].write(output=self.output, name="efficiency")
        return self.path

    def estimate_responsivity(self) -> signal.Signal:
        """
        It estimates the responsivity using the indicated :class:`~exosim.tasks.instrument.load_responsivity.LoadResponsivity`

        Returns
        -------
        :class:`~exosim.models.signal.Signal`
            channel responsivity

        Note
        ----
        The resulting information is also stored in the class under `responsivity` attribute.
        """
        responsivity_instance = find_and_run_task(
            self.parameters["qe"],
            "responsivity_task",
            instrument.LoadResponsivity,
        )

        self.responsivity = responsivity_instance(
            parameters=self.parameters,
            wavelength=self.wavelength,
            time=self.time,
        )
        self.responsivity.write(output=self.output, name="responsivity")
        return self.responsivity

    def propagate_foreground(self) -> dict:
        """
        It multiplies each radiance in the path by the solid angle.

        Returns
        -------
        dict
            dictionary of :class:`~exosim.models.signal.Radiance` and :class:`~exosim.models.signal.Dimensionless`,
            representing the radiance and efficiency of the path.

        Note
        ----
        it updates the `path` attribute of this class
        """

        propagateForegrounds = instrument.PropagateForegrounds()
        self.path = propagateForegrounds(
            light_path=self.path,
            parameters=self.parameters,
            responsivity=self.responsivity,
        )
        return self.path

    def define_sources(self, sources: dict) -> None:
        """
        It defines the sources for the channel

        Parameters
        __________
        sources: dict
            dictionary containing :class:`~exosim.models.signal.Sed`

        Returns
        -------
        dict
            dictionary containing :class:`~exosim.models.signal.Signal`
        """

        out_sources = {}

        for source in sources:
            out_sources[source] = signal.Sed(
                data=sources[source].data,
                spectral=sources[source].spectral,
                time=sources[source].time,
                metadata=sources[source].metadata,
            )
        self.sources = out_sources
        return out_sources

    def propagate_sources(self, Atel: ValueType) -> dict:
        """
        It propagates the sources though the channel,
        by applying :class:`~exosim.tasks.instrument.propagate_sources.PropagateSources`

        Parameters
        __________
        Atel:  :class:`~astropy.units.Quantity`
            effective telescope Area

        Returns
        -------
        dict
            dictionary containing :class:`~exosim.models.signal.Signal`
        """

        propagateSources = instrument.PropagateSources()
        self.sources = propagateSources(
            sources=self.sources,
            Atel=Atel,
            efficiency=self.path["efficiency"],
            responsivity=self.responsivity,
        )

        return self.sources

    def create_focal_planes(self) -> signal.Signal:
        """
        It produces the empty focal planes

        Returns
        -------
         :class:`~exosim.models.signal.Signal`
            focal plane array (with time evolution)
        """

        createFocalPlane = instrument.CreateFocalPlane()
        focal_plane = createFocalPlane(
            parameters=self.parameters,
            efficiency=self.path["efficiency"],
            time=self.time,
            output=self.output,
            group_name="focal_plane",
        )
        self.focal_plane = focal_plane
        self.frg_focal_plane = focal_plane.copy(dataset_name="frg_focal_plane")
        if len(self.sources) > 1:
            self.bkg_focal_plane = focal_plane.copy(dataset_name="bkg_focal_plane")
        return focal_plane

    def _focal_plane_wavelength_step(self) -> u.Quantity:
        """Wavelength width of each focal-plane spectral column."""
        x_wav_osr = self.focal_plane.spectral
        d_x_wav_osr = np.zeros_like(x_wav_osr)
        idx = np.where(x_wav_osr > 0.0)
        d_x_wav_osr[idx] = np.gradient(x_wav_osr[idx])
        if np.any(d_x_wav_osr < 0):
            d_x_wav_osr *= -1.0
        if (
            str(self.parameters.get("type", "")).lower() == "photometer"
            and idx[0].size > 1
        ):
            # The axis of a photometer runs from the first to the last
            # wavelength of its band, the two ends included: the samples are
            # the nodes of a trapezoid rule, the end ones weigh half a step.
            # With a full step each, the band would be widened by one step
            # (3% for 32 samples, ``1 / osf`` of it with oversampling).
            d_x_wav_osr[idx[0][0]] *= 0.5
            d_x_wav_osr[idx[0][-1]] *= 0.5
        return d_x_wav_osr * self.focal_plane.spectral_units

    def rescale_sources(self) -> None:
        """
        It rebins the sources onto the focal-plane wavelength grid and
        multiplies them by the wavelength width of each column, turning
        spectral densities into signals per column. Each source must be
        rescaled once, after it has been propagated.
        """
        if self.sources:
            d_spectral_wl = self._focal_plane_wavelength_step()
            for source in self.sources:
                self.sources[source].spectral_rebin(self.focal_plane.spectral)
                self.sources[source] *= d_spectral_wl

    def rescale_radiances(self) -> None:
        """
        It rebins the path radiances (foregrounds and optical-element
        emission) onto the focal-plane wavelength grid and multiplies them
        by the wavelength width of each column. The radiances must be
        rescaled once: when the sources are propagated again (for example
        after :meth:`define_sources`), use :meth:`rescale_sources` only.
        """
        if self.path:
            d_spectral_wl = self._focal_plane_wavelength_step()
            for rad in [k for k in self.path if "radiance" in k]:
                self.path[rad].spectral_rebin(self.focal_plane.spectral)
                self.path[rad] *= d_spectral_wl

    def rescale_contributions(self) -> None:
        """
        It rescales the sources and the path radiances onto the focal-plane
        wavelength grid: :meth:`rescale_sources` followed by
        :meth:`rescale_radiances`.
        """
        self.rescale_sources()
        self.rescale_radiances()

    @property
    def target_source(self):
        # TODO test this
        if not self.sources:
            return None
        if len(self.sources) == 1:
            return next(iter(self.sources.keys()))

        target = [
            source
            for source, param in self.sources.items()
            if param.metadata.get("parsed_parameters", {}).get("source_target")
        ]
        if not target:
            self.error(
                "Multiple sources are defined but none is flagged as the science "
                "target. Set 'source_target' on one source in the input file."
            )
            raise KeyError("no source is flagged as 'source_target'")
        if len(target) > 1:
            self.error(
                "More than one target source found. Please check your input file."
            )
            raise KeyError("more than one source is flagged as 'source_target'")
        self.debug(f"Target source is {target[0]}")
        return target[0]

    def populate_focal_plane(
        self, pointing: tuple[u.Quantity, u.Quantity] | None = None
    ) -> signal.Signal:
        """
        It populates the empty focal plane with monocromatic PSFs.

        Parameters
        -----------
        pointing: (:class:`astropy.units.Quantity`, :class:`astropy.units.Quantity`) (optional)
            telescope pointing direction, expressed ad a tuple of RA and DEC in degrees. Default is ``None``

        Returns
        -------
        :class:`~exosim.models.signal.Signal`
            focal plane array populated
        """

        # selecting the target source
        target = self.target_source

        if target is None:
            self.info("No sources defined, skipping focal plane population.")
            return self.focal_plane

        # storing binned target source
        sources_out = self.output.create_group("sources")
        self.sources[target].write(sources_out, name=target)

        # populates the focal plane
        populateFocalPlane = instrument.PopulateFocalPlane()
        focal_plane, psf = populateFocalPlane(
            parameters=self.parameters,
            focal_plane=self.focal_plane,
            sources={target: self.sources[target]},
            pointing=pointing,
            output=self.output,
        )
        self.psf = psf
        self.focal_plane = focal_plane
        return focal_plane

    def populate_bkg_focal_plane(
        self, pointing: tuple[u.Quantity, u.Quantity] | None = None
    ) -> signal.Signal:
        """
        It populates the empty background focal plane with monocromatic PSFs for each of the background sources.

        Parameters
        -----------
        pointing: (:class:`astropy.units.Quantity`, :class:`astropy.units.Quantity`) (optional)
            telescope pointing direction, expressed ad a tuple of RA and DEC in degrees. Default is ``None``

        Returns
        -------
        :class:`~exosim.models.signal.Signal`
            background focal plane array populated
        """
        if len(self.sources) > 1:
            # TODO test this
            # removing target source from source dictionary
            target = self.target_source
            self.sources.pop(target)

            # storing binned sources
            sources_out = self.output.create_group("sources")
            for source in self.sources:
                self.sources[source].write(sources_out, name=source)

            # populates the focal plane
            populateFocalPlane = instrument.PopulateFocalPlane()
            bkg_focal_plane, self.psf = populateFocalPlane(
                parameters=self.parameters,
                focal_plane=self.bkg_focal_plane,
                sources=self.sources,
                pointing=pointing,
                output=self.output,
                psf=self.psf,
            )
            self.bkg_focal_plane = bkg_focal_plane

        return self.bkg_focal_plane

    def apply_irf_to(
        self, focal_plane: signal.Signal, _kernel: tuple | None = None
    ) -> signal.Signal:
        """
        It applies the intra pixel response function (IRF) to an arbitrary
        focal plane.

        The IRF is what integrates the PSF over the area of a detector pixel.
        The kernel is built with the configured ``irf_task`` and convolved with
        the oversampled focal plane through the configured ``apply_irf_task``
        and ``convolution_method``. After the convolution every sub-pixel holds
        the flux integrated over a pixel-sized box, so readers must decimate
        the plane afterwards (``[osf // 2 :: osf]``) to obtain the physical
        pixels. Without it, an oversampled plane only samples the PSF at
        points and the brightest pixel of undersampled PSFs is overestimated.

        Parameters
        ----------
        focal_plane: :class:`~exosim.models.signal.Signal`
            oversampled focal plane to convolve (modified in place and returned)
        _kernel: tuple (optional)
            ``(kernel, delta_kernel)`` already built, to avoid recreating (and
            re-writing to the output) the kernel for several planes.

        Returns
        -------
        :class:`~exosim.models.signal.Signal`
            focal plane array with the IRF applied
        """
        kernel, delta_kernel = _kernel or self._create_irf_kernel()

        if "convolution_method" in self.parameters["detector"]:
            convolution_method = self.parameters["detector"]["convolution_method"]
        else:
            convolution_method = "fftconvolve"

        applyIntraPixelResponseFunction = find_and_run_task(
            self.parameters["detector"],
            "apply_irf_task",
            instrument.ApplyIntraPixelResponseFunction,
        )

        return applyIntraPixelResponseFunction(
            focal_plane=focal_plane,
            irf_kernel=kernel,
            irf_kernel_delta=delta_kernel,
            convolution_method=convolution_method,
        )

    def _create_irf_kernel(self) -> tuple:
        """
        Builds the IRF kernel with the configured ``irf_task``.

        The kernel depends only on the channel parameters, so it is built (and
        written to the output) once per channel and reused afterwards: writing
        it a second time would collide on the existing output dataset.
        """
        cached = getattr(self, "_irf_kernel", None)
        if cached is not None:
            return cached
        irf_instance = find_and_run_task(
            self.parameters["detector"],
            "irf_task",
            instrument.CreateIntrapixelResponseFunction,
        )

        self.parameters["psf_shape"] = self.psf.shape[1:]
        self._irf_kernel = irf_instance(parameters=self.parameters, output=self.output)
        return self._irf_kernel

    def apply_irf(self) -> signal.Signal:
        """
        It applies the intra pixel response function (IRF) to the focal plane
        and to the background focal plane (if any).
        See :func:`~exosim.models.channel.Channel.apply_irf_to`.

        Returns
        -------
        :class:`~exosim.models.signal.Signal`
            focal plane array
        """
        kernel = self._create_irf_kernel()
        focal_plane = self.apply_irf_to(self.focal_plane, _kernel=kernel)

        if self.bkg_focal_plane is not None:
            bkg_focal_plane = self.apply_irf_to(self.bkg_focal_plane, _kernel=kernel)
        else:
            bkg_focal_plane = None

        return focal_plane, bkg_focal_plane

    def populate_foreground_focal_plane(
        self,
    ) -> tuple[signal.Signal, signal.Signal]:
        """
        It adds the foreground contribution to the foreground focal plane

        Returns
        -------
        :class:`~exosim.models.signal.Signal`
            focal plane array
        """

        # populates the focal plane
        foregroundToFocalPlane = instrument.ForegroundsToFocalPlane()
        frg_focal_plane, sub_focal_planes = foregroundToFocalPlane(
            parameters=self.parameters,
            focal_plane=self.frg_focal_plane,
            path=self.path,
        )

        self.frg_focal_plane = frg_focal_plane
        self.frg_sub_focal_planes = sub_focal_planes
        return frg_focal_plane, sub_focal_planes
