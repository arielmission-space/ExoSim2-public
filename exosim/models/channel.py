from collections import OrderedDict
from typing import Tuple

import astropy.units as u
import numpy as np

import exosim.log as log
import exosim.models.signal as signal
import exosim.tasks.instrument as instrument
import exosim.tasks.parse as parse
from exosim.utils.klass_factory import find_and_run_task
from exosim.utils.types import ArrayType
from exosim.utils.types import OutputType
from exosim.utils.types import ValueType


class Channel(log.Logger):
    """
    It handles the channel gven the description

    Attributes
    ----------
    ch_name: str
        channel name
    path: dict
        dictionary of :class:`~exosim.models.signal.Radiance` and :class:`~exosim.models.signal.Dimensionless`,
        represeting the radiance and efficiency of the path.
    responsivity: :class:`~exosim.models.signal.Signal`
        channel responsivity
    sources: dict
        dictionary containing :class:`~exosim.models.signal.Signal`
    time: :class:`~astropy.units.Quantity`
        time grid.
    parameters: dict
        dictionary contained the optical element parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
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
            dictionary contained the optical element parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
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
            represeting the radiance and efficiency of the path.

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
        It estimates the responsivity using the indicated :class:`~exosim.tasks.instrument.loadResponsivity.LoadResponsivity`

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
            represeting the radiance and efficiency of the path.

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

    def propagate_sources(self, sources: OrderedDict, Atel: ValueType) -> dict:
        """
        It propagates the sources though the channel,
        by applying :class:`~exosim.tasks.instrument.propagateSources.PropagateSources`

        Parameters
        __________
        sources:  dict
            dictionary containing :class:`~exosim.models.signal.Sed`
        Atel:  :class:`~astropy.units.Quantity`
            effective telescope Area

        Returns
        -------
        dict
            dictionary containing :class:`~exosim.models.signal.Signal`
        """

        out_sources = {}

        for source in sources.keys():
            out_sources[source] = signal.Sed(
                data=sources[source].data,
                spectral=sources[source].spectral,
                time=sources[source].time,
                metadata=sources[source].metadata,
            )

        propagateSources = instrument.PropagateSources()
        self.sources = propagateSources(
            sources=out_sources,
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
            self.bkg_focal_plane = focal_plane.copy(
                dataset_name="bkg_focal_plane"
            )
        return focal_plane

    def rescale_contributions(self) -> None:
        """
        It updated the contributions (sources and path)
        by rebinning them to the wavelength solution grid
        and multipling them by the wl solution gradient
        """

        def wl_gradient(x_wav_osr):
            d_x_wav_osr = np.zeros_like(x_wav_osr)
            idx = np.where(x_wav_osr > 0.0)
            d_x_wav_osr[idx] = np.gradient(x_wav_osr[idx])
            if np.any(d_x_wav_osr < 0):
                d_x_wav_osr *= -1.0
            return d_x_wav_osr

        d_spectral_wl = (
            wl_gradient(self.focal_plane.spectral)
            * self.focal_plane.spectral_units
        )

        # multiply sources by gradient
        if self.sources:
            for source in self.sources.keys():
                self.sources[source].spectral_rebin(self.focal_plane.spectral)
                self.sources[source] *= d_spectral_wl

        # multiply radiances by gradient
        if self.path:
            for rad in [k for k in self.path.keys() if "radiance" in k]:
                self.path[rad].spectral_rebin(self.focal_plane.spectral)
                self.path[rad] *= d_spectral_wl

    @property
    def target_source(self):
        # TODO test this
        if len(self.sources) == 1:
            return list(self.sources.keys())[0]

        target = [
            source
            for source, param in self.sources.items()
            if "source_target" in param.metadata["parsed_parameters"].keys()
            and param.metadata["parsed_parameters"]["source_target"] == True
        ]
        if len(target) > 1:
            self.error(
                "More than one target source found. Please check your input file."
            )
        self.debug(f"Target source is {target[0]}")
        return target[0]

    def populate_focal_plane(
        self, pointing: Tuple[u.Quantity, u.Quantity] = None
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
        self, pointing: Tuple[u.Quantity, u.Quantity] = None
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
            for source in self.sources.keys():
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

    def apply_irf(self) -> signal.Signal:
        """
        It applies the intra pixel response function (IRF) to the focal plane


        Returns
        -------
        :class:`~exosim.models.signal.Signal`
            focal plane array
        """

        irf_instance = find_and_run_task(
            self.parameters["detector"],
            "irf_task",
            instrument.CreateIntrapixelResponseFunction,
        )

        self.parameters["psf_shape"] = self.psf.shape[1:]
        kernel, delta_kernel = irf_instance(
            parameters=self.parameters, output=self.output
        )

        if "convolution_method" in self.parameters["detector"]:
            convolution_method = self.parameters["detector"][
                "convolution_method"
            ]
        else:
            convolution_method = "fftconvolve"

        applyIntraPixelResponseFunction = find_and_run_task(
            self.parameters["detector"],
            "apply_irf_task",
            instrument.ApplyIntraPixelResponseFunction,
        )
        # apply IRF to target focal plane
        focal_plane = applyIntraPixelResponseFunction(
            focal_plane=self.focal_plane,
            irf_kernel=kernel,
            irf_kernel_delta=delta_kernel,
            convolution_method=convolution_method,
        )

        if self.bkg_focal_plane is not None:
            # apply IRF to background focal plane
            bkg_focal_plane = applyIntraPixelResponseFunction(
                focal_plane=self.bkg_focal_plane,
                irf_kernel=kernel,
                irf_kernel_delta=delta_kernel,
                convolution_method=convolution_method,
            )
        else:
            bkg_focal_plane = None

        return focal_plane, bkg_focal_plane

    def populate_foreground_focal_plane(
        self,
    ) -> Tuple[signal.Signal, signal.Signal]:
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
