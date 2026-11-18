"""
Tests for the pixel integration of the PSF in the radiometric modes.

A detector pixel collects the PSF integrated over its area. ExoSim does this by
oversampling the focal plane and convolving it with the intra pixel response
function (IRF); the radiometric readers then decimate with ``[osf // 2 :: osf]``.
These tests check that the source position does not depend on the oversampling
and that the IRF + decimation reproduces the pixel-integrated PSF.
"""

import astropy.units as u
import numpy as np
import pytest
from scipy.signal import fftconvolve
from scipy.special import j1

from exosim.models.channel import Channel
from exosim.models.signal import Signal
from exosim.recipes.radiometric_model import RadiometricModel
from exosim.tasks.instrument import CreateIntrapixelResponseFunction
from exosim.utils.focal_plane_locations import locate_wavelength_windows
from exosim.utils.psf import create_psf

PIX = 18.0  # um
WL = [0.6] * u.um
FNUM = 30.0  # lambda*F / pixel = 1.0 -> undersampled (FWHM ~ 1 pixel)
N_PIX = 41
NZERO = 12  # PSF array half-width ~ 12 pixels, inside the 41-pixel plane


class _Spectral:
    def __init__(self, size):
        self.size = size


class _FocalPlane:
    """Minimal focal plane exposing what locate_wavelength_windows reads."""

    def __init__(self, shape, osf=None, spatial=None):
        self.shape = shape
        self.data = None
        self.cached = True
        self.spectral = _Spectral(1)
        self.spatial = np.zeros(1) if spatial is None else spatial
        self.metadata = {} if osf is None else {"oversampling": osf}


def _airy_psf(osf):
    """4D Airy PSF (time, wl, y, x) sampled at the oversampled grid."""
    psf = create_psf(WL, (FNUM, FNUM), PIX / osf * u.um, nzero=NZERO, shape="airy")
    return psf[np.newaxis]


def _place(osf):
    """Oversampled plane with a centred Airy PSF placed by the library."""
    psf = _airy_psf(osf)
    plane = np.zeros((1, N_PIX * osf, N_PIX * osf))
    fp = _FocalPlane(plane.shape, osf)
    i0, j0 = locate_wavelength_windows(psf, fp, {"type": "photometer"})
    i0, j0 = int(i0[0]), int(j0[0])
    ny, nx = psf.shape[2:]
    plane[0, i0 : i0 + ny, j0 : j0 + nx] += psf[0, 0]
    return plane, psf


def _irf_kernel(osf):
    params = {
        "detector": {
            "oversampling": osf,
            "delta_pix": PIX * u.um,
            "diffusion_length": 0.0 * u.um,
            "intra_pix_distance": 0.0 * u.um,
        }
    }
    kernel, _ = CreateIntrapixelResponseFunction()(parameters=params)
    return kernel, params


def _physical_pixel(plane, osf):
    """Physical pixel (row, col) of the brightest decimated pixel."""
    dec = plane[0, osf // 2 :: osf, osf // 2 :: osf]
    return np.unravel_index(np.argmax(dec), dec.shape)


def _airy_intensity(x, y):
    r = np.hypot(x, y) / (WL[0].value * FNUM) * np.pi
    r = np.where(r == 0, 1e-12, r)
    return (2 * j1(r) / r) ** 2


def _reference_central_fraction(n_sub=61):
    """Brute-force pixel-integrated flux fraction in the central pixel."""
    edges = (np.arange(N_PIX + 1) - N_PIX / 2) * PIX  # pixel edges, centred
    # midpoint rule with n_sub x n_sub points per pixel
    sub = (np.arange(n_sub) + 0.5) / n_sub * PIX
    xs = (edges[:-1, None] + sub[None, :]).ravel()
    xx, yy = np.meshgrid(xs, xs)
    img = _airy_intensity(xx, yy).reshape(N_PIX, n_sub, N_PIX, n_sub)
    # same footprint as the (truncated) library PSF, so that both fractions are
    # normalised to the same flux
    half = (_airy_psf(1).shape[-1] // 2 + 0.5) * PIX
    inside = (np.abs(xx) <= half) & (np.abs(yy) <= half)
    img = (img.reshape(N_PIX * n_sub, N_PIX * n_sub) * inside).reshape(
        N_PIX, n_sub, N_PIX, n_sub
    )
    per_pixel = img.sum(axis=(1, 3))
    return per_pixel[N_PIX // 2, N_PIX // 2] / per_pixel.sum()


class TestPlacement:
    @pytest.mark.parametrize("ndim", [3, 4])
    def test_unit_oversampling_centres_on_pixel_n_over_2(self, ndim):
        """osf=1 (or missing metadata): the PSF centre is on pixel N // 2, where the
        apertures are centred (``shape / 2`` in the photutils convention)."""
        shape = (1, 64, 80)
        if ndim == 4:
            psf = np.zeros((1, 1, 7, 9))
            psf_x, psf_y = 9, 7
        else:
            psf = np.zeros((1, 7, 9))
            psf_x = 9
        for osf in (None, 1):
            fp = _FocalPlane(shape, osf)
            i0, j0 = locate_wavelength_windows(psf, fp, {"type": "photometer"})
            assert np.array_equal(j0, [shape[2] // 2 - psf_x // 2])
            if ndim == 4:
                assert np.array_equal(i0, [shape[1] // 2 - psf_y // 2])

        if ndim == 4:
            # spectrometer, no spatial wavelength solution
            fp = _FocalPlane(shape, 1)
            i0, _ = locate_wavelength_windows(psf, fp, {"type": "spectrometer"})
            assert np.array_equal(i0, [shape[1] // 2 - psf_y // 2])

    def test_spectrometer_spatial_axis_centred_on_same_pixel(self):
        psf = np.zeros((1, 1, 7, 9))
        centres = []
        for osf in (1, 3, 5):
            fp = _FocalPlane((1, 40 * osf, 60 * osf), osf)
            i0, _ = locate_wavelength_windows(psf, fp, {"type": "spectrometer"})
            centre_sub = i0[0] + psf.shape[2] // 2
            assert centre_sub % osf == osf // 2
            centres.append(centre_sub // osf)
        assert centres == [20, 20, 20]

    def test_same_physical_pixel_for_all_osf(self):
        pixels = []
        for osf in (1, 3, 5):
            plane, _ = _place(osf)
            pixels.append(_physical_pixel(plane, osf))
        assert pixels[0] == pixels[1] == pixels[2] == (N_PIX // 2, N_PIX // 2)


class TestIrfIntegration:
    def test_decimated_fraction_matches_pixel_integration(self):
        ref = _reference_central_fraction()
        plane1, _ = _place(1)
        point_sampled = plane1[0].max() / plane1[0].sum()
        # the point-sampled value is clearly biased high for this undersampled PSF
        assert point_sampled > ref + 0.05
        tol = {3: 0.03, 5: 0.02}
        for osf in (3, 5):
            plane, _ = _place(osf)
            kernel, _ = _irf_kernel(osf)
            assert kernel.sum() == pytest.approx(osf**2)
            conv = fftconvolve(plane[0], kernel, mode="same")
            dec = conv[osf // 2 :: osf, osf // 2 :: osf]
            frac = dec[N_PIX // 2, N_PIX // 2] / dec.sum()
            assert frac == pytest.approx(ref, abs=tol[osf]), (
                osf,
                frac,
                ref,
                point_sampled,
            )
            assert abs(frac - ref) < abs(point_sampled - ref)


def _channel(osf):
    params = {
        "detector": {
            "oversampling": osf,
            "delta_pix": PIX * u.um,
            "diffusion_length": 0.0 * u.um,
            "intra_pix_distance": 0.0 * u.um,
        }
    }
    ch = object.__new__(Channel)
    Channel.__bases__[0].__init__(ch)
    ch.parameters = params
    ch.psf = np.zeros((1, 3, 3))
    ch.output = None
    ch.bkg_focal_plane = None
    return ch


def _signal(plane):
    return Signal(
        spectral=[0.6] * u.um,
        time=[0] * u.hr,
        data=plane.copy(),
        data_units="ct/s",
        metadata={"focal_plane_delta": PIX / 3 * u.um, "oversampling": 3},
    )


class TestChannelApplyIrf:
    def test_apply_irf_to_matches_apply_irf(self):
        plane, _ = _place(3)
        a = _channel(3)
        a.focal_plane = _signal(plane)
        b = _channel(3)
        b.focal_plane = _signal(plane)

        via_apply, bkg = a.apply_irf()
        via_to = b.apply_irf_to(b.focal_plane)
        assert bkg is None
        assert np.allclose(via_apply.data, via_to.data)
        assert not np.allclose(via_to.data, plane)  # something was convolved
        assert via_to.data.sum() == pytest.approx(plane.sum() * 9, rel=1e-6)

    def test_apply_irf_also_convolves_background(self):
        plane, _ = _place(3)
        ch = _channel(3)
        ch.focal_plane = _signal(plane)
        ch.bkg_focal_plane = _signal(plane)
        fp, bkg = ch.apply_irf()
        assert np.allclose(fp.data, bkg.data)

    def test_kernel_built_and_written_once_per_channel(self, tmp_path):
        """A second apply_irf_to (e.g. on another plane) reuses the kernel and
        does not write it again to the output, which would collide."""
        from exosim.output import SetOutput

        plane, _ = _place(3)
        ch = _channel(3)
        ch.focal_plane = _signal(plane)
        other = _signal(plane)
        with SetOutput(str(tmp_path / "irf.h5")).use(append=True) as out:
            ch.output = out.create_group("CH")
            first = ch.apply_irf_to(ch.focal_plane)
            second = ch.apply_irf_to(other)
        assert np.allclose(first.data, second.data)


class TestTargetListOversampling:
    def _run(self):
        from unittest.mock import MagicMock, patch

        import exosim.log as log

        rm = object.__new__(RadiometricModel)
        log.Logger.__init__(rm)
        rm.output_file = "unused.h5"
        rm.mainConfig = {
            "sky": {"source": {"targetlist_filepath": "x", "source_type": None}},
            "detector": {"oversampling": 3},
        }
        rm.payloadConfig = {"channel": {"ch": {"detector": {"oversampling": 5}}}}
        output = MagicMock()
        output.getsize.return_value = 1.0
        output.use.return_value.__enter__.return_value = output
        with (
            patch("exosim.recipes.radiometric_model.LoadSourceList") as loader,
            patch("exosim.recipes.radiometric_model.SetOutput", return_value=output),
        ):
            loader.return_value.return_value = {}
            rm.target_list_pipeline(store_config=False)
        return rm

    def test_keeps_oversampling(self):
        rm = self._run()
        assert rm.mainConfig["detector"]["oversampling"] == 3
        assert rm.payloadConfig["channel"]["ch"]["detector"]["oversampling"] == 5
