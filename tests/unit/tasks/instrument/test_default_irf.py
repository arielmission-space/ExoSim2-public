"""
A pixel always integrates the light over its area, so a channel with an
``oversampling`` but without ``irf_task`` must keep the oversampling and use the
default intra pixel response function (a top-hat over the pixel when there is no
``diffusion_length``). It used to run silently at an oversampling of 1.
"""

import logging
from unittest.mock import MagicMock

import astropy.units as u
import numpy as np
import pytest
from scipy.special import j1

import exosim.log as log
from exosim.models.channel import Channel
from exosim.models.signal import Signal
from exosim.recipes.radiometric_model import RadiometricModel
from exosim.tasks.instrument import CreateIntrapixelResponseFunction
from exosim.tasks.instrument.create_focal_plane_array import CreateFocalPlaneArray
from exosim.tasks.instrument.create_intrapixel_response_function import (
    irf_extent_pixels,
)
from exosim.utils.focal_plane_locations import locate_wavelength_windows
from exosim.utils.psf import create_psf

PIX = 18.0  # um
WL = [0.6] * u.um
FNUM = 30.0  # F * lambda / pixel = 1: undersampled Airy
N_PIX = 41
OSF = 3
OSF_PRECISE = 9  # sub-pixel point sampling error of the Airy is 0.4% here


def _detector(osf, **extra):
    """Detector section with an oversampling and no ``irf_task``."""
    det = {
        "spatial_pix": 8,
        "spectral_pix": 12,
        "delta_pix": PIX * u.um,
        "oversampling": osf,
    }
    det.update(extra)
    return det


def test_focal_plane_array_keeps_oversampling_without_irf_task(caplog):
    x = np.linspace(-100, 800, 20) * u.um
    params = {
        "type": "spectrometer",
        "detector": _detector(OSF),
        "wl_solution": {
            "wl_solution_task": "LoadWavelengthSolution",
            "data": {
                "Wavelength": (2.0 + 0.004 * x.value) * u.um,
                "x": x,
                "y": np.zeros(x.size) * u.um,
            },
        },
        "wl_min": 2.0 * u.um,
        "wl_max": 5.0 * u.um,
    }
    assert "irf_task" not in params["detector"]
    task = CreateFocalPlaneArray()
    task.set_log_name()
    with caplog.at_level(logging.WARNING):
        fp = task(parameters=params, efficiency=None)
    assert fp.data.shape[-2:] == (8 * OSF, 12 * OSF)
    assert fp.metadata["oversampling"] == OSF
    assert u.isclose(fp.metadata["focal_plane_delta"], PIX / OSF * u.um)
    assert not [r for r in caplog.records if "irf_task" in r.getMessage()]


@pytest.mark.parametrize("osf", [1, 2, 3, 5, 8])
def test_default_kernel_without_diffusion_is_a_top_hat(osf):
    """No ``irf_task``, no ``diffusion_length``: K = 0 and a flat osf x osf box."""
    params = {"detector": _detector(osf)}
    assert "diffusion_length" not in params["detector"]
    kernel, delta = CreateIntrapixelResponseFunction()(parameters=params)
    kernel = np.asarray(kernel)
    assert irf_extent_pixels(PIX * u.um, 1e-20 * u.um) == 0
    assert kernel.shape == (osf, osf)
    assert u.isclose(delta, PIX / osf * u.um)
    np.testing.assert_allclose(kernel, 1.0, rtol=1e-12)
    assert kernel.sum() == pytest.approx(osf**2)


def _channel(osf):
    ch = object.__new__(Channel)
    Channel.__bases__[0].__init__(ch)
    ch.parameters = {"detector": _detector(osf)}  # no irf_task
    ch.psf = np.zeros((1, 3, 3))
    ch.output = None
    ch.bkg_focal_plane = None
    return ch


class _FocalPlane:
    """Minimal focal plane exposing what locate_wavelength_windows reads."""

    def __init__(self, shape, osf):
        self.shape = shape
        self.data = None
        self.cached = True
        self.spectral = type("S", (), {"size": 1})()
        self.spatial = np.zeros(1)
        self.metadata = {"oversampling": osf}


def _centred_airy_plane(osf):
    psf = create_psf(WL, (FNUM, FNUM), PIX / osf * u.um, nzero=12, shape="airy")[
        np.newaxis
    ]
    plane = np.zeros((1, N_PIX * osf, N_PIX * osf))
    fp = _FocalPlane(plane.shape, osf)
    i0, j0 = locate_wavelength_windows(psf, fp, {"type": "photometer"})
    i0, j0 = int(i0[0]), int(j0[0])
    ny, nx = psf.shape[2:]
    plane[0, i0 : i0 + ny, j0 : j0 + nx] += psf[0, 0]
    return plane, psf


def _pixel_integral_fraction(half_width, n_sub=81):
    """Fine-grid integral of the Airy PSF over the pixels: central pixel fraction."""
    edges = (np.arange(N_PIX + 1) - N_PIX / 2) * PIX
    sub = (np.arange(n_sub) + 0.5) / n_sub * PIX
    xs = (edges[:-1, None] + sub[None, :]).ravel()
    xx, yy = np.meshgrid(xs, xs)
    r = np.hypot(xx, yy) / (WL[0].value * FNUM) * np.pi
    r = np.where(r == 0, 1e-12, r)
    img = (2 * j1(r) / r) ** 2
    # same footprint as the (truncated) library PSF, so both are normalised alike
    img = img * ((np.abs(xx) <= half_width) & (np.abs(yy) <= half_width))
    per_pixel = img.reshape(N_PIX, n_sub, N_PIX, n_sub).sum(axis=(1, 3))
    return per_pixel[N_PIX // 2, N_PIX // 2] / per_pixel.sum()


@pytest.mark.parametrize("osf", [OSF, OSF_PRECISE])
def test_default_irf_integrates_the_psf_over_the_pixel(osf):
    plane, psf = _centred_airy_plane(osf)
    half = 0.5 * psf.shape[-1] / osf * PIX
    ref = _pixel_integral_fraction(half)

    ch = _channel(osf)
    assert "irf_task" not in ch.parameters["detector"]
    signal = Signal(
        spectral=WL,
        time=[0] * u.hr,
        data=plane.copy(),
        data_units="ct/s",
        metadata={"focal_plane_delta": PIX / osf * u.um, "oversampling": osf},
    )
    out = ch.apply_irf_to(signal)
    dec = out.data[0, osf // 2 :: osf, osf // 2 :: osf]
    frac = dec[N_PIX // 2, N_PIX // 2] / dec.sum()

    point = plane[0, osf // 2 :: osf, osf // 2 :: osf]
    point_frac = point.max() / point.sum()
    assert point_frac > 1.1 * ref  # point sampling is clearly too high
    assert abs(frac - ref) < abs(point_frac - ref) / 4
    if osf == OSF_PRECISE:
        assert frac == pytest.approx(ref, rel=0.01)
    # the top-hat only redistributes the flux: the total is conserved
    assert out.data.sum() == pytest.approx(plane.sum() * osf**2, rel=1e-6)


class TestRadiometricRecipeGate:
    """``populate_source_focal_plane`` applies the IRF for any osf > 1."""

    def _run(self, detector, population="sampled"):
        rm = object.__new__(RadiometricModel)
        log.Logger.__init__(rm)
        rm.payloadConfig = {"Telescope": {"Atel": 1.0}}
        channel = MagicMock()
        channel.parameters = {"detector": detector}
        channel.psf_population = population
        rm.populate_source_focal_plane({}, channel)
        return channel

    def test_applied_without_irf_task(self):
        ch = self._run(_detector(3))
        ch.apply_irf_to.assert_called_once_with(ch.focal_plane)

    def test_not_applied_at_osf_one(self):
        ch = self._run(_detector(1))
        ch.apply_irf_to.assert_not_called()

    def test_prf_population_uses_the_prf_path_without_irf(self):
        ch = self._run(_detector(3), population="prf")
        ch.rescale_radiances.assert_called_once_with()
        ch.populate_focal_plane_prf.assert_called_once_with()
        ch.rescale_contributions.assert_not_called()
        ch.populate_focal_plane.assert_not_called()
        ch.apply_irf_to.assert_not_called()

    def test_sampled_population_does_not_use_the_prf_path(self):
        ch = self._run(_detector(3), population="sampled")
        ch.rescale_contributions.assert_called_once_with()
        ch.populate_focal_plane.assert_called_once_with()
        ch.populate_focal_plane_prf.assert_not_called()
