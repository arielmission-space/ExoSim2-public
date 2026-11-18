"""The analytic PSF window: its size in rings and its default in PopulateFocalPlane."""

import astropy.units as u
import numpy as np
import pytest
from scipy.special import jn_zeros

from exosim.tasks.instrument.populate_focal_plane import PopulateFocalPlane, populate
from exosim.utils.focal_plane_locations import locate_wavelength_windows
from exosim.utils.psf import airy_rectangle_energy, create_psf


@pytest.mark.parametrize("nzero", [1, 2, 4, 8])
@pytest.mark.parametrize("fnum", [8.0, (8.0, 12.0)])
def test_window_reaches_the_nzero_th_dark_ring(nzero, fnum):
    wl = np.array([1.0, 2.0]) * u.um
    delta = 1.0 * u.um
    psf = create_psf(wl, fnum, delta, nzero=nzero, shape="airy")
    fx, fy = (fnum, fnum) if np.isscalar(fnum) else fnum
    ring = jn_zeros(1, nzero)[-1] / np.pi  # radius of the dark ring, in F * wl
    # half width of the window, in samples, from the centre sample
    half_x = psf.shape[1] // 2
    half_y = psf.shape[2] // 2
    assert half_x == pytest.approx(ring * fx * wl.max().value, abs=1.0)
    assert half_y == pytest.approx(ring * fy * wl.max().value, abs=1.0)


class _FocalPlane:
    """Minimal focal plane exposing what PopulateFocalPlane and
    locate_wavelength_windows read."""

    def __init__(self, n_spatial, n_spectral, delta, wl):
        self.data = np.zeros((1, n_spatial, n_spectral))
        self.spectral = np.asarray(wl.value)
        self.spectral_units = wl.unit
        self.spatial = np.zeros(1)
        self.cached = True
        self.time = np.array([0.0])
        self.metadata = {"focal_plane_delta": delta, "oversampling": 3}

    @property
    def shape(self):
        return self.data.shape


# FGS1-like photometer
FNUM = (18.7, 28.12)
WL = np.array([0.7]) * u.um
PIX = 18.0
N_PIX = 64
OSF = 3


def _total_in_focal_plane(psf_options):
    delta = PIX / OSF * u.um
    fp = _FocalPlane(N_PIX * OSF, N_PIX * OSF, delta, WL)
    parameters = {
        "type": "photometer",
        "Fnum_x": FNUM[0],
        "Fnum_y": FNUM[1],
        "psf": {"shape": "Airy", **psf_options},
    }
    psf = PopulateFocalPlane.load_psf(parameters, fp, None)
    i0, j0 = locate_wavelength_windows(psf, fp, parameters)
    source = np.ones((1, 1, 1))
    plane = populate(i0.copy(), j0.copy(), psf, fp.data, source)
    return plane[0].sum(), psf, delta


def test_default_window_is_the_detector_square():
    total, psf, delta = _total_in_focal_plane({})
    side = N_PIX * OSF - 1  # odd, centred, not larger than the detector
    assert psf.shape[2:] == (side, side)

    # the light inside the detector square (the full 192 samples; the window
    # is one sample shorter to stay odd and centred)
    u_ = 0.5 * N_PIX * OSF * delta.value / (FNUM[0] * WL.value[0])
    v_ = 0.5 * N_PIX * OSF * delta.value / (FNUM[1] * WL.value[0])
    expected = float(airy_rectangle_energy(u_, v_))
    assert total == pytest.approx(expected, rel=5e-3)
    assert total > 0.98


def test_explicit_nzero_gives_a_smaller_window_and_less_light():
    total_default, psf_default, _ = _total_in_focal_plane({})
    total_nzero, psf_nzero, _ = _total_in_focal_plane({"nzero": 4})
    assert psf_nzero.shape[2] < psf_default.shape[2]
    assert psf_nzero.shape[3] < psf_default.shape[3]
    assert total_nzero < total_default
    # nzero = 4 now reaches the 4th dark ring: ~ 95% of the light, not 91%
    assert 0.93 < total_nzero < 0.99


def test_explicit_size_is_respected():
    _, psf, _ = _total_in_focal_plane({"size_x": 41, "size_y": 21})
    assert psf.shape[2:] == (21, 41)


def test_even_focal_plane_gives_an_odd_window_inside_it():
    delta = PIX / OSF * u.um
    fp = _FocalPlane(60, 90, delta, WL)  # shorter side is even
    parameters = {
        "type": "photometer",
        "Fnum_x": FNUM[0],
        "Fnum_y": FNUM[1],
        "psf": {"shape": "Airy"},
    }
    psf = PopulateFocalPlane.load_psf(parameters, fp, None)
    assert psf.shape[2:] == (59, 59)
