"""create_psf: normalisation to the total energy of the profile."""

import astropy.units as u
import numpy as np
import pytest
from scipy.special import j1

from exosim.utils.psf import create_psf

WL = np.array([2.0]) * u.um
DELTA = 3.0 * u.um


def _airy(x, y, fx, fy, wl):
    """Independent implementation of the profile (x, y, wl in the same unit)."""
    r = np.sqrt(x**2 + (y * fx / fy) ** 2)
    arg = np.pi * r / (fx * wl) + 1e-20
    return (j1(arg) / arg) ** 2


def _fraction_in_window(shape, fx, fy, wl, delta, step=0.03):
    """Fraction of the total energy inside the window of the given shape,
    from a fine midpoint integration of the analytic profile over the window
    (box of +-(N + 1/2) samples, N = shape // 2), divided by the analytic
    total (fx * fy * wl**2 / pi for the Airy profile). The integration step is
    ``step * min(fx, fy) * wl``, independent of the sampling ``delta``."""
    nx, ny = (s // 2 for s in shape)
    h = step * min(fx, fy) * wl
    wx, wy = (nx + 0.5) * delta, (ny + 0.5) * delta
    mx, my = int(np.ceil(2 * wx / h)), int(np.ceil(2 * wy / h))
    hx, hy = 2 * wx / mx, 2 * wy / my
    gx = -wx + (np.arange(mx) + 0.5) * hx
    gy = -wy + (np.arange(my) + 0.5) * hy
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    integral = _airy(xx, yy, fx, fy, wl).sum() * hx * hy
    return integral / (fx * fy * wl**2 / np.pi)


@pytest.mark.parametrize("fnum", [8.0, (8.0, 8.0), (7.0, 11.0)])
@pytest.mark.parametrize("nzero", [4, 20])
def test_airy_total_normalisation_matches_integral(fnum, nzero):
    psf = create_psf(WL, fnum, DELTA, nzero=nzero, shape="airy")
    assert psf.shape[0] == 1
    fx, fy = (fnum, fnum) if np.isscalar(fnum) else fnum
    expected = _fraction_in_window(psf.shape[1:], fx, fy, WL.value[0], DELTA.value)
    assert psf.sum() < 1.0
    assert psf.sum() == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("nzero", [4, 40])
@pytest.mark.parametrize("fl_over_delta", [0.5, 0.73, 1.0, 2.0])
def test_airy_undersampled_sum_is_the_window_fraction(fl_over_delta, nzero):
    """The sum must not depend on the sampling: undersampled PSFs (F wl <
    delta, FGS1-like) used to sum to 1.6 instead of ~0.95."""
    fnum, wl = 18.7, 0.7 * u.um
    delta = fnum * wl.value / fl_over_delta * u.um
    psf = create_psf(wl[None], fnum, delta, nzero=nzero, shape="airy")
    expected = _fraction_in_window(psf.shape[1:], fnum, fnum, wl.value, delta.value)
    assert psf.sum() < 1.0
    assert psf.sum() == pytest.approx(expected, rel=3e-3)


def test_airy_table_is_consistent_with_encircled_energy():
    from scipy.special import j0

    from exosim.utils.psf import airy_rectangle_energy

    def circ(r):
        return 1.0 - (j0(np.pi * r) ** 2 + j1(np.pi * r) ** 2)

    for r in [0.1, 0.5, 1.0, 2.3, 7.0, 31.0, 150.0]:
        ee = float(airy_rectangle_energy(r, r))
        assert circ(r) <= ee <= circ(r * np.sqrt(2.0))
    assert airy_rectangle_energy(1e4, 1e4) == pytest.approx(1.0, abs=1e-3)
    # symmetric in its arguments, monotonic
    assert airy_rectangle_energy(1.3, 4.1) == pytest.approx(
        airy_rectangle_energy(4.1, 1.3)
    )
    assert airy_rectangle_energy(3.0, 3.0) > airy_rectangle_energy(2.9, 3.0)


def test_airy_large_window_converges_to_one():
    psf = create_psf(WL, 8.0, DELTA, nzero=150, shape="airy")
    assert psf.sum() == pytest.approx(1.0, abs=5e-3)


def test_known_encircled_energy_at_fourth_zero():
    # the 4th dark ring of the Airy pattern encloses 95.2% of the energy; the
    # square window of half side R_4 contains that circle, so it holds a bit
    # more (the window used to be half as wide and held 90-95%).
    psf = create_psf(WL, 8.0, 0.5 * u.um, nzero=4, shape="airy")
    assert 0.952 < psf.sum() < 0.97


@pytest.mark.parametrize("fnum", [8.0, (7.0, 11.0)])
def test_gauss_total_normalisation(fnum):
    small = create_psf(WL, fnum, DELTA, shape="gauss", nzero=1)
    large = create_psf(WL, fnum, DELTA, shape="gauss", nzero=6)
    assert small.sum() < 1.0 - 1e-3
    assert large.sum() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("delta", [0.5, 3.0, 12.0, 40.0])
@pytest.mark.parametrize("fnum", [8.0, (7.0, 11.0)])
def test_gauss_sum_is_the_exact_window_fraction(fnum, delta):
    from scipy.special import erf

    fx, ratio = (fnum, 1.0) if np.isscalar(fnum) else (fnum[0], fnum[1] / fnum[0])
    psf = create_psf(WL, fnum, delta * u.um, shape="gauss", nzero=2)
    nx, ny = psf.shape[1:]
    sx = 1.029 * fx * WL.value[0] / np.sqrt(8.0 * np.log(2.0))
    expected = erf(0.5 * nx * delta / (np.sqrt(2) * sx)) * erf(
        0.5 * ny * delta / (np.sqrt(2) * sx * ratio)
    )
    assert psf.sum() == pytest.approx(expected, rel=1e-12)


def test_zero_wavelength_is_zero():
    wl = np.array([0.0, 2.0]) * u.um
    psf = create_psf(wl, 8.0, DELTA, nzero=3)
    assert np.all(psf[0] == 0)
    assert psf[1].sum() > 0


def test_warning_when_much_energy_is_outside(caplog):
    with caplog.at_level("INFO", logger="exosim.utils.psf"):
        create_psf(WL, 8.0, DELTA, nzero=2, name="CH0")
    assert any(
        r.levelname == "WARNING" and "CH0" in r.getMessage() for r in caplog.records
    )
    assert any(r.levelname == "INFO" for r in caplog.records)
