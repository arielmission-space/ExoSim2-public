"""PSFs outside the wavelengths of the PAOS file: diffraction scaling of the edge PSF."""

import astropy.units as u
import numpy as np
import pytest
from scipy.interpolate import interp1d

from exosim.tasks.instrument.load_psf_paos import LoadPsfPaos
from exosim.utils.psf import scale_psf_image

DELTA = 18.0


def gauss(n, sigma, centre=None):
    c = n // 2
    i, j = np.indices((n, n))
    img = np.exp(-0.5 * ((i - c) ** 2 + (j - c) ** 2) / sigma**2)
    return img / img.sum()


def second_moment(img):
    c = img.shape[0] // 2
    i, j = np.indices(img.shape)
    r2 = (i - c) ** 2 + (j - c) ** 2
    return np.sqrt((img * r2).sum() / img.sum() / 2.0)


# ------------------------------------------------------------------ helper
@pytest.mark.parametrize("s", [1.3, 2.8])
@pytest.mark.parametrize("n", [101, 100])
def test_scale_conserves_energy_and_scales_width(s, n):
    img = gauss(n, 2.0) * 0.7
    out = scale_psf_image(img, s)
    assert out.min() >= 0.0
    assert out.sum() == pytest.approx(img.sum(), rel=1e-12)
    assert second_moment(out) == pytest.approx(s * second_moment(img), rel=2e-2)


def test_scale_identity_and_centre():
    img = gauss(61, 3.0)
    np.testing.assert_array_equal(scale_psf_image(img, 1.0), img)
    out = scale_psf_image(img, 1.7)
    c = img.shape[0] // 2
    i = np.arange(img.shape[0]) - c
    assert (out.sum(axis=1) * i).sum() == pytest.approx(0.0, abs=1e-12)
    assert np.unravel_index(out.argmax(), out.shape) == (c, c)


def test_scale_loses_energy_leaving_the_window():
    img = gauss(31, 5.0)
    out = scale_psf_image(img, 2.5)
    assert out.sum() < img.sum()
    assert out.min() >= 0.0


def test_scale_shrink_conserves_energy():
    img = gauss(61, 4.0)
    out = scale_psf_image(img, 0.6)
    assert out.sum() == pytest.approx(img.sum(), rel=1e-12)
    assert out.min() >= 0.0


# ------------------------------------------------------------------ sampled
def cube_and_wl():
    wl = np.array([0.4, 0.5, 0.6, 0.7]) * u.um
    cube = np.array([gauss(81, 3.0 * w / 0.4) for w in wl.value])
    return cube, wl


def test_wl_interpolate_inside_the_range_is_linear():
    cube, wl = cube_and_wl()
    req = np.array([0.41, 0.45, 0.55, 0.62, 0.7, 0.4]) * u.um
    out = LoadPsfPaos.wl_interpolate(cube, req, wl)
    ref = interp1d(wl.value, cube.reshape(4, -1), axis=0)(req.value).reshape(out.shape)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-15)


def test_wl_interpolate_outside_the_range():
    cube, wl = cube_and_wl()
    req = np.array([1.5, 0.2, 0.9, 2.0, 0.55]) * u.um
    out = LoadPsfPaos.wl_interpolate(cube, req, wl)
    assert out.min() >= 0.0
    # above: grows with wavelength, energy not larger than the edge one
    order = [2, 0, 3]  # 0.9, 1.5, 2.0
    widths = [second_moment(out[k]) for k in order]
    assert widths[0] < widths[1] < widths[2]
    for k in order:
        assert out[k].sum() <= cube[-1].sum() * (1 + 1e-12)
    assert out[2].sum() == pytest.approx(
        cube[-1].sum(), rel=1e-7
    )  # tail beyond the window only
    assert second_moment(out[2]) == pytest.approx(
        0.9 / 0.7 * second_moment(cube[-1]), rel=2e-2
    )
    # below: shrinks, energy conserved
    assert second_moment(out[1]) < second_moment(cube[0])
    assert out[1].sum() == pytest.approx(cube[0].sum(), rel=1e-12)
    # the order of the requested wavelengths is kept
    np.testing.assert_allclose(out[4], LoadPsfPaos.wl_interpolate(cube, req[4:], wl)[0])


def test_wl_interpolate_unsorted_sampled_wavelengths():
    cube, wl = cube_and_wl()
    req = np.array([1.2, 0.45]) * u.um
    a = LoadPsfPaos.wl_interpolate(cube, req, wl)
    b = LoadPsfPaos.wl_interpolate(cube[::-1], req, wl[::-1])
    np.testing.assert_allclose(a, b)


def test_single_psf_is_scaled_at_other_wavelengths():
    cube, wl = cube_and_wl()
    out = LoadPsfPaos.wl_interpolate(cube[:1], np.array([0.4, 0.8]) * u.um, wl[:1])
    np.testing.assert_array_equal(out[0], cube[0])
    assert second_moment(out[1]) == pytest.approx(2 * second_moment(cube[0]), rel=2e-2)
