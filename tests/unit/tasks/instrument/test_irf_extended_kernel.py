"""
Tests of the intra pixel response function kernel, which must include the
charge diffusing into the neighbouring pixels (Barron et al. 2007 tails).
"""

import astropy.units as u
import numpy as np
import pytest
from scipy.signal import fftconvolve
from scipy.special import j1

from exosim.tasks.instrument import (
    CreateIntrapixelResponseFunction,
    CreateOversampledIntrapixelResponseFunction,
)
from exosim.tasks.instrument.create_intrapixel_response_function import (
    irf_extent_pixels,
)
from exosim.utils.psf import create_psf

D = 18.0


def _params(osf, lx, ipd=0.0, d=D):
    return {
        "detector": {
            "oversampling": osf,
            "delta_pix": d * u.um,
            "diffusion_length": lx * u.um,
            "intra_pix_distance": ipd * u.um,
        }
    }


def _kernel(osf, lx, ipd=0.0):
    k, delta = CreateIntrapixelResponseFunction()(parameters=_params(osf, lx, ipd))
    return np.asarray(k), delta


def _old_kernel(osf, ipd=0.0, d=D):
    """Pre-fix kernel for vanishing diffusion: osf x osf samples inside the pixel."""
    lx = 1e-20
    scale = np.arange(0, osf) - osf / 2 + 0.5
    x = scale * d / osf
    xx, yy = np.meshgrid(x, x)

    def r(v):
        return np.arctan(np.tanh(0.5 * (0.5 * d - v) / lx)) - np.arctan(
            np.tanh(0.5 * (-0.5 * d - v) / lx)
        )

    k = r(xx) * r(yy)
    k[np.abs(yy) > 0.5 * (d - ipd)] = 0.0
    k[np.abs(xx) > 0.5 * (d - ipd)] = 0.0
    return k * osf * osf / k.sum()


def _resp(x, lx):
    return np.arctan(np.tanh(0.5 * (0.5 * D - x) / lx)) - np.arctan(
        np.tanh(0.5 * (-0.5 * D - x) / lx)
    )


def _truth(F, lam, lx, half=30):
    """Brute force: integral of the Airy PSF times R(x)R(y)/(pi/2)**2."""
    g = np.linspace(-half * D, half * D, 2403)
    X, Y = np.meshgrid(g, g)
    r = np.pi * np.sqrt(X**2 + Y**2) / (F * lam) + 1e-12
    psf = (2 * j1(r) / r) ** 2
    psf /= psf.sum()
    p = _resp(g, lx) / (np.pi / 2)
    return float((psf * p[None, :] * p[:, None]).sum())


def _chain(F, lam, lx, osf, npix=41):
    shape = npix * osf
    delta = D / osf * u.um
    psf = create_psf(
        np.array([lam]) * u.um,
        F,
        delta,
        shape="airy",
        nzero=40,
        max_array_size=(shape, shape),
        array_size=["full", "full"],
    )[0]
    fp = np.zeros((shape, shape))
    c = osf * (npix // 2) + osf // 2
    h0, h1 = psf.shape[0] // 2, psf.shape[1] // 2
    i0, j0 = c - h0, c - h1
    a0, a1 = max(i0, 0), max(j0, 0)
    b0, b1 = min(i0 + psf.shape[0], shape), min(j0 + psf.shape[1], shape)
    fp[a0:b0, a1:b1] = psf[a0 - i0 : b0 - i0, a1 - j0 : b1 - j0]
    fp /= fp.sum()
    k, _ = _kernel(osf, lx)
    conv = fftconvolve(fp, k, mode="same")
    dec = conv[osf // 2 :: osf, osf // 2 :: osf]
    return float(dec[npix // 2, npix // 2])


@pytest.mark.parametrize("osf", [1, 2, 3, 5, 6])
@pytest.mark.parametrize("ipd", [0.0, 2.0])
def test_no_diffusion_is_identical_to_old_kernel(osf, ipd):
    """lx=0: no tails, K=0: same kernel and same convolution as before the fix."""
    new, delta = _kernel(osf, 0.0, ipd)
    old = _old_kernel(osf, ipd)
    assert new.shape == (osf, osf)
    assert u.isclose(delta, D / osf * u.um)
    np.testing.assert_allclose(new, old, rtol=0, atol=1e-12)
    img = np.random.default_rng(1).random((40 * osf, 37 * osf))
    np.testing.assert_allclose(
        fftconvolve(img, new, mode="same"),
        fftconvolve(img, old, mode="same"),
        rtol=1e-10,
        atol=1e-10,
    )


def test_extent():
    assert irf_extent_pixels(D * u.um, 1e-20 * u.um) == 0
    assert irf_extent_pixels(D * u.um, 1.7 * u.um) >= 1
    k, _ = _kernel(3, 1.7)
    K = irf_extent_pixels(D * u.um, 1.7 * u.um)
    assert k.shape == (3 * (2 * K + 1),) * 2


@pytest.mark.parametrize("osf", [1, 3, 4, 7])
@pytest.mark.parametrize(("lx", "ipd"), [(1.7, 0.0), (1.7, 1.5), (0.0, 1.5)])
def test_flat_field_preserved(osf, lx, ipd):
    k, _ = _kernel(osf, lx, ipd)
    assert k.sum() == pytest.approx(osf**2)
    n = 30 * osf
    flat = np.ones((n, n))
    conv = fftconvolve(flat, k, mode="same")
    core = conv[k.shape[0] : -k.shape[0], k.shape[0] : -k.shape[0]]
    dec = core[osf // 2 :: osf, osf // 2 :: osf]
    np.testing.assert_allclose(dec, osf**2, rtol=1e-12)
    assert np.ptp(dec) < 1e-9


@pytest.mark.parametrize(
    ("F", "lam", "tol3", "tol7"),
    [(18.7, 0.7, 0.03, 0.01), (15.5, 2.9, 0.03, 0.01)],
)
def test_central_pixel_matches_direct_integration(F, lam, tol3, tol7):
    """F*lambda/d = 0.73 and 2.5, lx = 1.7 um: central pixel of a centred Airy."""
    lx = 1.7
    truth = _truth(F, lam, lx)
    r3 = _chain(F, lam, lx, 3) / truth
    r7 = _chain(F, lam, lx, 7) / truth
    msg = f"F*lam/d={F * lam / D:.2f} osf3 ratio {r3:.4f} osf7 ratio {r7:.4f}"
    assert abs(r3 - 1) < tol3, msg
    assert abs(r7 - 1) < tol7, msg


def test_gap_zeroed_around_every_pixel_boundary():
    osf, ipd = 9, 3.0
    k, _ = _kernel(osf, 1.7, ipd)
    K = irf_extent_pixels(D * u.um, 1.7 * u.um)
    assert K >= 1
    size = k.shape[0]
    x = (np.arange(size) - size / 2 + 0.5) * D / osf
    t = np.mod(x + D / 2, D)
    dist = np.minimum(t, D - t)
    gap = dist < ipd / 2
    assert gap.sum() > 2 * (2 * K + 1) - 1  # neighbour gaps exist too
    assert np.all(k[gap, :] == 0.0)
    assert np.all(k[:, gap] == 0.0)
    assert np.all(k[np.ix_(~gap, ~gap)] > 0.0)
    # old behaviour restricted to the own pixel would have left neighbour gaps non zero
    assert k.sum() == pytest.approx(osf**2)


def test_oversampled_task_has_tails():
    osf = 4
    p = _params(osf, 1.7, 1.0)
    p["psf_shape"] = (64, 64)
    k, _ = CreateOversampledIntrapixelResponseFunction()(parameters=p)
    k = np.asarray(k)
    assert k.shape == (64 * 8 * osf,) * 2
    assert k.sum() == pytest.approx((8 * osf) ** 2)
    # lx -> 0 : footprint is only the pixel (old behaviour)
    p0 = _params(osf, 0.0)
    p0["psf_shape"] = (64, 64)
    k0, _ = CreateOversampledIntrapixelResponseFunction()(parameters=p0)
    nz = np.count_nonzero(np.asarray(k0)[0])
    assert nz <= 8 * osf + 1  # the pixel footprint only
    nz_tail = np.count_nonzero(k[0])
    assert nz_tail > nz
