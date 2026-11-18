"""PAOS PSF: flux-conserving resampling."""

import astropy.units as u
import h5py
import numpy as np
import pytest
from scipy.special import erf

from exosim.tasks.instrument.load_psf_paos import LoadPsfPaos

PAOS_PIX = 0.5e-6  # m
SIGMA_UM = 3.0  # intensity sigma (um)


def _make_file(path, m=257, gain=0.8, dx=PAOS_PIX, dy=PAOS_PIX, sigma_um=SIGMA_UM):
    """Gaussian PAOS PSF of physical intensity sigma ``sigma_um``. Axis 0 of the
    array is sampled with the file ``dy`` and axis 1 with ``dx`` (the loader
    swaps them)."""
    c = m // 2
    i, j = np.indices((m, m))
    x0 = (i - c) * dy * 1e6
    x1 = (j - c) * dx * 1e6
    amp = gain * np.exp(-0.25 * (x0**2 + x1**2) / sigma_um**2)
    with h5py.File(path, "w") as f:
        g = f.create_group("0.7").create_group("surface_01")
        g["dx"] = dx
        g["dy"] = dy
        g["amplitude"] = amp
        g["wl"] = 0.7e-6
    return path, float((amp**2).sum())


def _load(path, size, osf):
    params = {"detector": {"spatial_pix": size, "spectral_pix": size}}
    task = LoadPsfPaos()
    task.set_log_name()
    with h5py.File(path, "r") as data:
        return task.load_imac("0.7", data, params, osf, (1.0 / osf) * u.um)


def _gauss_fraction(length, delta_um, sigma_um):
    """Fraction of a centred Gaussian inside the window of ``length``
    samples (cells of size delta around the samples, centre at length // 2)."""
    c = length // 2
    a, b = (-c - 0.5) * delta_um, (length - 1 - c + 0.5) * delta_um
    s = sigma_um * np.sqrt(2.0)
    return 0.5 * (erf(b / s) - erf(a / s))


@pytest.mark.parametrize("dxy", [(0.25e-6, 0.25e-6), (0.3e-6, 0.45e-6)])
@pytest.mark.parametrize("osf", [1, 3, 5])
@pytest.mark.parametrize("size", [8, 9])
def test_window_holds_its_exact_share_of_the_volume(tmp_path, osf, size, dxy):
    path, vol = _make_file(tmp_path / "p.h5", dx=dxy[0], dy=dxy[1], sigma_um=5.0)
    img = _load(path, size, osf)
    length = size * osf - 1
    assert img.shape == (length, length)
    expected = vol * _gauss_fraction(length, 1.0 / osf, 5.0) ** 2
    assert img.sum() < vol
    assert img.sum() == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("dxy", [(0.5e-6, 0.5e-6), (0.4e-6, 0.7e-6)])
@pytest.mark.parametrize("osf", [1, 3, 5])
def test_flux_conservation_when_psf_inside_window(tmp_path, osf, dxy):
    path, vol = _make_file(tmp_path / "p.h5", dx=dxy[0], dy=dxy[1], sigma_um=1.5)
    img = _load(path, 24, osf)
    assert img.sum() == pytest.approx(vol, rel=1e-6)
    assert img.min() >= 0.0


def test_rebinning_matches_brute_force_overlap(tmp_path):
    """Independent check: every PAOS cell split in many pieces, each piece
    given to the sub-pixel that contains it."""
    path, _ = _make_file(tmp_path / "p.h5", m=41, dx=0.4e-6, dy=0.7e-6, sigma_um=2.0)
    img = _load(path, 6, 3)
    with h5py.File(path, "r") as f:
        amp = f["0.7"]["surface_01"]["amplitude"][()]
    ima = amp**2
    delta = 1.0 / 3
    n = img.shape[0]
    nsub = 3000

    def brute_matrix(size_paos):
        """w[a, i]: fraction of PAOS cell i whose pieces fall in sub-pixel a."""
        centre = np.arange(41) * size_paos
        sub_c = (np.arange(n) - n // 2) * delta + centre[20]
        w = np.zeros((n, 41))
        pieces = (np.arange(nsub) + 0.5) / nsub - 0.5
        for i in range(41):
            pos = centre[i] + pieces * size_paos
            idx = np.floor((pos - sub_c[0] + 0.5 * delta) / delta).astype(int)
            idx = idx[(idx >= 0) & (idx < n)]
            np.add.at(w[:, i], idx, 1.0 / nsub)
        return w

    out = brute_matrix(0.7) @ ima @ brute_matrix(0.4).T  # axis 0 uses file dy
    np.testing.assert_allclose(img, out, rtol=0, atol=1e-3 * out.max())


def test_window_larger_than_paos_array_is_not_padded(tmp_path):
    # PAOS array is 32 um wide, the window 63 um: nothing may be added outside
    path, vol = _make_file(tmp_path / "p.h5", m=65, gain=1.0)
    img = _load(path, 64, 1)
    assert img.sum() == pytest.approx(vol, rel=1e-6)
    assert img[0].sum() == 0
    assert img[-1].sum() == 0
    assert img[:, 0].sum() == 0
    assert img[:, -1].sum() == 0


def test_model_keeps_only_the_window_share_of_the_volume(tmp_path):
    path, vol = _make_file(tmp_path / "p.h5")
    wl = np.array([0.7, 0.7]) * u.um
    tt = np.array([0.0, 1.0]) * u.hr
    params = {
        "detector": {
            "spatial_pix": 8,
            "spectral_pix": 8,
            "delta_pix": 1.0 * u.um,
        },
    }
    cube, _ = LoadPsfPaos()(
        filename=str(path), parameters=params, wavelength=wl, time=tt
    )
    assert np.all(cube.sum(axis=(-1, -2)) < 0.9 * vol)
