"""PAOS PSF alignment: the PSF centre must land on ``shape // 2``."""

import astropy.units as u
import h5py
import numpy as np
import pytest

from exosim.tasks.instrument.load_psf_paos import LoadPsfPaos

PAOS_PIX = 0.5e-6  # m


def _make_file(path, m=129, sigma=6.0, centre=None):
    c = m // 2 if centre is None else centre
    y, x = np.indices((m, m))
    amp = np.exp(-0.25 * ((x - c) ** 2 + (y - c) ** 2) / sigma**2)
    with h5py.File(path, "w") as f:
        g = f.create_group("0.7").create_group("surface_01")
        g["dx"] = PAOS_PIX
        g["dy"] = PAOS_PIX
        g["amplitude"] = amp
        g["wl"] = 0.7e-6
    return path


def _centroid(img):
    y, x = np.indices(img.shape)
    return (img * y).sum() / img.sum(), (img * x).sum() / img.sum()


@pytest.mark.parametrize("osf", [1, 3, 5])
@pytest.mark.parametrize("size", [20, 21])
def test_centroid_at_shape_over_2(tmp_path, osf, size):
    fname = _make_file(tmp_path / "psf.h5")
    params = {"detector": {"spatial_pix": size, "spectral_pix": size}}
    delta = 1.0 * u.um / osf
    task = LoadPsfPaos()
    task.set_log_name()
    with h5py.File(fname, "r") as data:
        img = task.load_imac("0.7", data, params, osf, delta)
    assert img.shape == (size * osf - 1,) * 2
    cy, cx = _centroid(img)
    assert abs(cy - img.shape[0] // 2) < 0.02
    assert abs(cx - img.shape[1] // 2) < 0.02


@pytest.mark.parametrize("osf", [1, 3, 5])
def test_old_alignment_index_was_off_by_one_for_even_size(osf):
    """The old code aligned at index size*osf//2; the returned image has
    size*osf-1 samples, whose ``shape // 2`` centre is one lower when
    size*osf is even."""
    n = 20 * osf
    length = n - 1
    assert n // 2 - length // 2 == (1 if n % 2 == 0 else 0)
