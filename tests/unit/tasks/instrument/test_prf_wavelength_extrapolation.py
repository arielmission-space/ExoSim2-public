"""PRF tables beyond the wavelengths of the PAOS file."""

import astropy.units as u
import h5py
import numpy as np
import pytest

from exosim.tasks.instrument.load_psf_paos import LoadPsfPaos
from exosim.tasks.instrument.populate_focal_plane_prf import (
    PopulateFocalPlanePrf,
    extend_prf_nodes,
)
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


def paos_multi(path, wls=(0.4, 0.55, 0.7), cell=2.0e-6, m=301):
    c = m // 2
    i, j = np.indices((m, m))
    with h5py.File(path, "w") as f:
        for w in wls:
            sigma = 10.0 * w / 0.7
            x0, x1 = (i - c) * cell * 1e6, (j - c) * cell * 1e6
            inten = np.exp(-0.5 * (x0**2 + x1**2) / sigma**2)
            inten /= inten.sum()
            g = f.create_group(f"{w}").create_group("surface_01")
            g["dx"] = cell
            g["dy"] = cell
            g["amplitude"] = np.sqrt(inten)
            g["wl"] = w * 1e-6
    return path


def prf_setup(path, hi, lo=0.45):
    from exosim.models.signal import Signal

    params = {
        "type": "photometer",
        "psf": {"psf_task": "LoadPsfPaos", "filename": str(path)},
        "Fnum_x": 20,
        "Fnum_y": 20,
        "detector": {
            "delta_pix": DELTA * u.um,
            "spatial_pix": 8,
            "spectral_pix": 8,
            "oversampling": 1,
        },
    }
    wl = np.linspace(lo, hi, 32) * u.um
    fp = Signal(
        spectral=wl,
        data=np.zeros((1, 8, 8, 1))[..., 0],
        time=np.zeros(1) * u.hr,
        metadata={"oversampling": 1},
    )
    return params, fp


def test_prf_nodes_cover_the_band(tmp_path):
    path = paos_multi(tmp_path / "p.h5")
    params, fp = prf_setup(path, hi=2.0, lo=0.3)
    nodes, cube = PopulateFocalPlanePrf.load_fine_psf(params, fp, 4)
    assert nodes[0] == pytest.approx(0.3)
    assert nodes[-1] == pytest.approx(2.0)
    assert np.all(np.diff(nodes) > 0)
    ratio = nodes[1:] / nodes[:-1]
    extra = (nodes[1:] <= 0.4 + 1e-12) | (nodes[:-1] >= 0.7 - 1e-12)
    assert ratio[extra].max() <= 1.05 + 1e-12
    assert cube.shape[0] == nodes.size
    assert cube.min() >= 0.0
    assert {0.4, 0.55, 0.7} <= set(np.round(nodes, 12))


def test_prf_nodes_unchanged_when_the_band_is_inside(tmp_path):
    path = paos_multi(tmp_path / "p.h5")
    params, fp = prf_setup(path, hi=0.7, lo=0.45)
    nodes, _cube = PopulateFocalPlanePrf.load_fine_psf(params, fp, 4)
    np.testing.assert_allclose(nodes, [0.4, 0.55, 0.7])


def test_extend_prf_nodes_uses_the_same_scaling():
    img = gauss(61, 3.0)
    nodes, cube = extend_prf_nodes(np.array([0.7]), img[None], 0.7, 1.4)
    assert nodes[-1] == pytest.approx(1.4)
    assert second_moment(cube[-1]) == pytest.approx(2.0 * second_moment(img), rel=2e-2)
    np.testing.assert_allclose(cube[-1], scale_psf_image(img, 1.4 / 0.7))


def test_sampled_and_prf_psfs_agree_beyond_the_paos_range(tmp_path):
    path = paos_multi(tmp_path / "p.h5")
    lam = 1.4
    params, fp = prf_setup(path, hi=lam)
    sampling = 4
    nodes, fine = PopulateFocalPlanePrf.load_fine_psf(params, fp, sampling)
    fine_lam = fine[np.argmin(abs(nodes - lam))]

    params["detector"]["oversampling"] = sampling
    task = LoadPsfPaos()
    task.set_log_name()
    sampled = task.model(str(path), params, np.array([lam]) * u.um, np.zeros(1) * u.hr)[
        0, 0
    ]
    # both come from the 0.7 um PSF scaled by 2, on grids of the same cell size
    assert second_moment(fine_lam) * DELTA / sampling == pytest.approx(
        second_moment(sampled) * DELTA / sampling, rel=0.05
    )
    assert fine_lam.sum() == pytest.approx(sampled.sum(), rel=0.03)


def test_cache_key_depends_on_the_band(tmp_path):
    path = paos_multi(tmp_path / "p.h5")
    p1, f1 = prf_setup(path, hi=1.0)
    p2, f2 = prf_setup(path, hi=2.0)
    assert PopulateFocalPlanePrf.cache_key(p1, f1) != PopulateFocalPlanePrf.cache_key(
        p2, f2
    )
