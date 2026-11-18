"""Inputs of the pixel response function population: fine PAOS PSFs and
unrounded pointing offsets."""

import astropy.units as u
import h5py
import numpy as np
import pytest
from scipy.special import erf

import exosim.tasks.instrument as instrument
from exosim.tasks.instrument.load_psf_paos import LoadPsfPaos

DELTA = 18.0


def paos_file(path, npix, sigma_um=10.0, cell=2.0e-6):
    """Gaussian PAOS PSF normalised to unit energy, 0.7 um."""
    m = 301
    c = m // 2
    i, j = np.indices((m, m))
    x0, x1 = (i - c) * cell * 1e6, (j - c) * cell * 1e6
    inten = np.exp(-0.5 * (x0**2 + x1**2) / sigma_um**2)
    inten /= inten.sum()
    with h5py.File(path, "w") as f:
        g = f.create_group("0.7").create_group("surface_01")
        g["dx"] = cell
        g["dy"] = cell
        g["amplitude"] = np.sqrt(inten)
        g["wl"] = 0.7e-6
    return path


def test_paos_fine_psf_has_the_exact_window_energy(tmp_path):
    """The fine PSF sums to the exact PAOS energy inside the window, also when
    the window cuts the PSF."""
    npix = 4
    path = paos_file(tmp_path / "paos.h5", npix, sigma_um=30.0)
    params = {
        "psf": {},
        "detector": {
            "delta_pix": DELTA * u.um,
            "spatial_pix": npix,
            "spectral_pix": npix,
        },
    }
    wl, cube = LoadPsfPaos().load_fine_psf(str(path), params, 8)
    # window: npix * 8 - 1 fine cells of 18 / 8 um, PSF sigma = 30 um
    n = npix * 8 - 1
    half = (n // 2 + 0.5) * DELTA / 8
    expected_x = erf(half / (np.sqrt(2) * 30.0))
    assert wl[0] == pytest.approx(0.7)
    assert cube.shape == (1, n, n)
    assert cube.sum() == pytest.approx(expected_x**2, rel=3e-3)


def test_paos_fine_psf_is_centred_on_the_central_sample(tmp_path):
    npix = 6
    path = paos_file(tmp_path / "paos.h5", npix, sigma_um=8.0)
    params = {
        "psf": {},
        "detector": {
            "delta_pix": DELTA * u.um,
            "spatial_pix": npix,
            "spectral_pix": npix,
        },
    }
    _, cube = LoadPsfPaos().load_fine_psf(str(path), params, 9)
    n = cube.shape[1]
    assert cube.shape[1] == cube.shape[2] == npix * 9 - 1
    assert np.unravel_index(cube[0].argmax(), cube[0].shape) == (n // 2, n // 2)
    assert np.allclose(cube[0], cube[0][::-1, ::-1], rtol=1e-9, atol=1e-15)


def test_pointing_offsets_can_be_returned_unrounded():
    params = {
        "detector": {
            "delta_pix": DELTA * u.um,
            "oversampling": 3,
            "plate_scale": 10 * u.mas / u.um,  # 180 mas per pixel
        }
    }
    metadata = {"parsed_parameters": {"ra": 0 * u.deg, "dec": 0 * u.deg}}
    pointing = ((0.37 * 0.18) * u.arcsec, 0 * u.deg)
    exact = instrument.ComputeSourcesPointingOffset()(
        source=metadata, pointing=pointing, parameters=params, exact=True
    )
    rounded = instrument.ComputeSourcesPointingOffset()(
        source=metadata, pointing=pointing, parameters=params
    )
    assert exact[0] == pytest.approx(0.37 * 3, abs=1e-6)
    assert exact[1] == pytest.approx(0.0, abs=1e-6)
    assert isinstance(exact[0], float)
    assert rounded == [1, 0]
    # without pointing or without coordinates: no offset
    assert instrument.ComputeSourcesPointingOffset()(
        source={"parsed_parameters": {}}, pointing=None, parameters=params, exact=True
    ) == [0.0, 0.0]


@pytest.mark.parametrize("cell_um", [2.48, 4.5, 5.5, 6.9])
def test_paos_fine_psf_is_a_cubic_interpolation_of_a_gaussian(tmp_path, cell_um):
    """An analytic Gaussian (0.56 pixel sigma) written in PAOS format, with PAOS
    cells of 2.5 to 7 um: the fine image of the PRF tables follows the Gaussian
    within 0.3% on the peak and 0.7% on the cells above 10% of the peak. The
    bilinear interpolation of the PAOS intensity flattens it (2% for the finest
    cell, up to 12% for the coarsest, per fine cell). The image is non
    negative and sums to the exact energy of the window."""
    npix, sampling, sigma = 16, 10, 0.56 * DELTA
    path = paos_file(tmp_path / "paos.h5", npix, sigma, cell_um * 1e-6)
    params = {
        "psf": {},
        "detector": {
            "delta_pix": DELTA * u.um,
            "spatial_pix": npix,
            "spectral_pix": npix,
        },
    }
    task = LoadPsfPaos()
    _, cube = task.load_fine_psf(str(path), params, sampling)
    image = cube[0]
    assert image.min() >= 0.0
    assert image.sum() == pytest.approx(1.0, abs=1e-6)  # the window holds 14 sigma
    assert 0.0 <= task.last_clipped_fraction < 1e-3

    h = DELTA / sampling
    x = (np.arange(image.shape[0]) - image.shape[0] // 2) * h

    def cell_energy(x0):
        edge = np.sqrt(2) * sigma
        return 0.5 * (erf((x0 + h / 2) / edge) - erf((x0 - h / 2) / edge))

    truth = np.outer(cell_energy(x), cell_energy(x))
    peak = image.shape[0] // 2
    assert image[peak, peak] / truth[peak, peak] - 1 == pytest.approx(0.0, abs=3e-3)
    core = truth > 0.1 * truth.max()
    assert np.abs(image[core] / truth[core] - 1).max() < 7e-3
