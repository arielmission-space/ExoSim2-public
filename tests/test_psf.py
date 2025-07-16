import logging
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from photutils.aperture import EllipticalAperture, RectangularAperture

import exosim.tasks.instrument as instrument
import exosim.utils.aperture as psf_util
from exosim.log import setLogLevel
from exosim.utils.klass_factory import find_task
from exosim.utils.psf import create_psf

setLogLevel(logging.DEBUG)


@pytest.fixture
def wl_grid():
    """Fixture per creare una griglia di lunghezze d'onda."""
    return np.linspace(1.95, 3.78, 10) * u.um


@pytest.mark.parametrize("shape", ["airy", "gauss"])
def test_create_psf_shape(wl_grid, shape):
    psf = create_psf(wl_grid, 15.5, 18 * u.um, shape=shape)
    assert psf.shape == (len(wl_grid), 15, 15)


@pytest.mark.parametrize("nzero", [4, 8])
def test_create_psf_nzero(wl_grid, nzero):
    psf = create_psf(wl_grid, 15.5, 18 * u.um, shape="airy", nzero=nzero)
    assert psf.shape[-1] > 4


@pytest.mark.parametrize(
    "max_array_size, expected_shape",
    [
        ((5, 5), (10, 5, 5)),
        ((4, 4), (10, 5, 5)),
    ],
)
def test_create_psf_max_size(wl_grid, max_array_size, expected_shape):
    psf = create_psf(wl_grid, 15.5, 18 * u.um, shape="airy", max_array_size=max_array_size)
    assert psf.shape == expected_shape


@pytest.mark.parametrize(
    "array_size, expected_shape",
    [
        ((31, 21), (10, 31, 21)),
        ((30, 20), (10, 31, 21)),
    ],
)
def test_create_psf_array_size(wl_grid, array_size, expected_shape):
    psf = create_psf(wl_grid, 15.5, 18 * u.um, shape="airy", array_size=array_size)
    assert psf.shape == expected_shape


def test_create_psf_plot(wl_grid, skip_plot):
    if skip_plot:
        pytest.skip("Skipping plot")
    psf = create_psf(wl_grid, 15.5, 18 * u.um, shape="airy")
    plt.imshow(psf[0])
    plt.title("PSF")
    plt.colorbar()
    plt.show()


@pytest.fixture
def psf():
    """Fixture per generare una PSF fissa."""
    return create_psf(4 * u.um, 15.5, 6 * u.um, shape="airy")


def test_energy_rectangular(psf):
    sizes, surf, ene = psf_util.find_rectangular_aperture(psf, 0.84)
    assert np.round(ene, decimals=2) >= 0.84



def test_aperture_plot_rectangular(psf, skip_plot):
    if skip_plot:
        pytest.skip("This test only produces plots")
    sizes, _, _ = psf_util.find_rectangular_aperture(psf, 0.84)
    aper = RectangularAperture(
        (psf.shape[1] // 2, psf.shape[0] // 2),
        h=sizes[1],
        w=sizes[0],
    )
    plt.imshow(psf)
    aper.plot(color="g", lw=2, label="Photometry aperture")
    plt.legend()
    plt.show()


def test_energy_elliptical(psf):
    sizes, _, ene = psf_util.find_elliptical_aperture(psf, 0.84)
    assert np.round(ene, decimals=2) >= 0.84


def test_aperture_plot_elliptical(psf, skip_plot):
    if skip_plot:
        pytest.skip("This test only produces plots")
    sizes, _, _ = psf_util.find_elliptical_aperture(psf, 0.84)
    aper = EllipticalAperture(
        (psf.shape[1] // 2, psf.shape[0] // 2),
        a=sizes[1],
        b=sizes[0],
    )
    plt.imshow(psf)
    aper.plot(color="g", lw=2, label="Photometry aperture")
    plt.legend()
    plt.show()


def test_find_task():
    task = find_task("LoadPsfPaos", instrument.LoadPsf)
    assert task is not None


@pytest.fixture
def paos_data(test_data_dir):
    """Fixture per fornire un file di dati PAOS."""
    return os.path.join(test_data_dir, "PAOS_ab0.h5")


def test_load_paos(paos_data):
    wl = np.linspace(1, 2.8, 5) * u.um
    tt = np.linspace(0, 10, 2) * u.hr
    parameters = {
        "detector": {
            "oversampling": 2,
            "delta_pix": 10 * u.um,
            "spatial_pix": 32,
            "spectral_pix": 32,
        }
    }
    loadPsfPaos = instrument.LoadPsfPaos()
    cube, norms = loadPsfPaos(
        filename=paos_data, parameters=parameters, wavelength=wl, time=tt
    )
    assert cube.shape[1] == len(wl)
    assert cube.shape[0] == len(tt)
