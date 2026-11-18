"""The wavelength of a physical pixel must not depend on the oversampling."""

import astropy.units as u
import numpy as np
import pytest

from exosim.tasks.instrument.create_focal_plane_array import CreateFocalPlaneArray


def _params(osf, center=None):
    x = np.linspace(-100, 800, 20) * u.um
    wl = 2.0 + 0.004 * x.value
    params = {
        "type": "spectrometer",
        "detector": {
            "spatial_pix": 6,
            "spectral_pix": 40,
            "delta_pix": 18 * u.um,
            "oversampling": osf,
        },
        "wl_solution": {
            "wl_solution_task": "LoadWavelengthSolution",
            "data": {
                "Wavelength": wl * u.um,
                "x": x,
                "y": np.zeros(x.size) * u.um,
            },
        },
        "wl_min": 2.0 * u.um,
        "wl_max": 5.0 * u.um,
    }
    if center is not None:
        params["wl_solution"]["center"] = center
    return params


def _plane(osf, center=None):
    task = CreateFocalPlaneArray()
    task.set_log_name()
    return task(parameters=_params(osf, center), efficiency=None)


def _physical_wl(osf, center=None):
    fp = _plane(osf, center)
    return np.asarray(fp.spectral)[osf // 2 :: osf]


@pytest.mark.parametrize("osf", [3, 5])
def test_no_center_wavelength_is_osf_independent(osf):
    np.testing.assert_allclose(_physical_wl(osf), _physical_wl(1), rtol=1e-12)


def test_no_center_osf1_is_unchanged():
    fp = _plane(1)
    pix = (np.arange(40) * 18 * u.um).to(u.um).value
    x = np.linspace(-100, 800, 20)
    par = np.polyfit(x, 2.0 + 0.004 * x, 2)
    np.testing.assert_allclose(np.asarray(fp.spectral), np.poly1d(par)(pix))


@pytest.mark.parametrize("osf", [3, 5])
def test_auto_center_wavelength_is_osf_independent(osf):
    np.testing.assert_allclose(
        _physical_wl(osf, "auto"), _physical_wl(1, "auto"), rtol=1e-12
    )
