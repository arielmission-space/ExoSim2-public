"""
Behavioural test for the :class:`~exosim.tasks.instrument.load_psf.LoadPsf`
base task: ``execute`` normalises the PSF cube and, when handed an output file,
writes the cube / norm / wavelength datasets (idempotently).
"""

import astropy.units as u
import h5py
import numpy as np

from exosim.output import SetOutput
from exosim.tasks.instrument.load_psf import LoadPsf


class _DummyLoadPsf(LoadPsf):
    """Minimal concrete PSF loader: a flat cube whose shape follows the grids."""

    def model(self, filename, parameters, wavelength, time):
        nt = max(len(np.atleast_1d(time)), 1)
        nw = len(wavelength)
        return np.ones((nt, nw, 4, 4)) / 16.0


def _call(output=None):
    wl = np.linspace(1.0, 2.0, 5) * u.um
    tt = np.array([0.0]) * u.hr
    return _DummyLoadPsf()(
        wavelength=wl,
        time=tt,
        parameters={},
        filename="unused",
        output=output,
    )


def test_returns_cube_and_normalisation():
    cube, norms = _call()
    assert cube.shape == (1, 5, 4, 4)
    # each PSF sums to 1, so every norm entry is 1
    np.testing.assert_allclose(norms, 1.0)


def test_writes_psf_group_to_the_output(tmp_path):
    fname = tmp_path / "psf.h5"
    with SetOutput(str(fname)).use(append=True) as out:
        _call(output=out)
    with h5py.File(fname, "r") as f:
        assert "psf/psf_cube" in f
        assert "psf/norm" in f
        assert f["psf"]["psf_cube"].attrs["time_axis"] == 0


def test_second_write_is_skipped_without_error(tmp_path):
    fname = tmp_path / "psf2.h5"
    with SetOutput(str(fname)).use(append=True) as out:
        _call(output=out)
        # writing again into the same group must hit the "already exists" guards
        _call(output=out)
    with h5py.File(fname, "r") as f:
        assert "psf/psf_cube" in f
