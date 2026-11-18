"""Tests of the table loading of EstimatePlanetarySignal beyond the table."""

import astropy.units as u
import numpy as np

from exosim.tasks.astrosignal.estimate_planetary_signal import (
    EstimatePlanetarySignal,
)
from exosim.utils.binning import rebin


def _write(path, wl, a, b=None):
    with open(path, "w") as f:
        f.write("Wavelength ldc1 ldc2\n" if b is not None else "Wavelength rp/rs\n")
        for i in range(wl.size):
            row = f"{wl[i]} {a[i]}" + (f" {b[i]}" if b is not None else "")
            f.write(row + "\n")


def test_load_ldc_beyond_table(tmp_path, monkeypatch):
    twl = np.arange(0.3, 3.5, 0.001)
    path = tmp_path / "ldc.csv"
    _write(path, twl, 0.5 - 0.1 * twl, 0.2 + 0.05 * twl)
    wl = np.linspace(0.5, 3.72, 64) * u.um
    task = EstimatePlanetarySignal()
    task.set_log_name()
    warnings = []
    monkeypatch.setattr(task, "warning", lambda msg, *a, **k: warnings.append(msg))
    ldc = task.load_ldc(str(path), wl)
    ldc = np.array(ldc)
    assert ldc.shape == (2, 64)
    assert np.isfinite(ldc).all()
    np.testing.assert_allclose(ldc[0, -1], 0.5 - 0.1 * 3.72, rtol=1e-3)
    out = " ".join(warnings)
    assert "extrapolated linearly" in out
    assert "ldc.csv" in out
    # bins with data are bit identical to rebin
    inside = (wl.value > twl.min()) & (wl.value < twl.max() - 0.05)
    ref = rebin(wl.value, twl, 0.5 - 0.1 * twl)
    np.testing.assert_array_equal(ldc[0][inside], ref[inside])


def test_load_rp_beyond_table(tmp_path):
    twl = np.arange(0.5, 3.0, 0.01)
    path = tmp_path / "rp.csv"
    _write(path, twl, 0.1 + 0.01 * twl)
    wl = np.linspace(0.4, 3.5, 40) * u.um
    task = EstimatePlanetarySignal()
    task.set_log_name()
    rp = task.load_rp(str(path), wl)
    assert np.isfinite(rp).all()
    np.testing.assert_allclose(rp[-1], 0.1 + 0.01 * 3.5, rtol=1e-3)
