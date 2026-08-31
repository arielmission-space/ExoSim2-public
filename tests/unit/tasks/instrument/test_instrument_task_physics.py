"""
Behavioural tests for the pure instrument tasks: saturation timing and source
propagation. These check the closed-form relations rather than mocking.
"""

import astropy.units as u
import numpy as np
import pytest

from exosim.models.signal import Sed, Signal
from exosim.tasks.instrument.compute_saturation import ComputeSaturation
from exosim.tasks.instrument.propagate_sources import PropagateSources


class TestComputeSaturation:
    """t_sat = f_well * well_depth / max(signal)."""

    def test_plain_array_input(self):
        fp = np.array([[10.0, 50.0], [5.0, 20.0]]) * u.ct / u.s
        t_sat, max_sig, min_sig = ComputeSaturation()(
            well_depth=100_000 * u.ct, focal_plane=fp
        )
        assert t_sat.to_value(u.s) == pytest.approx(100_000 / 50.0)
        assert max_sig.to_value(u.ct / u.s) == 50.0
        assert min_sig.to_value(u.ct / u.s) == 5.0

    def test_well_depth_fraction_scales_the_time(self):
        fp = np.full((4, 4), 25.0) * u.ct / u.s
        t_full, _, _ = ComputeSaturation()(well_depth=1000 * u.ct, focal_plane=fp)
        t_half, _, _ = ComputeSaturation()(
            well_depth=1000 * u.ct, f_well_depth=0.5, focal_plane=fp
        )
        ratio = float(t_half.to_value(u.s) / t_full.to_value(u.s))
        assert ratio == pytest.approx(0.5)

    def test_foreground_is_added_to_the_signal(self):
        fp = np.full((3, 3), 10.0) * u.ct / u.s
        frg = np.full((3, 3), 30.0) * u.ct / u.s
        t_sat, max_sig, _ = ComputeSaturation()(
            well_depth=800 * u.ct, focal_plane=fp, frg_focal_plane=frg
        )
        assert max_sig.to_value(u.ct / u.s) == 40.0
        assert t_sat.to_value(u.s) == pytest.approx(800 / 40.0)

    def test_signal_object_input(self):
        data = np.full((1, 4, 4), 8.0)
        fp = Signal(spectral=np.arange(4) * u.um, data=data, data_units=u.ct / u.s)
        t_sat, max_sig, _ = ComputeSaturation()(well_depth=400 * u.ct, focal_plane=fp)
        assert max_sig.to_value(u.ct / u.s) == 8.0
        assert t_sat.to_value(u.s) == pytest.approx(50.0)

    def test_brighter_source_saturates_faster(self):
        faint = np.full((3, 3), 1.0) * u.ct / u.s
        bright = np.full((3, 3), 100.0) * u.ct / u.s
        t_faint, _, _ = ComputeSaturation()(well_depth=1000 * u.ct, focal_plane=faint)
        t_bright, _, _ = ComputeSaturation()(well_depth=1000 * u.ct, focal_plane=bright)
        assert t_bright < t_faint


class TestPropagateSources:
    """Sources are multiplied by A_tel * efficiency * responsivity."""

    def _sed(self):
        return Sed(spectral=np.array([1.0, 2.0, 3.0]) * u.um, data=np.ones((1, 1, 3)))

    def test_defaults_leave_the_source_unchanged(self):
        out = PropagateSources()(sources={"star": self._sed()})
        np.testing.assert_allclose(out["star"].data.ravel(), [1.0, 1.0, 1.0])

    def test_area_scales_the_sed(self):
        out = PropagateSources()(
            sources={"star": self._sed()},
            Atel=3.0 * u.dimensionless_unscaled,
        )
        np.testing.assert_allclose(out["star"].data.ravel(), [3.0, 3.0, 3.0])

    def test_all_factors_multiply(self):
        out = PropagateSources()(
            sources={"star": self._sed()},
            Atel=2.0 * u.dimensionless_unscaled,
            efficiency=0.5 * u.dimensionless_unscaled,
            responsivity=10.0 * u.dimensionless_unscaled,
        )
        np.testing.assert_allclose(out["star"].data.ravel(), [10.0, 10.0, 10.0])

    def test_multiple_sources_are_all_propagated(self):
        out = PropagateSources()(
            sources={"a": self._sed(), "b": self._sed()},
            Atel=2.0 * u.dimensionless_unscaled,
        )
        assert set(out) == {"a", "b"}
        np.testing.assert_allclose(out["b"].data.ravel(), [2.0, 2.0, 2.0])
