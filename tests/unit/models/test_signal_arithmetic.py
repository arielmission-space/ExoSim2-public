"""
Behavioural tests for the Signal arithmetic dunders (including the reversed and
floor-division variants) and the unit handling of the typed Signal subclasses.
"""

import astropy.units as u
import numpy as np
import pytest

from exosim.models.signal import (
    Adu,
    Counts,
    CountsPerSecond,
    Dimensionless,
    Radiance,
    Sed,
    Signal,
)


def _signal(value=4.0, unit=u.ct / u.s):
    return Signal(
        spectral=np.array([1.0, 2.0, 3.0]) * u.um,
        data=np.full((1, 1, 3), value),
        data_units=unit,
    )


class TestReversedAndFloorOperators:
    def test_radd(self):
        out = 10 + _signal(2.0, u.dimensionless_unscaled)
        np.testing.assert_allclose(out.data.ravel(), 12.0)

    def test_rsub(self):
        out = 10 - _signal(2.0, u.dimensionless_unscaled)
        np.testing.assert_allclose(out.data.ravel(), 8.0)

    def test_rmul(self):
        out = 3 * _signal(2.0)
        np.testing.assert_allclose(out.data.ravel(), 6.0)

    def test_rtruediv(self):
        out = 12 / _signal(4.0)
        np.testing.assert_allclose(out.data.ravel(), 3.0)

    def test_floordiv(self):
        out = _signal(7.0) // 2
        np.testing.assert_allclose(out.data.ravel(), 3.0)

    def test_rfloordiv(self):
        out = 7 // _signal(2.0)
        np.testing.assert_allclose(out.data.ravel(), 3.0)

    def test_multiplication_combines_units(self):
        a = _signal(2.0, u.ct / u.s)
        b = _signal(3.0, u.s)
        out = a * b
        np.testing.assert_allclose(out.data.ravel(), 6.0)
        assert out.data_units == u.ct

    def test_division_combines_units(self):
        a = _signal(6.0, u.ct)
        b = _signal(2.0, u.s)
        out = a / b
        np.testing.assert_allclose(out.data.ravel(), 3.0)
        assert out.data_units == u.ct / u.s

    def test_sum_of_two_signals(self):
        out = _signal(2.0) + _signal(5.0)
        np.testing.assert_allclose(out.data.ravel(), 7.0)

    def test_add_a_quantity_converts_its_units(self):
        out = _signal(2.0, u.ct / u.s) + (3000.0 * u.ct / u.ks)
        # 3000 ct/ks == 3 ct/s
        np.testing.assert_allclose(out.data.ravel(), 5.0)

    def test_add_two_signals_with_convertible_units(self):
        a = _signal(2.0, u.ct / u.s)
        b = _signal(3600.0, u.ct / u.hr)  # == 1 ct/s
        out = a + b
        np.testing.assert_allclose(out.data.ravel(), 3.0, rtol=1e-6)


class TestTypedSubclasses:
    @pytest.mark.parametrize(
        ("klass", "unit"),
        [
            (Sed, u.W / u.m**2 / u.um),
            (Radiance, u.W / u.m**2 / u.um / u.sr),
            (CountsPerSecond, u.ct / u.s),
            (Counts, u.ct),
            (Adu, u.adu),
        ],
    )
    def test_data_with_convertible_unit_is_rescaled(self, klass, unit):
        # give the data a unit that is the target unit scaled by 1000
        scaled = (1000.0 * np.ones((1, 1, 3))) * (unit / 1000)
        sig = klass(spectral=np.array([1.0, 2.0, 3.0]) * u.um, data=scaled)
        assert sig.data_units == unit
        np.testing.assert_allclose(sig.data.ravel(), 1.0, rtol=1e-6)

    @pytest.mark.parametrize("klass", [Sed, Radiance, CountsPerSecond, Counts, Adu])
    def test_incompatible_unit_raises(self, klass):
        bad = np.ones((1, 1, 3)) * u.kg
        with pytest.raises(u.UnitsError):
            klass(spectral=np.array([1.0, 2.0, 3.0]) * u.um, data=bad)

    def test_dimensionless_accepts_plain_array(self):
        sig = Dimensionless(
            spectral=np.array([1.0, 2.0, 3.0]) * u.um, data=np.ones((1, 1, 3))
        )
        assert sig.data_units == u.dimensionless_unscaled

    @pytest.mark.parametrize(
        ("klass", "unit"),
        [
            (Sed, u.W / u.m**2 / u.um),
            (Radiance, u.W / u.m**2 / u.um / u.sr),
            (CountsPerSecond, u.ct / u.s),
            (Counts, u.ct),
            (Adu, u.adu),
            (Dimensionless, u.dimensionless_unscaled),
        ],
    )
    def test_copy_preserves_the_typed_class(self, klass, unit):
        sig = klass(
            spectral=np.array([1.0, 2.0, 3.0]) * u.um,
            data=np.ones((1, 1, 3)) * unit,
        )
        # copy() routes through _create_new_instance, which picks the class
        # from the data unit
        assert isinstance(sig.copy(), klass)


class TestUnitNormalisation:
    def test_none_unit_normalises_to_dimensionless(self):
        sig = _signal()
        assert sig._normalize_units(None) == u.dimensionless_unscaled

    def test_fractional_power_unit_is_recognised(self):
        sig = _signal()
        assert sig._normalize_units(u.hr**0.5) == u.hr**0.5


class TestTimeSlicing:
    def _timeseries(self):
        return Signal(
            spectral=np.array([1.0, 2.0]) * u.um,
            data=np.arange(5 * 1 * 2).reshape(5, 1, 2).astype(float),
            time=np.array([0.0, 1.0, 2.0, 3.0, 4.0]) * u.hr,
        )

    def test_get_slice_with_quantities(self):
        sig = self._timeseries()
        sl = sig.get_slice(1.0 * u.hr, 3.0 * u.hr)
        assert sl.shape[0] == 2

    def test_set_slice_replaces_the_window(self):
        sig = self._timeseries()
        sig.set_slice(1.0 * u.hr, 3.0 * u.hr, np.full((2, 1, 2), -1.0))
        np.testing.assert_allclose(sig.data[1:3], -1.0)

    def test_get_slice_accepts_bare_floats_as_hours(self):
        # the docstring promises "if a float is given, it's assumed to be in hours"
        sig = self._timeseries()
        by_float = sig.get_slice(1.0, 3.0)
        by_quantity = sig.get_slice(1.0 * u.hr, 3.0 * u.hr)
        np.testing.assert_array_equal(by_float, by_quantity)

    def test_set_slice_accepts_bare_floats(self):
        sig = self._timeseries()
        sig.set_slice(1.0, 3.0, np.full((2, 1, 2), 5.0))
        np.testing.assert_allclose(sig.data[1:3], 5.0)


class TestWriteWithoutOutput:
    def test_write_warns_when_no_output(self):
        sig = _signal()
        warnings = []
        sig.warning = warnings.append
        sig.write()  # no output configured
        assert len(warnings) == 1
        assert "No output" in warnings[0]
