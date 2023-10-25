import logging
import os
import unittest
from copy import deepcopy

import astropy.units as u
import numpy as np
from inputs import test_dir

from exosim.log import setLogLevel
from exosim.models.signal import Adu
from exosim.models.signal import Counts
from exosim.models.signal import CountsPerSecond
from exosim.models.signal import Dimensionless
from exosim.models.signal import Radiance
from exosim.models.signal import Sed
from exosim.models.signal import Signal

setLogLevel(logging.DEBUG)


class SignalUnitsTest(unittest.TestCase):
    cached = False

    def test_Signal_class(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )
        self.assertListEqual(list(wl.value), list(signal.spectral))
        self.assertEqual("um", signal.spectral_units)
        self.assertListEqual(list(time_grid.value), list(signal.time))
        self.assertEqual("hr", signal.time_units)

    def test_CountsPerSeconds_class(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        countpersecond = CountsPerSecond(
            spectral=wl,
            data=data * u.Unit("ct/s"),
            time=time_grid,
            cached=self.cached,
        )
        self.assertEqual("ct/s", countpersecond.data_units)

        countpersecond = CountsPerSecond(
            spectral=wl,
            data=data * u.Unit("ct/hr"),
            time=time_grid,
            cached=self.cached,
        )
        self.assertEqual("ct/s", countpersecond.data_units)

        with self.assertRaises(u.UnitsError):
            CountsPerSecond(
                spectral=wl,
                data=data * u.Unit("ct"),
                time=time_grid,
                cached=self.cached,
            )

    def test_Counts_class(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        counts = Counts(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )
        self.assertEqual("ct", counts.data_units)

        with self.assertRaises(u.UnitsError):
            Counts(
                spectral=wl,
                data=data * u.Unit("ct/s"),
                time=time_grid,
                cached=self.cached,
            )

    def test_Sed_class(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        sed = Sed(
            spectral=wl,
            data=data * u.W / u.m**2 / u.um,
            time=time_grid,
            cached=self.cached,
        )
        self.assertEqual("W/m**2/um", sed.data_units)

        sed = Sed(
            spectral=wl,
            data=data * u.W / u.cm**2 / u.um,
            time=time_grid,
            cached=self.cached,
        )
        self.assertEqual("W/m**2/um", sed.data_units)

        with self.assertRaises(u.UnitsError):
            Sed(
                spectral=wl,
                data=data * u.W / u.um,
                time=time_grid,
                cached=self.cached,
            )

    def test_Radiance_class(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        radiance = Radiance(
            spectral=wl,
            data=data * u.W / u.m**2 / u.um / u.sr,
            time=time_grid,
            cached=self.cached,
        )
        self.assertEqual("W/m**2/um/sr", radiance.data_units)

        radiance = Radiance(
            spectral=wl,
            data=data * u.W / u.cm**2 / u.um / u.sr,
            time=time_grid,
            cached=self.cached,
        )
        self.assertEqual("W/m**2/um/sr", radiance.data_units)
        with self.assertRaises(u.UnitsError):
            Radiance(
                spectral=wl,
                data=data * u.W / u.cm**2 / u.um,
                time=time_grid,
                cached=self.cached,
            )

    def test_Adu_class(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        aduPerSecond = Adu(
            spectral=wl,
            data=data * u.Unit("adu"),
            time=time_grid,
            cached=self.cached,
        )
        self.assertEqual("adu", aduPerSecond.data_units)

        with self.assertRaises(u.UnitsError):
            Adu(
                spectral=wl,
                data=data * u.Unit("adu/s"),
                time=time_grid,
                cached=self.cached,
            )

    def test_automatic_units_conversion(self):
        wl = np.linspace(0.1, 1, 10) * u.m
        time_grid = np.linspace(1, 5, 10) * u.s
        data = np.random.random_sample((10, 1, 10))
        signal = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )
        self.assertEqual("hr", signal.time_units)
        self.assertEqual("um", signal.spectral_units)

    def test_units_conversion(self):
        wl = np.linspace(0.1, 1, 10) * u.m
        time_grid = np.linspace(1, 5, 10) * u.s
        data = np.random.random_sample((10, 1, 10)) * u.m**2
        data_ = deepcopy(data)
        signal = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )
        signal.to(u.cm**2)
        self.assertEqual("cm^2", signal.data_units)
        np.testing.assert_array_equal(signal.data, data_.value * 10000.0)

    def test_automatic_units_application(self):
        wl = np.linspace(0.1, 1, 10)
        data = np.random.random_sample(10)
        signal = Signal(spectral=wl, data=data)
        self.assertEqual("um", signal.spectral_units)

        signal = Signal(spectral=wl * u.pix, data=data)
        self.assertEqual("pix", signal.spectral_units)

        data = np.random.random_sample(10) * u.s
        signal = Signal(spectral=wl, data=data)
        self.assertEqual("s", signal.data_units)

    def test_inappropriate_units_error(self):
        with self.assertRaises(u.UnitConversionError):
            wl = np.linspace(0.1, 1, 10) * u.s
            data = np.random.random_sample(10)
            Signal(spectral=wl, data=data)

    # def test_CountsPerSeconds_class(self):
    #     wl = np.linspace(0.1, 1, 10) * u.um
    #     data = np.random.random_sample((10, 1, 10))
    #     time_grid = np.linspace(1, 5, 10) * u.hr
    #
    #     countpersecond = CountsPerSecond(spectral=wl,
    #                                      data=data * u.Unit('ct/s'),
    #                                      time=time_grid, cached=self.cached)
    #     self.assertEqual('ct/s', countpersecond.data_units)


class SignalSliceTest(unittest.TestCase):
    cached = False

    def test_get_slice_edges(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.ones((10, 1, 10))
        time_grid = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5] * u.hr
        signal = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )
        start, stop = signal._find_slice(1 * u.hr, 2 * u.hr)
        np.testing.assert_array_equal(time_grid[start:stop], [1, 1.5] * u.hr)

    def test_get_slice(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.ones((10, 1, 10))
        time_grid = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5] * u.hr
        data[0:2, :, :] *= 2
        signal = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )
        np.testing.assert_array_equal(
            signal.get_slice(1 * u.hr, 2 * u.hr), np.ones((2, 1, 10)) * 2
        )

    def test_set_slice(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.ones((10, 1, 10))
        time_grid = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5] * u.hr
        signal = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )
        data_to_set = np.ones((2, 1, 10)) * 2
        data[0:2, :, :] *= 2
        signal.set_slice(1 * u.hr, 2 * u.hr, data_to_set)
        np.testing.assert_array_equal(signal.data, data)


class SignalRebinTest(unittest.TestCase):
    cached = False

    def test_spectral_interpolator(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.ones((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )
        new_wl = np.linspace(0.1, 1, 20) * u.um
        signal.spectral_rebin(new_wl)
        self.assertEqual((10, 1, 20), signal.data.shape)
        self.assertListEqual(list(new_wl.value), list(signal.spectral))

    def test_temporal_interpolator(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.ones((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )
        new_time = np.linspace(1, 5, 20) * u.hr
        signal.temporal_rebin(new_time)
        self.assertEqual((20, 1, 10), signal.data.shape)
        self.assertListEqual(list(new_time.value), list(signal.time))

    def test_temporal_replicator(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.ones((1, 1, 10))

        signal = Signal(spectral=wl, data=data, cached=self.cached)
        new_time = np.linspace(1, 5, 20) * u.hr
        signal.temporal_rebin(new_time)
        self.assertEqual((20, 1, 10), signal.data.shape)
        self.assertListEqual(list(new_time.value), list(signal.time))

    def test_spectral_binner(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.ones((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )
        new_wl = np.linspace(0.1, 1, 5) * u.um
        signal.spectral_rebin(new_wl)
        self.assertEqual((10, 1, 5), signal.data.shape)
        self.assertListEqual(list(new_wl.value), list(signal.spectral))

    def test_temporal_binner(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.ones((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )
        new_time = np.linspace(1, 5, 5) * u.hr
        signal.temporal_rebin(new_time)
        self.assertEqual((5, 1, 10), signal.data.shape)
        self.assertListEqual(list(new_time.value), list(signal.time))

    def test_wrong_interpolation_inputs(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.ones((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr
        new_time = np.linspace(1, 5, 20) * u.hr
        new_wl = np.linspace(0.1, 1, 20) * u.um

        with self.assertRaises(u.UnitConversionError):
            signal = Signal(
                spectral=wl, data=data, time=time_grid, cached=self.cached
            )
            signal.temporal_rebin(new_wl)
        with self.assertRaises(u.UnitConversionError):
            signal = Signal(
                spectral=wl, data=data, time=time_grid, cached=self.cached
            )
            signal.spectral_rebin(new_time)

    cached = False


class SignalOperationTest(unittest.TestCase):
    cached = False
    other_cached = False

    # testing the addition
    def test_add_signals(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        data = np.ones((10, 1, 10)) * 2
        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.other_cached
        )

        signal3 = signal1 + signal2
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 3, signal3.data)

        signal4 = signal1 + signal2 + signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 4, signal4.data)

    def test_add_array(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        data = np.ones((10, 1, 10)) * 2

        signal3 = signal1 + data
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 3, signal3.data)

        signal4 = signal1 + data + signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 4, signal4.data)

        signal5 = data + signal1 + data
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 5, signal5.data)

    def test_add_const(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        signal3 = signal1 + 2
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 3, signal3.data)

        signal3 = 2 + signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 3, signal3.data)

    def test_add_units(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.m
        )

        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.m
        )
        self.assertEqual((signal1 + signal2).data_units, "m")

        # conversion test
        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.cm
        )
        self.assertEqual((signal1 + signal2).data_units, "m")
        np.testing.assert_array_equal(
            np.ones((10, 1, 10)) + 0.01, (signal1 + signal2).data
        )

        # sum with array
        data = np.ones((10, 1, 10)) * u.m
        self.assertEqual((signal1 + data).data_units, "m")
        self.assertEqual((data + signal1).data_units, "m")

        # sum with scalar
        data = 1 * u.m
        self.assertEqual((signal1 + data).data_units, "m")
        self.assertEqual((data + signal1).data_units, "m")

        # testing different units compositions
        data = np.ones((10, 1, 10)) * u.s
        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        with self.assertRaises(u.UnitConversionError):
            signal1 + signal2
        with self.assertRaises(u.UnitConversionError):
            signal1 + data
        data = 1 * u.s
        with self.assertRaises(u.UnitConversionError):
            signal1 + data

    # testing the subtraction
    def test_sub_signals(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        data = np.ones((10, 1, 10)) * 2
        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.other_cached
        )

        signal3 = signal2 - signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)), signal3.data)

        signal4 = signal2 - signal1 - signal1
        np.testing.assert_array_equal(np.zeros((10, 1, 10)), signal4.data)

    def test_sub_array(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        data = np.ones((10, 1, 10)) * 2

        signal3 = data - signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)), signal3.data)

        signal4 = data - signal1 - signal1
        np.testing.assert_array_equal(np.zeros((10, 1, 10)), signal4.data)

        signal5 = signal1 - data - data
        np.testing.assert_array_equal(
            np.ones((10, 1, 10)) * (-3), signal5.data
        )

    def test_sub_const(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        signal3 = signal1 - 2
        np.testing.assert_array_equal(
            np.ones((10, 1, 10)) * (-1), signal3.data
        )

        signal3 = 2 - signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)), signal3.data)

    def test_sub_units(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.m
        )

        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.m
        )
        self.assertEqual((signal1 - signal2).data_units, "m")

        # conversion test
        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.cm
        )
        self.assertEqual((signal1 - signal2).data_units, "m")
        np.testing.assert_array_equal(
            np.ones((10, 1, 10)) - 0.01, (signal1 - signal2).data
        )

        # sum with array
        data = np.ones((10, 1, 10)) * u.m
        self.assertEqual((signal1 - data).data_units, "m")
        self.assertEqual((data - signal1).data_units, "m")

        # sum with scalar
        data = 1 * u.m
        self.assertEqual((signal1 - data).data_units, "m")
        self.assertEqual((data - signal1).data_units, "m")

        # testing different units compositions
        data = np.ones((10, 1, 10)) * u.s
        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        with self.assertRaises(u.UnitConversionError):
            signal1 - signal2
        with self.assertRaises(u.UnitConversionError):
            signal1 - data
        data = 1 * u.s
        with self.assertRaises(u.UnitConversionError):
            signal1 - data

    # testing the multiplication
    def test_mul_signals(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        data = np.ones((10, 1, 10)) * 2
        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.other_cached
        )

        signal3 = signal2 * signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal3.data)

        signal4 = signal2 * signal1 * signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal4.data)

    def test_mul_array(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        data = np.ones((10, 1, 10)) * 2

        signal3 = data * signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal3.data)

        signal4 = data * signal1 * signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal4.data)

        signal5 = signal1 * data * data
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * (4), signal5.data)

    def test_mul_const(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        signal3 = signal1 * 2
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal3.data)

        signal3 = 2 * signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal3.data)

    def test_mul_units(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.m
        )

        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.m
        )
        self.assertEqual((signal1 * signal2).data_units, "m**2")

        # sum with array
        data = np.ones((10, 1, 10)) * u.m
        self.assertEqual((signal1 * data).data_units, "m**2")
        self.assertEqual((data * signal1).data_units, "m**2")

        # sum with scalar
        data = 1 * u.m
        self.assertEqual((signal1 * data).data_units, "m**2")
        self.assertEqual((data * signal1).data_units, "m**2")

        # conversion test
        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.cm
        )
        self.assertEqual((signal1 * signal2).data_units, "cm m")
        data = 1 * u.cm
        self.assertEqual((signal1 * data).data_units, "cm m")
        data = np.ones((10, 1, 10)) * u.cm
        self.assertEqual((signal1 * data).data_units, "cm m")

    # testing the division
    def test_truediv_signals(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        data = np.ones((10, 1, 10)) * 2
        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.other_cached
        )

        signal3 = signal2 / signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal3.data)

        signal4 = signal2 / signal1 / signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal4.data)

    def test_truediv_array(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        data = np.ones((10, 1, 10)) * 2

        signal3 = data / signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal3.data)

        signal4 = data / signal1 / signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal4.data)

        signal5 = signal1 / data / data
        np.testing.assert_array_equal(np.ones((10, 1, 10)) / (4), signal5.data)

    def test_truediv_const(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        signal3 = signal1 / 2
        np.testing.assert_array_equal(np.ones((10, 1, 10)) / 2, signal3.data)

        signal3 = 2 / signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal3.data)

    def test_truediv_units(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.m
        )

        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.m
        )
        self.assertEqual((signal1 / signal2).data_units, u.Unit(""))

        # sum with array
        data = np.ones((10, 1, 10)) * u.m
        self.assertEqual((signal1 / data).data_units, u.Unit(""))
        self.assertEqual((data / signal1).data_units, u.Unit(""))

        # sum with scalar
        data = 1 * u.m
        self.assertEqual((signal1 / data).data_units, u.Unit(""))
        self.assertEqual((data / signal1).data_units, u.Unit(""))

        # conversion test
        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.cm
        )
        self.assertEqual((signal1 / signal2).data_units, "m/cm")
        data = 1 * u.cm
        self.assertEqual((signal1 / data).data_units, "m/cm")
        data = np.ones((10, 1, 10)) * u.cm
        self.assertEqual((signal1 / data).data_units, "m/cm")

    def test_floordiv_signals(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        data = np.ones((10, 1, 10)) * 2
        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.other_cached
        )

        signal3 = signal2 // signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal3.data)

        signal4 = signal2 // signal1 // signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal4.data)

    def test_floordiv_array(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        data = np.ones((10, 1, 10)) * 2

        signal3 = data // signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal3.data)

        signal4 = data // signal1 // signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal4.data)

        signal5 = signal1 // data // data
        np.testing.assert_array_equal(
            np.ones((10, 1, 10)) // (4), signal5.data
        )

    def test_floordiv_const(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, cached=self.cached
        )

        signal3 = signal1 // 2
        np.testing.assert_array_equal(np.ones((10, 1, 10)) // 2, signal3.data)

        signal3 = 2 // signal1
        np.testing.assert_array_equal(np.ones((10, 1, 10)) * 2, signal3.data)

    def test_floordiv_units(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.m
        )

        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.m
        )
        self.assertEqual((signal1 // signal2).data_units, u.Unit(""))

        # sum with array
        data = np.ones((10, 1, 10)) * u.m
        self.assertEqual((signal1 // data).data_units, u.Unit(""))
        self.assertEqual((data // signal1).data_units, u.Unit(""))

        # sum with scalar
        data = 1 * u.m
        self.assertEqual((signal1 // data).data_units, u.Unit(""))
        self.assertEqual((data // signal1).data_units, u.Unit(""))

        # conversion test
        signal2 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.cm
        )
        self.assertEqual((signal1 // signal2).data_units, "m/cm")
        data = 1 * u.cm
        self.assertEqual((signal1 // data).data_units, "m/cm")
        data = np.ones((10, 1, 10)) * u.cm
        self.assertEqual((signal1 // data).data_units, "m/cm")

    def test_units_class(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        # dimensionless
        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.Unit(""),
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.Unit(""),
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 + signal2), Dimensionless)
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.m,
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=1 / u.m,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 * signal2), Dimensionless)
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.m,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 // signal2), Dimensionless)
        self.assertEqual(type(signal1 / signal2), Dimensionless)

        # sed
        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.W / u.m**2 / u.um,
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.W / u.m**2 / u.um,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 + signal2), Sed)
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=1 / u.m**2 / u.um,
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.W,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 * signal2), Sed)
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.W / u.m**2,
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.um,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 // signal2), Sed)
        self.assertEqual(type(signal1 / signal2), Sed)

        # radiance
        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.W / u.m**2 / u.um / u.sr,
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.W / u.m**2 / u.um / u.sr,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 + signal2), Radiance)
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=1 / u.m**2 / u.um / u.sr,
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.W,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 * signal2), Radiance)
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.W / u.m**2 / u.sr,
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.um,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 // signal2), Radiance)
        self.assertEqual(type(signal1 / signal2), Radiance)

        # countspersecond
        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.ct / u.s,
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.ct / u.s,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 + signal2), CountsPerSecond)
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=1 / u.s,
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.ct,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 * signal2), CountsPerSecond)
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.ct,
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.s,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 // signal2), CountsPerSecond)
        self.assertEqual(type(signal1 / signal2), CountsPerSecond)

        # countspersecond
        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.ct / u.s,
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.ct / u.s,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 + signal2), CountsPerSecond)
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=1 / u.s,
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.ct,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 * signal2), CountsPerSecond)
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.ct,
            cached=self.cached,
        )
        signal2 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.s,
            cached=self.other_cached,
        )
        self.assertEqual(type(signal1 // signal2), CountsPerSecond)
        self.assertEqual(type(signal1 / signal2), CountsPerSecond)


class SignalCachedDataTest(
    SignalUnitsTest, SignalSliceTest, SignalOperationTest
):
    cached = True
    other_cached = True


# what if we mix cached signal with no cached signals?
class SignalCachedMixedDataTest(SignalOperationTest):
    cached = True
    other_cached = False


class SignalCachedMixedReversedDataTest(SignalOperationTest):
    cached = False
    other_cached = True


class SignalWriteTest(unittest.TestCase):
    def test_write(self):
        import h5py

        from exosim.output.hdf5.hdf5 import HDF5Output

        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.m
        )

        fname = os.path.join(test_dir, "output_test.h5")
        with HDF5Output(fname) as o:
            signal1.write(o, "test_signal")

        f = h5py.File(fname, "r")
        for a in ["data", "time", "spectral"]:
            np.testing.assert_array_equal(
                f["test_signal"][a][()], dict(signal1)[a]
            )
        for a in ["data_units", "time_units", "spectral_units"]:
            self.assertEqual(
                f["test_signal"][a][()].decode("utf-8"), dict(signal1)[a]
            )
        f.close()
        os.remove(fname)

    def test_load(self):
        import h5py

        from exosim.output.hdf5.hdf5 import HDF5Output
        from exosim.output.hdf5.utils import load_signal

        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))
        signal1 = Signal(
            spectral=wl, data=data, time=time_grid, data_units=u.m
        )

        fname = os.path.join(test_dir, "output_test.h5")
        with HDF5Output(fname) as o:
            signal1.write(o, "test_signal")

        f = h5py.File(fname, "r")
        signal = load_signal(f["test_signal"])
        for a in ["data", "time", "spectral"]:
            np.testing.assert_array_equal(dict(signal)[a], dict(signal1)[a])
        for a in ["data_units", "time_units", "spectral_units"]:
            self.assertEqual(dict(signal1)[a], dict(signal1)[a])
        f.close()
        os.remove(fname)

    def test_load_metadata(self):
        import h5py

        from exosim.output.hdf5.hdf5 import HDF5Output
        from exosim.output.hdf5.utils import load_signal

        wl = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr

        data = np.ones((10, 1, 10))

        meta = {
            "test": "this is test",
            "test_qua": 1 * u.um,
            "test_val": 2,
            "test_dict": {"test1": 1, "test2": "text"},
        }
        signal1 = Signal(
            spectral=wl,
            data=data,
            time=time_grid,
            data_units=u.m,
            metadata=meta,
        )

        fname = os.path.join(test_dir, "output_test.h5")
        with HDF5Output(fname) as o:
            signal1.write(o, "test_signal")

        f = h5py.File(fname, "r")
        signal = load_signal(f["test_signal"])

        print(signal.metadata)
        self.assertEqual(signal.metadata, meta)
        f.close()
        os.remove(fname)


class SignalCopyTest(unittest.TestCase):
    def test_clone(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Signal(spectral=wl, data=data, time=time_grid)

        signal1 = signal.copy()
        np.testing.assert_array_equal(signal.spectral, signal1.spectral)
        np.testing.assert_array_equal(signal.time, signal1.time)
        np.testing.assert_array_equal(signal.data, signal1.data)

    def test_overwrite(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Signal(
            spectral=wl, data=data, time=time_grid, metadata={"value": "test"}
        )

        signal1 = signal.copy(metadata={"value": "test1"})
        self.assertTrue(signal1.metadata["value"] == "test1")
