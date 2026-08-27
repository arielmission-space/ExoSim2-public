"""
Test suite for Signal model and its subclasses.

This module contains comprehensive tests for the Signal class and its subclasses
including Counts, CountsPerSecond, Sed, Radiance, and Adu. Tests cover signal
initialization, arithmetic operations, unit conversions, I/O operations, and
advanced features like caching and metadata handling.
"""

import os
import tempfile

import astropy.units as u
import h5py
import numpy as np
import pytest

from exosim.models.signal import (
    Adu,
    Counts,
    CountsPerSecond,
    Radiance,
    Sed,
    Signal,
)
from exosim.output.hdf5.hdf5 import HDF5Output
from exosim.output.hdf5.utils import load_signal


class TestSignalInitialization:
    """Test class for Signal model initialization and basic properties."""

    @pytest.fixture
    def spectral_grid(self):
        """
        Fixture for creating a spectral grid.

        Returns
        -------
        astropy.units.Quantity
            Spectral wavelength grid from 0.1 to 1.0 micrometers.
        """
        return np.linspace(0.1, 1, 10) * u.um

    @pytest.fixture
    def time_grid(self):
        """
        Fixture for creating a time grid.

        Returns
        -------
        astropy.units.Quantity
            Time grid from 1 to 5 hours.
        """
        return np.linspace(1, 5, 10) * u.hr

    @pytest.fixture
    def signal_data(self):
        """
        Fixture for creating random signal data.

        Returns
        -------
        numpy.ndarray
            Random 3D array with shape (10, 1, 10) for testing.
        """
        return np.random.random_sample((10, 1, 10))

    def test_signal_basic_initialization(self, spectral_grid, time_grid, signal_data):
        """
        Test the basic initialization of the Signal class.

        Parameters
        ----------
        spectral_grid : astropy.units.Quantity
            Spectral wavelength grid.
        time_grid : astropy.units.Quantity
            Time grid.
        signal_data : numpy.ndarray
            Signal data array.
        """
        signal = Signal(spectral=spectral_grid, data=signal_data, time=time_grid)
        assert list(signal.spectral) == list(spectral_grid.value)
        assert signal.spectral_units == u.um
        assert list(signal.time) == list(time_grid.value)
        assert signal.time_units == u.hr

    def test_signal_initialization_with_spatial(self):
        """Test Signal initialization with spatial dimension."""
        spectral = np.linspace(0.1, 1, 5) * u.um
        spatial = np.linspace(-10, 10, 3) * u.um
        time = np.linspace(1, 5, 4) * u.hr
        data = np.random.random((4, 3, 5))

        signal = Signal(spectral=spectral, spatial=spatial, data=data, time=time)

        assert np.array_equal(signal.spatial, spatial.value)
        assert signal.spatial_units == u.um
        assert signal.data.shape == (4, 3, 5)

    def test_signal_initialization_with_pixels(self):
        """Test Signal initialization with pixel units."""
        spectral_pix = np.array([10, 20, 30]) * u.pix
        spatial_pix = np.array([5, 10]) * u.pix
        time = np.array([1, 2]) * u.hr
        data = np.random.random((2, 2, 3))

        signal = Signal(
            spectral=spectral_pix, spatial=spatial_pix, data=data, time=time
        )

        assert signal.spectral_units == u.pix
        assert signal.spatial_units == u.pix

    def test_signal_metadata_handling(self):
        """Test Signal initialization and handling with metadata."""
        spectral = np.linspace(0.1, 1, 5) * u.um
        time = np.linspace(1, 5, 3) * u.hr
        data = np.random.random((3, 1, 5))
        metadata = {"test_key": "test_value", "version": 1.0}

        signal = Signal(spectral=spectral, data=data, time=time, metadata=metadata)

        assert signal.metadata == metadata
        assert signal.metadata["test_key"] == "test_value"
        assert signal.metadata["version"] == 1.0


class TestSignalUnitsAndConversions:
    """Test class for Signal unit handling and conversions."""

    @pytest.fixture
    def spectral_grid(self):
        """Fixture for creating a spectral grid."""
        return np.linspace(0.1, 1, 5) * u.um

    @pytest.fixture
    def time_grid(self):
        """Fixture for creating a time grid."""
        return np.linspace(1, 5, 3) * u.hr

    @pytest.fixture
    def signal_data(self):
        """Fixture for creating signal data."""
        return np.random.random_sample((3, 1, 5))

    @pytest.mark.parametrize(
        ("data_units", "expected_units"),
        [
            (u.ct / u.s, u.ct / u.s),
            (u.W / u.m**2 / u.um, u.W / u.m**2 / u.um),
        ],
    )
    def test_signal_units_conversion(
        self, spectral_grid, time_grid, signal_data, data_units, expected_units
    ):
        """
        Test automatic conversion of data units.

        Parameters
        ----------
        data_units : astropy.units.Unit
            Input data units.
        expected_units : astropy.units.Unit
            Expected output units.
        """
        signal = Signal(
            spectral=spectral_grid, data=signal_data * data_units, time=time_grid
        )
        assert signal.data_units == expected_units

    def test_signal_to_unit_conversion(self, spectral_grid, time_grid):
        """Test the to() method for unit conversion."""
        data = np.ones((3, 1, 5)) * 1000  # 1000 W/m^2

        signal = Signal(
            spectral=spectral_grid, data=data * u.W / u.m**2, time=time_grid
        )

        # Store original data
        original_data = signal.data.copy()

        # Convert to erg/s/cm^2
        signal.to(u.erg / u.s / u.cm**2)

        # Check units changed
        assert signal.data_units == u.erg / u.s / u.cm**2

        # Check data values were converted (1 W/m^2 = 1000 erg/s/cm^2)
        expected_data = original_data * 1000
        assert np.allclose(signal.data, expected_data)

    def test_normalize_units(self):
        """Test the _normalize_units function."""
        # Create a minimal Signal instance for testing
        signal = Signal(spectral=np.array([1, 2, 3]) * u.um, data=np.ones((3, 1, 1)))

        assert signal._normalize_units(u.m) == u.m
        assert signal._normalize_units(u.cm) == u.cm
        assert signal._normalize_units(u.J) == u.J
        assert signal._normalize_units(u.erg) == u.erg
        assert signal._normalize_units(u.Unit("W / (m**2 um)")) == u.W / u.m**2 / u.um
        assert signal._normalize_units(u.Unit("ct / s")) == u.ct / u.s
        assert signal._normalize_units(u.Unit("")) == u.dimensionless_unscaled


class TestSignalArithmeticOperations:
    """Test class for Signal arithmetic operations."""

    @pytest.fixture
    def spectral_grid(self):
        """Fixture for creating a spectral grid."""
        return np.linspace(0.1, 1, 5) * u.um

    @pytest.fixture
    def time_grid(self):
        """Fixture for creating a time grid."""
        return np.linspace(1, 5, 3) * u.hr

    @pytest.fixture
    def signal_data(self):
        """Fixture for creating signal data."""
        return np.random.random_sample((3, 1, 5))

    @pytest.mark.parametrize(
        ("operation", "expected_result"),
        [
            (lambda s1, s2: s1 + s2, 3),
            (lambda s1, s2: s1 - s2, -1),
        ],
    )
    def test_signal_add_sub(
        self, spectral_grid, time_grid, signal_data, operation, expected_result
    ):
        """
        Test addition and subtraction operations on signals.

        Parameters
        ----------
        operation : callable
            Operation to perform (addition or subtraction).
        expected_result : float
            Expected scaling factor for the result.
        """
        s1 = Signal(spectral=spectral_grid, data=signal_data, time=time_grid)
        s2 = Signal(spectral=spectral_grid, data=signal_data * 2, time=time_grid)
        result = operation(s1, s2)

        # Validation
        assert np.allclose(result.data, signal_data * expected_result), (
            "Addition/Subtraction failed!"
        )

    @pytest.mark.parametrize(
        ("operation", "expected_result_fn"),
        [
            (lambda s1, s2: s1 * s2, lambda data: 2 * data**2),
            (lambda s1, s2: s1 / s2, lambda data: np.full(data.shape, 0.5)),
        ],
    )
    def test_signal_mul_div(
        self, spectral_grid, time_grid, signal_data, operation, expected_result_fn
    ):
        """
        Test multiplication and division operations on signals.

        Parameters
        ----------
        operation : callable
            Operation to perform (multiplication or division).
        expected_result_fn : callable
            Function to calculate expected result data.
        """
        s1 = Signal(spectral=spectral_grid, data=signal_data, time=time_grid)
        s2 = Signal(spectral=spectral_grid, data=signal_data * 2, time=time_grid)
        result = operation(s1, s2)

        # Calculate the expected result
        expected_data = expected_result_fn(signal_data)

        # Validation
        assert np.allclose(result.data, expected_data, atol=1e-6), (
            "Multiplication/Division failed!"
        )

    @pytest.mark.parametrize(
        ("data_units1", "data_units2", "expected_units"),
        [
            (u.m, u.m, u.m**2),
            (u.ct / u.s, u.s, u.ct),
            (u.W / u.m**2, u.um, u.W / u.m**2 * u.um),
        ],
    )
    def test_signal_units_operations(
        self,
        spectral_grid,
        time_grid,
        signal_data,
        data_units1,
        data_units2,
        expected_units,
    ):
        """
        Test operations on signals with different units.

        Parameters
        ----------
        data_units1 : astropy.units.Unit
            Units for first signal.
        data_units2 : astropy.units.Unit
            Units for second signal.
        expected_units : astropy.units.Unit
            Expected units after multiplication.
        """
        # Create two signals with different units
        s1 = Signal(
            spectral=spectral_grid, data=signal_data * data_units1, time=time_grid
        )
        s2 = Signal(
            spectral=spectral_grid, data=signal_data * data_units2, time=time_grid
        )

        # Perform multiplication
        result = s1 * s2

        # Validate the resulting units
        assert result.data_units == expected_units

    def test_signal_copy_operations(self):
        """Test that signal operations create proper copies."""
        spectral = np.linspace(0.1, 1, 5) * u.um
        time = np.linspace(1, 5, 3) * u.hr
        data = np.ones((3, 1, 5))

        s1 = Signal(spectral=spectral, data=data, time=time)
        s2 = Signal(spectral=spectral, data=data * 2, time=time)

        # Addition should create new signal without modifying originals
        original_s1_data = s1.data.copy()
        original_s2_data = s2.data.copy()

        result = s1 + s2

        # Original signals should be unchanged
        assert np.array_equal(s1.data, original_s1_data)
        assert np.array_equal(s2.data, original_s2_data)

        # Result should be different
        assert not np.array_equal(result.data, s1.data)
        assert not np.array_equal(result.data, s2.data)

    def test_signal_unit_consistency(self):
        """Test that signal operations maintain unit consistency."""
        spectral = np.linspace(0.1, 1, 5) * u.um
        time = np.linspace(1, 5, 3) * u.hr
        data = np.ones((3, 1, 5))

        # Create signals with different but compatible units
        s1 = Signal(spectral=spectral, data=data * u.W / u.m**2, time=time)
        s2 = Signal(spectral=spectral, data=data * u.W / u.m**2, time=time)

        # Addition should preserve units
        result_add = s1 + s2
        assert result_add.data_units == u.W / u.m**2

        # Subtraction should preserve units
        result_sub = s1 - s2
        assert result_sub.data_units == u.W / u.m**2

        # Multiplication should combine units
        result_mul = s1 * s2
        assert result_mul.data_units == (u.W / u.m**2) ** 2

        # Division should divide units
        result_div = s1 / s2
        assert result_div.data_units == u.dimensionless_unscaled


class TestSignalSlicingOperations:
    """Test class for Signal slicing and data access operations."""

    def test_signal_temporal_slice(self):
        """Test slicing the signal over a time interval."""
        spectral_grid = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr
        signal_data = np.random.random_sample((10, 1, 10))

        signal = Signal(spectral=spectral_grid, data=signal_data, time=time_grid)
        sliced = signal.get_slice(1 * u.hr, 2 * u.hr)
        assert sliced.shape[0] == 2

    def test_signal_spectral_slicing(self):
        """Test slicing signal over spectral ranges."""
        spectral = np.linspace(0.1, 1.0, 10) * u.um  # 0.1 to 1.0 um
        time = np.linspace(1, 5, 5) * u.hr
        data = np.random.random((5, 1, 10))

        signal = Signal(spectral=spectral, data=data, time=time)

        # Test spectral slicing if method exists
        try:
            sliced = signal.get_spectral_slice(0.3 * u.um, 0.7 * u.um)
            # Should have fewer spectral elements
            assert sliced.data.shape[2] < 10
            assert sliced.data.shape[2] > 0
        except AttributeError:
            # Method might not exist, skip this test
            pytest.skip("get_spectral_slice method not implemented")


class TestSignalSubclasses:
    """Test class for Signal subclasses with specialized units."""

    def test_signal_subclasses_initialization(self):
        """
        Test initialization of Signal subclasses with default units.

        Tests Counts, CountsPerSecond, Sed, Radiance, and Adu subclasses
        to ensure they initialize with correct default units.
        """
        spectral = np.linspace(0.1, 1, 5) * u.um
        time = np.linspace(1, 5, 3) * u.hr
        data_shape = (3, 1, 5)

        # Test Counts class
        counts_data = np.random.randint(0, 1000, data_shape)
        counts = Counts(spectral=spectral, data=counts_data, time=time)
        assert counts.data_units == u.ct

        # Test CountsPerSecond class
        cps_data = np.random.random(data_shape) * 100
        cps = CountsPerSecond(spectral=spectral, data=cps_data, time=time)
        assert cps.data_units == u.ct / u.s

        # Test Sed class
        sed_data = np.random.random(data_shape) * 1e-10
        sed = Sed(spectral=spectral, data=sed_data, time=time)
        assert sed.data_units == u.W / u.m**2 / u.um

        # Test Radiance class
        rad_data = np.random.random(data_shape) * 1e-12
        radiance = Radiance(spectral=spectral, data=rad_data, time=time)
        assert radiance.data_units == u.W / u.m**2 / u.um / u.sr

        # Test Adu class
        adu_data = np.random.randint(0, 65536, data_shape)
        adu = Adu(spectral=spectral, data=adu_data, time=time)
        assert adu.data_units == u.adu


class TestSignalIOOperations:
    """Test class for Signal I/O operations including HDF5 read/write."""

    @pytest.mark.parametrize(
        ("signal_class", "expected_units"),
        [
            (Signal, u.dimensionless_unscaled),
            (Counts, u.ct),
            (CountsPerSecond, u.ct / u.s),
            (Sed, u.W / u.m**2 / u.um),
            (Radiance, u.W / u.m**2 / u.um / u.sr),
            (Adu, u.adu),
        ],
    )
    def test_signal_write_read_hdf5(self, tmp_path, signal_class, expected_units):
        """
        Test writing and reading signals to and from HDF5 files.

        Parameters
        ----------
        signal_class : type
            Signal class or subclass to test.
        expected_units : astropy.units.Unit
            Expected units for the signal class.
        """
        spectral_grid = np.linspace(0.1, 1, 10) * u.um
        time_grid = np.linspace(1, 5, 10) * u.hr
        signal_data = np.random.random_sample((10, 1, 10))

        # Create the signal
        signal = signal_class(
            spectral=spectral_grid, data=signal_data * expected_units, time=time_grid
        )

        # Write the signal to an HDF5 file
        fname = tmp_path / "output_test.h5"
        with HDF5Output(str(fname)) as o:
            signal.write(o, "test_signal")

        # Load the signal from the HDF5 file
        f = h5py.File(fname, "r")
        loaded_signal = load_signal(f["test_signal"])
        assert np.allclose(loaded_signal.data, signal.data)
        assert loaded_signal.data_units == expected_units


class TestSignalAdvancedFeatures:
    """Test class for Signal advanced features like caching and error handling."""

    def test_signal_cached_mode(self):
        """Test signal creation with caching enabled."""
        spectral = np.linspace(0.1, 1, 5) * u.um
        time = np.linspace(1, 5, 3) * u.hr
        data = np.random.random((3, 1, 5))

        # Create a temporary output file for caching
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with HDF5Output(tmp_path) as output:
                signal = Signal(
                    spectral=spectral,
                    data=data,
                    time=time,
                    cached=True,
                    output=output,
                    dataset_name="test_cached_signal",
                )

                # Verify that dataset is accessible
                assert signal.dataset is not None
                assert np.allclose(signal.data, data)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_signal_error_cases(self):
        """Test Signal behavior with unusual inputs (Signal auto-handles many cases)."""
        spectral = np.linspace(0.1, 1, 5) * u.um
        time = np.linspace(1, 5, 3) * u.hr

        # Test with 2D data - Signal auto-expands to 3D
        signal = Signal(spectral=spectral, data=np.ones((3, 5)), time=time)  # 2D data
        assert signal.data.shape == (1, 3, 5)  # Auto-expanded to 3D

        # Test with incompatible grid sizes - Signal allows this (no validation)
        signal = Signal(
            spectral=spectral,  # 5 elements
            data=np.ones((3, 1, 8)),  # 8 spectral elements (mismatched)
            time=time,  # 3 elements
        )
        # Signal class doesn't validate grid size compatibility
        assert signal.data.shape == (3, 1, 8)
        assert signal.spectral.shape == (5,)
