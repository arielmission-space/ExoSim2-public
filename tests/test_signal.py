import astropy.units as u
import numpy as np
import pytest

from exosim.models.signal import Adu, Counts, CountsPerSecond, Radiance, Sed, Signal


# Common fixtures
@pytest.fixture
def spectral_grid():
    """Fixture for creating a spectral grid."""
    return np.linspace(0.1, 1, 10) * u.um


@pytest.fixture
def time_grid():
    """Fixture for creating a time grid."""
    return np.linspace(1, 5, 10) * u.hr


@pytest.fixture
def signal_data():
    """Fixture for creating random signal data."""
    return np.random.random_sample((10, 1, 10))


# Test signal initialization
def test_signal_initialization(spectral_grid, time_grid, signal_data):
    """Test the initialization of the Signal class."""
    signal = Signal(spectral=spectral_grid, data=signal_data, time=time_grid)
    assert list(signal.spectral) == list(spectral_grid.value)
    assert signal.spectral_units == u.um
    assert list(signal.time) == list(time_grid.value)
    assert signal.time_units == u.hr


# Test automatic unit conversion
@pytest.mark.parametrize(
    "data_units, expected_units",
    [
        (u.ct / u.s, u.ct / u.s),
        (u.W / u.m**2 / u.um, u.W / u.m**2 / u.um),
    ],
)
def test_signal_units_conversion(spectral_grid, time_grid, signal_data, data_units, expected_units):
    """Test automatic conversion of data units."""
    signal = Signal(
        spectral=spectral_grid, data=signal_data * data_units, time=time_grid
    )
    assert signal.data_units == expected_units


# Test addition and subtraction operations
@pytest.mark.parametrize(
    "operation, expected_result",
    [
        (lambda s1, s2: s1 + s2, 3),
        (lambda s1, s2: s1 - s2, -1),
    ],
)
def test_signal_add_sub(spectral_grid, time_grid, signal_data, operation, expected_result):
    """Test addition and subtraction operations on signals."""
    s1 = Signal(spectral=spectral_grid, data=signal_data, time=time_grid)
    s2 = Signal(spectral=spectral_grid, data=signal_data * 2, time=time_grid)
    result = operation(s1, s2)

    # Debug information
    print("Operation: Addition/Subtraction")
    print("Signal 1 data:", s1.data)
    print("Signal 2 data:", s2.data)
    print("Result data:", result.data)
    print("Expected data:", signal_data * expected_result)

    # Validation
    assert np.allclose(result.data, signal_data * expected_result), "Addition/Subtraction failed!"


# Test multiplication and division operations
@pytest.mark.parametrize(
    "operation, expected_result_fn",
    [
        (lambda s1, s2: s1 * s2, lambda data: 2 * data**2),
        (lambda s1, s2: s1 / s2, lambda data: np.full(data.shape, 0.5)),
    ],
)
def test_signal_mul_div(spectral_grid, time_grid, signal_data, operation, expected_result_fn):
    """Test multiplication and division operations on signals."""
    s1 = Signal(spectral=spectral_grid, data=signal_data, time=time_grid)
    s2 = Signal(spectral=spectral_grid, data=signal_data * 2, time=time_grid)
    result = operation(s1, s2)

    # Calculate the expected result
    expected_data = expected_result_fn(signal_data)

    # Debug information
    print("Operation: Multiplication/Division")
    print("Signal 1 data:", s1.data)
    print("Signal 2 data:", s2.data)
    print("Result data:", result.data)
    print("Expected data:", expected_data)

    # Validation
    assert np.allclose(result.data, expected_data, atol=1e-6), "Multiplication/Division failed!"


# Test operations with units
@pytest.mark.parametrize(
    "data_units1, data_units2, expected_units",
    [
        (u.m, u.m, u.m**2),
        (u.ct / u.s, u.s, u.ct),
        (u.W / u.m**2, u.um, u.W / u.m**2 * u.um),
    ],
)
def test_signal_units_operations(
    spectral_grid, time_grid, signal_data, data_units1, data_units2, expected_units
):
    """Test operations on signals with different units."""
    # Create two signals with different units
    s1 = Signal(spectral=spectral_grid, data=signal_data * data_units1, time=time_grid)
    s2 = Signal(spectral=spectral_grid, data=signal_data * data_units2, time=time_grid)

    # Perform multiplication
    result = s1 * s2

    # Validate the resulting units
    assert result.data_units == expected_units


# Test temporal slicing
def test_signal_slice(spectral_grid, time_grid, signal_data):
    """Test slicing the signal over a time interval."""
    signal = Signal(spectral=spectral_grid, data=signal_data, time=time_grid)
    sliced = signal.get_slice(1 * u.hr, 2 * u.hr)
    assert sliced.shape[0] == 2


# Test writing and reading signals to/from HDF5
@pytest.mark.parametrize(
    "signal_class, expected_units",
    [
        (Signal, u.dimensionless_unscaled),
        (Counts, u.ct),
        (CountsPerSecond, u.ct / u.s),
        (Sed, u.W / u.m**2 / u.um),
        (Radiance, u.W / u.m**2 / u.um / u.sr),
        (Adu, u.adu),
    ],
)
def test_signal_write_read(tmp_path, spectral_grid, time_grid, signal_data, signal_class, expected_units):
    """Test writing and reading signals to and from HDF5 files."""
    import os

    import h5py

    from exosim.output.hdf5.hdf5 import HDF5Output
    from exosim.output.hdf5.utils import load_signal

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
    print(loaded_signal.data_units, expected_units)
    assert loaded_signal.data_units == expected_units

def test_normalize_units():
    """Test the _normalize_units function."""
    assert Signal._normalize_units(u.m) == u.m
    assert Signal._normalize_units(u.cm) == u.cm
    assert Signal._normalize_units(u.J) == u.J
    assert Signal._normalize_units(u.erg) == u.erg
    assert Signal._normalize_units(u.Unit("W / (m**2 um)")) == u.W / u.m**2 / u.um
    assert Signal._normalize_units(u.Unit("ct / s")) == u.ct / u.s
    assert Signal._normalize_units(u.Unit("")) == u.dimensionless_unscaled
