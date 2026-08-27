"""
Unit tests for the ChainTask abstract class.

Tests the ChainTask functionality by creating concrete implementations
and testing parameter handling, execution flow, and error conditions.
"""

import astropy.units as u
import numpy as np
import pytest

from exosim.models.signal import CountsPerSecond
from exosim.tasks.chain_task import ChainTask


class MockChainTask(ChainTask):
    """Mock implementation of ChainTask for testing purposes."""

    def __init__(self):
        """Initialize the mock chain task."""
        super().__init__()

    def model(self, signal, parameters, wavelength, time):
        """
        Mock model implementation.

        Applies a multiplier to the input signal data.
        """
        # Get multiplier from parameters, default to 1.0
        multiplier = parameters.get("multiplier", 1.0)
        output_data = signal.data * multiplier

        return CountsPerSecond(
            spectral=signal.spectral,
            data=output_data,
            time=signal.time if hasattr(signal, "time") else time,
        )


class FailingChainTask(ChainTask):
    """ChainTask that returns invalid output for error testing."""

    def __init__(self):
        super().__init__()

    def model(self, signal, parameters, wavelength, time):
        """Return invalid output (not a Signal) to trigger error."""
        return "not_a_signal_object"


class TestChainTaskInitialization:
    """Test ChainTask initialization and setup."""

    def test_chain_task_initialization(self):
        """Test ChainTask proper initialization."""
        task = MockChainTask()

        # Check that required task parameters are set up
        assert task._task_params is not None
        assert "signal" in task._task_params
        assert "wavelength" in task._task_params
        assert "time" in task._task_params
        assert "parameters" in task._task_params

    def test_task_parameter_structure(self):
        """Test that task parameters have proper structure."""
        task = MockChainTask()

        # Verify parameter structure and defaults
        for param_name in ["signal", "wavelength", "parameters"]:
            assert param_name in task._task_params

        # Time parameter should have default value of None
        assert "time" in task._task_params


class TestChainTaskExecution:
    """Test ChainTask execution behavior."""

    def test_successful_task_execution(self):
        """Test successful execution with valid inputs."""
        task = MockChainTask()

        # Create test signal data
        spectral = np.linspace(1, 10, 10) * u.um
        data = np.ones((1, 5, 10)) * u.ct / u.s  # 3D: (time, spatial, spectral)
        time = np.arange(5) * u.s

        input_signal = CountsPerSecond(spectral=spectral, data=data, time=time)
        parameters = {"multiplier": 2.0}

        # Execute the task
        result = task(
            signal=input_signal, parameters=parameters, wavelength=spectral, time=time
        )

        # Verify result
        assert isinstance(result, CountsPerSecond)
        expected_data = data.value * 2.0 if hasattr(data, "value") else data * 2.0
        result_data = (
            result.data.value if hasattr(result.data, "value") else result.data
        )
        np.testing.assert_array_equal(result_data, expected_data)

    def test_execution_with_default_parameters(self):
        """Test execution with default parameter values."""
        task = MockChainTask()

        # Create minimal test data with correct 3D shape: (time, spatial, spectral)
        spectral = np.linspace(1, 10, 5) * u.um
        data = np.ones((3, 1, 5)) * u.ct / u.s  # Added spatial dimension
        time = np.arange(3) * u.s

        input_signal = CountsPerSecond(spectral=spectral, data=data, time=time)

        # Execute with minimal parameters (using defaults)
        result = task(
            signal=input_signal,
            parameters={},  # Empty parameters dict
            wavelength=spectral,
            # time parameter omitted - should use default
        )

        # Should work with defaults
        assert isinstance(result, CountsPerSecond)
        # With multiplier defaulting to 1.0, data should be unchanged
        expected_data = data.value if hasattr(data, "value") else data
        result_data = (
            result.data.value if hasattr(result.data, "value") else result.data
        )
        np.testing.assert_array_equal(result_data, expected_data)


class TestChainTaskParameterHandling:
    """Test parameter validation and handling."""

    def test_parameter_validation_with_invalid_params(self):
        """Test parameter validation rejects unexpected parameters."""
        task = MockChainTask()

        # Create valid inputs
        spectral = np.linspace(1, 10, 5) * u.um
        data = np.ones((3, 5)) * u.ct / u.s
        time = np.arange(3) * u.s

        input_signal = CountsPerSecond(spectral=spectral, data=data, time=time)

        # Test with unexpected parameter - should raise ValueError
        with pytest.raises(
            ValueError, match="unexpected_parameter is not a valid parameter"
        ):
            task(
                signal=input_signal,
                parameters={},
                wavelength=spectral,
                time=time,
                unexpected_parameter="should_fail",
            )

    def test_parameter_access_methods(self):
        """Test parameter access methods work correctly."""
        task = MockChainTask()

        # Set up test data
        spectral = np.linspace(1, 10, 5) * u.um
        data = np.ones((3, 5)) * u.ct / u.s
        input_signal = CountsPerSecond(
            spectral=spectral, data=data, time=np.arange(3) * u.s
        )
        test_params = {"test_key": "test_value", "multiplier": 3.0}

        # Set task input manually for testing parameter access
        task._task_input = {
            "signal": input_signal,
            "parameters": test_params,
            "wavelength": spectral,
            "time": np.arange(3) * u.s,
        }

        # Test parameter access
        assert task.get_task_param("signal") is input_signal
        assert task.get_task_param("parameters") == test_params
        assert np.array_equal(task.get_task_param("wavelength"), spectral)


class TestChainTaskOutputHandling:
    """Test output handling functionality."""

    def test_get_and_set_output(self):
        """Test output getter and setter methods."""
        task = MockChainTask()

        # Test initial state (should be None)
        assert task.get_output() is None

        # Test setting and getting output
        test_output = "test_result_value"
        task.set_output(test_output)
        assert task.get_output() == test_output

        # Test setting different output type
        complex_output = {"key": "value", "number": 42}
        task.set_output(complex_output)
        assert task.get_output() == complex_output


class TestChainTaskErrorHandling:
    """Test error handling in ChainTask execution."""

    def test_invalid_output_type_raises_error(self):
        """Test that invalid output type raises appropriate error."""
        task = FailingChainTask()

        # Create valid inputs
        spectral = np.linspace(1, 10, 5) * u.um
        data = np.ones((3, 5)) * u.ct / u.s
        time = np.arange(3) * u.s

        input_signal = CountsPerSecond(spectral=spectral, data=data, time=time)

        # This should raise TypeError due to invalid model output
        with pytest.raises(TypeError, match="output is not a Signal"):
            task(signal=input_signal, parameters={}, wavelength=spectral, time=time)

    def test_missing_required_signal_parameter(self):
        """Test error when required signal parameter is missing."""
        task = MockChainTask()

        # Try to execute without signal parameter (pass None)
        with pytest.raises(
            AttributeError
        ):  # Model tries to access signal.data when signal=None
            task(
                signal=None,  # Explicit None signal
                parameters={},
                wavelength=np.linspace(1, 10, 5) * u.um,
                # signal parameter missing
            )


class TestChainTaskInheritancePattern:
    """Test ChainTask inheritance and abstract method pattern."""

    def test_chain_task_is_abstract(self):
        """Test that ChainTask has abstract model method."""
        # ChainTask can be instantiated but model method raises NotImplementedError
        task = ChainTask()
        with pytest.raises(NotImplementedError):
            task.model(parameters={}, wavelength=None, time=None)

    def test_concrete_implementation_required(self):
        """Test that concrete implementations must implement model method."""
        # MockChainTask should be instantiable as it implements model
        task = MockChainTask()
        assert hasattr(task, "model")
        assert callable(task.model)
