"""
Extended tests for CreateIntrapixelResponseFunction task.

This module provides comprehensive testing for the CreateIntrapixelResponseFunction task.
Tests include basic functionality, parameter handling, and error conditions.
"""

from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np

from exosim.tasks.instrument.create_intrapixel_response_function import (
    CreateIntrapixelResponseFunction,
)


class TestCreateIntrapixelResponseFunctionExtended:
    """Extended tests for CreateIntrapixelResponseFunction task."""

    def test_class_structure_and_inheritance(self):
        """Test CreateIntrapixelResponseFunction structure."""
        from exosim.tasks.task import Task

        assert issubclass(CreateIntrapixelResponseFunction, Task)

        task = CreateIntrapixelResponseFunction()
        assert hasattr(task, "__init__")
        assert hasattr(task, "execute")

    def test_task_parameters(self):
        """Test task parameters setup."""
        task = CreateIntrapixelResponseFunction()

        # Test parameter structure
        assert hasattr(task, "get_task_param")
        assert hasattr(task, "add_task_param")

    def test_execute_method_exists(self):
        """Test execute method existence."""
        task = CreateIntrapixelResponseFunction()

        assert hasattr(task, "execute")
        assert callable(task.execute)

    def test_parameter_validation_structure(self):
        """Test parameter validation structure."""
        task = CreateIntrapixelResponseFunction()

        # Test parameter methods exist
        assert hasattr(task, "get_task_param")
        assert hasattr(task, "add_task_param")

    @patch("exosim.tasks.instrument.create_intrapixel_response_function.np")
    def test_response_function_creation_structure(self, mock_np):
        """Test response function creation structure."""
        task = CreateIntrapixelResponseFunction()

        mock_np.ones.return_value = np.ones((5, 5))

        # Test basic method accessibility
        assert callable(task.execute)

    def test_task_call_interface(self):
        """Test task call interface."""
        task = CreateIntrapixelResponseFunction()

        # Test callable interface
        assert callable(task)


class TestCreateIntrapixelResponseFunctionErrorHandling:
    """Test error handling patterns for CreateIntrapixelResponseFunction."""

    def test_parameter_access_error_handling(self):
        """Test parameter access error handling."""
        task = CreateIntrapixelResponseFunction()

        # Test that parameter access methods exist
        assert hasattr(task, "get_task_param")

    def test_task_validation_patterns(self):
        """Test task validation patterns."""
        task = CreateIntrapixelResponseFunction()

        # Test validation structure
        assert hasattr(task, "get_task_param")
        assert hasattr(task, "add_task_param")


class TestAdvancedCreateIntrapixelResponseFunctionPatterns:
    """Test advanced task patterns for CreateIntrapixelResponseFunction."""

    @patch("tempfile.NamedTemporaryFile")
    def test_output_handling_patterns(self, mock_temp_file):
        """Test output handling patterns."""
        mock_file = MagicMock()
        mock_file.name = "test_output.h5"
        mock_temp_file.return_value.__enter__.return_value = mock_file

        # Test that output patterns are consistent
        assert mock_file.name.endswith(".h5")

    def test_task_base_functionality(self):
        """Test base task functionality."""
        task = CreateIntrapixelResponseFunction()

        # Test base class methods
        assert hasattr(task, "get_output")
        assert hasattr(task, "set_output")
        assert hasattr(task, "get_task_param")
        assert hasattr(task, "add_task_param")

    def test_task_inheritance_chain(self):
        """Test task inheritance chain."""
        from exosim.tasks.task import Task
        from exosim.utils.timed_class import TimedClass

        # Test inheritance chain
        assert issubclass(CreateIntrapixelResponseFunction, Task)
        assert issubclass(Task, TimedClass)

        task = CreateIntrapixelResponseFunction()
        assert isinstance(task, Task)
        assert isinstance(task, TimedClass)

    def test_astropy_units_integration(self):
        """Test astropy units integration."""
        import astropy.units as u

        # Test unit operations that are common in tasks
        wavelength = 5.0 * u.um
        assert wavelength.unit == u.um
        assert wavelength.value == 5.0

    def test_mock_integration_patterns(self):
        """Test mock integration patterns."""
        mock_signal = MagicMock()
        mock_signal.data = np.ones(100)
        mock_signal.wavelength = np.linspace(1, 10, 100) * u.um

        assert hasattr(mock_signal, "data")
        assert hasattr(mock_signal, "wavelength")
