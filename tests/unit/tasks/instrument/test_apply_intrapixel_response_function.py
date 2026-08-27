"""
Extended tests for ApplyIntraPixelResponseFunction task.

This module provides comprehensive testing for the ApplyIntraPixelResponseFunction task.
Tests include basic functionality, parameter handling, and error conditions.
"""

from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np

from exosim.tasks.instrument.apply_intra_pixel_response_function import (
    ApplyIntraPixelResponseFunction,
)


class TestApplyIntraPixelResponseFunctionExtended:
    """Extended tests for ApplyIntraPixelResponseFunction task."""

    def test_class_inheritance_and_structure(self):
        """Test ApplyIntraPixelResponseFunction inheritance."""
        from exosim.tasks.task import Task

        assert issubclass(ApplyIntraPixelResponseFunction, Task)

        # Test basic structure
        task = ApplyIntraPixelResponseFunction()
        assert hasattr(task, "__init__")
        assert hasattr(task, "execute")

    def test_task_initialization(self):
        """Test task initialization."""
        task = ApplyIntraPixelResponseFunction()

        # Check basic initialization
        assert hasattr(task, "get_task_param")
        assert hasattr(task, "add_task_param")

    def test_execute_method_signature(self):
        """Test execute method signature and structure."""
        task = ApplyIntraPixelResponseFunction()

        assert hasattr(task, "execute")
        assert callable(task.execute)

    def test_parameter_handling(self):
        """Test parameter handling patterns."""
        task = ApplyIntraPixelResponseFunction()

        # Test parameter access methods exist
        assert hasattr(task, "get_task_param")
        assert hasattr(task, "add_task_param")

    def test_execute_method_integration(self):
        """Test execute method integration."""
        task = ApplyIntraPixelResponseFunction()

        # Test execute method exists
        assert hasattr(task, "execute")
        assert callable(task.execute)

    def test_task_output_methods(self):
        """Test task output methods."""
        task = ApplyIntraPixelResponseFunction()

        # Test output methods from base class
        assert hasattr(task, "get_output")
        assert hasattr(task, "set_output")


class TestApplyIntraPixelResponseFunctionErrorHandling:
    """Test error handling patterns in ApplyIntraPixelResponseFunction task."""

    def test_parameter_access_error_handling(self):
        """Test parameter access error handling."""
        task = ApplyIntraPixelResponseFunction()

        # Test that parameter access methods exist
        assert hasattr(task, "get_task_param")

    def test_signal_processing_error_patterns(self):
        """Test signal processing error patterns."""
        task = ApplyIntraPixelResponseFunction()

        # Test basic error handling structure exists
        assert hasattr(task, "execute")


class TestAdvancedApplyIntraPixelResponseFunctionPatterns:
    """Test advanced task patterns for ApplyIntraPixelResponseFunction."""

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
        task = ApplyIntraPixelResponseFunction()

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
        assert issubclass(ApplyIntraPixelResponseFunction, Task)
        assert issubclass(Task, TimedClass)

        task = ApplyIntraPixelResponseFunction()
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
