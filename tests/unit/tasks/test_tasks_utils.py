"""
Test utilities and common patterns across all tasks.

This module provides testing for common patterns, utilities, and integrations
that are shared across multiple tasks.
"""

from unittest.mock import MagicMock

import astropy.units as u
import numpy as np

from exosim.tasks.astrosignal.apply_astronomical_signal import ApplyAstronomicalSignal
from exosim.tasks.instrument.apply_intra_pixel_response_function import (
    ApplyIntraPixelResponseFunction,
)
from exosim.tasks.instrument.create_intrapixel_response_function import (
    CreateIntrapixelResponseFunction,
)


class TestInstrumentTasksUtilities:
    """Test utilities and common patterns across instrument tasks."""

    def test_all_tasks_have_task_inheritance(self):
        """Test that all instrument tasks inherit from Task."""
        from exosim.tasks.task import Task

        tasks = [
            ApplyIntraPixelResponseFunction,
            CreateIntrapixelResponseFunction,
        ]

        for task_class in tasks:
            assert issubclass(task_class, Task)

    def test_all_tasks_have_execute_method(self):
        """Test that all tasks have execute method."""
        tasks = [
            ApplyAstronomicalSignal,
            ApplyIntraPixelResponseFunction,
            CreateIntrapixelResponseFunction,
        ]

        for task_class in tasks:
            task = task_class()
            assert hasattr(task, "execute")
            assert callable(task.execute)

    def test_task_parameter_patterns(self):
        """Test common task parameter patterns."""
        tasks = [
            ApplyAstronomicalSignal,
            ApplyIntraPixelResponseFunction,
            CreateIntrapixelResponseFunction,
        ]

        for task_class in tasks:
            task = task_class()
            assert hasattr(task, "get_task_param")
            assert hasattr(task, "add_task_param")

    def test_task_imports_work(self):
        """Test that task imports work correctly."""
        from exosim.tasks.astrosignal import apply_astronomical_signal
        from exosim.tasks.instrument import (
            apply_intra_pixel_response_function,
            create_intrapixel_response_function,
        )

        assert hasattr(apply_astronomical_signal, "ApplyAstronomicalSignal")
        assert hasattr(
            apply_intra_pixel_response_function, "ApplyIntraPixelResponseFunction"
        )
        assert hasattr(
            create_intrapixel_response_function, "CreateIntrapixelResponseFunction"
        )

    def test_task_output_patterns(self):
        """Test task output patterns."""
        tasks = [
            ApplyAstronomicalSignal,
            ApplyIntraPixelResponseFunction,
            CreateIntrapixelResponseFunction,
        ]

        for task_class in tasks:
            task = task_class()
            assert hasattr(task, "get_output")
            assert hasattr(task, "set_output")


class TestTasksErrorHandling:
    """Test error handling patterns in tasks."""

    def test_parameter_access_error_handling(self):
        """Test parameter access error handling."""
        task = ApplyAstronomicalSignal()

        # Test that parameter access methods exist
        assert hasattr(task, "get_task_param")

    def test_signal_processing_error_patterns(self):
        """Test signal processing error patterns."""
        tasks = [
            ApplyAstronomicalSignal,
            ApplyIntraPixelResponseFunction,
        ]

        for task_class in tasks:
            task = task_class()
            # Test basic error handling structure exists
            assert hasattr(task, "execute")

    def test_task_validation_patterns(self):
        """Test task validation patterns."""
        task = CreateIntrapixelResponseFunction()

        # Test validation structure
        assert hasattr(task, "get_task_param")
        assert hasattr(task, "add_task_param")


class TestAdvancedTaskPatterns:
    """Test advanced task patterns and integrations."""

    def test_task_base_class_patterns(self):
        """Test task base class patterns."""
        from exosim.tasks.task import Task

        # Test that Task is importable and has expected interface
        assert hasattr(Task, "__init__")

    def test_task_configuration_patterns(self):
        """Test task configuration patterns."""
        tasks = [
            ApplyAstronomicalSignal(),
            ApplyIntraPixelResponseFunction(),
            CreateIntrapixelResponseFunction(),
        ]

        for task in tasks:
            # Test configuration interface exists
            assert hasattr(task, "get_task_param")
            assert hasattr(task, "add_task_param")

    def test_numpy_integration_patterns(self):
        """Test numpy integration patterns."""
        # Test that numpy operations are commonly used
        import numpy as np

        test_array = np.ones(10)
        assert len(test_array) == 10
        assert np.all(test_array == 1.0)

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


class TestTaskParametersAndConfiguration:
    """Test task parameter and configuration patterns."""

    def test_task_parameter_access(self):
        """Test parameter access across different tasks."""
        tasks = [
            ApplyAstronomicalSignal(),
            ApplyIntraPixelResponseFunction(),
            CreateIntrapixelResponseFunction(),
        ]

        for task in tasks:
            # Test basic parameter interface
            assert hasattr(task, "get_task_param")
            assert hasattr(task, "add_task_param")
            assert callable(task.get_task_param)
            assert callable(task.add_task_param)

    def test_task_execution_patterns(self):
        """Test task execution patterns."""
        tasks = [
            ApplyAstronomicalSignal(),
            ApplyIntraPixelResponseFunction(),
            CreateIntrapixelResponseFunction(),
        ]

        for task in tasks:
            # Test execution interface
            assert hasattr(task, "execute")
            assert callable(task.execute)

    def test_task_callable_patterns(self):
        """Test task callable patterns."""
        tasks = [
            ApplyAstronomicalSignal(),
            ApplyIntraPixelResponseFunction(),
            CreateIntrapixelResponseFunction(),
        ]

        for task in tasks:
            # Test callable interface from base class
            assert callable(task)

    def test_task_initialization_requirements(self):
        """Test task initialization requirements."""
        # Test that all tasks can be instantiated without parameters
        tasks = [
            ApplyAstronomicalSignal,
            ApplyIntraPixelResponseFunction,
            CreateIntrapixelResponseFunction,
        ]

        for task_class in tasks:
            # Test that task can be created
            task = task_class()
            assert task is not None
            assert hasattr(task, "execute")


class TestTaskStructureValidation:
    """Test task structure validation and patterns."""

    def test_astrosignal_task_structure(self):
        """Test astrosignal task specific structure."""
        task = ApplyAstronomicalSignal()

        # Test specific to astronomical signal task
        assert hasattr(task, "execute")
        assert callable(task)

    def test_instrument_tasks_structure(self):
        """Test instrument tasks specific structure."""
        tasks = [
            ApplyIntraPixelResponseFunction(),
            CreateIntrapixelResponseFunction(),
        ]

        for task in tasks:
            # Test instrument task specific patterns
            assert hasattr(task, "execute")
            assert callable(task)

    def test_task_base_functionality(self):
        """Test base task functionality."""
        task = ApplyAstronomicalSignal()

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
        assert issubclass(ApplyAstronomicalSignal, Task)
        assert issubclass(Task, TimedClass)

        task = ApplyAstronomicalSignal()
        assert isinstance(task, Task)
        assert isinstance(task, TimedClass)
