"""
Unit tests for ExoSim utility modules.

This module tests various utility functions including:
- find_klass_in_file: Dynamic class loading from files
- load_klass: Class loading utilities
- find_task: Task class discovery
- find_and_run_task: Task instantiation
"""

import sys

import pytest

from exosim.utils.klass_factory import find_and_run_task, find_task


class TestFindTask:
    """Test find_task function for task discovery."""

    def test_find_task_by_name(self):
        """Test find_task with task name."""
        try:
            from exosim.tasks.task import Task

            # Test finding base task class by name
            result_class = find_task("Task", Task)
            assert result_class is Task

        except ImportError:
            pytest.skip("ExoSim Task class not available")

    def test_find_task_with_class(self):
        """Test find_task with actual class object."""
        try:
            from exosim.tasks.task import Task

            # Test with class object directly
            result_class = find_task(Task, Task)
            assert result_class is Task

        except ImportError:
            pytest.skip("ExoSim Task class not available")

    def test_find_task_invalid_input(self):
        """Test find_task with invalid input."""
        try:
            from exosim.tasks.task import Task

            # Test with invalid input type
            with pytest.raises(TypeError):
                find_task(123, Task)

        except ImportError:
            pytest.skip("ExoSim Task class not available")

    def test_find_task_nonexistent_class(self):
        """Test find_task with nonexistent class name."""
        try:
            from exosim.tasks.task import Task

            # Test with nonexistent task class name
            with pytest.raises(TypeError, match=r"not found in exosim.tasks"):
                find_task("NonexistentTaskClass", Task)

        except ImportError:
            pytest.skip("ExoSim Task class not available")


class TestFindAndRunTask:
    """Test find_and_run_task function for task instantiation."""

    def test_find_and_run_task_basic(self):
        """Test basic find_and_run_task functionality."""
        try:
            from exosim.tasks.task import Task

            parameters = {"task": "Task"}

            # Should return an instance of the Task class
            result_instance = find_and_run_task(parameters, "task", Task)
            assert isinstance(result_instance, Task)

        except ImportError:
            pytest.skip("ExoSim Task class not available")

    def test_find_and_run_task_default_class(self):
        """Test find_and_run_task with default class when key missing."""
        try:
            from exosim.tasks.task import Task

            parameters = {}  # No task key

            # Should use the default base class
            result_instance = find_and_run_task(parameters, "missing_key", Task)
            assert isinstance(result_instance, Task)

        except ImportError:
            pytest.skip("ExoSim Task class not available")

    def test_find_and_run_task_invalid_parameters(self):
        """Test find_and_run_task with invalid parameters."""
        try:
            from exosim.tasks.task import Task

            parameters = {"task": "InvalidTaskClass"}

            # Should raise TypeError for invalid task class
            with pytest.raises(TypeError, match=r"not found in exosim.tasks"):
                find_and_run_task(parameters, "task", Task)

        except ImportError:
            pytest.skip("ExoSim Task class not available")


class TestUtilsImports:
    """Test utility module imports and availability."""

    def test_essential_imports_available(self):
        """Test that essential utility imports are available."""
        try:
            from exosim.utils import checks

            assert hasattr(checks, "check_units")
        except ImportError:
            pytest.skip("ExoSim utils.checks not available")

    def test_klass_factory_imports(self):
        """Test klass_factory module imports."""
        try:
            from exosim.utils.klass_factory import find_and_run_task, find_task

            assert callable(find_task)
            assert callable(find_and_run_task)
        except ImportError:
            pytest.skip("ExoSim klass_factory not available")

    def test_task_module_availability(self):
        """Test that task modules are available."""
        try:
            from exosim.tasks.task import Task

            assert Task is not None
        except ImportError:
            pytest.skip("ExoSim task modules not available")


class TestUtilsErrorHandling:
    """Test error handling in utility functions."""

    def test_find_task_error_messages(self):
        """Test that find_task provides helpful error messages."""
        try:
            from exosim.tasks.task import Task

            # Test with invalid class name
            with pytest.raises(TypeError) as exc_info:
                find_task("CompletelyInvalidClassName", Task)

            # Error message should be informative
            error_msg = str(exc_info.value).lower()
            assert "not found" in error_msg or "invalid" in error_msg

        except ImportError:
            pytest.skip("ExoSim Task class not available")

    def test_graceful_degradation(self):
        """Test that utilities degrade gracefully on errors."""
        try:
            from exosim.tasks.task import Task

            # Test with various invalid inputs
            with pytest.raises(TypeError):
                find_task(None, Task)

        except ImportError:
            pytest.skip("ExoSim Task class not available")


class TestUtilsCompatibility:
    """Test utility function compatibility across Python versions."""

    def test_python_version_compatibility(self):
        """Test that utilities work with current Python version."""
        # Basic functionality should work in Python 3.12+
        assert sys.version_info >= (3, 12), "ExoSim requires Python 3.12+"

        # Test basic import functionality
        try:
            from exosim.utils.klass_factory import find_task

            assert callable(find_task)
        except ImportError:
            pytest.skip("ExoSim klass_factory not available")

    def test_module_loading_mechanism(self):
        """Test the underlying module loading mechanism."""
        # Test that importlib works as expected
        import importlib

        # Test dynamic import of standard library
        collections_module = importlib.import_module("collections")
        assert hasattr(collections_module, "OrderedDict")

        # Test getattr functionality
        OrderedDict = collections_module.OrderedDict
        from collections import OrderedDict as ExpectedOrderedDict

        assert OrderedDict is ExpectedOrderedDict
