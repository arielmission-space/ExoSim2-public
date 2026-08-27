"""
Unit tests for base Task class functionality.

This module contains tests for the core Task class behavior,
including parameter handling, initialization, and error conditions.
"""

import pytest

from exosim.tasks.task import Task


class ExampleEmptyTask(Task):
    """Example empty task for testing basic Task functionality."""


class ExampleInputTask(Task):
    """Example task with input parameters for testing parameter handling."""

    def __init__(self):
        """
        Initialize the task with required and optional parameters.

        This setup tests parameter registration functionality
        of the base Task class.
        """
        self.add_task_param("foo", "foo")
        self.add_task_param("bar", "bar", None)


class TestBaseTask:
    """Test suite for base Task class functionality."""

    def test_empty_task_initialization(self):
        """
        Test initialization of an empty task.

        This test verifies that tasks without custom initialization
        can be created successfully without errors.
        """
        example = ExampleEmptyTask()
        assert example is not None

    def test_missing_required_parameter(self):
        """
        Test error handling for missing required parameters.

        This test verifies that tasks raise ValueError when called
        with unexpected parameters instead of required ones.
        """
        example = ExampleInputTask()
        with pytest.raises(ValueError, match="test is not a valid parameter"):
            example(foo=1, test=0)

    def test_task_execution_with_parameters(self):
        """
        Test task execution with both required and optional parameters.

        This test verifies that tasks can be called with proper
        parameters without raising errors, even when no execute
        method is implemented.
        """
        example = ExampleInputTask()
        # This should not raise an error for parameter validation
        example(foo=1, bar=0)
