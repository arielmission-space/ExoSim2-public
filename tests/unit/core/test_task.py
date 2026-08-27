"""
Unit tests for the base Task class functionality.

Tests core Task behavior including parameter handling and execution.
"""

import pytest

from exosim.tasks.task import Task


class ExampleEmptyTask(Task):
    """Test task with no initialization."""


class ExampleParameterizedTask(Task):
    """Test task with parameters."""

    def __init__(self):
        self.add_task_param("foo", "Required foo parameter")
        self.add_task_param("bar", "Optional bar parameter", None)


class TestTaskCreation:
    """Test Task instantiation and basic functionality."""

    def test_empty_task_creation(self):
        """Test that empty task can be created successfully."""
        task = ExampleEmptyTask()
        assert task is not None
        assert isinstance(task, Task)

    def test_parameterized_task_creation(self):
        """Test that parameterized task can be created successfully."""
        task = ExampleParameterizedTask()
        assert task is not None
        assert isinstance(task, Task)


class TestTaskParameters:
    """Test Task parameter handling."""

    def test_missing_required_parameter_raises_error(self):
        """Test that missing required parameter raises ValueError."""
        task = ExampleParameterizedTask()
        with pytest.raises(ValueError, match="invalid_param is not a valid parameter"):
            task(
                foo=1, invalid_param=0
            )  # Missing 'bar' would be ok, but 'invalid_param' is not

    def test_valid_parameters_accepted(self):
        """Test that valid parameters are accepted."""
        task = ExampleParameterizedTask()
        # This should not raise an exception
        task(foo=1, bar=0)

    def test_optional_parameter_default(self):
        """Test that optional parameters use defaults."""
        task = ExampleParameterizedTask()
        # Should work with only required parameter
        task(foo=1)


class TestTaskExecution:
    """Test Task execution behavior."""

    def test_task_execution_without_implementation(self):
        """Test that task execution works even without execute() method."""
        task = ExampleParameterizedTask()
        # This should complete without error (base execute() is called)
        result = task(foo=1, bar=0)
        # Base Task.execute() returns None by default
        assert result is None
