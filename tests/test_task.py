import pytest

from exosim.tasks.task import Task


class ExampleEmpty(Task):
    pass


class ExampleInput(Task):
    def __init__(self):
        self.add_task_param("foo", "foo")
        self.add_task_param("bar", "bar", None)


def test_missing_init():
    example = ExampleEmpty()
    assert example is not None  # Verifica placeholder


def test_missing_par():
    example = ExampleInput()
    with pytest.raises(ValueError):
        example(foo=1, test=0)


def test_missing_exe():
    example = ExampleInput()
    example(foo=1, bar=0)
