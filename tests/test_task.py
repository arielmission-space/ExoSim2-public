import unittest

from exosim.tasks.task import Task


class ExampleEmpty(Task):
    pass


class ExampleInput(Task):
    def __init__(self):
        self.add_task_param("foo", "foo")
        self.add_task_param("bar", "bar", None)


class TaskTest(unittest.TestCase):
    def test_missing_init(self):
        example = ExampleEmpty()

    def test_missing_par(self):
        example = ExampleInput()
        with self.assertRaises(ValueError):
            example(foo=1, test=0)

    def test_missing_exe(self):
        example = ExampleInput()
        example(foo=1, bar=0)
