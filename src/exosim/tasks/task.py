from abc import ABCMeta, abstractmethod
from typing import Any

from exosim.utils.timed_class import TimedClass


class Task(TimedClass):
    """
    *Abstract class*

    Base class for tasks.
    """

    __metaclass__ = ABCMeta

    _output = None
    _task_input = None
    _task_params = None

    def __init__(self):
        """
        Class initialisation, needed to prepare the task inputs reader
        """
        super().__init__()
        self._output = None
        self._task_input = None
        self._task_params = None

    @abstractmethod
    def execute(self) -> None:
        """
        Class execution. It runs on call and executes all the task actions returning the outputs.
        It requires the input with correct keywords
        """

    def __call__(self, **kwargs):
        self.set_log_name()
        self._task_input = kwargs
        self._validate_input_params()
        self._populate_empty_param()
        self.trace("called")
        self.execute()
        self.trace("exited")
        return self.get_output()

    def _validate_input_params(self) -> None:
        for key in self._task_input:
            if key not in self._task_params:
                self.error(f"Unexpected Task input parameter: {key}")
                raise ValueError(f"{key} is not a valid parameter")

    def _populate_empty_param(self) -> None:
        for key in self._task_params:
            if key not in self._task_input:
                self._task_input[key] = self._task_params[key]["default"]

    def get_output(self) -> Any:
        """Returns the output values."""
        return self._output

    def set_output(self, product: Any) -> None:
        """It sets the values to return."""
        self._output = product

    def get_task_param(self, paramName: str) -> Any:
        """It get the value from the task parameter."""
        return self._task_input[paramName]

    def add_task_param(
        self, param_name: str, param_description: str, default: Any = None
    ) -> None:
        """It adds a parameter for the task."""
        if self._task_params is None:
            self._task_params = {}
        self._task_params[param_name] = {
            "parameters": param_description,
            "default": default,
        }
