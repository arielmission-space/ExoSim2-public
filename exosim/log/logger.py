# This code is inspired by the logger devolped for TauREx 3.1.
# Therefore, we attach here TauREx3.1 license:
#
# BSD 3-Clause License
#
# Copyright (c) 2019, Ahmed F. Al-Refaie, Quentin Changeat, Ingo Waldmann, Giovanna Tinetti
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the names of the copyright holders nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import logging
from typing import Any

from decorator import decorator

from exosim import __pkg_name__

__all__ = ["Logger"]

root_logger = logging.getLogger(__pkg_name__)
root_logger.propagate = False

# inspired by https://stackoverflow.com/a/56944256 on StackOverflow
# by sergey-pleshakov (https://stackoverflow.com/users/9150146/sergey-pleshakov)
# this is compliant to StackOverflow's CC BY-SA 3.0
# and its attribution requirement
# (https://stackoverflow.blog/2009/06/25/attribution-required/)


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    blue = "\x1b[34;20m"
    light_blue = "\x1b[36;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        10: grey + format + reset,
        15: green + format + reset,
        20: grey + format + reset,
        24: light_blue + format + reset,
        25: blue + format + reset,
        30: yellow + format + reset,
        40: red + format + reset,
        50: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
ch.setLevel(logging.DEBUG)
root_logger.addHandler(ch)
root_logger.setLevel(logging.DEBUG)


class Logger:
    """
    *Abstract class*

    Standard logging using logger library.
    It's an abtract class to be inherited to load its methods for logging.
    It define the logger name at the initialization, and then provides the logging methods.
    """

    def __init__(self):
        super().__init__()
        self.set_log_name()

    def set_log_name(self) -> None:
        """
        Produces the logger name and store it inside the class.
        The logger name is the name of the class that inherits this Logger class.
        """
        self._log_name = "{}.{}".format(__pkg_name__, self.__class__.__name__)
        self._logger = logging.getLogger(
            "{}.{}".format(__pkg_name__, self.__class__.__name__)
        )

    def announce(self, message, *args, **kwargs):
        """
        Produces ANNOUNCE level log
        See :class:`logging.Logger`
        """
        self._logger.announce(message, *args, **kwargs)

    def graphics(self, message, *args, **kwargs):
        """
        Produces INFO level log
        See :class:`logging.Logger`
        """
        self._logger.graphics(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """
        Produces INFO level log
        See :class:`logging.Logger`
        """
        self._logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs) -> None:
        """
        Produces WARNING level log
        See :class:`logging.Logger`
        """
        self._logger.warning(message, *args, **kwargs)

    def debug(self, message, *args, **kwargs) -> None:
        """
        Produces DEBUG level log
        See :class:`logging.Logger`
        """
        self._logger.debug(message, *args, **kwargs)

    def trace(self, message, *args, **kwargs) -> None:
        """
        Produces TRACE level log
        See :class:`logging.Logger`
        """
        self._logger.trace(message, *args, **kwargs)

    def error(self, message, *args, **kwargs) -> None:
        """
        Produces ERROR level log
        See :class:`logging.Logger`
        """
        self._logger.error(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs) -> None:
        """
        Produces CRITICAL level log
        See :class:`logging.Logger`
        """
        self._logger.critical(message, *args, **kwargs)


@decorator
def traced(obj: Any, *args, **kwargs) -> None:
    """
    Decorator to attach to functions and methods to log at TRACE level.
    Trace level produced a log anythime a function or a methos is entered and exited.
    """
    logger = logging.getLogger(generate_logger_name(obj))

    logger.trace("called")
    value = obj(*args, **kwargs)
    logger.trace("exited")

    return value


def generate_logger_name(obj: Any) -> str:
    """
    Given a class method or a function it returns the logger name.

    Parameters
    ----------
    obj
        class method or function

    Returns
    -------
    str
        logger name
    """
    parent_logger_name = obj.__module__
    return "{}.{}".format(
        parent_logger_name, getattr(obj, "__qualname__", obj.__name__)
    )
