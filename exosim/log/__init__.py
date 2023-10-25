# This code is inspired by the code devolped for TauREx 3.1.
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

from .logger import generate_logger_name
from .logger import Logger
from .logger import traced
from exosim import __pkg_name__

# these are imported here because they are to be imported from the logger module

last_log = logging.INFO


# producing TRACE log

logging.TRACE = 15
logging.addLevelName(logging.TRACE, "TRACE")


def trace(self, message: str, *args, **kws) -> None:
    """
    Log TRACE level.
    Trace level log should be produced anytime a function or a method is entered and exited.
    """
    if self.isEnabledFor(logging.TRACE):
        # Yes, logger takes its '*args' as 'args'.
        self._log(logging.TRACE, message, args, **kws)


logging.Logger.trace = trace

logging.ANNOUNCE = 25
logging.addLevelName(logging.ANNOUNCE, "ANNOUNCE")

# producing ANNOUNCE log


def announce(self, message, *args, **kws):
    """
    Log ANNOUNCE level.
    This level log should be produced for huge announcements, as the starting of a long process.
    """
    if self.isEnabledFor(logging.ANNOUNCE):
        # Yes, logger takes its '*args' as 'args'.
        self._log(logging.ANNOUNCE, message, args, **kws)


logging.Logger.announce = announce

# producing GRAPHICS log

logging.GRAPHICS = 24
logging.addLevelName(logging.GRAPHICS, "GRAPHICS")


def graphics(self, message, *args, **kws):
    """
    Log GRAPHICS level.
    This level log should be produced for graphical reasons only."""
    if self.isEnabledFor(logging.GRAPHICS):
        # Yes, logger takes its '*args' as 'args'.
        self._log(logging.GRAPHICS, message, args, **kws)


logging.Logger.graphics = graphics


def setLogLevel(level: int, log_id: int = 0) -> None:
    """
    Simple function to set the logger level

    Parameters
    ----------
    level: logging level
    log_id: int
        this is the index of the handler to edit. The basic handler index is 0.
        Every added handler is appended to the list. Default is 0.

    """
    global last_log
    from .logger import root_logger

    root_logger.handlers[log_id].setLevel(level)
    last_log = level


def disableLogging(log_id: int = 0) -> None:
    """
    It disables the logging setting the log level to ERROR.

    Parameters
    ----------
    log_id: int
        this is the index of the handler to edit. The basic handler index is 0.
        Every added handler is appended to the list. Default is 0.

    """
    setLogLevel(logging.ERROR, log_id)


def enableLogging(level: int = logging.INFO, log_id: int = 0) -> None:
    """
    It disables the logging setting the log level to ERROR.

    Parameters
    ----------
    level: logging level
        Default is logging.INFO.
    log_id: int
        this is the index of the handler to edit. The basic handler index is 0.
        Every added handler is appended to the list. Default is 0.

    """
    global last_log
    if last_log is None:
        last_log = level
    setLogLevel(level, log_id)


def addHandler(handler: logging.Handler) -> None:
    """
    It adds a handler to the logging handlers list.

    Parameters
    ----------
    handler: logging handler

    """
    from .logger import root_logger

    root_logger.addHandler(handler)


def addLogFile(
    fname: str = "{}.log".format(__pkg_name__),
    reset: bool = False,
    level: int = logging.DEBUG,
) -> None:
    """
    It adds a log file to the handlers list.

    Parameters
    ----------
    fname: str
        name for the log file. Default is exosim.log.
    reset: bool
        it reset the log file if it exists already. Default is False.
    level: logging level
        Default is logging.INFO.
    """
    if reset:
        import os

        try:
            os.remove(fname)
        except OSError:
            pass
    file_handler = logging.FileHandler(fname)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    addHandler(file_handler)
