import logging
import os
import re

import pytest

from exosim.log import Logger, addLogFile, setLogLevel


class LoggerExample(Logger):
    def __init__(self):
        super().__init__()

    def log_messages(self):
        self.info("info")
        self.debug("debug")
        self.warning("warning")
        self.critical("critical")
        self.error("error")
        self.trace("trace")
        self.announce("announce")
        self.graphics("graphics")


class TestLog:
    def test_logs_messages(self, caplog):
        import logging

        from exosim.log.logger import root_logger

        original_propagate = root_logger.propagate
        root_logger.propagate = True

        try:
            with caplog.at_level(logging.DEBUG):
                logger_example = LoggerExample()
                logger_example.log_messages()
                assert re.search(r"INFO     exosim.LoggerExample:logger.py:\d+ info", caplog.text)
                assert re.search(r"DEBUG    exosim.LoggerExample:logger.py:\d+ debug", caplog.text)
                assert re.search(r"TRACE    exosim.LoggerExample:__init__.py:\d+ trace", caplog.text)
                assert re.search(r"ANNOUNCE exosim.LoggerExample:__init__.py:\d+ announce", caplog.text)
                assert re.search(r"GRAPHICS exosim.LoggerExample:__init__.py:\d+ graphics", caplog.text)
                assert re.search(r"WARNING  exosim.LoggerExample:logger.py:\d+ warning", caplog.text)
                assert re.search(r"CRITICAL exosim.LoggerExample:logger.py:\d+ critical", caplog.text)
                assert re.search(r"ERROR    exosim.LoggerExample:logger.py:\d+ error", caplog.text)
        finally:
            root_logger.propagate = original_propagate

    def test_log_file(self, test_data_dir):
        fname = os.path.join(test_data_dir, "exosim.log")
        addLogFile(fname, reset=True)
        setLogLevel(logging.DEBUG, log_id=1)
        LoggerExample()
        os.remove(fname)
