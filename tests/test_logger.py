import logging
import os
import unittest

from inputs import test_dir

from exosim.log import addLogFile
from exosim.log import Logger
from exosim.log import setLogLevel


class LoggerExample(Logger):
    def __init__(self):
        super().__init__()
        self.info("info")
        self.debug("debug")
        self.warning("warning")
        self.critical("critical")
        self.error("error")
        self.trace("trace")
        self.announce("announce")
        self.graphics("graphics")


class LogTest(unittest.TestCase):
    def test_logs_messages(self):
        with self.assertLogs("exosim", level="DEBUG") as cm:
            LoggerExample()
            self.assertIn("INFO:exosim.LoggerExample:info", cm.output)
            self.assertIn("DEBUG:exosim.LoggerExample:debug", cm.output)
            self.assertIn("TRACE:exosim.LoggerExample:trace", cm.output)
            self.assertIn("ANNOUNCE:exosim.LoggerExample:announce", cm.output)
            self.assertIn("GRAPHICS:exosim.LoggerExample:graphics", cm.output)
            self.assertIn("WARNING:exosim.LoggerExample:warning", cm.output)
            self.assertIn("CRITICAL:exosim.LoggerExample:critical", cm.output)
            self.assertIn("ERROR:exosim.LoggerExample:error", cm.output)

    def test_log_file(self):
        fname = os.path.join(test_dir, "exosim.log")
        addLogFile(fname, reset=True)
        setLogLevel(logging.DEBUG, log_id=1)
        LoggerExample()
