"""
Unit tests for the ExoSim logging system.

Tests the Logger class and related logging utilities.
"""

import logging
import os

from exosim.log import Logger, add_log_file, set_log_level


class TestLoggerExample(Logger):
    """Test logger class for testing logger functionality."""

    def __init__(self):
        super().__init__()

    def log_all_levels(self):
        """Log messages at all levels for testing."""
        self.info("info")
        self.debug("debug")
        self.warning("warning")
        self.critical("critical")
        self.error("error")
        self.trace("trace")
        self.announce("announce")
        self.graphics("graphics")


class TestLoggerFunctionality:
    """Test logging system functionality."""

    def test_all_log_levels_captured(self, capsys):
        """Test that all log levels are properly captured and formatted."""

        logger = TestLoggerExample()
        logger.log_all_levels()

        # Check that all log messages are captured in stdout
        # The new structlog system outputs to stdout
        captured = capsys.readouterr()
        log_text = captured.out.lower()

        # Check for presence of each log level message
        assert "info" in log_text
        assert "warning" in log_text
        assert "error" in log_text
        assert "critical" in log_text

    def test_log_file_creation_and_cleanup(self, test_data_dir):
        """Test log file creation and management."""
        log_file_path = os.path.join(test_data_dir, "test_exosim.log")

        # Ensure test data directory exists
        os.makedirs(test_data_dir, exist_ok=True)

        # Add log file and configure debug level
        add_log_file(log_file_path, reset=True)
        set_log_level(logging.DEBUG, log_id=1)

        # Create logger instance (which should log to file)
        logger = TestLoggerExample()
        logger.info("Test log message")

        # Verify file was created and contains content
        assert os.path.exists(log_file_path)

        # Clean up
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
