"""
ExoSim 2.0 Modern Structured Logging

This module provides structured logging capabilities for ExoSim 2.0 using structlog.

Key Features:
- Structured JSON logs for production environments
- Human-readable colored logs for development
- Automatic context binding (class names, execution time, metadata)
- Performance metrics and operation timing
- Environment-based configuration

Usage:
    class MyTask(Task):
        def execute(self):
            # Context binding
            logger = self.bind(instrument="ARIEL", simulation_id="sim_001")
            logger.info("Simulation started")

            # Performance timing
            with self.time_operation("focal_plane_generation", channels=4):
                # ... simulation logic ...
                pass

Environment Variables:
    EXOSIM_LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    EXOSIM_DEV_MODE: Enable development mode with colors (true/false)
    EXOSIM_JSON_LOGS: Enable JSON structured output (true/false)
    EXOSIM_LOG_FILE: Path to log file for persistent logging
"""

import logging
import os
from pathlib import Path

__pkg_name__ = "exosim"

# Import modern structured logging implementation
from .logger import Logger
from .structlog_config import (
    ExoSimLogger,
    configure_structlog,
    disable_logging,
    enable_logging,
    with_logger,
)

# Auto-configure structured logging based on environment
_log_level = os.environ.get("EXOSIM_LOG_LEVEL", "INFO")
_development = os.environ.get("EXOSIM_DEV_MODE", "true").lower() == "true"
_json_logs = os.environ.get("EXOSIM_JSON_LOGS", "false").lower() == "true"
_log_file = os.environ.get("EXOSIM_LOG_FILE")

configure_structlog(
    log_level=_log_level,
    log_file=_log_file,
    development=_development,
    json_logs=_json_logs,
)


def configure_logging(
    log_level: str = "INFO",
    log_file: str | Path | None = None,
    json_logs: bool = False,
    development: bool = True,
) -> None:
    """
    Configure ExoSim logging with modern structured capabilities.

    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Optional[Union[str, Path]]
        Path to log file for persistent logging
    json_logs : bool
        Enable JSON structured log output
    development : bool
        Enable development-friendly formatting with colors

    Examples
    --------
    # Development setup with colors
    configure_logging("DEBUG", development=True)

    # Production setup with JSON logs
    configure_logging("INFO", "exosim.log", json_logs=True, development=False)
    """
    configure_structlog(
        log_level=log_level,
        log_file=log_file,
        json_logs=json_logs,
        development=development,
    )


def set_log_level(
    level: str | int,
    log_id: int = 0,
    logger_prefixes: list[str] | None = None,
) -> None:
    """
    Set the logger level.

    Parameters
    ----------
    level : Union[str, int]
        Logging level ("DEBUG", "INFO", etc. or numeric value)
    log_id : int
        Handler index (ignored, for backward compatibility)
    logger_prefixes : Optional[List[str]]
        List of logger name prefixes to configure. If None, uses default ('exosim', 'arielrad').
    """
    # Import here to avoid circular imports
    from .structlog_config import _set_log_level_internal

    _set_log_level_internal(level, logger_prefixes)


def add_log_file(
    fname: str | Path,
    reset: bool = False,
    level: str | int | None = None,
) -> None:
    """
    Add a log file handler to the logging system.

    This is a backward-compatibility function that wraps the modern
    configure_logging system.

    Parameters
    ----------
    fname : Union[str, Path]
        Path to the log file
    reset : bool
        If True, remove existing file handlers before adding new one
    level : Optional[Union[str, int]]
        Log level for the file handler (if None, uses current log level)
    """
    # Import here to avoid circular imports

    if reset:
        # Remove existing file handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)
                handler.close()

    # Add new file handler using the internal function
    from .structlog_config import _configure_file_logging

    if level is None:
        # Get current log level
        root_logger = logging.getLogger()
        level = root_logger.level

    _configure_file_logging(fname, json_logs=False, level=level)


# Export all public functions and classes (sorted alphabetically)
__all__ = [
    "ExoSimLogger",
    "Logger",
    "add_log_file",
    "configure_logging",
    "disable_logging",
    "enable_logging",
    "set_log_level",
    "with_logger",
]
