"""
Structured logging configuration using structlog.

This module provides modern structured logging capabilities for ExoSim 2.0
using structlog for better observability, structured data, and easier log analysis.

Key Features:
- Structured JSON logs for production
- Human-readable colored logs for development
- Automatic context binding (task name, execution time, etc.)
- Performance metrics and tracing
- Support for Task/Logger patterns
"""

import functools
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

# Custom log levels for ExoSim
TASK_LEVEL = 15  # Above DEBUG (10), below INFO (20) - for external tasks
GRAPHICS_LEVEL = 18  # Slightly below INFO (20)
ANNOUNCEMENT_LEVEL = 25  # Above INFO for important messages
logging.addLevelName(TASK_LEVEL, "TASK")
logging.addLevelName(GRAPHICS_LEVEL, "GRAPHICS")
logging.addLevelName(ANNOUNCEMENT_LEVEL, "ANNOUNCEMENT")

# Global variable to store active prefix filters
_active_prefix_filters = {}

# Global variable to store current logging context for task inheritance
_current_logging_context = "exosim"  # Default to exosim


def configure_structlog(
    log_level: str = "INFO",
    log_file: str | Path | None = None,
    json_logs: bool = False,
    development: bool = True,
) -> structlog.BoundLogger:
    """
    Configure structlog for ExoSim 2.0.

    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Optional[Union[str, Path]]
        Path to log file. If None, logs only to console
    json_logs : bool
        If True, output JSON structured logs. If False, human-readable format
    development : bool
        If True, use development-friendly formatting with colors

    Returns
    -------
    structlog.BoundLogger
        Configured logger instance
    """

    # Convert log level string to logging constant
    if log_level == "15":
        numeric_level = TASK_LEVEL  # Custom level between DEBUG (10) and INFO (20)
    else:
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Remove existing handlers to avoid conflicts
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Processors for all environments
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%dT%H:%M:%S", utc=True),
        # Note: CallsiteParameterAdder not included for simplicity
    ]

    # Add ExoSim-specific processors
    processors.append(_filter_by_logger_prefix)  # Add filtering first
    processors.append(_add_exosim_context)

    if development and not json_logs:
        # Development: Human-readable with colors
        # Custom colors for different log levels
        level_styles = structlog.dev.ConsoleRenderer.get_default_level_styles()
        level_styles["graphics"] = "\033[1;36m"  # Bright cyan, bold - for ASCII art
        level_styles["announcement"] = "\033[1;35m"  # Bright magenta, bold - stands out

        processors.extend(
            [
                structlog.dev.set_exc_info,
                structlog.dev.ConsoleRenderer(
                    colors=True,
                    exception_formatter=structlog.dev.better_traceback,
                    level_styles=level_styles,
                ),
            ]
        )
    else:
        # Production: JSON structured logs
        processors.extend(
            [structlog.processors.dict_tracebacks, structlog.processors.JSONRenderer()]
        )

    # Configure structlog - disable caching to allow level changes
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        cache_logger_on_first_use=False,  # Disable caching for dynamic levels
    )

    # Configure file logging if specified
    if log_file:
        _configure_file_logging(log_file, json_logs, numeric_level)

    return structlog.get_logger("exosim")


def _add_exosim_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Add ExoSim-specific context to log records.

    Extracts package name from logger name if available.
    Logger name should be in format "package.module.class" or just "package".

    Parameters
    ----------
    logger : Any
        Structlog logger instance
    method_name : str
        Name of logging method (debug, info, etc.)
    event_dict : dict[str, Any]
        Event dictionary to modify

    Returns
    -------
    dict[str, Any]
        Modified event dictionary with ExoSim context
    """
    # Extract package from logger_name if available
    logger_name = event_dict.get("logger_name", "")
    if logger_name:
        # Extract package name (first part before first dot)
        package_name = logger_name.split(".")[0]
        if package_name:
            event_dict["package"] = package_name

        # Remove logger_name from output (internal use only)
        event_dict.pop("logger_name", None)

    # Handle custom level overrides
    if event_dict.pop("_announcement", False):
        event_dict["level"] = "announcement"
    elif event_dict.pop("_graphics", False):
        event_dict["level"] = "graphics"

    return event_dict


def _filter_by_logger_prefix(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Custom processor to filter messages based on logger name prefixes.

    Uses the global _active_prefix_filters to determine which messages to pass through.
    """
    global _active_prefix_filters

    if not _active_prefix_filters:
        # No active filters, pass through all messages
        return event_dict

    # Get logger name from event
    logger_name = event_dict.get("logger_name", "")

    # Determine the actual level of the message
    # Handle custom level markers that come as flags in event_dict
    if event_dict.get("_announcement", False):
        current_level = ANNOUNCEMENT_LEVEL
    elif event_dict.get("_graphics", False):
        current_level = GRAPHICS_LEVEL
    else:
        # Use standard level mapping for regular methods
        level_mapping = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
            "trace": TASK_LEVEL,
        }
        current_level = level_mapping.get(method_name.lower(), logging.INFO)

    # Check if this logger matches any of the active filters
    for prefix, min_level in _active_prefix_filters.items():
        if (
            logger_name.startswith(prefix)
            or f".{prefix}" in logger_name
            or f"{prefix}." in logger_name
            or prefix in logger_name
        ):
            # This logger matches a filtered prefix
            if min_level == logging.ERROR:  # Special case: disable_logging
                # Block ALL messages for this prefix (it's disabled)
                raise structlog.DropEvent
            if current_level >= min_level:
                # Normal filtering: allow messages above threshold
                return event_dict
            # Block messages below threshold
            raise structlog.DropEvent

    # Logger doesn't match any filtered prefix, allow it through
    return event_dict


def _configure_file_logging(log_file: str | Path, json_logs: bool, level: int) -> None:
    """Configure file-based logging."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)

    # Configure formatter based on format preference
    if json_logs:
        # File logs are always JSON for easier parsing
        formatter = logging.Formatter("%(message)s")
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    file_handler.setFormatter(formatter)

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


class ExoSimLogger:
    """
    Modern structured logger for ExoSim 2.0.

    Provides structured logging capabilities with performance tracking
    and automatic context binding for Task classes.

    Usage:
        class MyTask(Task):
            def __init__(self):
                super().__init__()
                self.logger = ExoSimLogger.for_class(self)

            def execute(self):
                self.logger.info("Task started", task_type="simulation")
                # ... task logic ...
                self.logger.info("Task completed", duration_ms=123.45)
    """

    def __init__(self, name: str, context: dict[str, Any] | None = None):
        """
        Initialize structured logger.

        Parameters
        ----------
        name : str
            Logger name (typically class name)
        context : Optional[Dict[str, Any]]
            Initial context to bind to all log messages
        """
        self._base_logger = structlog.get_logger(name)
        self._context = context or {}
        self._start_time = time.perf_counter()

        # Store logger name for package inference
        self._logger_name = name

        # Bind initial context including logger name
        bind_context = {"logger_name": name}
        if self._context:
            bind_context.update(self._context)
        self._base_logger = self._base_logger.bind(**bind_context)

    @classmethod
    def for_class(cls, instance: Any, **extra_context) -> "ExoSimLogger":
        """
        Create logger for a class instance.

        Parameters
        ----------
        instance : Any
            Class instance to create logger for
        **extra_context
            Additional context to bind to logger

        Returns
        -------
        ExoSimLogger
            Configured logger instance
        """
        class_name = instance.__class__.__name__
        module_name = instance.__class__.__module__.split(".")[0]

        # Check if this is a task from an external package that should inherit caller context
        if module_name == "foo":
            # Try to find the caller's package by looking at current structlog loggers
            caller_package = cls._detect_caller_package()
            if caller_package:
                module_name = caller_package

        # Only include class_name (no module to keep logs clean)
        context = {"class_name": class_name, **extra_context}

        return cls(f"{module_name}.{class_name}", context)

    @classmethod
    def _detect_caller_package(cls) -> str | None:
        """
        Detect the package of the calling context by examining the call stack.

        Returns the first package found in the call stack.
        """
        import inspect
        import threading

        # Check if there's an active logger context in this thread
        thread_local = getattr(threading.current_thread(), "_logger_context", None)
        if thread_local and hasattr(thread_local, "package"):
            return thread_local.package

        # Fallback: examine call stack
        for frame_info in inspect.stack():
            frame_locals = frame_info.frame.f_locals
            frame_globals = frame_info.frame.f_globals

            # Look for existing logger instances in the frame
            for obj in frame_locals.values():
                if hasattr(obj, "_logger") and hasattr(obj._logger, "_logger_name"):
                    logger_name = obj._logger._logger_name
                    package = logger_name.split(".")[0]
                    # Return any valid package name that's not a system module
                    if (
                        package
                        and not package.startswith("_")
                        and package != "__main__"
                    ):
                        return package

            # Also check module name in globals
            module_name = frame_globals.get("__name__", "")
            if module_name and "." in module_name:
                package = module_name.split(".")[0]
                # Return any valid top-level package
                if package and not package.startswith("_"):
                    return package

        return None

    def bind(self, **kwargs) -> "ExoSimLogger":
        """Bind additional context to logger."""
        new_logger = ExoSimLogger(
            self._base_logger._context.get("logger_name", "exosim")
        )
        new_logger._base_logger = self._base_logger.bind(**kwargs)
        new_logger._context = {**self._context, **kwargs}
        return new_logger

    # Standard logging methods with structured support

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message with optional structured data."""
        if args:
            message = message % args
        self._base_logger.debug(message, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message with optional structured data."""
        if args:
            message = message % args
        self._base_logger.info(message, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message with optional structured data."""
        if args:
            message = message % args
        self._base_logger.warning(message, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message with optional structured data."""
        if args:
            message = message % args
        self._base_logger.error(message, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message with optional structured data."""
        if args:
            message = message % args
        self._base_logger.critical(message, **kwargs)

    def log(self, level: int, message: str, *args, **kwargs) -> None:
        """
        Log message at specified level.

        Parameters
        ----------
        level : int
            Logging level (logging.DEBUG, logging.INFO, etc.)
        message : str
            Log message
        *args
            Positional arguments for % formatting
        **kwargs
            Additional structured data
        """
        if args:
            message = message % args
        level_name = logging.getLevelName(level).lower()
        log_func = getattr(self, level_name, self.info)
        log_func(message, **kwargs)

    # ExoSim-specific logging methods

    def trace(self, message: str, *args, **kwargs) -> None:
        """Log trace-level message (maps to debug)."""
        if args:
            message = message % args
        self.debug(message, trace=True, **kwargs)

    def announcement(self, message: str, *args, **kwargs) -> None:
        """
        Log announcement message at ANNOUNCEMENT level.

        Announcements are displayed more prominently than regular info messages,
        useful for important events, milestones, or user-facing notifications.
        """
        if args:
            message = message % args
        # Temporarily override the log level in the event_dict
        # We'll use a custom processor to handle this
        kwargs["_announcement"] = True
        self._base_logger.info(message, **kwargs)

    def announce(self, message: str, *args, **kwargs) -> None:
        """Alias for announcement()."""
        self.announcement(message, *args, **kwargs)

    def graphics(self, message: str, **kwargs) -> None:
        """
        Log graphics-related message (ASCII art, banners, etc.).

        Uses custom GRAPHICS log level (18) - slightly below INFO.
        """
        # Use custom graphics level override
        kwargs["_graphics"] = True
        self._base_logger.info(message, **kwargs)

    # Performance tracking methods

    def log_runtime(self, message: str, level: str = "info", **kwargs) -> None:
        """
        Log runtime since last call or initialization.

        Parameters
        ----------
        message : str
            Log message
        level : str
            Log level (debug, info, warning, error, critical)
        **kwargs
            Additional structured data
        """
        current_time = time.perf_counter()
        duration = current_time - self._start_time
        self._start_time = current_time

        log_func = getattr(self, level, self.info)
        log_func(message, runtime_seconds=duration, **kwargs)

    def log_runtime_complete(self, message: str, level: str = "info", **kwargs) -> None:
        """
        Log total runtime since initialization.

        Parameters
        ----------
        message : str
            Log message
        level : str
            Log level
        **kwargs
            Additional structured data
        """
        total_duration = time.perf_counter() - self._start_time

        log_func = getattr(self, level, self.info)
        log_func(message, total_runtime_seconds=total_duration, **kwargs)

    # Context managers for performance tracking

    def time_operation(self, operation_name: str, **extra_context):
        """
        Context manager for timing operations.

        Usage:
            with logger.time_operation("data_processing", batch_size=1000):
                # ... processing logic ...
                pass
        """
        return _TimedOperation(self, operation_name, extra_context)


class _TimedOperation:
    """Context manager for timing operations with structured logging."""

    def __init__(
        self, logger: ExoSimLogger, operation_name: str, context: dict[str, Any]
    ):
        self.logger = logger
        self.operation_name = operation_name
        self.context = context
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.debug(
            "Operation started", operation=self.operation_name, **self.context
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time

        if exc_type is None:
            self.logger.info(
                "Operation completed",
                operation=self.operation_name,
                duration_seconds=duration,
                **self.context,
            )
        else:
            self.logger.error(
                "Operation failed",
                operation=self.operation_name,
                duration_seconds=duration,
                error_type=exc_type.__name__ if exc_type else None,
                error_message=str(exc_val) if exc_val else None,
                **self.context,
            )


# Default logger initialization
_default_logger = None


def get_default_logger() -> ExoSimLogger:
    """Get the default configured logger."""
    global _default_logger
    if _default_logger is None:
        # Configure with environment-based settings
        log_level = os.environ.get("EXOSIM_LOG_LEVEL", "INFO")
        development = os.environ.get("EXOSIM_DEV_MODE", "true").lower() == "true"
        json_logs = os.environ.get("EXOSIM_JSON_LOGS", "false").lower() == "true"

        configure_structlog(
            log_level=log_level, development=development, json_logs=json_logs
        )
        _default_logger = ExoSimLogger("exosim")

    return _default_logger


def _set_log_level_internal(
    level: str | int,
    logger_prefixes: list[str] | None = None,
    _explicit_prefixes: bool = False,
) -> None:
    """Internal function to set log level without circular imports."""
    # If logger_prefixes is None, this is a global call (reset everything)
    is_global_call = logger_prefixes is None

    if logger_prefixes is None:
        logger_prefixes = ["exosim"]

    # Convert level to string and numeric
    if isinstance(level, str):
        level_name = level.upper()
        numeric_level = getattr(logging, level_name, logging.INFO)
    else:
        numeric_level = level
        level_name = logging.getLevelName(level)

    # Set root logger level (this affects all Python loggers)
    logging.getLogger().setLevel(numeric_level)

    # If we're setting specific prefixes (explicit call), we need a different approach
    if not is_global_call:  # Specific prefixes case
        # Store the prefix filters globally for the custom processor to use
        global _active_prefix_filters
        # UPDATE existing filters, don't overwrite them
        for prefix in logger_prefixes:
            _active_prefix_filters[prefix] = numeric_level

        # Set root logger to DEBUG so our custom processor gets all messages
        logging.getLogger().setLevel(logging.DEBUG)

        # Set all existing loggers to DEBUG so filtering happens in our processor
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(logging.DEBUG)

    else:
        # Global case - apply to all loggers and clear filters
        _active_prefix_filters = {}

        # Set root logger and all existing loggers to the new level
        logging.getLogger().setLevel(numeric_level)
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(numeric_level)

        # Reconfigure structlog with the actual level (not DEBUG) for global case
        configure_structlog(log_level=level_name, development=True, json_logs=False)
        return  # Don't execute the DEBUG reconfigure below

    # Reconfigure structlog with TASK level (above DEBUG) to let our custom filtering work
    # but still allow most messages through for filtering
    configure_structlog(log_level="TASK", development=True, json_logs=False)


def disable_logging(
    log_id: int = 0,
    logger_prefixes: list[str] | None = None,
) -> None:
    """
    Disable logging by setting level to ERROR.

    Parameters
    ----------
    log_id : int
        Handler index (ignored, for backward compatibility)
    logger_prefixes : Optional[List[str]]
        List of logger name prefixes to disable. If None, uses default ('exosim').
    """
    # disable_logging is always explicit (never global)
    if logger_prefixes is None:
        logger_prefixes = ["exosim"]
    _set_log_level_internal(logging.ERROR, logger_prefixes)


def enable_logging(
    level: int = logging.INFO,
    log_id: int = 0,
    logger_prefixes: list[str] | None = None,
) -> None:
    """
    Enable logging at specified level.

    Parameters
    ----------
    level : int
        Logging level
    log_id : int
        Handler index (ignored, for backward compatibility)
    logger_prefixes : Optional[List[str]]
        List of logger name prefixes to enable. If None, uses default ('exosim').
    """
    # enable_logging is always explicit (never global)
    if logger_prefixes is None:
        logger_prefixes = ["exosim"]
    _set_log_level_internal(level, logger_prefixes)


def with_logger(func: Callable) -> Callable:
    """
    Decorator that automatically provides a logger to functions.

    The logger will be passed as a keyword argument 'logger' to the decorated function.
    If the function already receives a logger parameter, it will be used as-is.

    Usage:
        @with_logger
        def my_function(x, y, logger=None):
            logger.info("Processing data", x=x, y=y)
            return x + y

    Parameters
    ----------
    func : Callable
        Function to decorate with automatic logger injection

    Returns
    -------
    Callable
        Decorated function with logger parameter
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if logger is already provided
        if "logger" not in kwargs or kwargs["logger"] is None:
            # Create a logger for this function
            module_name = func.__module__.split(".")[0] if func.__module__ else "exosim"
            function_name = f"{module_name}.{func.__qualname__}"
            kwargs["logger"] = ExoSimLogger(function_name)
            kwargs["logger"]._base_logger._context["func_name"] = func.__name__

        return func(*args, **kwargs)

    return wrapper
