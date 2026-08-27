"""
Modern Logger class using structlog as backend.

This module provides a modern structured logging implementation
for ExoSim 2.0 Task classes using structlog for enhanced capabilities.
"""

from .structlog_config import ExoSimLogger, get_default_logger, with_logger


class Logger:
    """
    Modern structured logger for ExoSim 2.0 Tasks.

    This class uses structlog as the backend for structured logging capabilities,
    providing enhanced observability and better log analysis.
    """

    def __init__(self):
        """Initialize the logger with automatic name detection."""
        import contextlib

        with contextlib.suppress(TypeError):
            super().__init__()
        self.set_log_name()

    def set_log_name(self) -> None:
        """
        Set the logger name based on the class that inherits this Logger.

        Creates a structured logger with automatic class context detection.
        """
        class_name = self.__class__.__name__

        # Create structured logger with class context (no module to keep logs clean)
        self._logger = ExoSimLogger.for_class(self, class_name=class_name)

        # Store logger name
        self._log_name = f"exosim.{class_name}"

    # Standard logging methods

    def debug(self, message: str, *args, **kwargs) -> None:
        """
        Log debug message.

        Parameters
        ----------
        message : str
            Log message (supports % formatting)
        *args
            Positional arguments for message formatting
        **kwargs
            Additional structured data for the log record
        """
        formatted_message = self._format_message(message, args)
        self._logger.debug(formatted_message, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """
        Log info message.

        Parameters
        ----------
        message : str
            Log message
        *args
            Positional arguments for message formatting
        **kwargs
            Additional structured data
        """
        formatted_message = self._format_message(message, args)
        self._logger.info(formatted_message, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """
        Log warning message.

        Parameters
        ----------
        message : str
            Log message
        *args
            Positional arguments for message formatting
        **kwargs
            Additional structured data
        """
        formatted_message = self._format_message(message, args)
        self._logger.warning(formatted_message, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """
        Log error message.

        Parameters
        ----------
        message : str
            Log message
        *args
            Positional arguments for message formatting
        **kwargs
            Additional structured data
        """
        formatted_message = self._format_message(message, args)
        self._logger.error(formatted_message, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """
        Log critical message.

        Parameters
        ----------
        message : str
            Log message
        *args
            Positional arguments for message formatting
        **kwargs
            Additional structured data
        """
        formatted_message = self._format_message(message, args)
        self._logger.critical(formatted_message, **kwargs)

    # ExoSim-specific logging methods

    def trace(self, message: str, *args, **kwargs) -> None:
        """
        Log trace message.

        Maps to debug level with trace marker for structured logging.
        """
        formatted_message = self._format_message(message, args)
        self._logger.trace(formatted_message, **kwargs)

    def announce(self, message: str, *args, **kwargs) -> None:
        """
        Log announcement message.

        Maps to info level with announcement marker.
        """
        formatted_message = self._format_message(message, args)
        self._logger.announce(formatted_message, **kwargs)

    def graphics(self, message: str, *args, **kwargs) -> None:
        """
        Log graphics-related message.

        Maps to info level with graphics category marker.
        """
        formatted_message = self._format_message(message, args)
        self._logger.graphics(formatted_message, **kwargs)

    # Enhanced structured logging methods

    def bind(self, **context) -> "Logger":
        """
        Create a new logger with additional context bound to all messages.

        Parameters
        ----------
        **context
            Key-value pairs to bind to the logger context

        Returns
        -------
        Logger
            New logger instance with bound context

        Usage
        -----
        task_logger = self.bind(task_id="sim_001", instrument="ARIEL")
        task_logger.info("Starting simulation")  # Will include task_id and instrument
        """
        new_logger = Logger.__new__(Logger)
        new_logger._logger = self._logger.bind(**context)
        new_logger._log_name = self._log_name
        return new_logger

    def time_operation(self, operation_name: str, **context):
        """
        Context manager for timing operations with structured logging.

        Parameters
        ----------
        operation_name : str
            Name of the operation being timed
        **context
            Additional context to include in timing logs

        Usage
        -----
        with self.time_operation("focal_plane_generation", channels=4):
            # ... focal plane generation logic ...
            pass
        """
        return self._logger.time_operation(operation_name, **context)

    def log_runtime(self, message: str, level: str = "info") -> None:
        """
        Log runtime since last call.

        Parameters
        ----------
        message : str
            Log message
        level : str
            Log level (debug, info, warning, error, critical)
        """
        self._logger.log_runtime(message, level)

    def log_runtime_complete(self, message: str, level: str = "info") -> None:
        """
        Log total runtime since logger creation.

        Parameters
        ----------
        message : str
            Log message
        level : str
            Log level
        """
        self._logger.log_runtime_complete(message, level)

    # Utility methods

    def _format_message(self, message: str, args: tuple) -> str:
        """
        Format message with args using % formatting.

        Handles % formatting while gracefully handling formatting errors.
        """
        if args:
            try:
                return message % args
            except (TypeError, ValueError):
                # If formatting fails, return the original message
                return message
        return message

    # Properties

    @property
    def logger(self):
        """Access to the underlying structured logger."""
        return self._logger

    @property
    def log_name(self) -> str:
        """Get the logger name."""
        return self._log_name


def get_logger(name: str = "exosim"):
    """Get a logger instance."""
    return get_default_logger()


__all__ = ["Logger", "get_logger", "with_logger"]
