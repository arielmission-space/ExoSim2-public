"""
Behavioural tests for the structlog-based ExoSimLogger: every log level, the
context binding, the timing helpers, the timed-operation context manager (both
the success and the failure path) and the file-logging configuration.
"""

import logging

import pytest

from exosim.log.structlog_config import (
    ExoSimLogger,
    configure_structlog,
    disable_logging,
    enable_logging,
    get_default_logger,
    with_logger,
)


class TestExoSimLogger:
    def test_every_level_is_callable(self):
        log = ExoSimLogger("TestLogger")
        for method in ("debug", "info", "warning", "error", "critical", "trace"):
            getattr(log, method)("message", extra=1)
        log.announcement("hi")
        log.announce("hi")
        log.graphics("art")
        log.log(logging.INFO, "numeric level")

    def test_bind_returns_a_child_with_extra_context(self):
        log = ExoSimLogger("Parent")
        child = log.bind(run_id="abc")
        assert isinstance(child, ExoSimLogger)
        child.info("with context")

    def test_for_class_binds_the_instance_name(self):
        class Widget:
            pass

        log = ExoSimLogger.for_class(Widget())
        assert isinstance(log, ExoSimLogger)
        log.info("from a class logger")

    def test_runtime_helpers(self):
        log = ExoSimLogger("Timing")
        log.log_runtime("step done")
        log.log_runtime_complete("all done")

    def test_time_operation_success_path(self):
        log = ExoSimLogger("Op")
        with log.time_operation("processing", batch=10) as op:
            assert op.operation_name == "processing"

    def test_time_operation_failure_path_reraises(self):
        log = ExoSimLogger("Op")

        def _run():
            with log.time_operation("failing"):
                raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            _run()

    def test_percent_style_message_formatting(self):
        log = ExoSimLogger("Fmt")
        # every level accepts printf-style args
        log.debug("d=%s", 1)
        log.info("i=%d", 2)
        log.warning("w=%s", "x")
        log.error("e=%s/%s", "a", "b")
        log.critical("c=%s", 9)
        log.trace("t=%s", "z")
        log.log(logging.WARNING, "lvl=%s", "warn")


class TestModuleLevelHelpers:
    def test_get_default_logger_is_a_singleton(self):
        assert get_default_logger() is get_default_logger()

    def test_enable_and_disable_logging_run(self):
        # they adjust the exosim logger filtering; just exercise both paths
        disable_logging()
        disable_logging(logger_prefixes=["exosim.tasks"])
        enable_logging(logging.DEBUG)
        enable_logging(logging.INFO, logger_prefixes=["exosim"])

    def test_configure_structlog_with_a_file(self, tmp_path):
        log_file = tmp_path / "exosim.log"
        configure_structlog(log_level="DEBUG", log_file=str(log_file), json_logs=True)
        logging.getLogger("exosim").info("written to file")
        assert log_file.exists()
        # restore console-only config for the rest of the suite
        configure_structlog(log_level="INFO")

    def test_set_log_level_accepts_a_string_name(self):
        from exosim.log import set_log_level

        try:
            set_log_level("WARNING")
            set_log_level("debug", logger_prefixes=["exosim.tasks"])
        finally:
            set_log_level(logging.INFO)

    def test_with_logger_decorator_injects_a_logger(self):
        @with_logger
        def do_work(x, logger=None):
            assert logger is not None
            logger.info("working")
            return x * 2

        assert do_work(3) == 6
