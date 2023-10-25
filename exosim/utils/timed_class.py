import logging
import time

from exosim.log import generate_logger_name


class TimedClass:
    """
    This class adds methods to log the elapsed time
    """

    def __init__(self):
        super().__init__()
        self.start_time_gen = time.time()
        self.start_time = time.time()

    def log_runtime(self, message, level="info"):
        try:
            log_to_call = getattr(self, level)
            time_stamp = time.strftime(
                "%Hh%Mm%Ss", time.gmtime(time.time() - self.start_time)
            )
            full_message = message + ": {}".format(time_stamp)
            log_to_call(full_message)
        except AttributeError:
            logger.warning("calling class has no Logger's methods")

        self.start_time = time.time()

    def log_runtime_complete(self, message, level="info"):
        try:
            log_to_call = getattr(self, level)
            time_stamp = time.strftime(
                "%Hh%Mm%Ss", time.gmtime(time.time() - self.start_time_gen)
            )
            full_message = message + ": {}".format(time_stamp)
            log_to_call(full_message)
        except AttributeError:
            logger.warning("calling class has no Logger's methods")


logger = logging.getLogger(generate_logger_name(TimedClass))
