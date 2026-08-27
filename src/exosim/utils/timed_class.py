import logging
import time

from exosim.log import Logger

# Configure root logger to propagate messages for testing and capture in pytest
logging.getLogger("exosim").propagate = True
logging.getLogger("exosim").handlers = [logging.NullHandler()]


class TimedClass(Logger):
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
            full_message = message + f": {time_stamp}"
            # Log with both Logger and standard logging for test capture
            log_to_call(full_message)
            std_log = getattr(
                logging.getLogger("exosim." + self.__class__.__name__), level
            )
            std_log(full_message)
        except AttributeError:
            warning_msg = "calling class has no Logger's methods"
            self.warning(warning_msg)
            logging.getLogger("exosim." + self.__class__.__name__).warning(warning_msg)

        self.start_time = time.time()

    def log_runtime_complete(self, message, level="info"):
        try:
            log_to_call = getattr(self, level)
            time_stamp = time.strftime(
                "%Hh%Mm%Ss", time.gmtime(time.time() - self.start_time_gen)
            )
            full_message = message + f": {time_stamp}"
            # Log with both Logger and standard logging for test capture
            log_to_call(full_message)
            std_log = getattr(
                logging.getLogger("exosim." + self.__class__.__name__), level
            )
            std_log(full_message)
        except AttributeError:
            warning_msg = "calling class has no Logger's methods"
            self.warning(warning_msg)
            logging.getLogger("exosim." + self.__class__.__name__).warning(warning_msg)
