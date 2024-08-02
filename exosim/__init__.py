import importlib.metadata as metadata
import os.path
from datetime import date

__version__ = metadata.version("exosim")

# load package info
__pkg_name__ = metadata.metadata("exosim")["Name"]
__title__ = "ExoSim2"
__url__ = metadata.metadata("exosim")["Home-page"]
__author__ = metadata.metadata("exosim")["Author"]
__license__ = metadata.metadata("exosim")["license"]
__copyright__ = "2020-{:d}, {}".format(date.today().year, __author__)
__citation__ = None
__summary__ = metadata.metadata("exosim")["Summary"]

# load package commit number
try:
    __base_dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    __base_dir__ = None

__commit__ = None
__branch__ = None
if __base_dir__ is not None and os.path.exists(
    os.path.join(__base_dir__, ".git")
):
    git_folder = os.path.join(__base_dir__, ".git")
    with open(os.path.join(git_folder, "HEAD")) as fp:
        ref = fp.read().strip()
    ref_dir = ref[5:]
    __branch__ = ref[16:]
    try:
        with open(os.path.join(git_folder, ref_dir)) as fp:
            __commit__ = fp.read().strip()
    except FileNotFoundError:
        __commit__ = None


# initialise logger
import logging

logger = logging.getLogger(__pkg_name__)
logger.info("code version {}".format(__version__))

from exosim.log import setLogLevel

setLogLevel(logging.INFO)
