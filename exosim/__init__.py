import importlib.metadata as metadata
import os.path
from datetime import date


__version__ = metadata.version("exosim")

meta = metadata.metadata("exosim")
__pkg_name__ = meta.get("Name", "exosim")
__title__ = "ExoSim2"
__url__ = meta.get("Home-page", "https://github.com/arielmission-space/ExoSim2-public")
__author__ = meta.get("Author", "L. V. Mugnai")
__license__ = meta.get("License", "BSD-3-Clause")
__copyright__ = f"2020-{date.today().year}, {__author__}"

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
