import logging

# Import all metadata from centralized location
from exosim.__about__ import (
    __author__,
    __author_email__,
    __branch__,
    __citation__,
    __commit__,
    __copyright__,
    __description__,
    __license__,
    __pkg_name__,
    __title__,
    __url__,
    __version__,
    __version_info__,
    is_development_version,
    is_release_version,
)
from exosim.log import set_log_level

logger = logging.getLogger(__pkg_name__)
logger.info(f"code version {__version__}")
set_log_level(logging.INFO)
