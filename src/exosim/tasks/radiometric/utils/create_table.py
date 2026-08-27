from collections import OrderedDict

from astropy.table import QTable, vstack

from exosim.log import with_logger
from exosim.tasks import radiometric
from exosim.utils.klass_factory import find_task


@with_logger
def create_table(payloadConfig: dict, logger=None) -> QTable:
    """
    Produces the starting radiometric table with the spectral bins and their edges.
    It is based on :class:`~exosim.tasks.radiometric.estimateSpectralBinning.EstimateSpectralBinning` by default.
    """

    logger.debug("Entered create_table()")

    if "spectral_binning_task" in payloadConfig["channel"]:
        task_name = payloadConfig["channel"]["spectral_binning_task"]
        logger.debug("Custom spectral_binning_task requested: %r", task_name)
        task_cls = find_task(task_name, radiometric.EstimateSpectralBinning)
    else:
        logger.debug("Using default EstimateSpectralBinning task")
        task_cls = radiometric.EstimateSpectralBinning

    estimateSpectralBinning = task_cls()
    channels = payloadConfig["channel"]

    if isinstance(channels, OrderedDict):
        channel_items = channels.items()
    else:
        channel_items = [("single_channel", channels)]

    table_list = []
    for ch, params in channel_items:
        logger.debug("Estimating spectral binning for channel %r", ch)
        tbl = estimateSpectralBinning(parameters=params)
        table_list.append(tbl)

    combined = vstack(table_list)
    logger.info(
        "Created combined radiometric table with %d rows and %d columns",
        len(combined),
        len(combined.colnames),
    )

    return combined
