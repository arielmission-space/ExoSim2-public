from collections import OrderedDict

from astropy.table import QTable

from exosim.log import with_logger
from exosim.tasks import radiometric


@with_logger
def compute_multiaccum(table: QTable, payloadConfig: dict, logger=None) -> QTable:
    """
    It estimates the multiaccum gain factors using :class:`~exosim.tasks.radiometric.multiaccum.Multiaccum`.
    The multiaccum gain factors are computed for each channel in the table based on the channel description.

    Parameters
    ----------
    table: `~astropy.table.QTable`
        table containing the channel names and other relevant data
    payloadConfig: dict
        payload configuration dictionary containing channel descriptions

    Returns
    -------
    astropy.table.QTable:
        the input table with an additional columns for multiaccum read and shot gain
    astropy.table.QTable:
        multiaccum factors
    """

    logger.debug("Entered compute_multiaccum() with %d rows", len(table))

    read_gain, shot_gain = [], []
    multiaccum = radiometric.Multiaccum()

    channels = payloadConfig["channel"]

    if isinstance(channels, OrderedDict):
        channel_items = channels.items()
    else:
        channel_items = [("single_channel", channels)]

    for ch, desc in channel_items:
        radi_cfg = desc.get("radiometric", {})
        if "multiaccum" in radi_cfg:
            logger.debug("Channel %r: multiaccum parameters found", ch)
            read, shot = multiaccum(parameters=radi_cfg["multiaccum"])
            n = (
                len(table[table["ch_name"] == ch])
                if isinstance(channels, OrderedDict)
                else len(table)
            )
            logger.debug("Computed read=%r, shot=%r for %d rows", read, shot, n)
        else:
            logger.debug("Channel %r: no multiaccum, using defaults", ch)
            read, shot = 1, 1
            n = (
                len(table[table["ch_name"] == ch])
                if isinstance(channels, OrderedDict)
                else len(table)
            )

        read_gain += [read] * n
        shot_gain += [shot] * n

    table["multiaccum_read_gain"] = read_gain
    table["multiaccum_shot_gain"] = shot_gain

    logger.info(
        "Finished compute_multiaccum: added 'multiaccum_read_gain' and "
        "'multiaccum_shot_gain' columns"
    )

    return (
        table,
        table[
            "ch_name",
            "wavelength",
            "multiaccum_read_gain",
            "multiaccum_shot_gain",
        ],
    )
