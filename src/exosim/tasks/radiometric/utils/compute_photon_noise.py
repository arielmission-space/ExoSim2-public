from collections import OrderedDict

import numpy as np
from astropy.table import QTable

from exosim.log import with_logger
from exosim.tasks import radiometric


@with_logger
def compute_photon_noise(table: QTable, payloadConfig: dict, logger=None) -> QTable:
    """
    It computes and adds the photon noise to the radiometric table using
    :class:`~exosim.tasks.radiometric.computePhotonNoise.ComputePhotonNoise`,
    based on the signal in the aperture and the channel description.

    Parameters
    ----------
    table: `~astropy.table.QTable`
        table containing the signals in the aperture and other relevant data
    payloadConfig: dict
        payload configuration dictionary containing channel descriptions

    Returns
    -------
    astropy.table.QTable:
        the input table with additional columns for photon noise
    """

    logger.debug("Entered compute_photon_noise() with %d rows", len(table))

    computePhotonNoise = radiometric.ComputePhotonNoise()

    signals = [k for k in table.colnames if "_signal_in_aperture" in k]
    logger.debug("Found %d signal columns: %s", len(signals), signals)

    channels = payloadConfig["channel"]
    if isinstance(channels, OrderedDict):
        channel_dict = channels
    else:
        channel_dict = {"single_channel": channels}

    if "observation_efficiency" in table.colnames:
        observation_efficiency = table["observation_efficiency"]
    else:
        observation_efficiency = 1.0 * np.ones(len(table), dtype=float)

    for sig in signals:
        phot_noise = []
        logger.debug("Processing signal column %r", sig)

        for i, ch_name in enumerate(table["ch_name"]):
            ch_description = channel_dict.get(ch_name, channels)

            sig_scaled = table[sig][i] * observation_efficiency[i]

            noise = computePhotonNoise(
                signal=sig_scaled,
                description=ch_description,
                multiaccum_gain=table["multiaccum_shot_gain"][i],
            )
            phot_noise.append(noise)

            logger.debug(
                "Row %d, channel %r: signal=%r, photon_noise=%r",
                i,
                ch_name,
                table[sig][i],
                noise,
            )

        phot_noise = np.array([p.value for p in phot_noise]) * phot_noise[0].unit
        colname = sig.replace("_signal_in_aperture", "_photon_noise")
        table[colname] = phot_noise

        logger.debug("Added column %r with %d entries", colname, len(phot_noise))

    logger.info("Finished compute_photon_noise for all %d signal columns", len(signals))
    return table
