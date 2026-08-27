from collections import OrderedDict

import numpy as np
from astropy.table import QTable

from exosim.log import with_logger
from exosim.tasks import radiometric
from exosim.utils.klass_factory import find_task


@with_logger
def compute_saturation(
    table: QTable,
    payloadConfig: dict,
    input: str,
    channels_path=None,
    logger=None,
) -> QTable:
    """
    Computes and adds saturation metrics to the radiometric table for each channel.

    This function uses :class:`~exosim.tasks.radiometric.SaturationChannel`
    to determine, for each channel, the time to saturation, the integration time,
    and the maximum and minimum signal levels within each spectral bin.
    It also computes the frame time for each channel using the appropriate frame time task.

    Parameters
    ----------
    table : `~astropy.table.QTable`
        Table containing channel names, wavelengths, and other radiometric data.
    payloadConfig : dict
        Configuration dictionary for each channel. May be an OrderedDict
        mapping channel names to parameter dicts, or a single dict.
    input : str
        Path to the input file required by the saturation task.
    channels_path : str, optional
        Path to the channels in the input file.
    logger : `logging.Logger`, optional
        Injected logger instance for debug/info messages.

    Returns
    -------
    tuple of (`~astropy.table.QTable`, `~astropy.table.QTable`)
        - The full table with added columns:
          ``saturation_time``, ``integration_time``,
          ``max_signal_in_bin``, ``min_signal_in_bin``, ``observation_efficiency``, ``frame_time``.
        - A view of key columns:
          ``ch_name``, ``wavelength``, ``saturation_time``,
          ``integration_time``, ``max_signal_in_bin``, ``min_signal_in_bin``, ``observation_efficiency``, ``frame_time``.

    Notes
    -----
    The function processes all channels defined in the payload configuration.
    For each channel, it computes the saturation and frame time metrics and appends them as new columns to the
    radiometric table. The computation results for each channel are also returned as a separate table view
    containing only the relevant columns.
    """

    logger.debug("Entered compute_saturation() with %d rows", len(table))

    saturations = []
    integration_time = []
    max_signal_in_bin = []
    min_signal_in_bin = []

    saturation_task = radiometric.SaturationChannel()

    channels = payloadConfig["channel"]

    if isinstance(channels, OrderedDict):
        channel_items = channels.items()
    else:
        channel_items = [(None, channels)]

    for ch_name, params in channel_items:
        logger.debug("Computing saturation for channel %r", ch_name)
        sat, max_, min_ = saturation_task(
            table=table,
            description=params,
            input_file=input,
            channels_path=channels_path,
        )
        logger.debug(
            "Channel %r: sat=%r entries, sat sample=%r", ch_name, len(sat), sat[0]
        )
        saturations += sat
        max_signal_in_bin += max_
        min_signal_in_bin += min_

    table["saturation_time"] = saturations
    table["max_signal_in_bin"] = max_signal_in_bin
    table["min_signal_in_bin"] = min_signal_in_bin

    logger.info("Added saturation metrics: %d rows updated", len(table))

    integration_time = []
    for ch_name, params in channel_items:
        logger.debug("Computing integration time for channel %r", ch_name)

        # Check if radiometric key exists and contains integration_time_task
        radiometric_params = params.get("radiometric", {})
        compute_integration_time = (
            find_task(
                radiometric_params.get("integration_time_task"),
                radiometric.ComputeIntegrationTime(),
            )
            if "integration_time_task" in radiometric_params
            else radiometric.ComputeIntegrationTime()
        )

        integration_t = compute_integration_time(
            saturation_table=table,
            description=params,
            channel_name=ch_name if isinstance(channels, OrderedDict) else None,
        )
        integration_time += list(integration_t)
    table["integration_time"] = integration_time

    logger.info("Added integration_time column with %d entries", len(integration_time))

    observation_efficiency = []
    for ch_name, params in channel_items:
        logger.debug("Computing observation efficiency for channel %r", ch_name)

        # Check if radiometric key exists and contains observation_efficiency_task
        radiometric_params = params.get("radiometric", {})
        compute_observation_efficiency = (
            find_task(
                radiometric_params.get("observation_efficiency_task"),
                radiometric.ComputeObservationEfficiency(),
            )
            if "observation_efficiency_task" in radiometric_params
            else radiometric.ComputeObservationEfficiency()
        )

        efficiency = compute_observation_efficiency(
            radiometric_table=table,
            description=params,
            channel_name=ch_name if isinstance(channels, OrderedDict) else None,
        )

        n_bins = len(table[table["ch_name"] == ch_name]) if ch_name else len(table)
        observation_efficiency += list(efficiency * np.ones(n_bins, dtype=float))

    table["observation_efficiency"] = observation_efficiency

    # Convert to numpy arrays before division operation

    observation_efficiency_array = np.array(observation_efficiency, dtype=float)
    table["frame_time"] = table["integration_time"] / observation_efficiency_array

    logger.info("Added frame_time column with %d entries", len(table["frame_time"]))

    return (
        table,
        table[
            "ch_name",
            "wavelength",
            "saturation_time",
            "integration_time",
            "max_signal_in_bin",
            "min_signal_in_bin",
            "observation_efficiency",
            "frame_time",
        ],
    )
