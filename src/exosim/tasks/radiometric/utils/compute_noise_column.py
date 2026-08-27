from collections import OrderedDict

import numpy as np
from astropy.table import QTable

from exosim.log import with_logger
from exosim.utils.klass_factory import find_task


@with_logger
def compute_noise_column(
    table: QTable,
    payloadConfig: dict,
    noise_key: str,
    task_key: str,
    default_task,
    signal_col: str,
    gain_col: str,
    output_col: str,
    logger=None,
) -> QTable:
    """
    Generalized function to compute and add a noise column to the table.

    Parameters
    ----------
    table : QTable
        The radiometric table to add noise columns to
    payloadConfig : dict
        Payload configuration containing channel specifications
    noise_key : str
        Key in the radiometric configuration (e.g., 'dark_current', 'read_noise')
    task_key : str
        Key for custom task specification (e.g., 'dark_current_task')
    default_task : class
        Default task class to use if no custom task is specified
    signal_col : str
        Column name containing the signal data
    gain_col : str or None
        Column name containing gain data (None for noise types that don't use gain)
    output_col : str or None
        Name of the main output column to create. If None, only adds individual
        noise components from the noise_table
    logger : Logger
        Logger instance for debugging

    Returns
    -------
    QTable
        Updated table with new noise columns
    """
    logger.debug(
        f"Entered compute_noise_column() for {noise_key} with {len(table)} rows"
    )

    channels = payloadConfig["channel"]

    if isinstance(channels, OrderedDict):
        channel_items = channels.items()
    else:
        channel_items = [("single_channel", channels)]

    for ch_name, ch_description in channel_items:
        # Skip if noise configuration is not present
        if noise_key not in ch_description["radiometric"]:
            logger.debug(f"Channel {ch_name!r}: no {noise_key} configuration, skipping")
            continue
        # Skip if noise is disabled
        if not ch_description["radiometric"][noise_key]:
            logger.debug(f"Channel {ch_name!r}: {noise_key} is disabled, skipping")
            continue

        logger.info(f"Estimating {noise_key} on {ch_name}")

        # Get the noise computation task (custom or default)
        compute_noise_task = (
            find_task(
                ch_description["radiometric"].get(task_key),
                default_task,
            )
            if task_key in ch_description["radiometric"]
            else default_task
        )
        computeNoise = compute_noise_task()

        # Get rows for this channel
        mask = table["ch_name"] == ch_name

        # Handle different return patterns from noise computation tasks
        if gain_col is not None:
            # Standard pattern: tasks that need gain and return (noise_table, noise)
            noise_table, noise = computeNoise(
                signal=table[signal_col][mask],  # Pass all values, not just [0]
                aperture_table=table[mask],
                description=ch_description,
                multiaccum_gain=table[gain_col][mask],  # Pass all values, not just [0]
            )
        else:
            # Custom noise pattern: tasks that don't need gain
            noise_table, noise = computeNoise(
                wavelength=table["wavelength"][
                    mask
                ],  # Pass all wavelengths, not just [0]
                description=ch_description,
                radiometric_table=table[mask],
            )

        # Add the main noise column to the table (if output_col is specified)
        if output_col is not None:
            if output_col not in table.colnames:
                table[output_col] = np.nan * noise.unit
            table[output_col][mask] = noise  # Assign all values, remove [0]!

            logger.debug(
                f"Channel {ch_name!r}: {output_col} min={np.min(noise)!r}, max={np.max(noise)!r}"
            )

        # Add all individual noise components to the table
        for col_name in noise_table.colnames:
            # Skip the output column if it's already been handled above
            if output_col is not None and col_name == output_col:
                continue
            # Instead of skipping, update only the rows for this channel
            if col_name in table.colnames:
                table[col_name][mask] = noise_table[col_name]
            else:
                table[col_name] = 0 * noise_table[col_name].unit
                table[col_name][mask] = noise_table[col_name]
            logger.debug(
                f"Channel {ch_name!r}: updated component {col_name} min={np.min(noise_table[col_name])!r}, max={np.max(noise_table[col_name])!r}"
            )
        if output_col is None:
            logger.info(
                f"Channel {ch_name}: added {len(noise_table.colnames)} noise components"
            )

    logger.info(
        f"Finished compute_noise_column for {noise_key} on all {len(table)} rows"
    )
    return table
