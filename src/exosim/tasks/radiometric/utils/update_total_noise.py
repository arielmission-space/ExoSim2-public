from astropy.table import QTable

from exosim.log import with_logger
from exosim.tasks import radiometric


@with_logger
def update_total_noise(table: QTable, logger=None) -> QTable:
    """
    Updates the total noise column in the radiometric table.
    It uses the :class:`~exosim.tasks.radiometric.computeTotalNoise.ComputeTotalNoise`
    to compute the total noise based on the existing columns in the table.
    The total noise is computed from the photon noise and other relevant parameters.
    Parameters
    ----------
    table: `~astropy.table.QTable`
        table containing the photon noise and other relevant data

    Returns
    -------
    astropy.table.QTable:
        the input table with an additional column for total noise
    astropy.table.QTable:
        a view of the table with key columns: 'ch_name', 'wavelength', and 'total_noise'
    """
    logger.debug("Entered update_total_noise with %d rows", len(table))

    compute_total_noise = radiometric.ComputeTotalNoise()
    total_noise = compute_total_noise(table=table)
    logger.debug(
        "Computed total_noise array of length %d, dtype=%r",
        len(total_noise),
        getattr(total_noise, "dtype", None),
    )

    table["total_noise"] = total_noise
    logger.info("Added 'total_noise' column for %d channels", len(table["ch_name"]))

    # return full table plus a view of key columns
    return table, table["ch_name", "wavelength", "total_noise"]
