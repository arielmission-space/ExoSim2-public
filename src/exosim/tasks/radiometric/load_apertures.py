from astropy.table import QTable

from .estimate_apertures import EstimateApertures


class LoadApertures(EstimateApertures):
    """
    Loads aperture definitions from a file and returns a table with the required columns
    for photometric aperture extraction. This class is intended to standardize the input
    and output format for aperture definitions used in photometric extraction tasks.

    The user can subclass and overwrite the `model` method to implement custom logic.
    """

    def model(self, table, focal_plane, description, wl_grid):
        """
        Loads an aperture definition table from file, checks for required columns,
        and returns a new table containing only those columns.

        Parameters
        ----------
        table : astropy.table.QTable
            Input wavelength table (not used in this method).
        focal_plane : any
            Focal plane data (not used in this method).
        description : dict
            Dictionary containing the aperture photometry configuration.
            Must include the key "file_name" with the path to the aperture table file.
        wl_grid : any
            wavelength grid (not used in this method).

        Returns
        -------
        astropy.table.QTable
            Table containing only the required columns for aperture extraction:
            "spectral_center", "spectral_size", "spatial_center", "spatial_size", "aperture_shape".

        Raises
        ------
        ValueError
            If any of the required columns are missing from the loaded table.

        Notes
        -----
        The loaded table must contain all the required columns. Only these columns
        will be included in the returned table.
        """

        fname = description["file_name"]
        required_columns = [
            "spectral_center",
            "spectral_size",
            "spatial_center",
            "spatial_size",
            "aperture_shape",
            "aperture_size",
        ]

        # Load the table
        self.info(f"Loading aperture definitions from {fname}")
        loaded_tab = QTable.read(fname)

        # Check for missing columns
        missing = [col for col in required_columns if col not in loaded_tab.colnames]
        if missing:
            raise ValueError(f"Missing required columns in {fname}: {missing}")

        # Build new table with only the required columns
        new_tab = QTable()
        for col in required_columns:
            new_tab[col] = loaded_tab[col]

        return new_tab
