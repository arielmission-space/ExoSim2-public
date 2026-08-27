from collections import OrderedDict

import pandas as pd
from astropy import units as u

from exosim.tasks.task import Task


class LoadSourceList(Task):
    def __init__(self):
        """
        LoadSourceList is a task that reads a list of stellar sources from a CSV or Excel file
        and converts it into a structured OrderedDict for use in simulations.

        Each row in the file must contain information about a single star: its name, radius,
        distance, and effective temperature. The file may use arbitrary column names, which
        must be mapped to expected keys ('name', 'R', 'D', 'T', and 'M') via the `column_mapping` parameter.

        The default units for the radius, distance, temperature and mass are:
        - Radius: Solar radii (R_sun)
        - Distance: Parsecs (pc)
        - Temperature: Kelvin (K)
        - Mass: Solar masses (M_sun)


        Parameters
        __________
        targetilst_filepath: str
            path to the target list file, which contains stellar sources data.
        source_type: str (optional)
            type of the source, e.g., 'planck' for Planck's law. Default is 'planck'.
        column_mapping: dict (optional)
            mapping of column names in the input file to expected keys.
            Default is an empty dictionary, which means the default column names are used.


        Returns
        ---------
        OrderedDict
            Dictionary with star names as keys and dictionaries containing:
            - 'value': original star name (str)
            - 'source_type': value from source_type_column
            - 'R': stellar radius as an astropy Quantity in solar radii
            - 'D': distance as an astropy Quantity in parsecs
            - 'T': effective temperature as an astropy Quantity in kelvin
            - 'M': stellar mass as an astropy Quantity in solar masses

        Raises
        ------
        ValueError
            If the input file does not contain the required columns or if the column mapping is incorrect.
        """

        self.add_task_param("targetlist_filepath", "path to the target list file")
        self.add_task_param("source_type", "source type", "planck")
        self.add_task_param(
            "column_mapping", "mapping of column names in the input file", {}
        )

    def execute(self):
        filepath = self.get_task_param("targetlist_filepath")
        source_type = self.get_task_param("source_type")
        column_map = self.get_task_param("column_mapping")

        # Decide which pandas reader to use based on file extension
        if filepath.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)

        # Ensure the DataFrame has the expected columns
        required = {"name", "R", "D", "T", "M"}
        missing = required - set(column_map)
        if missing:
            ValueError(f"Missing mapped columns: {missing}")

        expected_cols = set(column_map.values())
        missing_cols = expected_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")

        sources = OrderedDict()
        for _, row in df.iterrows():
            name = str(row[column_map["name"]]).strip()
            # Parse units
            radius = row[column_map["R"]] * u.R_sun
            distance = row[column_map["D"]] * u.pc
            temperature = row[column_map["T"]] * u.K
            mass = row[column_map["M"]] * u.M_sun

            sources[name] = {
                "value": name,
                "source_type": source_type,
                "R": radius,
                "D": distance,
                "T": temperature,
                "M": mass,
                "metadata": {
                    col: row[col] for col in df.columns
                },  # include all original columns
            }
        self.info(f"Loaded {len(sources)} sources from {filepath}")
        self.set_output(sources)
