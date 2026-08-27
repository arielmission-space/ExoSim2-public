"""
Unit tests for source list loading functionality.

This module contains tests for loading stellar source lists from CSV files,
including column mapping, data type conversion, and proper formatting
of source parameters with astropy units.
"""

from collections import OrderedDict

import pandas as pd
import pytest
from astropy import units as u

from exosim.tasks.load.load_source_list import LoadSourceList


class TestSourceListLoading:
    """Test suite for source list loading functionality."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """
        Create a temporary CSV file with sample stellar data.

        This fixture creates a test CSV file containing stellar parameters
        for two well-known exoplanet host stars, providing known test data
        for validation of the source list loading functionality.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Temporary directory path provided by pytest

        Returns
        -------
        str
            Path to the created temporary CSV file
        """
        df = pd.DataFrame(
            {
                "Star": ["HD 209458", "GJ 1214"],
                "Radius": [1.18, 0.218],
                "Distance": [47, 13],
                "Temp": [6086, 3026],
                "Mass": [1.15, 0.176],  # Added Mass column for completeness
            }
        )
        file = tmp_path / "stars.csv"
        df.to_csv(file, index=False)
        return str(file)

    def test_csv_loading_with_column_mapping(self, sample_csv):
        """
        Test loading source list from CSV with custom column mapping.

        This test verifies that the LoadSourceList task can correctly:
        - Read CSV files with stellar data
        - Apply custom column mappings to match expected parameter names
        - Convert numerical values to appropriate astropy units
        - Generate properly formatted OrderedDict output

        The test uses known stellar parameters for HD 209458 and GJ 1214
        to validate correct data parsing and unit assignment.
        """
        # Instantiate the task
        task = LoadSourceList()

        # Define column mapping and source type
        column_map = {
            "name": "Star",
            "R": "Radius",
            "D": "Distance",
            "T": "Temp",
            "M": "Mass",
        }
        source_type = "planck"

        output = task(
            targetlist_filepath=sample_csv,
            source_type=source_type,
            column_mapping=column_map,
        )

        # Construct expected OrderedDict for validation
        expected = OrderedDict(
            [
                (
                    "HD 209458",
                    {
                        "value": "HD 209458",
                        "source_type": source_type,
                        "R": 1.18 * u.R_sun,
                        "D": 47 * u.pc,
                        "T": 6086 * u.K,
                    },
                ),
                (
                    "GJ 1214",
                    {
                        "value": "GJ 1214",
                        "source_type": source_type,
                        "R": 0.218 * u.R_sun,
                        "D": 13 * u.pc,
                        "T": 3026 * u.K,
                    },
                ),
            ]
        )

        # Validate each entry in the output matches expectations
        assert len(output) == len(expected), (
            "Output should have same number of entries as expected"
        )

        for name in expected:
            assert name in output, f"Star {name} should be present in output"
            out = output[name]
            exp = expected[name]

            assert out["value"] == exp["value"], f"Star name should match for {name}"
            assert out["source_type"] == exp["source_type"], (
                f"Source type should match for {name}"
            )
            assert out["R"] == exp["R"], f"Radius should match for {name}"
            assert out["D"] == exp["D"], f"Distance should match for {name}"
            assert out["T"] == exp["T"], f"Temperature should match for {name}"
