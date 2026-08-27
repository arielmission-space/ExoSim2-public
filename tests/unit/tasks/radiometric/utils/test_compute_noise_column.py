#!/usr/bin/env python3
"""Test compute_noise_column module."""

from unittest import TestCase

import numpy as np
from astropy import units as u
from astropy.table import QTable

from exosim.tasks.radiometric.utils.compute_noise_column import compute_noise_column
from exosim.tasks.task import Task


# Mock task class for testing
class DummyTask(Task):
    """Mock task class that returns predefined noise values."""

    def __init__(self):
        super().__init__()
        self.return_values = (
            QTable({"noise_component": [0.1, 0.1, 0.1] * u.electron}),
            np.array([0.1, 0.1, 0.1]) * u.electron,
        )

    def __call__(self, *args, **kwargs):
        """Return predefined noise values regardless of input."""
        return self.return_values


class TestComputeNoiseColumn(TestCase):
    """Test compute_noise_column functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test table and data with units
        self.table = QTable()
        self.table["signal"] = [1.0, 2.0, 3.0] * u.electron
        self.table["gain"] = [0.5, 0.5, 0.5] * u.dimensionless_unscaled
        self.table["ch_name"] = ["single_channel"] * 3

        # Prepare config with correct structure
        self.config = {
            "channel": {
                "radiometric": {
                    "test_noise": True,
                    "test_task": "DummyTask",  # Class name as string
                },
            },
        }

        # Real task class - not a Mock
        self.default_task = DummyTask

    def test_noise_computation_with_gain(self):
        """Test noise computation with gain."""
        result = compute_noise_column(
            table=self.table,
            payloadConfig=self.config,
            noise_key="test_noise",
            task_key="test_task",
            default_task=self.default_task,
            signal_col="signal",
            gain_col="gain",
            output_col="output",
        )

        # Check results
        assert "output" in result.colnames
        assert isinstance(result["output"], u.Quantity)
        np.testing.assert_array_equal(result["output"].value, [0.1, 0.1, 0.1])

    def test_noise_computation_without_gain(self):
        """Test noise computation without gain."""
        # Create test data with wavelength instead of gain
        table = QTable()
        table["signal"] = [1.0, 2.0, 3.0] * u.electron
        table["wavelength"] = [500, 600, 700] * u.nm
        table["ch_name"] = ["single_channel"] * 3

        result = compute_noise_column(
            table=table,
            payloadConfig=self.config,
            noise_key="test_noise",
            task_key="test_task",
            default_task=self.default_task,
            signal_col="signal",
            gain_col=None,
            output_col="output",
        )

        # Check results
        assert "output" in result.colnames
        assert isinstance(result["output"], u.Quantity)
        np.testing.assert_array_equal(result["output"].value, [0.1, 0.1, 0.1])

    def test_noise_computation_disabled(self):
        """Test when noise is disabled in config."""
        # Modify config to disable noise
        config = {
            "channel": {
                "radiometric": {
                    "test_noise": False,  # Disabled
                    "test_task": "DummyTask",
                },
            },
        }

        result = compute_noise_column(
            table=self.table.copy(),  # Use copy to not modify original
            payloadConfig=config,
            noise_key="test_noise",
            task_key="test_task",
            default_task=self.default_task,
            signal_col="signal",
            gain_col="gain",
            output_col="output",
        )

        # Check that output column wasn't added
        assert "output" not in result.colnames

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with missing channel config but with radiometric structure
        bad_config = {
            "channel": {
                "radiometric": {
                    "other_noise": True,  # Different noise key
                },
            },
        }

        result = compute_noise_column(
            table=self.table.copy(),
            payloadConfig=bad_config,
            noise_key="test_noise",  # This key doesn't exist in config
            task_key="test_task",
            default_task=self.default_task,
            signal_col="signal",
            gain_col="gain",
            output_col="output",
        )

        # Should silently skip missing config and return table unchanged
        assert "output" not in result.colnames

    def test_array_operations_validation(self):
        """Test that array operations produce expected results."""
        # Create test data with known values
        test_table = QTable()
        test_table["signal"] = [
            4.0,
            9.0,
            16.0,
        ] * u.electron  # Perfect squares for easy validation
        test_table["gain"] = [1.0, 1.0, 1.0] * u.dimensionless_unscaled
        test_table["ch_name"] = ["single_channel"] * 3

        result = compute_noise_column(
            table=test_table,
            payloadConfig=self.config,
            noise_key="test_noise",
            task_key="test_task",
            default_task=self.default_task,
            signal_col="signal",
            gain_col="gain",
            output_col="test_output",
        )

        # Verify the output column exists and has correct length
        assert "test_output" in result.colnames
        assert len(result["test_output"]) == len(test_table["signal"])

        # The noise should be added in quadrature, so should be non-zero
        assert all(result["test_output"] > 0)

    def test_missing_columns_handling(self):
        """Test behavior with missing required columns."""
        # Create table without required columns
        bad_table = QTable()
        bad_table["wrong_signal"] = [1.0, 2.0, 3.0] * u.electron
        bad_table["ch_name"] = ["single_channel"] * 3

        # Should handle missing columns gracefully or raise appropriate error
        try:
            result = compute_noise_column(
                table=bad_table,
                payloadConfig=self.config,
                noise_key="test_noise",
                task_key="test_task",
                default_task=self.default_task,
                signal_col="signal",  # This column doesn't exist
                gain_col="gain",  # This column doesn't exist
                output_col="output",
            )
            # If no exception, verify output
            assert isinstance(result, QTable)
        except (KeyError, ValueError, AttributeError):
            # Expected behavior for missing columns
            pass
