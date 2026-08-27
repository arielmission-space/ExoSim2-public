"""
Test suite for ComputeObservationEfficiencyFromDeadTime task.

This module tests the task that computes observation efficiency based on dead time,
which impacts the overall observation efficiency by introducing gaps between
integration periods.
"""

from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

from exosim.tasks.radiometric.compute_observation_efficiency_from_dead_time import (
    ComputeObservationEfficiencyFromDeadTime,
)


class TestComputeObservationEfficiencyFromDeadTime:
    """Test suite for ComputeObservationEfficiencyFromDeadTime task."""

    def test_task_creation(self):
        """Test that the task can be instantiated."""
        task = ComputeObservationEfficiencyFromDeadTime()
        assert task is not None

    def test_task_inheritance(self):
        """Test that the task inherits from ComputeObservationEfficiency."""
        from exosim.tasks.radiometric.compute_observation_efficiency import (
            ComputeObservationEfficiency,
        )

        assert issubclass(
            ComputeObservationEfficiencyFromDeadTime, ComputeObservationEfficiency
        )

    def test_model_with_dead_time_specified(self):
        """Test model method with dead time specified in description."""
        task = ComputeObservationEfficiencyFromDeadTime()

        # Create test data
        radiometric_table = QTable(
            {
                "ch_name": ["test_ch", "test_ch", "other_ch"],
                "integration_time": [1.0, 2.0, 3.0] * u.s,
            }
        )

        description = {"radiometric": {"dead_time": 0.5 * u.s}}

        channel_name = "test_ch"

        # Call the model method
        result = task.model(radiometric_table, description, channel_name)

        # Expected efficiency: t_int / (t_int + t_dead)
        # For t_int = [1.0, 2.0] s and t_dead = 0.5 s:
        # efficiency = [1.0/(1.0+0.5), 2.0/(2.0+0.5)] = [2/3, 4/5] = [0.6667, 0.8]
        expected = np.array([1.0 / 1.5, 2.0 / 2.5])

        # Check that result is array-like and values are correct
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_model_with_dead_time_zero(self):
        """Test model method with zero dead time (perfect efficiency)."""
        task = ComputeObservationEfficiencyFromDeadTime()

        radiometric_table = QTable(
            {
                "ch_name": ["test_ch", "test_ch"],
                "integration_time": [1.0, 2.0] * u.s,
            }
        )

        description = {"radiometric": {"dead_time": 0.0 * u.s}}

        channel_name = "test_ch"

        result = task.model(radiometric_table, description, channel_name)

        # With zero dead time, efficiency should be 1.0 for all apertures
        expected = np.array([1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_model_with_dead_time_equals_integration_time(self):
        """Test model method when dead time equals integration time (50% efficiency)."""
        task = ComputeObservationEfficiencyFromDeadTime()

        radiometric_table = QTable(
            {
                "ch_name": ["test_ch"],
                "integration_time": [1.0] * u.s,
            }
        )

        description = {"radiometric": {"dead_time": 1.0 * u.s}}

        channel_name = "test_ch"

        result = task.model(radiometric_table, description, channel_name)

        # When dead_time = integration_time, efficiency = 0.5
        expected = np.array([0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_model_no_radiometric_section(self):
        """Test model method when radiometric section is missing from description."""
        task = ComputeObservationEfficiencyFromDeadTime()

        radiometric_table = QTable(
            {
                "ch_name": ["test_ch"],
                "integration_time": [2.0] * u.s,
            }
        )

        description = {}  # No radiometric section

        channel_name = "test_ch"

        # Mock the warning method to verify it's called
        with patch.object(task, "warning") as mock_warning:
            result = task.model(radiometric_table, description, channel_name)

            # Should have issued a warning
            mock_warning.assert_called_once_with(
                "No dead time specified in the description. Assuming dead time = 0."
            )

        # With no dead time (defaults to 0), efficiency should be 1.0
        expected = np.array([1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_model_no_dead_time_key(self):
        """Test model method when radiometric section exists but dead_time key is missing."""
        task = ComputeObservationEfficiencyFromDeadTime()

        radiometric_table = QTable(
            {
                "ch_name": ["test_ch"],
                "integration_time": [1.5] * u.s,
            }
        )

        description = {"radiometric": {"other_param": "value"}}  # No dead_time key

        channel_name = "test_ch"

        # Mock the warning method to verify it's called
        with patch.object(task, "warning") as mock_warning:
            result = task.model(radiometric_table, description, channel_name)

            # Should have issued a warning
            mock_warning.assert_called_once_with(
                "No dead time specified in the description. Assuming dead time = 0."
            )

        # With no dead time (defaults to 0), efficiency should be 1.0
        expected = np.array([1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_model_channel_filtering(self):
        """Test that model method correctly filters by channel name."""
        task = ComputeObservationEfficiencyFromDeadTime()

        # Create table with multiple channels
        radiometric_table = QTable(
            {
                "ch_name": ["ch1", "ch2", "ch1", "ch3"],
                "integration_time": [1.0, 2.0, 3.0, 4.0] * u.s,
            }
        )

        description = {"radiometric": {"dead_time": 1.0 * u.s}}

        channel_name = "ch1"

        result = task.model(radiometric_table, description, channel_name)

        # Should only process ch1 entries (integration_time = [1.0, 3.0] s)
        # efficiency = [1.0/(1.0+1.0), 3.0/(3.0+1.0)] = [0.5, 0.75]
        expected = np.array([0.5, 0.75])
        np.testing.assert_array_almost_equal(result, expected)

    def test_model_single_aperture(self):
        """Test model method with single aperture."""
        task = ComputeObservationEfficiencyFromDeadTime()

        radiometric_table = QTable(
            {
                "ch_name": ["test_ch"],
                "integration_time": [4.0] * u.s,
            }
        )

        description = {"radiometric": {"dead_time": 1.0 * u.s}}

        channel_name = "test_ch"

        result = task.model(radiometric_table, description, channel_name)

        # efficiency = 4.0/(4.0+1.0) = 0.8
        expected = np.array([0.8])
        np.testing.assert_array_almost_equal(result, expected)

    def test_model_multiple_apertures_same_integration_time(self):
        """Test model method with multiple apertures having same integration time."""
        task = ComputeObservationEfficiencyFromDeadTime()

        radiometric_table = QTable(
            {
                "ch_name": ["test_ch", "test_ch", "test_ch"],
                "integration_time": [2.0, 2.0, 2.0] * u.s,
            }
        )

        description = {"radiometric": {"dead_time": 0.5 * u.s}}

        channel_name = "test_ch"

        result = task.model(radiometric_table, description, channel_name)

        # All apertures have same integration time, so same efficiency
        # efficiency = 2.0/(2.0+0.5) = 0.8
        expected = np.array([0.8, 0.8, 0.8])
        np.testing.assert_array_almost_equal(result, expected)

    def test_model_dead_time_with_units(self):
        """Test model method with dead time specified in different units."""
        task = ComputeObservationEfficiencyFromDeadTime()

        radiometric_table = QTable(
            {
                "ch_name": ["test_ch"],
                "integration_time": [1.0] * u.s,
            }
        )

        # Dead time in milliseconds
        description = {"radiometric": {"dead_time": 500.0 * u.ms}}

        channel_name = "test_ch"

        result = task.model(radiometric_table, description, channel_name)

        # 500 ms = 0.5 s, so efficiency = 1.0/(1.0+0.5) = 2/3
        expected = np.array([2.0 / 3.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_model_integration_time_with_units(self):
        """Test model method with integration time in different units."""
        task = ComputeObservationEfficiencyFromDeadTime()

        # Integration time in milliseconds
        radiometric_table = QTable(
            {
                "ch_name": ["test_ch"],
                "integration_time": [2000.0] * u.ms,
            }
        )

        description = {"radiometric": {"dead_time": 1.0 * u.s}}

        channel_name = "test_ch"

        result = task.model(radiometric_table, description, channel_name)

        # 2000 ms = 2.0 s, so efficiency = 2.0/(2.0+1.0) = 2/3
        expected = np.array([2.0 / 3.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_model_very_large_dead_time(self):
        """Test model method with very large dead time (low efficiency)."""
        task = ComputeObservationEfficiencyFromDeadTime()

        radiometric_table = QTable(
            {
                "ch_name": ["test_ch"],
                "integration_time": [1.0] * u.s,
            }
        )

        description = {"radiometric": {"dead_time": 99.0 * u.s}}

        channel_name = "test_ch"

        result = task.model(radiometric_table, description, channel_name)

        # efficiency = 1.0/(1.0+99.0) = 0.01
        expected = np.array([0.01])
        np.testing.assert_array_almost_equal(result, expected)

    def test_model_empty_filtered_table(self):
        """Test model method when no apertures match the channel name."""
        task = ComputeObservationEfficiencyFromDeadTime()

        radiometric_table = QTable(
            {
                "ch_name": ["other_ch1", "other_ch2"],
                "integration_time": [1.0, 2.0] * u.s,
            }
        )

        description = {"radiometric": {"dead_time": 0.5 * u.s}}

        channel_name = "test_ch"  # No matching channel

        result = task.model(radiometric_table, description, channel_name)

        # Should return empty array
        assert len(result) == 0

    @pytest.mark.parametrize(
        ("integration_times", "dead_time", "expected_efficiencies"),
        [
            ([1.0], 0.0, [1.0]),  # Zero dead time
            ([1.0], 1.0, [0.5]),  # Equal dead time
            ([2.0], 1.0, [2.0 / 3.0]),  # 2:1 ratio
            ([0.5, 1.0, 2.0], 0.5, [0.5, 2.0 / 3.0, 0.8]),  # Multiple apertures
            ([10.0], 0.1, [10.0 / 10.1]),  # Low dead time impact
        ],
    )
    def test_model_parametrized(
        self, integration_times, dead_time, expected_efficiencies
    ):
        """Parametrized test for various integration time and dead time combinations."""
        task = ComputeObservationEfficiencyFromDeadTime()

        radiometric_table = QTable(
            {
                "ch_name": ["test_ch"] * len(integration_times),
                "integration_time": integration_times * u.s,
            }
        )

        description = {"radiometric": {"dead_time": dead_time * u.s}}

        channel_name = "test_ch"

        result = task.model(radiometric_table, description, channel_name)

        np.testing.assert_array_almost_equal(result, expected_efficiencies, decimal=6)
