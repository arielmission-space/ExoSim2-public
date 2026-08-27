#!/usr/bin/env python3
"""Tests for constant_dark_current_noise module"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from astropy import units as u

from exosim.tasks.detector.add_constant_dark_current import AddConstantDarkCurrent


class TestConstantDarkCurrentNoise(unittest.TestCase):
    """Test constant_dark_current_noise module."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        task = AddConstantDarkCurrent()
        assert task is not None

    @patch("exosim.tasks.detector.add_constant_dark_current.check_units")
    @patch("exosim.tasks.detector.add_constant_dark_current.iterate_over_chunks")
    def test_add_dark_current_mock_numpy(self, mock_iterate_chunks, mock_check_units):
        """Test dark current addition with mocked numpy."""
        # Set up mocks
        task = AddConstantDarkCurrent()

        # Mock dark current
        dc_array = np.array([0.1], dtype=np.float64)
        mock_check_units.return_value = dc_array * u.ct / u.s
        mock_dc = Mock()
        mock_dc.astype = Mock(return_value=dc_array)
        mock_dc.unit = u.ct / u.s

        # Mock iterate_over_chunks to yield a single chunk covering the whole dataset
        mock_iterate_chunks.return_value = [
            (slice(0, 3, None), slice(None, None, None), slice(None, None, None))
        ]

        # Set up mock signal
        mock_signal = Mock()
        mock_signal.dataset = np.zeros((3, 10, 10))  # 3 time steps, 10x10 pixels
        mock_signal.output = Mock()
        mock_signal.output.flush = Mock()

        # Set up mock parameters
        mock_parameters = {"detector": {"dc_mean": mock_dc}}

        # Set up mock integration times as a Quantity array
        mock_times = np.array([1.0, 2.0, 3.0]) * u.s

        # Execute task
        task(
            subexposures=mock_signal,
            parameters=mock_parameters,
            integration_times=mock_times,
        )

        # Verify the interactions
        mock_dc.astype.assert_called_once()
        mock_check_units.assert_called_once_with(dc_array, "ct/s")
        mock_signal.output.flush.assert_called()
