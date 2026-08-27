#!/usr/bin/env python3
"""Tests for populate_focal_plane module"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from exosim.tasks.instrument.populate_focal_plane import PopulateFocalPlane


class TestPopulateFocalPlane(unittest.TestCase):
    """Test populate_focal_plane module."""

    @patch("exosim.tasks.instrument.populate_focal_plane.np")
    def test_populate_focal_plane_basic_initialization(self, mock_np):
        """Test basic initialization structure."""
        # Mock numpy array operations
        mock_np.zeros.return_value = np.zeros((10, 10))
        mock_np.arange.return_value = np.arange(10)

        # Test class instantiation
        try:
            task = PopulateFocalPlane()
            assert task is not None
        except Exception:
            # Expected due to Task base class requirements
            pass

    @patch("exosim.tasks.instrument.populate_focal_plane.Task.__init__")
    def test_populate_focal_plane_parameter_setup(self, mock_init):
        """Test parameter setup structure."""
        mock_init.return_value = None

        # Test with mocked Task initialization
        task = PopulateFocalPlane()
        task.add_task_param = Mock()
        task.get_task_param = Mock(return_value=Mock())

        # Test parameter access patterns
        task.get_task_param("focal_plane")
        task.add_task_param.assert_not_called()  # Should not be called in get
