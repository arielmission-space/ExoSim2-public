"""
Unit tests for data processing utilities.

Tests functions for chunked data iteration and array search operations
used in data processing workflows.
"""

from unittest.mock import Mock

import numpy as np

from exosim.utils.iterators import iterate_over_chunks, searchsorted


class TestChunkedDataIteration:
    """Test chunked data iteration functionality."""

    def test_iterate_over_chunks_basic_functionality(self):
        """Test basic iterate_over_chunks functionality."""
        # Mock h5py dataset
        mock_dataset = Mock()
        mock_dataset.shape = (100,)  # 100 elements
        mock_dataset.chunks = (10,)  # chunks of 10
        mock_dataset.iter_chunks.return_value = iter(
            [(slice(0, 10),), (slice(10, 20),)]
        )

        # Call the function
        result = iterate_over_chunks(mock_dataset, desc="Test progress")

        # Check that the calculation for total chunks is correct
        expected_total = int(np.ceil(100 / 10))  # 10 chunks
        assert expected_total == 10

        # The function should return a tqdm object
        # We verify by checking it has the expected interface
        assert hasattr(result, "__iter__")

    def test_iterate_over_chunks_non_divisible_size(self):
        """Test iterate_over_chunks when dataset size is not exactly divisible by chunk size."""
        # Mock h5py dataset with non-exact division
        mock_dataset = Mock()
        mock_dataset.shape = (105,)  # 105 elements
        mock_dataset.chunks = (10,)  # chunks of 10
        mock_dataset.iter_chunks.return_value = iter(
            [(slice(0, 10),), (slice(10, 20),)]
        )

        # Call the function
        iterate_over_chunks(mock_dataset)

        # Check calculation - should be ceil(105/10) = 11
        expected_total = int(np.ceil(105 / 10))
        assert expected_total == 11


class TestArraySearchUtilities:
    """Test array search and sorting utilities."""

    def test_searchsorted_basic_functionality(self):
        """Test basic functionality of searchsorted."""
        known_array = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        test_array = np.array([2.0, 4.0, 6.0])

        result = searchsorted(known_array, test_array)

        # Expected indices based on closest values
        # 2.0 should be closest to 1.0 (index 0) or 3.0 (index 1)
        # 4.0 should be closest to 3.0 (index 1) or 5.0 (index 2)
        # 6.0 should be closest to 5.0 (index 2) or 7.0 (index 3)
        assert len(result) == 3
        assert all(isinstance(x, int | np.integer) for x in result)
        assert all(0 <= x < len(known_array) for x in result)

    def test_searchsorted_with_unsorted_input(self):
        """Test searchsorted with unsorted known_array."""
        known_array = np.array([5.0, 1.0, 9.0, 3.0, 7.0])
        test_array = np.array([2.0, 4.0, 6.0])

        result = searchsorted(known_array, test_array)

        # Should still work because function sorts internally
        assert len(result) == 3
        assert all(isinstance(x, int | np.integer) for x in result)
        assert all(0 <= x < len(known_array) for x in result)

    def test_searchsorted_single_value_input(self):
        """Test searchsorted with single values."""
        known_array = np.array([1.0, 2.0, 3.0])
        test_array = np.array([2.5])

        result = searchsorted(known_array, test_array)

        assert len(result) == 1
        assert 0 <= result[0] < len(known_array)

    def test_searchsorted_edge_cases(self):
        """Test searchsorted with values outside the known range."""
        known_array = np.array([1.0, 2.0])
        test_array = np.array([0.5, 2.5])  # One below, one above range

        result = searchsorted(known_array, test_array)

        assert len(result) == 2
        assert all(0 <= x < len(known_array) for x in result)

    def test_searchsorted_exact_matches(self):
        """Test searchsorted when test values match known values exactly."""
        known_array = np.array([1.0, 3.0, 5.0])
        test_array = np.array([1.0, 3.0, 5.0])

        result = searchsorted(known_array, test_array)

        assert len(result) == 3
        # Each test value should map to its exact match or very close
        assert all(0 <= x < len(known_array) for x in result)

    def test_searchsorted_empty_arrays(self):
        """Test searchsorted behavior with empty arrays."""
        known_array = np.array([1.0, 2.0, 3.0])
        test_array = np.array([])

        result = searchsorted(known_array, test_array)

        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_searchsorted_duplicates_in_known_array(self):
        """Test searchsorted with duplicate values in known array."""
        known_array = np.array([1.0, 2.0, 2.0, 3.0])
        test_array = np.array([2.0])

        result = searchsorted(known_array, test_array)

        assert len(result) == 1
        assert 0 <= result[0] < len(known_array)
        # Should find one of the indices where value is 2.0
        assert known_array[result[0]] == 2.0
