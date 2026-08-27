"""
Unit tests for RunConfig functionality and random number generation.

This module contains tests for the ExoSim run configuration management,
including random seed handling, job configuration, and singleton pattern.
"""

import logging
import random

import numpy as np

from exosim.log import Logger, set_log_level
from exosim.utils import RunConfig

set_log_level(logging.DEBUG)


class TestRandom:
    """Test suite for random number generation with RunConfig."""

    def test_seed(self):
        """Test that random seed produces reproducible results."""
        # Generate some random numbers to change state
        a = np.random.uniform(0, 1)
        b = random.uniform(0, 1)

        # Set seed and test reproducibility
        RunConfig.random_seed = 1

        a = np.random.uniform(0, 1)
        b = random.uniform(0, 1)

        # Check expected values with seed=1
        assert a == 0.417022004702574
        assert b == 0.13436424411240122

        # Generate next values and check they're different but reproducible
        a = np.random.uniform(0, 1)
        b = random.uniform(0, 1)

        assert a == 0.7203244934421581
        assert b == 0.8474337369372327


class TestRunConfig:
    """Test suite for RunConfig singleton and configuration management."""

    def test_info(self):
        """Test that stats method executes without error."""
        RunConfig.stats()

    def test_singleton_pattern(self):
        """Test that RunConfig follows singleton pattern."""
        from exosim.utils.run_config import RunConfigInit

        # Multiple instances should return the same object
        instance1 = RunConfigInit()
        instance2 = RunConfigInit()
        assert instance1 is instance2

        # Should be the same as the global RunConfig
        assert instance1 is RunConfig

    def test_random_seed_property(self):
        """Test random seed getter and setter."""
        original_seed = RunConfig.random_seed

        try:
            # Test setting seed
            test_seed = 12345
            RunConfig.random_seed = test_seed
            assert RunConfig.random_seed == test_seed

            # Test that numpy and random modules are seeded
            import random

            import numpy as np

            RunConfig.random_seed = 42
            val1 = np.random.random()
            val2 = random.random()

            # Reset with same seed should give same values
            RunConfig.random_seed = 42
            val1_repeat = np.random.random()
            val2_repeat = random.random()

            assert val1 == val1_repeat
            assert val2 == val2_repeat

        finally:
            # Restore original seed
            RunConfig.random_seed = original_seed

    def test_random_generator_property(self):
        """Test random generator property."""
        original_seed = RunConfig.random_seed

        try:
            # Test with seed set
            RunConfig.random_seed = 100
            rng = RunConfig.random_generator
            assert rng is not None

            # Seed should be incremented after accessing generator
            assert RunConfig.random_seed == 101

            # Test with no seed (None)
            RunConfig.random_seed = None
            rng_no_seed = RunConfig.random_generator
            assert rng_no_seed is not None

        finally:
            RunConfig.random_seed = original_seed

    def test_n_job_property(self):
        """Test n_job getter and setter."""
        from exosim.utils.run_config import total_cpus

        original_n_job = RunConfig.n_job

        try:
            # Test setting a positive value (kept within the machine's CPU count)
            RunConfig.n_job = total_cpus
            assert RunConfig.n_job == total_cpus

            RunConfig.n_job = 1
            assert RunConfig.n_job == 1

            # Test setting a negative value (should use total_cpus + value)
            if total_cpus >= 2:
                RunConfig.n_job = -1
                assert RunConfig.n_job == total_cpus - 1

        finally:
            # Restore original value
            RunConfig.n_job = original_n_job

    def test_chunk_size_property(self):
        """Test chunk_size property."""
        original_chunk_size = RunConfig.chunk_size

        try:
            # Test setting chunk size
            RunConfig.chunk_size = 5
            assert RunConfig.chunk_size == 5

            # Test default value
            from exosim.utils.run_config import RunConfigInit

            new_instance = RunConfigInit()
            assert hasattr(new_instance, "chunk_size")

        finally:
            RunConfig.chunk_size = original_chunk_size

    def test_config_file_list(self):
        """Test config_file_list class variable."""
        # Should be a list
        assert isinstance(RunConfig.config_file_list, list)

        # Test adding items
        original_list = RunConfig.config_file_list.copy()
        try:
            RunConfig.config_file_list.append("test_config.xml")
            assert "test_config.xml" in RunConfig.config_file_list
        finally:
            RunConfig.config_file_list = original_list

    def test_stats_method(self):
        """Test stats method with and without logging."""
        # Test with logging enabled (default)
        stats_with_log = RunConfig.stats(log=True)
        assert isinstance(stats_with_log, dict)

        # Check required keys
        expected_keys = [
            "number of available cpus",
            "number of used cpus",
            "random seed",
            "chunk size (Mb)",
        ]
        for key in expected_keys:
            assert key in stats_with_log

        # Test with logging disabled
        stats_no_log = RunConfig.stats(log=False)
        assert isinstance(stats_no_log, dict)
        assert stats_no_log == stats_with_log  # Should have same content

    def test_dict_method(self):
        """Test custom __dict__ method."""
        config_dict = RunConfig.__dict__()
        assert isinstance(config_dict, dict)

        expected_attrs = ["n_job", "chunk_size", "random_seed", "config_file_list"]
        for attr in expected_attrs:
            assert attr in config_dict
            # Verify the value matches the actual attribute
            assert config_dict[attr] == getattr(RunConfig, attr)

    def test_logger_inheritance(self):
        """Test that RunConfig inherits from Logger."""

        assert isinstance(RunConfig, Logger)

        # Should have logging methods
        assert hasattr(RunConfig, "info")
        assert hasattr(RunConfig, "debug")
        assert hasattr(RunConfig, "warning")
        assert hasattr(RunConfig, "error")
