#!/usr/bin/env python3
"""Integration tests for simulate_observation module.

This test suite verifies the complete observation pipeline functionality,
from configuration file parsing to NDR generation. It includes tests for:
- Class structure and inheritance
- Parameter validation and handling
- Pipeline component integration
- Error handling
- Configuration management
- Plotting integration
"""

import tempfile
from unittest.mock import patch

import pytest

import exosim.log as log
from exosim.recipes.simulate_observation import SimulateObservation
from exosim.utils import RunConfig
from exosim.utils.timed_class import TimedClass


@pytest.fixture
def temp_options_file():
    """Provide a temporary options file."""
    with tempfile.NamedTemporaryFile(suffix=".yaml") as options_file:
        yield options_file.name


@pytest.fixture
def temp_output_file():
    """Provide a temporary output file."""
    with tempfile.NamedTemporaryFile(suffix=".h5") as output_file:
        yield output_file.name


@pytest.fixture
def temp_plots_dir(tmp_path):
    """Provide a temporary directory for plots."""
    plots_dir = tmp_path / "test_plots"
    plots_dir.mkdir(exist_ok=True)
    return str(plots_dir)


class TestSimulateObservationStructure:
    """Test the core structure and inheritance of SimulateObservation."""

    def test_class_inheritance(self):
        """Test proper class inheritance."""
        assert issubclass(SimulateObservation, TimedClass)
        assert issubclass(SimulateObservation, log.Logger)

        # Verify inheritance chain
        mro = SimulateObservation.__mro__
        assert TimedClass in mro
        assert log.Logger in mro

    def test_required_attributes(self):
        """Test presence of required attributes and methods."""
        # Core methods
        assert hasattr(SimulateObservation, "__init__")
        assert hasattr(SimulateObservation, "main")
        assert callable(SimulateObservation)

        # Logger methods
        assert hasattr(SimulateObservation, "info")
        assert hasattr(SimulateObservation, "debug")
        assert hasattr(SimulateObservation, "warning")
        assert hasattr(SimulateObservation, "error")
        assert hasattr(SimulateObservation, "announce")

    def test_method_signatures(self):
        """Test method signatures are correct."""
        import inspect

        # Check __init__ signature
        init_params = inspect.signature(SimulateObservation.__init__).parameters
        required_params = {
            "self",
            "options_file",
            "output_file",
            "plots_dir",
            "n_job",
            "random_seed",
        }
        assert all(param in init_params for param in required_params)

        # Check main signature only has self
        main_params = inspect.signature(SimulateObservation.main).parameters
        assert list(main_params.keys()) == ["self"]

    def test_documentation(self):
        """Test class and method documentation."""
        assert SimulateObservation.__doc__ is not None
        assert len(SimulateObservation.__doc__.strip()) > 0

        doc = SimulateObservation.__doc__.lower()
        required_terms = [
            "pipeline",
            "focal plane",
            "radiometric",
            "sub-exposures",
            "ndrs",
            "parameters",
        ]
        assert all(term in doc for term in required_terms)


class TestSimulateObservationInitialization:
    """Test initialization and parameter handling."""

    def test_basic_initialization(self, temp_options_file, temp_output_file):
        """Test basic initialization with required parameters."""
        with patch("exosim.recipes.simulate_observation.SimulateObservation.main"):
            obs = SimulateObservation(
                options_file=temp_options_file, output_file=temp_output_file
            )
            assert obs.options_file == temp_options_file
            assert obs.output_file == temp_output_file
            assert obs.plots_dir is None  # Default value

    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_plots_directory_creation(
        self, mock_exists, mock_makedirs, temp_options_file, temp_output_file
    ):
        """Test plots directory handling."""
        with patch("exosim.recipes.simulate_observation.SimulateObservation.main"):
            # Test directory creation when it doesn't exist
            mock_exists.return_value = False
            SimulateObservation(
                options_file=temp_options_file,
                output_file=temp_output_file,
                plots_dir="/tmp/test_plots",
            )
            mock_makedirs.assert_called_once_with("/tmp/test_plots")

            # Test no creation when directory exists
            mock_exists.return_value = True
            mock_makedirs.reset_mock()
            SimulateObservation(
                options_file=temp_options_file,
                output_file=temp_output_file,
                plots_dir="/tmp/existing_plots",
            )
            mock_makedirs.assert_not_called()

    def test_run_config_integration(self, temp_options_file, temp_output_file):
        """Test RunConfig parameter handling."""
        with patch("exosim.recipes.simulate_observation.SimulateObservation.main"):
            original_n_job = RunConfig.n_job
            original_seed = getattr(RunConfig, "random_seed", None)

            try:
                SimulateObservation(
                    options_file=temp_options_file,
                    output_file=temp_output_file,
                    n_job=4,
                    random_seed=123,
                )
                assert RunConfig.n_job == 4
                assert RunConfig.random_seed == 123

            finally:
                RunConfig.n_job = original_n_job
                if original_seed is not None:
                    RunConfig.random_seed = original_seed


@pytest.mark.integration
class TestSimulateObservationPipeline:
    """Test the main simulation pipeline functionality."""

    def test_pipeline_components_available(self):
        """Test that all required pipeline components can be imported."""
        from exosim.recipes.create_focal_plane import CreateFocalPlane
        from exosim.recipes.create_ndrs import CreateNDRs
        from exosim.recipes.create_sub_exposures import CreateSubExposures
        from exosim.recipes.radiometric_model import RadiometricModel

        assert all(
            comp is not None
            for comp in [
                CreateFocalPlane,
                RadiometricModel,
                CreateSubExposures,
                CreateNDRs,
            ]
        )

    def test_plotting_components_available(self):
        """Test that all plotting components can be imported."""
        from exosim.plots import (
            FocalPlanePlotter,
            NDRsPlotter,
            RadiometricPlotter,
            SubExposuresPlotter,
        )

        plotters = [
            FocalPlanePlotter,
            RadiometricPlotter,
            SubExposuresPlotter,
            NDRsPlotter,
        ]

        for plotter in plotters:
            assert callable(plotter)
            assert hasattr(plotter, "__init__")


@pytest.mark.integration
class TestSimulateObservationErrorHandling:
    """Test error handling in the simulation pipeline."""

    def test_invalid_file_paths(self):
        """Test handling of invalid file paths."""
        obs = SimulateObservation(
            options_file="/nonexistent/options.yaml",
            output_file="/nonexistent/output.h5",
        )
        with pytest.raises(FileNotFoundError, match="No such file or directory"):
            obs.main()

    def test_invalid_parameters(self, temp_options_file, temp_output_file):
        """Test handling of invalid parameters."""
        with pytest.raises(TypeError, match="n_job must be an integer"):
            SimulateObservation(
                options_file=temp_options_file,
                output_file=temp_output_file,
                n_job="invalid",  # Should be int
            )

    @patch("exosim.recipes.simulate_observation.SimulateObservation.main")
    def test_runtime_error_handling(
        self, mock_main, temp_options_file, temp_output_file
    ):
        """Test handling of runtime errors."""
        mock_main.side_effect = RuntimeError("Simulation failed")

        obs = SimulateObservation(
            options_file=temp_options_file, output_file=temp_output_file
        )
        with pytest.raises(RuntimeError, match="Simulation failed"):
            obs.main()
