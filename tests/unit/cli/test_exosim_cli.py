"""
End-to-end tests for the ExoSim CLI interface.

Tests the command-line interface from a user perspective,
verifying that commands work correctly with proper arguments.
"""

import logging
import os
import tempfile
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from exosim.exosim import (
    _plot_focal_plane,
    _plot_ndrs,
    _plot_radiometric,
    _plot_subexposures,
    _set_log,
    _set_threads,
    cli,
    common_options,
)
from exosim.utils import RunConfig


class TestCLIInterface:
    """Test the main CLI interface and commands."""

    def setup_method(self):
        """Set up test fixtures for each test."""
        self.runner = CliRunner()

    def test_cli_version_display(self):
        """Test that CLI version command displays version information."""
        result = self.runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "ExoSim" in result.output
        # Version should be displayed in the format "ExoSim v{version}"

    def test_cli_help_display(self):
        """Test that CLI help command shows available commands."""
        result = self.runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "ExoSim CLI" in result.output

        # Verify all main commands are listed
        expected_commands = ["focalplane", "radiometric", "sub-exposures", "ndrs"]
        for command in expected_commands:
            assert command in result.output

    def test_common_options_functionality(self):
        """Test that common_options decorator adds expected parameters."""

        @common_options
        def dummy_command():
            """Dummy command for testing decorator."""

        # Verify the decorator adds click parameters
        assert hasattr(dummy_command, "__click_params__")
        param_names = [param.name for param in dummy_command.__click_params__]

        expected_params = [
            "plot",
            "logger",
            "debug",
            "numberOfThreads",
            "output",
            "conf",
        ]

        for param in expected_params:
            assert param in param_names, f"Expected parameter '{param}' not found"


class TestFocalPlaneCommand:
    """Test the focal plane command functionality."""

    def setup_method(self):
        """Set up test fixtures for focal plane tests."""
        self.runner = CliRunner()

    @patch("exosim.exosim.recipes.CreateFocalPlane")
    @patch("exosim.exosim._plot_focal_plane")
    def test_focalplane_basic_execution(self, mock_plot, mock_create):
        """Test basic focal plane command without optional parameters."""
        with tempfile.NamedTemporaryFile(suffix=".xml") as temp_config:
            result = self.runner.invoke(
                cli, ["focalplane", "--configuration", temp_config.name]
            )

            assert result.exit_code == 0
            mock_create.assert_called_once()
            mock_plot.assert_not_called()

    @patch("exosim.exosim.recipes.CreateFocalPlane")
    @patch("exosim.exosim._plot_focal_plane")
    def test_focalplane_with_plotting(self, mock_plot, mock_create):
        """Test focal plane command with plotting enabled."""
        with (
            tempfile.NamedTemporaryFile(suffix=".xml") as temp_config,
            tempfile.NamedTemporaryFile(suffix=".h5") as temp_output,
        ):
            result = self.runner.invoke(
                cli,
                [
                    "focalplane",
                    "--configuration",
                    temp_config.name,
                    "--output",
                    temp_output.name,
                    "--plot",
                ],
            )

            assert result.exit_code == 0
            mock_create.assert_called_once()
            mock_plot.assert_called_once_with(temp_output.name, "linear")

    @patch("exosim.exosim.recipes.CreateFocalPlane")
    @patch("exosim.exosim._plot_focal_plane")
    def test_focalplane_with_db_scale(self, mock_plot, mock_create):
        """Test focal plane command with dB plot scale option."""
        with (
            tempfile.NamedTemporaryFile(suffix=".xml") as temp_config,
            tempfile.NamedTemporaryFile(suffix=".h5") as temp_output,
        ):
            result = self.runner.invoke(
                cli,
                [
                    "focalplane",
                    "--configuration",
                    temp_config.name,
                    "--output",
                    temp_output.name,
                    "--plot",
                    "--plot-scale",
                    "dB",
                ],
            )

            assert result.exit_code == 0
            mock_create.assert_called_once()
            mock_plot.assert_called_once_with(temp_output.name, "dB")

    def test_focalplane_missing_config_error(self):
        """Test focal plane command fails with non-existent config file."""
        result = self.runner.invoke(
            cli, ["focalplane", "--configuration", "nonexistent_file.xml"]
        )

        assert result.exit_code != 0
        # Should show error message about missing file


class TestRadiometricCommand:
    """Test the radiometric model command functionality."""

    def setup_method(self):
        """Set up test fixtures for radiometric tests."""
        self.runner = CliRunner()

    @patch("exosim.exosim.recipes.RadiometricModel")
    @patch("exosim.exosim._plot_radiometric")
    def test_radiometric_basic_execution(self, mock_plot, mock_radiometric):
        """Test basic radiometric command without optional parameters."""
        with tempfile.NamedTemporaryFile(suffix=".xml") as temp_config:
            result = self.runner.invoke(
                cli, ["radiometric", "--configuration", temp_config.name]
            )

            assert result.exit_code == 0
            mock_radiometric.assert_called_once()
            mock_plot.assert_not_called()

    @patch("exosim.exosim.recipes.RadiometricModel")
    @patch("exosim.exosim._plot_radiometric")
    def test_radiometric_with_plotting(self, mock_plot, mock_radiometric):
        """Test radiometric command with plotting enabled."""
        with (
            tempfile.NamedTemporaryFile(suffix=".xml") as temp_config,
            tempfile.NamedTemporaryFile(suffix=".h5") as temp_output,
        ):
            result = self.runner.invoke(
                cli,
                [
                    "radiometric",
                    "--configuration",
                    temp_config.name,
                    "--output",
                    temp_output.name,
                    "--plot",
                ],
            )

            assert result.exit_code == 0
            mock_radiometric.assert_called_once()
            mock_plot.assert_called_once_with(temp_output.name)


class TestUtilityFunctions:
    """Test CLI utility functions with comprehensive coverage."""

    @patch("exosim.exosim.set_log_level")
    @patch("exosim.exosim.add_log_file")
    def test_set_log_debug_only(self, mock_add_log, mock_set_level):
        """Test _set_log with debug mode only."""
        _set_log(debug=True, log=False, output=None)

        mock_set_level.assert_called_once_with(logging.DEBUG)
        mock_add_log.assert_not_called()

    @patch("exosim.exosim.set_log_level")
    @patch("exosim.exosim.add_log_file")
    def test_set_log_with_file_enabled(self, mock_add_log, mock_set_level):
        """Test _set_log with log file enabled."""
        _set_log(debug=False, log=True, output=None)

        mock_set_level.assert_not_called()
        mock_add_log.assert_called_once_with(fname="exosim.log")

    @patch("exosim.exosim.set_log_level")
    @patch("exosim.exosim.add_log_file")
    def test_set_log_with_output_directory(self, mock_add_log, mock_set_level):
        """Test _set_log with specific output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Ensure temp_dir is a string, not bytes
            temp_dir = str(temp_dir)
            output_file = os.path.join(temp_dir, "output.h5")
            expected_log = os.path.join(temp_dir, "exosim.log")

            _set_log(debug=False, log=True, output=output_file)

            mock_add_log.assert_called_once_with(fname=expected_log)

    @patch("exosim.exosim.set_log_level")
    @patch("exosim.exosim.add_log_file")
    def test_set_log_permission_error_fallback(self, mock_add_log, mock_set_level):
        """Test _set_log fallback when permission error occurs."""
        # First call raises PermissionError, second call succeeds
        mock_add_log.side_effect = [PermissionError(), None]

        _set_log(debug=False, log=True, output="/restricted/path/output.h5")

        assert mock_add_log.call_count == 2
        mock_add_log.assert_any_call(fname="/restricted/path/exosim.log")
        mock_add_log.assert_any_call(fname="exosim.log")

    def test_set_threads_functionality(self):
        """Test _set_threads function with different values."""
        original_value = getattr(RunConfig, "n_job", None)

        try:
            _set_threads(4)
            assert RunConfig.n_job == 4

            _set_threads(None)
            assert RunConfig.n_job == 4  # Should remain unchanged when None

            _set_threads(8)
            assert RunConfig.n_job == 8

        finally:
            # Restore original value
            if original_value is not None:
                RunConfig.n_job = original_value


class TestSubExposuresCommand:
    """Test the subexposures command functionality."""

    def setup_method(self):
        """Set up test fixtures for subexposures tests."""
        self.runner = CliRunner()

    @patch("exosim.exosim.recipes.CreateSubExposures")
    @patch("exosim.exosim._plot_subexposures")
    def test_subexposures_basic_execution(self, mock_plot, mock_create):
        """Test basic subexposures command execution."""
        with (
            tempfile.NamedTemporaryFile(suffix=".xml") as temp_config,
            tempfile.NamedTemporaryFile(suffix=".h5") as temp_input,
        ):
            result = self.runner.invoke(
                cli,
                [
                    "subexposures",
                    "--configuration",
                    temp_config.name,
                    "--input",
                    temp_input.name,
                ],
            )

            assert result.exit_code == 0
            mock_create.assert_called_once()
            mock_plot.assert_not_called()

    @patch("exosim.exosim.recipes.CreateSubExposures")
    @patch("exosim.exosim._plot_subexposures")
    def test_subexposures_with_plotting(self, mock_plot, mock_create):
        """Test subexposures command with plotting enabled."""
        with (
            tempfile.NamedTemporaryFile(suffix=".xml") as temp_config,
            tempfile.NamedTemporaryFile(suffix=".h5") as temp_input,
            tempfile.NamedTemporaryFile(suffix=".h5") as temp_output,
        ):
            result = self.runner.invoke(
                cli,
                [
                    "subexposures",
                    "--configuration",
                    temp_config.name,
                    "--input",
                    temp_input.name,
                    "--output",
                    temp_output.name,
                    "--plot",
                    "--chunk-size",
                    "4.0",
                ],
            )

            assert result.exit_code == 0
            mock_create.assert_called_once()
            mock_plot.assert_called_once_with(temp_output.name)
            assert RunConfig.chunk_size == 4.0

    def test_subexposures_missing_input_error(self):
        """Test subexposures command with missing input file."""
        with tempfile.NamedTemporaryFile(suffix=".xml") as temp_config:
            result = self.runner.invoke(
                cli,
                [
                    "subexposures",
                    "--configuration",
                    temp_config.name,
                    "--input",
                    "nonexistent.h5",
                ],
            )

            assert result.exit_code != 0


class TestNDRsCommand:
    """Test the ndrs command functionality."""

    def setup_method(self):
        """Set up test fixtures for NDRs tests."""
        self.runner = CliRunner()

    @patch("exosim.exosim.recipes.CreateNDRs")
    @patch("exosim.exosim._plot_ndrs")
    def test_ndrs_basic_execution(self, mock_plot, mock_create):
        """Test basic ndrs command execution."""
        with (
            tempfile.NamedTemporaryFile(suffix=".xml") as temp_config,
            tempfile.NamedTemporaryFile(suffix=".h5") as temp_input,
        ):
            result = self.runner.invoke(
                cli,
                [
                    "ndrs",
                    "--configuration",
                    temp_config.name,
                    "--input",
                    temp_input.name,
                ],
            )

            assert result.exit_code == 0
            mock_create.assert_called_once()
            mock_plot.assert_not_called()

    @patch("exosim.exosim.recipes.CreateNDRs")
    @patch("exosim.exosim._plot_ndrs")
    def test_ndrs_with_plotting(self, mock_plot, mock_create):
        """Test ndrs command with plotting enabled."""
        with (
            tempfile.NamedTemporaryFile(suffix=".xml") as temp_config,
            tempfile.NamedTemporaryFile(suffix=".h5") as temp_input,
            tempfile.NamedTemporaryFile(suffix=".h5") as temp_output,
        ):
            result = self.runner.invoke(
                cli,
                [
                    "ndrs",
                    "--configuration",
                    temp_config.name,
                    "--input",
                    temp_input.name,
                    "--output",
                    temp_output.name,
                    "--plot",
                    "--chunk-size",
                    "3.5",
                ],
            )

            assert result.exit_code == 0
            mock_create.assert_called_once()
            mock_plot.assert_called_once_with(temp_output.name)
            assert RunConfig.chunk_size == 3.5


class TestCommandParameterValidation:
    """Test command parameter validation and error handling."""

    def setup_method(self):
        """Set up test fixtures for parameter validation tests."""
        self.runner = CliRunner()

    def test_missing_required_configuration(self):
        """Test that commands fail without required configuration parameter."""
        commands = ["focalplane", "radiometric"]

        for command in commands:
            result = self.runner.invoke(cli, [command])
            assert result.exit_code != 0
            # Should show error about missing required option

    def test_invalid_plot_scale_parameter(self):
        """Test focal plane command with invalid plot scale."""
        with tempfile.NamedTemporaryFile(suffix=".xml") as temp_config:
            result = self.runner.invoke(
                cli,
                [
                    "focalplane",
                    "--configuration",
                    temp_config.name,
                    "--plot-scale",
                    "invalid_scale",
                ],
            )

            assert result.exit_code != 0
            # Should show error about invalid plot scale choice


class TestPlotFunctions:
    """Test plot functions for CLI commands."""

    @patch("exosim.plots.FocalPlanePlotter")
    @patch("os.makedirs")
    def test_plot_focal_plane_function(self, mock_makedirs, mock_plotter_class):
        """Test _plot_focal_plane function with linear scale."""
        mock_plotter = MagicMock()
        mock_plotter_class.return_value = mock_plotter

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = str(temp_dir)
            output_file = os.path.join(temp_dir, "output.h5")

            _plot_focal_plane(output_file, "linear")

            # Check that plotter was created with correct input
            mock_plotter_class.assert_called_once_with(input=output_file)

            # Check that plot methods were called
            mock_plotter.plot_focal_plane.assert_called_once_with(
                time_step=0, scale="linear"
            )
            mock_plotter.plot_efficiency.assert_called_once()

            # Check that save methods were called
            assert mock_plotter.save_fig.call_count == 2

            # Check that plots directory was created
            mock_makedirs.assert_called_once()

    @patch("exosim.plots.FocalPlanePlotter")
    @patch("os.makedirs")
    def test_plot_focal_plane_db_scale(self, mock_makedirs, mock_plotter_class):
        """Test _plot_focal_plane function with dB scale."""
        mock_plotter = MagicMock()
        mock_plotter_class.return_value = mock_plotter

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = str(temp_dir)
            output_file = os.path.join(temp_dir, "output.h5")

            _plot_focal_plane(output_file, "dB")

            mock_plotter.plot_focal_plane.assert_called_once_with(
                time_step=0, scale="dB"
            )

    @patch("exosim.plots.RadiometricPlotter")
    @patch("os.makedirs")
    def test_plot_radiometric_function(self, mock_makedirs, mock_plotter_class):
        """Test _plot_radiometric function."""
        mock_plotter = MagicMock()
        mock_plotter_class.return_value = mock_plotter

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = str(temp_dir)
            output_file = os.path.join(temp_dir, "output.h5")

            _plot_radiometric(output_file)

            mock_plotter_class.assert_called_once_with(input=output_file)
            mock_plotter.plot_table.assert_called_once()
            mock_plotter.plot_apertures.assert_called_once()
            assert mock_plotter.save_fig.call_count == 2

    @patch("exosim.plots.SubExposuresPlotter")
    @patch("os.makedirs")
    def test_plot_subexposures_function(self, mock_makedirs, mock_plotter_class):
        """Test _plot_subexposures function."""
        mock_plotter = MagicMock()
        mock_plotter_class.return_value = mock_plotter

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = str(temp_dir)
            output_file = os.path.join(temp_dir, "output.h5")
            expected_plot_dir = os.path.join(temp_dir, "plots")

            _plot_subexposures(output_file)

            mock_plotter_class.assert_called_once_with(input=output_file)
            mock_plotter.plot.assert_called_once_with(expected_plot_dir)

    @patch("exosim.plots.NDRsPlotter")
    @patch("os.makedirs")
    def test_plot_ndrs_function(self, mock_makedirs, mock_plotter_class):
        """Test _plot_ndrs function."""
        mock_plotter = MagicMock()
        mock_plotter_class.return_value = mock_plotter

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = str(temp_dir)
            output_file = os.path.join(temp_dir, "output.h5")
            expected_plot_dir = os.path.join(temp_dir, "plots")

            _plot_ndrs(output_file)

            mock_plotter_class.assert_called_once_with(input=output_file)
            mock_plotter.plot.assert_called_once_with(expected_plot_dir)


class TestMainEntryPoint:
    """Test the main entry point functionality."""

    def test_main_entry_point_exists(self):
        """Test that main entry point structure exists and is callable."""
        import exosim.exosim

        # Test that the cli function is callable
        assert callable(exosim.exosim.cli)

        # Test that the file has the main check
        import inspect

        source = inspect.getsource(exosim.exosim)
        assert 'if __name__ == "__main__"' in source
        assert "cli()" in source


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures for error handling tests."""
        self.runner = CliRunner()

    def test_invalid_plot_scale_error(self):
        """Test handling of invalid plot scale option."""
        with tempfile.NamedTemporaryFile(suffix=".xml") as temp_config:
            result = self.runner.invoke(
                cli,
                [
                    "focalplane",
                    "--configuration",
                    temp_config.name,
                    "--plot-scale",
                    "invalid",
                ],
            )

            assert result.exit_code != 0

    @patch("exosim.exosim.recipes.CreateFocalPlane")
    def test_command_with_exception(self, mock_create):
        """Test command behavior when underlying recipe raises exception."""
        mock_create.side_effect = Exception("Test exception")

        with tempfile.NamedTemporaryFile(suffix=".xml") as temp_config:
            result = self.runner.invoke(
                cli, ["focalplane", "--configuration", temp_config.name]
            )

            # Command should fail but not crash
            assert result.exit_code != 0
            assert (
                "Test exception" in str(result.output) or result.exception is not None
            )


class TestIntegrationScenarios:
    """Test integration scenarios with multiple CLI options."""

    def setup_method(self):
        """Set up test fixtures for integration tests."""
        self.runner = CliRunner()

    @patch("exosim.exosim.recipes.CreateFocalPlane")
    @patch("exosim.exosim._plot_focal_plane")
    @patch("exosim.exosim._set_log")
    @patch("exosim.exosim._set_threads")
    def test_focalplane_all_options_enabled(
        self, mock_threads, mock_log, mock_plot, mock_create
    ):
        """Test focalplane command with all options enabled."""
        with (
            tempfile.NamedTemporaryFile(suffix=".xml") as temp_config,
            tempfile.NamedTemporaryFile(suffix=".h5") as temp_output,
        ):
            result = self.runner.invoke(
                cli,
                [
                    "focalplane",
                    "--configuration",
                    temp_config.name,
                    "--output",
                    temp_output.name,
                    "--nThreads",
                    "8",
                    "--debug",
                    "--logger",
                    "--plot",
                    "--plot-scale",
                    "dB",
                ],
            )

            assert result.exit_code == 0
            mock_create.assert_called_once()
            mock_log.assert_called_once()
            mock_threads.assert_called_once()
            mock_plot.assert_called_once()

    @patch("exosim.exosim.recipes.RadiometricModel")
    @patch("exosim.exosim._plot_radiometric")
    @patch("exosim.exosim._set_log")
    @patch("exosim.exosim._set_threads")
    def test_radiometric_full_integration(
        self, mock_threads, mock_log, mock_plot, mock_radiometric
    ):
        """Test radiometric command with full integration options."""
        with (
            tempfile.NamedTemporaryFile(suffix=".xml") as temp_config,
            tempfile.NamedTemporaryFile(suffix=".h5") as temp_output,
        ):
            result = self.runner.invoke(
                cli,
                [
                    "radiometric",
                    "--configuration",
                    temp_config.name,
                    "--output",
                    temp_output.name,
                    "--nThreads",
                    "4",
                    "--debug",
                    "--logger",
                    "--plot",
                ],
            )

            assert result.exit_code == 0
            mock_radiometric.assert_called_once()
            mock_log.assert_called_once()
            mock_threads.assert_called_once()
            mock_plot.assert_called_once()
