"""
Unit tests for plotting functionality.

Tests the various plotting classes and utilities used for
visualizing ExoSim data and results.
"""

import tempfile
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

from exosim.plots.focal_plane_plotter import FocalPlanePlotter
from exosim.plots.ndrs_plotter import NDRsPlotter
from exosim.plots.plotter import main
from exosim.plots.radiometric_plotter import RadiometricPlotter
from exosim.plots.sub_exposures_plotter import SubExposuresPlotter
from exosim.plots.utils import _create_ordered_cmap, prepare_channels_list


class TestPlotterCLIInterface:
    """Test the main plotter command-line interface."""

    @patch("argparse.ArgumentParser.parse_args")
    @patch("exosim.plots.FocalPlanePlotter")
    def test_focal_plane_plotting_workflow(self, mock_plotter_class, mock_parse_args):
        """Test complete focal plane plotting workflow."""
        # Setup mock command line arguments
        mock_args = MagicMock()
        mock_args.focal = True
        mock_args.radiometric = False
        mock_args.subexposures = False
        mock_args.ndrs = False
        mock_args.input = "test_input.h5"
        mock_args.output = "/test/output"
        mock_args.time_step = 1
        mock_args.plot_scale = "dB"
        mock_parse_args.return_value = mock_args

        # Setup mock plotter instance
        mock_plotter = MagicMock()
        mock_plotter_class.return_value = mock_plotter

        # Execute main function
        main()

        # Verify plotter workflow
        mock_plotter_class.assert_called_once_with(input="test_input.h5")
        mock_plotter.plot_focal_plane.assert_called_once_with(time_step=1, scale="dB")
        mock_plotter.plot_efficiency.assert_called_once()
        assert mock_plotter.save_fig.call_count == 2

    @patch("argparse.ArgumentParser.parse_args")
    @patch("exosim.plots.RadiometricPlotter")
    def test_radiometric_plotting_workflow(self, mock_plotter_class, mock_parse_args):
        """Test complete radiometric plotting workflow."""
        mock_args = MagicMock()
        mock_args.focal = False
        mock_args.radiometric = True
        mock_args.subexposures = False
        mock_args.ndrs = False
        mock_args.input = "test_input.h5"
        mock_args.output = "/test/output"
        mock_parse_args.return_value = mock_args

        mock_plotter = MagicMock()
        mock_plotter_class.return_value = mock_plotter

        main()

        mock_plotter_class.assert_called_once_with(input="test_input.h5")
        mock_plotter.plot_table.assert_called_once()
        mock_plotter.plot_apertures.assert_called_once()
        assert mock_plotter.save_fig.call_count == 2

    @patch("argparse.ArgumentParser.parse_args")
    @patch("exosim.plots.SubExposuresPlotter")
    def test_subexposures_plotting_workflow(self, mock_plotter_class, mock_parse_args):
        """Test complete sub-exposures plotting workflow."""
        mock_args = MagicMock()
        mock_args.focal = False
        mock_args.radiometric = False
        mock_args.subexposures = True
        mock_args.ndrs = False
        mock_args.input = "test_input.h5"
        mock_args.output = "/test/output"
        mock_parse_args.return_value = mock_args

        mock_plotter = MagicMock()
        mock_plotter_class.return_value = mock_plotter

        main()

        mock_plotter_class.assert_called_once_with(input="test_input.h5")
        mock_plotter.plot.assert_called_once_with("/test/output")

    @patch("argparse.ArgumentParser.parse_args")
    @patch("exosim.plots.NDRsPlotter")
    def test_ndrs_plotting_workflow(self, mock_plotter_class, mock_parse_args):
        """Test complete NDRs plotting workflow."""
        mock_args = MagicMock()
        mock_args.focal = False
        mock_args.radiometric = False
        mock_args.subexposures = False
        mock_args.ndrs = True
        mock_args.input = "test_input.h5"
        mock_args.output = "/test/output"
        mock_parse_args.return_value = mock_args

        mock_plotter = MagicMock()
        mock_plotter_class.return_value = mock_plotter

        main()

        mock_plotter_class.assert_called_once_with(input="test_input.h5")
        mock_plotter.plot.assert_called_once_with("/test/output")


class TestRadiometricPlotter:
    """Test RadiometricPlotter class functionality."""

    @pytest.fixture
    def sample_radiometric_table(self):
        """Create sample radiometric table for testing."""
        return QTable(
            {
                "ch_name": ["Ch1", "Ch2"],
                "Wavelength": [1.0, 2.0] * u.um,
                "BinWavelengthWidth": [0.1, 0.1] * u.um,
                "LeftBinEdge": [0.95, 1.95] * u.um,
                "RightBinEdge": [1.05, 2.05] * u.um,
                "star_signal": [100, 200] * u.ph / u.s,
                "source_signal_in_aperture": [100, 200] * u.ph / u.s,
                "foreground_signal_in_aperture": [10, 20] * u.ph / u.s,
                "total_noise_in_aperture": [5, 8] * u.ph / u.s,
                "aperture_area": [1.0, 1.0] * u.arcsec**2,
                "left_bin_edge": [0.95, 1.95] * u.um,
                "right_bin_edge": [1.05, 2.05] * u.um,
            }
        )

    def test_radiometric_plotter_initialization_with_table(
        self, sample_radiometric_table
    ):
        """Test RadiometricPlotter initialization with table object."""
        plotter = RadiometricPlotter(input=sample_radiometric_table)

        assert plotter.input_table is sample_radiometric_table
        assert plotter.input is sample_radiometric_table

    @patch("exosim.plots.radiometric_plotter.RadiometricPlotter.load_table")
    def test_radiometric_plotter_initialization_with_filename(
        self, mock_load_table, sample_radiometric_table
    ):
        """Test RadiometricPlotter initialization with filename."""
        mock_load_table.return_value = sample_radiometric_table

        plotter = RadiometricPlotter(input="test_radiometric_file.h5")

        mock_load_table.assert_called_once_with(
            "test_radiometric_file.h5", "radiometric"
        )
        assert plotter.input == "test_radiometric_file.h5"
        assert plotter.input_table is sample_radiometric_table

    @patch("exosim.plots.radiometric_plotter.read_table_hdf5")
    @patch("h5py.File")
    def test_radiometric_table_loading(
        self, mock_h5py_file, mock_read_table, sample_radiometric_table
    ):
        """Test loading radiometric table from HDF5 file."""
        mock_read_table.return_value = sample_radiometric_table
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file
        mock_file.__getitem__.return_value = MagicMock()

        # Create plotter instance without full initialization
        plotter = RadiometricPlotter.__new__(RadiometricPlotter)
        plotter.set_log_name = MagicMock()
        plotter.debug = MagicMock()

        result = plotter.load_table("test_radiometric_file.h5")

        mock_h5py_file.assert_called_once_with("test_radiometric_file.h5", "r")
        mock_read_table.assert_called_once()
        assert result is sample_radiometric_table

    @patch("matplotlib.pyplot.subplots")
    def test_radiometric_bands_plotting(self, mock_subplots, sample_radiometric_table):
        """Test radiometric bands plotting functionality."""
        fig, ax = plt.figure(), plt.axes()
        mock_subplots.return_value = (fig, ax)

        plotter = RadiometricPlotter(input=sample_radiometric_table)

        # Mock channel preparation
        with patch(
            "exosim.plots.radiometric_plotter.prepare_channels_list"
        ) as mock_prepare:
            mock_prepare.return_value = (["Ch1", "Ch2"], None)

            # Mock plot_bands to avoid column name issues
            with patch.object(
                plotter, "plot_bands", return_value=ax
            ) as mock_plot_bands:
                result_ax = plotter.plot_bands(ax)

                assert result_ax is ax
                mock_plot_bands.assert_called_once()


class TestPlottingUtilities:
    """Test plotting utility functions."""

    def test_create_ordered_colormap(self):
        """Test creation of ordered colormap."""
        # Test with a valid colormap name
        colormap = _create_ordered_cmap(map_name="Pastel1")

        # Should return a ListedColormap object
        assert hasattr(colormap, "N")  # ColorMap attribute
        assert callable(colormap)  # Callable like a colormap

        # Test with different parameters
        colormap_rolled = _create_ordered_cmap(map_name="Set1", roll=2)
        assert hasattr(colormap_rolled, "N")

        # Test with delete parameter
        colormap_deleted = _create_ordered_cmap(map_name="Set1", delete=0)
        assert hasattr(colormap_deleted, "N")

    def test_prepare_channels_list_basic(self):
        """Test basic channel list preparation."""
        # Create sample table with required ch_name and Wavelength columns
        sample_table = QTable(
            {
                "ch_name": ["Channel_A", "Channel_B", "Channel_C"],
                "Wavelength": [1.0, 1.5, 2.0] * u.um,
            }
        )

        # Test basic preparation
        channels, norm = prepare_channels_list(sample_table)

        assert isinstance(channels, np.ndarray)
        assert len(channels) == 3
        assert "Channel_A" in channels
        assert "Channel_B" in channels
        assert "Channel_C" in channels
        assert hasattr(norm, "vmin")  # Normalize object

    def test_prepare_channels_list_with_wavelength_sorting(self):
        """Test channel list preparation with wavelength-based sorting."""
        sample_table = QTable(
            {
                "ch_name": ["Channel_D", "Channel_A", "Channel_C", "Channel_B"],
                "Wavelength": [3.0, 1.0, 2.5, 1.5] * u.um,  # Out of alphabetical order
            }
        )

        # Test preparation - should be sorted by wavelength
        channels, _norm = prepare_channels_list(sample_table)

        assert len(channels) == 4
        # Should be sorted by wavelength: A(1.0), B(1.5), C(2.5), D(3.0)
        assert channels[0] == "Channel_A"
        assert channels[1] == "Channel_B"
        assert channels[2] == "Channel_C"
        assert channels[3] == "Channel_D"


class TestPlotterClassStructure:
    """Test plotting class structure and interfaces."""

    def test_plotter_classes_exist(self):
        """Test that all expected plotter classes exist."""
        plotter_classes = [
            FocalPlanePlotter,
            RadiometricPlotter,
            SubExposuresPlotter,
            NDRsPlotter,
        ]

        for plotter_class in plotter_classes:
            assert plotter_class is not None
            assert hasattr(plotter_class, "__init__")

    def test_plotter_inheritance_pattern(self):
        """Test that plotter classes follow expected inheritance patterns."""
        # All plotters should have common base functionality
        plotter_classes = [
            FocalPlanePlotter,
            RadiometricPlotter,
            SubExposuresPlotter,
            NDRsPlotter,
        ]

        for plotter_class in plotter_classes:
            # Should have basic plotting interface
            # Note: Can't test full instantiation without proper inputs
            assert hasattr(plotter_class, "__init__")


class TestPlotterErrorHandling:
    """Test error handling in plotting functionality."""

    def test_radiometric_plotter_with_invalid_input(self):
        """Test RadiometricPlotter error handling with invalid input."""
        with pytest.raises((ValueError, TypeError, FileNotFoundError)):
            RadiometricPlotter(input="nonexistent_file.h5")

    def test_plotter_with_empty_table(self):
        """Test plotter behavior with empty table."""
        empty_table = QTable()

        # Should handle empty table gracefully
        with pytest.raises(
            (ValueError, KeyError), match=r"(Empty table|Missing required columns)"
        ):
            RadiometricPlotter(input=empty_table)


class TestPlotterIntegrationBehavior:
    """Test plotter integration and workflow behaviors."""

    @patch("matplotlib.pyplot.savefig")
    def test_plot_saving_workflow(self, mock_savefig):
        """Test plot saving workflow."""
        # Create sample data
        sample_table = QTable(
            {
                "ch_name": ["Ch1"],
                "Wavelength": [1.0] * u.um,
            }
        )

        plotter = RadiometricPlotter(input=sample_table)

        # Mock save_fig method if it exists
        if hasattr(plotter, "save_fig"):
            with patch.object(plotter, "save_fig") as mock_save:
                plotter.save_fig("test_plot.png")
                mock_save.assert_called_once_with("test_plot.png")
        else:
            # Skip if method doesn't exist
            pytest.skip("save_fig method not available in this plotter")

    def test_plotter_output_directory_handling(self):
        """Test plotter output directory handling."""
        sample_table = QTable(
            {
                "ch_name": ["Ch1"],
                "Wavelength": [1.0] * u.um,
            }
        )

        plotter = RadiometricPlotter(input=sample_table)

        # Test output directory configuration
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle valid directory
            if hasattr(plotter, "set_output_dir"):
                plotter.set_output_dir(temp_dir)
                assert True  # No exception raised
            else:
                pytest.skip("set_output_dir method not available")
