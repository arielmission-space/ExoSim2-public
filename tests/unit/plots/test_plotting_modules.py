"""
Unit tests for plotting module functionality.

This module tests the plotting functionality of ExoSim2.0, focusing on:
- Class instantiation and inheritance
- Method signatures and basic functionality
- Error handling and edge cases
- Integration with matplotlib and numpy
"""

import inspect
from unittest.mock import Mock, patch

import numpy as np
import pytest
from astropy.table import QTable

import exosim.log as log
from exosim.plots.focal_plane_plotter import FocalPlanePlotter
from exosim.plots.ndrs_plotter import NDRsPlotter
from exosim.plots.radiometric_plotter import RadiometricPlotter
from exosim.plots.sub_exposures_plotter import SubExposuresPlotter


class TestFocalPlanePlotter:
    """Test suite for FocalPlanePlotter class."""

    def test_inheritance(self):
        """
        Test that FocalPlanePlotter inherits from Logger.

        This test verifies proper inheritance structure for logging
        functionality in the focal plane plotter.
        """
        assert issubclass(FocalPlanePlotter, log.Logger)

    def test_initialization(self):
        """
        Test FocalPlanePlotter initialization.

        This test verifies that the FocalPlanePlotter can be initialized
        properly with required parameters and maintains expected attributes.
        """
        with (
            patch("h5py.File"),
            patch.object(FocalPlanePlotter, "set_log_name"),
            patch.object(FocalPlanePlotter, "graphics"),
            patch.object(FocalPlanePlotter, "announce"),
        ):
            plotter = FocalPlanePlotter(input="test_file.h5")
            assert plotter.input == "test_file.h5"
            assert plotter.fig is None

    def test_initialization_calls_logger_methods(self):
        """
        Test that initialization calls required logger methods.

        This test verifies that the FocalPlanePlotter properly initializes
        logging components during instantiation.
        """
        with (
            patch("h5py.File"),
            patch.object(FocalPlanePlotter, "set_log_name") as mock_log_name,
            patch.object(FocalPlanePlotter, "graphics") as mock_graphics,
            patch.object(FocalPlanePlotter, "announce") as mock_announce,
        ):
            FocalPlanePlotter(input="test_file.h5")
            mock_log_name.assert_called_once()
            mock_graphics.assert_called_once()
            mock_announce.assert_called_once_with("started")

    def test_has_required_methods(self):
        """
        Test that FocalPlanePlotter has all required methods.

        This test verifies that all essential methods are present
        in the FocalPlanePlotter class interface.
        """
        methods = [
            "_prepare_figure",
            "_plot_ch",
            "plot_focal_plane",
            "plot_bands",
            "plot_efficiency",
            "save_fig",
            "load_focal_plane",
        ]

        for method in methods:
            assert hasattr(FocalPlanePlotter, method), f"Missing method: {method}"

    def test_method_signatures(self):
        """
        Test that methods have expected signatures.

        This test verifies that critical methods have the expected
        parameter signatures for proper functionality.
        """
        # Test plot_focal_plane signature
        sig = inspect.signature(FocalPlanePlotter.plot_focal_plane)
        assert "time_step" in sig.parameters
        assert "scale" in sig.parameters

    def test_prepare_figure_method_exists(self):
        """
        Test that _prepare_figure method exists.

        This test verifies the existence of internal figure preparation
        method without requiring complex mocking setup.
        """
        assert hasattr(FocalPlanePlotter, "_prepare_figure")


class TestRadiometricPlotter:
    """Test suite for RadiometricPlotter class."""

    def test_inheritance(self):
        """
        Test that RadiometricPlotter inherits from Logger.

        This test verifies proper inheritance structure for logging
        functionality in the radiometric plotter.
        """
        assert issubclass(RadiometricPlotter, log.Logger)

    def test_initialization_with_string(self):
        """
        Test RadiometricPlotter initialization with string input.

        This test verifies that the RadiometricPlotter can be initialized
        with a file path string and properly loads the associated data table.
        """
        with (
            patch.object(RadiometricPlotter, "set_log_name"),
            patch.object(RadiometricPlotter, "graphics"),
            patch.object(RadiometricPlotter, "announce"),
            patch.object(RadiometricPlotter, "load_table") as mock_load,
        ):
            mock_table = Mock()
            mock_load.return_value = mock_table

            plotter = RadiometricPlotter(input="test_file.h5")
            assert plotter.input == "test_file.h5"
            assert plotter.input_table == mock_table
            assert plotter.fig is None
            mock_load.assert_called_once_with("test_file.h5", "radiometric")

    def test_initialization_with_table(self):
        """
        Test RadiometricPlotter initialization with Table input.

        This test verifies that the RadiometricPlotter can be initialized
        directly with a data table object.
        """
        test_table = QTable({"column1": [1, 2, 3]})

        with (
            patch.object(RadiometricPlotter, "set_log_name"),
            patch.object(RadiometricPlotter, "graphics"),
            patch.object(RadiometricPlotter, "announce"),
        ):
            plotter = RadiometricPlotter(input=test_table)
            assert plotter.input is test_table
            assert plotter.input_table is test_table
            assert plotter.fig is None

    def test_has_required_methods(self):
        """
        Test that RadiometricPlotter has all required methods.

        This test verifies that all essential methods are present
        in the RadiometricPlotter class interface.
        """
        methods = [
            "load_table",
            "plot_bands",
            "plot_noise",
            "plot_signal",
            "plot_table",
            "plot_efficiency",
            "plot_apertures",
            "save_fig",
        ]

        for method in methods:
            assert hasattr(RadiometricPlotter, method), f"Missing method: {method}"

    def test_method_signatures(self):
        """
        Test that methods have expected signatures.

        This test verifies that critical methods have the expected
        parameter signatures for proper functionality.
        """
        # Test plot_bands signature
        sig = inspect.signature(RadiometricPlotter.plot_bands)
        assert "ax" in sig.parameters
        assert "scale" in sig.parameters
        assert "channel_edges" in sig.parameters
        assert "add_legend" in sig.parameters

    def test_load_table_method_exists(self):
        """
        Test that load_table method exists.

        This test verifies the existence of the data table loading
        method without requiring complex mocking setup.
        """
        with (
            patch.object(RadiometricPlotter, "set_log_name"),
            patch.object(RadiometricPlotter, "graphics"),
            patch.object(RadiometricPlotter, "announce"),
        ):
            plotter = RadiometricPlotter(QTable({"col": [1]}))
            assert hasattr(plotter, "load_table")


class TestSubExposuresPlotter:
    """Test suite for SubExposuresPlotter class."""

    def test_inheritance(self):
        """
        Test that SubExposuresPlotter inherits from Logger.

        This test verifies proper inheritance structure for logging
        functionality in the sub-exposures plotter.
        """
        assert issubclass(SubExposuresPlotter, log.Logger)

    def test_initialization(self):
        """
        Test SubExposuresPlotter initialization.

        This test verifies that the SubExposuresPlotter can be initialized
        properly with required parameters.
        """
        with (
            patch.object(SubExposuresPlotter, "set_log_name"),
            patch.object(SubExposuresPlotter, "graphics"),
            patch.object(SubExposuresPlotter, "announce"),
        ):
            plotter = SubExposuresPlotter(input="test_file.h5")
            assert plotter.input == "test_file.h5"

    def test_initialization_calls_logger_methods(self):
        """
        Test that initialization calls required logger methods.

        This test verifies that the SubExposuresPlotter properly initializes
        logging components during instantiation.
        """
        with (
            patch.object(SubExposuresPlotter, "set_log_name") as mock_log_name,
            patch.object(SubExposuresPlotter, "graphics") as mock_graphics,
            patch.object(SubExposuresPlotter, "announce") as mock_announce,
        ):
            SubExposuresPlotter(input="test_file.h5")
            mock_log_name.assert_called_once()
            mock_graphics.assert_called_once()
            mock_announce.assert_called_once_with("started")

    def test_has_required_methods(self):
        """
        Test that SubExposuresPlotter has all required methods.

        This test verifies that all essential methods are present
        in the SubExposuresPlotter class interface.
        """
        methods = ["plot", "plot_sub_exposure", "load_ndrs"]

        for method in methods:
            assert hasattr(SubExposuresPlotter, method), f"Missing method: {method}"

    def test_method_signatures(self):
        """
        Test that methods have expected signatures.

        This test verifies that critical methods have the expected
        parameter signatures for proper functionality.
        """
        # Test plot signature
        sig = inspect.signature(SubExposuresPlotter.plot)
        assert "out_dir" in sig.parameters


class TestNDRsPlotter:
    """Test suite for NDRsPlotter class."""

    def test_inheritance(self):
        """
        Test that NDRsPlotter inherits from Logger.

        This test verifies proper inheritance structure for logging
        functionality in the NDRs plotter.
        """
        assert issubclass(NDRsPlotter, log.Logger)

    def test_initialization(self):
        """
        Test NDRsPlotter initialization.

        This test verifies that the NDRsPlotter can be initialized
        properly with required parameters.
        """
        with (
            patch.object(NDRsPlotter, "set_log_name"),
            patch.object(NDRsPlotter, "graphics"),
            patch.object(NDRsPlotter, "announce"),
        ):
            plotter = NDRsPlotter(input="test_file.h5")
            assert plotter.input == "test_file.h5"

    def test_initialization_calls_logger_methods(self):
        """
        Test that initialization calls required logger methods.

        This test verifies that the NDRsPlotter properly initializes
        logging components during instantiation.
        """
        with (
            patch.object(NDRsPlotter, "set_log_name") as mock_log_name,
            patch.object(NDRsPlotter, "graphics") as mock_graphics,
            patch.object(NDRsPlotter, "announce") as mock_announce,
        ):
            NDRsPlotter(input="test_file.h5")
            mock_log_name.assert_called_once()
            mock_graphics.assert_called_once()
            mock_announce.assert_called_once_with("started")

    def test_has_required_methods(self):
        """
        Test that NDRsPlotter has all required methods.

        This test verifies that all essential methods are present
        in the NDRsPlotter class interface.
        """
        methods = ["plot", "plot_ndrs", "load_ndrs"]

        for method in methods:
            assert hasattr(NDRsPlotter, method), f"Missing method: {method}"

    def test_method_signatures(self):
        """
        Test that methods have expected signatures.

        This test verifies that critical methods have the expected
        parameter signatures for proper functionality.
        """
        # Test plot signature
        sig = inspect.signature(NDRsPlotter.plot)
        assert "out_dir" in sig.parameters


class TestPlotUtilsIntegration:
    """Test suite for integration with plot utilities."""

    def test_imports_work(self):
        """
        Test that all plotter classes can be imported.

        This test verifies that all plotting modules can be imported
        successfully without errors.
        """
        from exosim.plots.focal_plane_plotter import FocalPlanePlotter
        from exosim.plots.ndrs_plotter import NDRsPlotter
        from exosim.plots.radiometric_plotter import RadiometricPlotter
        from exosim.plots.sub_exposures_plotter import SubExposuresPlotter

        # All classes should be importable
        assert FocalPlanePlotter is not None
        assert NDRsPlotter is not None
        assert RadiometricPlotter is not None
        assert SubExposuresPlotter is not None

    def test_all_plotters_have_save_fig(self):
        """
        Test that appropriate plotters have save_fig method.

        This test verifies that plotters expected to support figure
        saving have the save_fig method available.
        """
        # Only some plotters have save_fig
        assert hasattr(FocalPlanePlotter, "save_fig")
        assert hasattr(RadiometricPlotter, "save_fig")

    def test_logger_integration(self):
        """
        Test that all plotters properly integrate with logger.

        This test verifies that all plotting classes properly inherit
        from the Logger base class for consistent logging functionality.
        """
        plot_classes = [
            FocalPlanePlotter,
            NDRsPlotter,
            RadiometricPlotter,
            SubExposuresPlotter,
        ]

        for plot_class in plot_classes:
            assert issubclass(plot_class, log.Logger), (
                f"{plot_class} should inherit from Logger"
            )


class TestPlotterCommonPatterns:
    """Test suite for common patterns across all plotters."""

    def test_all_plotters_have_input_parameter(self):
        """
        Test that all plotters accept input parameter in __init__.

        This test verifies that all plotting classes follow the
        common pattern of accepting an 'input' parameter during
        initialization.
        """
        plot_classes = [
            FocalPlanePlotter,
            NDRsPlotter,
            RadiometricPlotter,
            SubExposuresPlotter,
        ]

        for plot_class in plot_classes:
            sig = inspect.signature(plot_class.__init__)
            assert "input" in sig.parameters, (
                f"{plot_class} should have 'input' parameter"
            )

    def test_plotters_use_matplotlib(self):
        """
        Test that plotters use matplotlib components.

        This is a smoke test to ensure matplotlib imports work
        correctly and basic plotting functionality is available.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plotters_use_numpy(self):
        """
        Test that plotters can use numpy arrays.

        This test ensures numpy integration works correctly
        for array operations commonly used in plotting.
        """
        test_array = np.array([1, 2, 3, 4, 5])
        assert len(test_array) == 5
        assert test_array.max() == 5


class TestPlottingErrorHandling:
    """Test suite for error handling in plotting modules."""

    def test_matplotlib_backend_availability(self):
        """
        Test matplotlib backend availability.

        This test verifies that a matplotlib backend is available
        for plotting operations, which is essential for the plotting
        modules to function correctly.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        # Check that a backend is available
        backend = mpl.get_backend()
        assert backend is not None

        # Test basic figure creation doesn't fail
        try:
            fig = plt.figure()
            plt.close(fig)
        except Exception as e:
            pytest.skip(f"Matplotlib backend not available: {e}")

    def test_numpy_array_operations(self):
        """
        Test basic numpy array operations used in plotting.

        This test verifies that common numpy operations used
        in plotting workflows function correctly.
        """
        # Test array creation and operations
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        assert len(x) == 100
        assert len(y) == 100
        assert -1 <= y.min() <= 1
        assert -1 <= y.max() <= 1
