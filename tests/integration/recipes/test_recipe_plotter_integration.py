"""
Integration tests for ExoSim recipes and plotters.

This module contains integration tests that verify the full workflow
from focal plane creation through NDR generation and plotting.
"""

import gc
import glob
import os
import shutil
import time

import h5py
import pytest

import exosim.plots as plots
import exosim.recipes as recipes
from exosim.log import disable_logging

disable_logging()


@pytest.fixture
def clean_test_environment(test_data_dir):
    """Clean test directories and create necessary folders."""
    # Clean ALL h5 files before test
    import contextlib

    for f in glob.glob(os.path.join(test_data_dir, "*.h5")):
        with contextlib.suppress(Exception):
            os.remove(f)

    # Clean plots folder
    plots_folder = os.path.join(test_data_dir, "plots")
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    else:
        for filename in os.listdir(plots_folder):
            file_path = os.path.join(plots_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception:
                pass

    yield plots_folder

    # Clean up after test
    # Force garbage collection to release any HDF5 file handles
    gc.collect()

    # Try to close any open HDF5 files
    for obj in gc.get_objects():
        try:
            if isinstance(obj, h5py.File):
                obj.close()
        except Exception:
            pass


@pytest.mark.skip(reason="HDF5 dataset creation conflict - needs investigation")
def test_recipes_plotters_full_workflow(
    clean_test_environment, test_data_dir, prepare_inputs_fixture
):
    """Test complete workflow: recipes + plotters integration."""
    plots_folder = clean_test_environment
    # Add microseconds to ensure unique filenames even in fast test runs
    timestr = (
        time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000000) % 1000000}"
    )

    # Define file names
    out_name = os.path.join(test_data_dir, f"test_data-{timestr}-fp.h5")
    se_out_name = os.path.join(test_data_dir, f"test_data-{timestr}-se.h5")
    ndr_out_name = os.path.join(test_data_dir, f"test_data-{timestr}-ndr.h5")

    # Ensure output files don't exist
    for fname in [out_name, se_out_name, ndr_out_name]:
        if os.path.exists(fname):
            import contextlib

            with contextlib.suppress(Exception):
                os.remove(fname)

    # Run tests
    mainConfig = prepare_inputs_fixture()

    # Debug the configuration
    # print("\nMain Config:")
    for key in ["sky", "source", "wl_grid"]:
        if key in mainConfig:
            pass
            # print(f"{key}: {mainConfig[key]}")

    recipes.CreateFocalPlane(mainConfig, out_name)
    assert os.path.isfile(out_name)

    # Force close any HDF5 file handles before next operation
    gc.collect()
    for obj in gc.get_objects():
        try:
            if isinstance(obj, h5py.File) and obj.filename == out_name:
                obj.close()
        except Exception:
            pass

    recipes.RadiometricModel(mainConfig, out_name)

    recipes.CreateSubExposures(
        input_file=out_name,
        output_file=se_out_name,
        options_file=mainConfig,
    )

    recipes.CreateNDRs(
        input_file=se_out_name,
        output_file=ndr_out_name,
        options_file=mainConfig,
    )

    focalPlanePlotter = plots.FocalPlanePlotter(input=out_name)
    focalPlanePlotter.plot_focal_plane(time_step=0)
    focalPlanePlotter.save_fig(os.path.join(plots_folder, "focal_plane.png"))

    focalPlanePlotter.plot_efficiency()
    focalPlanePlotter.save_fig(os.path.join(plots_folder, "efficiency.png"))

    radiometricPlotter = plots.RadiometricPlotter(input=out_name)
    radiometricPlotter.plot_table(contribs=False)
    radiometricPlotter.save_fig(os.path.join(plots_folder, "radiometric.png"))

    radiometricPlotter.plot_apertures()
    radiometricPlotter.save_fig(os.path.join(plots_folder, "apertures.png"))

    subExposuresPlotter = plots.SubExposuresPlotter(input=se_out_name)
    subExposuresPlotter.plot(plots_folder)

    ndrssPlotter = plots.NDRsPlotter(input=ndr_out_name)
    ndrssPlotter.plot(plots_folder)


@pytest.mark.skip(reason="HDF5 dataset creation conflict - needs investigation")
def test_recipes_plotters_single_channel_workflow(
    clean_test_environment, test_data_dir, prepare_inputs_fixture
):
    """Test single channel workflow: focal plane + radiometric + plotting."""
    plots_folder = clean_test_environment
    # Add microseconds to ensure unique filenames even in fast test runs
    timestr = (
        time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000000) % 1000000}"
    )

    # Define file names
    out_name = os.path.join(test_data_dir, f"test_data_single-{timestr}-fp.h5")
    se_out_name = os.path.join(test_data_dir, f"test_data_single-{timestr}-se.h5")
    ndr_out_name = os.path.join(test_data_dir, f"test_data_single-{timestr}-ndr.h5")

    # Prepare inputs
    mainConfig = prepare_inputs_fixture(single=True)

    # Run tests

    # Debug the configuration
    # print("\nMain Config:")
    for key in ["sky", "source", "wl_grid"]:
        if key in mainConfig:
            pass
            # print(f"{key}: {mainConfig[key]}")

    recipes.CreateFocalPlane(mainConfig, out_name)
    assert os.path.isfile(out_name)

    recipes.RadiometricModel(mainConfig, out_name)

    recipes.CreateSubExposures(
        input_file=out_name,
        output_file=se_out_name,
        options_file=mainConfig,
    )

    recipes.CreateNDRs(
        input_file=se_out_name,
        output_file=ndr_out_name,
        options_file=mainConfig,
    )

    focalPlanePlotter = plots.FocalPlanePlotter(input=out_name)
    focalPlanePlotter.plot_focal_plane(time_step=0)
    focalPlanePlotter.save_fig(os.path.join(plots_folder, "focal_plane.png"))

    focalPlanePlotter.plot_efficiency()
    focalPlanePlotter.save_fig(os.path.join(plots_folder, "efficiency.png"))

    radiometricPlotter = plots.RadiometricPlotter(input=out_name)
    radiometricPlotter.plot_table(contribs=False)
    radiometricPlotter.save_fig(os.path.join(plots_folder, "radiometric.png"))

    radiometricPlotter.plot_apertures()
    radiometricPlotter.save_fig(os.path.join(plots_folder, "apertures.png"))

    subExposuresPlotter = plots.SubExposuresPlotter(input=se_out_name)
    subExposuresPlotter.plot(plots_folder)

    ndrssPlotter = plots.NDRsPlotter(input=ndr_out_name)
    ndrssPlotter.plot(plots_folder)
