import glob
import os
import shutil
import time

import pytest

import exosim.plots as plots
import exosim.recipes as recipes
from exosim.log import disableLogging

disableLogging()
timestr = time.strftime("%Y%m%d-%H%M%S")


@pytest.fixture
def clean_test_environment(test_data_dir):
    """Clean test directories and create necessary folders."""
    # Clean data files
    for f in glob.glob(os.path.join(test_data_dir, "test_data-*.h5")):
        os.remove(f)

    for f in glob.glob(os.path.join(test_data_dir, "test_data_single-*.h5")):
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
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    return plots_folder


def test_recipes_plotters(clean_test_environment, test_data_dir, prepare_inputs_fixture, fast_test):
    if fast_test:
        pytest.skip("Skipping this test class in fast mode")
    plots_folder = clean_test_environment

    # Define file names
    out_name = os.path.join(test_data_dir, f"test_data-{timestr}-fp.h5")
    rm_out_name = os.path.join(test_data_dir, f"test_data-{timestr}-rm.h5")
    se_out_name = os.path.join(test_data_dir, f"test_data-{timestr}-se.h5")
    ndr_out_name = os.path.join(test_data_dir, f"test_data-{timestr}-ndr.h5")

    # Run tests
    mainConfig = prepare_inputs_fixture()
    recipes.CreateFocalPlane(mainConfig, out_name)
    assert os.path.isfile(out_name)

    recipes.RadiometricModel(mainConfig, out_name)
    recipes.RadiometricModel(mainConfig, rm_out_name)

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

    radiometricPlotter = plots.RadiometricPlotter(input=rm_out_name)
    radiometricPlotter.plot_table(contribs=False)
    radiometricPlotter.save_fig(os.path.join(plots_folder, "radiometric.png"))

    radiometricPlotter.plot_apertures()
    radiometricPlotter.save_fig(os.path.join(plots_folder, "apertures.png"))

    subExposuresPlotter = plots.SubExposuresPlotter(input=se_out_name)
    subExposuresPlotter.plot(plots_folder)

    ndrssPlotter = plots.NDRsPlotter(input=ndr_out_name)
    ndrssPlotter.plot(plots_folder)


def test_recipes_plotters_single_channel(clean_test_environment, test_data_dir, regression_data_dir, prepare_inputs_fixture, fast_test):
    if fast_test:
        pytest.skip("Skipping this test class in fast mode")
    plots_folder = clean_test_environment

    # Define file names
    out_name = os.path.join(test_data_dir, f"test_data_single-{timestr}-fp.h5")
    rm_out_name = os.path.join(test_data_dir, f"test_data_single-{timestr}-rm.h5")
    se_out_name = os.path.join(test_data_dir, f"test_data_single-{timestr}-se.h5")
    ndr_out_name = os.path.join(test_data_dir, f"test_data_single-{timestr}-ndr.h5")

    # Prepare inputs
    mainConfig = prepare_inputs_fixture(
        filename=os.path.join(regression_data_dir, "main_example_single.xml"),
        single=True,
    )

    # Run tests
    recipes.CreateFocalPlane(mainConfig, out_name)
    assert os.path.isfile(out_name)

    recipes.RadiometricModel(mainConfig, out_name)
    recipes.RadiometricModel(mainConfig, rm_out_name)

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

    radiometricPlotter = plots.RadiometricPlotter(input=rm_out_name)
    radiometricPlotter.plot_table(contribs=False)
    radiometricPlotter.save_fig(os.path.join(plots_folder, "radiometric.png"))

    radiometricPlotter.plot_apertures()
    radiometricPlotter.save_fig(os.path.join(plots_folder, "apertures.png"))

    subExposuresPlotter = plots.SubExposuresPlotter(input=se_out_name)
    subExposuresPlotter.plot(plots_folder)

    ndrssPlotter = plots.NDRsPlotter(input=ndr_out_name)
    ndrssPlotter.plot(plots_folder)
