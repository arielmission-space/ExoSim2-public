import glob
import os
import shutil
import time
import unittest

from inputs import fast_test
from inputs import prepare_inputs
from inputs import regression_dir
from inputs import test_dir

import exosim.plots as plots
import exosim.recipes as recipes
from exosim.log import disableLogging

disableLogging()
timestr = time.strftime("%Y%m%d-%H%M%S")


@unittest.skipIf(fast_test, "slow tests skipped")
class RecipesPlottersTest(unittest.TestCase):
    # clean the dir
    f_list = glob.glob(os.path.join(test_dir, "test_data-*.h5"))
    for f in f_list:
        os.remove(f)

    # clean the plots dir
    plots_folder = os.path.join(test_dir, "plots")
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    for filename in os.listdir(plots_folder):
        file_path = os.path.join(plots_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete {}. Reason: {}".format(file_path, e))

    # define new file names
    out_name = os.path.join(
        test_dir, "test_data-{}-{}.h5".format(timestr, "fp")
    )
    rm_out_name = os.path.join(
        test_dir, "test_data-{}-{}.h5".format(timestr, "rm")
    )
    se_out_name = os.path.join(
        test_dir, "test_data-{}-{}.h5".format(timestr, "se")
    )
    ndr_out_name = os.path.join(
        test_dir, "test_data-{}-{}.h5".format(timestr, "ndr")
    )

    def test_full_run(self):
        mainConfig = prepare_inputs()
        recipes.CreateFocalPlane(mainConfig, self.out_name)
        self.assertTrue(os.path.isfile(self.out_name))

        recipes.RadiometricModel(mainConfig, self.out_name)

        recipes.RadiometricModel(mainConfig, self.rm_out_name)

        recipes.CreateSubExposures(
            input_file=self.out_name,
            output_file=self.se_out_name,
            options_file=mainConfig,
        )

        recipes.CreateNDRs(
            input_file=self.se_out_name,
            output_file=self.ndr_out_name,
            options_file=mainConfig,
        )

        focalPlanePlotter = plots.FocalPlanePlotter(
            input=self.out_name,
        )
        focalPlanePlotter.plot_focal_plane(time_step=0)
        focalPlanePlotter.save_fig(
            os.path.join(self.plots_folder, "focal_plane.png")
        )

        focalPlanePlotter.plot_efficiency()
        focalPlanePlotter.save_fig(
            os.path.join(self.plots_folder, "efficiency.png")
        )

        radiometricPlotter = plots.RadiometricPlotter(input=self.rm_out_name)
        radiometricPlotter.plot_table(contribs=False)
        radiometricPlotter.save_fig(
            os.path.join(self.plots_folder, "radiometric.png")
        )

        radiometricPlotter.plot_apertures()
        radiometricPlotter.save_fig(
            os.path.join(self.plots_folder, "apertures.png")
        )

        subExposuresPlotter = plots.SubExposuresPlotter(input=self.se_out_name)
        subExposuresPlotter.plot(self.plots_folder)

        ndrssPlotter = plots.NDRsPlotter(input=self.ndr_out_name)
        ndrssPlotter.plot(self.plots_folder)


@unittest.skipIf(fast_test, "slow tests skipped")
class RecipesPlottersSingleChTest(unittest.TestCase):
    # clean the dir
    f_list = glob.glob(os.path.join(test_dir, "test_data_single-*.h5"))
    for f in f_list:
        os.remove(f)

    # clean the plots dir
    plots_folder = os.path.join(test_dir, "plots")
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    for filename in os.listdir(plots_folder):
        file_path = os.path.join(plots_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete {}. Reason: {}".format(file_path, e))

    # define new file names
    out_name = os.path.join(
        test_dir, "test_data_single-{}-{}.h5".format(timestr, "fp")
    )
    rm_out_name = os.path.join(
        test_dir, "test_data_single-{}-{}.h5".format(timestr, "rm")
    )
    se_out_name = os.path.join(
        test_dir, "test_data_single-{}-{}.h5".format(timestr, "se")
    )
    ndr_out_name = os.path.join(
        test_dir, "test_data_single-{}-{}.h5".format(timestr, "ndr")
    )

    mainConfig = prepare_inputs(
        filename=os.path.join(regression_dir, "main_example_single.xml"),
        single=True,
    )

    def test_full_run(self):
        recipes.CreateFocalPlane(self.mainConfig, self.out_name)

        self.assertTrue(os.path.isfile(self.out_name))

        recipes.RadiometricModel(self.mainConfig, self.out_name)

        recipes.RadiometricModel(self.mainConfig, self.rm_out_name)

        recipes.CreateSubExposures(
            input_file=self.out_name,
            output_file=self.se_out_name,
            options_file=self.mainConfig,
        )

        recipes.CreateNDRs(
            input_file=self.se_out_name,
            output_file=self.ndr_out_name,
            options_file=self.mainConfig,
        )

        focalPlanePlotter = plots.FocalPlanePlotter(
            input=self.out_name,
        )
        focalPlanePlotter.plot_focal_plane(time_step=0)
        focalPlanePlotter.save_fig(
            os.path.join(self.plots_folder, "focal_plane.png")
        )

        focalPlanePlotter.plot_efficiency()
        focalPlanePlotter.save_fig(
            os.path.join(self.plots_folder, "efficiency.png")
        )

        radiometricPlotter = plots.RadiometricPlotter(input=self.rm_out_name)
        radiometricPlotter.plot_table(contribs=False)
        radiometricPlotter.save_fig(
            os.path.join(self.plots_folder, "radiometric.png")
        )

        radiometricPlotter.plot_apertures()
        radiometricPlotter.save_fig(
            os.path.join(self.plots_folder, "apertures.png")
        )

        subExposuresPlotter = plots.SubExposuresPlotter(input=self.se_out_name)
        subExposuresPlotter.plot(self.plots_folder)

        ndrssPlotter = plots.NDRsPlotter(input=self.ndr_out_name)
        ndrssPlotter.plot(self.plots_folder)
