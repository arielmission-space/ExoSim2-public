import exosim.recipes as recipes
from exosim.plots import (
    FocalPlanePlotter,
    NDRsPlotter,
    RadiometricPlotter,
    SubExposuresPlotter,
)
from exosim.utils import RunConfig

# this will force the use of all the cpu except 2
RunConfig.n_job = -2
RunConfig.random_seed = 10


def main():
    # create focal plane
    recipes.CreateFocalPlane("main_example.xml", "./test_common.h5")
    # run focal plane plotter
    focal_plane_plotter = FocalPlanePlotter(input="./test_common.h5")
    focal_plane_plotter.plot_focal_plane(time_step=0, scale="linear")
    focal_plane_plotter.save_fig("plots/focal_plane.png")
    focal_plane_plotter.plot_efficiency()
    focal_plane_plotter.save_fig("plots/efficiency.png")

    # # run radiometric model
    recipes.RadiometricModel("main_example.xml", "./test_common.h5")
    # run radiometric plotter
    radiometric_plotter = RadiometricPlotter(input="./test_common.h5")
    radiometric_plotter.plot_table(contribs=False)
    radiometric_plotter.save_fig("plots/radiometric.png")
    radiometric_plotter.plot_apertures()
    radiometric_plotter.save_fig("plots/apertures.png")

    # create Sub-Exposures
    recipes.CreateSubExposures(
        input_file="./test_common.h5",
        output_file="./test_se.h5",
        options_file="main_example.xml",
    )
    # run Sub-Exposures plotter
    sub_exposures_plotter = SubExposuresPlotter(input="./test_se.h5")
    sub_exposures_plotter.plot("plots/subexposures")

    # create NDRs
    recipes.CreateNDRs(
        input_file="./test_se.h5",
        output_file="./test_ndr.h5",
        options_file="main_example.xml",
    )
    # run NDRs plotter
    ndrss_plotter = NDRsPlotter(input="./test_ndr.h5")
    ndrss_plotter.plot("plots/ndrs")


if __name__ == "__main__":
    main()
