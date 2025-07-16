import logging
import os

import exosim.plots as plots
from exosim import __version__ as version

logger = logging.getLogger("exosim")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ExoSim {}".format(version))
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        type=str,
        required=True,
        help="Input file to pass",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=True,
        default=None,
        help="Output file",
    )
    parser.add_argument(
        "-f",
        "--focal_plane",
        dest="focal",
        default=False,
        required=False,
        help="run to plot the focal plane ",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--radiometric",
        dest="radiometric",
        default=False,
        required=False,
        help="run to plot the radiometric table",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--subexposures",
        dest="subexposures",
        default=False,
        required=False,
        help="run to plot the sub-exposures",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--ndrs",
        dest="ndrs",
        default=False,
        required=False,
        help="run to plot the NDRs plotter",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--time_step",
        dest="time_step",
        type=int,
        default=0,
        required=False,
        help="focal plane time step to plot ",
    )

    parser.add_argument(
        "--plot-scale",
        dest="plot_scale",
        type=str,
        default="linear",
        required=False,
        help="plot scale. Can be 'linear' or 'dB'. Default is 'linear' ",
    )
    args = parser.parse_args()

    if args.focal:
        time = args.time_step
        scale = args.plot_scale
        focal_plane_plotter = plots.FocalPlanePlotter(input=args.input)
        focal_plane_plotter.plot_focal_plane(time_step=time, scale=scale)
        fig_name = os.path.join(args.output, "focal_plane_{}.png".format(time))
        focal_plane_plotter.save_fig(fig_name)

        focal_plane_plotter.plot_efficiency()
        fig_name = os.path.join(args.output, "efficiency.png")
        focal_plane_plotter.save_fig(fig_name)

    if args.radiometric:
        radiometric_plotter = plots.RadiometricPlotter(input=args.input)
        radiometric_plotter.plot_table()
        fig_name = os.path.join(args.output, "radiometric.png")
        radiometric_plotter.save_fig(fig_name)

        radiometric_plotter.plot_apertures()
        fig_name = os.path.join(args.output, "apertures.png")
        radiometric_plotter.save_fig(fig_name)

    if args.subexposures:
        sub_exposures_plotter = plots.SubExposuresPlotter(input=args.input)
        sub_exposures_plotter.plot(args.output)

    if args.ndrs:
        ndrs_plotter = plots.NDRsPlotter(input=args.input)
        ndrs_plotter.plot(args.output)
