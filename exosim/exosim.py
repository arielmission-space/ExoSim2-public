import importlib.metadata as metadata
import logging
import os

import exosim.recipes as recipes
from exosim import __summary__
from exosim import __version__
from exosim.log import addLogFile
from exosim.log import setLogLevel
from exosim.utils import RunConfig

logger = logging.getLogger("exosim")
code_name_and_version = "ExoSim {}".format(__version__)

# parser add_argument standard options
configuration_flags = ["-c", "--configuration"]
configuration = {
    "dest": "conf",
    "type": str,
    "required": True,
    "help": "Input configuration file to pass",
}

output_file_flags = ["-o", "--output"]
output_file = {
    "dest": "output",
    "type": str,
    "required": False,
    "default": None,
    "help": "Output file",
}

input_file_flags = ["-i", "--input"]
input_file = {
    "dest": "input",
    "type": str,
    "required": True,
    "default": None,
    "help": "Input file",
}

n_threads_flags = ["--nThreads"]
n_threads = {
    "dest": "numberOfThreads",
    "default": None,
    "type": int,
    "required": False,
    "help": "number of threads for parallel processing",
}

debug_conf_flags = ["-d", "--debug"]
debug_conf = {
    "dest": "debug",
    "default": False,
    "required": False,
    "help": "enable debug mode",
    "action": "store_true",
}

logger_conf_flags = ["-l", "--logger"]
logger_conf = {
    "dest": "log",
    "default": False,
    "required": False,
    "help": "save log file",
    "action": "store_true",
}

plot_options_flags = ["-P", "--plot"]
plot_options = {
    "dest": "plot",
    "default": False,
    "required": False,
    "help": "save sub-exposures plots",
    "action": "store_true",
}

chunk_options_flags = ["--chunk_size"]
chunk_options = {
    "dest": "chunk_size",
    "default": 2,
    "type": float,
    "required": False,
    "help": "h5file chunk size",
}


def help():
    import argparse

    print(code_name_and_version + "\n------\n" + __summary__)

    parser = argparse.ArgumentParser()
    parser.parse_args()


def focalplane():
    import argparse

    parser = argparse.ArgumentParser(description=code_name_and_version)
    parser.add_argument(*configuration_flags, **configuration)
    parser.add_argument(*output_file_flags, **output_file)
    parser.add_argument(*n_threads_flags, **n_threads)
    parser.add_argument(*debug_conf_flags, **debug_conf)
    parser.add_argument(*logger_conf_flags, **logger_conf)
    parser.add_argument(*plot_options_flags, **plot_options)

    parser.add_argument(
        "--plot-scale",
        dest="plot_scale",
        type=str,
        default="linear",
        required=False,
        help="plot scale. Can be 'linear' or 'dB'. Default is 'linear' ",
    )

    args = parser.parse_args()

    _set_log(args)
    _set_threads(args)

    recipes.CreateFocalPlane(options_file=args.conf, output_file=args.output)

    if args.plot:
        import exosim.plots as plots

        scale = args.plot_scale

        dir_name = os.path.join(os.path.dirname(args.output), "plots")

        focal_plane_plotter = plots.FocalPlanePlotter(input=args.output)
        focal_plane_plotter.plot_focal_plane(time_step=0, scale=scale)
        fig_name = os.path.join(dir_name, "focal_plane.png")
        focal_plane_plotter.save_fig(fig_name)

        focal_plane_plotter.plot_efficiency()
        fig_name = os.path.join(dir_name, "efficiency.png")
        focal_plane_plotter.save_fig(fig_name)


def radiometric():
    import argparse

    parser = argparse.ArgumentParser(description=code_name_and_version)
    parser.add_argument(*configuration_flags, **configuration)
    parser.add_argument(*output_file_flags, **output_file)
    parser.add_argument(*n_threads_flags, **n_threads)
    parser.add_argument(*debug_conf_flags, **debug_conf)
    parser.add_argument(*logger_conf_flags, **logger_conf)
    parser.add_argument(*plot_options_flags, **plot_options)

    args = parser.parse_args()

    _set_log(args)
    _set_threads(args)

    recipes.RadiometricModel(options_file=args.conf, input_file=args.output)

    if args.plot:
        import exosim.plots as plots

        dir_name = os.path.join(os.path.dirname(args.output), "plots")

        radiometric_plotter = plots.RadiometricPlotter(input=args.output)
        radiometric_plotter.plot_table()
        fig_name = os.path.join(dir_name, "radiometric.png")
        radiometric_plotter.save_fig(fig_name)

        radiometric_plotter.plot_apertures()
        fig_name = os.path.join(dir_name, "apertures.png")
        radiometric_plotter.save_fig(fig_name)


def subexposures():
    import argparse

    parser = argparse.ArgumentParser(description=code_name_and_version)
    parser.add_argument(*configuration_flags, **configuration)
    parser.add_argument(*input_file_flags, **input_file)
    parser.add_argument(*output_file_flags, **output_file)
    parser.add_argument(*n_threads_flags, **n_threads)
    parser.add_argument(*debug_conf_flags, **debug_conf)
    parser.add_argument(*logger_conf_flags, **logger_conf)
    parser.add_argument(*plot_options_flags, **plot_options)
    parser.add_argument(*chunk_options_flags, **chunk_options)

    args = parser.parse_args()

    _set_log(args)
    _set_threads(args)
    _set_chunk_size(args)

    recipes.CreateSubExposures(
        options_file=args.conf, input_file=args.input, output_file=args.output
    )

    if args.plot:
        import exosim.plots as plots

        dir_name = os.path.join(os.path.dirname(args.output), "plots")
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        sub_exposures_plotter = plots.SubExposuresPlotter(input=args.output)
        sub_exposures_plotter.plot(dir_name)


def ndrs():
    import argparse

    parser = argparse.ArgumentParser(description=code_name_and_version)
    parser.add_argument(*configuration_flags, **configuration)
    parser.add_argument(*input_file_flags, **input_file)
    parser.add_argument(*output_file_flags, **output_file)
    parser.add_argument(*n_threads_flags, **n_threads)
    parser.add_argument(*debug_conf_flags, **debug_conf)
    parser.add_argument(*logger_conf_flags, **logger_conf)
    parser.add_argument(*plot_options_flags, **plot_options)
    parser.add_argument(*chunk_options_flags, **chunk_options)

    args = parser.parse_args()

    _set_log(args)
    _set_threads(args)
    _set_chunk_size(args)

    recipes.CreateNDRs(
        options_file=args.conf, input_file=args.input, output_file=args.output
    )

    if args.plot:
        import exosim.plots as plots

        dir_name = os.path.join(os.path.dirname(args.output), "plots")
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        ndrs_plotter = plots.NDRsPlotter(input=args.output)
        ndrs_plotter.plot(dir_name)


def _set_log(args):
    if args.debug:
        setLogLevel(logging.DEBUG)
    if args.log:
        if isinstance(args.output, str):
            try:
                addLogFile(
                    fname="{}/exosim.log".format(os.path.dirname(args.output))
                )
            except PermissionError:
                addLogFile(fname="exosim.log")
        else:
            addLogFile()


def _set_threads(args):
    if args.numberOfThreads:
        RunConfig.n_job = args.numberOfThreads


def _set_chunk_size(args):
    if args.chunk_size:
        RunConfig.chunk_size = args.chunk_size
