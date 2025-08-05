import logging
import os

import click
from rich_click import RichGroup

import exosim.recipes as recipes
from exosim import __version__
from exosim.log import addLogFile, setLogLevel
from exosim.utils import RunConfig

# Logger configuration
logger = logging.getLogger("exosim")
code_name_and_version = "ExoSim {}".format(__version__)


# Common options decorator for shared command-line arguments
def common_options(func):
    """Decorator to add common options to commands."""
    options = [
        click.option(
            "-c",
            "--configuration",
            "conf",
            required=True,
            type=click.Path(exists=True),
            help="Input configuration file to pass.",
        ),
        click.option(
            "-o",
            "--output",
            "output",
            required=False,
            type=click.Path(),
            default=None,
            help="Output file.",
        ),
        click.option(
            "--nThreads",
            "numberOfThreads",
            required=False,
            type=int,
            default=None,
            help="Number of threads for parallel processing.",
        ),
        click.option(
            "-d",
            "--debug",
            is_flag=True,
            help="Enable debug mode.",
        ),
        click.option(
            "-l",
            "--logger",
            is_flag=True,
            help="Save log file.",
        ),
        click.option(
            "-P",
            "--plot",
            is_flag=True,
            help="Save plots.",
        ),
    ]
    for option in reversed(options):  # Add options in correct order
        func = option(func)
    return func


@click.group(cls=RichGroup)
@click.version_option(
    version=__version__,
    prog_name="ExoSim",
    message="%(prog)s v%(version)s",
)
def cli():
    """ExoSim CLI - A simulation toolkit for exoplanet characterisation."""


@cli.command()
@common_options
@click.option(
    "--plot-scale",
    default="linear",
    type=click.Choice(["linear", "dB"], case_sensitive=False),
    show_default=True,
    help="Plot scale. Can be 'linear' or 'dB'.",
)
def focalplane(conf, output, numberOfThreads, debug, logger, plot, plot_scale):
    """Create and plot the focal plane."""
    _set_log(debug, logger, output)
    _set_threads(numberOfThreads)

    recipes.CreateFocalPlane(options_file=conf, output_file=output)

    if plot:
        _plot_focal_plane(output, plot_scale)


@cli.command()
@common_options
def radiometric(conf, output, numberOfThreads, debug, logger, plot):
    """Create a radiometric model."""
    _set_log(debug, logger, output)
    _set_threads(numberOfThreads)

    recipes.RadiometricModel(options_file=conf, input_file=output)

    if plot:
        _plot_radiometric(output)


@cli.command()
@common_options
@click.option(
    "-i",
    "--input",
    "input",
    required=True,
    type=click.Path(exists=True),
    help="Input file.",
)
@click.option(
    "--chunk-size",
    default=2,
    type=float,
    show_default=True,
    help="H5 file chunk size.",
)
def subexposures(
    conf, input, output, numberOfThreads, debug, logger, plot, chunk_size
):
    """Create and plot sub-exposures."""
    _set_log(debug, logger, output)
    _set_threads(numberOfThreads)
    RunConfig.chunk_size = chunk_size

    recipes.CreateSubExposures(
        options_file=conf, input_file=input, output_file=output
    )

    if plot:
        _plot_subexposures(output)


@cli.command()
@common_options
@click.option(
    "-i",
    "--input",
    "input",
    required=True,
    type=click.Path(exists=True),
    help="Input file.",
)
@click.option(
    "--chunk-size",
    default=2,
    type=float,
    show_default=True,
    help="H5 file chunk size.",
)
def ndrs(
    conf, input, output, numberOfThreads, debug, logger, plot, chunk_size
):
    """Create and plot NDRs."""
    _set_log(debug, logger, output)
    _set_threads(numberOfThreads)
    RunConfig.chunk_size = chunk_size

    recipes.CreateNDRs(options_file=conf, input_file=input, output_file=output)

    if plot:
        _plot_ndrs(output)


# Support functions for shared behaviour
def _set_log(debug, log, output):
    """Configure logging based on options."""
    if debug:
        setLogLevel(logging.DEBUG)
    if log:
        log_file = "exosim.log"
        if output:
            log_dir = os.path.dirname(output)
            log_file = os.path.join(log_dir, "exosim.log")
        try:
            addLogFile(fname=log_file)
        except PermissionError:
            addLogFile(fname="exosim.log")


def _set_threads(numberOfThreads):
    """Set the number of threads for parallel processing."""
    if numberOfThreads:
        RunConfig.n_job = numberOfThreads


# Plot functions for specific outputs
def _plot_focal_plane(output, scale):
    """Generate focal plane plots."""
    import exosim.plots as plots

    dir_name = os.path.join(os.path.dirname(output), "plots")
    os.makedirs(dir_name, exist_ok=True)

    focal_plane_plotter = plots.FocalPlanePlotter(input=output)
    focal_plane_plotter.plot_focal_plane(time_step=0, scale=scale)
    focal_plane_plotter.save_fig(os.path.join(dir_name, "focal_plane.png"))
    focal_plane_plotter.plot_efficiency()
    focal_plane_plotter.save_fig(os.path.join(dir_name, "efficiency.png"))


def _plot_radiometric(output):
    """Generate radiometric model plots."""
    import exosim.plots as plots

    dir_name = os.path.join(os.path.dirname(output), "plots")
    os.makedirs(dir_name, exist_ok=True)

    radiometric_plotter = plots.RadiometricPlotter(input=output)
    radiometric_plotter.plot_table()
    radiometric_plotter.save_fig(os.path.join(dir_name, "radiometric.png"))
    radiometric_plotter.plot_apertures()
    radiometric_plotter.save_fig(os.path.join(dir_name, "apertures.png"))


def _plot_subexposures(output):
    """Generate sub-exposures plots."""
    import exosim.plots as plots

    dir_name = os.path.join(os.path.dirname(output), "plots")
    os.makedirs(dir_name, exist_ok=True)

    sub_exposures_plotter = plots.SubExposuresPlotter(input=output)
    sub_exposures_plotter.plot(dir_name)


def _plot_ndrs(output):
    """Generate NDRs plots."""
    import exosim.plots as plots

    dir_name = os.path.join(os.path.dirname(output), "plots")
    os.makedirs(dir_name, exist_ok=True)

    ndrs_plotter = plots.NDRsPlotter(input=output)
    ndrs_plotter.plot(dir_name)


if __name__ == "__main__":
    cli()
