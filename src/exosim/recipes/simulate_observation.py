import os

import exosim.log as log
from exosim.plots import (
    FocalPlanePlotter,
    NDRsPlotter,
    RadiometricPlotter,
    SubExposuresPlotter,
)
from exosim.utils import RunConfig
from exosim.utils.timed_class import TimedClass

from .create_focal_plane import CreateFocalPlane
from .create_ndrs import CreateNDRs
from .create_sub_exposures import CreateSubExposures
from .radiometric_model import RadiometricModel


class SimulateObservation(TimedClass, log.Logger):
    """
    Pipeline to run a full observation from options file to final NDRs.
    The pipeline includes the following steps:
    1. Create the focal plane using :class:`~exosim.recipes.create_focal_plane.CreateFocalPlane`
    2. Run the radiometric model using :class:`~exosim.recipes.radiometric_model.RadiometricModel`
    3. Create the sub-exposures using :class:`~exosim.recipes.create_sub_exposures.CreateSubExposures`
    4. Create the NDRs using :class:`~exosim.recipes.create_ndrs.CreateNDRs`
    5. Optionally, generate plots at each stage if a plots directory is provided.
    6. Clean up temporary files created during the process.

    Parameters
    ----------
    The class accepts various parameters via its initializer, including options file path,
    output file path, number of jobs, random seed, and an optional plots directory.

    Attributes
    ----------
    options_file: str
        Path to the options file.
    output_file: str
        Path to the output file. This will contain the final NDRs.
    plots_dir: str, optional
        Directory to save the plots. If None, no plots are saved. Default is None.

    Examples
    --------
    >>> from exosim.recipes import SimulateObservation
    >>> full_observation = SimulateObservation(
    ...     options_file="path/to/options_file.yaml",
    ...     output_file="path/to/output_file.h5",
    ...     plots_dir="path/to/plots_dir",
    ...     n_job=4,
    ...     random_seed=42,
    ... )

    """

    def __init__(
        self,
        options_file: str,
        output_file: str,
        plots_dir: str | None = None,
        n_job: int = 1,
        random_seed=None,
    ):
        """
        Parameters
        ----------
        options_file: str
            Path to the options file.
        output_file: str
            Path to the output file. This will contain the final NDRs.
        plots_dir: str, optional
            Directory to save the plots. If None, no plots are saved. Default is None.
        n_job: int, optional
            Number of parallel jobs to run. Default is 1.
        random_seed: int, optional
            Random seed for reproducibility. Default is None.
        """
        super().__init__()
        self.options_file = options_file
        self.output_file = output_file
        self.plots_dir = plots_dir
        RunConfig.n_job = n_job
        if self.plots_dir is not None and not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        if random_seed is not None:
            RunConfig.random_seed = random_seed
        self.info(
            f"Running SimulateObservation with:\n options_file: {self.options_file}\n output_file: {self.output_file}\n plots_dir: {self.plots_dir}\n n_job: {RunConfig.n_job}\n random_seed: {RunConfig.random_seed}"
        )

    def main(self):
        """Main method to run the full observation pipeline."""
        self.announce("Starting SimulateObservation pipeline")
        # create focal plane
        CreateFocalPlane(self.options_file, "./test_common.h5")
        # run focal plane plotter
        if self.plots_dir is not None:
            focal_plane_plotter = FocalPlanePlotter(input="./test_common.h5")
            focal_plane_plotter.plot_focal_plane(time_step=0, scale="linear")
            focal_plane_plotter.save_fig(f"{self.plots_dir}/focal_plane.png")

        # Try to plot efficiency if data is available
        if self.plots_dir is not None:
            focal_plane_plotter.plot_efficiency()
            focal_plane_plotter.save_fig(f"{self.plots_dir}/efficiency.png")

        # # run radiometric model
        RadiometricModel(self.options_file, "./test_common.h5")
        # run radiometric plotter
        if self.plots_dir is not None:
            radiometric_plotter = RadiometricPlotter(input="./test_common.h5")
            radiometric_plotter.plot_table(contribs=False)
            radiometric_plotter.save_fig(f"{self.plots_dir}/radiometric.png")
            radiometric_plotter.plot_apertures()
            radiometric_plotter.save_fig(f"{self.plots_dir}/apertures.png")

        # create Sub-Exposures
        CreateSubExposures(
            input_file="./test_common.h5",
            output_file="./test_se.h5",
            options_file=self.options_file,
        )
        # run Sub-Exposures plotter
        if self.plots_dir is not None:
            sub_exposures_plotter = SubExposuresPlotter(input="./test_se.h5")
            sub_exposures_plotter.plot(f"{self.plots_dir}/subexposures")

        # create NDRs
        CreateNDRs(
            input_file="./test_se.h5",
            output_file=self.output_file,
            options_file=self.options_file,
        )
        # run NDRs plotter
        if self.plots_dir is not None:
            ndrss_plotter = NDRsPlotter(input=self.output_file)
            ndrss_plotter.plot(f"{self.plots_dir}/ndrs")

        self.info("Cleaning up temporary files")
        os.remove("./test_common.h5")
        os.remove("./test_se.h5")
