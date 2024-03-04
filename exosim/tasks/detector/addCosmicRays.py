from copy import deepcopy
from fractions import Fraction
from typing import List
from typing import Tuple

import astropy.units as u
import numpy as np
from astropy.table import Table

from exosim.models.signal import Signal
from exosim.output import Output
from exosim.tasks.task import Task
from exosim.utils import RunConfig
from exosim.utils.checks import check_units
from exosim.utils.iterators import iterate_over_chunks
from exosim.utils.operations import operate_over_axis
from exosim.utils.types import ArrayType


class AddCosmicRays(Task):
    """
    Task to simulate the addition of cosmic rays to sub-exposures in a detector.

    This task models the impact of cosmic rays on a detector during the exposure time. The model assumes
    that the cosmic rays can interact with the detector pixels in various predefined shapes like a cross,
    horizontal rectangle, and vertical rectangle. The number of cosmic ray events and their impact on the
    detector pixels are calculated based on the given cosmic ray flux, detector characteristics, and
    integration times for the sub-exposures.

    Notes
    -----
    - This task assumes that the cosmic ray events saturate the affected pixels, setting their value to the full well depth.
    - The probabilities for the interaction shapes are configurable. The task issues a warning if the sum of provided probabilities is not 1.
    - The cosmic ray flux is specified in ct/s/cm^2 and is scaled based on the pixel size and detector dimensions.

    This is a default class with standardised inputs and outputs.
    The user can load this class and overwrite the "model" method
    to implement a custom Task to replace this.


    """

    def __init__(self):
        """
        Parameters
        ----------
        subexposures: :class:`~exosim.models.signal.Counts`
            sub-exposures cached signal
        parameters: dict
            channel parameters dictionary
        integration_times: :class:`~astropy.units.Quantity`
            sub-exposures integration times
        outputs: :class:`~exosim.output.output.Output` (optional)
            output file
        """

        self.add_task_param("subexposures", "sub-exposures cached signal")
        self.add_task_param("parameters", "channel parameters dictionary")
        self.add_task_param(
            "integration_times",
            "subexposures integration times",
        )
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("adding cosmic rays")
        subexposures = self.get_task_param("subexposures")
        parameters = self.get_task_param("parameters")
        integration_times = self.get_task_param("integration_times")
        output = self.get_task_param("output")

        self.model(subexposures, parameters, integration_times, output)

    def model(
        self,
        subexposures: Signal,
        parameters: dict,
        integration_times: ArrayType,
        output=None,
    ) -> None:
        """
        Default model to simulate the addition of cosmic rays to the sub-exposures.

        This method saturates the hit pixels in the sub-exposure data based on the cosmic ray rate,
        detector properties, and the given integration times. The cosmic ray interactions could be in various shapes
        like single pixel, lines, squares, etc., which are defined in the configuration.

        Parameters
        ----------
        subexposures : Signal
            The sub-exposures' cached signal.
        parameters : dict
            The channel parameters dictionary containing detector information.
        integration_times : ArrayType
            The integration times for the sub-exposures.
        output : Output, optional
            The output file where the cosmic ray events will be stored.

        Notes
        -----
        The method assumes multiple possible shapes for the interaction:
        - Single pixel
        - Vertical line: Saturates two pixels vertically aligned.
        - Horizontal line: Saturates two pixels horizontally aligned.
        - Square: Saturates four pixels in a square.
        - Cross: Saturates five pixels in a cross shape.
        - Horizontal rectangle: Saturates six pixels in a horizontal rectangle shape.
        - Vertical rectangle: Saturates six pixels in a vertical rectangle shape.
        """
        # Extract relevant parameters from the provided dictionary
        rate = parameters["detector"]["cosmic_rays_rate"].astype(np.float64)
        rate = check_units(rate, "ct/s/cm^2")
        saturation_rate = parameters["detector"]["saturation_rate"]
        spatial_pix = parameters["detector"]["spatial_pix"]
        spectral_pix = parameters["detector"]["spectral_pix"]
        well_depth = parameters["detector"]["well_depth"]
        well_depth = check_units(well_depth, "ct", force=True).value
        pixel_size = parameters["detector"]["delta_pix"].astype(np.float64)

        # Calculate the number of events to be added to each sub-exposure
        events_counter = self.count_events(
            rate,
            pixel_size,
            spectral_pix,
            spatial_pix,
            integration_times,
            saturation_rate,
        )
        self.info(f"cosmic events_counter: {events_counter}")

        # Get interaction shapes and their probabilities
        shapes, probs = self.shapes_and_probs(parameters["detector"])

        # Initialize a table to store information about the cosmic ray events
        storing_info = Table(
            names=(
                "Sub-exposure",
                "spectral_pix",
                "spatial_pix",
                "saturated_pix_rand",
                "saturated_pix_seed",
                "saturated_pix",
            ),
            dtype=("i4", "i4", "i4", "f8", "i4", "U16"),
        )

        # Iterate over the sub-exposures and add cosmic ray events
        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="adding cosmic rays"
        ):
            data = deepcopy(subexposures.dataset[chunk])
            events_counter_chunk = events_counter[chunk[0]]
            for t in range(data.shape[0]):
                # Randomly select pixels for the cosmic ray events
                x_pix = RunConfig.random_generator.choice(
                    spectral_pix, events_counter_chunk[t]
                )
                y_pix = RunConfig.random_generator.choice(
                    spatial_pix, events_counter_chunk[t]
                )

                for x, y in zip(x_pix, y_pix):
                    info_row = [chunk[0].start + t, x, y]

                    # Randomly select a shape for the cosmic ray event
                    rand = RunConfig.random_generator.uniform(0, 1)
                    info_row.append(rand)
                    info_row.append(RunConfig.random_seed)

                    r_ = 0
                    for s, r in zip(shapes, probs):
                        r_ += r
                        if rand <= r_:
                            shape = s
                            break

                    shape_str = ""
                    for i, j in zip(shape[0], shape[1]):
                        shape_str += f"({i + x}, {j + y})"
                        try:
                            data[t, j + y, i + x] = well_depth
                        except IndexError:
                            continue

                    info_row.append(shape_str)
                    storing_info.add_row(info_row)

            # Update the sub-exposure data
            subexposures.dataset[chunk] = data
            subexposures.output.flush()

        # Write the cosmic ray events information to the output file if provided
        if output and issubclass(output.__class__, Output):
            out_grp = output.create_group("cosmic rays")
            out_grp.write_table("cosmic rays events", storing_info)
            out_grp.write_array("cosmic events counter", events_counter)

    def count_events(
        self,
        rate: float,
        pixel_size: u.Quantity,
        spatial_pix: int,
        spectral_pix: int,
        integration_times: ArrayType,
        saturation_rate: float,
    ) -> int:
        """
        Calculate the number of cosmic ray events in each sub-exposure.

        Parameters
        ----------
        rate : float
            Cosmic rays flux rate in ct/s/cm^2.
        pixel_size : float
            Size of a detector pixel in cm^2.
        spatial_pix : int
            Number of spatial pixels in the detector.
        spectral_pix : int
            Number of spectral pixels in the detector.
        integration_times : :class:`~astropy.units.Quantity`
            Sub-exposure integration times.
        saturation_rate : float
            Saturation rate for the detector.

        Returns
        -------
        events_counter_i : np.ndarray
            An array containing the number of events for each sub-exposure, rounded to the nearest integer.

        Notes
        -----
        The method first scales the rate based on the pixel size and the number of pixels in both spatial
        and spectral dimensions. It then multiplies the scaled rate by the integration times and saturation rate
        to get the number of events. The fractional part of the events is handled probabilistically.
        """

        # Scale the rate based on detector characteristics
        scaled_rate = (
            rate * pixel_size * pixel_size * spatial_pix * spectral_pix
        )
        scaled_rate = check_units(scaled_rate, "ct/s")

        # Log the calculated rate
        self.info("cosmic rays rate: {}".format(scaled_rate))

        # Calculate the expected number of events for each sub-exposure
        events_counter = scaled_rate * integration_times * saturation_rate

        # Handle the fractional part of the events
        events_counter_d, events_counter_i = np.modf(events_counter.value)
        fract = [Fraction(e).limit_denominator() for e in events_counter_d]
        ref_odds = np.array([[f.numerator, f.denominator] for f in fract])
        ref, odds = ref_odds[:, 0], ref_odds[:, 1]

        # Probabilistically round the number of events
        rand = np.random.choice(odds, 1)
        id_ev = np.where(rand <= ref)[0]
        events_counter_i[id_ev] += 1

        return np.array(events_counter_i).astype(int)

    def shapes_and_probs(self, parameters: dict) -> Tuple[List, List]:
        """
        Generate cosmic ray interaction shapes and their corresponding probabilities.

        Parameters
        ----------
        parameters : dict
            A dictionary containing optional 'interaction_shapes' which is a sub-dictionary
            specifying shapes and their probabilities.

        Returns
        -------
        shapes : list of tuples
            A list of tuples representing shapes of cosmic ray interactions.
        probs : list of float
            A list of probabilities corresponding to the shapes.

        Raises
        ------
        ValueError
            If the sum of provided probabilities for all shapes is greater than 1.

        Examples
        --------
        >>> shapes, probs = AddCosmicRays.shapes_and_probs({"interaction_shapes": {"single": 0.5, "line_h": 0.3, "line_v": 0.2}})
        >>> print(shapes)
        [([0], [0]), ([0, 1], [0, 0]), ([0, 0], [0, 1]), ([0, 0, 1, 1], [0, 1, 0, 1]), ([0, 0, 0, -1, 1], [0, 1, -1, 0, 0])]
        >>> print(probs)
        [0.5, 0.3, 0.2, 0.1, 0.1]

        >>> shapes, probs = AddCosmicRays.shapes_and_probs({"interaction_shapes": {"single": 0.5, "line_h": 0.3, "line_v": 0.2}})
        >>> print(shapes)

        Warnings
        --------
        A warning is issued if the sum of provided probabilities for all shapes is not 1.
        """

        # Predefined cosmic ray interaction shapes
        predefined_shapes = {
            "single": ([0], [0]),
            "line_h": ([0, 1], [0, 0]),
            "line_v": ([0, 0], [0, 1]),
            "square": ([0, 0, 1, 1], [0, 1, 0, 1]),
            "cross": ([0, 0, 0, -1, 1], [0, 1, -1, 0, 0]),
            "rect_h": ([0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1]),
            "rect_v": ([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]),
        }

        # Initialize lists to store shapes and probabilities
        shapes, probs = [], []

        # Check if interaction shapes are defined in the parameters
        if "interaction_shapes" in parameters:
            for shape_key, prob in parameters["interaction_shapes"].items():
                if shape_key in predefined_shapes:
                    shapes.append(predefined_shapes[shape_key])
                    probs.append(prob)

            # Check if probabilities sum up to 1
            remaining_prob = 1.0 - sum(probs)
            if remaining_prob < 0:
                self.error("Total probability greater than 1.")
                raise ValueError("Total probability greater than 1.")

            # Adjust the probability for the 'single' shape if needed
            if remaining_prob != parameters["interaction_shapes"].get(
                "single", 0
            ):
                self.warning(
                    "Total probability of all interaction shapes should sum to 1. "
                    "Adjusting 'single' interaction shape to make the sum equal to 1."
                )
                shapes.append(predefined_shapes["single"])
                probs.append(remaining_prob)

            return shapes, probs

        # Default to 'single' shape with probability 1 if no shapes are defined
        return [predefined_shapes["single"]], [1.0]
