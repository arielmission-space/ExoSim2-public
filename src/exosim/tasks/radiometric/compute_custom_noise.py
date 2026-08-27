from collections import OrderedDict

import numpy as np
from astropy import units as u
from astropy.table import QTable

from exosim.models.signal import Signal
from exosim.tasks.task import Task
from exosim.utils.klass_factory import find_task


class ComputeCustomNoise(Task):
    r"""
    Computes custom noise contributions for radiometric calculations.

    This task processes user-defined noise sources that can be added to the
    radiometric noise budget. It supports three different input formats to
    accommodate various noise specification methods:

    1. **Spectral data tables**: wavelength-dependent noise from data files
    2. **Multiple noise sources**: Using OrderedDict for several contributions
    3. **Single noise source**: Simple dictionary for one contribution

    The task combines multiple noise contributions using quadrature addition:
    :math:`\\sigma_{total} = \\sqrt{\\sigma_1^2 + \\sigma_2^2 + ... + \\sigma_n^2}`


    Parameters
    ----------
    wavelength : astropy.units.Quantity
        wavelength array defining the spectral grid for noise calculation.
        Must have units of length (e.g., micron, nm, angstrom).
    description : dict, optional
        Channel description dictionary containing the custom noise specification
        under ``description["radiometric"]["custom_noise"]``. If None or if
        custom_noise is not present, returns zero noise.
    radiometric_table : astropy.table.QTable, optional
        Radiometric table that may be used for noise calculations if needed.

    Returns
    -------
    astropy.table.QTable
        Table containing:
        - ``custom_noise`` : Combined noise from all sources (units: hr**0.5)
        - Individual noise columns for each named source (units: hr**0.5)

    Notes
    -----
    - For spectral data, noise values are rebinned to the target wavelength grid
      using the Signal class spectral_rebin method
    - For constant values (OrderedDict and simple dict), the ``value`` field is
      assumed to be in ppm and is converted using a 1e-6 factor
    - All noise contributions are combined in quadrature
    - The output table includes both individual noise components and the total
    """

    def __init__(self):
        """Initialize the ComputeCustomNoise task with required parameters."""
        self.add_task_param("wavelength", "wavelength array")
        self.add_task_param("description", "channel description", None)
        self.add_task_param("radiometric_table", "radiometric table", None)

    def execute(self):
        self.debug("compute custom noise")
        wavelength = self.get_task_param("wavelength")
        description = self.get_task_param("description")
        radiometric_table = self.get_task_param("radiometric_table")

        noise_table, total_noise = self.model(
            wavelength, description, radiometric_table
        )

        if not isinstance(noise_table, QTable):
            self.error("noise table must be a QTable")
            raise TypeError("noise table must be a QTable")

        self.set_output([noise_table, total_noise])

    def model(self, wl, description, radiometric_table):
        """
        Compute custom noise contributions for the given wavelengths.

        Parameters
        ----------
        wl : astropy.units.Quantity
            wavelength array with shape (N,) and units of length.
        description : dict or None
            Channel specification dictionary. Custom noise should be located at
            ``description["radiometric"]["custom_noise"]``.
        radiometric_table : astropy.table.QTable or None
            Radiometric table for potential use in noise calculations.

        Returns
        -------
        tuple
            (noise_table, total_custom_noise) where:
            - noise_table : QTable with individual noise contributions (no total column)
            - total_custom_noise : Quantity array with total combined noise
        """
        # Initialize custom noise table (without total column)
        if wl.size == 0:
            self.error("empty wavelength array provided")
            raise ValueError("empty wavelength array provided")

        custom_noise_table = QTable()
        total_variance = np.zeros(wl.size) * (u.hr**0.5) ** 2  # Accumulate variance

        # Check if custom noise is specified in the description
        if description is None or "radiometric" not in description:
            self.debug("no radiometric section in description")
            total_custom_noise = np.sqrt(total_variance)
            return custom_noise_table, total_custom_noise

        if "custom_noise" not in description["radiometric"]:
            self.debug("no custom noise specified in radiometric section")
            total_custom_noise = np.sqrt(total_variance)
            return custom_noise_table, total_custom_noise

        custom = description["radiometric"]["custom_noise"]

        if isinstance(custom, dict) and not isinstance(custom, OrderedDict):
            self.debug("processing single custom noise source from dict")
            custom_noise_table, total_variance = self.process_contribution(
                custom, wl, custom_noise_table, total_variance, radiometric_table
            )
        elif isinstance(custom, OrderedDict):
            self.debug(
                f"processing {len(custom)} custom noise sources from OrderedDict"
            )
            for contrib_data in custom.values():
                custom_noise_table, total_variance = self.process_contribution(
                    contrib_data,
                    wl,
                    custom_noise_table,
                    total_variance,
                    radiometric_table,
                )
        else:
            self.error("custom noise must be an OrderedDict or dict")
            raise TypeError("custom noise must be an OrderedDict or dict")

        total_custom_noise = np.sqrt(total_variance)

        self.debug(
            f"Total custom noise computed: min={np.min(total_custom_noise)}, "
            f"max={np.max(total_custom_noise)}"
        )
        return custom_noise_table, total_variance**0.5

    def process_contribution(
        self,
        contrib_data,
        wl,
        custom_noise_table,
        total_variance,
        radiometric_table=None,
    ):
        # Extract data from file
        if "data" in contrib_data:
            custom_noise_table, partial_total_variance = self.load_from_file(
                contrib_data, wl, custom_noise_table
            )
            total_variance += partial_total_variance
        # Extract from task (not implemented)
        elif "task" in contrib_data:
            self.debug("processing custom noise source from task")
            compute_noise_task = find_task(
                contrib_data["task"],
                baseclass=ComputeCustomNoise,
            )
            computeNoise = compute_noise_task()
            custom_noise_table, noise = computeNoise(
                wavelength=wl,
                description=contrib_data,
                radiometric_table=radiometric_table,
            )
            total_variance += noise**2

        # Extract constant value
        else:
            self.debug("processing single custom noise source")

            noise_contribution = self.parse_constants(contrib_data, wl)

            # Generate column name
            base_name = contrib_data.get("value", "custom")
            if "name" in contrib_data:
                base_name = contrib_data["name"]

            noise_name = f"{base_name}_noise" if "noise" not in base_name else base_name

            self.debug(
                f"Added single custom noise '{noise_name}': {noise_contribution} "
            )

            # Store individual contribution and add to total variance
            custom_noise_table[noise_name] = noise_contribution
            total_variance += noise_contribution**2

        return custom_noise_table, total_variance

    def parse_constants(self, contrib_data, wl):
        # Extract value and convert from ppm
        if isinstance(contrib_data, dict):
            # Try different possible keys for the noise value
            if "noise_level" in contrib_data:
                if isinstance(contrib_data["noise_level"], dict):
                    scale_factor = contrib_data["noise_level"].get("scale", 1)
                    noise_value = contrib_data["noise_level"]["value"] * float(
                        scale_factor
                    )
                else:
                    noise_value = contrib_data["noise_level"]
            elif "value" in contrib_data:
                # For simple XML: <custom_noise> 20 <name>test1</name> </custom_noise>
                noise_value = contrib_data["value"]
            else:
                raise KeyError(f"noise value not found in contrib_data: {contrib_data}")
        else:
            noise_value = contrib_data
        return noise_value * np.ones(wl.size) * u.hr**0.5

    def load_from_file(self, custom, wl, custom_noise_table):
        self.debug("processing spectral custom noise data")
        col_names = [col for col in custom["data"] if "wavelength" not in col]
        total_variance = np.zeros(wl.size) * (u.hr**0.5) ** 2  # Accumulate variance

        for col_name in col_names:
            # Create Signal object and perform spectral rebinning
            custom_noise_signal = Signal(
                custom["data"]["wavelength"], custom["data"][col_name]
            )
            custom_noise_signal.spectral_rebin(wl)
            noise_contribution = (
                custom_noise_signal.data[0, 0, :] * custom_noise_signal.data_units
            )

            # Generate unique column name
            base_name = custom.get("name", "custom")
            base_name += f"_{col_name}"
            noise_name = f"{base_name}_noise" if "noise" not in base_name else base_name

            self.debug(
                f"Added spectral custom noise '{noise_name}' from column '{col_name}': "
                f"min={np.min(noise_contribution)}, max={np.max(noise_contribution)}"
            )
            # Store individual contribution
            custom_noise_table[noise_name] = noise_contribution
            total_variance += noise_contribution**2

        return custom_noise_table, total_variance
