import os
from collections import OrderedDict
from typing import List
from typing import Tuple
from typing import Union

import astropy.units as u
import h5py
import numpy as np

import exosim.log as log
import exosim.tasks.detector as detector
from exosim.models.signal import Counts
from exosim.output import HDF5OutputGroup
from exosim.output import SetOutput
from exosim.output.hdf5.utils import copy_file
from exosim.output.hdf5.utils import recursively_read_dict_contents
from exosim.utils.ascii_arts import astronomer3
from exosim.utils.checks import look_for_key
from exosim.utils.klass_factory import find_and_run_task
from exosim.utils.prepare_recipes import copy_input_files
from exosim.utils.prepare_recipes import load_options
from exosim.utils.runConfig import RunConfig
from exosim.utils.timed_class import TimedClass


class CreateNDRs(TimedClass, log.Logger):
    """
    Pipeline to create the observation NDRs.
    This pipeline loads the configuration file and the Sub-Exposures.

    Examples
    --------

    >>> import exosim.recipes as recipes
    >>> recipes.CreateNDRs(input_file='./input_file.h5',
    >>>                    output_file = 'ndrs_file.h5',
    >>>                    options_file='main _configuration.xml')

    """

    def __init__(
        self, input_file: str, output_file: str, options_file: Union[str, dict]
    ) -> None:
        """
        Parameters
        ----------
        input_file : str
            Sub-Exposure ExoSim product
        output_file : str
            output file name
        options_file: str or dict
            input configuration file
        """

        super().__init__()
        self.graphics(astronomer3)
        RunConfig.stats()

        self.announce("started")
        self.input = input_file
        self.mainConfig, self.payloadConfig = load_options(options_file)
        copy_input_files(os.path.dirname(os.path.abspath(output_file)))

        output = SetOutput(output_file)

        with h5py.File(self.input, "r") as f:
            ch_list = list(f["channels"].keys())
            ch_list.sort()
        with output.use(append=True, cache=True) as out:
            output_grp = out.create_group("channels")
            for ch in ch_list:
                ch_description = (
                    self.payloadConfig["channel"][ch]
                    if isinstance(self.payloadConfig["channel"], OrderedDict)
                    else self.payloadConfig["channel"]
                )
                # prepare output
                self.announce("iterating over {}".format(ch))
                ch_grp = output_grp.create_group(ch)

                # import exposures data
                (
                    spectral,
                    spatial,
                    mid_freq_time_line,
                    integration_times,
                    number_of_exposures,
                    n_subexposures_per_groups,
                    n_groups_per_ramp,
                ) = self.load_subexposure_data(ch)

                # preparing outputs
                sub_ndrs, sim_grp = self.prepare_output(
                    spectral,
                    spatial,
                    mid_freq_time_line,
                    integration_times,
                    ch,
                    ch_grp,
                )

                # add dark current
                if look_for_key(
                    ch_description["detector"], "dark_current", True
                ):
                    dc_instance = find_and_run_task(
                        ch_description["detector"],
                        "dc_task",
                        detector.AddConstantDarkCurrent,
                    )
                    dc_instance(
                        subexposures=sub_ndrs,
                        parameters=ch_description,
                        integration_times=integration_times,
                        output=sim_grp,
                    )

                # add shot noise
                if look_for_key(
                    ch_description["detector"], "shot_noise", True
                ):
                    addShotNoise = detector.AddShotNoise()
                    addShotNoise(subexposures=sub_ndrs, output=sim_grp)

                # add cosmic rays
                if look_for_key(
                    ch_description["detector"], "cosmic_rays", True
                ):
                    cosmic_instance = find_and_run_task(
                        ch_description["detector"],
                        "cosmic_rays_task",
                        detector.AddCosmicRays,
                    )
                    cosmic_instance(
                        subexposures=sub_ndrs,
                        parameters=ch_description,
                        integration_times=integration_times,
                        output=sim_grp,
                    )

                # prepare accumulation
                n_subexposures_per_ramp = (
                    sub_ndrs.dataset.shape[0] // number_of_exposures
                )
                base = np.arange(0, number_of_exposures, 1)
                state_machine = np.repeat(base, n_subexposures_per_ramp)

                # accumulate ndrs
                accumulateSubExposures = detector.AccumulateSubExposures()
                accumulateSubExposures(
                    subexposures=sub_ndrs, state_machine=state_machine
                )

                # add reset bias
                if look_for_key(
                    ch_description["detector"], "ktc_offset", True
                ):
                    reset_instance = find_and_run_task(
                        ch_description["detector"],
                        "ktc_offset_task",
                        detector.AddKTC,
                    )
                    reset_instance(
                        subexposures=sub_ndrs,
                        parameters=ch_description,
                        state_machine=state_machine,
                        output=sim_grp,
                    )

                # apply dead pixels map
                if look_for_key(
                    ch_description["detector"], "dead_pixels", True
                ):
                    dp_map_instance = find_and_run_task(
                        ch_description["detector"],
                        "dp_map_task",
                        detector.ApplyDeadPixelsMap,
                    )
                    dp_map_instance(
                        subexposures=sub_ndrs,
                        parameters=ch_description,
                        output=sim_grp,
                    )

                # apply pixel non linearity
                if look_for_key(
                    ch_description["detector"], "pixel_non_linearity", True
                ):
                    pnl_map_instance = find_and_run_task(
                        ch_description["detector"],
                        "pnl_map_task",
                        detector.LoadPixelsNonLinearityMap,
                    )
                    pnl_map = pnl_map_instance(
                        parameters=ch_description,
                        output=sim_grp,
                    )
                    pnl_apply_instance = find_and_run_task(
                        ch_description["detector"],
                        "pnl_task",
                        detector.ApplyPixelsNonLinearity,
                    )
                    pnl_apply_instance(
                        subexposures=sub_ndrs, parameters=pnl_map
                    )

                # apply pixel saturation
                if look_for_key(
                    ch_description["detector"], "saturation", True
                ):
                    sat_instance = find_and_run_task(
                        ch_description["detector"],
                        "sat_task",
                        detector.ApplySimpleSaturation,
                    )
                    sat_instance(
                        subexposures=sub_ndrs, parameters=ch_description
                    )

                # add read noise
                if look_for_key(
                    ch_description["detector"], "read_noise", True
                ):
                    read_instance = find_and_run_task(
                        ch_description["detector"],
                        "read_noise_task",
                        detector.AddNormalReadNoise,
                    )
                    read_instance(
                        subexposures=sub_ndrs,
                        parameters=ch_description,
                        output=sim_grp,
                    )

                # add gain drift
                if look_for_key(
                    ch_description["detector"], "gain_drift", True
                ):
                    gain_instance = find_and_run_task(
                        ch_description["detector"],
                        "gain_drift_task",
                        detector.addGainDrift,
                    )
                    gain_instance(
                        subexposures=sub_ndrs,
                        parameters=ch_description,
                        output=sim_grp,
                    )

                # converting to digital
                if look_for_key(ch_description["detector"], "ADC", True):
                    analogToDigital = detector.AnalogToDigital()
                    sub_ndrs = analogToDigital(
                        subexposures=sub_ndrs,
                        parameters=ch_description,
                        output=sim_grp,
                    )

                # merging groups' ndrs
                mergeGroups = detector.MergeGroups()
                merged_ndrs = mergeGroups(
                    subexposures=sub_ndrs,
                    n_groups=n_groups_per_ramp * number_of_exposures,
                    n_ndrs=n_subexposures_per_groups,
                    output=ch_grp,
                )

                merged_ndrs.metadata.update(
                    {"n_groups_per_ramp": n_groups_per_ramp}
                )
                merged_ndrs.write()

                # clean the output from redundant information
                self.clean_output_tree(ch_grp, ["simulation_data/sub_ndrs"])

                self.info("{} completed.".format(ch))
                self.log_runtime("{} ended in".format(ch), "info")

        self.refactor_output(output_file)
        self.info("output {} size: {:.3f}".format(output_file, out.getsize()))
        self.log_runtime_complete("recipe ended", "info")
        self.announce("ended")

    def load_subexposure_data(
        self, ch: str
    ) -> Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, int, int, int]:
        """It loads the sub-exposures data from file

        Parameters
        ----------
        ch : str
            channel name

        Returns
        -------
        :class:`astropy.units.Quantity`
            spectral axes array
        :class:`astropy.units.Quantity`
            spatial axes array
        :class:`astropy.units.Quantity`
            temporal array
        :class:`astropy.units.Quantity`
            integration times
        int
            number of exposures
        int
            number of NDRs per ramp
        int
            number of NDRs per group
        """
        with h5py.File(self.input, "r") as f:
            file_path = os.path.join("channels", ch)
            se = f[os.path.join(file_path, "SubExposures")]
            spectral = se["spectral"] * u.Unit(
                se["spectral_units"][()].decode("utf-8")
            )
            spatial = se["spatial"] * u.Unit(
                se["spatial_units"][()].decode("utf-8")
            )
            time_line = se["time"] * u.Unit(
                se["time_units"][()].decode("utf-8")
            )
            integration_times = se["metadata"]["integration_times"]["value"][
                ()
            ] * u.Unit(
                se["metadata"]["integration_times"]["unit"][()].decode("utf-8")
            )
            params = dict(
                f[os.path.join(file_path, "instantaneous_readout_params")]
            )
            try:
                number_of_exposures = params["number_of_exposures"][()]
            except TypeError:
                number_of_exposures = params["number_of_exposures"]["value"][
                    ()
                ]

            params = dict(f[os.path.join(file_path, "reading_scheme_params")])
            n_subexposures_per_groups = params["n_NRDs_per_group"][()]
            n_groups_per_ramp = params["n_GRPs"][()]

        return (
            spectral,
            spatial,
            time_line,
            integration_times,
            number_of_exposures,
            n_subexposures_per_groups,
            n_groups_per_ramp,
        )

    def prepare_output(
        self,
        spectral: u.Quantity,
        spatial: u.Quantity,
        time_line: u.Quantity,
        integration_times: u.Quantity,
        ch: str,
        ch_grp: HDF5OutputGroup,
    ) -> Tuple[Counts, HDF5OutputGroup]:
        """It produces the NDRs output per channel.

        Parameters
        ----------
        spectral : u.Quantity
            spectral axes array
        spatial : u.Quantity
            spatial axes array
        time_line : u.Quantity
            temporal array
        integration_times : u.Quantity
            integration times
        ch : str
            channel name
        ch_grp : h5py.Group
            output group

        Returns
        -------
        :class:`exosim.models.signal.Counts`
            NDRs signal class
        :class:`exosim.output.hdf5.h5df.HDF5OutputGroup`
            output group
        """

        sim_grp = ch_grp.create_group("simulation_data")

        with h5py.File(self.input, "r") as f:
            file_path = os.path.join("channels", ch)

            # copy simulation data
            sim_data = os.path.join(file_path, "simulation_data")
            for key in f[sim_data].keys():
                f.copy(os.path.join(sim_data, key), sim_grp._entry)
            f.copy("info", sim_grp._entry, name="sub_exposures_info")

            inst_rdo_data = os.path.join(
                file_path, "instantaneous_readout_params"
            )
            keys_to_keep = [
                "spe_jit_averaged",
                "spa_jit_averaged",
                "y_jit",
                "x_jit",
                "jit_indexes",
                "y_jit_averaged",
                "x_jit_averaged",
                "effective_osf",
            ]
            for key in f[inst_rdo_data].keys():
                if key in keys_to_keep:
                    f.copy(os.path.join(inst_rdo_data, key), sim_grp._entry)
            if "pointing_jitter" in f.keys():
                for key in f["pointing_jitter"]:
                    f.copy(
                        os.path.join("pointing_jitter", key), sim_grp._entry
                    )

            self.debug("simulation data copied to output")

            se = f[os.path.join(file_path, "SubExposures")]
            subexposures = se["data"]

            metadata = recursively_read_dict_contents(se["metadata"])
            # create sub_ndrs data
            sub_ndrs = Counts(
                spectral=spectral,
                time=time_line,
                data=None,
                spatial=spatial,
                shape=(
                    subexposures.shape[0],
                    subexposures.shape[1],
                    subexposures.shape[2],
                ),
                cached=True,
                output=sim_grp,
                dataset_name="sub_ndrs",
                output_path=None,
                metadata={"integration_times": integration_times},
                dtype=np.float64,
            )
            sub_ndrs.metadata.update(metadata)

            for chunk in sub_ndrs.dataset.iter_chunks():
                sub_ndrs.dataset[chunk] = subexposures[chunk]
                sub_ndrs.output.flush()
            sub_ndrs.write()

        return sub_ndrs, sim_grp

    def clean_output_tree(
        self, out: HDF5OutputGroup, key_list: List[str]
    ) -> None:
        """
        Remove specified keys and their associated data from the output object.

        This method iterates through a list of keys and deletes each key-value pair from the output object.
        Useful for cleaning up temporary or unnecessary data from the output.

        Parameters
        ----------
        out : HDF5OutputGroup
            The output object where key-value pairs are stored.
        key_list : list of str
            A list of keys that need to be removed from the output object.

        Examples
        --------
        >>> out = Output({'key1': 'value1', 'key2': 'value2', 'key3': 'value3'})
        >>> clean_output(out, ['key1', 'key3'])
        >>> out.data
        {'key2': 'value2'}

        Notes
        -----
        The method assumes that the keys specified in the `key_list` are present in the output object.
        If a key is not found, the `delete_data` method should handle it gracefully.
        """
        self.info("Cleaning the output tree")
        for key in key_list:
            out.delete_data(key)

    def refactor_output(self, output_file: str) -> None:
        """
        Renames the original output file and creates a new one with its contents.

        This method takes the original output file, renames it by appending '_unrefactored' to its name,
        and then copies its content into a new file with the original name. The refactored (old version)
        file is deleted after the operation.

        Parameters
        ----------
        output_file : str
            The path of the original output file.
        """

        # Log the refactoring process
        self.info("Refactoring output")

        # Split the output_file into filename and extension
        filename, file_extension = os.path.splitext(output_file)

        # Create a new name for the original file by appending '_unrefactored'
        new_name = "{}_unrefactored{}".format(filename, file_extension)

        # Rename the original file
        os.rename(output_file, new_name)

        # Open the renamed file and create a new output file
        # Then copy the contents from the renamed file to the new file
        with h5py.File(new_name, "r") as f:
            with h5py.File(output_file, "w") as g:
                copy_file(f, g)

        # Remove the renamed (old version) file
        os.remove(new_name)
