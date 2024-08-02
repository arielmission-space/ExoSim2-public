from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

import exosim.tasks.instrument as instrument
from exosim.tasks.task import Task
from exosim.utils.focal_plane_locations import locate_wavelength_windows
from exosim.utils.klass_factory import find_task
from exosim.utils.psf import create_psf


class PopulateFocalPlane(Task):
    """
    It populates the empty focal plane with monocromatic PSFs.

    Returns
    -------
    :class:`~exosim.models.signal.Signal`
        focal plane array populated
    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            channel parameter dictionary. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        focal_plane: :class:`~exosim.models.signal.Signal`
            focal plane array (with time evolution)
        sources:  dict
            dictionary containing :class:`~exosim.models.signal.Sed`
        pointing: (:class:`astropy.units.Quantity`, :class:`astropy.units.Quantity`) (optional)
            telescope pointing direction, expressed ad a tuple of RA and DEC in degrees. Default is ``None``
        psf: :class:`numpy.ndarray` (optional)
            PSF array. Default is ``None``
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        """

        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param("focal_plane", "focal plane")
        self.add_task_param("sources", "sources")
        self.add_task_param("psf", "psf array", None)
        self.add_task_param("output", "output file", None)
        self.add_task_param("pointing", "telescope pointing", None)

    def execute(self):
        parameters = self.get_task_param("parameters")
        focal_plane = self.get_task_param("focal_plane")
        sources = self.get_task_param("sources")
        output = self.get_task_param("output")
        psf = self.get_task_param("psf")
        pointing = self.get_task_param("pointing")

        # load the psf
        psf = (
            self.load_psf(parameters, focal_plane, output)
            if psf is None
            else psf
        )

        i0_, j0_ = locate_wavelength_windows(psf, focal_plane, parameters)

        if sources:
            for source in sources.keys():
                self.info("populating focal plane with {}".format(source))

                j0 = deepcopy(j0_)
                i0 = deepcopy(i0_)

                # applying pointing offset in units of sub pixels
                compute_offset = instrument.ComputeSourcesPointingOffset()
                offset_spectral, offset_spatial = compute_offset(
                    source=sources[source].metadata,
                    pointing=pointing,
                    parameters=parameters,
                )

                j0 += offset_spectral
                i0 += offset_spatial

                focal_plane.data = populate(
                    i0, j0, psf, focal_plane.data, sources[source].data
                )

            focal_plane.data_units = sources[source].data_units
        # focal_plane.data.astype(np.float64)

        self.set_output([focal_plane, psf])

    @staticmethod
    def load_psf(parameters, focal_plane, output):
        """
        Loads the PSF as indicated in the configuration file or it creates them.

        Parameters
        ----------
        parameters: dict
            channel parameter dictionary. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        focal_plane: :class:`~exosim.models.signal.Signal`
            focal plane array (with time evolution)
        output: :class:`~exosim.output.output.Output` (optional)
            output file

        Returns
        -------
        :class:`~numpy.ndarray`
            four dimensional array: axis 0 is time, axis 1 is wavelength,
            axis 2 is spatial, axis 3 is spectral.


        """
        if "psf_task" in parameters["psf"].keys():
            psf_task = find_task(
                parameters["psf"]["psf_task"], instrument.LoadPsf
            )
            psf_instance = psf_task()
            psf, _ = psf_instance(
                filename=parameters["psf"]["filename"],
                parameters=parameters,
                wavelength=focal_plane.spectral * focal_plane.spectral_units,
                time=focal_plane.time * focal_plane.time_units,
                output=output,
            )
        # create the psf
        else:
            nzero = (
                parameters["psf"]["nzero"]
                if "nzero" in parameters["psf"].keys()
                else 4
            )
            if "size" in parameters["psf"].keys():
                array_size = [
                    parameters["psf"]["size_y"],
                    parameters["psf"]["size_x"],
                ]
            else:
                array_size = None

            psf = create_psf(
                focal_plane.spectral * focal_plane.spectral_units,
                (parameters["Fnum_x"], parameters["Fnum_y"]),
                focal_plane.metadata["focal_plane_delta"],
                shape=parameters["psf"]["shape"].lower(),
                nzero=nzero,
                max_array_size=focal_plane.data[0].shape,
                array_size=array_size,
            )
            psf = psf[np.newaxis, ...]
            psf = np.repeat(psf, focal_plane.time.size, axis=0)

            # store the psf
            if output is not None:
                output_group = output.create_group("psf")
                output_group.write_array(
                    "psf_cube",
                    psf,
                    metadata={
                        "spatial fp_axis": 1,
                        "spectral_fp_axis": 2,
                        "wavelength_axis": 0,
                    },
                )
                norms = psf.sum(axis=0).sum(axis=0)
                output_group.write_array("norm", norms)
                output_group.write_quantity(
                    "wavelength",
                    focal_plane.spectral * focal_plane.spectral_units,
                )

        return psf


def populate(
    i0: np.array,
    j0: np.array,
    psf: np.array,
    focal_plane: np.array,
    source: np.array,
) -> np.array:
    """it populates the focal plane adding the pfs"""
    for k in tqdm(range(len(i0)), desc="populating"):
        # this section is to avoid the psf to be out of the focal plane
        # i and j are the indexes of the psf on the focal plane
        # i_start and j_start are the indexes of the start cropped psf on the psf array
        # i_stop and j_stop are the indexes of the end cropped psf on the psf array

        # set the expected end of the psf
        i1 = i0[k] + psf.shape[2]
        j1 = j0[k] + psf.shape[3]

        # psf starting index
        i_start, j_start = 0, 0
        # check that teh expected start is not out of the focal plane
        if i0[k] < 0:
            i_start = -i0[k]
            i0[k] = 0
        if j0[k] < 0:
            j_start = -j0[k]
            j0[k] = 0

        # set the actual end of the psf
        i_stop, j_stop = i_start + psf.shape[2], j_start + psf.shape[3]

        # check that the expected end is not out of the focal plane
        if i1 > focal_plane.shape[1]:
            i_stop = i_start + focal_plane.shape[1] - i0[k]
            i1 = focal_plane.shape[1]
        if j1 > focal_plane.shape[2]:
            j_stop = j_start + focal_plane.shape[2] - j0[k]
            j1 = focal_plane.shape[2]

        for i in range(focal_plane.shape[0]):
            psfi = i
            if psf.shape[0] == 1:
                psfi = 0
            focal_plane[i, i0[k] : i1, j0[k] : j1] += (
                psf[psfi, k, i_start:i_stop, j_start:j_stop] * source[i, 0, k]
            )
    return focal_plane
