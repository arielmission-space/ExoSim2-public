import gc
import os
import re
from copy import deepcopy

import astropy.units as u
import h5py
import numpy as np

import exosim.tasks.instrument.loadPsfPaos as loadPsfPaos


class LoadPsfPaosTimeInterp(loadPsfPaos.LoadPsfPaos):
    """
    It loads the PSFs from two PAOS files and interpolates over time between them according to the given parameters.
    Each PAOS file contains a PSF vs wavelength.
    The task loads the PSF cubes provided by the `filename` parameter.
    The PSFs are then interpolated over a grid matching the one used to produce the focal planes,
    to convert them into the physical units.
    Then the total volume of the interpolated PSF is rescaled to the total volume of the original one.
    This allow to take into account for loss in the transmission due to the optical path.
    The PSF are then interpolated over a wavelength grid matching the one used to for the focal plane, producing the cube.

    Then an example of time dependency is implemented in the model method.

    The PSF cube is then cropped to removed unused edges, where the signal is less than 1/100 of the PSF peak.
    This would fasten up the successive `ExoSim` step.

    Returns
    -------
     :class:`~numpy.ndarray`
        cube of psfs.  axis=0 is time, axis=1 is wavelength, axis=2 is spatial direction, axis=3 is spectral direction.

    Note
    -----
    `filename` must be a string of filenames separated by a comma and a space.
    To write your version, define a new class that inherits from this one in a dedicated python file.
    Then copy the `model` method and replace the indicated section (where `REPLACE THIS WITH YOUR MODE` is written).

    """

    def model(self, filename, parameters, wavelength, time):
        self.debug("loading PAOS psf with time dependency")

        psfs = self.load_psfs(filename, wavelength, parameters)

        # prepare output ipercube
        psf_out = np.empty(
            (time.size, wavelength.size, psfs[0].shape[1], psfs[0].shape[2])
        )

        ################################
        # time evolution example.
        # REPLACE THIS WITH YOUR MODEL
        t0 = parameters["psf"]["t0"]
        for i, t in enumerate(time):
            psf_out[i] = (t / t0) * psfs[0] + (1 - t / t0) * psfs[1]

        ################################
        # clean memory
        del psfs
        gc.collect()

        # crop image stack
        psf_out_cropped = self.crop_image_stack(psf_out, time_iteration=True)

        # re-normalize output
        psf_out_cropped = self.normalise(psf_out, psf_out_cropped)

        return psf_out_cropped

    @staticmethod
    def normalise(psf_out, psf_out_cropped):
        old_norms = psf_out.sum(axis=-1).sum(axis=-1)
        new_norms = psf_out_cropped.sum(axis=-1).sum(axis=-1)
        dim_array = np.ones((1, psf_out.ndim), int).ravel()
        dim_array[0] = psf_out.shape[0]
        dim_array[1] = psf_out.shape[1]

        del psf_out  # clean memory
        gc.collect()

        old_norms_reshaped = old_norms.reshape(dim_array)
        new_norms_reshaped = new_norms.reshape(dim_array)
        psf_out_cropped_norm_ = psf_out_cropped / new_norms_reshaped

        del psf_out_cropped  # clean memory
        gc.collect()

        psf_out_cropped = psf_out_cropped_norm_ * old_norms_reshaped
        return psf_out_cropped

    def load_psfs(self, filename, wavelength, parameters):
        """

        Parameters
        ----------
         parameters: dict
             dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
         wavelength: :class:`~astropy.units.Quantity`
             wavelength grid.
         filename: str
             string containing PSF input files separated by a comma

        Returns
        -------
        list
            list of psf cubes. axis=0 is wavelength, axis=1 is spatial direction, axis=2 is spectral direction.


        """
        if "oversampling" in parameters["detector"].keys():
            oversampling = parameters["detector"]["oversampling"]
        else:
            oversampling = 1
        delta_pix = parameters["detector"]["delta_pix"] / oversampling
        filename = filename.split(", ")

        psfs = []
        for file in filename:
            with h5py.File(os.path.expanduser(file), "r") as data:
                # find the sampled wl
                temp = re.compile(r"\d+(?:\.\d*)")
                wl_sampled_h5 = [ele for ele in data.keys() if temp.match(ele)]

                psf_cube = [
                    self.load_imac(
                        wl, data, parameters, oversampling, delta_pix
                    )
                    for wl in wl_sampled_h5
                ]
                wl_sampled = [
                    self.load_wl_sampled(wl, data) for wl in wl_sampled_h5
                ]

            wl_sampled = np.array(wl_sampled) * u.m
            psf_cube = np.array(psf_cube)

            # wl_interpolate to the output wl grid
            wl_unsorted = deepcopy(wavelength)
            wavelength.sort()
            psf_cube_interp = self.wl_interpolate(
                psf_cube, wavelength, wl_sampled.to(u.um)
            )
            # reorder the array
            idx = [int(np.where(wavelength == wl_)[0]) for wl_ in wl_unsorted]
            psf_cube_interp = psf_cube_interp[idx, ...]

            psfs.append(psf_cube_interp)

        return psfs
