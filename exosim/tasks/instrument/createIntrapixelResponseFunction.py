import astropy.units as u
import numpy as np

import exosim.output as output
from exosim.tasks.task import Task


class CreateIntrapixelResponseFunction(Task):
    """
    It creates the intrapixel response function to be used in the convolution with the focal plane array.


    Returns
    -------
    kernel	: 2D array
            the kernel image
    kernel_delta  : scalar
                    the kernel sampling interval in microns


    Notes
    -----
    This is a default class with standardised inputs and outputs.
    The user can load this class and overwrite the "model" method
    to implement a custom Task to replace this.
    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            channel parameter dictionary. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        output: :class:`~exosim.output.output.Output` (optional)
           output file
        """

        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("creating pixel respone function")
        parameters = self.get_task_param("parameters")

        kernel, kernel_delta = self.model(parameters)

        output_file = self.get_task_param("output")
        if output_file and issubclass(output_file.__class__, output.Output):
            output_group = output_file.create_group(
                "intrapixel_response_function"
            )
            output_group.write_array(
                "kernel",
                kernel,
            )
            output_group.write_quantity(
                "kernel_delta",
                kernel_delta,
            )

        self.set_output([kernel, kernel_delta])

    def model(self, parameters):
        """
        This model creates the intrapixel response function to be used in the convolution with the focal plane array.
        Thsi intrapixel response function is compatible with the Scipy convolution functions.

        Estimate the detector pixel response function with the prescription of
        Barron et al., PASP, 119, 466-475 (2007).

        Parameters
        ----------
        oversampling: int
                number of samples in each resolving element. The
                final shape of the response function would be shape*osf
        delta_pix: :class:`astropy.units.Quantity`
                Phisical size of the detector pixel in microns
        diffusion_length: :class:`astropy.units.Quantity`
                diffusion length in microns
        intra_pix_distance: :class:`astropy.units.Quantity`
                distance between two adjacent detector pixels
                in microns

        Returns
        -------
        kernel	: 2D array
                the kernel image
        kernel_delta  : :class:`astropy.units.Quantity`
                        the kernel sampling interval in microns
        """
        if "oversampling" in parameters["detector"].keys():
            osf = parameters["detector"][
                "oversampling"
            ]  # activate for old convolution
        else:
            osf = 1  # activate for old convolution

        delta = parameters["detector"]["delta_pix"]
        if "diffusion_length" in parameters["detector"].keys():
            lx = parameters["detector"]["diffusion_length"]
        else:
            lx = 0.0 * u.um
        if "intra_pix_distance" in parameters["detector"].keys():
            ipd = parameters["detector"]["intra_pix_distance"]
        else:
            ipd = 0 * u.um
        lx += 1e-20 * u.um  # to avoid problems if user pass lx=0
        lx = lx.to(delta.unit)

        kernel = np.zeros((osf, osf))
        self.debug("kernel size: {}".format(kernel.shape))
        # prepare the kernel stamp grid
        kernel_delta = delta / osf
        scale = np.arange(0, 2 * osf / 2) - osf / 2 + 0.5
        yy = scale * kernel_delta
        xx = scale * kernel_delta

        xx, yy = np.meshgrid(xx, yy)

        # compute kernel stamp
        kernel_stamp = np.arctan(
            np.tanh((0.5 * (0.5 * delta - xx) / lx).value)
        ) - np.arctan(np.tanh((0.5 * (-0.5 * delta - xx) / lx).value))

        kernel_stamp *= np.arctan(
            np.tanh((0.5 * (0.5 * delta - yy) / lx).value)
        ) - np.arctan(np.tanh((0.5 * (-0.5 * delta - yy) / lx).value))

        # deal with border
        i_mask_xx = np.where(np.abs(xx) > 0.5 * (delta - ipd))
        i_mask_yy = np.where(np.abs(yy) > 0.5 * (delta - ipd))
        # set the unused area of kernel stamp to zero
        kernel_stamp[i_mask_yy] = 0.0
        kernel_stamp[i_mask_xx] = 0.0

        # # deal with fractional pixels
        # border = kernel_delta - ipd
        # pix, fract = divmod(border.value,1)

        # if fract != 0:
        #     idx = np.where(kernel_stamp!=0)
        #     kernel_stamp[:, idx[0][0]] *= 1-fract
        #     kernel_stamp[idx[1][0], :] *= 1-fract
        #     kernel_stamp[:, idx[0][-1]] *= 1-fract
        #     kernel_stamp[idx[1][-1], :] *= 1-fract
        # Normalise the kernel such that the pixel has QE=1
        kernel_stamp /= kernel_stamp.sum()
        kernel_stamp *= osf * osf

        return kernel_stamp, kernel_delta

        # Author note: We finally moved to a more general approach using scipy fftconvolve.
        # In thi vision we can simulate a kernel of a single pixel.
        # The problem with previuos approach is that if the kernel was not zeri padded as the PSF the results will have aliasing effects.
        # 2022-12-01 L.V.M

        # Author note: previous version was computing the pixel irf shape on a kernel with the size of the PSF shape.
        # This was numerical intensive and not necessary. The new version is much faster and more accurate.
        # The kernel is now a square with the size of 4 pixels. The kernel is then rolled to use for the fast fft convolve.
        # Previous version can be tracked in the git history.
        # 2022-11-30 L.V.M

        # Author note: previous version was computing the pixel irf shape on the full kernel and then setting to 0 all other pixels.
        # That method resulted in a very slow computation of arctan and huge memory requiring for high osf.
        # This "kernel stamp" solution return the same kernel (with relative differences under 1e-8 - tested on 2022/05/19)
        # but very lower computing time.
        # TEST: using osf 12 psf =(64,64): previous method took 4.2s, new method takes 6ms with)
        # This method removes the memory problem: not much of RAM used for this computation
        # tested on jupyter notebook on 2022/05/19. L.V.M.
