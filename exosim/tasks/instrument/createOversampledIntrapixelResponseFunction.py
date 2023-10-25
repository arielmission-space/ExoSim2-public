import astropy.units as u
import numpy as np

from exosim.tasks.instrument.createIntrapixelResponseFunction import (
    CreateIntrapixelResponseFunction,
)


class CreateOversampledIntrapixelResponseFunction(
    CreateIntrapixelResponseFunction
):
    def model(self, parameters):
        """
        This class produces an oversampled version of the intrapixel response function.
        This kernel is zero-padded to the size of the PSF.
        This version is compatible with :func:`exosim.utils.convolution.fast_convolution` convolution function.


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
            osf = (
                8 * parameters["detector"]["oversampling"]
            )  # activate for old convolution
        else:
            osf = 7  # activate for old convolution

        # TODO find a better zeroing method. It should be enought if the zeros are twice the size of the kernel.
        psf_shape = parameters["psf_shape"]

        delta = parameters["detector"]["delta_pix"]
        if "diffusion_length" in parameters["detector"].keys():
            lx = parameters["detector"]["diffusion_length"]
        else:
            lx = 0.0 * u.um
        if "intra_pix_distance" in parameters["detector"].keys():
            ipd = parameters["detector"]["intra_pix_distance"]
        else:
            ipd = 0 * u.um

        if type(osf) != int:
            osf = int(osf)

        lx += 1e-20 * u.um  # to avoid problems if user pass lx=0
        lx = lx.to(delta.unit)

        kernel = np.zeros((psf_shape[0] * osf, psf_shape[1] * osf))
        self.debug("kernel size: {}".format(kernel.shape))

        # prepare the kernel stamp grid
        kernel_delta = delta / osf
        yc, xc = np.array(kernel.shape) // 2
        yy = (np.arange(-1 * osf, 1 * osf)) * kernel_delta
        xx = (np.arange(-1 * osf, 1 * osf)) * kernel_delta

        # inverse mask to select every other pixel but the central one
        i_mask_xx = np.where(np.abs(xx) > 0.5 * (delta - ipd))
        i_mask_yy = np.where(np.abs(yy) > 0.5 * (delta - ipd))

        xx, yy = np.meshgrid(xx, yy)

        # compute kernel stamp
        kernel_stamp = np.arctan(
            np.tanh((0.5 * (0.5 * delta - xx) / lx).value)
        ) - np.arctan(np.tanh((0.5 * (-0.5 * delta - xx) / lx).value))

        kernel_stamp *= np.arctan(
            np.tanh((0.5 * (0.5 * delta - yy) / lx).value)
        ) - np.arctan(np.tanh((0.5 * (-0.5 * delta - yy) / lx).value))

        # set the unused area of kernel stamp to zero
        kernel_stamp[i_mask_yy, ...] = 0.0
        kernel_stamp[..., i_mask_xx] = 0.0

        # Normalise the kernel such that the pixel has QE=1
        kernel_stamp *= osf * osf / kernel_stamp.sum()
        # put back the kernel stamp
        x_off = kernel.shape[1] // 2 - kernel_stamp.shape[1] // 2
        y_off = kernel.shape[0] // 2 - kernel_stamp.shape[0] // 2

        kernel[
            y_off : y_off + kernel_stamp.shape[0],
            x_off : x_off + kernel_stamp.shape[1],
        ] = kernel_stamp

        # roll to use for the fast fft convolve
        kernel = np.roll(kernel, -xc, axis=1)
        kernel = np.roll(kernel, -yc, axis=0)

        return kernel, kernel_delta

        # Author note: previous version was computing the pixel irf shape on the full kernel and then setting to 0 all other pixels.
        # That method resulted in a very slow computation of arctan and huge memory requiring for high osf.
        # This "kernel stamp" solution return the same kernel (with relative differences under 1e-8 - tested on 2022/05/19)
        # but very lower computing time.
        # TEST: using osf 12 psf =(64,64): previous method took 4.2s, new method takes 6ms with)
        # This method removes the memory problem: not much of RAM used for this computation
        # tested on jupyter notebook on 2022/05/19. L.V.M.
