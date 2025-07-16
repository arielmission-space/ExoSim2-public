import numpy as np
from scipy.interpolate import RectBivariateSpline


def fast_convolution(im, delta_im, ker, delta_ker):
    """fast_convolution.
    Convolve an image with a kernel. Image and kernel can be sampled on different
    grids defined.

    Parameters
    __________
    im: :class:`~numpy.ndarray`
        the image to be convolved
    delta_im: float
        image sampling interval
    ker: :class:`~numpy.ndarray`
        the convolution kernel
    delta_ker: float
        Kernel sampling interval
    Returns
    -------
    :class:`~numpy.ndarray`
        the image convolved with the kernel.
    """
    # Fourier transform the kernel
    kerf = np.fft.rfft2(ker)
    ker_k = [
        np.fft.fftfreq(ker.shape[0], d=delta_ker),
        np.fft.rfftfreq(ker.shape[1], d=delta_ker),
    ]
    ker_k[0] = np.fft.fftshift(ker_k[0])
    kerf = np.fft.fftshift(kerf, axes=0)

    # Fourier transform the image
    imf = np.fft.rfft2(im)
    im_k = [
        np.fft.fftfreq(im.shape[0], d=delta_im),
        np.fft.rfftfreq(im.shape[1], d=delta_im),
    ]
    im_k[0] = np.fft.fftshift(im_k[0])
    imf = np.fft.fftshift(imf, axes=0)

    # Interpolate kernel
    kerf_r = RectBivariateSpline(ker_k[0], ker_k[1], kerf.real)
    kerf_i = RectBivariateSpline(ker_k[0], ker_k[1], kerf.imag)
    # Convolve
    imf = imf * (kerf_r(im_k[0], im_k[1]) + 1j * kerf_i(im_k[0], im_k[1]))

    imf = np.fft.ifftshift(imf, axes=0)

    return np.fft.irfft2(imf) * (delta_ker / delta_im) ** 2
