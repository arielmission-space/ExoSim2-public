import numpy as np
from joblib import delayed
from joblib import Parallel
from tqdm.auto import tqdm

import exosim.output as output
from exosim.tasks.task import Task
from exosim.utils import RunConfig


class ApplyIntraPixelResponseFunction(Task):
    """
    It applies the intra pixel response function to the plane

    Returns
    -------
    :class:`~exosim.models.signal.Signal`
        focal plane array (irf applied)
    """

    def __init__(self):
        """
        Parameters
        __________
        focal_plane: :class:`~exosim.models.signal.Signal`
            focal plane array
        irf_kernel	: 2D array
            the irf kernel image
        irf_kernel_delta  : scalar
            the irf kernel sampling interval in microns
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        convolution_method: str (optional)
            convolution method to use. Supported methods are:
            `fftconvolve` (:func:`scipy.signal.fftconvolve`),
            `convolve` (:func:`scipy.signal.convolve`),
            `ndimage.convolve` (:func:`scipy.ndimage.convolve`),
            `fast_convolution` (:func:`exosim.utils.convolution.fast_convolution`)
        """

        self.add_task_param("focal_plane", "focal plane")
        self.add_task_param("irf_kernel", " the irf kernel image")
        self.add_task_param(
            "irf_kernel_delta", "the irf kernel sampling interval in microns"
        )
        self.add_task_param("output", "output file", None)
        self.add_task_param(
            "convolution_method", "convolution method", "fftconvolve"
        )

    def execute(self):
        focal_plane = self.get_task_param("focal_plane")
        irf_kernel = self.get_task_param("irf_kernel")
        irf_kernel_delta = self.get_task_param("irf_kernel_delta")
        convolution_method = self.get_task_param("convolution_method")

        self.info(
            "applying pixel response function to {}".format(
                focal_plane.dataset_name
            )
        )

        output_file = self.get_task_param("output")
        if output_file and issubclass(output_file.__class__, output.Output):
            output_group = output_file.create_group("irf")
            output_group.write_array("irf_kernel", irf_kernel)
            output_group.write_array("irf_kernel_delta", irf_kernel_delta)

        img_delta = focal_plane.metadata["focal_plane_delta"]
        convolution_func, kwargs = self.select_convoltion_func(
            convolution_method, irf_kernel, irf_kernel_delta, img_delta
        )

        Parallel(n_jobs=RunConfig.n_job, require="sharedmem")(
            delayed(self._apply_to_time_slice)(
                focal_plane, t, convolution_func, kwargs
            )
            for t in tqdm(
                range(focal_plane.time.size),
                total=focal_plane.time.size,
                desc="applying {}".format(convolution_method),
            )
        )

        self.debug("looking for negative values")
        focal_plane.data = np.where(
            focal_plane.data >= 0.0, focal_plane.data, 1e-30
        )  # remove negative values

        self.set_output(focal_plane)

    def select_convoltion_func(self, method, irf, irf_delta, img_delta):
        func, kwargs = None, None

        if method == "fftconvolve":
            from scipy.signal import fftconvolve

            kwargs = {"in2": irf, "mode": "same"}
            func = fftconvolve
        elif method == "convolve":
            from scipy.signal import convolve

            kwargs = {"in2": irf, "mode": "same"}
            func = convolve
        elif method == "ndimage.convolve":
            # TODO: test this
            # TODO document this
            from scipy.ndimage import convolve

            kwargs = {"weights": irf, "mode": "wrap"}
            func = convolve
        elif method == "fast_convolution":
            from exosim.utils.convolution import fast_convolution

            kwargs = {
                "delta_im": img_delta,
                "ker": irf,
                "delta_ker": irf_delta,
            }
            func = fast_convolution
        else:
            raise ValueError(
                "{} convolution method not supported".format(method)
            )
        self.debug(
            "convolution method: {}. Selected function is {}".format(
                method, func
            )
        )
        self.debug("keywords: {}".format(kwargs))
        return func, kwargs

    def _apply_to_time_slice(self, focal_plane, t, convolution_func, kwargs):
        focal_plane.data[t] = convolution_func(focal_plane.data[t], **kwargs)
