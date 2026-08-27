===================================
Focal plane
===================================

Create focal planes
---------------------

The next step in the :class:`~exosim.models.channel.Channel` pipeline is to create an empty focal plane to populate.
This can be done with the method :func:`~exosim.models.channel.Channel.create_focal_planes`.

.. code-block:: python

        channel.create_focal_planes()

This method calls the :class:`~exosim.tasks.instrument.create_focal_plane.CreateFocalPlane` task.
This task first builds the focal plane array by using :class:`~exosim.tasks.instrument.create_focal_plane_array.CreateFocalPlaneArray`.

.. _detector geometry:

Detector geometry
^^^^^^^^^^^^^^^^^^^

The first step is the detector geometry, which needs to be specified in the channel detector description:

.. code-block:: xml

    <channel> channel_name

        <detector>
            <delta_pix unit="micron"> 18.0 </delta_pix>
            <spatial_pix>64</spatial_pix>
            <spectral_pix>364</spectral_pix>
            <oversampling>3</oversampling>
        </detector>

    </channel>

In this case we are building a detector with 64 pixels in the spatial direction and 364 pixels in the dispersion direction, with pixels 18 micron wide.
The `oversampling` key allows us to use sub-pixels. In this case we split each pixel into 3 in each direction, yielding 9 sub-pixels.

.. note::
The main reason to have an oversampling factor is the jitter effect (see :ref:`Instantaneous readout`).
    The oversampling factor is needed to ensure that the PSF is Nyquist sampled (at least 2 per FWHM) and to correctly represent intra-pixel response.
    The oversampling factor can be any number, but for efficiency reasons it should be a power of an odd value.


Wavelength solution
^^^^^^^^^^^^^^^^^^^^
If the channel is a `spectrometer`, then the wavelength solution is used to find the wavelength collected by each pixel in the dispersion direction.
The solution can be specified as

.. code-block:: xml

    <channel> channel_name
        <wl_solution>
            <wl_solution_task>LoadWavelengthSolution</wl_solution_task>
            <datafile>__ConfigPath__/wl_sol.ecsv</datafile>
            <center>auto</center>
        </wl_solution>
    </channel>

The `wl_sol.ecsv` file is a table with 3 columns: `Wavelength`, `x`, `y`, where `x` is the dispersion direction and `y` is the spatial direction.
If `y` is set to 0 for each wavelength, the source light is assumed to be dispersed only along the dispersion direction; otherwise it is also dispersed in the spatial direction.
The wavelengths associated with each pixel in the spectral and spatial directions are stored along the focal plane in the :class:`~exosim.models.signal.Signal` class in the `spectral` and `spatial` attributes.

The `wl_solution_task` indicates the task to use to load the wavelength solution.
By default, the :class:`~exosim.tasks.instrument.load_wavelength_solution.LoadWavelengthSolution` is used.
This :class:`~exosim.tasks.task.Task` can be customised, as described in :ref:`Custom Tasks`.

The **center** key is used to set the central pixel in the spectral direction.
If "auto" it sets the central wavelength of the channel in the centre of the pixel array.
If a wavelength is indicated, it centres the wavelength solution on that wavelength.
Else, it shifts the pixel array by the indicated number of pixels.

If the channel is a `photometer` there is no need to specify the wavelength solution.
The :class:`~exosim.tasks.instrument.create_focal_plane_array.CreateFocalPlaneArray` task will use the detector responsivity to estimate a wavelength solution to use for the next step (:ref:`rescale contribution`).

Source and foregrounds Focal planes
-----------------------------------------
Once the array is built, the :class:`~exosim.tasks.instrument.create_focal_plane.CreateFocalPlane` task creates a stack of arrays along the temporal direction.

.. image:: _static/signal_class.png
    :align: center

Finally, :func:`~exosim.models.channel.Channel.create_focal_planes` duplicates it to produce a focal plane for the foreground contributions.
This method populates the `focal_plane` and `frg_focal_plane` attributes in the :class:`~exosim.models.channel.Channel` class.


.. _rescale contribution:

Rescale Contributions
-----------------------

Knowing the size of the focal planes and the wavelength solutions, we can rescale the incoming signals to convert them from signal densities (:math:`counts/s/\mu m`) into proper signals (:math:`counts/s/pixel`).

.. code-block:: python

        channel.rescale_contributions()

The :func:`~exosim.models.channel.Channel.rescale_contributions` method updates the `sources` and `path` keys in the :class:`~exosim.models.channel.Channel` class by rebinning the signals according to the focal plane dispersion binning.
Then it estimates the wavelength solution gradient from the pixel wavelength solution and multiplies the signal by this gradient.

Populate focal plane
----------------------

Next it is time to populate the source focal plane. We follow the following scheme:

.. image:: _static/focal_plane_population.png
    :align: center

First we need to produce a monochromatic PSF for each wavelength sampled in the pixel wavelength solution.
Then we multiply the PSF by the source signal at the respective wavelength and add the result to the relevant pixel.
On the now populated focal plane, we then apply the Intra-pixel Response Function (IRF).


The first steps are handled by

.. code-block:: python

        channel.populate_focal_plane()

The :func:`~exosim.models.channel.Channel.populate_focal_plane` method calls the :class:`~exosim.tasks.instrument.populate_focal_plane.PopulateFocalPlane` task.

PSF
^^^^^^^^
The first step mentioned above is the production of the Point Spread Function hypercube.

.. image:: _static/psf_ipercube.png
    :align: center

For each temporal step, the PSF cube is defined as in the following figure:

.. image:: _static/psf_cube.png
    :align: center

The PSF specifics are to be listed in the `psf` section of the `.xml` channel description.
The simplest PSFs are described by the `Airy` or `Gauss` functions.

.. code-block:: xml

    <channel> channel_name
        <psf>
            <shape>Airy</shape>
        </psf>
    </channel>

In this case, the :class:`~exosim.tasks.instrument.populate_focal_plane.PopulateFocalPlane` task calls :func:`~exosim.utils.psf.create_psf`.
This function produces a PSF cube as the one shown before, where the volume of each PSF is normalised to unity:

.. image:: _static/airy_es.png
    :width: 500
    :align: center

The `psf` section can be customised by adding the following keys:

.. code-block:: xml

    <channel> channel_name
        <psf>
            <shape>Airy</shape>
            <nzero> 8 </nzero>
            <size_y> 64 </size_y>
            <size_x> 64 </size_x>
        </psf>
    </channel>

Where `nzero` indicates the number of zeros in the Airy function, `size_x` and `size_y` are the sizes of the PSF cube in the spectral and spatial directions.
`size_x` and `size_y` can also be set to `full` to use the full size of the focal plane.

However, the user may want to load specific PSF shapes.
This can be done by writing a dedicated :class:`~exosim.tasks.instrument.loadPsf.LoadPsf` task.
:class:`~exosim.tasks.instrument.loadPsf.LoadPsf` task produces a hypercube, where each temporal step of the focal plane is associated with a PSF cube as in the previous picture.
The native PSF format supported by `ExoSim` is the PAOS format, and the functionality is provided by :class:`~exosim.tasks.instrument.loadPsfPaos.LoadPsfPaos`.
In this case the user shall specify it in the `.xml` file as

.. code-block:: xml

    <channel> channel_name
        <psf>
            <psf_task>LoadPsfPaos</psf_task>
            <filename>__ConfigPath__/paos_file.h5</filename>
        </psf>
    </channel>

The :class:`~exosim.tasks.instrument.loadPsfPaos.LoadPsfPaos` task loads the PSF cube provided by the `filename` data.
The PSFs are then interpolated over a grid matching the one used to produce the focal planes, to convert them into physical units.
Then the total volume of the interpolated PSF is rescaled to the total volume of the original one.
This allows accounting for losses in transmission due to the optical path.
The PSFs are then interpolated over a wavelength grid matching the one used for the focal plane, producing the cube.
This speeds up the subsequent `ExoSim` steps.
The default :class:`~exosim.tasks.instrument.loadPsfPaos.LoadPsfPaos` task does not include a temporal dependency,
and therefore the PSF cube is repeated along the temporal axis.

.. note::

For long observations with a small "low frequencies variation"
    the memory needed to keep the repeated PAOS PSF could be very high.
    It is possible to store only one PSF, switching
    to False the `time_dependence` parameter in the `psf` section,
    e.g.:

    .. code-block:: xml

        <channel> channel_name
            <psf>
                <psf_task>LoadPsfPaos</psf_task>
                <filename>__ConfigPath__/paos_file.h5</filename>
                 <time_dependence>False</time_dependence>
            </psf>
        </channel>

The user can define a temporal dependence by using a custom :class:`~exosim.tasks.instrument.loadPsf.LoadPsf` task.
An example using PAOS PSFs is reported in :class:`~exosim.tasks.instrument.loadPsfPaosTimeInterp.LoadPsfPaosTimeInterp`.

Finally, the obtained PSFs are stored in the output file.

Adding PSF to the focal plane
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once the PSF cube is ready, for each temporal step of the focal plane, we add a monochromatic PSF to the relevant pixel, multiplying it by the relative intensity of the source signal at the same temporal step.
This allows us to produce a dispersed image in the case of a `spectrometer` or to accumulate the PSF in the case of a `photometer`.
Also, if the source signal has a time-dependent variation, this is propagated to the image on the focal plane thanks to the use of the same temporal step in both the focal plane and the source signal.
The result will be an oversampled focal plane.

Intra-pixel Response Function
--------------------------------

The pixels on the focal plane do not have a uniform responsivity to the incoming light on their surfaces.
They are known to be more responsive at the centre and less at the edges.
This effect can be represented in `ExoSim` by introducing the IRF.

This is handled by the :func:`~exosim.models.channel.Channel.apply_irf` method:

.. code-block:: python

        channel.apply_irf()

Create IRF
^^^^^^^^^^^^^
The task to use to estimate the IRF is indicated as

.. code-block:: xml

    <channel> channel_name
        <detector>
            <irf_task>CreateIntrapixelResponseFunction</irf_task>
        </detector>
    </channel>

where :class:`~exosim.tasks.instrument.create_intrapixel_response_function.CreateIntrapixelResponseFunction` is the default class.
This task implements the equation presented in Barron et al., PASP, 119, 466–475, 2007 (https://doi.org/10.1086/517620).
It requires the pixel `diffusion length` and the `intra-pixel distance`:

.. code-block:: xml

    <channel> channel_name
        <detector>
            <irf_task>CreateIntrapixelResponseFunction</irf_task>
            <diffusion_length unit="micron">1.7</diffusion_length>
            <intra_pix_distance unit="micron">0.0</intra_pix_distance>
        </detector>
    </channel>

Two other default tasks are available to create the IRF:
:class:`~exosim.tasks.instrument.create_oversampled_intrapixel_response_function.CreateOversampledIntrapixelResponseFunction`.
The first one is a simple oversampling of the IRF, while the second one is an oversampling of the IRF with a larger size.

The user can, however, specify their own tasks and the relative parameters.
Notice that the IRF volume is expected to be normalised to unity.
Here is an example of a resulting IRF:

.. image:: _static/pixel_response_es.png
    :width: 500
    :align: center

.. caution::
    If no `irf_task` key is provided in the channel description,
    the :func:`~exosim.models.channel.Channel.apply_irf` method
    automatically uses the default :class:`~exosim.tasks.instrument.create_intrapixel_response_function.CreateIntrapixelResponseFunction` task.

IRF application
^^^^^^^^^^^^^^^^^

When the pixel response function is produced, we apply it using the :class:`~exosim.tasks.instrument.apply_intra_pixel_response_function.ApplyIntraPixelResponseFunction`.
This task performs a convolution between the focal plane and the IRF.

Now the source focal plane is completed.

.. note::

    In the default recipe (:ref:`focal plane recipe`), if no `irf_task` key is provided in the channel description, the IRF step is skipped.

The user can specify the convolution method to use:

.. code-block:: xml

    <channel> channel_name
        <detector>
            <convolution_method>fftconvolve</convolution_method>
        </detector>
    </channel>

The available methods are `fftconvolve` (:func:`scipy.signal.fftconvolve`), `convolve` (:func:`scipy.signal.convolve`), `ndimage.convolve` (:func:`scipy.ndimage.convolve`) and `fast_convolution` (:func:`exosim.utils.convolution.fast_convolution`).
If no convolution_method is specified, the default is `fftconvolve`.

.. note::

The `fast_convolution` method is the same as implemented in `Sarkar et al., 2021 <https://link.springer.com/article/10.1007/s10686-020-09690-9>`__.
    It is very accurate but slower than the other methods and requires a lot of memory.
    It is therefore recommended to use it only for small oversampling factors.

The :class:`~exosim.tasks.instrument.create_intrapixel_response_function.CreateIntrapixelResponseFunction` task creates a kernel compatible with both `fftconvolve` (:func:`scipy.signal.fftconvolve`), `convolve` (:func:`scipy.signal.convolve`) and `ndimage.convolve` (:func:`scipy.ndimage.convolve`).
The task :class:`~exosim.tasks.instrument.create_oversampled_intrapixel_response_function.CreateOversampledIntrapixelResponseFunction` is instead compatible with `fast_convolution` (:func:`exosim.utils.convolution.fast_convolution`), which is a method developed specifically for ExoSim.

Populate foreground focal plane
--------------------------------

To populate the foreground focal plane, we can call the :func:`~exosim.models.channel.Channel.populate_foreground_focal_plane` method:

.. code-block:: python

        channel.populate_foreground_focal_plane()

This involves the :class:`~exosim.tasks.instrument.foregrounds_to_focal_plane.ForegroundsToFocalPlane` task,
which simply adds the foreground contributions, stored in the `path` attribute, to the foreground focal plane, stored in the `frg_focal_plane` attribute.

If the `path` element to add is before a slit, the signal is dispersed.
Therefore the contribution signal is convolved with a kernel of the width of the slit expressed as a number of pixels, and then summed to the full array.
If the slit width expressed as a number of pixels at the focal plane is :math:`L`, and the spectral resolving power computed at a certain :math:`\lambda_0` is :math:`R(\lambda_0)`,
the detector receives diffuse radiation over the wavelength range :math:`\left( \lambda_j - \frac{L \lambda_0}{4 R(\lambda_0)} \, , \, \lambda_j  + \frac{L \lambda_0}{4 R(\lambda_0)} \right)`,
and not over the full range of wavelengths accepted by the filter. So, the :math:`j`-th pixel sampling the :math:`\lambda_j` wavelength the collected signal is

.. math::
    S(j) = \int_{\lambda_j - \frac{L \lambda_0}{4 R(\lambda_0)}}^{\lambda_j  + \frac{L \lambda_0}{4 R(\lambda_0)}} S_{for} (\lambda) d \lambda


If the `path` element to add is after a slit, or if no slit is in the path, the signal integrated over the full wavelength range is simply added to each pixel:

.. math::
    S = \int S_{for} (\lambda) d \lambda

Now the foreground focal plane is completed.

.. _sub focal planes:

Foreground sub focal planes
^^^^^^^^^^^^^^^^^^^^^^^^^^^
If at least one optical element has

.. code-block:: xml

    <optical_path>
        <opticalElement>
            ...
            <isolate> True <isolate>
        </opticalElement>
    </optical_path>

Then the sub-focal planes are computed. The same :func:`~exosim.models.channel.Channel.populate_foreground_focal_plane` method also populates a `frg_sub_focal_planes` attribute.
This is a dictionary containing all the foreground signal contributions, highlighting the ones marked with ``isolate=True``.
The sum of all the sub-focal planes matches `frg_focal_plane`.

This mode allows the user to investigate the effects of a single optical surface.
