===========
Focal plane
===========

Create the focal planes
-----------------------

The next step in the :class:`~exosim.models.channel.Channel` pipeline is to
create an empty focal plane to populate, with
:func:`~exosim.models.channel.Channel.create_focal_planes`:

.. code-block:: python

        channel.create_focal_planes()

This calls the
:class:`~exosim.tasks.instrument.create_focal_plane.CreateFocalPlane` task,
which first builds the focal-plane array with
:class:`~exosim.tasks.instrument.create_focal_plane_array.CreateFocalPlaneArray`.

.. _detector geometry:

Detector geometry
^^^^^^^^^^^^^^^^^

The first input is the detector geometry, given in the channel detector
description:

.. code-block:: xml

    <channel> channel_name

        <detector>
            <delta_pix unit="micron"> 18.0 </delta_pix>
            <spatial_pix>64</spatial_pix>
            <spectral_pix>364</spectral_pix>
            <oversampling>3</oversampling>
        </detector>

    </channel>

This builds a detector with 64 pixels in the spatial direction and 364 pixels in
the dispersion direction, each 18 micron wide. The `oversampling` key adds
sub-pixels: here each pixel is split into 3 in each direction, giving 9
sub-pixels.

.. note::
    The main reason for an oversampling factor is the jitter effect (see
    :ref:`Instantaneous readout`). Oversampling ensures the PSF is Nyquist
    sampled (at least 2 samples per FWHM) and lets the intra-pixel response be
    represented correctly. The factor can be any number, but for efficiency it
    should be a power of an odd value.


Wavelength solution
^^^^^^^^^^^^^^^^^^^

For a `spectrometer` channel, the wavelength solution gives the wavelength
collected by each pixel in the dispersion direction. It is specified as:

.. code-block:: xml

    <channel> channel_name
        <wl_solution>
            <wl_solution_task>LoadWavelengthSolution</wl_solution_task>
            <datafile>__ConfigPath__/wl_sol.ecsv</datafile>
            <center>auto</center>
        </wl_solution>
    </channel>

The `wl_sol.ecsv` file is a table with three columns, `Wavelength`, `x` and `y`,
where `x` is the dispersion direction and `y` is the spatial direction. If `y`
is 0 at every wavelength, the source light is dispersed only along the
dispersion direction; otherwise it is also dispersed spatially. The wavelengths
assigned to each pixel are stored on the focal plane, in the `spectral` and
`spatial` attributes of the :class:`~exosim.models.signal.Signal`.

`wl_solution_task` names the task that loads the wavelength solution; the default
is
:class:`~exosim.tasks.instrument.load_wavelength_solution.LoadWavelengthSolution`,
which can be customised (see :ref:`Custom Tasks`).

The **center** key sets the central pixel in the spectral direction:

- `auto` puts the central wavelength of the channel at the centre of the pixel
  array;
- a wavelength value centres the wavelength solution on that wavelength;
- an integer shifts the pixel array by that many pixels.

A `photometer` channel needs no wavelength solution. The
:class:`~exosim.tasks.instrument.create_focal_plane_array.CreateFocalPlaneArray`
task uses the detector responsivity to derive a wavelength solution for the next
step (:ref:`rescale contribution`).

Source and foreground focal planes
----------------------------------

Once the array is built, the
:class:`~exosim.tasks.instrument.create_focal_plane.CreateFocalPlane` task
stacks copies of it along the temporal axis.

.. image:: _static/signal_class.png
    :align: center

Finally, :func:`~exosim.models.channel.Channel.create_focal_planes` duplicates
the stack to create a focal plane for the foreground contributions. This
populates the `focal_plane` and `frg_focal_plane` attributes of
:class:`~exosim.models.channel.Channel`.


.. _rescale contribution:

Rescale contributions
---------------------

With the focal-plane size and the wavelength solutions known, the incoming
signals can be rescaled from signal densities (:math:`counts/s/\mu m`) to proper
signals (:math:`counts/s/pixel`):

.. code-block:: python

        channel.rescale_contributions()

:func:`~exosim.models.channel.Channel.rescale_contributions` updates the
`sources` and `path` keys of :class:`~exosim.models.channel.Channel` by rebinning
the signals onto the focal-plane dispersion binning. It then estimates the
wavelength-solution gradient from the pixel wavelength solution and multiplies
the signal by it.

Populate the focal plane
------------------------

Next, populate the source focal plane, following this scheme:

.. image:: _static/focal_plane_population.png
    :align: center

First, produce a monochromatic PSF for each wavelength sampled in the pixel
wavelength solution. Then multiply each PSF by the source signal at its
wavelength and add the result to the relevant pixel. Finally, apply the
intra-pixel response function (IRF) to the populated focal plane.

The first steps are handled by:

.. code-block:: python

        channel.populate_focal_plane()

:func:`~exosim.models.channel.Channel.populate_focal_plane` calls the
:class:`~exosim.tasks.instrument.populate_focal_plane.PopulateFocalPlane` task.

PSF
^^^

The first step is building the point-spread-function hypercube.

.. image:: _static/psf_ipercube.png
    :align: center

For each temporal step, the PSF cube is defined as in the figure below:

.. image:: _static/psf_cube.png
    :align: center

The PSF is specified in the `psf` section of the `.xml` channel description. The
simplest PSFs are the `Airy` and `Gauss` functions:

.. code-block:: xml

    <channel> channel_name
        <psf>
            <shape>Airy</shape>
        </psf>
    </channel>

In this case the
:class:`~exosim.tasks.instrument.populate_focal_plane.PopulateFocalPlane` task
calls :func:`~exosim.utils.psf.create_psf`, which produces a PSF cube like the
one above, with each PSF normalised to unit volume:

.. image:: _static/airy_es.png
    :width: 500
    :align: center

The `psf` section accepts these extra keys:

.. code-block:: xml

    <channel> channel_name
        <psf>
            <shape>Airy</shape>
            <nzero> 8 </nzero>
            <size_y> 64 </size_y>
            <size_x> 64 </size_x>
        </psf>
    </channel>

`nzero` is the number of zeros of the Airy function; `size_x` and `size_y` are
the sizes of the PSF cube in the spectral and spatial directions. `size_x` and
`size_y` can also be set to `full` to use the full size of the focal plane.

You may instead want to load specific PSF shapes. Do this with a dedicated
:class:`~exosim.tasks.instrument.loadPsf.LoadPsf` task, which produces a
hypercube where each temporal step of the focal plane has its own PSF cube. The
native PSF format supported by `ExoSim` is the PAOS format, handled by
:class:`~exosim.tasks.instrument.loadPsfPaos.LoadPsfPaos`. Specify it as:

.. code-block:: xml

    <channel> channel_name
        <psf>
            <psf_task>LoadPsfPaos</psf_task>
            <filename>__ConfigPath__/paos_file.h5</filename>
        </psf>
    </channel>

:class:`~exosim.tasks.instrument.loadPsfPaos.LoadPsfPaos` loads the PSF cube from
`filename`. The PSFs are interpolated onto a grid matching the one used to build
the focal planes, to convert them to physical units, and their total volume is
rescaled to that of the originals, which accounts for transmission losses in the
optical path. They are also interpolated onto a wavelength grid matching the
focal plane, producing the cube; this speeds up the later `ExoSim` steps. The
default :class:`~exosim.tasks.instrument.loadPsfPaos.LoadPsfPaos` task has no
time dependence, so the PSF cube is repeated along the temporal axis.

.. note::
    For a long observation with a small low-frequency variation, keeping the
    repeated PAOS PSF in memory can be expensive. You can store a single PSF by
    setting `time_dependence` to False in the `psf` section:

    .. code-block:: xml

        <channel> channel_name
            <psf>
                <psf_task>LoadPsfPaos</psf_task>
                <filename>__ConfigPath__/paos_file.h5</filename>
                 <time_dependence>False</time_dependence>
            </psf>
        </channel>

To add a time dependence, use a custom
:class:`~exosim.tasks.instrument.loadPsf.LoadPsf` task. An example with PAOS PSFs
is
:class:`~exosim.tasks.instrument.loadPsfPaosTimeInterp.LoadPsfPaosTimeInterp`.

Finally, the PSFs are stored in the output file.

Adding the PSF to the focal plane
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the PSF cube is ready, for each temporal step of the focal plane a
monochromatic PSF is added to the relevant pixel, multiplied by the intensity of
the source signal at the same temporal step. This produces a dispersed image for
a `spectrometer` or an accumulated PSF for a `photometer`. Because the focal
plane and the source signal share the same temporal steps, any time variation in
the source signal is carried through to the focal-plane image. The result is an
oversampled focal plane.

Intra-pixel response function
-----------------------------

A pixel does not respond uniformly across its surface: it is more responsive at
the centre than at the edges. `ExoSim` represents this with the intra-pixel
response function (IRF), applied by
:func:`~exosim.models.channel.Channel.apply_irf`:

.. code-block:: python

        channel.apply_irf()

Create the IRF
^^^^^^^^^^^^^^

The task that estimates the IRF is named as:

.. code-block:: xml

    <channel> channel_name
        <detector>
            <irf_task>CreateIntrapixelResponseFunction</irf_task>
        </detector>
    </channel>

The default class is
:class:`~exosim.tasks.instrument.create_intrapixel_response_function.CreateIntrapixelResponseFunction`.
It implements the equation of Barron et al., PASP, 119, 466–475, 2007
(https://doi.org/10.1086/517620), and needs the pixel `diffusion length` and the
`intra-pixel distance`:

.. code-block:: xml

    <channel> channel_name
        <detector>
            <irf_task>CreateIntrapixelResponseFunction</irf_task>
            <diffusion_length unit="micron">1.7</diffusion_length>
            <intra_pix_distance unit="micron">0.0</intra_pix_distance>
        </detector>
    </channel>

One alternative default task is available,
:class:`~exosim.tasks.instrument.create_oversampled_intrapixel_response_function.CreateOversampledIntrapixelResponseFunction`.
It produces an oversampled version of the IRF, zero-padded to the size of the
PSF, for use with the `fast_convolution` method.

You can also supply your own task and parameters. Note that the IRF is expected
to have unit volume. Here is an example IRF:

.. image:: _static/pixel_response_es.png
    :width: 500
    :align: center

.. caution::
    If no `irf_task` key is given in the channel description,
    :func:`~exosim.models.channel.Channel.apply_irf` uses the default
    :class:`~exosim.tasks.instrument.create_intrapixel_response_function.CreateIntrapixelResponseFunction`
    task.

Apply the IRF
^^^^^^^^^^^^^

Once the pixel response function is ready, it is applied with
:class:`~exosim.tasks.instrument.apply_intra_pixel_response_function.ApplyIntraPixelResponseFunction`,
which convolves the focal plane with the IRF.

The source focal plane is now complete.

.. note::

    In the default recipe (:ref:`focal plane recipe`), if no `irf_task` key is
    given in the channel description, the IRF step is skipped.

You can choose the convolution method:

.. code-block:: xml

    <channel> channel_name
        <detector>
            <convolution_method>fftconvolve</convolution_method>
        </detector>
    </channel>

The available methods are `fftconvolve` (:func:`scipy.signal.fftconvolve`),
`convolve` (:func:`scipy.signal.convolve`), `ndimage.convolve`
(:func:`scipy.ndimage.convolve`) and `fast_convolution`
(:func:`exosim.utils.convolution.fast_convolution`). The default is
`fftconvolve`.

.. note::
    The `fast_convolution` method is the one implemented in `Sarkar et al., 2021
    <https://link.springer.com/article/10.1007/s10686-020-09690-9>`__. It is
    very accurate but slower than the others and memory-hungry, so use it only
    for small oversampling factors.

The
:class:`~exosim.tasks.instrument.create_intrapixel_response_function.CreateIntrapixelResponseFunction`
task creates a kernel compatible with `fftconvolve`, `convolve` and
`ndimage.convolve`. The
:class:`~exosim.tasks.instrument.create_oversampled_intrapixel_response_function.CreateOversampledIntrapixelResponseFunction`
task instead produces a kernel for `fast_convolution`, a method developed
specifically for ExoSim.

Populate the foreground focal plane
-----------------------------------

Populate the foreground focal plane with
:func:`~exosim.models.channel.Channel.populate_foreground_focal_plane`:

.. code-block:: python

        channel.populate_foreground_focal_plane()

This uses the
:class:`~exosim.tasks.instrument.foregrounds_to_focal_plane.ForegroundsToFocalPlane`
task, which adds the foreground contributions stored in the `path` attribute to
the foreground focal plane stored in `frg_focal_plane`.

If a `path` element sits before a slit, its signal is dispersed: the
contribution is convolved with a kernel as wide as the slit (in pixels) and then
added to the full array. If the slit width at the focal plane is :math:`L`
pixels and the spectral resolving power at some :math:`\lambda_0` is
:math:`R(\lambda_0)`, the detector receives diffuse radiation over the
wavelength range
:math:`\left( \lambda_j - \frac{L \lambda_0}{4 R(\lambda_0)} \, , \, \lambda_j  + \frac{L \lambda_0}{4 R(\lambda_0)} \right)`,
not over the full range passed by the filter. So the :math:`j`-th pixel,
sampling wavelength :math:`\lambda_j`, collects

.. math::
    S(j) = \int_{\lambda_j - \frac{L \lambda_0}{4 R(\lambda_0)}}^{\lambda_j  + \frac{L \lambda_0}{4 R(\lambda_0)}} S_{for} (\lambda) d \lambda


If a `path` element sits after a slit, or there is no slit in the path, the
signal integrated over the full wavelength range is simply added to each pixel:

.. math::
    S = \int S_{for} (\lambda) d \lambda

The foreground focal plane is now complete.

.. _sub focal planes:

Foreground sub focal planes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If at least one optical element is marked with

.. code-block:: xml

    <optical_path>
        <opticalElement>
            ...
            <isolate> True </isolate>
        </opticalElement>
    </optical_path>

then the sub-focal planes are computed. The same
:func:`~exosim.models.channel.Channel.populate_foreground_focal_plane` method
also populates a `frg_sub_focal_planes` attribute: a dictionary of all the
foreground signal contributions, singling out the ones marked
``isolate=True``. The sum of all the sub-focal planes equals `frg_focal_plane`.

This mode lets you study the effect of a single optical surface.
