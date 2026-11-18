===========
Changelog
===========

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (keepachangelog_), and this project adheres
to Semantic Versioning (semver_).

Unreleased changes live as fragments in ``changelog.d/`` and are merged here by
``nox -s release``. See ``docs/source/contributing/releasing.rst``.

.. scriv-insert-here

.. _changelog-2.2.5:

[2.2.5_] - Focal-plane fixes
============================

Added
-----

- ``Channel.apply_irf_to(focal_plane)`` applies the intra pixel response
  function to an arbitrary focal plane; ``Channel.apply_irf()`` uses it for the
  focal plane and the background focal plane.

Fixed
-----

- Radiometric target-list mode no longer forces an oversampling of 1 by
  default. With ``osf = 1`` the PSF is point-sampled at the pixel centres and
  not integrated over the pixel, so the brightest pixel of an undersampled PSF
  was overestimated (about 30% for an FGS-like Airy PSF). The payload
  oversampling is now kept and the intra pixel response function is applied, as
  in the full simulator.
- ``locate_wavelength_windows`` placed the PSF centre on an edge sub-pixel when
  ``osf > 1`` (photometer axes and spectrometer spatial axis without spatial
  wavelength solution), so the source sat ``(osf // 2) / osf`` pixel off the
  centre of the physical pixel read by the decimation ``[osf // 2 :: osf]``.
  The PSF is now centred on the centre sub-pixel of the same physical pixel
  used with ``osf = 1``, independently of the oversampling. Results for
  ``osf = 1`` are unchanged.
- The intra pixel response function kernel
  (``CreateIntrapixelResponseFunction``, and
  ``CreateOversampledIntrapixelResponseFunction``) was sampled only inside the
  pixel and then renormalised to ``osf**2``. With a non-zero diffusion length
  the Barron et al. (2007) response has tails outside the pixel (about 11% of
  the 1D response for 18 um pixels and 1.7 um diffusion length): the charge
  diffusing into the neighbours was put back into the pixel, so the signal of a
  centred star was overestimated (about +3% for ``F*lambda/d = 2.5`` and
  +19-20% for ``F*lambda/d = 0.73``, not converging with the oversampling). The
  kernel now covers the pixel and as many neighbours as needed (response below
  1e-6 of the peak, ``K = 0`` without diffusion), the inter-pixel gap is zeroed
  around every pixel boundary, and it is still normalised to sum ``osf**2``
  (flat field flux is conserved). The central pixel now agrees with a direct
  integration of the PSF times the pixel response within 1% at ``osf = 7``.
  ``ApplyAstronomicalSignal`` decimates the model by the oversampling factor
  rather than by the kernel size. **This changes the output of the full
  focal-plane simulator (every focal plane, hence every simulated image and
  light curve of a channel using the IRF with diffusion), not only the
  radiometric model.** Results with ``diffusion_length = 0`` are unchanged.
- ``LoadPsfPaos.load_imac`` (and ``LoadPsfPaosTimeInterp``) aligned the PAOS PSF
  centre with the sample ``size * osf // 2`` of an array of ``size * osf - 1``
  samples, while the focal plane population and
  ``locate_wavelength_windows`` take ``shape // 2`` as the PSF centre. With an
  even ``size * osf`` the PSF was therefore placed one sample off on both axes:
  one full pixel with ``osf = 1`` and ``1/osf`` pixel otherwise (the AIRS-CH0
  spectrum of the Ariel payload was shifted by one column at ``osf = 1``).
  The PSF is now aligned at the index actually used as its centre. **This
  changes the output of the full focal-plane simulator (position of every PAOS
  PSF on the focal plane), not only the radiometric model.**
- The wavelength (and spatial) axes of the focal plane were anchored on the
  first sub-pixel (``x_j = j * delta / osf``), while physical pixel ``m`` is
  read on sub-pixel ``osf * m + osf // 2``: the spectrum was shifted by
  ``(osf // 2) / osf`` pixel with respect to ``osf = 1`` and the wavelength of
  a physical pixel depended on the oversampling. The axes are now
  ``x_j = (j - osf // 2) * delta / osf``, so the wavelength read at physical
  pixel ``m`` is the same for every ``osf``. Results for ``osf = 1`` (and for
  ``center: auto`` with odd ``osf``) are unchanged. **This changes the
  wavelength axis of the full focal-plane simulator for ``osf > 1``.**
- ``InstantaneousReadOut`` (no-jitter branch) and ``AddForegrounds``
  decimated the oversampled focal plane with ``[::osf]``, while every other
  reader uses ``[osf // 2 :: osf]``. The sub-exposures were therefore read
  ``(osf // 2) / osf`` pixel off. Both now use ``[osf // 2 :: osf]``. The jitter
  branch already sampled the plane centred on ``osf // 2`` and is unchanged.
- ``InstantaneousReadOut`` converted the pointing jitter to sub-pixels with
  ``astype(int)`` (truncation toward zero), which shrank the jitter amplitude
  (rms of the brightest-pixel series 7-12% too low). The offsets are now
  rounded to the nearest sub-pixel (``floor(x + 0.5)``, symmetric in sign, no
  bias). **This changes the jittered sub-exposures.**
- The wavelength widths used to turn the spectral densities of the sources
  and of the path radiances into signals per focal-plane column
  (``Channel.rescale_sources`` / ``rescale_radiances``) gave a full step to the
  two end samples of a photometer axis, although the axis runs from the first
  to the last wavelength of the band, the two ends included. The band was
  widened by one step, hence the photometer signal was too high by about
  ``3.2% / osf`` (+2.1% at ``osf = 1`` and +0.3% at ``osf = 7`` for the example
  photometer, against the independent integral of the spectrum over the band).
  The end samples now weigh half a step (trapezoid rule): the signal agrees
  with the integral within 0.04% at any oversampling. Spectrometers are
  unchanged. **This lowers the photometer signal and background by a few
  per cent at small oversampling.**

- **The absolute signals of the focal-plane simulator change** (every focal
  plane, hence every simulated image and light curve, and the radiometric
  model built on it): PSFs truncated to a finite window were normalised to
  unit sum *inside* the window, so the energy falling outside it was put back
  into the core.

  - ``create_psf`` (analytic Airy and Gauss PSFs) now sets the sum of each PSF
    to the fraction of the *total* analytic energy that falls inside the
    window (the shape is the sampled profile, rescaled). The fraction is
    computed analytically and does not use the samples, so it is independent
    of the sampling (also for undersampled PSFs, ``F wl < delta``, where the
    sum of the point samples differs from the integral): exact
    ``erf(wx/(sqrt(2) sx)) erf(wy/(sqrt(2) sy))`` for the Gaussian, and for the
    Airy pattern a lazily built table of the energy inside a rectangle,
    interpolated in ``log(u), log(v)`` (``u, v`` the window half widths in
    ``F wl``). With the default ``nzero = 4`` the Airy window holds 91-97% of
    the energy, hence every pixel and aperture received 3-9% too much light
    (3.5-5.5% in fixed apertures of the Ariel channels). The fraction of
    energy outside the window is logged per channel (INFO), with a WARNING
    above 1% suggesting a larger ``nzero``.
  - ``LoadPsfPaos.load_imac`` (and ``LoadPsfPaosTimeInterp``) rebin the PAOS
    PSF with an exact flux-conserving separable overlap of the PAOS cells with
    the sub-pixels (each PAOS sample is the integral over its own cell), so
    the window holds its true share of the PAOS volume at any oversampling,
    instead of the volume of the whole PAOS array. Each sub-pixel holds the
    flux in its cell (the IRF box sum then gives exact pixel integrals). The
    subtraction of the minimum of the window is removed, the PSF is zero
    outside the PAOS array, and ``LoadPsfPaosTimeInterp`` does not
    renormalise the cube after cropping it. For the Ariel PAOS files the
    window holds 99.0-100% of the PAOS energy at ``osf = 1`` and ``3``.

- PAOS PSFs outside the wavelengths of the PAOS file. A channel can have light
  at wavelengths where the PAOS file has no PSF (for instance a photometer with
  a red leak in its efficiency: the Ariel FGS-VISPhot has PSFs from 0.40 to
  0.70 um and light up to 2 um). The PSF images were extrapolated linearly,
  which gives negative, unphysical images (at 1.5 to 2 um the negative energy
  was two to three times the total energy): with a 3330 K star the brightest
  pixel of VISPhot was 10% too low and did not converge with the oversampling.
  Now, inside the PAOS range the images are interpolated linearly as before,
  and outside it the PSF of the nearest PAOS wavelength is scaled in size in
  proportion to the wavelength, conserving its energy
  (``exosim.utils.psf.scale_psf_image``). This is exact for the diffraction
  pattern and keeps the aberrations of the edge wavelength.
  ``LoadPsfPaosTimeInterp`` is fixed too. A PAOS file with a single wavelength
  is now the edge PSF scaled at every other wavelength, instead of a constant
  PSF.

- A channel with an ``<oversampling>`` in its ``<detector>`` but no
  ``<irf_task>`` ran silently at an oversampling of 1:
  ``CreateFocalPlaneArray`` set it to 1 with only a warning, and the
  recipes (``CreateFocalPlane``, ``RadiometricModel``) applied the intra pixel
  response function (IRF) only if ``irf_task`` was present. The PSF was then
  sampled at the pixel centres and not integrated over the pixel, so the
  brightest pixel of an undersampled Airy PSF was 18-29% too high. A pixel
  always integrates the light over its area, so the configured oversampling is
  now always kept and, whenever it is larger than 1, the IRF is applied with the
  default task, ``CreateIntrapixelResponseFunction``, if ``irf_task`` is not
  given. Without ``diffusion_length`` the default kernel is a top-hat over the
  pixel (``osf x osf`` equal values, summing to ``osf**2``). There is no switch
  to restore the old behaviour: use an earlier release. **This changes the
  results for payloads that set an oversampling without ``irf_task``: the focal
  planes, the signals and the saturation times of every such channel.**

- **The absolute signals of channels with an analytic (Airy or Gauss) PSF
  change.** Two faults of the PSF window are fixed. There is no switch to
  restore the old behaviour: use an earlier release.

  - ``create_psf``: the window was half as wide as documented. Its half width
    was :math:`j_{1,n} F\lambda / (2\pi)`, but the Airy pattern of the code,
    :math:`J_1(\pi r / F\lambda)`, has its n-th dark ring at
    :math:`j_{1,n} F\lambda / \pi`. With the default ``nzero = 4`` the window
    reached :math:`2.1 F\lambda` and held 91% of the light, not
    the fourth dark ring at :math:`4.2 F\lambda`. The half width is now
    :math:`j_{1,n} F\lambda / \pi`, so ``nzero`` is the number of dark rings
    that the window reaches. The Gaussian PSF uses the same window and it is
    also twice as wide. **A configuration with an explicit ``nzero`` gets a
    window twice as wide, and more light.**
  - ``PopulateFocalPlane``: the default window lost light that lands on the
    detector. Without ``nzero`` and ``size_x``/``size_y`` in ``<psf>``, the
    window is now a square whose side is the shorter side of the oversampled
    focal plane (made odd and centred, never larger than the focal plane), the
    rule that ``LoadPsfPaos`` already uses. The light lost is now only the
    light that falls outside the detector square, computed by the same
    analytic energy fraction. An explicit ``nzero`` or ``size_x``/``size_y`` is
    respected. ``size_x`` and ``size_y`` were also ignored unless a ``size``
    key was present; they are now read as documented.

  On the Ariel payload (1 h, oversampling 3), the NIRSpec total star signal
  rises by 2.6% (7.705e6 to 7.902e6 in the sum of the decimated focal plane)
  and the brightest pixel holds 1.145% of it instead of 1.174%. The PSF cube of
  a channel grows from the old default window to the detector square (NIRSpec:
  47 x 71 to 179 x 179 sub-pixels), and with it the memory of the PSF cube and
  the run time of the population step. Channels with a PAOS PSF are unchanged.

- The source of a `photometer` was placed on physical pixel ``N // 2 - 1``, one
  pixel below and to the left of the pixel ``N // 2`` on which every aperture
  is centred (``EstimateApertures`` and ``find_elliptical_aperture`` use
  ``shape / 2``, which in the :mod:`photutils` convention is the centre of pixel
  ``N // 2``; the aperture files of the Ariel payload do the same). The offset
  dates back to the first version of the code and most likely compensated the
  PAOS PSF being one sample off, fixed earlier in this release: with both, the
  star sat on pixel ``N // 2`` at ``osf = 1``. After that fix the offset moved
  the star one pixel away from the aperture centre, and the star signal in the
  aperture of the photometers was 2.5 to 4.5% too low (toy payload of the
  ArielRad2 comparison). The source of a photometer is now on pixel
  ``N // 2``, as for the spectrometers. **This changes the star signal in the
  aperture and the brightest pixel of the photometers.**

- The jitter magnification grid of ``InstantaneousReadOut.oversample`` did not
  keep the centre of the pixels fixed. The centre of the physical pixel ``m``
  is the sub-pixel ``osf * m + osf // 2`` of the focal plane, and
  ``o * m + o // 2`` after the magnification ``mag``, with ``o = osf * mag``.
  The old grid placed the new sub-pixels at ``-(mag - 1) / (2 * mag)`` from the
  old ones, which matches the pixel centres only when ``osf`` and ``mag`` are
  both odd. In the other cases the jittered scene was shifted by a fraction of
  an old sub-pixel (for ``osf = 3`` and ``mag = 4``, about 0.04 pixels). The new
  sub-pixel ``n`` now samples the old focal plane at
  ``osf // 2 + (n - o // 2) / mag``, so that the centres coincide for any
  ``osf`` and ``mag``. ``get_oversample_factors`` and ``oversample`` take the
  base oversampling factor ``osf`` as a new required argument. For odd ``osf``
  and odd ``mag`` the result is unchanged. **This changes the position of the
  jittered scene when ``osf`` or ``mag`` is even.**
- ``force_power_conservation`` divided the power of the original focal plane by
  the square of the magnified oversampling factor ``osf * mag`` instead of
  ``osf``, so the target power was wrong by a factor ``mag**2`` whenever the
  jitter magnification was larger than one. It now uses the base oversampling
  factor. **This changes the signal of the jittered channels when
  ``force_power_conservation`` is enabled and the magnification is above one.**

.. _changelog-2.2.4:

[2.2.4_] - Rescale sources API
==============================

Added
-----

- ``Channel.rescale_sources()`` and ``Channel.rescale_radiances()``: the two
  halves of ``Channel.rescale_contributions()``, which now calls both and
  behaves as before. A pipeline that propagates the sources a second time
  (for example after ``define_sources``) can rescale only the sources, without
  multiplying the path radiances by the wavelength step again.

.. _changelog-2.2.3:

[2.2.3_] - Elliptical aperture area
===================================

Fixed
-----

- elliptical apertures: ``aperture_size`` (the pixel count used by
  ``ComputeConstantDarkCurrentNoise`` and ``ComputeConstantReadNoise``) was
  computed as :math:`\pi (a/2)(b/2)`, four times smaller than the area
  integrated by the photometry, which uses ``a`` and ``b`` as the semi-axes of
  the ``photutils`` ``EllipticalAperture``. It is now :math:`\pi a b`.
  ``find_elliptical_aperture`` likewise returns :math:`\pi a b` as its area
  (it returned ``a * b``) and its docstring now states that the sizes are
  semi-axes. Aperture tables generated with earlier versions have a wrong
  ``aperture_size`` for elliptical rows and should be regenerated

.. _changelog-2.2.2:

[2.2.2_] - Fix ACES solar-metallicity SED download
==================================================

Changed
-------

- the documentation home page now points readers who only need signal and
  noise budgets to ExoRad 2, the radiometric model ExoSim 2 builds on

Fixed
-----

- ``DownloadSed`` with ``model_name="phoenix-aces"`` built a 404 URL for a
  solar-metallicity star: the Goettingen server names the directory and files
  ``Z-0.0`` / ``lte...-0.0...``, but the URL builder formatted ``0.0`` as
  ``+0.0``. Metallicity formatting is now routed through ``_feh_str``.

.. _changelog-2.2.1:

[2.2.1_] - Pipeline hardening
=============================

Changed
-------

- user guide and developer guide reworked for clarity: tighter prose, corrected
  grammar, consistent sentence-case headings, and each mechanism now explains
  the reason behind it, not just the behaviour
- home page, user-guide and developer-guide landing pages now share the same
  card-based navigation with consistent line icons, rendered from
  reStructuredText instead of hand-written HTML and clip-art

Removed
-------

- unused documentation dependencies from the ``docs`` extra
  (``sphinx-rtd-theme``, ``sphinx-panels``, ``sphinxcontrib-napoleon``,
  ``nbsphinx``, ``myst-parser`` and a few transitive-only pins)

- dead duplicate ``PrepareInstantaneousReadOut.force_power_conservation`` method
  (the live implementation is in ``InstantaneousReadOut``)

Fixed
-----

- ``nox -s docs`` and ``nox -s docs-live`` now install the ``docs`` extra, so the
  documentation builds on a clean checkout (Sphinx and the theme were missing)
- two broken links on the user-guide landing page
- several ``note`` and ``warning`` admonitions that were not rendering because
  their content was not indented
- broken cross-references to the installation instructions from the quick-start
  and FAQ pages
- malformed XML in the custom-task example (``unit`` attributes now quoted) and a
  duplicate documentation label for pixel non-linearity
- stale continuous-integration details in the contributing guide (workflow name
  and supported Python versions)
- title overline/underline mismatch on the sky-sources page that newer Sphinx
  rejects
- corrupted LaTeX in several task docstrings (``\frac``, ``\times`` and similar
  were being read as escape characters), so the rendered formulae for read
  noise, dark current noise and the dark-current map are now correct
- the ExoSim logo no longer overflows and clips the page header

- ``FocalPlanePlotter`` failed with ``KeyError: 'channels'`` on a normal
  focal-plane output: it looked for ``channels/channels`` instead of
  ``channels``

- ``exosim.__branch__`` and ``exosim.__commit__`` were always ``None``: the git
  metadata reader looked for ``.git`` one directory too shallow

- ``InstantaneousReadOut`` in memory-saving mode (``slicing=True``) crashed with
  ``AttributeError: 'InstantaneousReadOut' object has no attribute
  'getOversampleFactors'`` whenever the pointing jitter needed sub-pixel
  magnification; the call now uses the renamed ``get_oversample_factors``

- the target-list mode of ``RadiometricModel`` (a ``targetlist_filepath`` under
  ``sky/source``) crashed with ``KeyError: component not found``: it rebinned
  the channel efficiencies from the file root instead of from the per-target
  ``targets/<name>/channels`` group. The recipe now runs to completion and
  writes a radiometric table for every target in the list.

- ``RadiometricModel`` and ``CreateFocalPlane`` failed on a payload with a
  single, unwrapped ``channel`` entry (``KeyError: component not found`` /
  ``KeyError: 'Photometer'``): the efficiency-rebinning and the
  foreground/source signal helpers iterated the channel dictionary as if it
  were the multi-channel mapping. Single-channel configurations now run through
  the whole radiometric pipeline.

- ``Signal.get_slice`` / ``Signal.set_slice`` now accept a plain ``float``
  start/end time as documented (assumed to be hours); before they raised
  ``AttributeError``
- ``AddGainDrift`` raises a clear ``KeyError`` when neither
  ``gain_drift_amplitude`` nor the amplitude range is configured, instead of an
  ``UnboundLocalError``
- ``EstimatePlanetarySignal`` raises a clear ``ValueError`` when the planet
  ``rp`` is neither a file path nor a number, instead of an
  ``UnboundLocalError``
- ``Channel.target_source`` raises an informative ``KeyError`` when several
  sources are defined but none (or more than one) is flagged as the science
  target, instead of an ``IndexError``
- ``RadiometricModel`` now passes the per-channel configuration to the
  sub-foreground signal task, so the ``<contribution>_total_signal`` columns are
  computed for every channel type
- ``CreateFocalPlane`` infers the wavelength grid from the channel passbands
  (``channel``/``wl_min``/``wl_max``) when ``wl_grid`` is absent from the
  configuration; the previous fallback looked up a non-existent ``channels`` key
  and always used the 1-2 micron default
- the spectrometer focal-plane builder no longer relies on a fragile
  buffer-level ``array == zeros`` comparison to detect a flat wavelength axis
  (in ``CreateFocalPlaneArray._wav_osr`` and the PSF window locator); it now
  checks the values explicitly
- ``locate_wavelength_windows`` raises a clear error for an unsupported PSF
  dimensionality or channel type instead of returning ``None`` (which made the
  caller fail with an opaque unpacking ``TypeError``)
- ``EstimateZodi`` no longer loops forever when the bundled ``data`` directory
  cannot be found by walking up from the module; the search is bounded and
  raises ``OSError`` if it fails
- ``AccumulateSubExposures`` corrupted the ramp of a single-exposure
  acquisition: the first chunk read ``state_machine[-1]`` / ``dataset[-1]`` (the
  wrapped-around last frame) as its carry-in offset. The first chunk now starts
  from zero
- ``AnalogToDigital`` clips the converted signal to the ADC full scale; a bright
  pixel above full scale with a fixed ADC gain used to wrap around on the
  integer cast (e.g. 200000 counts became 3392 on a 16-bit ADC)
- ``AddCosmicRays`` no longer fails with an ``UnboundLocalError`` when the
  configured ``interaction_shapes`` probabilities sum to less than one; the
  event shape falls back to the last configured shape
- ``MergeGroups`` raises a clear error when the number of sub-exposures is not
  ``n_groups * n_ndrs`` instead of silently mis-averaging a truncated group or
  raising ``IndexError``
- ``ComputeSourcesPointingOffset`` computes the source-to-pointing offset with
  ``SkyCoord.spherical_offsets_to``, so it handles the RA 0/360 wrap and the
  ``cos(dec)`` foreshortening (a plain RA/Dec difference was wrong away from the
  equator and near RA = 0); the offset is now rounded to the nearest sub-pixel
  rather than truncated

Security
--------

- refreshed the locked dependency versions to pull in security fixes for
  ``pillow``, ``requests``, ``urllib3``, ``cryptography``, ``idna``,
  ``jupyter-server``, ``jupyterlab``, ``notebook``, ``nbconvert``, ``mistune``,
  ``tornado``, ``pygments`` and the ``safety`` toolchain (``authlib``,
  ``nltk``); this clears the outstanding Dependabot alerts on ``uv.lock``. The
  release still installs the highest compatible versions for end users, so this
  only affects reproducible development and CI environments

.. _changelog-2.2.0:

[2.2.0_] - End-to-end simulations
=================================

Added
-----

- new recipe ``SimulateObservation`` to run the full simulation pipeline from an options file to the NDRs
- radiometric model now supports target-list mode
- new documentation section for the radiometric model with a detailed description of the pipeline and operating modes
- observation efficiency can now be specified in the options file
- ``slim_output`` option for recipes to reduce output file size by dropping intermediate data products
- SED download task to fetch stellar spectra from online databases (currently Phoenix models only)

Changed
-------

- ``fast_convolution`` now uses ``scipy.signal.fftconvolve`` for better performance and accuracy
- saturation time now includes the well-depth fraction; integration time is the minimum saturation time across the detector; frame time is integration time divided by observation efficiency
- photon noise is now computed on the observation-efficiency-corrected signal
- packaging and dependency management moved from Poetry to uv
- logging moved to ``structlog``
- ``wl_interpolate`` reworked to use argsort + inverse mapping, avoiding O(n^2) lookups and extra copies
- ExoSim can now run without any source (no input SED)
- versioning is now derived from git tags via setuptools-scm; releases are cut with ``nox -s release``

Fixed
-----

- normalization in ``Signal`` now handles custom units correctly
- foreground is now decimated before being added to the focal plane instead of summed
- inverted parameters in the wavelength-centering equation, to avoid a polynomial inversion
- wrong wavelength order for ``loadPsfPaos``

[2.1.1_] - CLI Fixed
====================
Fixed
-----
- missing summary for cli.

[2.1.0_] - Hello Python 3.12
============================
Added
-----
- added unit normalization in Signal

Changed
-------
- dropped support for Python 3.9 and 3.10. Now supports only Python >3.12
- test framework changed into pytest
- added default value for low frequency time grid



[2.0.1_] - stable release
=========================
Added
-----
- improved exosim CLI with new click commands
- option to load optical elements from .hdf5 files (LoadOpticalElementHDF5)
- option to parse options file replacing a custom keyword with a value (LoadOptions)
- added codemeta.json file for better metadata management
- warning for misuse of oversampling factor and intra-pixel response function
- added support for wavelength dependent limb darkening
Changed
-------
- updated citation information
- rp in astronomical signal is now binned with the same binning method as Signal
Fixed
-----
- Custom SED units and documentation (added missing angle in loadCustom.py)
- find keys is not case sensitive anymore
- cosmic rays now work for low rates
- fixed elliptical PSF from PAOS
- ADC offset implemented as trigger offset (no negative values)
- astronomical signal now is weighted by the stellar flux on the detector

[2.0.0-rc2_] - ADC automode fix and minor improvements
======================================================
Added
-----
- boundaries for random amplitude in gain noise
- option to force non-time-dependent PSF for LoadPsfPaos (`time_dependence` parameter to `False`)
- option to slice jitter to save memory when the focal plane is sampled at high frequencies (`slicing` parameter to `True`)
Changed
-------
- dependencies versions updated to support `Poetry` for Python 3.10
Fixed
-----
- added offset and "auto" mode to ADC
- removed Numba in populateFocalPlane (better numerical accuracy)
- interp2d replaced with RectBivariateSpline in LoadPsfPaos to support Scipy 1.10+

[2.0.0-rc1_] - Scipy compatibility fix
======================================
Added
-----
- .npy input support for pixel non-linearity coefficients (LoadPixelsNonLinearityMapNumpy)
- multiplicative noise simulator (AddGainNoise)

Fixed
-----
- replaced scipy.convolve with scipy.signal.convolve
- ADC now works with unsigned integers

[v2.0.0-rc0_] - Release Candidate
================================
Cleaned repository

.. _v2.0.0-rc0: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.0.0-rc0
.. _2.0.0-rc1: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.0.0-rc1
.. _2.0.0-rc2: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.0.0-rc2
.. _2.0.1: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.0.1
.. _2.1.0: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.1.0
.. _2.1.1: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.1.1

.. _2.2.0: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.2.0

.. _2.2.1: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.2.1

.. _2.2.2: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.2.2

.. _2.2.3: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.2.3

.. _2.2.4: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.2.4

.. _2.2.5: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.2.5

.. _keepachangelog: https://keepachangelog.com/en/1.0.0/
.. _semver: https://semver.org/spec/v2.0.0.html
