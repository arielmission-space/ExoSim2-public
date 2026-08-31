===========
Changelog
===========

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (keepachangelog_), and this project adheres
to Semantic Versioning (semver_).

Unreleased changes live as fragments in ``changelog.d/`` and are merged here by
``nox -s release``. See ``docs/source/contributing/releasing.rst``.

.. scriv-insert-here

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

.. _keepachangelog: https://keepachangelog.com/en/1.0.0/
.. _semver: https://semver.org/spec/v2.0.0.html
