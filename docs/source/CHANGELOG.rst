===========
Changelog
===========

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (keepachangelog_), and this project adheres
to Semantic Versioning (semver_).


[2.0.1_] - stable release
=======================================================
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
=======================================================
Added
------
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
=======================================================
Added
------
- .npy input support for pixel non-linearity coefficients (LoadPixelsNonLinearityMapNumpy)
- multiplicative noise simulator (AddGainNoise)

Fixed
-----
- replaced scipy.convolve with scipy.signal.convolve
- ADC now works with unsigned integers

[v2.0.0-rc0_] - Release Candidate
=======================================================
Cleaned repository

.. _v2.0.0-rc0: https://github.com/arielmission-space/ExoSim2-public/releases/tag/v2.0.0-rc0
.. _2.0.0-rc1: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.0.0-rc1
.. _2.0.0-rc2: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.0.0-rc2
.. _2.0.1: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v2.0.1


.. _keepachangelog: https://keepachangelog.com/en/1.0.0/
.. _semver: https://semver.org/spec/v2.0.0.html
