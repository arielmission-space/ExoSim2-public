===========
Changelog
===========

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (keepachangelog_), and this project adheres
to Semantic Versioning (semver_).

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
Fix
----
- added offset and "auto" mode to ADC
- removed Numba in populateFocalPlane (better numerical accuracy)
- interp2d replaced with RectBivariateSpline in LoadPsfPaos to support Scipy 1.10+

[2.0.0-rc1_] - Scipy compatibility fix
=======================================================
Added
------
- .npy input support for pixel non-linearity coefficients (LoadPixelsNonLinearityMapNumpy)
- multiplicative noise simulator (AddGainNoise)

Fix
----
- replaced scipy.convolve with scipy.signal.convolve
- ADC now works with unsigned integers

[v2.0.0-rc0_] - Release Candidate
=======================================================
Cleaned repository

.. _v2.0.0-rc0: https://github.com/arielmission-space/ExoSim2-public/releases/tag/v2.0.0-rc0
.. _2.0.0-rc1: https://github.com/arielmission-space/ExoSim2.0-public/releases/tag/v2.0.0-rc1
.. _2.0.0-rc2: https://github.com/arielmission-space/ExoSim2.0-public/releases/tag/v2.0.0-rc2

.. _keepachangelog: https://keepachangelog.com/en/1.0.0/
.. _semver: https://semver.org/spec/v2.0.0.html
