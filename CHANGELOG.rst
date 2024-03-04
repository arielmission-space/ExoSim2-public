===========
Changelog
===========

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (keepachangelog_), and this project adheres
to Semantic Versioning (semver_).

[v2.0.0-rc1_] - Scipy compatibility fix
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
.. _v2.0.0-rc1: https://github.com/arielmission-space/ExoSim2-public/releases/tag/vv2.0.0-rc1

.. _keepachangelog: https://keepachangelog.com/en/1.0.0/
.. _semver: https://semver.org/spec/v2.0.0.html
