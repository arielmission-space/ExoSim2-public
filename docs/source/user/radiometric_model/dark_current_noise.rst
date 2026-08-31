.. _dark_current_noise_radiometric:

==================
Dark current noise
==================

Dark current noise comes from electrons that are generated thermally in the
detector material even when no light reaches it. `ExoSim` includes a default task
to add its contribution to the radiometric model, and you can replace it with
your own when needed.

Default task
------------

The default task
:class:`~exosim.tasks.radiometric.compute_constant_dark_current_noise.ComputeConstantDarkCurrentNoise`
computes the dark current noise for each aperture in the radiometric table. It
treats the dark current as a constant per-pixel rate that scales with the
aperture area and the exposure time.

Model
~~~~~

The dark current variance is

.. math::

    \mathrm{dark\_current\_variance} = G \cdot \mu_{\mathrm{DC}} \cdot A

where:

- :math:`G` is the multiaccum gain factor,
- :math:`\mu_{\mathrm{DC}}` is the mean dark current rate per pixel, in ct/s,
- :math:`A` is the aperture area, in pixels.

The variance above is a rate, in counts per second. Multiplying it by the
exposure time (3600 s) gives the total counts, and its square root gives the
Poisson noise for a 1-hour exposure:

.. math::

    \mathrm{dark\_current\_noise} = \sqrt{\mathrm{dark\_current\_variance} \cdot 3600 \, \mathrm{s}}

which is then normalised by the signal,

.. math::

    \mathrm{dark\_current\_noise\_norm} = \frac{\mathrm{dark\_current\_noise}}{S}

where :math:`S` is the signal.

Inputs
~~~~~~

The task needs:

- **signal**: the signal array used for the normalisation
  (:class:`astropy.units.Quantity`).
- **aperture_table**: a table with the aperture information, with the column
  ``aperture_size`` (the aperture area, in pixels).
- **description**: the channel description, which must contain
  ``detector.dark_current`` and ``detector.dc_mean``, the mean dark current rate
  per pixel (:class:`astropy.units.Quantity`, in ct/s).
- **multiaccum_gain**: the multiaccum gain factor (NumPy ndarray or float).

Output
~~~~~~

The task returns:

1. the aperture table with two new columns, ``dark_current_variance`` and
   ``darkcurrent_noise`` (the latter for a 1-hour exposure, normalised by the
   signal), and
2. an array of dark current noise values for each aperture, normalised by the
   signal.

Configuration
-------------

To use the default task, set the mean dark current rate in the detector section
and enable the dark current in the radiometric section:

.. code-block:: xml

    <detector>
        <!-- other detector parameters -->
        <dc_mean unit="ct/s"> 0.1 </dc_mean>
    </detector>

    <radiometric>
        <dark_current> True </dark_current>
        <!-- optional: pick a custom task -->
        <!-- <dark_current_task> ComputeConstantDarkCurrentNoise </dark_current_task> -->
    </radiometric>

Custom task
-----------

To model the dark current noise differently, inherit from
:class:`~exosim.tasks.radiometric.compute_constant_dark_current_noise.ComputeConstantDarkCurrentNoise`.
The custom task should take the same inputs as the default one, implement the
``model()`` method, and return the same output (the table and the noise array).
Point to it in the configuration:

.. code-block:: xml

    <radiometric>
        <dark_current> True </dark_current>
        <dark_current_task> path/to/my_custom_dark_current_noise.py </dark_current_task>
        <!-- any extra custom parameters -->
    </radiometric>

Make sure the custom task class is importable and on the Python path when you run
`ExoSim`.

Where it runs in the pipeline
-----------------------------

When it is enabled, the dark current noise is computed automatically as part of
the noise sequence:

1. multiaccum gain calculation,
2. photon noise,
3. **dark current noise** (this task),
4. read noise,
5. custom noise, if any,
6. total noise combination.

The dark current contribution is then included in the total noise budget for
each aperture and spectral bin.
