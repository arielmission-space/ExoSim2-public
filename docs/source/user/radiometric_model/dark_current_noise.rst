Dark Current Noise
==================

Overview
--------

Dark current noise is a fundamental noise component in detector systems that arises from thermally generated electrons in the detector material, even when no photons are incident. ExoSim 2 includes a default task to compute the dark current noise contribution to the radiometric model, which can be customised by users when needed.

Default Task: ComputeConstantDarkCurrentNoise
---------------------------------------------

The default task :class:`~exosim.tasks.radiometric.compute_constant_dark_current_noise.ComputeConstantDarkCurrentNoise` computes the dark current noise for each aperture in the radiometric table. It models the dark current as a constant per-pixel rate that scales with the aperture area and exposure time.

Mathematical Model
~~~~~~~~~~~~~~~~~~

The dark current variance is computed using the following formula:

.. math::

    \mathrm{dark\_current\_variance} = G \cdot \mu_{\mathrm{DC}} \cdot A

where:

- :math:`G` is the multiaccum gain factor
- :math:`\mu_{\mathrm{DC}}` is the mean dark current rate per pixel (in ct/s)
- :math:`A` is the aperture area (in pixels)

The dark current noise for a 1-hour exposure is then:

.. math::

    \mathrm{dark\_current\_noise} = \sqrt{\mathrm{dark\_current\_variance} \cdot 3600 \, \mathrm{s}}

The result is normalised by the input signal:

.. math::

    \mathrm{dark\_current\_noise\_norm} = \frac{\mathrm{dark\_current\_noise}}{S}

where :math:`S` is the signal. Note that ``dark_current_variance`` is a rate in counts per second, so multiplying by the exposure time (3600 s) yields the total counts, and the square root gives the Poisson noise.

Task Parameters
~~~~~~~~~~~~~~~

The task requires the following parameters:

- **signal**: Signal array for normalisation (Astropy units.Quantity)
- **aperture_table**: Table containing aperture information with required columns:

  - ``aperture_size``: area of the aperture (in pixels)

- **description**: Channel description dictionary containing:

  - ``detector.dark_current``: dark current description (required key)
- ``detector.dc_mean``: mean dark current rate per pixel (Astropy units.Quantity, in ct/s)

- **multiaccum_gain**: Multiaccum gain factor (NumPy ndarray or float)

Task Output
~~~~~~~~~~~

The task returns:

1. An updated aperture table with two additional columns:

   - ``dark_current_variance``: computed dark current variance for each aperture
   - ``darkcurrent_noise``: computed dark current noise for 1-hour exposure, normalised by the signal

2. An array of dark current noise values for each aperture (normalised by the signal)

Configuration
-------------

XML Configuration for Default Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use the default dark current noise task in your instrument configuration, add the following to your channel configuration in the XML file:

.. code-block:: xml

    <detector>
        <!-- Other detector parameters -->

        <!-- Specify the mean dark current rate per pixel -->
        <dc_mean unit="ct/s"> 0.1 </dc_mean>
    </detector>

    <radiometric>
        <!-- Dark current noise will be computed automatically if enabled above -->
        <dark_current> True </dark_current>

        <!-- Optional: specify custom noise computation task -->
        <!-- <dark_current_task> ComputeConstantDarkCurrentNoise </dark_current_task> -->
    </radiometric>


Custom Dark Current Noise Tasks
--------------------------------


Users can implement custom dark current noise tasks by inheriting from the :class:`~exosim.tasks.radiometric.compute_constant_dark_current_noise.ComputeConstantDarkCurrentNoise` base class. The custom task should:

1. Accept the same input parameters as the default task
2. Implement the ``model()`` method for the actual computation
3. Return the same output format (table and noise array)


To use your custom dark current noise task, specify it in the XML configuration:

.. code-block:: xml

    <radiometric>
        <dark_current> True </dark_current>
        <dark_current_task> path/to/my_custom_dark_current_noise.py </dark_current_task>
        <!-- Additional custom parameters as needed -->
    </radiometric>

Make sure your custom task class is properly imported and available in the Python path when running ExoSim 2.

Integration with Radiometric Model
-----------------------------------

The dark current noise computation is automatically integrated into the radiometric model pipeline when enabled. The task is called as part of the noise computation sequence, which typically includes:

1. Multiaccum gain calculation
2. Photon noise computation
3. **Dark current noise computation** (this task)
4. Read noise computation
5. Custom noise computation (if specified)
6. Total noise combination

The dark current noise contribution is automatically included in the total noise budget for each aperture and spectral bin.
