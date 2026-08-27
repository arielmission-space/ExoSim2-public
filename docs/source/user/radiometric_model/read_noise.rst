Read Noise
==========

Overview
--------

The read noise is a fundamental noise component in detector systems that originates from the electronics during the readout process. ExoSim2 includes a default task to compute the read noise contribution to the radiometric model, which can be customized by users when needed.

Default Task: ComputeConstantReadNoise
--------------------------------------

The default task :class:`~exosim.tasks.radiometric.compute_constant_read_noise.ComputeConstantReadNoise` computes the read noise for each aperture in the radiometric table. It models the read noise as a constant per-pixel noise that scales with the aperture area and frame time.

Mathematical Model
~~~~~~~~~~~~~~~~~~

The read noise variance is computed using the following formula:

.. math::

    \mathrm{read\_noise\_variance} = G \cdot \sigma_{\mathrm{RN}}^2 \cdot \frac{A}{t_{\mathrm{frame}}}

where:

- :math:`G` is the multiaccum gain factor
- :math:`\sigma_{\mathrm{RN}}` is the read noise per pixel (in counts)
- :math:`A` is the aperture area (in pixels)
- :math:`t_{\mathrm{frame}}` is the frame time (in seconds)

The final read noise is then:

.. math::

    \mathrm{read\_noise} = \sqrt{\mathrm{read\_noise\_variance}}

The result is normalized by the input signal for a 1-hour exposure:

.. math::

    \mathrm{read\_noise\_norm} = \frac{\mathrm{read\_noise}}{S}

where :math:`S` is the signal.

Task Parameters
~~~~~~~~~~~~~~~

The task requires the following parameters:

- **signal**: Signal array for normalization (astropy.units.Quantity)
- **aperture_table**: Table containing aperture information with required columns:

  - ``aperture_size``: area of the aperture (in pixels)
  - ``frame_time``: frame time for each aperture (in seconds)

- **description**: Channel description dictionary containing:

  - ``detector.read_noise_sigma``: read noise per pixel (astropy.units.Quantity, in ct)

- **multiaccum_gain**: Multiaccum gain factor (float or Quantity)

Task Output
~~~~~~~~~~~

The task returns:

1. An updated aperture table with two additional columns:

   - ``read_noise_variance``: computed read noise variance for each aperture
   - ``read_noise``: computed total read noise for each aperture, normalized by the signal

2. An array of read noise values for each aperture (normalized by the signal)

Configuration
-------------

XML Configuration for Default Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use the default read noise task in your instrument configuration, add the following to your channel configuration in the XML file:

.. code-block:: xml

    <detector>
        <!-- Other detector parameters -->

        <!-- Specify the read noise sigma per pixel -->
        <read_noise_sigma unit="ct"> 10 </read_noise_sigma>

    </detector>

    <radiometric>
        <!-- Read noise will be computed automatically if enabled above -->
        <read_noise> True </read_noise>

        <!-- Optional: specify custom task (if not specified, uses ComputeConstantReadNoise) -->
        <!-- <read_noise_task> ComputeConstantReadNoise </read_noise_task> -->
    </radiometric>


Creating a Custom Task
~~~~~~~~~~~~~~~~~~~~~~~

Users can implement custom read noise tasks by inheriting from the :class:`~exosim.tasks.radiometric.compute_constant_read_noise.ComputeConstantReadNoise` base class. The custom task should:

1. Accept the same input parameters as the default task
2. Implement the ``model()`` method
3. Return the same output format (table and noise array)
