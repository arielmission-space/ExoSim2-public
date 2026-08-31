.. _read_noise_radiometric:

==========
Read noise
==========

Read noise is the noise added by the detector electronics every time the array is
read out. `ExoSim` includes a default task to add its contribution to the
radiometric model, and you can replace it with your own when needed.

Default task
------------

The default task
:class:`~exosim.tasks.radiometric.compute_constant_read_noise.ComputeConstantReadNoise`
computes the read noise for each aperture in the radiometric table. It treats the
read noise as a constant per-pixel value that scales with the aperture area and
the frame time.

Model
~~~~~

The read-noise variance is

.. math::

    \mathrm{read\_noise\_variance} = G \cdot \sigma_{\mathrm{RN}}^2 \cdot \frac{A}{t_{\mathrm{frame}}}

where:

- :math:`G` is the multiaccum gain factor,
- :math:`\sigma_{\mathrm{RN}}` is the read noise per pixel, in counts,
- :math:`A` is the aperture area, in pixels,
- :math:`t_{\mathrm{frame}}` is the frame time, in seconds.

The read noise is the square root of the variance,

.. math::

    \mathrm{read\_noise} = \sqrt{\mathrm{read\_noise\_variance}}

and is then normalised by the signal for a 1-hour exposure,

.. math::

    \mathrm{read\_noise\_norm} = \frac{\mathrm{read\_noise}}{S}

where :math:`S` is the signal.

Inputs
~~~~~~

The task needs:

- **signal**: the signal array used for the normalisation
  (:class:`astropy.units.Quantity`).
- **aperture_table**: a table with the aperture information, with the columns

  - ``aperture_size``: the aperture area, in pixels,
  - ``frame_time``: the frame time for each aperture, in seconds.

- **description**: the channel description, which must contain
  ``detector.read_noise_sigma``, the read noise per pixel
  (:class:`astropy.units.Quantity`, in ct).
- **multiaccum_gain**: the multiaccum gain factor (float or Quantity).

Output
~~~~~~

The task returns:

1. the aperture table with two new columns, ``read_noise_variance`` and
   ``read_noise`` (the latter normalised by the signal), and
2. an array of read-noise values for each aperture, normalised by the signal.

Configuration
-------------

To use the default task, set the read noise per pixel in the detector section
and enable the read noise in the radiometric section:

.. code-block:: xml

    <detector>
        <!-- other detector parameters -->
        <read_noise_sigma unit="ct"> 10 </read_noise_sigma>
    </detector>

    <radiometric>
        <read_noise> True </read_noise>
        <!-- optional: pick a custom task; defaults to ComputeConstantReadNoise -->
        <!-- <read_noise_task> ComputeConstantReadNoise </read_noise_task> -->
    </radiometric>

Custom task
-----------

To model the read noise differently, inherit from
:class:`~exosim.tasks.radiometric.compute_constant_read_noise.ComputeConstantReadNoise`.
The custom task should take the same inputs as the default one, implement the
``model()`` method, and return the same output (the table and the noise array).
