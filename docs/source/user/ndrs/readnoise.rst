.. _readnoise:

==========
Read noise
==========

Every time a pixel is read by the electronics, an error is introduced: the
`read noise`. It is the noise of the amplifier that converts the counts into an
analogue voltage change for the ADC. By default it is modelled by
:class:`~exosim.tasks.detector.addReadNoise.AddNormalReadNoise`, which simulates
the read noise as a normal distribution whose parameters are set in the
configuration file:

.. code-block:: xml

    <channel> channel
        <detector>
            <read_noise> True </read_noise>
            <read_noise_task> AddNormalReadNoise </read_noise_task>
            <read_noise_sigma unit="ct"> 10 </read_noise_sigma>
        </detector>
    </channel>

A separate draw from the same distribution is added to every pixel of every
sub-exposure:

.. math::
    S_{meas} = S_{meas} + \mathcal{N}(\mu = 0, \sigma = \sigma_{RN})

Alternatively, you can use a per-pixel map of measured read noise. A default
task is provided for a NumPy array input (see the `NumPy documentation
<https://numpy.org/devdocs/reference/generated/numpy.lib.format.html>`_):
:class:`~exosim.tasks.detector.addReadNoiseMapNumpy.AddReadNoiseMapNumpy`.

.. code-block:: xml

    <channel> channel
        <detector>
            <read_noise> True </read_noise>
            <read_noise_task> AddReadNoiseMapNumpy </read_noise_task>
            <read_noise_filename> read_noise_map.npy </read_noise_filename>
        </detector>
    </channel>

.. note::
    You can develop custom versions of this task (see :ref:`Custom Tasks`).

.. note::
    For reproducibility, the random-generator seed can be set as described in
    :ref:`random_seed`. When multiple chunks are used, the seed used for each
    chunk is stored in the output file.
