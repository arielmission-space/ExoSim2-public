.. _readnoise:

===================================
Read out noise
===================================

Every time a pixel is read by the electronic, an error is introduced. This is called `read noise`.
This is the noise of the amplifier which converts the counts into a change in analog voltage for the ADC.
This kind of uncertainty is represented by default by  :class:`~exosim.tasks.detector.addReadNoise.AddNormalReadNoise`.
This :class:`~exosim.tasks.task.Task` simulates the read noise as a normal distribution whose parameters can be defined in the configuration file.

.. code-block:: xml

    <channel> channel
        <detector>
            <read_noise> True </read_noise>
            <read_noise_task> AddNormalReadNoise </read_noise_task>
            <read_noise_sigma unit="ct"> 10 </read_noise_sigma>
        <detector>
    </channel>

A different realization of the same distribution is added to each pixel of each sub-exposure.

.. math::
    S_{meas} = S_{meas} + \mathcal{N}(\mu = 0, \sigma = \sigma_{RN})

Alternatively, a map of read noise measured for each pixel can be used. A default Task is provided for this scope assuming numpy array (see `numpy documentation <https://numpy.org/devdocs/reference/generated/numpy.lib.format.html>`_) as input: :class:`~exosim.tasks.detector.addReadNoiseMapNumpy.AddReadNoiseMapNumpy`

.. code-block:: xml

    <channel> channel
        <detector>
            <read_noise> True </read_noise>
            <read_noise_task> AddReadNoiseMapNumpy </read_noise_task>
            <read_noise_filename> read_noise_map.npy </read_noise_filename>
        <detector>
    </channel>
    
.. note::
    Other custom realizations of this Task can be developed by the user (see :ref:`Custom Tasks`).

.. note::
    For reproducibility, the seed for the random generator can be set as described in :ref:`random_seed`. 
    Remember that in the case of multiple chunks used, the random seed used in any chunk is stored in the output file for reproducibility.
