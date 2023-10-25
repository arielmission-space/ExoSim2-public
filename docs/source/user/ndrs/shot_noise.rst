
.. _shot noise:

===================================
Shot Noise
===================================

The addition of shot noise to each sub-exposures is managed by :class:`~exosim.tasks.detector.addShotNoise.AddShotNoise`.

This functionality can be enabled in the channel configuration file as

.. code-block:: xml

    <channel> channel
        <detector>
            <shot_noise> True </shot_noise>
        <detector>
    </channel>

or disabled by setting the `shot_noise` keyword to `False`

This :class:`~exosim.tasks.task.Task` replace the values of each pixel with random number distributed around its true value according to a Poisson distribution.

.. math::

    S_{meas} = \mathcal{P}(S_{true})

Where :math:`S_{meas}` is the new value, which represents the measured value, and :math:`S_{true}` is the true pixel count value, which also is the original one.

.. note::
    For reproducibility, the seed for the random generator can be set as described in :ref:`random_seed`. 
    Remember that in the case of multiple chunks used, the random seed used in any chunk is stored in the output file for reproducibility.
