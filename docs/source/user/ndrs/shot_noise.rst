.. _shot noise:

==========
Shot noise
==========

Shot noise is added to each sub-exposure by
:class:`~exosim.tasks.detector.addShotNoise.AddShotNoise`.

Enable it in the channel configuration file:

.. code-block:: xml

    <channel> channel
        <detector>
            <shot_noise> True </shot_noise>
        </detector>
    </channel>

or disable it by setting `shot_noise` to `False`.

This :class:`~exosim.tasks.task.Task` replaces each pixel value with a random
draw from a Poisson distribution centred on its true value:

.. math::

    S_{meas} = \mathcal{P}(S_{true})

where :math:`S_{meas}` is the new, measured value and :math:`S_{true}` is the
true (original) pixel count.

.. note::
    For reproducibility, the random-generator seed can be set as described in
    :ref:`random_seed`. When multiple chunks are used, the seed used for each
    chunk is stored in the output file.
