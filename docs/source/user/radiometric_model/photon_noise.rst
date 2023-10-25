.. _photon noise:

=======================
Photon Noise
=======================

For each signal in the radiometric table is possible to compute the photon noise.
The photon noise is computed by :class:`~exosim.tasks.radiometric.computePhotonNoise.ComputePhotonNoise`

Given the incoming signal :math:`S` the resulting photon noise variance is :math:`Var[S]=S`.

If photon gain factor :math:`gain_{phot}` has been computed with multiaccum equation (see :ref:`multiaccum`), then  :math:`Var[S]= gain_{phot} \cdot Var[S]`.

The user can also add a margin to the photon noise as

.. code-block:: xml

    <channel> channel_name
        <radiometric>
            <photon_margin> 0.4 </photon_margin>
        </radiometric>
    </channel>

If photon noise margin, :math:`\chi`, is found in the description, then  :math:`Var[S]= (1+\chi) \cdot Var[S]`.
The noise returned is :math:`\sigma = \sqrt{Var[S]}`

For each channel can be run in a script as

.. code-block:: python

    import exosim.tasks.radiometric as radiometric

    computePhotonNoise = radiometric.ComputePhotonNoise()
    phot_noise = computePhotonNoise(signal=table['signal_name'],
                                    description=description,
                                    multiaccum_gain=table['multiaccum_shot_gain'])
