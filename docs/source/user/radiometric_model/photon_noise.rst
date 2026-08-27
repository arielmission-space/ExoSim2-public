.. _photon noise:

=======================
Photon Noise
=======================

For each signal in the radiometric table, it is possible to compute the photon noise.
The photon noise is computed by :class:`~exosim.tasks.radiometric.computePhotonNoise.ComputePhotonNoise`

Before computing the photon noise, the incoming signal :math:`S` is first multiplied by the duty cycle efficiency :math:`\eta_{duty}`
to account for observing time losses due to shutters, choppers, or other mechanisms that interrupt the observation
(see :ref:`timing_model` for details on duty cycle and timing calculations).

The effective signal is then: :math:`S_{eff} = \eta_{duty} \cdot S`

Given the effective signal :math:`S_{eff}`, the resulting photon noise variance is :math:`Var[S_{eff}]=S_{eff}`.

If photon gain factor :math:`gain_{phot}` has been computed with multiaccum equation (see :ref:`multiaccum`), then  :math:`Var[S_{eff}]= gain_{phot} \cdot Var[S_{eff}]`.

The user can also add a margin to the photon noise as

.. code-block:: xml

    <channel> channel_name
        <radiometric>
            <photon_margin> 0.4 </photon_margin>
        </radiometric>
    </channel>

The duty cycle efficiency can be specified in the radiometric configuration:

.. code-block:: xml

    <channel> channel_name
        <radiometric>
            <duty_cycle> 0.8 </duty_cycle>
        </radiometric>
    </channel>

If not specified, the duty cycle defaults to 1.0 (100% observing efficiency).

If photon noise margin, :math:`\chi`, is found in the description, then  :math:`Var[S_{eff}]= (1+\chi) \cdot Var[S_{eff}]`.
The noise returned is :math:`\sigma = \sqrt{Var[S_{eff}]}`

For each channel, it can be run in a script as

.. code-block:: python

    import exosim.tasks.radiometric as radiometric

    computePhotonNoise = radiometric.ComputePhotonNoise()
    phot_noise = computePhotonNoise(signal=table['signal_name'],
                                    description=description,
                                    multiaccum_gain=table['multiaccum_shot_gain'])
