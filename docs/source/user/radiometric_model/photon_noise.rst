.. _photon noise:

============
Photon noise
============

For every signal in the radiometric table, `ExoSim` can compute the photon
noise. This is done by
:class:`~exosim.tasks.radiometric.computePhotonNoise.ComputePhotonNoise`.

First the incoming signal :math:`S` is scaled by the duty-cycle efficiency
:math:`\eta_{duty}`, to account for the observing time lost to shutters,
choppers, or other mechanisms that interrupt the observation (see
:ref:`timing_model` for the duty cycle and the timing calculations). The
effective signal is

.. math:: S_{eff} = \eta_{duty} \cdot S

The photon noise is Poisson, so the variance of the effective signal is
:math:`Var[S_{eff}] = S_{eff}`.

If the photon gain factor :math:`gain_{phot}` has been computed with the
multiaccum equation (see :ref:`multiaccum`), the variance is scaled by it:
:math:`Var[S_{eff}] = gain_{phot} \cdot Var[S_{eff}]`.

You can also add a margin to the photon noise:

.. code-block:: xml

    <channel> channel_name
        <radiometric>
            <photon_margin> 0.4 </photon_margin>
        </radiometric>
    </channel>

If a photon-noise margin :math:`\chi` is found in the description, the variance
becomes :math:`Var[S_{eff}] = (1+\chi) \cdot Var[S_{eff}]`. The noise returned is
:math:`\sigma = \sqrt{Var[S_{eff}]}`.

The duty-cycle efficiency is set in the radiometric configuration:

.. code-block:: xml

    <channel> channel_name
        <radiometric>
            <duty_cycle> 0.8 </duty_cycle>
        </radiometric>
    </channel>

If it is not specified, the duty cycle defaults to 1.0, that is 100% observing
efficiency.

For each channel, the task can be run from a script:

.. code-block:: python

    import exosim.tasks.radiometric as radiometric

    computePhotonNoise = radiometric.ComputePhotonNoise()
    phot_noise = computePhotonNoise(signal=table['signal_name'],
                                    description=description,
                                    multiaccum_gain=table['multiaccum_shot_gain'])
