.. _adc_gain:

==================
ADC gain estimator
==================

The measured sub-exposure signals are in counts, but the detector output is
reported in :math:`adu`. `ExoSim` simulates the analog-to-digital converter
(ADC) with :class:`~exosim.tasks.detector.analogToDigital.AnalogToDigital`, which
converts the :math:`counts` of the sub-exposures into the :math:`adu` of the NDRs
(see :ref:`analogtodigtital`).

The ADC output is an unsigned integer stored in a fixed number of bits. Since it
can only represent values up to :math:`2^{n_{bits}} - 1`, a conversion factor is
needed to rescale the floating-point NDRs into that range. This factor is the
``ADC_gain``:

.. math::

    g_{ADC} = \frac{2^{n_{bits}}-1 }{ADC_{max}}

where :math:`n_{bits}` is the number of bits of the ADC and :math:`ADC_{max}` is
the largest value the ADC should handle. For a 16-bit ADC,
:math:`2^{16} - 1 = 65535`; with a target :math:`ADC_{max} = 120000`, this gives
:math:`g_{ADC} = \frac{65535}{120000} = 0.546125`.

Set the two parameters in the tool configuration file:

.. code-block:: xml

    <channel> channel_name
        <detector>
            <ADC_num_bit> 16 </ADC_num_bit>
            <ADC_max_value> 120000 </ADC_max_value>
        </detector>
    </channel>
