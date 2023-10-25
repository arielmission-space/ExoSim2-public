.. _adc_gain:

===================================
ADC gain estimator
===================================

The sub-exposure measured signals are in counts units, however we know that the detector output is reported in :math:`adu` units.
ExoSim simulates the Analog to Digital Converter (ADC) thanks to  :class:`~exosim.tasks.detector.analogToDigital.AnalogToDigital`,
which converts the :math:`counts` units of sub-exposures into :math:`adu` units of NDRs (see :ref:`analogtodigtital`)

ADC output are unsigned integers defined in a certain number of bits.
Because this objects can represent numbers up to a maximum value of :math:`2^{n_{bits}} -1`,
we need a conversion factor to rescale the floats value NDRs to fit in the new data type range.
This conversion factor is defined by `ADC_gain`. as:

.. math::

    g_{ADC} = \frac{2^{n_{bits}}-1 }{ADC_{max}}

where :math:`n_{bits}` is the number of bits set for the ADC and
:math:`ADC_{max}` is the maximum value we want the ADC to handle.
Assuming a 16 bits ADC, :math:`2^{16} -1 = 65535` and a desired :math:`ADC_{max} = 120000`,
then we have :math:`g_{ADC} = \frac{65535}{120000} = 0.546125`.

This condition can be expressed in the tools configuration files as:

.. code-block:: xml

    <channel> channel_name
        <detector>
            <ADC_num_bit> 16 </ADC_num_bit>
            <ADC_max_value> 120000 </ADC_max_value>
        </detector>
    </channel>
