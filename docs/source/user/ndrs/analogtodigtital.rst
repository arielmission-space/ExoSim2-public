.. _analogtodigtital:

===================================
Analog to Digital conversion
===================================

At this point the NDRs are stored as `float64`, however, we know that the detector output is reported in integers in :math:`adu` units.
Here we simulate the Analog to Digital Converter (ADC) thanks to  :class:`~exosim.tasks.detector.analogToDigital.AnalogToDigital`,
which converts the :math:`counts` units of sub-exposures into :math:`adu` units of NDRs.
This task needs two inputs from the channel configuration file:

+ the number of bits of the output integer (e.g. 16 bits)
+ ADC gain factor

These can be injected as

.. code-block:: xml

    <channel> channel
        <detector>
            <ADC> True </ADC>
            <ADC_num_bit> 16 </ADC_num_bit>
            <ADC_gain> 0.5 </ADC_gain>
            <ADC_round_method>floor</ADC_round_method>
            <ADC_offset> 1000 </ADC_offset>
        <detector>
    </channel>

To enable the conversion set `True` for the `ADC` keyword, as in the example.
If `False`, this step will be skipped.

In this example, we want the ADC to convert the NDRs into 16 bits unsigned integers.
Because integers can represent numbers up to a maximum value of :math:`2^{16} -1 = 65535`,
we need a conversion factor to rescale our float NDRs to fit in the new data type range.
This conversion factor is defined by `ADC_gain`, such that the float focal plane is multiplied by this gain (:math:`g_{ADC}`) before the conversion.
If this gain is not known, an estimate is provided by the :ref:`adc_gain` tool.

.. math::

    S_{out} = [ ADC_{gain} \cdot( S_{meas} - ADC_{offset}) ]_{int}

Note that the offset is subtracted from the NDRs. All the resulting negative values are set to zero.
The user can input any integer number of bits up to 32. 
The :class:`~exosim.tasks.detector.analogToDigital.AnalogToDigital` chooses the minimum Python 
data type to store the desired output to minimize the size of the output product and to be more representative of the expected result.

`ADC_round_method` keyword indicates which method the ADC should use to cast the float into integers. Three options are available:

- `floor` which uses :func:`numpy.floor`; 
- `ceil` which uses :func:`numpy.ceil`; 
- `round` which uses :func:`numpy.round`; 

The default is `floor`.

Automatic ADC
---------------

ExoSim can help you setting this values automatically as 

.. code-block:: xml

    <channel> channel
        <detector>
            <ADC> True </ADC>
            <ADC_num_bit> 16 </ADC_num_bit>
            <ADC_gain> auto </ADC_gain>
            <ADC_offset> auto </ADC_offset>
        <detector>
    </channel>

With this configuration, ExoSim will compute the offset (:code:`<ADC_offset> auto </ADC_offset>`) as the minimum value in all the datacube, and the gain (:code:`<ADC_gain> auto </ADC_gain>`) as 

.. math::

    g_{ADC} = \frac{2^{n_{bits}}-1 }{ADC_{max}-offset}


The used offset and gain are reported in the output metadata.
