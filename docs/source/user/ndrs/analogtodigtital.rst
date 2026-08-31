.. _analogtodigtital:

============================
Analog-to-digital conversion
============================

At this point the NDRs are stored as `float64`, but a real detector reports its
output as integers in :math:`adu`. The analog-to-digital converter (ADC) is
simulated by :class:`~exosim.tasks.detector.analogToDigital.AnalogToDigital`,
which converts the :math:`counts` of the sub-exposures into the :math:`adu` of
the NDRs. It needs two inputs from the channel configuration file:

+ the number of bits of the output integer (e.g. 16);
+ the ADC gain factor.

.. code-block:: xml

    <channel> channel
        <detector>
            <ADC> True </ADC>
            <ADC_num_bit> 16 </ADC_num_bit>
            <ADC_gain> 0.5 </ADC_gain>
            <ADC_round_method>floor</ADC_round_method>
            <ADC_offset> 1000 </ADC_offset>
        </detector>
    </channel>

Set the `ADC` keyword to `True` to enable the conversion, or `False` to skip it.

In this example the ADC converts the NDRs into 16-bit unsigned integers. Since a
16-bit integer can hold values up to :math:`2^{16} - 1 = 65535`, a conversion
factor is needed to rescale the float NDRs into that range. That factor is
`ADC_gain` (:math:`g_{ADC}`): the floating-point focal plane is multiplied by it
before the conversion. If the gain is not known, the :ref:`adc_gain` tool can
estimate it.

.. math::

    S_{out} = [ ADC_{gain} \cdot( S_{meas} - ADC_{offset}) ]_{int}

The offset is subtracted from the NDRs, any resulting negative value is set to
zero, and any value above the ADC full scale (:math:`2^{n_{bits}} - 1`) is
clipped to it, as a real converter saturates rather than wrapping around. You
can request any integer number of bits up to 32.
:class:`~exosim.tasks.detector.analogToDigital.AnalogToDigital` then picks the
smallest Python data type that can hold the output, to keep the output product
small and representative of the real result.

The `ADC_round_method` keyword sets how floating-point values are cast to
integers:

- `floor` uses :func:`numpy.floor`;
- `ceil` uses :func:`numpy.ceil`;
- `round` uses :func:`numpy.round`.

The default is `floor`.

Automatic ADC
-------------

ExoSim can set these values for you:

.. code-block:: xml

    <channel> channel
        <detector>
            <ADC> True </ADC>
            <ADC_num_bit> 16 </ADC_num_bit>
            <ADC_gain> auto </ADC_gain>
            <ADC_offset> auto </ADC_offset>
        </detector>
    </channel>

With ``auto``, ExoSim computes the offset as the minimum value in the whole
datacube, and the gain as

.. math::

    g_{ADC} = \frac{2^{n_{bits}}-1 }{ADC_{max}-offset}

The offset and gain actually used are recorded in the output metadata.
