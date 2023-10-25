.. _tools:

===================================
ExoSim Tools
===================================

.. image:: _static/exosim_tools.png
    :width: 300
    :align: center

In this section we present and describe a set of tool we included in ExoSim 2 to help the user to perform its simulations.

All the parameters to run the tools are parsed from and `.xml` file. In these examples, we assume the input file is called `tools_input_example.xml`.
This file should mimic the general ExoSim input:

.. code-block:: xml

    <root>

        <ConfigPath> path/to/your/configs </ConfigPath>

        <channel> channel 1
            ...
        </channel>

        <channel> channel 2
            ...
        </channel>

    </root>


List of tools
---------------
.. toctree::
    :maxdepth: 1

    Quantum efficiency variation map   <qe_map>
    Pixels Non-Linearity <pixel_non_linearity>
    Readout Scheme Calculator   <readout_scheme_calculator>
    Create dead pixels map  <dead_pixels>
    ADC Gain estimator  <adc_gain>
