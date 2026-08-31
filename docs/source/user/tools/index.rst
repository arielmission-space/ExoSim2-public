.. _tools:

============
ExoSim tools
============

.. image:: _static/exosim_tools.png
    :width: 300
    :align: center

This section describes a set of tools that `ExoSim 2` provides to help you
prepare a simulation. Each one produces an input that a task then consumes during
the run: a map, a set of coefficients, or a reading scheme.

Every tool reads its parameters from an ``.xml`` file. In the examples that
follow, this file is called ``tools_input_example.xml``. It mirrors the structure
of a normal `ExoSim` input:

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
-------------

.. toctree::
    :maxdepth: 1

    Quantum efficiency variation map <qe_map>
    Pixel non-linearity <pixel_non_linearity>
    Readout scheme calculator <readout_scheme_calculator>
    Create dead pixels map <dead_pixels>
    ADC gain estimator <adc_gain>
