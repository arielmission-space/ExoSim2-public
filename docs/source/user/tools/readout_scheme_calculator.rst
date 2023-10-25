.. _readout_scheme_calculator:

===================================
Readout Scheme Calculator
===================================

Let's assume we want to produce the following readout scheme, but know only the following information.

.. image:: _static/reading_ramp.png
   :width: 600
   :align: center

The ramp is sampled at `readout_frequency` cadence, defined in :ref:`sub-exposures creation`.
In this example we want to spend :math:`0.2 \, s` seconds in ground (GND) state and
:math:`0.2 \, s` before the reset state (RST).
The NDRs are read at a constant cadence of :math:`0.1 \, s`.
Then we want to have 3 groups which divide the residual ramp into equal parts.
Each group consists of 2 NDRs separated the time needed to read a NDR.


To produce the reading scheme, we need to know the time spent between groups
(which is reported in the bottom part of the figure expressed as time from the start of the simulation).
But we also need to translate our human-readable units into the simulation clock unis needed for :class:`~exosim.tasks.subexposures.computeReadingScheme.ComputeReadingScheme`,
as described in :ref:`reading_scheme`.
Finally, we want to make good use of the ramp sampling, and therefore we don't want to saturate the detector.
All of this is handled by :class:`~exosim.tools.readoutSchemeCalculator.ReadoutSchemeCalculator`.


First we need to translate all the known parameter in the following description in the channel section of the tool input file:

.. code-block:: xml

    <channel> channel name
        <readout>
            <n_NRDs_per_group> 2 </n_NRDs_per_group>
            <n_groups>  3 </n_groups>
            <readout_frequency unit ='s'> 0.1 </readout_frequency>
            <Ground_time unit ='s'> 0.2 </Ground_time>
            <Reset_time unit ='s'> 0.2 </Reset_time>
        </readout>
    </channel>

The user can also set the `readout_frequency` in units of :math:`Hz` instead of :math:`s`.

Obviously, to estimate the saturation time some other input is needed: the focal planes.
We assume here that the focal plane are stored in `input_file.h5`:

.. code-block:: python

    import exosim.tools as tools

    tools.ReadoutSchemeCalculator(options_file = 'tools_input_example.xml',
                                  input_file='input_file.h5')

The code will then suggest the inputs to write on the payload configuration file.
In this case, according to the figure, the results will be


.. code-block:: xml

    <channel> channel name
        <readout>
            <readout_frequency unit="Hz">10</readout_frequency>
            <n_NRDs_per_group> 2 </n_NRDs_per_group>
            <n_groups> 3 </n_groups>
            <n_sim_clocks_Ground> 2 </n_sim_clocks_Ground>
            <n_sim_clocks_first_NDR> 1 </n_sim_clocks_first_NDR>
            <n_sim_clocks_Reset> 2 </n_sim_clocks_Reset>
            <n_sim_clocks_groups> 296 </n_sim_clocks_groups>
        </readout>
    </channel>

Which will results in the following scheme

.. image:: _static/reading_ramp_nclock.png
   :width: 600
   :align: center


Also, the user can set a custom exposure time to use instead of the saturation time:

.. code-block:: xml

    <channel> channel name
        <readout>
            <exposure_time unit="s"> 60 </exposure_time>
        </readout>
    </channel>
