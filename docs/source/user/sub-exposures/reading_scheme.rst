.. _reading_scheme:

==============
Reading scheme
==============

Once the jitter timelines are ready, define the detector reading scheme.

.. note::
    This model considers only *instantaneous readout* of the detector.

Suppose we want to reproduce the reading scheme from
:ref:`sub-exposures creation`, where a :math:`60.3 \,s` exposure time is sampled
by 6 NDRs in 3 groups.

.. image:: ../tools/_static/reading_ramp_nclock.png
    :width: 600
    :align: center


The ramp is sampled at the ``readout_frequency`` cadence defined in
:ref:`sub-exposures creation`. In this example we assume:

+ the ground (GND) state lasts :math:`0.2\,s`, i.e. 2 simulation clocks at
  :math:`10\,Hz`;
+ the first NDR is read after 1 clock (:math:`0.1\,s`);
+ NDRs within a group are spaced by 1 clock;
+ groups are spaced by 296 simulation clocks;
+ the reset (RST) state lasts 2 clocks (:math:`0.2\,s`).

With these parameters, the NDRs fall at the following clock indices:

+ first NDR: starts at clock 2 (after GND), ends at 3;
+ second NDR: starts at 4, ends at 5;
+ third NDR: starts at 300 (= 4 + 296), ends at 301;
+ fourth NDR: starts at 302, ends at 303;
+ fifth NDR: starts at 598 (= 302 + 296), ends at 599;
+ sixth NDR: starts at 600, ends at 601.

The RST state then completes the ramp at clocks 602–603.

That is 603 simulation clocks at :math:`0.1\,s` resolution, i.e. exactly
:math:`60.3\,s`.

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

You can also give `readout_frequency` in :math:`Hz` instead of :math:`s`.

The reading scheme is computed by
:class:`~exosim.tasks.subexposures.compute_reading_scheme.ComputeReadingScheme`:

.. code-block:: python

        import exosim.tasks.subexposures as subexposures
        computeReadingScheme = subexposures.ComputeReadingScheme()
        clock, base_mask, frame_sequence, number_of_exposures = computeReadingScheme(
            parameters=parameters,
            main_parameters=main_parameters,
            focal_plane=focal_plane,
            frg_focal_plane=frg_focal_plane)

The outputs of this :class:`~exosim.tasks.task.Task` can look cryptic, because
they are shaped to optimise the next step of the sub-exposures procedure. Each
one is:

+ ``clock``: the simulation frequency, the inverse of the
  `high_frequencies_resolution` defined in :ref:`sub-exposures creation`;
+ ``base_mask``: the state machine for reading the ramp. A ramp is made of three
  states: ground (GND), reset (RST) and read (NDR). This mask is a list of 0s and
  1s, with 1 marking a read operation. For the image above, the base is
  ``[0, 1, 1, 1, 1, 1, 1, 0]``;
+ ``frame_sequence``: the full list of simulation steps for each step on the
  ramp, repeated for every ramp, e.g. ``[2, 1, 1, 296, 1, 296, 1, 2]``;
+ ``number_of_exposures``: the number of exposures needed to sample the whole
  observation with ramps of the exposure-time length. To get this, the
  :class:`~exosim.tasks.task.Task` computes the saturation time with
  :class:`~exosim.tasks.instrument.compute_saturation.ComputeSaturation`, which
  is why it needs the focal planes.


The exposure time is computed from the configuration with logic equivalent to a
hardware implementation (an FPGA, say), counting clocks for each operation:

.. code-block:: python

    # define exposure time in seconds
    exposure_time = (
        n_clk_GND                          # Ground state
        + n_clk_NDR0                       # First NDR
        + n_clk_NDR * (n_NRDs_per_group - 1)                    # Remaining NDRs in first group
        + (n_clk_GRP + n_clk_NDR * (n_NRDs_per_group - 1)) * (n_GRPs - 1)  # Other groups
        + n_clk_RST                        # Reset state
    ) * clock                              # Convert to seconds

This mirrors how readout operations are sequenced in a detector control system
or in programmable logic, making the timing and event spacing fully transparent.

For testing, since sampling the whole observation can be slow and produce many
sub-exposures, you can force the number of exposures:

.. code-block:: xml

    <channel> channel name
        <type> channel type </type>
        <readout>
            <n_exposures> 2 </n_exposures>
        </readout>
    </channel>

.. note::
    To help you design the detector reading scheme, `ExoSim` includes a
    dedicated tool: :ref:`readout_scheme_calculator`.

The readout scheme, together with everything needed for the instantaneous
readout, is computed by
:class:`~exosim.tasks.subexposures.PrepareInstantaneousReadOut.PrepareInstantaneousReadOut`.
