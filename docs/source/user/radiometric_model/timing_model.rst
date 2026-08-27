.. _timing_model:

=============
Timing Model
=============

Overview
--------

The timing model computes fundamental timing parameters that determine detector operation and observation efficiency. These parameters directly influence noise calculations and detector performance:

- saturation time,
- integration time,
- duty cycle,
- frame time.

.. _saturation_time:

Saturation Time
---------------

The default Task :class:`~exosim.tasks.radiometric.saturation_channel.SaturationChannel` computes the maximum exposure time before detector saturation occurs.
It uses :class:`~exosim.tasks.radiometric.compute_saturation.ComputeSaturation` to perform the calculation.


.. math::

    t_{\mathrm{sat}} = \frac{f_{\mathrm{well}} \times W_{\mathrm{depth}}}{S_{\mathrm{max}}}

where:

- :math:`t_{\mathrm{sat}}` is the saturation time
- :math:`f_{\mathrm{well}}` is the fraction of well depth to use (safety factor)
- :math:`W_{\mathrm{depth}}` is the detector well depth
- :math:`S_{\mathrm{max}}` is the maximum signal rate in the focal plane

The task combines source and foreground signals, searches the entire focal plane for maximum and minimum signal levels, and applies the saturation formula.

To configure the well depth and fraction, use:

.. code-block:: xml

    <channel> channel_name
        <detector>
            <well_depth unit="ct"> 100000 </well_depth>
            <f_well_depth> 0.8 </f_well_depth>
        </detector>
    </channel>

And it can be run in a script as

.. code-block:: python

    import exosim.tasks.radiometric as radiometric

    saturation_task = radiometric.SaturationChannel()
    sat_time, max_signal, min_signal = saturation_task(
        table=wavelength_table,
        description=channel_description,
        input_file=focal_plane_file
    )

This task can be customized by subclassing the :class:`~exosim.tasks.radiometric.saturation_channel.SaturationChannel` class and overriding its methods.

Integration Time
----------------


The Task :class:`~exosim.tasks.radiometric.compute_integration_time.ComputeIntegrationTime` determines the actual integration time used for observations.

The default implementation uses the minimum saturation time across all spectral bins:

.. math::

    t_{\mathrm{integration}} = \min(t_{\mathrm{sat}, i})

where :math:`t_{\mathrm{sat}, i}` is the saturation time for spectral bin :math:`i`.

This is computed automatically from saturation analysis.


Duty Cycle/Observation Efficiency
-------------------------------------

Computes the observing efficiency accounting for time losses due to overheads, resets, shutters, choppers, calibration sequences, or other interruptions.
The default implementation :class:`~exosim.tasks.radiometric.compute_observation_efficiency.ComputeObservationEfficiency`, reads the observation efficiency from the configuration file:

.. code-block:: xml

    <channel> channel_name
        <radiometric>
            <observation_efficiency> 0.8 </observation_efficiency>
        </radiometric>
    </channel>

If not specified, the observation efficiency is set to 1.0 (100% efficiency).

The observation efficiency computation can be customized by implementing a custom task inheriting from :class:`~exosim.tasks.radiometric.compute_observation_efficiency.ComputeObservationEfficiency`.
Such custom task should then be specified in the XML configuration:

.. code-block:: xml

    <channel> channel_name
        <radiometric>
            <observation_efficiency_task> custom_observation_efficiency_task.py </observation_efficiency_task>
        </radiometric>
    </channel>

Dead Time Based Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~

ExoSim includes a built-in implementation :class:`~exosim.tasks.radiometric.compute_observation_efficiency_from_dead_time.ComputeObservationEfficiencyFromDeadTime` that computes observation efficiency based on detector dead time. This is more accurate for detectors with significant readout overhead.

The efficiency is calculated as:

.. math::

    \eta = \frac{t_{\mathrm{int}}}{t_{\mathrm{int}} + t_{\mathrm{dead}}}

where :math:`t_{\mathrm{int}}` is the integration time and :math:`t_{\mathrm{dead}}` is the detector dead time.

To use this implementation, specify the dead time in the configuration:

.. code-block:: xml

    <channel> channel_name
        <radiometric>
            <observation_efficiency_task>
                ComputeObservationEfficiencyFromDeadTime
            </observation_efficiency_task>
            <dead_time unit="s"> 0.1 </dead_time>
        </radiometric>
    </channel>

This approach accounts for the fact that longer integration times result in higher observing efficiency, as the fixed dead time has proportionally less impact.

Frame Time
-----------

This is calculated automatically in :func:`~exosim.tasks.radiometric.utils.compute_saturation.compute_saturation` as the total time per detector frame, including integration time and readout overheads.

.. math::

    t_{\mathrm{frame}} = \frac{t_{\mathrm{integration}}}{\eta_{\mathrm{duty}}}

Frame time is computed automatically from integration time and duty cycle.
