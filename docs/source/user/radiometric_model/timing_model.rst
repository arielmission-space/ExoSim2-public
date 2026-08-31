.. _timing_model:

============
Timing model
============

The timing model computes the timing parameters that set how the detector is
operated and how much of the observation is spent collecting light. These
parameters feed straight into the noise calculations, so getting them right
matters for the performance estimate:

- saturation time,
- integration time,
- duty cycle,
- frame time.

.. _saturation_time:

Saturation time
---------------

The saturation time is the longest exposure the detector can take before a pixel
fills up. The default task
:class:`~exosim.tasks.radiometric.saturation_channel.SaturationChannel` computes
it, delegating the calculation to
:class:`~exosim.tasks.radiometric.compute_saturation.ComputeSaturation`:

.. math::

    t_{\mathrm{sat}} = \frac{f_{\mathrm{well}} \times W_{\mathrm{depth}}}{S_{\mathrm{max}}}

where:

- :math:`t_{\mathrm{sat}}` is the saturation time,
- :math:`f_{\mathrm{well}}` is the fraction of the well depth to use, a safety
  factor that keeps the detector away from the non-linear regime,
- :math:`W_{\mathrm{depth}}` is the detector well depth,
- :math:`S_{\mathrm{max}}` is the highest signal rate found anywhere in the
  focal plane.

The task combines the source and foreground signals, scans the whole focal plane
for the maximum and minimum signal levels, and applies the formula above.

Set the well depth and the fraction in the configuration:

.. code-block:: xml

    <channel> channel_name
        <detector>
            <well_depth unit="ct"> 100000 </well_depth>
            <f_well_depth> 0.8 </f_well_depth>
        </detector>
    </channel>

The task can also be run from a script:

.. code-block:: python

    import exosim.tasks.radiometric as radiometric

    saturation_task = radiometric.SaturationChannel()
    sat_time, max_signal, min_signal = saturation_task(
        table=wavelength_table,
        description=channel_description,
        input_file=focal_plane_file
    )

To customise it, subclass
:class:`~exosim.tasks.radiometric.saturation_channel.SaturationChannel` and
override its methods.

Integration time
----------------

The task
:class:`~exosim.tasks.radiometric.compute_integration_time.ComputeIntegrationTime`
sets the integration time actually used for the observation. The default
implementation takes the shortest saturation time across all the spectral bins,
so that no bin saturates:

.. math::

    t_{\mathrm{integration}} = \min(t_{\mathrm{sat}, i})

where :math:`t_{\mathrm{sat}, i}` is the saturation time for spectral bin
:math:`i`. It is computed automatically from the saturation analysis.

Duty cycle and observation efficiency
-------------------------------------

The duty cycle is the fraction of the observation actually spent integrating,
once overheads, resets, shutters, choppers, calibration sequences and other
interruptions are taken out. The default implementation,
:class:`~exosim.tasks.radiometric.compute_observation_efficiency.ComputeObservationEfficiency`,
reads it from the configuration file:

.. code-block:: xml

    <channel> channel_name
        <radiometric>
            <observation_efficiency> 0.8 </observation_efficiency>
        </radiometric>
    </channel>

If it is not specified, the observation efficiency is set to 1.0, that is 100%
efficiency.

To compute it differently, write a custom task that inherits from
:class:`~exosim.tasks.radiometric.compute_observation_efficiency.ComputeObservationEfficiency`
and point to it in the XML configuration:

.. code-block:: xml

    <channel> channel_name
        <radiometric>
            <observation_efficiency_task> custom_observation_efficiency_task.py </observation_efficiency_task>
        </radiometric>
    </channel>

Dead-time-based efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ExoSim` also ships a built-in alternative,
:class:`~exosim.tasks.radiometric.compute_observation_efficiency_from_dead_time.ComputeObservationEfficiencyFromDeadTime`,
which derives the efficiency from the detector dead time. It is more accurate for
detectors with a significant readout overhead:

.. math::

    \eta = \frac{t_{\mathrm{int}}}{t_{\mathrm{int}} + t_{\mathrm{dead}}}

where :math:`t_{\mathrm{int}}` is the integration time and :math:`t_{\mathrm{dead}}`
is the detector dead time. Specify the dead time in the configuration:

.. code-block:: xml

    <channel> channel_name
        <radiometric>
            <observation_efficiency_task>
                ComputeObservationEfficiencyFromDeadTime
            </observation_efficiency_task>
            <dead_time unit="s"> 0.1 </dead_time>
        </radiometric>
    </channel>

Because the dead time is fixed, a longer integration time makes it weigh
proportionally less, so the observing efficiency grows with the integration
time.

Frame time
----------

The frame time is the total time per detector frame, integration time plus
readout overheads. It is computed automatically in
:func:`~exosim.tasks.radiometric.utils.compute_saturation.compute_saturation`
from the integration time and the duty cycle:

.. math::

    t_{\mathrm{frame}} = \frac{t_{\mathrm{integration}}}{\eta_{\mathrm{duty}}}
