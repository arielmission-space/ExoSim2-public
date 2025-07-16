.. _add_gain_drift:

====================================
Gain Drift
====================================

The :class:`~exosim.tasks.detector.addGainDrift.AddGainDrift` task, part of the ExoSim simulation package, is designed to model and apply gain drift to a detector simulator.
The gain drift is constructed as a polynomial trend dependent on time and wavelength.

The polynomial coefficients are randomly generated within specified ranges. Finally, the resulting amplitude is rescaled according to the input parameters.

Usage and Parameters
--------------------

To apply gain drift using the :class:`~exosim.tasks.detector.addGainDrift.AddGainDrift` task, the following parameters should be specified in the configuration file.
Here we include also some example values.
- ``gain_coeff_order_t``: Order of the polynomial used for the time-dependent trend.
- ``gain_coeff_t_min`` and ``gain_coeff_t_max``: Minimum and maximum values for the randomly generated coefficients of the time-dependent polynomial trend.
- ``gain_coeff_order_w``: Order of the polynomial used for the wavelength-dependent trend.
- ``gain_coeff_w_min`` and ``gain_coeff_w_max``: Minimum and maximum values for the randomly generated coefficients of the wavelength-dependent polynomial trend.
- ``gain_drift_amplitude``: gain drift desired maximum amplitude relative to the signal.

These parameters control the characteristics of the gain noise, allowing for detailed modeling of the detector's response.

.. code-block:: xml

    <channel>
        <detector>
            <gain_drift> True </gain_drift>
            <gain_drift_task> AddGainDrift </gain_drift_task>

            <gain_drift_amplitude> 1e-2 </gain_drift_amplitude>

            <gain_coeff_order_t> 5 </gain_coeff_order_t>
            <gain_coeff_t_min> -1.0 </gain_coeff_t_min>
            <gain_coeff_t_max> 1.0 </gain_coeff_t_max>

            <gain_coeff_order_w> 5 </gain_coeff_order_w>
            <gain_coeff_w_min> -1.0 </gain_coeff_w_min>
            <gain_coeff_w_max> 1.0 </gain_coeff_w_max>
        </detector>
    </channel>


Alternatively, the :class:`~exosim.tasks.detector.addGainDrift.AddGainDrift` task can also randomly estimate the amplitude of the gain drift by using the range defined with the keywords: ``gain_drift_amplitude_range_min``, ``gain_drift_amplitude_range_max``:

.. code-block:: xml

    <channel>
        <detector>
            <gain_drift> True </gain_drift>
            <gain_drift_task> AddGainDrift </gain_drift_task>

            <gain_drift_amplitude_range_min> 1e-2 </gain_drift_amplitude_range_min>
            <gain_drift_amplitude_range_max> 5e-2 </gain_drift_amplitude_range_max>
        </detector>
    </channel>

Customization
-------------

The :class:`~exosim.tasks.detector.addGainDrift.AddGainDrift` task is designed for flexibility and can be customized or replaced by a user-defined implementation as needed.

.. note::
    Users are encouraged to develop their own custom realizations of this task to fit specific simulation requirements (see :ref:`Custom Tasks`).
