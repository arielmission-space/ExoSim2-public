.. _add_gain_drift:

==========
Gain drift
==========

The :class:`~exosim.tasks.detector.addGainDrift.AddGainDrift` task models and
applies a gain drift to the detector. The drift is a polynomial trend in time
and wavelength: the polynomial coefficients are drawn at random within given
ranges, and the resulting amplitude is rescaled according to the input
parameters.

Usage and parameters
--------------------

To apply a gain drift with
:class:`~exosim.tasks.detector.addGainDrift.AddGainDrift`, set the following
parameters in the configuration file (example values are shown below):

- ``gain_coeff_order_t``: order of the polynomial for the time-dependent trend;
- ``gain_coeff_t_min``, ``gain_coeff_t_max``: range for the random coefficients
  of the time-dependent polynomial;
- ``gain_coeff_order_w``: order of the polynomial for the wavelength-dependent
  trend;
- ``gain_coeff_w_min``, ``gain_coeff_w_max``: range for the random coefficients
  of the wavelength-dependent polynomial;
- ``gain_drift_amplitude``: desired maximum gain-drift amplitude, relative to
  the signal.

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


:class:`~exosim.tasks.detector.addGainDrift.AddGainDrift` can also draw the
amplitude at random from a range, set with
``gain_drift_amplitude_range_min`` and ``gain_drift_amplitude_range_max``:

.. code-block:: xml

    <channel>
        <detector>
            <gain_drift> True </gain_drift>
            <gain_drift_task> AddGainDrift </gain_drift_task>

            <gain_drift_amplitude_range_min> 1e-2 </gain_drift_amplitude_range_min>
            <gain_drift_amplitude_range_max> 5e-2 </gain_drift_amplitude_range_max>
        </detector>
    </channel>

Customisation
-------------

:class:`~exosim.tasks.detector.addGainDrift.AddGainDrift` can be customised or
replaced by a user-defined implementation.

.. note::
    You are encouraged to develop custom versions of this task for your own
    simulation needs (see :ref:`Custom Tasks`).
