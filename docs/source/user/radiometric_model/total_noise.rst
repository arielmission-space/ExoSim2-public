.. _total noise:

=======================
Total Noise
=======================

The total relative noise is estimated for a :math:`1 \, hr` time scale observation.
Therefore it has units of :math:`\sqrt{hr}`.

The task dedicated to this job is :class:`~exosim.tasks.radiometric.computeTotalNoise.ComputeTotalNoise`.

We start from an empy array of variances :math:` Var_{1 \, hr}(\lambda)`.

This tasks iterates over the column of the radiometric table looking for noise sources.
If a column name contains the word `noise` then is handled by this task.
Assuming that the code founds a column name `X_noise`.
If the column `X_noise` units are :math:`ct/s` then

.. math::

    Var_{1 \, hr}(\lambda) = Var_{1 \, hr}(\lambda) + \frac{[\sigma_{X}(\lambda)]^2}{\Delta T_{int}}

where :math:`\Delta T_{int} = 3600 \, s` for the :math:`1 \, hr` integration time.

When the noise from all the columns that has such units has been added, the total variance is converted into relative noise.

.. note::
    To avoid confusion, only the noise from the source and the cumulative foreground are added to the total noise.

Then the relative noise is

.. math::
    \sigma_{1 \, hr}(\lambda) = \frac{Var_{1 \, hr}(\lambda)}{S_{source}(\lambda)}

where :math:`S_{source}` is the source signal in the radiometric table.

Assuming that the code founds also a column name `Y_noise` and that the column `Y_noise` has no units.
This is a relative noise already.
So, the code updates the total noise as

.. math::
    \sigma_{1 \, hr}(\lambda) = \sqrt{[\sigma_{1 \, hr}(\lambda)]^2 + [\sigma_{Y}(\lambda)]^2}

When also this check is concluded, the total relative noise is added to the radiometric table.

To run the task from script

.. code-block:: python

        import exosim.tasks.radiometric as radiometric

        computeTotalNoise = radiometric.ComputeTotalNoise()
        total_noise = computeTotalNoise(table)
