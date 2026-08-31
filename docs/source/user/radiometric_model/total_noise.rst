.. _total noise:

===========
Total noise
===========

The total relative noise is estimated for a :math:`1 \, hr` observation, so it
has units of :math:`\sqrt{hr}`. The task in charge of it is
:class:`~exosim.tasks.radiometric.computeTotalNoise.ComputeTotalNoise`.

It starts from an empty array of variances :math:`Var_{1 \, hr}(\lambda)` and
scans the columns of the radiometric table for noise sources: a column counts as
a noise source if its name contains the word ``noise``.

Take a column called ``X_noise``. If its units are :math:`ct/s`, the variance is
updated as

.. math::

    Var_{1 \, hr}(\lambda) = Var_{1 \, hr}(\lambda) + \frac{[\sigma_{X}(\lambda)]^2}{\Delta T_{int}}

with :math:`\Delta T_{int} = 3600 \, s` for the 1-hour integration time. Once
every column with those units has been added, the total variance is turned into
relative noise,

.. math::
    \sigma_{1 \, hr}(\lambda) = \frac{Var_{1 \, hr}(\lambda)}{S_{source}(\lambda)}

where :math:`S_{source}` is the source signal in the radiometric table.

.. note::
    To avoid confusion, only the noise from the source and from the cumulative
    foreground are added to the total noise.

Now take a column called ``Y_noise`` that has no units. This is already a
relative noise, so it is combined directly:

.. math::
    \sigma_{1 \, hr}(\lambda) = \sqrt{[\sigma_{1 \, hr}(\lambda)]^2 + [\sigma_{Y}(\lambda)]^2}

Once this pass is done too, the total relative noise is written back to the
radiometric table.

To run the task from a script:

.. code-block:: python

    import exosim.tasks.radiometric as radiometric

    computeTotalNoise = radiometric.ComputeTotalNoise()
    total_noise = computeTotalNoise(table)
