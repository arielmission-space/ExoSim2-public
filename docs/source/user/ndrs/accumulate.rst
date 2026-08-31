.. _accumulate:

===============
Accumulate NDRs
===============

So far we have worked with sub-exposures. Each sub-exposure collects the signal
that arrives after the previous sub-exposure has finished collecting, as shown
in this figure from :ref:`sub-exposures creation`:

.. image:: ../sub-exposures/_static/reding_ramp_se_explained.png
    :width: 600
    :align: center

Now we accumulate the successive sub-exposures of the same exposure to build the
ramp. Starting from the first sub-exposure of the ramp, each sub-exposure
becomes itself plus the previous one. For an exposure of :math:`N`
sub-exposures:

.. math::

    Sub_0 = Sub_0

and for every other sub-exposure of the same ramp:

.. math::

    Sub_i = Sub_i + Sub_{i-1}

This is handled by
:class:`~exosim.tasks.detector.accumulateSubExposures.AccumulateSubExposures`,
which overwrites the input cached dataset with the accumulated one.
