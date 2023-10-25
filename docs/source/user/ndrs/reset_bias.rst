.. _reset_bias:

===================================
KTC noise
===================================

When the detector is reset, the offset signal in each pixel of each frame can be different.
This is the kTC noise and can be included in the simulation as

.. code-block:: xml

    <channel> channel
        <detector>
            <ktc_offset> True </ktc_offset>
        <detector>
    </channel>

or disabled by setting the `ktc_offset` keyword to `False`.

By default, this :class:`~exosim.tasks.task.Task` used to add the reset bias is :class:`~exosim.tasks.detector.addKTC.AddKTC`,
which adds a random number of counts to each pixel of the same ramp, normally distributed according to the given mean and standard deviation:

.. code-block:: xml

    <channel> channel
        <detector>
            <ktc_offset> True </ktc_offset>
            <ktc_offset_task> AddKTC </ktc_offset_task>
            <ktc_sigma unit="ct"> 10 </ktc_sigma>
        <detector>
    </channel>


.. math::

    S_{meas} = S_{meas} + \mathcal{N}(\mu = 0, \sigma = \sigma_{KTC})

.. note::
    Other custom realizations of this Task can be developed by the user (see :ref:`Custom Tasks`).
