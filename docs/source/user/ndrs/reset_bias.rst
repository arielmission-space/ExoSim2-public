.. _reset_bias:

=========
KTC noise
=========

When the detector is reset, the offset signal in each pixel can differ from
frame to frame. This is the kTC noise. Include it in the simulation with:

.. code-block:: xml

    <channel> channel
        <detector>
            <ktc_offset> True </ktc_offset>
        </detector>
    </channel>

or disable it by setting `ktc_offset` to `False`.

By default, the reset bias is added by
:class:`~exosim.tasks.detector.addKTC.AddKTC`, which adds a random number of
counts to each pixel of a ramp, drawn from a normal distribution with the given
mean and standard deviation:

.. code-block:: xml

    <channel> channel
        <detector>
            <ktc_offset> True </ktc_offset>
            <ktc_offset_task> AddKTC </ktc_offset_task>
            <ktc_sigma unit="ct"> 10 </ktc_sigma>
        </detector>
    </channel>


.. math::

    S_{meas} = S_{meas} + \mathcal{N}(\mu = 0, \sigma = \sigma_{KTC})

.. note::
    You can develop custom versions of this task (see :ref:`Custom Tasks`).
