.. _darkcurrent:

==============
Dark Current
==============

The dark current signal can be added to each sub-exposure using the `dark_current` keyword

.. code-block:: xml

    <channel> channel
        <detector>
            <dark_current> True </dark_current>
        <detector>
    </channel>

or disabled by setting the `dark_current` keyword to `False`.

By default, this :class:`~exosim.tasks.task.Task` used to add the dark current is :class:`~exosim.tasks.detector.addConstantDarkCurrent.AddConstantDarkCurrent`,
which, as the name suggests, adds a constant flux to each pixel.

It can be set as

.. code-block:: xml

    <channel> channel
        <detector>
            <dark_current> True </dark_current>
            <dc_task> AddConstantDarkCurrent </dc_task>
            <dc_mean unit="ct/s"> 5 </dc_mean>
        <detector>
    </channel>

Using the configuration reported in the example, the codes adds to each pixel
:math:`5 \, ct/s \times t_{s, \,int}` where :math:`t_{s, \,int}` is the sub-exposure integration time.

It is always possible to replace this function with one using a dark current map, to add a different dark current to each pixel which can also evolve in time.
A custom task can be used to replace :class:`~exosim.tasks.detector.addConstantDarkCurrent.AddConstantDarkCurrent` (see :ref:`Custom Tasks`).

An implementation of dark current map has been prepared assuming numpy array (see `numpy documentation <https://numpy.org/devdocs/reference/generated/numpy.lib.format.html>`_)
 as input (:class:`~exosim.tasks.detector.addDarkCurrentMapNumpy.AddDarkCurrentMapNumpy`). It can be used as

.. code-block:: xml

    <channel> channel
        <detector>
            <dark_current> True </dark_current>
            <dc_task> AddDarkCurrentMapNumpy </dc_task>
            <dc_map_filename> dark_map.npy </dc_map_filename>
        <detector>
    </channel>

.. note::
    Other custom realizations of this Task can be developed by the user (see :ref:`Custom Tasks`).
