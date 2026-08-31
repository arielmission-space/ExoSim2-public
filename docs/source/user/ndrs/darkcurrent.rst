.. _darkcurrent:

============
Dark current
============

The dark current is added to each sub-exposure with the `dark_current` keyword:

.. code-block:: xml

    <channel> channel
        <detector>
            <dark_current> True </dark_current>
        </detector>
    </channel>

or disabled by setting `dark_current` to `False`.

By default the dark current is added by
:class:`~exosim.tasks.detector.addConstantDarkCurrent.AddConstantDarkCurrent`,
which, as the name says, adds a constant flux to every pixel:

.. code-block:: xml

    <channel> channel
        <detector>
            <dark_current> True </dark_current>
            <dc_task> AddConstantDarkCurrent </dc_task>
            <dc_mean unit="ct/s"> 5 </dc_mean>
        </detector>
    </channel>

With this configuration, the code adds
:math:`5 \, ct/s \times t_{s, \,int}` to each pixel, where
:math:`t_{s, \,int}` is the sub-exposure integration time.

You can replace this with a dark current map, to give each pixel a different
dark current that can also evolve in time. A custom task can replace
:class:`~exosim.tasks.detector.addConstantDarkCurrent.AddConstantDarkCurrent`
(see :ref:`Custom Tasks`).

A map-based implementation is provided for NumPy array input (see the `NumPy
documentation
<https://numpy.org/devdocs/reference/generated/numpy.lib.format.html>`_),
:class:`~exosim.tasks.detector.addDarkCurrentMapNumpy.AddDarkCurrentMapNumpy`:

.. code-block:: xml

    <channel> channel
        <detector>
            <dark_current> True </dark_current>
            <dc_task> AddDarkCurrentMapNumpy </dc_task>
            <dc_map_filename> dark_map.npy </dc_map_filename>
        </detector>
    </channel>

.. note::
    You can develop custom versions of this task (see :ref:`Custom Tasks`).
