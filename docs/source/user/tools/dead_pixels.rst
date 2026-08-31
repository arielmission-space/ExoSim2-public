.. _dead_pixel_map:

===============
Dead pixels map
===============

The :class:`~exosim.tools.deadPixelsMap.DeadPixelsMap` tool creates dead-pixel
maps for use in `ExoSim`. Give it the number of dead pixels in the channel and it
scatters them at random over the focal plane.

Set the number in the tool input parameters:

.. code-block:: xml

    <channel> channel_name
        <detector>
            <dp_mean> 10 </dp_mean>
        </detector>
    </channel>

Then run the tool:

.. code-block:: python

    import exosim.tools as tools

    tools.DeadPixelsMap(options_file='tools_input_example.xml',
                        output='data/payload')

It writes one ``.csv`` file per channel with the coordinates of the dead pixels.
The result looks like this:

.. image:: _static/dp_map.png
    :width: 500
    :align: center

If the exact number of dead pixels is not known, add its uncertainty:

.. code-block:: xml

    <channel> channel_name
        <detector>
            <dp_mean> 10 </dp_mean>
            <dp_sigma> 1 </dp_sigma>
        </detector>
    </channel>

The tool then draws the number of dead pixels from a normal distribution centred
on ``dp_mean`` with standard deviation ``dp_sigma``.

The ``.csv`` files can be passed to
:class:`~exosim.tasks.detector.applyDeadPixelMap.ApplyDeadPixelsMap`, described in
:ref:`dead_pixels`.
