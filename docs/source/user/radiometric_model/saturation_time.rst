.. _saturation time:

=======================
Saturation time
=======================

The saturation time is estimated in each channel by :class:`~exosim.tasks.radiometric.saturationChannel.SaturationChannel`.
This :class:`~exosim.tasks.task.Task` sums the source and the foreground focal plane and look for the maximum and minimum signals.
From the detector details, it then estimates the saturation time.
The channel description must include the well depth and the fraction of well depth to use:

.. code-block:: xml

    <channel> channel_name

        <detector>
            <well_depth> 100000 </well_depth>
            <f_well_depth> 0.9 </f_well_depth>
        </detector>

    </channel>

The saturation time is estimated as the well depth divided by the maximum signal in a pixel of the array.
Then the integration time is the saturation time multiplied by the fraction of the well depth to use.

Inside :class:`~exosim.recipes.radiometricModel.RadiometricModel` this tasks
is handled by the :func:`~exosim.recipes.radiometricModel.RadiometricModel.compute_saturation` method.
To use the default task in a script on a channel the user can write:

.. code-block:: python

    import exosim.tasks.radiometric as radiometric

    saturationChannel = radiometric.SaturationChannel()
    sat, t_int, max_sig, min_sign = saturationChannel(table=table,
                                                      description=description
                                                      input_file=input)

Where `table` is the wavelength radiometric table with aperture, `ch_name` is the channel name,
`description` is the dictionary containing the channel information from the `xml` file, and
`input_file` is the input hdf5 file containing the focal planes.
