.. _estimate signals:

=======================
Estimate signals
=======================

There are different signals to estimate for the radiometric model, that depends on different focal planes:

+ source
+ foreground
+ sub foregrounds


Source and Foreground signal
-------------------------------

Source and foreground signal are estimated using aperture photometry in the same way and on the same apertures starting from their focal planes.
They are estimated for each channel by using :class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel` task by default.

:class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel` needs a radiometric table with apertures listed and a focal plane,
then it runs :class:`~exosim.tasks.radiometric.aperturePhotometry.AperturePhotometry` and returns its results.

.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>

        <radiometric>
            <signal_task> ComputeSignalsChannel </signal_task>
            ...
        </radiometric>

    </channel>

Inside :class:`~exosim.recipes.radiometricModel.RadiometricModel` this tasks is handled by the :func:`~exosim.recipes.radiometricModel.RadiometricModel.compute_source_signals` method
for the source focal plane and by the :func:`~exosim.recipes.radiometricModel.RadiometricModel.compute_foreground_signals` method for the foreground.
To use the default task in a script on a channel the user can write:

.. code-block:: python

    import exosim.tasks.instrument as instrument

    computeSignalsChannel = instrument.ComputeSignalsChannel()
    photometry = computeSignalsChannel(table=table,
                                       focal_plane=focal_plane)

Where `table` is the wavelength radiometric table with apertures and `focal plane` is the channel source or foreground focal plane array.

.. caution::
    If the user doesn't include the `signal_task` keyword in the channel description,
    the default :class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel` task is used.
    To develop a custom :class:`~exosim.tasks.task.Task`, please refer to :ref:`Custom Tasks`.

The default :class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel` task
uses the apertures center, sizes and shapes in the radiometric table to perform aperture photometry
with the appropriate apertures using :func:`photutils.aperture.aperture_photometry`.


Foreground sub focal plane signals
--------------------------------------

If at least one of the foreground has the `isolate` option enable,
there will be contributions to the focal plane to estimate for the radiometric table.
As mentioned already in :ref:`sub focal planes`, these focal planes are stored in a dedicated directory.
To estimate their contribution to the radiometric signal, a default :class:`~exosim.tasks.task.Task` has been developed:
:class:`~exosim.tasks.radiometric.computeSubFrgSignalsChannel.ComputeSubFrgSignalsChannel`.
As  :class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel`,
this task use :class:`~exosim.tasks.radiometric.aperturePhotometry.AperturePhotometry` to perform aperture photometry
on the same apertures used for source and general foreground focal planes.
This task should be indicated in the description document as

.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>

        <radiometric>
            <sub_frg_signal_task> ComputeSubFrgSignalsChannel </sub_frg_signal_task>
            ...
        </radiometric>

    </channel>

Inside :class:`~exosim.recipes.radiometricModel.RadiometricModel` this tasks
is handled by the :func:`~exosim.recipes.radiometricModel.RadiometricModel.compute_sub_foregrounds_signals` method.
To use the default task in a script on a channel the user can write:

.. code-block:: python

    import exosim.tasks.radiometric as radiometric

    computeFrgSignalsChannel = radiometric.ComputeSubFrgSignalsChannel()
    signal_table = computeFrgSignalsChannel(table=table,
                                            ch_name=ch,
                                            input_file=input,
                                            parameters=description)

Where `table` is th wavelength radiometric table with aperture, `ch_name` is the channel name,
`input_file` is the input hdf5 file containing the focal planes, and
`parameters` is the dictionary containing the aperture photometry information from the `xml` file.

.. caution::
    If the user doesn't include the `sub_frg_signal_task` keyword in the channel description,
    the default :class:`~exosim.tasks.radiometric.computeSubFrgSignalsChannel.ComputeSubFrgSignalsChannel` task is used.
    To develop a custom :class:`~exosim.tasks.task.Task`, please refer to :ref:`Custom Tasks`.
