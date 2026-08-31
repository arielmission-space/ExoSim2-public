.. _estimate signals:

================
Estimate signals
================

The radiometric model estimates several signals, each from a different focal
plane:

+ source
+ foreground
+ sub-foregrounds


Source and foreground signal
----------------------------

The source and foreground signals are estimated with aperture photometry, the
same way and on the same apertures, starting from their focal planes. By default
this is done for each channel with the
:class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel`
task.

:class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel`
needs a radiometric table with the apertures listed and a focal plane; it then
runs :class:`~exosim.tasks.radiometric.aperture_photometry.AperturePhotometry`
and returns its results.

.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>

        <radiometric>
            <signal_task> ComputeSignalsChannel </signal_task>
            ...
        </radiometric>

    </channel>

Inside :class:`~exosim.recipes.radiometric_model.RadiometricModel` this task is
handled by
:func:`~exosim.recipes.radiometric_model.RadiometricModel.compute_source_signals`
for the source focal plane and by
:func:`~exosim.recipes.radiometric_model.RadiometricModel.compute_foreground_signals`
for the foreground. To use the default task in a script on a channel:

.. code-block:: python

    import exosim.tasks.radiometric as radiometric

    computeSignalsChannel = radiometric.ComputeSignalsChannel()
    photometry = computeSignalsChannel(table=table,
                                       focal_plane=focal_plane)

where `table` is the wavelength radiometric table with apertures and
`focal_plane` is the channel source or foreground focal-plane array.

.. caution::
    If you omit the `signal_task` keyword from the channel description, the
    default
    :class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel`
    task is used. See :ref:`Custom Tasks` to develop a custom
    :class:`~exosim.tasks.task.Task`.

The default
:class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel`
task uses the aperture centres, sizes and shapes in the radiometric table to run
aperture photometry with the right apertures, via
:func:`photutils.aperture.aperture_photometry`.


Foreground sub-focal-plane signals
----------------------------------

If at least one foreground has the `isolate` option enabled, there are extra
focal-plane contributions to estimate for the radiometric table. As mentioned in
:ref:`sub focal planes`, these focal planes are stored in a dedicated group. A
default :class:`~exosim.tasks.task.Task`,
:class:`~exosim.tasks.radiometric.computeSubFrgSignalsChannel.ComputeSubFrgSignalsChannel`,
estimates their contribution. Like
:class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel`,
it uses :class:`~exosim.tasks.radiometric.aperture_photometry.AperturePhotometry`
on the same apertures used for the source and the general foreground focal
planes. Name this task in the description document:

.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>

        <radiometric>
            <sub_frg_signal_task> ComputeSubFrgSignalsChannel </sub_frg_signal_task>
            ...
        </radiometric>

    </channel>

Inside :class:`~exosim.recipes.radiometric_model.RadiometricModel` this task is
handled by
:func:`~exosim.recipes.radiometric_model.RadiometricModel.compute_sub_foregrounds_signals`.
To use the default task in a script on a channel:

.. code-block:: python

    import exosim.tasks.radiometric as radiometric

    computeFrgSignalsChannel = radiometric.ComputeSubFrgSignalsChannel()
    signal_table = computeFrgSignalsChannel(table=table,
                                            ch_name=ch,
                                            input_file=input,
                                            parameters=description)

where `table` is the wavelength radiometric table with apertures, `ch_name` is
the channel name, `input_file` is the input HDF5 file with the focal planes, and
`parameters` is the dictionary with the aperture-photometry information from the
`xml` file.

.. caution::
    If you omit the `sub_frg_signal_task` keyword from the channel description,
    the default
    :class:`~exosim.tasks.radiometric.computeSubFrgSignalsChannel.ComputeSubFrgSignalsChannel`
    task is used. See :ref:`Custom Tasks` to develop a custom
    :class:`~exosim.tasks.task.Task`.
