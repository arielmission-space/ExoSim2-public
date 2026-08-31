.. _wavelength bin:

==================
Wavelength binning
==================

The first step builds the starting radiometric table, with the spectral bins and
their edges. By default this is done for each channel by
:class:`~exosim.tasks.radiometric.estimateSpectralBinning.EstimateSpectralBinning`;
you can name your own :class:`~exosim.tasks.task.Task` in the channel
description, as described below.

Inside :class:`~exosim.recipes.radiometric_model.RadiometricModel` this task is
handled by
:func:`~exosim.recipes.radiometric_model.RadiometricModel.create_table`. To use
the default task in a script, on a channel parsed into a dictionary:

.. code-block:: python

    import exosim.tasks.radiometric as radiometric

    estimateSpectralBinning = radiometric.EstimateSpectralBinning()
    table = estimateSpectralBinning(parameters=channel_dict)


.. caution::
    If you omit the `spectral_binning_task` keyword from the channel
    description, the default
    :class:`~exosim.tasks.radiometric.estimateSpectralBinning.EstimateSpectralBinning`
    task is used. See :ref:`Custom Tasks` to develop a custom
    :class:`~exosim.tasks.task.Task`.


This :class:`~exosim.tasks.task.Task` returns an
:class:`astropy.table.QTable` per channel, with these columns:

====================    ====================================================
keyword                 content
====================    ====================================================
ch_name                 channel name
wavelength              central bin wavelength in :math:`\mu m`
bandwidth               band width of the spectral bin in :math:`\mu m`
left_bin_edge           left edge of the spectral bin
right_bin_edge          right edge of the spectral bin
====================    ====================================================

:class:`~exosim.tasks.radiometric.estimateSpectralBinning.EstimateSpectralBinning`
offers several ways to estimate the spectral binning, tuned in the channel
description.

Photometer
^^^^^^^^^^

For a photometer, the description XML file looks like this:

.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>
        ...
    </channel>

Here the radiometric table has a single bin at the central wavelength of the
photometer, with a width equal to the wavelength band. Give the minimum and
maximum wavelengths, with units, in the `xml` file:

.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>

        <spectral_binning_task> EstimateSpectralBinning </spectral_binning_task>
        <wl_min unit="micron"> 0.5 </wl_min>
        <wl_max unit="micron"> 0.6 </wl_max>

        ...
    </channel>

Spectrometer
^^^^^^^^^^^^

For a spectrometer, the description XML file looks like this:

.. code-block:: xml

    <channel> channel_name
        <type> spectrometer </type>

        ...
    </channel>

The wavelength grid can be estimated in two modes:

- **native**: if `targetR` is `native`, the grid is the pixel-level wavelength
  grid, with one bin per pixel;
- **fixed R**: if `targetR` is a constant value, the grid is built with
  :func:`~exosim.utils.grids.wl_grid`.

Give the mode in the configuration `xml` file, along with the minimum and
maximum wavelengths. The `native` configuration:

.. code-block:: xml

    <channel> channel_name
        <type> spectrometer </type>

        <spectral_binning_task> EstimateSpectralBinning </spectral_binning_task>
        <wl_min unit="micron"> 2 </wl_min>
        <wl_max unit="micron"> 6 </wl_max>
        <targetR> native </targetR>

        ...
    </channel>

The `fixed R` configuration:

.. code-block:: xml

    <channel> channel_name
        <type> spectrometer </type>
        <spectral_binning_task> EstimateSpectralBinning </spectral_binning_task>
        <wl_min unit="micron"> 2 </wl_min>
        <wl_max unit="micron"> 6 </wl_max>
        <targetR> 50 </targetR>

        ...
    </channel>
