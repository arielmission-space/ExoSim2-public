.. _wavelength bin:

=======================
Wavelength binning
=======================

The first step is to produce the starting radiometric table with the spectral bins and their edges.
This is done for each channel by the default :class:`~exosim.tasks.radiometric.estimateSpectralBinning.EstimateSpectralBinning`.
The user can specify a dedicated :class:`~exosim.tasks.task.Task` in the channel description as described later in this section.

Inside :class:`~exosim.recipes.radiometric_model.RadiometricModel` this task is handled by the :func:`~exosim.recipes.radiometric_model.RadiometricModel.create_table` method.
To use the default task in a script on a channel parsed in a dictionary, the user can write:

.. code-block:: python

    import exosim.tasks.radiometric as radiometric

    estimateSpectralBinning = radiometric.EstimateSpectralBinning()
    table = estimateSpectralBinning(parameters=channel_dict)


.. caution::
    If the user doesn't include the `spectral_binning_task` keyword in the channel description,
    the default :class:`~exosim.tasks.radiometric.estimateSpectralBinning.EstimateSpectralBinning` task is used.
    To develop a custom :class:`~exosim.tasks.task.Task`, please refer to :ref:`Custom Tasks`.


This :class:`~exosim.tasks.task.Task` returns an :class:`astropy.table.QTable` for each channel. The table has only the following keywords:

====================    ====================================================
keyword                 content
====================    ====================================================
ch_name                 channel name
wavelength              central bin wavelength in :math:`\mu m`
bandwidth               band width of the spectral bin in :math:`\mu m`
left_bin_edge           left edge of the spectral bin
right_bin_edge          right edge of the spectral bin
====================    ====================================================

The :class:`~exosim.tasks.radiometric.estimateSpectralBinning.EstimateSpectralBinning` task includes different methods to estimate the spectral binning,
which can be tuned in the channel description document.

Photometer
^^^^^^^^^^^

For a photometer, the description XML file should look like this:

.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>
        ...
    </channel>

In this case the radiometric table is estimated as the central wavelength of the photometer
with a bin width equal to the wavelength band.
Therefore the maximum and minimum wavelengths must be indicated along with the units in the `xml` file:

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

For a spectrometer, the description XML file should look like this:

.. code-block:: xml

    <channel> channel_name
        <type> spectrometer </type>

        ...
    </channel>

The wavelength grid can be estimated in 2 modes:

- `native` mode. If `targetR` is set to `native` the wavelength grid computed is the pixel level wavelength grid, where each bin is of the size of a pixel;
- `fixed R` mode. If `targetR` is set to a constant value, the wavelength grid is estimated using :func:`~exosim.utils.grids.wl_grid`.

The modes must be indicated in the configuration `xml` file along with the maximum and minimum wavelengths.
The `native` configuration will look like this

.. code-block:: xml

    <channel> channel_name
        <type> spectrometer </type>

        <spectral_binning_task> EstimateSpectralBinning </spectral_binning_task>
        <wl_min unit="micron"> 2 </wl_min>
        <wl_max unit="micron"> 6 </wl_max>
        <targetR> native </targetR>

        ...
    </channel>

The `fixed R` configuration will be like

.. code-block:: xml

    <channel> channel_name
        <type> spectrometer </type>
        <spectral_binning_task> EstimateSpectralBinning </spectral_binning_task>
        <wl_min unit="micron"> 2 </wl_min>
        <wl_max unit="micron"> 6 </wl_max>
        <targetR> 50 </targetR>

        ...
    </channel>
