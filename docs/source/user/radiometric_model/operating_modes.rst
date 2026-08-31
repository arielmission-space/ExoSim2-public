.. _recipe:
.. _modes:

=================================
Radiometric model operating modes
=================================

The radiometric model has three operating modes, selected automatically from the
configuration and the input files. This is handled by the
:class:`~exosim.recipes.radiometric_model.RadiometricModel` class.

Usage
-----

.. code-block:: python

    from exosim import recipes
    recipes.RadiometricModel(options_file='your_config_file.xml',
                             output_file='output_file.h5')

:class:`~exosim.recipes.radiometric_model.RadiometricModel` can also be run from
the console:

.. code-block:: console

    exosim-radiometric -c your_config_file.xml -o output_file.h5

Add ``-P`` to also run
:class:`~exosim.plots.radiometric_plotter.RadiometricPlotter` (documented in
:ref:`radiometric plotter`):

.. code-block:: console

    exosim-radiometric -c your_config_file.xml -o output_file.h5 -P

The radiometric pipeline stores its products in the output file by default, with
:func:`exosim.recipes.radiometric_model.RadiometricModel.write`. If the output
format is the default HDF5_, see :ref:`loadHDF5` in the :ref:`FAQs` section for
how to read the data, and :ref:`load signal table` in particular for how to cast
the focal plane into a :class:`~exosim.models.signal.Signal`.

.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/

How the mode is selected
------------------------

The mode is chosen from the configuration and the input files, in this order:

1. **Target-list mode**: when ``targetlist_filepath`` is set in the sky
   configuration.
2. **Existing focal-plane mode**: when the output file already exists and
   contains focal-plane data, and no target list is set.
3. **Single-source mode**: when neither of the above applies, so a new focal
   plane is built for a single source.

.. _target_list_mode:

Target-list mode
----------------

When ``targetlist_filepath`` is set in the sky configuration, the radiometric
model processes several targets in one run. This is the mode for survey planning
and for comparing targets.

For each target the pipeline:

1. reads the target parameters from the CSV file,
2. builds a focal plane for that target,
3. computes its apertures, signals and noise,
4. stores its radiometric table and plots.

Configuration
~~~~~~~~~~~~~~

Point the sky section at a target list. By default it is loaded by
:class:`~exosim.tasks.load.load_source_list.LoadSourceList`:

.. code-block:: xml

    <sky>
        <source> targetlist
            <targetlist_filepath>targets.csv</targetlist_filepath>
            <source_type>planck</source_type>
            <column_mapping>
                <name>star name</name>
                <T>star Teff [K]</T>
                <R>star R [R_sun]</R>
                <D>star D [pc]</D>
                <M>star M [M_sun]</M>
            </column_mapping>
        </source>
    </sky>

Target-list format
~~~~~~~~~~~~~~~~~~~

The target list is a CSV file, for example:

.. csv-table:: Example target list
   :header: "star name", "star Teff [K]", "star R [R_sun]", "star D [pc]", "star M [M_sun]"
   :widths: 20, 15, 15, 15, 15

   "GJ 1214", "3250", "0.211", "14.55", "0.176"
   "HD 209458", "6086", "1.18", "47.5", "1.175"
   "HD 219134", "4699", "0.778", "6.5", "0.778"

The column names in the file are mapped to the physical parameters through the
``column_mapping`` section of the configuration.

Output structure
~~~~~~~~~~~~~~~~~

In target-list mode the output file is organised per target:

.. code-block::

    output_file.h5
    ├── targets/
    │   ├── GJ_1214/
    │   │   ├── channels/
    │   │   │   ├── Photometer/
    │   │   │   └── Spectrometer/
    │   │   └── configuration/
    │   ├── HD_209458/
    │   │   ├── channels/
    │   │   └── configuration/
    │   └── HD_219134/
    │       ├── channels/
    │       └── configuration/
    ├── radiometric/
    │   ├── apertures/
    │   ├── GJ_1214/
    │   ├── HD_209458/
    │   └── HD_219134/
    └── ...

One CSV file is also written per target:

- ``GJ_1214_radiometric_table.ecsv``
- ``HD_209458_radiometric_table.ecsv``
- ``HD_219134_radiometric_table.ecsv``

Example
~~~~~~~

.. code-block:: python

    from exosim.recipes import RadiometricModel

    rm = RadiometricModel(
        options_file='config_with_targets.xml',
        output_file='target_list_results.h5'
    )
    rm.run()

.. _existing_fp:

Existing focal-plane mode
-------------------------

When a focal-plane file already exists and no target list is set, the radiometric
model loads that focal plane and computes the radiometric estimates directly.

.. image:: _static/radiometric-full.png
    :align: center

Starting from the existing focal plane, the pipeline runs the signal estimation:

1. build the wavelength table (see :ref:`wavelength bin`);
2. estimate the aperture sizes and pixel counts (see :ref:`estimate apertures`);
3. estimate the sub-foreground signals, if any (see :ref:`estimate signals`);
4. estimate the total foreground signals (see :ref:`estimate signals`);
5. estimate the source signals in the apertures (see :ref:`estimate signals`);
6. estimate the saturation times (see :ref:`saturation_time`);

then the noise estimation:

1. compute the multiaccum factors (see :ref:`multiaccum`);
2. estimate the photon noise (see :ref:`photon noise`);
3. compute the detector noise sources;
4. compute the total noise (see :ref:`total noise`).

The radiometric table is then written to the output file.

.. code-block:: python

    # 'existing_focal_plane.h5' already contains focal-plane data
    rm = recipes.RadiometricModel(
        options_file='config.xml',
        output_file='existing_focal_plane.h5'
    )

.. _non_existing_fp:

Single-source mode
------------------

If no focal plane is available as input,
:class:`~exosim.recipes.radiometric_model.RadiometricModel` builds one.

.. image:: _static/non_existing_fp.png
    :align: center

Following the figure, the pipeline loads the input configuration file, removes
the temporal dimension (the radiometric model does not need it), isolates each
optical element so it can estimate its contribution, and builds the focal plane
with the :ref:`focal plane recipe`. From there, the :ref:`existing_fp` pipeline
runs on the new focal plane.

Output files
------------

What the model writes depends on the mode:

- **Target-list mode**: a new HDF5 file with a radiometric table for every
  target.
- **Existing focal-plane mode**: the input HDF5 file, with the radiometric
  information added next to the focal-plane data.
- **Single-source mode**: the radiometric results in the output file you
  specify.

Every output holds the full radiometric table, with the wavelength grid, the
signal estimates and the noise for each channel.
