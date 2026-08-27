Custom Noise
============

Overview
--------

Custom noise allows users to include additional noise contributions in the radiometric model beyond the standard photon, dark current, and read noise components. ExoSim 2 provides a flexible task to compute custom noise contributions that can be specified in multiple formats to accommodate various noise modelling requirements.

Default Task
---------------------------------

The default task :class:`~exosim.tasks.radiometric.compute_custom_noise.ComputeCustomNoise` computes user-defined noise contributions for the radiometric noise budget. It supports three different input formats to accommodate various noise specification methods:

1. **Single noise source**: Simple dictionary for one contribution
2. **Multiple noise sources**: Using OrderedDict for several contributions
3. **Spectral data tables**: Wavelength-dependent noise from data files


The task returns a tuple containing:

1. **noise_table**: Astropy table.QTable with individual noise contributions

   - Individual noise columns for each named source (units: hr**0.5)
   - Does not include a total column (total is returned separately)

2. **total_custom_noise**: Combined noise from all sources using quadrature addition (Astropy units.Quantity with units hr**0.5)

Configuration Formats
----------------------

Format 1: Single Noise Source (Simple Dictionary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a single constant noise contribution:

.. code-block:: xml

    <radiometric>
        <custom_noise> thermal_drift
            <noise_level scale="1e-6">100 </noise_level>
        </custom_noise>
    </radiometric>

This corresponds to the following internal structure:

.. code-block:: python

    {
        "value": "thermal_drift",
        "noise_level": {
            "value": 100,
            "scale": 1e-6
        }
    }

Format 2: Multiple Noise Sources (Multiple XML Entries)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For multiple constant noise contributions, specify multiple custom_noise entries:

.. code-block:: xml

    <radiometric>
        <custom_noise> electronics
            <noise_level scale="1e-6">50 </noise_level>
        </custom_noise>
        <custom_noise> thermal
            <noise_level scale="1e-6">30 </noise_level>
        </custom_noise>
        <custom_noise> mechanical
            <noise_level scale="1e-6">25 </noise_level>
        </custom_noise>
    </radiometric>

This creates multiple independent noise sources that are combined in quadrature.

Format 3: Spectral Data (Wavelength-Dependent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For wavelength-dependent noise from data tables, the custom_noise section should contain a ``data`` field with spectral information:

.. code-block:: xml

    <radiometric>
        <custom_noise>
            <data>__ConfigPath__/custom_noise_data.ecsv</data>
            <name>spectral_systematics</name>
        </custom_noise>
    </radiometric>

The data file should contain wavelength and one or more noise columns:

.. code-block:: text

    # Spectral custom noise data
    # %ECSV 1.0
    # ---
    # datatype:
    # - {name: Wavelength, unit: micron, datatype: float64}
    # - {name: systematic_noise, unit: 1 / hr(1/2), datatype: float64}
    # - {name: detector_drift, unit: 1 / hr(1/2), datatype: float64}
    Wavelength systematic_noise detector_drift
    0.5 0.001 0.0005
    1.0 0.002 0.0008
    2.0 0.003 0.0012
    3.0 0.004 0.0015
    4.0 0.005 0.0018
    5.0 0.004 0.0015
    6.0 0.003 0.0012

All non-wavelength columns are treated as separate noise contributions and are combined in quadrature.

XML Configuration Examples
---------------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

To enable custom noise computation in your instrument configuration:

.. code-block:: xml

    <radiometric>
        <!-- Other noise sources -->
        <photon_noise> True </photon_noise>
        <dark_current> True </dark_current>
        <read_noise> True </read_noise>

        <!-- Custom noise contributions -->
        <custom_noise> 100
            <name>systematic_uncertainty</name>
        </custom_noise>
    </radiometric>

Advanced Configuration with Multiple Sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: xml

    <radiometric>
        <!-- Standard noise sources -->
        <photon_noise> True </photon_noise>
        <dark_current> True </dark_current>
        <read_noise> True </read_noise>

        <!-- Multiple custom noise sources -->
        <custom_noise> 50
            <name>thermal_drift</name>
        </custom_noise>
        <custom_noise> 30
            <name>electronics_stability</name>
        </custom_noise>
        <custom_noise> 20
            <name>mechanical_vibration</name>
        </custom_noise>

        <!-- Spectral custom noise from file -->
        <custom_noise>
            <data>__ConfigPath__/wavelength_dependent_systematics.ecsv</data>
            <name>spectral_systematics</name>
        </custom_noise>
    </radiometric>

Column Naming Convention
~~~~~~~~~~~~~~~~~~~~~~~~

The task automatically generates column names for the output table:

- If a ``name`` field is provided: ``{name}_noise``
- If ``name`` contains "noise": uses the name as-is
- If no name provided: defaults to ``custom_noise``
- For spectral data: ``{name}_{column_name}_noise`` for each data column

Integration with Radiometric Model
-----------------------------------

The custom noise computation is automatically integrated into the radiometric model pipeline when custom noise sources are specified. The task is called as part of the noise computation sequence:

1. Multiaccum gain calculation
2. Photon noise computation
3. Dark current noise computation
4. Read noise computation
5. **Custom noise computation** (this task)
6. Total noise combination

The custom noise contributions are automatically included in the total noise budget using quadrature addition with all other noise sources.

Custom Task Implementation
--------------------------

Users can implement custom noise tasks by inheriting from the :class:`~exosim.tasks.radiometric.compute_custom_noise.ComputeCustomNoise` base class. The custom task should:

1. Accept the same input parameters as the default task (wavelength, description)
2. Implement the ``model()`` method for the actual computation
3. Return the same output format (noise_table, total_noise)

Example custom task structure:

.. code-block:: python

    from exosim.tasks.radiometric.compute_custom_noise import ComputeCustomNoise

    class MyCustomNoiseTask(ComputeCustomNoise):

        def __init__(self):
            super().__init__()
            # Add any additional parameters if needed

        def model(self, wavelength, description):
            # Implement custom noise calculation logic
            # Must return (noise_table, total_noise)
            # where noise_table is QTable and total_noise is Quantity

            # Example: call parent method and modify results
            noise_table, total_noise = super().model(wavelength, description)

            # Add custom modifications here

            return noise_table, total_noise

To use your custom task, specify it in the XML configuration:

.. code-block:: xml

    <radiometric>
        <custom_noise_task>path.to.my_custom_noise_task.MyCustomNoiseTask</custom_noise_task>
        <!-- Custom noise specifications -->
        <custom_noise> 100
            <name>my_custom_source</name>
        </custom_noise>
    </radiometric>

The custom noise system provides maximum flexibility for modelling instrument-specific systematic effects while maintaining consistency with ExoSim 2's overall noise budget framework.
