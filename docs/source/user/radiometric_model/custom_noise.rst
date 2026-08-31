.. _custom_noise_radiometric:

============
Custom noise
============

Custom noise lets you add noise terms to the radiometric model beyond the
standard photon, dark current and read noise. `ExoSim` provides a flexible task
that accepts several input formats, so you can describe most instrument-specific
systematics without writing code.

Default task
------------

The default task
:class:`~exosim.tasks.radiometric.compute_custom_noise.ComputeCustomNoise`
computes the user-defined contributions. It accepts three input formats:

1. a single noise source, as a simple dictionary,
2. several noise sources, as multiple XML entries,
3. wavelength-dependent noise, from a spectral data table.

It returns a tuple:

1. **noise_table**: an :class:`astropy.table.QTable` with one column per named
   source (units ``hr**0.5``). It does not contain a total column.
2. **total_custom_noise**: the sources combined in quadrature
   (:class:`astropy.units.Quantity`, units ``hr**0.5``).

Configuration formats
---------------------

Single noise source
~~~~~~~~~~~~~~~~~~~~~

A single constant contribution:

.. code-block:: xml

    <radiometric>
        <custom_noise> thermal_drift
            <noise_level scale="1e-6">100 </noise_level>
        </custom_noise>
    </radiometric>

which corresponds to the internal structure

.. code-block:: python

    {
        "value": "thermal_drift",
        "noise_level": {
            "value": 100,
            "scale": 1e-6
        }
    }

Several noise sources
~~~~~~~~~~~~~~~~~~~~~~

Add one ``custom_noise`` entry per contribution. They are treated as independent
and combined in quadrature:

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

Wavelength-dependent noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For noise that varies with wavelength, point the ``custom_noise`` section at a
``data`` file:

.. code-block:: xml

    <radiometric>
        <custom_noise>
            <data>__ConfigPath__/custom_noise_data.ecsv</data>
            <name>spectral_systematics</name>
        </custom_noise>
    </radiometric>

The file holds a wavelength column and one or more noise columns:

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

Every column other than the wavelength is treated as a separate contribution and
combined in quadrature.

Enabling custom noise
---------------------

Custom noise is computed whenever one or more sources are present. A basic
configuration alongside the standard noise sources:

.. code-block:: xml

    <radiometric>
        <photon_noise> True </photon_noise>
        <dark_current> True </dark_current>
        <read_noise> True </read_noise>

        <custom_noise> 100
            <name>systematic_uncertainty</name>
        </custom_noise>
    </radiometric>

and a configuration with several constant sources plus a spectral one:

.. code-block:: xml

    <radiometric>
        <photon_noise> True </photon_noise>
        <dark_current> True </dark_current>
        <read_noise> True </read_noise>

        <custom_noise> 50
            <name>thermal_drift</name>
        </custom_noise>
        <custom_noise> 30
            <name>electronics_stability</name>
        </custom_noise>
        <custom_noise> 20
            <name>mechanical_vibration</name>
        </custom_noise>

        <custom_noise>
            <data>__ConfigPath__/wavelength_dependent_systematics.ecsv</data>
            <name>spectral_systematics</name>
        </custom_noise>
    </radiometric>

Column naming
~~~~~~~~~~~~~

The output column names are generated automatically:

- with a ``name`` field: ``{name}_noise``,
- if ``name`` already contains "noise": the name is used as is,
- with no name: ``custom_noise``,
- for spectral data: ``{name}_{column_name}_noise`` for each data column.

Where it runs in the pipeline
-----------------------------

When custom sources are present, the custom noise is computed automatically as
part of the noise sequence:

1. multiaccum gain calculation,
2. photon noise,
3. dark current noise,
4. read noise,
5. **custom noise** (this task),
6. total noise combination.

The custom contributions are added in quadrature with the other noise sources in
the total budget.

Custom task
-----------

To go further, inherit from
:class:`~exosim.tasks.radiometric.compute_custom_noise.ComputeCustomNoise`. The
custom task should take the same inputs as the default one (``wavelength``,
``description``), implement the ``model()`` method, and return the same output
(``noise_table``, ``total_noise``):

.. code-block:: python

    from exosim.tasks.radiometric.compute_custom_noise import ComputeCustomNoise

    class MyCustomNoiseTask(ComputeCustomNoise):

        def __init__(self):
            super().__init__()
            # add any extra parameters here

        def model(self, wavelength, description):
            noise_table, total_noise = super().model(wavelength, description)
            # apply your modifications here
            return noise_table, total_noise

Point to it in the configuration:

.. code-block:: xml

    <radiometric>
        <custom_noise_task>path.to.my_custom_noise_task.MyCustomNoiseTask</custom_noise_task>
        <custom_noise> 100
            <name>my_custom_source</name>
        </custom_noise>
    </radiometric>
