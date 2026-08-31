.. role:: xml(code)
    :language: xml

.. _general settings:

================
General settings
================

This section explains how to set up a simulation. To simulate a specific
observation you need to describe two things: the astronomical scene and the
instrument payload. In ExoSim you describe both in XML files.

The starting point is a **main configuration** `.xml` file. It acts as an index
that lists your settings and points to the other configuration files.
:class:`~exosim.tasks.load.load_options.LoadOptions` parses it into a dictionary.

Configuration path
------------------

The first thing to set is the configuration path:

.. code-block:: xml

    <root>
        <ConfigPath> path/to/your/configs
            <comment>Main directory for the configuration files</comment>
        </ConfigPath>
    </root>

This is the directory that holds all the data the simulation needs. Every time
you write ``__ConfigPath__`` in an `.xml` file, the parser
(:class:`~exosim.tasks.load.load_options.LoadOptions`) replaces it with this
path.

You can define your own substitution keywords in the same way:

.. code-block:: xml

    <root>
        <__CustomKeyword> your_value </__CustomKeyword>
        <another_keyword> __CustomKeyword__ </another_keyword>
    </root>

The code then replaces every ``__CustomKeyword__`` with `your_value`.

.. _wavelength grid:

Wavelength grid
---------------

Next, set up a wavelength grid. It is used to produce the emissions and signals
of every source in the simulation; these quantities are later rebinned onto the
instrument wavelength grid when the focal plane is built.

.. code-block:: xml

    <root>
        <wl_grid>
            <wl_min unit="micron">0.45</wl_min>
            <wl_max unit="micron">10.0</wl_max>
            <logbin_resolution unit="">6000</logbin_resolution>
        </wl_grid>
    </root>

This data is passed to :func:`~exosim.utils.grids.wl_grid`, which builds the
grid. The wavelength at the centre of each spectral bin is

.. math::

    \lambda_c = \frac{1}{2} (\lambda_j + \lambda_{j+1} )

where :math:`\lambda_j` is the wavelength at the bin edge, given by the recursive
relation below, and :math:`R` is the `logbin_resolution` set by the user:

.. math::

    \lambda_{j+1} = \lambda_{j} \left( 1 + \frac{1}{R} \right)

Given the minimum and maximum wavelengths, the number of bins is

.. math::

    n_{bins} = \frac{\log \left( \frac{\lambda_{max}}{\lambda_{min}} \right) } {\log \left( 1 + \frac{1}{R}\right)} + 1

In Python, the wavelength grid is parsed as follows:

.. code-block:: python

    import exosim.tasks.load as load
    import exosim.utils as utils

    loadOption = load.LoadOptions()
    mainConfig = loadOption(filename='your_config_file.xml')

    wl_grid = utils.grids.wl_grid(mainConfig['wl_grid']['wl_min'],
                                  mainConfig['wl_grid']['wl_max'],
                                  mainConfig['wl_grid']['logbin_resolution'])


.. _temporal grid:

Temporal grid
-------------

Now set the temporal grid.

.. code-block:: xml

    <root>
        <time_grid>
            <start_time unit="hour">0.0</start_time>
            <end_time unit="hour">10.0</end_time>
            <low_frequencies_resolution unit="second">60.0</low_frequencies_resolution>
        </time_grid>
    </root>

This is the focal-plane temporal grid, and it should only be used for
low-frequency variations. High-frequency dependencies are handled by a dedicated
pipeline, discussed later. This data is passed to
:func:`~exosim.utils.grids.time_grid`, which produces an evenly sampled grid.

.. code-block:: python

    import exosim.tasks.load as load
    import exosim.utils as utils

    loadOption = load.LoadOptions()
    mainConfig = loadOption(filename='your_config_file.xml')

    time_grid = utils.grids.time_grid(mainConfig['time_grid']['start_time'],
                                           mainConfig['time_grid']['end_time'],
                                           mainConfig['time_grid']['low_frequencies_resolution'])

If no `<low_frequencies_resolution>` is provided (or its value is `None`), the
function returns a single-element array containing only the `start_time`.

.. _configuration file:

Sky and payload
---------------

Finally, describe the astronomical scene and the instrument payload:

.. code-block:: xml

    <root>
        <sky>
            <config>__ConfigPath__/sky_example.xml</config>
        </sky>

        <payload>
            <config>__ConfigPath__/payload_example.xml</config>
        </payload>
    </root>

Here we use two separate `.xml` files for the sky and the payload, and
``__ConfigPath__`` to point at files inside the configuration directory. The
`config` keyword tells the parser
(:class:`~exosim.tasks.load.load_options.LoadOptions`) to load another `.xml`
file.

The `sky` root holds everything about the light sources and the sky foregrounds.
The `payload` root describes the instrument.

The `payload` root can contain both the common part of the instrument and the
channel-specific parts. In the example below, the payload has one common optical
path (the telescope) and two channels, each described in its own `.xml` file:

.. code-block:: xml

    <root>
        <Telescope> Common optics
            <config>__ConfigPath__/telescope.xml</config>
        </Telescope>

        <channel> channel 1
            <config>__ConfigPath__/channel_1.xml</config>
        </channel>
        <channel> channel 2
            <config>__ConfigPath__/channel_2.xml</config>
        </channel>
    </root>

.. _prepare output:

Preparing the output
--------------------

`ExoSim` can store all its products in an output file. At the time of writing,
only `.hdf5` files are supported.

Prepare the output with:

.. code-block:: python

    from exosim.output import SetOutput

    output = SetOutput('output_file.h5')

This sets `output_file.h5` as the output file. To use it, call
:func:`~exosim.output.setOutput.SetOutput.use`, which returns an
:class:`~exosim.output.output.Output` object:

.. code-block:: python

        with output.use(append=True, cache=True) as out:
            ...

With the file in use, create sub-groups with:

.. code-block:: python

    out_group = out.create_group('group name')

For the rest of the functionality, see the
:class:`~exosim.output.output.Output` class.
