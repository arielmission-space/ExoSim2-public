.. role:: xml(code)
    :language: xml

.. _general settings:

=======================
General settings
=======================

Here we will explore how to set up your machine for the simulation.
To simulate a secific observation you need to define both the astroscene and the instrument payload.
In ExoSim the user can set them both using xml file.

To start planing your simulation, you first need to setup a main configuration `.xml` file.
This is an index where your settings are to be listed.
This file will be parsed by :class:`~exosim.tasks.load.loadOptions.LoadOptions` into a dictionary,

Configuration path
--------------------

The first thing to set is the configuration path:

.. code-block:: xml

    <root>
        <ConfigPath> path/to/your/configs
            <comment>Main directory for the configuration files</comment>
        </ConfigPath>
    </root>

This is the path that contains all the data you will need for the simulation.
The `ConfigPath` content will be replaced by the parser (:class:`~exosim.tasks.load.loadOptions.LoadOptions`) in your string everytime you write ``__ConfigPath__`` in your `.xml`.

.. _wavelength grid:

Wavelength grid
------------------

Then we need to set up a wavelength grid. This will be used for the production of emissions and signals from all the source in the simulation.
These quantities will then be rebinned into the instrument wavelength grid when the focal plane is produced.

.. code-block:: xml

    <root>
        <wl_grid>
            <wl_min unit="micron">0.45</wl_min>
            <wl_max unit="micron">10.0</wl_max>
            <logbin_resolution unit="">6000</logbin_resolution>
        </wl_grid>
    </root>

This data will be fed into the :func:`~exosim.utils.grids.wl_grid` to produce the wavelength grid.
The wavelength at the center of the spectral bins is defined as

.. math::

    \lambda_c = \frac{1}{2} (\lambda_j + \lambda_{j+1} )

where :math:`\lambda_j` is the wavelength at the bin edge defined by the recursive relation,
and :math:`R` is the `logbin_resolution` defined by the user.

.. math::

    \lambda_{j+1} = \lambda_{j} \left( 1 + \frac{1}{R} \right)

And, given the maximum and minimum wavelengths, provided by the user, the number of bins is

.. math::

    n_{bins} = \frac{\log \left( \frac{\lambda_{max}}{\lambda_{min}} \right) } {\log \left( 1 + \frac{1}{R}\right)} + 1

The python code to parse the wavelength grid will be:

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
------------------

Now we need to set the temporal grid.

.. code-block:: xml

    <root>
        <time_grid>
            <start_time unit="hour">0.0</start_time>
            <end_time unit="hour">10.0</end_time>
            <low_frequencies_resolution unit="second">60.0</low_frequencies_resolution>
        </time_grid>
    </root>

This is going to be the focal plane temporal grid and should only use for low frequencies variation.
For high frequency dependency a dedicated pipeline will be discussed later.
This data will be fed into the :func:`~exosim.utils.grids.time_grid` to produce an equally sampled grid.

.. code-block:: python

    import exosim.tasks.load as load
    import exosim.utils as utils

    loadOption = load.LoadOptions()
    mainConfig = loadOption(filename='your_config_file.xml')

    time_grid = utils.grids.time_grid(mainConfig['time_grid']['start_time'],
                                           mainConfig['time_grid']['end_time'],
                                           mainConfig['time_grid']['low_frequencies_resolution'])

If no time details are provided a single time step is assumed.

.. _configuration file:

Sky and payload
------------------
Then we can describe the astroscene and the instrument payload by filling the keywords:

.. code-block:: xml

    <root>
        <sky>
            <config>__ConfigPath__/sky_example.xml</config>
        </sky>

        <payload>
            <config>__ConfigPath__/payload_example.xml</config>
        </payload>
    </root>

In this example we use two different `.xml` files to describe the sky and the payload.
We make use of the ``__ConfigPath__`` to point to file contained in the directory mentioned above.
The `config` keyword tells to the parser (:class:`~exosim.tasks.load.loadOptions.LoadOptions`) to look for another `.xml` file.

The `sky` root contains all the information about the light sources and the sky foregrounds.
The `payload` root contains the description of the instrument.

In particular, the `payload` root can contain both the common part of the instrument and the channel dedicated parts.
In the following example, the payload contains a common optics path, which is the telescope,
and two separated channels. Each of these part is described in a dedicated `.xml` configuration file.

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

Preparing output
=================

`ExoSim` can store all its product into an output file. At the moment of writing only `.hdf5` file are supported as output.


To prepare the output the following script can be used:

.. code-block:: python

    from exosim.output import SetOutput

    output = SetOutput('output_file.h5')

This will set `output_file.h5` as the output file.
To use the file the method :func:`~exosim.output.setOutput.SetOutput.use` can be use as it return an :class:`~exosim.output.output.Output` class:

.. code-block:: python

        with output.use(append=True, cache=True) as out:
            ...

With the file in use, to produce sub-folders in the file the user can use.

.. code-block:: python

    out_group = out.create_group('group name')

For other functionalities refer to the :class:`~exosim.output.output.Output` class.
