.. role:: xml(code)
    :language: xml

.. _sources:

=======
Sources
=======

Inputs: describe the source
---------------------------

The target source is described in an `.xml` file under `sky`, using the keyword
`source`.

In the following example we simulate HD 209458:

.. code-block:: xml

        <source> HD 209458
        </source>

:class:`~exosim.tasks.load.load_options.LoadOptions` parses this file into a
dictionary, storing the star name under the keyword `value`.

.. code-block:: python

    loadOptions = LoadOptions()
    options = loadOptions(filename = 'path/to/file.xml')
    options['value'] = 'HD 209458'

The dictionary is then passed to the
:class:`~exosim.tasks.parse.parseSource.ParseSource` task, which returns the
source :class:`~exosim.models.signal.Sed`.

ExoSim supports three source types:

+ :ref:`planck`
+ :ref:`phoenix`
+ :ref:`custom`

The source type is set with:

.. code-block:: xml

        <source> HD 209458
            <source_type> planck </source_type>
        </source>

Depending on the type, :class:`~exosim.tasks.parse.parseSource.ParseSource` calls
a different :class:`~exosim.tasks.task.Task`.

.. _planck:

Planck star
^^^^^^^^^^^

For a `planck` source, ExoSim needs some extra parameters:

.. code-block:: xml

        <source> HD 209458
            <source_type> planck </source_type>
            <R unit="R_sun"> 1.18 </R>
            <T unit="K"> 6086 </T>
            <D unit="pc"> 47 </D>
        </source>

The Planck star SED is created by
:class:`~exosim.tasks.sed.createPlanckStar.CreatePlanckStar`. The star emission
is modelled with :class:`astropy.modeling.physical_models.BlackBody`; the
resulting SED is converted into :math:`W/m^2/sr/\mu m` and scaled by the solid
angle :math:`\pi \left( \frac{R}{D} \right)^2`.

For example:

.. code-block:: python

    from exosim.tasks.sed import CreatePlanckStar
    import astropy.units as u
    import numpy as np
    createPlanckStar = CreatePlanckStar()
    wl = np.linspace(0.5, 7.8, 10000) * u.um
    T = 6086 * u.K
    R = 1.18 * u.R_sun
    D = 47 * u.au
    sed = createPlanckStar(wavelength=wl, T=T, R=R, D=D)

    import matplotlib.pyplot as plt
    plt.plot(sed.spectral, sed.data[0,0])
    plt.ylabel(sed.data_units)
    plt.xlabel(sed.spectral_units)
    plt.show()

.. plot:: mpl_examples/createPlanckStar.py

.. _phoenix:

Phoenix star
^^^^^^^^^^^^

For a `phoenix` source, `ExoSim` uses the Phoenix spectral irradiances. You can
either point to a specific Phoenix file with the `filename` keyword:

.. code-block:: xml

        <source> HD 209458
            <source_type>phoenix </source_type>
            <filename> phoenix_filename </filename>

            <R unit="R_sun"> 1.18 </R>
            <D unit="pc"> 47 </D>
        </source>

or point `ExoSim` to a directory containing the whole Phoenix spectra library
and give it the parameters it needs to pick the best-matching spectrum:

.. code-block:: xml

        <source> HD 209458
            <source_type>phoenix </source_type>
            <path> phoenix_path </path>

            <R unit="R_sun"> 1.18 </R>
            <M unit="M_sun"> 1.17 </M>
            <T unit="K"> 6086 </T>
            <D unit="pc"> 47 </D>
            <z unit=""> 0.0 </z>
        </source>

The Phoenix star SED is created by
:class:`~exosim.tasks.sed.loadPhoenix.LoadPhoenix`. The Phoenix SED has units of
:math:`W/m^2/\mu m` and is scaled by :math:`\left( \frac{R}{D} \right)^2`.

.. _download_sed:

Download SED (online)
^^^^^^^^^^^^^^^^^^^^^

The :class:`~exosim.tasks.sed.download_sed.DownloadSed` task fetches stellar
SEDs from online model repositories. It supports two backends:

- **Göttingen PHOENIX-ACES**: high-resolution FITS files served by the
  Göttingen server (use ``model_name='phoenix-aces'``).
- **SVO models**: the Spanish Virtual Observatory provides a range of
  pre-computed model grids (e.g. ``bt-settl``, ``bt-settl-cifist``).

By default the task uses the SVO BT-Settl CIFIST models (``bt-settl-cifist``).
Files are downloaded with :func:`astropy.utils.data.download_file` and kept in
the Astropy download cache, so repeated calls do not re-download identical
files. The task logs the URL of each downloaded file (info level) and further
selection details in debug mode.

.. code-block:: python

        from exosim.tasks.sed import DownloadSed
        import astropy.units as u

        downloader = DownloadSed()
        sed = downloader(
                T=3016 * u.K,
                R=0.218 * u.R_sun,
                D=12.975 * u.pc,
                logg=4.5,
                model_name='bt-settl-cifist',
        )

Use the ``model_name`` parameter to select a different backend or SVO grid.

.. note::

    You can retrieve the list of models available on the Spanish Virtual
    Observatory dynamically using :func:`get_svo_models`::

        >>> from exosim.tasks.sed.download_sed import get_svo_models
        >>> get_svo_models()

    Or from the shell (prints one model per line)::

        python -c "from exosim.tasks.sed.download_sed import get_svo_models; print('\n'.join(get_svo_models()))"

    The ``DownloadSed`` task will attempt an SVO lookup for any ``model_name``
    that is not ``phoenix-aces``; if the requested model is not present on
    SVO a ``ValueError`` is raised explaining how to list available models.


.. _custom:

Custom star
^^^^^^^^^^^

For a `custom` source, `ExoSim` uses a custom
:class:`~exosim.tasks.task.Task` if `source_task` is present in the
configuration file (see :ref:`Custom Tasks`), and otherwise the default
:class:`~exosim.tasks.sed.loadCustom.LoadCustom`. The task loads a custom SED
from a file and scales it by the solid angle
:math:`\pi \left( \frac{R}{D} \right)^2`.

The default :class:`~exosim.tasks.sed.loadCustom.LoadCustom` needs the name of
the file that holds the :class:`~exosim.models.signal.Sed` to use.

.. code-block:: xml

        <source> HD 209458
            <source_type>custom </source_type>
            <filename> custom_sed_filename </filename>

            <R unit="R_sun"> 1.18 </R>
            <D unit="pc"> 47 </D>
        </source>

The custom SED file must be an `.ecsv` file with two columns, `Wavelength` and
`Sed`, where the SED has units of :math:`W/m^2/sr/\mu m`.

.. note::
    Depending on the computing power available, you can use a different number
    of wavelength and temporal points to simulate the source, trading speed for
    accuracy.


.. _sed_units_note:

.. note:: **Spectral Irradiance vs. Spectral Radiance**

    The distinction between **Phoenix SEDs** and the **Planck/Custom SEDs** lies in their physical definition:

    - **Phoenix SEDs** represent **spectral irradiance**, with units of :math:`W/m^2/\mu m`. They describe the flux received per unit area at a given distance.
    - **Planck and Custom SEDs** represent **spectral radiance**, with units of :math:`W/m^2/sr/\mu m`. These include the angular distribution of emitted radiation.

    To ensure consistency, ExoSim applies a scaling factor of :math:`\left( \frac{R}{D} \right)^2` to all SEDs. However, only Planck and Custom SEDs include an additional factor of :math:`\pi`, accounting for the assumption of isotropic emission over a hemisphere.


Load star parameters from online databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`ExoSim` can load star parameters from an online database. At the moment only
exodb_ is supported.

Instead of the stellar parameters, point to the online database:

.. code-block:: xml

        <source> HD 209458
            <source_type>phoenix </source_type>
            <path>/usr/local/project_data/sed </path>

            <online_database>
                <url>https://exodb.space/api/v1/star</url>
                <x-access-tokens> your_token_here </x-access-tokens>
            </online_database>

        </source>

Create your own source
^^^^^^^^^^^^^^^^^^^^^^

You can also build your own source with a customizable
:class:`~exosim.tasks.task.Task` (see :ref:`Custom Tasks`). The entry point is
:class:`~exosim.tasks.sed.createCustomSource.CreateCustomSource`.

As an example, here is the default
:class:`~exosim.tasks.sed.createCustomSource.CreateCustomSource` task. To enable
it, write the following in your `.xml` file:

.. code-block:: xml

    <source> HD 209458
        <source_task> CreateCustomSource </source_task>
        <R unit="R_sun"> 1.17967 </R>
        <T unit="K"> 6086 </T>
        <D unit="pc"> 47.4567 </D>
        <wl_min unit="um">0.5</wl_min>
        <wl_max unit="um">8</wl_max>
        <n_points >1000</n_points>
    </source>

The `source_task` keyword tells the code which
:class:`~exosim.tasks.task.Task` to use; here it is the default task. If you
write your own version, put the path to your script in this keyword. The default
:class:`~exosim.tasks.sed.createCustomSource.CreateCustomSource` task simply
creates a Planck star from the input parameters.

Outputs: prepare the sources
----------------------------

Single source
^^^^^^^^^^^^^

For the Planck case, the `.xml` file parsed by
:class:`~exosim.tasks.load.load_options.LoadOptions` returns a dictionary such
as

.. code-block:: python

    source_in = {
            'value': 'HD 209458',
            'source_type': 'planck',
            'R': 1.18 * u.R_sun,
            'D': 47 * u.pc,
            'T': 6086 * u.K,
            }

The wavelength grid to use comes from the :ref:`wavelength grid`.

We then use the :class:`~exosim.tasks.parse.parseSource.ParseSource` task to
produce the :class:`~exosim.models.signal.Sed`. The result is a dictionary keyed
by star name, with :class:`~exosim.models.signal.Sed` objects as values.

.. code-block:: python

    from exosim.tasks.parse import ParseSource
    import astropy.units as u
    import numpy as np
    parseSource = ParseSource()
    wl = np.linspace(0.5, 7.8, 10000) * u.um
    tt = np.linspace(0.5, 1, 10) * u.hr

    source_out = parseSource(parameters=source_in,
                             wavelength=wl,
                             time=tt)

    import matplotlib.pyplot as plt

    plt.plot(source_out['HD 209458'].spectral, source_out['HD 209458'].data[0,0])
    plt.ylabel(source_out['HD 209458'].data_units)
    plt.xlabel(source_out['HD 209458'].spectral_units)
    plt.show()

.. plot:: mpl_examples/parseSource.py

More sources
^^^^^^^^^^^^

With more than one source, the `.xml` file looks like this:

.. code-block:: xml

        <source> HD 209458
            <source_type> planck </source_type>
            <R unit="R_sun"> 1.18 </R>
            <T unit="K"> 6086 </T>
            <D unit="pc"> 47 </D>
        </source>

        <source> GJ 1214
            <source_type> planck </source_type>
            <R unit="R_sun"> 0.218 </R>
            <T unit="K"> 3026 </T>
            <D unit="pc"> 13 </D>
        </source>


The parsed dictionary is then:

.. code-block:: python

    from collections import OrderedDict
    sources_in = OrderedDict({'HD 209458': {'value': 'HD 209458',
                                        'source_type': 'planck',
                                        'R': 1.18 * u.R_sun,
                                        'D': 47 * u.pc,
                                        'T': 6086 * u.K,
                                        },
                                'GJ 1214': {'value': 'GJ 1214',
                                        'source_type': 'planck',
                                        'R': 0.218 * u.R_sun,
                                        'D': 13 * u.pc,
                                        'T': 3026 * u.K,
                                        },})

This dictionary is passed to
:class:`~exosim.tasks.parse.parseSource.ParseSources` to produce the
:class:`~exosim.models.signal.Sed`:

.. code-block:: python

    import astropy.units as u
    import numpy as np
    from exosim.tasks.parse import ParseSources

    wl = np.linspace(0.5, 7.8, 10000) * u.um
    tt = np.linspace(0.5, 1, 10) * u.hr

    parseSources = ParseSources()
    sources_out = parseSources(parameters=sources_in,
                               wavelength=wl,
                               time=tt)

    import matplotlib.pyplot as plt

    for key in sources_out.keys():
        plt.plot(sources_out[key].spectral, sources_out[key].data[0, 0], label=key)
    plt.ylabel(sources_out[key].data_units)
    plt.xlabel(sources_out[key].spectral_units)
    plt.legend()
    plt.show()

.. plot:: mpl_examples/parseSources.py

.. note::
    In this example the sources are superimposed. If the sources sit at
    different positions in the sky, see :ref:`pointing`, which explains how to
    simulate multiple sources together with the telescope pointing.

.. _sky from xml:

Parse from xml
^^^^^^^^^^^^^^

Assuming the wavelength and temporal grids have already been built (see
:ref:`wavelength grid` and :ref:`temporal grid`), you can parse the
configuration file into a dictionary of sources:

.. code-block:: python

    import exosim.tasks.parse as parse

    with output.use(append=True, cache=True) as out:

        out_sky = out.create_group('sky')

        parseSources = parse.ParseSources()
        sources = parseSources(parameters=mainConfig['sky']['source'],
                               wavelength=wl_grid,
                               time=time_grid,
                               output=out_sky)

This also assumes you have selected an output file (see :ref:`prepare output`)
and want to store the products in a dedicated sub-group.

.. _exodb: https://exodb.space/
