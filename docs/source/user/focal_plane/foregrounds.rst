.. _foregrounds:

===================================
Foregrounds
===================================

Defining the foregrounds
---------------------------

The foregrounds are to be listed in the `sky` `.xml` file, along with the `source`,
but under the keyword `foregrounds`.

.. code-block:: xml

    <foregrounds>
    </foregrounds>

Foregrounds are parsed as optical elements, like the optics in the payload, by :class:`~exosim.tasks.parse.parseOpticalElement.ParseOpticalElement`.
Multiple foregrounds form an optical path, and therefore are parsed by :class:`~exosim.tasks.parse.parsePath.ParsePath`.

.. code-block:: xml

    <foregrounds>
        <opticalElement> first_foreground_name
        </opticalElement>

        <opticalElement> second_foreground_name
        </opticalElement>
    </foregrounds>

By default, `ExoSim` supports user-defined foregrounds and zodiacal foregrounds.

.. _user foreground:

User-defined foreground
^^^^^^^^^^^^^^^^^^^^^^^^^

An example of a user-defined foreground is reported in the package `examples` directory:

.. code-block:: xml

    <foregrounds>
        <opticalElement> earthsky
            <task_model>LoadOpticalElement</task_model>
            <datafile>__ConfigPath__/foreground_file.ecsv</datafile>
            <wavelength_key>Wavelength</wavelength_key>
            <radiance_key>Radiance</radiance_key>
            <efficiency_key>Transmission</efficiency_key>
        </opticalElement>
    </foregrounds>

In this case `ExoSim` finds a foreground called `earthsky`, and uses the :class:`~exosim.tasks.task.Task` indicated in `task_model` to load it.
The indicated :class:`~exosim.tasks.load.load_optical_element.LoadOpticalElement` is the default :class:`~exosim.tasks.task.Task` included in `ExoSim` to load an optical element.
An optical element is defined by its radiance and efficiency as a function of the wavelength.
Hence, this default class looks into the indicated `datafile` to load the three quantities.
The data file should contain a table such that the three quantities are identified by the keys reported in the `.xml` description.
In this case the wavelength is reported under a column called `Wavelength`, the radiance under the column `Radiance`, and the efficiency under the column `Transmission`.

The user can write a custom :class:`~exosim.tasks.task.Task` to load or estimate the foreground differently.
This can be done by writing a new class inheriting from the default :class:`~exosim.tasks.load.load_optical_element.LoadOpticalElement`
and indicating in the `task_model` key the Python file containing such a new class.
The user shall only overwrite the `model` method in the new class.
The output of a custom model method, as indicated in the :class:`~exosim.tasks.load.load_optical_element.LoadOpticalElement` documentation,
shall be a :class:`~exosim.models.signal.Radiance` and a :class:`~exosim.models.signal.Dimensionless`.
The first contains the foreground radiance, and the second the foreground transmission.
The two classes should be binned to the general :ref:`wavelength grid` and the :ref:`temporal grid`.
Notice that the binning can be handled by the :func:`~exosim.models.signal.Signal.spectral_rebin` and :func:`~exosim.models.signal.Signal.temporal_rebin` methods of the :class:`~exosim.models.signal.Signal` class.
To learn more about customizing tasks, please refer to :ref:`Custom Tasks`.

.. caution::
    If the user doesn't include the `task_model` keyword in the optical element description,
    the default :class:`~exosim.tasks.load.load_optical_element.LoadOpticalElement` task is used.

Zodiacal foreground
^^^^^^^^^^^^^^^^^^^^^^^

If the foreground name is `zodi` or `zodiacal`, the code will parse the element using :class:`~exosim.tasks.foregrounds.estimateZodi.EstimateZodi` instead of :class:`~exosim.tasks.parse.parseOpticalElement.ParseOpticalElement`.

The zodiacal foreground radiance is estimated using a modified version of the JWST-MIRI Zodiacal model (Glasse et al., 2010),
scaled according to the target position in the sky and the Zodi model of Kelsall et al. (1998):

.. math::

    I_{zodi}(\lambda) = A \left( 3.5 \cdot 10^{-14} BB(\lambda, 5500 \, K) + 3.52 \cdot 10^{-8} BB(\lambda, 270 \, K) \right)

where :math:`BB(\lambda, T)` is the Planck black body law and :math:`A` is the fitted coefficient.

The user can either specify the coefficient to use, as in the example:

.. code-block:: xml

    <foregrounds>
        <opticalElement> zodiacal
            <zodiacal_factor>2.5</zodiacal_factor>
        </opticalElement>
    </foregrounds>

or can specify the coordinates in RA and Dec:

.. code-block:: xml

    <foregrounds>
        <opticalElement> zodiacal
            <coordinates> (ra, dec) </coordinates>
        </opticalElement>
    </foregrounds>

In this case the :math:`A` coefficient is selected from a precompiled grid.
The grid has been estimated by fitting our model with Kelsall et al. (1998) data.
A custom map can be provided to replace the default one, as long as it matches the format, by adding the keyword `zodi_map`.

.. _foreground propagation:

Foreground propagation
-------------------------

Each parsed foreground contains a radiance in units of :math:`W/m^2/\mu m/sr`, contained in a :class:`~exosim.models.signal.Radiance` class,
and a transmission, contained in a :class:`~exosim.models.signal.Dimensionless` class. Both classes are children of the :class:`~exosim.models.signal.Signal` class.

If more than one foreground is listed, the :class:`~exosim.tasks.parse.parsePath.ParsePath` class orders them in the same order used by the user in the `.xml` file
and propagates their light top to bottom. So the first element radiance is multiplied by the second element transmission, then the second element radiance is summed.
The obtained radiance is then multiplied by the third element transmission and the third element radiance is summed to the result.
The final transmission is the product of all transmissions. At the end of the pipeline we have a resulting radiance, still expressed in units of :math:`W/m^2/\mu m/sr`,
and still contained in a :class:`~exosim.models.signal.Radiance` class, which is the resulting radiance at the end of the chain, and a transmission that is the equivalent transmission of the full chain.
This can be considered as a foreground equivalent to the full foregrounds chain.

.. image:: _static/foregrounds.png
    :width: 600
    :align: center

The problem can be expressed with the recursive equation

.. math::

    I_{for, i+1} = I_{for, i+1} + I_{for, i} \cdot \Phi_{for, i+1}

.. math::

    \Phi_{for,i+1} = \Phi_{for,i+1} \cdot \Phi_{for,i}

Where :math:`I_{for, i}` is the radiance of :math:`i` foreground and :math:`\Phi_{for,i}` is its transmission.

.. note::

    Because of the way the light path is parsed, it is important to be careful with the order of writing for the optical elements. Optical elements further from the detector should be written first in the `.xml` file.

Following the process presented in :ref:`sky from xml`, we can parse the foregrounds as

.. code-block:: python

    import exosim.tasks.parse as parse

    with output.use(append=True, cache=True) as out:

        out_sky = out.create_group('sky')

        parsePath = parse.ParsePath()
        for_contrib = parsePath(parameters=mainConfig['sky']['foregrounds'],
                                wavelength=wl_grid, time=time_grid,
                                output=out_sky,
                                group_name='foregrounds')

In this case, thanks to the `group_name` keyword, the contributions are saved in a dedicated folder called `foregrounds`.

The `for_contrib` element shall be propagated now through the telescope. `ExoSim 2` handles this as the first optical element of the telescope optical chain.
