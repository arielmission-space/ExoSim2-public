.. _foregrounds:

===========
Foregrounds
===========

Defining the foregrounds
------------------------

The foregrounds are listed in the `sky` `.xml` file, next to the `source`, under
the keyword `foregrounds`:

.. code-block:: xml

    <foregrounds>
    </foregrounds>

Foregrounds are parsed as optical elements, just like the optics in the payload,
by :class:`~exosim.tasks.parse.parseOpticalElement.ParseOpticalElement`. Several
foregrounds together form an optical path, so they are parsed by
:class:`~exosim.tasks.parse.parsePath.ParsePath`.

.. code-block:: xml

    <foregrounds>
        <opticalElement> first_foreground_name
        </opticalElement>

        <opticalElement> second_foreground_name
        </opticalElement>
    </foregrounds>

Out of the box, `ExoSim` supports user-defined foregrounds and the zodiacal
foreground.

.. _user foreground:

User-defined foreground
^^^^^^^^^^^^^^^^^^^^^^^^

The package `examples` directory contains a user-defined foreground:

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

Here `ExoSim` finds a foreground called `earthsky` and loads it with the
:class:`~exosim.tasks.task.Task` named in `task_model`. The default task,
:class:`~exosim.tasks.load.load_optical_element.LoadOpticalElement`, loads an
optical element from a data file. An optical element is defined by its radiance
and its efficiency as functions of wavelength, so the task reads three columns
from `datafile`, identified by the keys in the `.xml` description: `Wavelength`
for the wavelength, `Radiance` for the radiance, and `Transmission` for the
efficiency.

You can write a custom :class:`~exosim.tasks.task.Task` to load or estimate the
foreground differently. Write a new class that inherits from the default
:class:`~exosim.tasks.load.load_optical_element.LoadOpticalElement`, and point
the `task_model` key at the Python file that contains it. You only need to
override the `model` method. As described in the
:class:`~exosim.tasks.load.load_optical_element.LoadOpticalElement`
documentation, `model` must return a :class:`~exosim.models.signal.Radiance`
(the foreground radiance) and a :class:`~exosim.models.signal.Dimensionless`
(the foreground transmission). Both must be binned to the :ref:`wavelength grid`
and the :ref:`temporal grid`; the
:func:`~exosim.models.signal.Signal.spectral_rebin` and
:func:`~exosim.models.signal.Signal.temporal_rebin` methods of
:class:`~exosim.models.signal.Signal` can do this for you. See :ref:`Custom
Tasks` for more on customizing tasks.

.. caution::
    If you omit the `task_model` keyword from an optical-element description,
    the default
    :class:`~exosim.tasks.load.load_optical_element.LoadOpticalElement` task is
    used.

Zodiacal foreground
^^^^^^^^^^^^^^^^^^^^

If the foreground is named `zodi` or `zodiacal`, the code parses it with
:class:`~exosim.tasks.foregrounds.estimateZodi.EstimateZodi` instead of
:class:`~exosim.tasks.parse.parseOpticalElement.ParseOpticalElement`.

The zodiacal radiance is estimated from a modified version of the JWST-MIRI
zodiacal model (Glasse et al., 2010), scaled to the target position in the sky
using the zodiacal model of Kelsall et al. (1998):

.. math::

    I_{zodi}(\lambda) = A \left( 3.5 \cdot 10^{-14} BB(\lambda, 5500 \, K) + 3.52 \cdot 10^{-8} BB(\lambda, 270 \, K) \right)

where :math:`BB(\lambda, T)` is the Planck black-body law and :math:`A` is the
fitted coefficient.

You can either set the coefficient explicitly,

.. code-block:: xml

    <foregrounds>
        <opticalElement> zodiacal
            <zodiacal_factor>2.5</zodiacal_factor>
        </opticalElement>
    </foregrounds>

or give the target coordinates in RA and Dec:

.. code-block:: xml

    <foregrounds>
        <opticalElement> zodiacal
            <coordinates> (ra, dec) </coordinates>
        </opticalElement>
    </foregrounds>

In the second case :math:`A` is read from a pre-computed grid, obtained by
fitting our model to the Kelsall et al. (1998) data. You can replace the default
grid with your own by adding the `zodi_map` keyword, as long as it matches the
expected format.

.. _foreground propagation:

Foreground propagation
----------------------

Each parsed foreground carries a radiance in units of :math:`W/m^2/\mu m/sr`, in
a :class:`~exosim.models.signal.Radiance` object, and a transmission, in a
:class:`~exosim.models.signal.Dimensionless` object. Both are subclasses of
:class:`~exosim.models.signal.Signal`.

When several foregrounds are listed,
:class:`~exosim.tasks.parse.parsePath.ParsePath` keeps them in the order they
appear in the `.xml` file and propagates the light from top to bottom: the
radiance of the first element is multiplied by the transmission of the second,
then the radiance of the second is added; that result is multiplied by the
transmission of the third, and the radiance of the third is added; and so on.
The final transmission is the product of all the individual transmissions. The
result is a single radiance (still in :math:`W/m^2/\mu m/sr`, still a
:class:`~exosim.models.signal.Radiance`) and a single transmission, together
equivalent to the whole foreground chain.

.. image:: _static/foregrounds.png
    :width: 600
    :align: center

This is the recursive relation:

.. math::

    I_{for, i+1} = I_{for, i+1} + I_{for, i} \cdot \Phi_{for, i+1}

.. math::

    \Phi_{for,i+1} = \Phi_{for,i+1} \cdot \Phi_{for,i}

where :math:`I_{for, i}` is the radiance of foreground :math:`i` and
:math:`\Phi_{for,i}` is its transmission.

.. note::

    Because of how the light path is parsed, the order of the optical elements
    matters. Elements further from the detector must be written first in the
    `.xml` file.

Following the process shown in :ref:`sky from xml`, the foregrounds are parsed
as:

.. code-block:: python

    import exosim.tasks.parse as parse

    with output.use(append=True, cache=True) as out:

        out_sky = out.create_group('sky')

        parsePath = parse.ParsePath()
        for_contrib = parsePath(parameters=mainConfig['sky']['foregrounds'],
                                wavelength=wl_grid, time=time_grid,
                                output=out_sky,
                                group_name='foregrounds')

The `group_name` keyword stores the contributions in a dedicated group called
`foregrounds`.

The `for_contrib` element is then propagated through the telescope: `ExoSim 2`
treats it as the first optical element of the telescope optical chain.
