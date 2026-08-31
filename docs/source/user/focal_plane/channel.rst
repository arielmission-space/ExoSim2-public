.. _channel:

=======
Channel
=======

Recall the road to the focal plane from :ref:`Focal plane creation`: after
parsing the sources, the foregrounds and the common optics, the simulation moves
inside each channel.

.. image:: _static/road_to_focal_plane.png
    :width: 600
    :align: center

Channels are described in the `.xml` file as:

.. code-block:: xml

    <channel> channel_name
        <type>spectrometer</type>
    </channel>

The channel `type` is either `spectrometer` or `photometer`. Channels are
handled by the :class:`~exosim.models.channel.Channel` class.

As mentioned in :ref:`channel optical path`, initialise
:class:`~exosim.models.channel.Channel` as:

.. code-block:: python

    from exosim.models import Channel

    with output.use(append=True, cache=True) as out:

        channel = Channel(parameters=payloadConfig['channel']['channel_name'],
                          wavelength=wl_grid, time=time_grid, output=out)

The output context manager and the `output` keyword are only needed if you want
to store the products.

Optical path
------------

The first thing to do with a channel is parse its optical path, as shown in
:ref:`channel optical path`. The channel optical path is described under the
channel section of the `.xml` configuration file:

.. code-block:: xml

    <channel> channel_name

        <optical_path>
            <opticalElement> first_optical_element
            </opticalElement>

            <opticalElement> second_optical_element
            </opticalElement>
        </optical_path>

    </channel>

The following line is appended to the previous script and assumes every earlier
step has run:

.. code-block:: python

        channel.parse_path(light_path=common_path)

This parses the channel-specific optical path and attaches it after the common
optical path stored in `common_path`.
:meth:`~exosim.models.channel.Channel.parse_path` populates the `path`
attribute. The resulting transmission and radiances are datacubes with the
shape of the :ref:`wavelength grid` by the :ref:`temporal grid`, wrapped in
:class:`~exosim.models.signal.Signal` objects (see :ref:`signal`). This format
allows wavelength- and time-dependent contributions.

.. _responsivity:

Estimate responsivity
---------------------

The channel responsivity is derived from the detector quantum efficiency (QE).
It is defined under the channel section of the `.xml` configuration file:

.. code-block:: xml

    <channel> channel_name

        <qe>
            <responsivity_task>LoadResponsivity</responsivity_task>
            <datafile>__ConfigPath__/qe.ecsv</datafile>
        </qe>

    </channel>

As described in :ref:`user foreground` and :ref:`optical element`, the
`responsivity_task` key names a customisable task that estimates the detector
responsivity, typically by loading the quantum efficiency and converting it. See
:ref:`Custom Tasks` to customise this.

The default task is
:class:`~exosim.tasks.instrument.load_responsivity.LoadResponsivity`. It loads
the `.csv` or `.ecsv` file given by `datafile`, which must contain:

- a first column named `Wavelength` (in units convertible to metres);
- one or more columns, named after the payload channels, holding dimensionless
  QE values as a function of wavelength.

The default implementation selects the column for the channel, rebins the QE
onto the simulation grids, and converts it to responsivity with:

.. math::

   R(\lambda) = \frac{QE(\lambda) \cdot \lambda}{h \cdot c}

The result is in :math:`\text{counts/Joule}`, returned as a datacube over the
:ref:`wavelength grid` and the :ref:`temporal grid`, wrapped in a
:class:`~exosim.models.signal.Signal` (see :ref:`signal`). Responsivity can
therefore vary with both wavelength and time.

In code, call:

.. code-block:: python

    channel.estimate_responsivity()

.. caution::

    If the `responsivity_task` keyword is omitted from the channel description,
    :func:`~exosim.models.channel.Channel.estimate_responsivity` uses the
    default task,
    :class:`~exosim.tasks.instrument.load_responsivity.LoadResponsivity`.


Propagate foreground
--------------------

Propagating the foregrounds means multiplying the result of the optical path by
the detector responsivity and the correct solid angle:

.. math::

    S_{path, i} = A_{pix} \cdot \Omega_{pix} \cdot \nu \cdot I_{path, i}

where :math:`A_{pix}` is the pixel area, :math:`\Omega_{pix}` is the solid
angle, and :math:`\nu` is the detector responsivity. The pixel area
:math:`A_{pix}` is computed from the detector section of the `.xml` file:

.. code-block:: xml

    <channel> channel_name
        <detector>
            <delta_pix unit="micron"> 18.0 </delta_pix>
        </detector>
    </channel>


.. image:: _static/detector_irradiation.png
    :width: 600
    :align: center


The solid angle :math:`\Omega_{pix}`, as discussed in
:ref:`supported optical elements`, depends on where the optical element sits
relative to the detector. Everything from the pixel field of view is multiplied
by the solid angle subtended by an on-axis elliptical aperture. The algorithm is
Equation 56 of John T. Conway, *Nuclear Instruments and Methods in Physics
Research Section A*, 614(1), 17–27, 2010
(https://doi.org/10.1016/j.nima.2009.11.075). It needs the f-numbers in the two
directions:

.. code-block:: xml

    <channel> channel_name
       <Fnum_x>15.5</Fnum_x>
       <Fnum_y>15.5</Fnum_y>
    </channel>

where `x` is the dispersion direction and `y` is the spatial direction. If only
`x` is given, the two are assumed equal, and the solid angle is estimated for a
circular aperture.

Light from the optics box is multiplied by :math:`\pi - \Omega_{pix}`; light
from the back of the detector (the `detector box`) is multiplied by :math:`\pi`.

The result is a dictionary holding the contributions of all the foregrounds (the
light paths), in :math:`counts / s / \mu m`.

To run this step, call
:func:`~exosim.models.channel.Channel.propagate_foreground`:

.. code-block:: python

        channel.propagate_foreground()

This updates the `path` attribute.

Propagate source
----------------

Propagating a source means multiplying the source SED by the instrument
efficiency to get the density signal, in :math:`counts / s / \mu m`.

For each source parsed as in :ref:`sky from xml`, the density signal is

.. math::

    S_{source, i} = A_{tel} \cdot \Phi_{tot} \cdot \nu \cdot I_{source, i}

where :math:`A_{tel}` is the telescope aperture, given in the `common optics`
description in the `.xml` file:

.. code-block:: xml

    <Telescope>
        <Atel unit="m**2">  0.63  </Atel>
        <optical_path>
            ...
        </optical_path>
    </Telescope>

:math:`\Phi_{tot}` is the final transmission of the optical chain and
:math:`\nu` is the detector responsivity.

To run this step, call
:func:`~exosim.models.channel.Channel.propagate_sources`:

.. code-block:: python

        channel.propagate_sources(sources = sources,
                                  Atel = payloadConfig['Telescope']['Atel'])

This updates the `sources` attribute.
