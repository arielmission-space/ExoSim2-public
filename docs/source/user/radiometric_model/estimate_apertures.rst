.. _estimate apertures:

==================
Estimate apertures
==================

Once the wavelength table is ready, the next step is to estimate the aperture
sizes for the photometry. By default this is done by
:class:`~exosim.tasks.radiometric.estimate_apertures.EstimateApertures`, which
has many options; you can also replace it with a custom task (see
:ref:`Custom Tasks`). Name the aperture task under the `radiometric` keyword:

.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>

        <radiometric>
            <aperture_photometry>
                <apertures_task> EstimateApertures </apertures_task>
            </aperture_photometry>
            ...
        </radiometric>

    </channel>

Inside :class:`~exosim.recipes.radiometric_model.RadiometricModel` this task is
handled by
:func:`~exosim.recipes.radiometric_model.RadiometricModel.compute_apertures`. To
use the default task in a script on a channel:

.. code-block:: python

    import exosim.tasks.radiometric as radiometric

    estimateApertures = radiometric.EstimateApertures()
    aperture_table = estimateApertures(table=table,
                                       focal_plane=focal_plane,
                                       description=description['radiometric']['aperture_photometry'],
                                       wl_grid=wl_grid)

where `table` is the wavelength radiometric table, `focal_plane` is the channel
source focal-plane array, `description` is the dictionary with the
aperture-photometry information from the `xml` file, and `wl_grid` is the
focal-plane wavelength grid.

.. caution::
    If you omit the `apertures_task` keyword from the channel description, the
    default
    :class:`~exosim.tasks.radiometric.estimate_apertures.EstimateApertures` task
    is used. See :ref:`Custom Tasks` to develop a custom
    :class:`~exosim.tasks.task.Task`.

This :class:`~exosim.tasks.task.Task` returns a
:class:`~astropy.table.QTable` with the centres, sizes and shapes of the channel
apertures.

====================    ====================================================
keyword                 content
====================    ====================================================
spectral_center         centre of the aperture in the spectral direction
spectral_size           size of the aperture in the spectral direction
spatial_center          centre of the aperture in the spatial direction
spatial_size            size of the aperture in the spatial direction
aperture_shape          shape of the aperture (rectangular or elliptical)
====================    ====================================================

The rest of this page covers the :class:`~exosim.tasks.task.Task` options.

Spectral and spatial modes
==========================

The spectral and spatial modes control how the focal-plane data are summed in
the two directions. Combining them gives you full control over the summing
method.

Spectral modes
--------------

The spectral mode is set with the `spectral_mode` keyword.

Rows
^^^^

With `spectral_mode` set to `row`, the aperture in the spectral direction is
sized to sum the full pixel row.

.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>

        <radiometric>
            <aperture_photometry>
                <apertures_task> EstimateApertures </apertures_task>
                <spectral_mode> row </spectral_mode>
            </aperture_photometry>
            ...
        </radiometric>

    </channel>


.. image:: _static/aperture_row.png
    :align: center



Wavelength solution
^^^^^^^^^^^^^^^^^^^^

With `spectral_mode` set to `wl_solution`, the aperture in the spectral
direction is sized from the spectral bin width in the radiometric table (see
:ref:`wavelength bin`).

.. code-block:: xml

    <channel> channel_name
        <type> spectrometer </type>

        <radiometric>
            <aperture_photometry>
                <apertures_task> EstimateApertures </apertures_task>
                <spectral_mode> wl_solution </spectral_mode>
            </aperture_photometry>
            ...
        </radiometric>

    </channel>

.. image:: _static/aperture_wl_solution.png
    :align: center

Spatial modes
-------------

Only one spatial mode is available. With `spatial_mode` set to `column`, the
aperture in the spatial direction is sized to sum the full pixel column.


.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>

        <radiometric>
            <aperture_photometry>
                <apertures_task> EstimateApertures </apertures_task>
                <spatial_mode> column </spatial_mode>
            </aperture_photometry>
            ...
        </radiometric>

    </channel>

.. image:: _static/aperture_column.png
    :align: center

Use-case examples
-----------------

To sum all the pixel values for a photometer, you can either use the automatic
`full` mode (shown below),

.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>

        <radiometric>
            <aperture_photometry>
                <apertures_task> EstimateApertures </apertures_task>
                <auto_mode> full </auto_mode>
            </aperture_photometry>
            ...
        </radiometric>

    </channel>

or set the mode in each direction and let
:class:`~exosim.tasks.radiometric.estimate_apertures.EstimateApertures` sum all
the rows and columns:

.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>

        <radiometric>
            <aperture_photometry>
                <apertures_task> EstimateApertures </apertures_task>
                <spectral_mode> row </spectral_mode>
                <spatial_mode> column </spatial_mode>
            </aperture_photometry>
            ...
        </radiometric>

    </channel>

To sum all the pixels along the columns of a spectral bin for a spectrometer,
combine the `column` and `wl_solution` modes:

.. code-block:: xml

    <channel> channel_name
        <type> spectrometer </type>

        <radiometric>
            <aperture_photometry>
                <apertures_task> EstimateApertures </apertures_task>
                <spectral_mode> wl_solution </spectral_mode>
                <spatial_mode> column </spatial_mode>
            </aperture_photometry>
            ...
        </radiometric>

    </channel>

Automatic modes
===============

:class:`~exosim.tasks.radiometric.estimate_apertures.EstimateApertures` also has
automatic modes that search for the best aperture.

Elliptical apertures
--------------------

Set with `auto_mode` equal to `elliptical`. This runs
:func:`~exosim.utils.aperture.find_elliptical_aperture`, which looks for the
elliptical aperture on the focal plane that encloses at least the encircled
energy set by the `EnE` keyword, with the fewest pixels.

.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>

        <radiometric>
            <aperture_photometry>
                <apertures_task> EstimateApertures </apertures_task>
                <auto_mode> elliptical </auto_mode>
                <EnE> 0.91 </EnE>
            </aperture_photometry>
            ...
        </radiometric>

    </channel>

.. image:: _static/aperture_elliptical.png
    :align: center

Rectangular apertures
---------------------

Set with `auto_mode` equal to `rectangular`. This runs
:func:`~exosim.utils.aperture.find_rectangular_aperture`, which looks for the
rectangular aperture on the focal plane that encloses at least the encircled
energy set by the `EnE` keyword, with the fewest pixels.


.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>

        <radiometric>
            <aperture_photometry>
                <apertures_task> EstimateApertures </apertures_task>
                <auto_mode> rectangular </auto_mode>
                <EnE> 0.91 </EnE>
            </aperture_photometry>
            ...
        </radiometric>

    </channel>

.. image:: _static/aperture_rectangular.png
    :align: center

Spectral-bin apertures
----------------------

Set with `auto_mode` equal to `bin`. This runs
:func:`~exosim.utils.aperture.find_bin_aperture`, which looks for a rectangular
aperture with a fixed spectral size that encloses at least the encircled energy
set by the `EnE` keyword, with the fewest pixels.


.. code-block:: xml

    <channel> channel_name
        <type> spectrometer </type>

        <radiometric>
            <aperture_photometry>
                <apertures_task> EstimateApertures </apertures_task>
                <auto_mode> bin </auto_mode>
                <EnE> 0.91 </EnE>
            </aperture_photometry>
            ...
        </radiometric>

    </channel>

.. image:: _static/aperture_autobin.png
    :align: center

Full aperture
-------------

Set with `auto_mode` equal to `full`. This creates a rectangular aperture the
size of the whole focal plane.

.. code-block:: xml

    <channel> channel_name
        <type> photometer </type>

        <radiometric>
            <aperture_photometry>
                <apertures_task> EstimateApertures </apertures_task>
                <auto_mode> full </auto_mode>
            </aperture_photometry>
            ...
        </radiometric>

    </channel>
