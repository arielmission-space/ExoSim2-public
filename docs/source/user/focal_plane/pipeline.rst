.. _focal plane recipe:

============================
Focal plane automatic recipe
============================

Appending all the scripts shown in the previous sections gives a full pipeline
for building the focal plane.

.. image:: _static/road_to_focal_plane.png
    :width: 600
    :align: center

These scripts are already collected in a ready-made pipeline, under
:py:mod:`~exosim.recipes`:

.. code-block:: python

    from exosim import recipes
    recipes.CreateFocalPlane(options_file='your_config_file.xml',
                                 output_file='output_file.h5')

:class:`~exosim.recipes.create_focal_plane.CreateFocalPlane` can also be run from
the console:

.. code-block:: console

    exosim-focalplane -c your_config_file.xml -o output_file.h5

Add ``-P`` to also run the
:class:`~exosim.plots.focalPlanePlotter.FocalPlanePlotter` (documented in
:ref:`focal plane plotter`):

.. code-block:: console

    exosim-focalplane -c your_config_file.xml -o output_file.h5 -P
