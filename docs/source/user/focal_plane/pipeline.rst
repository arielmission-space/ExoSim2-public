.. _focal plane recipe:

==============================
Focal plane automatic Recipe
==============================

By appending all the scripts shown previously we produce a pipeline for the production of the focal plane.

.. image:: _static/road_to_focal_plane.png
    :width: 600
    :align: center

The scripts are already collected in a pre-made pipeline.
This is under the :py:mod:`~exosim.recipes` of `ExoSim`.

.. code-block:: python

    from exosim import recipes
    recipes.CreateFocalPlane(options_file='your_config_file.xml',
                                 output_file='output_file.h5')

The :class:`~exosim.recipes.createFocalPlane.CreateFocalPlane` can also be run from console as

.. code-block:: console

    exosim-focalplane -c your_config_file.xml -o output_file.h5

or

.. code-block:: console

    exosim-focalplane -c your_config_file.xml -o output_file.h5 -P

to also run ExoSim :class:`~exosim.plots.focalPlanePlotter.FocalPlanePlotter`, which is documented in :ref:`focal plane plotter`.
