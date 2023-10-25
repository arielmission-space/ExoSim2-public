.. _subexposure recipe:

==============================
Sub-Exposures automatic Recipe
==============================

Al the steps needed to the production of sub-exposures are already collected in a pre-made pipeline.
This is under the :py:mod:`~exosim.recipes` of `ExoSim`.

.. code-block:: python

    from exosim import recipes
    recipes.CreateSubExposures(input_file='./input_file.h5',
                               output_file='./output_file.h5',
                               options_file='your_config_file.xml')

The :class:`~exosim.recipes.createSubExposures.CreateSubExposures` can also be run from console as

.. code-block:: console

    exosim-sub-exposures -c your_config_file.xml -i input_file.h5 -o output_file.h5

or

.. code-block:: console

    exosim-sub-exposures -c your_config_file.xml -i input_file.h5 -o output_file.h5 -P

to also run ExoSim :class:`~exosim.plots.subExposuresPlotter.SubExposuresPlotter`, which is documented in :ref:`sub-exposures plotter`.

The user can also set the chunk size (see :ref:`Instantaneous readout`) using

.. code-block:: console

    exosim-sub-exposures -c your_config_file.xml -i input_file.h5 -o output_file.h5 --chunk_size N

where `N` is the desired size expressed in Mbs.
