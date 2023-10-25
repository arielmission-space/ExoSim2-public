.. _ndrs recipe:

==============================
NDRs automatic Recipe
==============================

Al the steps needed to the production of sub-exposures are already collected in a pre-made pipeline.
This is under the :py:mod:`~exosim.recipes` of `ExoSim`.

.. code-block:: python

    from exosim import recipes
    recipes.CreateNDRs(input_file='./input_file.h5',
                       output_file='./output_file.h5',
                       options_file='your_config_file.xml')

The :class:`~exosim.recipes.createNDRs.CreateNDRs` can also be run from console as

.. code-block:: console

    exosim-ndrs -c your_config_file.xml -i input_file.h5 -o output_file.h5

or

.. code-block:: console

    exosim-ndrs -c your_config_file.xml -i input_file.h5 -o output_file.h5 -P

to also run ExoSim :class:`~exosim.plots.ndrsPlotter.NDRsPlotter`, which is documented in :ref:`ndrs plotter`.

The user can also set the chunk size (see :ref:`Instantaneous readout`) using

.. code-block:: console

    exosim-ndrs -c your_config_file.xml -i input_file.h5 -o output_file.h5 --chunk_size N

where `N` is the desired size expressed in Mbs.
