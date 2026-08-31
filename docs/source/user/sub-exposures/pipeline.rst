.. _subexposure recipe:

==============================
Sub-exposures automatic recipe
==============================

All the steps needed to produce the sub-exposures are collected in a ready-made
pipeline, under :py:mod:`~exosim.recipes`:

.. code-block:: python

    from exosim import recipes
    recipes.CreateSubExposures(input_file='./input_file.h5',
                               output_file='./output_file.h5',
                               options_file='your_config_file.xml')

:class:`~exosim.recipes.createSubExposures.CreateSubExposures` can also be run
from the console:

.. code-block:: console

    exosim-sub-exposures -c your_config_file.xml -i input_file.h5 -o output_file.h5

Add ``-P`` to also run
:class:`~exosim.plots.subExposuresPlotter.SubExposuresPlotter` (documented in
:ref:`sub-exposures plotter`):

.. code-block:: console

    exosim-sub-exposures -c your_config_file.xml -i input_file.h5 -o output_file.h5 -P

You can also set the chunk size (see :ref:`Instantaneous readout`):

.. code-block:: console

    exosim-sub-exposures -c your_config_file.xml -i input_file.h5 -o output_file.h5 --chunk_size N

where `N` is the size in MB.
