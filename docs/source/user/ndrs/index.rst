.. _ndrs creation:

====
NDRs
====

Finally, we can produce the NDRs.

The flow is summarised in the figure below, where the tasks are grouped into
blocks to make it easier to follow.

.. image:: _static/ndrs.png
    :align: center

The following pages discuss each step in turn.

.. note::
    All of these steps operate on cached data. The result of each step is not a
    new dataset but the input dataset with its values overwritten.

The NDRs creation is automated by a recipe,
:class:`~exosim.recipes.createNDRs.CreateNDRs`.

`ExoSim` also has a dedicated plotter,
:class:`~exosim.plots.ndrsPlotter.NDRsPlotter`, described in :ref:`ndrs plotter`.

.. toctree::
    :maxdepth: 1

    Dark current <darkcurrent>
    Shot noise <shot_noise>
    Cosmic rays <cosmic_rays>
    Accumulate NDRs <accumulate>
    KTC noise <reset_bias>
    Dead pixels <dead_pixels>
    Pixel non-linearity and saturation <pixel_non_linearity>
    Gain drift <gain_drift>
    Read noise <readnoise>
    Analog-to-digital conversion <analogtodigtital>
    Merge NDRs and results <results>
    Automatic recipe <pipeline>
