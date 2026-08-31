.. _Contributing:

===============
Developer guide
===============

This guide is for anyone extending `ExoSim 2` or contributing to it: the coding
and documentation conventions, how tasks and signals work internally, how to
configure a run, and how a release is cut.

.. grid:: 2 2 3 3
    :gutter: 3

    .. grid-item-card:: :octicon:`checklist;1.3em;sd-mr-1` Guidelines
        :link: guidelines
        :link-type: doc

        Coding and documentation conventions, testing, logging, versioning and
        the branch model.

    .. grid-item-card:: :octicon:`package;1.3em;sd-mr-1` Task structure
        :link: tasks
        :link-type: doc

        How an `ExoSim` task is built, run and logged.

    .. grid-item-card:: :octicon:`gear;1.3em;sd-mr-1` Custom tasks
        :link: custom_tasks
        :link-type: doc

        Replace a default task with your own version of the same step.

    .. grid-item-card:: :octicon:`pulse;1.3em;sd-mr-1` Signals
        :link: signals
        :link-type: doc

        The :class:`~exosim.models.signal.Signal` class, its derived classes and
        cached mode.

    .. grid-item-card:: :octicon:`sliders;1.3em;sd-mr-1` Run configuration
        :link: runConfig
        :link-type: doc

        Shared settings: parallel processing, chunk size and the random seed.

    .. grid-item-card:: :octicon:`tag;1.3em;sd-mr-1` Releasing
        :link: releasing
        :link-type: doc

        How a new version goes from a commit to a package on PyPI.

.. toctree::
   :hidden:
   :maxdepth: 1

   Guidelines <guidelines>
   Task structure <tasks>
   Custom tasks <custom_tasks>
   Signals <signals>
   Run configuration <runConfig>
   Releasing <releasing>
