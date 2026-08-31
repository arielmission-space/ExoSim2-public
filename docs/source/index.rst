.. _index:

======================
ExoSim 2 documentation
======================

**ExoSim 2** is an end-to-end, time-domain simulator for exoplanet observations
from space telescopes, ground-based observatories, and sub-orbital platforms.
It models the full observation chain, from the astronomical sources through the
instrument optics, the dispersers, the detector arrays and the readout
electronics, reproducing both the astrophysical signal and the instrumental
systematics. It is written in Python 3 with a modular, object-oriented design, so
almost every component can be swapped for a custom implementation without
rewriting the rest of the pipeline.

This guide walks through the simulation steps with worked examples and explains
how to configure ExoSim 2 for the instrument you want to model.

.. grid:: 2 2 2 2
    :gutter: 3

    .. grid-item-card:: Installation
        :text-align: center
        :shadow: md
        :link: installation
        :link-type: doc

        :octicon:`download;2.5em;sd-text-primary`
        ^^^
        New to ExoSim? Start here.

    .. grid-item-card:: User guide
        :text-align: center
        :shadow: md
        :link: user/index
        :link-type: doc

        :octicon:`book;2.5em;sd-text-primary`
        ^^^
        How to run ExoSim, one pipeline at a time.

    .. grid-item-card:: Developer guide
        :text-align: center
        :shadow: md
        :link: contributing/index
        :link-type: doc

        :octicon:`tools;2.5em;sd-text-primary`
        ^^^
        How to extend or customise ExoSim.

    .. grid-item-card:: API guide
        :text-align: center
        :shadow: md
        :link: api/exosim/index
        :link-type: doc

        :octicon:`code-square;2.5em;sd-text-primary`
        ^^^
        The complete API reference.


.. grid:: 3 3 3 3
    :gutter: 3

    .. grid-item-card:: FAQs
        :text-align: center
        :shadow: md
        :link: FAQs
        :link-type: doc

        :octicon:`question;2em;sd-text-secondary`
        ^^^
        Frequently asked questions

    .. grid-item-card:: License
        :text-align: center
        :shadow: md
        :link: license
        :link-type: doc

        :octicon:`law;2em;sd-text-secondary`
        ^^^
        The BSD 3-Clause licence

    .. grid-item-card:: Changelog
        :text-align: center
        :shadow: md
        :link: CHANGELOG
        :link-type: doc

        :octicon:`history;2em;sd-text-secondary`
        ^^^
        What changed in each release

.. toctree::
    :hidden:
    :maxdepth: 1

    Installation <installation>
    User guide <user/index>
    Developer guide <contributing/index>
    API guide <api/index>
    FAQs <FAQs>
    License <license>
    Changelog <CHANGELOG>

.. note::

    Found a mistake or something unclear? Please contact the developers, or open
    an issue on the `GitHub repository
    <https://github.com/arielmission-space/ExoSim2-public/issues>`__.

What can ExoSim 2 do?
=====================

- **Radiometric modelling**: fast estimates of signal and noise budgets,
  aperture photometry, saturation analysis, and multiaccum readout factors for
  performance prediction and observing-time calculations.
- **Full time-domain simulation**: focal-plane evolution, sub-exposures with
  pointing jitter, and realistic detector readouts (NDRs) to test data reduction
  pipelines against a known truth.
- **Comprehensive noise budget**: photon noise, read noise, dark current, and
  user-defined noise sources with realistic detector behaviour.
- **Low-frequency systematics**: pointing jitter, thermal fluctuations, and
  detector drifts captured across the full observation timescale.
- **Multi-instrument support**: configured out of the box for the Ariel space
  mission, with community-contributed configurations for JWST and ground-based
  spectrographs.
- **Flexible configuration and outputs**: XML and YAML parameter files with
  inheritance and full unit handling via Astropy, and structured,
  self-documenting HDF5 outputs.

Who should use ExoSim 2?
========================

- **Mission planners and instrument scientists** exploring design trade-offs
  before the hardware is built.
- **Observers** estimating signal-to-noise ratios and feasibility before
  submitting a proposal.
- **Data-pipeline developers** validating reduction algorithms against
  known-truth synthetic data.
- **Students** gaining hands-on experience with realistic observation scenarios
  without needing telescope time.

Cite
====

If you use this software, please cite:

Mugnai et al., 2025, "`ExoSim 2: the new exoplanet observation simulator applied to the Ariel space mission <https://link.springer.com/article/10.1007/s10686-024-09976-2>`__", Exp. Astron, 59, 9. DOI:10.1007/s10686-024-09976-2.

Acknowledgments
===============

ExoSim 2 has been developed under the umbrella of the `Ariel Space Mission <https://arielmission.space/>`__,
with the support of the Ariel Consortium and the members of the Simulator Software, Management and Documentation (S2MD) Working Group.

..  image:: _static/ariel.png
        :width: 120
        :class: dark-light
..  image:: _static/S2MD.png
        :width: 90
        :class: dark-light


During the development of the first alpha and beta versions of this software,
`L. V. Mugnai <https://www.lorenzomugnai.com/>`__ was affiliated to `Sapienza University of Rome <https://www.phys.uniroma1.it/fisica/en-welcome>`__
and supported by `ASI <https://www.asi.it/en/>`__.


We thank Ahmed Al-Refaie for his support during the development and the inspiration provided by his code: `TauREx3 <https://arxiv.org/abs/1912.07759>`__ .
