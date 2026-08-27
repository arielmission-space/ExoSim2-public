.. _index:

===================================
Welcome to ExoSim 2's documentation!
===================================

**Version**: |version|


ExoSim is a time-domain simulator for exoplanet observations designed to be easy to use and largely customisable: almost every part of the code can be customised by the user.

This guide will walk you through the simulation steps with examples and descriptions of the simulation strategy.
The guide aims to train the user to customise the simulator according to the instrument required for the observation.

What is ExoSim 2?
=================

**ExoSim 2** is an end-to-end time-domain simulator for exoplanet observations, designed to produce realistic synthetic data from space telescopes, ground-based observatories, and sub-orbital platforms. It models the complete observation chain from astronomical sources through instrument optics to detector readouts, capturing both astrophysical signals and instrumental systematic effects.

Built from the ground up in Python 3 with object-oriented architecture, ExoSim 2 offers unprecedented flexibility: researchers can replace individual components with custom implementations while maintaining the integrity of the full simulation pipeline.

What problems does ExoSim 2 solve?
-----------------------------------

**Mission and instrument design**: ExoSim 2 enables systematic exploration of design trade-offs before hardware is built, answering questions such as "How will detector noise affect transit depth precision?" or "What wavelength coverage is needed to characterise this atmosphere?"

**Performance prediction**: Generate fast radiometric estimates of signal-to-noise ratios, saturation times, and noise budgets (photon noise, read noise, dark current) for specific observing scenarios. Evaluate feasibility and optimise observing strategies before committing telescope time.

**Full time-domain simulation**: Produce complete synthetic observation datasets including focal plane evolution, sub-exposures with pointing jitter, and realistic detector readouts (NDRs). Test data reduction pipelines against known-truth inputs and quantify systematic biases.

**Training and education**: Provide students and early-career researchers with hands-on experience of realistic observation scenarios without requiring telescope time or expensive hardware access.

Key capabilities
----------------

- **Radiometric modelling**: Fast estimation of signal and noise budgets, aperture photometry, saturation analysis, and multiaccum readout factors for performance prediction and observing-time calculations
- **Complete observation chain**: From source spectra through telescope optics, dispersive elements, detector arrays, and readout electronics to synthetic detector frames
- **Time-domain modelling**: Capture low-frequency systematic effects such as pointing jitter, thermal fluctuations, and detector drifts across observation timescales
- **Comprehensive noise budget**: Model photon noise, read noise, dark current, and custom noise sources with realistic detector behaviour
- **Modular architecture**: Replace or extend individual components (optical elements, noise sources, readout schemes) without rewriting the pipeline
- **Multi-instrument support**: Configured out-of-the-box for the Ariel space mission, with community-contributed configurations for JWST, ground-based spectrographs, and other platforms
- **Flexible configuration**: XML and YAML parameter files with inheritance, path resolution, and full unit handling via Astropy
- **HDF5 outputs**: Structured, self-documenting outputs compatible with standard analysis tools

Who should use ExoSim 2?
-------------------------

- **Mission planners** designing new exoplanet observatories and evaluating instrument concepts
- **Instrument scientists** optimising detector parameters, readout schemes, and observing strategies
- **Observers** estimating signal-to-noise ratios, exposure times, and feasibility before proposal submission
- **Data pipeline developers** validating reduction algorithms against known-truth synthetic data
- **Theorists** generating mock catalogues to test atmospheric retrieval codes and bias-correction methods

.. grid:: 2 2 2 2
    :gutter: 3

    .. grid-item-card:: Installation
        :text-align: center
        :shadow: md
        :link: installation
        :link-type: doc

        .. image:: _static/install.png
            :width: 250
            :align: center

        +++

        New to ExoSim? Install ExoSim to start.

    .. grid-item-card:: User Guide
        :text-align: center
        :shadow: md
        :link: user/index
        :link-type: doc

        .. image:: _static/user.png
            :width: 250
            :align: center

        +++

        Learn how to use ExoSim.

    .. grid-item-card:: Developer Guide
        :text-align: center
        :shadow: md
        :link: contributing/index
        :link-type: doc

        .. image:: _static/developers.png
            :width: 250
            :align: center

        +++

        Learn how to improve or customise ExoSim.

    .. grid-item-card:: API Guide
        :text-align: center
        :shadow: md
        :link: api/exosim/index
        :link-type: doc

        .. image:: _static/api.png
            :width: 250
            :align: center

        +++

        Dig into the complete API guide.


.. grid:: 3 3 3 3
    :gutter: 3

    .. grid-item-card:: FAQs
        :text-align: center
        :shadow: md
        :link: FAQs
        :link-type: doc

        .. image:: _static/faqs.png
            :width: 150
            :align: center

        +++

        Go To FAQs

    .. grid-item-card:: License
        :text-align: center
        :shadow: md
        :link: license
        :link-type: doc

        .. image:: _static/license.png
            :width: 150
            :align: center

        +++

        Go To License

    .. grid-item-card:: Changelog
        :text-align: center
        :shadow: md
        :link: CHANGELOG
        :link-type: doc

        .. image:: _static/changelog.png
            :width: 150
            :align: center

        +++

        Go To Changelog

.. toctree::
    :hidden:
    :maxdepth: 1

    Installation <installation>
    User Guide <user/index>
    Developer Guide  <contributing/index>
    API Guide <api/index>
    FAQs <FAQs>
    License <license>
    Changelog <CHANGELOG>

.. warning::

    This documentation is not complete yet. If you find any issue or difficulty,
    please contact the developers for help.

Cite
------

If you use this software please cite:

Mugnai et al., 2025, "`ExoSim 2: the new exoplanet observation simulator applied to the Ariel space mission <https://link.springer.com/article/10.1007/s10686-024-09976-2>`__", Exp. Astron, 59, 9. DOI:10.1007/s10686-024-09976-2.

Acknowledgments
------------------

ExoSim 2 has been developed under the umbrella of `Ariel Space Mission <https://arielmission.space/>`__,
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


.. The icons used in this page are from `Font PNGEgg <https://https://www.pngegg.com>`__,
    and all the icons are licensed under `Non commercial license <https://www.pngegg.com/tos>`__.
    Non-commercial means something is not primarily intended for, or directed towards, commercial advantage or monetary compensation by an individual or organisation.
    As ExoSim is a publicly available software, under BSD-3-Clause license, we are allowed to use these icons in our documentation, because they are not used for commercial purposes.
We believe that our work does not conflict with the legitimate interests of the creator of the artistic icons.
    We also edited some of the icon to fit our needs.
