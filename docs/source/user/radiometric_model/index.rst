.. _radiometric:

===================================
Radiometric Model
===================================

The radiometric model in `ExoSim` provides a fast estimate of channel parameters and performance metrics.
The radiometric model is handled as a recipe: :class:`~exosim.recipes.radiometric_model.RadiometricModel` (see :ref:`recipe`).

For a complete description of the steps involved, this guide refers to the pipeline encapsulated in the :class:`~exosim.recipes.radiometric_model.RadiometricModel` class,
listing its methods and discussing the involved tasks.

Operating Modes
^^^^^^^^^^^^^^^^

The radiometric model has **three operating modes** that are automatically determined based on the configuration and input files:

1. **Target list mode**: Used when a ``targetlist_filepath`` is specified in the sky configuration. Processes multiple targets from a CSV file, creating individual focal planes and radiometric estimates for each target. See :ref:`target_list_mode` for details.

2. **Existing focal plane mode**: Used when the output file already exists and contains focal plane data, but no target list is specified. Loads the existing focal plane and computes radiometric estimates directly. See :ref:`existing_fp` for details.

3. **Single source mode**: Used for single source configurations when no existing focal plane is found. Creates a new focal plane for the source and then computes radiometric estimates. See :ref:`non_existing_fp` for details.

Common Radiometric Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Regardless of the operating mode, all radiometric models follow the same core pipeline for signal estimation and noise computation.

**Signal Estimation Process:**

Starting from a focal plane (either existing or newly created), the first steps are to create and populate the radiometric table:

.. image:: _static/prepare-radiometric.png
    :width: 600
    :align: center

The signal estimation process includes:

1. **Wavelength binning**: Create the wavelength grid for the radiometric table (see :ref:`wavelength bin`)
2. **Aperture estimation**: Estimate aperture sizes and pixel counts for photometry (see :ref:`estimate apertures`)
3. **Signal calculation**: Estimate source and background signals using aperture photometry (see :ref:`estimate signals`)
4. **Saturation analysis**: Calculate detector saturation times (see :ref:`saturation_time`)

**Noise Estimation Process:**

Once the radiometric table is built, the next step is to estimate the noise from various sources:

.. image:: _static/radiometric-common.png
    :width: 600
    :align: center

The noise estimation process includes:

1. **Multiaccum factors**: Calculate factors for multiple accumulation readout schemes (see :ref:`multiaccum`)
2. **Photon noise**: Estimate photon noise contributions (see :ref:`photon noise`)
3. **Total noise**: Combine all noise sources for comprehensive noise analysis (see :ref:`total noise`)

Guide Contents
^^^^^^^^^^^^^^

This guide covers all aspects of the radiometric model, from operating modes to detailed signal and noise calculations:

.. toctree::
   :maxdepth: 1

   Operating Modes <operating_modes>
   Wavelength binning <wavelength_bin>
   Estimate apertures <estimate_apertures>
   Estimate signal <estimate_signal>
   Timing model <timing_model>
   Multiaccum <multiaccum>
   Photon noise <photon_noise>
   Read Noise <read_noise>
   Dark Current Noise <dark_current_noise>
   Custom Noise <custom_noise>
   Total Noise <total_noise>
