.. _common-config-edits:

===========================
Common configuration edits
===========================

An ExoSim configuration is an XML (or YAML) file built from **sections**
(``wl_grid``, ``time_grid``, ``pointing``, ``jitter``, ``sky``, ``payload`` …),
with ``<config>`` tags pulling in shared includes and a ``unit`` attribute on
every physical value. The bundled ``examples/main_example.xml`` is a complete,
working starting point; the edits below are the ones people make most often.

Each recipe gives the **section path** to change and shows two ways to apply it:
by hand in the XML, or by asking an AI assistant through the
:doc:`MCP server <mcp>` (``config_requirements`` to see the sections and keys,
``draft_config`` to apply a set of changes and validate the result).

.. tip::
   After any edit, load the file once to check it:
   ``exosim focalplane -c my_config.xml -o /tmp/out.h5`` fails fast on a broken
   configuration, and the MCP ``validate_config`` tool reports the parsed
   sections and any error without running anything.

Change the wavelength range
===========================

Section ``wl_grid`` — ``wl_min``, ``wl_max`` (and ``logbin_resolution`` for the
sampling).

.. code-block:: xml

    <wl_grid>
        <wl_min unit="micron">0.5</wl_min>
        <wl_max unit="micron">7.8</wl_max>
        <logbin_resolution unit="">6000</logbin_resolution>
    </wl_grid>

Via the assistant: *"draft a focalplane config with wl_grid.wl_min = 0.5 micron
and wl_grid.wl_max = 7.8 micron"*.

Change the telescope collecting area
====================================

Section ``payload.Telescope`` — ``Atel`` (effective collecting area). The
per-channel optics live under ``payload.Telescope.optical_path``.

.. code-block:: xml

    <Telescope>Telescope
        <Atel unit="m2">1.1</Atel>
    </Telescope>

``payload.Telescope.Atel = 1.1 m2``.

Change the observation duration and timing
==========================================

Section ``time_grid`` — ``start_time``, ``end_time`` set the simulated window;
``low_frequencies_resolution`` sets how often the low-frequency state (pointing,
jitter envelope) is recomputed.

.. code-block:: xml

    <time_grid>
        <start_time unit="hour">0.0</start_time>
        <end_time unit="hour">6.0</end_time>
        <low_frequencies_resolution unit="hour">4</low_frequencies_resolution>
    </time_grid>

Adjust a detector
=================

Section ``payload.channel.<name>.detector`` — ``spatial_pix`` /
``spectral_pix`` (array size), ``delta_pix`` (pixel pitch), ``plate_scale``.
Name the channel because a payload usually has more than one:

.. code-block:: xml

    <channel>Photometer
        <detector>
            <spatial_pix unit="">64</spatial_pix>
            <spectral_pix unit="">64</spectral_pix>
        </detector>
    </channel>

``payload.channel.Photometer.detector.spatial_pix = 64``.

Change the target star
======================

Section ``sky.source.<name>`` — ``T`` (temperature), ``R`` (radius), ``M``
(mass), ``D`` (distance), ``z`` (metallicity), ``ra`` / ``dec``. The source is
identified by its name, so pass it as the path segment:

.. code-block:: xml

    <source>HD 209458
        <T unit="K">6100</T>
        <D unit="pc">48.0</D>
    </source>

``sky.source.HD 209458.T = 6100 K``.

Switch the retrieval backend for grounded answers
=================================================

Not a configuration edit, but relevant when driving ExoSim from an assistant:
installing ``exosim[mcp-embeddings]`` and setting
``EXOSIM_MCP_RETRIEVAL=hybrid`` adds semantic matching to the knowledge base, so
paraphrased questions ("shot noise" for "photon noise") still find the right
page. See the *How grounding works* section of the :doc:`MCP guide <mcp>`.
