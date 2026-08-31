.. _Focal plane creation:

====================
Focal plane creation
====================

The first step of an `ExoSim` simulation is building the instrument focal
planes. Here, `focal plane` means a **time-dependent** focal plane that captures
low-frequency variations over the observation. The recipe
:class:`~exosim.recipes.create_focal_plane.CreateFocalPlane` automates the whole
process; this section explains the steps it goes through.

.. image:: _static/road_to_focal_plane.png
    :width: 600
    :align: center

As the figure shows, the simulation starts from the light sources. It then adds
the foregrounds and the optical contributions common to the whole instrument.
Finally, for each channel, it estimates the channel optical path and produces the
detector focal plane.

.. toctree::
   :maxdepth: 1

   General settings <general>
   Sources <sky_sources>
   Foregrounds <foregrounds>
   Optical paths <optical_paths>
   Channel <channel>
   Focal plane <focal_plane_array>
   Resulting focal planes <resulting_focal_plane>
   Automatic pipeline <pipeline>

Further capabilities of the focal-plane creation process are documented in:

.. toctree::
   :maxdepth: 1

   Telescope pointing and multiple sources <pointing>
