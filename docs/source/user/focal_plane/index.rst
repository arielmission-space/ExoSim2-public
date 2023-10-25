.. _Focal plane creation:

===================================
Focal plane creation
===================================

The first step af an `ExoSim` simulation is the creation of the instrument focal planes.
With `focal plane` here we are referring to the production of a time dependent focal plane that could take into account for low frequency time dependences.
The focal plane creation is automatised by a recipe: :class:`~exosim.recipes.createFocalPlane.CreateFocalPlane`.
In this section we explain each of the steps that lead to the focal plane creation.

.. image:: _static/road_to_focal_plane.png
    :width: 600
    :align: center

First, we discuss the definition the light sources and their propagation

As explained in the figure above, we start from the light source. Next we explore the foreground and the common optic contributions.
Then, for each channel we estimate the channel optical path and we produce the detector focal plane.

.. toctree::
   :maxdepth: 1

    General settings <general>
    Sources <sky_sources>
    Foregrounds <foregrounds>
    Optical Paths <optical_paths>
    Channel <channel>
    Focal plane <focal_plane_array>
    Telescope pointing and multiple sources <pointing>
    Resulting focal planes <resulting_focal_plane>
    Automatic pipeline <pipeline>

Other useful capabilities of the focal plane creation process are documented in:

.. toctree::
   :maxdepth: 1

    Telescope pointing and multiple sources <pointing>
