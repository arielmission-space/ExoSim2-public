.. _installation:

=======================
Installation & updates
=======================

The following notes guide you toward the installation of `ExoSim` using a Python virtual environment.
You have to have Python and `pip`` installed already. Ask your computer administrator in case you need to install these components.

.. note::
    The current implementation of ExoSim 2 is compatible with Python 3.8, 3.9 and 3.10.

Create the Virtual Environment
====================================
You can either create a Python Virtual Environment in your anaconda python or in a standard Python installation.

.. tip::
    The Anaconda solution is cross-platform: the following instructions should work for Windows, iOS and Linux.

.. tabbed:: Anaconda |condaLogo|


    .. |condaLogo| image:: _static/conda-logo.png
                :width: 50
                :class: dark-light

    Assuming you have `Anaconda <https://www.anaconda.com/>`__ installed in your system, you can simply install ExoSimVE following this procedure.
    Open the Anaconda command shell, or if you are on a Unix system just open the console.

    You can create a Virtual Environment as

    .. code-block:: console

        conda create --name ExoSimVE python=3.9

    The program will ask you if you want to install some standard packages: accept them.

    You can now activate or deactivate the Virtual Environment as

    .. code-block:: console

        conda activate ExoSimVE
        conda deactivate


.. tabbed:: Python venv |pythonLogo|



    .. |pythonLogo| image:: _static/python-logo.png
                    :width: 50
                    :class: dark-light


    If you have a standard Python installation, you still can work with Virtual Environment.
    You have to have Python `virtualenv`` installed.
    For Linux you can do that as:

    .. code-block:: console

        mkdir ExoSimVE
        virtualenv -p /usr/bin/python3.9 ExoSimVE

    Then activate the virtual environment. If using csh, type

    .. code-block:: console

        source ExoSimVE/bin/activate.csh

    (check virtual environment documentation when using a different shell)

If you don't want to use a virtual environment, check :ref:`noVirtualEnv`

.. _raw_installation:

ExoSim package installation
====================================

Instal ExoSim
----------------

.. tabbed:: Install from PiPy |PypiLogo|

    .. _install pip:

    .. |PypiLogo| image:: _static/pypi-logo.png
                    :width: 50
                    :class: dark-light

    The ExoSim package is hosted on Pypi repository. You can install it by

    .. code-block:: console

        pip install exosim

.. tabbed:: Install from Git |GitLogo|

    .. _install git:

    .. |GitLogo| image:: _static/Git-logo.png
                    :width: 50
                    :class: dark-light

    You can clone ExoSim from our main git repository

    .. code-block:: console

        git clone https://github.com/arielmission-space/ExoSim2-public.git

    Move into the ExoSim folder

    .. code-block:: console

        cd /your_path/ExoSim2.0

    Then, just do

    .. code-block:: console

        pip install .

To test for correct setup you can do

.. code-block:: console

    python -c "import exosim"

If no errors appeared then it was successfully installed. Additionally the `exosim` program
should now be available in the command line

.. code-block:: console

    exosim


Uninstall ExoSim
-------------------

ExoSim is installed in your system as a standard python package:
you can uninstall it from your Environment as

.. code-block:: console

    pip uninstall exosim


Upgrade ExoSim
---------------

.. tabbed:: Upgrade from PiPy |PypiLogo|


    If you have installed ExoSim from PyPi, now you can update the package simply as

    .. code-block:: console

        pip install exosim --upgrade

.. tabbed:: Upgrade from Git |GitLogo|


    If you have installed ExoSim from Git, you can download or pull a newer version of ExoSim over the old one, replacing all modified data.

    Then you have to place yourself inside the installation directory with the console

    .. code-block:: console

        cd /your_path/ExoSim2.0

    Now you can update ExoSim simply as

    .. code-block:: console

        pip install . --upgrade

    or simply

    .. code-block:: console

        pip install .


Modify ExoSim
---------------

You can modify ExoSim main code, editing it as you prefer, but in order to make the changes effective

.. code-block:: console

    pip install . --upgrade

or simply

.. code-block:: console

    pip install .

To produce new `ExoSim` functionalities and contribute to the code, please see :ref:`guidelines`.
