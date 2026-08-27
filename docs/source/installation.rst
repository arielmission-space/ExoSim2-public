.. _installation:

=======================
Installation & updates
=======================

The following notes guide you toward the installation of `ExoSim` using a Python virtual environment.
You must have Python and `pip` installed already. Ask your computer administrator in case you need to install these components.

.. note::
    The current implementation of ExoSim 2 is compatible with Python >3.12.

Create the Virtual Environment
====================================
You can either create a Python Virtual Environment in your anaconda python or in a standard Python installation.

.. tip::
    The Anaconda solution is cross-platform: the following instructions should work for Windows, macOS, and Linux.

.. tab-set::

    .. tab-item:: Anaconda |condaLogo|


        .. |condaLogo| image:: _static/conda-logo.png
                    :width: 50
                    :class: dark-light

        Assuming you have `Anaconda <https://www.anaconda.com/>`__ installed on your system, you can simply install ExoSimVE following this procedure.
        Open the Anaconda command shell, or if you are on a Unix system, just open the console.

        You can create a Virtual Environment as

        .. code-block:: console

            conda create --name ExoSimVE python=3.12

        The program will ask if you want to install some standard packages: accept them.

        You can now activate or deactivate the Virtual Environment as

        .. code-block:: console

            conda activate ExoSimVE
            conda deactivate


    .. tab-item:: Python venv |pythonLogo|



        .. |pythonLogo| image:: _static/python-logo.png
                        :width: 50
                        :class: dark-light


        If you have a standard Python installation, you can still work with a virtual environment.
        You must have Python `virtualenv` installed.
        For Linux, you can do that as:

        .. code-block:: console

            mkdir ExoSimVE
            virtualenv -p /usr/bin/python3.12 ExoSimVE

        Then activate the virtual environment. If using csh, type

        .. code-block:: console

            source ExoSimVE/bin/activate.csh

        (check the virtual environment documentation when using a different shell)

If you don't want to use a virtual environment, check :ref:`noVirtualEnv`

.. _raw_installation:

ExoSim package installation
====================================

Install ExoSim
----------------

.. _install pip:
.. _install git:

.. tab-set::

    .. tab-item:: Install from PyPI |PypiLogo|
        :sync: pypi

        .. |PypiLogo| image:: _static/pypi-logo.png
                        :width: 50
                        :class: dark-light

        The ExoSim package is hosted on the PyPI repository. You can install it by:

        .. code-block:: console

            pip install exosim

    .. tab-item:: Install from Git |GitLogo|
        :sync: git

        .. |GitLogo| image:: _static/Git-logo.png
                        :width: 50
                        :class: dark-light

        You can clone ExoSim from our main Git repository:

        .. code-block:: console

            git clone https://github.com/arielmission-space/ExoSim2-public.git

        Move into the ExoSim folder:

        .. code-block:: console

            cd /your_path/ExoSim2.0

        ExoSim uses **uv** for dependency management and package installation. If you haven't installed uv yet, you can do so by following the `official uv documentation <https://docs.astral.sh/uv/getting-started/installation/>`_.

        Once uv is installed, you can proceed with installing ExoSim::

            uv sync --extra dev

To test for correct setup you can do

.. code-block:: console

    python -c "import exosim"

If no errors appeared, then it was successfully installed. Additionally, the `exosim` program
should now be available on the command line:

.. code-block:: console

    exosim


Uninstall ExoSim
-------------------

ExoSim is installed on your system as a standard Python package:
you can uninstall it from your environment as:

.. code-block:: console

    pip uninstall exosim


Upgrade ExoSim
---------------

.. tab-set::

    .. tab-item:: Upgrade from PyPI |PypiLogo|
        :sync: pypi



        If you have installed ExoSim from PyPI, you can now update the package simply as:

        .. code-block:: console

            pip install exosim --upgrade

    .. tab-item:: Upgrade from Git |GitLogo|
        :sync: git


        If you have installed ExoSim from Git, you can download or pull a newer version of ExoSim over the old one, replacing all modified data.

        Then you must place yourself inside the installation directory with the console:

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

You can modify ExoSim's main code, editing it as you prefer, but in order to make the changes effective:

.. code-block:: console

    pip install . --upgrade

or simply

.. code-block:: console

    pip install .

To produce new `ExoSim` functionalities and contribute to the code, please see :ref:`guidelines`.
