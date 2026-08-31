.. _installation:

======================
Installation & updates
======================

These notes walk you through installing `ExoSim` inside a Python virtual
environment. You need Python and `pip` already installed; ask your system
administrator if you have to add them.

.. note::
    ExoSim 2 requires Python 3.12 or newer.

Create the virtual environment
==============================

You can create the virtual environment either with Anaconda or with a standard
Python installation.

.. tip::
    The Anaconda solution is cross-platform: the instructions below work on
    Windows, macOS, and Linux.

.. tab-set::

    .. tab-item:: Anaconda |condaLogo|


        .. |condaLogo| image:: _static/conda-logo.png
                    :width: 50
                    :class: dark-light

        Assuming `Anaconda <https://www.anaconda.com/>`__ is installed on your
        system, open the Anaconda command shell (on a Unix system, just open a
        terminal).

        Create the virtual environment with:

        .. code-block:: console

            conda create --name ExoSimVE python=3.12

        The installer asks whether you want to install some standard packages:
        accept them.

        You can then activate or deactivate the virtual environment with:

        .. code-block:: console

            conda activate ExoSimVE
            conda deactivate


    .. tab-item:: Python venv |pythonLogo|



        .. |pythonLogo| image:: _static/python-logo.png
                        :width: 50
                        :class: dark-light


        With a standard Python installation you can still work in a virtual
        environment, but you need Python `virtualenv` installed. On Linux:

        .. code-block:: console

            mkdir ExoSimVE
            virtualenv -p /usr/bin/python3.12 ExoSimVE

        Then activate the virtual environment. With csh, type:

        .. code-block:: console

            source ExoSimVE/bin/activate.csh

        (see the ``virtualenv`` documentation for other shells).

If you would rather not use a virtual environment, see :ref:`noVirtualEnv`.

.. _raw_installation:

Install the ExoSim package
==========================

Install ExoSim
--------------

.. _install pip:
.. _install git:

.. tab-set::

    .. tab-item:: Install from PyPI |PypiLogo|
        :sync: pypi

        .. |PypiLogo| image:: _static/pypi-logo.png
                        :width: 50
                        :class: dark-light

        ExoSim is published on PyPI. Install it with:

        .. code-block:: console

            pip install exosim

    .. tab-item:: Install from Git |GitLogo|
        :sync: git

        .. |GitLogo| image:: _static/Git-logo.png
                        :width: 50
                        :class: dark-light

        Clone ExoSim from the main Git repository:

        .. code-block:: console

            git clone https://github.com/arielmission-space/ExoSim2-public.git

        Move into the ExoSim folder:

        .. code-block:: console

            cd /your_path/ExoSim2.0

        ExoSim uses **uv** for dependency management and package installation.
        If you do not have uv yet, install it by following the `official uv
        documentation <https://docs.astral.sh/uv/getting-started/installation/>`_.

        Once uv is available, install ExoSim with its development dependencies::

            uv sync --extra dev

To check that the installation succeeded, run:

.. code-block:: console

    python -c "import exosim"

If no error is raised, ExoSim was installed correctly. The `exosim` command is
now also available on the command line:

.. code-block:: console

    exosim


Uninstall ExoSim
----------------

ExoSim is installed as a standard Python package, so you can remove it from your
environment with:

.. code-block:: console

    pip uninstall exosim


Upgrade ExoSim
--------------

.. tab-set::

    .. tab-item:: Upgrade from PyPI |PypiLogo|
        :sync: pypi



        If you installed ExoSim from PyPI, update it with:

        .. code-block:: console

            pip install exosim --upgrade

    .. tab-item:: Upgrade from Git |GitLogo|
        :sync: git


        If you installed ExoSim from Git, pull the latest version over the old
        one, then move into the installation directory:

        .. code-block:: console

            cd /your_path/ExoSim2.0

        Update the environment with:

        .. code-block:: console

            uv sync --extra dev


Modify ExoSim
-------------

You can edit ExoSim's source code as you like. An editable install (``uv sync``,
or ``pip install -e .``) makes your changes take effect immediately, with no
reinstall needed.

To develop new `ExoSim` features and contribute them back to the project, see
:ref:`guidelines`.
