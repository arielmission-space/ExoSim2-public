.. _FAQs:

FAQs
=====

.. _noReqPack:

What if ExoSim cannot find the required packages?
---------------------------------------------------

ExoSim should install the required packages automatically. If for some reason it will not, you can do that manually.
To install required packages in the virtual environment use

.. code-block:: console

    cd /your_path/ExoSim
    pip install -r requirements.txt

.. _noVirtualEnv:

What if I don't want to create a python virtual environment?
------------------------------------------------------------------
Sure you can avoid the use of virtual environment, if you want.
In this case you have to follow all the step described :ref:`installation`
except the environment sections.

Note that if you are installing ExoSim in your standard Python Environment,
the system may ask you for administration privileges.

.. _failedCheck:

ExoSim is installed but not working: what can I do?
-----------------------------------------------------

In our experience, mostly the errors raised after an failed run are of three different kinds:

1. Python rasie an :code:`ImportError: No module named ####`, for it cannot import the module used.

    Generally you can fix it installing the missing dependency, as

    .. code-block:: console

        pip install ####

    if the installation return errors, you may need to use administration privileges. For a Unix system you can use:

    .. code-block:: console

        sudo pip install ####


2. Python raise an :code:`ImportError: No module named ExoSim`.

    This means that ExoSim is not installed in the environment you are using.
    Try the installation procedure again or see the solution offered in :ref:`noVirtualEnv`


.. _ver:

What versioning system is used?
--------------------------------

We are using a versioning system compliant to PEP440_ standards.
Given a version number as X.Y.Z

- X is major modification identifier. Changing this number mean that we have refactored big part of the code. It's not compatible with previous versions.
- Y is minor modification. This number changes when we add functionalities or we change how part of the code work. A User will notice these variations and should check the documentation if some errors occur.
- Z is patches identification. We fixed some problem or optimized something. Probably the user won't notice the difference.

Other used versioning is in the form X.YbZ. The only difference is thar *b* stands for *beta*.
You may also find *c*, meaning *release candidate*, or simply *r*, meaning *release* (this might be omitted) to indicate stable distributable versions.

.. _PEP440: https://www.python.org/dev/peps/pep-0440/

How can I check which ExoSim version I'm using?
-------------------------------------------------

You can do this in multiple ways.
The easiest way is to open the documentation (you already did) and look under the logo on the left panel or go to the :doc:`changelog <CHANGELOG>`:
there you will find all the versions listed.
But this actually only refers to the directory you downloaded. To be sure that the version you are using is the same of the one you downloaded,
run from the ExoSim Virtual Environment

.. code-block:: console

    pip show ExoSim

you will see all the installation details for ExoSim, including a *Version* line.
Or you can check it from your python Virtual Environment

.. code-block:: python

    import ExoSim
    ExoSim.__version__

or you can find it into the output files af an ExoSim module, looking for the ``ExoSim version`` metadata.
Finally, you can even find that information inside the ``ExoSim.log`` file.

Be sure that version is the same reported in the documentation.
If not, upgrade your installation with

.. code-block:: console

    pip install exosim --upgrade

if you installed `ExoSim` using pip (see :ref:`install pip`).
If you used the source code from GitHub (see :ref:`install git`) go in your `ExoSim` directory,
pull the last change and update your installation:

.. code-block:: console

    cd /your_path/ExoSim
    git pull
    pip install . --upgrade

and check again.

.. tip::
    If you are using Anaconda Python, there must be a IDE listing all the installed package for your Virtual Environments and their versions.

.. _loadHDF5:

How can I load HDF5 data into my code?
-------------------------------------------------
Once you have produced your dataset and it is stored into an `.h5` file,
you can use the data using the python package h5py_.
Assuming you data file is called `data_file.h5`, you can include it in your code as

.. code-block:: python

    import h5py

    with h5py.File('data_file.h5', 'r+') as input_file:
        ...

Now the file can be navigated as a python dictionary.
To read and use the data the user can refer to the documentation (https://docs.h5py.org/en/stable/high/dataset.html#reading-writing-data),
but here is an example:


.. code-block:: python

    import h5py

    with h5py.File('data_file.h5', 'r+') as input_file:
        data = input_file['first_level']['second_level']['dataset_name'][()]

This script navigates the file looking for the dataset called `dataset_name` that is under `first_level/second_level`
and it loads all the dataset content into the `data` variable.

.. _load signal table:

Load signals and tables
^^^^^^^^^^^^^^^^^^^^^^^^^^
You can load the data stored into HDF5 file into their original python classes.
In particular, you can cast a stored table into an :class:`~astropy.table.QTable`, using :func:`astropy.io.misc.hdf5.read_table_hdf5`:

.. code-block:: python

    import h5py
    from astropy.io.misc.hdf5 import read_table_hdf5

    with h5py.File('data_file.h5', 'r+') as input_file:
        table_data = input_file['first_level']['table_group']
        table = read_table_hdf5(table_data)

where `table_data` is a dictionary loaded from the hdf5 that contains both the table and the table metadata,
stored in the fle as `.__table_column_meta__`.

In the case of :class:`~exosim.models.signal.Signal` class, you can use the :func:`exosim.output.hdf5.utils.load_signal`:

.. code-block:: python

    import h5py
    from exosim.output.hdf5.utils import load_signal

    with h5py.File('data_file.h5', 'r+') as input_file:
        signal_group = input_file['first_level']['stored_signal_name']
        signal = load_signal(signal_group)

.. _h5py: https://docs.h5py.org/en/stable/
