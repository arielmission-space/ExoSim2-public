.. _FAQs:

====
FAQs
====

.. _noVirtualEnv:

What if I don't want to create a Python virtual environment?
------------------------------------------------------------

You can skip the virtual environment if you prefer. In that case, follow every
step in :ref:`installation` except the ones that create or activate the
environment.

Note that installing ExoSim into your system Python may require administrator
privileges.

.. _failedCheck:

ExoSim is installed but not working: what can I do?
---------------------------------------------------

In our experience, most errors raised after a failed run fall into three groups:

1. Python raises :code:`ImportError: No module named ####` because it cannot
   import a module ExoSim depends on.

    You can usually fix this by installing the missing dependency:

    .. code-block:: console

        pip install ####

    If the installation reports errors, you may need administrator privileges.
    On a Unix system:

    .. code-block:: console

        sudo pip install ####


2. Python raises :code:`ImportError: No module named exosim`.

    This means ExoSim is not installed in the environment you are using. Run the
    installation procedure again, or see the solution in :ref:`noVirtualEnv`.


.. _ver:

What versioning system is used?
-------------------------------

We use a versioning scheme compliant with the PEP440_ standard. Given a version
number ``X.Y.Z``:

- **X** is the major identifier. A change here means a large part of the code has
  been refactored, and the release is not compatible with previous versions.
- **Y** is the minor identifier. It changes when we add features or change how
  part of the code works. Users will notice these changes and should check the
  documentation if errors occur.
- **Z** is the patch identifier. We fixed a bug or optimised something, and users
  will probably not notice the difference.

You may also see versions of the form ``X.YbZ``, where *b* stands for *beta*.
Other suffixes are *c* (release candidate) and *r* (release, usually omitted),
which mark stable, distributable versions.

.. _PEP440: https://www.python.org/dev/peps/pep-0440/

How can I check which ExoSim version I'm using?
-----------------------------------------------

There are several ways.

The quickest is to open this documentation and look under the logo in the
left-hand panel, or to open the :doc:`changelog <CHANGELOG>`, which lists every
release. This tells you the version the documentation describes, not necessarily
the one you have installed.

To check the installed version, run this from the ExoSim virtual environment:

.. code-block:: console

    pip show exosim

The output includes a *Version* line. From a Python session you can instead run:

.. code-block:: python

    import exosim
    exosim.__version__

The version is also recorded in the output files of every ExoSim module, in the
``ExoSim version`` metadata, and in the ``exosim.log`` file.

Make sure the installed version matches the one in the documentation. If it does
not, upgrade your installation.

If you installed `ExoSim` with pip (see :ref:`install from PyPI <install pip>`):

.. code-block:: console

    pip install exosim --upgrade

If you installed from the GitHub source (see :ref:`install from Git <install git>`), go to your
`ExoSim` directory, pull the latest changes, and update the environment:

.. code-block:: console

    cd /your_path/ExoSim2.0
    git pull
    uv sync --extra dev

Then check the version again.

.. tip::
    If you use Anaconda Python, its IDE can list every package installed in each
    virtual environment together with its version.

.. _loadHDF5:

How can I load HDF5 data into my code?
--------------------------------------

Once your dataset is stored in an `.h5` file, you can read it with the h5py_
package. Assuming the file is called `data_file.h5`:

.. code-block:: python

    import h5py

    with h5py.File('data_file.h5', 'r+') as input_file:
        ...

The file can now be navigated like a Python dictionary. The h5py documentation
covers reading and writing data
(https://docs.h5py.org/en/stable/high/dataset.html#reading-writing-data); here is
a short example:


.. code-block:: python

    import h5py

    with h5py.File('data_file.h5', 'r+') as input_file:
        data = input_file['first_level']['second_level']['dataset_name'][()]

This navigates the file to the dataset `dataset_name` under
`first_level/second_level` and loads its full content into the `data` variable.

.. _load signal table:

Load signals and tables
^^^^^^^^^^^^^^^^^^^^^^^

You can also load stored data back into its original Python class. For example,
you can cast a stored table into an :class:`~astropy.table.QTable` with
:func:`astropy.io.misc.hdf5.read_table_hdf5`:

.. code-block:: python

    import h5py
    from astropy.io.misc.hdf5 import read_table_hdf5

    with h5py.File('data_file.h5', 'r+') as input_file:
        table_data = input_file['first_level']['table_group']
        table = read_table_hdf5(table_data)

Here `table_data` is the group loaded from the HDF5 file; it holds both the table
and its metadata, stored as `.__table_column_meta__`.

For the :class:`~exosim.models.signal.Signal` class, use
:func:`exosim.output.hdf5.utils.load_signal`:

.. code-block:: python

    import h5py
    from exosim.output.hdf5.utils import load_signal

    with h5py.File('data_file.h5', 'r+') as input_file:
        signal_group = input_file['first_level']['stored_signal_name']
        signal = load_signal(signal_group)

.. _h5py: https://docs.h5py.org/en/stable/
