.. _guidelines:

===================================
Contributing guidelines
===================================

Code bugs and issues
------------------------
If you notice a bug or an issue, the best thing to do is to open an issue on the `GitHub repository <https://github.com/arielmission-space/ExoSim2-public/issues>`__.

Coding conventions
-----------------------

The code has been developed following the PEP 8_ standard and the Python Zen_.
If you have any doubts, try

.. code-block:: python

    import this


Documentation
-----------------------
Every function or class should be documented using docstrings which follow the numpydoc_ structure.
This web page is written using the reStructuredText_ format, which is parsed by Sphinx_.
If you want to contribute to this documentation, please refer to the Sphinx_ documentation first.
You can improve these pages by digging into the `docs` directory in the source.

To help the contributor in writing the documentation, we have created two `nox <https://nox.thea.codes/en/stable/>`__ sessions:

.. code-block:: bash

    $ nox -s docs
    $ nox -s docs-live

The first will build the documentation; the second will build the documentation and open a live server to see the changes in real time.
The live server can be accessed at http://localhost:8000/

.. note::
    To run a ``nox`` session, you need to install it first. You can do it by running:

    .. code-block:: bash

        $ pip install nox

Testing
-----------------------
Unit testing is very important for a code as large as `ExoSim 2`.
At the moment, `ExoSim` is tested using pytest_.
If you add functionalities, please also add a dedicated test into the `tests` directory.
All the tests can be run with:

.. code-block:: console

    pytest

.. _logging:

Logging
--------------
Logging is important when coding, hence we include a :class:`exosim.log.logger.Logger` class to inherit.

.. code-block:: python

    import exosim.log as log

    class MyClass(log.Logger):
        ...

Now the new class has the following methods from the main :class:`~exosim.log.logger.Logger` class:

.. code-block:: python

    self.info()
    self.debug()
    self.warning()
    self.error()
    self.critical()

where the arguments shall be strings.
The logger output will be printed during the run or stored in the log file, if the log file option is enabled.
To enable the log file, use :func:`exosim.log.add_log_file`.

.. note::

    The logger here produced is inspired by the logging classes in TauREx3_ developed by Ahmed Al-Refaie.

The user can also set the level of the printed messages using :func:`exosim.log.set_log_level`, or enable or disable the messages with :func:`exosim.log.enableLogging` or :func:`exosim.log.disable_logging`

If the contributor wants to trace every time a function is called, the :func:`exosim.log.logger.traced` decorator_ is useful:

.. code-block:: python

    import exosim.log as log

    @log.traced
    def my_func(args):
        ...

This will produce a log every time the function is entered and exited, with a `TRACE` logging level.

Versioning conventions
-----------------------

The versioning convention used is the one described in Semantic Versioning (semver_) and is compliant with the PEP440_ standard.
In the [Major].[minor].[patch] scheme, for each modification to the previous release, we increase one of the numbers.

+ `Major` is increased only if the code is no longer compatible with the previous version. This is considered a Major change.
+ `minor` is increased for minor changes. These are for the addition of new features that may change the results from previous versions. These are still hard edits, but not enough to justify the increase of a `Major`.
+ `patch` are the patches. This number should increase for any bug fixes, or minor additions or changes to the code. It won't affect the user experience in any way.

Additional information can be added to the version number using the following scheme: [Major].[minor].[patch]-[Tag].[update].

+ `Major` is increased only if the code is no longer compatible with the previous version. This is considered a Major change.
+ `minor` is increased for minor changes. These are for the addition of new features that may change the results from previous versions. These are still hard edits, but not enough to justify the increase of a `Major`.
+ `patch` are the patches. This number should increase for any bug fixes, or minor addition or change to the code. It won't affect the user experience in any way.
+ `Tag` is a string that can be added to the version number. It can be used to indicate the type of release, or the type of change. For example, `alpha`, `beta`, `release`, or `dev` can be used to indicate that the version is not stable yet.
+ `updated` is a number to increase for all the changes that are not related to the code patch. This is useful for development purposes, to keep track of the number of updates since the last release.

See also :ref:`ver`.

.. _PEP440: https://www.python.org/dev/peps/pep-0440/

Automatic version numbers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The version number is **not** stored in the repository: it is derived from the
git history by `setuptools-scm <https://setuptools-scm.readthedocs.io/>`__.
Every commit on ``develop`` gets a unique ``X.Y.(Z+1).devN`` version
automatically (no commits, no ``-devN`` tags), and a release tag ``vX.Y.Z``
produces the clean ``X.Y.Z``.

Cutting a release is a single command:

.. code-block:: console

    nox -s release

which asks whether the bump is ``major``, ``minor`` or ``patch``, collects the
``changelog.d/`` fragments into the Changelog, updates the metadata files, and
pushes the ``vX.Y.Z`` tag. The tag triggers the automated pipeline that builds
the package, publishes it to PyPI and creates the GitHub Release.

The full procedure — the branch model, how to write changelog entries, and the
one-time PyPI setup — is documented in :ref:`releasing`.

Source Control
------------------


The code is hosted on GitHub (https://github.com/arielmission-space/ExoSim2-public) and is structured as follows:

The source has two long-lived branches:

+ ``main``: holds the latest stable release only. It is updated (fast-forwarded) by the release process and should never be committed to directly.
+ ``develop``: the working branch where new features land. Releases are cut from here with ``nox -s release`` (see :ref:`releasing`).

Adding new features
^^^^^^^^^^^^^^^^^^^^^^

New features can be added to the code following the schemes designed above.

If the contributor has writing rights to the repository, they should create a new branch starting from the `develop` one.
In the new `feature` branch, the user should produce the new functionalities according to the above guidelines.
When the feature is ready, the branch can be merged into the official `develop` one.

.. image:: _static/github.png
    :align: center
    :width: 600

To create the new feature starting from the current develop version, the contributor should run

.. code-block:: console

    git checkout develop
    git checkout -b feature/<branchname>

The completed feature can then be added to the develop.
This can be done in two ways: by a merge or a pull_ request.

Merge
++++++

A merge is a soft way to add a new feature to another branch.
Performing a merge means that the change will be applied if the two branches are compatible, according to GitHub.

.. code-block:: console

    git merge develop
    git checkout develop
    git merge feature/<branchname>
    git push

Once a feature is completed and merged, the contributor should `archive` the branch and remove it to keep the repository clean.
The usual procedure is:

.. code-block:: console

    git tag archive/<branchname> feature/<branchname>
    git push --tags
    git branch -d feature/<branchname>

Remember to delete the branch from the remote repository as well.
If needed, the feature branch can be restored as

.. code-block:: console

    git checkout -b <branchname> archive/<branchname>

Fixing bug
^^^^^^^^^^^

The procedure to fix a bug is similar to the one for adding a new feature.

Create the new branch starting from the current develop version

.. code-block:: console

    git checkout develop
    git checkout -b fix/<branchname>

Then, once the bug is fixed, the branch can be merged into the official `develop` one.

.. code-block:: console

    git merge develop
    git checkout develop
    git merge fix/<branchname>
    git push

    git tag archive/fix/<branchname> fix/<branchname>
    git push --tags
    git branch -d fix/<branchname>

Pull request
++++++++++++++++
A similar result can be obtained via the GitHub web interface.
When the feature is completed, the contributor can visit the branches_ tab of the GitHub page.
From there, it is possible to open a pull_ request by clicking on the button on the right of the branch we want to merge.
Then select `develop` as the destination branch and confirm.
GitHub will run all the Python tests written for `ExoSim 2` and check for compatibility between the two branches.
If everything is okay, a merge can be confirmed.

Then, on the branches_ page, it is possible to delete the new feature branch if it is no longer useful.

Fork and Pull
++++++++++++++

If the contributor does not have writing rights to the repository, they should use the Fork-and-Pull_ model.
The contributor should fork_ the main repository and clone it. Then the new features can be implemented.
When the code is ready, a pull_ request can be raised.

.. image:: _static/fork_pull.png
    :align: center
    :width: 600

Derived projects
^^^^^^^^^^^^^^^^^

If the contributor wants to maintain a custom forked repository or derived project, the following naming convention should be followed:
the forked repository should be named after the original repository plus an identifying name.
The following picture shows an example of a possible growth of the ExoSim family:

.. image:: _static/Exosim_Family_fork.png
    :align: center
    :width: 600

Automatic actions
^^^^^^^^^^^^^^^^^^^^
Every time a commit is ``pushed`` into the `develop` or `main` branch, some automatic actions_ are run by GitHub.
The available actions are stored in the `.github/workflows` directory in this repository.
The basic actions are three:

    - Linux OS (ci_linux.yml): this action runs all the tests implemented in the repository for an Ubuntu virtual machine with Python 3.11 to 3.14.

If all the tests for each action are passed, a green badge will be added to the repository README.


.. _PEP 8: https://www.python.org/dev/peps/pep-0008/
.. _Zen: https://www.python.org/dev/peps/pep-0020/
.. _reStructuredText: https://docutils.sourceforge.io/rst.html
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/
.. _Fork-and-Pull: https://en.wikipedia.org/wiki/Fork_and_pull_model
.. _fork: https://docs.github.com/en/get-started/quickstart/fork-a-repo
.. _pull: https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
.. _semver: https://semver.org/spec/v2.0.0.html
.. _pytest: https://docs.pytest.org/en/latest/
.. _decorator: https://realpython.com/primer-on-python-decorators/
.. _TauREx3: https://taurex3-public.readthedocs.io/en/latest/
.. _actions: https://github.com/features/actions
.. _branches: https://github.com/arielmission-space/ExoSim2-public/branches
