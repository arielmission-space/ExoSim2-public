.. _run_config:


==========================
Run configuration
==========================

To handle shared information in the simulation, we can use the ``RunConfig`` class,
which is a singleton initialized by :class:`~exosim.utils.runConfig.RunConfigInit`.

.. code-block:: python

    from exosim.utils import RunConfig

.. _parallel:

Parallel processing
======================

Parallel processing is important for such demanding simulations.
The number of parallel processes to use can be set using

.. code-block:: python

    from exosim.utils import RunConfig

    RunConfig.n_job = N

This number is then set for both `joblib` and `numba` libraries.

.. _chunk_size:

Chunk size
=============

The chunk size is the size of the chunk of the cached dataset (see :ref:`cached`).
This value can be set as

.. code-block:: python

        from exosim.utils import RunConfig

        RunConfig.chunk_size = N

where `N` is the desired size of the chunk in MB, which will be set for the environment.


.. _random_seed:

Random seed and Random generators
=====================================

The initial random seed can be set as:

.. code-block:: python

    from exosim.utils import RunConfig

    RunConfig.random_seed = N

where `N` is the desired seed number.
By default, the seed is set to `None`, and therefore each simulation is unique.

ExoSim also provides a default random generator (:class:`numpy.random.Generator`) already initialized with the set random seed.
The random generator can be accessed as:

.. code-block:: python

    from exosim.utils import RunConfig

    rng = RunConfig.random_generator

and it can be used as any other random generator.

.. code-block:: python

    from exosim.utils import RunConfig

    # uniform distribution:
    RunConfig.random_generator.uniform(-1,0,1000)

    # normal distribution:
    RunConfig.random_generator.normal(0,1,1000)

    # Poisson distribution:
    RunConfig.random_generator.poisson(5, 1000)

More examples are available in the `numpy.random.Generator documentation <https://numpy.org/doc/stable/reference/random/generator.html>`_.

Because ExoSim works with chunks of data and the generator may be used in loops, if the seed is not `None`, :func:`~exosim.utils.runConfig.RunConfigInit.random_generator` updates the seed at every call, by adding 1 to the given value.
