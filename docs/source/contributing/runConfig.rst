.. _run_config:

=================
Run configuration
=================

Information that is shared across the whole simulation is held in the
``RunConfig`` class, a singleton initialised by
:class:`~exosim.utils.run_config.RunConfigInit`:

.. code-block:: python

    from exosim.utils import RunConfig

.. _parallel:

Parallel processing
-------------------

`ExoSim` simulations are demanding, so parallel processing matters. Set the
number of parallel processes with:

.. code-block:: python

    from exosim.utils import RunConfig

    RunConfig.n_job = N

The value is applied to both `joblib` and `numba`.

.. _chunk_size:

Chunk size
----------

The chunk size is the size of a chunk of a cached dataset (see :ref:`cached`).
Set it with:

.. code-block:: python

        from exosim.utils import RunConfig

        RunConfig.chunk_size = N

where ``N`` is the chunk size in MB, applied for the whole environment.

.. _random_seed:

Random seed and random generators
---------------------------------

Set the initial random seed with:

.. code-block:: python

    from exosim.utils import RunConfig

    RunConfig.random_seed = N

where ``N`` is the seed. By default the seed is ``None``, so each simulation is
unique.

`ExoSim` also provides a default random generator
(:class:`numpy.random.Generator`), already initialised with the current seed:

.. code-block:: python

    from exosim.utils import RunConfig

    rng = RunConfig.random_generator

It is used like any other NumPy generator:

.. code-block:: python

    from exosim.utils import RunConfig

    # uniform distribution:
    RunConfig.random_generator.uniform(-1,0,1000)

    # normal distribution:
    RunConfig.random_generator.normal(0,1,1000)

    # Poisson distribution:
    RunConfig.random_generator.poisson(5, 1000)

More examples are in the `numpy.random.Generator documentation
<https://numpy.org/doc/stable/reference/random/generator.html>`_.

`ExoSim` works on chunks of data and the generator may be called inside loops, so
when the seed is not ``None``,
:func:`~exosim.utils.run_config.RunConfigInit.random_generator` adds 1 to the
seed at every call. This keeps the draws independent between chunks while staying
reproducible.
