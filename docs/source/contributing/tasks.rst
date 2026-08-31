.. _tasks:

==================
The task structure
==================

Instead of plain functions, `ExoSim` uses a task system. A
:class:`exosim.tasks.task.Task` is a class that carries out one operation and,
through :class:`~exosim.log.logger.Logger`, comes with logging built in. Tasks
have a fixed shape, so they can be swapped for custom versions (see
:ref:`Custom Tasks`).

Write a task
------------

First, create a class that inherits from :class:`~exosim.tasks.task.Task`:

.. code-block:: python

    from exosim.tasks.task import Task

    class ExampleTask(Task):
        """
        This is an example Task
        """

Then declare the inputs in ``__init__``, with
:func:`~exosim.tasks.task.Task.add_task_param`:

.. code-block:: python

        def __init__(self):
            """
             Parameters
             __________
             parameters: dict
                 dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`
             wavelength: :class:`~astropy.units.Quantity`
                 wavelength grid.
             output: :class:`~exosim.output.output.Output` (optional)
                output file
             """

            self.add_task_param('parameters', 'channel parameters dict')
            self.add_task_param('wavelength', 'wavelength grid')
            self.add_task_param('output', 'output file', None)

Here the task takes three inputs: a dictionary, a wavelength grid and an output
file. The last one is optional, since it is given the default value ``None``.

Then describe what the task does, in ``execute``:

.. code-block:: python

    def execute(self):
        parameters = self.get_task_param('parameters')
        wavelength = self.get_task_param('wavelength')
        output = self.get_task_param('output')

        ...

        variable = None
        self.set_output(variable)

:func:`~exosim.tasks.task.Task.get_task_param` returns the value associated with
the input name. After the work is done, the result is handed back with
:func:`~exosim.tasks.task.Task.set_output`. To return several values, pass a
list:

.. code-block:: python

        variable1 = None
        variable2 = None
        self.set_output([variable1, variable2])

Logging
-------

The :class:`~exosim.tasks.task.Task` class provides the same logging methods as
:class:`~exosim.log.logger.Logger`:

.. code-block:: python

    self.info("info message")
    self.debug("debug message")
    self.warning("warning message")
    self.error("error message")
    self.critical("critical message")

They can be used in any method of the task.

Use a task
----------

Initialise the task, then call it with its parameters:

.. code-block:: python

    exampleTask = ExampleTask()
    variable = exampleTask(parameters=par_dic, wavelength=wl_grid)
