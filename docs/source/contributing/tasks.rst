.. _tasks:


==========================
The Task structure
==========================

Instead of functions, `ExoSim` uses a tasks system.
The :class:`exosim.tasks.task.Task` is a class with logging properties (from :class:`~exosim.log.logger.Logger`) which executes some operations.

Write a Task
--------------

To write a task, first we need to create a class that inherits from the :class:`~exosim.tasks.task.Task` class:

.. code-block:: python

    from exosim.tasks.task import Task

    class ExampleTask(Task):
        """
        This is an example Task
        """


Then we need to define the inputs. This must be done in the class `__init__` using the :func:`~exosim.tasks.task.Task.add_task_param` method:

.. code-block:: python

        def __init__(self):
            """
             Parameters
             __________
             parameters: dict
                 dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
             wavelength: :class:`~astropy.units.Quantity`
                 wavelength grid.
             output: :class:`~exosim.output.output.Output` (optional)
                output file
             """

            self.add_task_param('parameters', 'channel parameters dict')
            self.add_task_param('wavelength', 'wavelength grid')
            self.add_task_param('output', 'output file', None)

In this example we want the task to have 3 inputs: a dictionary, a wavelength grid and an output file.
The latter is optional, in fact we have set ``None`` as a default value.

Then we can move to describe the operations that this Task is gonna do:

.. code-block:: python

    def execute(self):
        parameters = self.get_task_param('parameters')
        parameters = self.get_task_param('wavelength')
        parameters = self.get_task_param('output')

        ...

        variable = None
        self.set_output(variable)

The :func:`~exosim.tasks.task.Task.get_task_param` method allow to retrieve the variable associated to the string used as argument.
Then some operations are done and the output is set with the :func:`~exosim.tasks.task.Task.set_output` method.
If a list of variables are expected as output the code will be

.. code-block:: python

        variable1 = None
        variable2 = None
        self.set_output([variable1,variable2])

Logging
--------------
Logging is important when producing a new task, hence we include some logging options into the :class:`~exosim.tasks.task.Task` class.
Here are some examples of how to use them, but you can have a better understanding by looking at the :class:`~exosim.log.logger.Logger` class.

.. code-block:: python

    self.info("info message")
    self.debug("debug message")
    self.warning("warning message")
    self.error("error message")
    self.critical("critical message")

These lines can be include in every method inside the :class:`~exosim.tasks.task.Task` class.

Use a Task
--------------
To use a :class:`~exosim.tasks.task.Task` we first need to initialise it, and the call it with it's parameters:

.. code-block:: python

    exampleTask = ExampleTask()
    variable = exampleTask(parameters = par_dic, wavelength=wl_grid)
