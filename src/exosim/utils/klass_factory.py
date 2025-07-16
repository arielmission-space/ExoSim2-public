from importlib import import_module


def find_klass_in_file(python_file, baseclass):
    """
    It finds in the indicated python file a class that is a sublcass of the given one.

    Parameters
    ----------
    python_file: str
        python file name
    baseclass: class
        reference class to search for

    Returns
    -------
    class:
        class found in the python file.
    """

    import importlib.util
    import inspect

    spec = importlib.util.spec_from_file_location("foo", python_file)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    classes = [
        m[1]
        for m in inspect.getmembers(foo, inspect.isclass)
        if m[1] is not baseclass and issubclass(m[1], baseclass)
    ]

    if len(classes) == 0:
        # logger = logging.getLogger(generate_logger_name(obj))
        # logger.error('Could not find class of type %s in file %s',
        #           baseclass, python_file)
        raise Exception(
            f"No class inheriting from {baseclass} in " f"{python_file}"
        )
    return classes[0]


def load_klass(input, baseclass):
    """
    It returns a class that is a subclass of the given base class.

    Parameters
    ----------
    input: str or class
        if is a string, :func:`find_klass_in_file` is used to return the right class.
        If is a class, it checks whether it is an eligible class or not.
    baseclass: class
        reference class to search for

    Returns
    -------
    class:
        subclass of baseclass

    """

    if isinstance(input, str):
        return find_klass_in_file(input, baseclass)
    else:
        raise TypeError("task model in the wrong format")


def find_task(input, baseclass):
    """
    It looks for a class that is a subclass of the base class indicated.

    Parameters
    ----------
    input: str or object
        can either be a string indicating a class name, a python file, or it can be a class.

    baseclass: object
        reference class

    Returns
    -------
    object

    """
    if isinstance(input, str):
        if input == baseclass.__name__:
            # import the base class
            klass = baseclass
        elif input.endswith(".py"):
            # import from file
            klass = load_klass(input, baseclass)
        else:
            # look for task class by name in exosim.tasks submodules
            for sub in dir(import_module("exosim.tasks")):
                try:
                    klass = getattr(
                        import_module("exosim.tasks.{}".format(sub)), input
                    )
                except (ModuleNotFoundError, AttributeError):
                    continue
    elif issubclass(input, baseclass):
        klass = input
    else:
        raise TypeError
    return klass


def find_and_run_task(parameters, key, baseclass):
    """
    It looks in the input parameters for a class that is a subclass of the base class indicated, and it initialises it.

    Parameters
    ----------
    parameters: dict
        input dictionaty
    key: str
        string indicating the keyword for the class name

    baseclass: object
        reference class

    Returns
    -------
    callable
    """
    try:
        task = (
            find_task(parameters[key], baseclass)
            if key in parameters.keys()
            else baseclass
        )
    except UnboundLocalError as exc:
        raise Exception(
            "unable to find and instantiate a {} class".format(baseclass)
        ) from exc
    return task()
