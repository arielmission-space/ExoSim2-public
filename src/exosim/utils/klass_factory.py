import inspect
import pkgutil
from importlib import import_module

_KLASS_CACHE = {}


def find_klass_in_file(python_file, baseclass):
    """
    It finds in the indicated python file a class that is a subclass of the given one.

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

    spec = importlib.util.spec_from_file_location("foo", python_file)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    classes = [
        m[1]
        for m in inspect.getmembers(foo, inspect.isclass)
        if m[1] is not baseclass and issubclass(m[1], baseclass)
    ]

    if len(classes) == 0:
        raise ImportError(f"No class inheriting from {baseclass} in {python_file}")
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
    raise TypeError("task model in the wrong format")


def _find_class_in_module_recursive(module_path, class_name, baseclass, parent_path=""):
    """
    Recursively search for a class in a module and its submodules.

    Parameters
    ----------
    module_path: str
        The module path to search in
    class_name: str
        The name of the class to find
    baseclass: class
        The base class that the target class should inherit from
    parent_path: str
        The parent module path (used internally for recursion)

    Returns
    -------
    class or None
        The found class or None if not found
    """
    try:
        full_path = f"{parent_path}.{module_path}" if parent_path else module_path
        module = import_module(full_path)

        # Check if class exists in current module
        if hasattr(module, class_name):
            klass = getattr(module, class_name)
            if inspect.isclass(klass) and issubclass(klass, baseclass):
                return klass

        # Recursively search in submodules
        if hasattr(module, "__path__"):
            for module_info in pkgutil.iter_modules(module.__path__):
                result = _find_class_in_module_recursive(
                    module_info.name, class_name, baseclass, full_path
                )
                if result is not None:
                    return result

    except (ModuleNotFoundError, AttributeError, TypeError, ImportError):
        pass

    return None


def find_task(input, baseclass, module_path="exosim.tasks"):
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
            # Check cache first
            cache_key = (module_path, input)
            if cache_key in _KLASS_CACHE:
                return _KLASS_CACHE[cache_key]

            # Search recursively for task class by name
            klass = _find_class_in_module_recursive(module_path, input, baseclass)

            if klass is not None:
                _KLASS_CACHE[cache_key] = klass
            else:
                raise TypeError(f"Class '{input}' not found in {module_path} modules")
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
        task = find_task(parameters[key], baseclass) if key in parameters else baseclass
    except UnboundLocalError as exc:
        raise Exception(f"unable to find and instantiate a {baseclass} class") from exc
    return task()
