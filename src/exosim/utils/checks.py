import astropy.units as u
import numpy as np


def check_units(input_data, desired_units, calling_class=None, force=False):
    """
    It checks the units of the inputs and returns the quantity rescaled to the desired units.

    Parameters
    ----------
    input_data: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity`
        input class to rescale.
    desired_units: :class:`~astropy.units.Quantity`
        desired unit for the input qquantity.
    calling_class: :class:`~exosim.log.logger.Logger` (optional)
        calling class. This is needed to print the eventual debug message inside the calling class.
    force: bool (optional)
        if True, if the input data has no units, it assumes is expressed in the desired units. Default is False.

    Returns
    -------
    :class:`~astropy.units.Quantity`
        scaled input quantity.

    Raises
    ------
    UnitConversionError
        if it cannot convert the original units into the desired ones
    """

    def _has_no_unit(_input_data, _desired_units, _force):
        """If input has no unit, it assigns the desired one, if force is True,
        or just convert the input data to a :class:`~astropy.units.Quantity` with no unit.
        """
        if calling_class:
            calling_class.debug("input data has no unit")
            calling_class.debug(f"input data type: {type(_input_data)}")
            calling_class.debug(f"input data value: {_input_data!r}")

        # Try to convert to a numeric type if it's a string
        if isinstance(_input_data, str):
            try:
                _input_data = float(_input_data)
                if calling_class:
                    calling_class.debug(f"converted string to float: {_input_data}")
            except ValueError:
                if calling_class:
                    calling_class.debug(
                        f"failed to convert string to float: {_input_data!r}"
                    )
                # If it fails, leave it as is and let numpy handle it

        if _force:
            if calling_class:
                calling_class.debug(f"forcing {_desired_units} units")
            _input_data = np.array(_input_data) * u.Unit(_desired_units)
        else:
            if calling_class:
                calling_class.debug("forcing no units")
            _input_data = np.array(_input_data) * u.Unit("")
        return _input_data

    def _has_different_unit(_input_data, _desired_units):
        """If input has different unit, it converts the input data into the desired one."""
        try:
            output_data = _input_data.to(u.Unit(_desired_units))
            if calling_class:
                calling_class.debug(f"converted {_input_data.unit} to {_desired_units}")
                calling_class.debug(f"converted data: {output_data}")
            return output_data

        except u.UnitConversionError:
            if (_input_data.unit == u.s and _desired_units == u.Hz) or (
                _input_data.unit == u.Hz and _desired_units == u.s
            ):
                return check_units(1 / _input_data, _desired_units)
            msg = f"impossible to convert {_input_data.unit} into {_desired_units}"
            raise u.UnitConversionError(msg) from None

    if desired_units is None:
        desired_units = ""

    if not hasattr(input_data, "unit") or input_data.unit is None:
        input_data = _has_no_unit(input_data, desired_units, force)
    else:
        input_unit = input_data.unit
        if input_unit == 1 / u.Hz:
            input_unit = u.s
        if input_unit == 1 / u.s:
            input_unit = u.Hz
        if not input_unit:
            input_unit = u.Unit("")
        input_data = np.array(input_data) * u.Unit(input_unit)

    if calling_class:
        calling_class.debug(f"input data: {input_data}")

    if input_data.unit != u.Unit(desired_units):
        return _has_different_unit(input_data, desired_units)

    return input_data


def find_key(input_class_keys, key_list, calling_class=None):
    """
    Finds which key from key_list is contained in input_class_keys, ignoring case.

    Parameters
    ----------
    input_class_keys : list
        List of the input class keys.
    key_list : list
        List of the desired keys.
    calling_class : :class:`~exosim.log.logger.Logger`, optional
        Calling class. This is needed to print the eventual debug message inside the calling class.

    Returns
    -------
    str
        Found key.

    Raises
    -------
    KeyError
        If no matching key is found.
    """
    # Map lower-case key -> original key
    key_map = {key.lower(): key for key in input_class_keys}

    try:
        return next(key_map[k.lower()] for k in key_list if k.lower() in key_map)
    except StopIteration:
        msg = "no matching key found"
        if calling_class:
            calling_class.error(msg)
        raise KeyError(msg) from None


def look_for_key(input_dict, key, value, foo=False):
    """
    Returns ``True`` if a certain key is in the dictionary and has a certain value.

    Parameters
    ----------
    input_dict: dict
        input dictionary
    key: str
        key to search
    value
        key content to check

    Returns
    -------
    bool

    """
    for k, val in input_dict.items():
        if k == key and val == value:
            foo = True
            break
        if isinstance(input_dict[k], dict):
            foo = look_for_key(input_dict[k], key, value, foo)
    return foo
