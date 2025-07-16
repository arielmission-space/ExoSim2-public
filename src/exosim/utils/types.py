from typing import TypeVar

import astropy.units as u
import numpy as np

ArrayType = TypeVar("ArrayType", np.ndarray, u.Quantity)
ValueType = TypeVar("ValueType", float, u.Quantity)
UnitType = TypeVar("UnitType", str, u.Quantity)
HDF5OutputType = TypeVar("HDF5OutputType", "HDF5Output", "HDF5OutputGroup")  # type: ignore # noqa: F821
OutputType = TypeVar("OutputType", "Output", "OutputGroup")  # type: ignore # noqa: F821
