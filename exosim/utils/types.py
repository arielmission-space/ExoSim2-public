from typing import TypeVar

import astropy.units as u
import numpy as np

from exosim.output.hdf5 import HDF5Output
from exosim.output.hdf5 import HDF5OutputGroup
from exosim.output.output import Output
from exosim.output.output import OutputGroup

ArrayType = TypeVar("ArrayType", np.ndarray, u.Quantity)
ValueType = TypeVar("ValueType", float, u.Quantity)
UnitType = TypeVar("UnitType", str, u.Quantity)
HDF5OutputType = TypeVar("HDF5OutputType", HDF5Output, HDF5OutputGroup)
OutputType = TypeVar("OutputType", Output, OutputGroup)
