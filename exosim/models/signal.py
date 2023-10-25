from __future__ import annotations

import copy

import astropy.units as u
import numpy as np

import exosim.log as log
import exosim.utils.binning as binning
import exosim.utils.checks as checks
from exosim.models.utils.cachedData import CachedData
from exosim.output.output import Output
from exosim.utils.types import ArrayType
from exosim.utils.types import UnitType
from exosim.utils.types import ValueType

_invalid_units = "invalid units"


class Signal(log.Logger):
    """
    This class handles data cubes. The cube axes are time (0), spatial (1) and spectral (2) directions.

    Attributes
    -----------
    spectral: :class:`~numpy.ndarray`
        spectral direction grid.
    data: :class:`~numpy.ndarray`
        data table. It has 3 axes: 0. time axis, 1. spatial axis, 2. spectral axis.
    dataset: :class:`h5py.Dataset`
        If cached mode is enabled, it contains the data table. It has 3 axes: 0. time axis, 1. spatial axis, 2. spectral axis.
    time: :class:`~numpy.ndarray`
        time grid.
    spatial: :class:`~numpy.ndarray`
        spatial direction grid.
    spectral_units: str
        define the spectral direction units. Default is `um`
    data_units: str
        define the data units.
    time_units: str
        define the time direction units. Default is `hr`.
    spatial_units: str
        define the spatial direction units. Default is `um`.
    cached: bool
        it tells if cache mode is enabled.
    output: str or  :class:`~exosim.output.hdf5.hdf5.HDF5Output`
        h5py open file used for caching
    dataset_name: :class:`h5py.Dataset`
        h5py dataset used to store the data
    output_path: str
        path where is stored the dataset inside the output file.
    metadata: dict
        signal metadata attached to the class

    Notes
    -----
    To understand caching mode, please look to :class:`~exosim.models.utils.cachedData.CachedData`
    """

    def __init__(
        self,
        spectral: ArrayType = [0] * u.um,
        data: ArrayType = None,
        time: ArrayType = [0] * u.hr,
        data_units: UnitType = None,
        spatial: ArrayType = [0] * u.um,
        shape: tuple[int, int, int] = None,
        cached: bool = False,
        output: str = None,
        output_path: str = None,
        dataset_name: str = None,
        metadata: dict = None,
        dtype: np.dtype = np.float64,
    ) -> None:
        """
        Parameters
        ----------
        spectral: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity` (optional)
            wavelength grid. Must have a single axes. Default is [0] um.
            If the input data is a not :class:`~astropy.units.Quantity`, it's assumed to be expressed in microns,
            otherwhise is converted into microns or pixels.
        data: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity`
            data table. It must have 3 axes: 0. time axis, 1. spatial axis, 2. spectral axis.
            If data is :class:`~astropy.units.Quantity`, data_units are parsed automatically.
        time: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity` (optional)
            time grid. Must have a single axes. Default is [0] hr.
        spatial: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity` (optional)
            spatial direction grid. Must have a single axes.
            If the input data is a not :class:`~astropy.units.Quantity`, it's assumed to be expressed in micron.
            Default is [0] um.
        data_units: str or :class:`~astropy.units.Quantity` (optional)
            define the data units.
            If data is :class:`~astropy.units.Quantity`, data_units are parsed automatically.
        shape: tuple (optional)
            data shape. If data is None, it's used to create a data table with the desired shape.
        cached: bool (optional)
            it enable the cache mode. Default is `False`.
        output: str or :class:`~exosim.output.hdf5.hdf5.HDF5Output` (optional)
            h5 file to use for caching. If `None` a temporary file will be generated. Default is `None`.
        output_path: str (optional)
            path where to store the dataset inside the output file. Deafault is `None`.
        dataset_name: str (optional)
            name to use to store the Dataset into the h5 file named after 'output'. If `None` a random name will be generated. Default is `None`.
        metadata: dict (optional)
            dictionary of metadata.
        dtype: :class:`~numpy.dtype` (optional)
            data type of the data table. Default is `np.float64`.

        """
        self.set_log_name()
        try:
            self.spectral = checks.check_units(
                spectral, "um", self, force=True
            ).value
            self.spectral_units = u.um
        except u.UnitConversionError:
            self.spectral = checks.check_units(spectral, "pix", self).value
            self.spectral_units = u.pix

        self.time = checks.check_units(time, "hr", self, force=True).value
        self.time_units = u.hr

        try:
            self.spatial = checks.check_units(
                spatial, "um", self, force=True
            ).value
            self.spatial_units = u.um
        except u.UnitConversionError:
            self.spatial = checks.check_units(spatial, "pix", self).value
            self.spatial_units = u.pix

        # TODO set up the use of a standard Hdf5Output instead of a common file:
        #  we can use the same file use for output instead of creating a new one
        self.cached = cached
        self.shape = shape
        self.output = output
        self.output_path = output_path
        self.dataset_name = dataset_name

        if data_units is not None:
            self.data_units = u.Unit(data_units)
        elif hasattr(data, "unit"):
            self.data_units = data.unit
        else:
            self.data_units = u.Unit("")

        if cached:
            if shape is not None:
                self._data = self._prepare_cached_dataset(
                    fname=output,
                    def_name=dataset_name,
                    data=data,
                    shape=shape,
                    dtype=dtype,
                )
            elif data is not None:
                data = checks.check_units(
                    data, self.data_units, self, force=True
                ).value
                self._data = self._prepare_cached_dataset(
                    fname=output,
                    def_name=dataset_name,
                    data=data,
                    shape=data.shape,
                    dtype=data.dtype,
                )
        else:
            self._data = self._check_data_size(data)
            self._data = checks.check_units(
                self._data, self.data_units, self, force=True
            ).value
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = {}

        self.data = self._data.chunked_dataset[:] if cached else self._data
        self.dataset = self._data.chunked_dataset if cached else None

    def to(self, units: u.Unit) -> None:
        """
        It converts the Signal data into the desired units.

        Parameters
        ----------
        units: :class:`~astropy.units.Unit`
            desired unit

        Examples
        ---------
        >>> import numpy as np
        >>> import astropy.units as u
        >>> from exosim.models.signal import Signal
        >>> wl = np.linspace(0.1, 1, 10) * u.m
        >>> time_grid = np.linspace(1, 5, 10) * u.s
        >>> data = np.random.random_sample((10, 1, 10))*u.m**2
        >>> signal = Signal(spectral=wl, data=data, time=time_grid)
        >>> signal.to(u.cm**2)
        """

        self.data *= self.data_units.to(units)
        self.debug("converted data: {}".format(self.data))
        self.data_units = units

    def _check_data_size(self, data: ArrayType) -> ArrayType:
        while data.ndim < 3:
            data = np.expand_dims(data, axis=0)
        return data

    def _prepare_cached_dataset(
        self,
        fname: str,
        def_name: str,
        data: ArrayType,
        shape: tuple[int, int, int],
        dtype,
    ) -> CachedData:
        """
        If cached option is enable, this function initializes a :class:`~exosim.modules.utils.cachedData.CachedData` class.
        """
        cached_data = CachedData(
            axis0=shape[0],
            axis1=shape[1],
            axis2=shape[2],
            output=fname,
            output_path=self.output_path,
            dataset_name=def_name,
            dtype=dtype,
        )
        if data is not None:
            cached_data.chunked_dataset[:] = data

        return cached_data

    def spectral_rebin(
        self, new_wavelength: ArrayType, fill_value: ValueType = 0.0, **kwargs
    ) -> None:
        """
        It bins the class data over the spectral direction and changes the wavelegth attributes.
        This method is based on :func:`~exosim.utils.binning.rebin`.

        Parameters
        ----------
        new_wavelength: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity`
            new wavelength grid. If no units are attached is considered expressed in 'um'
        fill_value: float or :class:`~astropy.units.Quantity`
            filla value for empry bins. If no units are attached is considered expressed in 'um'.

        Examples
        ----------
        >>> from exosim.models.signal import Signal
        >>> import numpy as np
        >>> import astropy.units as u

        We first define the initial values:

        >>> wavelength = np.linspace(0.1, 1, 10) * u.um
        >>> data = np.ones((10, 1, 10))
        >>> time_grid = np.linspace(1, 5, 10) * u.hr
        >>> signal = Signal(spectral=wavelength, data=data, time=time_grid)
        >>> print(signal.data.shape)
        (10,1,10)

        We now interpolates at a finer wavelength grid:

        >>> new_wl = np.linspace(0.1, 1, 20) * u.um
        >>> signal.spectral_rebin(new_wl)
        >>> print(signal.data.shape)
        (10,1,20)

        We now bin down the to a new wavelength grid:

        >>> signal = Signal(spectral=wavelength, data=data, time=time_grid)
        >>> new_wl = np.linspace(0.1, 1, 5) * u.um
        >>> signal.spectral_rebin(new_wl)
        >>> print(signal.data.shape)
        (10,1,5)

        """
        new_wavelength_ = checks.check_units(
            new_wavelength, "um", self, force=True
        ).value
        self.data = binning.rebin(
            new_wavelength_,
            self.spectral,
            self.data,
            axis=2,
            fill_value=fill_value,
            **kwargs,
        )
        idx = np.where(np.isnan(self.data))
        self.data[idx] = 0.0
        self.spectral = new_wavelength_

    def temporal_rebin(
        self, new_time: ArrayType, fill_value="extrapolate", **kwargs
    ) -> None:
        """
        It bins the class data over the temporal direction and changes the time attributes.
        This method is based on :func:`~exosim.utils.binning.rebin`.


        Parameters
        ----------
        new_time: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity`
            new time grid. If no units are attached is considered expressed in 'hr'

        Examples
        ----------
        >>> from exosim.models.signal import Signal
        >>> import numpy as np
        >>> import astropy.units as u

        We first define the initial values:

        >>> wavelength = np.linspace(0.1, 1, 10) * u.um
        >>> data = np.ones((10, 1, 10))
        >>> time_grid = np.linspace(1, 5, 10) * u.hr
        >>> signal = Signal(spectral=wavelength, data=data, time=time_grid)
        >>> print(signal.data.shape)
        (10,1,10)

        We now interpolates at a finer time grid:

        >>> new_time = np.linspace(1, 5, 20) * u.hr
        >>> signal.temporal_rebin(new_time)
        >>> print(signal.data.shape)
        (20,1,10)

        We now bin down the to a new wavelength grid:

        >>> signal = Signal(spectral=wavelength, data=data, time=time_grid)
        >>> new_time = np.linspace(1, 5, 5) * u.hr
        >>> signal.temporal_rebin(new_time)
        >>> print(signal.data.shape)
        (5,1,10)

        """
        new_time_ = checks.check_units(new_time, "hr", self, force=True).value
        if self.time.size == 1:
            # if a rebin is called on a signal with no time dependence,
            # the data array will be repeated to match the desired size
            a = copy.deepcopy(self.data[0, :, :])
            self.data = np.repeat(a[np.newaxis, :, :], new_time.size, axis=0)

        else:
            # if the rebin is called on a signal with time dependence,
            # the data grid will be rebinned
            self.data = binning.rebin(
                new_time_,
                self.time,
                self.data,
                axis=0,
                fill_value=fill_value,
                **kwargs,
            )

        idx = np.where(np.isnan(self.data))
        self.data[idx] = 0.0
        self.time = new_time_

    # TODO if cached just rename or move. Do not write it again!
    def write(self, output: Output = None, name: str = None) -> None:
        """
        It writes the Signal class into an :class:`~exosim.output.output.Output`.
        The signal class is stored as a dictionary.

        Parameters
        ----------
        output: :class:`~exosim.output.output.Output`
            container to use to write the class

        name: str
            name to use to store

        Examples
        --------
        >>> from exosim.models.signal import Signal
        >>> import numpy as np
        >>> import astropy.units as u

        We first define the initial values:

        >>> wavelength = np.linspace(0.1, 1, 10) * u.um
        >>> data = np.ones((10, 1, 10))
        >>> time_grid = np.linspace(1, 5, 10) * u.hr
        >>> signal = Signal(spectral=wavelength, data=data, time=time_grid)

        Then we store it in a test output HDF5 file

        >>> from exosim.output.hdf5.hdf5 import HDF5Output
        >>> output = os.path.join(test_dir, 'output_test.h5')
        >>> with HDF5Output(output) as o:
        >>>     signal.write(o, 'test_signal')

        Inside the file is now stored a dictionary like the one we can obtain by

        >>> dict(signal)
        """
        if not output:
            output = self.output
        if not name:
            name = self.dataset_name

        if output is None:
            self.warning("No output indicated")
            return

        to_store = dict(self)
        if self.cached:
            to_store.pop("data", None)
        to_store["datatype"] = str(self.__class__)
        output.store_dictionary(to_store, group_name=name)
        self.debug("{} saved".format(name))

    def _find_slice(
        self, start_time: ValueType, end_time: ValueType
    ) -> tuple[int, int]:
        """This function finds a data slice given a time interval."""
        if isinstance(start_time, u.Quantity):
            start_time = start_time.to(u.hr)
        else:
            self.debug("start time assumed to be expressed in hr")
        if isinstance(end_time, u.Quantity):
            end_time = end_time.to(u.hr)
        else:
            self.debug("start time assumed to be expressed in hr")
        start = np.argmin(np.abs(self.time - start_time.value))
        stop = np.argmin(np.abs(self.time - end_time.value))
        return start, stop

    def get_slice(
        self, start_time: ValueType, end_time: ValueType
    ) -> np.ndarray:
        """
        It returnes the data relative to a time slice:

        Parameters
        ----------
        start_time: float or :class:`~astropy.units.Quantity`
            if a float is given, it's assume to be expressed in hours.

        end_time:  float or :class:`~astropy.units.Quantity`
            if a float is given, it's assume to be expressed in hours.

        Returns
        --------
        :class:`~numpy.ndarray`
        """
        start, stop = self._find_slice(start_time, end_time)
        return self.data[start:stop, :, :]

    def set_slice(
        self, start_time: ValueType, end_time: ValueType, data: ArrayType
    ) -> None:
        """
        It replaces the class data values in a specific a time slice:

        Parameters
        ----------
        start_time: float or :class:`~astropy.units.Quantity`
            if a float is given, it's assume to be expressed in hours.

        end_time:  float or :class:`~astropy.units.Quantity`
            if a float is given, it's assume to be expressed in hours.

        data: :class:`~numpy.ndarray`
            data to use as replacement for the existing ones.
        """
        start, end = self._find_slice(start_time, end_time)
        # if self.cached:
        #     self.cached_data.dataset[start:end, :, :] = data
        # else:
        self.data[start:end, :, :] = data

    # Here we define the class iterators. These are usefull to write data

    def __iter__(self):
        for a in ["data", "time", "spectral", "spatial", "metadata", "cached"]:
            yield a, self.__getattribute__(a)
        for a in [
            "data_units",
            "time_units",
            "spectral_units",
            "spatial_units",
        ]:
            yield a, str(self.__getattribute__(a))

    # Here we define the class maths.

    __numpy_ufunc__ = None  # Numpy up to 13.0
    __array_ufunc__ = None  # Numpy 13.0 and above

    def _check_other_val_sum(self, other: ArrayType) -> ArrayType:
        """extracts value and units from the other element and performs the required conversions"""
        if hasattr(other, "unit"):
            val = other.to(self.data_units).value
        elif hasattr(other, "data_units"):
            try:
                val = other.data * other.data_units.to(self.data_units)
            except TypeError:
                val = other._data * other.data_units.to(self.data_units)
        elif hasattr(other, "cached_data"):
            val = other.cached_data
        # elif hasattr(other, 'data'):
        #     val = other.data
        else:
            val = other
        return val

    def _check_other_val_product(self, other: ArrayType) -> ArrayType:
        """extracts value and units from the other element"""
        if hasattr(other, "unit"):
            return other.value, other.unit
        elif hasattr(other, "data_units"):
            return other.data, other.data_units
        else:
            units = u.Unit("")
            val = other
            return val, units

    def _create_new_instance(
        self,
        unit: UnitType = None,
        cached: bool = None,
        shape: tuple[int, int, int] = None,
        output: str | Output = None,
        output_path: str = None,
        metadata: dict = None,
        dataset_name: str = None,
    ) -> Signal:
        # check if overwrite data
        if not cached:
            cached = self.cached
        if not shape:
            shape = self.shape
        if not output:
            output = self.output
        if not output_path:
            output_path = self.output_path
        if not metadata:
            metadata = self.metadata

        # check on cached data
        # if self.cached:
        #     data = self._data.dataset
        # else:
        #
        data = self.data

        # if no unit provided use the first class unit
        if not unit:
            unit = self.data_units

        # given the unit, check the right class to instantiate
        if unit == u.W / u.m**2 / u.um:
            klass = Sed
        elif unit == u.W / u.m**2 / u.um / u.sr:
            klass = Radiance
        elif unit == u.ct / u.s:
            klass = CountsPerSecond
        elif unit == u.ct:
            klass = Counts
        elif unit == u.adu:
            klass = Adu
        elif unit == u.Unit(""):
            klass = Dimensionless
        else:
            klass = Signal
        return klass(
            spectral=self.spectral * self.spectral_units,
            data=data * unit,
            time=self.time * self.time_units,
            spatial=self.spatial * self.spatial_units,
            cached=cached,
            shape=shape,
            dataset_name=dataset_name,
            output=output,
            output_path=output_path,
            metadata=metadata,
        )

    def copy(self, **kwargs) -> Signal:
        """
        It returns a copy of the class.

        Parameters
        ----------
        kwargs: dict
            Dictionary of parameters to overwrite. The paramaters that can be included in the list are `cached`, `metadata`, `dataset_name`, 'output`, `output_path`.

        Returns
        -------
            :class:`~exosim.models.signal.Signal`

        Examples
        ----------

        >>> import numpy as np
        >>> import astropy.units as u
        >>> from exosim.models.signal import Signal
        >>> wl = np.linspace(0.1, 1, 10) * u.um
        >>> data = np.random.random_sample((10, 1, 10))
        >>> time_grid = np.linspace(1, 5, 10) * u.hr
        >>> signal = Signal(spectral=wl, data=data, time=time_grid)
        >>> signal_new = signal.copy()

        """
        return self._create_new_instance(**kwargs)

    def __add__(self, other: ArrayType | Signal | ValueType) -> Signal:
        val = self._check_other_val_sum(other)
        out_class = self._create_new_instance()
        out_class.data = self.data + val
        if isinstance(out_class.data, CachedData):
            out_class.cached = True
        return out_class

    def __radd__(self, other: ArrayType | Signal | ValueType) -> Signal:
        return self + other

    def __sub__(self, other):
        val = self._check_other_val_sum(other)
        out_class = self._create_new_instance()
        out_class.data = self.data - val
        if isinstance(out_class.data, CachedData):
            out_class.cached = True
        return out_class

    def __rsub__(self, other: ArrayType | Signal | ValueType) -> Signal:
        return (-1) * (self - other)

    def __mul__(self, other: ArrayType | Signal | ValueType) -> Signal:
        val, units = self._check_other_val_product(other)
        out_class = self._create_new_instance(unit=self.data_units * units)
        out_class.data = self.data * val
        if hasattr(out_class.data, "unit"):
            out_class.data = out_class.data.value
        return out_class

    def __rmul__(self, other: ArrayType | Signal | ValueType) -> Signal:
        return self * other

    def __truediv__(self, other: ArrayType | Signal | ValueType) -> Signal:
        val, units = self._check_other_val_product(other)
        out_class = self._create_new_instance(unit=self.data_units / units)
        out_class.data = self.data / val
        if hasattr(out_class.data, "unit"):
            out_class.data = out_class.data.value
        return out_class

    def __rtruediv__(self, other: ArrayType | Signal | ValueType) -> Signal:
        val, units = self._check_other_val_product(other)
        out_class = self._create_new_instance(unit=units / self.data_units)
        out_class.data = val / self.data
        if hasattr(out_class.data, "unit"):
            out_class.data = out_class.data.value
        return out_class

    def __floordiv__(self, other: ArrayType | Signal | ValueType) -> Signal:
        val, units = self._check_other_val_product(other)
        out_class = self._create_new_instance(unit=self.data_units / units)
        out_class.data = self.data // val
        if hasattr(out_class.data, "unit"):
            out_class.data = out_class.data.value
        return out_class

    def __rfloordiv__(self, other: ArrayType | Signal | ValueType) -> Signal:
        val, units = self._check_other_val_product(other)
        out_class = self._create_new_instance(unit=units / self.data_units)
        out_class.data = val // self.data
        if hasattr(out_class.data, "unit"):
            out_class.data = out_class.data.value
        return out_class


class Sed(Signal):
    r"""
    It's a Signal class with data having units of :math:`[W m^{-2} \mu m^{-1}]`
    """

    def __init__(
        self,
        spectral: ArrayType = [0] * u.um,
        data: ArrayType = None,
        time: ArrayType = [0] * u.hr,
        spatial: ArrayType = [0] * u.um,
        *args,
        **kwargs,
    ) -> None:
        self.set_log_name()

        if hasattr(data, "unit") and data.unit != u.W / u.m**2 / u.um:
            try:
                data = data.to(u.W / u.m**2 / u.um)
            except u.UnitConversionError:
                self.error(_invalid_units)
                raise u.UnitsError(_invalid_units)
        kwargs.pop("data_units", None)

        super().__init__(
            spectral,
            data,
            time,
            data_units=u.W / u.m**2 / u.um,
            spatial=spatial,
            *args,
            **kwargs,
        )


class Radiance(Signal):
    r"""
    It's a Signal class with data having units of :math:`[W m^{-2} \mu m^{-1} sr^{-1}]`
    """

    def __init__(
        self,
        spectral: ArrayType = [0] * u.um,
        data: ArrayType = None,
        time: ArrayType = [0] * u.hr,
        spatial: ArrayType = [0] * u.um,
        *args,
        **kwargs,
    ) -> None:
        self.set_log_name()

        if hasattr(data, "unit") and data.unit != u.W / u.m**2 / u.um / u.sr:
            try:
                data = data.to(u.W / u.m**2 / u.um / u.sr)
            except u.UnitConversionError:
                self.error(_invalid_units)
                raise u.UnitsError(_invalid_units)
        kwargs.pop("data_units", None)

        super().__init__(
            spectral,
            data,
            time,
            data_units=u.W / u.m**2 / u.um / u.sr,
            spatial=spatial,
            *args,
            **kwargs,
        )


class CountsPerSecond(Signal):
    r"""
    It's a Signal class with data having units of :math:`[ct/s]`
    """

    def __init__(
        self,
        spectral: ArrayType = [0] * u.um,
        data: ArrayType = None,
        time: ArrayType = [0] * u.hr,
        spatial: ArrayType = [0] * u.um,
        *args,
        **kwargs,
    ) -> None:
        self.set_log_name()

        if hasattr(data, "unit") and data.unit != u.ct / u.s:
            try:
                data = data.to(u.ct / u.s)
            except u.UnitConversionError:
                self.error(_invalid_units)
                raise u.UnitsError(_invalid_units)
        kwargs.pop("data_units", None)

        super().__init__(
            spectral,
            data,
            time,
            data_units=u.ct / u.s,
            spatial=spatial,
            *args,
            **kwargs,
        )


class Counts(Signal):
    r"""
    It's a Signal class with data having units of :math:`[ct]`
    """

    def __init__(
        self,
        spectral: ArrayType = [0] * u.um,
        data: ArrayType = None,
        time: ArrayType = [0] * u.hr,
        spatial: ArrayType = [0] * u.um,
        *args,
        **kwargs,
    ) -> None:
        self.set_log_name()

        if hasattr(data, "unit") and data.unit != u.ct:
            try:
                data = data.to(u.ct)
            except u.UnitConversionError:
                self.error(_invalid_units)
                raise u.UnitsError(_invalid_units)
        kwargs.pop("data_units", None)

        super().__init__(
            spectral,
            data,
            time,
            data_units=u.ct,
            spatial=spatial,
            *args,
            **kwargs,
        )


class Adu(Signal):
    r"""
    It's a Signal class with data having units of :math:`[adu/s]`
    """

    def __init__(
        self,
        spectral: ArrayType = [0] * u.um,
        data: ArrayType = None,
        time: ArrayType = [0] * u.hr,
        spatial: ArrayType = [0] * u.um,
        *args,
        **kwargs,
    ) -> None:
        self.set_log_name()

        if hasattr(data, "unit") and data.unit != u.adu:
            try:
                data = data.to(u.adu)
            except u.UnitConversionError:
                self.error(_invalid_units)
                raise u.UnitsError(_invalid_units)
        kwargs.pop("data_units", None)

        super().__init__(
            spectral,
            data,
            time,
            data_units=u.adu,
            spatial=spatial,
            *args,
            **kwargs,
        )


class Dimensionless(Signal):
    """
    It's a Signal class with data having no units
    """

    def __init__(
        self,
        spectral: ArrayType = [0] * u.um,
        data: ArrayType = None,
        time: ArrayType = [0] * u.hr,
        spatial: ArrayType = [0] * u.um,
        *args,
        **kwargs,
    ) -> None:
        self.set_log_name()

        if hasattr(data, "unit") and data.unit != u.Unit(""):
            data.to(u.Unit(""))
        kwargs.pop("data_units", None)

        super().__init__(
            spectral, data, time, u.Unit(""), spatial=spatial, *args, **kwargs
        )
