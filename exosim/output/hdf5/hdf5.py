# This code is inspired by the code developed for TauREx 3.1.
# Therefore, we attach here TauREx3.1 license:
#
# BSD 3-Clause License
#
# Copyright (c) 2019, Ahmed F. Al-Refaie, Quentin Changeat, Ingo Waldmann, Giovanna Tinetti
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the names of the copyright holders nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import datetime
import os
from typing import List
from typing import Union

import astropy.units as u
import h5py
import numpy as np
from astropy.table import meta
from astropy.table import QTable
from astropy.table import Table

import exosim.output.output as output
from exosim import __author__
from exosim import __branch__
from exosim import __citation__
from exosim import __commit__
from exosim import __copyright__
from exosim import __license__
from exosim import __pkg_name__
from exosim import __title__
from exosim import __url__
from exosim import __version__
from exosim.utils.runConfig import RunConfig

META_KEY = "__table_column_meta__"


class HDF5OutputGroup(output.OutputGroup):
    def __init__(
        self,
        entry: h5py.Group,
        fd: h5py.File,
        cache: bool = False,
        filename: str = None,
    ) -> None:
        self.set_log_name()
        self.fd = fd
        self._entry = entry
        self._cache = cache
        self.filename = filename

    def write_array(
        self, array_name: str, array: np.ndarray, metadata: dict = None
    ) -> None:
        if isinstance(array, list):
            for idx, a in enumerate(array):
                self.write_array("{}{}".format(array_name, idx), a, metadata)
            return
        ds = self._entry.create_dataset(
            str(array_name), data=array, shape=array.shape, dtype=array.dtype
        )

        ds.attrs["datatype"] = str(array.__class__)
        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = v

    def write_table(
        self,
        table_name: str,
        table: Union[Table, QTable],
        metadata: dict = None,
        replace: bool = False,
    ) -> None:
        if replace:
            if str(table_name) in self._entry.keys():
                del self._entry[str(table_name)]
                try:
                    del self._entry[str(table_name) + "." + META_KEY]
                except KeyError:
                    pass
                try:
                    del self._entry["{}_to_group".format(str(table_name))]
                except KeyError:
                    pass

        table = _encode_mixins(table)
        if any(col.info.dtype.kind == "U" for col in table.itercols()):
            table = table.copy(copy_data=False)
            table.convert_unicode_to_bytestring()

        self._entry.create_dataset(str(table_name), data=table.as_array())
        header_yaml = meta.get_yaml_from_table(table)

        header_encoded = [h.encode("utf-8") for h in header_yaml]
        self._entry.create_dataset(
            str(table_name) + "." + META_KEY, data=header_encoded
        )

        tg = self._entry.create_group("{}_to_group".format(str(table_name)))
        for col in table.keys():
            tg_c = tg.create_group(str(col))
            tg_c.create_dataset("value", data=table[col])
            if table[col].unit is not None:
                tg_c.create_dataset("unit", data=str(table[col].unit))

        tg.attrs["datatype"] = str(table.__class__)
        if metadata:
            for k, v in metadata.items():
                tg.attrs[k] = v
            for k, v in table.meta.items():
                tg.attrs[k] = v

    def write_scalar(
        self, scalar_name: str, scalar: float, metadata: dict = None
    ) -> None:
        ds = self._entry.create_dataset(str(scalar_name), data=scalar)

        ds.attrs["datatype"] = str(scalar.__class__)
        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = v

    def write_string(
        self, string_name: str, string: str, metadata: dict = None
    ) -> None:
        # h5py new string hanlder support bites. So we convert strings into bites
        string = string.encode("utf-8")
        ds = self._entry.create_dataset(str(string_name), data=string)

        ds.attrs["datatype"] = str(string.__class__)
        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = v

    def create_group(self, group_name: str) -> "HDF5OutputGroup":
        """
        it creates an :class:`~exosim.output.hdf5.hdf5.HDF5OutputGroup`.

        Parameters
        ----------
        group_name: str
            group name

        Returns
        -------
        OutputGroup

        """
        entry = None
        if self._entry:
            try:
                entry = self._entry.create_group(str(group_name))
            except ValueError:
                entry = self._entry[str(group_name)]
        return HDF5OutputGroup(
            entry, fd=self.fd, cache=self._cache, filename=self.filename
        )

    def delete_data(self, key):
        try:
            del self._entry[key]
        except KeyError:
            self.error("key {} not found".format(key))
            raise KeyError

    def flush(self) -> None:
        self.fd.flush()

    def write_string_array(
        self, string_name: str, string_array: List[str], metadata: dict = None
    ) -> None:
        string_array = string_array.astype("S64")
        ds = self._entry.create_dataset(str(string_name), data=string_array)

        ds.attrs["datatype"] = str(string_array.__class__)
        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = v

    def write_quantity(
        self, quantity_name: str, quantity: u.Quantity, metadata: dict = None
    ) -> None:
        if quantity_name == "value":
            qg_c = self._entry.create_dataset(
                "value",
                data=quantity.value,
            )
        else:
            qg_c = self._entry.create_group(str(quantity_name))
            qg_c.create_dataset(
                "value",
                data=quantity.value,
            )
            qg_c.create_dataset("unit", data=str(quantity.unit))
            pass
        qg_c.attrs["datatype"] = str(quantity.__class__)

        if metadata:
            for k, v in metadata.items():
                qg_c.attrs[k] = v


def _encode_mixins(tbl):
    from astropy.table import serialize
    from astropy.utils.data_info import serialize_context_as

    with serialize_context_as("hdf5"):
        encode_tbl = serialize.represent_mixins_as_columns(tbl)

    return encode_tbl


class HDF5Output(output.Output):
    def __init__(self, filename, append=False, cache=False):
        super().__init__()
        self.filename = filename
        self._append = append
        self._cache = cache
        self.fd = None

    def open(self) -> None:
        self.fd = self._openFile(self.filename)
        self.debug("opened {}".format(self.filename))

    def _openFile(self, fname: str) -> h5py.File:
        mode = "w"
        if self._append:
            mode = "a"

        rdcc_w0 = None
        if self._cache:
            rdcc_w0 = 1

        attrs = {
            "file_name": fname,
            "file_time": datetime.datetime.now().isoformat(),
            "creator": self.__class__.__name__,
            "HDF5_Version": h5py.version.hdf5_version,
            "h5py_version": h5py.version.version,
            "program_name": str(__title__),
            "package name": str(__pkg_name__),
            "program_version": str(__version__),
            "author": str(__author__),
            "copyright": str(__copyright__),
            "license": str(__license__),
            "url": str(__url__),
            "citation": str(__citation__),
            "git commit": str(__commit__),
            "git branch": str(__branch__),
        }

        fd = h5py.File(fname, mode=mode, rdcc_w0=rdcc_w0)
        for key in attrs:
            fd.attrs[key] = attrs[key]

        if mode == "w" or "info" not in fd.keys():
            try:
                gd_ = fd["info"]
            except KeyError:
                gd_ = fd.create_group("info")
            gd = HDF5OutputGroup(gd_, fd)
            gd.store_dictionary(attrs, "exosim")
            gd.store_dictionary(RunConfig.__dict__(), "RunConfig")

        fd.flush()

        return fd

    def add_info(self, attrs: dict, name: str = None) -> None:
        gd = self.create_group("info")
        gd.store_dictionary(attrs, name)

    def flush(self) -> None:
        self.fd.flush()

    def create_group(self, group_name: str) -> "HDF5OutputGroup":
        """
        it creates an :class:`~exosim.output.hdf5.hdf5.HDF5OutputGroup`.

        Parameters
        ----------
        group_name: str
            group name

        Returns
        -------
        OutputGroup

        """

        entry = None
        if self.fd:
            try:
                entry = self.fd.create_group(str(group_name))
            except ValueError:
                entry = self.fd[str(group_name)]
        return HDF5OutputGroup(
            entry, cache=self._cache, filename=self.filename, fd=self.fd
        )

    def store_dictionary(
        self, dictionary: dict, group_name: str = None
    ) -> None:
        """
        it stores a full dictionary inside the :class:`HDF5Output`, forcing the flush.

        Parameters
        ----------
        dictionary: dict
            data to store
        group_name: str (optional)
            group name for the data
        """

        from exosim.output.utils import (
            recursively_save_dict_contents_to_output,
        )

        out = self
        if group_name is not None:
            out = self.create_group(group_name)

        recursively_save_dict_contents_to_output(out, dictionary)

        self.fd.flush()

    def close(self) -> None:
        if self.fd:
            self.fd.flush()
            self.fd.close()
            self.debug("closed {}".format(self.filename))

    def getsize(self) -> u.Quantity:
        """It returns the output file size"""
        size = os.path.getsize(self.filename) * u.b
        if size.to(u.Tb).value > 1.0:
            size = size.to(u.Tb)
        elif size.to(u.Gb).value > 1.0:
            size = size.to(u.Gb)
        elif size.to(u.Mb).value > 1.0:
            size = size.to(u.Mb)
        return size

    def delete_data(self, key):
        try:
            del self.fd[key]
        except KeyError:
            self.error("key {} not found".format(key))
            raise KeyError
