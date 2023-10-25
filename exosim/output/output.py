# This code is inspired by the code devolped for TauREx 3.1.
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
from abc import ABC

import numpy as np

import exosim.log as log


class Output(log.Logger):
    def __init__(self):
        super().__init__()

    def open(self):
        """It opens the output file"""
        raise NotImplementedError

    def create_group(self, group_name):
        """
        it creates an :class:`~exosim.output.output.OutputGroup`.

        Parameters
        ----------
        group_name: str
            group name

        Returns
        -------
        OutputGroup

        """
        raise NotImplementedError

    def close(self):
        """It closes the output file"""
        raise NotImplementedError

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def store_dictionary(self, dictionary, group_name=None):
        """
        it stores a full dictionary inside an :class:`~exosim.output.output.OutputGroup`.

        Parameters
        ----------
        dictionary: dict
            data to store
        group_name: str
            group name for the data
        """

        from exosim.output.utils import (
            recursively_save_dict_contents_to_output,
        )

        out = self
        if group_name is not None:
            out = self.create_group(group_name)

        recursively_save_dict_contents_to_output(out, dictionary)

    def getsize(self):
        """It returns the output file size"""
        raise NotImplementedError

    def delete_data(self, key):
        raise NotImplementedError


class OutputGroup(Output, ABC):
    def __init__(self, name):
        super().__init__(name)
        self._name = name

    def write_array(self, array_name, array, metadata=None):
        """
        Method to store :class:`~numpy.ndarray`.

        Parameters
        ----------
        array_name: str
            dataset name
        array: :class:`~numpy.ndarray`
            data to store
        metadata: dict
            metadata to attach
        """
        raise NotImplementedError

    def write_list(self, list_name, list_array, metadata=None):
        """
        Method to store lists.

        Parameters
        ----------
        list_name: str
            dataset name
        list_array: list
            data to store
        metadata: dict
            metadata to attach
        """
        arr = np.array(list_array)
        self.write_array(list_name, arr)

    def write_scalar(self, scalar_name, scalar, metadata=None):
        """
        Method to store scalars.

        Parameters
        ----------
        scalar_name: str
            dataset name
        scalar: int or float
            data to store
        metadata: dict
            metadata to attach
        """
        raise NotImplementedError

    def write_string(self, string_name, string, metadata=None, replace=False):
        """
        Method to store strings.

        Parameters
        ----------
        string_name: str
            dataset name
        string: str
            data to store
        metadata: dict
            metadata to attach
        """
        raise NotImplementedError

    def write_string_array(self, string_name, string_array, metadata=None):
        """
        Method to store :class:`~numpy.ndarray` of strings.

        Parameters
        ----------
        string_name: str
            dataset name
        string_array: :class:`~numpy.ndarray`
            data to store
        metadata: dict
            metadata to attach
        """
        raise NotImplementedError

    def write_table(self, table_name, table, metadata=None, replace=False):
        """
        Method to store :class:`~astropy.table.Table` or :class:`~astropy.table.QTable`.

        Parameters
        ----------
        table_name: str
            dataset name
        table: :class:`~astropy.table.Table` or :class:`~astropy.table.QTable`
            data to store
        metadata: dict
            metadata to attach
        replace: bool
            if ``True``, it replaces the table in the outpur file if already existing.
            Default is ``False``
        """
        raise NotImplementedError

    def write_quantity(self, quantity_name, quantity, metadata=None):
        """
        Method to store :class:`~astropy.units.Quantity`.

        Parameters
        ----------
        quantity_name: str
            dataset name
        quantity: :class:`~astropy.units.Quantity`
            data to store
        metadata: dict
            metadata to attach
        """
        raise NotImplementedError

    def create_group(self, group_name):
        """
        it creates an :class:`OutputGroup` as a subgroup

        Parameters
        ----------
        group_name: str
            group name

        Returns
        -------
        OutputGroup

        """
        raise NotImplementedError

    def delete_data(self, key):
        """
        It deletes a specific key and its associated data from the data store.

        Parameters
        ----------
        key : str
            The key identifying the data to be deleted.

        Raises
        ------
        KeyError
            If the key is not found in the data store
        """
        raise NotImplementedError
