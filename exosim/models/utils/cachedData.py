import os
import tempfile
import uuid
from pathlib import Path
from typing import Union

import h5py
import numpy as np

import exosim.log as log
from exosim.output import HDF5Output
from exosim.output import HDF5OutputGroup
from exosim.utils import RunConfig
from exosim.utils.types import HDF5OutputType


class CachedData(log.Logger):
    """
    This class caches data cube into an h5 file. The cube data are chunked toward the first axis.
    In this class are also defined a set of operation to operate on the dataset using the chinks system.

    Attributes
    ----------
    axis0: int
        first axis size
    axis1: int
        second axis size
    axis2: int
        third axis size
    output: str or :class:`~exosim.output.HDF5Output`
        name of the file used for caching.
    dataset_name: str
        name used to store the dataset into the h5 file.
    output: :class:`h5py.File` or  :class:`~exosim.output.hdf5.hdf5.HDF5Output` or :class:`~exosim.output.hdf5.hdf5.HDF5OutputGroup`
        h5py open file used for caching
    dataset_path: str
         path where is stored the dataset inside the output file.
    chunked_dataset: :class:`h5py.Dataset`
        h5py dataset used to store the data

    Notes
    -----
    The cached data may be stored in a temporary file. To delete temporary files we included a garbage collector.
    Please, remember to delete the class when done as in the following example

    >>> myClass = CachedData(1,1,1)
    >>> del myClass
    """

    def __init__(
        self,
        axis0: int,
        axis1: int,
        axis2: int,
        output: Union[str, HDF5OutputType] = None,
        output_path: str = None,
        dataset_name: str = None,
        dtype: np.dtype = np.float64,
    ) -> None:
        """
        Parameters
        ----------
        axis0: int
            first axis size
        axis1: int
            second axis size
        axis2: int
            third axis size
        output: str or :class:`~exosim.output.hdf5.hdf5.HDF5Output` or :class:`~exosim.output.hdf5.hdf5.HDF5OutputGroup`
            file name to use for caching. If `None` a temporary file will be generated. Default is `None`.
        output_path: str (optional)
            path where to store the dataset inside the output file. Default is `None`.
        dataset_name: str
            name to use to store the dataset into the h5 file. If `None` a random name will be generated. Default is `None`.
        """
        self.set_log_name()

        self.axis0 = axis0
        self.axis1 = axis1
        self.axis2 = axis2

        # we want to get a name for the dataset to store it in the h5 file
        if dataset_name is None:
            dataset_name = str(uuid.uuid4())[:8]
        self.dataset_name = dataset_name
        self.debug("data will be stored as: {}".format(self.dataset_name))

        # if a file name is given, we use that as output, otherwise we use a temporary file
        self.tmp = False
        if output:
            # if is a string or an already existing temporary file, we open it as an h5py
            if isinstance(output, str):
                self.output = h5py.File(output, "a", rdcc_w0=1)
                self.fname = output
            elif isinstance(output, tempfile._TemporaryFileWrapper):
                self.output = h5py.File(output, "a", rdcc_w0=1)
                self.fname = output
                self.tmp = True
            # if it is an Output class, we use it
            elif isinstance(output, HDF5Output):
                if not output._cache:
                    self.error("output file not set for caching")
                    raise OSError("output file not set for caching")
                self.fname = output.filename
                self.output = output.fd
            elif isinstance(output, HDF5OutputGroup):
                if not output._cache:
                    self.error("output file not set for caching")
                    raise OSError("output file not set for caching")
                self.fname = output.filename
                self.output = output._entry

            else:
                self.error("unsupported output format")
                raise OSError("unsupported output format")

            self.debug("data stored in: {}".format(self.fname))
        else:
            path = Path(os.path.join(os.getcwd(), "tmp"))
            tempfile.tempdir = path
            tempfile.tempdir.mkdir(parents=True, exist_ok=True)
            self.fname = tempfile.NamedTemporaryFile(
                suffix=".h5", delete=False
            )
            self.output = h5py.File(self.fname, "a", overrite=True, rdcc_w0=1)
            self.tmp = True
            self.debug("temporary file created: {}".format(self.fname.name))

        # we define the address of the dataset inside the file
        self.output_path = output_path
        if self.output_path:
            self.dataset_path = os.path.join(output_path, self.dataset_name)
        else:
            self.dataset_path = self.dataset_name + "/data"

        # chunk size fixed to 2Mb
        mem_size = RunConfig.chunk_size * 1e6  # convert to Mbs
        image_size = axis1 * axis2 * np.dtype(dtype).itemsize
        axis0_chunk = (
            int(mem_size // image_size)
            if int(mem_size // image_size) < axis0
            else axis0
        )
        # we create an empty dataset (full of zeros) of a given shape, chunked of the first axis
        self.chunked_dataset = self.output.create_dataset(
            self.dataset_path,
            shape=(axis0, axis1, axis2),
            chunks=(axis0_chunk, axis1, axis2),
            dtype=dtype,
            compression=None,
        )

    def rename_dataset(self, new_name: str) -> None:
        """It renames the dataset in the HDF5 file.

        Parameters
        -----------
        new_name: str
           new name for the dataset
        """
        self.output[new_name] = self.output[self.dataset_name]
        del self.output[self.dataset_name]
        self.dataset_name = new_name

    # to perform any operation, we iterate over the chunks built on the first axis
    # def _get_val_in_slice(self, other, i):
    #     """Returs the values contained in chunk built on the first axis."""
    #     if hasattr(other, 'dataset'):
    #         return other.dataset[i, :, :]
    #     elif hasattr(other, 'shape'):
    #         return other[i, :, :]
    #     else:
    #         return np.ones(self.dataset[i, :, :].shape) * other
    #
    # def _create_new_instance(self):
    #     return CachedData(*self.dataset.shape, output=self.fname,
    #                       output_path=self.output_path)
    #
    # @property
    # def dataset(self):
    #     """h5py dataset used to store the data"""
    #     return self._dataset
    #
    # @dataset.setter
    # def dataset(self, other):
    #     """ This setter implements the slicing/caching system"""
    #     self._dataset = other
    #     self.output.flush()
    #
    # @dataset.deleter
    # def dataset(self):
    #     """It empties the dataset"""
    #     del self._dataset

    def __del__(self):
        """Garbage collector"""
        # close the file

        # try to flush and close the output file
        if hasattr(self.output, "flush"):
            try:
                self.output.flush()
            except ValueError:
                pass
        if hasattr(self.output, "close"):
            self.output.close()

        if self.tmp:
            # remove temp file if exist
            try:
                os.remove(self.fname)
                self.debug("file deleted: {}".format(self.fname))
            except TypeError:
                os.remove(self.fname.name)
                self.debug(
                    "temporary file deleted: {}".format(self.fname.name)
                )
            except FileNotFoundError:
                pass

            # check if the file has been correctly removed
            try:
                if Path(self.fname).is_file():
                    self.warning("file not deleted: {}".format(self.fname))
            except TypeError:
                if Path(self.fname.name).is_file():
                    self.warning(
                        "file not deleted: {}".format(self.fname.name)
                    )

            # if temp dir is empty, delete it
            try:
                if not any(tempfile.tempdir.iterdir()):
                    os.rmdir(tempfile.tempdir)
                    self.debug(
                        "temporary dir deleted: {}".format(tempfile.tempdir)
                    )
            except FileNotFoundError:
                pass
            except AttributeError:
                pass
