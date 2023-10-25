import os
import tempfile
from pathlib import Path

import h5py

import exosim.log as log
from .hdf5 import HDF5Output


class SetOutput(log.Logger):
    """
    It sets the output for the code.
    This class created and initializes the output file.
    If a file name is provided, it loads the relative :class:`~exosim.output.output.Output` class is instatiated.
    Otherwise an :class:`~exosim.output.hdf5.hdf5.HDF5Output` is used by default for a temporary file.
    """

    def __init__(self, filename: str = None, replace: bool = True):
        """

        Parameters
        ----------
        filename: str (optional)
            output file name for. If `None` a temporary file is produced.
        """
        super().__init__()

        self.tmp = False
        if filename is None:
            path = Path(os.path.join(os.getcwd(), "tmp"))
            tempfile.tempdir = path
            tempfile.tempdir.mkdir(parents=True, exist_ok=True)
            filename = tempfile.NamedTemporaryFile(
                suffix=".h5", delete=False
            ).name
            self.tmp = True
        self.info("file name set: {}".format(filename))
        dir_name = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            self.debug("created {}".format(dir_name))

        self.fname = filename
        if replace:
            if os.path.exists(filename):
                os.remove(filename)

    def use(self, append: bool = True, cache: bool = False) -> HDF5Output:
        """
        It returns the :class:`~exosim.output.output.Output` with file opened and ready to write

        Parameters
        ----------
        append: bool (optional)
            True to append data to already existing file. Default is True.
        cache: bool (optional)
            True to write data in caching mode. Default is False.

        Returns
        -------
        :class:`~exosim.output.output.Output`
            output class instantiated.

        """
        if self.fname.endswith(".h5"):
            return HDF5Output(self.fname, append, cache)

    def open(self) -> h5py.File:
        """
        It returns the :class:`~exosim.output.output.Output` with file opened and ready to read

        Returns
        -------
        :class:`~exosim.output.output.Output`
            output class instantiated.
        """

        if self.fname.endswith(".h5"):
            f = HDF5Output(self.fname, append=True)
            f.open()
            return f.fd

    def __del__(self) -> None:
        """
        Garbage collector: it deletes the file when not in use by the context.
        """

        if self.tmp:
            self.delete()

    def delete(self) -> None:
        """
        It deletes the output file created.
        """

        try:
            os.remove(self.fname)
            self.debug("file deleted: {}".format(self.fname))
        except FileNotFoundError:
            pass

        # check if the file has been correctly removed
        if Path(self.fname).is_file():
            self.warning("file not deleted: {}".format(self.fname))

        # if temp dir is empty, delete it
        try:
            if not any(tempfile.tempdir.iterdir()):
                os.rmdir(tempfile.tempdir)
                self.debug(
                    "temporary dir deleted: {}".format(tempfile.tempdir)
                )
        except FileNotFoundError:
            pass
        except FileNotFoundError:
            pass
        except AttributeError:
            pass
