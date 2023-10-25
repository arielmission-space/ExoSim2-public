import random

import numpy as np
from numba import get_num_threads
from numba import set_num_threads

from exosim.log import Logger

total_cpus = get_num_threads()


# Singleton Meta-Class method
# https://refactoring.guru/design-patterns/singleton/python/example


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class RunConfigInit(Logger, metaclass=SingletonMeta):
    """
    Class used to propagate values through the code.
    """

    _n_job = 1
    chunk_size = 2
    _random_seed = None
    config_file_list = []

    def __init__(self):
        self.set_log_name()

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self._random_seed = seed

    @random_seed.deleter
    def random_seed(self):
        del self._random_seed

    @property
    def n_job(self):
        return self._n_job

    @n_job.setter
    def n_job(self, value):
        value = value if value > 0 else total_cpus + value
        set_num_threads(value)
        self._n_job = value

    @property
    def random_generator(self):
        """Returns a random generator with the current random seed updated by one if the random seed is set"""
        if self.random_seed is not None:
            self.random_seed += 1
        return np.random.default_rng(seed=self.random_seed)

    def __dict__(self):
        out = {}
        for a in ["n_job", "chunk_size", "random_seed", "config_file_list"]:
            out[a] = self.__getattribute__(a)
        return out

    def stats(self, log=True):
        """
        It returns a dictionary with the updated run configurations.

        Parameters
        ----------
        log: bool (optional)
            if True, it prints the run configuration stats. Default is True.

        Returns
        -------

        """
        out_dict = {
            "number of available cpus": total_cpus,
            "number of used cpus": self.n_job,
            "random seed": self.random_seed,
            "chunk size (Mb)": self.chunk_size,
        }
        if log:
            self.info("-- RUN CONFIGURATION ---------")
            for key, val in out_dict.items():
                self.info("{}: {}".format(key, val))
            self.info("------------------------------")

        return out_dict


RunConfig = RunConfigInit()
RunConfig.random_seed = random.randrange(0, 2**32 - 1)
