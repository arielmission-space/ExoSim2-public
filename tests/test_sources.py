import logging
import os
from collections import OrderedDict

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import constants as cc
from astropy.io import ascii

from exosim.log import setLogLevel
from exosim.tasks.parse import ParseSource, ParseSources
from exosim.tasks.sed import (
    CreateCustomSource,
    CreatePlanckStar,
    LoadCustom,
    LoadPhoenix,
    PrepareSed,
)

# Set logging level
setLogLevel(logging.DEBUG)


def exolib_bb_model(wl, T):
    """Blackbody model using pre-defined constants."""
    a = np.float64(1.191042768e8) * u.um**5 * u.W / u.m**2 / u.sr / u.um
    b = np.float64(14387.7516) * 1 * u.um * u.K
    x = b / (wl * T)
    bb = a / wl**5 / (np.exp(x) - 1.0)
    return bb


@pytest.mark.usefixtures("phoenix_stellar_model")
class TestLoadPhoenix:
    """Tests for loading Phoenix stellar models."""

    @pytest.fixture(autouse=True)
    def _init(self, phoenix_stellar_model):
        if not os.path.isdir(phoenix_stellar_model):
            pytest.skip("Phoenix model directory not found")
        self.loadPhoenix = LoadPhoenix()
        self.phoenix_stellar_model=phoenix_stellar_model

    def test_load_star_from_dir(self):
        D = 12.975 * u.pc
        T = 3016 * u.K
        M = 0.15 * u.Msun
        R = 0.218 * u.Rsun
        z = 0.0
        g = (cc.G * M.si / R.si**2).to(u.cm / u.s**2)
        logg = np.log10(g.value)
        sed = self.loadPhoenix(
            path=self.phoenix_stellar_model, T=T, D=D, R=R, z=z, logg=logg
        )
        assert sed is not None

    def test_compare(self,phoenix_file):
        D = 12.975 * u.pc
        T = 3016 * u.K
        M = 0.15 * u.Msun
        R = 0.218 * u.Rsun
        z = 0.0
        g = (cc.G * M.si / R.si**2).to(u.cm / u.s**2)
        logg = np.log10(g.value)

        sed_dir = self.loadPhoenix(
            path=self.phoenix_stellar_model, T=T, D=D, R=R, z=z, logg=logg
        )
        sed_file = self.loadPhoenix(filename=phoenix_file, D=D, R=R)

        np.testing.assert_array_equal(sed_dir.data, sed_file.data)

    def test_error_handling(self):
        with pytest.raises(IOError):
            self.loadPhoenix(path="invalid_path", T=3000 * u.K, D=1 * u.pc, R=1 * u.Rsun)


class TestLoadCustom:

    def test_load(self, example_dir):
        custom_file = os.path.join(example_dir, "customsed.csv")
        loadCustom = LoadCustom()
        if not os.path.isfile(custom_file):
            pytest.skip("Custom file not found")
        D = 1 * u.au
        R = 1 * u.Rsun
        sed = loadCustom(filename=custom_file, D=D, R=R)

        ph = ascii.read(custom_file, format="ecsv")
        ph_sed = ph["Sed"].data * ph["Sed"].unit
        ph_sed *= np.pi * (R.to(u.m) / D.to(u.m)) ** 2 * u.sr

        np.testing.assert_equal(sed.data_units, ph_sed.unit)
        np.testing.assert_array_equal(sed.data[0, 0], ph_sed.value)


class TestLoadSed:
    wl = np.linspace(0.5, 7.8, 10000) * u.um
    T = 5778 * u.K
    R = 1 * u.R_sun
    D = 1 * u.au

    def test_values(self):
        createPlanckStar = CreatePlanckStar()
        sed = createPlanckStar(wavelength=self.wl, T=self.T, R=self.R, D=self.D)
        omega_star = np.pi * (self.R.si / self.D.si) ** 2 * u.sr
        sed_exolib = omega_star * exolib_bb_model(self.wl, self.T)
        np.testing.assert_array_almost_equal(
            sed_exolib.value / sed.data, np.ones_like(sed.data), decimal=5
        )


class TestPlotAndBinning:
    """Tests for plotting and spectral binning."""
    def setup_method(self, skip_plot):
        if skip_plot:
            pytest.skip("Skipping plot")
        self.T = 5778 * u.K
        self.R = 1 * u.R_sun
        self.D = 1 * u.au
        self.wl = np.linspace(0.5, 7.8, 1000) * u.um

    def test_plot_planck(self):
        createPlanckStar = CreatePlanckStar()
        sed_planck = createPlanckStar(wavelength=self.wl, T=self.T, R=self.R, D=self.D)

        plt.plot(sed_planck.spectral, sed_planck.data[0, 0], label="Planck")
        sed_planck.spectral_rebin(self.wl)
        plt.plot(sed_planck.spectral, sed_planck.data[0, 0], ls=":", label="Binned Planck")
        plt.legend()
        plt.xlim(0, 8)
        plt.show()
