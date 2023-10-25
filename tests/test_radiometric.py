import os
import unittest

import astropy.units as u
import h5py
import numpy as np
from astropy.table import hstack
from astropy.table import QTable
from inputs import test_dir

import exosim.utils.grids as grids
from exosim.models.signal import CountsPerSecond
from exosim.output import HDF5Output
from exosim.output import SetOutput
from exosim.tasks.radiometric import AperturePhotometry
from exosim.tasks.radiometric import ComputePhotonNoise
from exosim.tasks.radiometric import ComputeSignalsChannel
from exosim.tasks.radiometric import ComputeSubFrgSignalsChannel
from exosim.tasks.radiometric import ComputeTotalNoise
from exosim.tasks.radiometric import EstimateApertures
from exosim.tasks.radiometric import EstimateSpectralBinning
from exosim.tasks.radiometric import Multiaccum
from exosim.tasks.radiometric import SaturationChannel
from exosim.utils.psf import create_psf


class SpectralBinnerTest(unittest.TestCase):
    estimateSpectralBinning = EstimateSpectralBinning()

    def test_spectrometer(self):
        # fixed R
        parameters = {
            "type": "spectrometer",
            "value": "test_ch",
            "targetR": 50,
            "wl_min": 1 * u.um,
            "wl_max": 10 * u.um,
        }
        table = self.estimateSpectralBinning(parameters=parameters)
        wl_bin_c, wl_bin_width = grids.wl_grid(
            parameters["wl_min"],
            parameters["wl_max"],
            parameters["targetR"],
            return_bin_width=True,
        )
        np.testing.assert_array_equal(wl_bin_c, table["Wavelength"])

        # Native R
        parameters = {
            "type": "spectrometer",
            "value": "test_ch",
        }
        wl_grid = np.logspace(1, 10, 1000) * u.um
        table = self.estimateSpectralBinning(
            parameters=parameters, wl_grid=wl_grid
        )
        np.testing.assert_array_equal(wl_grid, table["Wavelength"])

        # wrong input
        parameters = {
            "type": "spectrometer",
            "value": "test_ch",
            "targetR": "test",
        }
        with self.assertRaises(KeyError):
            self.estimateSpectralBinning(parameters=parameters)

    def test_photometer(self):
        parameters = {
            "type": "photometer",
            "value": "test_ch",
            "wl_min": 1 * u.um,
            "wl_max": 10 * u.um,
        }
        wl_c = 5.5 * u.um
        table = self.estimateSpectralBinning(parameters=parameters)
        np.testing.assert_array_equal(wl_c, table["Wavelength"])


class EstimateAperturesTest(unittest.TestCase):
    psf = create_psf(5.5 * u.um, 15, 18 * u.um, nzero=4, shape="gauss")

    wl_grid = np.arange(0, psf.shape[1]) * u.um

    estimateSpectralBinning = EstimateSpectralBinning()

    parameters_spec = {
        "type": "spectrometer",
        "value": "test_ch",
        "targetR": 2,
        "wl_min": 1 * u.um,
        "wl_max": 17 * u.um,
    }
    table_spec = estimateSpectralBinning(parameters=parameters_spec)

    parameters_phot = {
        "type": "photometer",
        "value": "test_ch",
        "wl_min": 1 * u.um,
        "wl_max": 10 * u.um,
    }
    table_phot = estimateSpectralBinning(parameters=parameters_phot)

    estimateApertures = EstimateApertures()

    def test_rows_columns(self):
        # correct
        description = {"spectral_mode": "row", "spatial_mode": "column"}
        self.estimateApertures(
            table=self.table_phot,
            focal_plane=self.psf,
            description=description,
        )

        # wrong spatial
        with self.assertRaises(IOError):
            description = {"spectral_mode": "row", "spatial_mode": "columns"}
            self.estimateApertures(
                table=self.table_phot,
                focal_plane=self.psf,
                description=description,
            )
        # wrong spectral
        with self.assertRaises(IOError):
            description = {"spectral_mode": "rows", "spatial_mode": "column"}
            self.estimateApertures(
                table=self.table_phot,
                focal_plane=self.psf,
                description=description,
            )

    def test_full(self):
        description = {"auto_mode": "full"}
        self.estimateApertures(
            table=self.table_phot,
            focal_plane=self.psf,
            description=description,
        )

    def test_rectangular(self):
        description = {"auto_mode": "rectangular", "EnE": 0.91}
        self.estimateApertures(
            table=self.table_phot,
            focal_plane=self.psf,
            description=description,
        )

    def test_elliptical(self):
        description = {"auto_mode": "elliptical", "EnE": 0.91}
        self.estimateApertures(
            table=self.table_phot,
            focal_plane=self.psf,
            description=description,
        )

    def test_wl_solution(self):
        description = {
            "spectral_mode": "wl_solution",
            "spatial_mode": "column",
        }
        self.estimateApertures(
            table=self.table_spec,
            focal_plane=self.psf,
            description=description,
            wl_grid=self.wl_grid,
        )
        # missing input
        with self.assertRaises(IOError):
            self.estimateApertures(
                table=self.table_spec,
                focal_plane=self.psf,
                description=description,
            )
        # wrong input
        with self.assertRaises(TypeError):
            self.estimateApertures(
                table="test",
                focal_plane=self.psf,
                description=description,
                wl_grid=self.wl_grid,
            )

    def test_bin(self):
        description = {"auto_mode": "bin", "EnE": 0.91}
        self.estimateApertures(
            table=self.table_spec,
            focal_plane=self.psf,
            description=description,
            wl_grid=self.wl_grid,
        )
        # missing input
        with self.assertRaises(IOError):
            self.estimateApertures(
                table=self.table_spec,
                focal_plane=self.psf,
                description=description,
            )


class AperturePhotometryTest(unittest.TestCase):
    psf = create_psf(5.5 * u.um, 15, 18 * u.um, nzero=4, shape="gauss")

    wl_grid = np.arange(0, psf.shape[1]) * u.um

    focal_plane = CountsPerSecond(
        data=psf, spectral=wl_grid, metadata={"oversampling": 1}
    )

    parameters_phot = {
        "type": "photometer",
        "value": "test_ch",
        "wl_min": 1 * u.um,
        "wl_max": 10 * u.um,
    }
    estimateSpectralBinning = EstimateSpectralBinning()
    table_phot = estimateSpectralBinning(parameters=parameters_phot)
    table_phot["spectral_center"] = psf.shape[1] // 2
    table_phot["spectral_size"] = psf.shape[1] // 3
    table_phot["spatial_size"] = psf.shape[0] // 3
    table_phot["spatial_center"] = psf.shape[0] // 2
    table_phot["aperture_shape"] = "rectangular"

    def test_photometry(self):
        aperturePhotometry = AperturePhotometry()
        aperturePhotometry(focal_plane=self.psf, table=self.table_phot)

    def test_signal(self):
        fname = os.path.join(test_dir, "output_test.h5")
        with HDF5Output(fname) as o:
            self.focal_plane.write(o, "focal_plane")

        f = h5py.File(fname, "r")
        computeSignalsChannel = ComputeSignalsChannel()
        computeSignalsChannel(
            focal_plane=f["focal_plane"],
            table=self.table_phot,
            parameters=self.parameters_phot,
        )
        f.close()
        os.remove(fname)

    def test_subforeg(self):
        fname = os.path.join(test_dir, "output_test.h5")
        with HDF5Output(fname) as o:
            self.focal_plane.write(
                o, "channels/ch/sub_focal_planes/focal_plane"
            )
        input = SetOutput(fname, replace=False)

        computeSubFrgSignalsChannel = ComputeSubFrgSignalsChannel()
        computeSubFrgSignalsChannel(
            input_file=input,
            ch_name="ch",
            table=self.table_phot,
            parameters=self.parameters_phot,
        )
        os.remove(fname)


class SaturationTest(unittest.TestCase):
    psf = create_psf(5.5 * u.um, 15, 18 * u.um, nzero=4, shape="gauss")

    wl_grid = np.arange(0, psf.shape[1]) * u.um

    frg = CountsPerSecond(
        data=np.zeros_like(psf), spectral=wl_grid, metadata={"oversampling": 1}
    )

    focal_plane = CountsPerSecond(
        data=psf, spectral=wl_grid, metadata={"oversampling": 1}
    )

    def test_saturation_phot(self):
        parameters_phot = {
            "type": "photometer",
            "value": "ch",
            "wl_min": 1 * u.um,
            "wl_max": 10 * u.um,
            "detector": {"well_depth": 1000 * u.ct, "f_well_depth": 0.9},
        }
        estimateSpectralBinning = EstimateSpectralBinning()

        table_phot = estimateSpectralBinning(parameters=parameters_phot)
        table_phot["spectral_center"] = self.psf.shape[1] // 2
        table_phot["spectral_size"] = self.psf.shape[1] // 3
        table_phot["spatial_size"] = self.psf.shape[0] // 3
        table_phot["spatial_center"] = self.psf.shape[0] // 2
        table_phot["aperture_shape"] = "rectangular"

        fname = os.path.join(test_dir, "output_test.h5")
        with HDF5Output(fname) as o:
            self.focal_plane.write(o, "channels/ch/focal_plane")
            self.frg.write(o, "channels/ch/frg_focal_plane")
        input = SetOutput(fname, replace=False)

        saturationChannel = SaturationChannel()
        saturationChannel(
            input_file=input, table=table_phot, description=parameters_phot
        )
        os.remove(fname)

    def test_saturation_spec(self):
        parameters = {
            "type": "spectrometer",
            "value": "ch",
            "targetR": 50,
            "wl_min": 1 * u.um,
            "wl_max": 10 * u.um,
            "detector": {"well_depth": 1000 * u.ct, "f_well_depth": 0.9},
        }

        estimateSpectralBinning = EstimateSpectralBinning()
        table_spec = estimateSpectralBinning(parameters=parameters)
        description = {
            "spectral_mode": "wl_solution",
            "spatial_mode": "column",
        }

        estimateApertures = EstimateApertures()
        aper = estimateApertures(
            table=table_spec,
            focal_plane=self.psf,
            description=description,
            wl_grid=self.wl_grid,
        )
        table = hstack([table_spec, aper])

        fname = os.path.join(test_dir, "output_test.h5")
        with HDF5Output(fname) as o:
            self.focal_plane.write(o, "channels/ch/focal_plane")
            self.frg.write(o, "channels/ch/frg_focal_plane")
        input = SetOutput(fname, replace=False)

        saturationChannel = SaturationChannel()
        saturationChannel(
            input_file=input, table=table, description=parameters
        )
        os.remove(fname)


class PhotonNoiseTest(unittest.TestCase):
    signal = 4 * np.ones(10) * u.ct / u.s

    def test_simple(self):
        computePhotonNoise = ComputePhotonNoise()
        noise = computePhotonNoise(signal=self.signal)
        expected = 2 * np.ones(10) * u.ct / u.s
        np.testing.assert_array_equal(noise, expected)

    def test_margin(self):
        description = {"radiometric": {"photon_margin": 0.5}}

        computePhotonNoise = ComputePhotonNoise()
        noise = computePhotonNoise(signal=self.signal, description=description)
        expected = 2 * np.ones(10) * u.ct / u.s
        expected *= np.sqrt(1.5)
        np.testing.assert_array_equal(noise, expected)

    def test_multiaccum(self):
        multiaccum = 1.5 * np.ones(10)

        computePhotonNoise = ComputePhotonNoise()
        noise = computePhotonNoise(
            signal=self.signal, multiaccum_gain=multiaccum
        )
        expected = 2 * np.ones(10) * u.ct / u.s
        expected *= np.sqrt(1.5)
        np.testing.assert_array_equal(noise, expected)

    def test_combined(self):
        multiaccum = 1.5 * np.ones(10)
        description = {"radiometric": {"photon_margin": 0.5}}

        computePhotonNoise = ComputePhotonNoise()
        noise = computePhotonNoise(
            signal=self.signal,
            description=description,
            multiaccum_gain=multiaccum,
        )
        expected = 2 * np.ones(10) * u.ct / u.s
        expected *= 1.5
        np.testing.assert_array_equal(noise, expected)


class TotalNoiseTest(unittest.TestCase):
    table = QTable()
    table["Wavelength"] = np.linspace(1, 10, 10) * u.um

    computeTotalNoise = ComputeTotalNoise()

    def test_photon_noise(self):
        table = QTable()
        table["Wavelength"] = np.linspace(1, 10, 10) * u.um
        table["source_signal_in_aperture"] = np.ones(10) * 10000.0 * u.ct / u.s
        table["source_photon_noise"] = np.ones(10) * 100.0 * u.ct / u.s
        table["foreground_photon_noise"] = np.ones(10) * 36.0 * u.ct / u.s
        table["mirror_photon_noise"] = np.ones(10) * 16.0 * u.ct / u.s

        total_noise = self.computeTotalNoise(table=table)

        expected_noise = (
            (
                (
                    table["foreground_photon_noise"] ** 2
                    + table["source_photon_noise"] ** 2
                )
                / 3600
            )
            ** 0.5
            / table["source_signal_in_aperture"]
            * u.hr**0.5
        )
        np.testing.assert_array_equal(total_noise.value, expected_noise.value)
        self.assertEqual(total_noise.unit, expected_noise.unit)

    def test_relative_noise(self):
        table = QTable()
        table["Wavelength"] = np.linspace(1, 10, 10) * u.um
        table["source_signal_in_aperture"] = np.ones(10) * 10000.0 * u.ct / u.s
        table["gain_noise"] = 100.0
        total_noise = self.computeTotalNoise(table=table)

        expected_noise = np.ones(10) * 100.0 * u.hr**0.5
        print(total_noise, expected_noise)
        np.testing.assert_array_equal(total_noise.value, expected_noise.value)
        self.assertEqual(total_noise.unit, expected_noise.unit)


class MultiaccumTest(unittest.TestCase):
    parameteters = {"n": 1, "m": 1, "tf": 1 * u.s, "tg": 1 * u.s}

    def test_multiaccum(self):
        multiaccum = Multiaccum()
        res = multiaccum(parameters=self.parameteters)
        print(res)
