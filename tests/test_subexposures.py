import logging
import os

import astropy.units as u
import numpy as np
import pytest

from exosim.log import setLogLevel
from exosim.models.signal import Counts, CountsPerSecond, Dimensionless
from exosim.output import SetOutput
from exosim.tasks.subexposures import (
    AddForegrounds,
    ApplyQeMap,
    ComputeReadingScheme,
    EstimateChJitter,
    EstimatePointingJitter,
    LoadQeMap,
)

setLogLevel(logging.DEBUG)


@pytest.fixture
def output_file(test_data_dir):
    fname = os.path.join(test_data_dir, "output_test.h5")
    yield fname
    if os.path.exists(fname):
        os.remove(fname)


def test_pointing_jitter_units():
    parameters = {
        "time_grid": {
            "start_time": 0.0 * u.hr,
            "end_time": 2.0 * u.hr,
            "low_frequencies_resolution": 2.0 * u.hr,
        },
        "jitter": {
            "jitter_task": "EstimatePointingJitter",
            "spatial": 0.2 * u.arcsec,
            "spectral": 0.4 * u.arcsec,
            "frequency_resolution": 0.01 * u.s,
        },
    }
    estimatePointingJitter = EstimatePointingJitter()
    jitter_spa, jitter_spe, jitter_time = estimatePointingJitter(parameters=parameters)
    assert jitter_spa.unit == "deg"
    assert jitter_spe.unit == "deg"


def test_pointing_jitter_std():
    parameters = {
        "time_grid": {
            "start_time": 0.0 * u.hr,
            "end_time": 2.0 * u.hr,
            "low_frequencies_resolution": 2.0 * u.hr,
        },
        "jitter": {
            "jitter_task": "EstimatePointingJitter",
            "spatial": 0.2 * u.arcsec,
            "spectral": 0.4 * u.arcsec,
            "frequency_resolution": 0.01 * u.s,
        },
    }
    estimatePointingJitter = EstimatePointingJitter()
    jitter_spa, jitter_spe, jitter_time = estimatePointingJitter(parameters=parameters)
    np.testing.assert_allclose(
        jitter_spa.to(u.arcsec).std().value,
        parameters["jitter"]["spatial"].to(u.arcsec).value,
        atol=0.002,
    )
    np.testing.assert_allclose(
        jitter_spe.to(u.arcsec).std().value,
        parameters["jitter"]["spectral"].to(u.arcsec).value,
        atol=0.002,
    )


def test_ch_jitter_std():
    jitter_time = np.arange(0, 60, 0.005) * u.s
    jitter_spa = np.random.normal(0, 0.01, jitter_time.size) * u.arcsec
    jitter_spe = np.random.normal(0, 0.05, jitter_time.size) * u.arcsec
    parameters = {
        "detector": {
            "plate_scale": {
                "spatial": 0.01 * u.arcsec / u.pixel,
                "spectral": 0.05 * u.arcsec / u.pixel,
            },
            "delta_pix": 0,
            "oversampling": 1,
        },
        "readout": {"readout_frequency": 200 * u.Hz},
    }
    estimateChJitter = EstimateChJitter()
    jitter_spe, jitter_spa, jit_y, jit_x, new_jit_time = estimateChJitter(
        parameters=parameters,
        pointing_jitter=(jitter_spa, jitter_spe, jitter_time),
    )
    np.testing.assert_allclose(np.std(jit_y), 1, atol=0.1)
    np.testing.assert_allclose(np.std(jit_x), 1, atol=0.1)
    np.testing.assert_allclose(np.std(jitter_spe), 0.05 * u.arcsec, atol=0.01)
    np.testing.assert_allclose(np.std(jitter_spa), 0.01 * u.arcsec, atol=0.01)


def test_add_foregrounds(output_file):
    frg = CountsPerSecond(
        spectral=np.arange(0, 10),
        data=np.ones((10, 10)) * 2,
        metadata={"oversampling": 1},
    )
    integration_time = np.ones(10) * u.s
    data = np.ones((10, 10, 10))
    output = SetOutput(output_file)
    with output.use(cache=True) as out:
        input = Counts(
            spectral=np.arange(0, 10),
            time=np.arange(0, 10) * u.hr,
            data=data,
            shape=data.shape,
            cached=True,
            output=out,
            dataset_name="SubExposures",
            output_path=None,
            dtype=np.float64,
        )
        addForegrounds = AddForegrounds()
        input = addForegrounds(
            subexposures=input,
            frg_focal_plane=frg,
            integration_time=integration_time,
        )
        np.testing.assert_array_equal(
            input.dataset[0], np.ones((10, 10), dtype=np.float64) * 3
        )
        assert np.sum(input.dataset) == 3 * 10 * 10 * 10


def test_apply_qe_map(output_file):
    qe_map = Dimensionless(
        spectral=np.arange(0, 10), data=np.ones((1, 10, 10)) * 2
    )
    data = np.ones((10, 10, 10))
    output = SetOutput(output_file)
    with output.use(cache=True) as out:
        input = Counts(
            spectral=np.arange(0, 10),
            time=np.arange(0, 10) * u.hr,
            data=data,
            shape=data.shape,
            cached=True,
            output=out,
            dataset_name="SubExposures",
            output_path=None,
            dtype=np.float64,
        )
        applyQeMap = ApplyQeMap()
        input = applyQeMap(subexposures=input, qe_map=qe_map)
        np.testing.assert_array_equal(
            input.dataset[0], np.ones((10, 10), dtype=np.float64) * 2
        )
        assert np.sum(input.dataset) == 2 * 10 * 10 * 10
