import logging
import os
from collections import OrderedDict

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from exosim.log import setLogLevel
from exosim.tasks.astrosignal.applyAstronomicalSignal import (
    ApplyAstronomicalSignal,
)
from exosim.tasks.astrosignal.estimatePlanetarySignal import (
    EstimatePlanetarySignal,
)
from exosim.tasks.astrosignal.findAstronomicalSignals import (
    FindAstronomicalSignals,
)

setLogLevel(logging.DEBUG)


class TestFindAstronomicalSignal:
    findAstronomicalSignal = FindAstronomicalSignals()

    def test_no_signal(self):
        parameters = {
            "source": {
                "value": "test_star",
                "source_type": "phoenix",
                "R": 1 * u.R_sun,
                "M": 1 * u.M_sun,
                "D": 10 * u.pc,
                "T": 6000 * u.K,
                "z": 0.0,
            }
        }
        out_dict = self.findAstronomicalSignal(sky_parameters=parameters)
        assert out_dict == {}

    def test_single_signal(self):
        signal = {"signal_task": EstimatePlanetarySignal}
        parameters = {
            "source": {
                "value": "test_star",
                "source_type": "phoenix",
                "R": 1 * u.R_sun,
                "M": 1 * u.M_sun,
                "D": 10 * u.pc,
                "T": 6000 * u.K,
                "z": 0.0,
                "example_signal": signal,
            }
        }
        out_dict = self.findAstronomicalSignal(sky_parameters=parameters)
        assert out_dict == {
            "test_star": {
                "example_signal": {
                    "task": EstimatePlanetarySignal,
                    "parsed_parameters": parameters["source"],
                }
            }
        }

    def test_double_signals(self):
        signal = {"signal_task": EstimatePlanetarySignal}
        parameters = {
            "source": {
                "value": "test_star",
                "source_type": "phoenix",
                "R": 1 * u.R_sun,
                "M": 1 * u.M_sun,
                "D": 10 * u.pc,
                "T": 6000 * u.K,
                "z": 0.0,
                "example_signal1": signal,
                "example_signal2": signal,
            }
        }
        out_dict = self.findAstronomicalSignal(sky_parameters=parameters)
        assert out_dict == {
            "test_star": {
                "example_signal1": {
                    "task": EstimatePlanetarySignal,
                    "parsed_parameters": parameters["source"],
                },
                "example_signal2": {
                    "task": EstimatePlanetarySignal,
                    "parsed_parameters": parameters["source"],
                },
            }
        }

    def test_double_sources_double_signals(self):
        signal = {"signal_task": EstimatePlanetarySignal}
        star1 = {
            "source_type": "phoenix",
            "R": 1 * u.R_sun,
            "M": 1 * u.M_sun,
            "D": 10 * u.pc,
            "T": 6000 * u.K,
            "z": 0.0,
            "example_signal1": signal,
        }
        star2 = {
            "source_type": "phoenix",
            "R": 1 * u.R_sun,
            "M": 1 * u.M_sun,
            "D": 10 * u.pc,
            "T": 6000 * u.K,
            "z": 0.0,
            "example_signal2": signal,
        }

        parameters = {
            "source": OrderedDict(
                {
                    "test_star1": star1,
                    "test_star2": star2,
                }
            )
        }

        with pytest.raises(ValueError):
            out_dict = self.findAstronomicalSignal(sky_parameters=parameters)

    def test_double_sources_one_signal(self):
        signal = {"signal_task": EstimatePlanetarySignal}
        star1 = {
            "source_type": "phoenix",
            "R": 1 * u.R_sun,
            "M": 1 * u.M_sun,
            "D": 10 * u.pc,
            "T": 6000 * u.K,
            "z": 0.0,
            "example_signal1": signal,
        }
        star2 = {
            "source_type": "phoenix",
            "R": 1 * u.R_sun,
            "M": 1 * u.M_sun,
            "D": 10 * u.pc,
            "T": 6000 * u.K,
            "z": 0.0,
        }

        parameters = {
            "source": OrderedDict(
                {
                    "test_star1": star1,
                    "test_star2": star2,
                }
            )
        }
        out_dict = self.findAstronomicalSignal(sky_parameters=parameters)
        assert out_dict == {
            "test_star1": {
                "example_signal1": {
                    "task": EstimatePlanetarySignal,
                    "parsed_parameters": parameters["source"]["test_star1"],
                }
            }
        }


class TestPlanetarySignal:
    # TODO include the source signal in the test
    estimatePlanetarySignal = EstimatePlanetarySignal()

    main_parameters = {
        "planet": {
            "radius": 0.1,
            "t0": 0 * u.s,
            "period": 1 * u.s,
            "sma": 15,
            "inc": 87 * u.deg,
            "ecc": 0,
            "w": 90 * u.deg,
            "limb_darkening": "linear",
            "limb_darkening_coefficients": "[0]",
        }
    }
    parameters = {}
    timeline = np.arange(-0.05, 0.05, 0.0001) * u.s
    wl_grid = np.logspace(0.5, 8, 100)

    def test_lightcurve_flat(self):
        self.main_parameters["planet"]["rp"] = 0.12
        new_timeline, model = self.estimatePlanetarySignal(
            timeline=self.timeline,
            wl_grid=self.wl_grid,
            ch_parameters=self.parameters,
            source_parameters=self.main_parameters,
        )

        batman_model_ = batman_model(
            self.main_parameters["planet"], new_timeline, rp=[0.12]
        )
        np.testing.assert_allclose(model[0], batman_model_[0])
        expected = batman_model_[0]
        batman_model_ = np.repeat(
            expected[np.newaxis, :], model.shape[0], axis=0
        )

        np.testing.assert_allclose(model, batman_model_)


# class TestApplyAstronomicalSignal:
#     applyAstronomicalSignal = ApplyAstronomicalSignal()

#     params = {'type': 'spectrometer', 'detector': {}}
#     source = {'parsed_parameters': {
#         "value": "test_star",
#         "source_type": "phoenix",
#         "R": 1 * u.R_sun,
#         "M": 1 * u.M_sun,
#         "D": 10 * u.pc,
#         "T": 6000 * u.K,
#         'planet': {
#             'radius': 0.1,
#             't0': 0 * u.s,
#             'period': 1 * u.s,
#             'sma': 15,
#             'inc': 87 * u.deg,
#             'ecc': 0,
#             'w': 90 * u.deg,
#             'limb_darkening': 'nonlinear',
#             'limb_darkening_coefficients': [0.5, 0.1, 0.1, -0.1]
#         }
#     }}

#     time = np.arange(-0.05, 0.05, 0.0001) * u.s
#     wl_grid = np.linspace(0.5, 8, 20) * u.um

#     # use a sin func to simulate a signal
#     model = np.sin(2 * np.pi * np.arange(0, 1000) / 1000)
#     model = model[np.newaxis, ...]
#     model = np.repeat(model, wl_grid.size, axis=0)
#     for i in range(wl_grid.size):
#         model[i] = model[i] * wl_grid[i].value

#     data = np.ones((1000, 10, 20))

#     # single pixel psf
#     psf = create_psf(wl=wl_grid, fnum=5, delta=18 * u.um, nzero=1, shape='airy')
#     psf = psf[np.newaxis, ...]

#     fname = os.path.join('.', "astro_test.h5")
#     output = SetOutput(fname)

#     def test_values(self):
#         with self.output.use(cache=True) as out:
#             se = Counts(
#                 spectral=self.wl_grid,
#                 time=self.time,
#                 data=self.data,
#                 shape=self.data.shape,
#                 cached=True,
#                 output=out,
#                 dataset_name="SubExposures",
#                 output_path=None,
#                 dtype=np.float64,
#                 metadata={
#                     'integration_times': np.zeros(self.time.size) * u.s,
#                     'focal_plane_time_indexes': np.zeros(self.time.size).astype(int)
#                 }
#             )

#             se = self.applyAstronomicalSignal(
#                 model=self.model,
#                 subexposures=se,
#                 psf=self.psf,
#                 timeline=self.time,
#                 source=self.source,
#                 ch_parameters=self.params,
#             )
#         print(se)

#     def test_select_chunk_range(self):
#         func = self.applyAstronomicalSignal.select_chunk_range

#         # test slice with right start and stop
#         chunk = slice(0, 100, 1)
#         new_chunk = func(chunk, 0, 100)
#         assert new_chunk == slice(0, 100, 1)

#         # test slice with start and stop inside the range
#         new_chunk = func(chunk, 50, 60)
#         assert new_chunk == slice(50, 60, 1)

#         # test slice with start and stop larger the range
#         new_chunk = func(chunk, -1, 101)
#         assert new_chunk == chunk

#         # test mixed situation
#         new_chunk = func(chunk, -1, 60)
#         assert new_chunk == slice(0, 60, 1)

#         new_chunk = func(chunk, 50, 101)
#         assert new_chunk == slice(50, 100, 1)

#         # test out of boundary
#         new_chunk = func(chunk, 101, 200)
#         assert new_chunk is None

def batman_model(parameters, input_timeline, rp=0):
    import json

    import batman

    from exosim.utils.checks import check_units

    params = batman.TransitParams()
    params.t0 = check_units(
        parameters["t0"], input_timeline.unit, None, True
    ).value
    params.per = check_units(
        parameters["period"], input_timeline.unit, None, True
    ).value
    params.a = parameters["sma"]  # (in units of stellar radii)
    params.inc = check_units(parameters["inc"], u.deg, None, True).value
    params.ecc = parameters["ecc"]
    params.w = check_units(parameters["w"], u.deg, None, True).value
    params.limb_dark = parameters["limb_darkening"]
    raw_u = parameters["limb_darkening_coefficients"]
    params.u = json.loads(raw_u)

    out_model = np.zeros((len(rp), len(input_timeline)))

    for i in range(len(rp)):
        params.rp = rp[i]  # planet radius (in units of stellar radii)

        # initialise batman model
        m = batman.TransitModel(params, input_timeline)
        out_model[i] = m.light_curve(params)

    return out_model
