"""
Behavioural tests for the preparation tools that build detector maps and
non-linearity coefficients: they take a configuration dictionary and produce a
map/array whose statistics or closed form can be checked directly.
"""

from math import factorial

import numpy as np
import pytest
from astropy import units as u

from exosim.tools.dark_current_map import DarkCurrentMap
from exosim.tools.pixels_non_linearity import PixelsNonLinearity
from exosim.tools.pixels_non_linearity_from_correction import (
    PixelsNonLinearityFromCorrection,
)
from exosim.tools.quantum_efficiency_map import QuantumEfficiencyMap
from exosim.utils import RunConfig


def _time_grid_cfg():
    return {
        "start_time": 0.0 * u.hr,
        "end_time": 5.0 * u.hr,
        "low_frequencies_resolution": 1.0 * u.hr,
    }


@pytest.fixture(autouse=True)
def _restore_run_config():
    from exosim.utils.run_config import RunConfig

    seed = RunConfig.random_seed
    files = list(RunConfig.config_file_list)
    try:
        yield
    finally:
        RunConfig.random_seed = seed
        RunConfig.config_file_list = files


class TestDarkCurrentMap:
    """DarkCurrentMap draws a per-pixel dark current from N(dc_mean, dc_sigma)."""

    def _detector(self, **over):
        det = {
            "dc_mean": 5.0 * u.ct / u.s,
            "dc_sigma": 1.0 * u.ct / u.s,
            "spatial_pix": 40,
            "spectral_pix": 40,
            "oversampling": 1,
        }
        det.update(over)
        return {"detector": det}

    def test_map_statistics_match_configuration(self):
        RunConfig.random_seed = 42
        dc_map = DarkCurrentMap()(parameters=self._detector(), time=[0.0] * u.hr)
        data = dc_map.data
        assert data.shape == (1, 40, 40)
        assert np.mean(data) == pytest.approx(5.0, abs=0.2)
        assert np.std(data) == pytest.approx(1.0, abs=0.2)
        assert np.all(data > 0)  # non-positive draws are clipped to ~0

    def test_oversampling_repeats_pixels(self):
        RunConfig.random_seed = 1
        dc_map = DarkCurrentMap()(
            parameters=self._detector(oversampling=3, spatial_pix=10, spectral_pix=10),
            time=[0.0] * u.hr,
        )
        assert dc_map.data.shape == (1, 30, 30)

    def test_missing_dc_sigma_raises(self):
        det = self._detector()
        del det["detector"]["dc_sigma"]
        with pytest.raises(KeyError, match="dc_sigma"):
            DarkCurrentMap()(parameters=det, time=[0.0] * u.hr)

    def test_aging_adds_time_dimension(self):
        RunConfig.random_seed = 7
        det = self._detector(dc_aging_factor=0.1, dc_aging_time_scale=5.0 * u.hr)
        time = np.array([0.0, 2.5, 5.0]) * u.hr
        dc_map = DarkCurrentMap()(parameters=det, time=time)
        assert dc_map.data.shape[0] == 3

    def test_dc_median_is_converted_to_a_mean(self):
        RunConfig.random_seed = 3
        det = self._detector()
        del det["detector"]["dc_mean"]
        det["detector"]["dc_median"] = 4.0 * u.ct / u.s
        dc_map = DarkCurrentMap()(parameters=det, time=[0.0] * u.hr)
        # log-normal mean sits above the median
        assert np.mean(dc_map.data) > 3.5

    def test_map_is_written_to_the_output(self, tmp_path):
        import h5py

        from exosim.output import SetOutput

        RunConfig.random_seed = 5
        fname = tmp_path / "dc.h5"
        with SetOutput(str(fname)).use(append=True) as out:
            DarkCurrentMap()(parameters=self._detector(), time=[0.0] * u.hr, output=out)
        with h5py.File(fname, "r") as f:
            assert "dc_map" in f


class TestQuantumEfficiencyMap:
    """QuantumEfficiencyMap draws a per-pixel QE around 1 with spread qe_sigma."""

    def _cfg(self, **det_over):
        det = {"qe_sigma": 0.05, "spatial_pix": 32, "spectral_pix": 32}
        det.update(det_over)
        return {
            "time_grid": _time_grid_cfg(),
            "channel": {"value": "ch", "detector": det},
        }

    def test_map_centered_on_unity(self):
        RunConfig.random_seed = 11
        tool = QuantumEfficiencyMap(self._cfg())
        qe = tool.results["ch"].data
        assert np.mean(qe) == pytest.approx(1.0, abs=0.02)
        assert np.std(qe) == pytest.approx(0.05, abs=0.02)

    def test_aging_produces_time_evolution(self):
        RunConfig.random_seed = 3
        cfg = self._cfg(qe_aging_factor=0.02, qe_aging_time_scale=5.0 * u.hr)
        tool = QuantumEfficiencyMap(cfg)
        qe = tool.results["ch"]
        assert qe.data.shape[0] == qe.time.size
        # ageing only lowers QE: later map has mean <= initial map
        assert np.mean(qe.data[-1]) <= np.mean(qe.data[0]) + 1e-6


class TestPixelsNonLinearity:
    """The non-linearity coefficients follow the Taylor expansion of the model."""

    def test_coefficients_match_taylor_expansion(self):
        q_wd = 25000.0
        cfg = {
            "channel": {
                "value": "ch",
                "detector": {
                    "well_depth": q_wd * u.ct,
                    "spatial_pix": 8,
                    "spectral_pix": 8,
                },
            }
        }
        tool = PixelsNonLinearity(cfg, show_results=False)
        coeff = tool.results["ch"]["coeff"]
        constant = 0.103479 / q_wd
        expected = [1.0] + [
            (-1) ** i / factorial(i + 1) * constant**i for i in range(1, 5)
        ]
        np.testing.assert_allclose(coeff, expected, rtol=1e-9)
        assert tool.results["ch"]["saturation"] == pytest.approx(q_wd)

    def test_show_results_path_runs(self):
        # show_results=True exercises the printing + plotting branch
        import matplotlib as mpl

        mpl.use("Agg")
        from unittest.mock import patch

        cfg = {
            "channel": {
                "value": "ch",
                "detector": {
                    "well_depth": 25000 * u.ct,
                    "spatial_pix": 4,
                    "spectral_pix": 4,
                },
            }
        }
        with patch("matplotlib.pyplot.show"):
            tool = PixelsNonLinearity(cfg, show_results=True)
        assert "coeff" in tool.results["ch"]

    def test_map_has_detector_shape(self):
        cfg = {
            "channel": {
                "value": "ch",
                "detector": {
                    "well_depth": 20000 * u.ct,
                    "spatial_pix": 6,
                    "spectral_pix": 10,
                    "pnl_coeff_std": 0.01,
                },
            }
        }
        tool = PixelsNonLinearity(cfg, show_results=False)
        pnl_map = np.asarray(tool.results["ch"]["map"])
        # a coefficient map over the detector: last two axes are the pixel grid
        assert pnl_map.shape[-2:] == (6, 10)


class TestPixelsNonLinearityFromCorrection:
    """Recover the a_i coefficients from measured b_i correction coefficients."""

    def _cfg(self, **det_over):
        det = {
            "well_depth": 25000 * u.ct,
            "spatial_pix": 6,
            "spectral_pix": 6,
            "pnl_coeff_a": 1.00117667e00,
            "pnl_coeff_b": -5.41836850e-07,
            "pnl_coeff_c": 4.57790820e-11,
            "pnl_coeff_d": 7.66734616e-16,
            "pnl_coeff_e": -2.32026578e-19,
            "pnl_correction_operator": "/",
        }
        det.update(det_over)
        return {"channel": {"value": "ch", "detector": det}}

    def test_recovers_fourth_order_coefficients(self):
        tool = PixelsNonLinearityFromCorrection(self._cfg(), show_results=False)
        coeff = np.asarray(tool.results["ch"]["coeff"])
        assert coeff.shape[0] >= 4
        assert np.all(np.isfinite(coeff))
        # the retrieved model saturates (5% from linear) near the well depth
        assert tool.results["ch"]["saturation"] > 0

    def test_missing_coefficients_raise(self):
        cfg = self._cfg()
        for k in ("a", "b", "c", "d", "e"):
            cfg["channel"]["detector"].pop(f"pnl_coeff_{k}")
        with pytest.raises(KeyError, match="coefficients missing"):
            PixelsNonLinearityFromCorrection(cfg, show_results=False)

    def test_show_results_runs_the_print_and_plot_branch(self):
        import matplotlib as mpl

        mpl.use("Agg")
        from unittest.mock import patch

        with patch("matplotlib.pyplot.show"):
            tool = PixelsNonLinearityFromCorrection(self._cfg(), show_results=True)
        assert "coeff" in tool.results["ch"]
