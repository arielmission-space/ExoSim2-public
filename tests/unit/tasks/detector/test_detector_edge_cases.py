"""
Behavioural tests for detector-task branches the existing suite does not reach:
the analog-to-digital conversion (gain, offset, dtype, rounding, guard rails),
the seed-saving output paths of the noise tasks, and group merging.
"""

import numpy as np
import pytest
from astropy import units as u

from exosim.models.signal import Counts
from exosim.output import SetOutput
from exosim.tasks.detector import (
    AddGainDrift,
    AddNormalReadNoise,
    AddShotNoise,
    AnalogToDigital,
    MergeGroups,
)
from exosim.utils import RunConfig


def _cached_counts(out, data, name="SubExposures"):
    return Counts(
        spectral=np.arange(data.shape[2]),
        data=data,
        shape=data.shape,
        cached=True,
        output=out,
        dataset_name=name,
        output_path=None,
        dtype=np.float64,
    )


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


class TestAnalogToDigital:
    def _convert(self, tmp_path, data, detector, fname="adc.h5"):
        out = SetOutput(str(tmp_path / fname))
        with out.use(cache=True) as o:
            se = _cached_counts(o, data)
            ndrs = AnalogToDigital()(
                subexposures=se, parameters={"detector": detector}, output=o
            )
            return np.array(ndrs.dataset), ndrs.dataset.dtype, dict(ndrs.metadata)

    def test_offset_and_gain_applied(self, tmp_path):
        data = np.full((4, 6, 6), 1000.0)
        res, _, _ = self._convert(
            tmp_path, data, {"ADC_num_bit": 16, "ADC_gain": 2.0, "ADC_offset": 100}
        )
        assert np.all(res == 1800)  # (1000 - 100) * 2

    def test_dtype_follows_bit_depth(self, tmp_path):
        data = np.full((2, 4, 4), 10.0)
        _, dt8, _ = self._convert(
            tmp_path, data, {"ADC_num_bit": 8, "ADC_gain": 1.0}, "a.h5"
        )
        _, dt12, _ = self._convert(
            tmp_path, data, {"ADC_num_bit": 12, "ADC_gain": 1.0}, "b.h5"
        )
        assert dt8 == np.uint8
        assert dt12 == np.uint16

    def test_auto_gain_uses_full_range(self, tmp_path):
        data = np.linspace(0, 500, 4 * 4 * 4).reshape(4, 4, 4)
        res, _, meta = self._convert(
            tmp_path, data, {"ADC_num_bit": 8, "ADC_gain": "auto", "ADC_offset": 0}
        )
        assert res.max() == 255
        assert meta["ADC"]["gain"] == pytest.approx(255 / 500, rel=1e-3)

    def test_round_method_ceil(self, tmp_path):
        data = np.full((2, 3, 3), 10.4)
        res, _, _ = self._convert(
            tmp_path,
            data,
            {
                "ADC_num_bit": 16,
                "ADC_gain": 1.0,
                "ADC_offset": 0,
                "ADC_round_method": "ceil",
            },
        )
        assert np.all(res == 11)

    def test_bad_round_method_raises(self, tmp_path):
        with pytest.raises(ValueError, match="round, floor or ceil"):
            self._convert(
                tmp_path,
                np.ones((2, 3, 3)),
                {"ADC_num_bit": 16, "ADC_round_method": "banker"},
            )

    def test_bit_depth_above_32_raises(self, tmp_path):
        with pytest.raises(ValueError, match="exceeds maximum"):
            self._convert(tmp_path, np.ones((2, 3, 3)), {"ADC_num_bit": 40})

    def test_non_integer_bit_depth_raises(self, tmp_path):
        with pytest.raises(TypeError, match="should be integer"):
            self._convert(tmp_path, np.ones((2, 3, 3)), {"ADC_num_bit": 12.5})

    def test_over_range_pixels_saturate_instead_of_wrapping(self, tmp_path):
        # a bright pixel well past the ADC full scale with a fixed gain: the
        # cast used to wrap around (200000 -> 3392 on uint16)
        data = np.full((2, 4, 4), 200000.0)
        data[0, 0, 0] = 5.0
        res, dtype, _ = self._convert(
            tmp_path, data, {"ADC_num_bit": 16, "ADC_gain": 1.0, "ADC_offset": 0}
        )
        assert dtype == np.uint16
        assert res[0, 1, 1] == 65535  # clipped to full scale
        assert res[0, 0, 0] == 5  # in-range pixel untouched


class TestNoiseTaskOutputs:
    def test_shot_noise_runs_with_output_group(self, tmp_path):
        out = SetOutput(str(tmp_path / "shot.h5"))
        RunConfig.random_seed = 1
        with out.use(cache=True) as o:
            se = _cached_counts(o, np.full((5, 5, 20), 100.0))
            AddShotNoise()(subexposures=se, output=o.create_group("run"))
            # Poisson noise was applied: mean stays ~100, spread ~sqrt(100)
            assert np.std(np.asarray(se.dataset)) == pytest.approx(10.0, rel=0.15)

    def test_shot_noise_clips_non_positive_pixels(self, tmp_path):
        out = SetOutput(str(tmp_path / "shot2.h5"))
        RunConfig.random_seed = 2
        with out.use(cache=True) as o:
            data = np.full((4, 4, 10), 50.0)
            data[0] = -5.0  # negative pixels -> replaced with 1e-10 (+ warning)
            se = _cached_counts(o, data)
            AddShotNoise()(subexposures=se)
            assert np.all(np.asarray(se.dataset) >= 0)

    def test_read_noise_runs_with_output_group(self, tmp_path):
        out = SetOutput(str(tmp_path / "read.h5"))
        RunConfig.random_seed = 3
        with out.use(cache=True) as o:
            se = _cached_counts(o, np.full((5, 5, 10), 100.0))
            AddNormalReadNoise()(
                subexposures=se,
                parameters={"detector": {"read_noise_sigma": 2 * u.ct}},
                output=o.create_group("run"),
            )
            assert np.std(np.asarray(se.dataset)) == pytest.approx(2.0, rel=0.2)


class TestAddGainDrift:
    def _params(self, **det_over):
        det = {
            "gain_coeff_order_t": 1,
            "gain_coeff_t_min": 1.0,
            "gain_coeff_t_max": 1.01,
            "gain_coeff_order_w": 1,
            "gain_coeff_w_min": 1.0,
            "gain_coeff_w_max": 1.01,
        }
        det.update(det_over)
        return {"detector": det}

    def _subexposures(self, o):
        data = np.ones((20, 8, 8)) * 100.0
        return Counts(
            spectral=np.arange(8),
            data=data,
            time=np.arange(20),
            shape=data.shape,
            cached=True,
            output=o,
            dataset_name="SubExposures",
            output_path=None,
            dtype=np.float64,
            metadata={"integration_times": np.ones(20)},
        )

    def test_fixed_amplitude_sets_the_gain_dynamic_range(self, tmp_path):
        RunConfig.random_seed = 5
        out = SetOutput(str(tmp_path / "g.h5"))
        with out.use(cache=True) as o:
            se = self._subexposures(o)
            AddGainDrift()(
                subexposures=se, parameters=self._params(gain_drift_amplitude=1e-2)
            )
            data = np.asarray(se.dataset)
            rng = (data.max() - data.min()) / data.min()
            assert rng == pytest.approx(1e-2, abs=1e-4)

    def test_amplitude_range_draws_a_random_amplitude_and_writes_output(self, tmp_path):
        RunConfig.random_seed = 9
        out = SetOutput(str(tmp_path / "g2.h5"))
        with out.use(cache=True) as o:
            se = self._subexposures(o)
            AddGainDrift()(
                subexposures=se,
                parameters=self._params(
                    gain_drift_amplitude_range_min=1e-3,
                    gain_drift_amplitude_range_max=5e-3,
                ),
                output=o,
            )
            data = np.asarray(se.dataset)
        rng = (data.max() - data.min()) / data.min()
        assert 1e-3 - 1e-4 <= rng <= 5e-3 + 1e-4
        import h5py

        with h5py.File(str(tmp_path / "g2.h5"), "r") as f:
            assert "gain noise/amplitude" in f

    def test_missing_amplitude_definition_raises_keyerror(self, tmp_path):
        RunConfig.random_seed = 1
        out = SetOutput(str(tmp_path / "g3.h5"))
        with out.use(cache=True) as o:
            se = self._subexposures(o)
            with pytest.raises(KeyError, match="gain_drift_amplitude"):
                AddGainDrift()(subexposures=se, parameters=self._params())

    def test_polynomial_helpers_are_normalised(self):
        tt = np.arange(10) * u.s
        y = AddGainDrift._pol_t(tt, np.array([2.0, 1.0]))
        assert y[0] == pytest.approx(2.0)  # x=0 -> constant term
        assert y[-1] == pytest.approx(3.0)  # x=1 -> c0 + c1
        yw = AddGainDrift._pol_w(np.arange(5), np.array([0.0, 4.0]))
        assert yw[0] == pytest.approx(0.0)
        assert yw[-1] == pytest.approx(4.0)

    def test_psd_and_noise_generator_shapes(self):
        f = np.linspace(0.0, 1.0, 65)
        psd = AddGainDrift._psd(f, w0=1.0, f0=0.1)
        assert psd[0] == 0.0
        assert np.all(psd[1:] > 0)
        RunConfig.random_seed = 3
        noise = AddGainDrift._noise_generator(f, psd)
        assert noise.shape == (2 * psd.size - 2,)
        assert np.all(np.isfinite(noise))


class TestMergeGroups:
    def test_merges_ndrs_by_averaging(self, tmp_path):
        out = SetOutput(str(tmp_path / "merge.h5"))
        with out.use(cache=True) as o:
            data = np.stack([np.full((3, 3), float(v)) for v in [1, 3, 5, 7, 9, 11]])
            se = _cached_counts(o, data)
            merged = MergeGroups()(subexposures=se, n_groups=3, n_ndrs=2, output=o)
            vals = np.asarray(merged.dataset)[:, 0, 0]
            np.testing.assert_allclose(vals, [2.0, 6.0, 10.0])

    def test_single_ndr_per_group_copies_through(self, tmp_path):
        out = SetOutput(str(tmp_path / "merge1.h5"))
        with out.use(cache=True) as o:
            data = np.stack([np.full((3, 3), float(v)) for v in [10, 20, 30]])
            se = _cached_counts(o, data)
            merged = MergeGroups()(subexposures=se, n_groups=3, n_ndrs=1, output=o)
            np.testing.assert_allclose(
                np.asarray(merged.dataset)[:, 0, 0], [10, 20, 30]
            )

    def test_frame_count_mismatch_raises(self, tmp_path):
        out = SetOutput(str(tmp_path / "merge_bad.h5"))
        with out.use(cache=True) as o:
            data = np.stack([np.full((3, 3), float(v)) for v in [1, 2, 3, 4, 5]])
            se = _cached_counts(o, data)
            # 5 sub-exposures cannot form 3 groups of 2
            with pytest.raises(ValueError, match="not n_groups"):
                MergeGroups()(subexposures=se, n_groups=3, n_ndrs=2, output=o)
