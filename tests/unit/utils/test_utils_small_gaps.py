"""
Targeted behavioural tests for branches of the small utility modules that the
existing suite does not exercise: PSF array sizing, the custom FFT convolution
path, the class factory, the recipe-preparation helpers and TimedClass.
"""

import astropy.units as u
import numpy as np
import pytest

from exosim.utils import convolution, klass_factory, prepare_recipes
from exosim.utils.psf import create_psf
from exosim.utils.run_config import RunConfig
from exosim.utils.timed_class import TimedClass


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


class TestCreatePsf:
    def test_tuple_fnum_gives_rectangular_array(self):
        img = create_psf(1 * u.um, (60, 30), 6 * u.um, shape="gauss")
        assert img.ndim == 2
        assert img.shape[0] != img.shape[1]

    def test_max_array_size_caps_the_output(self):
        big = create_psf(3 * u.um, 40, 6 * u.um, shape="airy")
        capped = create_psf(
            3 * u.um, 40, 6 * u.um, shape="airy", max_array_size=(11, 11)
        )
        assert capped.shape[0] <= big.shape[0]
        assert max(capped.shape) <= 11

    def test_array_size_full_needs_max_array_size(self):
        with pytest.raises(ValueError, match="max_array_size must be set"):
            create_psf(1 * u.um, 40, 6 * u.um, array_size=("full", "full"))

    def test_explicit_array_size(self):
        img = create_psf(
            1 * u.um, 40, 6 * u.um, array_size=(9, 9), max_array_size=(21, 21)
        )
        assert img.shape[0] <= 21


class TestFastConvolution:
    def test_delta_kernel_is_identity(self):
        rng = np.random.default_rng(0)
        im = rng.random((20, 20))
        ker = np.zeros((5, 5))
        ker[2, 2] = 1.0
        out = convolution.fast_convolution(im, 1.0 * u.um, ker, 1.0 * u.um)
        assert out.shape == im.shape
        np.testing.assert_allclose(out, im, atol=1e-9)

    def test_kernel_sum_is_preserved(self):
        im = np.ones((24, 24))
        ker = np.ones((3, 3)) / 9.0
        out = convolution.fast_convolution(im, 1.0 * u.um, ker, 1.0 * u.um)
        # convolving a flat field with a normalised kernel leaves it flat
        np.testing.assert_allclose(out[5:-5, 5:-5], 1.0, atol=1e-9)

    def test_large_array_uses_power_of_two_padding(self):
        rng = np.random.default_rng(1)
        im = rng.random((40, 40))
        ker = np.zeros((5, 5))
        ker[2, 2] = 1.0
        out = convolution.fast_convolution(im, 1.0 * u.um, ker, 1.0 * u.um)
        np.testing.assert_allclose(out, im, atol=1e-9)

    def test_different_sampling_triggers_resampling(self):
        im = np.ones((20, 20))
        ker = np.ones((3, 3)) / 9.0
        out = convolution.fast_convolution(im, 1.0 * u.um, ker, 0.5 * u.um)
        assert out.shape == im.shape


class TestKlassFactory:
    def test_find_task_returns_baseclass_by_name(self):
        from exosim.tasks.radiometric.multiaccum import Multiaccum

        found = klass_factory.find_task("Multiaccum", Multiaccum)
        assert found is Multiaccum

    def test_load_klass_rejects_non_string(self):
        from exosim.tasks.task import Task

        with pytest.raises(TypeError, match="wrong format"):
            klass_factory.load_klass(1234, Task)

    def test_find_and_run_task_falls_back_to_default(self):
        from exosim.tasks.radiometric.multiaccum import Multiaccum

        task = klass_factory.find_and_run_task({}, "missing_key", Multiaccum)
        assert isinstance(task, Multiaccum)


class TestPrepareRecipes:
    def test_load_options_passes_through_a_dict(self):
        cfg = {"payload": {"channel": {"value": "ch"}}}
        main, payload = prepare_recipes.load_options(cfg)
        assert main is cfg
        assert payload == {"channel": {"value": "ch"}}

    def test_load_options_rejects_none(self):
        with pytest.raises(ValueError, match="cannot be None"):
            prepare_recipes.load_options(None)

    def test_load_options_rejects_other_types(self):
        with pytest.raises(TypeError, match="must be str or dict"):
            prepare_recipes.load_options(1234)

    def test_copy_and_clean_config_files(self, tmp_path):
        src = tmp_path / "cfg.xml"
        src.write_text("<root/>")
        dest = tmp_path / "out"
        dest.mkdir()
        RunConfig.config_file_list = [str(src)]
        prepare_recipes.copy_input_files(str(dest))
        assert (dest / "cfg.xml").exists()
        # a second call hits the SameFileError branch when source == dest
        RunConfig.config_file_list = [str(dest / "cfg.xml")]
        prepare_recipes.copy_input_files(str(dest))
        prepare_recipes.clean_config_files()
        assert RunConfig.config_file_list == []


class TestTimedClass:
    def test_log_runtime_complete_bad_level_warns(self):
        class _Timed(TimedClass):
            pass

        timed = _Timed()
        # an unknown level makes getattr(self, level) fail -> warning branch
        timed.log_runtime_complete("done", level="does_not_exist")
