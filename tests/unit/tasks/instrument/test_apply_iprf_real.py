"""
Behavioural tests for ApplyIntraPixelResponseFunction: the convolution-method
selector (every branch plus the error) and an end-to-end run with a
delta-function kernel, which must leave the focal plane unchanged.
"""

import astropy.units as u
import numpy as np
import pytest

from exosim.models.signal import Signal
from exosim.tasks.instrument.apply_intra_pixel_response_function import (
    ApplyIntraPixelResponseFunction,
)
from exosim.utils import RunConfig


@pytest.fixture(autouse=True)
def _one_job():
    n = RunConfig.n_job
    RunConfig.n_job = 1
    yield
    RunConfig.n_job = n


class TestConvolutionSelector:
    @pytest.mark.parametrize(
        "method", ["fftconvolve", "convolve", "ndimage.convolve", "fast_convolution"]
    )
    def test_each_method_returns_a_callable(self, method):
        task = ApplyIntraPixelResponseFunction()
        task.set_log_name()
        ker = np.ones((3, 3)) / 9.0
        func, kwargs = task.select_convoltion_func(method, ker, 1.0, 1.0)
        assert callable(func)
        assert isinstance(kwargs, dict)

    def test_unknown_method_raises(self):
        task = ApplyIntraPixelResponseFunction()
        task.set_log_name()
        with pytest.raises(ValueError, match="not supported"):
            task.select_convoltion_func("nope", np.ones((3, 3)), 1.0, 1.0)


class TestApplyToFocalPlane:
    def _focal_plane(self):
        rng = np.random.default_rng(0)
        data = rng.random((2, 12, 12))
        fp = Signal(
            spectral=np.arange(12) * u.um,
            data=data,
            time=np.array([0.0, 1.0]) * u.hr,
            data_units=u.ct / u.s,
        )
        fp.dataset_name = "focal_plane"
        fp.metadata["focal_plane_delta"] = 1.0 * u.um
        return fp

    def test_delta_kernel_leaves_the_focal_plane_unchanged(self):
        fp = self._focal_plane()
        before = fp.data.copy()
        ker = np.zeros((5, 5))
        ker[2, 2] = 1.0
        out = ApplyIntraPixelResponseFunction()(
            focal_plane=fp,
            irf_kernel=ker,
            irf_kernel_delta=1.0 * u.um,
            convolution_method="fftconvolve",
        )
        np.testing.assert_allclose(out.data, before, atol=1e-9)

    def test_negative_values_are_clipped(self):
        fp = self._focal_plane()
        fp.data[0, 0, 0] = -5.0
        ker = np.zeros((3, 3))
        ker[1, 1] = 1.0
        out = ApplyIntraPixelResponseFunction()(
            focal_plane=fp,
            irf_kernel=ker,
            irf_kernel_delta=1.0 * u.um,
            convolution_method="convolve",
        )
        assert np.all(out.data >= 0)

    def test_kernel_is_written_to_the_output_and_second_run_is_idempotent(
        self, tmp_path
    ):
        import h5py

        from exosim.output import SetOutput

        ker = np.zeros((3, 3))
        ker[1, 1] = 1.0
        fname = tmp_path / "irf.h5"
        with SetOutput(str(fname)).use(append=True) as out:
            ApplyIntraPixelResponseFunction()(
                focal_plane=self._focal_plane(),
                irf_kernel=ker,
                irf_kernel_delta=1.0 * u.um,
                convolution_method="convolve",
                output=out,
            )
            # a second call into the same group must hit the "already exists" guards
            ApplyIntraPixelResponseFunction()(
                focal_plane=self._focal_plane(),
                irf_kernel=ker,
                irf_kernel_delta=1.0 * u.um,
                convolution_method="convolve",
                output=out,
            )
        with h5py.File(fname, "r") as f:
            assert "irf/irf_kernel" in f
            assert "irf/irf_kernel_delta" in f
