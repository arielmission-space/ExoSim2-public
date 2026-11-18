"""The jitter offsets are rounded to the nearest sub-pixel, not truncated."""

import numpy as np

from exosim.tasks.subexposures.instantaneous_read_out import InstantaneousReadOut

rnd = InstantaneousReadOut.round_to_subpixel


def test_rounds_to_nearest_symmetrically():
    x = np.array([-1.6, -1.4, -0.6, -0.4, 0.0, 0.4, 0.6, 1.4, 1.6])
    np.testing.assert_array_equal(rnd(x), [-2, -1, -1, 0, 0, 0, 1, 1, 2])
    assert rnd(x).dtype.kind == "i"


def test_sinusoid_rms_and_no_bias():
    t = np.linspace(0, 1, 200001)[:-1]
    amp = 6.3  # sub-pixels
    x = amp * np.sin(2 * np.pi * 37 * t)
    q = rnd(x)
    # unquantised rms amp/sqrt(2); rounding adds 1/12 in variance
    expected = np.sqrt(amp**2 / 2 + 1 / 12)
    assert abs(q.mean()) < 1e-2
    assert abs(q.std() - expected) < 0.06  # arcsine law: Sheppard is approximate
    assert abs(q.std() - x.std()) < 0.05  # quantisation error only
    # truncation toward zero would shrink the rms (the former behaviour)
    assert np.std(x.astype(int)) < x.std() - 0.1


def test_gaussian_jitter_rms_and_no_bias():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.45, 400000)  # sub-pixel rms comparable to the step
    q = rnd(x)
    assert abs(q.mean()) < 3e-3
    assert abs(q.std() - np.sqrt(0.45**2 + 1 / 12)) < 0.02
