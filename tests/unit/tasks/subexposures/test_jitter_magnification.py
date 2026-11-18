"""The jitter magnification keeps the pixel centres fixed and the power right."""

import astropy.units as u
import numpy as np
import pytest

from exosim.models.signal import CountsPerSecond
from exosim.output import SetOutput
from exosim.tasks.subexposures import (
    InstantaneousReadOut,
    PrepareInstantaneousReadOut,
)

CASES = [(3, 2), (3, 4), (3, 3), (5, 2), (4, 3), (1, 4)]
NPIX = 20
SIGMA_PIX = 1.7  # pixels
# the centre of the Gaussian, in pixels from the centre of pixel 0
X0, Y0 = 9.3, 10.4


def _gauss(x, y, osf):
    """Gaussian of centre (X0, Y0) pixels, evaluated at old sub-pixel
    coordinates x, y. Pixel m has its centre at sub-pixel osf * m + osf // 2."""
    s = SIGMA_PIX * osf
    cx = X0 * osf + osf // 2
    cy = Y0 * osf + osf // 2
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))


def _old_grid(shape, mag):
    """Previous formula of ``get_oversample_factors`` (right for odd/odd only)."""
    step = 1.0 / mag
    start = -step * np.float64((mag - 1) / 2)
    xout = np.arange(start, start + step * shape[1] * mag, step)
    yout = np.arange(start, start + step * shape[0] * mag, step)
    return xout, yout


def _plane(osf):
    n = NPIX * osf
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return _gauss(xx, yy, osf)


def _centroid(img):
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].astype(float)
    return (xx * img).sum() / img.sum(), (yy * img).sum() / img.sum()


@pytest.mark.parametrize(("osf", "mag"), CASES)
def test_new_subpixel_of_pixel_centre_matches_old_one(osf, mag):
    fp = _plane(osf)
    new = InstantaneousReadOut.oversample(fp, mag, osf)
    o = osf * mag
    assert new.shape == (fp.shape[0] * mag, fp.shape[1] * mag)
    m = np.arange(NPIX)
    expected = _gauss((osf * m + osf // 2)[None, :], (osf * m + osf // 2)[:, None], osf)
    got = new[(o * m + o // 2)[:, None], (o * m + o // 2)[None, :]]
    np.testing.assert_allclose(got, expected, atol=1e-3 * expected.max())


@pytest.mark.parametrize(("osf", "mag"), CASES)
def test_centroid_is_unchanged(osf, mag):
    fp = _plane(osf)
    new = InstantaneousReadOut.oversample(fp, mag, osf)
    o = osf * mag
    before = np.array(_centroid(fp[osf // 2 :: osf, osf // 2 :: osf]))
    after = np.array(_centroid(new[o // 2 :: o, o // 2 :: o]))
    np.testing.assert_allclose(after, before, atol=1e-3)


@pytest.mark.parametrize(("osf", "mag"), [(3, 3), (5, 3), (1, 3), (3, 1), (5, 5)])
def test_odd_odd_is_identical_to_the_old_grid(osf, mag):
    fp = _plane(osf)
    _xin, _yin, xout, yout = InstantaneousReadOut.get_oversample_factors(fp, mag, osf)
    xold, yold = _old_grid(fp.shape, mag)
    np.testing.assert_allclose(xout, xold, atol=1e-12)
    np.testing.assert_allclose(yout, yold, atol=1e-12)


def test_old_grid_was_shifted_for_even_cases():
    # documents the bug: osf = 3, mag = 4 -> 0.125 old sub-pixels
    osf, mag = 3, 4
    fp = _plane(osf)
    _x, _y, xout, _yo = InstantaneousReadOut.get_oversample_factors(fp, mag, osf)
    xold, _ = _old_grid(fp.shape, mag)
    o = osf * mag
    assert xout[o // 2] == pytest.approx(osf // 2)
    assert xold[o // 2] - osf // 2 == pytest.approx(0.125)


def test_force_power_conservation_with_magnification(tmp_path):
    osf, npix = 3, 24
    grid = np.arange(npix * osf)
    yy, xx = np.meshgrid(grid, grid)
    sigma = 3 * osf / 2.355
    cen = npix * osf / 2
    g = 49 * np.exp(-((xx - cen) ** 2 + (yy - cen) ** 2) / (2 * sigma**2))
    fp = CountsPerSecond(
        data=(g * u.ct / u.s)[np.newaxis, ...],
        spectral=grid,
        metadata={"oversampling": osf},
    )
    main_parameters = {
        "time_grid": {
            "start_time": 0.0 * u.hr,
            "end_time": 2.0 * u.hr,
            "low_frequencies_resolution": 2.0 * u.hr,
        }
    }
    parameters = {
        "detector": {
            "well_depth": 1000 * u.ct,
            "f_well_depth": 1,
            "delta_pix": 18 * u.um,
            "oversampling": osf,
            "jitter_resampler_mag": 2,
            "plate_scale": {
                "spatial": 0.05 * u.arcsec / u.pixel,
                "spectral": 0.05 * u.arcsec / u.pixel,
            },
        },
        "readout": {
            "readout_frequency": 100 * u.Hz,
            "n_NRDs_per_group": 1,
            "n_groups": 2,
            "n_sim_clocks_Ground": 5,
            "n_sim_clocks_first_NDR": 5,
            "n_sim_clocks_NDR": 1,
            "n_sim_clocks_Reset": 3,
            "n_sim_clocks_groups": 200,
            "n_exposures": 1,
        },
        "value": "test_channel",
        "force_power_conservation": True,
    }
    rng = np.random.default_rng(3)
    jt = np.arange(0, 5, 0.1) * u.s
    pj = (
        rng.normal(0, 0.05, jt.size) * u.arcsec,
        rng.normal(0, 0.05, jt.size) * u.arcsec,
        jt,
    )
    with SetOutput(str(tmp_path / "mag_power.h5")).use(append=True, cache=True) as out:
        readout_parameters, _ = PrepareInstantaneousReadOut()(
            main_parameters=main_parameters,
            parameters=parameters,
            focal_plane=fp,
            pointing_jitter=pj,
            output_file=out,
        )
        assert readout_parameters["mag"] > 1
        dset = InstantaneousReadOut()(
            readout_parameters=readout_parameters,
            parameters=parameters,
            focal_plane=fp,
            pointing_jitter=pj,
            output_file=out,
        )
        data = np.asarray(dset.dataset)
        times = readout_parameters["ndr_integration_times"].to_value(u.s)

    # total per unit time equals the sum of the decimated plane
    expected = fp.data[0][osf // 2 :: osf, osf // 2 :: osf].sum()
    # the requested power is the whole plane divided by osf**2
    requested = np.asarray(fp.data[0]).sum() / osf**2
    per_second = data.sum(axis=(1, 2)) / times
    np.testing.assert_allclose(per_second, requested, rtol=1e-6)
    assert abs(requested - expected) / expected < 1e-3
