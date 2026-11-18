"""Pixel response function tables and deposition (exosim.utils.prf)."""

import astropy.units as u
import numpy as np
import pytest
from scipy.special import erf

from exosim.utils import prf as prf_utils
from exosim.utils.prf import (
    PrfTables,
    build_prf_tables,
    deposit_prf,
    fine_pixel_response,
    get_population,
    node_weights,
    prf_unsupported_reason,
    quadrature_kernel,
    wavelength_coverage,
    wavelength_to_subpixel,
)

DELTA = 18.0 * u.um


def _gaussian_cells(sampling, sigma_px, half_px):
    """Energy per fine cell of a centred Gaussian PSF: the point samples of the
    profile times the cell area (the form of the fine PSFs of the tables)."""
    n = 2 * int(half_px * sampling) + 1
    x = (np.arange(n) - n // 2) / sampling
    line = np.exp(-0.5 * (x / sigma_px) ** 2) / (
        np.sqrt(2 * np.pi) * sigma_px * sampling
    )
    return np.outer(line, line)


def _gaussian_prf(offset_y, offset_x, sigma_px):
    """Exact signal of a unit-flux Gaussian in a top-hat pixel at an offset."""

    def axis(d):
        a, b = d - 0.5, d + 0.5
        return 0.5 * (
            erf(b / (np.sqrt(2) * sigma_px)) - erf(a / (np.sqrt(2) * sigma_px))
        )

    return axis(np.asarray(offset_y)) * axis(np.asarray(offset_x))


@pytest.fixture(scope="module")
def gaussian_tables():
    sampling, sigma = 10, 0.45
    psf = _gaussian_cells(sampling, sigma, 6)[None]
    response = fine_pixel_response(DELTA, 0.0 * u.um, 0.0 * u.um, sampling)
    tables = build_prf_tables(psf, np.array([1.0]), response, sampling)
    return tables, sigma


_DET = {"delta_pix": 18 * u.um}


def test_population_default_and_validation():
    assert get_population({"detector": _DET}) == "prf"
    assert get_population({"detector": _DET, "psf": {"shape": "airy"}}) == "prf"
    assert get_population({"detector": _DET, "psf": {"population": " PRF "}}) == "prf"
    with pytest.raises(ValueError, match="population"):
        get_population({"detector": _DET, "psf": {"population": "exact"}})


def test_population_explicit_sampled_is_silent():
    calls = []
    pars = {"detector": _DET, "psf": {"population": "sampled"}}
    assert get_population(pars, on_fallback=calls.append) == "sampled"
    assert calls == []
    # explicit sampled is also silent on an unsupported configuration
    pars["detector"] = {
        **_DET,
        "irf_task": "CreateOversampledIntrapixelResponseFunction",
    }
    assert get_population(pars, on_fallback=calls.append) == "sampled"
    assert calls == []


def test_population_falls_back_to_sampled_with_a_reason():
    calls = []
    pars = {
        "detector": {**_DET, "irf_task": "CreateOversampledIntrapixelResponseFunction"},
        "psf": {"shape": "airy"},
    }
    assert get_population(pars, on_fallback=calls.append) == "sampled"
    assert len(calls) == 1
    assert "irf_task" in calls[0]
    assert prf_unsupported_reason(pars) == calls[0]


def test_population_falls_back_for_time_dependent_psf():
    pars = {
        "detector": _DET,
        "psf": {"psf_task": "LoadPsfPaosTimeInterp", "filename": "x.h5"},
    }
    assert get_population(pars) == "sampled"
    assert "time dependent" in prf_unsupported_reason(pars)


def test_population_falls_back_without_delta_pix():
    assert get_population({"detector": {}}) == "sampled"
    assert "delta_pix" in prf_unsupported_reason({"detector": {}})


def test_population_prf_with_default_irf_task_and_paos():
    pars = {
        "detector": {**_DET, "irf_task": "CreateIntrapixelResponseFunction"},
        "psf": {"psf_task": "LoadPsfPaos", "filename": "x.h5"},
    }
    assert prf_unsupported_reason(pars) is None
    assert get_population(pars) == "prf"


def test_explicit_prf_with_custom_irf_still_raises():
    from exosim.tasks.instrument import PopulateFocalPlanePrf

    pars = {
        "detector": {**_DET, "irf_task": "CreateOversampledIntrapixelResponseFunction"},
        "psf": {"population": "prf"},
    }
    assert get_population(pars) == "prf"
    with pytest.raises(ValueError, match="custom"):
        PopulateFocalPlanePrf.check_configuration(pars)


def test_channel_warns_once_on_fallback():
    from unittest.mock import MagicMock

    from exosim.models.channel import Channel

    ch = object.__new__(Channel)
    ch.ch_name = "chA"
    ch.warning = MagicMock()
    ch.parameters = {
        "detector": {**_DET, "irf_task": "CreateOversampledIntrapixelResponseFunction"}
    }
    for _ in range(3):
        assert ch.psf_population == "sampled"
    ch.warning.assert_called_once()
    assert "<population>sampled</population>" in ch.warning.call_args[0][0]
    ch2 = object.__new__(Channel)
    ch2.ch_name = "chB"
    ch2.warning = MagicMock()
    ch2.parameters = {"detector": _DET}
    assert ch2.psf_population == "prf"
    ch2.warning.assert_not_called()


@pytest.mark.parametrize("lx", [0.0, 1.7, 6.0])
@pytest.mark.parametrize("sampling", [7, 10])
def test_response_normalisation_and_symmetry(lx, sampling):
    resp = fine_pixel_response(DELTA, lx * u.um, 0.0 * u.um, sampling)
    n_sub = prf_utils.RESPONSE_SUBSAMPLES
    assert resp.size % n_sub == 0
    assert (resp.size // n_sub) % 2 == 1
    # unit mean over a pixel for a uniform illumination
    assert resp.sum() / n_sub == pytest.approx(sampling, rel=1e-12)
    assert np.allclose(resp, resp[::-1], rtol=1e-12, atol=1e-12)
    kernel = quadrature_kernel(resp, n_sub)
    assert kernel.sum() == pytest.approx(sampling, rel=1e-9)
    # without diffusion the response is a top-hat one pixel wide
    if lx == 0.0:
        assert resp.size // n_sub == sampling + (1 - sampling % 2)
        assert resp.max() == pytest.approx(1.0, rel=1e-9)


def test_response_gap_is_zeroed():
    resp = fine_pixel_response(DELTA, 0.0 * u.um, 2.0 * u.um, 10)
    n_sub = prf_utils.RESPONSE_SUBSAMPLES
    assert resp.sum() / n_sub == pytest.approx(10, rel=1e-12)
    assert resp[: n_sub // 2].max() == 0.0


def test_node_weights():
    nodes = np.array([1.0, 2.0, 4.0])
    idx, w = node_weights([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 9.0], nodes)
    assert list(idx) == [0, 0, 0, 0, 1, 1, 1]
    assert np.allclose(w, [1.0, 1.0, 0.5, 0.0, 0.5, 0.0, 0.0])
    idx, w = node_weights([3.0], np.array([2.0]))
    assert idx[0] == 0
    assert w[0] == 1.0


def test_wavelength_to_subpixel_increasing_and_decreasing():
    axis = 2.0 + 0.1 * np.arange(10)
    pos, _ = wavelength_to_subpixel([2.0, 2.25, 2.9], axis)
    assert np.allclose(pos, [0.0, 2.5, 9.0])
    pos, _ = wavelength_to_subpixel([2.25, 2.9], axis[::-1])
    assert np.allclose(pos, [6.5, 0.0])
    # linear extrapolation beyond the axis
    pos, _ = wavelength_to_subpixel([1.8, 3.1], axis)
    assert np.allclose(pos, [-2.0, 11.0])
    with pytest.raises(ValueError, match="monotonic"):
        wavelength_to_subpixel([2.0], np.array([1.0, 3.0, 2.0]))


def test_wavelength_coverage_fractions():
    axis = np.array([1.0, 2.0, 3.0])  # cell edges 0.5 and 3.5
    wl = np.array([0.0, 0.5, 1.0, 3.5, 4.0])
    w = wavelength_coverage(wl, np.ones(5), axis)
    assert np.allclose(w, [0.0, 0.5, 1.0, 0.5, 0.0])
    assert np.all(wavelength_coverage(wl, np.ones(5), np.array([2.0, 2.0])) == 1.0)


@pytest.mark.parametrize("osf", [1, 3, 5, 4])
def test_physical_axis_edges_do_not_depend_on_the_oversampling(osf):
    """The wavelength of the physical pixel edges (-1/2 and N - 1/2), whose
    pixel m is the sub-pixel osf * m + osf // 2, is the same at any osf."""
    n_pix = 10
    j = np.arange(n_pix * osf)
    axis = 1.0 + 0.01 * (j - osf // 2) / osf  # linear: pixel m is at 1 + 0.01 m
    lo, hi = prf_utils.physical_axis_edges(axis, osf)
    assert lo == pytest.approx(1.0 - 0.005)
    assert hi == pytest.approx(1.0 + 0.01 * (n_pix - 1) + 0.005)
    # descending axis, and invalid (zero) wavelengths at the ends
    assert prf_utils.physical_axis_edges(axis[::-1] * 1.0, osf) == pytest.approx(
        (lo, hi), abs=0.01 * 1.01 / osf
    )
    cut = axis.copy()
    cut[: osf * 2] = 0.0
    lo_c, hi_c = prf_utils.physical_axis_edges(cut, osf)
    assert hi_c == pytest.approx(hi)
    assert lo_c == pytest.approx(1.0 + 0.01 * (2 * osf - osf // 2 - 0.5) / osf)


def test_tables_on_grid_match_analytic_gaussian(gaussian_tables):
    tables, sigma = gaussian_tables
    assert isinstance(tables, PrfTables)
    cy, cx = tables.centre
    for dy in (0, 1, -1, 2):
        for dx in (0, 1, -2, 3):
            value = tables.tables[0, cy + 10 * dy, cx + 10 * dx]
            assert value == pytest.approx(
                float(_gaussian_prf(dy, dx, sigma)), rel=5e-3, abs=1e-5
            )


@pytest.mark.parametrize("osf", [1, 3, 4, 5])
def test_deposit_matches_analytic_gaussian_at_any_offset(osf):
    """Every sub-pixel holds the signal of a pixel centred on it, for any
    sub-pixel position of the source and any oversampling (the interpolation of
    the tables is accurate to 0.1% for a PSF with sigma = 0.6 pixel, 0.3% for
    sigma = 0.45 pixel and ``prf_sampling = 10``)."""
    sampling, sigma = 10, 0.6
    psf = _gaussian_cells(sampling, sigma, 6)[None]
    response = fine_pixel_response(DELTA, 0.0 * u.um, 0.0 * u.um, sampling)
    tables = build_prf_tables(psf, np.array([1.0]), response, sampling)
    n = 16
    qy, qx = 7.0 + 0.23, 8.0 - 0.37
    plane = np.zeros((1, n * osf, n * osf))
    deposit_prf(
        plane, tables, np.array([0]), np.array([1.0]),
        np.array([qy]), np.array([qx]), np.array([[2.0]]), osf,
    )  # fmt: skip
    a = np.arange(n * osf)
    centres = (a - osf // 2) / osf
    expected = 2.0 * _gaussian_prf(
        (centres - qy)[:, None], (centres - qx)[None, :], sigma
    )
    assert np.abs(plane[0] - expected).max() < 5e-4 * expected.max()
    big = expected > 1e-2 * expected.max()
    assert np.abs(plane[0][big] / expected[big] - 1).max() < 1e-3


def test_deposit_is_linear_in_flux_and_wavelength_nodes():
    sampling = 6
    psf = np.stack([_gaussian_cells(sampling, s, 5) for s in (0.4, 0.6)])
    response = fine_pixel_response(DELTA, 0.0 * u.um, 0.0 * u.um, sampling)
    tables = build_prf_tables(psf, np.array([1.0, 2.0]), response, sampling)
    plane = np.zeros((2, 12, 12))
    flux = np.array([[1.0, 2.0], [0.5, 0.0]])
    deposit_prf(
        plane, tables, np.array([0, 0]), np.array([0.25, 1.0]),
        np.array([6.0, 5.0]), np.array([6.0, 5.5]), flux, 1,
    )  # fmt: skip
    plane_a = np.zeros((1, 12, 12))
    deposit_prf(
        plane_a, tables, np.array([0]), np.array([0.25]),
        np.array([6.0]), np.array([6.0]), np.array([[1.0]]), 1,
    )  # fmt: skip
    plane_b = np.zeros((1, 12, 12))
    deposit_prf(
        plane_b, tables, np.array([0]), np.array([1.0]),
        np.array([5.0]), np.array([5.5]), np.array([[2.0]]), 1,
    )  # fmt: skip
    assert np.allclose(plane[0], plane_a[0] + plane_b[0], rtol=1e-12, atol=1e-15)
    assert np.allclose(plane[1], 0.5 * plane_a[0], rtol=1e-12, atol=1e-15)


def test_deposit_outside_the_plane_and_validation(gaussian_tables):
    tables, _ = gaussian_tables
    plane = np.zeros((1, 8, 8))
    # a source far outside the plane adds nothing, one on the border is cropped
    deposit_prf(
        plane, tables, np.array([0, 0]), np.array([1.0, 1.0]),
        np.array([50.0, 0.0]), np.array([4.0, 0.0]), np.ones((1, 2)), 1,
    )  # fmt: skip
    assert plane[0, 0, 0] > 0
    assert 0.7 < plane.sum() < 0.8  # a quadrant of the source
    with pytest.raises(ValueError, match="flux must have shape"):
        deposit_prf(plane, tables, [0], [1.0], [1.0], [1.0], np.ones((2, 1)), 1)
    with pytest.raises(ValueError, match="multiple of the oversampling"):
        deposit_prf(plane, tables, [0], [1.0], [1.0], [1.0], np.ones((1, 1)), 3)
    with pytest.raises(ValueError, match="3D"):
        deposit_prf(plane[0], tables, [0], [1.0], [1.0], [1.0], np.ones((1, 1)), 1)


def test_build_tables_validation():
    response = fine_pixel_response(DELTA, 0.0 * u.um, 0.0 * u.um, 4)
    psf = np.ones((2, 9, 9))
    with pytest.raises(ValueError, match="3D"):
        build_prf_tables(psf[0], np.array([1.0]), response, 4)
    with pytest.raises(ValueError, match="one wavelength"):
        build_prf_tables(psf, np.array([1.0]), response, 4)
    with pytest.raises(ValueError, match="increasing"):
        build_prf_tables(psf, np.array([2.0, 1.0]), response, 4)


# ------------------------------------------------------------------- cache
def _tables(n_bytes=8):
    return prf_utils.PrfTables(
        nodes=np.array([1.0, 2.0]),
        tables=np.zeros((2, 1, n_bytes // 8), np.float32),
        sampling=1,
        centre=(0, 0),
        response=np.ones(1),
        energy=np.ones(2),
    )


@pytest.fixture
def empty_cache():
    prf_utils.clear_prf_cache()
    yield
    prf_utils.clear_prf_cache()


def test_cache_is_lru_and_bounded(empty_cache):
    built = []

    def get(key):
        return prf_utils.cached_tables(key, lambda: built.append(key) or _tables())

    for key in "abcd":
        get(key)
    get("a")  # a is the most recent now
    get("e")  # evicts b, the least recently used
    assert prf_utils.prf_cache_info()["entries"] == prf_utils.PRF_CACHE_ENTRIES
    assert built == list("abcde")
    get("a")
    get("c")
    get("e")
    assert built == list("abcde")
    get("b")
    assert built == list("abcdeb")
    assert prf_utils.prf_cache_info()["hits"] == 4


def test_cache_is_bounded_in_bytes(empty_cache, monkeypatch):
    monkeypatch.setattr(prf_utils, "PRF_CACHE_MAX_BYTES", 2 * _tables(800).nbytes)
    for key in "abc":
        prf_utils.cached_tables(key, lambda: _tables(800))
    assert prf_utils.prf_cache_info()["entries"] == 2
    # the newest entry is kept even if it alone exceeds the limit
    monkeypatch.setattr(prf_utils, "PRF_CACHE_MAX_BYTES", 1)
    prf_utils.cached_tables("d", lambda: _tables(800))
    assert prf_utils.prf_cache_info()["entries"] == 1


def test_cache_can_be_disabled(empty_cache):
    n = []
    for _ in range(2):
        prf_utils.cached_tables("a", lambda: n.append(1) or _tables(), enabled=False)
    assert len(n) == 2
    assert prf_utils.prf_cache_info()["entries"] == 0


def test_cache_option_parsing():
    assert prf_utils.prf_cache_enabled({}) is True
    assert prf_utils.prf_cache_enabled({"psf": {"prf_cache": False}}) is False
    assert prf_utils.prf_cache_enabled({"psf": {"prf_cache": " False "}}) is False
    assert prf_utils.prf_cache_enabled({"psf": {"prf_cache": "true"}}) is True
    with pytest.raises(ValueError, match="prf_cache"):
        prf_utils.prf_cache_enabled({"psf": {"prf_cache": "maybe"}})


def test_cache_key_is_canonical():
    a = {"x": 1.0 * u.um, "y": [1, 2.0000000000001], "z": {"b": 1, "a": 2}}
    b = {"z": {"a": 2, "b": 1}, "y": [1, 2.0000000000002], "x": 1.0 * u.um}
    assert prf_utils.prf_cache_key(a) == prf_utils.prf_cache_key(b)
    assert prf_utils.prf_cache_key(a) != prf_utils.prf_cache_key({**a, "x": 2.0 * u.um})


def test_cache_in_a_forked_process(empty_cache):
    """A forked worker has its own copy of the cache and a usable lock."""
    import multiprocessing

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("no fork")
    prf_utils.cached_tables("a", _tables)
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(1) as pool:
        info = pool.apply(_worker_info)
    assert info == {"entries": 2, "hits": 1}
    assert prf_utils.prf_cache_info()["entries"] == 1  # the parent is unchanged


def _worker_info():
    prf_utils.cached_tables("a", _tables)  # inherited: a hit
    prf_utils.cached_tables("b", _tables)
    info = prf_utils.prf_cache_info()
    return {"entries": info["entries"], "hits": info["hits"]}
