"""
Population of the focal plane with the pixel response function (PRF).

The PRF plane is checked against brute-force integrals of the PSF over the
pixels (computed on a much finer grid, independently of the PRF tables), against
the sampled population (point-sampled PSFs followed by the IRF), against itself at
different oversampling factors and for the conservation of the flux.
"""

import astropy.units as u
import numpy as np
import pytest
from scipy.special import erf, expit, j1

import exosim.tasks.instrument as instrument
from exosim.models.signal import Sed, Signal
from exosim.tasks.instrument import (
    ApplyIntraPixelResponseFunction,
    CreateIntrapixelResponseFunction,
    PopulateFocalPlane,
    PopulateFocalPlanePrf,
)
from exosim.tasks.instrument.create_intrapixel_response_function import (
    irf_axis_response,
)
from exosim.utils.prf import clear_prf_cache, prf_cache_info
from exosim.utils.psf import create_psf

from .test_prf_inputs import paos_file

DELTA = 18.0


@pytest.fixture(autouse=True)
def _empty_prf_cache():
    clear_prf_cache()
    yield
    clear_prf_cache()


# ----------------------------------------------------------------- helpers
def make_parameters(fnum, lx, osf=1, npix=32, ch_type="photometer", **psf):
    return {
        "type": ch_type,
        "value": "test",
        "Fnum_x": fnum,
        "Fnum_y": fnum,
        "psf": {"shape": "airy", **psf},
        "detector": {
            "delta_pix": DELTA * u.um,
            "diffusion_length": lx * u.um,
            "intra_pix_distance": 0 * u.um,
            "oversampling": osf,
            "spatial_pix": npix,
            "spectral_pix": npix,
            "irf_task": "CreateIntrapixelResponseFunction",
        },
    }


def make_focal_plane(parameters, spectral):
    osf = parameters["detector"]["oversampling"]
    npix = parameters["detector"]["spatial_pix"]
    return Signal(
        spectral=np.asarray(spectral) * u.um,
        spatial=np.zeros(len(spectral)) * u.um,
        data=np.zeros((1, npix * osf, npix * osf)),
        metadata={"focal_plane_delta": DELTA * u.um / osf, "oversampling": osf},
    )


def make_source(wavelength, spectrum, **parsed):
    """A source after PropagateSources: a spectrum in ct/s/um."""
    source = Sed(
        spectral=np.asarray(wavelength) * u.um,
        data=np.asarray(spectrum)[None, None, :] * u.W / u.m**2 / u.um,
        metadata={"parsed_parameters": parsed},
    )
    source.data_units = u.ct / u.s / u.um
    return source


def flat_source(lo=0.9, hi=1.1, n=801):
    wl = np.linspace(lo, hi, n)
    return make_source(wl, np.ones(n))


def prf_plane(parameters, focal_plane, source, pointing=None):
    out, prf = PopulateFocalPlanePrf()(
        parameters=parameters,
        focal_plane=focal_plane,
        sources={"s": source},
        pointing=pointing,
    )
    return out.data[0], prf


def sampled_plane(parameters, focal_plane, source, pointing=None):
    """Sampled population: columns rescaling, point-sampled PSFs, IRF; decimated."""
    osf = parameters["detector"]["oversampling"]
    x = focal_plane.spectral
    step = np.zeros_like(x)
    positive = np.where(x > 0)
    step[positive] = np.gradient(x[positive])
    source.spectral_rebin(x)
    source *= np.abs(step) * u.um
    out, _ = PopulateFocalPlane()(
        parameters=parameters,
        focal_plane=focal_plane,
        sources={"s": source},
        pointing=pointing,
    )
    if osf > 1:
        kernel, delta = CreateIntrapixelResponseFunction()(parameters=parameters)
        out = ApplyIntraPixelResponseFunction()(
            focal_plane=out, irf_kernel=kernel, irf_kernel_delta=delta
        )
    return out.data[0][osf // 2 :: osf, osf // 2 :: osf]


def default_side(npix, sampling=10):
    """Fine samples of the default PSF window: a square as large as the detector
    side, odd, so that it is centred on the source."""
    side = npix * sampling
    return side if side % 2 == 1 else side - 1


def window_half(fnum, wl, npix, sampling=10):
    """Half width, in pixels, of the default PSF window of the tables: the light
    outside a square of the side of the detector is lost. It does not depend on
    the f-number or the wavelength."""
    return default_side(npix, sampling) / 2 / sampling


def brute_airy(fnum, lx, wl, y0, x0, rows, cols, half_window, ps_ref=21):
    """
    Energy collected by the pixels centred on ``rows`` x ``cols`` (pixel units)
    from an Airy point source at (y0, x0), per unit flux, inside a square window
    of half width ``half_window`` centred on the source. Direct sum over cells
    of 1 / ps_ref pixel (odd, so that the pixel edges fall on the cell edges).
    """
    h = 1.0 / ps_ref
    k_ext = 3
    ylo, yhi = min(rows) - k_ext - 1, max(rows) + k_ext + 1
    xlo, xhi = min(cols) - k_ext - 1, max(cols) + k_ext + 1
    yc = (np.arange(round((yhi - ylo) * ps_ref)) + 0.5) * h + ylo
    xc = (np.arange(round((xhi - xlo) * ps_ref)) + 0.5) * h + xlo
    yy, xx = np.meshgrid(yc, xc, indexing="ij")
    arg = np.pi * np.hypot(yy - y0, xx - x0) * DELTA / (fnum * wl) + 1e-20
    energy = (j1(arg) / arg) ** 2 * (h * DELTA) ** 2 / ((1 / np.pi) * (fnum * wl) ** 2)
    energy *= (np.abs(yy - y0) <= half_window) & (np.abs(xx - x0) <= half_window)
    sub = (np.arange(7) + 0.5) / 7 - 0.5
    lxq = (lx + 1e-20) * u.um

    def response(cells, centre):
        pts = ((cells - centre)[:, None] + sub[None, :] * h) * DELTA
        resp = irf_axis_response(pts.ravel() * u.um, DELTA * u.um, lxq, 0 * u.um)
        return resp.reshape(-1, sub.size).mean(axis=1)

    big = np.arange(-40 * ps_ref, 40 * ps_ref + 1) * h
    norm = response(big, 0.0).sum() * h
    py = np.array([response(yc, c) / norm for c in rows])
    px = np.array([response(xc, c) / norm for c in cols])
    return py @ energy @ px.T


def monochromatic(wl):
    """Unit flux source at one wavelength and the matching one-wavelength axis."""
    w = np.array([wl - 0.002, wl, wl + 0.002])
    return make_source(w, np.ones(3) / np.abs(np.gradient(w)).sum())


# --------------------------------------------------------------- brute force
@pytest.mark.parametrize(
    ("f_lambda", "sampling", "limit"),
    [(2.5, 10, 1.5e-3), (0.73, 20, 3e-3), (0.73, 10, 7e-3)],
)
@pytest.mark.parametrize("lx", [0.0, 1.7])
def test_centred_airy_matches_brute_force(f_lambda, sampling, limit, lx):
    """Centred Airy photometer, osf = 1: the central 5x5 pixels agree with the
    direct integration of PSF x pixel response (on a grid of 41 cells per pixel,
    61 for the sharp PSF). The faint pixels of the undersampled PSF
    (F wl / delta = 0.73), which hold 1% of the peak, need ``prf_sampling = 20``
    to reach 0.3%; with 10 they are within 0.7% and every pixel within 0.05% of
    the peak."""
    wl, npix = 1.0, 32
    fnum = f_lambda * DELTA / wl
    params = make_parameters(fnum, lx, 1, npix, prf_sampling=sampling)
    plane, _ = prf_plane(
        params, make_focal_plane(params, [wl] * npix), monochromatic(wl)
    )
    c = npix // 2
    sub = plane[c - 2 : c + 3, c - 2 : c + 3]
    truth = brute_airy(
        fnum,
        lx,
        wl,
        0.0,
        0.0,
        range(-2, 3),
        range(-2, 3),
        window_half(fnum, wl, npix, sampling),
        61 if f_lambda < 1 else 41,
    )
    big = truth > 1e-3 * truth.max()
    assert np.abs(sub[big] / truth[big] - 1).max() < limit
    assert np.abs(sub - truth).max() < 5e-4 * truth.max()
    # the flux in the window is the one of the table
    assert sub.sum() == pytest.approx(truth.sum(), rel=1e-3)


# ------------------------------------------------------------ osf independence
@pytest.mark.parametrize("lx", [0.0, 1.7])
def test_prf_does_not_depend_on_the_oversampling(lx):
    fnum = 2.5 * DELTA
    results = {}
    for osf in (1, 3, 5):
        n = 32 * osf
        params = make_parameters(fnum, lx, osf, 32)
        focal_plane = make_focal_plane(params, np.linspace(0.99, 1.01, n))
        plane, _ = prf_plane(params, focal_plane, flat_source())
        # the photometer band is the range of its axis: 0.02 um at any osf
        results[osf] = plane[osf // 2 :: osf, osf // 2 :: osf] / 0.02
    for osf in (3, 5):
        assert np.abs(results[osf] / results[1] - 1)[results[1] > 1e-3].max() < 2e-3
        assert np.abs(results[osf] - results[1]).max() < 1e-4 * results[1].max()


def test_subpixels_hold_the_signal_of_a_pixel_centred_on_them():
    """The sub-pixel 3m + 0 of an osf = 3 plane is a pixel centred 1/3 pixel to
    the left of the pixel m: it agrees with the brute-force integral."""
    wl, npix, lx = 1.0, 24, 1.7
    fnum = 2.5 * DELTA / wl
    params = make_parameters(fnum, lx, 3, npix)
    plane, _ = prf_plane(
        params, make_focal_plane(params, [wl] * (3 * npix)), monochromatic(wl)
    )
    c = npix // 2
    rows = [-1, 0, 1]
    cols = [-1 - 1 / 3, -1 / 3, 1 - 1 / 3]
    truth = brute_airy(fnum, lx, wl, 0.0, 0.0, rows, cols, window_half(fnum, wl, npix))
    got = plane[3 * c + 1 - 3 : 3 * c + 1 + 6 : 3, 3 * c - 3 : 3 * c + 4 : 3]
    assert np.abs(got / truth - 1).max() < 3e-3


# ----------------------------------------------------------- sampled population
@pytest.mark.parametrize(
    ("f_lambda", "lx", "limit"),
    [(0.73, 0.0, 6e-3), (0.73, 1.7, 5e-3), (2.5, 0.0, 5e-3), (2.5, 1.7, 5e-3)],
)
def test_sampled_population_converges_to_the_prf(f_lambda, lx, limit):
    """The sampled population at osf = 9 is within 0.5% (of the peak) of the PRF
    at osf = 1; at osf = 1 the point sampling is off by 7-30%."""
    wl, npix = 1.0, 32
    fnum = f_lambda * DELTA / wl

    def axis(osf):
        n = npix * osf
        return np.linspace(0.99, 1.01, n), 0.02 * n / (n - 1)

    params = make_parameters(fnum, lx, 1, npix)
    ax, flux = axis(1)
    prf, _ = prf_plane(params, make_focal_plane(params, ax), flat_source())
    prf /= 0.02  # the band of a photometer is the range of its axis
    params9 = make_parameters(fnum, lx, 9, npix)
    ax9, flux9 = axis(9)
    sampled9 = sampled_plane(params9, make_focal_plane(params9, ax9), flat_source())
    assert np.abs(sampled9 / flux9 - prf).max() < limit * prf.max()
    paramsn = make_parameters(fnum, lx, 1, npix)
    sampled1 = sampled_plane(paramsn, make_focal_plane(paramsn, ax), flat_source())
    assert sampled1.max() / flux > 1.05 * prf.max()


# ------------------------------------------------------------------ flux
@pytest.mark.parametrize("lx", [0.0, 1.7])
@pytest.mark.parametrize("f_lambda", [0.73, 2.5])
def test_flux_is_the_source_flux_times_the_energy_in_the_window(f_lambda, lx):
    npix = 32
    fnum = f_lambda * DELTA / 1.0
    wl = np.array([0.95, 1.0, 1.05])
    spectrum = np.array([1.0, 2.0, 0.5])
    params = make_parameters(fnum, lx, 1, npix)
    plane, _ = prf_plane(
        params,
        make_focal_plane(params, np.linspace(0.9, 1.1, npix)),
        make_source(wl, spectrum),
    )
    # the default window is a square of the detector side (the light outside it
    # is lost), the same at every wavelength
    side = default_side(npix)
    psfs = create_psf(
        wl * u.um,
        (fnum, fnum),
        DELTA * u.um / 10,
        array_size=[side, side],
        max_array_size=(npix * 10, npix * 10),
    )
    width = np.abs(np.gradient(wl))
    expected = sum(
        s * w * p.sum() for s, w, p in zip(spectrum, width, psfs, strict=True)
    )
    assert plane.sum() == pytest.approx(expected, rel=1e-3)


# ------------------------------------------------------------- pointing offset
def _offset_setup(shift_px, osf, npix=32):
    """Photometer, source at RA = 0, pointing displaced by shift_px pixels."""
    params = make_parameters(2.5 * DELTA, 1.7, osf, npix)
    params["detector"]["plate_scale"] = 10 * u.mas / u.um  # 180 mas per pixel
    focal_plane = make_focal_plane(params, np.linspace(0.99, 1.01, npix * osf))
    source = flat_source()
    source.metadata["parsed_parameters"] = {"ra": 0 * u.deg, "dec": 0 * u.deg}
    pointing = ((shift_px * 0.18) * u.arcsec, 0 * u.deg)
    return params, focal_plane, source, pointing


def _centroid(plane, osf):
    cols = (np.arange(plane.shape[1]) - osf // 2) / osf
    rows = (np.arange(plane.shape[0]) - osf // 2) / osf
    return (rows * plane.sum(1)).sum() / plane.sum(), (
        cols * plane.sum(0)
    ).sum() / plane.sum()


@pytest.mark.parametrize("osf", [1, 3])
def test_subpixel_pointing_offset_moves_the_centroid(osf):
    """A source offset by 0.37 pixel moves the centroid of the PRF plane by 0.37
    pixel; the sampled population moves it by the offset rounded to a whole
    number of sub-pixels."""
    centroids = {}
    for label, shift in (("ref", 0.0), ("shifted", 0.37)):
        params, focal_plane, source, pointing = _offset_setup(shift, osf)
        plane, _ = prf_plane(params, focal_plane, source, pointing)
        centroids[label] = _centroid(plane[osf // 2 :: osf, osf // 2 :: osf], 1)
    assert centroids["shifted"][1] - centroids["ref"][1] == pytest.approx(
        0.37, abs=0.01
    )
    assert centroids["shifted"][0] - centroids["ref"][0] == pytest.approx(0.0, abs=0.01)

    sampled = {}
    for label, shift in (("ref", 0.0), ("shifted", 0.37)):
        params, focal_plane, source, pointing = _offset_setup(shift, osf)
        sampled[label] = _centroid(
            sampled_plane(params, focal_plane, source, pointing), 1
        )
    rounded = round(0.37 * osf) / osf
    assert sampled["shifted"][1] - sampled["ref"][1] == pytest.approx(rounded, abs=0.01)


# ------------------------------------------------------------- spectral edge
def _edge_setup(osf, lx, npix=24, f_lambda=2.0):
    lam0, dpix, wlc = 1.9, 0.02, 2.0
    q_edge, width = 12.0, 0.34  # logistic: 10-90% over 1.5 pixel

    def spectrum(wl):
        return expit(((wl - lam0) / dpix - q_edge) / width)

    fnum = f_lambda * DELTA / wlc
    params = make_parameters(fnum, lx, osf, npix, ch_type="spectrometer")
    # wavelength of the centre of every sub-pixel, pixel m centred on osf*m + osf//2
    axis = lam0 + (np.arange(npix * osf) - osf // 2) / osf * dpix
    focal_plane = make_focal_plane(params, axis)
    wl = lam0 + np.arange(-3, npix + 3, 0.05) * dpix
    return params, focal_plane, make_source(wl, spectrum(wl)), (lam0, dpix, npix, fnum)


@pytest.mark.parametrize("lx", [0.0, 1.7])
def test_spectral_edge_matches_brute_force(lx):
    """A logistic edge over 1.5 pixel crossing the spectrum of a spectrometer:
    the column sums of the PRF plane agree with a brute-force reference built
    from a very fine spectral grid and fine pixels within 0.3% at the edge."""
    params, focal_plane, source, (lam0, dpix, npix, fnum) = _edge_setup(1, lx)
    plane, _ = prf_plane(params, focal_plane, source)
    rows = list(range(-7, 8))
    cols = list(range(5, 20))
    centre = npix // 2
    got = plane[centre + rows[0] : centre + rows[-1] + 1, cols[0] : cols[-1] + 1]

    wl = np.asarray(source.spectral)
    spectrum = source.data[0, 0]
    width = np.abs(np.gradient(wl))
    qx = (wl - lam0) / dpix
    half = window_half(fnum, focal_plane.spectral.max(), npix)
    ref = np.zeros(got.shape)
    for k in range(wl.size):
        # only the wavelengths that fall on the focal plane are populated
        if not -0.5 <= qx[k] <= npix - 0.5 or spectrum[k] < 1e-9:
            continue
        ref += (
            spectrum[k]
            * width[k]
            * brute_airy(
                fnum,
                lx,
                wl[k],
                float(centre),
                qx[k],
                [centre + r for r in rows],
                cols,
                half,
                11,
            )
        )
    ratio = got.sum(0) / ref.sum(0)
    signal = ref.sum(0) > 1e-2 * ref.sum(0).max()
    strong = ref.sum(0) > 1e-1 * ref.sum(0).max()
    assert strong.sum() >= 6
    assert np.abs(ratio[strong] - 1).max() < 1.5e-3
    assert np.abs(ratio[signal] - 1).max() < 6e-3
    assert np.abs(got - ref).max() < 1e-3 * ref.max()


def test_spectral_edge_does_not_depend_on_the_oversampling():
    """With a wavelength axis that places the physical pixel m at the same
    position at every oversampling, the PRF decimated at osf = 3 and 5 agrees
    with osf = 1 across a steep edge."""
    res = {}
    for osf in (1, 3, 5):
        params, focal_plane, source, _ = _edge_setup(osf, 1.7)
        plane, _ = prf_plane(params, focal_plane, source)
        res[osf] = plane[osf // 2 :: osf, osf // 2 :: osf]
    for osf in (3, 5):
        assert np.abs(res[osf] - res[1]).max() < 1e-3 * res[1].max()
        big = res[1] > 1e-2 * res[1].max()
        assert np.abs(res[osf][big] / res[1][big] - 1).max() < 5e-3


def _spectrometer_from_wl_solution(osf, w0, center=None, ncol=48, npix=32):
    """Spectrometer with a wavelength solution (0.0005 um/um, starting at
    ``w0``) and the focal plane array built as in a real channel, where the
    wavelength of every sub-pixel comes from the solution, so that the first
    and last sub-pixel centres depend on the oversampling."""
    from exosim.tasks.instrument.create_focal_plane_array import CreateFocalPlaneArray

    x = np.linspace(-100, 1000, 30) * u.um
    params = make_parameters(2.5 * DELTA, 1.7, osf, npix, ch_type="spectrometer")
    params["detector"]["spectral_pix"] = ncol
    params["wl_solution"] = {
        "wl_solution_task": "LoadWavelengthSolution",
        "data": {
            "Wavelength": (w0 + 0.0005 * x.value) * u.um,
            "x": x,
            "y": np.zeros(30) * u.um,
        },
    }
    params["wl_min"] = w0 * u.um
    params["wl_max"] = (w0 + 0.4) * u.um
    if center:
        params["wl_solution"]["center"] = center
    task = CreateFocalPlaneArray()
    task.set_log_name()
    return params, task(parameters=params, efficiency=None)


@pytest.mark.parametrize("w0", [1.0, 1.2])
@pytest.mark.parametrize("center", [None, "auto"])
def test_spectrometer_from_wavelength_solution_does_not_depend_on_osf(w0, center):
    """The wavelength axis of a real channel has first and last sub-pixel
    centres that depend on the oversampling: the nodes of the tables, the window
    of the PSF (proportional to the longest wavelength, rounded to a number of
    fine cells) and the coverage of the source must be defined on the physical
    pixels. With a spectrum starting at 1 um (F * lambda / d = 2.5) the window
    of osf 3 and 5 was one fine cell larger than the one of osf 1, and the
    signal differed by up to 0.8% above 5% of the peak."""
    res = {}
    for osf in (1, 3, 5):
        params, focal_plane = _spectrometer_from_wl_solution(osf, w0, center)
        # steep edge (logistic over 0.2 pixel) at the column 24.3
        edge = w0 + 0.0005 * 24.3 * DELTA
        wl = np.linspace(0.8, 2.2, 14001)
        source = make_source(wl, expit((wl - edge) / (0.2 * 0.009)))
        plane, prf = prf_plane(params, focal_plane, source)
        res[osf] = (plane[osf // 2 :: osf, osf // 2 :: osf], prf)
    for osf in (3, 5):
        assert res[osf][1].tables.shape == res[1][1].tables.shape
        np.testing.assert_allclose(res[osf][1].nodes, res[1][1].nodes, rtol=1e-9)
        diff = np.abs(res[osf][0] - res[1][0])
        assert diff[res[1][0] > 1e-3 * res[1][0].max()].max() < 1e-4 * res[1][0].max()


@pytest.mark.parametrize("lx", [0.0, 1.7])
def test_photometer_does_not_depend_on_the_oversampling(lx):
    """The wavelength axis of a photometer has ``32 * osf`` samples over the
    band: the part of the spectrum on the focal plane must not depend on it."""
    from exosim.tasks.instrument.create_focal_plane_array import CreateFocalPlaneArray

    res = {}
    for osf in (1, 3, 5):
        params = make_parameters(2.5 * DELTA, lx, osf, 32)
        wl_eff = np.linspace(0.9, 1.4, 200)
        efficiency = Signal(
            spectral=wl_eff * u.um,
            data=np.ones((1, 1, wl_eff.size)),
        )
        task = CreateFocalPlaneArray()
        task.set_log_name()
        focal_plane = task(parameters=params, efficiency=efficiency)
        wl = np.linspace(0.8, 1.6, 4001)
        source = make_source(wl, 1.0 + 0.5 * (wl - 1.0) ** 2)
        plane, _ = prf_plane(params, focal_plane, source)
        res[osf] = plane[osf // 2 :: osf, osf // 2 :: osf]
    for osf in (3, 5):
        diff = np.abs(res[osf] - res[1])
        assert diff[res[1] > 1e-3 * res[1].max()].max() < 1e-4 * res[1].max()


# ------------------------------------------------------------------ cache
@pytest.fixture
def count_builds(monkeypatch):
    """Counter of the calls that build PRF tables."""
    import exosim.tasks.instrument.populate_focal_plane_prf as module

    calls = []
    original = module.build_prf_tables

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "build_prf_tables", counting)
    return calls


def _populate(lx=1.7, scale=1.0, **psf):
    params = make_parameters(2.5 * DELTA, lx, 1, 32, **psf)
    focal_plane = make_focal_plane(params, np.linspace(0.9, 1.1, 32))
    source = flat_source(0.85, 1.15)
    source.data = source.data * scale
    return prf_plane(params, focal_plane, source)


def test_prf_tables_are_built_once_per_channel(count_builds):
    """Populating several targets with the same channel (new focal plane and
    source every time) builds the tables once, and the planes are identical to
    the ones without the cache."""
    first, tables = _populate()
    second, tables_again = _populate(scale=2.0)
    assert len(count_builds) == 1
    assert tables_again is tables
    assert prf_cache_info()["hits"] == 1
    np.testing.assert_array_equal(second, 2.0 * first)

    reference, _ = _populate(prf_cache=False)
    assert len(count_builds) == 2  # not cached: built again
    np.testing.assert_array_equal(reference, first)
    _populate(prf_cache="False")
    assert len(count_builds) == 3
    assert prf_cache_info()["entries"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"lx": 0.0},
        {"prf_sampling": 6},
        {"prf_nodes": 5},
        {"nzero": 6},
        {"shape": "gauss"},
    ],
)
def test_prf_cache_is_invalidated_by_the_psf_configuration(count_builds, change):
    _populate()
    _populate()
    assert len(count_builds) == 1
    kwargs = dict(change)
    lx = kwargs.pop("lx", 1.7)
    plane, _ = _populate(lx=lx, **kwargs)
    assert len(count_builds) == 2
    assert prf_cache_info()["entries"] == 2
    # and the new configuration is cached in turn
    again, _ = _populate(lx=lx, **kwargs)
    assert len(count_builds) == 2
    np.testing.assert_array_equal(again, plane)


def test_prf_cache_depends_on_the_f_number_and_the_band(count_builds):
    params = make_parameters(2.5 * DELTA, 1.7, 1, 32)
    for fnum, band in [(2.5, (0.9, 1.1)), (2.5, (0.9, 1.1)), (2.0, (0.9, 1.1))]:
        p = make_parameters(fnum * DELTA, 1.7, 1, 32)
        prf_plane(p, make_focal_plane(p, np.linspace(*band, 32)), flat_source())
    assert len(count_builds) == 2
    prf_plane(
        params, make_focal_plane(params, np.linspace(0.9, 1.2, 32)), flat_source()
    )
    assert len(count_builds) == 3  # a different band: different nodes and window


def test_prf_cache_of_a_paos_file_follows_the_file(tmp_path, count_builds):
    """The key holds the file name, size and modification time."""
    path = paos_file(tmp_path / "paos.h5", 16, 10.0)
    params = make_parameters(10.0, 0.0, 1, 16)
    params["psf"] = {"psf_task": "LoadPsfPaos", "filename": str(path)}
    for _ in range(2):
        prf_plane(
            params,
            make_focal_plane(params, np.linspace(0.65, 0.75, 16)),
            flat_source(0.6, 0.8, 401),
        )
    assert len(count_builds) == 1
    paos_file(path, 16, 14.0)  # rewritten: another PSF
    prf_plane(
        params,
        make_focal_plane(params, np.linspace(0.65, 0.75, 16)),
        flat_source(0.6, 0.8, 401),
    )
    assert len(count_builds) == 2


# -------------------------------------------------------------- PAOS PSF
def test_paos_psf_flux_and_shape(tmp_path):
    npix, sigma_um = 16, 10.0
    path = paos_file(tmp_path / "paos.h5", npix, sigma_um)
    params = make_parameters(10.0, 0.0, 1, npix)
    params["psf"] = {"psf_task": "LoadPsfPaos", "filename": str(path)}
    focal_plane = make_focal_plane(params, np.linspace(0.65, 0.75, npix))
    # light within 0.3% of the PAOS wavelength: the scaled PSF is the PAOS one
    plane, prf = prf_plane(params, focal_plane, flat_source(0.698, 0.702, 401))
    flux = 0.00401  # 0.5 per um over the 0.008 um, plus the bin edge
    # the nodes cover the band, the PSF beyond the PAOS wavelength is scaled
    assert prf.nodes[0] == pytest.approx(0.65)
    assert prf.nodes[-1] == pytest.approx(0.75)
    assert prf.energy[0] == pytest.approx(1.0, abs=2e-4)
    assert plane.sum() == pytest.approx(flux, rel=2e-3)
    # an exact Gaussian: the signal of the pixels is an error function
    sigma = sigma_um / DELTA

    def axis(d):
        return 0.5 * (
            erf((d + 0.5) / (np.sqrt(2) * sigma))
            - erf((d - 0.5) / (np.sqrt(2) * sigma))
        )

    c = npix // 2  # the source of a photometer is on the pixel N // 2
    d = np.arange(-3, 4)
    truth = flux * np.outer(axis(d), axis(d))
    got = plane[c - 3 : c + 4, c - 3 : c + 4]
    assert np.abs(got - truth).max() < 6e-3 * truth.max()


# ------------------------------------------------------------------- errors
def test_custom_irf_task_is_refused():
    params = make_parameters(20.0, 1.7)
    params["detector"]["irf_task"] = (
        instrument.CreateOversampledIntrapixelResponseFunction
    )
    with pytest.raises(ValueError, match="custom irf_task"):
        prf_plane(params, make_focal_plane(params, [1.0] * 32), monochromatic(1.0))


def test_time_dependent_and_unknown_psf_tasks_are_refused():
    params = make_parameters(20.0, 0.0)
    params["psf"] = {"psf_task": "LoadPsfPaosTimeInterp", "filename": "x.h5"}
    with pytest.raises(NotImplementedError, match="time"):
        prf_plane(params, make_focal_plane(params, [1.0] * 32), monochromatic(1.0))

    class MyPsf(instrument.LoadPsf):
        def model(self, filename, parameters, wavelength, time):
            return np.ones((1, 1, 3, 3))

    params["psf"] = {"psf_task": MyPsf, "filename": "x.h5"}
    with pytest.raises(NotImplementedError, match="LoadPsfPaos"):
        prf_plane(params, make_focal_plane(params, [1.0] * 32), monochromatic(1.0))


def test_plane_is_flagged_and_the_irf_is_not_applied_again():
    params = make_parameters(20.0, 1.7, 3, 16)
    focal_plane = make_focal_plane(params, np.linspace(0.99, 1.01, 48))
    out, prf = PopulateFocalPlanePrf()(
        parameters=params,
        focal_plane=focal_plane,
        sources={"s": flat_source()},
        pointing=None,
    )
    assert out.metadata["pixel_response_applied"] is True
    assert out.metadata["psf_population"] == "prf"
    assert out.metadata["prf_sampling"] == prf.sampling == 10
    before = out.data.copy()
    kernel, delta = CreateIntrapixelResponseFunction()(parameters=params)
    again = ApplyIntraPixelResponseFunction()(
        focal_plane=out, irf_kernel=kernel, irf_kernel_delta=delta
    )
    assert np.array_equal(again.data, before)
    assert out.data_units == u.ct / u.s
