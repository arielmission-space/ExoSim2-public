"""
Pixel response function (PRF) population of the focal plane.

The signal collected by a pixel centred at :math:`r_p` from a point source at
:math:`r_s` and wavelength :math:`\\lambda` is

.. math::
    \\mathrm{PRF}_\\lambda(r_p - r_s) = \\int \\mathrm{PSF}_\\lambda(r - r_s)\\,
    P(r - r_p)\\, d^2 r ,

where :math:`P(x, y) = p(x)\\,p(y)` is the probability that a photon landing at
:math:`(x, y)` from the centre of a pixel is collected by that pixel (the
extended Barron et al. 2007 response of
:func:`~exosim.tasks.instrument.create_intrapixel_response_function.irf_axis_response`,
normalised such that a uniform illumination of one unit per pixel gives one
unit per pixel).

This module tabulates the PRF on a fine grid of offsets and for a set of
wavelength nodes, and deposits it on the (oversampled) focal plane. Every
sub-pixel of the plane receives the signal that a physical pixel *centred on
that sub-pixel* would collect, which is what the sampled PSF followed by the
IRF convolution produces, but without the dependence on the oversampling
factor and with an exact pixel integration.
"""

import hashlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from numba import njit, prange
from scipy.ndimage import convolve1d

POPULATIONS = ("sampled", "prf")
DEFAULT_PRF_SAMPLING = 10
DEFAULT_PRF_NODES = 24
RESPONSE_SUBSAMPLES = 63
PRF_CACHE_ENTRIES = 4
PRF_CACHE_MAX_BYTES = 1 << 30


# ------------------------------------------------------------------- cache
# The tables depend only on the PSF configuration of the channel (not on the
# source, the target or the time), hence the tables of a channel are kept in a
# process-level cache: populating several targets or time steps with the same
# channel builds them once. Every process (e.g. every worker of a joblib or
# multiprocessing pool) has its own cache.
_CACHE = OrderedDict()
_CACHE_LOCK = threading.RLock()
_CACHE_STATS = {"hits": 0, "misses": 0}


def _reset_cache_lock():
    # a forked child must not inherit a lock held by another thread
    global _CACHE_LOCK
    _CACHE_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_cache_lock)


def prf_cache_enabled(parameters) -> bool:
    """
    Whether the PRF tables of a channel are cached: ``<psf><prf_cache>`` (default
    ``True``).
    """
    psf = parameters.get("psf") if hasattr(parameters, "get") else None
    value = True
    if psf is not None and hasattr(psf, "get"):
        value = psf.get("prf_cache", True)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("false", "0", "no", "off"):
            return False
        if text in ("true", "1", "yes", "on"):
            return True
        raise ValueError(f"<psf><prf_cache> must be True or False, got {value!r}")
    return bool(value)


def canonical(value):
    """
    Canonical, hashable and platform independent form of a configuration value
    (quantities in their own unit, floats rounded to 12 digits, dictionaries
    sorted, files by path, size and modification time).
    """
    if isinstance(value, u.Quantity):
        unit = value.unit
        return ("q", canonical(np.asarray(value.value).tolist()), str(unit))
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(f"{float(value):.12g}")
    if isinstance(value, dict) or hasattr(value, "items"):
        return tuple(sorted((str(k), canonical(v)) for k, v in value.items()))
    if isinstance(value, list | tuple):
        return tuple(canonical(v) for v in value)
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    return str(value)


def file_signature(filename):
    """Path, size and modification time of a file (the path only if missing)."""
    path = os.path.realpath(os.path.expanduser(str(filename).strip()))
    try:
        stat = os.stat(path)
    except OSError:
        return (path,)
    return (path, stat.st_size, stat.st_mtime_ns)


def prf_cache_key(config) -> str:
    """
    Hash of the inputs of the PRF tables, see
    :meth:`~exosim.tasks.instrument.populate_focal_plane_prf.PopulateFocalPlanePrf.cache_key`.
    """
    return hashlib.sha256(repr(canonical(config)).encode()).hexdigest()


def cached_tables(key, builder, enabled=True):
    """
    Returns the tables of the cache for a key, building them with
    ``builder()`` when missing.

    The cache keeps the last :data:`PRF_CACHE_ENTRIES` entries (least recently
    used first out) and no more than :data:`PRF_CACHE_MAX_BYTES` (the newest
    entry is always kept).

    Parameters
    ----------
    key: str
        key from :func:`prf_cache_key`
    builder: callable
        function without arguments that returns the :class:`PrfTables`
    enabled: bool
        if ``False`` the cache is bypassed

    Returns
    -------
    :class:`PrfTables`
        tables. They are shared, they must not be modified.
    """
    if not enabled:
        return builder()
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            _CACHE_STATS["hits"] += 1
            return _CACHE[key]
        _CACHE_STATS["misses"] += 1
    tables = builder()
    with _CACHE_LOCK:
        _CACHE[key] = tables
        _CACHE.move_to_end(key)
        while len(_CACHE) > 1 and (
            len(_CACHE) > PRF_CACHE_ENTRIES
            or sum(t.nbytes for t in _CACHE.values()) > PRF_CACHE_MAX_BYTES
        ):
            _CACHE.popitem(last=False)
    return tables


def clear_prf_cache():
    """Empties the cache of PRF tables of this process and resets its counters."""
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE_STATS.update(hits=0, misses=0)


def prf_cache_info() -> dict:
    """Entries, bytes, hits and misses of the cache of this process."""
    with _CACHE_LOCK:
        return {
            "entries": len(_CACHE),
            "bytes": sum(t.nbytes for t in _CACHE.values()),
            **_CACHE_STATS,
        }


def prf_unsupported_reason(parameters):
    """
    Why the PRF population cannot reproduce a channel configuration.

    The PRF integrates the analytic pixel response of the detector over the
    PSF. It cannot be used when:

    - ``detector`` has an ``irf_task`` that is not the default
      :class:`~exosim.tasks.instrument.create_intrapixel_response_function.CreateIntrapixelResponseFunction`
      (a custom response cannot be folded into the analytic one);
    - ``psf`` has a ``psf_task`` that is time dependent
      (:class:`~exosim.tasks.instrument.load_psf_paos_time_interp.LoadPsfPaosTimeInterp`)
      or that is not :class:`~exosim.tasks.instrument.load_psf_paos.LoadPsfPaos`
      (its output is on the oversampled grid);
    - ``detector`` has no ``delta_pix``.

    Parameters
    ----------
    parameters: dict
        channel parameters

    Returns
    -------
    str or None
        the reason, or ``None`` if the PRF population is supported.
    """
    # lazy imports: the tasks import this module
    import exosim.tasks.instrument as instrument
    from exosim.utils.klass_factory import find_task

    detector = parameters.get("detector") if hasattr(parameters, "get") else None
    if detector is None or not hasattr(detector, "get"):
        detector = {}
    if "irf_task" in detector:
        klass = find_task(
            detector["irf_task"], instrument.CreateIntrapixelResponseFunction
        )
        if klass is not instrument.CreateIntrapixelResponseFunction:
            return (
                f"a custom irf_task ({detector['irf_task']}) is used: the pixel "
                "response of the PRF is computed analytically from "
                "diffusion_length, intra_pix_distance and delta_pix"
            )
    psf = parameters.get("psf") if hasattr(parameters, "get") else None
    if psf is not None and hasattr(psf, "get") and "psf_task" in psf:
        psf_task = find_task(psf["psf_task"], instrument.LoadPsf)
        if issubclass(psf_task, instrument.LoadPsfPaosTimeInterp):
            return f"the PSF is time dependent ({psf['psf_task']})"
        if not issubclass(psf_task, instrument.LoadPsfPaos):
            return (
                f"the psf_task ({psf['psf_task']}) is not LoadPsfPaos: its output "
                "is on the oversampled grid and cannot be used to tabulate the PRF"
            )
    if "delta_pix" not in detector:
        return "delta_pix is not defined in the detector section"
    return None


def get_population(parameters, on_fallback=None) -> str:
    """
    Focal plane population mode of a channel: ``"prf"`` (default) or
    ``"sampled"``.

    It reads ``parameters["psf"]["population"]``. When the key is absent the
    population is ``"prf"``, unless the configuration is one the PRF cannot
    reproduce (see :func:`prf_unsupported_reason`): then it is ``"sampled"``.
    An explicit value is always used as given.

    Parameters
    ----------
    parameters: dict
        channel parameters
    on_fallback: callable (optional)
        called with the reason (a string) when the default falls back to
        ``"sampled"``. It is called at every evaluation: the caller decides
        how often to report it.
    """
    psf = parameters.get("psf") if hasattr(parameters, "get") else None
    if psf is not None and hasattr(psf, "get") and "population" in psf:
        mode = str(psf["population"]).strip().lower()
        if mode not in POPULATIONS:
            raise ValueError(
                f"<psf><population> must be one of {POPULATIONS}, got {mode!r}"
            )
        return mode
    reason = prf_unsupported_reason(parameters)
    if reason is None:
        return "prf"
    if on_fallback is not None:
        on_fallback(reason)
    return "sampled"


def fine_pixel_response(delta, lx, ipd, sampling, n_sub=RESPONSE_SUBSAMPLES):
    """
    1D pixel response :math:`p`, sampled ``n_sub`` times per fine cell.

    The array is centred (odd number of fine cells) and spans the pixel and the
    ``K`` neighbouring pixels on each side, as in
    :class:`~exosim.tasks.instrument.create_intrapixel_response_function.CreateIntrapixelResponseFunction`
    (the response is a step function for a vanishing diffusion length, which
    is why it is sampled much more finely than the PRF tables). It is
    normalised such that the mean over a pixel of a uniform illumination is the
    unit: ``sum / n_sub = sampling``. The gaps between the pixels are zeroed
    before the normalisation, as for the IRF.

    Parameters
    ----------
    delta: :class:`astropy.units.Quantity`
        pixel pitch
    lx: :class:`astropy.units.Quantity`
        diffusion length (``0`` allowed)
    ipd: :class:`astropy.units.Quantity`
        inter-pixel distance (gap)
    sampling: int
        fine samples (cells) per physical pixel
    n_sub: int
        samples per fine cell

    Returns
    -------
    :class:`numpy.ndarray`
        response samples, ``n_cells * n_sub`` of them. The cell average is
        ``response.reshape(-1, n_sub).mean(axis=1)``.
    """
    # imported here: the task package imports this module
    from exosim.tasks.instrument.create_intrapixel_response_function import (
        irf_axis_response,
        irf_extent_pixels,
    )

    lx = (lx + 1e-20 * u.um).to(delta.unit)
    n_ext = irf_extent_pixels(delta, lx)
    n = sampling * (2 * n_ext + 1)
    if n % 2 == 0:
        n += 1
    h = delta / sampling
    pos = (np.arange(n * n_sub) - (n * n_sub - 1) / 2.0) / n_sub * h
    resp = irf_axis_response(pos, delta, lx, ipd)
    return resp * (sampling * n_sub / resp.sum())


def node_weights(wavelength, nodes):
    """
    Linear interpolation weights of wavelengths between sorted nodes.

    Returns the index ``i`` of the lower node and the weight ``w`` of that
    node, the upper node (``i + 1``) having the weight ``1 - w``. Wavelengths
    outside the nodes are clamped to the first/last node.
    """
    nodes = np.asarray(nodes, float)
    wavelength = np.asarray(wavelength, float)
    if nodes.size == 1:
        return np.zeros(wavelength.shape, int), np.ones(wavelength.shape)
    idx = np.clip(np.searchsorted(nodes, wavelength) - 1, 0, nodes.size - 2)
    w_up = np.clip((wavelength - nodes[idx]) / (nodes[idx + 1] - nodes[idx]), 0.0, 1.0)
    return idx.astype(int), 1.0 - w_up


@dataclass
class PrfTables:
    """
    Tabulated pixel response function.

    Attributes
    ----------
    nodes: :class:`numpy.ndarray`
        wavelengths of the tables, in um, ascending
    tables: :class:`numpy.ndarray`
        ``(n_nodes, ny, nx)`` array. Element ``[n, cy + j, cx + i]`` is the
        signal collected by a pixel whose centre is ``(j, i) / sampling``
        pixels away from the source (axis 0 is the spatial, axis 1 the
        spectral direction of the focal plane), per unit of source flux.
    sampling: int
        fine samples per physical pixel
    centre: tuple
        ``(cy, cx)`` index of the zero offset
    response: :class:`numpy.ndarray`
        1D pixel response used to build the tables
    energy: :class:`numpy.ndarray`
        energy of the fine PSF of every node (fraction of the total inside the
        PSF window)
    """

    nodes: np.ndarray
    tables: np.ndarray
    sampling: int
    centre: tuple
    response: np.ndarray
    energy: np.ndarray

    @property
    def extent_pixels(self) -> tuple:
        """Half extent of the tables, in physical pixels, along each axis."""
        return (
            (self.tables.shape[1] - 1 - self.centre[0]) / self.sampling,
            (self.tables.shape[2] - 1 - self.centre[1]) / self.sampling,
        )

    @property
    def nbytes(self) -> int:
        return int(self.tables.nbytes)


def quadrature_kernel(response, n_sub=RESPONSE_SUBSAMPLES):
    """
    Discrete kernel that integrates a PSF sampled on the fine cells against
    the pixel response.

    The samples of the PSF are interpolated with the same cubic convolution
    kernel :math:`K` (Keys, a = -0.5) used to read the tables, and the
    interpolant is integrated against the response:
    ``kernel[c] = 1/h \\int p(x) K(x/h - c) dx``.
    The plain sum of the fine cells times the (cell averaged) response makes a
    first order error where a pixel edge falls on a fine cell; this
    quadrature, instead, is third order accurate, and it is local (a
    Fourier-space convolution with the continuous transform of the response
    is accurate for band limited PSFs but rings at the truncation of the PSF
    window). The sum of the kernel is that of the response (``sampling``).

    Parameters
    ----------
    response: :class:`numpy.ndarray`
        response from :func:`fine_pixel_response` (odd number of cells)
    n_sub: int
        samples per fine cell of ``response`` (odd)

    Returns
    -------
    :class:`numpy.ndarray`
        kernel on the fine cells (odd number of cells, centred), 4 cells
        longer than the response
    """
    n_sub = int(n_sub)
    if n_sub % 2 == 0 or (response.size // n_sub) % 2 == 0:
        raise ValueError("n_sub and the number of cells must be odd")
    t = np.abs(np.arange(-2 * n_sub, 2 * n_sub + 1)) / n_sub
    keys = np.where(
        t < 1,
        1.5 * t**3 - 2.5 * t**2 + 1.0,
        np.where(t < 2, -0.5 * t**3 + 2.5 * t**2 - 4.0 * t + 2.0, 0.0),
    )
    smooth = np.convolve(response, keys, mode="full") / n_sub
    n_cells = response.size // n_sub
    c = np.arange(-((n_cells - 1) // 2 + 2), (n_cells - 1) // 2 + 3)
    return smooth[n_sub * (c + 2) + (response.size - 1) // 2]


def build_prf_tables(
    fine_psf, nodes, response, sampling, centre=None, n_sub=RESPONSE_SUBSAMPLES
):
    """
    PRF tables from fine PSFs.

    The PRF is the convolution of the PSF with the pixel response, computed
    with the :func:`quadrature_kernel`.

    Parameters
    ----------
    fine_psf: :class:`numpy.ndarray`
        ``(n_nodes, ny, nx)`` array. Each image holds the *energy* in the fine
        cell (cells of size ``pixel / sampling``), so that its sum is the
        fraction of the energy of the PSF inside the window.
    nodes: :class:`numpy.ndarray`
        wavelength of each PSF, in um, ascending
    response: :class:`numpy.ndarray`
        response from :func:`fine_pixel_response`
    sampling: int
        fine samples per physical pixel
    centre: tuple (optional)
        ``(cy, cx)`` index of the PSF centre. Default is ``(ny // 2, nx // 2)``.
    n_sub: int
        samples per fine cell of ``response``

    Returns
    -------
    :class:`PrfTables`
    """
    fine_psf = np.asarray(fine_psf)
    if fine_psf.ndim != 3:
        raise ValueError(f"fine_psf must be 3D, got shape {fine_psf.shape}")
    nodes = np.asarray(nodes, float)
    if nodes.size != fine_psf.shape[0]:
        raise ValueError("one wavelength node is required for every fine PSF")
    if np.any(np.diff(nodes) <= 0):
        raise ValueError("the wavelength nodes must be strictly increasing")
    n, ny, nx = fine_psf.shape
    if centre is None:
        centre = (ny // 2, nx // 2)
    kernel = quadrature_kernel(response, n_sub)
    # the extra 2 samples hold the zeros the cubic interpolation needs
    pad = kernel.size // 2 + 2
    shape = (ny + 2 * pad, nx + 2 * pad)
    tables = np.zeros((n, *shape), dtype=np.float32)
    buf = np.zeros(shape)
    for i in range(n):
        buf[:] = 0.0
        buf[pad : pad + ny, pad : pad + nx] = fine_psf[i]
        tmp = convolve1d(buf, kernel, axis=0, mode="constant")
        tables[i] = convolve1d(tmp, kernel, axis=1, mode="constant")
    if n == 1:
        # a single wavelength: duplicate it, so that the interpolation works
        tables = np.repeat(tables, 2, axis=0)
        nodes = np.array([nodes[0], nodes[0] * (1 + 1e-6)])
    return PrfTables(
        nodes=nodes,
        tables=tables,
        sampling=int(sampling),
        centre=(centre[0] + pad, centre[1] + pad),
        response=response.reshape(-1, n_sub).mean(axis=1),
        energy=fine_psf.sum(axis=(1, 2)),
    )


@njit(cache=True)
def _keys_weights(f, w):
    """Cubic convolution (Keys, a = -0.5) weights for the fractional part f."""
    f2 = f * f
    f3 = f2 * f
    w[0] = -0.5 * f3 + f2 - 0.5 * f
    w[1] = 1.5 * f3 - 2.5 * f2 + 1.0
    w[2] = -1.5 * f3 + 2.0 * f2 + 0.5 * f
    w[3] = 0.5 * f3 - 0.5 * f2


@njit(cache=True, parallel=True)
def _deposit(plane, tables, node0, w0, qy, qx, flux, osf, ps, cy, cx):
    n_t, n_rows, n_cols = plane.shape
    n_e = qy.shape[0]
    ty = tables.shape[1]
    tx = tables.shape[2]
    half = osf // 2
    for a in prange(n_rows):
        wy = np.empty(4)
        wx = np.empty(4)
        ya = (a - half) / osf
        for e in range(n_e):
            uy = cy + (ya - qy[e]) * ps
            if uy < 1.0 or uy >= ty - 2.0:
                continue
            iy = int(np.floor(uy))
            _keys_weights(uy - iy, wy)
            blo = int(np.ceil(half + (qx[e] + (1.0 - cx) / ps) * osf))
            bhi = int(np.ceil(half + (qx[e] + ((tx - 2.0) - cx) / ps) * osf)) - 1
            if blo < 0:
                blo = 0
            if bhi > n_cols - 1:
                bhi = n_cols - 1
            n0 = node0[e]
            ww0 = w0[e]
            for b in range(blo, bhi + 1):
                ux = cx + ((b - half) / osf - qx[e]) * ps
                ix = int(np.floor(ux))
                if ix < 1 or ix > tx - 3:
                    continue
                _keys_weights(ux - ix, wx)
                v0 = 0.0
                for i in range(4):
                    r = 0.0
                    for j in range(4):
                        r += wx[j] * tables[n0, iy + i - 1, ix + j - 1]
                    v0 += wy[i] * r
                val = ww0 * v0
                if ww0 < 1.0:
                    v1 = 0.0
                    for i in range(4):
                        r = 0.0
                        for j in range(4):
                            r += wx[j] * tables[n0 + 1, iy + i - 1, ix + j - 1]
                        v1 += wy[i] * r
                    val += (1.0 - ww0) * v1
                for t in range(n_t):
                    plane[t, a, b] += flux[t, e] * val


def deposit_prf(plane, prf, node0, w0, qy, qx, flux, osf):
    """
    Adds the PRF of point sources to an oversampled focal plane.

    The sub-pixel ``(a, b)`` of the plane is the centre of the physical pixel
    at ``((a - osf // 2) / osf, (b - osf // 2) / osf)`` in units of physical
    pixels (the centre of the physical pixel ``m`` is the sub-pixel
    ``osf * m + osf // 2``, the one picked by the decimation
    ``[osf // 2 :: osf]``). Every sub-pixel receives
    ``flux * PRF(centre - source position)``.

    Parameters
    ----------
    plane: :class:`numpy.ndarray`
        ``(time, rows, cols)`` plane, modified in place and returned
    prf: :class:`PrfTables`
        PRF tables
    node0: :class:`numpy.ndarray`
        for every source sample, index of the lower wavelength node
    w0: :class:`numpy.ndarray`
        for every source sample, weight of the node ``node0``
        (``1 - w0`` is the one of ``node0 + 1``)
    qy, qx: :class:`numpy.ndarray`
        position of every source sample along the spatial (rows) and
        spectral (columns) axis, in physical pixels
    flux: :class:`numpy.ndarray`
        ``(time, samples)`` flux of every source sample
    osf: int
        oversampling factor of the plane

    Returns
    -------
    :class:`numpy.ndarray`
        the plane
    """
    if plane.ndim != 3:
        raise ValueError(f"the plane must be 3D (time, rows, cols), got {plane.shape}")
    flux = np.ascontiguousarray(flux, dtype=np.float64)
    if flux.shape != (plane.shape[0], np.size(qy)):
        raise ValueError(
            f"flux must have shape (time, samples) = {(plane.shape[0], np.size(qy))}, "
            f"got {flux.shape}"
        )
    if plane.shape[1] % osf or plane.shape[2] % osf:
        raise ValueError("the plane size is not a multiple of the oversampling factor")
    _deposit(
        plane,
        prf.tables,
        np.ascontiguousarray(node0, dtype=np.int64),
        np.ascontiguousarray(w0, dtype=np.float64),
        np.ascontiguousarray(qy, dtype=np.float64),
        np.ascontiguousarray(qx, dtype=np.float64),
        flux,
        int(osf),
        float(prf.sampling),
        float(prf.centre[0]),
        float(prf.centre[1]),
    )
    return plane


def wavelength_to_subpixel(wavelength, axis_wavelength):
    """
    Continuous sub-pixel index of the wavelengths, given the wavelength of
    every sub-pixel of an axis.

    Parameters
    ----------
    wavelength: :class:`numpy.ndarray`
        wavelengths to locate, in um
    axis_wavelength: :class:`numpy.ndarray`
        wavelength of every sub-pixel of the axis, in um. Values lower or equal
        to zero are ignored. It must be monotonic.

    Returns
    -------
    :class:`numpy.ndarray`
        sub-pixel index (continuous, linearly extrapolated beyond the axis)
    :class:`numpy.ndarray`
        boolean mask of the wavelengths that fall on the axis (that is, inside
        the cell edges of the first and last sub-pixel)
    """
    axis_wavelength = np.asarray(axis_wavelength, float)
    index = np.arange(axis_wavelength.size, dtype=float)
    ok = axis_wavelength > 0.0
    if ok.sum() < 2:
        raise ValueError("at least two sub-pixels with a positive wavelength needed")
    wl_ax, idx_ax = axis_wavelength[ok], index[ok]
    steps = np.diff(wl_ax)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        raise ValueError("the wavelength solution is not monotonic")
    order = np.argsort(wl_ax)
    wl_ax, idx_ax = wl_ax[order], idx_ax[order]
    wavelength = np.asarray(wavelength, float)
    pos = np.interp(wavelength, wl_ax, idx_ax)
    lo = (idx_ax[1] - idx_ax[0]) / (wl_ax[1] - wl_ax[0])
    hi = (idx_ax[-1] - idx_ax[-2]) / (wl_ax[-1] - wl_ax[-2])
    pos = np.where(wavelength < wl_ax[0], idx_ax[0] + (wavelength - wl_ax[0]) * lo, pos)
    pos = np.where(
        wavelength > wl_ax[-1], idx_ax[-1] + (wavelength - wl_ax[-1]) * hi, pos
    )
    inside = (pos >= idx_ax.min() - 0.5) & (pos <= idx_ax.max() + 0.5)
    return pos, inside


def physical_axis_edges(axis_wavelength, osf=1):
    """
    Wavelengths of the two edges of the physical pixel axis.

    The sub-pixel :math:`j` of an axis oversampled by ``osf`` is the centre of
    the physical pixel at :math:`q = (j - \\mathrm{osf} // 2) / \\mathrm{osf}`
    (the centre of the physical pixel :math:`m` is the sub-pixel
    ``osf * m + osf // 2``), so that the physical pixels span
    :math:`q \\in [-1/2, N - 1/2]`. The wavelengths of these edges (linearly
    extrapolated from the end sub-pixels) are the same at every ``osf`` (up to the curvature
    of the wavelength solution), whereas the first and last sub-pixel centres
    are not: they depend on the oversampling. Sub-pixels with a non-positive
    wavelength are ignored, and the edge is moved to the outer edge of the
    first/last valid sub-pixel when the end of the axis is not valid.

    Parameters
    ----------
    axis_wavelength: :class:`numpy.ndarray`
        wavelength of every sub-pixel of the axis, in um
    osf: int
        oversampling factor of the axis

    Returns
    -------
    float
        lower wavelength of the edges, in um
    float
        upper wavelength of the edges, in um
    """
    axis = np.asarray(axis_wavelength, float)
    osf = int(osf)
    index = np.arange(axis.size, dtype=float)
    ok = axis > 0.0
    if ok.sum() < 2:
        raise ValueError("at least two sub-pixels with a positive wavelength needed")
    wl_ax, idx_ax = axis[ok], index[ok]
    n_phys = axis.size / osf
    # physical coordinate of the first/last valid sub-pixel outer edges
    q_lo = -0.5 if idx_ax[0] == 0 else (idx_ax[0] - osf // 2 - 0.5) / osf
    q_hi = (
        n_phys - 0.5
        if idx_ax[-1] == axis.size - 1
        else (idx_ax[-1] - osf // 2 + 0.5) / osf
    )
    j_edges = np.array([q_lo, q_hi]) * osf + osf // 2
    # linear interpolation of the valid values, linear extrapolation beyond
    wl_edges = np.interp(j_edges, idx_ax, wl_ax)
    lo_slope = (wl_ax[1] - wl_ax[0]) / (idx_ax[1] - idx_ax[0])
    hi_slope = (wl_ax[-1] - wl_ax[-2]) / (idx_ax[-1] - idx_ax[-2])
    wl_edges = np.where(
        j_edges < idx_ax[0], wl_ax[0] + (j_edges - idx_ax[0]) * lo_slope, wl_edges
    )
    wl_edges = np.where(
        j_edges > idx_ax[-1], wl_ax[-1] + (j_edges - idx_ax[-1]) * hi_slope, wl_edges
    )
    return float(wl_edges.min()), float(wl_edges.max())


def wavelength_coverage(wavelength, bin_width, axis_wavelength, osf=1, edges=None):
    """
    Fraction of the spectral bin of every wavelength that falls on a
    wavelength axis, that is, inside the edges of its physical pixels (see
    :func:`physical_axis_edges`: independent of the oversampling). The bin of
    the wavelength :math:`\\lambda_k` is
    :math:`\\lambda_k \\pm \\Delta\\lambda_k / 2`. An axis with a single
    wavelength covers everything.

    Parameters
    ----------
    wavelength: :class:`numpy.ndarray`
        wavelengths of the source samples, in um
    bin_width: :class:`numpy.ndarray`
        width of the spectral bin of every sample, in um
    axis_wavelength: :class:`numpy.ndarray`
        wavelength axis, in um
    osf: int (optional)
        oversampling of the axis. Default is 1.
    edges: (float, float) (optional)
        wavelength limits to use instead of the edges of the axis

    Returns
    -------
    :class:`numpy.ndarray`
        weights between 0 and 1
    """
    axis = np.sort(np.asarray(axis_wavelength, float))
    axis = axis[axis > 0.0]
    wavelength = np.asarray(wavelength, float)
    if edges is not None:
        lo, hi = edges
    elif axis.size < 2 or axis[0] == axis[-1]:
        return np.ones(wavelength.shape)
    else:
        lo, hi = physical_axis_edges(axis_wavelength, osf)
    bin_lo, bin_hi = wavelength - 0.5 * bin_width, wavelength + 0.5 * bin_width
    overlap = np.clip(np.minimum(bin_hi, hi) - np.maximum(bin_lo, lo), 0.0, None)
    return overlap / bin_width
