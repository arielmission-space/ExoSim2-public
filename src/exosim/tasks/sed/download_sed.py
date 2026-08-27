"""
DownloadPhoenix task: fetches stellar SEDs from online Phoenix spectral databases.

Two backends are supported, selected automatically by *model_name*:

* **phoenix-aces**
  PHOENIX-ACES-AGSS-COND-2011 from the Goettingen server
  (https://phoenix.astro.physik.uni-goettingen.de).
  The parameter grid is hardcoded; only astropy and numpy are required.

* **SVO** (bt-settl, bt-settl-cifist, bt-nextgen, nextgen, ...)
  The Spanish Virtual Observatory (https://svo2.cab.inta-csic.es/theory/newov2/)
  is queried at run time using stdlib ``urllib`` and ``html.parser``; no
  extra packages are required.

In both cases ``astropy.utils.data.download_file`` is used for the actual HTTP
transfer; files are cached in the default astropy cache directory so they are
not re-downloaded on subsequent calls.
"""

import logging
import os
import re
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.utils.data import download_file

import exosim.models.signal as signal
import exosim.utils.checks as checks
from exosim.tasks.task import Task

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Goettingen Phoenix-ACES backend
# ---------------------------------------------------------------------------

_GOETTINGEN_BASE = "https://phoenix.astro.physik.uni-goettingen.de/data/v2.0/HiResFITS/"
_ACES_MODEL = "PHOENIX-ACES-AGSS-COND-2011"
_ACES_WAVE_URL = f"{_GOETTINGEN_BASE}WAVE_{_ACES_MODEL}.fits"

# Full parameter grids for PHOENIX-ACES-AGSS-COND-2011
_ACES_TEFF = np.concatenate([np.arange(2300, 7100, 100), np.arange(7200, 12200, 200)])
_ACES_LOGG = np.arange(0.0, 6.5, 0.5)
_ACES_FEH = np.array([-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])


def _snap(value, grid):
    """Return the element of *grid* nearest to *value*."""
    return grid[int(np.argmin(np.abs(grid - value)))]


def _aces_url(teff, logg, feh):
    """Build the Goettingen HiResFITS URL and return (url, snapped params)."""
    t = int(_snap(teff, _ACES_TEFF))
    g = float(_snap(logg, _ACES_LOGG))
    z = float(_snap(feh, _ACES_FEH))
    # Filename convention: lte{TTTTT}-{logg:.2f}{feh:+.1f}.MODEL-HiRes.fits
    # The '-' before logg is a literal separator, not a sign.
    filename = f"lte{t:05d}-{g:.2f}{z:+.1f}.{_ACES_MODEL}-HiRes.fits"
    url = f"{_GOETTINGEN_BASE}{_ACES_MODEL}/Z{z:+.1f}/{filename}"
    return url, t, g, z


def _fetch_aces(teff, logg, feh):
    """Download Phoenix-ACES wave + flux from Goettingen and return Quantities."""
    url, t, g, z = _aces_url(teff, logg, feh)

    # Check Goettingen server reachability before attempting downloads
    if not _host_is_reachable(_GOETTINGEN_BASE):
        raise ConnectionError(
            f"Cannot reach Goettingen server at {_GOETTINGEN_BASE}; no network"
        )

    logger.info(f"Downloading Phoenix files: wave={_ACES_WAVE_URL}, flux={url}")
    logger.debug("Downloading Phoenix WAVE: %s", _ACES_WAVE_URL)
    logger.debug("Downloading Phoenix flux: %s", url)

    wave_path = download_file(_ACES_WAVE_URL, pkgname="exosim", cache=True)
    flux_path = download_file(url, pkgname="exosim", cache=True)

    with fits.open(wave_path) as hdul:
        wl = np.asarray(hdul[0].data) * u.AA

    with fits.open(flux_path) as hdul:
        # Surface flux density: erg s-1 cm-2 cm-1
        flux = np.asarray(hdul[0].data) * (u.erg / u.s / u.cm**2 / u.cm)

    meta = {
        "teff_grid": t,
        "logg_grid": g,
        "feh_grid": z,
        "model": _ACES_MODEL,
        "source": _GOETTINGEN_BASE,
    }
    # Prefer the remote filename (from URL path); fall back to cached file name
    try:
        fname = os.path.basename(urllib.parse.urlparse(url).path) or None
        if not fname:
            fname = os.path.basename(flux_path)
        meta.update({"filename": fname})
    except Exception:
        pass
    return wl, flux, meta


# ---------------------------------------------------------------------------
# SVO backend
# ---------------------------------------------------------------------------

_SVO_BASE = "https://svo2.cab.inta-csic.es/theory/newov2/"


def _host_is_reachable(base_url: str, timeout: float = 3.0) -> bool:
    """Return True if the host for *base_url* accepts TCP connections.

    This attempts a short TCP connection to port 443 for HTTPS (or 80 for
    HTTP). It may give false negatives in captive or filtered networks.
    """
    try:
        parsed = urllib.parse.urlparse(base_url)
        host = parsed.hostname
        if not host:
            return False
        port = 443 if parsed.scheme == "https" else 80
        conn = socket.create_connection((host, port), timeout=timeout)
        conn.close()
        return True
    except Exception:
        return False


class _SVOModelsParser(HTMLParser):
    """Parse SVO index HTML to extract available model identifiers.

    The parser looks for ``<select name="models">``/``<option value="...">``
    and for inputs named ``reqmodels[]`` with a ``value`` attribute.
    """

    def __init__(self):
        super().__init__()
        self._in_models_select = False
        self.models: set[str] = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "select" and attrs.get("name", "") in ("models", "reqmodels[]"):
            self._in_models_select = True
        elif (tag == "option" and self._in_models_select) or (
            tag == "input" and attrs.get("name") == "reqmodels[]"
        ):
            val = attrs.get("value")
            if val:
                self.models.add(val)
        elif tag == "a":
            # Links like index.php?models=bt-settl appear as model selectors
            href = attrs.get("href", "")
            if "models=" in href:
                try:
                    qs = urllib.parse.urlparse(href).query
                    params = urllib.parse.parse_qs(qs)
                    vals = params.get("models") or params.get("reqmodels[]")
                    if vals:
                        for v in vals:
                            if v:
                                self.models.add(v)
                except Exception:
                    pass

    def handle_endtag(self, tag):
        if tag == "select":
            self._in_models_select = False


def get_svo_models() -> list[str]:
    """Query the SVO theory index page and return a sorted list of model ids.

    This performs a live HTTP GET against the SVO index and parses the HTML
    for model identifiers. Network errors will propagate as exceptions.
    """
    url = urllib.parse.urljoin(_SVO_BASE, "index.php")
    # Quick connectivity check to fail fast if offline
    if not _host_is_reachable(_SVO_BASE):
        raise ConnectionError(f"Cannot reach SVO service at {_SVO_BASE}; no network")

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    parser = _SVOModelsParser()
    parser.feed(html)
    return sorted(parser.models)


class _SVOTableParser(HTMLParser):
    """Minimal HTML parser that extracts rows from the SVO results table.

    The SVO HTML uses:
    - ``<td class="tabcab">`` for column headers
    - ``<td class="tabfld">`` for data cells; download links live inside
      an ``<a href="...">`` within the 3rd-from-last data cell of each row.
    """

    def __init__(self):
        super().__init__()
        self.headers: list[str] = []
        self._rows: list[list[tuple[str, str | None]]] = []
        self._row: list[tuple[str, str | None]] | None = None
        self._mode = ""  # "hdr" | "data" | ""
        self._cell_text = ""
        self._cell_link: str | None = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag == "td":
            cls = attrs.get("class", "")
            if cls == "tabcab":
                self._mode = "hdr"
            elif cls == "tabfld":
                self._mode = "data"
            else:
                self._mode = ""
            self._cell_text = ""
            self._cell_link = None
        elif tag == "a" and self._mode == "data":
            href = attrs.get("href", "")
            if href:
                self._cell_link = href

    def handle_endtag(self, tag):
        if tag == "td":
            text = self._cell_text.strip()
            if self._mode == "hdr" and text:
                self.headers.append(text)
            elif self._mode == "data" and self._row is not None:
                self._row.append((text, self._cell_link))
            self._mode = ""
        elif tag == "tr":
            if self._row:
                self._rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._mode in ("hdr", "data"):
            self._cell_text += data

    def models(self):
        """Return list of dicts with teff/logg/feh/alpha/url for each row."""

        def _col(name, default):
            # case-insensitive partial match in headers
            name_l = name.lower()
            for i, h in enumerate(self.headers):
                if name_l in h.lower():
                    return i
            return None

        i_teff = _col("teff", None)
        i_logg = _col("logg", None)
        i_meta = _col("metallicit", None)
        i_alpha = _col("alpha", None)

        if i_teff is None or i_logg is None:
            return []

        results = []
        for row in self._rows:
            try:
                teff = float(row[i_teff][0])
                logg = float(row[i_logg][0])
                feh = float(row[i_meta][0]) if i_meta is not None else 0.0
                alpha = float(row[i_alpha][0]) if i_alpha is not None else 0.0
            except (IndexError, ValueError):
                continue
            # Download link is in the 3rd-from-last cell
            link = None
            for _, lnk in reversed(row):
                if lnk:
                    link = lnk
                    break
            if link is None:
                continue
            results.append(
                {
                    "teff": teff,
                    "logg": logg,
                    "feh": feh,
                    "alpha": alpha,
                    "url": urllib.parse.urljoin(_SVO_BASE, link),
                }
            )
        return results


def _svo_nearest(model_name, teff, logg, feh, alpha):
    """POST to SVO, parse the HTML table, return the nearest model entry."""
    post_body = urllib.parse.urlencode(
        {
            "models": model_name,
            "oby": "",
            "odesc": "",
            "sbut": "",
            # Wide ranges so we get the full grid back
            "params[bt-settl][teff][min]": "0",
            "params[bt-settl][teff][max]": "999999",
            "params[bt-settl][logg][min]": "-99",
            "params[bt-settl][logg][max]": "99",
            "params[bt-settl][meta][min]": "-99",
            "params[bt-settl][meta][max]": "99",
            "params[bt-settl][alpha][min]": "-99",
            "params[bt-settl][alpha][max]": "99",
            "nres": "all",
            "boton": "Search",
            "reqmodels[]": model_name,
        }
    ).encode("utf-8")

    # Check SVO reachability before making the POST
    if not _host_is_reachable(_SVO_BASE):
        raise ConnectionError(f"Cannot reach SVO service at {_SVO_BASE}; no network")

    req = urllib.request.Request(
        urllib.parse.urljoin(_SVO_BASE, "index.php"),
        data=post_body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    parser = _SVOTableParser()
    parser.feed(html)
    candidates = parser.models()

    if not candidates:
        raise ValueError(
            f"SVO returned no spectra for model '{model_name}'. "
            "Check the model name or its availability on the SVO service."
        )

    # Nearest neighbour in normalised parameter space
    teff_a = np.array([m["teff"] for m in candidates])
    logg_a = np.array([m["logg"] for m in candidates])
    feh_a = np.array([m["feh"] for m in candidates])
    alpha_a = np.array([m["alpha"] for m in candidates])
    dist = (
        ((teff_a - teff) / 100.0) ** 2
        + ((logg_a - logg) / 0.5) ** 2
        + ((feh_a - feh) / 0.5) ** 2
        + ((alpha_a - alpha) / 0.2) ** 2
    )
    return candidates[int(np.argmin(dist))]


def air_to_vacuum_wavelength(lambda_air, max_iter=20, tol=1e-12):
    """
    Convert air wavelengths to vacuum wavelengths using the Ciddor relation.

    Wavelengths must be in Angstrom. The relation is applied only for
    finite positive wavelengths greater than 2000 Angstrom.
    """

    lambda_air = np.asarray(lambda_air, dtype=float)
    lambda_vac = np.full_like(lambda_air, np.nan, dtype=float)

    valid = np.isfinite(lambda_air) & (lambda_air > 2000.0)

    if not np.any(valid):
        return lambda_vac

    lam_air_valid = lambda_air[valid]
    lam_vac_valid = lam_air_valid.copy()

    for _ in range(max_iter):
        sigma2 = (1e4 / lam_vac_valid) ** 2

        f = 1.0 + 0.05792105 / (238.0185 - sigma2) + 0.00167917 / (57.362 - sigma2)

        new_lam_vac_valid = lam_air_valid * f

        rel_diff = np.abs(new_lam_vac_valid - lam_vac_valid) / lam_vac_valid

        lam_vac_valid = new_lam_vac_valid

        if np.all(rel_diff < tol):
            break

    lambda_vac[valid] = lam_vac_valid

    return lambda_vac


def _fetch_svo(model_name, teff, logg, feh, alpha):
    """Download an SVO ASCII spectrum and return Quantities."""
    best = _svo_nearest(model_name, teff, logg, feh, alpha)

    logger.info("Downloading SVO file: %s", best["url"])
    logger.debug("SVO selection: %s", best)

    # Check SVO host before attempting to download
    if not _host_is_reachable(_SVO_BASE):
        raise ConnectionError(f"Cannot reach SVO service at {_SVO_BASE}; no network")

    # Try to determine a meaningful remote filename using HEAD (Content-Disposition
    # or final redirected URL). If the remote name is generic (ssap.php, index.php)
    # build a descriptive fallback name including the model and parameters.
    def _remote_filename(url: str) -> str | None:
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as resp:
                final = resp.geturl()
                cd = resp.headers.get("Content-Disposition") or resp.headers.get(
                    "content-disposition"
                )
            if cd:
                # Try filename* (RFC5987) first, then plain filename
                m = re.search(
                    r"filename\*=(?:UTF-8''?)?(?P<f>[^;\n]+)", cd, flags=re.IGNORECASE
                )
                if not m:
                    m = re.search(
                        r'filename=(?P<f>"[^"]+"|[^;\n]+)', cd, flags=re.IGNORECASE
                    )
                if m:
                    f = m.group("f").strip().strip('"')
                    return os.path.basename(f)
            return os.path.basename(urllib.parse.urlparse(final).path) or None
        except Exception:
            return None

    remote_fname = _remote_filename(best["url"]) or None
    local = download_file(best["url"], pkgname="exosim", cache=True)

    wl_list, flux_list = [], []
    with open(local) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    wl_list.append(float(parts[0]))
                    flux_list.append(float(parts[1]))
                except ValueError:
                    continue

    if not wl_list:
        raise ValueError(
            f"Could not parse any spectral data from SVO file: {best['url']}"
        )

    wl = np.array(wl_list) * u.AA

    # SVO BT-Settl files use erg s-1 cm-2 AA-1
    flux = np.array(flux_list) * (u.erg / u.s / u.cm**2 / u.AA)

    meta = {k: best[k] for k in ("teff", "logg", "feh", "alpha")}
    # include model/source and prefer the remote filename from the URL; if that
    # is generic (ssap.php, index.php, contents) use a constructed descriptive name
    meta.update({"model": model_name, "source": _SVO_BASE})
    try:
        generic_names = {"ssap.php", "index.php", "contents", ""}
        fname = remote_fname or os.path.basename(local)
        if fname and fname.lower() in generic_names:
            # construct descriptive fallback name
            safe_model = re.sub(r"[^A-Za-z0-9._+-]", "-", model_name)
            fname = (
                f"{safe_model}_Teff{int(teff):05d}_logg{logg:.2f}_feh{feh:+.1f}.spec"
            )
        meta.update({"filename": fname})
    except Exception:
        pass
    return wl, flux, meta


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------


class DownloadSed(Task):
    """
    Downloads a stellar SED from an online Phoenix spectral database,
    selects the nearest grid model to the requested stellar parameters,
    and returns the spectrum scaled to the apparent flux at the observer.

        Two backends are available, chosen automatically by *model_name*:

        * ``"phoenix-aces"`` - PHOENIX-ACES-AGSS-COND-2011 from the Goettingen
            server.  The parameter grid is hardcoded; only ``astropy`` and
            ``numpy`` are required.

        * Any SVO model name (``"bt-settl"``, ``"bt-settl-cifist"`,
            ``"bt-nextgen-agss2009"``, ``"nextgen"`, ...) - files are fetched from
            the Spanish Virtual Observatory using stdlib ``urllib``; no extra
            packages are needed.

        Default: ``bt-settl-cifist`` (SVO BT-Settl CIFIST).

    Files are cached in the astropy download cache
    (``~/.astropy/cache/download/``) so repeated calls with the same
    parameters do not re-download.

    Returns
    -------
    :class:`~exosim.models.signal.Sed`
        Star SED (apparent flux at observer, units W m-2 um-1)

    Examples
    --------
    Example (default model ``bt-settl-cifist``):

    >>> from exosim.tasks.sed import DownloadSed
    >>> import astropy.units as u, astropy.constants as cc, numpy as np
    >>> downloadSed = DownloadSed()
    >>> D = 12.975 * u.pc
    >>> T = 3016 * u.K
    >>> M = 0.15 * u.Msun
    >>> R = 0.218 * u.Rsun
    >>> g = (cc.G * M.si / R.si**2).to(u.cm / u.s**2)
    >>> logg = np.log10(g.value)
    >>> sed = downloadSed(T=T, D=D, R=R, logg=logg)

    BT-Settl CIFIST from the SVO (explicit):

    >>> sed = downloadSed(
    ...     T=T, D=D, R=R, logg=logg, model_name="bt-settl-cifist"
    ... )

    Raises
    ------
    KeyError
        if a required stellar parameter is missing.
    ValueError
        if *model_name* is not recognised or the online query returns no data.
    """

    def __init__(self):
        """
        Parameters
        ----------
        R : :class:`~astropy.units.Quantity` or float
            Star radius.  If dimensionless, metres are assumed.
        D : :class:`~astropy.units.Quantity` or float
            Star distance.  If dimensionless, metres are assumed.
        T : :class:`~astropy.units.Quantity` or float
            Effective temperature.  If dimensionless, Kelvin are assumed.
        logg : float
            log10 of the surface gravity in cgs (log10 g [cm/s2]).
        z : float, optional
            Metallicity [Fe/H].  Default is 0.0.
        alpha : float, optional
            Alpha-element enhancement [alpha/Fe] (SVO models only).
            Default is 0.0.
        model_name : str, optional
            Phoenix or SVO model to download. Defaults to ``bt-settl-cifist``
            (SVO BT-Settl CIFIST). Use ``"phoenix-aces"`` for
            PHOENIX-ACES-AGSS-COND-2011 from the Goettingen server, or any
            SVO model name (e.g. ``"bt-settl"``, ``"bt-settl-cifist"``).
        """
        self.add_task_param("R", "star radius")
        self.add_task_param("D", "star distance")
        self.add_task_param("T", "star temperature")
        self.add_task_param("logg", "star logG")
        self.add_task_param("z", "star metallicity", 0.0)
        self.add_task_param("alpha", "alpha element enhancement", 0.0)
        self.add_task_param(
            "model_name",
            "phoenix or SVO model name",
            "bt-settl-cifist",
        )

    def execute(self):
        R = self.get_task_param("R")
        D = self.get_task_param("D")
        T = self.get_task_param("T")
        logg = self.get_task_param("logg")
        z = self.get_task_param("z")
        alpha = self.get_task_param("alpha")
        model_name = self.get_task_param("model_name")

        # --- validate required parameters --------------------------------
        for name, val in (("R", R), ("D", D), ("T", T), ("logg", logg)):
            if val is None:
                self.error(f"star {name} missing")
                raise KeyError(f"star {name} missing")

        if hasattr(T, "unit"):
            T = T.to(u.K)
        else:
            T = T * u.K
            self.debug("no units found for T: Kelvin are assumed.")

        R = checks.check_units(R, u.m, self)
        D = checks.check_units(D, u.m, self)

        teff = float(T.to(u.K).value)
        logg = float(logg)
        z = float(z) if z is not None else 0.0
        alpha = float(alpha) if alpha is not None else 0.0

        # --- dispatch to the appropriate backend -------------------------
        self.info(
            f"downloading spectrum: Teff={teff:.0f} K, "
            f"logg={logg:.2f}, [Fe/H]={z:.2f}, model='{model_name}'"
        )
        if model_name == "phoenix-aces":
            wl_ph, sed_ph, meta = _fetch_aces(teff, logg, z)
        else:
            # Attempt an SVO fetch for any other model name. If the SVO
            # query returns no candidates a ValueError will be raised.
            try:
                wl_ph_air, sed_ph, meta = _fetch_svo(model_name, teff, logg, z, alpha)
                # SVO BT-Settl files use air wavelengths in Angstrom; convert to vacuum
                self.debug("converting SVO wavelengths from air to vacuum")
                wl_ph = air_to_vacuum_wavelength(wl_ph_air.value) * u.AA

            except ValueError as exc:
                msg = (
                    f"Unknown or unavailable SVO model '{model_name}'. "
                    "Check the model name or list available models with "
                    "get_svo_models()."
                )
                # Log the higher-level message but re-raise the original
                # ValueError so callers (and tests) can inspect the original
                # error message from the SVO backend.
                self.error(msg + f" Details: {exc}")
                raise

        meta.update({"model_name": model_name})
        self.debug(f"spectrum downloaded: {len(wl_ph)} points; meta={meta}")

        # --- convert to W m-2 um-1 ---------------------------------------
        wl_ph = wl_ph.to(u.um, equivalencies=u.spectral())
        sed_ph = sed_ph.to(
            u.W / u.m**2 / u.um,
            equivalencies=u.spectral_density(wl_ph),
        )

        # Remove duplicates and sort by wavelength
        idx = np.nonzero(np.diff(wl_ph))[0]
        wl_ph = wl_ph[idx]
        sed_ph = sed_ph[idx]
        sort_idx = np.argsort(wl_ph)
        wl_ph = wl_ph[sort_idx]
        sed_ph = sed_ph[sort_idx]

        sed = signal.Sed(spectral=wl_ph, data=sed_ph)
        for k, v in meta.items():
            sed.metadata[k] = v

        # --- geometric dilution (R/D)^2 ----------------------------------
        sed *= (R / D) ** 2

        self.debug(f"phoenix sed scaled: {sed.data}")
        self.set_output(sed)
