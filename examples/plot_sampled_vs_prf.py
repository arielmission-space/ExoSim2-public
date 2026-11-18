"""Figures of the documentation page "Sampled and PRF populations".

A generic case, independent of any payload: a monochromatic Airy PSF at 1 um on a
detector of 24 x 24 pixels of 18 um, a photometer-like channel with the source on the
pixel centre ``N // 2``. The focal planes are made with the ExoSim tasks
(:class:`PopulateFocalPlane` with the IRF, and :class:`PopulateFocalPlanePrf`), and the
truth is the direct integral of the Airy over the pixel, computed here on cells of
1/21 pixel without either population.

    python plot_sampled_vs_prf.py   # writes docs/source/user/focal_plane/_static/sampled_vs_prf_*.png
"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1

from exosim.models.signal import Sed, Signal
from exosim.tasks.instrument import (
    ApplyIntraPixelResponseFunction,
    CreateIntrapixelResponseFunction,
    PopulateFocalPlane,
    PopulateFocalPlanePrf,
)
from exosim.utils.prf import clear_prf_cache

DELTA = 18.0  # pixel size, um
NPIX = 24
WL = 1.0  # um
BAND = 0.002  # narrow flat band around WL: monochromatic to 1e-3
OUT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "source"
    / "user"
    / "focal_plane"
    / "_static"
)

C_SAMPLED, C_PRF, C_TRUE = "#2a78d6", "#1a9e77", "#222222"


# ------------------------------------------------------------------ ExoSim planes
def parameters(flam, osf):
    """Channel parameters for an Airy PSF with F lambda / delta = flam."""
    fnum = flam * DELTA / WL
    return {
        "type": "photometer",
        "value": "example",
        "Fnum_x": fnum,
        "Fnum_y": fnum,
        "psf": {"shape": "airy"},
        "detector": {
            "delta_pix": DELTA * u.um,
            "diffusion_length": 0 * u.um,
            "intra_pix_distance": 0 * u.um,
            "oversampling": osf,
            "spatial_pix": NPIX,
            "spectral_pix": NPIX,
            "irf_task": "CreateIntrapixelResponseFunction",
        },
    }


def empty_plane(osf):
    ax = np.linspace(WL - BAND / 2, WL + BAND / 2, NPIX * osf)
    return Signal(
        spectral=ax * u.um,
        spatial=np.zeros(ax.size) * u.um,
        data=np.zeros((1, NPIX * osf, NPIX * osf)),
        metadata={"focal_plane_delta": DELTA * u.um / osf, "oversampling": osf},
    )


def source():
    """Flat 1 ct/s/um."""
    w = np.linspace(WL - 0.01, WL + 0.01, 2001)
    s = Sed(
        spectral=w * u.um,
        data=np.ones(w.size)[None, None, :] * u.W / u.m**2 / u.um,
        metadata={"parsed_parameters": {}},
    )
    s.data_units = u.ct / u.s / u.um
    return s


def prf_plane(flam, osf):
    """PRF focal plane, per unit flux."""
    clear_prf_cache()
    out, _ = PopulateFocalPlanePrf()(
        parameters=parameters(flam, osf),
        focal_plane=empty_plane(osf),
        sources={"s": source()},
    )
    return np.asarray(out.data[0], float) / BAND


def sampled_plane(flam, osf):
    """Sampled focal plane (point-sampled PSF, then the IRF when osf > 1), per unit flux."""
    p = parameters(flam, osf)
    fp = empty_plane(osf)
    s = source()
    x = fp.spectral
    s.spectral_rebin(x)
    s *= np.abs(np.gradient(x)) * u.um
    out, _ = PopulateFocalPlane()(parameters=p, focal_plane=fp, sources={"s": s})
    if osf > 1:
        k, d = CreateIntrapixelResponseFunction()(parameters=p)
        out = ApplyIntraPixelResponseFunction()(
            focal_plane=out, irf_kernel=k, irf_kernel_delta=d
        )
    n = x.size
    return np.asarray(out.data[0], float) / (BAND * n / (n - 1))


# ------------------------------------------------------------------ the truth
def airy(r, flam):
    """Airy intensity [1/pixel^2] at r pixels, unit integral."""
    a = np.pi * r / flam + 1e-20
    return (2 * j1(a) / a) ** 2 / (4 * flam**2 / np.pi)


def exact_pixel(flam, dy, dx, ps=41):
    """Light collected by the pixel centred on the origin from a source at (dy, dx) pixels."""
    c = (np.arange(ps) + 0.5) / ps - 0.5
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return float(airy(np.hypot(yy - dy, xx - dx), flam).sum() / ps**2)


# ------------------------------------------------------------------ figures
def fig_concept():
    """1D sketch: the integral over the pixel and its Riemann sum on osf samples."""
    flam = 0.8
    x = np.linspace(-1.6, 1.6, 801)
    prof = airy(np.abs(x), flam)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, osf in zip(axs, (3, 7), strict=False):
        ax.plot(x, prof, color=C_TRUE, lw=1.4)
        inside = np.abs(x) <= 0.5
        ax.fill_between(
            x[inside],
            prof[inside],
            color=C_PRF,
            alpha=0.25,
            lw=0,
            label="PRF: exact integral over the pixel",
        )
        xs = (np.arange(osf) - osf // 2) / osf
        ax.bar(
            xs,
            airy(np.abs(xs), flam),
            width=1 / osf,
            fill=False,
            edgecolor=C_SAMPLED,
            lw=1.3,
            label=f"sampled: {osf} point samples x 1/{osf} pixel",
        )
        ax.plot(xs, airy(np.abs(xs), flam), "o", color=C_SAMPLED, ms=4)
        ax.axvspan(-0.5, 0.5, color="0.5", alpha=0.06)
        exact = np.trapezoid(prof[inside], x[inside])
        riemann = airy(np.abs(xs), flam).sum() / osf
        ax.set_title(f"oversampling {osf}: sum / integral = {riemann / exact:.3f}")
        ax.set_xlabel("position across the pixel [pixels]")
        ax.legend(fontsize=8, loc="upper left", frameon=False)
    axs[0].set_ylim(0, 1.75)
    axs[0].set_ylabel("PSF cut (arbitrary units)")
    fig.tight_layout()
    fig.savefig(OUT / "sampled_vs_prf_concept.png", dpi=150)
    plt.close(fig)


def fig_oversampling():
    """Brightest pixel against the oversampling factor, for three PSF widths."""
    osfs = (1, 3, 5, 7, 9)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for flam, ls in ((0.7, "-"), (1.4, "--"), (2.8, ":")):
        truth = exact_pixel(flam, 0, 0)
        for f, col, _lab in (
            (sampled_plane, C_SAMPLED, "sampled"),
            (prf_plane, C_PRF, "PRF"),
        ):
            y = []
            for osf in osfs:
                plane = f(flam, osf)
                c = osf * (NPIX // 2) + osf // 2
                y.append(plane[c, c] / truth)
            ax.plot(osfs, y, ls, marker="o", color=col, ms=4)
    ax.axhline(1, color=C_TRUE, lw=0.8)
    ax.set_xlabel("oversampling factor")
    ax.set_ylabel("brightest pixel / exact integral")
    ax.set_xticks(osfs)
    from matplotlib.lines import Line2D

    l1 = ax.legend(
        [
            Line2D([], [], color=C_SAMPLED, marker="o", ms=4),
            Line2D([], [], color=C_PRF, marker="o", ms=4),
        ],
        ["sampled", "PRF"],
        fontsize=8,
        frameon=False,
        loc="upper right",
    )
    ax.add_artist(l1)
    ax.legend(
        [Line2D([], [], color="0.3", ls=ls) for ls in ("-", "--", ":")],
        [f"F$\\lambda/\\delta$ = {f}" for f in (0.7, 1.4, 2.8)],
        fontsize=8,
        frameon=False,
        loc="upper center",
    )
    fig.tight_layout()
    fig.savefig(OUT / "sampled_vs_prf_oversampling.png", dpi=150)
    plt.close(fig)


def fig_phases(flam=0.7, osf=3):
    """The pixel read at every sub-pixel phase, against the source displacement."""
    d = np.linspace(-0.5, 0.5, 201)
    truth = np.array([exact_pixel(flam, 0, -t) for t in d])
    peak = truth.max()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(d, truth / peak, color=C_TRUE, lw=1.3, label="exact integral")
    c = osf * (NPIX // 2) + osf // 2
    k = np.arange(osf) - osf // 2
    for f, col, mk, lab in (
        (sampled_plane, C_SAMPLED, "s", "sampled"),
        (prf_plane, C_PRF, "D", "PRF"),
    ):
        plane = f(flam, osf)
        # reading sub-pixel c + k is the pixel centred k / osf away: the source moved by -k / osf
        ax.plot(
            -k / osf,
            plane[c, c + k] / peak,
            mk,
            color=col,
            ms=7,
            mec="white",
            mew=1,
            ls="none",
            label=f"{lab}, oversampling {osf}",
        )
    ax.set_xlabel("source displacement along the row [pixels]")
    ax.set_ylabel("pixel signal / exact peak")
    ax.set_title(f"Airy, F$\\lambda/\\delta$ = {flam}")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "sampled_vs_prf_phases.png", dpi=150)
    plt.close(fig)


def numbers():
    """The numbers quoted in the text."""
    for flam in (0.7, 1.4, 2.8):
        t = exact_pixel(flam, 0, 0)
        row = [f"F lambda/delta {flam}: exact {t:.4f}"]
        for osf in (1, 3, 5, 7):
            c = osf * (NPIX // 2) + osf // 2
            row.append(
                f"osf {osf}: sampled {sampled_plane(flam, osf)[c, c] / t:.4f} "
                f"prf {prf_plane(flam, osf)[c, c] / t:.4f}"
            )
        print(" | ".join(row))
    osf, flam = 3, 0.7
    c = osf * (NPIX // 2) + osf // 2
    for f, lab in ((sampled_plane, "sampled"), (prf_plane, "prf")):
        plane = f(flam, osf)
        errs = [
            abs(plane[c, c + k] - exact_pixel(flam, 0, -k / osf))
            / exact_pixel(flam, 0, 0)
            for k in (-1, 0, 1)
        ]
        print(f"phases osf 3 F 0.7 {lab}: max error / peak {max(errs):.4f}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_concept()
    fig_oversampling()
    fig_phases()
    numbers()
    print("written", *sorted(p.name for p in OUT.glob("sampled_vs_prf_*.png")))
