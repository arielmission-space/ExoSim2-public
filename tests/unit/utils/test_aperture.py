"""
Behavioural tests for the aperture-finding utilities.

Each function optimises an aperture on a PSF so that it collects a requested
fraction of the encircled energy. The tests check that the returned aperture
actually collects that fraction, that the reported area is consistent, and that
asking for more energy yields a larger aperture.
"""

import numpy as np
import pytest

from exosim.utils.aperture import (
    find_bin_aperture,
    find_elliptical_aperture,
    find_rectangular_aperture,
)


def _gaussian_psf(shape=(41, 41), sigma=4.0):
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    psf = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return psf / psf.sum()


class TestFindRectangularAperture:
    def test_collects_requested_energy(self):
        psf = _gaussian_psf()
        (w, h), area, ene = find_rectangular_aperture(psf, 0.80)
        assert ene == pytest.approx(0.80, abs=0.02)
        assert area == pytest.approx(w * h)
        assert w > 0
        assert h > 0

    def test_more_energy_needs_larger_aperture(self):
        psf = _gaussian_psf()
        _, area_lo, _ = find_rectangular_aperture(psf, 0.6)
        _, area_hi, _ = find_rectangular_aperture(psf, 0.9)
        assert area_hi > area_lo


class TestFindEllipticalAperture:
    def test_collects_requested_energy(self):
        psf = _gaussian_psf()
        (a, b), area, ene = find_elliptical_aperture(psf, 0.75)
        assert ene == pytest.approx(0.75, abs=0.03)
        assert area == pytest.approx(a * b)


class TestFindBinAperture:
    def test_collects_requested_fraction_of_the_column(self):
        psf = _gaussian_psf(shape=(41, 41), sigma=4.0)
        spatial_with = psf.shape[1]  # full spectral row
        h, area, ene = find_bin_aperture(psf, 0.85, spatial_with)
        assert ene == pytest.approx(0.85, abs=0.03)
        assert area == pytest.approx(h * spatial_with)

    def test_more_energy_needs_taller_bin(self):
        psf = _gaussian_psf()
        h_lo, _, _ = find_bin_aperture(psf, 0.6, psf.shape[1])
        h_hi, _, _ = find_bin_aperture(psf, 0.95, psf.shape[1])
        assert h_hi > h_lo

    def test_accepts_explicit_center(self):
        psf = _gaussian_psf()
        c = (psf.shape[1] / 2, psf.shape[0] / 2)
        h, _, ene = find_bin_aperture(psf, 0.8, psf.shape[1], center=c)
        assert ene == pytest.approx(0.8, abs=0.03)
        assert h > 0
