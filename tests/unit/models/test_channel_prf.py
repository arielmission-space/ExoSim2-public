"""
Channel and CreateFocalPlane with ``<psf><population>prf</population></psf>``.

The default population (sampled PSFs + IRF) must stay bit-identical to the one
of the base branch: ``tests/data/prf/sampled_reference.npz`` holds the focal
planes produced by the base branch (commit fda2d5cb: the photometer plane
was regenerated after the half-step weights of its end samples) for the same small run.
"""

import os
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pytest

from exosim import recipes
from exosim.models.channel import Channel
from exosim.tasks.load import LoadOptions
from exosim.utils import RunConfig

REFERENCE = Path(__file__).parents[2] / "data" / "prf" / "sampled_reference.npz"
EXAMPLES = Path(__file__).parents[3] / "examples"


def _config(osf=3, npix=16, population=None, drop_irf_task=False):
    config = LoadOptions()(
        filename=str(EXAMPLES / "main_example.xml"), config_path=str(EXAMPLES)
    )
    config["time_grid"]["end_time"] = 1.0 * u.hr
    config["time_grid"]["low_frequencies_resolution"] = 1.0 * u.hr
    for channel in config["payload"]["channel"].values():
        detector = channel["detector"]
        detector["spatial_pix"] = npix
        detector["spectral_pix"] = npix
        detector["oversampling"] = osf
        if drop_irf_task:
            detector.pop("irf_task")
        if population:
            channel["psf"]["population"] = population
    return config


def _run(tmp_path, name, **kwargs):
    RunConfig.n_job = 1
    RunConfig.random_seed = 42
    out = tmp_path / f"{name}.h5"
    recipes.CreateFocalPlane(_config(**kwargs), str(out))
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("prf_runs")
    owd = os.getcwd()
    os.chdir(EXAMPLES.parent)
    try:
        yield {
            # pinned: the default population is prf, the reference is sampled
            "sampled": _run(tmp_path, "sampled", population="sampled"),
            "prf3": _run(tmp_path, "prf3", population="prf"),
            "prf1": _run(tmp_path, "prf1", population="prf", osf=1),
            "prf3_no_irf_task": _run(
                tmp_path, "prf3b", population="prf", drop_irf_task=True
            ),
        }
    finally:
        os.chdir(owd)


def _planes(path, key="focal_plane"):
    with h5py.File(path) as f:
        return {ch: f["channels"][ch][key]["data"][0] for ch in f["channels"]}


def test_sampled_population_is_unchanged(runs):
    """The sampled population gives the planes of the base branch."""
    reference = np.load(REFERENCE)
    for channel, plane in _planes(runs["sampled"]).items():
        np.testing.assert_allclose(
            plane, reference[f"{channel}_focal_plane"][0], rtol=1e-10, atol=0
        )
    for channel, plane in _planes(runs["sampled"], "frg_focal_plane").items():
        np.testing.assert_allclose(
            plane, reference[f"{channel}_frg_focal_plane"][0], rtol=1e-10, atol=0
        )
    with h5py.File(runs["sampled"]) as f:
        for ch in f["channels"]:
            assert "intrapixel_response_function" in f["channels"][ch]
            assert "prf" not in f["channels"][ch]
            assert (
                "pixel_response_applied"
                not in f["channels"][ch]["focal_plane"]["metadata"]
            )


def test_prf_planes_are_flagged_and_have_no_irf(runs):
    with h5py.File(runs["prf3"]) as f:
        for ch in f["channels"]:
            group = f["channels"][ch]
            meta = group["focal_plane"]["metadata"]
            assert bool(meta["pixel_response_applied"][()])
            assert meta["oversampling"][()] == 3
            assert meta["prf_sampling"][()] == 10
            assert "intrapixel_response_function" not in group
            assert "irf" not in group
            assert "prf" in group
            assert "wavelength" in group["prf"]
            assert "sources" in group
            # the foreground plane is not a source plane
            assert "pixel_response_applied" not in group["frg_focal_plane"]["metadata"]


def test_prf_output_stores_the_same_psf_cube_as_sampled_and_ils_reads_it(runs):
    from exosim.tasks.subexposures import LoadILS

    with h5py.File(runs["sampled"]) as fs, h5py.File(runs["prf3"]) as fp:
        for ch in fs["channels"]:
            gs, gp = fs["channels"][ch]["psf"], fp["channels"][ch]["psf"]
            for key in ("psf_cube", "norm"):
                np.testing.assert_array_equal(gp[key][()], gs[key][()])
            np.testing.assert_array_equal(
                gp["wavelength"]["value"][()], gs["wavelength"]["value"][()]
            )
            assert dict(gp["psf_cube"].attrs) == dict(gs["psf_cube"].attrs)
    # LoadILS reads the PSF of a PRF file, as it does for a sampled one
    config = _config(population="prf")
    for ch in config["payload"]["channel"]:
        pars = config["payload"]["channel"][ch]
        with h5py.File(runs["prf3"]) as f:
            wl = f["channels"][ch]["psf"]["wavelength"]
            wl_grid = wl["value"][()] * u.Unit(wl["unit"][()])
        ils = LoadILS()(input_file=str(runs["prf3"]), ch_param=pars, wl_grid=wl_grid)
        assert ils.shape[1] == wl_grid.size


def test_prf_plane_agrees_with_sampled_and_is_osf_independent(runs):
    sampled = _planes(runs["sampled"])
    prf3 = _planes(runs["prf3"])
    prf1 = _planes(runs["prf1"])
    for channel in sampled:
        decimated = prf3[channel][1::3, 1::3]
        sampled_dec = sampled[channel][1::3, 1::3]
        peak = prf1[channel].max()
        # smooth spectra: the PRF is the same at osf 1 and 3 (the wavelength axis
        # of a spectrometer is shifted by 1/3 pixel with the oversampling)
        assert np.abs(decimated - prf1[channel]).max() < 2e-3 * peak
        # the sampled population at osf = 3 is within a few percent
        assert np.abs(sampled_dec - prf1[channel]).max() < 0.05 * peak
        assert prf3[channel].sum() / 9 == pytest.approx(prf1[channel].sum(), rel=2e-3)


def test_oversampling_is_kept_without_irf_task_for_prf(runs):
    """The PRF does not need an ``irf_task`` to use an oversampled plane."""
    planes = _planes(runs["prf3_no_irf_task"])
    for channel, plane in planes.items():
        assert plane.shape == (48, 48)
        np.testing.assert_allclose(plane, _planes(runs["prf3"])[channel], rtol=1e-12)


def test_apply_irf_is_skipped_on_prf_planes():
    channel = Channel(
        parameters={"value": "ch", "psf": {"population": "prf"}, "detector": {}},
        wavelength=np.linspace(1, 2, 3) * u.um,
        time=[0] * u.hr,
    )
    assert channel.psf_population == "prf"
    from exosim.models.signal import Signal

    plane = Signal(
        spectral=np.linspace(1, 2, 4) * u.um,
        data=np.ones((1, 4, 4)),
        metadata={"oversampling": 2, "pixel_response_applied": True},
    )
    # apply_irf_to returns the plane untouched, without building any kernel
    assert channel.apply_irf_to(plane) is plane
    assert np.all(plane.data == 1.0)
    channel.focal_plane = plane
    assert channel.apply_irf() == (plane, None)


def test_population_is_validated():
    channel = Channel(
        parameters={"value": "ch", "psf": {"population": "bogus"}},
        wavelength=np.linspace(1, 2, 3) * u.um,
        time=[0] * u.hr,
    )
    with pytest.raises(ValueError, match="population"):
        channel.psf_population  # noqa: B018


def test_custom_irf_task_with_prf_fails_in_the_recipe(tmp_path):
    from exosim.tasks.instrument import CreateOversampledIntrapixelResponseFunction

    config = _config(population="prf")
    for channel in config["payload"]["channel"].values():
        channel["detector"]["irf_task"] = CreateOversampledIntrapixelResponseFunction
    RunConfig.n_job = 1
    owd = os.getcwd()
    os.chdir(EXAMPLES.parent)
    try:
        with pytest.raises(ValueError, match="custom irf_task"):
            recipes.CreateFocalPlane(config, str(tmp_path / "x.h5"))
    finally:
        os.chdir(owd)
