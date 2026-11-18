"""A PRF-populated plane goes through the jitter of the sub-exposures."""

import astropy.units as u
import numpy as np
import pytest

from exosim.models.signal import CountsPerSecond
from exosim.output import SetOutput
from exosim.tasks.subexposures import InstantaneousReadOut, PrepareInstantaneousReadOut

from .test_populate_focal_plane_prf import (
    flat_source,
    make_focal_plane,
    make_parameters,
    prf_plane,
)


@pytest.mark.parametrize("osf", [3, 4])
def test_instantaneous_readout_preserves_the_flux_of_a_prf_plane(tmp_path, osf):
    npix = 16
    params = make_parameters(2.5 * 18.0, 1.7, osf, npix)
    params["detector"].update(
        {
            "well_depth": 1000 * u.ct,
            "f_well_depth": 1,
            "plate_scale": {
                "spatial": 0.05 * u.arcsec / u.pixel,
                "spectral": 0.05 * u.arcsec / u.pixel,
            },
        }
    )
    params["readout"] = {
        "readout_frequency": 100 * u.Hz,
        "n_NRDs_per_group": 1,
        "n_groups": 2,
        "n_sim_clocks_Ground": 5,
        "n_sim_clocks_first_NDR": 5,
        "n_sim_clocks_NDR": 1,
        "n_sim_clocks_Reset": 3,
        "n_sim_clocks_groups": 100,
        "n_exposures": 1,
    }
    plane, _ = prf_plane(
        params,
        make_focal_plane(params, np.linspace(0.99, 1.01, npix * osf)),
        flat_source(),
    )
    plane = plane * 1.0e4  # ct/s
    focal_plane = CountsPerSecond(
        data=plane[np.newaxis],
        spectral=np.linspace(0.99, 1.01, npix * osf),
        metadata={"oversampling": osf, "pixel_response_applied": True},
    )

    np.random.seed(7)
    jitter_time = np.arange(0, 5, 0.1) * u.s
    jitter = (
        np.random.normal(0, 0.05, jitter_time.size) * u.arcsec,
        np.random.normal(0, 0.05, jitter_time.size) * u.arcsec,
        jitter_time,
    )
    main = {
        "time_grid": {
            "start_time": 0.0 * u.hr,
            "end_time": 2.0 * u.hr,
            "low_frequencies_resolution": 2.0 * u.hr,
        }
    }
    output = SetOutput(str(tmp_path / "jitter.h5"))
    with output.use(append=True, cache=True) as out:
        readout_parameters, integration_time = PrepareInstantaneousReadOut()(
            main_parameters=main,
            parameters=params,
            focal_plane=focal_plane,
            pointing_jitter=jitter,
            output_file=out,
        )
        dset = InstantaneousReadOut()(
            readout_parameters=readout_parameters,
            parameters=params,
            focal_plane=focal_plane,
            pointing_jitter=jitter,
            output_file=out,
        )
        frames = dset.dataset[:]

    assert frames.shape == (integration_time.size, npix, npix)
    # every sub-exposure collects the flux of the plane: a pixel centred on each
    # sub-pixel is a full pixel, whatever the jitter shift
    expected = plane[osf // 2 :: osf, osf // 2 :: osf].sum()
    per_second = frames.sum(axis=(1, 2)) / integration_time.to(u.s).value
    assert np.all(np.isfinite(frames))
    assert per_second.min() > 0.98 * expected
    assert per_second.max() < 1.02 * expected
