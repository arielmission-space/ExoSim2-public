"""Read-out decimation convention: physical pixel m is sub-pixel osf*m + osf//2."""

import astropy.units as u
import numpy as np

from exosim.models.signal import CountsPerSecond
from exosim.tasks.subexposures.instantaneous_read_out import InstantaneousReadOut


def test_replicating_uses_physical_pixel_values():
    osf = 3
    phys = np.arange(12.0).reshape(3, 4) + 1
    fine = np.zeros((1, 3 * osf, 4 * osf))
    fine[0, osf // 2 :: osf, osf // 2 :: osf] = phys
    # junk on the other sub-pixels must not be read
    mask = np.ones(fine.shape[1:], bool)
    mask[osf // 2 :: osf, osf // 2 :: osf] = False
    fine[0][mask] = -99.0
    dec = fine[:, osf // 2 :: osf, osf // 2 :: osf]
    out = InstantaneousReadOut.replicating_the_focalplane(
        dec, np.arange(2), np.zeros(2, dtype=int)
    )
    np.testing.assert_array_equal(out[0], phys)
    np.testing.assert_array_equal(out[1], phys)


def test_no_jitter_readout_end_to_end(tmp_path):
    from exosim.output import SetOutput

    osf = 3
    phys = np.arange(12.0).reshape(3, 4) + 1
    fine = np.full((1, 3 * osf, 4 * osf), -99.0)
    fine[0, osf // 2 :: osf, osf // 2 :: osf] = phys
    fp = CountsPerSecond(
        spectral=np.arange(4 * osf) * u.um,
        spatial=np.arange(3 * osf) * u.um,
        data=fine,
        time=np.zeros(1) * u.hr,
        metadata={"oversampling": osf},
    )
    ro = {
        "ndr_integration_times": np.ones(2) * u.s,
        "simulation_clock": 1 * u.s,
        "fp_time": np.zeros(2, dtype=int),
        "ndr_start_cumulative_sequence": np.arange(2),
        "ndr_end_cumulative_sequence": np.arange(2) + 1,
    }
    task = InstantaneousReadOut()
    task.set_log_name()
    with SetOutput(str(tmp_path / "o.h5")).use(append=True, cache=True) as out_file:
        res = task(
            parameters={"value": "ch"},
            readout_parameters=ro,
            focal_plane=fp,
            pointing_jitter=(None, None, None),
            output_file=out_file,
        )
        np.testing.assert_array_equal(res.dataset[0], phys)
        np.testing.assert_array_equal(res.dataset[1], phys)
