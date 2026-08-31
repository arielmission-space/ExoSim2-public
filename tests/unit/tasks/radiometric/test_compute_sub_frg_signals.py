"""
Behavioural test for :class:`~exosim.tasks.radiometric.compute_sub_frg_signals_channel.ComputeSubFrgSignalsChannel`.

The task walks the ``sub_focal_planes`` group written by the sub-exposure stage
and runs aperture photometry on each foreground contribution. A tiny HDF5 file
with the expected layout is enough to drive both the photometer and the
spectrometer branches.
"""

import h5py
import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

from exosim.output import SetOutput
from exosim.tasks.radiometric.compute_sub_frg_signals_channel import (
    ComputeSubFrgSignalsChannel,
)


def _write_sub_focal_planes(path, ch="Photometer", osf=2):
    with h5py.File(path, "w") as f:
        sfp = f.create_group(f"channels/{ch}/sub_focal_planes")
        for name in ("frg_zodi", "frg_skybackground"):
            g = sfp.create_group(name)
            data = np.ones((1, 4 * osf, 6 * osf))
            g.create_dataset("data", data=data)
            g.create_dataset("data_units", data="ct / s")
            g.create_group("metadata").create_dataset("oversampling", data=osf)


def _aperture_table():
    return QTable(
        {
            "wavelength": [1.0, 2.0] * u.um,
            "spectral_center": [1.5, 4.5],
            "spectral_size": [3.0, 3.0],
            "spatial_center": [2.0, 2.0],
            "spatial_size": [3.0, 3.0],
            "aperture_shape": ["rectangular", "rectangular"],
        }
    )


@pytest.mark.parametrize("kind", ["photometer", "spectrometer"])
def test_model_builds_a_signal_table_per_foreground(tmp_path, kind):
    fname = tmp_path / "sfp.h5"
    _write_sub_focal_planes(str(fname))

    task = ComputeSubFrgSignalsChannel()
    out = task(
        table=_aperture_table(),
        ch_name="Photometer",
        input_file=SetOutput(str(fname), replace=False),
        parameters={"type": kind},
    )

    assert isinstance(out, QTable)
    # one "<name>_signal_in_aperture" column per foreground, plus a matching
    # "<name>_total_signal" column for both channel types
    assert "zodi_signal_in_aperture" in out.colnames
    assert "skybackground_signal_in_aperture" in out.colnames
    assert "zodi_total_signal" in out.colnames
    assert np.all(out["zodi_signal_in_aperture"].value > 0)


def test_no_sub_focal_planes_yields_an_empty_table(tmp_path):
    fname = tmp_path / "empty.h5"
    with h5py.File(str(fname), "w") as f:
        f.create_group("channels/Photometer/focal_plane")

    out = ComputeSubFrgSignalsChannel()(
        table=_aperture_table(),
        ch_name="Photometer",
        input_file=SetOutput(str(fname), replace=False),
        parameters={"type": "photometer"},
    )
    assert isinstance(out, QTable)
    assert len(out.colnames) == 0
