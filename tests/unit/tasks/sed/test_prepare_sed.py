"""
Behavioural tests for :class:`~exosim.tasks.sed.prepare_sed.PrepareSed`, the
dispatcher that turns a ``source_type`` string into a concrete SED by delegating
to the Planck / Phoenix / custom / download backends.
"""

from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u

from exosim.models.signal import Sed
from exosim.tasks.sed.prepare_sed import PrepareSed

_ECSV = """# %ECSV 0.9
# ---
# datatype:
# - {name: Wavelength, unit: um, datatype: float64}
# - {name: Sed, unit: W / (m2 sr um), datatype: float64}
# schema: astropy-2.0
Wavelength Sed
0.5 1.0
1.0 2.0
2.0 3.0
"""


class TestPlanckDispatch:
    def test_planck_source_type_builds_a_blackbody_sed(self):
        wl = np.linspace(0.5, 7.0, 200) * u.um
        sed = PrepareSed()(
            source_type="planck",
            wavelength=wl,
            T=5778 * u.K,
            R=1 * u.R_sun,
            D=1 * u.au,
        )
        assert isinstance(sed, Sed)
        assert sed.data.shape[-1] == wl.size
        assert np.all(sed.data >= 0)


class TestCustomDispatch:
    def test_custom_source_type_reads_the_file_and_scales_by_solid_angle(
        self, tmp_path
    ):
        f = tmp_path / "sed.ecsv"
        f.write_text(_ECSV)
        R, D = 1 * u.R_sun, 10 * u.pc
        sed = PrepareSed()(source_type="custom", filename=str(f), R=R, D=D)
        assert isinstance(sed, Sed)
        expected = (
            np.array([1.0, 2.0, 3.0]) * (np.pi * (R.to(u.m) / D.to(u.m)) ** 2).value
        )
        np.testing.assert_allclose(sed.data.flatten(), expected, rtol=1e-6)

    def test_case_insensitive_source_type(self, tmp_path):
        f = tmp_path / "sed.ecsv"
        f.write_text(_ECSV)
        sed = PrepareSed()(
            source_type="CUSTOM", filename=str(f), R=1 * u.R_sun, D=10 * u.pc
        )
        assert isinstance(sed, Sed)


class TestDownloadDispatch:
    def test_download_sed_source_type_delegates_to_downloadsed(self):
        sentinel = object()
        with patch("exosim.tasks.sed.prepare_sed.DownloadSed") as mock_cls:
            mock_cls.return_value.return_value = sentinel
            out = PrepareSed()(
                source_type="download_sed",
                R=1 * u.R_sun,
                D=10 * u.pc,
                T=5000 * u.K,
                logg=4.5,
                z=0.0,
                model_name="bt-settl",
            )
        assert out is sentinel
        _, kwargs = mock_cls.return_value.call_args
        assert kwargs["model_name"] == "bt-settl"
        assert kwargs["T"] == 5000 * u.K


class TestPhoenixDispatch:
    def test_phoenix_source_type_delegates_to_loadphoenix(self):
        sentinel = object()
        with patch("exosim.tasks.sed.prepare_sed.LoadPhoenix") as mock_cls:
            mock_cls.return_value.return_value = sentinel
            out = PrepareSed()(
                source_type="phoenix",
                path="/some/grid",
                R=1 * u.R_sun,
                D=10 * u.pc,
                T=5000 * u.K,
                logg=4.5,
                z=0.0,
            )
        assert out is sentinel
        _, kwargs = mock_cls.return_value.call_args
        assert kwargs["path"] == "/some/grid"


class TestUnknownDispatch:
    def test_unknown_source_type_raises_keyerror(self):
        with pytest.raises(KeyError, match="not supported source type"):
            PrepareSed()(source_type="banana", R=1 * u.R_sun, D=10 * u.pc)
