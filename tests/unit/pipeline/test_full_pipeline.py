"""
End-to-end pipeline test on the bundled single-channel example configuration.

One session-scoped fixture runs the full chain once ::

    CreateFocalPlane -> RadiometricModel -> CreateSubExposures -> CreateNDRs

and the individual tests assert on each product and exercise the four plotters
against the real outputs. This drives the recipe orchestrators, most of the
instrument / sub-exposure / detector tasks, the HDF5 output layer and the
plotting code that unit-level tests cannot reach in isolation.
"""

import os
from collections import OrderedDict
from pathlib import Path

import astropy.units as u
import h5py
import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

from exosim import recipes
from exosim.plots import (
    FocalPlanePlotter,
    NDRsPlotter,
    RadiometricPlotter,
    SubExposuresPlotter,
)
from exosim.utils import RunConfig


@pytest.fixture(scope="session")
def pipeline_products():
    """Run the whole pipeline once for a single (photometer) channel.

    Uses a private temp directory (not tmp_path_factory) so the shared basetemp
    retention policy cannot delete the products mid-session.
    """
    import shutil
    import tempfile

    from exosim.tasks.load import LoadOptions

    RunConfig.n_job = 1
    RunConfig.random_seed = 42

    root = os.path.abspath("examples")
    config = LoadOptions()(
        filename=os.path.join(root, "main_example.xml"), config_path=root
    )
    channels = config["payload"]["channel"]
    if isinstance(channels, OrderedDict):
        name = "Photometer" if "Photometer" in channels else next(iter(channels))
        config["payload"]["channel"] = OrderedDict({name: channels[name]})

    out = Path(tempfile.mkdtemp(prefix="exosim_pipeline_"))
    fp = out / "fp.h5"
    se = out / "se.h5"
    ndr = out / "ndr.h5"

    recipes.CreateFocalPlane(config, str(fp))
    recipes.RadiometricModel(config, str(fp))
    recipes.CreateSubExposures(
        input_file=str(fp), output_file=str(se), options_file=config
    )
    recipes.CreateNDRs(input_file=str(se), output_file=str(ndr), options_file=config)
    yield {"dir": out, "fp": fp, "se": se, "ndr": ndr, "config": config}
    shutil.rmtree(out, ignore_errors=True)


@pytest.fixture(scope="session")
def spectrometer_focal_plane():
    """Build a spectrometer focal plane and its radiometric table once.

    Only ``CreateFocalPlane`` + ``RadiometricModel`` are run (the jittered
    sub-exposure stage is too slow for a spectrometer at unit-test scale). The
    spectral resolution is coarsened so the run takes a few seconds; this drives
    the spectrometer branches of the focal-plane array builder, the PSF loader,
    the foreground painter and the radiometric spectrometer code that the
    photometer pipeline never touches.
    """
    import shutil
    import tempfile

    from exosim.tasks.load import LoadOptions

    RunConfig.n_job = 1
    RunConfig.random_seed = 42

    root = os.path.abspath("examples")
    config = LoadOptions()(
        filename=os.path.join(root, "main_example.xml"), config_path=root
    )
    config["wl_grid"]["logbin_resolution"] = 100.0
    ch = config["payload"]["channel"]["Spectrometer"]
    ch["detector"]["oversampling"] = 2
    config["payload"]["channel"] = OrderedDict({"Spectrometer": ch})

    out = Path(tempfile.mkdtemp(prefix="exosim_spectrometer_"))
    fp = out / "fp.h5"
    recipes.CreateFocalPlane(config, str(fp))
    recipes.RadiometricModel(config, str(fp))
    yield {"dir": out, "fp": fp, "config": config}
    shutil.rmtree(out, ignore_errors=True)


class TestPipelineProducts:
    def test_focal_plane_file(self, pipeline_products):
        with h5py.File(pipeline_products["fp"], "r") as f:
            assert "channels" in f
            ch = next(iter(f["channels"]))
            fp = f["channels"][ch]["focal_plane"]["data"][()]
            assert fp.ndim == 3
            assert np.all(np.isfinite(fp))
            assert fp.max() > 0  # the source put light on the array

    def test_radiometric_table_added(self, pipeline_products):
        with h5py.File(pipeline_products["fp"], "r") as f:
            assert "radiometric" in f
            cols = set(f["radiometric"]["table_to_group"].keys())
            # the radiometric table carries the per-bin signal and noise
            assert "source_signal_in_aperture" in cols
            assert "total_noise" in cols
            assert "saturation_time" in cols

    def test_sub_exposures_file(self, pipeline_products):
        with h5py.File(pipeline_products["se"], "r") as f:
            grp = f.get("channels", f)
            assert len(list(grp.keys())) > 0

    def test_ndrs_file(self, pipeline_products):
        with h5py.File(pipeline_products["ndr"], "r") as f:
            assert len(list(f.keys())) > 0


class TestPlottersOnRealData:
    def test_focal_plane_plotter(self, pipeline_products):
        plotter = FocalPlanePlotter(input=str(pipeline_products["fp"]))
        fig = plotter.plot_focal_plane(time_step=0, scale="linear")
        assert fig is not None
        plotter.plot_focal_plane(time_step=0, scale="dB")
        plotter.save_fig(str(pipeline_products["dir"] / "fp.png"))
        assert (pipeline_products["dir"] / "fp.png").exists()

    @pytest.mark.parametrize(
        "efficiency",
        [
            "optical efficiency",
            "responsivity",
            "quantum efficiency",
            "photon conversion efficiency",
            "all",
        ],
    )
    def test_focal_plane_efficiency_plot_every_kind(
        self, pipeline_products, efficiency
    ):
        import matplotlib.pyplot as plt

        plotter = FocalPlanePlotter(input=str(pipeline_products["fp"]))
        plotter.plot_efficiency(
            efficiency=efficiency, scale="linear", channel_edges=True, ch_lengend=True
        )
        plt.close("all")

    def test_focal_plane_efficiency_rejects_unknown_kind(self, pipeline_products):
        plotter = FocalPlanePlotter(input=str(pipeline_products["fp"]))
        with pytest.raises(ValueError, match="invalid efficiency option"):
            plotter.plot_efficiency(efficiency="nonsense")

    def test_save_fig_without_a_figure_logs_an_error(self, pipeline_products):
        plotter = FocalPlanePlotter(input=str(pipeline_products["fp"]))
        # nothing plotted yet -> AttributeError swallowed, error logged
        plotter.save_fig(str(pipeline_products["dir"] / "never.png"))
        assert not (pipeline_products["dir"] / "never.png").exists()

    def test_radiometric_plotter(self, pipeline_products):
        import matplotlib.pyplot as plt

        plotter = RadiometricPlotter(input=str(pipeline_products["fp"]))
        plotter.plot_table(contribs=False)
        plotter.save_fig(str(pipeline_products["dir"] / "rm.png"))
        assert (pipeline_products["dir"] / "rm.png").exists()
        plotter.plot_apertures()
        plotter.plot_efficiency(scale="linear", channel_edges=True)
        plt.close("all")

    def test_radiometric_plotter_option_matrix(self, pipeline_products):
        import matplotlib.pyplot as plt

        plotter = RadiometricPlotter(input=str(pipeline_products["fp"]))

        plotter.plot_table(
            contribs=True,
            scale="linear",
            channel_edges=True,
            signal_ylim=(1e-3, 1e9),
            noise_ylim=(1e-3, 1e9),
        )
        plt.close("all")

        _fig, ax = plt.subplots()
        plotter.plot_bands(ax, scale="linear", channel_edges=True, add_legend=False)
        plotter.plot_noise(ax, contribs=True, ylim=(1e-6, 1.0), channel_edges=True)
        plt.close("all")

        _fig, ax = plt.subplots()
        plotter.plot_signal(ax, contribs=True, ylim=(1e-3, 1e12), channel_edges=True)
        plt.close("all")

    def test_radiometric_plotter_accepts_a_table_object(self, pipeline_products):
        import matplotlib.pyplot as plt

        # a plotter built from a file exposes the loaded table; feeding that
        # table straight back in must exercise the non-str branch of __init__
        table = RadiometricPlotter(input=str(pipeline_products["fp"])).input_table
        plotter = RadiometricPlotter(input=table)
        _fig, ax = plt.subplots()
        plotter.plot_signal(ax)
        plt.close("all")

    def test_radiometric_plotter_rejects_an_empty_table(self):
        from astropy.table import QTable

        with pytest.raises(ValueError, match="Empty table"):
            RadiometricPlotter(input=QTable())

    def test_focal_plane_plotter_bands(self, pipeline_products):
        import matplotlib.pyplot as plt

        plotter = FocalPlanePlotter(input=str(pipeline_products["fp"]))
        _fig, ax = plt.subplots()
        plotter.plot_bands(ax)
        plt.close("all")

    def test_sub_exposures_plotter(self, pipeline_products):
        plotter = SubExposuresPlotter(input=str(pipeline_products["se"]))
        outdir = pipeline_products["dir"] / "se_plots"
        outdir.mkdir(exist_ok=True)
        plotter.plot(str(outdir))
        assert any(outdir.iterdir())

    def test_ndrs_plotter(self, pipeline_products):
        plotter = NDRsPlotter(input=str(pipeline_products["ndr"]))
        outdir = pipeline_products["dir"] / "ndr_plots"
        outdir.mkdir(exist_ok=True)
        plotter.plot(str(outdir))
        assert any(outdir.iterdir())


class TestRadiometricModelModes:
    def test_single_source_mode_builds_its_own_focal_plane(self, pipeline_products):
        # output file does not exist and there is no target list -> single-source
        rm_file = pipeline_products["dir"] / "rm_single.h5"
        recipes.RadiometricModel(pipeline_products["config"], str(rm_file))
        with h5py.File(rm_file, "r") as f:
            assert "radiometric" in f
            assert "channels" in f  # it created the focal plane itself

    def test_radiometric_model_with_plotter_flag(self, pipeline_products):
        # exercise the recipe's built-in plotting hook
        rm_file = pipeline_products["dir"] / "rm_plot.h5"
        recipes.RadiometricModel(pipeline_products["config"], str(rm_file), plot=True)
        assert rm_file.exists()

    def test_single_unwrapped_channel_config_runs(self, pipeline_products):
        # a payload whose "channel" is one bare dict (not a mapping of channels)
        import copy

        config = copy.deepcopy(pipeline_products["config"])
        only = next(iter(config["payload"]["channel"].values()))
        config["payload"]["channel"] = dict(only)
        config["payload"]["channel"]["value"] = "Photometer"

        fp = pipeline_products["dir"] / "fp_single.h5"
        recipes.CreateFocalPlane(config, str(fp))
        recipes.RadiometricModel(config, str(fp))
        with h5py.File(fp, "r") as f:
            assert list(f["channels"].keys()) == ["Photometer"]
            cols = set(f["radiometric"]["table_to_group"].keys())
            assert "source_signal_in_aperture" in cols
            assert "transmission" in cols  # rebin_efficiencies ran

    def test_sub_foreground_signals_pass_the_per_channel_config(
        self, pipeline_products
    ):
        # give the focal-plane file a sub_focal_planes group, then run just the
        # sub-foreground step: the "<name>_total_signal" column only appears if
        # the task received the channel config (with its "type"), not the whole
        # channel mapping.
        import shutil

        from astropy.table import QTable

        from exosim.output import SetOutput

        src = pipeline_products["fp"]
        work = pipeline_products["dir"] / "fp_subfrg.h5"
        shutil.copy(src, work)
        with h5py.File(work, "a") as f:
            ch = next(iter(f["channels"]))
            osf = int(f["channels"][ch]["focal_plane"]["metadata"]["oversampling"][()])
            g = f["channels"][ch].create_group("sub_focal_planes/frg_zodi")
            shp = f["channels"][ch]["frg_focal_plane"]["data"].shape
            g.create_dataset("data", data=np.ones(shp))
            g.create_dataset("data_units", data="ct / s")
            g.create_group("metadata").create_dataset("oversampling", data=osf)

        rm = recipes.RadiometricModel.__new__(recipes.RadiometricModel)
        from exosim.log import Logger

        Logger.__init__(rm)
        rm.output = SetOutput(str(work), replace=False)
        rm.payloadConfig = pipeline_products["config"]["payload"]
        with h5py.File(work, "r") as f:
            wl = f["radiometric"]["table_to_group"]["wavelength"]["value"][()]
        n = len(wl)
        rm.table = QTable(
            {
                "ch_name": [next(iter(rm.payloadConfig["channel"]))] * n,
                "wavelength": wl * u.um,
                "spectral_center": np.arange(n, dtype=float),
                "spectral_size": np.full(n, 3.0),
                "spatial_center": np.full(n, float(shp[1] // 2)),
                "spatial_size": np.full(n, 3.0),
                "aperture_shape": ["rectangular"] * n,
                "left_bin_edge": wl * u.um - 0.01 * u.um,
                "right_bin_edge": wl * u.um + 0.01 * u.um,
            }
        )
        _full, view = rm.compute_sub_foregrounds_signals()
        assert "zodi_signal_in_aperture" in view.colnames
        assert "zodi_total_signal" in view.colnames  # proves "type" was seen

    def test_target_list_mode_writes_a_table_per_target(self, pipeline_products):
        import copy

        import pandas as pd

        config = copy.deepcopy(pipeline_products["config"])
        csv = pipeline_products["dir"] / "targets.csv"
        pd.DataFrame(
            {
                "Star": ["HD 209458", "GJ 1214"],
                "Radius": [1.18, 0.218],
                "Distance": [47, 13],
                "Temp": [6086, 3026],
                "Mass": [1.15, 0.176],
            }
        ).to_csv(csv, index=False)
        config["sky"]["source"] = {
            "targetlist_filepath": str(csv),
            "source_type": "planck",
            "column_mapping": {
                "name": "Star",
                "R": "Radius",
                "D": "Distance",
                "T": "Temp",
                "M": "Mass",
            },
        }
        rm_file = pipeline_products["dir"] / "rm_targetlist.h5"
        recipes.RadiometricModel(config, str(rm_file))
        with h5py.File(rm_file, "r") as f:
            assert set(f["targets"].keys()) == {"HD 209458", "GJ 1214"}
            for name in ("HD 209458", "GJ 1214"):
                cols = f["radiometric"][name]["table_to_group"]
                assert "source_signal_in_aperture" in set(cols.keys())


class TestSpectrometerFocalPlane:
    def test_focal_plane_is_dispersed(self, spectrometer_focal_plane):
        with h5py.File(spectrometer_focal_plane["fp"], "r") as f:
            data = f["channels"]["Spectrometer"]["focal_plane"]["data"][()]
            assert data.ndim == 3
            assert np.all(np.isfinite(data))
            # a spectrometer disperses the source along the spectral axis, so
            # more than a handful of columns carry signal
            lit_columns = np.count_nonzero(data[0].sum(axis=0) > data.max() * 1e-6)
            assert lit_columns > 10

    def test_radiometric_table_has_spectrometer_bins(self, spectrometer_focal_plane):
        with h5py.File(spectrometer_focal_plane["fp"], "r") as f:
            cols = f["radiometric"]["table_to_group"]
            wl = cols["wavelength"]["value"][()]
            # several wavelength bins across the 1-3.5 micron passband
            assert len(wl) > 5
            assert wl.min() >= 1.0
            assert wl.max() <= 3.5
            assert "source_signal_in_aperture" in set(cols.keys())

    def test_plotters_run_on_the_spectrometer_output(self, spectrometer_focal_plane):
        import matplotlib.pyplot as plt

        fpp = FocalPlanePlotter(input=str(spectrometer_focal_plane["fp"]))
        fpp.plot_focal_plane(time_step=0, scale="linear")

        rmp = RadiometricPlotter(input=str(spectrometer_focal_plane["fp"]))
        rmp.plot_table(contribs=True)
        _fig, ax = plt.subplots()
        rmp.plot_signal(ax, contribs=True)
        rmp.plot_noise(ax, contribs=True)
        plt.close("all")


class TestReadoutSchemeCalculator:
    def test_suggests_a_scheme_from_the_focal_plane(self, pipeline_products):
        import copy

        from exosim.tools import ReadoutSchemeCalculator

        config = copy.deepcopy(pipeline_products["config"])
        ch = next(iter(config["payload"]["channel"].values()))
        ch["readout"].update(
            {
                "readout_frequency": 10 * u.Hz,
                "n_NRDs_per_group": 2,
                "n_groups": 3,
                "Ground_time": 0.2 * u.s,
                "Reset_time": 0.2 * u.s,
            }
        )
        tool = ReadoutSchemeCalculator(
            config["payload"], input_file=str(pipeline_products["fp"])
        )
        name = next(iter(config["payload"]["channel"]))
        scheme = tool.results[name]
        assert scheme["n_groups"] == 3
        assert scheme["n_sim_clocks_groups"] > 0
