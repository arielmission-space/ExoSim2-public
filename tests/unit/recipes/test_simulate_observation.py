"""
Behavioural test for the SimulateObservation orchestrator: it chains the four
recipes and, when a plots directory is given, the four plotters. The recipes and
plotters are mocked so the test is fast; it checks the wiring, the RunConfig
side effects, and the temp-file cleanup.
"""

from unittest.mock import MagicMock, patch

import pytest

from exosim.recipes.simulate_observation import SimulateObservation
from exosim.utils import RunConfig


@pytest.fixture(autouse=True)
def _restore_run_config():
    n_job, seed = RunConfig.n_job, RunConfig.random_seed
    yield
    RunConfig.n_job, RunConfig.random_seed = n_job, seed


def test_init_applies_run_config_and_creates_plot_dir(tmp_path):
    plots = tmp_path / "plots"
    obs = SimulateObservation(
        options_file="cfg.xml",
        output_file=str(tmp_path / "out.h5"),
        plots_dir=str(plots),
        n_job=1,
        random_seed=7,
    )
    assert plots.is_dir()
    assert RunConfig.random_seed == 7
    assert obs.output_file.endswith("out.h5")


def test_main_chains_recipes_and_plotters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # the recipe removes ./test_common.h5 and ./test_se.h5 at the end
    (tmp_path / "test_common.h5").write_text("")
    (tmp_path / "test_se.h5").write_text("")

    mod = "exosim.recipes.simulate_observation"
    with (
        patch(f"{mod}.CreateFocalPlane") as fp,
        patch(f"{mod}.RadiometricModel") as rm,
        patch(f"{mod}.CreateSubExposures") as se,
        patch(f"{mod}.CreateNDRs") as ndr,
        patch(f"{mod}.FocalPlanePlotter", return_value=MagicMock()),
        patch(f"{mod}.RadiometricPlotter", return_value=MagicMock()),
        patch(f"{mod}.SubExposuresPlotter", return_value=MagicMock()),
        patch(f"{mod}.NDRsPlotter", return_value=MagicMock()),
    ):
        obs = SimulateObservation(
            options_file="cfg.xml",
            output_file=str(tmp_path / "out.h5"),
            plots_dir=str(tmp_path / "plots"),
        )
        obs.main()

    fp.assert_called_once()
    rm.assert_called_once()
    se.assert_called_once()
    ndr.assert_called_once()


def test_main_without_plots_skips_the_plotters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test_common.h5").write_text("")
    (tmp_path / "test_se.h5").write_text("")

    mod = "exosim.recipes.simulate_observation"
    with (
        patch(f"{mod}.CreateFocalPlane"),
        patch(f"{mod}.RadiometricModel"),
        patch(f"{mod}.CreateSubExposures"),
        patch(f"{mod}.CreateNDRs"),
        patch(f"{mod}.FocalPlanePlotter") as fpp,
    ):
        SimulateObservation(
            options_file="cfg.xml", output_file=str(tmp_path / "out.h5")
        ).main()
    fpp.assert_not_called()
