"""validate_config runs in worker threads: the include lists must not mix."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from exosim.mcp import configuration
from exosim.utils import RunConfig

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_concurrent_loads_do_not_mix_include_lists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Deterministic: a slow loader would interleave without the lock."""

    class SlowLoader:
        def __call__(self, filename: str, config_path: str) -> dict[str, Any]:
            RunConfig.config_file_list.append(os.path.abspath(filename))
            time.sleep(0.05)  # let another thread run between append and diff
            RunConfig.config_file_list.append(f"{filename}.include")
            return {"section": 1}

    monkeypatch.setattr(configuration, "LoadOptions", SlowLoader)
    files = []
    for name in ("a.xml", "b.xml", "c.xml", "d.xml"):
        path = tmp_path / name
        path.write_text("<root/>")
        files.append(path)

    barrier = threading.Barrier(len(files))

    def run(path: Path) -> dict[str, Any]:
        barrier.wait()
        return configuration.validate_configuration_file(path, repo_root=tmp_path)

    with ThreadPoolExecutor(len(files)) as pool:
        results = list(pool.map(run, files))

    for path, result in zip(files, results, strict=True):
        assert result["included_files"] == [
            os.path.abspath(path),
            f"{path}.include",
        ]


def test_concurrent_validation_of_real_examples() -> None:
    names = ["examples/main_example.xml", "examples/radiometric/main_radiometric.xml"]

    def run(name: str) -> list[str]:
        return configuration.validate_configuration_file(name, repo_root=REPO_ROOT)[
            "included_files"
        ]

    serial = {name: run(name) for name in names}
    with ThreadPoolExecutor(8) as pool:
        futures = {i: pool.submit(run, names[i % 2]) for i in range(8)}
    for i, future in futures.items():
        assert future.result() == serial[names[i % 2]]
