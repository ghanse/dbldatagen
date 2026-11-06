"""Worker must import dbldatagen_profiler from the candidate worktree."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import dbldatagen_profiler
from profiler_bench.runner.identity import identify_harness
from profiler_bench.runner.policy import DEFAULT_POLICY, load_policy
from profiler_bench.runner.worker import _prepare_profiler_import
from profiler_bench.runner.worker import main as worker_main


REPO = Path(__file__).resolve().parents[3]


def test_prepare_profiler_import_uses_worktree_not_site_packages(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate-wt"
    src = candidate / "profiler" / "src" / "dbldatagen_profiler"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("TOKEN = 'from-candidate-worktree'\n")
    installed_file = Path(dbldatagen_profiler.__file__).resolve()
    assert "from-candidate-worktree" not in installed_file.read_text()

    _prepare_profiler_import(candidate)
    loaded = importlib.import_module("dbldatagen_profiler")
    assert loaded.__file__ is not None
    loaded_file = Path(loaded.__file__).resolve()
    assert loaded_file == (src / "__init__.py").resolve()
    assert "from-candidate-worktree" in loaded_file.read_text()
    assert loaded_file != installed_file
    for name in list(sys.modules):
        if name == "dbldatagen_profiler" or name.startswith("dbldatagen_profiler."):
            del sys.modules[name]
    importlib.invalidate_caches()
    importlib.import_module("dbldatagen_profiler")


def test_worker_records_profiler_module_under_worktree(tmp_path: Path, catalog_cache: Path) -> None:
    policy = load_policy(DEFAULT_POLICY)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    out = tmp_path / "worker-out.json"
    code = worker_main(
        [
            "--worktree",
            str(REPO),
            "--dataset",
            "numeric",
            "--tier",
            "xs",
            "--seed",
            "0",
            "--policy-json",
            str(policy_path),
            "--out",
            str(out),
            "--cache-dir",
            str(catalog_cache),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["ok"] is True
    module_file = Path(payload["profiler_module_file"]).resolve()
    assert module_file.is_relative_to((REPO / "profiler" / "src").resolve())


def test_identify_harness_uses_package_checkout() -> None:
    policy = load_policy(DEFAULT_POLICY)
    identity = identify_harness(policy=policy)
    assert Path(identity.harness_root).exists()
    assert identity.hashed_roots
    for root in identity.hashed_roots:
        path = Path(root)
        assert path.exists()
        assert path.name in {"profiler_bench", "policies"} or "profiler_bench" in path.as_posix()
