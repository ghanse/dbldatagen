"""Cross-experiment cache of baseline-build speed and fidelity samples.

An entry holds every baseline sample of one (kind, dataset, tier) case. Its file name is the
SHA-256 of the key components, and the components are stored in the entry and re-checked on
load, so any mismatch or corrupt file is a miss. Writes take an fcntl lock and use a temp
file plus `os.replace`, like the ceiling cache.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from profiler_bench.catalog import REAL_DATASETS, materialize
from profiler_bench.catalog.real import SPLIT
from profiler_bench.catalog.store import get


def config_hash(value: object) -> str:
    """Returns the SHA-256 of a JSON value in canonical form."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def entry_path(root: Path, components: dict[str, Any]) -> Path:
    """Returns the cache file for `components`."""
    return root / f"{components['kind']}-{config_hash(components)}.json"


def load_entry(root: Path, components: dict[str, Any]) -> dict[str, Any] | None:
    """Returns a valid cached entry for `components`, or None on a miss or corrupt file."""
    try:
        entry = json.loads(entry_path(root, components).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict) or entry.get("components") != components:
        return None
    if not isinstance(entry.get("source_experiment_id"), str):
        return None
    records = entry.get("records")
    if not isinstance(records, list) or not _is_complete(records, components):
        return None
    return entry


def store_entry(
    root: Path,
    components: dict[str, Any],
    *,
    source_experiment_id: str,
    records: Sequence[dict[str, Any]],
    profile: dict[str, Any] | None = None,
) -> None:
    """Writes an entry under an exclusive lock; skips sets that are incomplete or failed."""
    if not _is_complete(list(records), components):
        return
    entry: dict[str, Any] = {
        "components": components,
        "source_experiment_id": source_experiment_id,
        "records": list(records),
    }
    if profile is not None:
        entry["profile"] = profile
    path = entry_path(root, components)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            tmp_path.write_text(json.dumps(entry, sort_keys=True, default=str))
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _is_complete(records: list[Any], components: dict[str, Any]) -> bool:
    """Returns whether `records` are the complete, successful, ordered baseline set for `components`."""
    if components["kind"] == "fidelity":
        index_field, indices = "seed", list(components["sampler_seeds"])
    else:
        index_field, indices = "sample_index", list(range(int(components["samples"])))
    if not indices or len(records) != len(indices):
        return False
    return all(
        isinstance(record, dict)
        and record.get("ok") is True
        and record.get("kind") == components["kind"]
        and record.get("role") == "baseline"
        and record.get("dataset") == components["dataset"]
        and record.get("tier") == components["tier"]
        and record.get(index_field) == index
        for record, index in zip(records, indices, strict=True)
    )


ENVIRONMENT_PACKAGES = ("pyspark", "numpy", "pandas", "scikit-learn", "pyarrow")


def environment(*, java: bool) -> dict[str, str]:
    """Returns the Python and library versions, plus the Java version when `java` is set."""
    versions = {"python": platform.python_version()}
    for package in ENVIRONMENT_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "missing"
    if java:
        versions["java"] = java_version()
    return versions


def java_version() -> str:
    """Returns the first line of `java -version` for `$JAVA_HOME/bin/java` (or `java` on PATH)."""
    java_home = os.environ.get("JAVA_HOME")
    exe = str(Path(java_home) / "bin" / "java") if java_home else "java"
    try:
        completed = subprocess.run([exe, "-version"], check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return "missing"
    lines = (completed.stderr or completed.stdout).strip().splitlines()
    return lines[0] if lines else "missing"


def dataset_identity(dataset: str, tier: str, seed: int, root: Path | None) -> dict[str, Any]:
    """Returns the catalog generator fingerprint and instance content hash, or a real dataset's pins and split."""
    if dataset in REAL_DATASETS:
        return {"real_pinned_hashes": dict(REAL_DATASETS[dataset].hashes), "real_split": SPLIT}
    instance = materialize(dataset, tier, seed=seed, root=root, check_hash=False)
    return {"catalog_fingerprint": get(dataset).fingerprint(), "catalog_content_hash": instance.content_hash}
