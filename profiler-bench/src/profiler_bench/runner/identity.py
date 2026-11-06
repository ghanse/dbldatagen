"""Build and harness identity: git commit, dirty flag, and content hashes."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import profiler_bench
from profiler_bench.runner.policy import policy_hash


@dataclass(frozen=True)
class BuildIdentity:
    """Identity of one profiler build under test.

    Attributes:
        role: `baseline` or `candidate`.
        worktree: Absolute path to the git worktree that contains `profiler/src`.
        commit: Git commit SHA at that worktree, or `unknown` when unavailable.
        dirty: Whether the worktree has uncommitted changes.
        profiler_src_hash: SHA-256 of the sorted contents of `profiler/src`.
    """

    role: str
    worktree: str
    commit: str
    dirty: bool
    profiler_src_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class HarnessIdentity:
    """Identity of the frozen profiler-bench harness that is executing.

    Attributes:
        commit: Git commit SHA of the executing harness checkout.
        dirty: Whether that checkout has uncommitted changes.
        harness_hash: SHA-256 of the listed harness roots and metrics config digest.
        metrics_config_hash: Hash of the metrics configuration when present, else empty.
        policy_hash: Hash of the frozen policy used for this experiment.
        harness_root: Absolute path of the git checkout that owns the executing package.
        hashed_roots: Absolute paths whose contents were included in `harness_hash`.
    """

    commit: str
    dirty: bool
    harness_hash: str
    metrics_config_hash: str
    policy_hash: str
    harness_root: str
    hashed_roots: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return asdict(self)


def identify_build(role: str, worktree: Path) -> BuildIdentity:
    """Returns the identity of a profiler build at `worktree`."""
    root = worktree.resolve()
    src = root / "profiler" / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"profiler/src not found under {root}")
    return BuildIdentity(
        role=role,
        worktree=str(root),
        commit=_git_commit(root),
        dirty=_git_dirty(root),
        profiler_src_hash=hash_tree(src),
    )


def identify_harness(*, policy: dict[str, Any]) -> HarnessIdentity:
    """Returns the identity of the executing harness checkout.

    Commit, dirty flag, and source hashes all come from the same checkout resolved
    from the installed `profiler_bench` package location. `--repo-root` is never used.
    """
    harness_root, package_root, policies_root = _harness_paths()
    metrics = package_root / "fidelity" / "metrics_config.json"
    metrics_hash = hashlib.sha256(metrics.read_bytes()).hexdigest() if metrics.is_file() else ""
    hashed_roots: list[Path] = []
    digest = hashlib.sha256()
    if package_root.is_dir():
        hashed_roots.append(package_root)
        digest.update(hash_tree(package_root).encode())
    if policies_root.is_dir():
        hashed_roots.append(policies_root)
        digest.update(hash_tree(policies_root).encode())
    digest.update(metrics_hash.encode())
    return HarnessIdentity(
        commit=_git_commit(harness_root),
        dirty=_git_dirty(harness_root),
        harness_hash=digest.hexdigest(),
        metrics_config_hash=metrics_hash,
        policy_hash=policy_hash(policy),
        harness_root=str(harness_root),
        hashed_roots=tuple(str(path) for path in hashed_roots),
    )


def machine_info() -> dict[str, Any]:
    """Returns host facts recorded in `run.json`."""
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
    }


def hash_tree(root: Path) -> str:
    """Returns a SHA-256 over relative paths and file bytes under `root`."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def profile_content_hash(profile: dict[str, Any]) -> str:
    """Returns a SHA-256 of a profile with volatile fields removed."""
    normalized = json.loads(json.dumps(profile, sort_keys=True, default=str))
    normalized.pop("created_at", None)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _harness_paths() -> tuple[Path, Path, Path]:
    """Returns `(harness_git_root, package_root, policies_root)` for the executing package."""
    package_root = Path(profiler_bench.__file__).resolve().parent
    bench_dist_root = package_root.parents[1]  # .../profiler-bench
    harness_root = package_root.parents[2]  # repository checkout that owns profiler-bench
    return harness_root, package_root, bench_dist_root / "policies"


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_dirty(root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else True
