"""Hook for optional real public datasets: download once, verify a checksum, skip when offline.

Public datasets have no planted ground truth, so they carry no manifest. They complement the
synthetic catalog for realism and speed measurements. Register one by adding a `PublicDataset`
to `PUBLIC_DATASETS`.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from profiler_bench.catalog.store import cache_dir


@dataclass(frozen=True)
class PublicDataset:
    """A real dataset fetched from a public URL.

    Attributes:
        name: Dataset name.
        description: What the dataset contains and why it is in the catalog.
        url: Download URL.
        sha256: Expected SHA-256 of the downloaded file.
        filename: Name of the file in the cache.
        license: License the data is distributed under.
        read: Function from the downloaded file and a Spark session to DataFrames by table name.
    """

    name: str
    description: str
    url: str
    sha256: str
    filename: str
    license: str
    read: Callable[[Path, SparkSession], dict[str, DataFrame]]


PUBLIC_DATASETS: dict[str, PublicDataset] = {}


def fetch(dataset: PublicDataset, root: Path | None = None, timeout: float = 60.0) -> Path | None:
    """Returns the cached file of `dataset`, downloading it first if needed.

    Args:
        dataset: Public dataset to fetch.
        root: Cache root; defaults to `cache_dir()`.
        timeout: Download timeout in seconds.

    Returns:
        Path of the verified file, or None when the URL cannot be reached (offline).

    Raises:
        urllib.error.HTTPError: If the server answers with an error status.
        ValueError: If the downloaded file does not match `dataset.sha256`.
    """
    target = (root or cache_dir()) / "public" / dataset.name / dataset.filename
    if target.exists() and _sha256(target) == dataset.sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".tmp-", delete=False) as staging:
        staging_path = Path(staging.name)
        try:
            with urllib.request.urlopen(dataset.url, timeout=timeout) as response:
                shutil.copyfileobj(response, staging)
        except urllib.error.HTTPError:
            staging_path.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, TimeoutError):
            staging_path.unlink(missing_ok=True)
            return None
    actual = _sha256(staging_path)
    if actual != dataset.sha256:
        staging_path.unlink(missing_ok=True)
        raise ValueError(f"{dataset.name}: checksum {actual} != {dataset.sha256}")
    staging_path.replace(target)
    return target


def load_public(name: str, spark: SparkSession, root: Path | None = None) -> dict[str, DataFrame] | None:
    """Returns the tables of a public dataset, or None when it cannot be downloaded.

    Raises:
        KeyError: If no public dataset has that name.
    """
    dataset = PUBLIC_DATASETS[name]
    path = fetch(dataset, root)
    return None if path is None else dataset.read(path, spark)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
