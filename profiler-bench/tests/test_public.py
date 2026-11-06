import hashlib
import socket
from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession

from profiler_bench.catalog import public
from profiler_bench.catalog.public import PublicDataset, fetch, load_public


def _read(path: Path, spark: SparkSession) -> dict[str, DataFrame]:
    return {"t": spark.read.option("header", True).csv(str(path))}


def _dataset(url: str, sha256: str) -> PublicDataset:
    return PublicDataset(
        name="tiny", description="test", url=url, sha256=sha256, filename="t.csv", license="none", read=_read
    )


@pytest.fixture
def source(tmp_path: Path) -> tuple[str, str]:
    path = tmp_path / "source.csv"
    path.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")
    return path.as_uri(), hashlib.sha256(path.read_bytes()).hexdigest()


def test_fetch_downloads_and_verifies(source: tuple[str, str], tmp_path: Path) -> None:
    url, sha256 = source
    path = fetch(_dataset(url, sha256), tmp_path / "cache")
    assert path is not None
    assert path.read_text(encoding="utf-8").startswith("a,b")
    assert fetch(_dataset("file:///missing/now", sha256), tmp_path / "cache") == path


def test_fetch_rejects_a_bad_checksum(source: tuple[str, str], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checksum"):
        fetch(_dataset(source[0], "0" * 64), tmp_path / "cache")
    assert not (tmp_path / "cache" / "public" / "tiny" / "t.csv").exists()


def test_fetch_returns_none_when_offline(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert fetch(_dataset(f"http://127.0.0.1:{port}/t.csv", "0" * 64), tmp_path, timeout=5) is None


def test_load_public_reads_registered_dataset(
    source: tuple[str, str], tmp_path: Path, spark: SparkSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(public.PUBLIC_DATASETS, "tiny", _dataset(*source))
    frames = load_public("tiny", spark, tmp_path)
    assert frames is not None
    assert frames["t"].count() == 2
