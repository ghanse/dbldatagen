import io
import json
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import pyarrow as pa
import pytest

from profiler_bench.catalog import __main__ as cli
from profiler_bench.catalog import real
from profiler_bench.catalog.real import RealDataset, RealTable, Unavailable, instance_dir, materialize
from profiler_bench.catalog.registry import spark_type


COLUMNS = "d decimal(7,2), ts timestamp, m map<string,bigint>, s struct<a:int,b:string>, a array<bigint>"
TABLE = pa.table(
    {
        "d": pa.array([Decimal("1.25"), None, Decimal("-99999.99")], pa.decimal128(7, 2)),
        "ts": pa.array(
            [datetime(2026, 1, 2, 3, 4, 5, 6, tzinfo=timezone.utc), None, datetime(1999, 12, 31)],
            pa.timestamp("us", tz="UTC"),
        ),
        "m": pa.array([[("x", 1)], None, []], pa.map_(pa.string(), pa.int64())),
        "s": pa.array(
            [{"a": 1, "b": "p"}, None, {"a": None, "b": "q"}], pa.struct([("a", pa.int32()), ("b", pa.string())])
        ),
        "a": pa.array([[1, 2], None, [None]], pa.list_(pa.int64())),
    }
)


def _dataset(rows: int = TABLE.num_rows) -> RealDataset:
    return RealDataset(
        name="probe",
        description="One table with nested and typed columns.",
        license="none",
        terms="none",
        tables=(RealTable("t", "x.y.t", rows, COLUMNS),),
    )


def _stream(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return bytes(sink.getvalue().to_pybytes())


def _forbidden(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=io.BytesIO(b""))  # type: ignore[arg-type]


class FakeWarehouse:
    """Answers Statement Execution API calls and presigned-link downloads for one table split into chunks."""

    def __init__(self, table: pa.Table, chunks: int = 3, count: int | None = None) -> None:
        bounds = [table.num_rows * i // chunks for i in range(chunks + 1)]
        self.parts = [_stream(table.slice(lo, hi - lo)) for lo, hi in pairwise(bounds)]
        self.count = table.num_rows if count is None else count
        self.statements: list[str] = []
        self.link_requests: Counter[int] = Counter()
        self.failures: dict[int, list[Exception]] = {}

    def urlopen(self, request: urllib.request.Request | str, timeout: float | None = None) -> io.BytesIO:
        if isinstance(request, str):
            index = int(request.split("/")[3])
            if self.failures.get(index):
                raise self.failures[index].pop(0)
            return io.BytesIO(self.parts[index])
        if request.get_method() == "POST":
            statement = json.loads(request.data)["statement"]  # type: ignore[arg-type]
            self.statements.append(statement)
            if statement.startswith("SELECT COUNT(*)"):
                result = {
                    "statement_id": "c",
                    "status": {"state": "SUCCEEDED"},
                    "result": {"data_array": [[str(self.count)]]},
                }
            else:
                result = {
                    "statement_id": "d",
                    "status": {"state": "SUCCEEDED"},
                    "manifest": {"total_chunk_count": len(self.parts)},
                }
            return io.BytesIO(json.dumps(result).encode())
        index = int(request.full_url.rsplit("/", 1)[1])
        self.link_requests[index] += 1
        link = f"https://links.example/{index}/{self.link_requests[index]}"
        return io.BytesIO(json.dumps({"external_links": [{"external_link": link}]}).encode())


@pytest.fixture
def warehouse(monkeypatch: pytest.MonkeyPatch) -> FakeWarehouse:
    fake = FakeWarehouse(TABLE)
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    monkeypatch.setattr(real, "RETRY_SECONDS", 0.0)
    monkeypatch.setenv(real.WAREHOUSE_ENV, "w")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.setenv("DATABRICKS_TOKEN", "t")
    return fake


def test_export_fetches_every_chunk_and_counts_the_source(warehouse: FakeWarehouse) -> None:
    tables = real.export(_dataset(), real.connect())
    assert tables["t"].to_pylist() == TABLE.to_pylist()
    assert warehouse.link_requests == {0: 1, 1: 1, 2: 1}
    assert warehouse.statements[0] == "SELECT COUNT(*) FROM x.y.t"


def test_expired_link_is_refreshed(warehouse: FakeWarehouse) -> None:
    warehouse.failures[1] = [_forbidden("https://links.example/1/1")]
    assert real.export(_dataset(), real.connect())["t"].num_rows == TABLE.num_rows
    assert warehouse.link_requests[1] == 2


def test_link_that_keeps_expiring_is_unavailable(warehouse: FakeWarehouse) -> None:
    warehouse.failures[1] = [_forbidden("https://links.example/1") for _ in range(real.LINK_ATTEMPTS)]
    with pytest.raises(Unavailable, match="expired on 3 attempts"):
        real.export(_dataset(), real.connect())
    assert warehouse.link_requests[1] == real.LINK_ATTEMPTS


@pytest.mark.parametrize("failure", [urllib.error.URLError("network is down"), ConnectionResetError("reset by peer")])
def test_offline_mid_download_skips_cleanly(
    warehouse: FakeWarehouse,
    failure: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    warehouse.failures[2] = [failure]
    monkeypatch.setitem(cli.REAL_DATASETS, "probe", _dataset())
    assert cli.main(["real", "probe", "--tier", "full", "--cache-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.startswith("skipped probe: chunk 2 download failed")
    assert not instance_dir(_dataset(), "full", tmp_path).exists()


def test_source_count_must_match_received_rows(warehouse: FakeWarehouse) -> None:
    warehouse.count = TABLE.num_rows + 1
    with pytest.raises(ValueError, match=r"source COUNT\(\*\) 4, received 3, declared 3"):
        real.export(_dataset(), real.connect())


def test_declared_rows_must_match_the_source(warehouse: FakeWarehouse) -> None:
    with pytest.raises(ValueError, match="declared 5"):
        real.export(_dataset(rows=5), real.connect())


def test_types_survive_arrow_to_parquet(warehouse: FakeWarehouse, tmp_path: Path) -> None:
    instance = materialize(_dataset(), "full", tmp_path)
    stored = instance.read_table("t")
    assert [spark_type(f.type) for f in stored.schema] == [c.type for c in instance.manifest.tables[0].columns]
    assert [c.type for c in instance.manifest.tables[0].columns] == [
        "decimal(7,2)",
        "timestamp",
        "map<string,bigint>",
        "struct<a:int,b:string>",
        "array<bigint>",
    ]
    assert stored.to_pylist() == TABLE.to_pylist()
