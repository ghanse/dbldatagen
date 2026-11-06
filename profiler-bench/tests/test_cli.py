import json
from pathlib import Path

import pytest

from profiler_bench.catalog import DATASETS
from profiler_bench.catalog.__main__ import main


def test_list_shows_every_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in DATASETS)
    assert "xs=5,000" in out


def test_materialize_prints_content_hashes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["materialize", "numeric", "edge", "--tier", "xs", "--cache-dir", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines] == ["numeric", "edge"]
    assert all(len(line.split()[3]) == 64 for line in lines)
    assert (tmp_path / "numeric" / "xs" / "seed=0").is_dir()


def test_materialize_rejects_unknown_datasets(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["materialize", "missing", "--cache-dir", str(tmp_path)])


def test_describe_prints_the_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["describe", "retail", "--tier", "xs", "--seed", "2", "--cache-dir", str(tmp_path)]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["dataset"] == "retail"
    assert manifest["seed"] == 2
    assert {t["name"] for t in manifest["tables"]} == {"customers", "products", "orders", "order_items"}
