import json
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest
from conftest import EXAMPLES

from dbldatagen_profiler.cli import main
from dbldatagen_profiler.model import render_json_schema


def test_console_script_is_registered() -> None:
    (script,) = entry_points(group="console_scripts", name="dbldatagen-profile")
    assert script.value == "dbldatagen_profiler.cli:main"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_validate_accepts_example(path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(path)]) == 0
    assert "valid profile (spec 0.1.0" in capsys.readouterr().out


def test_validate_rejects_invalid_profile(
    tmp_path: Path, minimal_profile: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    minimal_profile["tables"][0]["row_count"] = -1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(minimal_profile), encoding="utf-8")
    assert main(["validate", str(path)]) == 1
    assert "row_count" in capsys.readouterr().err


@pytest.mark.parametrize("content", [None, "{not json"])
def test_validate_rejects_unreadable_file(tmp_path: Path, content: str | None) -> None:
    path = tmp_path / "profile.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert main(["validate", str(path)]) == 1


def test_schema_prints_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema"]) == 0
    assert capsys.readouterr().out == render_json_schema()


def test_schema_writes_to_file(tmp_path: Path) -> None:
    output = tmp_path / "schema.json"
    assert main(["schema", "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == render_json_schema()


def test_missing_subcommand_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


def test_validate_strict_rejects_unknown_fields(
    tmp_path: Path, minimal_profile: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    minimal_profile["future_block"] = {}
    path = tmp_path / "future.json"
    path.write_text(json.dumps(minimal_profile), encoding="utf-8")
    assert main(["validate", str(path)]) == 0
    assert main(["validate", "--strict", str(path)]) == 1
    assert "future_block" in capsys.readouterr().err
