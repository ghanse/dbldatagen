import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from profiler_bench.fidelity import MetricsConfig, load_config
from profiler_bench.fidelity.config import DEFAULT_CONFIG


def test_packaged_config_loads_and_hash_is_stable() -> None:
    first, second = load_config(), load_config()
    assert first.hash() == second.hash()
    assert len(first.hash()) == 64


def test_hash_ignores_formatting_but_not_values(tmp_path: Path) -> None:
    data = json.loads(DEFAULT_CONFIG.read_text())
    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps(data))
    assert load_config(compact).hash() == load_config().hash()
    data["discriminator"]["max_iter"] = 101
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(data))
    assert load_config(changed).hash() != load_config().hash()


@pytest.mark.parametrize("key", ["reconstruction_weights", "structure_weights"])
def test_weights_must_sum_to_one(key: str) -> None:
    data = json.loads(DEFAULT_CONFIG.read_text())
    first = next(iter(data[key]))
    data[key][first] += 0.5
    with pytest.raises(ValidationError, match="sum to 1"):
        MetricsConfig.model_validate(data)


def test_ceiling_and_holdout_need_different_sample_seeds() -> None:
    data = json.loads(DEFAULT_CONFIG.read_text())
    data["sampling"]["holdout_sample_offset"] = data["sampling"]["ceiling_sample_offset"]
    with pytest.raises(ValidationError, match="different sample seeds"):
        MetricsConfig.model_validate(data)
