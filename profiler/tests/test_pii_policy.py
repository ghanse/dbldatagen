import json

import pytest

from dbldatagen_profiler.cli import main
from dbldatagen_profiler.model import PiiClass, Semantic, Sensitivity
from dbldatagen_profiler.pii import (
    PiiAction,
    PiiPolicy,
    PiiRule,
    dump_policy_json,
    expected_correlation_attenuation,
    load_policy,
)


def semantic(pii_class: PiiClass, sensitivity: Sensitivity, confidence: float = 1.0) -> Semantic:
    return Semantic(pii_class=pii_class, sensitivity=sensitivity, confidence=confidence)


def test_default_policy_yaml_and_json_round_trip() -> None:
    policy = PiiPolicy()
    yaml_text = policy.to_yaml()
    assert load_policy(yaml_text) == policy
    assert load_policy(dump_policy_json(policy)) == policy
    assert load_policy(json.loads(dump_policy_json(policy))) == policy
    assert all(policy.sensitivity[kind].action == PiiAction.SUBSTITUTE for kind in Sensitivity)
    assert policy.min_confidence == 0.25
    assert PiiAction.PERTURB not in {rule.action for rule in policy.sensitivity.values()}


def test_yaml_round_trip_preserves_overrides_and_parameters() -> None:
    policy = PiiPolicy(
        pii_class={PiiClass.POSTAL_CODE: PiiRule(action=PiiAction.GENERALIZE, parameters={"prefix_length": 3})},
        columns={
            "people.city": PiiRule(action=PiiAction.SUPPRESS),
            "people.contact": PiiRule(action=PiiAction.SUBSTITUTE, pii_class=PiiClass.EMAIL),
        },
    )
    assert load_policy(policy.to_yaml()) == policy


def test_long_json_string_is_parsed_as_content() -> None:
    text = dump_policy_json(
        PiiPolicy(columns={f"table.column_{index}": PiiRule(action=PiiAction.KEEP) for index in range(30)})
    )
    assert len(text) > 255
    assert load_policy(text).columns == load_policy(json.loads(text)).columns


def test_column_then_class_then_sensitivity_precedence() -> None:
    policy = PiiPolicy(
        sensitivity={kind: PiiRule(action=PiiAction.KEEP) for kind in Sensitivity},
        pii_class={PiiClass.CITY: PiiRule(action=PiiAction.SUPPRESS)},
        columns={"people.city": PiiRule(action=PiiAction.DROP)},
    )
    city = semantic(PiiClass.CITY, Sensitivity.QUASI_IDENTIFIER)
    assert policy.resolve("people.city", city)[1] == "column"
    assert policy.resolve("other.city", city)[1] == "pii_class"
    assert policy.resolve("other.zip", semantic(PiiClass.POSTAL_CODE, Sensitivity.QUASI_IDENTIFIER))[1] == "sensitivity"


@pytest.mark.parametrize("sensitivity", list(Sensitivity))
@pytest.mark.parametrize("action", list(PiiAction))
def test_each_action_can_be_configured_for_each_sensitivity(sensitivity: Sensitivity, action: PiiAction) -> None:
    if action == PiiAction.GENERALIZE:
        parameters: dict[str, str | int | float | bool] = {"prefix_length": 3}
    elif action == PiiAction.PERTURB:
        parameters = {"mode": "noise", "noise_ratio": 0.1}
    else:
        parameters = {}
    policy = PiiPolicy(
        sensitivity={
            kind: (
                PiiRule(action=action, parameters=parameters) if kind == sensitivity else PiiRule(action=PiiAction.KEEP)
            )
            for kind in Sensitivity
        }
    )
    rule, source = policy.resolve("t.value", semantic(PiiClass.OTHER, sensitivity))
    assert (rule.action, source) == (action, "sensitivity")


def test_generalize_requires_parameters() -> None:
    with pytest.raises(ValueError, match="generalize requires parameters"):
        PiiRule(action=PiiAction.GENERALIZE)


def test_perturb_parameter_validation_and_attenuation() -> None:
    assert PiiRule(action=PiiAction.PERTURB).perturb_mode() == "noise"
    assert PiiRule(action=PiiAction.PERTURB).perturb_noise_ratio() == 0.1
    noise = PiiRule(action=PiiAction.PERTURB, parameters={"mode": "noise", "noise_ratio": 0.1})
    affine = PiiRule(action=PiiAction.PERTURB, parameters={"mode": "affine"})
    assert expected_correlation_attenuation(noise) == pytest.approx(1.0 / (1.01**0.5))
    assert expected_correlation_attenuation(affine) == 1.0
    with pytest.raises(ValueError, match="noise_ratio must satisfy"):
        PiiRule(action=PiiAction.PERTURB, parameters={"noise_ratio": 0.0})
    with pytest.raises(ValueError, match="noise_ratio must satisfy"):
        PiiRule(action=PiiAction.PERTURB, parameters={"noise_ratio": 1.5})
    with pytest.raises(ValueError, match="mode must be"):
        PiiRule(action=PiiAction.PERTURB, parameters={"mode": "shuffle"})
    with pytest.raises(ValueError, match="affine mode does not use noise_ratio"):
        PiiRule(action=PiiAction.PERTURB, parameters={"mode": "affine", "noise_ratio": 0.1})
    with pytest.raises(ValueError, match="does not accept parameters"):
        PiiRule(action=PiiAction.PERTURB, parameters={"mode": "noise", "sigma": 1})


def test_perturb_yaml_and_json_round_trip() -> None:
    policy = PiiPolicy(
        sensitivity={
            Sensitivity.DIRECT_IDENTIFIER: PiiRule(action=PiiAction.SUBSTITUTE),
            Sensitivity.QUASI_IDENTIFIER: PiiRule(
                action=PiiAction.PERTURB, parameters={"mode": "noise", "noise_ratio": 0.2}
            ),
            Sensitivity.SECRET: PiiRule(action=PiiAction.SUBSTITUTE),
            Sensitivity.FREE_TEXT: PiiRule(action=PiiAction.SUBSTITUTE),
        },
        columns={"t.score": PiiRule(action=PiiAction.PERTURB, parameters={"mode": "affine"})},
    )
    assert load_policy(policy.to_yaml()) == policy
    assert load_policy(dump_policy_json(policy)) == policy


def test_cli_prints_default_policy(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["pii", "policy", "--default"]) == 0
    assert load_policy(capsys.readouterr().out) == PiiPolicy()
