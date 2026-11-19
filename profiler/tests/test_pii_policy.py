import json

import pytest

from dbldatagen_profiler.cli import main
from dbldatagen_profiler.model import PiiClass, Semantic, Sensitivity
from dbldatagen_profiler.pii import PiiAction, PiiPolicy, PiiRule, dump_policy_json, load_policy


def semantic(pii_class: PiiClass, sensitivity: Sensitivity, confidence: float = 1.0) -> Semantic:
    return Semantic(pii_class=pii_class, sensitivity=sensitivity, confidence=confidence)


def test_default_policy_yaml_and_json_round_trip() -> None:
    policy = PiiPolicy()
    yaml_text = policy.to_yaml()
    assert load_policy(yaml_text) == policy
    assert load_policy(dump_policy_json(policy)) == policy
    assert load_policy(json.loads(dump_policy_json(policy))) == policy
    assert all(policy.sensitivity[kind].action == PiiAction.SUBSTITUTE for kind in Sensitivity)
    assert policy.min_confidence == 0.5


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
    parameters = {"prefix_length": 3} if action == PiiAction.GENERALIZE else {}
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


def test_cli_prints_default_policy(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["pii", "policy", "--default"]) == 0
    assert load_policy(capsys.readouterr().out) == PiiPolicy()
