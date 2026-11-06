"""Paired-bootstrap evaluator and experiment reporting."""

from profiler_bench.evaluate.bootstrap import paired_bootstrap_ci, stratified_dataset_mean_ci
from profiler_bench.evaluate.evaluate import evaluate_experiment
from profiler_bench.evaluate.gates import GateResult, evaluate_gates, expected_sample_keys


__all__ = [
    "GateResult",
    "evaluate_experiment",
    "evaluate_gates",
    "expected_sample_keys",
    "paired_bootstrap_ci",
    "stratified_dataset_mean_ci",
]
