"""Fidelity metrics: reconstruction accuracy, structure recovery, and the privacy guard.

`score_reconstruction` compares synthetic tables with the real ones, `score_structure` compares a profile with the
dataset's ground-truth manifest, and `score_privacy` checks that neither the profile nor the synthetic data gives away
real rows or sensitive values. `ceiling` scores two catalog seeds against each other: the best achievable
reconstruction score. Every score is in [0, 1], JSON-serializable, and stamped with the hash of the metrics
configuration that produced it.
"""

from profiler_bench.fidelity.config import MetricsConfig, load_config
from profiler_bench.fidelity.privacy import score_privacy
from profiler_bench.fidelity.reconstruction import ceiling, score_reconstruction
from profiler_bench.fidelity.scores import OverallScore, PrivacyScore, ReconstructionScore, StructureScore, overall
from profiler_bench.fidelity.structure import score_structure


__all__ = [
    "MetricsConfig",
    "OverallScore",
    "PrivacyScore",
    "ReconstructionScore",
    "StructureScore",
    "ceiling",
    "load_config",
    "overall",
    "score_privacy",
    "score_reconstruction",
    "score_structure",
]
