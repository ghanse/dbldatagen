"""Dataset catalog with planted, declared ground truth.

Each dataset is a seeded numpy/Arrow generator plus a manifest describing the structure it planted.
`load(name, tier, seed, spark)` materializes the dataset to a parquet cache and returns DataFrames;
`verify(manifest, frames)` checks that the manifest is true of the data.
"""

from profiler_bench.catalog.datasets import DATASETS
from profiler_bench.catalog.manifest import Manifest
from profiler_bench.catalog.registry import DEFAULT_TIER, TIERS, Dataset, generate
from profiler_bench.catalog.store import Materialized, cache_dir, get, load, materialize
from profiler_bench.catalog.verification import verify


__all__ = [
    "DATASETS",
    "DEFAULT_TIER",
    "TIERS",
    "Dataset",
    "Manifest",
    "Materialized",
    "cache_dir",
    "generate",
    "get",
    "load",
    "materialize",
    "verify",
]
