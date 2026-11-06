"""Catalog datasets by name."""

from profiler_bench.catalog.datasets import edge, nested, numeric, retail, strings, temporal
from profiler_bench.catalog.registry import Dataset


DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in (retail.DATASET, numeric.DATASET, strings.DATASET, temporal.DATASET, nested.DATASET, edge.DATASET)
}
