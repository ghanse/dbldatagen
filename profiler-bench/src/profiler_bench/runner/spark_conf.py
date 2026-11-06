"""Fixed local SparkSession configuration for speed samples."""

from __future__ import annotations

from typing import Any

from pyspark.sql import SparkSession


def build_spark(policy: dict[str, Any], *, app_name: str = "profiler-bench-rsi") -> SparkSession:
    """Returns a local SparkSession with the policy's fixed configuration."""
    spark_policy = policy.get("spark", {})
    master = str(spark_policy.get("master", "local[2]"))
    configs = dict(spark_policy.get("configs", {}))
    builder = SparkSession.builder.master(master).appName(app_name)
    for key, value in configs.items():
        builder = builder.config(str(key), str(value))
    return builder.getOrCreate()
