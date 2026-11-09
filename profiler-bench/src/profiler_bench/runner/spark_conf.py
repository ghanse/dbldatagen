"""Fixed local SparkSession configuration for speed and fidelity workers."""

from __future__ import annotations

import re
from typing import Any

from pyspark.sql import SparkSession


DEFAULT_FIDELITY_DRIVER_MEMORY = "8g"
DEFAULT_FIDELITY_MASTER = "local[2]"


def build_spark(policy: dict[str, Any], *, app_name: str = "profiler-bench-rsi") -> SparkSession:
    """Returns a local SparkSession with the policy's fixed configuration."""
    spark_policy = policy.get("spark", {})
    master = str(spark_policy.get("master", "local[2]"))
    configs = dict(spark_policy.get("configs", {}))
    return _session(master=master, configs=configs, app_name=app_name)


def fidelity_spark_config(
    *,
    driver_memory: str = DEFAULT_FIDELITY_DRIVER_MEMORY,
    master: str = DEFAULT_FIDELITY_MASTER,
) -> dict[str, Any]:
    """Returns the fixed Spark config used by fidelity workers (not speed timing)."""
    return {
        "master": master,
        "configs": {
            "spark.driver.memory": driver_memory,
            "spark.sql.shuffle.partitions": str(local_shuffle_partitions(master)),
            "spark.sql.session.timeZone": "UTC",
            "spark.ui.enabled": "false",
        },
    }


def build_fidelity_spark(
    fidelity_spark: dict[str, Any],
    *,
    app_name: str = "profiler-bench-fidelity",
) -> SparkSession:
    """Returns a local SparkSession with the recorded fidelity configuration."""
    master = str(fidelity_spark.get("master", DEFAULT_FIDELITY_MASTER))
    configs = dict(fidelity_spark.get("configs", {}))
    return _session(master=master, configs=configs, app_name=app_name)


def local_shuffle_partitions(master: str) -> int:
    """Returns a shuffle partition count sized for a local[*]/local[N] master."""
    match = re.fullmatch(r"local\[(\d+)\]", master.strip())
    if match is not None:
        return max(1, int(match.group(1)))
    return 2


def _session(*, master: str, configs: dict[str, Any], app_name: str) -> SparkSession:
    builder = SparkSession.builder.master(master).appName(app_name)
    for key, value in configs.items():
        builder = builder.config(str(key), str(value))
    return builder.getOrCreate()
