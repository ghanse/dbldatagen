"""Profiles Spark DataFrames into the engine-neutral profile model."""

from dbldatagen_profiler.engines.spark.engine import profile_dataframe, profile_tables


__all__ = ["profile_dataframe", "profile_tables"]
