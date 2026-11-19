"""Profiles Spark Structured Streaming DataFrames incrementally from mergeable per-batch states."""

from dbldatagen_profiler.engines.streaming.profiler import StreamingProfiler
from dbldatagen_profiler.engines.streaming.state import (
    DISTINCT_RELATIVE_ERROR,
    ColumnState,
    TableState,
    empty_state,
    merge_states,
    profile_states,
    state_for_dataframe,
)


__all__ = [
    "DISTINCT_RELATIVE_ERROR",
    "ColumnState",
    "StreamingProfiler",
    "TableState",
    "empty_state",
    "merge_states",
    "profile_states",
    "state_for_dataframe",
]
