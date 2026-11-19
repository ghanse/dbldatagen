"""Incremental profiling of Spark Structured Streaming DataFrames."""

from __future__ import annotations

import os
import pickle
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql.streaming import StreamingQuery

from dbldatagen_profiler.engines.streaming.state import (
    PII_SAMPLE_ROWS,
    PiiAnalysis,
    TableState,
    analyze_sample,
    empty_state,
    merge_states,
    profile_states,
    state_for_dataframe,
)
from dbldatagen_profiler.model import PrivacyPolicy, Profile


PII_WARMUP_ROWS = 10_000
AVAILABLE_NOW = "availableNow"
_STATE_FILE = "state.pkl"


@dataclass(frozen=True)
class Checkpoint:
    """Persisted profiling state of one streaming table.

    Attributes:
        batch_id: Last micro-batch merged into `state`, or -1 before the first.
        state: Merged state of every merged micro-batch.
        pii: PII analysis frozen after the warmup rows, or None before.
    """

    batch_id: int
    state: TableState
    pii: PiiAnalysis | None = None


class StreamingProfiler:
    """Profiles streaming DataFrames incrementally with `foreachBatch`.

    Each micro-batch is reduced to a mergeable state and merged into the table's state, which is written atomically
    to `checkpoint_dir/<table>/state.pkl` together with the batch id. Spark's own checkpoint lives next to it in
    `checkpoint_dir/<table>/spark`. A batch that Spark replays after a crash has an id the state already holds and is
    skipped, so restarts neither double-count nor lose batches.

    PII detection runs on the batch engine's bounded deterministic sample (the `pii_sample_rows` rows with the
    smallest xxhash64 key), maintained across batches. When a table has seen `pii_warmup_rows` rows, the detection
    result is frozen and the sample is dropped from the checkpoint; until then the checkpoint holds raw sample rows.

    Args:
        tables: Streaming DataFrames keyed by profile table name.
        checkpoint_dir: Local directory for profiling and Spark checkpoints.
        privacy: Optional privacy policy applied to snapshots.
        pii_policy: Reserved for the PII handling policy (P2.3); must be None.
        trigger: `"availableNow"` or a processing-time interval such as `"10 seconds"`.
        detect_pii: Whether to detect sensitive semantics and quasi-identifiers (default true).
        pii_warmup_rows: Rows after which PII detection results freeze.
        pii_sample_rows: Maximum PII detection sample size.

    Raises:
        ValueError: If no tables are supplied, or a checkpoint belongs to a different schema.
        NotImplementedError: If a `pii_policy` is supplied.
    """

    def __init__(
        self,
        tables: Mapping[str, DataFrame],
        *,
        checkpoint_dir: str | Path,
        privacy: PrivacyPolicy | None = None,
        pii_policy: object | None = None,
        trigger: str = AVAILABLE_NOW,
        detect_pii: bool = True,
        pii_warmup_rows: int = PII_WARMUP_ROWS,
        pii_sample_rows: int = PII_SAMPLE_ROWS,
    ) -> None:
        if not tables:
            raise ValueError("at least one table is required")
        if pii_policy is not None:
            raise NotImplementedError("pii_policy is not supported yet")
        self._tables = dict(tables)
        self._directory = Path(checkpoint_dir)
        self._privacy = privacy
        self._trigger = trigger
        self._detect_pii = detect_pii
        self._warmup_rows = pii_warmup_rows
        self._lock = threading.Lock()
        self._queries: list[StreamingQuery] = []
        self._checkpoints = {
            name: _read_checkpoint(self._directory / name, frame, pii_sample_rows)
            for name, frame in self._tables.items()
        }

    def start(self) -> None:
        """Starts one streaming query per table."""
        for name, frame in self._tables.items():
            writer = frame.writeStream.foreachBatch(partial(self._process, name)).option(
                "checkpointLocation", str(self._directory / name / "spark")
            )
            writer = (
                writer.trigger(availableNow=True)
                if self._trigger == AVAILABLE_NOW
                else writer.trigger(processingTime=self._trigger)
            )
            self._queries.append(writer.start())

    def await_termination(self, timeout: int | None = None) -> bool:
        """Waits for every query to finish.

        Args:
            timeout: Optional seconds to wait for each query.

        Returns:
            Whether every query has finished.

        Raises:
            StreamingQueryException: If a query failed.
        """
        return all(query.awaitTermination(timeout) is not False for query in self._queries)

    def stop(self) -> None:
        """Stops every query."""
        for query in self._queries:
            query.stop()
        self._queries = []

    def profile(self) -> Profile:
        """Returns a validated profile of every row merged so far."""
        with self._lock:
            checkpoints = dict(self._checkpoints)
        pii = (
            {name: checkpoint.pii or analyze_sample(checkpoint.state) for name, checkpoint in checkpoints.items()}
            if self._detect_pii
            else None
        )
        states = {name: checkpoint.state for name, checkpoint in checkpoints.items()}
        return profile_states(states, privacy=self._privacy, pii=pii)

    @property
    def pii_frozen(self) -> dict[str, bool]:
        """Whether each table's PII detection result is frozen."""
        with self._lock:
            return {name: checkpoint.pii is not None for name, checkpoint in self._checkpoints.items()}

    def _process(self, name: str, batch: DataFrame, batch_id: int) -> None:
        with self._lock:
            current = self._checkpoints[name]
        if batch_id <= current.batch_id:
            return
        collect_sample = self._detect_pii and current.pii is None
        batch.persist()
        try:
            state = state_for_dataframe(batch, sample_size=current.state.sample_size, collect_sample=collect_sample)
            merged = merge_states(current.state, state, spark=batch.sparkSession)
        finally:
            batch.unpersist()
        pii = current.pii
        if collect_sample and merged.row_count >= self._warmup_rows:
            pii = analyze_sample(merged, batch.sparkSession)
            merged = replace(merged, sample=())
        checkpoint = Checkpoint(batch_id, merged, pii)
        _write_checkpoint(self._directory / name, checkpoint)
        with self._lock:
            self._checkpoints[name] = checkpoint


def _read_checkpoint(directory: Path, frame: DataFrame, sample_size: int) -> Checkpoint:
    path = directory / _STATE_FILE
    if not path.exists():
        return Checkpoint(-1, empty_state(frame.schema, sample_size=sample_size))
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, Checkpoint) or checkpoint.state.schema != frame.schema:
        raise ValueError(f"checkpoint {path} does not match the table schema")
    return checkpoint


def _write_checkpoint(directory: Path, checkpoint: Checkpoint) -> None:
    """Writes a checkpoint atomically: to a temporary file that then replaces the previous one."""
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f"{_STATE_FILE}.tmp"
    with temporary.open("wb") as handle:
        pickle.dump(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, directory / _STATE_FILE)
