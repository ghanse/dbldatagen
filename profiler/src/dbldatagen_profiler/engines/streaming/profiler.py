"""Incremental profiling of Spark Structured Streaming DataFrames."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path

from pyspark.errors import AnalysisException
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery

from dbldatagen_profiler.engines.streaming.checkpoint import Checkpoint, add_batch, read_checkpoint, write_checkpoint
from dbldatagen_profiler.engines.streaming.state import (
    analyze_sample,
    empty_state,
    merge_states,
    profile_states,
    require_streaming_support,
    state_for_dataframe,
)
from dbldatagen_profiler.model import PrivacyPolicy, Profile


PII_WARMUP_ROWS = 10_000
AVAILABLE_NOW = "availableNow"
STATE_FILE = "state.json"
SOURCE_ORDER_COLUMNS = ("__profile_file_time", "__profile_file_path", "__profile_file_row")


class StreamingProfiler:
    """Profiles streaming DataFrames incrementally with `foreachBatch`.

    Each micro-batch is reduced to a mergeable state and merged into the table's state, which is written atomically
    as versioned JSON to `checkpoint_dir/<table>/state.json` together with the set of merged batch ids. Spark's own
    checkpoint lives next to it in `checkpoint_dir/<table>/spark`. A batch whose id is already in the set (a replay
    after a crash) is skipped, so restarts neither double-count nor lose batches, whatever order batches arrive in.

    PII detection runs on the first `pii_warmup_rows` rows in source order and freezes once a table has seen that
    many rows; the raw rows are then dropped from the checkpoint. For file sources, source order is
    (file modification time, file path, row position in the file), which is the order in which Spark's file source
    takes files, so the frozen result does not depend on how files are split into micro-batches. Row position is
    the Parquet row index; for formats without one, rows within a file are ordered by the batch engine's xxhash64
    sampling key. For other sources, rows are ordered by that sampling key alone: the result is deterministic for
    a given set of rows, but which rows have arrived when the warmup is reached depends on the source.

    Args:
        tables: Streaming DataFrames keyed by profile table name.
        checkpoint_dir: Local directory for profiling and Spark checkpoints.
        privacy: Optional privacy policy applied to snapshots.
        pii_policy: Reserved for the PII handling policy (P2.3); must be None.
        trigger: `"availableNow"` or a processing-time interval such as `"10 seconds"`.
        detect_pii: Whether to detect sensitive semantics and quasi-identifiers (default true).
        pii_warmup_rows: Rows after which PII detection results freeze; also the PII sample size.

    Raises:
        RuntimeError: If PySpark is older than 3.5.
        ValueError: If no tables are supplied, or a checkpoint is invalid or belongs to a different schema.
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
    ) -> None:
        require_streaming_support()
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
            name: _load(self._directory / name / STATE_FILE, frame, pii_warmup_rows)
            for name, frame in self._tables.items()
        }

    def start(self) -> None:
        """Starts one streaming query per table."""
        for name, frame in self._tables.items():
            writer = (
                _with_source_order(frame)
                .writeStream.foreachBatch(partial(self._process, name))
                .option("checkpointLocation", str(self._directory / name / "spark"))
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
        if current.has_batch(batch_id):
            return
        collect_sample = self._detect_pii and current.pii is None
        order_columns = [column for column in SOURCE_ORDER_COLUMNS if column in batch.columns]
        batch.persist()
        try:
            state = state_for_dataframe(
                batch,
                sample_size=current.state.sample_size,
                collect_sample=collect_sample,
                order_columns=order_columns,
            )
            merged = merge_states(current.state, state, spark=batch.sparkSession)
        finally:
            batch.unpersist()
        pii = current.pii
        if collect_sample and merged.row_count >= self._warmup_rows:
            pii = analyze_sample(merged, batch.sparkSession)
            merged = replace(merged, sample=())
        checkpoint = Checkpoint(add_batch(current.batches, batch_id), merged, pii)
        write_checkpoint(self._directory / name / STATE_FILE, checkpoint)
        with self._lock:
            self._checkpoints[name] = checkpoint


def _with_source_order(frame: DataFrame) -> DataFrame:
    """Adds the file source order columns when the source exposes file metadata."""
    metadata = F.col("_metadata")
    file_columns: list[Column] = [
        F.unix_micros(metadata.file_modification_time).alias(SOURCE_ORDER_COLUMNS[0]),
        metadata.file_path.alias(SOURCE_ORDER_COLUMNS[1]),
    ]
    for columns in ([*file_columns, metadata.row_index.alias(SOURCE_ORDER_COLUMNS[2])], file_columns):
        try:
            return frame.select("*", *columns)
        except AnalysisException:
            continue
    return frame


def _load(path: Path, frame: DataFrame, sample_size: int) -> Checkpoint:
    if not path.exists():
        return Checkpoint((), empty_state(frame.schema, sample_size=sample_size))
    checkpoint = read_checkpoint(path)
    if checkpoint.state.schema != frame.schema:
        raise ValueError(f"checkpoint {path} does not match the table schema")
    return checkpoint
