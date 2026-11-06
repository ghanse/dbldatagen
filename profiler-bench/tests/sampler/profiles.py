from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from dbldatagen_profiler.model import ColumnProfile, EngineInfo, LogicalType, PrivacyPolicy, Profile, TableProfile


def column(name: str, kind: LogicalType, **values: Any) -> ColumnProfile:  # noqa: ANN401
    """Builds a concise column profile for sampler tests."""
    return ColumnProfile(
        name=name,
        logical_type=kind,
        source_type=values.pop("source_type", kind.value),
        nullable=values.pop("nullable", False),
        null_fraction=values.pop("null_fraction", 0.0),
        **values,
    )


def profile(*tables: TableProfile) -> Profile:
    """Builds a deterministic profile envelope for sampler tests."""
    return Profile(
        spec_version="0.1.0",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        engine=EngineInfo(name="test", version="1"),
        privacy=PrivacyPolicy(allow_expression_constraints=True),
        tables=list(tables),
    )
