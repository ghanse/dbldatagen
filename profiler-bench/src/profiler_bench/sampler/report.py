"""Reports how the reference sampler consumed a profile."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnReport:
    """Records profile fields consumed, approximated, ignored, or unmet for one column."""

    used: tuple[str, ...] = ()
    approximated: tuple[str, ...] = ()
    ignored: tuple[str, ...] = ()
    unmet: tuple[str, ...] = ()


@dataclass(frozen=True)
class SampleReport:
    """Records sampler coverage by ``table.column`` and table-level unmet constraints."""

    columns: dict[str, ColumnReport]
    unmet_constraints: dict[str, tuple[str, ...]] = field(default_factory=dict)


class ReportBuilder:
    """Accumulates immutable sampler report entries."""

    def __init__(self) -> None:
        self._fields: dict[str, dict[str, set[str]]] = {}
        self.unmet_constraints: dict[str, list[str]] = {}

    def add(self, path: str, category: str, *fields: str) -> None:
        """Adds fields under a report category for a column path."""
        entry = self._fields.setdefault(path, {name: set() for name in ColumnReport.__dataclass_fields__})
        entry[category].update(fields)

    def unmet(self, table: str, constraint: str) -> None:
        """Records a constraint that could not be fully enforced."""
        self.unmet_constraints.setdefault(table, []).append(constraint)

    def build(self) -> SampleReport:
        """Returns the stable, immutable report."""
        columns = {
            path: ColumnReport(**{category: tuple(sorted(values)) for category, values in entry.items()})
            for path, entry in sorted(self._fields.items())
        }
        unmet = {table: tuple(values) for table, values in sorted(self.unmet_constraints.items())}
        return SampleReport(columns=columns, unmet_constraints=unmet)
