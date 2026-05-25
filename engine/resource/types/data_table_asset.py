"""Data table asset for structured tabular game data."""
from __future__ import annotations

from typing import Callable

from engine.resource.types.base_asset import BaseAsset

__all__ = ["DataTableAsset"]


class DataTableAsset(BaseAsset):
    """A table of structured row data (e.g. item databases, loot tables)."""

    __slots__ = ("_columns", "_rows", "_row_type", "_loaded")

    def __init__(
        self,
        asset_id: int,
        name: str,
        path: str,
        size_bytes: int,
        columns: list[str],
        row_type: str,
        rows: list[dict] | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(asset_id, name, path, size_bytes, version)
        self._columns = list(columns)
        self._row_type = row_type
        self._rows: list[dict] = rows or []
        self._loaded = False

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def rows(self) -> list[dict]:
        return list(self._rows)

    @property
    def row_type(self) -> str:
        return self._row_type

    # --- BaseAsset interface ---

    def load(self, data: bytes) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    # --- table helpers ---

    def get_row(self, index: int) -> dict:
        """Return row at the given index."""
        if index < 0 or index >= len(self._rows):
            raise IndexError(f"Row index {index} out of range [0, {len(self._rows)})")
        return dict(self._rows[index])

    def get_column_values(self, col: str) -> list:
        """Return all values for a given column name."""
        if col not in self._columns:
            raise KeyError(f"Unknown column {col!r}")
        return [row.get(col) for row in self._rows]

    def find_rows(self, predicate: Callable[[dict], bool]) -> list[dict]:
        """Return all rows matching the predicate."""
        return [dict(r) for r in self._rows if predicate(r)]
