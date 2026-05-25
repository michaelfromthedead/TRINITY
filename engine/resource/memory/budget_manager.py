"""Memory budget tracking per asset category."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from engine.resource.constants import (
    DEFAULT_AUDIO_BUDGET,
    DEFAULT_MESH_BUDGET,
    DEFAULT_TEXTURE_BUDGET,
)


class AssetCategory(Enum):
    """Categories of game assets for budget tracking."""

    TEXTURE = auto()
    MESH = auto()
    AUDIO = auto()
    ANIMATION = auto()
    SHADER = auto()
    MATERIAL = auto()
    OTHER = auto()


_DEFAULT_BUDGETS: dict[AssetCategory, int] = {
    AssetCategory.TEXTURE: DEFAULT_TEXTURE_BUDGET,
    AssetCategory.MESH: DEFAULT_MESH_BUDGET,
    AssetCategory.AUDIO: DEFAULT_AUDIO_BUDGET,
}


@dataclass
class BudgetEntry:
    """Tracks budget usage for a single asset category."""

    __slots__ = ("category", "budget_bytes", "used_bytes", "peak_bytes")

    category: AssetCategory
    budget_bytes: int
    used_bytes: int
    peak_bytes: int

    def __init__(
        self,
        category: AssetCategory,
        budget_bytes: int = 0,
        used_bytes: int = 0,
        peak_bytes: int = 0,
    ) -> None:
        self.category = category
        self.budget_bytes = budget_bytes
        self.used_bytes = used_bytes
        self.peak_bytes = peak_bytes


class BudgetManager:
    """Tracks memory budgets per asset category."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[AssetCategory, BudgetEntry] = {}
        for cat, budget in _DEFAULT_BUDGETS.items():
            self._entries[cat] = BudgetEntry(category=cat, budget_bytes=budget)

    def set_budget(self, category: AssetCategory, budget_bytes: int) -> None:
        """Set or update the memory budget for a category."""
        if category in self._entries:
            self._entries[category].budget_bytes = budget_bytes
        else:
            self._entries[category] = BudgetEntry(
                category=category, budget_bytes=budget_bytes
            )

    def allocate(self, category: AssetCategory, size_bytes: int) -> bool:
        """Allocate memory. Returns False if allocation would exceed budget."""
        entry = self._entries.get(category)
        if entry is None:
            return False
        if entry.used_bytes + size_bytes > entry.budget_bytes:
            return False
        entry.used_bytes += size_bytes
        if entry.used_bytes > entry.peak_bytes:
            entry.peak_bytes = entry.used_bytes
        return True

    def free(self, category: AssetCategory, size_bytes: int) -> None:
        """Free previously allocated memory."""
        entry = self._entries.get(category)
        if entry is None:
            return
        entry.used_bytes = max(0, entry.used_bytes - size_bytes)

    def get_usage(self, category: AssetCategory) -> BudgetEntry:
        """Get budget entry for a category."""
        entry = self._entries.get(category)
        if entry is None:
            raise KeyError(f"No budget set for {category}")
        return entry

    def get_total_usage(self) -> dict[AssetCategory, BudgetEntry]:
        """Get all budget entries."""
        return dict(self._entries)

    def is_over_budget(self, category: AssetCategory) -> bool:
        """Check if a category exceeds its budget."""
        entry = self._entries.get(category)
        if entry is None:
            return False
        return entry.used_bytes > entry.budget_bytes

    def get_pressure(self) -> float:
        """Overall memory pressure as ratio 0.0-1.0."""
        total_budget = sum(e.budget_bytes for e in self._entries.values())
        if total_budget == 0:
            return 0.0
        total_used = sum(e.used_bytes for e in self._entries.values())
        return min(1.0, total_used / total_budget)
