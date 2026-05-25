"""Manages which assets are memory-resident."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from collections.abc import Callable

from engine.resource.memory.budget_manager import AssetCategory, BudgetManager
from engine.resource.memory.eviction import (
    EvictionCandidate,
    EvictionManager,
    EvictionPolicy,
)


class ResidencyState(Enum):
    """Lifecycle states for asset residency."""

    NON_RESIDENT = auto()
    LOADING = auto()
    RESIDENT = auto()
    EVICTING = auto()


@dataclass
class ResidencyInfo:
    """Tracks residency information for a single asset."""

    __slots__ = ("asset_id", "state", "size_bytes", "last_access", "priority")

    asset_id: int
    state: ResidencyState
    size_bytes: int
    last_access: float
    priority: int

    def __init__(
        self,
        asset_id: int,
        state: ResidencyState = ResidencyState.NON_RESIDENT,
        size_bytes: int = 0,
        last_access: float = 0.0,
        priority: int = 0,
    ) -> None:
        self.asset_id = asset_id
        self.state = state
        self.size_bytes = size_bytes
        self.last_access = last_access
        self.priority = priority


class ResidencyManager:
    """Manages asset residency with budget and eviction integration."""

    __slots__ = (
        "_assets",
        "_budget_manager",
        "_eviction_manager",
        "_category",
        "_time_fn",
    )

    def __init__(
        self,
        budget_manager: BudgetManager,
        eviction_policy: EvictionPolicy,
        category: AssetCategory = AssetCategory.OTHER,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._assets: dict[int, ResidencyInfo] = {}
        self._budget_manager = budget_manager
        self._eviction_manager = EvictionManager(eviction_policy)
        self._category = category
        self._time_fn = time_fn or time.monotonic

    def request_residency(
        self, asset_id: int, size_bytes: int, priority: int = 0
    ) -> bool:
        """Request an asset become resident. Returns False if budget disallows."""
        now = self._time_fn()
        if asset_id in self._assets:
            info = self._assets[asset_id]
            if info.state == ResidencyState.RESIDENT:
                info.last_access = now
                return True
            if info.state in (ResidencyState.LOADING, ResidencyState.EVICTING):
                return False

        if not self._budget_manager.allocate(self._category, size_bytes):
            return False

        info = ResidencyInfo(
            asset_id=asset_id,
            state=ResidencyState.RESIDENT,
            size_bytes=size_bytes,
            last_access=now,
            priority=priority,
        )
        self._assets[asset_id] = info
        self._eviction_manager.add_candidate(
            EvictionCandidate(
                asset_id=asset_id,
                size_bytes=size_bytes,
                last_access_time=now,
                access_count=1,
                priority=priority,
            )
        )
        return True

    def release_residency(self, asset_id: int) -> None:
        """Release an asset from residency."""
        info = self._assets.pop(asset_id, None)
        if info is None:
            return
        info.state = ResidencyState.NON_RESIDENT
        self._budget_manager.free(self._category, info.size_bytes)
        self._eviction_manager.remove_candidate(asset_id)

    def touch(self, asset_id: int) -> None:
        """Update last access time for an asset."""
        info = self._assets.get(asset_id)
        if info is not None:
            info.last_access = self._time_fn()

    def get_state(self, asset_id: int) -> ResidencyState:
        """Get residency state. Returns NON_RESIDENT if unknown."""
        info = self._assets.get(asset_id)
        if info is None:
            return ResidencyState.NON_RESIDENT
        return info.state

    def get_resident_count(self) -> int:
        """Number of currently resident assets."""
        return sum(
            1
            for info in self._assets.values()
            if info.state == ResidencyState.RESIDENT
        )

    def get_resident_bytes(self) -> int:
        """Total bytes of resident assets."""
        return sum(
            info.size_bytes
            for info in self._assets.values()
            if info.state == ResidencyState.RESIDENT
        )

    def update(self) -> list[int]:
        """Run eviction if over budget. Returns list of evicted asset_ids."""
        if not self._budget_manager.is_over_budget(self._category):
            return []
        entry = self._budget_manager.get_usage(self._category)
        overage = entry.used_bytes - entry.budget_bytes
        evicted_ids = self._eviction_manager.run_eviction(overage)
        for aid in evicted_ids:
            info = self._assets.pop(aid, None)
            if info is not None:
                self._budget_manager.free(self._category, info.size_bytes)
        return evicted_ids
