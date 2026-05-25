"""Eviction policies and manager for memory-resident assets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass
class EvictionCandidate:
    """An asset that may be evicted."""

    __slots__ = ("asset_id", "size_bytes", "last_access_time", "access_count", "priority")

    asset_id: int
    size_bytes: int
    last_access_time: float
    access_count: int
    priority: int

    def __init__(
        self,
        asset_id: int,
        size_bytes: int,
        last_access_time: float = 0.0,
        access_count: int = 0,
        priority: int = 0,
    ) -> None:
        self.asset_id = asset_id
        self.size_bytes = size_bytes
        self.last_access_time = last_access_time
        self.access_count = access_count
        self.priority = priority


class EvictionPolicy(ABC):
    """Abstract base for eviction policies."""

    @abstractmethod
    def select_for_eviction(
        self, candidates: Sequence[EvictionCandidate], bytes_needed: int
    ) -> list[int]:
        """Select asset_ids to evict to free at least bytes_needed."""
        ...


class LRUEviction(EvictionPolicy):
    """Evict least recently used assets first."""

    def select_for_eviction(
        self, candidates: Sequence[EvictionCandidate], bytes_needed: int
    ) -> list[int]:
        sorted_candidates = sorted(candidates, key=lambda c: c.last_access_time)
        return _collect_until(sorted_candidates, bytes_needed)


class LFUEviction(EvictionPolicy):
    """Evict least frequently used assets first."""

    def select_for_eviction(
        self, candidates: Sequence[EvictionCandidate], bytes_needed: int
    ) -> list[int]:
        sorted_candidates = sorted(candidates, key=lambda c: c.access_count)
        return _collect_until(sorted_candidates, bytes_needed)


class SizeEviction(EvictionPolicy):
    """Evict largest assets first."""

    def select_for_eviction(
        self, candidates: Sequence[EvictionCandidate], bytes_needed: int
    ) -> list[int]:
        sorted_candidates = sorted(candidates, key=lambda c: -c.size_bytes)
        return _collect_until(sorted_candidates, bytes_needed)


class PriorityEviction(EvictionPolicy):
    """Evict lowest priority assets first."""

    def select_for_eviction(
        self, candidates: Sequence[EvictionCandidate], bytes_needed: int
    ) -> list[int]:
        sorted_candidates = sorted(candidates, key=lambda c: c.priority)
        return _collect_until(sorted_candidates, bytes_needed)


def _collect_until(
    sorted_candidates: list[EvictionCandidate], bytes_needed: int
) -> list[int]:
    """Collect asset_ids until bytes_needed is satisfied."""
    result: list[int] = []
    freed = 0
    for c in sorted_candidates:
        if freed >= bytes_needed:
            break
        result.append(c.asset_id)
        freed += c.size_bytes
    return result


class EvictionManager:
    """Manages eviction candidates and runs eviction via a policy."""

    __slots__ = ("_policy", "_candidates")

    def __init__(self, policy: EvictionPolicy) -> None:
        self._policy = policy
        self._candidates: dict[int, EvictionCandidate] = {}

    @property
    def policy(self) -> EvictionPolicy:
        return self._policy

    def add_candidate(self, candidate: EvictionCandidate) -> None:
        self._candidates[candidate.asset_id] = candidate

    def remove_candidate(self, asset_id: int) -> None:
        self._candidates.pop(asset_id, None)

    def run_eviction(self, bytes_needed: int) -> list[int]:
        """Run eviction and return list of evicted asset_ids."""
        if not self._candidates:
            return []
        evicted = self._policy.select_for_eviction(
            list(self._candidates.values()), bytes_needed
        )
        for aid in evicted:
            self._candidates.pop(aid, None)
        return evicted

    def candidate_count(self) -> int:
        return len(self._candidates)
