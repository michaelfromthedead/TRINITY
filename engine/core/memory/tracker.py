"""Memory tracking — allocation stats, leak detection, per-tag budgeting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from engine.core.memory.allocator import AllocationInfo, MemoryTag

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryStats:
    """Aggregate statistics for a memory tag (or total)."""
    allocated: int = 0
    freed: int = 0
    peak: int = 0
    current: int = 0


class MemoryTracker:
    """Tracks live allocations and computes per-tag statistics."""

    def __init__(self) -> None:
        self._live: Dict[int, AllocationInfo] = {}
        self._stats: Dict[MemoryTag, MemoryStats] = {}
        self._total_current: int = 0
        self._total_peak: int = 0

    def track_allocation(self, info: AllocationInfo) -> None:
        self._live[info.offset] = info
        stats = self._stats_for(info.tag)
        stats.allocated += info.size
        stats.current += info.size
        if stats.current > stats.peak:
            stats.peak = stats.current
        self._total_current += info.size
        if self._total_current > self._total_peak:
            self._total_peak = self._total_current
        logger.debug("track alloc offset=%d size=%d tag=%s", info.offset, info.size, info.tag.name)

    def track_free(self, offset: int) -> None:
        info = self._live.pop(offset, None)
        if info is None:
            logger.warning("track_free: unknown offset %d", offset)
            return
        stats = self._stats_for(info.tag)
        stats.freed += info.size
        stats.current -= info.size
        self._total_current -= info.size
        logger.debug("track free offset=%d size=%d tag=%s", offset, info.size, info.tag.name)

    def get_stats(self, tag: MemoryTag) -> MemoryStats:
        return self._stats_for(tag)

    def get_total_stats(self) -> MemoryStats:
        total = MemoryStats()
        for s in self._stats.values():
            total.allocated += s.allocated
            total.freed += s.freed
            total.current += s.current
        total.peak = self._total_peak
        return total

    def get_live_allocations(self) -> List[AllocationInfo]:
        return list(self._live.values())

    # -- Internal -------------------------------------------------------------

    def _stats_for(self, tag: MemoryTag) -> MemoryStats:
        if tag not in self._stats:
            self._stats[tag] = MemoryStats()
        return self._stats[tag]
