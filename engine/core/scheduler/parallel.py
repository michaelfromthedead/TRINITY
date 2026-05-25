"""Parallel dispatch with conflict detection based on component access declarations."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SystemId = int


@dataclass
class SystemAccess:
    """Declares which component types a system reads and writes."""
    reads: Set[type] = field(default_factory=set)
    writes: Set[type] = field(default_factory=set)


def can_run_parallel(a: SystemAccess, b: SystemAccess) -> bool:
    """Two systems can run in parallel if neither writes to components the other accesses.

    read/read   -> OK
    read/write  -> CONFLICT
    write/write -> CONFLICT
    """
    # a.writes vs b.reads or b.writes
    if a.writes & (b.reads | b.writes):
        return False
    # b.writes vs a.reads
    if b.writes & a.reads:
        return False
    return True


def compute_parallel_groups(
    system_ids: List[SystemId],
    access_map: Dict[SystemId, SystemAccess],
) -> List[List[SystemId]]:
    """Partition an ordered list of systems into groups that can run concurrently.

    Systems within a group have no read/write conflicts. Order between groups
    is preserved from the input ordering.
    """
    groups: List[List[SystemId]] = []
    for sid in system_ids:
        access = access_map.get(sid, SystemAccess())
        placed = False
        # Try to add to the last group
        if groups:
            conflict = False
            for existing_id in groups[-1]:
                existing_access = access_map.get(existing_id, SystemAccess())
                if not can_run_parallel(access, existing_access):
                    conflict = True
                    break
            if not conflict:
                groups[-1].append(sid)
                placed = True
        if not placed:
            groups.append([sid])
    return groups


class ParallelDispatcher:
    """Executes non-conflicting systems concurrently using ThreadPoolExecutor."""

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def shutdown(self) -> None:
        """Shut down the reusable executor."""
        self._executor.shutdown(wait=False)

    def dispatch(
        self,
        groups: List[List[SystemId]],
        run_system: Callable[[SystemId], None],
    ) -> None:
        """Execute system groups. Systems within a group run in parallel;
        groups run sequentially."""
        for group in groups:
            if len(group) == 1:
                run_system(group[0])
            else:
                futures = {self._executor.submit(run_system, sid): sid for sid in group}
                try:
                    for future in as_completed(futures):
                        exc = future.exception()
                        if exc is not None:
                            # Cancel remaining futures before re-raising
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            logger.error(
                                "System %s raised: %s", futures[future], exc
                            )
                            raise exc
                except Exception:
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    raise
