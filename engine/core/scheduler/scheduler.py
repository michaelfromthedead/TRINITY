"""SystemScheduler: register systems, build execution graph, run by phase."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .graph import SystemGraph
from .parallel import ParallelDispatcher, SystemAccess, compute_parallel_groups
from .phases import DEFAULT_PHASE_ORDER, Phase, PhaseGroup

logger = logging.getLogger(__name__)

SystemId = int


@dataclass
class _SystemEntry:
    """Internal record for a registered system."""
    id: int
    system: Any  # callable or object with update()
    phase: Phase
    run_if: Optional[Callable[[], bool]] = None
    access: SystemAccess = field(default_factory=SystemAccess)


class SystemScheduler:
    """Phase-based system scheduler with dependency ordering and parallel dispatch."""

    def __init__(self, parallel_dispatch: bool = True, max_workers: Optional[int] = None) -> None:
        self._next_id: int = 0
        self._systems: Dict[SystemId, _SystemEntry] = {}
        self._graph = SystemGraph()
        self._phase_order: List[Phase] = list(DEFAULT_PHASE_ORDER)
        self._parallel_dispatch = parallel_dispatch
        self._dispatcher = ParallelDispatcher(max_workers=max_workers)
        # Cached sorted order per phase (built on first run or after invalidation)
        self._sorted_cache: Optional[Dict[Phase, List[SystemId]]] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_system(
        self,
        system: Any,
        phase: Union[str, Phase] = Phase.UPDATE,
        run_if: Optional[Callable[[], bool]] = None,
        access: Optional[SystemAccess] = None,
    ) -> SystemId:
        """Register a system callable or object. Returns a unique SystemId."""
        sid = self._next_id
        self._next_id += 1
        if isinstance(phase, str):
            phase = Phase[phase.upper()]
        entry = _SystemEntry(
            id=sid,
            system=system,
            phase=phase,
            run_if=run_if,
            access=access or SystemAccess(),
        )
        self._systems[sid] = entry
        self._graph.add_node(sid)
        self._sorted_cache = None
        return sid

    def add_dependency(self, from_id: SystemId, to_id: SystemId) -> None:
        """from_id must run before to_id."""
        self._graph.add_edge(from_id, to_id)
        self._sorted_cache = None

    def set_phase(self, system_id: SystemId, phase: Union[str, Phase]) -> None:
        if isinstance(phase, str):
            phase = Phase[phase.upper()]
        self._systems[system_id].phase = phase
        self._sorted_cache = None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, world: Any, delta_time: float) -> None:
        """Execute all phases in order."""
        for phase in self._phase_order:
            self.run_phase(world, phase, delta_time)

    def run_phase(self, world: Any, phase: Union[str, Phase], delta_time: float) -> None:
        """Execute a single phase."""
        if isinstance(phase, str):
            phase = Phase[phase.upper()]
        sorted_ids = self._get_sorted_for_phase(phase)
        if not sorted_ids:
            return

        if self._parallel_dispatch:
            access_map = {sid: self._systems[sid].access for sid in sorted_ids}
            groups = compute_parallel_groups(sorted_ids, access_map)
            self._dispatcher.dispatch(
                groups, lambda sid: self._invoke(sid, world, delta_time)
            )
        else:
            for sid in sorted_ids:
                self._invoke(sid, world, delta_time)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_sorted(self) -> Dict[Phase, List[SystemId]]:
        """Build topologically sorted system lists per phase."""
        global_order = self._graph.topological_sort()
        order_index = {sid: i for i, sid in enumerate(global_order)}
        per_phase: Dict[Phase, List[SystemId]] = {p: [] for p in Phase}
        for sid, entry in self._systems.items():
            per_phase[entry.phase].append(sid)
        for phase in per_phase:
            per_phase[phase].sort(key=lambda s: order_index.get(s, 0))
        return per_phase

    def _get_sorted_for_phase(self, phase: Phase) -> List[SystemId]:
        if self._sorted_cache is None:
            self._sorted_cache = self._build_sorted()
        return self._sorted_cache.get(phase, [])

    def _invoke(self, system_id: SystemId, world: Any, delta_time: float) -> None:
        entry = self._systems[system_id]
        if entry.run_if is not None and not entry.run_if():
            return
        system = entry.system
        if callable(system) and not hasattr(system, "update"):
            system(world, delta_time)
        elif hasattr(system, "update"):
            system.update(world, delta_time)
        else:
            raise TypeError(f"System {system} is not callable and has no update() method")
