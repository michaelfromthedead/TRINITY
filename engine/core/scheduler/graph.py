"""Dependency graph with topological sort (Kahn's algorithm) and parallel group detection."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class CycleDetectedError(Exception):
    """Raised when a cycle is found in the dependency graph."""


@dataclass
class SystemGraph:
    """Directed acyclic graph of system dependencies.

    Nodes are system IDs (ints). An edge from A -> B means A must run before B.
    """
    _nodes: Set[int] = field(default_factory=set)
    _edges: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    _reverse: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))

    def add_node(self, node: int) -> None:
        self._nodes.add(node)

    def add_edge(self, from_id: int, to_id: int) -> None:
        """Add dependency: from_id must run before to_id."""
        self._nodes.add(from_id)
        self._nodes.add(to_id)
        self._edges[from_id].add(to_id)
        self._reverse[to_id].add(from_id)

    def topological_sort(self) -> List[int]:
        """Kahn's algorithm. Raises CycleDetectedError if cycle exists."""
        in_degree: Dict[int, int] = {n: 0 for n in self._nodes}
        for node in self._nodes:
            for succ in self._edges.get(node, set()):
                in_degree[succ] = in_degree.get(succ, 0) + 1

        queue = deque(sorted(n for n in self._nodes if in_degree[n] == 0))
        result: List[int] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for succ in sorted(self._edges.get(node, set())):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(result) != len(self._nodes):
            raise CycleDetectedError(
                f"Cycle detected: sorted {len(result)} of {len(self._nodes)} nodes"
            )
        return result

    def detect_cycles(self) -> None:
        """Raises CycleDetectedError if cycle found."""
        self.topological_sort()

    def get_parallel_groups(self) -> List[List[int]]:
        """Return groups of nodes that can run concurrently (same topological level).

        Nodes at the same level have no dependencies on each other.
        """
        if not self._nodes:
            return []

        in_degree: Dict[int, int] = {n: 0 for n in self._nodes}
        for node in self._nodes:
            for succ in self._edges.get(node, set()):
                in_degree[succ] = in_degree.get(succ, 0) + 1

        queue = deque(sorted(n for n in self._nodes if in_degree[n] == 0))
        groups: List[List[int]] = []

        while queue:
            level = sorted(queue)
            groups.append(level)
            next_queue: deque[int] = deque()
            for node in level:
                for succ in sorted(self._edges.get(node, set())):
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        next_queue.append(succ)
            queue = next_queue

        total = sum(len(g) for g in groups)
        if total != len(self._nodes):
            raise CycleDetectedError("Cycle detected during parallel group computation")
        return groups
