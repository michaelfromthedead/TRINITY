"""DAG of asset dependencies with topological sort and cycle detection."""
from __future__ import annotations

from collections import deque

from engine.resource.asset.asset_handle import AssetId

__all__ = ["DependencyGraph"]


class DependencyGraph:
    """Directed acyclic graph tracking asset dependencies."""
    __slots__ = ("_deps", "_rdeps")

    def __init__(self) -> None:
        # asset_id -> set of ids it depends on
        self._deps: dict[AssetId, set[AssetId]] = {}
        # asset_id -> set of ids that depend on it (reverse)
        self._rdeps: dict[AssetId, set[AssetId]] = {}

    def add_dependency(self, asset_id: AssetId, depends_on_id: AssetId) -> None:
        """Declare that *asset_id* depends on *depends_on_id*. Raises ValueError on cycle."""
        # Check for cycle: would adding this edge create a path from depends_on_id back to asset_id?
        if asset_id == depends_on_id:
            raise ValueError(f"Self-dependency: {asset_id}")
        if self._has_path(depends_on_id, asset_id):
            raise ValueError(
                f"Cycle detected: adding {asset_id} -> {depends_on_id} "
                f"would create a cycle"
            )
        self._deps.setdefault(asset_id, set()).add(depends_on_id)
        self._rdeps.setdefault(depends_on_id, set()).add(asset_id)
        # Ensure nodes exist in both maps
        self._deps.setdefault(depends_on_id, set())
        self._rdeps.setdefault(asset_id, set())

    def _has_path(self, from_id: AssetId, to_id: AssetId) -> bool:
        """BFS to check if there's a path from from_id to to_id following dependency edges."""
        visited: set[AssetId] = set()
        queue: deque[AssetId] = deque()
        # from_id depends on ... follow deps
        queue.append(from_id)
        while queue:
            current = queue.popleft()
            if current == to_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            for dep in self._deps.get(current, ()):
                queue.append(dep)
        return False

    def remove(self, asset_id: AssetId) -> None:
        """Remove an asset and all its dependency edges."""
        # Remove forward edges
        for dep in self._deps.pop(asset_id, set()):
            rdeps = self._rdeps.get(dep)
            if rdeps:
                rdeps.discard(asset_id)
        # Remove reverse edges
        for dependent in self._rdeps.pop(asset_id, set()):
            deps = self._deps.get(dependent)
            if deps:
                deps.discard(asset_id)

    def get_dependents(self, asset_id: AssetId) -> set[AssetId]:
        """Return the set of asset ids that directly depend on *asset_id*."""
        return set(self._rdeps.get(asset_id, ()))

    def get_load_order(self, asset_ids: list[AssetId]) -> list[AssetId]:
        """Return a topological ordering: dependencies before dependents.

        Only includes *asset_ids* and their transitive dependencies.
        """
        # Gather all relevant nodes
        relevant: set[AssetId] = set()
        queue: deque[AssetId] = deque(asset_ids)
        while queue:
            node = queue.popleft()
            if node in relevant:
                continue
            relevant.add(node)
            for dep in self._deps.get(node, ()):
                queue.append(dep)

        # Kahn's algorithm on the subgraph
        in_degree: dict[AssetId, int] = {n: 0 for n in relevant}
        for node in relevant:
            for dep in self._deps.get(node, ()):
                if dep in relevant:
                    in_degree[node] += 1  # node depends on dep -> in-degree of node

        start: deque[AssetId] = deque(n for n, d in in_degree.items() if d == 0)
        result: list[AssetId] = []
        while start:
            node = start.popleft()
            result.append(node)
            for dependent in self._rdeps.get(node, ()):
                if dependent in relevant:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        start.append(dependent)

        if len(result) != len(relevant):
            raise ValueError("Cycle detected in dependency graph")
        return result
