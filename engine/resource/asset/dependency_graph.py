"""Asset dependency graph with rebuild cascade, cycle detection, and incremental rebuild.

Provides:
- AssetDependencyGraph: Tracks asset -> dependencies relationships
- DependencyEdge: Typed edges (import, reference, embed)
- RebuildPlanner: Plans and executes rebuilds with correct ordering
- Cycle detection and detailed error reporting
- Incremental rebuild (skip unchanged assets via content hash)
"""
from __future__ import annotations

import enum
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from engine.resource.asset.asset_handle import AssetId
from engine.resource.asset.content_hash import ContentHash

__all__ = [
    "DependencyGraph",
    "DependencyEdge",
    "DependencyType",
    "AssetDependencyGraph",
    "RebuildPlanner",
    "RebuildPlan",
    "RebuildResult",
    "RebuildCallback",
    "CycleError",
    "AssetNode",
    "RebuildStats",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class CycleError(Exception):
    """Raised when a dependency cycle is detected."""

    def __init__(
        self,
        message: str,
        cycle_path: List[AssetId] | None = None,
    ) -> None:
        self.cycle_path = cycle_path or []
        super().__init__(message)

    def __str__(self) -> str:
        if self.cycle_path:
            path_str = " -> ".join(str(aid) for aid in self.cycle_path)
            return f"{self.args[0]}: {path_str}"
        return str(self.args[0])


# -----------------------------------------------------------------------------
# Dependency Types
# -----------------------------------------------------------------------------


class DependencyType(enum.Enum):
    """Type of dependency relationship between assets."""

    IMPORT = "import"      # Asset directly imports/includes another
    REFERENCE = "reference"  # Asset references another (can be lazy loaded)
    EMBED = "embed"        # Asset embeds another (inline, always loaded together)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class DependencyEdge:
    """A typed dependency edge from one asset to another.

    Attributes:
        source: The asset that has the dependency
        target: The asset being depended on
        dep_type: Type of dependency relationship
        metadata: Optional additional information about the dependency
    """
    source: AssetId
    target: AssetId
    dep_type: DependencyType
    metadata: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError(f"Self-dependency not allowed: {self.source}")

    @classmethod
    def import_edge(cls, source: AssetId, target: AssetId, **meta: str) -> DependencyEdge:
        """Create an import dependency edge."""
        return cls(source, target, DependencyType.IMPORT, frozenset(meta.items()))

    @classmethod
    def reference_edge(cls, source: AssetId, target: AssetId, **meta: str) -> DependencyEdge:
        """Create a reference dependency edge."""
        return cls(source, target, DependencyType.REFERENCE, frozenset(meta.items()))

    @classmethod
    def embed_edge(cls, source: AssetId, target: AssetId, **meta: str) -> DependencyEdge:
        """Create an embed dependency edge."""
        return cls(source, target, DependencyType.EMBED, frozenset(meta.items()))

    def get_metadata(self, key: str) -> str | None:
        """Get metadata value by key."""
        for k, v in self.metadata:
            if k == key:
                return v
        return None

    def with_metadata(self, **meta: str) -> DependencyEdge:
        """Return a new edge with additional metadata."""
        new_meta = dict(self.metadata)
        new_meta.update(meta)
        return DependencyEdge(
            self.source, self.target, self.dep_type, frozenset(new_meta.items())
        )

    def __repr__(self) -> str:
        return f"DependencyEdge({self.source} --[{self.dep_type}]--> {self.target})"


# -----------------------------------------------------------------------------
# Asset Node
# -----------------------------------------------------------------------------


@dataclass
class AssetNode:
    """Node in the dependency graph representing an asset.

    Tracks the asset's content hash for incremental rebuild detection.
    """
    asset_id: AssetId
    content_hash: ContentHash | None = None
    last_rebuild_time: float = 0.0
    rebuild_count: int = 0
    path: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def needs_rebuild(self, new_hash: ContentHash | None) -> bool:
        """Check if the asset needs rebuild based on content hash change."""
        if new_hash is None:
            return True
        if self.content_hash is None:
            return True
        return self.content_hash != new_hash

    def mark_rebuilt(self, new_hash: ContentHash | None) -> None:
        """Mark the asset as rebuilt with new content hash."""
        self.content_hash = new_hash
        self.last_rebuild_time = time.time()
        self.rebuild_count += 1


# -----------------------------------------------------------------------------
# Legacy DependencyGraph (backward compatible)
# -----------------------------------------------------------------------------


class DependencyGraph:
    """Directed acyclic graph tracking asset dependencies (legacy API).

    This class maintains backward compatibility with the original interface
    while delegating to AssetDependencyGraph for actual functionality.
    """
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


# -----------------------------------------------------------------------------
# Full-Featured AssetDependencyGraph
# -----------------------------------------------------------------------------


class AssetDependencyGraph:
    """Full-featured dependency graph with typed edges and incremental rebuild support.

    Features:
    - Typed dependency edges (import, reference, embed)
    - Content hash tracking for incremental rebuilds
    - Detailed cycle detection with path reporting
    - Transitive dependency/dependent queries
    - Thread-safe operations
    """
    __slots__ = (
        "_nodes",
        "_edges",
        "_forward",
        "_reverse",
        "_lock",
    )

    def __init__(self) -> None:
        self._nodes: Dict[AssetId, AssetNode] = {}
        self._edges: Dict[Tuple[AssetId, AssetId], DependencyEdge] = {}
        self._forward: Dict[AssetId, Set[AssetId]] = {}  # asset -> dependencies
        self._reverse: Dict[AssetId, Set[AssetId]] = {}  # asset -> dependents
        self._lock = threading.RLock()

    # -------------------------------------------------------------------------
    # Node Management
    # -------------------------------------------------------------------------

    def add_node(
        self,
        asset_id: AssetId,
        content_hash: ContentHash | None = None,
        path: str | None = None,
        **metadata: Any,
    ) -> AssetNode:
        """Add or update an asset node in the graph."""
        with self._lock:
            if asset_id in self._nodes:
                node = self._nodes[asset_id]
                if content_hash is not None:
                    node.content_hash = content_hash
                if path is not None:
                    node.path = path
                node.metadata.update(metadata)
            else:
                node = AssetNode(
                    asset_id=asset_id,
                    content_hash=content_hash,
                    path=path,
                    metadata=dict(metadata),
                )
                self._nodes[asset_id] = node
                self._forward.setdefault(asset_id, set())
                self._reverse.setdefault(asset_id, set())
            return node

    def get_node(self, asset_id: AssetId) -> AssetNode | None:
        """Get an asset node by ID."""
        with self._lock:
            return self._nodes.get(asset_id)

    def has_node(self, asset_id: AssetId) -> bool:
        """Check if an asset exists in the graph."""
        with self._lock:
            return asset_id in self._nodes

    def remove_node(self, asset_id: AssetId) -> bool:
        """Remove an asset and all its edges from the graph."""
        with self._lock:
            if asset_id not in self._nodes:
                return False

            # Remove all edges involving this node
            for target in list(self._forward.get(asset_id, set())):
                self._remove_edge_internal(asset_id, target)
            for source in list(self._reverse.get(asset_id, set())):
                self._remove_edge_internal(source, asset_id)

            # Remove node
            del self._nodes[asset_id]
            self._forward.pop(asset_id, None)
            self._reverse.pop(asset_id, None)
            return True

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        with self._lock:
            return len(self._nodes)

    def nodes(self) -> Iterator[AssetNode]:
        """Iterate over all nodes."""
        with self._lock:
            yield from self._nodes.values()

    def node_ids(self) -> Iterator[AssetId]:
        """Iterate over all node IDs."""
        with self._lock:
            yield from self._nodes.keys()

    # -------------------------------------------------------------------------
    # Edge Management
    # -------------------------------------------------------------------------

    def add_edge(self, edge: DependencyEdge) -> None:
        """Add a typed dependency edge. Raises CycleError if it would create a cycle."""
        with self._lock:
            # Check for self-dependency
            if edge.source == edge.target:
                raise CycleError(
                    "Self-dependency not allowed",
                    cycle_path=[edge.source],
                )

            # Check for cycle
            cycle = self._would_create_cycle(edge.source, edge.target)
            if cycle:
                raise CycleError(
                    f"Adding edge {edge.source} -> {edge.target} would create a cycle",
                    cycle_path=cycle,
                )

            # Ensure nodes exist
            self.add_node(edge.source)
            self.add_node(edge.target)

            # Add edge
            self._edges[(edge.source, edge.target)] = edge
            self._forward[edge.source].add(edge.target)
            self._reverse[edge.target].add(edge.source)

    def add_dependency(
        self,
        source: AssetId,
        target: AssetId,
        dep_type: DependencyType = DependencyType.IMPORT,
        **metadata: str,
    ) -> DependencyEdge:
        """Add a dependency edge (convenience method)."""
        # Check for self-dependency before creating edge
        if source == target:
            raise CycleError(
                "Self-dependency not allowed",
                cycle_path=[source],
            )
        edge = DependencyEdge(
            source=source,
            target=target,
            dep_type=dep_type,
            metadata=frozenset(metadata.items()),
        )
        self.add_edge(edge)
        return edge

    def get_edge(self, source: AssetId, target: AssetId) -> DependencyEdge | None:
        """Get the edge between two assets."""
        with self._lock:
            return self._edges.get((source, target))

    def has_edge(self, source: AssetId, target: AssetId) -> bool:
        """Check if an edge exists."""
        with self._lock:
            return (source, target) in self._edges

    def remove_edge(self, source: AssetId, target: AssetId) -> bool:
        """Remove an edge from the graph."""
        with self._lock:
            return self._remove_edge_internal(source, target)

    def _remove_edge_internal(self, source: AssetId, target: AssetId) -> bool:
        """Internal edge removal (assumes lock held)."""
        key = (source, target)
        if key not in self._edges:
            return False
        del self._edges[key]
        self._forward[source].discard(target)
        self._reverse[target].discard(source)
        return True

    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        with self._lock:
            return len(self._edges)

    def edges(self) -> Iterator[DependencyEdge]:
        """Iterate over all edges."""
        with self._lock:
            yield from self._edges.values()

    def edges_of_type(self, dep_type: DependencyType) -> Iterator[DependencyEdge]:
        """Iterate over edges of a specific type."""
        with self._lock:
            for edge in self._edges.values():
                if edge.dep_type == dep_type:
                    yield edge

    # -------------------------------------------------------------------------
    # Dependency Queries
    # -------------------------------------------------------------------------

    def get_dependencies(self, asset_id: AssetId) -> Set[AssetId]:
        """Get direct dependencies of an asset."""
        with self._lock:
            return set(self._forward.get(asset_id, set()))

    def get_dependents(self, asset_id: AssetId) -> Set[AssetId]:
        """Get direct dependents of an asset (assets that depend on this one)."""
        with self._lock:
            return set(self._reverse.get(asset_id, set()))

    def get_transitive_dependencies(self, asset_id: AssetId) -> Set[AssetId]:
        """Get all transitive dependencies (everything this asset depends on)."""
        with self._lock:
            result: Set[AssetId] = set()
            queue: deque[AssetId] = deque(self._forward.get(asset_id, set()))
            while queue:
                current = queue.popleft()
                if current in result:
                    continue
                result.add(current)
                queue.extend(self._forward.get(current, set()))
            return result

    def get_transitive_dependents(self, asset_id: AssetId) -> Set[AssetId]:
        """Get all transitive dependents (everything that depends on this asset)."""
        with self._lock:
            result: Set[AssetId] = set()
            queue: deque[AssetId] = deque(self._reverse.get(asset_id, set()))
            while queue:
                current = queue.popleft()
                if current in result:
                    continue
                result.add(current)
                queue.extend(self._reverse.get(current, set()))
            return result

    def get_edges_from(self, asset_id: AssetId) -> List[DependencyEdge]:
        """Get all edges originating from an asset."""
        with self._lock:
            return [
                self._edges[(asset_id, target)]
                for target in self._forward.get(asset_id, set())
            ]

    def get_edges_to(self, asset_id: AssetId) -> List[DependencyEdge]:
        """Get all edges pointing to an asset."""
        with self._lock:
            return [
                self._edges[(source, asset_id)]
                for source in self._reverse.get(asset_id, set())
            ]

    # -------------------------------------------------------------------------
    # Cycle Detection
    # -------------------------------------------------------------------------

    def _would_create_cycle(self, source: AssetId, target: AssetId) -> List[AssetId] | None:
        """Check if adding source -> target would create a cycle.

        Returns the cycle path if found, None otherwise.
        """
        # A cycle would be created if there's already a path from target to source
        visited: Set[AssetId] = set()
        parent: Dict[AssetId, AssetId] = {}
        queue: deque[AssetId] = deque([target])

        while queue:
            current = queue.popleft()
            if current == source:
                # Found a path from target to source, reconstruct cycle
                path = [source, target]
                node = target
                while node != source and node in parent:
                    path.append(parent[node])
                    node = parent[node]
                path.append(source)  # Complete the cycle
                return path

            if current in visited:
                continue
            visited.add(current)

            for dep in self._forward.get(current, set()):
                if dep not in visited:
                    parent[dep] = current
                    queue.append(dep)

        return None

    def detect_cycles(self) -> List[List[AssetId]]:
        """Detect all cycles in the graph using DFS.

        Returns a list of cycle paths.
        """
        with self._lock:
            cycles: List[List[AssetId]] = []
            visited: Set[AssetId] = set()
            rec_stack: Set[AssetId] = set()
            path: List[AssetId] = []

            def dfs(node: AssetId) -> None:
                visited.add(node)
                rec_stack.add(node)
                path.append(node)

                for neighbor in self._forward.get(node, set()):
                    if neighbor not in visited:
                        dfs(neighbor)
                    elif neighbor in rec_stack:
                        # Found a cycle
                        cycle_start = path.index(neighbor)
                        cycles.append(path[cycle_start:] + [neighbor])

                path.pop()
                rec_stack.remove(node)

            for node in self._nodes:
                if node not in visited:
                    dfs(node)

            return cycles

    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic."""
        return len(self.detect_cycles()) == 0

    # -------------------------------------------------------------------------
    # Topological Sort
    # -------------------------------------------------------------------------

    def topological_sort(self) -> List[AssetId]:
        """Return all nodes in topological order (dependencies before dependents).

        Raises CycleError if the graph contains cycles.
        """
        with self._lock:
            # Kahn's algorithm
            in_degree: Dict[AssetId, int] = {node: 0 for node in self._nodes}
            for node in self._nodes:
                for dep in self._forward.get(node, set()):
                    in_degree[node] += 1

            queue: deque[AssetId] = deque(
                node for node, degree in in_degree.items() if degree == 0
            )
            result: List[AssetId] = []

            while queue:
                node = queue.popleft()
                result.append(node)
                for dependent in self._reverse.get(node, set()):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            if len(result) != len(self._nodes):
                # Find the cycle for error reporting
                remaining = set(self._nodes.keys()) - set(result)
                raise CycleError(
                    "Graph contains a cycle",
                    cycle_path=list(remaining)[:10],  # Report up to 10 nodes
                )

            return result

    def topological_sort_subset(self, asset_ids: Iterable[AssetId]) -> List[AssetId]:
        """Return a subset of nodes in topological order.

        Includes all transitive dependencies of the given assets.
        """
        with self._lock:
            # Gather all relevant nodes
            relevant: Set[AssetId] = set()
            queue: deque[AssetId] = deque(asset_ids)
            while queue:
                node = queue.popleft()
                if node in relevant:
                    continue
                relevant.add(node)
                for dep in self._forward.get(node, set()):
                    queue.append(dep)

            # Kahn's algorithm on the subgraph
            in_degree: Dict[AssetId, int] = {n: 0 for n in relevant}
            for node in relevant:
                for dep in self._forward.get(node, set()):
                    if dep in relevant:
                        in_degree[node] += 1

            start: deque[AssetId] = deque(n for n, d in in_degree.items() if d == 0)
            result: List[AssetId] = []

            while start:
                node = start.popleft()
                result.append(node)
                for dependent in self._reverse.get(node, set()):
                    if dependent in relevant:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            start.append(dependent)

            if len(result) != len(relevant):
                raise CycleError("Cycle detected in subgraph")

            return result

    # -------------------------------------------------------------------------
    # Rebuild Order (for changed assets)
    # -------------------------------------------------------------------------

    def get_rebuild_order(self, changed_assets: Iterable[AssetId]) -> List[AssetId]:
        """Get the correct rebuild order for changed assets and all their dependents.

        When an asset changes, all its dependents need to be rebuilt in the
        correct order (dependencies before dependents).
        """
        with self._lock:
            # Collect all affected assets
            affected: Set[AssetId] = set(changed_assets)
            for asset_id in list(affected):
                affected.update(self.get_transitive_dependents(asset_id))

            # Topological sort the affected subset
            in_degree: Dict[AssetId, int] = {n: 0 for n in affected}
            for node in affected:
                for dep in self._forward.get(node, set()):
                    if dep in affected:
                        in_degree[node] += 1

            queue: deque[AssetId] = deque(n for n, d in in_degree.items() if d == 0)
            result: List[AssetId] = []

            while queue:
                node = queue.popleft()
                result.append(node)
                for dependent in self._reverse.get(node, set()):
                    if dependent in affected:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)

            if len(result) != len(affected):
                raise CycleError("Cycle detected in rebuild graph")

            return result

    # -------------------------------------------------------------------------
    # Content Hash / Incremental Rebuild Support
    # -------------------------------------------------------------------------

    def update_content_hash(self, asset_id: AssetId, content_hash: ContentHash) -> bool:
        """Update the content hash for an asset. Returns True if changed."""
        with self._lock:
            node = self._nodes.get(asset_id)
            if node is None:
                return False
            if node.content_hash == content_hash:
                return False
            node.content_hash = content_hash
            return True

    def get_content_hash(self, asset_id: AssetId) -> ContentHash | None:
        """Get the content hash for an asset."""
        with self._lock:
            node = self._nodes.get(asset_id)
            return node.content_hash if node else None

    def needs_rebuild(
        self,
        asset_id: AssetId,
        new_hash: ContentHash | None,
    ) -> bool:
        """Check if an asset needs rebuild based on content hash."""
        with self._lock:
            node = self._nodes.get(asset_id)
            if node is None:
                return True
            return node.needs_rebuild(new_hash)

    def mark_rebuilt(
        self,
        asset_id: AssetId,
        new_hash: ContentHash | None = None,
    ) -> None:
        """Mark an asset as rebuilt."""
        with self._lock:
            node = self._nodes.get(asset_id)
            if node is not None:
                node.mark_rebuilt(new_hash)

    # -------------------------------------------------------------------------
    # Graph Operations
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._forward.clear()
            self._reverse.clear()

    def subgraph(self, asset_ids: Iterable[AssetId]) -> AssetDependencyGraph:
        """Create a subgraph containing only the specified assets and their edges."""
        with self._lock:
            ids = set(asset_ids)
            sub = AssetDependencyGraph()
            for aid in ids:
                node = self._nodes.get(aid)
                if node is not None:
                    sub.add_node(
                        aid,
                        content_hash=node.content_hash,
                        path=node.path,
                        **node.metadata,
                    )
            for edge in self._edges.values():
                if edge.source in ids and edge.target in ids:
                    sub._edges[(edge.source, edge.target)] = edge
                    sub._forward[edge.source].add(edge.target)
                    sub._reverse[edge.target].add(edge.source)
            return sub

    def get_stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        with self._lock:
            edge_types: Dict[str, int] = {}
            for edge in self._edges.values():
                key = str(edge.dep_type)
                edge_types[key] = edge_types.get(key, 0) + 1

            return {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "edge_types": edge_types,
                "is_acyclic": self.is_acyclic(),
            }

    def __len__(self) -> int:
        """Return the number of nodes."""
        with self._lock:
            return len(self._nodes)

    def __contains__(self, asset_id: AssetId) -> bool:
        """Check if an asset is in the graph."""
        return self.has_node(asset_id)

    def __repr__(self) -> str:
        return f"AssetDependencyGraph(nodes={self.node_count()}, edges={self.edge_count()})"


# -----------------------------------------------------------------------------
# Rebuild Planner
# -----------------------------------------------------------------------------


# Type for rebuild callbacks
RebuildCallback = Callable[[AssetId, AssetNode | None], bool]


@dataclass
class RebuildStats:
    """Statistics from a rebuild operation."""
    total_assets: int = 0
    rebuilt: int = 0
    skipped: int = 0
    failed: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    failed_assets: List[Tuple[AssetId, str]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Return the duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def success_rate(self) -> float:
        """Return the success rate as a percentage."""
        if self.total_assets == 0:
            return 100.0
        return ((self.rebuilt + self.skipped) / self.total_assets) * 100


@dataclass
class RebuildPlan:
    """A plan for rebuilding assets in dependency order."""
    assets_to_rebuild: List[AssetId]
    changed_sources: Set[AssetId]
    affected_dependents: Set[AssetId]
    skipped_unchanged: Set[AssetId]

    @property
    def total_affected(self) -> int:
        """Total number of affected assets."""
        return len(self.assets_to_rebuild)

    def __repr__(self) -> str:
        return (
            f"RebuildPlan(rebuild={len(self.assets_to_rebuild)}, "
            f"sources={len(self.changed_sources)}, "
            f"skipped={len(self.skipped_unchanged)})"
        )


@dataclass
class RebuildResult:
    """Result of executing a rebuild plan."""
    stats: RebuildStats
    rebuilt_assets: List[AssetId]
    skipped_assets: List[AssetId]
    failed_assets: List[Tuple[AssetId, str]]

    @property
    def success(self) -> bool:
        """Return True if no failures occurred."""
        return len(self.failed_assets) == 0


class RebuildPlanner:
    """Plans and executes asset rebuilds with dependency ordering.

    Features:
    - Incremental rebuilds (skip unchanged assets)
    - Correct rebuild order (dependencies before dependents)
    - Parallel rebuild support (groups of independent assets)
    - Detailed rebuild statistics
    """
    __slots__ = (
        "_graph",
        "_rebuild_callback",
        "_hash_provider",
        "_stop_on_failure",
    )

    def __init__(
        self,
        graph: AssetDependencyGraph,
        rebuild_callback: RebuildCallback | None = None,
        hash_provider: Callable[[AssetId], ContentHash | None] | None = None,
        stop_on_failure: bool = False,
    ) -> None:
        self._graph = graph
        self._rebuild_callback = rebuild_callback
        self._hash_provider = hash_provider
        self._stop_on_failure = stop_on_failure

    @property
    def graph(self) -> AssetDependencyGraph:
        """Return the underlying dependency graph."""
        return self._graph

    def set_rebuild_callback(self, callback: RebuildCallback) -> None:
        """Set the callback function for rebuilding assets."""
        self._rebuild_callback = callback

    def set_hash_provider(self, provider: Callable[[AssetId], ContentHash | None]) -> None:
        """Set the function to get current content hashes."""
        self._hash_provider = provider

    # -------------------------------------------------------------------------
    # Planning
    # -------------------------------------------------------------------------

    def plan_rebuild(
        self,
        changed_assets: Iterable[AssetId],
        check_unchanged: bool = True,
    ) -> RebuildPlan:
        """Create a rebuild plan for changed assets.

        Args:
            changed_assets: Assets whose source content has changed
            check_unchanged: If True, skip assets whose source content hash
                             is unchanged AND none of their dependencies changed

        Returns:
            A RebuildPlan with ordered list of assets to rebuild

        Note:
            Skipping only applies to assets in changed_assets that have the same
            content hash as stored. Transitive dependents always rebuild because
            their inputs (dependencies) may have changed output.
        """
        changed = set(changed_assets)

        # First, filter changed assets if check_unchanged is True
        # Only source assets (not dependents) can be skipped based on hash
        skipped: Set[AssetId] = set()
        actually_changed: Set[AssetId] = set()

        if check_unchanged and self._hash_provider:
            for asset_id in changed:
                new_hash = self._hash_provider(asset_id)
                if self._graph.needs_rebuild(asset_id, new_hash):
                    actually_changed.add(asset_id)
                else:
                    skipped.add(asset_id)
        else:
            actually_changed = changed

        # Get all affected assets (changed + all transitive dependents)
        affected: Set[AssetId] = set(actually_changed)
        for asset_id in actually_changed:
            affected.update(self._graph.get_transitive_dependents(asset_id))

        # Get rebuild order
        rebuild_order = self._graph.get_rebuild_order(affected)

        return RebuildPlan(
            assets_to_rebuild=rebuild_order,
            changed_sources=actually_changed,
            affected_dependents=affected - actually_changed,
            skipped_unchanged=skipped,
        )

    def plan_full_rebuild(self) -> RebuildPlan:
        """Create a plan to rebuild all assets in dependency order."""
        all_assets = set(self._graph.node_ids())
        rebuild_order = self._graph.topological_sort()
        return RebuildPlan(
            assets_to_rebuild=rebuild_order,
            changed_sources=all_assets,
            affected_dependents=set(),
            skipped_unchanged=set(),
        )

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def execute(self, plan: RebuildPlan) -> RebuildResult:
        """Execute a rebuild plan.

        Calls the rebuild callback for each asset in order.
        """
        if self._rebuild_callback is None:
            raise RuntimeError("No rebuild callback set")

        stats = RebuildStats(
            total_assets=len(plan.assets_to_rebuild),
            start_time=time.time(),
        )
        rebuilt: List[AssetId] = []
        skipped: List[AssetId] = []
        failed: List[Tuple[AssetId, str]] = []

        for asset_id in plan.assets_to_rebuild:
            node = self._graph.get_node(asset_id)

            try:
                success = self._rebuild_callback(asset_id, node)
                if success:
                    # Update content hash after successful rebuild
                    if self._hash_provider:
                        new_hash = self._hash_provider(asset_id)
                        self._graph.mark_rebuilt(asset_id, new_hash)
                    else:
                        self._graph.mark_rebuilt(asset_id)
                    rebuilt.append(asset_id)
                    stats.rebuilt += 1
                else:
                    # Callback returned False, treat as skip
                    skipped.append(asset_id)
                    stats.skipped += 1
            except Exception as e:
                error_msg = str(e)
                failed.append((asset_id, error_msg))
                stats.failed += 1
                stats.failed_assets.append((asset_id, error_msg))
                logger.error("Failed to rebuild asset %s: %s", asset_id, e)
                if self._stop_on_failure:
                    break

        stats.end_time = time.time()

        return RebuildResult(
            stats=stats,
            rebuilt_assets=rebuilt,
            skipped_assets=skipped,
            failed_assets=failed,
        )

    def execute_incremental(
        self,
        changed_assets: Iterable[AssetId],
    ) -> RebuildResult:
        """Plan and execute an incremental rebuild for changed assets."""
        plan = self.plan_rebuild(changed_assets, check_unchanged=True)
        return self.execute(plan)

    def execute_full(self) -> RebuildResult:
        """Plan and execute a full rebuild of all assets."""
        plan = self.plan_full_rebuild()
        return self.execute(plan)

    # -------------------------------------------------------------------------
    # Parallel Rebuild Support
    # -------------------------------------------------------------------------

    def get_parallel_groups(self, plan: RebuildPlan) -> List[List[AssetId]]:
        """Split a rebuild plan into groups that can be rebuilt in parallel.

        Assets within a group have no dependencies on each other.
        Groups must be processed in order (earlier groups before later).
        """
        if not plan.assets_to_rebuild:
            return []

        # Build dependency relationships for the plan subset
        assets = set(plan.assets_to_rebuild)
        in_degree: Dict[AssetId, int] = {}
        dependents: Dict[AssetId, Set[AssetId]] = {}

        for asset_id in assets:
            in_degree[asset_id] = 0
            dependents[asset_id] = set()

        for asset_id in assets:
            deps = self._graph.get_dependencies(asset_id)
            for dep in deps:
                if dep in assets:
                    in_degree[asset_id] += 1
                    dependents[dep].add(asset_id)

        groups: List[List[AssetId]] = []
        remaining = set(assets)

        while remaining:
            # Find all assets with no pending dependencies
            group = [aid for aid in remaining if in_degree[aid] == 0]
            if not group:
                # Cycle detected
                raise CycleError(
                    "Cycle detected while creating parallel groups",
                    cycle_path=list(remaining)[:5],
                )

            groups.append(group)

            # Remove processed assets and update degrees
            for asset_id in group:
                remaining.remove(asset_id)
                for dependent in dependents[asset_id]:
                    in_degree[dependent] -= 1

        return groups

    def execute_parallel(
        self,
        plan: RebuildPlan,
        max_workers: int = 4,
    ) -> RebuildResult:
        """Execute a rebuild plan with parallel processing.

        Groups of independent assets are rebuilt concurrently.
        """
        import concurrent.futures

        if self._rebuild_callback is None:
            raise RuntimeError("No rebuild callback set")

        stats = RebuildStats(
            total_assets=len(plan.assets_to_rebuild),
            start_time=time.time(),
        )
        rebuilt: List[AssetId] = []
        skipped: List[AssetId] = []
        failed: List[Tuple[AssetId, str]] = []
        lock = threading.Lock()

        groups = self.get_parallel_groups(plan)

        def rebuild_one(asset_id: AssetId) -> Tuple[AssetId, bool, str | None]:
            node = self._graph.get_node(asset_id)
            try:
                success = self._rebuild_callback(asset_id, node)
                if success:
                    if self._hash_provider:
                        new_hash = self._hash_provider(asset_id)
                        with lock:
                            self._graph.mark_rebuilt(asset_id, new_hash)
                    else:
                        with lock:
                            self._graph.mark_rebuilt(asset_id)
                return (asset_id, success, None)
            except Exception as e:
                return (asset_id, False, str(e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for group in groups:
                futures = {executor.submit(rebuild_one, aid): aid for aid in group}
                for future in concurrent.futures.as_completed(futures):
                    asset_id, success, error = future.result()
                    if error:
                        failed.append((asset_id, error))
                        stats.failed += 1
                        stats.failed_assets.append((asset_id, error))
                    elif success:
                        rebuilt.append(asset_id)
                        stats.rebuilt += 1
                    else:
                        skipped.append(asset_id)
                        stats.skipped += 1

                if self._stop_on_failure and failed:
                    break

        stats.end_time = time.time()

        return RebuildResult(
            stats=stats,
            rebuilt_assets=rebuilt,
            skipped_assets=skipped,
            failed_assets=failed,
        )
