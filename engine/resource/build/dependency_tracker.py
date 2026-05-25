"""Build dependency tracker — tracks file dependencies for incremental builds."""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class FileRecord:
    """Record of a tracked file."""

    path: str
    mtime: float
    content_hash: str
    dependencies: set[str] = field(default_factory=set)


class BuildDependencyTracker:
    """Tracks file dependencies and determines dirty files for incremental builds."""

    __slots__ = ("_records",)

    def __init__(self) -> None:
        self._records: dict[str, FileRecord] = {}

    @staticmethod
    def _compute_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def record_file(
        self,
        path: str,
        mtime: float,
        content: bytes,
        dependencies: set[str] | None = None,
    ) -> FileRecord:
        """Record or update a file entry."""
        record = FileRecord(
            path=path,
            mtime=mtime,
            content_hash=self._compute_hash(content),
            dependencies=dependencies or set(),
        )
        self._records[path] = record
        return record

    def is_dirty(self, path: str, current_mtime: float, current_content: bytes) -> bool:
        """Check if a file has changed since last recorded."""
        record = self._records.get(path)
        if record is None:
            return True
        if current_mtime != record.mtime:
            current_hash = self._compute_hash(current_content)
            return current_hash != record.content_hash
        return False

    def get_dirty_files(
        self,
        file_states: dict[str, tuple[float, bytes]],
    ) -> list[str]:
        """Return paths that are dirty given current states {path: (mtime, content)}."""
        dirty: list[str] = []
        for path, (mtime, content) in file_states.items():
            if self.is_dirty(path, mtime, content):
                dirty.append(path)
        return dirty

    def get_build_order(self, dirty: list[str]) -> list[str]:
        """Return a topological build order for the given dirty files."""
        # Build adjacency from dependencies
        all_files = set(dirty)
        # Include transitive dependents
        for path, record in self._records.items():
            if record.dependencies & all_files:
                all_files.add(path)

        # Kahn's algorithm
        in_degree: dict[str, int] = {f: 0 for f in all_files}
        adj: dict[str, list[str]] = {f: [] for f in all_files}

        for path in all_files:
            record = self._records.get(path)
            if record is None:
                continue
            for dep in record.dependencies:
                if dep in all_files:
                    adj[dep].append(path)
                    in_degree[path] += 1

        queue: deque[str] = deque(p for p, d in in_degree.items() if d == 0)
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbour in adj.get(node, []):
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if len(order) < len(all_files):
            raise ValueError("Cyclic dependency detected")
        return order

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()

    def get_record(self, path: str) -> FileRecord | None:
        """Get the record for a path."""
        return self._records.get(path)
