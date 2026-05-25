"""AST cache for FlowForge Backend.

Caches parsed NodeGraph results per file path, invalidating when the
file's modification time changes.
"""

from __future__ import annotations

import os
from typing import Optional

from .graph_types import NodeGraph


class ASTCache:
    """In-memory cache mapping file paths to parsed NodeGraph results.

    Each entry stores the file's mtime at parse time.  A cache hit is
    only returned when the file's current mtime matches the stored one.
    """

    def __init__(self) -> None:
        # key: abs path -> (mtime, NodeGraph)
        self._entries: dict[str, tuple[float, NodeGraph]] = {}

    def get(self, path: str) -> Optional[NodeGraph]:
        """Return cached NodeGraph if the file hasn't been modified.

        Args:
            path: File path (will be normalised to absolute).

        Returns:
            Cached NodeGraph or None on miss / stale entry.
        """
        key = os.path.abspath(path)
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_mtime, graph = entry
        try:
            current_mtime = os.path.getmtime(key)
        except OSError:
            # File gone – drop entry
            self._entries.pop(key, None)
            return None
        if current_mtime != stored_mtime:
            self._entries.pop(key, None)
            return None
        return graph

    def put(self, path: str, graph: NodeGraph) -> None:
        """Store a NodeGraph for the given file path.

        Args:
            path: File path (will be normalised to absolute).
            graph: The parsed NodeGraph to cache.
        """
        key = os.path.abspath(path)
        try:
            mtime = os.path.getmtime(key)
        except OSError:
            return
        self._entries[key] = (mtime, graph)

    def invalidate(self, path: str) -> None:
        """Remove a single cache entry.

        Args:
            path: File path to invalidate.
        """
        key = os.path.abspath(path)
        self._entries.pop(key, None)

    def clear(self) -> None:
        """Remove all cache entries."""
        self._entries.clear()


# Module-level singleton so the cache persists across calls.
_default_cache = ASTCache()


def get_default_cache() -> ASTCache:
    """Return the module-level default ASTCache instance."""
    return _default_cache
