"""
Migration Registry - Schema migration path finding and execution.

Part of Core Foundation Layer 0. Enables:
- Hot reload detection (hash changed → migration needed)
- Versioned save files
- Network protocol compatibility
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Optional


MigrateFn = Callable[[dict], dict]


class MigrationRegistry:
    """
    Registry for schema migrations between different versions.

    Migrations are registered as (from_hash, to_hash) → migrate_fn pairs.
    The registry can find paths through multiple migrations using BFS.
    """

    def __init__(self) -> None:
        self._migrations: dict[tuple[str, str], MigrateFn] = {}
        self._graph: dict[str, set[str]] = {}  # Adjacency list for path finding

    def register(self, from_hash: str, to_hash: str, migrate_fn: MigrateFn) -> None:
        """
        Register a migration function between two schema versions.

        Args:
            from_hash: The source schema hash (16-char hex).
            to_hash: The target schema hash (16-char hex).
            migrate_fn: Function that transforms data from old to new format.
                        Takes a dict, returns a dict.
        """
        key = (from_hash, to_hash)
        self._migrations[key] = migrate_fn

        # Update adjacency list
        if from_hash not in self._graph:
            self._graph[from_hash] = set()
        self._graph[from_hash].add(to_hash)

    def has_migration(self, from_hash: str, to_hash: str) -> bool:
        """Check if a direct migration exists between two hashes."""
        return (from_hash, to_hash) in self._migrations

    def has_path(self, from_hash: str, to_hash: str) -> bool:
        """Check if any migration path exists between two hashes."""
        return self._find_path(from_hash, to_hash) is not None

    def _find_path(self, from_hash: str, to_hash: str) -> Optional[list[tuple[str, str]]]:
        """
        Find a migration path from from_hash to to_hash using BFS.

        Returns:
            List of (from, to) tuples representing the path, or None if no path exists.
        """
        if from_hash == to_hash:
            return []

        if from_hash not in self._graph:
            return None

        # BFS to find shortest path
        queue: deque[tuple[str, list[tuple[str, str]]]] = deque()
        queue.append((from_hash, []))
        visited: set[str] = {from_hash}

        while queue:
            current, path = queue.popleft()

            for neighbor in self._graph.get(current, set()):
                new_path = path + [(current, neighbor)]

                if neighbor == to_hash:
                    return new_path

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))

        return None

    def migrate(self, data: dict, from_hash: str, to_hash: str) -> dict:
        """
        Migrate data from one schema version to another.

        Finds the shortest path through registered migrations and applies them.

        Args:
            data: The data dictionary to migrate.
            from_hash: The current schema hash of the data.
            to_hash: The target schema hash.

        Returns:
            The migrated data dictionary.

        Raises:
            ValueError: If no migration path exists.
        """
        if from_hash == to_hash:
            return data

        path = self._find_path(from_hash, to_hash)
        if path is None:
            raise ValueError(f"No migration path from {from_hash} to {to_hash}")

        result = data
        for step_from, step_to in path:
            migrate_fn = self._migrations[(step_from, step_to)]
            result = migrate_fn(result)

        return result

    def get_path(self, from_hash: str, to_hash: str) -> Optional[list[tuple[str, str]]]:
        """
        Get the migration path between two schema versions.

        Returns:
            List of (from, to) tuples, or None if no path exists.
        """
        return self._find_path(from_hash, to_hash)

    def clear(self) -> None:
        """Clear all registered migrations."""
        self._migrations.clear()
        self._graph.clear()


# Global singleton instance
_registry = MigrationRegistry()


def register_migration(from_hash: str, to_hash: str, migrate_fn: MigrateFn) -> None:
    """Register a migration in the global registry."""
    _registry.register(from_hash, to_hash, migrate_fn)


def migrate(data: dict, from_hash: str, to_hash: str) -> dict:
    """Migrate data using the global registry."""
    return _registry.migrate(data, from_hash, to_hash)


def has_migration_path(from_hash: str, to_hash: str) -> bool:
    """Check if a migration path exists in the global registry."""
    return _registry.has_path(from_hash, to_hash)


def get_migration_path(from_hash: str, to_hash: str) -> Optional[list[tuple[str, str]]]:
    """Get the migration path from the global registry."""
    return _registry.get_path(from_hash, to_hash)


def clear_migrations() -> None:
    """Clear the global migration registry."""
    _registry.clear()


__all__ = [
    "MigrationRegistry",
    "MigrateFn",
    "register_migration",
    "migrate",
    "has_migration_path",
    "get_migration_path",
    "clear_migrations",
]
