"""
Delta Sync - Compute and apply minimal change patches.
Part of Core Foundation. Used for network efficiency.

This module provides utilities for computing minimal differences between
dictionary states and applying those differences as patches. This is
particularly useful for network synchronization where sending only
changes is more efficient than sending full state.

Examples:
    sync = DeltaSync()

    # Compute what changed
    old_state = {"health": 100, "position": {"x": 0, "y": 0}}
    new_state = {"health": 80, "position": {"x": 5, "y": 0}}
    delta = sync.compute_delta(old_state, new_state)
    # delta.changes = [("health", 80), ("position.x", 5)]

    # Apply changes to another copy
    target = {"health": 100, "position": {"x": 0, "y": 0}}
    sync.apply_delta(target, delta)
    # target now equals new_state
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from foundation.paths import parse_path, set_path


@dataclass
class DeltaPatch:
    """
    Represents a minimal delta between two states.

    Attributes:
        changes: List of (path, new_value) tuples for changed/added values.
        removes: List of paths that were removed.

    Examples:
        >>> patch = DeltaPatch()
        >>> patch.is_empty()
        True
        >>> patch.changes.append(("health", 80))
        >>> patch.is_empty()
        False
        >>> len(patch)
        1
    """
    changes: list[tuple[str, Any]] = field(default_factory=list)
    removes: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Check if patch has no changes."""
        return not self.changes and not self.removes

    def __len__(self) -> int:
        """Number of operations in this patch."""
        return len(self.changes) + len(self.removes)


class DeltaSync:
    """
    Compute and apply minimal deltas between dict states.

    This class provides methods to efficiently track changes between
    dictionary states. It recursively compares nested dictionaries and
    produces a minimal patch that can be applied to transform one state
    into another.

    Usage:
        sync = DeltaSync()
        delta = sync.compute_delta(old_state, new_state)
        result = sync.apply_delta(target, delta)

    Note:
        - Only dict structures are recursively compared
        - Lists and other types are compared by equality
        - Paths use dotted notation (e.g., "a.b.c")
    """

    def compute_delta(self, old: dict, new: dict, prefix: str = "") -> DeltaPatch:
        """
        Compute minimal delta between two states.

        Args:
            old: Original state dict.
            new: New state dict.
            prefix: Path prefix for recursion (internal use).

        Returns:
            DeltaPatch with changes and removes.

        Examples:
            >>> sync = DeltaSync()
            >>> old = {"a": 1, "b": {"c": 2}}
            >>> new = {"a": 1, "b": {"c": 3}, "d": 4}
            >>> delta = sync.compute_delta(old, new)
            >>> delta.changes
            [('b.c', 3), ('d', 4)]
            >>> delta.removes
            []
        """
        patch = DeltaPatch()

        old_keys = set(old.keys())
        new_keys = set(new.keys())

        # Removed keys
        for key in old_keys - new_keys:
            path = f"{prefix}.{key}" if prefix else key
            patch.removes.append(path)

        # Added or changed keys
        for key in new_keys:
            path = f"{prefix}.{key}" if prefix else key
            new_val = new[key]

            if key not in old:
                # Added
                patch.changes.append((path, new_val))
            else:
                old_val = old[key]
                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    # Recurse into nested dicts
                    nested = self.compute_delta(old_val, new_val, path)
                    patch.changes.extend(nested.changes)
                    patch.removes.extend(nested.removes)
                elif old_val != new_val:
                    # Value changed
                    patch.changes.append((path, new_val))

        return patch

    def apply_delta(self, target: dict, delta: DeltaPatch) -> dict:
        """
        Apply a delta patch to a target dict.

        Args:
            target: Dict to apply changes to (modified in place).
            delta: DeltaPatch to apply.

        Returns:
            The modified target dict.

        Examples:
            >>> sync = DeltaSync()
            >>> target = {"a": 1, "b": {"c": 2}}
            >>> delta = DeltaPatch(changes=[("b.c", 3)], removes=[])
            >>> result = sync.apply_delta(target, delta)
            >>> result["b"]["c"]
            3
        """
        # Apply removes first
        for path in delta.removes:
            self._remove_path(target, path)

        # Apply changes
        for path, value in delta.changes:
            set_path(target, path, value, create_intermediate=True)

        return target

    def _remove_path(self, obj: dict, path: str) -> None:
        """
        Remove a value at the given path.

        Args:
            obj: Dict to remove from.
            path: Dotted path to the value to remove.
        """
        segments = parse_path(path)
        if not segments:
            return

        # Navigate to parent
        parent = obj
        for segment in segments[:-1]:
            if isinstance(segment, int):
                if isinstance(parent, list) and 0 <= segment < len(parent):
                    parent = parent[segment]
                else:
                    return  # Path doesn't exist, nothing to remove
            elif isinstance(parent, dict):
                if segment in parent:
                    parent = parent[segment]
                else:
                    return  # Path doesn't exist, nothing to remove
            else:
                parent = getattr(parent, segment, None)
                if parent is None:
                    return  # Path doesn't exist

        # Remove final key
        final = segments[-1]
        if isinstance(final, int) and isinstance(parent, list):
            if 0 <= final < len(parent):
                parent.pop(final)
        elif isinstance(parent, dict) and final in parent:
            del parent[final]


__all__ = ["DeltaPatch", "DeltaSync"]
