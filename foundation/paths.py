"""
Path Utilities - Parse and navigate dotted paths for reflection.
Part of Core Foundation Layer 0.

This module provides utilities for navigating nested object structures
using dotted path notation with array index support.

Examples:
    "a.b.c" -> Navigate through nested attributes
    "items[0]" -> Access array element
    "data[0].name" -> Mixed navigation
"""
from __future__ import annotations
from typing import Any, Union
import re

# Sentinel for missing default (allows None as valid default)
_MISSING = object()


class PathError(Exception):
    """Raised when path navigation fails."""
    pass


# Pattern to match path segments: either a name, or a name followed by indices
# Examples: "foo", "items[0]", "data[0][1]"
_SEGMENT_PATTERN = re.compile(r'([^.\[\]]+)|\[(\d+)\]')


def parse_path(path: str) -> list[Union[str, int]]:
    """
    Parse a dotted path string into segments.

    Args:
        path: Dotted path string with optional array indices.

    Returns:
        List of path segments (strings for names, ints for indices).

    Examples:
        >>> parse_path("a.b.c")
        ['a', 'b', 'c']
        >>> parse_path("items[0]")
        ['items', 0]
        >>> parse_path("data[0].name")
        ['data', 0, 'name']
        >>> parse_path("a[0][1]")
        ['a', 0, 1]
        >>> parse_path("x[2].y[3].z")
        ['x', 2, 'y', 3, 'z']
    """
    if not path:
        return []

    segments: list[Union[str, int]] = []

    for match in _SEGMENT_PATTERN.finditer(path):
        name_group = match.group(1)
        index_group = match.group(2)

        if name_group is not None:
            # It's a name segment - may contain dots, so split further
            for part in name_group.split('.'):
                if part:  # Skip empty strings from leading/trailing dots
                    segments.append(part)
        elif index_group is not None:
            # It's an array index
            segments.append(int(index_group))

    return segments


def _get_segment(obj: Any, segment: Union[str, int]) -> Any:
    """Get a single segment from an object."""
    if isinstance(segment, int):
        # Array index access
        try:
            return obj[segment]
        except (TypeError, IndexError, KeyError) as e:
            raise PathError(f"Cannot access index [{segment}] on {type(obj).__name__}: {e}")
    else:
        # String segment - try dict first, then attribute
        if isinstance(obj, dict):
            if segment in obj:
                return obj[segment]
            raise PathError(f"Key '{segment}' not found in dict")

        if hasattr(obj, segment):
            return getattr(obj, segment)

        # Try subscript access as fallback (for dict-like objects)
        try:
            return obj[segment]
        except (TypeError, KeyError):
            pass

        raise PathError(f"Attribute '{segment}' not found on {type(obj).__name__}")


def get_path(obj: Any, path: str, default: Any = _MISSING) -> Any:
    """
    Get value at dotted path.

    Args:
        obj: Root object to navigate from.
        path: Dotted path string.
        default: Default value if path not found (raises PathError if not provided).

    Returns:
        The value at the specified path.

    Raises:
        PathError: If path cannot be navigated and no default provided.

    Examples:
        >>> get_path({"a": {"b": 1}}, "a.b")
        1
        >>> get_path(player, "inventory.items[0].damage")
        10
        >>> get_path(obj, "missing.path", default=None)
        None
    """
    if not path:
        return obj

    segments = parse_path(path)
    current = obj

    try:
        for segment in segments:
            current = _get_segment(current, segment)
        return current
    except PathError:
        if default is not _MISSING:
            return default
        raise


def _set_segment(obj: Any, segment: Union[str, int], value: Any) -> None:
    """Set a single segment on an object."""
    if isinstance(segment, int):
        # Array index access
        try:
            obj[segment] = value
        except (TypeError, IndexError) as e:
            raise PathError(f"Cannot set index [{segment}] on {type(obj).__name__}: {e}")
    else:
        # String segment - try dict first, then attribute
        if isinstance(obj, dict):
            obj[segment] = value
        elif hasattr(obj, segment):
            setattr(obj, segment, value)
        else:
            # Try subscript as fallback, else set attribute
            try:
                obj[segment] = value
            except TypeError:
                setattr(obj, segment, value)


def set_path(obj: Any, path: str, value: Any, create_intermediate: bool = False) -> None:
    """
    Set value at dotted path.

    Args:
        obj: Root object to navigate from.
        path: Dotted path string.
        value: Value to set.
        create_intermediate: If True, create missing dicts along the way.

    Raises:
        PathError: If path cannot be navigated (unless create_intermediate=True for dicts).

    Examples:
        >>> data = {"a": {"b": 1}}
        >>> set_path(data, "a.b", 2)
        >>> data["a"]["b"]
        2
        >>> set_path(data, "x.y.z", 5, create_intermediate=True)
        >>> data["x"]["y"]["z"]
        5
    """
    if not path:
        raise PathError("Cannot set value on empty path")

    segments = parse_path(path)

    if not segments:
        raise PathError("Cannot set value on empty path")

    # Navigate to parent, creating intermediate dicts if needed
    current = obj
    for segment in segments[:-1]:
        try:
            current = _get_segment(current, segment)
        except PathError:
            if create_intermediate and isinstance(segment, str):
                # Create intermediate dict
                new_dict: dict[str, Any] = {}
                _set_segment(current, segment, new_dict)
                current = new_dict
            else:
                raise

    # Set the final value
    _set_segment(current, segments[-1], value)


__all__ = ["PathError", "parse_path", "get_path", "set_path"]
