"""
Selection - Multi-selection system with marquee, picking, and grouping.

Provides:
- Selection management with add/remove/toggle operations
- Selection sets for storing named selections
- Marquee (box) selection
- Selection filtering by type/properties
- Selection grouping for organizing objects
- Picking results from viewport interaction
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, Iterable, Optional, Set, TypeVar

from engine.tooling.editor.app_shell import editor, reloadable

T = TypeVar('T')


class SelectionOperation(Enum):
    """Operations for modifying selection."""
    SET = auto()      # Replace selection
    ADD = auto()      # Add to selection
    REMOVE = auto()   # Remove from selection
    TOGGLE = auto()   # Toggle in selection


@editor(category="Selection")
@reloadable()
class PickingResult:
    """Result of picking an object in the viewport."""
    __slots__ = ("hit", "object", "position", "normal", "distance",
                 "face_index", "uv")

    def __init__(self, hit: bool = False, object: Any = None,
                 position: tuple = (0.0, 0.0, 0.0), normal: tuple = (0.0, 1.0, 0.0),
                 distance: float = float('inf'), face_index: int = -1,
                 uv: tuple = (0.0, 0.0)):
        self.hit = hit
        self.object = object
        self.position = position
        self.normal = normal
        self.distance = distance
        self.face_index = face_index
        self.uv = uv

    @classmethod
    def miss(cls) -> "PickingResult":
        """Create a miss result."""
        return cls(hit=False)


@editor(category="Selection")
@reloadable()
class SelectionFilter:
    """Filter for selecting objects by type or properties."""
    __slots__ = ("allowed_types", "excluded_types", "custom_filter",
                 "name", "enabled")

    def __init__(self, name: str = ""):
        self.name = name
        self.allowed_types: set[type] = set()
        self.excluded_types: set[type] = set()
        self.custom_filter: Optional[Callable[[Any], bool]] = None
        self.enabled: bool = True

    def allow_type(self, obj_type: type) -> "SelectionFilter":
        """Add a type to allowed types."""
        self.allowed_types.add(obj_type)
        return self

    def exclude_type(self, obj_type: type) -> "SelectionFilter":
        """Add a type to excluded types."""
        self.excluded_types.add(obj_type)
        return self

    def set_custom_filter(self, func: Callable[[Any], bool]) -> "SelectionFilter":
        """Set a custom filter function."""
        self.custom_filter = func
        return self

    def accepts(self, obj: Any) -> bool:
        """Check if an object passes the filter."""
        if not self.enabled:
            return True

        obj_type = type(obj)

        # Check excluded types first
        for excluded in self.excluded_types:
            if isinstance(obj, excluded):
                return False

        # Check allowed types if specified
        if self.allowed_types:
            allowed = False
            for allow_type in self.allowed_types:
                if isinstance(obj, allow_type):
                    allowed = True
                    break
            if not allowed:
                return False

        # Check custom filter
        if self.custom_filter:
            return self.custom_filter(obj)

        return True

    def reset(self) -> None:
        """Reset filter to default state."""
        self.allowed_types.clear()
        self.excluded_types.clear()
        self.custom_filter = None
        self.enabled = True


@editor(category="Selection")
@reloadable()
class Selection(Generic[T]):
    """A set of selected objects with change notification."""
    __slots__ = ("_items", "_primary", "on_changed", "_filter")

    def __init__(self):
        self._items: set[T] = set()
        self._primary: Optional[T] = None
        self.on_changed: Optional[Callable[[set[T]], None]] = None
        self._filter: Optional[SelectionFilter] = None

    @property
    def items(self) -> frozenset[T]:
        """Get immutable set of selected items."""
        return frozenset(self._items)

    @property
    def primary(self) -> Optional[T]:
        """Get the primary selected item."""
        return self._primary

    @property
    def count(self) -> int:
        """Get number of selected items."""
        return len(self._items)

    @property
    def empty(self) -> bool:
        """Check if selection is empty."""
        return len(self._items) == 0

    def set_filter(self, filter: Optional[SelectionFilter]) -> None:
        """Set the selection filter."""
        self._filter = filter

    def _notify(self) -> None:
        """Notify listeners of selection change."""
        if self.on_changed:
            self.on_changed(set(self._items))

    def _can_select(self, item: T) -> bool:
        """Check if an item can be selected."""
        if self._filter:
            return self._filter.accepts(item)
        return True

    def select(self, item: T, operation: SelectionOperation = SelectionOperation.SET) -> bool:
        """Select an item with the given operation. Returns True if selection changed."""
        if not self._can_select(item):
            return False

        changed = False

        if operation == SelectionOperation.SET:
            if self._items != {item}:
                self._items = {item}
                self._primary = item
                changed = True
        elif operation == SelectionOperation.ADD:
            if item not in self._items:
                self._items.add(item)
                if self._primary is None:
                    self._primary = item
                changed = True
        elif operation == SelectionOperation.REMOVE:
            if item in self._items:
                self._items.discard(item)
                if self._primary == item:
                    self._primary = next(iter(self._items), None)
                changed = True
        elif operation == SelectionOperation.TOGGLE:
            if item in self._items:
                self._items.discard(item)
                if self._primary == item:
                    self._primary = next(iter(self._items), None)
            else:
                self._items.add(item)
                if self._primary is None:
                    self._primary = item
            changed = True

        if changed:
            self._notify()
        return changed

    def select_all(self, items: Iterable[T]) -> bool:
        """Select all items, replacing current selection."""
        filtered = {item for item in items if self._can_select(item)}
        if self._items != filtered:
            self._items = filtered
            self._primary = next(iter(self._items), None)
            self._notify()
            return True
        return False

    def add_all(self, items: Iterable[T]) -> bool:
        """Add all items to selection."""
        filtered = {item for item in items if self._can_select(item)}
        new_items = filtered - self._items
        if new_items:
            self._items.update(new_items)
            if self._primary is None:
                self._primary = next(iter(new_items), None)
            self._notify()
            return True
        return False

    def remove_all(self, items: Iterable[T]) -> bool:
        """Remove all items from selection."""
        to_remove = set(items) & self._items
        if to_remove:
            self._items -= to_remove
            if self._primary in to_remove:
                self._primary = next(iter(self._items), None)
            self._notify()
            return True
        return False

    def clear(self) -> bool:
        """Clear the selection."""
        if self._items:
            self._items.clear()
            self._primary = None
            self._notify()
            return True
        return False

    def set_primary(self, item: T) -> bool:
        """Set the primary selected item."""
        if item in self._items:
            self._primary = item
            self._notify()
            return True
        return False

    def contains(self, item: T) -> bool:
        """Check if an item is selected."""
        return item in self._items

    def __contains__(self, item: T) -> bool:
        return item in self._items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


@editor(category="Selection")
@reloadable()
class SelectionSet:
    """A named set of selected objects that can be saved/restored."""
    __slots__ = ("name", "items", "locked", "color")

    def __init__(self, name: str, items: Optional[set] = None):
        self.name = name
        self.items: set = items or set()
        self.locked: bool = False
        self.color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    def add(self, item: Any) -> bool:
        """Add an item to the set."""
        if self.locked:
            return False
        if item not in self.items:
            self.items.add(item)
            return True
        return False

    def remove(self, item: Any) -> bool:
        """Remove an item from the set."""
        if self.locked:
            return False
        if item in self.items:
            self.items.discard(item)
            return True
        return False

    def clear(self) -> bool:
        """Clear the set."""
        if self.locked:
            return False
        if self.items:
            self.items.clear()
            return True
        return False

    def lock(self) -> None:
        """Lock the set to prevent modifications."""
        self.locked = True

    def unlock(self) -> None:
        """Unlock the set."""
        self.locked = False

    @property
    def count(self) -> int:
        """Get number of items."""
        return len(self.items)


@editor(category="Selection")
@reloadable()
class MarqueeSelection:
    """Marquee (box) selection for viewport."""
    __slots__ = ("start_x", "start_y", "end_x", "end_y", "active",
                 "operation", "_candidates")

    def __init__(self):
        self.start_x: int = 0
        self.start_y: int = 0
        self.end_x: int = 0
        self.end_y: int = 0
        self.active: bool = False
        self.operation: SelectionOperation = SelectionOperation.SET
        self._candidates: set = set()

    @property
    def rect(self) -> tuple[int, int, int, int]:
        """Get normalized rectangle (min_x, min_y, max_x, max_y)."""
        min_x = min(self.start_x, self.end_x)
        min_y = min(self.start_y, self.end_y)
        max_x = max(self.start_x, self.end_x)
        max_y = max(self.start_y, self.end_y)
        return (min_x, min_y, max_x, max_y)

    @property
    def width(self) -> int:
        """Get marquee width."""
        return abs(self.end_x - self.start_x)

    @property
    def height(self) -> int:
        """Get marquee height."""
        return abs(self.end_y - self.start_y)

    def begin(self, x: int, y: int, operation: SelectionOperation = SelectionOperation.SET) -> None:
        """Begin marquee selection."""
        self.start_x = x
        self.start_y = y
        self.end_x = x
        self.end_y = y
        self.active = True
        self.operation = operation
        self._candidates.clear()

    def update(self, x: int, y: int) -> None:
        """Update marquee end position."""
        if self.active:
            self.end_x = x
            self.end_y = y

    def end(self) -> set:
        """End marquee selection and return candidates."""
        self.active = False
        result = set(self._candidates)
        self._candidates.clear()
        return result

    def cancel(self) -> None:
        """Cancel marquee selection."""
        self.active = False
        self._candidates.clear()

    def add_candidate(self, obj: Any) -> None:
        """Add an object as a marquee candidate."""
        self._candidates.add(obj)

    def remove_candidate(self, obj: Any) -> None:
        """Remove an object from marquee candidates."""
        self._candidates.discard(obj)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside the marquee."""
        min_x, min_y, max_x, max_y = self.rect
        return min_x <= x <= max_x and min_y <= y <= max_y


@editor(category="Selection")
@reloadable()
class SelectionGroup:
    """A group of objects that can be selected as a unit."""
    __slots__ = ("id", "name", "_members", "locked", "color",
                 "visible", "expanded", "_parent_ref")

    def __init__(self, id: str, name: str = ""):
        self.id = id
        self.name = name or f"Group_{id}"
        self._members: list[Any] = []
        self.locked: bool = False
        self.color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
        self.visible: bool = True
        self.expanded: bool = True
        self._parent_ref: Optional[weakref.ref] = None

    @property
    def members(self) -> list[Any]:
        """Get group members."""
        return list(self._members)

    @property
    def count(self) -> int:
        """Get number of members."""
        return len(self._members)

    def add(self, obj: Any) -> bool:
        """Add an object to the group."""
        if self.locked or obj in self._members:
            return False
        self._members.append(obj)
        return True

    def remove(self, obj: Any) -> bool:
        """Remove an object from the group."""
        if self.locked or obj not in self._members:
            return False
        self._members.remove(obj)
        return True

    def clear(self) -> bool:
        """Clear all members."""
        if self.locked:
            return False
        self._members.clear()
        return True

    def contains(self, obj: Any) -> bool:
        """Check if object is in the group."""
        return obj in self._members

    def lock(self) -> None:
        """Lock the group."""
        self.locked = True

    def unlock(self) -> None:
        """Unlock the group."""
        self.locked = False

    def reorder(self, obj: Any, new_index: int) -> bool:
        """Reorder an object within the group."""
        if self.locked or obj not in self._members:
            return False
        self._members.remove(obj)
        new_index = max(0, min(new_index, len(self._members)))
        self._members.insert(new_index, obj)
        return True


@editor(category="Selection")
@reloadable(preserve=["_selection", "_sets", "_groups"])
class SelectionManager:
    """Central manager for selection operations."""
    __slots__ = ("_selection", "_sets", "_groups", "_history",
                 "_history_index", "_max_history", "on_selection_changed",
                 "_filter", "_tracker_ref")

    def __init__(self, max_history: int = 50):
        self._selection: Selection = Selection()
        self._sets: dict[str, SelectionSet] = {}
        self._groups: dict[str, SelectionGroup] = {}
        self._history: list[set] = []
        self._history_index: int = -1
        self._max_history = max_history
        self.on_selection_changed: Optional[Callable[[set], None]] = None
        self._filter: Optional[SelectionFilter] = None
        self._tracker_ref: Any = None

        # Wire up selection change callback
        self._selection.on_changed = self._on_internal_change

    @property
    def selection(self) -> Selection:
        """Get the current selection."""
        return self._selection

    @property
    def selected_items(self) -> frozenset:
        """Get currently selected items."""
        return self._selection.items

    @property
    def primary(self) -> Any:
        """Get the primary selected item."""
        return self._selection.primary

    @property
    def count(self) -> int:
        """Get selection count."""
        return self._selection.count

    def set_tracker(self, tracker: Any) -> None:
        """Set the Foundation Tracker for undo integration."""
        self._tracker_ref = tracker

    def set_filter(self, filter: Optional[SelectionFilter]) -> None:
        """Set the selection filter."""
        self._filter = filter
        self._selection.set_filter(filter)

    def _on_internal_change(self, items: set) -> None:
        """Handle internal selection change."""
        # Record history
        self._record_history(items)

        # Notify external listeners
        if self.on_selection_changed:
            self.on_selection_changed(items)

    def _record_history(self, items: set) -> None:
        """Record selection state in history."""
        # Truncate forward history if we're not at the end
        if self._history_index < len(self._history) - 1:
            self._history = self._history[:self._history_index + 1]

        self._history.append(set(items))
        self._history_index = len(self._history) - 1

        # Limit history size
        while len(self._history) > self._max_history:
            self._history.pop(0)
            self._history_index = max(0, self._history_index - 1)

    def select(self, obj: Any, operation: SelectionOperation = SelectionOperation.SET) -> bool:
        """Select an object."""
        return self._selection.select(obj, operation)

    def select_all(self, items: Iterable[Any]) -> bool:
        """Select all items."""
        return self._selection.select_all(items)

    def add_to_selection(self, obj: Any) -> bool:
        """Add to selection."""
        return self._selection.select(obj, SelectionOperation.ADD)

    def remove_from_selection(self, obj: Any) -> bool:
        """Remove from selection."""
        return self._selection.select(obj, SelectionOperation.REMOVE)

    def toggle_selection(self, obj: Any) -> bool:
        """Toggle selection."""
        return self._selection.select(obj, SelectionOperation.TOGGLE)

    def clear_selection(self) -> bool:
        """Clear selection."""
        return self._selection.clear()

    def is_selected(self, obj: Any) -> bool:
        """Check if object is selected."""
        return self._selection.contains(obj)

    # Selection sets
    def create_set(self, name: str, from_current: bool = True) -> SelectionSet:
        """Create a named selection set."""
        items = set(self._selection.items) if from_current else set()
        selection_set = SelectionSet(name, items)
        self._sets[name] = selection_set
        return selection_set

    def delete_set(self, name: str) -> bool:
        """Delete a selection set."""
        return self._sets.pop(name, None) is not None

    def get_set(self, name: str) -> Optional[SelectionSet]:
        """Get a selection set by name."""
        return self._sets.get(name)

    def restore_set(self, name: str) -> bool:
        """Restore selection from a set."""
        selection_set = self._sets.get(name)
        if selection_set:
            return self._selection.select_all(selection_set.items)
        return False

    def add_set_to_selection(self, name: str) -> bool:
        """Add a selection set to current selection."""
        selection_set = self._sets.get(name)
        if selection_set:
            return self._selection.add_all(selection_set.items)
        return False

    @property
    def set_names(self) -> list[str]:
        """Get all selection set names."""
        return list(self._sets.keys())

    # Groups
    def create_group(self, name: str = "", from_selection: bool = True) -> SelectionGroup:
        """Create a selection group."""
        import uuid
        group_id = str(uuid.uuid4())[:8]
        group = SelectionGroup(group_id, name or f"Group_{len(self._groups)}")
        if from_selection:
            for item in self._selection.items:
                group.add(item)
        self._groups[group_id] = group
        return group

    def delete_group(self, group_id: str) -> bool:
        """Delete a group."""
        return self._groups.pop(group_id, None) is not None

    def get_group(self, group_id: str) -> Optional[SelectionGroup]:
        """Get a group by ID."""
        return self._groups.get(group_id)

    def find_groups_containing(self, obj: Any) -> list[SelectionGroup]:
        """Find all groups containing an object."""
        return [g for g in self._groups.values() if g.contains(obj)]

    @property
    def groups(self) -> list[SelectionGroup]:
        """Get all groups."""
        return list(self._groups.values())

    # History navigation
    def undo_selection(self) -> bool:
        """Undo to previous selection state."""
        if self._history_index > 0:
            self._history_index -= 1
            self._selection._items = set(self._history[self._history_index])
            self._selection._primary = next(iter(self._selection._items), None)
            if self.on_selection_changed:
                self.on_selection_changed(set(self._selection._items))
            return True
        return False

    def redo_selection(self) -> bool:
        """Redo to next selection state."""
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._selection._items = set(self._history[self._history_index])
            self._selection._primary = next(iter(self._selection._items), None)
            if self.on_selection_changed:
                self.on_selection_changed(set(self._selection._items))
            return True
        return False

    @property
    def can_undo_selection(self) -> bool:
        """Check if selection undo is available."""
        return self._history_index > 0

    @property
    def can_redo_selection(self) -> bool:
        """Check if selection redo is available."""
        return self._history_index < len(self._history) - 1

    # Filtering helpers
    def select_by_type(self, obj_type: type) -> int:
        """Select all objects of a type. Returns count selected."""
        # This would need scene integration to find all objects
        count = 0
        # Placeholder - would iterate scene objects
        return count

    def invert_selection(self, all_objects: Iterable[Any]) -> bool:
        """Invert selection relative to all objects."""
        current = set(self._selection.items)
        inverted = {obj for obj in all_objects if obj not in current}
        return self._selection.select_all(inverted)
