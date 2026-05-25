"""
Observable collections for UI data binding.

Provides collections that emit change notifications for reactive UI updates.
Includes ObservableList, ObservableDict, and virtualization support.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    MutableMapping,
    MutableSequence,
    Optional,
    TypeVar,
    Union,
    overload,
)

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

# Default configuration constants
DEFAULT_ITEM_HEIGHT = 20.0

# Callback type for collection changes
CollectionChangeCallback = Callable[["CollectionChangeEvent"], None]


class CollectionChangeAction(Enum):
    """Types of changes that can occur in a collection."""

    ADD = auto()
    REMOVE = auto()
    REPLACE = auto()
    MOVE = auto()
    RESET = auto()


@dataclass
class CollectionChangeEvent(Generic[T]):
    """Event data for collection changes."""

    action: CollectionChangeAction
    new_items: List[T] = field(default_factory=list)
    old_items: List[T] = field(default_factory=list)
    new_starting_index: int = -1
    old_starting_index: int = -1

    @classmethod
    def add(cls, items: List[T], index: int) -> "CollectionChangeEvent[T]":
        """Create an add event."""
        return cls(
            action=CollectionChangeAction.ADD,
            new_items=items,
            new_starting_index=index,
        )

    @classmethod
    def remove(cls, items: List[T], index: int) -> "CollectionChangeEvent[T]":
        """Create a remove event."""
        return cls(
            action=CollectionChangeAction.REMOVE,
            old_items=items,
            old_starting_index=index,
        )

    @classmethod
    def replace(
        cls, old_item: T, new_item: T, index: int
    ) -> "CollectionChangeEvent[T]":
        """Create a replace event."""
        return cls(
            action=CollectionChangeAction.REPLACE,
            new_items=[new_item],
            old_items=[old_item],
            new_starting_index=index,
            old_starting_index=index,
        )

    @classmethod
    def move(
        cls, item: T, old_index: int, new_index: int
    ) -> "CollectionChangeEvent[T]":
        """Create a move event."""
        return cls(
            action=CollectionChangeAction.MOVE,
            new_items=[item],
            old_items=[item],
            new_starting_index=new_index,
            old_starting_index=old_index,
        )

    @classmethod
    def reset(cls) -> "CollectionChangeEvent[T]":
        """Create a reset event."""
        return cls(action=CollectionChangeAction.RESET)


class IObservableCollection(ABC, Generic[T]):
    """Interface for observable collections."""

    @abstractmethod
    def add_listener(self, callback: CollectionChangeCallback) -> None:
        """Add a listener for collection changes."""
        pass

    @abstractmethod
    def remove_listener(self, callback: CollectionChangeCallback) -> None:
        """Remove a listener."""
        pass

    @abstractmethod
    def suspend_notifications(self) -> None:
        """Suspend change notifications."""
        pass

    @abstractmethod
    def resume_notifications(self) -> None:
        """Resume change notifications."""
        pass


class ObservableList(MutableSequence[T], IObservableCollection[T]):
    """
    A list that emits change notifications.

    Supports all standard list operations with automatic change notification.
    Provides virtualization support for efficient rendering of large lists.
    """

    def __init__(self, initial: Optional[List[T]] = None):
        self._data: List[T] = list(initial) if initial else []
        self._listeners: List[CollectionChangeCallback] = []
        self._lock = threading.RLock()
        self._notifications_suspended = False
        self._pending_events: List[CollectionChangeEvent[T]] = []

    def __repr__(self) -> str:
        return f"ObservableList({self._data!r})"

    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> List[T]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T, List[T]]:
        return self._data[index]

    @overload
    def __setitem__(self, index: int, value: T) -> None:
        ...

    @overload
    def __setitem__(self, index: slice, value: List[T]) -> None:
        ...

    def __setitem__(self, index: Union[int, slice], value: Union[T, List[T]]) -> None:
        with self._lock:
            if isinstance(index, slice):
                old_items = self._data[index]
                self._data[index] = value  # type: ignore
                # Notify reset for slice assignment
                self._notify(CollectionChangeEvent.reset())
            else:
                old_item = self._data[index]
                self._data[index] = value  # type: ignore
                self._notify(CollectionChangeEvent.replace(old_item, value, index))  # type: ignore

    def __delitem__(self, index: Union[int, slice]) -> None:
        with self._lock:
            if isinstance(index, slice):
                old_items = self._data[index]
                del self._data[index]
                self._notify(CollectionChangeEvent.reset())
            else:
                old_item = self._data[index]
                del self._data[index]
                self._notify(CollectionChangeEvent.remove([old_item], index))

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __contains__(self, item: object) -> bool:
        return item in self._data

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ObservableList):
            return self._data == other._data
        if isinstance(other, list):
            return self._data == other
        return False

    def insert(self, index: int, value: T) -> None:
        """Insert an item at a given position."""
        with self._lock:
            self._data.insert(index, value)
            self._notify(CollectionChangeEvent.add([value], index))

    def append(self, value: T) -> None:
        """Append an item to the end."""
        with self._lock:
            index = len(self._data)
            self._data.append(value)
            self._notify(CollectionChangeEvent.add([value], index))

    def extend(self, values: List[T]) -> None:
        """Extend the list with multiple items."""
        with self._lock:
            index = len(self._data)
            self._data.extend(values)
            self._notify(CollectionChangeEvent.add(list(values), index))

    def pop(self, index: int = -1) -> T:
        """Remove and return an item at the given position."""
        with self._lock:
            if index < 0:
                index = len(self._data) + index
            item = self._data.pop(index)
            self._notify(CollectionChangeEvent.remove([item], index))
            return item

    def remove(self, value: T) -> None:
        """Remove the first occurrence of a value."""
        with self._lock:
            index = self._data.index(value)
            del self._data[index]
            self._notify(CollectionChangeEvent.remove([value], index))

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._data.clear()
            self._notify(CollectionChangeEvent.reset())

    def index(self, value: T, start: int = 0, stop: int = None) -> int:  # type: ignore
        """Return the index of the first occurrence of a value."""
        if stop is None:
            stop = len(self._data)
        return self._data.index(value, start, stop)

    def count(self, value: T) -> int:
        """Return the number of occurrences of a value."""
        return self._data.count(value)

    def reverse(self) -> None:
        """Reverse the list in place."""
        with self._lock:
            self._data.reverse()
            self._notify(CollectionChangeEvent.reset())

    def sort(self, *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False) -> None:
        """Sort the list in place."""
        with self._lock:
            self._data.sort(key=key, reverse=reverse)
            self._notify(CollectionChangeEvent.reset())

    def copy(self) -> List[T]:
        """Return a shallow copy of the list."""
        return self._data.copy()

    def move(self, old_index: int, new_index: int) -> None:
        """Move an item from one position to another."""
        with self._lock:
            if old_index == new_index:
                return
            item = self._data.pop(old_index)
            self._data.insert(new_index, item)
            self._notify(CollectionChangeEvent.move(item, old_index, new_index))

    # Observable interface
    def add_listener(self, callback: CollectionChangeCallback) -> None:
        """Add a listener for collection changes."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_listener(self, callback: CollectionChangeCallback) -> None:
        """Remove a listener."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def suspend_notifications(self) -> None:
        """Suspend change notifications."""
        with self._lock:
            self._notifications_suspended = True

    def resume_notifications(self) -> None:
        """Resume change notifications and flush pending events."""
        with self._lock:
            self._notifications_suspended = False
            # Flush pending events
            if self._pending_events:
                # Collapse to a single reset if multiple changes
                self._notify(CollectionChangeEvent.reset())
                self._pending_events.clear()

    def _notify(self, event: CollectionChangeEvent[T]) -> None:
        """Notify all listeners of a change."""
        if self._notifications_suspended:
            self._pending_events.append(event)
            return

        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                pass  # Silently ignore listener errors

    # Virtualization support
    def get_range(self, start: int, count: int) -> List[T]:
        """Get a range of items for virtualized rendering."""
        end = min(start + count, len(self._data))
        return self._data[start:end]

    @property
    def total_count(self) -> int:
        """Total number of items (for virtualization)."""
        return len(self._data)


class ObservableDict(MutableMapping[K, V], IObservableCollection[V]):
    """
    A dictionary that emits change notifications.

    Supports all standard dict operations with automatic change notification.
    """

    def __init__(self, initial: Optional[dict] = None):
        self._data: dict[K, V] = dict(initial) if initial else {}
        self._listeners: List[CollectionChangeCallback] = []
        self._lock = threading.RLock()
        self._notifications_suspended = False
        self._pending_events: List[CollectionChangeEvent] = []

    def __repr__(self) -> str:
        return f"ObservableDict({self._data!r})"

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            is_new = key not in self._data
            old_value = self._data.get(key)
            self._data[key] = value
            if is_new:
                self._notify(CollectionChangeEvent.add([value], -1))
            else:
                self._notify(CollectionChangeEvent.replace(old_value, value, -1))

    def __delitem__(self, key: K) -> None:
        with self._lock:
            value = self._data[key]
            del self._data[key]
            self._notify(CollectionChangeEvent.remove([value], -1))

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ObservableDict):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return False

    def keys(self):
        """Return a view of the keys."""
        return self._data.keys()

    def values(self):
        """Return a view of the values."""
        return self._data.values()

    def items(self):
        """Return a view of the items."""
        return self._data.items()

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get a value with an optional default."""
        return self._data.get(key, default)

    def pop(self, key: K, *args) -> V:
        """Remove and return a value."""
        with self._lock:
            if key in self._data:
                value = self._data.pop(key)
                self._notify(CollectionChangeEvent.remove([value], -1))
                return value
            elif args:
                return args[0]
            raise KeyError(key)

    def popitem(self) -> tuple[K, V]:
        """Remove and return an arbitrary key-value pair."""
        with self._lock:
            key, value = self._data.popitem()
            self._notify(CollectionChangeEvent.remove([value], -1))
            return key, value

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._data.clear()
            self._notify(CollectionChangeEvent.reset())

    def update(self, other: Optional[dict] = None, **kwargs) -> None:
        """Update the dictionary."""
        with self._lock:
            if other:
                self._data.update(other)
            self._data.update(kwargs)
            self._notify(CollectionChangeEvent.reset())

    def setdefault(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get a value, setting it to default if not present."""
        with self._lock:
            if key not in self._data:
                self._data[key] = default  # type: ignore
                self._notify(CollectionChangeEvent.add([default], -1))
            return self._data[key]

    def copy(self) -> dict[K, V]:
        """Return a shallow copy."""
        return self._data.copy()

    # Observable interface
    def add_listener(self, callback: CollectionChangeCallback) -> None:
        """Add a listener for collection changes."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_listener(self, callback: CollectionChangeCallback) -> None:
        """Remove a listener."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def suspend_notifications(self) -> None:
        """Suspend change notifications."""
        with self._lock:
            self._notifications_suspended = True

    def resume_notifications(self) -> None:
        """Resume change notifications and flush pending events."""
        with self._lock:
            self._notifications_suspended = False
            if self._pending_events:
                self._notify(CollectionChangeEvent.reset())
                self._pending_events.clear()

    def _notify(self, event: CollectionChangeEvent) -> None:
        """Notify all listeners of a change."""
        if self._notifications_suspended:
            self._pending_events.append(event)
            return

        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                pass


class VirtualizedListView(Generic[T]):
    """
    Provides a virtualized view of an observable list.

    Optimizes rendering by only tracking visible items and recycling
    widget instances as items scroll in/out of view.
    """

    def __init__(
        self,
        source: ObservableList[T],
        visible_count: int,
        item_height: float = DEFAULT_ITEM_HEIGHT,
    ):
        self._source = source
        self._disposed = False
        self._visible_count = visible_count
        self._item_height = item_height
        self._scroll_offset = 0
        self._recycled_widgets: List[Any] = []
        self._active_widgets: dict[int, Any] = {}
        self._listeners: List[Callable[[int, int], None]] = []
        self._lock = threading.RLock()

        # Subscribe to source changes
        source.add_listener(self._on_source_changed)

    @property
    def scroll_offset(self) -> int:
        """Current scroll position (first visible item index)."""
        return self._scroll_offset

    @scroll_offset.setter
    def scroll_offset(self, value: int) -> None:
        """Set scroll position."""
        with self._lock:
            max_offset = max(0, len(self._source) - self._visible_count)
            self._scroll_offset = max(0, min(value, max_offset))
            self._notify_range_change()

    @property
    def visible_range(self) -> tuple[int, int]:
        """Return the range of visible item indices (start, end)."""
        start = self._scroll_offset
        end = min(start + self._visible_count, len(self._source))
        return start, end

    @property
    def visible_items(self) -> List[T]:
        """Return the currently visible items."""
        start, end = self.visible_range
        return list(self._source[start:end])

    @property
    def total_height(self) -> float:
        """Total scrollable height."""
        return len(self._source) * self._item_height

    @property
    def viewport_height(self) -> float:
        """Height of the visible viewport."""
        return self._visible_count * self._item_height

    def scroll_by(self, delta: int) -> None:
        """Scroll by a number of items."""
        self.scroll_offset = self._scroll_offset + delta

    def scroll_to_item(self, index: int) -> None:
        """Scroll to make an item visible."""
        if index < self._scroll_offset:
            self.scroll_offset = index
        elif index >= self._scroll_offset + self._visible_count:
            self.scroll_offset = index - self._visible_count + 1

    def acquire_widget(self) -> Optional[Any]:
        """Get a recycled widget or None if pool is empty."""
        with self._lock:
            if self._recycled_widgets:
                return self._recycled_widgets.pop()
            return None

    def recycle_widget(self, widget: Any) -> None:
        """Return a widget to the recycling pool."""
        with self._lock:
            self._recycled_widgets.append(widget)

    def bind_widget(self, index: int, widget: Any) -> None:
        """Bind a widget to a specific item index."""
        with self._lock:
            self._active_widgets[index] = widget

    def unbind_widget(self, index: int) -> Optional[Any]:
        """Unbind and return a widget from an index."""
        with self._lock:
            return self._active_widgets.pop(index, None)

    def get_bound_widget(self, index: int) -> Optional[Any]:
        """Get the widget bound to an index, if any."""
        return self._active_widgets.get(index)

    def add_range_listener(self, callback: Callable[[int, int], None]) -> None:
        """Add a listener for visible range changes."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_range_listener(self, callback: Callable[[int, int], None]) -> None:
        """Remove a range change listener."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def _on_source_changed(self, event: CollectionChangeEvent) -> None:
        """Handle source collection changes."""
        if self._disposed:
            return
        with self._lock:
            # Adjust scroll offset if needed
            max_offset = max(0, len(self._source) - self._visible_count)
            if self._scroll_offset > max_offset:
                self._scroll_offset = max_offset
            self._notify_range_change()

    def _notify_range_change(self) -> None:
        """Notify listeners of visible range change."""
        start, end = self.visible_range
        for listener in list(self._listeners):
            try:
                listener(start, end)
            except Exception:
                pass

    def dispose(self) -> None:
        """Clean up resources."""
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            try:
                self._source.remove_listener(self._on_source_changed)
            except (ValueError, ReferenceError):
                pass  # Source may already be gone or listener already removed
            self._listeners.clear()
            self._recycled_widgets.clear()
            self._active_widgets.clear()


__all__ = [
    "CollectionChangeAction",
    "CollectionChangeCallback",
    "CollectionChangeEvent",
    "IObservableCollection",
    "ObservableDict",
    "ObservableList",
    "VirtualizedListView",
]
