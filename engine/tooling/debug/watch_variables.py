"""
Watch Variables - Variable watch window, breakpoints, and conditional watches.

Provides tools for monitoring variable values, setting breakpoints,
and triggering actions on value changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, ClassVar, Optional, Any, Generic, TypeVar
import threading
import time
import weakref


T = TypeVar('T')


class WatchType(Enum):
    """Types of watches."""
    VALUE = auto()       # Simple value watch
    EXPRESSION = auto()  # Expression watch
    CONDITIONAL = auto() # Conditional watch
    BREAKPOINT = auto()  # Breakpoint


class BreakpointType(Enum):
    """Types of breakpoints."""
    ALWAYS = auto()      # Always break
    CONDITIONAL = auto() # Break on condition
    HIT_COUNT = auto()   # Break after N hits
    LOG_ONLY = auto()    # Log but don't break


class ValueChangeType(Enum):
    """Types of value changes."""
    ANY = auto()        # Any change
    INCREASE = auto()   # Value increased
    DECREASE = auto()   # Value decreased
    EQUALS = auto()     # Value equals target
    NOT_EQUALS = auto() # Value not equals target
    GREATER = auto()    # Value greater than target
    LESS = auto()       # Value less than target


@dataclass
class WatchHistoryEntry:
    """Historical entry for a watch."""
    timestamp: float
    value: Any
    frame: int = 0


@dataclass
class WatchVariable:
    """Represents a watched variable."""
    name: str
    getter: Callable[[], Any]
    watch_type: WatchType = WatchType.VALUE
    enabled: bool = True
    category: str = "default"
    format_string: str = "{value}"
    history: list[WatchHistoryEntry] = field(default_factory=list)
    max_history: int = 100
    last_value: Any = None
    update_count: int = 0

    def get_value(self) -> Any:
        """Get the current value."""
        try:
            return self.getter()
        except Exception as e:
            return f"<error: {e}>"

    def format_value(self, value: Any) -> str:
        """Format the value for display."""
        try:
            return self.format_string.format(value=value)
        except Exception:
            return str(value)

    def update(self, frame: int = 0) -> bool:
        """Update the watch. Returns True if value changed."""
        if not self.enabled:
            return False

        new_value = self.get_value()
        changed = new_value != self.last_value

        if changed:
            self.history.append(WatchHistoryEntry(
                timestamp=time.time(),
                value=new_value,
                frame=frame,
            ))

            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

            self.last_value = new_value

        self.update_count += 1
        return changed

    def clear_history(self) -> None:
        """Clear value history."""
        self.history.clear()


@dataclass
class Breakpoint:
    """Represents a breakpoint."""
    breakpoint_id: str
    name: str
    condition: Callable[[], bool]
    breakpoint_type: BreakpointType = BreakpointType.ALWAYS
    enabled: bool = True
    hit_count: int = 0
    target_hit_count: int = 1
    action: Optional[Callable[[], None]] = None
    log_message: str = ""
    triggered: bool = False
    last_triggered_time: float = 0.0

    def check(self) -> bool:
        """Check if breakpoint should trigger."""
        if not self.enabled:
            return False

        try:
            condition_met = self.condition()
        except Exception:
            return False

        if not condition_met:
            return False

        self.hit_count += 1

        if self.breakpoint_type == BreakpointType.HIT_COUNT:
            if self.hit_count < self.target_hit_count:
                return False

        self.triggered = True
        self.last_triggered_time = time.time()

        if self.action:
            self.action()

        return self.breakpoint_type != BreakpointType.LOG_ONLY

    def reset(self) -> None:
        """Reset hit count and triggered state."""
        self.hit_count = 0
        self.triggered = False


class ConditionalWatch(WatchVariable, Generic[T]):
    """Watch that triggers on specific conditions."""

    __slots__ = (
        '_change_type',
        '_target_value',
        '_on_trigger',
        '_triggered',
        '_trigger_count',
    )

    def __init__(
        self,
        name: str,
        getter: Callable[[], T],
        change_type: ValueChangeType = ValueChangeType.ANY,
        target_value: Optional[T] = None,
        on_trigger: Optional[Callable[[T, T], None]] = None,
        **kwargs,
    ):
        super().__init__(name=name, getter=getter, watch_type=WatchType.CONDITIONAL, **kwargs)
        self._change_type = change_type
        self._target_value = target_value
        self._on_trigger = on_trigger
        self._triggered = False
        self._trigger_count = 0

    @property
    def change_type(self) -> ValueChangeType:
        return self._change_type

    @change_type.setter
    def change_type(self, value: ValueChangeType) -> None:
        self._change_type = value

    @property
    def target_value(self) -> Optional[T]:
        return self._target_value

    @target_value.setter
    def target_value(self, value: Optional[T]) -> None:
        self._target_value = value

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    def update(self, frame: int = 0) -> bool:
        """Update and check condition."""
        if not self.enabled:
            return False

        old_value = self.last_value
        changed = super().update(frame)

        if changed:
            self._check_condition(old_value, self.last_value)

        return changed

    def _check_condition(self, old_value: Any, new_value: Any) -> None:
        """Check if the change condition is met."""
        should_trigger = False

        if self._change_type == ValueChangeType.ANY:
            should_trigger = True
        elif self._change_type == ValueChangeType.INCREASE:
            try:
                should_trigger = new_value > old_value
            except TypeError:
                pass
        elif self._change_type == ValueChangeType.DECREASE:
            try:
                should_trigger = new_value < old_value
            except TypeError:
                pass
        elif self._change_type == ValueChangeType.EQUALS:
            should_trigger = new_value == self._target_value
        elif self._change_type == ValueChangeType.NOT_EQUALS:
            should_trigger = new_value != self._target_value
        elif self._change_type == ValueChangeType.GREATER:
            try:
                should_trigger = new_value > self._target_value
            except TypeError:
                pass
        elif self._change_type == ValueChangeType.LESS:
            try:
                should_trigger = new_value < self._target_value
            except TypeError:
                pass

        if should_trigger:
            self._triggered = True
            self._trigger_count += 1
            if self._on_trigger:
                self._on_trigger(old_value, new_value)
        else:
            self._triggered = False

    def set_on_trigger(self, callback: Callable[[T, T], None]) -> None:
        """Set trigger callback."""
        self._on_trigger = callback

    def reset_trigger_count(self) -> None:
        """Reset trigger count."""
        self._trigger_count = 0
        self._triggered = False


class VariableTracker:
    """Tracks object properties automatically."""

    __slots__ = ('_object_ref', '_property_name', '_last_value')

    def __init__(self, obj: Any, property_name: str):
        # Cannot create weak references to built-in types like int, float, str, bool, None, dict, list, tuple
        if isinstance(obj, (int, float, str, bool, type(None), dict, list, tuple)):
            self._object_ref = lambda: obj
        else:
            self._object_ref = weakref.ref(obj)
        self._property_name = property_name
        self._last_value = None

    def get_value(self) -> Any:
        """Get the tracked value."""
        obj = self._object_ref()
        if obj is None:
            return "<object destroyed>"

        try:
            if hasattr(obj, self._property_name):
                return getattr(obj, self._property_name)
            elif isinstance(obj, dict) and self._property_name in obj:
                return obj[self._property_name]
            elif hasattr(obj, '__getitem__'):
                return obj[self._property_name]
            else:
                return f"<no property: {self._property_name}>"
        except Exception as e:
            return f"<error: {e}>"

    @property
    def is_valid(self) -> bool:
        """Check if the tracked object is still valid."""
        return self._object_ref() is not None


class WatchWindow:
    """Main watch window managing all watches and breakpoints."""

    _instance: ClassVar[Optional["WatchWindow"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    __slots__ = (
        '_watches',
        '_breakpoints',
        '_enabled',
        '_visible',
        '_paused',
        '_current_frame',
        '_categories',
        '_on_breakpoint_hit',
        '_update_interval',
        '_last_update',
    )

    def __init__(self):
        self._watches: dict[str, WatchVariable] = {}
        self._breakpoints: dict[str, Breakpoint] = {}
        self._enabled = True
        self._visible = False
        self._paused = False
        self._current_frame = 0
        self._categories: set[str] = {"default"}
        self._on_breakpoint_hit: list[Callable[[Breakpoint], None]] = []
        self._update_interval = 0.0  # 0 = every frame
        self._last_update = 0.0

    @classmethod
    def get_instance(cls) -> "WatchWindow":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def show(self) -> None:
        self._visible = True

    def hide(self) -> None:
        self._visible = False

    def toggle(self) -> bool:
        """Toggle visibility. Returns new state."""
        self._visible = not self._visible
        return self._visible

    @property
    def is_visible(self) -> bool:
        return self._visible

    def pause(self) -> None:
        """Pause watch updates."""
        self._paused = True

    def resume(self) -> None:
        """Resume watch updates."""
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    def set_update_interval(self, interval: float) -> None:
        """Set update interval in seconds (0 = every frame)."""
        self._update_interval = max(0.0, interval)

    # Watch Management

    def add_watch(
        self,
        name: str,
        getter: Callable[[], Any],
        category: str = "default",
        format_string: str = "{value}",
        max_history: int = 100,
    ) -> WatchVariable:
        """Add a simple value watch."""
        watch = WatchVariable(
            name=name,
            getter=getter,
            category=category,
            format_string=format_string,
            max_history=max_history,
        )
        self._watches[name] = watch
        self._categories.add(category)
        return watch

    def add_conditional_watch(
        self,
        name: str,
        getter: Callable[[], Any],
        change_type: ValueChangeType = ValueChangeType.ANY,
        target_value: Any = None,
        on_trigger: Optional[Callable[[Any, Any], None]] = None,
        category: str = "default",
    ) -> ConditionalWatch:
        """Add a conditional watch."""
        watch = ConditionalWatch(
            name=name,
            getter=getter,
            change_type=change_type,
            target_value=target_value,
            on_trigger=on_trigger,
            category=category,
        )
        self._watches[name] = watch
        self._categories.add(category)
        return watch

    def add_property_watch(
        self,
        name: str,
        obj: Any,
        property_name: str,
        category: str = "default",
    ) -> WatchVariable:
        """Add a watch on an object property."""
        tracker = VariableTracker(obj, property_name)
        return self.add_watch(
            name=name,
            getter=tracker.get_value,
            category=category,
        )

    def remove_watch(self, name: str) -> Optional[WatchVariable]:
        """Remove a watch."""
        return self._watches.pop(name, None)

    def get_watch(self, name: str) -> Optional[WatchVariable]:
        """Get a watch by name."""
        return self._watches.get(name)

    def get_watches_by_category(self, category: str) -> list[WatchVariable]:
        """Get all watches in a category."""
        return [w for w in self._watches.values() if w.category == category]

    def clear_watches(self) -> None:
        """Clear all watches."""
        self._watches.clear()

    # Breakpoint Management

    def add_breakpoint(
        self,
        breakpoint_id: str,
        name: str,
        condition: Callable[[], bool],
        breakpoint_type: BreakpointType = BreakpointType.ALWAYS,
        action: Optional[Callable[[], None]] = None,
        log_message: str = "",
    ) -> Breakpoint:
        """Add a breakpoint."""
        breakpoint = Breakpoint(
            breakpoint_id=breakpoint_id,
            name=name,
            condition=condition,
            breakpoint_type=breakpoint_type,
            action=action,
            log_message=log_message,
        )
        self._breakpoints[breakpoint_id] = breakpoint
        return breakpoint

    def add_value_breakpoint(
        self,
        breakpoint_id: str,
        name: str,
        getter: Callable[[], Any],
        target_value: Any,
        comparison: ValueChangeType = ValueChangeType.EQUALS,
    ) -> Breakpoint:
        """Add a breakpoint that triggers on value comparison."""
        def condition() -> bool:
            value = getter()
            if comparison == ValueChangeType.EQUALS:
                return value == target_value
            elif comparison == ValueChangeType.NOT_EQUALS:
                return value != target_value
            elif comparison == ValueChangeType.GREATER:
                return value > target_value
            elif comparison == ValueChangeType.LESS:
                return value < target_value
            return False

        return self.add_breakpoint(
            breakpoint_id=breakpoint_id,
            name=name,
            condition=condition,
            breakpoint_type=BreakpointType.CONDITIONAL,
        )

    def remove_breakpoint(self, breakpoint_id: str) -> Optional[Breakpoint]:
        """Remove a breakpoint."""
        return self._breakpoints.pop(breakpoint_id, None)

    def get_breakpoint(self, breakpoint_id: str) -> Optional[Breakpoint]:
        """Get a breakpoint by ID."""
        return self._breakpoints.get(breakpoint_id)

    def enable_breakpoint(self, breakpoint_id: str) -> bool:
        """Enable a breakpoint."""
        bp = self._breakpoints.get(breakpoint_id)
        if bp:
            bp.enabled = True
            return True
        return False

    def disable_breakpoint(self, breakpoint_id: str) -> bool:
        """Disable a breakpoint."""
        bp = self._breakpoints.get(breakpoint_id)
        if bp:
            bp.enabled = False
            return True
        return False

    def clear_breakpoints(self) -> None:
        """Clear all breakpoints."""
        self._breakpoints.clear()

    def on_breakpoint_hit(self, callback: Callable[[Breakpoint], None]) -> None:
        """Register callback for when breakpoints are hit."""
        self._on_breakpoint_hit.append(callback)

    # Update

    def update(self) -> list[Breakpoint]:
        """Update all watches and check breakpoints. Returns triggered breakpoints."""
        if not self._enabled or self._paused:
            return []

        current_time = time.time()
        if self._update_interval > 0:
            if (current_time - self._last_update) < self._update_interval:
                return []
            self._last_update = current_time

        self._current_frame += 1
        triggered_breakpoints = []

        # Update watches
        for watch in self._watches.values():
            watch.update(self._current_frame)

        # Check breakpoints
        for bp in self._breakpoints.values():
            if bp.check():
                triggered_breakpoints.append(bp)
                for callback in self._on_breakpoint_hit:
                    callback(bp)

        return triggered_breakpoints

    # Query

    @property
    def watch_count(self) -> int:
        return len(self._watches)

    @property
    def breakpoint_count(self) -> int:
        return len(self._breakpoints)

    @property
    def categories(self) -> set[str]:
        return self._categories.copy()

    @property
    def current_frame(self) -> int:
        return self._current_frame

    def get_all_values(self) -> dict[str, Any]:
        """Get current values of all watches."""
        return {name: watch.get_value() for name, watch in self._watches.items()}

    def get_triggered_breakpoints(self) -> list[Breakpoint]:
        """Get all currently triggered breakpoints."""
        return [bp for bp in self._breakpoints.values() if bp.triggered]

    def reset_all_breakpoints(self) -> None:
        """Reset all breakpoint states."""
        for bp in self._breakpoints.values():
            bp.reset()

    # Rendering

    def render(self) -> dict[str, Any]:
        """Render watch window data."""
        if not self._enabled or not self._visible:
            return {}

        watches_data = []
        for name, watch in self._watches.items():
            value = watch.get_value()
            watches_data.append({
                "name": name,
                "value": value,
                "formatted_value": watch.format_value(value),
                "category": watch.category,
                "enabled": watch.enabled,
                "update_count": watch.update_count,
                "history_length": len(watch.history),
                "type": watch.watch_type.name,
                "triggered": watch._triggered if isinstance(watch, ConditionalWatch) else False,
            })

        breakpoints_data = []
        for bp_id, bp in self._breakpoints.items():
            breakpoints_data.append({
                "id": bp_id,
                "name": bp.name,
                "type": bp.breakpoint_type.name,
                "enabled": bp.enabled,
                "hit_count": bp.hit_count,
                "triggered": bp.triggered,
                "log_message": bp.log_message,
            })

        return {
            "type": "watch_window",
            "visible": self._visible,
            "paused": self._paused,
            "current_frame": self._current_frame,
            "categories": list(self._categories),
            "watches": watches_data,
            "breakpoints": breakpoints_data,
        }
