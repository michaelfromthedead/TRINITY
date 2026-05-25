"""
Inspector - Object visualization and editing system. Core Foundation Layer 3.
Provides views, navigation, and type-appropriate widgets for any object.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from foundation.constants import INDENT_SPACES
from foundation.mirror import mirror, FieldInfo

try:
    from foundation.serializer import to_dict as _to_dict
except ImportError:
    _to_dict = None


@runtime_checkable
class View(Protocol):
    """Protocol for object views."""
    name: str
    def can_render(self, obj: Any) -> bool: ...
    def render(self, obj: Any, ctx: 'UIContext') -> str: ...


@runtime_checkable
class UIContext(Protocol):
    """Protocol for UI rendering context."""
    def text(self, content: str) -> None: ...
    def label(self, text: str, value: Any) -> None: ...
    def input(self, name: str, value: Any, on_change: Callable[[Any], None]) -> None: ...
    def button(self, label: str, on_click: Callable[[], None]) -> None: ...
    def group(self, title: str) -> 'UIContext': ...


class TextUIContext:
    """Simple text-based UI context for testing and CLI output."""
    __slots__ = ("_lines", "_indent", "_inputs", "_buttons")

    def __init__(self, indent: int = 0):
        self._lines: list[str] = []
        self._indent = indent
        self._inputs: dict[str, tuple[Any, Callable]] = {}
        self._buttons: dict[str, Callable] = {}

    def _prefix(self) -> str: return " " * INDENT_SPACES * self._indent
    def text(self, content: str) -> None: self._lines.append(f"{self._prefix()}{content}")
    def label(self, text: str, value: Any) -> None: self._lines.append(f"{self._prefix()}{text}: {value!r}")

    def input(self, name: str, value: Any, on_change: Callable[[Any], None]) -> None:
        self._lines.append(f"{self._prefix()}[{name}]: {value!r}")
        self._inputs[name] = (value, on_change)

    def button(self, label: str, on_click: Callable[[], None]) -> None:
        self._lines.append(f"{self._prefix()}[{label}]")
        self._buttons[label] = on_click

    def group(self, title: str) -> 'TextUIContext':
        self._lines.append(f"{self._prefix()}{title}:")
        return TextUIContext(self._indent + 1)

    def get_output(self) -> str: return "\n".join(self._lines)

    def set_value(self, name: str, value: Any) -> None:
        if name in self._inputs: self._inputs[name][1](value)

    def click_button(self, label: str) -> None:
        if label in self._buttons: self._buttons[label]()


class FieldsView:
    """Default view - shows editable fields with type-appropriate widgets."""
    name: str = "Fields"

    def can_render(self, obj: Any) -> bool:
        return not isinstance(obj, (type, str, int, float, bool, type(None), list, tuple, dict, set, frozenset))

    def render(self, obj: Any, ctx: UIContext) -> str:
        m = mirror(obj)
        ctx.text(f"=== {m.type_name} ===")
        for name, fi in m.fields.items():
            try: value = m.get(name)
            except (AttributeError, KeyError): value = "<unset>"
            if fi.metadata.get("hidden"): continue
            if fi.metadata.get("readonly"): ctx.label(name, value)
            else: ctx.input(name, value, lambda v, n=name: m.set(n, v))
        return ctx.get_output() if hasattr(ctx, "get_output") else ""


class RawView:
    """Shows obj.__dict__ as tree."""
    name: str = "Raw"

    def can_render(self, obj: Any) -> bool:
        return hasattr(obj, "__dict__") or hasattr(obj, "__slots__")

    def render(self, obj: Any, ctx: UIContext) -> str:
        ctx.text(f"=== Raw: {type(obj).__name__} ===")
        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items(): ctx.label(k, v)
        if hasattr(obj, "__slots__"):
            slots = obj.__slots__
            for s in ((slots,) if isinstance(slots, str) else slots):
                if hasattr(obj, s): ctx.label(s, getattr(obj, s))
        return ctx.get_output() if hasattr(ctx, "get_output") else ""


class JSONView:
    """Shows serialized form (uses Serializer)."""
    name: str = "JSON"

    def can_render(self, obj: Any) -> bool: return _to_dict is not None

    def render(self, obj: Any, ctx: UIContext) -> str:
        ctx.text(f"=== JSON: {type(obj).__name__} ===")
        if _to_dict:
            try:
                for line in json.dumps(_to_dict(obj), indent=2, default=str).split("\n"):
                    ctx.text(line)
            except Exception as e: ctx.text(f"Serialization error: {e}")
        return ctx.get_output() if hasattr(ctx, "get_output") else ""


class CollectionView:
    """Table view for lists/dicts."""
    name: str = "Collection"

    def can_render(self, obj: Any) -> bool:
        return isinstance(obj, (list, tuple, dict, set, frozenset))

    def render(self, obj: Any, ctx: UIContext) -> str:
        ctx.text(f"=== {type(obj).__name__} ({len(obj)} items) ===")
        if isinstance(obj, dict):
            for k, v in obj.items(): ctx.label(str(k), v)
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj): ctx.label(f"[{i}]", item)
        else:  # set, frozenset
            for i, item in enumerate(obj): ctx.label(f"({i})", item)
        return ctx.get_output() if hasattr(ctx, "get_output") else ""


@dataclass
class HistoryEntry:
    """An entry in the navigation history."""
    obj: Any
    view_name: Optional[str] = None


class InspectorPanel:
    """Panel for inspecting a single object with navigation history."""
    __slots__ = ("_inspector", "_target", "_current_view", "_history", "_history_pos")

    def __init__(self, inspector: 'Inspector', target: Any):
        self._inspector = inspector
        self._target = target
        self._history: list[HistoryEntry] = [HistoryEntry(target)]
        self._history_pos: int = 0
        views = self._inspector.get_views(target)
        self._current_view: Optional[View] = views[0] if views else None

    @property
    def target(self) -> Any: return self._target
    @property
    def current_view(self) -> Optional[View]: return self._current_view
    @property
    def views(self) -> list[View]: return self._inspector.get_views(self._target)
    @property
    def history(self) -> list[HistoryEntry]: return self._history.copy()
    @property
    def can_go_back(self) -> bool: return self._history_pos > 0
    @property
    def can_go_forward(self) -> bool: return self._history_pos < len(self._history) - 1

    def set_view(self, view_name: str) -> bool:
        for view in self.views:
            if view.name == view_name:
                self._current_view = view
                return True
        return False

    def navigate_to(self, obj: Any) -> None:
        self._history = self._history[:self._history_pos + 1]
        self._history.append(HistoryEntry(obj))
        self._history_pos = len(self._history) - 1
        self._target = obj
        views = self._inspector.get_views(obj)
        self._current_view = views[0] if views else None

    def back(self) -> bool:
        if not self.can_go_back: return False
        self._history_pos -= 1
        entry = self._history[self._history_pos]
        self._target = entry.obj
        if entry.view_name: self.set_view(entry.view_name)
        else:
            views = self._inspector.get_views(self._target)
            self._current_view = views[0] if views else None
        return True

    def forward(self) -> bool:
        if not self.can_go_forward: return False
        self._history_pos += 1
        entry = self._history[self._history_pos]
        self._target = entry.obj
        if entry.view_name: self.set_view(entry.view_name)
        else:
            views = self._inspector.get_views(self._target)
            self._current_view = views[0] if views else None
        return True

    def render(self, ctx: Optional[UIContext] = None) -> str:
        if ctx is None: ctx = TextUIContext()
        if self._current_view is None:
            ctx.text(f"No view available for {type(self._target).__name__}")
            return ctx.get_output() if hasattr(ctx, "get_output") else ""
        return self._current_view.render(self._target, ctx)


def _widget_bool(name: str, value: Any, ctx: UIContext, on_change: Callable) -> None:
    ctx.input(name, f"[{'x' if value else ' '}]", on_change)

def _widget_number(name: str, value: Any, ctx: UIContext, on_change: Callable, meta: dict) -> None:
    r = meta.get("range")
    ctx.input(name, f"{value} (range: {r[0]}-{r[1]})" if r and len(r) >= 2 else value, on_change)

def _widget_str(name: str, value: Any, ctx: UIContext, on_change: Callable, meta: dict) -> None:
    ctx.input(name, f"[multiline] {value!r}" if meta.get("multiline") else value, on_change)

def _widget_enum(name: str, value: Any, ctx: UIContext, on_change: Callable) -> None:
    ctx.input(name, f"{value.name} (choices: {', '.join(e.name for e in type(value))})", on_change)

def _widget_list(name: str, value: Any, ctx: UIContext, on_change: Callable) -> None:
    ctx.label(name, f"[{type(value).__name__}: {len(value)} items]")

def _widget_object(name: str, value: Any, ctx: UIContext, navigate: Callable[[Any], None]) -> None:
    ctx.button(f"{name}: {type(value).__name__} ->", lambda: navigate(value))


class Inspector:
    """Main inspector class - provides views and widget mapping for any object."""
    __slots__ = ("_views", "_widgets")

    def __init__(self):
        self._views: list[View] = [CollectionView(), FieldsView(), RawView(), JSONView()]
        self._widgets: dict[type, Callable] = {
            bool: _widget_bool, int: _widget_number, float: _widget_number,
            str: _widget_str, list: _widget_list, tuple: _widget_list, dict: _widget_list,
        }

    def inspect(self, obj: Any) -> InspectorPanel:
        """Create an inspector panel for an object."""
        return InspectorPanel(self, obj)

    def register_view(self, view: View) -> None:
        """Register a custom view (takes priority over built-ins)."""
        self._views.insert(0, view)

    def register_widget(self, type_: type, widget_factory: Callable) -> None:
        """Register a custom widget factory for a type."""
        self._widgets[type_] = widget_factory

    def get_views(self, obj: Any) -> list[View]:
        """Get all views that can render the given object."""
        return [v for v in self._views if v.can_render(obj)]

    def get_widget(self, field_type: type) -> Optional[Callable]:
        """Get the widget factory for a given type."""
        if field_type in self._widgets: return self._widgets[field_type]
        if isinstance(field_type, type) and issubclass(field_type, Enum): return _widget_enum
        for t, w in self._widgets.items():
            if isinstance(field_type, type) and issubclass(field_type, t): return w
        return _widget_object


inspector = Inspector()

__all__ = [
    "View", "UIContext", "TextUIContext", "FieldsView", "RawView", "JSONView",
    "CollectionView", "InspectorPanel", "Inspector", "inspector", "HistoryEntry",
]
