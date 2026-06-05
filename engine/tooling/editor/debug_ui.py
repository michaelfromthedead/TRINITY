"""
Debug UI - ImGui/egui-style debug interface framework.

Provides:
- DebugUI class wrapping imgui/egui-like Python bindings
- AutoInspector generating UI from TrinityMirror introspection
- Widget registry for common types (int, float, vec3, color, enum)
- PropertyPanel for editing component properties
- CollapsibleSection for organizing debug panels
- Integration with editor modes

This implementation provides a mock interface that can be replaced
with real imgui/egui bindings when available.
"""
from __future__ import annotations

import uuid
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from engine.tooling.editor.app_shell import editor, reloadable

# Try to import Mirror for introspection
try:
    from foundation.mirror import ObjectMirror, ClassMirror, FieldInfo
    MIRROR_AVAILABLE = True
except ImportError:
    MIRROR_AVAILABLE = False
    ObjectMirror = None
    ClassMirror = None
    FieldInfo = None


T = TypeVar("T")


# =============================================================================
# Core Types and Enums
# =============================================================================


class WidgetType(Enum):
    """Types of debug UI widgets."""
    LABEL = auto()
    TEXT_INPUT = auto()
    INT_SLIDER = auto()
    FLOAT_SLIDER = auto()
    INT_INPUT = auto()
    FLOAT_INPUT = auto()
    CHECKBOX = auto()
    DROPDOWN = auto()
    COLOR_PICKER = auto()
    VEC2_INPUT = auto()
    VEC3_INPUT = auto()
    VEC4_INPUT = auto()
    BUTTON = auto()
    SEPARATOR = auto()
    TREE_NODE = auto()
    COLLAPSING_HEADER = auto()
    CUSTOM = auto()


class UIState(Enum):
    """State of UI elements."""
    NORMAL = auto()
    HOVERED = auto()
    ACTIVE = auto()
    DISABLED = auto()
    FOCUSED = auto()


@dataclass
class Vec2:
    """2D vector for UI positions and sizes."""
    x: float = 0.0
    y: float = 0.0

    def __iter__(self):
        yield self.x
        yield self.y

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Vec3:
    """3D vector for positions and directions."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Vec4:
    """4D vector for colors and quaternions."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
        yield self.w

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)


@dataclass
class Color:
    """RGBA color representation."""
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0

    def __iter__(self):
        yield self.r
        yield self.g
        yield self.b
        yield self.a

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)

    def to_hex(self) -> str:
        """Convert to hex string."""
        return "#{:02x}{:02x}{:02x}{:02x}".format(
            int(self.r * 255),
            int(self.g * 255),
            int(self.b * 255),
            int(self.a * 255),
        )

    @classmethod
    def from_hex(cls, hex_str: str) -> "Color":
        """Create color from hex string."""
        hex_str = hex_str.lstrip("#")
        if len(hex_str) == 6:
            r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
            return cls(r / 255.0, g / 255.0, b / 255.0, 1.0)
        elif len(hex_str) == 8:
            r, g, b, a = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16), int(hex_str[6:8], 16)
            return cls(r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        raise ValueError(f"Invalid hex color: {hex_str}")


@dataclass
class WidgetStyle:
    """Styling configuration for widgets."""
    padding: Vec2 = field(default_factory=lambda: Vec2(4.0, 2.0))
    margin: Vec2 = field(default_factory=lambda: Vec2(0.0, 2.0))
    border_radius: float = 2.0
    font_size: float = 14.0
    background_color: Optional[Color] = None
    text_color: Optional[Color] = None
    border_color: Optional[Color] = None
    hover_color: Optional[Color] = None
    active_color: Optional[Color] = None


@dataclass
class WidgetConfig:
    """Configuration for widget creation."""
    label: str = ""
    tooltip: str = ""
    width: Optional[float] = None
    height: Optional[float] = None
    enabled: bool = True
    visible: bool = True
    style: Optional[WidgetStyle] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Widget Base Classes
# =============================================================================


@editor(category="DebugUI")
@reloadable()
class Widget(ABC):
    """Base class for all debug UI widgets."""
    __slots__ = (
        "id", "widget_type", "config", "state", "value", "_parent_ref",
        "on_change", "on_click", "on_hover", "_dirty", "_last_value", "__weakref__"
    )

    _id_counter = 0

    def __init__(
        self,
        widget_type: WidgetType,
        config: Optional[WidgetConfig] = None,
    ):
        Widget._id_counter += 1
        self.id = f"widget_{Widget._id_counter}"
        self.widget_type = widget_type
        self.config = config or WidgetConfig()
        self.state = UIState.NORMAL
        self.value: Any = None
        self._parent_ref: Optional[weakref.ref] = None
        self.on_change: Optional[Callable[[Any], None]] = None
        self.on_click: Optional[Callable[[], None]] = None
        self.on_hover: Optional[Callable[[], None]] = None
        self._dirty = False
        self._last_value: Any = None

    @property
    def label(self) -> str:
        return self.config.label

    @label.setter
    def label(self, value: str):
        self.config.label = value

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @enabled.setter
    def enabled(self, value: bool):
        self.config.enabled = value
        if not value:
            self.state = UIState.DISABLED

    @property
    def visible(self) -> bool:
        return self.config.visible

    @visible.setter
    def visible(self, value: bool):
        self.config.visible = value

    @property
    def parent(self) -> Optional["Widget"]:
        return self._parent_ref() if self._parent_ref else None

    def set_value(self, value: Any) -> bool:
        """Set widget value. Returns True if changed."""
        if value != self.value:
            self._last_value = self.value
            self.value = value
            self._dirty = True
            if self.on_change:
                self.on_change(value)
            return True
        return False

    def is_dirty(self) -> bool:
        """Check if value changed since last render."""
        return self._dirty

    def clear_dirty(self) -> None:
        """Clear dirty flag."""
        self._dirty = False

    @abstractmethod
    def render(self, ctx: "DebugUIContext") -> None:
        """Render the widget."""
        pass

    def handle_input(self, event: "UIEvent") -> bool:
        """Handle input event. Returns True if consumed."""
        return False


@editor(category="DebugUI")
@reloadable()
class ContainerWidget(Widget):
    """Widget that can contain other widgets."""
    __slots__ = ("_children", "_layout")

    def __init__(
        self,
        widget_type: WidgetType,
        config: Optional[WidgetConfig] = None,
    ):
        super().__init__(widget_type, config)
        self._children: List[Widget] = []
        self._layout = "vertical"

    @property
    def children(self) -> List[Widget]:
        return list(self._children)

    def add_child(self, widget: Widget) -> None:
        """Add a child widget."""
        widget._parent_ref = weakref.ref(self)
        self._children.append(widget)

    def remove_child(self, widget: Widget) -> bool:
        """Remove a child widget."""
        if widget in self._children:
            self._children.remove(widget)
            widget._parent_ref = None
            return True
        return False

    def clear_children(self) -> None:
        """Remove all children."""
        for child in self._children:
            child._parent_ref = None
        self._children.clear()

    def find_child(self, widget_id: str) -> Optional[Widget]:
        """Find child by ID recursively."""
        for child in self._children:
            if child.id == widget_id:
                return child
            if isinstance(child, ContainerWidget):
                found = child.find_child(widget_id)
                if found:
                    return found
        return None

    def render(self, ctx: "DebugUIContext") -> None:
        """Render container and all children."""
        if not self.visible:
            return
        for child in self._children:
            if child.visible:
                child.render(ctx)


# =============================================================================
# Concrete Widget Implementations
# =============================================================================


@editor(category="DebugUI")
@reloadable()
class LabelWidget(Widget):
    """Simple text label widget."""

    def __init__(self, text: str = "", config: Optional[WidgetConfig] = None):
        cfg = config or WidgetConfig()
        cfg.label = text
        super().__init__(WidgetType.LABEL, cfg)
        self.value = text

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        ctx.draw_text(self.value, ctx.cursor_pos)


@editor(category="DebugUI")
@reloadable()
class TextInputWidget(Widget):
    """Single-line text input widget."""
    __slots__ = ("max_length", "placeholder", "password")

    def __init__(
        self,
        label: str = "",
        initial_value: str = "",
        max_length: int = 256,
        placeholder: str = "",
        password: bool = False,
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = label
        super().__init__(WidgetType.TEXT_INPUT, cfg)
        self.value = initial_value
        self.max_length = max_length
        self.placeholder = placeholder
        self.password = password

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        ctx.draw_text_input(
            self.id,
            self.label,
            self.value,
            self.max_length,
            self.placeholder,
            self.password,
        )


@editor(category="DebugUI")
@reloadable()
class IntSliderWidget(Widget):
    """Integer slider widget."""
    __slots__ = ("min_value", "max_value", "step")

    def __init__(
        self,
        label: str = "",
        initial_value: int = 0,
        min_value: int = 0,
        max_value: int = 100,
        step: int = 1,
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = label
        super().__init__(WidgetType.INT_SLIDER, cfg)
        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step

    def set_value(self, value: Any) -> bool:
        """Set value with clamping."""
        clamped = max(self.min_value, min(self.max_value, int(value)))
        return super().set_value(clamped)

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        ctx.draw_slider_int(
            self.id,
            self.label,
            self.value,
            self.min_value,
            self.max_value,
        )


@editor(category="DebugUI")
@reloadable()
class FloatSliderWidget(Widget):
    """Float slider widget."""
    __slots__ = ("min_value", "max_value", "step", "precision")

    def __init__(
        self,
        label: str = "",
        initial_value: float = 0.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
        step: float = 0.01,
        precision: int = 3,
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = label
        super().__init__(WidgetType.FLOAT_SLIDER, cfg)
        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.precision = precision

    def set_value(self, value: Any) -> bool:
        """Set value with clamping."""
        clamped = max(self.min_value, min(self.max_value, float(value)))
        return super().set_value(round(clamped, self.precision))

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        ctx.draw_slider_float(
            self.id,
            self.label,
            self.value,
            self.min_value,
            self.max_value,
            self.precision,
        )


@editor(category="DebugUI")
@reloadable()
class IntInputWidget(Widget):
    """Integer input field widget."""
    __slots__ = ("min_value", "max_value", "step")

    def __init__(
        self,
        label: str = "",
        initial_value: int = 0,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        step: int = 1,
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = label
        super().__init__(WidgetType.INT_INPUT, cfg)
        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step

    def set_value(self, value: Any) -> bool:
        """Set value with optional clamping."""
        v = int(value)
        if self.min_value is not None:
            v = max(self.min_value, v)
        if self.max_value is not None:
            v = min(self.max_value, v)
        return super().set_value(v)

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        ctx.draw_input_int(self.id, self.label, self.value, self.step)


@editor(category="DebugUI")
@reloadable()
class FloatInputWidget(Widget):
    """Float input field widget."""
    __slots__ = ("min_value", "max_value", "step", "precision")

    def __init__(
        self,
        label: str = "",
        initial_value: float = 0.0,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: float = 0.1,
        precision: int = 3,
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = label
        super().__init__(WidgetType.FLOAT_INPUT, cfg)
        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.precision = precision

    def set_value(self, value: Any) -> bool:
        """Set value with optional clamping."""
        v = float(value)
        if self.min_value is not None:
            v = max(self.min_value, v)
        if self.max_value is not None:
            v = min(self.max_value, v)
        return super().set_value(round(v, self.precision))

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        ctx.draw_input_float(
            self.id, self.label, self.value, self.step, self.precision
        )


@editor(category="DebugUI")
@reloadable()
class CheckboxWidget(Widget):
    """Boolean checkbox widget."""

    def __init__(
        self,
        label: str = "",
        initial_value: bool = False,
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = label
        super().__init__(WidgetType.CHECKBOX, cfg)
        self.value = initial_value

    def toggle(self) -> None:
        """Toggle the checkbox value."""
        self.set_value(not self.value)

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        ctx.draw_checkbox(self.id, self.label, self.value)


@editor(category="DebugUI")
@reloadable()
class DropdownWidget(Widget):
    """Dropdown/combo box widget."""
    __slots__ = ("options", "_selected_index")

    def __init__(
        self,
        label: str = "",
        options: Optional[List[str]] = None,
        selected_index: int = 0,
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = label
        super().__init__(WidgetType.DROPDOWN, cfg)
        self.options = options or []
        self._selected_index = selected_index
        if self.options and 0 <= selected_index < len(self.options):
            self.value = self.options[selected_index]
        else:
            self.value = None

    @property
    def selected_index(self) -> int:
        return self._selected_index

    @selected_index.setter
    def selected_index(self, index: int) -> None:
        if 0 <= index < len(self.options):
            self._selected_index = index
            self.set_value(self.options[index])

    def set_options(self, options: List[str], keep_selection: bool = True) -> None:
        """Update options list."""
        old_value = self.value
        self.options = options
        if keep_selection and old_value in options:
            self._selected_index = options.index(old_value)
            self.value = old_value
        elif options:
            self._selected_index = 0
            self.set_value(options[0])
        else:
            self._selected_index = -1
            self.set_value(None)

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        ctx.draw_dropdown(
            self.id, self.label, self.options, self._selected_index
        )


@editor(category="DebugUI")
@reloadable()
class ColorPickerWidget(Widget):
    """RGBA color picker widget."""
    __slots__ = ("show_alpha", "show_hex")

    def __init__(
        self,
        label: str = "",
        initial_color: Optional[Color] = None,
        show_alpha: bool = True,
        show_hex: bool = True,
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = label
        super().__init__(WidgetType.COLOR_PICKER, cfg)
        self.value = initial_color or Color()
        self.show_alpha = show_alpha
        self.show_hex = show_hex

    def set_rgb(self, r: float, g: float, b: float) -> None:
        """Set RGB values keeping alpha."""
        new_color = Color(r, g, b, self.value.a)
        self.set_value(new_color)

    def set_rgba(self, r: float, g: float, b: float, a: float) -> None:
        """Set full RGBA values."""
        self.set_value(Color(r, g, b, a))

    def set_hex(self, hex_str: str) -> None:
        """Set color from hex string."""
        self.set_value(Color.from_hex(hex_str))

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        ctx.draw_color_picker(
            self.id, self.label, self.value, self.show_alpha, self.show_hex
        )


@editor(category="DebugUI")
@reloadable()
class Vec2InputWidget(Widget):
    """2D vector input widget."""
    __slots__ = ("labels", "min_value", "max_value", "step", "precision")

    def __init__(
        self,
        label: str = "",
        initial_value: Optional[Vec2] = None,
        labels: Tuple[str, str] = ("X", "Y"),
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: float = 0.1,
        precision: int = 3,
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = label
        super().__init__(WidgetType.VEC2_INPUT, cfg)
        self.value = initial_value or Vec2()
        self.labels = labels
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.precision = precision

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        ctx.draw_vec2_input(
            self.id, self.label, self.value, self.labels, self.step, self.precision
        )


@editor(category="DebugUI")
@reloadable()
class Vec3InputWidget(Widget):
    """3D vector input widget."""
    __slots__ = ("labels", "min_value", "max_value", "step", "precision")

    def __init__(
        self,
        label: str = "",
        initial_value: Optional[Vec3] = None,
        labels: Tuple[str, str, str] = ("X", "Y", "Z"),
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: float = 0.1,
        precision: int = 3,
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = label
        super().__init__(WidgetType.VEC3_INPUT, cfg)
        self.value = initial_value or Vec3()
        self.labels = labels
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.precision = precision

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        ctx.draw_vec3_input(
            self.id, self.label, self.value, self.labels, self.step, self.precision
        )


@editor(category="DebugUI")
@reloadable()
class Vec4InputWidget(Widget):
    """4D vector input widget."""
    __slots__ = ("labels", "min_value", "max_value", "step", "precision")

    def __init__(
        self,
        label: str = "",
        initial_value: Optional[Vec4] = None,
        labels: Tuple[str, str, str, str] = ("X", "Y", "Z", "W"),
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: float = 0.1,
        precision: int = 3,
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = label
        super().__init__(WidgetType.VEC4_INPUT, cfg)
        self.value = initial_value or Vec4()
        self.labels = labels
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.precision = precision

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        ctx.draw_vec4_input(
            self.id, self.label, self.value, self.labels, self.step, self.precision
        )


@editor(category="DebugUI")
@reloadable()
class ButtonWidget(Widget):
    """Clickable button widget."""
    __slots__ = ("icon",)

    def __init__(
        self,
        label: str = "",
        icon: str = "",
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = label
        super().__init__(WidgetType.BUTTON, cfg)
        self.icon = icon
        self.value = False  # True when clicked this frame

    def click(self) -> None:
        """Programmatically click the button."""
        self.value = True
        if self.on_click:
            self.on_click()

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        self.value = False  # Reset each frame
        ctx.draw_button(self.id, self.label, self.icon)


@editor(category="DebugUI")
@reloadable()
class SeparatorWidget(Widget):
    """Visual separator widget."""
    __slots__ = ("horizontal",)

    def __init__(self, horizontal: bool = True, config: Optional[WidgetConfig] = None):
        super().__init__(WidgetType.SEPARATOR, config)
        self.horizontal = horizontal

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        ctx.draw_separator(self.horizontal)


# =============================================================================
# Collapsible Section
# =============================================================================


@editor(category="DebugUI")
@reloadable()
class CollapsibleSection(ContainerWidget):
    """Collapsible section for organizing debug panels."""
    __slots__ = ("expanded", "default_expanded", "icon")

    def __init__(
        self,
        title: str,
        expanded: bool = True,
        icon: str = "",
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = title
        super().__init__(WidgetType.COLLAPSING_HEADER, cfg)
        self.expanded = expanded
        self.default_expanded = expanded
        self.icon = icon

    @property
    def title(self) -> str:
        return self.config.label

    @title.setter
    def title(self, value: str):
        self.config.label = value

    def toggle(self) -> None:
        """Toggle expanded state."""
        self.expanded = not self.expanded

    def expand(self) -> None:
        """Expand the section."""
        self.expanded = True

    def collapse(self) -> None:
        """Collapse the section."""
        self.expanded = False

    def reset(self) -> None:
        """Reset to default expanded state."""
        self.expanded = self.default_expanded

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        # Draw header
        clicked = ctx.draw_collapsing_header(
            self.id, self.title, self.expanded, self.icon
        )
        if clicked:
            self.toggle()
        # Draw children if expanded
        if self.expanded:
            ctx.indent()
            for child in self._children:
                if child.visible:
                    child.render(ctx)
            ctx.unindent()


# =============================================================================
# Property Panel
# =============================================================================


@dataclass
class PropertyBinding:
    """Binds a widget to an object property."""
    widget: Widget
    target: Any
    property_name: str
    getter: Optional[Callable[[], Any]] = None
    setter: Optional[Callable[[Any], None]] = None
    readonly: bool = False

    def sync_from_target(self) -> None:
        """Update widget value from target."""
        if self.getter:
            value = self.getter()
        else:
            value = getattr(self.target, self.property_name, None)
        self.widget.set_value(value)

    def sync_to_target(self) -> None:
        """Update target from widget value."""
        if self.readonly:
            return
        if self.setter:
            self.setter(self.widget.value)
        else:
            setattr(self.target, self.property_name, self.widget.value)


@editor(category="DebugUI")
@reloadable()
class PropertyPanel(ContainerWidget):
    """Panel for editing object properties."""
    __slots__ = ("_target", "_bindings", "_auto_sync", "title")

    def __init__(
        self,
        title: str = "Properties",
        target: Any = None,
        auto_sync: bool = True,
        config: Optional[WidgetConfig] = None,
    ):
        cfg = config or WidgetConfig()
        cfg.label = title
        super().__init__(WidgetType.TREE_NODE, cfg)
        self.title = title
        self._target = target
        self._bindings: Dict[str, PropertyBinding] = {}
        self._auto_sync = auto_sync

    @property
    def target(self) -> Any:
        return self._target

    @target.setter
    def target(self, value: Any) -> None:
        self._target = value
        self.rebuild()

    def add_property(
        self,
        name: str,
        widget: Widget,
        getter: Optional[Callable[[], Any]] = None,
        setter: Optional[Callable[[Any], None]] = None,
        readonly: bool = False,
    ) -> PropertyBinding:
        """Add a property binding."""
        binding = PropertyBinding(
            widget=widget,
            target=self._target,
            property_name=name,
            getter=getter,
            setter=setter,
            readonly=readonly,
        )
        self._bindings[name] = binding
        self.add_child(widget)

        if not readonly:
            original_on_change = widget.on_change
            def on_value_change(value: Any) -> None:
                binding.sync_to_target()
                if original_on_change:
                    original_on_change(value)
            widget.on_change = on_value_change

        return binding

    def remove_property(self, name: str) -> bool:
        """Remove a property binding."""
        if name in self._bindings:
            binding = self._bindings.pop(name)
            self.remove_child(binding.widget)
            return True
        return False

    def get_binding(self, name: str) -> Optional[PropertyBinding]:
        """Get binding by property name."""
        return self._bindings.get(name)

    def sync_all_from_target(self) -> None:
        """Sync all widgets from target values."""
        for binding in self._bindings.values():
            binding.sync_from_target()

    def sync_all_to_target(self) -> None:
        """Sync all widget values to target."""
        for binding in self._bindings.values():
            if not binding.readonly:
                binding.sync_to_target()

    def rebuild(self) -> None:
        """Rebuild property panel for current target."""
        self.clear_children()
        self._bindings.clear()

    def render(self, ctx: "DebugUIContext") -> None:
        if not self.visible:
            return
        # Draw panel header
        ctx.draw_text(self.title, ctx.cursor_pos)
        ctx.draw_separator(True)
        # Sync values if auto_sync enabled
        if self._auto_sync:
            self.sync_all_from_target()
        # Draw all property widgets
        for child in self._children:
            if child.visible:
                child.render(ctx)


# =============================================================================
# Widget Registry
# =============================================================================


@editor(category="DebugUI")
@reloadable()
class WidgetRegistry:
    """Registry mapping types to widget factories."""
    __slots__ = ("_type_widgets", "_name_widgets", "_fallback")

    def __init__(self):
        self._type_widgets: Dict[type, Callable[..., Widget]] = {}
        self._name_widgets: Dict[str, Callable[..., Widget]] = {}
        self._fallback: Optional[Callable[..., Widget]] = None
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default type-to-widget mappings."""
        # Basic types
        self.register_type(int, self._create_int_widget)
        self.register_type(float, self._create_float_widget)
        self.register_type(str, self._create_string_widget)
        self.register_type(bool, self._create_bool_widget)

        # Vector types
        self.register_type(Vec2, self._create_vec2_widget)
        self.register_type(Vec3, self._create_vec3_widget)
        self.register_type(Vec4, self._create_vec4_widget)
        self.register_type(Color, self._create_color_widget)

        # By name patterns (for fields named "color", "position", etc.)
        self.register_name("color", self._create_color_widget)
        self.register_name("colour", self._create_color_widget)
        self.register_name("position", self._create_vec3_widget)
        self.register_name("rotation", self._create_vec3_widget)
        self.register_name("scale", self._create_vec3_widget)

    def register_type(
        self, type_cls: type, factory: Callable[..., Widget]
    ) -> None:
        """Register a widget factory for a type."""
        self._type_widgets[type_cls] = factory

    def register_name(
        self, name_pattern: str, factory: Callable[..., Widget]
    ) -> None:
        """Register a widget factory for field names matching pattern."""
        self._name_widgets[name_pattern.lower()] = factory

    def unregister_type(self, type_cls: type) -> bool:
        """Unregister a type mapping."""
        if type_cls in self._type_widgets:
            del self._type_widgets[type_cls]
            return True
        return False

    def unregister_name(self, name_pattern: str) -> bool:
        """Unregister a name pattern."""
        key = name_pattern.lower()
        if key in self._name_widgets:
            del self._name_widgets[key]
            return True
        return False

    def set_fallback(self, factory: Callable[..., Widget]) -> None:
        """Set fallback widget factory."""
        self._fallback = factory

    def get_widget_factory(
        self,
        field_type: Optional[type],
        field_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Callable[..., Widget]]:
        """Get appropriate widget factory for type/name."""
        metadata = metadata or {}

        # Check explicit widget hint in metadata
        if "widget" in metadata:
            widget_type = metadata["widget"]
            if widget_type in self._name_widgets:
                return self._name_widgets[widget_type]

        # Check name patterns
        name_lower = field_name.lower()
        for pattern, factory in self._name_widgets.items():
            if pattern in name_lower:
                return factory

        # Check type mappings
        if field_type is not None:
            # Handle enums specially
            if isinstance(field_type, type) and issubclass(field_type, Enum):
                return lambda **kw: self._create_enum_widget(field_type, **kw)

            if field_type in self._type_widgets:
                return self._type_widgets[field_type]

            # Check for generic types (List, Dict, etc.)
            origin = get_origin(field_type)
            if origin in self._type_widgets:
                return self._type_widgets[origin]

        return self._fallback

    def create_widget(
        self,
        field_type: Optional[type],
        field_name: str = "",
        label: str = "",
        initial_value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Widget]:
        """Create appropriate widget for field."""
        factory = self.get_widget_factory(field_type, field_name, metadata)
        if factory is None:
            return None

        kwargs = {
            "label": label or field_name,
            "initial_value": initial_value,
        }

        # Add metadata hints
        if metadata:
            if "range" in metadata:
                kwargs["min_value"] = metadata["range"][0]
                kwargs["max_value"] = metadata["range"][1]
            if "choices" in metadata:
                kwargs["options"] = metadata["choices"]
            if "readonly" in metadata:
                kwargs["readonly"] = metadata["readonly"]

        return factory(**kwargs)

    # Default widget factories
    def _create_int_widget(
        self,
        label: str = "",
        initial_value: int = 0,
        min_value: int = -1000000,
        max_value: int = 1000000,
        **kwargs,
    ) -> Widget:
        # Use slider if range is reasonable
        if max_value - min_value <= 1000:
            return IntSliderWidget(
                label, initial_value, min_value, max_value
            )
        return IntInputWidget(label, initial_value, min_value, max_value)

    def _create_float_widget(
        self,
        label: str = "",
        initial_value: float = 0.0,
        min_value: float = -1000000.0,
        max_value: float = 1000000.0,
        **kwargs,
    ) -> Widget:
        # Use slider if range is reasonable
        if max_value - min_value <= 100:
            return FloatSliderWidget(
                label, initial_value, min_value, max_value
            )
        return FloatInputWidget(label, initial_value)

    def _create_string_widget(
        self, label: str = "", initial_value: str = "", **kwargs
    ) -> Widget:
        return TextInputWidget(label, initial_value)

    def _create_bool_widget(
        self, label: str = "", initial_value: bool = False, **kwargs
    ) -> Widget:
        return CheckboxWidget(label, initial_value)

    def _create_vec2_widget(
        self, label: str = "", initial_value: Optional[Vec2] = None, **kwargs
    ) -> Widget:
        return Vec2InputWidget(label, initial_value)

    def _create_vec3_widget(
        self, label: str = "", initial_value: Optional[Vec3] = None, **kwargs
    ) -> Widget:
        return Vec3InputWidget(label, initial_value)

    def _create_vec4_widget(
        self, label: str = "", initial_value: Optional[Vec4] = None, **kwargs
    ) -> Widget:
        return Vec4InputWidget(label, initial_value)

    def _create_color_widget(
        self, label: str = "", initial_value: Optional[Color] = None, **kwargs
    ) -> Widget:
        return ColorPickerWidget(label, initial_value)

    def _create_enum_widget(
        self,
        enum_type: Type[Enum],
        label: str = "",
        initial_value: Optional[Enum] = None,
        **kwargs,
    ) -> Widget:
        options = [e.name for e in enum_type]
        selected = 0
        if initial_value is not None:
            try:
                selected = options.index(initial_value.name)
            except ValueError:
                pass
        return DropdownWidget(label, options, selected)


# =============================================================================
# Auto Inspector
# =============================================================================


@editor(category="DebugUI")
@reloadable()
class AutoInspector:
    """Generates UI from TrinityMirror introspection."""
    __slots__ = ("_registry", "_panels", "_excluded_fields", "_read_only_fields")

    def __init__(self, registry: Optional[WidgetRegistry] = None):
        self._registry = registry or WidgetRegistry()
        self._panels: Dict[int, PropertyPanel] = {}
        self._excluded_fields: set = {"_", "__"}
        self._read_only_fields: set = set()

    @property
    def registry(self) -> WidgetRegistry:
        return self._registry

    def exclude_field(self, field_name: str) -> None:
        """Exclude a field from auto-inspection."""
        self._excluded_fields.add(field_name)

    def include_field(self, field_name: str) -> None:
        """Re-include a previously excluded field."""
        self._excluded_fields.discard(field_name)

    def set_read_only(self, field_name: str, readonly: bool = True) -> None:
        """Mark a field as read-only."""
        if readonly:
            self._read_only_fields.add(field_name)
        else:
            self._read_only_fields.discard(field_name)

    def _should_include_field(self, field_name: str, field_info: Any) -> bool:
        """Check if field should be included in inspector."""
        # Check exclusions
        for pattern in self._excluded_fields:
            if field_name.startswith(pattern):
                return False

        # Check metadata for hidden flag
        if hasattr(field_info, "metadata"):
            if field_info.metadata.get("hidden", False):
                return False
            if field_info.metadata.get("transient", False):
                return False

        return True

    def _get_display_label(self, field_name: str, field_info: Any) -> str:
        """Get display label for field."""
        # Check for explicit label in metadata
        if hasattr(field_info, "metadata"):
            if "label" in field_info.metadata:
                return field_info.metadata["label"]

        # Convert snake_case to Title Case
        return field_name.replace("_", " ").title()

    def inspect_object(self, obj: Any, title: Optional[str] = None) -> PropertyPanel:
        """Create property panel for object using introspection."""
        obj_id = id(obj)

        # Return cached panel if exists
        if obj_id in self._panels:
            panel = self._panels[obj_id]
            panel._target = obj
            panel.sync_all_from_target()
            return panel

        # Create new panel
        class_name = type(obj).__name__
        panel = PropertyPanel(title or class_name, obj)

        # Get fields via Mirror if available
        if MIRROR_AVAILABLE and ObjectMirror is not None:
            try:
                mirror = ObjectMirror(obj)
                self._populate_from_mirror(panel, obj, mirror)
            except Exception:
                self._populate_from_annotations(panel, obj)
        else:
            self._populate_from_annotations(panel, obj)

        self._panels[obj_id] = panel
        return panel

    def _populate_from_mirror(
        self, panel: PropertyPanel, obj: Any, mirror: Any
    ) -> None:
        """Populate panel using Mirror introspection."""
        for field_name, field_info in mirror.fields.items():
            if not self._should_include_field(field_name, field_info):
                continue

            field_type = field_info.type
            metadata = field_info.metadata if hasattr(field_info, "metadata") else {}
            label = self._get_display_label(field_name, field_info)
            readonly = (
                field_name in self._read_only_fields
                or metadata.get("readonly", False)
            )

            try:
                value = getattr(obj, field_name, None)
            except (AttributeError, TypeError):
                value = None

            widget = self._registry.create_widget(
                field_type, field_name, label, value, metadata
            )

            if widget:
                panel.add_property(field_name, widget, readonly=readonly)

    def _populate_from_annotations(self, panel: PropertyPanel, obj: Any) -> None:
        """Populate panel using type annotations."""
        hints = {}
        try:
            hints = get_type_hints(type(obj))
        except (TypeError, NameError):
            pass

        # Also check __dict__ for runtime attributes
        attrs = set(hints.keys())
        if hasattr(obj, "__dict__"):
            attrs.update(obj.__dict__.keys())

        for field_name in sorted(attrs):
            if not self._should_include_field(field_name, None):
                continue

            field_type = hints.get(field_name)
            label = self._get_display_label(field_name, None)
            readonly = field_name in self._read_only_fields

            try:
                value = getattr(obj, field_name, None)
            except (AttributeError, TypeError):
                value = None

            # Infer type from value if annotation missing
            if field_type is None and value is not None:
                field_type = type(value)

            widget = self._registry.create_widget(
                field_type, field_name, label, value, {}
            )

            if widget:
                panel.add_property(field_name, widget, readonly=readonly)

    def inspect_class(self, cls: type, title: Optional[str] = None) -> PropertyPanel:
        """Create property panel for class definition."""
        panel = PropertyPanel(title or cls.__name__, None)

        if MIRROR_AVAILABLE and ClassMirror is not None:
            try:
                mirror = ClassMirror(cls)
                for field_name, field_info in mirror.fields.items():
                    if not self._should_include_field(field_name, field_info):
                        continue

                    label = self._get_display_label(field_name, field_info)
                    field_type = field_info.type
                    default = field_info.default if field_info.has_default else None
                    metadata = field_info.metadata if hasattr(field_info, "metadata") else {}

                    widget = self._registry.create_widget(
                        field_type, field_name, label, default, metadata
                    )
                    if widget:
                        panel.add_property(field_name, widget, readonly=True)
            except Exception:
                pass

        return panel

    def clear_cache(self) -> None:
        """Clear cached panels."""
        self._panels.clear()

    def remove_cached(self, obj: Any) -> bool:
        """Remove specific object from cache."""
        obj_id = id(obj)
        if obj_id in self._panels:
            del self._panels[obj_id]
            return True
        return False


# =============================================================================
# Debug UI Context (Mock Rendering)
# =============================================================================


@dataclass
class UIEvent:
    """Input event for debug UI."""
    type: str  # "mouse_down", "mouse_up", "mouse_move", "key_down", "key_up"
    x: int = 0
    y: int = 0
    button: int = 0
    key: str = ""
    modifiers: set = field(default_factory=set)


@editor(category="DebugUI")
@reloadable()
class DebugUIContext:
    """Rendering context for debug UI (mock implementation)."""
    __slots__ = (
        "cursor_pos", "indent_level", "_draw_commands", "_widget_states",
        "screen_width", "screen_height", "scale", "_frame_count"
    )

    def __init__(self, width: int = 800, height: int = 600, scale: float = 1.0):
        self.cursor_pos = Vec2(0, 0)
        self.indent_level = 0
        self._draw_commands: List[Dict[str, Any]] = []
        self._widget_states: Dict[str, UIState] = {}
        self.screen_width = width
        self.screen_height = height
        self.scale = scale
        self._frame_count = 0

    def begin_frame(self) -> None:
        """Begin a new frame."""
        self._draw_commands.clear()
        self.cursor_pos = Vec2(0, 0)
        self.indent_level = 0
        self._frame_count += 1

    def end_frame(self) -> List[Dict[str, Any]]:
        """End frame and return draw commands."""
        return list(self._draw_commands)

    def indent(self, amount: int = 1) -> None:
        """Increase indent level."""
        self.indent_level += amount
        self.cursor_pos.x += 20 * amount

    def unindent(self, amount: int = 1) -> None:
        """Decrease indent level."""
        self.indent_level = max(0, self.indent_level - amount)
        self.cursor_pos.x = max(0, self.cursor_pos.x - 20 * amount)

    def next_line(self, height: float = 20.0) -> None:
        """Move cursor to next line."""
        self.cursor_pos.y += height
        self.cursor_pos.x = 20 * self.indent_level

    def get_widget_state(self, widget_id: str) -> UIState:
        """Get state for widget."""
        return self._widget_states.get(widget_id, UIState.NORMAL)

    def set_widget_state(self, widget_id: str, state: UIState) -> None:
        """Set state for widget."""
        self._widget_states[widget_id] = state

    # Draw commands (mock implementations that record commands)

    def draw_text(self, text: str, pos: Vec2) -> None:
        """Draw text at position."""
        self._draw_commands.append({
            "type": "text",
            "text": text,
            "x": pos.x,
            "y": pos.y,
        })
        self.next_line()

    def draw_text_input(
        self,
        widget_id: str,
        label: str,
        value: str,
        max_length: int,
        placeholder: str,
        password: bool,
    ) -> None:
        """Draw text input widget."""
        self._draw_commands.append({
            "type": "text_input",
            "id": widget_id,
            "label": label,
            "value": value,
            "max_length": max_length,
            "placeholder": placeholder,
            "password": password,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        self.next_line()

    def draw_slider_int(
        self,
        widget_id: str,
        label: str,
        value: int,
        min_value: int,
        max_value: int,
    ) -> None:
        """Draw integer slider widget."""
        self._draw_commands.append({
            "type": "slider_int",
            "id": widget_id,
            "label": label,
            "value": value,
            "min": min_value,
            "max": max_value,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        self.next_line()

    def draw_slider_float(
        self,
        widget_id: str,
        label: str,
        value: float,
        min_value: float,
        max_value: float,
        precision: int,
    ) -> None:
        """Draw float slider widget."""
        self._draw_commands.append({
            "type": "slider_float",
            "id": widget_id,
            "label": label,
            "value": value,
            "min": min_value,
            "max": max_value,
            "precision": precision,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        self.next_line()

    def draw_input_int(
        self, widget_id: str, label: str, value: int, step: int
    ) -> None:
        """Draw integer input widget."""
        self._draw_commands.append({
            "type": "input_int",
            "id": widget_id,
            "label": label,
            "value": value,
            "step": step,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        self.next_line()

    def draw_input_float(
        self,
        widget_id: str,
        label: str,
        value: float,
        step: float,
        precision: int,
    ) -> None:
        """Draw float input widget."""
        self._draw_commands.append({
            "type": "input_float",
            "id": widget_id,
            "label": label,
            "value": value,
            "step": step,
            "precision": precision,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        self.next_line()

    def draw_checkbox(self, widget_id: str, label: str, value: bool) -> None:
        """Draw checkbox widget."""
        self._draw_commands.append({
            "type": "checkbox",
            "id": widget_id,
            "label": label,
            "value": value,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        self.next_line()

    def draw_dropdown(
        self,
        widget_id: str,
        label: str,
        options: List[str],
        selected_index: int,
    ) -> None:
        """Draw dropdown widget."""
        self._draw_commands.append({
            "type": "dropdown",
            "id": widget_id,
            "label": label,
            "options": options,
            "selected": selected_index,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        self.next_line()

    def draw_color_picker(
        self,
        widget_id: str,
        label: str,
        color: Color,
        show_alpha: bool,
        show_hex: bool,
    ) -> None:
        """Draw color picker widget."""
        self._draw_commands.append({
            "type": "color_picker",
            "id": widget_id,
            "label": label,
            "color": color.to_tuple(),
            "show_alpha": show_alpha,
            "show_hex": show_hex,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        self.next_line(40)  # Color picker is taller

    def draw_vec2_input(
        self,
        widget_id: str,
        label: str,
        value: Vec2,
        labels: Tuple[str, str],
        step: float,
        precision: int,
    ) -> None:
        """Draw Vec2 input widget."""
        self._draw_commands.append({
            "type": "vec2_input",
            "id": widget_id,
            "label": label,
            "value": value.to_tuple(),
            "component_labels": labels,
            "step": step,
            "precision": precision,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        self.next_line()

    def draw_vec3_input(
        self,
        widget_id: str,
        label: str,
        value: Vec3,
        labels: Tuple[str, str, str],
        step: float,
        precision: int,
    ) -> None:
        """Draw Vec3 input widget."""
        self._draw_commands.append({
            "type": "vec3_input",
            "id": widget_id,
            "label": label,
            "value": value.to_tuple(),
            "component_labels": labels,
            "step": step,
            "precision": precision,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        self.next_line()

    def draw_vec4_input(
        self,
        widget_id: str,
        label: str,
        value: Vec4,
        labels: Tuple[str, str, str, str],
        step: float,
        precision: int,
    ) -> None:
        """Draw Vec4 input widget."""
        self._draw_commands.append({
            "type": "vec4_input",
            "id": widget_id,
            "label": label,
            "value": value.to_tuple(),
            "component_labels": labels,
            "step": step,
            "precision": precision,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        self.next_line()

    def draw_button(self, widget_id: str, label: str, icon: str) -> bool:
        """Draw button widget. Returns True if clicked."""
        self._draw_commands.append({
            "type": "button",
            "id": widget_id,
            "label": label,
            "icon": icon,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        self.next_line()
        return False  # Mock: never clicked

    def draw_separator(self, horizontal: bool) -> None:
        """Draw separator line."""
        self._draw_commands.append({
            "type": "separator",
            "horizontal": horizontal,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        if horizontal:
            self.next_line(8)

    def draw_collapsing_header(
        self, widget_id: str, label: str, expanded: bool, icon: str
    ) -> bool:
        """Draw collapsing header. Returns True if clicked."""
        self._draw_commands.append({
            "type": "collapsing_header",
            "id": widget_id,
            "label": label,
            "expanded": expanded,
            "icon": icon,
            "x": self.cursor_pos.x,
            "y": self.cursor_pos.y,
        })
        self.next_line()
        return False  # Mock: never clicked


# =============================================================================
# Debug UI Manager
# =============================================================================


@editor(category="DebugUI")
@reloadable()
class DebugUI:
    """Main debug UI manager integrating all components."""
    __slots__ = (
        "_context", "_registry", "_inspector", "_panels", "_root",
        "_visible", "_active_mode", "_mode_panels"
    )

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        scale: float = 1.0,
    ):
        self._context = DebugUIContext(width, height, scale)
        self._registry = WidgetRegistry()
        self._inspector = AutoInspector(self._registry)
        self._panels: Dict[str, PropertyPanel] = {}
        self._root = ContainerWidget(WidgetType.TREE_NODE)
        self._visible = True
        self._active_mode: Optional[str] = None
        self._mode_panels: Dict[str, List[str]] = {}

    @property
    def context(self) -> DebugUIContext:
        return self._context

    @property
    def registry(self) -> WidgetRegistry:
        return self._registry

    @property
    def inspector(self) -> AutoInspector:
        return self._inspector

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = value

    def toggle_visibility(self) -> None:
        """Toggle debug UI visibility."""
        self._visible = not self._visible

    # Panel management

    def create_panel(
        self, panel_id: str, title: str, target: Any = None
    ) -> PropertyPanel:
        """Create a new property panel."""
        panel = PropertyPanel(title, target)
        self._panels[panel_id] = panel
        self._root.add_child(panel)
        return panel

    def get_panel(self, panel_id: str) -> Optional[PropertyPanel]:
        """Get panel by ID."""
        return self._panels.get(panel_id)

    def remove_panel(self, panel_id: str) -> bool:
        """Remove a panel."""
        if panel_id in self._panels:
            panel = self._panels.pop(panel_id)
            self._root.remove_child(panel)
            return True
        return False

    def list_panels(self) -> List[str]:
        """List all panel IDs."""
        return list(self._panels.keys())

    # Auto-inspection

    def inspect(self, obj: Any, panel_id: Optional[str] = None) -> PropertyPanel:
        """Inspect object and create/update panel."""
        panel = self._inspector.inspect_object(obj)
        if panel_id:
            self._panels[panel_id] = panel
            self._root.add_child(panel)
        return panel

    # Section management

    def create_section(
        self, title: str, expanded: bool = True, icon: str = ""
    ) -> CollapsibleSection:
        """Create a collapsible section."""
        section = CollapsibleSection(title, expanded, icon)
        self._root.add_child(section)
        return section

    # Mode integration

    def register_mode_panel(self, mode_name: str, panel_id: str) -> None:
        """Register a panel to show when mode is active."""
        if mode_name not in self._mode_panels:
            self._mode_panels[mode_name] = []
        if panel_id not in self._mode_panels[mode_name]:
            self._mode_panels[mode_name].append(panel_id)

    def unregister_mode_panel(self, mode_name: str, panel_id: str) -> bool:
        """Unregister a panel from mode."""
        if mode_name in self._mode_panels:
            if panel_id in self._mode_panels[mode_name]:
                self._mode_panels[mode_name].remove(panel_id)
                return True
        return False

    def set_active_mode(self, mode_name: Optional[str]) -> None:
        """Set active editor mode, showing/hiding relevant panels."""
        old_mode = self._active_mode
        self._active_mode = mode_name

        # Hide panels from old mode
        if old_mode and old_mode in self._mode_panels:
            for panel_id in self._mode_panels[old_mode]:
                panel = self._panels.get(panel_id)
                if panel:
                    panel.visible = False

        # Show panels for new mode
        if mode_name and mode_name in self._mode_panels:
            for panel_id in self._mode_panels[mode_name]:
                panel = self._panels.get(panel_id)
                if panel:
                    panel.visible = True

    def get_mode_panels(self, mode_name: str) -> List[str]:
        """Get panel IDs registered for mode."""
        return list(self._mode_panels.get(mode_name, []))

    # Widget creation helpers

    def add_label(self, text: str) -> LabelWidget:
        """Add a label widget."""
        widget = LabelWidget(text)
        self._root.add_child(widget)
        return widget

    def add_button(
        self, label: str, on_click: Optional[Callable[[], None]] = None
    ) -> ButtonWidget:
        """Add a button widget."""
        widget = ButtonWidget(label)
        widget.on_click = on_click
        self._root.add_child(widget)
        return widget

    def add_checkbox(
        self,
        label: str,
        initial_value: bool = False,
        on_change: Optional[Callable[[bool], None]] = None,
    ) -> CheckboxWidget:
        """Add a checkbox widget."""
        widget = CheckboxWidget(label, initial_value)
        widget.on_change = on_change
        self._root.add_child(widget)
        return widget

    def add_slider_int(
        self,
        label: str,
        initial_value: int = 0,
        min_value: int = 0,
        max_value: int = 100,
        on_change: Optional[Callable[[int], None]] = None,
    ) -> IntSliderWidget:
        """Add an integer slider widget."""
        widget = IntSliderWidget(label, initial_value, min_value, max_value)
        widget.on_change = on_change
        self._root.add_child(widget)
        return widget

    def add_slider_float(
        self,
        label: str,
        initial_value: float = 0.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
        on_change: Optional[Callable[[float], None]] = None,
    ) -> FloatSliderWidget:
        """Add a float slider widget."""
        widget = FloatSliderWidget(label, initial_value, min_value, max_value)
        widget.on_change = on_change
        self._root.add_child(widget)
        return widget

    def add_dropdown(
        self,
        label: str,
        options: List[str],
        selected_index: int = 0,
        on_change: Optional[Callable[[str], None]] = None,
    ) -> DropdownWidget:
        """Add a dropdown widget."""
        widget = DropdownWidget(label, options, selected_index)
        widget.on_change = on_change
        self._root.add_child(widget)
        return widget

    def add_color_picker(
        self,
        label: str,
        initial_color: Optional[Color] = None,
        on_change: Optional[Callable[[Color], None]] = None,
    ) -> ColorPickerWidget:
        """Add a color picker widget."""
        widget = ColorPickerWidget(label, initial_color)
        widget.on_change = on_change
        self._root.add_child(widget)
        return widget

    def add_separator(self) -> SeparatorWidget:
        """Add a separator widget."""
        widget = SeparatorWidget()
        self._root.add_child(widget)
        return widget

    # Rendering

    def begin_frame(self) -> None:
        """Begin a new UI frame."""
        self._context.begin_frame()

    def render(self) -> List[Dict[str, Any]]:
        """Render all UI and return draw commands."""
        if not self._visible:
            return []
        self._context.begin_frame()
        self._root.render(self._context)
        return self._context.end_frame()

    def end_frame(self) -> List[Dict[str, Any]]:
        """End frame and return draw commands."""
        return self._context.end_frame()

    # Input handling

    def handle_event(self, event: UIEvent) -> bool:
        """Handle input event. Returns True if consumed."""
        if not self._visible:
            return False
        return self._root.handle_input(event)

    # Cleanup

    def clear(self) -> None:
        """Clear all widgets and panels."""
        self._root.clear_children()
        self._panels.clear()
        self._inspector.clear_cache()
        self._mode_panels.clear()


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Core types
    "WidgetType",
    "UIState",
    "Vec2",
    "Vec3",
    "Vec4",
    "Color",
    "WidgetStyle",
    "WidgetConfig",
    "UIEvent",
    # Base widgets
    "Widget",
    "ContainerWidget",
    # Concrete widgets
    "LabelWidget",
    "TextInputWidget",
    "IntSliderWidget",
    "FloatSliderWidget",
    "IntInputWidget",
    "FloatInputWidget",
    "CheckboxWidget",
    "DropdownWidget",
    "ColorPickerWidget",
    "Vec2InputWidget",
    "Vec3InputWidget",
    "Vec4InputWidget",
    "ButtonWidget",
    "SeparatorWidget",
    # Containers
    "CollapsibleSection",
    "PropertyPanel",
    "PropertyBinding",
    # Registry and inspection
    "WidgetRegistry",
    "AutoInspector",
    # Context and manager
    "DebugUIContext",
    "DebugUI",
]
