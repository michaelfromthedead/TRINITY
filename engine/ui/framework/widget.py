"""
Base Widget class for the UI framework.

Provides the foundation for all UI widgets with:
- Hierarchical parent-child relationships
- Transform and layout support
- Event handling integration
- Lifecycle hooks for mount/unmount/update/render
- Dirty tracking for efficient re-rendering

Widget Properties:
    - parent: Parent widget reference
    - children: List of child widgets
    - transform: Position, size, and rotation
    - visible: Whether widget is rendered
    - enabled: Whether widget receives input
    - focusable: Whether widget can receive focus
    - tooltip: Tooltip text to display on hover
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from engine.ui.framework.coordinate import (
    Anchor,
    Margins,
    Point,
    Rect,
    Size,
    Transform2D,
)
from engine.ui.framework.events import (
    EventHandler,
    EventPhase,
    EventType,
    FocusEvent,
    KeyboardEvent,
    MouseEvent,
    UIEvent,
)

if TYPE_CHECKING:
    from engine.ui.framework.focus import FocusManager


T = TypeVar("T")


class TrackedDescriptor(Generic[T]):
    """
    Descriptor that tracks changes to trigger re-rendering.

    When a value changes, marks the widget as dirty and optionally
    calls registered change callbacks.
    """

    __slots__ = ("_name", "_default", "_private_name")

    def __init__(self, default: T = None) -> None:
        self._default = default
        self._name: str = ""
        self._private_name: str = ""

    def __set_name__(self, owner: Type, name: str) -> None:
        self._name = name
        self._private_name = f"_tracked_{name}"

    def __get__(self, obj: Optional[object], objtype: Type = None) -> T:
        if obj is None:
            return self  # type: ignore
        return getattr(obj, self._private_name, self._default)

    def __set__(self, obj: object, value: T) -> None:
        old_value = getattr(obj, self._private_name, self._default)
        if old_value != value:
            setattr(obj, self._private_name, value)
            if hasattr(obj, "_mark_dirty"):
                obj._mark_dirty(self._name)
            if hasattr(obj, "_on_property_changed"):
                obj._on_property_changed(self._name, old_value, value)


# Global widget ID counter
_widget_id_counter = itertools.count(start=1)


def _generate_widget_id() -> int:
    """Generate a unique widget ID."""
    return next(_widget_id_counter)


@dataclass
class WidgetStyle:
    """
    Style properties for a widget.

    Contains visual styling that can be themed or customized.
    """

    background_color: Optional[str] = None
    border_color: Optional[str] = None
    border_width: float = 0.0
    corner_radius: float = 0.0
    opacity: float = 1.0
    padding: Margins = field(default_factory=Margins.zero)

    def merge_with(self, other: "WidgetStyle") -> "WidgetStyle":
        """Create a new style by merging this with another (other takes precedence)."""
        return WidgetStyle(
            background_color=other.background_color or self.background_color,
            border_color=other.border_color or self.border_color,
            border_width=other.border_width if other.border_width > 0 else self.border_width,
            corner_radius=other.corner_radius if other.corner_radius > 0 else self.corner_radius,
            opacity=other.opacity if other.opacity < 1.0 else self.opacity,
            padding=other.padding if other.padding != Margins.zero() else self.padding,
        )


@dataclass
class LayoutConstraints:
    """
    Layout constraints for a widget.

    Defines minimum, maximum, and preferred sizes.
    """

    min_width: float = 0.0
    min_height: float = 0.0
    max_width: float = float("inf")
    max_height: float = float("inf")
    preferred_width: Optional[float] = None
    preferred_height: Optional[float] = None

    def constrain_size(self, size: Size) -> Size:
        """Apply constraints to a size."""
        width = max(self.min_width, min(size.width, self.max_width))
        height = max(self.min_height, min(size.height, self.max_height))
        return Size(width, height)


class Widget:
    """
    Base class for all UI widgets.

    Provides hierarchy management, event handling, layout support,
    and lifecycle hooks. Uses TrackedDescriptor for dirty tracking.
    """

    # Tracked properties that trigger re-render when changed
    local_x: float = TrackedDescriptor(0.0)
    local_y: float = TrackedDescriptor(0.0)
    width: float = TrackedDescriptor(0.0)
    height: float = TrackedDescriptor(0.0)
    rotation: float = TrackedDescriptor(0.0)
    scale_x: float = TrackedDescriptor(1.0)
    scale_y: float = TrackedDescriptor(1.0)
    visible: bool = TrackedDescriptor(True)
    enabled: bool = TrackedDescriptor(True)
    focusable: bool = TrackedDescriptor(False)
    tooltip: Optional[str] = TrackedDescriptor(None)
    tab_index: int = TrackedDescriptor(0)
    z_index: int = TrackedDescriptor(0)

    __slots__ = (
        "_id",
        "_name",
        "_parent",
        "_children",
        "_dirty_flags",
        "_is_mounted",
        "_event_handlers",
        "_capture_handlers",
        "_style",
        "_layout_constraints",
        "_anchor",
        "_pivot",
        "_stretch",
        "_clip_children",
        "_cached_global_rect",
        "_cached_global_rect_valid",
        # Private storage for TrackedDescriptor
        "_tracked_local_x",
        "_tracked_local_y",
        "_tracked_width",
        "_tracked_height",
        "_tracked_rotation",
        "_tracked_scale_x",
        "_tracked_scale_y",
        "_tracked_visible",
        "_tracked_enabled",
        "_tracked_focusable",
        "_tracked_tooltip",
        "_tracked_tab_index",
        "_tracked_z_index",
    )

    def __init__(
        self,
        name: str = "",
        x: float = 0.0,
        y: float = 0.0,
        width: float = 0.0,
        height: float = 0.0,
        visible: bool = True,
        enabled: bool = True,
        focusable: bool = False,
        tooltip: Optional[str] = None,
    ) -> None:
        """
        Initialize a widget.

        Args:
            name: Optional name for identification.
            x: Local x position.
            y: Local y position.
            width: Widget width.
            height: Widget height.
            visible: Whether widget is visible.
            enabled: Whether widget receives input.
            focusable: Whether widget can receive focus.
            tooltip: Tooltip text.
        """
        self._id = _generate_widget_id()
        self._name = name or f"Widget_{self._id}"
        self._parent: Optional[Widget] = None
        self._children: List[Widget] = []
        self._dirty_flags: Set[str] = set()
        self._is_mounted = False
        self._event_handlers: Dict[EventType, List[EventHandler]] = {}
        self._capture_handlers: Dict[EventType, List[EventHandler]] = {}
        self._style = WidgetStyle()
        self._layout_constraints = LayoutConstraints()
        self._anchor = Anchor.TOP_LEFT
        self._pivot = Point(0.0, 0.0)
        self._stretch = False
        self._clip_children = False
        self._cached_global_rect: Optional[Rect] = None
        self._cached_global_rect_valid = False

        # Initialize tracked properties
        self.local_x = x
        self.local_y = y
        self.width = width
        self.height = height
        self.visible = visible
        self.enabled = enabled
        self.focusable = focusable
        self.tooltip = tooltip

    # ============================================
    # Identity Properties
    # ============================================

    @property
    def id(self) -> int:
        """Unique widget identifier."""
        return self._id

    @property
    def name(self) -> str:
        """Widget name for identification."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    # ============================================
    # Hierarchy Properties
    # ============================================

    @property
    def parent(self) -> Optional["Widget"]:
        """Parent widget, or None if this is a root."""
        return self._parent

    @property
    def children(self) -> List["Widget"]:
        """List of child widgets (read-only copy)."""
        return self._children.copy()

    @property
    def child_count(self) -> int:
        """Number of direct children."""
        return len(self._children)

    @property
    def has_children(self) -> bool:
        """Whether widget has any children."""
        return len(self._children) > 0

    @property
    def is_root(self) -> bool:
        """Whether this widget is a root (no parent)."""
        return self._parent is None

    @property
    def is_mounted(self) -> bool:
        """Whether widget is mounted in the hierarchy."""
        return self._is_mounted

    # ============================================
    # Transform Properties
    # ============================================

    @property
    def position(self) -> Point:
        """Local position as Point."""
        return Point(self.local_x, self.local_y)

    @position.setter
    def position(self, value: Point) -> None:
        self.local_x = value.x
        self.local_y = value.y

    @property
    def size(self) -> Size:
        """Widget size."""
        return Size(self.width, self.height)

    @size.setter
    def size(self, value: Size) -> None:
        self.width = value.width
        self.height = value.height

    @property
    def transform(self) -> Transform2D:
        """Full 2D transform."""
        return Transform2D(
            position=Point(self.local_x, self.local_y),
            rotation=self.rotation,
            scale=Point(self.scale_x, self.scale_y),
        )

    @transform.setter
    def transform(self, value: Transform2D) -> None:
        self.local_x = value.position.x
        self.local_y = value.position.y
        self.rotation = value.rotation
        self.scale_x = value.scale.x
        self.scale_y = value.scale.y

    @property
    def local_rect(self) -> Rect:
        """Local bounding rectangle."""
        return Rect(self.local_x, self.local_y, self.width, self.height)

    @property
    def global_x(self) -> float:
        """Global x position (computed from parent chain)."""
        if self._parent is None:
            return self.local_x
        return self.local_x + self._parent.global_x

    @property
    def global_y(self) -> float:
        """Global y position (computed from parent chain)."""
        if self._parent is None:
            return self.local_y
        return self.local_y + self._parent.global_y

    @property
    def global_position(self) -> Point:
        """Global position."""
        return Point(self.global_x, self.global_y)

    @property
    def global_rect(self) -> Rect:
        """Global bounding rectangle (cached)."""
        if not self._cached_global_rect_valid:
            self._cached_global_rect = Rect(
                self.global_x, self.global_y, self.width, self.height
            )
            self._cached_global_rect_valid = True
        return self._cached_global_rect

    # ============================================
    # Style Properties
    # ============================================

    @property
    def style(self) -> WidgetStyle:
        """Widget style properties."""
        return self._style

    @style.setter
    def style(self, value: WidgetStyle) -> None:
        self._style = value
        self._mark_dirty("style")

    @property
    def layout_constraints(self) -> LayoutConstraints:
        """Layout constraints."""
        return self._layout_constraints

    @layout_constraints.setter
    def layout_constraints(self, value: LayoutConstraints) -> None:
        self._layout_constraints = value
        self._mark_dirty("layout")

    @property
    def anchor(self) -> Anchor:
        """Anchor point within parent."""
        return self._anchor

    @anchor.setter
    def anchor(self, value: Anchor) -> None:
        self._anchor = value
        self._invalidate_global_rect()

    @property
    def pivot(self) -> Point:
        """Pivot point for rotation/scaling (0-1 range)."""
        return self._pivot

    @pivot.setter
    def pivot(self, value: Point) -> None:
        self._pivot = value
        self._invalidate_global_rect()

    @property
    def clip_children(self) -> bool:
        """Whether to clip children to this widget's bounds."""
        return self._clip_children

    @clip_children.setter
    def clip_children(self, value: bool) -> None:
        self._clip_children = value
        self._mark_dirty("clip")

    # ============================================
    # Hierarchy Management
    # ============================================

    def add_child(self, child: "Widget") -> "Widget":
        """
        Add a child widget.

        Args:
            child: Widget to add as child.

        Returns:
            The added child (for chaining).

        Raises:
            ValueError: If child already has a parent or is an ancestor.
        """
        if child._parent is not None:
            raise ValueError(f"Widget '{child.name}' already has a parent")

        if child is self:
            raise ValueError("Cannot add widget as its own child")

        # Check for circular reference
        ancestor = self._parent
        while ancestor is not None:
            if ancestor is child:
                raise ValueError("Cannot add ancestor as child (circular reference)")
            ancestor = ancestor._parent

        child._parent = self
        self._children.append(child)

        # Sort by z_index
        self._sort_children_by_z()

        # Mount if we're mounted
        if self._is_mounted:
            child._mount()

        self._mark_dirty("children")
        return child

    def remove_child(self, child: "Widget") -> bool:
        """
        Remove a child widget.

        Args:
            child: Widget to remove.

        Returns:
            True if child was removed, False if not found.
        """
        if child not in self._children:
            return False

        # Unmount if mounted
        if child._is_mounted:
            child._unmount()

        child._parent = None
        self._children.remove(child)
        self._mark_dirty("children")
        return True

    def remove_all_children(self) -> int:
        """
        Remove all child widgets.

        Returns:
            Number of children removed.
        """
        count = len(self._children)
        for child in self._children.copy():
            self.remove_child(child)
        return count

    def find_child(self, name: str, recursive: bool = False) -> Optional["Widget"]:
        """
        Find a child widget by name.

        Args:
            name: Name to search for.
            recursive: Whether to search descendants.

        Returns:
            Found widget or None.
        """
        for child in self._children:
            if child.name == name:
                return child
            if recursive:
                found = child.find_child(name, recursive=True)
                if found is not None:
                    return found
        return None

    def find_child_by_id(self, widget_id: int, recursive: bool = False) -> Optional["Widget"]:
        """
        Find a child widget by ID.

        Args:
            widget_id: ID to search for.
            recursive: Whether to search descendants.

        Returns:
            Found widget or None.
        """
        for child in self._children:
            if child.id == widget_id:
                return child
            if recursive:
                found = child.find_child_by_id(widget_id, recursive=True)
                if found is not None:
                    return found
        return None

    def find_children_by_type(
        self, widget_type: Type["Widget"], recursive: bool = False
    ) -> List["Widget"]:
        """
        Find all children of a specific type.

        Args:
            widget_type: Type to search for.
            recursive: Whether to search descendants.

        Returns:
            List of matching widgets.
        """
        results: List[Widget] = []
        for child in self._children:
            if isinstance(child, widget_type):
                results.append(child)
            if recursive:
                results.extend(child.find_children_by_type(widget_type, recursive=True))
        return results

    def get_child_at(self, index: int) -> Optional["Widget"]:
        """Get child at index, or None if out of bounds."""
        if 0 <= index < len(self._children):
            return self._children[index]
        return None

    def get_child_index(self, child: "Widget") -> int:
        """Get index of child, or -1 if not found."""
        try:
            return self._children.index(child)
        except ValueError:
            return -1

    def move_child(self, child: "Widget", new_index: int) -> bool:
        """
        Move a child to a new index.

        Args:
            child: Child to move.
            new_index: Target index.

        Returns:
            True if moved successfully.
        """
        if child not in self._children:
            return False

        self._children.remove(child)
        new_index = max(0, min(new_index, len(self._children)))
        self._children.insert(new_index, child)
        self._mark_dirty("children")
        return True

    def bring_to_front(self) -> None:
        """Bring this widget to front of siblings."""
        if self._parent is not None:
            max_z = max((c.z_index for c in self._parent._children), default=0)
            self.z_index = max_z + 1

    def send_to_back(self) -> None:
        """Send this widget to back of siblings."""
        if self._parent is not None:
            min_z = min((c.z_index for c in self._parent._children), default=0)
            self.z_index = min_z - 1

    def get_root(self) -> "Widget":
        """Get the root widget of the hierarchy."""
        widget = self
        while widget._parent is not None:
            widget = widget._parent
        return widget

    def get_ancestors(self) -> List["Widget"]:
        """Get list of ancestors from parent to root."""
        ancestors: List[Widget] = []
        widget = self._parent
        while widget is not None:
            ancestors.append(widget)
            widget = widget._parent
        return ancestors

    def get_descendants(self) -> Iterator["Widget"]:
        """Iterate over all descendants (depth-first)."""
        for child in self._children:
            yield child
            yield from child.get_descendants()

    def _sort_children_by_z(self) -> None:
        """Sort children by z_index (stable sort)."""
        self._children.sort(key=lambda c: c.z_index)

    # ============================================
    # Dirty Tracking
    # ============================================

    def _mark_dirty(self, flag: str = "render") -> None:
        """Mark widget as needing update."""
        self._dirty_flags.add(flag)
        self._invalidate_global_rect()

        # Propagate layout dirty flag to parent
        if flag == "layout" and self._parent is not None:
            self._parent._mark_dirty("layout")

    def _clear_dirty(self, flag: Optional[str] = None) -> None:
        """Clear dirty flag(s)."""
        if flag is None:
            self._dirty_flags.clear()
        else:
            self._dirty_flags.discard(flag)

    def is_dirty(self, flag: Optional[str] = None) -> bool:
        """Check if widget is dirty."""
        if flag is None:
            return len(self._dirty_flags) > 0
        return flag in self._dirty_flags

    def _invalidate_global_rect(self) -> None:
        """Invalidate cached global rect for this and descendants."""
        self._cached_global_rect_valid = False
        for child in self._children:
            child._invalidate_global_rect()

    def _on_property_changed(self, name: str, old_value: Any, new_value: Any) -> None:
        """Called when a tracked property changes."""
        # Re-sort children when z_index changes
        if name == "z_index" and self._parent is not None:
            self._parent._sort_children_by_z()

    # ============================================
    # Layout
    # ============================================

    def layout(self, available_rect: Optional[Rect] = None) -> None:
        """
        Perform layout for this widget and children.

        Override in subclasses for custom layout behavior.

        Args:
            available_rect: Available space for layout.
        """
        # Apply constraints to size
        if self._layout_constraints:
            constrained = self._layout_constraints.constrain_size(self.size)
            if constrained != self.size:
                self.size = constrained

        # Layout children
        for child in self._children:
            if child.visible:
                child.layout(self.local_rect)

        self._clear_dirty("layout")

    def measure(self) -> Size:
        """
        Measure preferred size of this widget.

        Override in subclasses for custom measurement.

        Returns:
            Preferred size.
        """
        constraints = self._layout_constraints
        width = constraints.preferred_width or self.width
        height = constraints.preferred_height or self.height
        return Size(width, height)

    # ============================================
    # Hit Testing
    # ============================================

    def hit_test(self, point: Point) -> Optional["Widget"]:
        """
        Find the deepest widget at a point.

        Args:
            point: Point in global coordinates.

        Returns:
            Deepest widget containing the point, or None.
        """
        if not self.visible:
            return None

        # Check if point is in our bounds
        if not self.global_rect.contains_point(point):
            return None

        # Convert to local coordinates for children
        local_point = Point(point.x - self.global_x, point.y - self.global_y)

        # Check children in reverse order (front to back)
        for child in reversed(self._children):
            # Convert local point to global for child hit test
            hit = child.hit_test(point)
            if hit is not None:
                return hit

        # No child hit, return self if enabled
        return self if self.enabled else None

    def hit_test_all(self, point: Point) -> List["Widget"]:
        """
        Find all widgets at a point (from front to back).

        Args:
            point: Point in global coordinates.

        Returns:
            List of widgets containing the point.
        """
        results: List[Widget] = []

        if not self.visible:
            return results

        if not self.global_rect.contains_point(point):
            return results

        # Check children in reverse order
        for child in reversed(self._children):
            results.extend(child.hit_test_all(point))

        # Add self if enabled
        if self.enabled:
            results.append(self)

        return results

    def contains_point(self, point: Point, local: bool = False) -> bool:
        """
        Check if point is inside widget bounds.

        Args:
            point: Point to check.
            local: If True, point is in local coordinates.

        Returns:
            True if point is inside bounds.
        """
        if local:
            return Rect(0, 0, self.width, self.height).contains_point(point)
        return self.global_rect.contains_point(point)

    # ============================================
    # Event Handling
    # ============================================

    def add_event_handler(
        self,
        event_type: EventType,
        handler: EventHandler,
        capture: bool = False,
    ) -> None:
        """
        Add an event handler.

        Args:
            event_type: Type of event to handle.
            handler: Handler function.
            capture: If True, handle during capture phase.
        """
        handlers = self._capture_handlers if capture else self._event_handlers
        if event_type not in handlers:
            handlers[event_type] = []
        if handler not in handlers[event_type]:
            handlers[event_type].append(handler)

    def remove_event_handler(
        self,
        event_type: EventType,
        handler: EventHandler,
        capture: bool = False,
    ) -> bool:
        """
        Remove an event handler.

        Args:
            event_type: Type of event.
            handler: Handler to remove.
            capture: If True, remove from capture handlers.

        Returns:
            True if handler was removed.
        """
        handlers = self._capture_handlers if capture else self._event_handlers
        if event_type in handlers and handler in handlers[event_type]:
            handlers[event_type].remove(handler)
            return True
        return False

    def _dispatch_to_handlers(self, event: UIEvent, capture: bool) -> None:
        """Dispatch event to registered handlers."""
        handlers = self._capture_handlers if capture else self._event_handlers
        if event.event_type in handlers:
            for handler in handlers[event.event_type].copy():
                if event.is_stopped_immediate:
                    break
                handler(event)

    def dispatch_event(self, event: UIEvent) -> bool:
        """
        Dispatch an event to this widget.

        Handles capture and bubble phases according to W3C model.

        Args:
            event: Event to dispatch.

        Returns:
            True if event was not cancelled.
        """
        from engine.ui.framework.events import EventDispatcher
        return EventDispatcher.dispatch(event, self)

    # ============================================
    # Lifecycle Hooks
    # ============================================

    def _mount(self) -> None:
        """Internal mount handler."""
        if self._is_mounted:
            return

        self._is_mounted = True
        self.on_mount()

        # Mount children
        for child in self._children:
            child._mount()

    def _unmount(self) -> None:
        """Internal unmount handler."""
        if not self._is_mounted:
            return

        # Unmount children first
        for child in self._children:
            child._unmount()

        self._is_mounted = False
        self.on_unmount()

    def on_mount(self) -> None:
        """
        Called when widget is mounted to the hierarchy.

        Override in subclasses for initialization.
        """
        pass

    def on_unmount(self) -> None:
        """
        Called when widget is unmounted from the hierarchy.

        Override in subclasses for cleanup.
        """
        pass

    def on_update(self, delta_time: float) -> None:
        """
        Called every frame for updates.

        Override in subclasses for per-frame logic.

        Args:
            delta_time: Time since last update in seconds.
        """
        pass

    def on_render(self, context: Any) -> None:
        """
        Called to render the widget.

        Override in subclasses for custom rendering.

        Args:
            context: Render context (renderer-specific).
        """
        pass

    def update(self, delta_time: float) -> None:
        """
        Update widget and all children.

        Args:
            delta_time: Time since last update.
        """
        if not self.visible:
            return

        self.on_update(delta_time)

        for child in self._children:
            child.update(delta_time)

    def render(self, context: Any) -> None:
        """
        Render widget and all children.

        Args:
            context: Render context.
        """
        if not self.visible:
            return

        self.on_render(context)
        self._clear_dirty("render")

        for child in self._children:
            child.render(context)

    # ============================================
    # Coordinate Conversion
    # ============================================

    def local_to_global(self, point: Point) -> Point:
        """Convert local coordinates to global."""
        return Point(point.x + self.global_x, point.y + self.global_y)

    def global_to_local(self, point: Point) -> Point:
        """Convert global coordinates to local."""
        return Point(point.x - self.global_x, point.y - self.global_y)

    def local_to_parent(self, point: Point) -> Point:
        """Convert local coordinates to parent coordinates."""
        return Point(point.x + self.local_x, point.y + self.local_y)

    def parent_to_local(self, point: Point) -> Point:
        """Convert parent coordinates to local."""
        return Point(point.x - self.local_x, point.y - self.local_y)

    # ============================================
    # String Representation
    # ============================================

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self._id}, "
            f"name='{self._name}', "
            f"rect={self.local_rect})"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}('{self._name}')"

    def debug_tree(self, indent: int = 0) -> str:
        """Return a debug string showing widget hierarchy."""
        prefix = "  " * indent
        lines = [f"{prefix}{self}"]
        for child in self._children:
            lines.append(child.debug_tree(indent + 1))
        return "\n".join(lines)


__all__ = [
    # Descriptor
    "TrackedDescriptor",
    # Data classes
    "WidgetStyle",
    "LayoutConstraints",
    # Main class
    "Widget",
]
