"""
Container widget that can hold and layout children.

Provides specialized container behaviors:
- Child management with automatic layout
- Layout delegation to layout managers
- Clipping support for overflow
- Padding and spacing configuration

Container Types:
    - Container: Basic container with manual layout
    - Box: Linear layout (horizontal or vertical)
    - Stack: Layered children (z-order stacking)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional

from engine.ui.framework.coordinate import Margins, Point, Rect, Size
from engine.ui.framework.widget import LayoutConstraints, Widget, WidgetStyle


class LayoutDirection(Enum):
    """Layout direction for containers."""

    HORIZONTAL = auto()  # Left to right
    VERTICAL = auto()    # Top to bottom


class Alignment(Enum):
    """Alignment options for layout."""

    START = auto()       # Left/Top
    CENTER = auto()      # Center
    END = auto()         # Right/Bottom
    STRETCH = auto()     # Fill available space
    SPACE_BETWEEN = auto()  # Even spacing between items
    SPACE_AROUND = auto()   # Even spacing around items
    SPACE_EVENLY = auto()   # Equal spacing everywhere


class CrossAlignment(Enum):
    """Cross-axis alignment for layout."""

    START = auto()
    CENTER = auto()
    END = auto()
    STRETCH = auto()


@dataclass
class LayoutConfig:
    """
    Configuration for container layout.

    Controls spacing, alignment, and sizing behavior.
    """

    direction: LayoutDirection = LayoutDirection.VERTICAL
    main_alignment: Alignment = Alignment.START
    cross_alignment: CrossAlignment = CrossAlignment.START
    padding: Margins = field(default_factory=Margins.zero)
    spacing: float = 0.0
    wrap: bool = False
    reverse: bool = False

    @classmethod
    def horizontal(
        cls,
        spacing: float = 0.0,
        main_align: Alignment = Alignment.START,
        cross_align: CrossAlignment = CrossAlignment.CENTER,
    ) -> "LayoutConfig":
        """Create horizontal layout config."""
        return cls(
            direction=LayoutDirection.HORIZONTAL,
            main_alignment=main_align,
            cross_alignment=cross_align,
            spacing=spacing,
        )

    @classmethod
    def vertical(
        cls,
        spacing: float = 0.0,
        main_align: Alignment = Alignment.START,
        cross_align: CrossAlignment = CrossAlignment.START,
    ) -> "LayoutConfig":
        """Create vertical layout config."""
        return cls(
            direction=LayoutDirection.VERTICAL,
            main_alignment=main_align,
            cross_alignment=cross_align,
            spacing=spacing,
        )


class Container(Widget):
    """
    Container widget that manages child layout.

    Extends Widget with:
    - Layout configuration
    - Automatic child positioning
    - Clipping support
    - Content size calculation
    """

    __slots__ = (
        "_layout_config",
        "_content_rect",
        "_needs_layout",
    )

    def __init__(
        self,
        name: str = "",
        x: float = 0.0,
        y: float = 0.0,
        width: float = 0.0,
        height: float = 0.0,
        layout_config: Optional[LayoutConfig] = None,
        clip_children: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize a container.

        Args:
            name: Container name.
            x: X position.
            y: Y position.
            width: Width.
            height: Height.
            layout_config: Layout configuration.
            clip_children: Whether to clip children to bounds.
            **kwargs: Additional widget arguments.
        """
        super().__init__(
            name=name,
            x=x,
            y=y,
            width=width,
            height=height,
            **kwargs,
        )
        self._layout_config = layout_config or LayoutConfig()
        self._clip_children = clip_children
        self._content_rect = Rect.zero()
        self._needs_layout = True

    @property
    def layout_config(self) -> LayoutConfig:
        """Layout configuration."""
        return self._layout_config

    @layout_config.setter
    def layout_config(self, value: LayoutConfig) -> None:
        self._layout_config = value
        self._needs_layout = True
        self._mark_dirty("layout")

    @property
    def padding(self) -> Margins:
        """Container padding."""
        return self._layout_config.padding

    @padding.setter
    def padding(self, value: Margins) -> None:
        self._layout_config.padding = value
        self._needs_layout = True
        self._mark_dirty("layout")

    @property
    def spacing(self) -> float:
        """Spacing between children."""
        return self._layout_config.spacing

    @spacing.setter
    def spacing(self, value: float) -> None:
        self._layout_config.spacing = value
        self._needs_layout = True
        self._mark_dirty("layout")

    @property
    def content_rect(self) -> Rect:
        """Rectangle containing all content (may exceed bounds)."""
        return self._content_rect

    @property
    def content_size(self) -> Size:
        """Size of all content."""
        return self._content_rect.size

    @property
    def inner_rect(self) -> Rect:
        """Rectangle inside padding."""
        return self._layout_config.padding.apply_to_rect(
            Rect(0, 0, self.width, self.height)
        )

    def _on_property_changed(self, name: str, old_value: Any, new_value: Any) -> None:
        """Handle property changes."""
        super()._on_property_changed(name, old_value, new_value)

        # Mark layout needed when size changes
        if name in ("width", "height"):
            self._needs_layout = True

    def add_child(self, child: Widget) -> Widget:
        """Add child and mark layout needed."""
        result = super().add_child(child)
        self._needs_layout = True
        return result

    def remove_child(self, child: Widget) -> bool:
        """Remove child and mark layout needed."""
        result = super().remove_child(child)
        if result:
            self._needs_layout = True
        return result

    def layout(self, available_rect: Optional[Rect] = None) -> None:
        """
        Perform layout for all children.

        Args:
            available_rect: Available space for layout.
        """
        if not self._needs_layout and not self.is_dirty("layout"):
            return

        config = self._layout_config
        padding = config.padding

        # Calculate inner area
        inner_x = padding.left
        inner_y = padding.top
        inner_width = max(0, self.width - padding.horizontal)
        inner_height = max(0, self.height - padding.vertical)

        # Get visible children
        children = [c for c in self._children if c.visible]

        if not children:
            self._content_rect = Rect(inner_x, inner_y, 0, 0)
            self._needs_layout = False
            self._clear_dirty("layout")
            return

        # Reverse order if needed
        if config.reverse:
            children = list(reversed(children))

        # Calculate layout based on direction
        if config.direction == LayoutDirection.HORIZONTAL:
            self._layout_horizontal(children, inner_x, inner_y, inner_width, inner_height)
        else:
            self._layout_vertical(children, inner_x, inner_y, inner_width, inner_height)

        # Layout children recursively
        for child in children:
            if isinstance(child, Container):
                child.layout(child.local_rect)

        self._needs_layout = False
        self._clear_dirty("layout")

    def _layout_horizontal(
        self,
        children: List[Widget],
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        """Layout children horizontally."""
        config = self._layout_config
        spacing = config.spacing

        # Measure total children size
        total_width = sum(c.width for c in children)
        total_spacing = spacing * max(0, len(children) - 1)
        total_size = total_width + total_spacing
        remaining = width - total_size

        # Calculate starting position based on alignment
        current_x = x
        gap = spacing

        if config.main_alignment == Alignment.CENTER:
            current_x = x + remaining / 2
        elif config.main_alignment == Alignment.END:
            current_x = x + remaining
        elif config.main_alignment == Alignment.SPACE_BETWEEN:
            if len(children) > 1:
                gap = (width - total_width) / (len(children) - 1)
        elif config.main_alignment == Alignment.SPACE_AROUND:
            around = remaining / len(children)
            current_x = x + around / 2
            gap = spacing + around
        elif config.main_alignment == Alignment.SPACE_EVENLY:
            even = remaining / (len(children) + 1)
            current_x = x + even
            gap = spacing + even

        # Position each child
        max_height = 0.0
        for child in children:
            child.local_x = current_x

            # Cross-axis alignment
            if config.cross_alignment == CrossAlignment.START:
                child.local_y = y
            elif config.cross_alignment == CrossAlignment.CENTER:
                child.local_y = y + (height - child.height) / 2
            elif config.cross_alignment == CrossAlignment.END:
                child.local_y = y + height - child.height
            elif config.cross_alignment == CrossAlignment.STRETCH:
                child.local_y = y
                child.height = height

            current_x += child.width + gap
            max_height = max(max_height, child.height)

        # Update content rect
        content_width = current_x - gap - x
        self._content_rect = Rect(x, y, content_width, max_height)

    def _layout_vertical(
        self,
        children: List[Widget],
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        """Layout children vertically."""
        config = self._layout_config
        spacing = config.spacing

        # Measure total children size
        total_height = sum(c.height for c in children)
        total_spacing = spacing * max(0, len(children) - 1)
        total_size = total_height + total_spacing
        remaining = height - total_size

        # Calculate starting position based on alignment
        current_y = y
        gap = spacing

        if config.main_alignment == Alignment.CENTER:
            current_y = y + remaining / 2
        elif config.main_alignment == Alignment.END:
            current_y = y + remaining
        elif config.main_alignment == Alignment.SPACE_BETWEEN:
            if len(children) > 1:
                gap = (height - total_height) / (len(children) - 1)
        elif config.main_alignment == Alignment.SPACE_AROUND:
            around = remaining / len(children)
            current_y = y + around / 2
            gap = spacing + around
        elif config.main_alignment == Alignment.SPACE_EVENLY:
            even = remaining / (len(children) + 1)
            current_y = y + even
            gap = spacing + even

        # Position each child
        max_width = 0.0
        for child in children:
            child.local_y = current_y

            # Cross-axis alignment
            if config.cross_alignment == CrossAlignment.START:
                child.local_x = x
            elif config.cross_alignment == CrossAlignment.CENTER:
                child.local_x = x + (width - child.width) / 2
            elif config.cross_alignment == CrossAlignment.END:
                child.local_x = x + width - child.width
            elif config.cross_alignment == CrossAlignment.STRETCH:
                child.local_x = x
                child.width = width

            current_y += child.height + gap
            max_width = max(max_width, child.width)

        # Update content rect
        content_height = current_y - gap - y
        self._content_rect = Rect(x, y, max_width, content_height)

    def measure(self) -> Size:
        """
        Measure preferred size based on children.

        Returns:
            Preferred size to contain all children.
        """
        if not self._children:
            return Size(
                self._layout_config.padding.horizontal,
                self._layout_config.padding.vertical,
            )

        config = self._layout_config
        padding = config.padding
        spacing = config.spacing

        visible_children = [c for c in self._children if c.visible]

        if config.direction == LayoutDirection.HORIZONTAL:
            width = sum(c.width for c in visible_children)
            width += spacing * max(0, len(visible_children) - 1)
            height = max((c.height for c in visible_children), default=0)
        else:
            width = max((c.width for c in visible_children), default=0)
            height = sum(c.height for c in visible_children)
            height += spacing * max(0, len(visible_children) - 1)

        return Size(
            width + padding.horizontal,
            height + padding.vertical,
        )

    def fit_to_content(self) -> None:
        """Resize container to fit its content."""
        preferred = self.measure()
        self.width = preferred.width
        self.height = preferred.height


class HBox(Container):
    """Horizontal container (children laid out left to right)."""

    def __init__(
        self,
        name: str = "",
        spacing: float = 0.0,
        main_align: Alignment = Alignment.START,
        cross_align: CrossAlignment = CrossAlignment.CENTER,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            layout_config=LayoutConfig.horizontal(
                spacing=spacing,
                main_align=main_align,
                cross_align=cross_align,
            ),
            **kwargs,
        )


class VBox(Container):
    """Vertical container (children laid out top to bottom)."""

    def __init__(
        self,
        name: str = "",
        spacing: float = 0.0,
        main_align: Alignment = Alignment.START,
        cross_align: CrossAlignment = CrossAlignment.START,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            layout_config=LayoutConfig.vertical(
                spacing=spacing,
                main_align=main_align,
                cross_align=cross_align,
            ),
            **kwargs,
        )


class Stack(Container):
    """
    Stacked container where children overlap.

    Children are stacked on top of each other, with z_index
    determining the visual order. Layout is not applied to
    children positions.
    """

    def __init__(
        self,
        name: str = "",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)

    def layout(self, available_rect: Optional[Rect] = None) -> None:
        """
        Layout stacked children.

        Children maintain their individual positions but are
        constrained to container bounds if clip_children is True.
        """
        padding = self._layout_config.padding
        inner_rect = padding.apply_to_rect(Rect(0, 0, self.width, self.height))

        # Calculate content rect from all children
        if self._children:
            min_x = min((c.local_x for c in self._children), default=0)
            min_y = min((c.local_y for c in self._children), default=0)
            max_x = max((c.local_x + c.width for c in self._children), default=0)
            max_y = max((c.local_y + c.height for c in self._children), default=0)
            self._content_rect = Rect(min_x, min_y, max_x - min_x, max_y - min_y)
        else:
            self._content_rect = Rect.zero()

        # Layout children recursively
        for child in self._children:
            if isinstance(child, Container):
                child.layout(child.local_rect)

        self._needs_layout = False
        self._clear_dirty("layout")


class ScrollContainer(Container):
    """
    Container that supports scrolling content.

    Provides scroll offset and viewport clipping for content
    that exceeds the container bounds.
    """

    __slots__ = (
        "_scroll_x",
        "_scroll_y",
        "_scroll_enabled_x",
        "_scroll_enabled_y",
    )

    def __init__(
        self,
        name: str = "",
        scroll_x: bool = False,
        scroll_y: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name=name, clip_children=True, **kwargs)
        self._scroll_x = 0.0
        self._scroll_y = 0.0
        self._scroll_enabled_x = scroll_x
        self._scroll_enabled_y = scroll_y

    @property
    def scroll_x(self) -> float:
        """Horizontal scroll offset."""
        return self._scroll_x

    @scroll_x.setter
    def scroll_x(self, value: float) -> None:
        if not self._scroll_enabled_x:
            return
        max_scroll = max(0, self._content_rect.width - self.inner_rect.width)
        self._scroll_x = max(0, min(value, max_scroll))
        self._mark_dirty("scroll")

    @property
    def scroll_y(self) -> float:
        """Vertical scroll offset."""
        return self._scroll_y

    @scroll_y.setter
    def scroll_y(self, value: float) -> None:
        if not self._scroll_enabled_y:
            return
        max_scroll = max(0, self._content_rect.height - self.inner_rect.height)
        self._scroll_y = max(0, min(value, max_scroll))
        self._mark_dirty("scroll")

    @property
    def scroll_offset(self) -> Point:
        """Current scroll offset."""
        return Point(self._scroll_x, self._scroll_y)

    @scroll_offset.setter
    def scroll_offset(self, value: Point) -> None:
        self.scroll_x = value.x
        self.scroll_y = value.y

    @property
    def max_scroll_x(self) -> float:
        """Maximum horizontal scroll value."""
        return max(0, self._content_rect.width - self.inner_rect.width)

    @property
    def max_scroll_y(self) -> float:
        """Maximum vertical scroll value."""
        return max(0, self._content_rect.height - self.inner_rect.height)

    @property
    def viewport_rect(self) -> Rect:
        """Visible viewport rectangle in content space."""
        inner = self.inner_rect
        return Rect(
            self._scroll_x,
            self._scroll_y,
            inner.width,
            inner.height,
        )

    def scroll_to(self, x: float, y: float, animated: bool = False) -> None:
        """
        Scroll to a specific position.

        Args:
            x: Target x scroll position.
            y: Target y scroll position.
            animated: Whether to animate the scroll (future).
        """
        self.scroll_x = x
        self.scroll_y = y

    def scroll_by(self, dx: float, dy: float) -> None:
        """
        Scroll by a delta amount.

        Args:
            dx: Horizontal scroll delta.
            dy: Vertical scroll delta.
        """
        self.scroll_x = self._scroll_x + dx
        self.scroll_y = self._scroll_y + dy

    def scroll_to_child(self, child: Widget) -> None:
        """
        Scroll to make a child visible.

        Args:
            child: Child widget to scroll to.
        """
        if child not in self._children:
            return

        viewport = self.viewport_rect
        child_rect = child.local_rect

        # Scroll horizontally if needed
        if self._scroll_enabled_x:
            if child_rect.left < viewport.left:
                self.scroll_x = child_rect.left
            elif child_rect.right > viewport.right:
                self.scroll_x = child_rect.right - viewport.width

        # Scroll vertically if needed
        if self._scroll_enabled_y:
            if child_rect.top < viewport.top:
                self.scroll_y = child_rect.top
            elif child_rect.bottom > viewport.bottom:
                self.scroll_y = child_rect.bottom - viewport.height

    def hit_test(self, point: Point) -> Optional[Widget]:
        """Hit test accounting for scroll offset."""
        if not self.visible:
            return None

        if not self.global_rect.contains_point(point):
            return None

        # Adjust point for scroll offset
        local = self.global_to_local(point)
        scrolled = Point(local.x + self._scroll_x, local.y + self._scroll_y)

        # Check if in inner bounds
        if not self.inner_rect.contains_point(local):
            return self if self.enabled else None

        # Check children with scrolled coordinates
        for child in reversed(self._children):
            if not child.visible:
                continue
            child_local = Point(
                scrolled.x - child.local_x,
                scrolled.y - child.local_y,
            )
            if Rect(0, 0, child.width, child.height).contains_point(child_local):
                hit = child.hit_test(
                    Point(
                        point.x - self._scroll_x,
                        point.y - self._scroll_y,
                    )
                )
                if hit is not None:
                    return hit

        return self if self.enabled else None


__all__ = [
    # Enums
    "LayoutDirection",
    "Alignment",
    "CrossAlignment",
    # Data classes
    "LayoutConfig",
    # Container classes
    "Container",
    "HBox",
    "VBox",
    "Stack",
    "ScrollContainer",
]
