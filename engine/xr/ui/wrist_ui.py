"""Wrist-attached UI for XR (watch-style interface).

Provides a wrist-mounted quick access menu with:
- Watch-style circular or rectangular layout
- Automatic positioning based on wrist tracking
- Visibility toggle based on wrist orientation
- Quick action buttons
- Status indicators
- Haptic feedback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Any
import math

from engine.xr.config import XR_CONFIG


class WristUIPosition(Enum):
    """Which wrist the UI is attached to."""
    LEFT = auto()
    RIGHT = auto()


class WristUILayout(Enum):
    """Layout style for wrist UI."""
    CIRCULAR = auto()  # Watch face style
    RECTANGULAR = auto()  # Smartwatch app grid
    RADIAL = auto()  # Pie menu style


class WristUIVisibilityMode(Enum):
    """How wrist UI visibility is controlled."""
    ALWAYS = auto()  # Always visible when active
    LOOK_AT = auto()  # Visible when user looks at wrist
    PALM_UP = auto()  # Visible when palm faces up
    MANUAL = auto()  # Manually toggled


@dataclass(slots=True)
class WristMenuItem:
    """A menu item on the wrist UI.

    Attributes:
        id: Unique identifier
        label: Display text
        icon: Icon identifier or path
        action: Callback when selected
        is_enabled: Whether item can be selected
        is_highlighted: Whether item is visually highlighted
        badge_count: Number for notification badge (0 = no badge)
        color: Item color (r, g, b, a)
    """
    id: str
    label: str = ""
    icon: str = ""
    action: Optional[Callable[[], None]] = None
    is_enabled: bool = True
    is_highlighted: bool = False
    is_hovered: bool = False
    badge_count: int = 0
    color: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0)
    _index: int = 0

    def execute(self) -> bool:
        """Execute the menu item action.

        Returns:
            True if action was executed
        """
        if self.is_enabled and self.action:
            self.action()
            return True
        return False


@dataclass(slots=True)
class WristUIStyle:
    """Visual style for wrist UI."""
    background_color: tuple[float, float, float, float] = (0.1, 0.1, 0.15, 0.9)
    item_color: tuple[float, float, float, float] = (0.25, 0.25, 0.3, 1.0)
    item_hover_color: tuple[float, float, float, float] = (0.35, 0.35, 0.45, 1.0)
    item_disabled_color: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.5)
    highlight_color: tuple[float, float, float, float] = (0.2, 0.5, 0.8, 1.0)
    text_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    badge_color: tuple[float, float, float, float] = (0.9, 0.2, 0.2, 1.0)
    border_radius: float = 0.005  # Meters
    item_size: float = 0.025  # Meters
    item_spacing: float = 0.005  # Meters
    font_size: float = 0.008  # Meters


@dataclass(slots=True)
class WristUI:
    """Wrist-attached UI component.

    Provides a quick-access menu attached to the user's wrist,
    similar to a smartwatch interface.

    Attributes:
        position: Which wrist to attach to
        layout: UI layout style
        visibility_mode: How visibility is controlled
        size: Diameter/width of the UI in meters
        items: Menu items
        is_visible: Current visibility state
        is_active: Whether UI is active and accepting input
        look_threshold: Angle threshold for look-at visibility (degrees)
        palm_up_threshold: Angle threshold for palm-up visibility (degrees)
    """
    position: WristUIPosition = WristUIPosition.LEFT
    layout: WristUILayout = WristUILayout.CIRCULAR
    visibility_mode: WristUIVisibilityMode = WristUIVisibilityMode.LOOK_AT
    size: float = XR_CONFIG.ui.WRIST_UI_SIZE  # Meters (diameter for circular, width for rectangular)
    items: list[WristMenuItem] = field(default_factory=list)
    is_visible: bool = False
    is_active: bool = True
    look_threshold: float = XR_CONFIG.ui.WRIST_LOOK_THRESHOLD_DEGREES  # Degrees
    palm_up_threshold: float = XR_CONFIG.ui.WRIST_PALM_UP_THRESHOLD_DEGREES  # Degrees
    style: WristUIStyle = field(default_factory=WristUIStyle)
    _wrist_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _wrist_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    _head_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _head_forward: tuple[float, float, float] = (0.0, 0.0, -1.0)
    _hovered_item: Optional[WristMenuItem] = None
    _on_visibility_changed: Optional[Callable[[bool], None]] = None
    _on_item_selected: Optional[Callable[[WristMenuItem], None]] = None
    _parent: Any = None
    _time_visible: float = 0.0

    @property
    def center_position(self) -> tuple[float, float, float]:
        """Get center position of the wrist UI in world space."""
        # Offset from wrist based on which hand
        wx, wy, wz = self._wrist_position
        offset = XR_CONFIG.ui.WRIST_OFFSET_X if self.position == WristUIPosition.LEFT else -XR_CONFIG.ui.WRIST_OFFSET_X
        return (wx + offset, wy + XR_CONFIG.ui.WRIST_OFFSET_Y, wz)

    @property
    def item_count(self) -> int:
        """Get number of menu items."""
        return len(self.items)

    @property
    def max_items(self) -> int:
        """Get maximum items based on layout."""
        if self.layout == WristUILayout.CIRCULAR:
            return 8  # Around the edge
        elif self.layout == WristUILayout.RADIAL:
            return 8  # Pie slices
        else:
            return 12  # 4x3 grid

    def add_item(self, item: WristMenuItem) -> bool:
        """Add a menu item.

        Args:
            item: Menu item to add

        Returns:
            True if item was added
        """
        if len(self.items) >= self.max_items:
            return False

        item._index = len(self.items)
        self.items.append(item)
        return True

    def remove_item(self, item_id: str) -> bool:
        """Remove a menu item by ID.

        Args:
            item_id: ID of item to remove

        Returns:
            True if item was removed
        """
        for i, item in enumerate(self.items):
            if item.id == item_id:
                self.items.pop(i)
                # Update indices
                for j in range(i, len(self.items)):
                    self.items[j]._index = j
                return True
        return False

    def get_item(self, item_id: str) -> Optional[WristMenuItem]:
        """Get menu item by ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def clear_items(self) -> None:
        """Remove all menu items."""
        self.items.clear()
        self._hovered_item = None

    def show(self) -> None:
        """Show the wrist UI."""
        was_visible = self.is_visible
        self.is_visible = True
        self._time_visible = 0.0

        if not was_visible and self._on_visibility_changed:
            self._on_visibility_changed(True)

    def hide(self) -> None:
        """Hide the wrist UI."""
        was_visible = self.is_visible
        self.is_visible = False
        self._clear_hover()

        if was_visible and self._on_visibility_changed:
            self._on_visibility_changed(False)

    def toggle(self) -> None:
        """Toggle visibility."""
        if self.is_visible:
            self.hide()
        else:
            self.show()

    def activate(self) -> None:
        """Activate UI for input."""
        self.is_active = True

    def deactivate(self) -> None:
        """Deactivate UI (ignores input)."""
        self.is_active = False
        self._clear_hover()

    def on_visibility_changed(self, callback: Callable[[bool], None]) -> None:
        """Set visibility change callback."""
        self._on_visibility_changed = callback

    def on_item_selected(self, callback: Callable[[WristMenuItem], None]) -> None:
        """Set item selection callback."""
        self._on_item_selected = callback

    def update_wrist_tracking(
        self,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        """Update wrist tracking data.

        Args:
            position: Wrist position in world space
            orientation: Wrist orientation as quaternion
        """
        self._wrist_position = position
        self._wrist_orientation = orientation

    def update_head_tracking(
        self,
        position: tuple[float, float, float],
        forward: tuple[float, float, float],
    ) -> None:
        """Update head tracking data for look-at detection.

        Args:
            position: Head/eye position in world space
            forward: Head forward direction (normalized)
        """
        self._head_position = position
        self._head_forward = forward

    def update(self, delta_time: float) -> bool:
        """Update wrist UI state.

        Args:
            delta_time: Time since last update

        Returns:
            True if visibility changed
        """
        if not self.is_active:
            return False

        old_visible = self.is_visible

        if self.visibility_mode == WristUIVisibilityMode.ALWAYS:
            self.is_visible = True

        elif self.visibility_mode == WristUIVisibilityMode.LOOK_AT:
            # Check if user is looking at wrist
            self.is_visible = self._check_look_at()

        elif self.visibility_mode == WristUIVisibilityMode.PALM_UP:
            # Check if palm is facing up
            self.is_visible = self._check_palm_up()

        # MANUAL mode doesn't auto-update visibility

        if self.is_visible:
            self._time_visible += delta_time

        if old_visible != self.is_visible:
            if self._on_visibility_changed:
                self._on_visibility_changed(self.is_visible)
            return True

        return False

    def _check_look_at(self) -> bool:
        """Check if user is looking at wrist.

        Returns:
            True if looking at wrist
        """
        # Direction from head to wrist
        hx, hy, hz = self._head_position
        wx, wy, wz = self.center_position

        dx = wx - hx
        dy = wy - hy
        dz = wz - hz

        # Normalize
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length < XR_CONFIG.ui.WRIST_LENGTH_EPSILON:
            return False

        dx /= length
        dy /= length
        dz /= length

        # Dot product with head forward
        fx, fy, fz = self._head_forward
        dot = dx*fx + dy*fy + dz*fz

        # Convert to angle
        angle = math.degrees(math.acos(max(-1, min(1, dot))))

        return angle < self.look_threshold

    def _check_palm_up(self) -> bool:
        """Check if palm is facing up.

        Returns:
            True if palm is up
        """
        # Extract up vector from quaternion
        # Simplified - full implementation would use proper quaternion math
        qx, qy, qz, qw = self._wrist_orientation

        # Palm up vector (assuming standard hand orientation)
        # This is simplified; real implementation needs proper rotation
        up_y = 1 - 2*(qx*qx + qz*qz)

        # Check if up vector points upward
        angle = math.degrees(math.acos(max(-1, min(1, up_y))))

        return angle < self.palm_up_threshold

    def _clear_hover(self) -> None:
        """Clear hover state on all items."""
        for item in self.items:
            item.is_hovered = False
        self._hovered_item = None

    def hover_item(self, index: int) -> Optional[WristMenuItem]:
        """Set hover on item at index.

        Args:
            index: Item index

        Returns:
            Hovered item or None
        """
        self._clear_hover()

        if 0 <= index < len(self.items):
            item = self.items[index]
            if item.is_enabled:
                item.is_hovered = True
                self._hovered_item = item
                return item

        return None

    def hover_at_position(
        self,
        local_x: float,
        local_y: float
    ) -> Optional[WristMenuItem]:
        """Set hover based on local position on UI.

        Args:
            local_x: X position relative to center (-1 to 1)
            local_y: Y position relative to center (-1 to 1)

        Returns:
            Hovered item or None
        """
        index = self._position_to_index(local_x, local_y)
        return self.hover_item(index) if index >= 0 else None

    def _position_to_index(self, x: float, y: float) -> int:
        """Convert position to item index based on layout.

        Args:
            x: X position (-1 to 1)
            y: Y position (-1 to 1)

        Returns:
            Item index or -1 if not on an item
        """
        if not self.items:
            return -1

        if self.layout == WristUILayout.CIRCULAR:
            # Items arranged around edge
            angle = math.atan2(y, x)
            if angle < 0:
                angle += 2 * math.pi

            # Map angle to item index
            index = int(angle / (2 * math.pi) * len(self.items))
            return min(index, len(self.items) - 1)

        elif self.layout == WristUILayout.RADIAL:
            # Pie slices
            angle = math.atan2(y, x)
            if angle < 0:
                angle += 2 * math.pi

            index = int(angle / (2 * math.pi) * len(self.items))
            return min(index, len(self.items) - 1)

        else:  # RECTANGULAR
            # Grid layout
            cols = min(4, len(self.items))
            rows = (len(self.items) + cols - 1) // cols

            col = int((x + 1) / 2 * cols)
            row = int((1 - y) / 2 * rows)

            col = max(0, min(col, cols - 1))
            row = max(0, min(row, rows - 1))

            index = row * cols + col
            return index if index < len(self.items) else -1

    def select_hovered(self) -> bool:
        """Select the currently hovered item.

        Returns:
            True if an item was selected
        """
        if self._hovered_item and self._hovered_item.is_enabled:
            if self._on_item_selected:
                self._on_item_selected(self._hovered_item)
            return self._hovered_item.execute()
        return False

    def select_item(self, item_id: str) -> bool:
        """Select item by ID.

        Args:
            item_id: ID of item to select

        Returns:
            True if item was selected
        """
        item = self.get_item(item_id)
        if item and item.is_enabled:
            if self._on_item_selected:
                self._on_item_selected(item)
            return item.execute()
        return False

    def get_item_position(self, index: int) -> tuple[float, float]:
        """Get local position of item at index.

        Args:
            index: Item index

        Returns:
            Local position (x, y) in meters from center
        """
        if index < 0 or index >= len(self.items):
            return (0.0, 0.0)

        if self.layout == WristUILayout.CIRCULAR:
            # Arrange around edge
            angle = (index / len(self.items)) * 2 * math.pi - math.pi / 2
            radius = self.size / 2 - self.style.item_size / 2
            return (math.cos(angle) * radius, math.sin(angle) * radius)

        elif self.layout == WristUILayout.RADIAL:
            # Pie slice centers
            angle = (index / len(self.items)) * 2 * math.pi - math.pi / 2
            radius = self.size / 4
            return (math.cos(angle) * radius, math.sin(angle) * radius)

        else:  # RECTANGULAR
            # Grid positions
            cols = min(4, len(self.items))
            col = index % cols
            row = index // cols

            total_width = cols * self.style.item_size + (cols - 1) * self.style.item_spacing
            total_height = ((len(self.items) + cols - 1) // cols) * self.style.item_size

            x = -total_width / 2 + col * (self.style.item_size + self.style.item_spacing) + self.style.item_size / 2
            y = total_height / 2 - row * (self.style.item_size + self.style.item_spacing) - self.style.item_size / 2

            return (x, y)


class WristUIManager:
    """Manages wrist UI instances for both hands."""

    __slots__ = ('_left_ui', '_right_ui', '_active_hand')

    def __init__(self):
        """Initialize wrist UI manager."""
        self._left_ui: Optional[WristUI] = None
        self._right_ui: Optional[WristUI] = None
        self._active_hand: Optional[WristUIPosition] = None

    def create_left(self, **kwargs) -> WristUI:
        """Create left wrist UI."""
        self._left_ui = WristUI(position=WristUIPosition.LEFT, **kwargs)
        return self._left_ui

    def create_right(self, **kwargs) -> WristUI:
        """Create right wrist UI."""
        self._right_ui = WristUI(position=WristUIPosition.RIGHT, **kwargs)
        return self._right_ui

    @property
    def left(self) -> Optional[WristUI]:
        """Get left wrist UI."""
        return self._left_ui

    @property
    def right(self) -> Optional[WristUI]:
        """Get right wrist UI."""
        return self._right_ui

    def get(self, position: WristUIPosition) -> Optional[WristUI]:
        """Get wrist UI by position."""
        if position == WristUIPosition.LEFT:
            return self._left_ui
        return self._right_ui

    def update(
        self,
        delta_time: float,
        left_wrist_pos: Optional[tuple[float, float, float]] = None,
        left_wrist_rot: Optional[tuple[float, float, float, float]] = None,
        right_wrist_pos: Optional[tuple[float, float, float]] = None,
        right_wrist_rot: Optional[tuple[float, float, float, float]] = None,
        head_pos: Optional[tuple[float, float, float]] = None,
        head_forward: Optional[tuple[float, float, float]] = None,
    ) -> None:
        """Update both wrist UIs with tracking data.

        Args:
            delta_time: Time since last update
            left_wrist_pos: Left wrist position
            left_wrist_rot: Left wrist rotation
            right_wrist_pos: Right wrist position
            right_wrist_rot: Right wrist rotation
            head_pos: Head position
            head_forward: Head forward direction
        """
        if self._left_ui:
            if left_wrist_pos:
                self._left_ui.update_wrist_tracking(
                    left_wrist_pos,
                    left_wrist_rot or (0, 0, 0, 1)
                )
            if head_pos:
                self._left_ui.update_head_tracking(
                    head_pos,
                    head_forward or (0, 0, -1)
                )
            self._left_ui.update(delta_time)

        if self._right_ui:
            if right_wrist_pos:
                self._right_ui.update_wrist_tracking(
                    right_wrist_pos,
                    right_wrist_rot or (0, 0, 0, 1)
                )
            if head_pos:
                self._right_ui.update_head_tracking(
                    head_pos,
                    head_forward or (0, 0, -1)
                )
            self._right_ui.update(delta_time)

    def show_all(self) -> None:
        """Show both wrist UIs."""
        if self._left_ui:
            self._left_ui.show()
        if self._right_ui:
            self._right_ui.show()

    def hide_all(self) -> None:
        """Hide both wrist UIs."""
        if self._left_ui:
            self._left_ui.hide()
        if self._right_ui:
            self._right_ui.hide()
