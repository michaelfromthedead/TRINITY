"""
Focus management system for UI framework.

Handles:
- Focus state tracking
- Tab navigation order
- Focus groups for logical groupings
- Focus trapping for modals/dialogs
- Focus events and callbacks

Focus Navigation:
    - Tab: Move to next focusable widget
    - Shift+Tab: Move to previous focusable widget
    - Arrow keys: Navigate within groups (optional)
    - Escape: Exit focus trap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
)
# Note: weakref imports removed - they were unused (WeakSet, ref, ReferenceType)

from engine.ui.config import FOCUS
from engine.ui.framework.events import (
    EventType,
    FocusEvent,
    KeyboardEvent,
    KeyModifier,
)

if TYPE_CHECKING:
    from engine.ui.framework.widget import Widget


class FocusDirection(Enum):
    """Direction of focus navigation."""

    NEXT = auto()      # Tab forward
    PREVIOUS = auto()  # Shift+Tab backward
    UP = auto()        # Arrow up
    DOWN = auto()      # Arrow down
    LEFT = auto()      # Arrow left
    RIGHT = auto()     # Arrow right


class FocusReason(Enum):
    """Reason for focus change."""

    PROGRAMMATIC = auto()  # Set by code
    TAB = auto()           # Tab navigation
    CLICK = auto()         # Mouse click
    KEY = auto()           # Keyboard navigation
    RESTORE = auto()       # Restored from saved state


@dataclass
class FocusGroup:
    """
    A logical group of focusable widgets.

    Groups allow related widgets to be navigated as a unit
    and can have their own tab order.
    """

    name: str
    widgets: List["Widget"] = field(default_factory=list)
    tab_order: List["Widget"] = field(default_factory=list)
    wrap_navigation: bool = True
    arrow_navigation: bool = False

    def add(self, widget: "Widget") -> None:
        """Add widget to group."""
        if widget not in self.widgets:
            self.widgets.append(widget)
            self._rebuild_tab_order()

    def remove(self, widget: "Widget") -> bool:
        """Remove widget from group."""
        if widget in self.widgets:
            self.widgets.remove(widget)
            self._rebuild_tab_order()
            return True
        return False

    def clear(self) -> None:
        """Remove all widgets from group."""
        self.widgets.clear()
        self.tab_order.clear()

    def _rebuild_tab_order(self) -> None:
        """Rebuild tab order based on widget tab_index."""
        self.tab_order = sorted(
            [w for w in self.widgets if w.focusable and w.visible and w.enabled],
            key=lambda w: w.tab_index,
        )

    def get_first(self) -> Optional["Widget"]:
        """Get first focusable widget in group."""
        self._rebuild_tab_order()
        return self.tab_order[0] if self.tab_order else None

    def get_last(self) -> Optional["Widget"]:
        """Get last focusable widget in group."""
        self._rebuild_tab_order()
        return self.tab_order[-1] if self.tab_order else None

    def get_next(self, current: Optional["Widget"]) -> Optional["Widget"]:
        """Get next widget after current."""
        self._rebuild_tab_order()
        if not self.tab_order:
            return None

        if current is None or current not in self.tab_order:
            return self.tab_order[0]

        idx = self.tab_order.index(current)
        next_idx = idx + 1

        if next_idx >= len(self.tab_order):
            return self.tab_order[0] if self.wrap_navigation else None

        return self.tab_order[next_idx]

    def get_previous(self, current: Optional["Widget"]) -> Optional["Widget"]:
        """Get previous widget before current."""
        self._rebuild_tab_order()
        if not self.tab_order:
            return None

        if current is None or current not in self.tab_order:
            return self.tab_order[-1]

        idx = self.tab_order.index(current)
        prev_idx = idx - 1

        if prev_idx < 0:
            return self.tab_order[-1] if self.wrap_navigation else None

        return self.tab_order[prev_idx]


@dataclass
class FocusTrap:
    """
    A focus trap that prevents focus from leaving a container.

    Used for modals, dialogs, and other overlay widgets that
    should contain focus until dismissed.
    """

    container: "Widget"
    previous_focus: Optional["Widget"] = None
    restore_on_exit: bool = True
    initial_focus: Optional["Widget"] = None
    group: Optional[FocusGroup] = None

    def __post_init__(self) -> None:
        if self.group is None:
            self.group = FocusGroup(name=f"trap_{id(self.container)}")
            self._populate_group()

    def _populate_group(self) -> None:
        """Populate focus group from container descendants."""
        if self.group is None:
            return

        self.group.clear()

        def collect_focusable(widget: "Widget") -> None:
            if widget.focusable and widget.visible and widget.enabled:
                self.group.add(widget)
            for child in widget.children:
                collect_focusable(child)

        collect_focusable(self.container)


# Type for focus change callbacks
FocusCallback = Callable[["Widget", Optional["Widget"], FocusReason], None]


class FocusManager:
    """
    Singleton manager for UI focus state.

    Handles:
    - Tracking current focused widget
    - Tab navigation between focusable widgets
    - Focus groups for logical organization
    - Focus trapping for modals
    - Focus history for restoration

    Usage:
        focus_manager = FocusManager.get_instance()
        focus_manager.set_focus(my_widget)
        focus_manager.navigate(FocusDirection.NEXT)
    """

    _instance: Optional["FocusManager"] = None

    def __init__(self) -> None:
        """Initialize focus manager (use get_instance() instead)."""
        self._focused: Optional["Widget"] = None
        self._focus_history: List["Widget"] = []
        self._max_history = FOCUS.MAX_HISTORY_SIZE
        self._groups: Dict[str, FocusGroup] = {}
        self._traps: List[FocusTrap] = []
        self._default_group = FocusGroup(name="default")
        self._callbacks: List[FocusCallback] = []
        self._root: Optional["Widget"] = None

    @classmethod
    def get_instance(cls) -> "FocusManager":
        """Get the singleton focus manager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    # ============================================
    # Focus State
    # ============================================

    @property
    def focused(self) -> Optional["Widget"]:
        """Currently focused widget."""
        return self._focused

    @property
    def has_focus(self) -> bool:
        """Whether any widget has focus."""
        return self._focused is not None

    def set_root(self, root: "Widget") -> None:
        """
        Set the root widget for focus navigation.

        Args:
            root: Root widget of the UI tree.
        """
        self._root = root
        self._rebuild_default_group()

    def _rebuild_default_group(self) -> None:
        """Rebuild default focus group from root."""
        self._default_group.clear()
        if self._root is None:
            return

        def collect(widget: "Widget") -> None:
            if widget.focusable and widget.visible and widget.enabled:
                self._default_group.add(widget)
            for child in widget.children:
                collect(child)

        collect(self._root)

    # ============================================
    # Focus Control
    # ============================================

    def set_focus(
        self,
        widget: Optional["Widget"],
        reason: FocusReason = FocusReason.PROGRAMMATIC,
    ) -> bool:
        """
        Set focus to a widget.

        Args:
            widget: Widget to focus, or None to clear focus.
            reason: Reason for focus change.

        Returns:
            True if focus changed, False if blocked.
        """
        # Check if within active focus trap
        if self._traps and widget is not None:
            trap = self._traps[-1]
            if trap.group and widget not in trap.group.widgets:
                # Widget is outside trap, block focus
                return False

        old_focus = self._focused

        # Skip if already focused
        if old_focus is widget:
            return False

        # Validate widget is focusable
        if widget is not None and not self._can_focus(widget):
            return False

        # Update focus
        self._focused = widget

        # Add to history
        if old_focus is not None:
            self._add_to_history(old_focus)

        # Dispatch events
        self._dispatch_focus_change(old_focus, widget, reason)

        # Notify callbacks
        for callback in self._callbacks:
            callback(widget, old_focus, reason)

        return True

    def clear_focus(self, reason: FocusReason = FocusReason.PROGRAMMATIC) -> None:
        """Clear current focus."""
        self.set_focus(None, reason)

    def _can_focus(self, widget: "Widget") -> bool:
        """Check if widget can receive focus."""
        return (
            widget.focusable and
            widget.visible and
            widget.enabled
        )

    def _dispatch_focus_change(
        self,
        old_widget: Optional["Widget"],
        new_widget: Optional["Widget"],
        reason: FocusReason,
    ) -> None:
        """Dispatch focus events to affected widgets."""
        if old_widget is not None:
            event = FocusEvent.focus_out(
                target=old_widget,
                related_target=new_widget,
            )
            old_widget.dispatch_event(event)

        if new_widget is not None:
            event = FocusEvent.focus_in(
                target=new_widget,
                related_target=old_widget,
            )
            new_widget.dispatch_event(event)

    # ============================================
    # Navigation
    # ============================================

    def navigate(self, direction: FocusDirection) -> bool:
        """
        Navigate focus in a direction.

        Args:
            direction: Direction to navigate.

        Returns:
            True if focus changed.
        """
        # Get active group
        group = self._get_active_group()

        # Get next/previous widget
        if direction == FocusDirection.NEXT:
            target = group.get_next(self._focused)
        elif direction == FocusDirection.PREVIOUS:
            target = group.get_previous(self._focused)
        elif direction in (FocusDirection.UP, FocusDirection.DOWN,
                          FocusDirection.LEFT, FocusDirection.RIGHT):
            if not group.arrow_navigation:
                return False
            target = self._find_spatial_neighbor(direction)
        else:
            return False

        if target is None:
            return False

        reason = FocusReason.TAB if direction in (FocusDirection.NEXT, FocusDirection.PREVIOUS) else FocusReason.KEY
        return self.set_focus(target, reason)

    def _get_active_group(self) -> FocusGroup:
        """Get the currently active focus group."""
        # If trapped, use trap's group
        if self._traps:
            trap = self._traps[-1]
            if trap.group:
                return trap.group

        # Use default group
        self._default_group._rebuild_tab_order()
        return self._default_group

    def _find_spatial_neighbor(self, direction: FocusDirection) -> Optional["Widget"]:
        """Find nearest widget in spatial direction."""
        if self._focused is None:
            return None

        group = self._get_active_group()
        candidates = [w for w in group.tab_order if w is not self._focused]

        if not candidates:
            return None

        # Get current widget center
        current_rect = self._focused.global_rect
        cx = current_rect.center.x
        cy = current_rect.center.y

        best: Optional["Widget"] = None
        best_dist = float("inf")

        for widget in candidates:
            rect = widget.global_rect
            wx = rect.center.x
            wy = rect.center.y

            # Check direction
            dx = wx - cx
            dy = wy - cy

            in_direction = False
            if direction == FocusDirection.UP and dy < 0:
                in_direction = True
            elif direction == FocusDirection.DOWN and dy > 0:
                in_direction = True
            elif direction == FocusDirection.LEFT and dx < 0:
                in_direction = True
            elif direction == FocusDirection.RIGHT and dx > 0:
                in_direction = True

            if not in_direction:
                continue

            # Calculate distance
            dist = (dx * dx + dy * dy) ** 0.5

            if dist < best_dist:
                best_dist = dist
                best = widget

        return best

    def focus_first(self) -> bool:
        """Focus the first focusable widget."""
        group = self._get_active_group()
        first = group.get_first()
        if first:
            return self.set_focus(first, FocusReason.PROGRAMMATIC)
        return False

    def focus_last(self) -> bool:
        """Focus the last focusable widget."""
        group = self._get_active_group()
        last = group.get_last()
        if last:
            return self.set_focus(last, FocusReason.PROGRAMMATIC)
        return False

    # ============================================
    # Focus History
    # ============================================

    def _add_to_history(self, widget: "Widget") -> None:
        """Add widget to focus history."""
        # Remove if already in history
        if widget in self._focus_history:
            self._focus_history.remove(widget)

        # Add to front
        self._focus_history.insert(0, widget)

        # Trim if too long
        while len(self._focus_history) > self._max_history:
            self._focus_history.pop()

    def restore_previous_focus(self) -> bool:
        """
        Restore focus to previous widget in history.

        Returns:
            True if focus was restored.
        """
        while self._focus_history:
            widget = self._focus_history.pop(0)
            if self._can_focus(widget):
                return self.set_focus(widget, FocusReason.RESTORE)
        return False

    # ============================================
    # Focus Groups
    # ============================================

    def create_group(
        self,
        name: str,
        widgets: Optional[List["Widget"]] = None,
        wrap: bool = True,
        arrow_nav: bool = False,
    ) -> FocusGroup:
        """
        Create a new focus group.

        Args:
            name: Group name.
            widgets: Initial widgets.
            wrap: Whether navigation wraps.
            arrow_nav: Whether arrow keys navigate.

        Returns:
            Created focus group.
        """
        group = FocusGroup(
            name=name,
            wrap_navigation=wrap,
            arrow_navigation=arrow_nav,
        )

        if widgets:
            for widget in widgets:
                group.add(widget)

        self._groups[name] = group
        return group

    def get_group(self, name: str) -> Optional[FocusGroup]:
        """Get a focus group by name."""
        return self._groups.get(name)

    def remove_group(self, name: str) -> bool:
        """Remove a focus group."""
        if name in self._groups:
            del self._groups[name]
            return True
        return False

    # ============================================
    # Focus Trapping
    # ============================================

    def push_trap(
        self,
        container: "Widget",
        initial_focus: Optional["Widget"] = None,
        restore_on_exit: bool = True,
    ) -> FocusTrap:
        """
        Push a focus trap for a container.

        Focus will be constrained to the container's descendants
        until the trap is popped.

        Args:
            container: Container to trap focus within.
            initial_focus: Widget to focus initially.
            restore_on_exit: Whether to restore previous focus on pop.

        Returns:
            Created focus trap.
        """
        trap = FocusTrap(
            container=container,
            previous_focus=self._focused,
            restore_on_exit=restore_on_exit,
            initial_focus=initial_focus,
        )

        self._traps.append(trap)

        # Set initial focus
        if initial_focus and self._can_focus(initial_focus):
            self.set_focus(initial_focus, FocusReason.PROGRAMMATIC)
        elif trap.group:
            first = trap.group.get_first()
            if first:
                self.set_focus(first, FocusReason.PROGRAMMATIC)

        return trap

    def pop_trap(self) -> Optional[FocusTrap]:
        """
        Pop the current focus trap.

        Returns:
            Popped trap, or None if no trap active.
        """
        if not self._traps:
            return None

        trap = self._traps.pop()

        # Restore previous focus
        if trap.restore_on_exit and trap.previous_focus:
            if self._can_focus(trap.previous_focus):
                self.set_focus(trap.previous_focus, FocusReason.RESTORE)

        return trap

    @property
    def is_trapped(self) -> bool:
        """Whether focus is currently trapped."""
        return len(self._traps) > 0

    @property
    def current_trap(self) -> Optional[FocusTrap]:
        """Current active focus trap."""
        return self._traps[-1] if self._traps else None

    # ============================================
    # Callbacks
    # ============================================

    def add_callback(self, callback: FocusCallback) -> None:
        """
        Add focus change callback.

        Args:
            callback: Function called on focus change.
                      Receives (new_widget, old_widget, reason).
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: FocusCallback) -> bool:
        """Remove a focus callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    # ============================================
    # Keyboard Handling
    # ============================================

    def handle_key_event(self, event: KeyboardEvent) -> bool:
        """
        Handle keyboard event for focus navigation.

        Args:
            event: Keyboard event.

        Returns:
            True if event was handled.
        """
        if event.event_type != EventType.KEY_DOWN:
            return False

        key = event.key.lower()

        # Tab navigation
        if key == "tab":
            if event.is_shift:
                return self.navigate(FocusDirection.PREVIOUS)
            return self.navigate(FocusDirection.NEXT)

        # Arrow navigation
        if key == "arrowup":
            return self.navigate(FocusDirection.UP)
        if key == "arrowdown":
            return self.navigate(FocusDirection.DOWN)
        if key == "arrowleft":
            return self.navigate(FocusDirection.LEFT)
        if key == "arrowright":
            return self.navigate(FocusDirection.RIGHT)

        # Escape to exit trap
        if key == "escape" and self.is_trapped:
            self.pop_trap()
            return True

        return False

    # ============================================
    # Widget Notifications
    # ============================================

    def on_widget_removed(self, widget: "Widget") -> None:
        """
        Called when a widget is removed from the tree.

        Cleans up focus state and groups.

        Args:
            widget: Removed widget.
        """
        # Clear focus if this widget was focused
        if self._focused is widget:
            self.clear_focus()

        # Remove from history
        if widget in self._focus_history:
            self._focus_history.remove(widget)

        # Remove from groups
        for group in self._groups.values():
            group.remove(widget)

        self._default_group.remove(widget)

    def on_widget_visibility_changed(self, widget: "Widget") -> None:
        """
        Called when widget visibility changes.

        Args:
            widget: Widget with changed visibility.
        """
        if not widget.visible and self._focused is widget:
            # Try to move to next focusable
            if not self.navigate(FocusDirection.NEXT):
                self.clear_focus()

        # Rebuild groups
        self._rebuild_default_group()

    def on_widget_enabled_changed(self, widget: "Widget") -> None:
        """
        Called when widget enabled state changes.

        Args:
            widget: Widget with changed enabled state.
        """
        if not widget.enabled and self._focused is widget:
            if not self.navigate(FocusDirection.NEXT):
                self.clear_focus()


# Convenience function to get focus manager
def get_focus_manager() -> FocusManager:
    """Get the global focus manager instance."""
    return FocusManager.get_instance()


__all__ = [
    # Enums
    "FocusDirection",
    "FocusReason",
    # Data classes
    "FocusGroup",
    "FocusTrap",
    # Manager
    "FocusManager",
    # Convenience
    "get_focus_manager",
]
