"""
Keyboard navigation system for UI accessibility.

Provides comprehensive keyboard navigation support:
- Tab order management for sequential focus navigation
- Arrow key navigation for lists, grids, and menus
- Keyboard shortcuts with modifier key support
- Skip links for quick navigation to landmarks

Reference (ARCHITECTURE_UI.md):
- Tab Order: Sequential navigation
- Arrow Keys: Directional navigation
- Explicit: Widget links
- Auto: Spatial analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


class FocusDirection(Enum):
    """Direction for focus movement."""
    NEXT = auto()      # Tab forward
    PREVIOUS = auto()  # Tab backward / Shift+Tab
    UP = auto()        # Arrow up
    DOWN = auto()      # Arrow down
    LEFT = auto()      # Arrow left
    RIGHT = auto()     # Arrow right
    FIRST = auto()     # Home
    LAST = auto()      # End
    PAGE_UP = auto()   # Page up
    PAGE_DOWN = auto()  # Page down


class NavigationMode(Enum):
    """Navigation mode for keyboard focus."""
    SEQUENTIAL = auto()  # Tab order
    DIRECTIONAL = auto()  # Arrow keys
    GRID = auto()        # 2D grid navigation
    ROVING = auto()      # Roving tabindex (single tab stop, arrows within)


@dataclass
class TabStop:
    """
    A focusable element in the tab order.

    Represents a widget that can receive keyboard focus.
    """
    widget_id: str
    tab_index: int = 0  # 0 = natural order, positive = explicit order, -1 = skip
    focusable: bool = True
    group: Optional[str] = None  # Navigation group ID

    # Spatial position for directional navigation
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    # Callbacks
    on_focus: Optional[Callable[[], None]] = None
    on_blur: Optional[Callable[[], None]] = None

    def is_tabbable(self) -> bool:
        """Check if this stop is in the tab order."""
        return self.focusable and self.tab_index >= 0

    def center_x(self) -> float:
        """Get center X coordinate."""
        return self.x + self.width / 2

    def center_y(self) -> float:
        """Get center Y coordinate."""
        return self.y + self.height / 2

    def contains_point(self, px: float, py: float) -> bool:
        """Check if a point is within this tab stop's bounds."""
        return (
            self.x <= px <= self.x + self.width and
            self.y <= py <= self.y + self.height
        )


@dataclass
class NavigationGroup:
    """
    A group of focusable elements with specific navigation behavior.

    Groups allow different navigation modes within the same UI.
    For example, a list uses arrow navigation while the overall
    page uses tab navigation.
    """
    group_id: str
    mode: NavigationMode = NavigationMode.SEQUENTIAL
    wrap: bool = True  # Wrap around at ends

    # Grid-specific
    columns: int = 1
    rows: int = 0  # 0 = auto-calculate

    # Members
    members: list[str] = field(default_factory=list)  # Widget IDs

    # Focus tracking within group
    active_index: int = 0

    # Focus trap - keep focus within group
    trap_focus: bool = False

    # Skip when tabbing (for roving tabindex)
    single_tab_stop: bool = False

    def add_member(self, widget_id: str) -> None:
        """Add a widget to this group."""
        if widget_id not in self.members:
            self.members.append(widget_id)

    def remove_member(self, widget_id: str) -> None:
        """Remove a widget from this group."""
        if widget_id in self.members:
            self.members.remove(widget_id)

    def get_member_count(self) -> int:
        """Get the number of members in this group."""
        return len(self.members)

    def get_active_member(self) -> Optional[str]:
        """Get the currently active member."""
        if 0 <= self.active_index < len(self.members):
            return self.members[self.active_index]
        return None

    def set_active_by_id(self, widget_id: str) -> bool:
        """Set the active member by widget ID."""
        try:
            self.active_index = self.members.index(widget_id)
            return True
        except ValueError:
            return False


@dataclass
class SkipLink:
    """
    A skip link for quick navigation to landmarks.

    Skip links allow users to jump directly to main content areas,
    navigation, or other landmarks without tabbing through everything.
    """
    link_id: str
    label: str  # Visible/announced label
    target_widget_id: str
    shortcut: Optional[str] = None  # e.g., "Alt+1"
    visible_on_focus: bool = True  # Only show when focused

    # Position (usually at top of page)
    x: float = 0.0
    y: float = 0.0


@dataclass
class KeyboardShortcut:
    """
    A keyboard shortcut binding.

    Defines a key combination that triggers an action.
    """
    key: str  # e.g., "A", "F1", "Enter"
    action: Callable[[], None]
    description: str = ""

    # Modifiers
    ctrl: bool = False
    alt: bool = False
    shift: bool = False
    meta: bool = False  # Windows/Command key

    # Scope
    global_shortcut: bool = False  # Works regardless of focus
    group_id: Optional[str] = None  # Only works in this group
    widget_id: Optional[str] = None  # Only works when this widget focused

    # State
    enabled: bool = True

    def matches(
        self,
        key: str,
        ctrl: bool = False,
        alt: bool = False,
        shift: bool = False,
        meta: bool = False,
    ) -> bool:
        """Check if the given key combination matches this shortcut."""
        return (
            self.enabled and
            self.key.lower() == key.lower() and
            self.ctrl == ctrl and
            self.alt == alt and
            self.shift == shift and
            self.meta == meta
        )

    def get_shortcut_string(self) -> str:
        """Get a human-readable shortcut string."""
        parts = []
        if self.ctrl:
            parts.append("Ctrl")
        if self.alt:
            parts.append("Alt")
        if self.shift:
            parts.append("Shift")
        if self.meta:
            parts.append("Meta")
        parts.append(self.key)
        return "+".join(parts)


class TabOrder:
    """
    Manages the tab order for a set of widgets.

    Handles both explicit tab indices and natural document order.
    """

    __slots__ = ("_stops", "_sorted_cache", "_dirty")

    def __init__(self) -> None:
        self._stops: dict[str, TabStop] = {}
        self._sorted_cache: list[TabStop] = []
        self._dirty: bool = True

    def add(self, stop: TabStop) -> None:
        """Add a tab stop."""
        self._stops[stop.widget_id] = stop
        self._dirty = True

    def remove(self, widget_id: str) -> None:
        """Remove a tab stop."""
        if widget_id in self._stops:
            del self._stops[widget_id]
            self._dirty = True

    def get(self, widget_id: str) -> Optional[TabStop]:
        """Get a tab stop by widget ID."""
        return self._stops.get(widget_id)

    def update(self, widget_id: str, **kwargs: Any) -> None:
        """Update a tab stop's properties."""
        stop = self._stops.get(widget_id)
        if stop:
            for key, value in kwargs.items():
                if hasattr(stop, key):
                    setattr(stop, key, value)
            self._dirty = True

    def _rebuild_cache(self) -> None:
        """Rebuild the sorted tab order cache."""
        tabbable = [s for s in self._stops.values() if s.is_tabbable()]

        # Sort by tab_index (0 = natural order at end), then by position
        self._sorted_cache = sorted(
            tabbable,
            key=lambda s: (
                0 if s.tab_index > 0 else 1,  # Explicit indices first
                s.tab_index if s.tab_index > 0 else 0,
                s.y,
                s.x,
            ),
        )
        self._dirty = False

    def get_sorted(self) -> list[TabStop]:
        """Get the sorted tab order."""
        if self._dirty:
            self._rebuild_cache()
        return self._sorted_cache.copy()

    def get_next(self, current_id: Optional[str], wrap: bool = True) -> Optional[TabStop]:
        """Get the next tab stop after the current one."""
        if self._dirty:
            self._rebuild_cache()

        if not self._sorted_cache:
            return None

        if current_id is None:
            return self._sorted_cache[0]

        # Find current index
        current_index = -1
        for i, stop in enumerate(self._sorted_cache):
            if stop.widget_id == current_id:
                current_index = i
                break

        if current_index == -1:
            return self._sorted_cache[0]

        next_index = current_index + 1
        if next_index >= len(self._sorted_cache):
            return self._sorted_cache[0] if wrap else None

        return self._sorted_cache[next_index]

    def get_previous(self, current_id: Optional[str], wrap: bool = True) -> Optional[TabStop]:
        """Get the previous tab stop before the current one."""
        if self._dirty:
            self._rebuild_cache()

        if not self._sorted_cache:
            return None

        if current_id is None:
            return self._sorted_cache[-1]

        # Find current index
        current_index = -1
        for i, stop in enumerate(self._sorted_cache):
            if stop.widget_id == current_id:
                current_index = i
                break

        if current_index == -1:
            return self._sorted_cache[-1]

        prev_index = current_index - 1
        if prev_index < 0:
            return self._sorted_cache[-1] if wrap else None

        return self._sorted_cache[prev_index]

    def get_first(self) -> Optional[TabStop]:
        """Get the first tab stop."""
        if self._dirty:
            self._rebuild_cache()
        return self._sorted_cache[0] if self._sorted_cache else None

    def get_last(self) -> Optional[TabStop]:
        """Get the last tab stop."""
        if self._dirty:
            self._rebuild_cache()
        return self._sorted_cache[-1] if self._sorted_cache else None

    def clear(self) -> None:
        """Clear all tab stops."""
        self._stops.clear()
        self._sorted_cache.clear()
        self._dirty = True


class KeyboardNavigator:
    """
    Main keyboard navigation manager.

    Coordinates tab order, navigation groups, skip links,
    and keyboard shortcuts.
    """

    __slots__ = (
        "_tab_order",
        "_groups",
        "_skip_links",
        "_shortcuts",
        "_current_focus",
        "_current_group",
        "_focus_callbacks",
        "_enabled",
    )

    def __init__(self) -> None:
        self._tab_order = TabOrder()
        self._groups: dict[str, NavigationGroup] = {}
        self._skip_links: list[SkipLink] = []
        self._shortcuts: list[KeyboardShortcut] = []
        self._current_focus: Optional[str] = None
        self._current_group: Optional[str] = None
        self._focus_callbacks: list[Callable[[Optional[str], Optional[str]], None]] = []
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        """Check if keyboard navigation is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable keyboard navigation."""
        self._enabled = value

    @property
    def current_focus(self) -> Optional[str]:
        """Get the currently focused widget ID."""
        return self._current_focus

    @property
    def current_group(self) -> Optional[str]:
        """Get the current navigation group ID."""
        return self._current_group

    @property
    def tab_order(self) -> TabOrder:
        """Get the tab order manager."""
        return self._tab_order

    # Tab stop management
    def register_focusable(
        self,
        widget_id: str,
        tab_index: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 0.0,
        height: float = 0.0,
        group: Optional[str] = None,
        on_focus: Optional[Callable[[], None]] = None,
        on_blur: Optional[Callable[[], None]] = None,
    ) -> TabStop:
        """Register a widget as focusable."""
        stop = TabStop(
            widget_id=widget_id,
            tab_index=tab_index,
            x=x,
            y=y,
            width=width,
            height=height,
            group=group,
            on_focus=on_focus,
            on_blur=on_blur,
        )
        self._tab_order.add(stop)

        if group:
            self._add_to_group(widget_id, group)

        return stop

    def unregister_focusable(self, widget_id: str) -> None:
        """Unregister a focusable widget."""
        stop = self._tab_order.get(widget_id)
        if stop and stop.group:
            self._remove_from_group(widget_id, stop.group)

        self._tab_order.remove(widget_id)

        if self._current_focus == widget_id:
            self._current_focus = None

    def update_focusable(self, widget_id: str, **kwargs: Any) -> None:
        """Update a focusable widget's properties."""
        self._tab_order.update(widget_id, **kwargs)

    # Group management
    def create_group(
        self,
        group_id: str,
        mode: NavigationMode = NavigationMode.SEQUENTIAL,
        wrap: bool = True,
        columns: int = 1,
        trap_focus: bool = False,
        single_tab_stop: bool = False,
    ) -> NavigationGroup:
        """Create a navigation group."""
        group = NavigationGroup(
            group_id=group_id,
            mode=mode,
            wrap=wrap,
            columns=columns,
            trap_focus=trap_focus,
            single_tab_stop=single_tab_stop,
        )
        self._groups[group_id] = group
        return group

    def get_group(self, group_id: str) -> Optional[NavigationGroup]:
        """Get a navigation group by ID."""
        return self._groups.get(group_id)

    def remove_group(self, group_id: str) -> None:
        """Remove a navigation group."""
        self._groups.pop(group_id, None)

    def _add_to_group(self, widget_id: str, group_id: str) -> None:
        """Add a widget to a group."""
        group = self._groups.get(group_id)
        if group:
            group.add_member(widget_id)

    def _remove_from_group(self, widget_id: str, group_id: str) -> None:
        """Remove a widget from a group."""
        group = self._groups.get(group_id)
        if group:
            group.remove_member(widget_id)

    # Skip links
    def add_skip_link(
        self,
        link_id: str,
        label: str,
        target_widget_id: str,
        shortcut: Optional[str] = None,
    ) -> SkipLink:
        """Add a skip link."""
        link = SkipLink(
            link_id=link_id,
            label=label,
            target_widget_id=target_widget_id,
            shortcut=shortcut,
        )
        self._skip_links.append(link)
        return link

    def get_skip_links(self) -> list[SkipLink]:
        """Get all skip links."""
        return self._skip_links.copy()

    def remove_skip_link(self, link_id: str) -> None:
        """Remove a skip link."""
        self._skip_links = [l for l in self._skip_links if l.link_id != link_id]

    def activate_skip_link(self, link_id: str) -> bool:
        """Activate a skip link and move focus to its target."""
        for link in self._skip_links:
            if link.link_id == link_id:
                return self.set_focus(link.target_widget_id)
        return False

    # Shortcuts
    def register_shortcut(
        self,
        key: str,
        action: Callable[[], None],
        description: str = "",
        ctrl: bool = False,
        alt: bool = False,
        shift: bool = False,
        meta: bool = False,
        global_shortcut: bool = False,
        group_id: Optional[str] = None,
        widget_id: Optional[str] = None,
    ) -> KeyboardShortcut:
        """Register a keyboard shortcut."""
        shortcut = KeyboardShortcut(
            key=key,
            action=action,
            description=description,
            ctrl=ctrl,
            alt=alt,
            shift=shift,
            meta=meta,
            global_shortcut=global_shortcut,
            group_id=group_id,
            widget_id=widget_id,
        )
        self._shortcuts.append(shortcut)
        return shortcut

    def unregister_shortcut(self, shortcut: KeyboardShortcut) -> None:
        """Unregister a keyboard shortcut."""
        if shortcut in self._shortcuts:
            self._shortcuts.remove(shortcut)

    def get_shortcuts(self) -> list[KeyboardShortcut]:
        """Get all registered shortcuts."""
        return self._shortcuts.copy()

    def handle_key(
        self,
        key: str,
        ctrl: bool = False,
        alt: bool = False,
        shift: bool = False,
        meta: bool = False,
    ) -> bool:
        """
        Handle a key press.

        Returns True if the key was handled by a shortcut.
        """
        if not self._enabled:
            return False

        # Check shortcuts
        for shortcut in self._shortcuts:
            if not shortcut.matches(key, ctrl, alt, shift, meta):
                continue

            # Check scope
            if shortcut.global_shortcut:
                shortcut.action()
                return True

            if shortcut.widget_id and shortcut.widget_id == self._current_focus:
                shortcut.action()
                return True

            if shortcut.group_id and shortcut.group_id == self._current_group:
                shortcut.action()
                return True

            if not shortcut.widget_id and not shortcut.group_id:
                shortcut.action()
                return True

        return False

    # Focus management
    def set_focus(self, widget_id: str) -> bool:
        """Set focus to a specific widget."""
        if not self._enabled:
            return False

        stop = self._tab_order.get(widget_id)
        if not stop or not stop.focusable:
            return False

        old_focus = self._current_focus

        # Blur old focus
        if old_focus:
            old_stop = self._tab_order.get(old_focus)
            if old_stop and old_stop.on_blur:
                old_stop.on_blur()

        # Set new focus
        self._current_focus = widget_id
        self._current_group = stop.group

        # Focus new widget
        if stop.on_focus:
            stop.on_focus()

        # Update group active index
        if stop.group:
            group = self._groups.get(stop.group)
            if group:
                group.set_active_by_id(widget_id)

        # Notify callbacks
        for callback in self._focus_callbacks:
            callback(old_focus, widget_id)

        return True

    def clear_focus(self) -> None:
        """Clear the current focus."""
        if self._current_focus:
            stop = self._tab_order.get(self._current_focus)
            if stop and stop.on_blur:
                stop.on_blur()

        old_focus = self._current_focus
        self._current_focus = None
        self._current_group = None

        for callback in self._focus_callbacks:
            callback(old_focus, None)

    def move_focus(self, direction: FocusDirection) -> bool:
        """
        Move focus in the specified direction.

        Returns True if focus was moved.
        """
        if not self._enabled:
            return False

        # Get current group for directional navigation
        group = self._groups.get(self._current_group) if self._current_group else None

        # Handle directional navigation within groups
        if group and direction in (
            FocusDirection.UP,
            FocusDirection.DOWN,
            FocusDirection.LEFT,
            FocusDirection.RIGHT,
        ):
            return self._move_within_group(group, direction)

        # Handle tab navigation
        if direction == FocusDirection.NEXT:
            next_stop = self._tab_order.get_next(
                self._current_focus,
                wrap=group.wrap if group else True,
            )
            if next_stop:
                return self.set_focus(next_stop.widget_id)

        elif direction == FocusDirection.PREVIOUS:
            prev_stop = self._tab_order.get_previous(
                self._current_focus,
                wrap=group.wrap if group else True,
            )
            if prev_stop:
                return self.set_focus(prev_stop.widget_id)

        elif direction == FocusDirection.FIRST:
            first_stop = self._tab_order.get_first()
            if first_stop:
                return self.set_focus(first_stop.widget_id)

        elif direction == FocusDirection.LAST:
            last_stop = self._tab_order.get_last()
            if last_stop:
                return self.set_focus(last_stop.widget_id)

        return False

    def _move_within_group(self, group: NavigationGroup, direction: FocusDirection) -> bool:
        """Handle directional navigation within a group."""
        if not group.members:
            return False

        current_index = group.active_index
        new_index = current_index

        if group.mode == NavigationMode.GRID:
            # Grid navigation
            cols = group.columns
            rows = (len(group.members) + cols - 1) // cols

            row = current_index // cols
            col = current_index % cols

            if direction == FocusDirection.UP:
                row = (row - 1) % rows if group.wrap else max(0, row - 1)
            elif direction == FocusDirection.DOWN:
                row = (row + 1) % rows if group.wrap else min(rows - 1, row + 1)
            elif direction == FocusDirection.LEFT:
                col = (col - 1) % cols if group.wrap else max(0, col - 1)
            elif direction == FocusDirection.RIGHT:
                col = (col + 1) % cols if group.wrap else min(cols - 1, col + 1)

            new_index = row * cols + col
            if new_index >= len(group.members):
                new_index = len(group.members) - 1

        else:
            # Sequential/directional navigation
            if direction in (FocusDirection.UP, FocusDirection.LEFT):
                if group.wrap:
                    new_index = (current_index - 1) % len(group.members)
                else:
                    new_index = max(0, current_index - 1)
            elif direction in (FocusDirection.DOWN, FocusDirection.RIGHT):
                if group.wrap:
                    new_index = (current_index + 1) % len(group.members)
                else:
                    new_index = min(len(group.members) - 1, current_index + 1)

        if new_index != current_index and 0 <= new_index < len(group.members):
            widget_id = group.members[new_index]
            return self.set_focus(widget_id)

        return False

    # Focus callbacks
    def add_focus_callback(
        self,
        callback: Callable[[Optional[str], Optional[str]], None],
    ) -> None:
        """Add a callback for focus changes (old_id, new_id)."""
        self._focus_callbacks.append(callback)

    def remove_focus_callback(
        self,
        callback: Callable[[Optional[str], Optional[str]], None],
    ) -> None:
        """Remove a focus change callback."""
        if callback in self._focus_callbacks:
            self._focus_callbacks.remove(callback)

    # Directional navigation helpers
    def find_nearest(
        self,
        from_id: str,
        direction: FocusDirection,
        candidates: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Find the nearest focusable widget in a direction.

        Uses spatial analysis to find the best candidate.
        """
        from_stop = self._tab_order.get(from_id)
        if not from_stop:
            return None

        # Get candidate stops
        if candidates:
            stops = [self._tab_order.get(c) for c in candidates]
            stops = [s for s in stops if s and s.focusable]
        else:
            stops = [s for s in self._tab_order.get_sorted() if s.widget_id != from_id]

        if not stops:
            return None

        # Filter by direction
        fx, fy = from_stop.center_x(), from_stop.center_y()

        valid_candidates = []
        for stop in stops:
            sx, sy = stop.center_x(), stop.center_y()

            if direction == FocusDirection.UP and sy < fy:
                valid_candidates.append(stop)
            elif direction == FocusDirection.DOWN and sy > fy:
                valid_candidates.append(stop)
            elif direction == FocusDirection.LEFT and sx < fx:
                valid_candidates.append(stop)
            elif direction == FocusDirection.RIGHT and sx > fx:
                valid_candidates.append(stop)

        if not valid_candidates:
            return None

        # Find nearest by distance
        def distance(stop: TabStop) -> float:
            dx = stop.center_x() - fx
            dy = stop.center_y() - fy
            return dx * dx + dy * dy

        nearest = min(valid_candidates, key=distance)
        return nearest.widget_id

    # Utility
    def clear(self) -> None:
        """Clear all navigation data."""
        self._tab_order.clear()
        self._groups.clear()
        self._skip_links.clear()
        self._shortcuts.clear()
        self._current_focus = None
        self._current_group = None
