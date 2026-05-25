"""
Screen reader support for UI accessibility.

Provides ARIA (Accessible Rich Internet Applications) support for the game engine UI:
- ARIA roles for semantic meaning
- ARIA properties for widget attributes
- ARIA states for dynamic state tracking
- Focus announcements for navigation
- Live regions for dynamic content updates

Reference (ARCHITECTURE_UI.md):
- Widget Announce: Name + role
- State Change: Value updates
- Focus Announce: Navigation
- Live Regions: Polite/Assertive announcements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


class AriaRole(Enum):
    """
    ARIA roles for semantic widget identification.

    Roles define the type of widget and its purpose for screen readers.
    Based on WAI-ARIA 1.2 specification.
    """
    # Widget roles
    BUTTON = auto()
    CHECKBOX = auto()
    DIALOG = auto()
    GRIDCELL = auto()
    LINK = auto()
    LISTBOX = auto()
    LISTITEM = auto()
    MENUITEM = auto()
    MENUITEMCHECKBOX = auto()
    MENUITEMRADIO = auto()
    OPTION = auto()
    PROGRESSBAR = auto()
    RADIO = auto()
    SCROLLBAR = auto()
    SEARCHBOX = auto()
    SLIDER = auto()
    SPINBUTTON = auto()
    SWITCH = auto()
    TAB = auto()
    TABPANEL = auto()
    TEXTBOX = auto()
    TREEITEM = auto()

    # Composite widget roles
    COMBOBOX = auto()
    GRID = auto()
    LISTBOX_COMPOSITE = auto()
    MENU = auto()
    MENUBAR = auto()
    RADIOGROUP = auto()
    TABLIST = auto()
    TREE = auto()
    TREEGRID = auto()

    # Document structure roles
    APPLICATION = auto()
    ARTICLE = auto()
    CELL = auto()
    COLUMNHEADER = auto()
    DEFINITION = auto()
    DIRECTORY = auto()
    DOCUMENT = auto()
    FEED = auto()
    FIGURE = auto()
    GROUP = auto()
    HEADING = auto()
    IMG = auto()
    LIST = auto()
    LISTBOX_STRUCTURE = auto()
    MATH = auto()
    NONE = auto()
    NOTE = auto()
    PRESENTATION = auto()
    ROW = auto()
    ROWGROUP = auto()
    ROWHEADER = auto()
    SEPARATOR = auto()
    TABLE = auto()
    TERM = auto()
    TOOLBAR = auto()
    TOOLTIP = auto()

    # Landmark roles
    BANNER = auto()
    COMPLEMENTARY = auto()
    CONTENTINFO = auto()
    FORM = auto()
    MAIN = auto()
    NAVIGATION = auto()
    REGION = auto()
    SEARCH = auto()

    # Live region roles
    ALERT = auto()
    LOG = auto()
    MARQUEE = auto()
    STATUS = auto()
    TIMER = auto()

    # Window roles
    ALERTDIALOG = auto()
    DIALOG_WINDOW = auto()


class LiveRegionPoliteness(Enum):
    """
    Politeness levels for live region announcements.

    Determines how urgently the screen reader should announce changes.
    """
    OFF = auto()      # No announcements
    POLITE = auto()   # Announce after current speech completes
    ASSERTIVE = auto()  # Interrupt current speech immediately


@dataclass
class AriaProperty:
    """
    ARIA property for widget attributes.

    Properties provide additional information about a widget that
    doesn't change based on user interaction.
    """
    label: Optional[str] = None  # aria-label
    labelledby: Optional[str] = None  # aria-labelledby (ID reference)
    describedby: Optional[str] = None  # aria-describedby (ID reference)
    description: Optional[str] = None  # aria-description
    roledescription: Optional[str] = None  # aria-roledescription

    # Value properties
    valuemin: Optional[float] = None  # aria-valuemin
    valuemax: Optional[float] = None  # aria-valuemax
    valuenow: Optional[float] = None  # aria-valuenow
    valuetext: Optional[str] = None  # aria-valuetext

    # Relationship properties
    controls: Optional[str] = None  # aria-controls (ID reference)
    owns: Optional[str] = None  # aria-owns (ID reference)
    flowto: Optional[str] = None  # aria-flowto (ID reference)

    # Structure properties
    level: Optional[int] = None  # aria-level (heading level)
    setsize: Optional[int] = None  # aria-setsize (items in set)
    posinset: Optional[int] = None  # aria-posinset (position in set)
    colcount: Optional[int] = None  # aria-colcount
    rowcount: Optional[int] = None  # aria-rowcount
    colindex: Optional[int] = None  # aria-colindex
    rowindex: Optional[int] = None  # aria-rowindex
    colspan: Optional[int] = None  # aria-colspan
    rowspan: Optional[int] = None  # aria-rowspan

    # Modal property
    modal: bool = False  # aria-modal

    # Other properties
    keyshortcuts: Optional[str] = None  # aria-keyshortcuts
    placeholder: Optional[str] = None  # aria-placeholder
    autocomplete: Optional[str] = None  # aria-autocomplete
    orientation: Optional[str] = None  # aria-orientation (horizontal/vertical)

    def get_accessible_name(self) -> Optional[str]:
        """Get the accessible name for screen readers."""
        return self.label or self.valuetext

    def get_accessible_description(self) -> Optional[str]:
        """Get the accessible description for screen readers."""
        return self.description


@dataclass
class AriaState:
    """
    ARIA state for dynamic widget state tracking.

    States represent the current condition of a widget that may
    change based on user interaction or application state.
    """
    # Selection states
    checked: Optional[bool] = None  # aria-checked (tri-state: True, False, None=mixed)
    selected: bool = False  # aria-selected
    pressed: Optional[bool] = None  # aria-pressed (tri-state)

    # Expansion states
    expanded: Optional[bool] = None  # aria-expanded

    # Focus states
    current: Optional[str] = None  # aria-current (page, step, location, date, time, true)

    # Validity states
    invalid: bool = False  # aria-invalid
    errormessage: Optional[str] = None  # aria-errormessage (ID reference)

    # Interactive states
    disabled: bool = False  # aria-disabled
    readonly: bool = False  # aria-readonly
    required: bool = False  # aria-required

    # Visibility states
    hidden: bool = False  # aria-hidden

    # Busy state
    busy: bool = False  # aria-busy

    # Grab states (for drag and drop)
    grabbed: Optional[bool] = None  # aria-grabbed (deprecated but supported)
    dropeffect: Optional[str] = None  # aria-dropeffect (deprecated but supported)

    # Sort state
    sort: Optional[str] = None  # aria-sort (ascending, descending, none, other)

    # Multi-selection
    multiselectable: bool = False  # aria-multiselectable

    # Active descendant (for composite widgets)
    activedescendant: Optional[str] = None  # aria-activedescendant (ID reference)

    def has_state_change(self, other: "AriaState") -> bool:
        """Check if state has changed from another state."""
        return (
            self.checked != other.checked or
            self.selected != other.selected or
            self.pressed != other.pressed or
            self.expanded != other.expanded or
            self.disabled != other.disabled or
            self.hidden != other.hidden or
            self.busy != other.busy
        )


@dataclass
class AriaLiveRegion:
    """
    Live region for dynamic content announcements.

    Live regions allow screen readers to announce changes to content
    without requiring user focus.
    """
    politeness: LiveRegionPoliteness = LiveRegionPoliteness.POLITE
    atomic: bool = False  # aria-atomic - announce entire region or just changes
    relevant: str = "additions text"  # aria-relevant - what changes to announce

    # Pending announcements queue
    _pending_announcements: list[str] = field(default_factory=list)

    def announce(self, message: str) -> None:
        """Queue an announcement for the live region."""
        if self.politeness != LiveRegionPoliteness.OFF:
            self._pending_announcements.append(message)

    def get_pending(self) -> list[str]:
        """Get and clear pending announcements."""
        announcements = self._pending_announcements.copy()
        self._pending_announcements.clear()
        return announcements

    def clear(self) -> None:
        """Clear all pending announcements."""
        self._pending_announcements.clear()


@dataclass
class FocusAnnouncement:
    """
    Focus announcement for navigation feedback.

    Provides information to announce when a widget receives focus.
    """
    widget_id: str
    role: AriaRole
    name: str
    state: Optional[AriaState] = None
    position: Optional[str] = None  # e.g., "1 of 5"
    hint: Optional[str] = None  # Additional instruction

    def build_announcement(self) -> str:
        """Build the full announcement string."""
        parts = []

        # Name
        if self.name:
            parts.append(self.name)

        # Role
        role_name = self.role.name.lower().replace("_", " ")
        parts.append(role_name)

        # State
        if self.state:
            if self.state.checked is True:
                parts.append("checked")
            elif self.state.checked is False:
                parts.append("not checked")
            elif self.state.checked is None and self.role in (AriaRole.CHECKBOX, AriaRole.SWITCH):
                parts.append("mixed")

            if self.state.expanded is True:
                parts.append("expanded")
            elif self.state.expanded is False:
                parts.append("collapsed")

            if self.state.selected:
                parts.append("selected")

            if self.state.disabled:
                parts.append("disabled")

        # Position
        if self.position:
            parts.append(self.position)

        # Hint
        if self.hint:
            parts.append(self.hint)

        return ", ".join(parts)


class AccessibilityManager:
    """
    Singleton manager for accessibility features.

    Coordinates screen reader support, focus management, and
    accessibility announcements across the UI system.
    """

    _instance: Optional["AccessibilityManager"] = None

    def __new__(cls) -> "AccessibilityManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True

        # Widget accessibility info storage
        self._widget_roles: dict[str, AriaRole] = {}
        self._widget_properties: dict[str, AriaProperty] = {}
        self._widget_states: dict[str, AriaState] = {}

        # Live regions
        self._live_regions: dict[str, AriaLiveRegion] = {}

        # Announcement callbacks
        self._announcement_handlers: list[Callable[[str, LiveRegionPoliteness], None]] = []

        # Focus tracking
        self._current_focus: Optional[str] = None
        self._focus_history: list[str] = []
        self._max_history: int = 50

        # Screen reader detection
        self._screen_reader_active: bool = False
        self._screen_reader_name: Optional[str] = None

        # Enabled state
        self._enabled: bool = True

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    @property
    def enabled(self) -> bool:
        """Check if accessibility features are enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable accessibility features."""
        self._enabled = value

    @property
    def screen_reader_active(self) -> bool:
        """Check if a screen reader is detected."""
        return self._screen_reader_active

    @property
    def screen_reader_name(self) -> Optional[str]:
        """Get the name of the detected screen reader."""
        return self._screen_reader_name

    @property
    def current_focus(self) -> Optional[str]:
        """Get the currently focused widget ID."""
        return self._current_focus

    def detect_screen_reader(self) -> bool:
        """
        Attempt to detect if a screen reader is running.

        Returns:
            True if a screen reader was detected
        """
        # Platform-specific detection would go here
        # This is a placeholder implementation
        # Windows: Check for NVDA, JAWS, Narrator
        # macOS: Check VoiceOver
        # Linux: Check Orca
        return self._screen_reader_active

    def set_screen_reader_active(self, active: bool, name: Optional[str] = None) -> None:
        """Set screen reader detection state (for testing or manual override)."""
        self._screen_reader_active = active
        self._screen_reader_name = name

    # Role management
    def set_role(self, widget_id: str, role: AriaRole) -> None:
        """Set the ARIA role for a widget."""
        self._widget_roles[widget_id] = role

    def get_role(self, widget_id: str) -> Optional[AriaRole]:
        """Get the ARIA role for a widget."""
        return self._widget_roles.get(widget_id)

    def remove_role(self, widget_id: str) -> None:
        """Remove the ARIA role for a widget."""
        self._widget_roles.pop(widget_id, None)

    # Property management
    def set_property(self, widget_id: str, property: AriaProperty) -> None:
        """Set ARIA properties for a widget."""
        self._widget_properties[widget_id] = property

    def get_property(self, widget_id: str) -> Optional[AriaProperty]:
        """Get ARIA properties for a widget."""
        return self._widget_properties.get(widget_id)

    def update_property(self, widget_id: str, **kwargs: Any) -> None:
        """Update specific ARIA properties for a widget."""
        prop = self._widget_properties.get(widget_id)
        if prop is None:
            prop = AriaProperty()
            self._widget_properties[widget_id] = prop

        for key, value in kwargs.items():
            if hasattr(prop, key):
                setattr(prop, key, value)

    def remove_property(self, widget_id: str) -> None:
        """Remove ARIA properties for a widget."""
        self._widget_properties.pop(widget_id, None)

    # State management
    def set_state(self, widget_id: str, state: AriaState) -> None:
        """Set ARIA state for a widget."""
        old_state = self._widget_states.get(widget_id)
        self._widget_states[widget_id] = state

        # Announce state changes
        if old_state and state.has_state_change(old_state) and self._enabled:
            self._announce_state_change(widget_id, old_state, state)

    def get_state(self, widget_id: str) -> Optional[AriaState]:
        """Get ARIA state for a widget."""
        return self._widget_states.get(widget_id)

    def update_state(self, widget_id: str, **kwargs: Any) -> None:
        """Update specific ARIA state for a widget."""
        state = self._widget_states.get(widget_id)
        if state is None:
            state = AriaState()
            self._widget_states[widget_id] = state

        old_state = AriaState(**{k: getattr(state, k) for k in state.__dataclass_fields__})

        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)

        if state.has_state_change(old_state) and self._enabled:
            self._announce_state_change(widget_id, old_state, state)

    def remove_state(self, widget_id: str) -> None:
        """Remove ARIA state for a widget."""
        self._widget_states.pop(widget_id, None)

    def _announce_state_change(self, widget_id: str, old: AriaState, new: AriaState) -> None:
        """Announce a state change to screen readers."""
        changes = []

        if old.checked != new.checked:
            if new.checked is True:
                changes.append("checked")
            elif new.checked is False:
                changes.append("not checked")
            else:
                changes.append("mixed")

        if old.expanded != new.expanded:
            changes.append("expanded" if new.expanded else "collapsed")

        if old.selected != new.selected:
            changes.append("selected" if new.selected else "not selected")

        if old.disabled != new.disabled:
            changes.append("disabled" if new.disabled else "enabled")

        if changes:
            announcement = ", ".join(changes)
            self._broadcast_announcement(announcement, LiveRegionPoliteness.POLITE)

    # Live region management
    def create_live_region(
        self,
        region_id: str,
        politeness: LiveRegionPoliteness = LiveRegionPoliteness.POLITE,
        atomic: bool = False,
        relevant: str = "additions text",
    ) -> AriaLiveRegion:
        """Create a live region for dynamic announcements."""
        region = AriaLiveRegion(
            politeness=politeness,
            atomic=atomic,
            relevant=relevant,
        )
        self._live_regions[region_id] = region
        return region

    def get_live_region(self, region_id: str) -> Optional[AriaLiveRegion]:
        """Get a live region by ID."""
        return self._live_regions.get(region_id)

    def remove_live_region(self, region_id: str) -> None:
        """Remove a live region."""
        self._live_regions.pop(region_id, None)

    def announce_live(self, region_id: str, message: str) -> None:
        """Announce a message through a live region."""
        region = self._live_regions.get(region_id)
        if region and self._enabled:
            region.announce(message)
            self._broadcast_announcement(message, region.politeness)

    def announce_polite(self, message: str) -> None:
        """Make a polite announcement (after current speech)."""
        if self._enabled:
            self._broadcast_announcement(message, LiveRegionPoliteness.POLITE)

    def announce_assertive(self, message: str) -> None:
        """Make an assertive announcement (interrupt current speech)."""
        if self._enabled:
            self._broadcast_announcement(message, LiveRegionPoliteness.ASSERTIVE)

    # Focus announcements
    def announce_focus(self, widget_id: str) -> None:
        """Announce when a widget receives focus."""
        if not self._enabled:
            return

        # Track focus
        self._current_focus = widget_id
        self._focus_history.append(widget_id)
        if len(self._focus_history) > self._max_history:
            self._focus_history.pop(0)

        # Build announcement
        role = self.get_role(widget_id)
        prop = self.get_property(widget_id)
        state = self.get_state(widget_id)

        if role is None:
            return

        name = prop.get_accessible_name() if prop else None

        announcement = FocusAnnouncement(
            widget_id=widget_id,
            role=role,
            name=name or "",
            state=state,
        )

        self._broadcast_announcement(
            announcement.build_announcement(),
            LiveRegionPoliteness.POLITE,
        )

    def clear_focus(self) -> None:
        """Clear the current focus."""
        self._current_focus = None

    def get_focus_history(self) -> list[str]:
        """Get the focus history."""
        return self._focus_history.copy()

    # Announcement handlers
    def add_announcement_handler(
        self,
        handler: Callable[[str, LiveRegionPoliteness], None],
    ) -> None:
        """Add a handler for accessibility announcements."""
        self._announcement_handlers.append(handler)

    def remove_announcement_handler(
        self,
        handler: Callable[[str, LiveRegionPoliteness], None],
    ) -> None:
        """Remove an announcement handler."""
        if handler in self._announcement_handlers:
            self._announcement_handlers.remove(handler)

    def _broadcast_announcement(self, message: str, politeness: LiveRegionPoliteness) -> None:
        """Broadcast an announcement to all handlers."""
        for handler in self._announcement_handlers:
            handler(message, politeness)

    # Cleanup
    def remove_widget(self, widget_id: str) -> None:
        """Remove all accessibility data for a widget."""
        self.remove_role(widget_id)
        self.remove_property(widget_id)
        self.remove_state(widget_id)

        if self._current_focus == widget_id:
            self._current_focus = None

    def clear(self) -> None:
        """Clear all accessibility data."""
        self._widget_roles.clear()
        self._widget_properties.clear()
        self._widget_states.clear()
        self._live_regions.clear()
        self._current_focus = None
        self._focus_history.clear()
