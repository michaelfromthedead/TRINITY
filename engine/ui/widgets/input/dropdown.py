"""
Dropdown Widget Implementation.

A dropdown/select widget with support for:
- Options list (value, label pairs)
- Selected item tracking
- Open/close state
- Keyboard navigation (arrows, enter, escape)
- Search/filter functionality
- Multi-select option
- Placeholder when nothing selected
- Custom item rendering

Follows the Trinity Pattern with TrackedDescriptor for state changes
and ObservableDescriptor for event subscriptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from time import time
from typing import Any, Callable, Generic, Optional, TypeVar


T = TypeVar('T')


class DropdownState(Enum):
    """Visual interaction states for the dropdown."""
    NORMAL = auto()
    CLOSED = NORMAL
    HOVERED = auto()
    FOCUSED = auto()
    OPEN = auto()
    DISABLED = auto()


@dataclass(slots=True)
class DropdownOption(Generic[T]):
    """Represents an option in the dropdown.

    Attributes:
        value: The actual value of the option
        label: Display text for the option
        disabled: Whether this option is disabled
        icon: Optional icon identifier
        group: Optional group name for grouping options
    """
    value: T
    label: str
    disabled: bool = False
    icon: Optional[str] = None
    group: Optional[str] = None

    def matches_filter(self, filter_text: str) -> bool:
        """Check if this option matches a filter string.

        Args:
            filter_text: Text to filter by

        Returns:
            True if the label contains the filter text (case-insensitive)
        """
        return filter_text.lower() in self.label.lower()


@dataclass(slots=True)
class DropdownStyle:
    """Style configuration for dropdown appearance.

    Attributes:
        background_color: Background color of the dropdown button
        hover_color: Background color when hovered
        open_color: Background color when open
        disabled_color: Background color when disabled
        text_color: Text color
        placeholder_color: Placeholder text color
        disabled_text_color: Text color when disabled
        border_color: Border color
        focused_border_color: Border color when focused
        border_width: Border thickness
        corner_radius: Corner rounding
        padding_horizontal: Horizontal padding
        padding_vertical: Vertical padding
        font_size: Text font size
        dropdown_background: Dropdown menu background color
        dropdown_border_color: Dropdown menu border color
        item_hover_color: Background color of hovered item
        item_selected_color: Background color of selected item
        item_height: Height of each dropdown item
        max_visible_items: Maximum items visible before scrolling
        arrow_color: Color of the dropdown arrow
        arrow_size: Size of the dropdown arrow
        search_placeholder: Placeholder text for search input
        checkbox_color: Color of checkboxes in multi-select mode
    """
    background_color: str = "#FFFFFF"
    hover_color: str = "#F5F5F5"
    open_color: str = "#E8E8E8"
    disabled_color: str = "#F0F0F0"
    text_color: str = "#333333"
    placeholder_color: str = "#AAAAAA"
    disabled_text_color: str = "#999999"
    border_color: str = "#CCCCCC"
    focused_border_color: str = "#4A90D9"
    border_width: float = 1.0
    corner_radius: float = 4.0
    padding_horizontal: float = 12.0
    padding_vertical: float = 8.0
    font_size: float = 14.0
    dropdown_background: str = "#FFFFFF"
    dropdown_border_color: str = "#CCCCCC"
    item_hover_color: str = "#E8F4FF"
    item_selected_color: str = "#D0E8FF"
    item_height: float = 36.0
    max_visible_items: int = 8
    arrow_color: str = "#666666"
    arrow_size: float = 8.0
    search_placeholder: str = "Search..."
    checkbox_color: str = "#4A90D9"


@dataclass(slots=True)
class SelectionChangeEvent(Generic[T]):
    """Event emitted when selection changes.

    Attributes:
        dropdown: Reference to the dropdown widget
        timestamp: Time of the change
        selected_values: List of selected values
        previous_values: List of previously selected values
        is_user_action: True if triggered by user interaction
    """
    dropdown: "Dropdown[T]"
    timestamp: float
    selected_values: list[T]
    previous_values: list[T]
    is_user_action: bool = True

    @property
    def new_value(self) -> Optional[T]:
        """Get the new value (first if multi-select)."""
        return self.selected_values[0] if self.selected_values else None

    @property
    def previous_value(self) -> Optional[T]:
        """Get the previous value (first if multi-select)."""
        return self.previous_values[0] if self.previous_values else None


@dataclass(slots=True)
class OpenStateChangeEvent:
    """Event emitted when dropdown open state changes.

    Attributes:
        dropdown: Reference to the dropdown widget
        timestamp: Time of the change
        is_open: Whether dropdown is now open
    """
    dropdown: "Dropdown"
    timestamp: float
    is_open: bool


class Dropdown(Generic[T]):
    """Interactive dropdown/select widget.

    A dropdown allows users to select one or more options from a list.
    Supports filtering, keyboard navigation, and multi-select.

    Type Parameters:
        T: Type of the option values

    Attributes:
        options: List of available options
        selected_values: Currently selected value(s)
        placeholder: Text shown when nothing is selected
        is_open: Whether the dropdown is currently open
        multi_select: Whether multiple items can be selected
        searchable: Whether search/filter is enabled
        enabled: Whether the dropdown is interactive
        visible: Whether the dropdown is rendered

    Events:
        on_selection_change: Fired when selection changes
        on_open: Fired when dropdown opens/closes

    Example:
        options = [
            DropdownOption("us", "United States"),
            DropdownOption("uk", "United Kingdom"),
            DropdownOption("ca", "Canada"),
        ]
        dropdown = Dropdown(options=options, placeholder="Select country...")
        dropdown.on_selection_change(lambda e: print(f"Selected: {e.selected_values}"))
    """

    __slots__ = (
        '_id', '_options', '_selected_indices', '_placeholder',
        '_is_open', '_multi_select', '_searchable',
        '_enabled', '_visible', '_focusable',
        '_visual_state', '_style',
        '_x', '_y', '_width', '_height',
        '_filter_text', '_highlighted_index', '_scroll_offset',
        '_on_selection_change_handlers', '_on_open_handlers', '_on_close_handlers',
        '_custom_renderer',
        '_is_hovered', '_is_focused',
        '_dirty', '_cached_mesh'
    )

    # Class-level ID counter
    _next_id: int = 0

    def __init__(
        self,
        options: Optional[list[DropdownOption[T]]] = None,
        selected_value: Optional[T] = None,
        selected_values: Optional[list[T]] = None,
        placeholder: str = "Select...",
        multi_select: bool = False,
        searchable: bool = False,
        enabled: bool = True,
        visible: bool = True,
        style: Optional[DropdownStyle] = None,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 200.0,
        height: float = 40.0,
    ):
        """Initialize a dropdown widget.

        Args:
            options: List of options
            selected_value: Initially selected value (single-select)
            selected_values: Initially selected values (multi-select)
            placeholder: Placeholder text
            multi_select: Enable multi-select mode
            searchable: Enable search/filter
            enabled: Initial enabled state
            visible: Initial visibility
            style: Style configuration
            x: X position
            y: Y position
            width: Widget width
            height: Widget height (button only, not including dropdown)
        """
        self._id = Dropdown._next_id
        Dropdown._next_id += 1

        self._options: list[DropdownOption[T]] = self._normalize_options(options) if options else []
        self._placeholder = placeholder
        self._is_open = False
        self._multi_select = multi_select
        self._searchable = searchable
        self._enabled = enabled
        self._visible = visible
        self._focusable = True
        self._visual_state = DropdownState.NORMAL if enabled else DropdownState.DISABLED
        self._style = style or DropdownStyle()

        self._x = x
        self._y = y
        self._width = width
        self._height = height

        self._filter_text = ""
        self._highlighted_index = -1
        self._scroll_offset = 0

        # Initialize selection
        self._selected_indices: list[int] = []
        if selected_values and multi_select:
            for val in selected_values:
                idx = self._find_option_index(val)
                if idx >= 0 and idx not in self._selected_indices:
                    self._selected_indices.append(idx)
        elif selected_value is not None:
            idx = self._find_option_index(selected_value)
            if idx >= 0:
                self._selected_indices = [idx]

        self._on_selection_change_handlers: list[Callable[[SelectionChangeEvent[T]], None]] = []
        self._on_open_handlers: list[Callable[[OpenStateChangeEvent], None]] = []
        self._on_close_handlers: list[Callable[[OpenStateChangeEvent], None]] = []
        self._custom_renderer: Optional[Callable[[DropdownOption[T], bool, bool], Any]] = None

        self._is_hovered = False
        self._is_focused = False

        self._dirty = True
        self._cached_mesh: Any = None

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the ID counter. Used for testing."""
        cls._next_id = 0

    def _normalize_option(self, opt) -> DropdownOption[T]:
        """Convert option to DropdownOption if needed."""
        if isinstance(opt, DropdownOption):
            return opt
        return DropdownOption(value=opt, label=str(opt))

    def _normalize_options(self, options: list) -> list[DropdownOption[T]]:
        """Convert all options to DropdownOption objects."""
        return [self._normalize_option(o) for o in options] if options else []

    def _get_option_value(self, opt) -> T:
        """Get the value from an option (handles both DropdownOption and plain values)."""
        return opt.value if isinstance(opt, DropdownOption) else opt

    def _get_option_label(self, opt) -> str:
        """Get the label from an option (handles both DropdownOption and plain values)."""
        return opt.label if isinstance(opt, DropdownOption) else str(opt)

    def _find_option_index(self, value: T) -> int:
        """Find the index of an option by value.

        Args:
            value: Value to find

        Returns:
            Index of the option, or -1 if not found
        """
        for i, opt in enumerate(self._options):
            if self._get_option_value(opt) == value:
                return i
        return -1

    @property
    def id(self) -> int:
        """Get the unique widget ID."""
        return self._id

    @property
    def options(self) -> list[DropdownOption[T]]:
        """Get the list of options."""
        return self._options

    @options.setter
    def options(self, value: list[DropdownOption[T]]) -> None:
        """Set the options list."""
        self._options = self._normalize_options(value)
        # Validate current selection
        valid_indices = []
        for idx in self._selected_indices:
            if 0 <= idx < len(self._options):
                valid_indices.append(idx)
        self._selected_indices = valid_indices
        self._highlighted_index = -1
        self._scroll_offset = 0
        self._dirty = True

    def add_option(self, option) -> None:
        """Add an option to the dropdown."""
        self._options.append(self._normalize_option(option))
        self._dirty = True

    def remove_option(self, value: T) -> None:
        """Remove an option by value."""
        idx = self._find_option_index(value)
        if idx >= 0:
            self._options.pop(idx)
            # Adjust selection indices
            self._selected_indices = [
                i - 1 if i > idx else i
                for i in self._selected_indices
                if i != idx
            ]
            self._dirty = True

    def clear_options(self) -> None:
        """Clear all options."""
        self._options = []
        self._selected_indices = []
        self._highlighted_index = -1
        self._scroll_offset = 0
        self._dirty = True

    def set_options(self, options: list) -> None:
        """Replace all options."""
        self.options = options

    def get_option(self, value: T):
        """Get an option by value."""
        idx = self._find_option_index(value)
        if idx >= 0:
            return self._options[idx]
        return None

    def _matches_filter(self, opt, filter_text: str) -> bool:
        """Check if option matches filter text."""
        if isinstance(opt, DropdownOption):
            return opt.matches_filter(filter_text)
        return filter_text.lower() in str(opt).lower()

    @property
    def _filtered_options_with_indices(self) -> list[tuple[int, DropdownOption[T]]]:
        """Get options filtered by current search text with indices.

        Returns:
            List of (original_index, option) tuples
        """
        if not self._filter_text:
            return list(enumerate(self._options))
        return [
            (i, opt) for i, opt in enumerate(self._options)
            if self._matches_filter(opt, self._filter_text)
        ]

    @property
    def filtered_options(self) -> list[DropdownOption[T]]:
        """Get options filtered by current search text."""
        if not self._filter_text:
            return self._options[:]
        return [
            opt for opt in self._options
            if self._matches_filter(opt, self._filter_text)
        ]

    @property
    def grouped_options(self) -> dict[str, list[DropdownOption[T]]]:
        """Get options grouped by their group attribute."""
        groups: dict[str, list[DropdownOption[T]]] = {}
        for opt in self._options:
            group = getattr(opt, 'group', None)
            if group:
                if group not in groups:
                    groups[group] = []
                groups[group].append(opt)
        return groups

    @property
    def has_groups(self) -> bool:
        """Check if any options have a group defined."""
        return any(getattr(opt, 'group', None) for opt in self._options)

    @property
    def selected_value(self) -> Optional[T]:
        """Get the selected value (first if multi-select)."""
        if self._selected_indices:
            return self._get_option_value(self._options[self._selected_indices[0]])
        return None

    @selected_value.setter
    def selected_value(self, value: Optional[T]) -> None:
        """Set the selected value (single-select mode)."""
        if value is None:
            self._set_selection([], is_user_action=False)
        else:
            idx = self._find_option_index(value)
            if idx < 0:
                raise ValueError(f"Value {value!r} not in options")
            self._set_selection([idx], is_user_action=False)

    @property
    def selected_values(self) -> list[T]:
        """Get all selected values."""
        return [self._get_option_value(self._options[i]) for i in self._selected_indices]

    @selected_values.setter
    def selected_values(self, values: list[T]) -> None:
        """Set the selected values (multi-select mode)."""
        indices = []
        for val in values:
            idx = self._find_option_index(val)
            if idx >= 0 and idx not in indices:
                indices.append(idx)
        self._set_selection(indices, is_user_action=False)

    @property
    def selected_option(self) -> Optional[DropdownOption[T]]:
        """Get the selected option (first if multi-select)."""
        if self._selected_indices:
            return self._options[self._selected_indices[0]]
        return None

    @property
    def selected_options(self) -> list[DropdownOption[T]]:
        """Get all selected options."""
        return [self._options[i] for i in self._selected_indices]

    @property
    def selected_label(self) -> Optional[str]:
        """Get the label of the selected option (first if multi-select)."""
        if self._selected_indices:
            return self._get_option_label(self._options[self._selected_indices[0]])
        return None

    @property
    def display_text(self) -> str:
        """Get the text to display in the dropdown button."""
        if not self._selected_indices:
            return self._placeholder

        if self._multi_select:
            count = len(self._selected_indices)
            if count == 1:
                return self._get_option_label(self._options[self._selected_indices[0]])
            return f"{count} selected"
        else:
            return self._get_option_label(self._options[self._selected_indices[0]])

    @property
    def placeholder(self) -> str:
        """Get the placeholder text."""
        return self._placeholder

    @placeholder.setter
    def placeholder(self, value: str) -> None:
        """Set the placeholder text."""
        if self._placeholder != value:
            self._placeholder = value
            self._dirty = True

    @property
    def is_open(self) -> bool:
        """Check if dropdown is open."""
        return self._is_open

    @property
    def multi_select(self) -> bool:
        """Check if multi-select mode is enabled."""
        return self._multi_select

    @multi_select.setter
    def multi_select(self, value: bool) -> None:
        """Set multi-select mode."""
        if self._multi_select != value:
            self._multi_select = value
            # In single-select, keep only first selection
            if not value and len(self._selected_indices) > 1:
                self._selected_indices = self._selected_indices[:1]
            self._dirty = True

    @property
    def searchable(self) -> bool:
        """Check if search is enabled."""
        return self._searchable

    @searchable.setter
    def searchable(self, value: bool) -> None:
        """Set searchable mode."""
        if self._searchable != value:
            self._searchable = value
            self._filter_text = ""
            self._dirty = True

    @property
    def filter_text(self) -> str:
        """Get the current filter text."""
        return self._filter_text

    @filter_text.setter
    def filter_text(self, value: str) -> None:
        """Set the filter text."""
        if self._filter_text != value:
            self._filter_text = value
            self._highlighted_index = -1
            self._scroll_offset = 0
            self._dirty = True

    @property
    def search_text(self) -> str:
        """Alias for filter_text."""
        return self._filter_text

    @search_text.setter
    def search_text(self, value: str) -> None:
        """Alias for filter_text."""
        self.filter_text = value

    @property
    def highlighted_index(self) -> int:
        """Get the currently highlighted option index."""
        return self._highlighted_index

    @highlighted_index.setter
    def highlighted_index(self, value: int) -> None:
        """Set the highlighted option index."""
        if 0 <= value < len(self._options):
            self._highlighted_index = value
            self._dirty = True

    @property
    def enabled(self) -> bool:
        """Check if dropdown is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set enabled state."""
        if self._enabled != value:
            self._enabled = value
            if not value and self._is_open:
                self.close()
            self._update_visual_state()
            self._dirty = True

    @property
    def visible(self) -> bool:
        """Check if dropdown is visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set visibility."""
        if self._visible != value:
            self._visible = value
            if not value and self._is_open:
                self.close()
            self._dirty = True

    @property
    def focusable(self) -> bool:
        """Check if dropdown can receive focus."""
        return self._focusable and self._enabled

    @focusable.setter
    def focusable(self, value: bool) -> None:
        """Set focusable state."""
        self._focusable = value

    @property
    def visual_state(self) -> DropdownState:
        """Get current visual state."""
        return self._visual_state

    @property
    def style(self) -> DropdownStyle:
        """Get dropdown style."""
        return self._style

    @style.setter
    def style(self, value: DropdownStyle) -> None:
        """Set dropdown style."""
        self._style = value
        self._dirty = True

    @property
    def custom_renderer(self) -> Optional[Callable[[DropdownOption[T], bool, bool], Any]]:
        """Get the custom item renderer function."""
        return self._custom_renderer

    @custom_renderer.setter
    def custom_renderer(self, value: Optional[Callable[[DropdownOption[T], bool, bool], Any]]) -> None:
        """Set a custom item renderer function.

        The function receives (option, is_selected, is_highlighted) and should
        return render data appropriate for the rendering system.
        """
        self._custom_renderer = value
        self._dirty = True

    @property
    def x(self) -> float:
        """Get X position."""
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        """Set X position."""
        if self._x != value:
            self._x = value
            self._dirty = True

    @property
    def y(self) -> float:
        """Get Y position."""
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        """Set Y position."""
        if self._y != value:
            self._y = value
            self._dirty = True

    @property
    def width(self) -> float:
        """Get widget width."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set widget width."""
        if value < 0:
            raise ValueError("width must be >= 0")
        if self._width != value:
            self._width = value
            self._dirty = True

    @property
    def height(self) -> float:
        """Get widget height (button only)."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set widget height."""
        if value < 0:
            raise ValueError("height must be >= 0")
        if self._height != value:
            self._height = value
            self._dirty = True

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Get button bounds (x, y, width, height)."""
        return (self._x, self._y, self._width, self._height)

    @property
    def dropdown_bounds(self) -> tuple[float, float, float, float]:
        """Get the dropdown menu bounds when open."""
        visible_count = min(len(self.filtered_options), self._style.max_visible_items)
        search_height = self._height if self._searchable else 0
        dropdown_height = visible_count * self._style.item_height + search_height
        return (self._x, self._y + self._height, self._width, dropdown_height)

    @property
    def total_bounds(self) -> tuple[float, float, float, float]:
        """Get total bounds including dropdown when open."""
        if not self._is_open:
            return self.bounds
        db = self.dropdown_bounds
        return (self._x, self._y, self._width, self._height + db[3])

    @property
    def is_dirty(self) -> bool:
        """Check if dropdown needs re-rendering."""
        return self._dirty

    def mark_clean(self) -> None:
        """Mark the dropdown as rendered."""
        self._dirty = False

    def _update_visual_state(self) -> None:
        """Update visual state based on current conditions."""
        if not self._enabled:
            self._visual_state = DropdownState.DISABLED
        elif self._is_open:
            self._visual_state = DropdownState.OPEN
        elif self._is_focused:
            self._visual_state = DropdownState.FOCUSED
        elif self._is_hovered:
            self._visual_state = DropdownState.HOVERED
        else:
            self._visual_state = DropdownState.NORMAL

    def _set_selection(self, indices: list[int], is_user_action: bool = True) -> None:
        """Set the selection to the given indices.

        Args:
            indices: List of indices to select
            is_user_action: Whether this is a user action
        """
        # Validate indices
        valid = [i for i in indices if 0 <= i < len(self._options) and not getattr(self._options[i], 'disabled', False)]

        if self._selected_indices != valid:
            previous_values = self.selected_values
            self._selected_indices = valid
            self._dirty = True
            self._emit_selection_change(previous_values, is_user_action)

    def _emit_selection_change(self, previous_values: list[T], is_user_action: bool = True) -> None:
        """Emit selection change event to all handlers."""
        event = SelectionChangeEvent(
            dropdown=self,
            timestamp=time(),
            selected_values=self.selected_values,
            previous_values=previous_values,
            is_user_action=is_user_action,
        )
        for handler in self._on_selection_change_handlers:
            handler(event)

    def _emit_open_state(self) -> None:
        """Emit open state change event to all handlers."""
        event = OpenStateChangeEvent(
            dropdown=self,
            timestamp=time(),
            is_open=self._is_open,
        )
        if self._is_open:
            for handler in self._on_open_handlers:
                handler(event)
        else:
            for handler in self._on_close_handlers:
                handler(event)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within the widget bounds.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside bounds
        """
        if self._is_open:
            tb = self.total_bounds
            return (
                tb[0] <= x <= tb[0] + tb[2] and
                tb[1] <= y <= tb[1] + tb[3]
            )
        return (
            self._x <= x <= self._x + self._width and
            self._y <= y <= self._y + self._height
        )

    def contains_point_in_button(self, x: float, y: float) -> bool:
        """Check if a point is within the button bounds.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside button
        """
        return (
            self._x <= x <= self._x + self._width and
            self._y <= y <= self._y + self._height
        )

    def contains_point_in_dropdown(self, x: float, y: float) -> bool:
        """Check if a point is within the dropdown menu.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside dropdown menu
        """
        if not self._is_open:
            return False
        db = self.dropdown_bounds
        return (
            db[0] <= x <= db[0] + db[2] and
            db[1] <= y <= db[1] + db[3]
        )

    # Event subscription methods
    def on_selection_change(
        self, handler: Callable[[SelectionChangeEvent[T]], None]
    ) -> Callable[[], None]:
        """Subscribe to selection change events.

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_selection_change_handlers.append(handler)
        return lambda: self._on_selection_change_handlers.remove(handler)

    def on_open(self, handler: Callable[[OpenStateChangeEvent], None]) -> Callable[[], None]:
        """Subscribe to open state change events.

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_open_handlers.append(handler)
        return lambda: self._on_open_handlers.remove(handler)

    def on_close(self, handler: Callable[[OpenStateChangeEvent], None]) -> Callable[[], None]:
        """Subscribe to close state change events.

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_close_handlers.append(handler)
        return lambda: self._on_close_handlers.remove(handler)

    # Dropdown control methods
    def open(self) -> None:
        """Open the dropdown."""
        if not self._enabled or self._is_open or not self._options:
            return

        self._is_open = True
        self._filter_text = ""
        self._scroll_offset = 0

        # Highlight first selected or first option
        if self._selected_indices:
            self._highlighted_index = self._selected_indices[0]
        elif self._options:
            self._highlighted_index = 0
        else:
            self._highlighted_index = -1

        self._update_visual_state()
        self._dirty = True
        self._emit_open_state()

    def close(self) -> None:
        """Close the dropdown."""
        if not self._is_open:
            return

        self._is_open = False
        self._filter_text = ""
        self._highlighted_index = -1
        self._update_visual_state()
        self._dirty = True
        self._emit_open_state()

    def toggle(self) -> None:
        """Toggle the dropdown open/closed."""
        if self._is_open:
            self.close()
        else:
            self.open()

    def navigate_down(self) -> None:
        """Navigate to the next option, wrapping and skipping disabled."""
        if not self._options:
            return
        start = self._highlighted_index
        count = len(self._options)
        for _ in range(count):
            self._highlighted_index = (self._highlighted_index + 1) % count
            if not getattr(self._options[self._highlighted_index], 'disabled', False):
                self._dirty = True
                return
        self._highlighted_index = start

    def navigate_up(self) -> None:
        """Navigate to the previous option, wrapping and skipping disabled."""
        if not self._options:
            return
        start = self._highlighted_index
        count = len(self._options)
        for _ in range(count):
            self._highlighted_index = (self._highlighted_index - 1) % count
            if not getattr(self._options[self._highlighted_index], 'disabled', False):
                self._dirty = True
                return
        self._highlighted_index = start

    def select_option(self, index: int) -> None:
        """Select an option by index.

        Args:
            index: Index of the option to select
        """
        if not self._enabled:
            return

        if index < 0 or index >= len(self._options):
            return

        option = self._options[index]
        if option.disabled:
            return

        if self._multi_select:
            # Toggle selection
            if index in self._selected_indices:
                indices = [i for i in self._selected_indices if i != index]
            else:
                indices = self._selected_indices + [index]
            self._set_selection(indices)
        else:
            # Single select - close after selection
            self._set_selection([index])
            self.close()

    def select_highlighted(self) -> None:
        """Select the currently highlighted option."""
        if self._highlighted_index >= 0:
            # Map filtered index to real index
            filtered = self._filtered_options_with_indices
            for i, (real_idx, _) in enumerate(filtered):
                if i == self._highlighted_index:
                    self.select_option(real_idx)
                    break

    def clear_selection(self) -> None:
        """Clear all selections."""
        if self._selected_indices:
            self._set_selection([])

    def select_all(self) -> None:
        """Select all options (multi-select only)."""
        if not self._multi_select:
            return

        indices = [
            i for i, opt in enumerate(self._options)
            if not opt.disabled
        ]
        self._set_selection(indices)

    # Navigation methods
    def highlight_next(self) -> None:
        """Highlight the next option."""
        filtered = self.filtered_options
        if not filtered:
            return

        if self._highlighted_index < 0:
            self._highlighted_index = 0
        else:
            self._highlighted_index = min(
                self._highlighted_index + 1,
                len(filtered) - 1
            )

        self._ensure_highlighted_visible()
        self._dirty = True

    def highlight_previous(self) -> None:
        """Highlight the previous option."""
        filtered = self.filtered_options
        if not filtered:
            return

        if self._highlighted_index < 0:
            self._highlighted_index = len(filtered) - 1
        else:
            self._highlighted_index = max(self._highlighted_index - 1, 0)

        self._ensure_highlighted_visible()
        self._dirty = True

    def highlight_first(self) -> None:
        """Highlight the first option."""
        filtered = self.filtered_options
        if filtered:
            self._highlighted_index = 0
            self._scroll_offset = 0
            self._dirty = True

    def highlight_last(self) -> None:
        """Highlight the last option."""
        filtered = self.filtered_options
        if filtered:
            self._highlighted_index = len(filtered) - 1
            self._ensure_highlighted_visible()
            self._dirty = True

    def _ensure_highlighted_visible(self) -> None:
        """Ensure the highlighted item is visible by scrolling if needed."""
        max_visible = self._style.max_visible_items
        if self._highlighted_index < self._scroll_offset:
            self._scroll_offset = self._highlighted_index
        elif self._highlighted_index >= self._scroll_offset + max_visible:
            self._scroll_offset = self._highlighted_index - max_visible + 1

    def get_item_at_position(self, x: float, y: float) -> int:
        """Get the index of the item at a position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Filtered index of the item, or -1 if none
        """
        if not self.contains_point_in_dropdown(x, y):
            return -1

        db = self.dropdown_bounds
        search_height = self._height if self._searchable else 0
        relative_y = y - db[1] - search_height

        if relative_y < 0:
            return -1  # In search area

        item_index = int(relative_y / self._style.item_height) + self._scroll_offset
        filtered = self.filtered_options

        if 0 <= item_index < len(filtered):
            return item_index
        return -1

    # Input event handlers
    def handle_mouse_enter(self) -> None:
        """Handle mouse entering the dropdown area."""
        if not self._enabled:
            return
        self._is_hovered = True
        self._update_visual_state()
        self._dirty = True

    def handle_mouse_leave(self) -> None:
        """Handle mouse leaving the dropdown area."""
        self._is_hovered = False
        self._update_visual_state()
        self._dirty = True

    def handle_mouse_down(self, x: float, y: float) -> bool:
        """Handle mouse button press.

        Args:
            x: Mouse X position
            y: Mouse Y position

        Returns:
            True if event was consumed
        """
        if not self._enabled:
            return False
        return self.contains_point_in_button(x, y) or (self._is_open and self.contains_point_in_dropdown(x, y))

    def handle_mouse_up(self, x: float, y: float) -> bool:
        """Handle mouse button release.

        Args:
            x: Mouse X position
            y: Mouse Y position

        Returns:
            True if event was consumed
        """
        if not self._enabled:
            return False

        # Click on button
        if self.contains_point_in_button(x, y):
            self.toggle()
            return True

        # Click on dropdown item
        if self._is_open and self.contains_point_in_dropdown(x, y):
            item_idx = self.get_item_at_position(x, y)
            if item_idx >= 0:
                self.select_option(item_idx)
            return True

        # Click outside - close
        if self._is_open:
            self.close()
            return True

        return False

    def handle_click_outside(self) -> None:
        """Handle click outside the dropdown."""
        if self._is_open:
            self.close()

    def handle_option_click(self, index: int) -> None:
        """Handle click on an option by index."""
        self.select_option(index)

    def handle_option_hover(self, index: int) -> None:
        """Handle hover over an option by index."""
        if 0 <= index < len(self._options):
            self._highlighted_index = index
            self._dirty = True

    def handle_mouse_move(self, x: float, y: float) -> bool:
        """Handle mouse movement.

        Args:
            x: Mouse X position
            y: Mouse Y position

        Returns:
            True if event was consumed
        """
        if not self._is_open:
            return False

        item_idx = self.get_item_at_position(x, y)
        if item_idx >= 0 and item_idx != self._highlighted_index:
            self._highlighted_index = item_idx
            self._dirty = True
            return True
        return False

    def handle_mouse_scroll(self, delta: float) -> bool:
        """Handle mouse scroll.

        Args:
            delta: Scroll delta (positive = up)

        Returns:
            True if event was consumed
        """
        if not self._is_open:
            return False

        filtered = self.filtered_options
        max_scroll = max(0, len(filtered) - self._style.max_visible_items)

        if delta > 0:
            self._scroll_offset = max(0, self._scroll_offset - 1)
        else:
            self._scroll_offset = min(max_scroll, self._scroll_offset + 1)

        self._dirty = True
        return True

    def handle_focus_gained(self) -> None:
        """Handle receiving keyboard focus."""
        if not self._enabled:
            return
        self._is_focused = True
        self._update_visual_state()
        self._dirty = True

    def handle_focus_lost(self) -> None:
        """Handle losing keyboard focus."""
        self._is_focused = False
        if self._is_open:
            self.close()
        self._update_visual_state()
        self._dirty = True

    def handle_key_down(self, key: str, shift: bool = False, ctrl: bool = False, alt: bool = False) -> bool:
        """Handle keyboard key press.

        Args:
            key: Key identifier
            shift: Shift modifier state
            ctrl: Ctrl modifier state
            alt: Alt modifier state (reserved for future use)

        Returns:
            True if event was consumed
        """
        if not self._enabled:
            return False

        # When open, keys work even without focus
        if self._is_open:
            if key == "escape":
                self.close()
                return True
            if key in ("enter", "return"):
                self.select_highlighted()
                if not self._multi_select:
                    self.close()
                return True
            elif key == "down":
                self.highlight_next()
                return True
            elif key == "up":
                self.highlight_previous()
                return True
            elif key == "home":
                self.highlight_first()
                return True
            elif key == "end":
                self.highlight_last()
                return True
            elif key == "pagedown":
                for _ in range(self._style.max_visible_items):
                    self.highlight_next()
                return True
            elif key == "pageup":
                for _ in range(self._style.max_visible_items):
                    self.highlight_previous()
                return True
            elif key == "space" and self._multi_select:
                self.select_highlighted()
                return True
            elif ctrl and key == "a" and self._multi_select:
                self.select_all()
                return True
            elif key == "backspace" and self._searchable:
                return self.handle_search_backspace()
        else:
            # Dropdown closed
            if key in ("enter", "return", "space", "down"):
                self.open()
                return True

        return False

    def handle_text_input(self, text: str) -> bool:
        """Handle text input for search.

        Args:
            text: Input text

        Returns:
            True if event was consumed
        """
        if not self._is_open or not self._searchable:
            return False

        if text.isprintable():
            self._filter_text += text
            self._highlighted_index = 0 if self.filtered_options else -1
            self._scroll_offset = 0
            self._dirty = True
            return True
        return False

    def handle_search_backspace(self) -> bool:
        """Handle backspace in search field.

        Returns:
            True if event was consumed
        """
        if not self._is_open or not self._searchable or not self._filter_text:
            return False

        self._filter_text = self._filter_text[:-1]
        self._highlighted_index = 0 if self.filtered_options else -1
        self._scroll_offset = 0
        self._dirty = True
        return True

    # Visual state helpers
    def get_current_background_color(self) -> str:
        """Get the background color for current state.

        Returns:
            Color string for current state
        """
        if self._visual_state == DropdownState.DISABLED:
            return self._style.disabled_color
        elif self._visual_state == DropdownState.OPEN:
            return self._style.open_color
        elif self._visual_state == DropdownState.HOVERED:
            return self._style.hover_color
        else:
            return self._style.background_color

    def get_current_text_color(self) -> str:
        """Get the text color for current state.

        Returns:
            Text color string for current state
        """
        if self._visual_state == DropdownState.DISABLED:
            return self._style.disabled_text_color
        elif not self._selected_indices:
            return self._style.placeholder_color
        else:
            return self._style.text_color

    def get_current_border_color(self) -> str:
        """Get the border color for current state.

        Returns:
            Border color string for current state
        """
        if self._visual_state in (DropdownState.FOCUSED, DropdownState.OPEN):
            return self._style.focused_border_color
        return self._style.border_color

    def get_item_background(self, filtered_index: int) -> str:
        """Get background color for an item.

        Args:
            filtered_index: Index in filtered list

        Returns:
            Background color for the item
        """
        if filtered_index < 0:
            return self._style.dropdown_background

        filtered = self._filtered_options_with_indices
        if filtered_index >= len(filtered):
            return self._style.dropdown_background

        real_idx, _ = filtered[filtered_index]
        is_selected = real_idx in self._selected_indices
        is_highlighted = filtered_index == self._highlighted_index

        if is_highlighted:
            return self._style.item_hover_color
        elif is_selected:
            return self._style.item_selected_color
        else:
            return self._style.dropdown_background

    def get_visible_items(self) -> list[tuple[int, DropdownOption[T], bool, bool]]:
        """Get the visible items for rendering.

        Returns:
            List of (real_index, option, is_selected, is_highlighted) tuples
        """
        filtered = self._filtered_options_with_indices
        start = self._scroll_offset
        end = min(start + self._style.max_visible_items, len(filtered))

        result = []
        for i in range(start, end):
            real_idx, opt = filtered[i]
            is_selected = real_idx in self._selected_indices
            is_highlighted = i == self._highlighted_index
            result.append((real_idx, opt, is_selected, is_highlighted))
        return result

    def to_dict(self) -> dict:
        """Serialize dropdown state to dictionary."""
        options_data = []
        for opt in self._options:
            if isinstance(opt, DropdownOption):
                options_data.append({
                    "value": opt.value,
                    "label": opt.label,
                    "disabled": opt.disabled,
                    "icon": opt.icon,
                    "group": opt.group,
                })
            else:
                options_data.append({"value": opt, "label": str(opt)})

        return {
            "options": options_data,
            "selected_value": self.selected_value,
            "selected_values": self.selected_values,
            "placeholder": self._placeholder,
            "multi_select": self._multi_select,
            "searchable": self._searchable,
            "enabled": self._enabled,
            "visible": self._visible,
            "x": self._x,
            "y": self._y,
            "width": self._width,
            "height": self._height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Dropdown[T]":
        """Create dropdown from dictionary."""
        options = []
        for opt_data in data.get("options", []):
            if isinstance(opt_data, dict):
                options.append(DropdownOption(
                    value=opt_data.get("value"),
                    label=opt_data.get("label", ""),
                    disabled=opt_data.get("disabled", False),
                    icon=opt_data.get("icon"),
                    group=opt_data.get("group"),
                ))
            else:
                options.append(opt_data)

        return cls(
            options=options,
            selected_value=data.get("selected_value"),
            selected_values=data.get("selected_values"),
            placeholder=data.get("placeholder", "Select..."),
            multi_select=data.get("multi_select", False),
            searchable=data.get("searchable", False),
            enabled=data.get("enabled", True),
            visible=data.get("visible", True),
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            width=data.get("width", 200.0),
            height=data.get("height", 40.0),
        )
