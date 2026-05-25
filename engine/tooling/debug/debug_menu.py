"""
Debug Menu - Hierarchical debug menu system with categories.

Provides an in-game debug menu for toggling features, adjusting values,
and executing debug actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, ClassVar, Optional, Any, Generic, TypeVar
import threading
import time


T = TypeVar('T')


class MenuCategory(Enum):
    """Categories for menu organization."""
    GENERAL = auto()
    RENDERING = auto()
    PHYSICS = auto()
    AI = auto()
    GAMEPLAY = auto()
    AUDIO = auto()
    NETWORK = auto()
    PERFORMANCE = auto()
    CHEATS = auto()
    CUSTOM = auto()


class MenuItemType(Enum):
    """Types of menu items."""
    SUBMENU = auto()
    TOGGLE = auto()
    SLIDER = auto()
    ACTION = auto()
    TEXT = auto()
    SEPARATOR = auto()
    DROPDOWN = auto()
    COLOR_PICKER = auto()


@dataclass
class MenuStyle:
    """Visual style for menus."""
    background_color: tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.9)
    text_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    highlight_color: tuple[float, float, float, float] = (0.3, 0.5, 0.8, 1.0)
    disabled_color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
    separator_color: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0)
    font_size: float = 14.0
    item_height: float = 24.0
    padding: float = 8.0
    indent: float = 16.0


class MenuItem(ABC):
    """Base class for menu items."""

    __slots__ = (
        '_id',
        '_label',
        '_enabled',
        '_visible',
        '_tooltip',
        '_shortcut',
        '_category',
        '_parent',
    )

    def __init__(
        self,
        item_id: str,
        label: str,
        enabled: bool = True,
        visible: bool = True,
        tooltip: str = "",
        shortcut: str = "",
        category: MenuCategory = MenuCategory.GENERAL,
    ):
        self._id = item_id
        self._label = label
        self._enabled = enabled
        self._visible = visible
        self._tooltip = tooltip
        self._shortcut = shortcut
        self._category = category
        self._parent: Optional["SubMenu"] = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def visible(self) -> bool:
        return self._visible

    def show(self) -> None:
        self._visible = True

    def hide(self) -> None:
        self._visible = False

    @property
    def tooltip(self) -> str:
        return self._tooltip

    @tooltip.setter
    def tooltip(self, value: str) -> None:
        self._tooltip = value

    @property
    def shortcut(self) -> str:
        return self._shortcut

    @property
    def category(self) -> MenuCategory:
        return self._category

    @property
    def parent(self) -> Optional["SubMenu"]:
        return self._parent

    @abstractmethod
    def get_type(self) -> MenuItemType:
        """Get the menu item type."""
        pass

    @abstractmethod
    def render(self) -> dict[str, Any]:
        """Render the menu item."""
        pass


class MenuToggle(MenuItem):
    """Toggle/checkbox menu item."""

    __slots__ = ('_value', '_on_change')

    def __init__(
        self,
        item_id: str,
        label: str,
        value: bool = False,
        on_change: Optional[Callable[[bool], None]] = None,
        **kwargs,
    ):
        super().__init__(item_id=item_id, label=label, **kwargs)
        self._value = value
        self._on_change = on_change

    def get_type(self) -> MenuItemType:
        return MenuItemType.TOGGLE

    @property
    def value(self) -> bool:
        return self._value

    @value.setter
    def value(self, new_value: bool) -> None:
        if self._value != new_value:
            self._value = new_value
            if self._on_change:
                self._on_change(new_value)

    def toggle(self) -> bool:
        """Toggle the value. Returns new value."""
        self.value = not self._value
        return self._value

    def set_on_change(self, callback: Callable[[bool], None]) -> None:
        """Set the change callback."""
        self._on_change = callback

    def render(self) -> dict[str, Any]:
        return {
            "type": "toggle",
            "id": self._id,
            "label": self._label,
            "value": self._value,
            "enabled": self._enabled,
            "visible": self._visible,
            "tooltip": self._tooltip,
            "shortcut": self._shortcut,
        }


class MenuSlider(MenuItem, Generic[T]):
    """Slider menu item for numeric values."""

    __slots__ = ('_value', '_min_value', '_max_value', '_step', '_format_string', '_on_change')

    def __init__(
        self,
        item_id: str,
        label: str,
        value: T,
        min_value: T,
        max_value: T,
        step: Optional[T] = None,
        format_string: str = "{value}",
        on_change: Optional[Callable[[T], None]] = None,
        **kwargs,
    ):
        super().__init__(item_id=item_id, label=label, **kwargs)
        self._value = value
        self._min_value = min_value
        self._max_value = max_value
        self._step = step
        self._format_string = format_string
        self._on_change = on_change

    def get_type(self) -> MenuItemType:
        return MenuItemType.SLIDER

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, new_value: T) -> None:
        # Clamp to range
        if isinstance(new_value, (int, float)):
            new_value = max(self._min_value, min(self._max_value, new_value))  # type: ignore

        if self._value != new_value:
            self._value = new_value
            if self._on_change:
                self._on_change(new_value)

    @property
    def min_value(self) -> T:
        return self._min_value

    @property
    def max_value(self) -> T:
        return self._max_value

    @property
    def step(self) -> Optional[T]:
        return self._step

    def increment(self) -> T:
        """Increment value by step."""
        if self._step is not None:
            self.value = self._value + self._step  # type: ignore
        return self._value

    def decrement(self) -> T:
        """Decrement value by step."""
        if self._step is not None:
            self.value = self._value - self._step  # type: ignore
        return self._value

    def get_formatted_value(self) -> str:
        """Get formatted value string."""
        return self._format_string.format(value=self._value)

    def set_on_change(self, callback: Callable[[T], None]) -> None:
        """Set the change callback."""
        self._on_change = callback

    def render(self) -> dict[str, Any]:
        return {
            "type": "slider",
            "id": self._id,
            "label": self._label,
            "value": self._value,
            "min_value": self._min_value,
            "max_value": self._max_value,
            "step": self._step,
            "formatted_value": self.get_formatted_value(),
            "enabled": self._enabled,
            "visible": self._visible,
            "tooltip": self._tooltip,
        }


class MenuAction(MenuItem):
    """Action button menu item."""

    __slots__ = ('_callback',)

    def __init__(
        self,
        item_id: str,
        label: str,
        callback: Callable[[], Any],
        **kwargs,
    ):
        super().__init__(item_id=item_id, label=label, **kwargs)
        self._callback = callback

    def get_type(self) -> MenuItemType:
        return MenuItemType.ACTION

    def execute(self) -> Any:
        """Execute the action."""
        if self._enabled and self._callback:
            return self._callback()
        return None

    def set_callback(self, callback: Callable[[], Any]) -> None:
        """Set the action callback."""
        self._callback = callback

    def render(self) -> dict[str, Any]:
        return {
            "type": "action",
            "id": self._id,
            "label": self._label,
            "enabled": self._enabled,
            "visible": self._visible,
            "tooltip": self._tooltip,
            "shortcut": self._shortcut,
        }


class MenuText(MenuItem):
    """Static text/label menu item."""

    __slots__ = ('_text',)

    def __init__(
        self,
        item_id: str,
        text: str,
        **kwargs,
    ):
        super().__init__(item_id=item_id, label=text, **kwargs)
        self._text = text

    def get_type(self) -> MenuItemType:
        return MenuItemType.TEXT

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        self._text = value
        self._label = value

    def render(self) -> dict[str, Any]:
        return {
            "type": "text",
            "id": self._id,
            "text": self._text,
            "visible": self._visible,
        }


class MenuSeparator(MenuItem):
    """Separator line menu item."""

    def __init__(self, item_id: str = ""):
        super().__init__(item_id=item_id or f"separator_{id(self)}", label="")

    def get_type(self) -> MenuItemType:
        return MenuItemType.SEPARATOR

    def render(self) -> dict[str, Any]:
        return {
            "type": "separator",
            "id": self._id,
            "visible": self._visible,
        }


class MenuDropdown(MenuItem, Generic[T]):
    """Dropdown selection menu item."""

    __slots__ = ('_options', '_selected_index', '_on_change')

    def __init__(
        self,
        item_id: str,
        label: str,
        options: list[tuple[str, T]],  # (display_name, value)
        selected_index: int = 0,
        on_change: Optional[Callable[[T], None]] = None,
        **kwargs,
    ):
        super().__init__(item_id=item_id, label=label, **kwargs)
        self._options = options
        self._selected_index = max(0, min(selected_index, len(options) - 1)) if options else 0
        self._on_change = on_change

    def get_type(self) -> MenuItemType:
        return MenuItemType.DROPDOWN

    @property
    def options(self) -> list[tuple[str, T]]:
        return self._options

    @property
    def selected_index(self) -> int:
        return self._selected_index

    @selected_index.setter
    def selected_index(self, index: int) -> None:
        if 0 <= index < len(self._options) and index != self._selected_index:
            self._selected_index = index
            if self._on_change:
                self._on_change(self._options[index][1])

    @property
    def selected_value(self) -> Optional[T]:
        if 0 <= self._selected_index < len(self._options):
            return self._options[self._selected_index][1]
        return None

    @property
    def selected_label(self) -> str:
        if 0 <= self._selected_index < len(self._options):
            return self._options[self._selected_index][0]
        return ""

    def select_next(self) -> None:
        """Select next option."""
        if self._options:
            self.selected_index = (self._selected_index + 1) % len(self._options)

    def select_previous(self) -> None:
        """Select previous option."""
        if self._options:
            self.selected_index = (self._selected_index - 1) % len(self._options)

    def set_on_change(self, callback: Callable[[T], None]) -> None:
        """Set the change callback."""
        self._on_change = callback

    def render(self) -> dict[str, Any]:
        return {
            "type": "dropdown",
            "id": self._id,
            "label": self._label,
            "options": [opt[0] for opt in self._options],
            "selected_index": self._selected_index,
            "selected_label": self.selected_label,
            "enabled": self._enabled,
            "visible": self._visible,
            "tooltip": self._tooltip,
        }


class SubMenu(MenuItem):
    """Submenu containing other menu items."""

    __slots__ = ('_items', '_expanded')

    def __init__(
        self,
        item_id: str,
        label: str,
        items: Optional[list[MenuItem]] = None,
        expanded: bool = False,
        **kwargs,
    ):
        super().__init__(item_id=item_id, label=label, **kwargs)
        self._items: list[MenuItem] = []
        self._expanded = expanded

        if items:
            for item in items:
                self.add_item(item)

    def get_type(self) -> MenuItemType:
        return MenuItemType.SUBMENU

    @property
    def items(self) -> list[MenuItem]:
        return self._items.copy()

    @property
    def expanded(self) -> bool:
        return self._expanded

    def expand(self) -> None:
        """Expand the submenu."""
        self._expanded = True

    def collapse(self) -> None:
        """Collapse the submenu."""
        self._expanded = False

    def toggle_expanded(self) -> bool:
        """Toggle expansion. Returns new state."""
        self._expanded = not self._expanded
        return self._expanded

    def add_item(self, item: MenuItem) -> None:
        """Add an item to the submenu."""
        item._parent = self
        self._items.append(item)

    def remove_item(self, item_id: str) -> Optional[MenuItem]:
        """Remove an item by ID."""
        for i, item in enumerate(self._items):
            if item.id == item_id:
                removed = self._items.pop(i)
                removed._parent = None
                return removed
        return None

    def get_item(self, item_id: str) -> Optional[MenuItem]:
        """Get an item by ID (recursive search)."""
        for item in self._items:
            if item.id == item_id:
                return item
            if isinstance(item, SubMenu):
                found = item.get_item(item_id)
                if found:
                    return found
        return None

    def clear(self) -> None:
        """Clear all items."""
        for item in self._items:
            item._parent = None
        self._items.clear()

    def add_separator(self) -> MenuSeparator:
        """Add a separator."""
        sep = MenuSeparator()
        self.add_item(sep)
        return sep

    def add_toggle(
        self,
        item_id: str,
        label: str,
        value: bool = False,
        on_change: Optional[Callable[[bool], None]] = None,
        **kwargs,
    ) -> MenuToggle:
        """Add a toggle item."""
        toggle = MenuToggle(item_id, label, value, on_change, **kwargs)
        self.add_item(toggle)
        return toggle

    def add_slider(
        self,
        item_id: str,
        label: str,
        value: float,
        min_value: float,
        max_value: float,
        step: Optional[float] = None,
        on_change: Optional[Callable[[float], None]] = None,
        **kwargs,
    ) -> MenuSlider:
        """Add a slider item."""
        slider = MenuSlider(item_id, label, value, min_value, max_value, step, on_change=on_change, **kwargs)
        self.add_item(slider)
        return slider

    def add_action(
        self,
        item_id: str,
        label: str,
        callback: Callable[[], Any],
        **kwargs,
    ) -> MenuAction:
        """Add an action item."""
        action = MenuAction(item_id, label, callback, **kwargs)
        self.add_item(action)
        return action

    def add_submenu(
        self,
        item_id: str,
        label: str,
        **kwargs,
    ) -> "SubMenu":
        """Add a submenu."""
        submenu = SubMenu(item_id, label, **kwargs)
        self.add_item(submenu)
        return submenu

    @property
    def item_count(self) -> int:
        return len(self._items)

    def render(self) -> dict[str, Any]:
        return {
            "type": "submenu",
            "id": self._id,
            "label": self._label,
            "expanded": self._expanded,
            "items": [item.render() for item in self._items if item.visible],
            "enabled": self._enabled,
            "visible": self._visible,
            "tooltip": self._tooltip,
        }


class DebugMenu:
    """Main debug menu system."""

    _instance: ClassVar[Optional["DebugMenu"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    __slots__ = (
        '_root',
        '_enabled',
        '_visible',
        '_style',
        '_selected_item',
        '_shortcut_map',
    )

    def __init__(self):
        self._root = SubMenu("root", "Debug Menu")
        self._enabled = True
        self._visible = False
        self._style = MenuStyle()
        self._selected_item: Optional[str] = None
        self._shortcut_map: dict[str, str] = {}  # shortcut -> item_id

        self._create_default_menus()

    @classmethod
    def get_instance(cls) -> "DebugMenu":
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

    def _create_default_menus(self) -> None:
        """Create default debug menu structure."""
        # Rendering submenu
        rendering = self._root.add_submenu("rendering", "Rendering", category=MenuCategory.RENDERING)
        rendering.add_toggle("wireframe", "Wireframe Mode")
        rendering.add_toggle("show_bounds", "Show Bounding Boxes")
        rendering.add_toggle("show_lod", "Show LOD Levels")
        rendering.add_toggle("show_overdraw", "Show Overdraw")

        # Physics submenu
        physics = self._root.add_submenu("physics", "Physics", category=MenuCategory.PHYSICS)
        physics.add_toggle("show_colliders", "Show Colliders")
        physics.add_toggle("show_contacts", "Show Contact Points")
        physics.add_toggle("show_raycasts", "Show Raycasts")

        # AI submenu
        ai = self._root.add_submenu("ai", "AI", category=MenuCategory.AI)
        ai.add_toggle("show_paths", "Show AI Paths")
        ai.add_toggle("show_perception", "Show Perception Radius")
        ai.add_toggle("show_navmesh", "Show Nav Mesh")

        # Performance submenu
        performance = self._root.add_submenu("performance", "Performance", category=MenuCategory.PERFORMANCE)
        performance.add_toggle("show_fps", "Show FPS")
        performance.add_toggle("show_memory", "Show Memory Usage")
        performance.add_toggle("show_stats", "Show Statistics")

    def enable(self) -> None:
        """Enable the menu."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the menu."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def show(self) -> None:
        """Show the menu."""
        self._visible = True

    def hide(self) -> None:
        """Hide the menu."""
        self._visible = False

    def toggle(self) -> bool:
        """Toggle menu visibility. Returns new state."""
        self._visible = not self._visible
        return self._visible

    @property
    def is_visible(self) -> bool:
        return self._visible

    @property
    def root(self) -> SubMenu:
        return self._root

    @property
    def style(self) -> MenuStyle:
        return self._style

    @style.setter
    def style(self, value: MenuStyle) -> None:
        self._style = value

    def get_item(self, item_id: str) -> Optional[MenuItem]:
        """Get an item by ID."""
        return self._root.get_item(item_id)

    def add_item(self, item: MenuItem, parent_id: Optional[str] = None) -> bool:
        """Add an item to the menu."""
        if parent_id:
            parent = self._root.get_item(parent_id)
            if isinstance(parent, SubMenu):
                parent.add_item(item)
                return True
            return False
        else:
            self._root.add_item(item)
            return True

    def remove_item(self, item_id: str) -> Optional[MenuItem]:
        """Remove an item."""
        return self._root.remove_item(item_id)

    def register_shortcut(self, shortcut: str, item_id: str) -> None:
        """Register a keyboard shortcut for an item."""
        self._shortcut_map[shortcut] = item_id

    def handle_shortcut(self, shortcut: str) -> bool:
        """Handle a keyboard shortcut. Returns True if handled."""
        if shortcut in self._shortcut_map:
            item = self.get_item(self._shortcut_map[shortcut])
            if item and item.enabled:
                if isinstance(item, MenuToggle):
                    item.toggle()
                    return True
                elif isinstance(item, MenuAction):
                    item.execute()
                    return True
                elif isinstance(item, SubMenu):
                    item.toggle_expanded()
                    return True
        return False

    def select_item(self, item_id: str) -> bool:
        """Select an item."""
        item = self.get_item(item_id)
        if item:
            self._selected_item = item_id
            return True
        return False

    @property
    def selected_item(self) -> Optional[str]:
        return self._selected_item

    def activate_selected(self) -> bool:
        """Activate the currently selected item."""
        if self._selected_item:
            item = self.get_item(self._selected_item)
            if item and item.enabled:
                if isinstance(item, MenuToggle):
                    item.toggle()
                    return True
                elif isinstance(item, MenuAction):
                    item.execute()
                    return True
                elif isinstance(item, SubMenu):
                    item.toggle_expanded()
                    return True
        return False

    def render(self) -> dict[str, Any]:
        """Render the menu."""
        if not self._enabled or not self._visible:
            return {}

        return {
            "type": "debug_menu",
            "visible": self._visible,
            "style": {
                "background": self._style.background_color,
                "text_color": self._style.text_color,
                "highlight_color": self._style.highlight_color,
                "disabled_color": self._style.disabled_color,
                "font_size": self._style.font_size,
                "item_height": self._style.item_height,
                "padding": self._style.padding,
            },
            "root": self._root.render(),
            "selected_item": self._selected_item,
        }

    def get_all_toggles(self) -> dict[str, bool]:
        """Get all toggle values."""
        toggles = {}
        self._collect_toggles(self._root, toggles)
        return toggles

    def _collect_toggles(self, menu: SubMenu, toggles: dict[str, bool]) -> None:
        """Recursively collect toggle values."""
        for item in menu.items:
            if isinstance(item, MenuToggle):
                toggles[item.id] = item.value
            elif isinstance(item, SubMenu):
                self._collect_toggles(item, toggles)

    def set_toggle(self, item_id: str, value: bool) -> bool:
        """Set a toggle value."""
        item = self.get_item(item_id)
        if isinstance(item, MenuToggle):
            item.value = value
            return True
        return False

    def set_slider(self, item_id: str, value: float) -> bool:
        """Set a slider value."""
        item = self.get_item(item_id)
        if isinstance(item, MenuSlider):
            item.value = value
            return True
        return False
