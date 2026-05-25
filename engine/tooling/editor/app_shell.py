"""
Application Shell - Main editor application with docking, tabs, and panels.

Provides the core editor window management including:
- Docking system for flexible panel layout
- Tab groups for organizing content
- Menu bar with hierarchical menus
- Toolbars with tool buttons
- Status bar with information display
"""
from __future__ import annotations

import uuid
import weakref
from typing import TYPE_CHECKING
from enum import Enum, auto
from typing import Any, Callable, Optional, Union


def editor(category: str = "General", hidden: bool = False):
    """Decorator marking editor-only classes."""
    def decorator(cls):
        cls._editor = True
        cls._editor_category = category
        cls._editor_hidden = hidden
        return cls
    return decorator


def reloadable(enabled: bool = True, preserve: list[str] = None,
               reinitialize: list[str] = None, validate: Callable = None):
    """Decorator for hot-reload support."""
    def decorator(cls):
        cls._reloadable = enabled
        cls._reload_preserve = preserve or []
        cls._reload_reinitialize = reinitialize or []
        cls._reload_validate = validate
        return cls
    return decorator


class PanelPosition(Enum):
    """Position for docking panels."""
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()
    CENTER = auto()
    FLOATING = auto()


@editor(category="Shell")
@reloadable(preserve=["id", "title"])
class Panel:
    """A dockable panel in the editor."""
    __slots__ = ("id", "title", "position", "visible", "minimized", "width",
                 "height", "min_width", "min_height", "content", "on_close",
                 "on_focus", "_parent_ref")

    def __init__(self, id: str, title: str, position: PanelPosition = PanelPosition.LEFT,
                 visible: bool = True, minimized: bool = False, width: int = 300,
                 height: int = 400, min_width: int = 100, min_height: int = 100,
                 content: Any = None, on_close: Optional[Callable] = None,
                 on_focus: Optional[Callable] = None):
        self.id = id
        self.title = title
        self.position = position
        self.visible = visible
        self.minimized = minimized
        self.width = width
        self.height = height
        self.min_width = min_width
        self.min_height = min_height
        self.content = content
        self.on_close = on_close
        self.on_focus = on_focus
        self._parent_ref = None

    def show(self) -> None:
        """Show the panel."""
        self.visible = True
        self.minimized = False

    def hide(self) -> None:
        """Hide the panel."""
        self.visible = False

    def toggle(self) -> None:
        """Toggle panel visibility."""
        self.visible = not self.visible

    def minimize(self) -> None:
        """Minimize the panel."""
        self.minimized = True

    def restore(self) -> None:
        """Restore from minimized state."""
        self.minimized = False

    def close(self) -> None:
        """Close the panel, triggering on_close callback."""
        self.visible = False
        if self.on_close:
            self.on_close()

    def focus(self) -> None:
        """Focus the panel, triggering on_focus callback."""
        if self.on_focus:
            self.on_focus()

    def resize(self, width: int, height: int) -> None:
        """Resize panel respecting minimum dimensions."""
        self.width = max(width, self.min_width)
        self.height = max(height, self.min_height)


@editor(category="Shell")
@reloadable(preserve=["id", "label"])
class Tab:
    """A tab within a tab group."""
    __slots__ = ("id", "label", "tooltip", "closable", "content", "dirty",
                 "icon", "on_close", "on_select", "_group_ref")

    def __init__(self, id: str, label: str, tooltip: str = "", closable: bool = True,
                 content: Any = None, dirty: bool = False, icon: Optional[str] = None,
                 on_close: Optional[Callable] = None, on_select: Optional[Callable] = None):
        self.id = id
        self.label = label
        self.tooltip = tooltip
        self.closable = closable
        self.content = content
        self.dirty = dirty
        self.icon = icon
        self.on_close = on_close
        self.on_select = on_select
        self._group_ref = None

    def mark_dirty(self) -> None:
        """Mark tab as having unsaved changes."""
        self.dirty = True

    def mark_clean(self) -> None:
        """Mark tab as saved."""
        self.dirty = False

    def close(self) -> bool:
        """Attempt to close the tab. Returns True if closed."""
        if not self.closable:
            return False
        if self.on_close:
            return self.on_close()
        return True

    def select(self) -> None:
        """Select this tab."""
        if self.on_select:
            self.on_select()


@editor(category="Shell")
@reloadable(preserve=["id"])
class TabGroup:
    """A group of tabs that can be switched between."""
    __slots__ = ("id", "_tabs", "_active_tab_id", "on_tab_change", "__weakref__")

    def __init__(self, id: str):
        self.id = id
        self._tabs: dict[str, Tab] = {}
        self._active_tab_id: Optional[str] = None
        self.on_tab_change: Optional[Callable[[Optional[Tab]], None]] = None

    @property
    def tabs(self) -> list[Tab]:
        """Get all tabs in order."""
        return list(self._tabs.values())

    @property
    def active_tab(self) -> Optional[Tab]:
        """Get the currently active tab."""
        if self._active_tab_id:
            return self._tabs.get(self._active_tab_id)
        return None

    @property
    def tab_count(self) -> int:
        """Get number of tabs."""
        return len(self._tabs)

    def add_tab(self, tab: Tab) -> None:
        """Add a tab to the group."""
        tab._group_ref = weakref.ref(self)
        self._tabs[tab.id] = tab
        if self._active_tab_id is None:
            self._active_tab_id = tab.id

    def remove_tab(self, tab_id: str) -> Optional[Tab]:
        """Remove a tab from the group."""
        tab = self._tabs.pop(tab_id, None)
        if tab:
            tab._group_ref = None
            if self._active_tab_id == tab_id:
                self._active_tab_id = next(iter(self._tabs.keys()), None)
                if self.on_tab_change:
                    self.on_tab_change(self.active_tab)
        return tab

    def get_tab(self, tab_id: str) -> Optional[Tab]:
        """Get a tab by ID."""
        return self._tabs.get(tab_id)

    def set_active_tab(self, tab_id: str) -> bool:
        """Set the active tab by ID. Returns True if successful."""
        if tab_id in self._tabs:
            old_active = self._active_tab_id
            self._active_tab_id = tab_id
            self._tabs[tab_id].select()
            if old_active != tab_id and self.on_tab_change:
                self.on_tab_change(self.active_tab)
            return True
        return False

    def close_tab(self, tab_id: str) -> bool:
        """Close a tab. Returns True if closed."""
        tab = self._tabs.get(tab_id)
        if tab and tab.close():
            self.remove_tab(tab_id)
            return True
        return False

    def close_all_tabs(self, force: bool = False) -> int:
        """Close all tabs. Returns number closed."""
        closed = 0
        for tab_id in list(self._tabs.keys()):
            tab = self._tabs[tab_id]
            if force or tab.close():
                self.remove_tab(tab_id)
                closed += 1
        return closed

    def has_dirty_tabs(self) -> bool:
        """Check if any tabs have unsaved changes."""
        return any(tab.dirty for tab in self._tabs.values())


@editor(category="Shell")
@reloadable(preserve=["label", "action"])
class MenuItem:
    """A menu item in the menu bar."""
    __slots__ = ("label", "action", "shortcut", "enabled", "checked",
                 "icon", "submenu", "separator", "tooltip")

    def __init__(self, label: str = "", action: Optional[Callable] = None,
                 shortcut: Optional[str] = None, enabled: bool = True,
                 checked: Optional[bool] = None, icon: Optional[str] = None,
                 submenu: Optional[list["MenuItem"]] = None, separator: bool = False,
                 tooltip: str = ""):
        self.label = label
        self.action = action
        self.shortcut = shortcut
        self.enabled = enabled
        self.checked = checked
        self.icon = icon
        self.submenu = submenu
        self.separator = separator
        self.tooltip = tooltip

    @classmethod
    def create_separator(cls) -> "MenuItem":
        """Create a separator menu item."""
        return cls(separator=True)

    def execute(self) -> bool:
        """Execute the menu item action. Returns True if executed."""
        if self.enabled and self.action:
            self.action()
            return True
        return False

    def toggle(self) -> None:
        """Toggle checked state if checkable."""
        if self.checked is not None:
            self.checked = not self.checked


@editor(category="Shell")
@reloadable(preserve=["_menus"])
class MenuBar:
    """The main menu bar of the editor."""
    __slots__ = ("_menus", "on_menu_action")

    def __init__(self):
        self._menus: dict[str, list[MenuItem]] = {}
        self.on_menu_action: Optional[Callable[[str, MenuItem], None]] = None

    @property
    def menu_names(self) -> list[str]:
        """Get all menu names."""
        return list(self._menus.keys())

    def add_menu(self, name: str, items: list[MenuItem] = None) -> None:
        """Add a menu to the menu bar."""
        self._menus[name] = items or []

    def remove_menu(self, name: str) -> Optional[list[MenuItem]]:
        """Remove a menu from the menu bar."""
        return self._menus.pop(name, None)

    def get_menu(self, name: str) -> Optional[list[MenuItem]]:
        """Get a menu by name."""
        return self._menus.get(name)

    def add_item(self, menu_name: str, item: MenuItem, index: int = -1) -> bool:
        """Add an item to a menu. Returns True if successful."""
        menu = self._menus.get(menu_name)
        if menu is not None:
            if index < 0:
                menu.append(item)
            else:
                menu.insert(index, item)
            return True
        return False

    def remove_item(self, menu_name: str, label: str) -> Optional[MenuItem]:
        """Remove an item from a menu by label."""
        menu = self._menus.get(menu_name)
        if menu:
            for i, item in enumerate(menu):
                if item.label == label:
                    return menu.pop(i)
        return None

    def find_item(self, menu_name: str, label: str) -> Optional[MenuItem]:
        """Find an item in a menu by label."""
        menu = self._menus.get(menu_name)
        if menu:
            for item in menu:
                if item.label == label:
                    return item
        return None

    def execute_item(self, menu_name: str, label: str) -> bool:
        """Execute a menu item by menu name and label."""
        item = self.find_item(menu_name, label)
        if item:
            result = item.execute()
            if result and self.on_menu_action:
                self.on_menu_action(menu_name, item)
            return result
        return False


@editor(category="Shell")
@reloadable(preserve=["_tools"])
class ToolBar:
    """A toolbar with tool buttons."""
    __slots__ = ("id", "name", "_tools", "visible", "orientation", "on_tool_click")

    def __init__(self, id: str, name: str = ""):
        self.id = id
        self.name = name or id
        self._tools: list[dict[str, Any]] = []
        self.visible: bool = True
        self.orientation: str = "horizontal"  # or "vertical"
        self.on_tool_click: Optional[Callable[[str], None]] = None

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Get all tools."""
        return list(self._tools)

    def add_tool(self, id: str, icon: str, tooltip: str = "",
                 action: Optional[Callable] = None,
                 enabled: bool = True, checkable: bool = False,
                 checked: bool = False, separator_before: bool = False) -> None:
        """Add a tool button."""
        self._tools.append({
            "id": id,
            "icon": icon,
            "tooltip": tooltip,
            "action": action,
            "enabled": enabled,
            "checkable": checkable,
            "checked": checked,
            "separator_before": separator_before,
        })

    def remove_tool(self, id: str) -> bool:
        """Remove a tool by ID."""
        for i, tool in enumerate(self._tools):
            if tool["id"] == id:
                self._tools.pop(i)
                return True
        return False

    def get_tool(self, id: str) -> Optional[dict[str, Any]]:
        """Get a tool by ID."""
        for tool in self._tools:
            if tool["id"] == id:
                return tool
        return None

    def set_tool_enabled(self, id: str, enabled: bool) -> bool:
        """Set tool enabled state."""
        tool = self.get_tool(id)
        if tool:
            tool["enabled"] = enabled
            return True
        return False

    def set_tool_checked(self, id: str, checked: bool) -> bool:
        """Set tool checked state."""
        tool = self.get_tool(id)
        if tool and tool.get("checkable"):
            tool["checked"] = checked
            return True
        return False

    def click_tool(self, id: str) -> bool:
        """Click a tool button."""
        tool = self.get_tool(id)
        if tool and tool["enabled"]:
            if tool.get("checkable"):
                tool["checked"] = not tool["checked"]
            if tool.get("action"):
                tool["action"]()
            if self.on_tool_click:
                self.on_tool_click(id)
            return True
        return False

    def add_separator(self) -> None:
        """Add a separator."""
        self._tools.append({"separator": True})


@editor(category="Shell")
@reloadable()
class StatusBar:
    """The status bar at the bottom of the editor."""
    __slots__ = ("_sections", "_messages", "visible")

    def __init__(self):
        self._sections: dict[str, dict[str, Any]] = {}
        self._messages: list[tuple[str, float]] = []  # (message, timestamp)
        self.visible: bool = True

    def add_section(self, id: str, text: str = "", width: int = -1,
                    stretch: bool = False, tooltip: str = "") -> None:
        """Add a section to the status bar."""
        self._sections[id] = {
            "text": text,
            "width": width,
            "stretch": stretch,
            "tooltip": tooltip,
        }

    def remove_section(self, id: str) -> bool:
        """Remove a section."""
        return self._sections.pop(id, None) is not None

    def set_text(self, section_id: str, text: str) -> bool:
        """Set section text."""
        section = self._sections.get(section_id)
        if section:
            section["text"] = text
            return True
        return False

    def get_text(self, section_id: str) -> Optional[str]:
        """Get section text."""
        section = self._sections.get(section_id)
        return section["text"] if section else None

    def show_message(self, message: str, duration_ms: int = 3000) -> None:
        """Show a temporary message."""
        import time
        self._messages.append((message, time.time() + duration_ms / 1000))

    def get_current_message(self) -> Optional[str]:
        """Get the current temporary message."""
        import time
        now = time.time()
        # Clean expired messages
        self._messages = [(msg, exp) for msg, exp in self._messages if exp > now]
        return self._messages[-1][0] if self._messages else None

    @property
    def sections(self) -> dict[str, dict[str, Any]]:
        """Get all sections."""
        return dict(self._sections)


@editor(category="Shell")
@reloadable(preserve=["_panels", "_layout"])
class DockingManager:
    """Manages panel docking and layout."""
    __slots__ = ("_panels", "_layout", "_floating_panels", "on_layout_change", "__weakref__")

    def __init__(self):
        self._panels: dict[str, Panel] = {}
        self._layout: dict[PanelPosition, list[str]] = {
            pos: [] for pos in PanelPosition
        }
        self._floating_panels: list[str] = []
        self.on_layout_change: Optional[Callable[[], None]] = None

    @property
    def panels(self) -> list[Panel]:
        """Get all panels."""
        return list(self._panels.values())

    def register_panel(self, panel: Panel) -> None:
        """Register a panel with the docking manager."""
        self._panels[panel.id] = panel
        panel._parent_ref = weakref.ref(self)
        self._layout[panel.position].append(panel.id)
        if panel.position == PanelPosition.FLOATING:
            self._floating_panels.append(panel.id)

    def unregister_panel(self, panel_id: str) -> Optional[Panel]:
        """Unregister a panel."""
        panel = self._panels.pop(panel_id, None)
        if panel:
            panel._parent_ref = None
            for pos_list in self._layout.values():
                if panel_id in pos_list:
                    pos_list.remove(panel_id)
            if panel_id in self._floating_panels:
                self._floating_panels.remove(panel_id)
        return panel

    def get_panel(self, panel_id: str) -> Optional[Panel]:
        """Get a panel by ID."""
        return self._panels.get(panel_id)

    def dock_panel(self, panel_id: str, position: PanelPosition) -> bool:
        """Dock a panel to a position."""
        panel = self._panels.get(panel_id)
        if not panel:
            return False

        # Remove from current position
        old_pos = panel.position
        if panel_id in self._layout[old_pos]:
            self._layout[old_pos].remove(panel_id)
        if panel_id in self._floating_panels:
            self._floating_panels.remove(panel_id)

        # Add to new position
        panel.position = position
        self._layout[position].append(panel_id)
        if position == PanelPosition.FLOATING:
            self._floating_panels.append(panel_id)

        if self.on_layout_change:
            self.on_layout_change()
        return True

    def float_panel(self, panel_id: str) -> bool:
        """Make a panel floating."""
        return self.dock_panel(panel_id, PanelPosition.FLOATING)

    def get_panels_at_position(self, position: PanelPosition) -> list[Panel]:
        """Get all panels at a position."""
        return [self._panels[pid] for pid in self._layout[position]
                if pid in self._panels]

    def get_floating_panels(self) -> list[Panel]:
        """Get all floating panels."""
        return [self._panels[pid] for pid in self._floating_panels
                if pid in self._panels]

    def save_layout(self) -> dict:
        """Save the current layout configuration."""
        return {
            "panels": {
                pid: {
                    "position": panel.position.name,
                    "visible": panel.visible,
                    "width": panel.width,
                    "height": panel.height,
                    "minimized": panel.minimized,
                }
                for pid, panel in self._panels.items()
            },
            "layout_order": {
                pos.name: list(panel_ids)
                for pos, panel_ids in self._layout.items()
            }
        }

    def load_layout(self, layout: dict) -> None:
        """Load a layout configuration."""
        panel_data = layout.get("panels", {})
        for pid, data in panel_data.items():
            panel = self._panels.get(pid)
            if panel:
                panel.position = PanelPosition[data.get("position", "LEFT")]
                panel.visible = data.get("visible", True)
                panel.width = data.get("width", panel.width)
                panel.height = data.get("height", panel.height)
                panel.minimized = data.get("minimized", False)

        layout_order = layout.get("layout_order", {})
        for pos_name, panel_ids in layout_order.items():
            pos = PanelPosition[pos_name]
            self._layout[pos] = [pid for pid in panel_ids if pid in self._panels]

        if self.on_layout_change:
            self.on_layout_change()


@editor(category="Shell")
@reloadable(preserve=["_docking", "_menu_bar", "_toolbars", "_status_bar"])
class EditorApplication:
    """Main editor application class."""
    __slots__ = ("title", "width", "height", "_docking", "_menu_bar",
                 "_tab_groups", "_toolbars", "_status_bar", "_running",
                 "on_startup", "on_shutdown", "on_update")

    def __init__(self, title: str = "Editor", width: int = 1920, height: int = 1080):
        self.title = title
        self.width = width
        self.height = height
        self._docking = DockingManager()
        self._menu_bar = MenuBar()
        self._tab_groups: dict[str, TabGroup] = {}
        self._toolbars: dict[str, ToolBar] = {}
        self._status_bar = StatusBar()
        self._running = False
        self.on_startup: Optional[Callable[[], None]] = None
        self.on_shutdown: Optional[Callable[[], None]] = None
        self.on_update: Optional[Callable[[float], None]] = None

    @property
    def docking(self) -> DockingManager:
        """Get the docking manager."""
        return self._docking

    @property
    def menu_bar(self) -> MenuBar:
        """Get the menu bar."""
        return self._menu_bar

    @property
    def status_bar(self) -> StatusBar:
        """Get the status bar."""
        return self._status_bar

    @property
    def is_running(self) -> bool:
        """Check if the application is running."""
        return self._running

    def add_toolbar(self, toolbar: ToolBar) -> None:
        """Add a toolbar."""
        self._toolbars[toolbar.id] = toolbar

    def remove_toolbar(self, id: str) -> Optional[ToolBar]:
        """Remove a toolbar."""
        return self._toolbars.pop(id, None)

    def get_toolbar(self, id: str) -> Optional[ToolBar]:
        """Get a toolbar by ID."""
        return self._toolbars.get(id)

    @property
    def toolbars(self) -> list[ToolBar]:
        """Get all toolbars."""
        return list(self._toolbars.values())

    def add_tab_group(self, group: TabGroup) -> None:
        """Add a tab group."""
        self._tab_groups[group.id] = group

    def remove_tab_group(self, id: str) -> Optional[TabGroup]:
        """Remove a tab group."""
        return self._tab_groups.pop(id, None)

    def get_tab_group(self, id: str) -> Optional[TabGroup]:
        """Get a tab group by ID."""
        return self._tab_groups.get(id)

    @property
    def tab_groups(self) -> list[TabGroup]:
        """Get all tab groups."""
        return list(self._tab_groups.values())

    def startup(self) -> None:
        """Start the editor application."""
        if self._running:
            return
        self._running = True
        if self.on_startup:
            self.on_startup()

    def shutdown(self) -> None:
        """Shutdown the editor application."""
        if not self._running:
            return
        self._running = False
        if self.on_shutdown:
            self.on_shutdown()

    def update(self, delta_time: float) -> None:
        """Update the editor application."""
        if not self._running:
            return
        if self.on_update:
            self.on_update(delta_time)

    def create_panel(self, id: str, title: str,
                     position: PanelPosition = PanelPosition.LEFT) -> Panel:
        """Create and register a new panel."""
        panel = Panel(id=id, title=title, position=position)
        self._docking.register_panel(panel)
        return panel

    def create_tab_group(self, id: str) -> TabGroup:
        """Create and register a new tab group."""
        group = TabGroup(id=id)
        self.add_tab_group(group)
        return group
