"""
Comprehensive tests for the Editor Application Shell.

Tests cover:
- Tab management (create, close, switch, dirty tracking)
- Docking system (dock, undock, float, layout save/load)
- Panel visibility and state
- Menu bar operations
- Toolbar functionality
- Status bar messages
- Editor application lifecycle
"""
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.editor.app_shell import (
    EditorApplication,
    DockingManager,
    Panel,
    PanelPosition,
    Tab,
    TabGroup,
    MenuBar,
    MenuItem,
    ToolBar,
    StatusBar,
)


class TestPanel:
    """Tests for Panel class."""

    def test_panel_creation(self):
        """Panel should be created with default values."""
        panel = Panel(id="test", title="Test Panel")
        assert panel.id == "test"
        assert panel.title == "Test Panel"
        assert panel.visible is True
        assert panel.minimized is False
        assert panel.position == PanelPosition.LEFT

    def test_panel_show_hide_toggle(self):
        """Panel visibility can be toggled."""
        panel = Panel(id="test", title="Test")
        assert panel.visible is True

        panel.hide()
        assert panel.visible is False

        panel.show()
        assert panel.visible is True
        assert panel.minimized is False

        panel.toggle()
        assert panel.visible is False

        panel.toggle()
        assert panel.visible is True

    def test_panel_minimize_restore(self):
        """Panel can be minimized and restored."""
        panel = Panel(id="test", title="Test")

        panel.minimize()
        assert panel.minimized is True

        panel.restore()
        assert panel.minimized is False

    def test_panel_resize_respects_minimum(self):
        """Panel resize should respect minimum dimensions."""
        panel = Panel(id="test", title="Test", min_width=100, min_height=100)

        panel.resize(50, 50)
        assert panel.width == 100
        assert panel.height == 100

        panel.resize(200, 300)
        assert panel.width == 200
        assert panel.height == 300

    def test_panel_close_callback(self):
        """Panel close should trigger callback."""
        closed = []
        panel = Panel(id="test", title="Test", on_close=lambda: closed.append(True))

        panel.close()
        assert panel.visible is False
        assert len(closed) == 1

    def test_panel_focus_callback(self):
        """Panel focus should trigger callback."""
        focused = []
        panel = Panel(id="test", title="Test", on_focus=lambda: focused.append(True))

        panel.focus()
        assert len(focused) == 1


class TestTab:
    """Tests for Tab class."""

    def test_tab_creation(self):
        """Tab should be created with default values."""
        tab = Tab(id="tab1", label="Tab 1")
        assert tab.id == "tab1"
        assert tab.label == "Tab 1"
        assert tab.closable is True
        assert tab.dirty is False

    def test_tab_dirty_tracking(self):
        """Tab dirty state can be tracked."""
        tab = Tab(id="tab1", label="Tab 1")

        tab.mark_dirty()
        assert tab.dirty is True

        tab.mark_clean()
        assert tab.dirty is False

    def test_tab_close_closable(self):
        """Closable tab can be closed."""
        tab = Tab(id="tab1", label="Tab 1", closable=True)
        assert tab.close() is True

    def test_tab_close_not_closable(self):
        """Non-closable tab cannot be closed."""
        tab = Tab(id="tab1", label="Tab 1", closable=False)
        assert tab.close() is False

    def test_tab_close_callback_prevents_close(self):
        """Close callback can prevent tab closing."""
        tab = Tab(id="tab1", label="Tab 1", on_close=lambda: False)
        assert tab.close() is False

        tab.on_close = lambda: True
        assert tab.close() is True

    def test_tab_select_callback(self):
        """Select should trigger callback."""
        selected = []
        tab = Tab(id="tab1", label="Tab 1", on_select=lambda: selected.append(True))

        tab.select()
        assert len(selected) == 1


class TestTabGroup:
    """Tests for TabGroup class."""

    def test_tabgroup_creation(self):
        """TabGroup should be created empty."""
        group = TabGroup(id="group1")
        assert group.id == "group1"
        assert group.tab_count == 0
        assert group.active_tab is None

    def test_tabgroup_add_tab(self):
        """Adding tab should make it active if first."""
        group = TabGroup(id="group1")
        tab = Tab(id="tab1", label="Tab 1")

        group.add_tab(tab)
        assert group.tab_count == 1
        assert group.active_tab == tab

    def test_tabgroup_multiple_tabs(self):
        """Multiple tabs can be added."""
        group = TabGroup(id="group1")
        tab1 = Tab(id="tab1", label="Tab 1")
        tab2 = Tab(id="tab2", label="Tab 2")

        group.add_tab(tab1)
        group.add_tab(tab2)

        assert group.tab_count == 2
        assert group.active_tab == tab1  # First added stays active

    def test_tabgroup_remove_tab(self):
        """Removing tab should update active if needed."""
        group = TabGroup(id="group1")
        tab1 = Tab(id="tab1", label="Tab 1")
        tab2 = Tab(id="tab2", label="Tab 2")

        group.add_tab(tab1)
        group.add_tab(tab2)

        removed = group.remove_tab("tab1")
        assert removed == tab1
        assert group.tab_count == 1
        assert group.active_tab == tab2

    def test_tabgroup_set_active_tab(self):
        """Active tab can be changed."""
        group = TabGroup(id="group1")
        tab1 = Tab(id="tab1", label="Tab 1")
        tab2 = Tab(id="tab2", label="Tab 2")

        group.add_tab(tab1)
        group.add_tab(tab2)

        assert group.set_active_tab("tab2") is True
        assert group.active_tab == tab2

        assert group.set_active_tab("nonexistent") is False

    def test_tabgroup_close_tab(self):
        """Closing tab should remove it from group."""
        group = TabGroup(id="group1")
        tab = Tab(id="tab1", label="Tab 1", closable=True)

        group.add_tab(tab)
        assert group.close_tab("tab1") is True
        assert group.tab_count == 0

    def test_tabgroup_close_all_tabs(self):
        """Close all should close closable tabs."""
        group = TabGroup(id="group1")
        tab1 = Tab(id="tab1", label="Tab 1", closable=True)
        tab2 = Tab(id="tab2", label="Tab 2", closable=False)

        group.add_tab(tab1)
        group.add_tab(tab2)

        closed = group.close_all_tabs()
        assert closed == 1
        assert group.tab_count == 1

    def test_tabgroup_has_dirty_tabs(self):
        """Should detect dirty tabs."""
        group = TabGroup(id="group1")
        tab1 = Tab(id="tab1", label="Tab 1")
        tab2 = Tab(id="tab2", label="Tab 2")

        group.add_tab(tab1)
        group.add_tab(tab2)

        assert group.has_dirty_tabs() is False

        tab1.mark_dirty()
        assert group.has_dirty_tabs() is True

    def test_tabgroup_tab_change_callback(self):
        """Tab change should trigger callback."""
        changes = []
        group = TabGroup(id="group1")
        group.on_tab_change = lambda t: changes.append(t)

        tab1 = Tab(id="tab1", label="Tab 1")
        tab2 = Tab(id="tab2", label="Tab 2")

        group.add_tab(tab1)
        group.add_tab(tab2)
        group.set_active_tab("tab2")

        assert len(changes) == 1
        assert changes[0] == tab2


class TestMenuItem:
    """Tests for MenuItem class."""

    def test_menuitem_creation(self):
        """MenuItem should be created with defaults."""
        item = MenuItem(label="File")
        assert item.label == "File"
        assert item.enabled is True
        assert item.separator is False

    def test_menuitem_separator(self):
        """Separator can be created."""
        sep = MenuItem.create_separator()
        assert sep.separator is True

    def test_menuitem_execute(self):
        """Execute should call action."""
        executed = []
        item = MenuItem(label="Test", action=lambda: executed.append(True))

        assert item.execute() is True
        assert len(executed) == 1

    def test_menuitem_execute_disabled(self):
        """Disabled item should not execute."""
        executed = []
        item = MenuItem(label="Test", action=lambda: executed.append(True), enabled=False)

        assert item.execute() is False
        assert len(executed) == 0

    def test_menuitem_toggle_checkable(self):
        """Checkable item can be toggled."""
        item = MenuItem(label="Option", checked=False)

        item.toggle()
        assert item.checked is True

        item.toggle()
        assert item.checked is False

    def test_menuitem_toggle_non_checkable(self):
        """Non-checkable item toggle does nothing."""
        item = MenuItem(label="Option", checked=None)

        item.toggle()
        assert item.checked is None


class TestMenuBar:
    """Tests for MenuBar class."""

    def test_menubar_creation(self):
        """MenuBar should be created empty."""
        menubar = MenuBar()
        assert len(menubar.menu_names) == 0

    def test_menubar_add_menu(self):
        """Menu can be added."""
        menubar = MenuBar()
        menubar.add_menu("File", [MenuItem(label="New")])

        assert "File" in menubar.menu_names
        menu = menubar.get_menu("File")
        assert len(menu) == 1

    def test_menubar_remove_menu(self):
        """Menu can be removed."""
        menubar = MenuBar()
        menubar.add_menu("File", [MenuItem(label="New")])

        removed = menubar.remove_menu("File")
        assert removed is not None
        assert "File" not in menubar.menu_names

    def test_menubar_add_item(self):
        """Item can be added to menu."""
        menubar = MenuBar()
        menubar.add_menu("File", [])

        item = MenuItem(label="New")
        assert menubar.add_item("File", item) is True

        menu = menubar.get_menu("File")
        assert len(menu) == 1

    def test_menubar_find_item(self):
        """Item can be found by label."""
        menubar = MenuBar()
        menubar.add_menu("File", [MenuItem(label="New"), MenuItem(label="Open")])

        item = menubar.find_item("File", "Open")
        assert item is not None
        assert item.label == "Open"

    def test_menubar_execute_item(self):
        """Menu item can be executed by name."""
        executed = []
        menubar = MenuBar()
        menubar.add_menu("File", [MenuItem(label="New", action=lambda: executed.append(True))])

        assert menubar.execute_item("File", "New") is True
        assert len(executed) == 1


class TestToolBar:
    """Tests for ToolBar class."""

    def test_toolbar_creation(self):
        """ToolBar should be created empty."""
        toolbar = ToolBar(id="main", name="Main Toolbar")
        assert toolbar.id == "main"
        assert toolbar.name == "Main Toolbar"
        assert len(toolbar.tools) == 0

    def test_toolbar_add_tool(self):
        """Tool can be added."""
        toolbar = ToolBar(id="main")
        toolbar.add_tool("select", "select_icon", "Select Tool")

        assert len(toolbar.tools) == 1
        tool = toolbar.get_tool("select")
        assert tool is not None
        assert tool["icon"] == "select_icon"

    def test_toolbar_remove_tool(self):
        """Tool can be removed."""
        toolbar = ToolBar(id="main")
        toolbar.add_tool("select", "select_icon")

        assert toolbar.remove_tool("select") is True
        assert toolbar.get_tool("select") is None

    def test_toolbar_click_tool(self):
        """Clicking tool should trigger action."""
        clicked = []
        toolbar = ToolBar(id="main")
        toolbar.add_tool("select", "icon", action=lambda: clicked.append(True))

        assert toolbar.click_tool("select") is True
        assert len(clicked) == 1

    def test_toolbar_click_disabled_tool(self):
        """Disabled tool should not be clickable."""
        clicked = []
        toolbar = ToolBar(id="main")
        toolbar.add_tool("select", "icon", action=lambda: clicked.append(True), enabled=False)

        assert toolbar.click_tool("select") is False
        assert len(clicked) == 0

    def test_toolbar_checkable_tool(self):
        """Checkable tool can be toggled."""
        toolbar = ToolBar(id="main")
        toolbar.add_tool("grid", "icon", checkable=True, checked=False)

        toolbar.click_tool("grid")
        tool = toolbar.get_tool("grid")
        assert tool["checked"] is True

    def test_toolbar_set_tool_enabled(self):
        """Tool enabled state can be changed."""
        toolbar = ToolBar(id="main")
        toolbar.add_tool("select", "icon")

        toolbar.set_tool_enabled("select", False)
        tool = toolbar.get_tool("select")
        assert tool["enabled"] is False


class TestStatusBar:
    """Tests for StatusBar class."""

    def test_statusbar_creation(self):
        """StatusBar should be created visible."""
        statusbar = StatusBar()
        assert statusbar.visible is True
        assert len(statusbar.sections) == 0

    def test_statusbar_add_section(self):
        """Section can be added."""
        statusbar = StatusBar()
        statusbar.add_section("status", "Ready")

        assert "status" in statusbar.sections
        assert statusbar.get_text("status") == "Ready"

    def test_statusbar_set_text(self):
        """Section text can be changed."""
        statusbar = StatusBar()
        statusbar.add_section("status", "Ready")

        assert statusbar.set_text("status", "Working...") is True
        assert statusbar.get_text("status") == "Working..."

    def test_statusbar_remove_section(self):
        """Section can be removed."""
        statusbar = StatusBar()
        statusbar.add_section("status", "Ready")

        assert statusbar.remove_section("status") is True
        assert "status" not in statusbar.sections

    def test_statusbar_temporary_message(self):
        """Temporary messages can be shown."""
        statusbar = StatusBar()
        statusbar.show_message("File saved!", 5000)

        assert statusbar.get_current_message() == "File saved!"


class TestDockingManager:
    """Tests for DockingManager class."""

    def test_docking_creation(self):
        """DockingManager should be created empty."""
        docking = DockingManager()
        assert len(docking.panels) == 0

    def test_docking_register_panel(self):
        """Panel can be registered."""
        docking = DockingManager()
        panel = Panel(id="hierarchy", title="Hierarchy", position=PanelPosition.LEFT)

        docking.register_panel(panel)
        assert len(docking.panels) == 1
        assert docking.get_panel("hierarchy") == panel

    def test_docking_unregister_panel(self):
        """Panel can be unregistered."""
        docking = DockingManager()
        panel = Panel(id="hierarchy", title="Hierarchy")

        docking.register_panel(panel)
        removed = docking.unregister_panel("hierarchy")

        assert removed == panel
        assert len(docking.panels) == 0

    def test_docking_dock_panel(self):
        """Panel can be docked to position."""
        docking = DockingManager()
        panel = Panel(id="hierarchy", title="Hierarchy", position=PanelPosition.LEFT)

        docking.register_panel(panel)
        assert docking.dock_panel("hierarchy", PanelPosition.RIGHT) is True
        assert panel.position == PanelPosition.RIGHT

    def test_docking_float_panel(self):
        """Panel can be floated."""
        docking = DockingManager()
        panel = Panel(id="hierarchy", title="Hierarchy", position=PanelPosition.LEFT)

        docking.register_panel(panel)
        assert docking.float_panel("hierarchy") is True
        assert panel.position == PanelPosition.FLOATING
        assert panel in docking.get_floating_panels()

    def test_docking_get_panels_at_position(self):
        """Panels at position can be retrieved."""
        docking = DockingManager()
        panel1 = Panel(id="p1", title="P1", position=PanelPosition.LEFT)
        panel2 = Panel(id="p2", title="P2", position=PanelPosition.LEFT)
        panel3 = Panel(id="p3", title="P3", position=PanelPosition.RIGHT)

        docking.register_panel(panel1)
        docking.register_panel(panel2)
        docking.register_panel(panel3)

        left_panels = docking.get_panels_at_position(PanelPosition.LEFT)
        assert len(left_panels) == 2

    def test_docking_save_load_layout(self):
        """Layout can be saved and loaded."""
        docking = DockingManager()
        panel = Panel(id="hierarchy", title="Hierarchy", position=PanelPosition.LEFT)
        docking.register_panel(panel)

        # Save layout
        layout = docking.save_layout()
        assert "panels" in layout
        assert "hierarchy" in layout["panels"]

        # Modify panel
        panel.position = PanelPosition.RIGHT
        panel.width = 500

        # Load layout
        docking.load_layout(layout)
        assert panel.position == PanelPosition.LEFT


class TestEditorApplication:
    """Tests for EditorApplication class."""

    def test_application_creation(self):
        """EditorApplication should be created with defaults."""
        app = EditorApplication(title="Test Editor")
        assert app.title == "Test Editor"
        assert app.is_running is False

    def test_application_startup_shutdown(self):
        """Application can start and stop."""
        app = EditorApplication()

        started = []
        stopped = []
        app.on_startup = lambda: started.append(True)
        app.on_shutdown = lambda: stopped.append(True)

        app.startup()
        assert app.is_running is True
        assert len(started) == 1

        app.shutdown()
        assert app.is_running is False
        assert len(stopped) == 1

    def test_application_update_when_running(self):
        """Application update is called when running."""
        app = EditorApplication()
        updates = []
        app.on_update = lambda dt: updates.append(dt)

        app.update(0.016)  # Not running
        assert len(updates) == 0

        app.startup()
        app.update(0.016)
        assert len(updates) == 1

    def test_application_docking_manager(self):
        """Application has docking manager."""
        app = EditorApplication()
        assert app.docking is not None
        assert isinstance(app.docking, DockingManager)

    def test_application_menu_bar(self):
        """Application has menu bar."""
        app = EditorApplication()
        assert app.menu_bar is not None
        assert isinstance(app.menu_bar, MenuBar)

    def test_application_status_bar(self):
        """Application has status bar."""
        app = EditorApplication()
        assert app.status_bar is not None
        assert isinstance(app.status_bar, StatusBar)

    def test_application_add_toolbar(self):
        """Toolbars can be added and retrieved."""
        app = EditorApplication()
        toolbar = ToolBar(id="main", name="Main")

        app.add_toolbar(toolbar)
        assert app.get_toolbar("main") == toolbar
        assert toolbar in app.toolbars

    def test_application_remove_toolbar(self):
        """Toolbars can be removed."""
        app = EditorApplication()
        toolbar = ToolBar(id="main", name="Main")

        app.add_toolbar(toolbar)
        removed = app.remove_toolbar("main")

        assert removed == toolbar
        assert app.get_toolbar("main") is None

    def test_application_create_panel(self):
        """Panel can be created and registered."""
        app = EditorApplication()
        panel = app.create_panel("hierarchy", "Hierarchy", PanelPosition.LEFT)

        assert panel is not None
        assert app.docking.get_panel("hierarchy") == panel

    def test_application_create_tab_group(self):
        """Tab group can be created and registered."""
        app = EditorApplication()
        group = app.create_tab_group("main_content")

        assert group is not None
        assert app.get_tab_group("main_content") == group
        assert group in app.tab_groups


class TestEditorDecorators:
    """Tests for editor decorators."""

    def test_editor_decorator_applied(self):
        """@editor decorator sets attributes."""
        assert Panel._editor is True
        assert Panel._editor_category == "Shell"
        assert Panel._editor_hidden is False

    def test_reloadable_decorator_applied(self):
        """@reloadable decorator sets attributes."""
        assert Panel._reloadable is True
        assert "id" in Panel._reload_preserve
        assert "title" in Panel._reload_preserve
