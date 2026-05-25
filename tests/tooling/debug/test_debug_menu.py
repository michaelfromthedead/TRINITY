"""Tests for debug menu - menu navigation, toggles."""

import pytest
from engine.tooling.debug.debug_menu import (
    DebugMenu,
    MenuCategory,
    MenuItem,
    MenuToggle,
    MenuSlider,
    MenuAction,
    MenuText,
    MenuSeparator,
    MenuDropdown,
    SubMenu,
    MenuStyle,
    MenuItemType,
)


class TestMenuStyle:
    """Tests for MenuStyle class."""

    def test_default_style(self):
        style = MenuStyle()
        assert style.font_size == 14.0
        assert style.item_height == 24.0

    def test_custom_style(self):
        style = MenuStyle(
            font_size=16.0,
            padding=10.0,
            background_color=(1.0, 0.0, 0.0, 0.8),
        )
        assert style.font_size == 16.0
        assert style.padding == 10.0


class TestMenuToggle:
    """Tests for MenuToggle class."""

    def test_toggle_creation(self):
        toggle = MenuToggle(
            item_id="wireframe",
            label="Wireframe Mode",
            value=False,
        )
        assert toggle.id == "wireframe"
        assert toggle.value is False
        assert toggle.get_type() == MenuItemType.TOGGLE

    def test_toggle_value(self):
        toggle = MenuToggle("test", "Test", value=False)
        toggle.value = True
        assert toggle.value is True

    def test_toggle_method(self):
        toggle = MenuToggle("test", "Test", value=False)
        new_value = toggle.toggle()
        assert new_value is True
        assert toggle.value is True

    def test_toggle_callback(self):
        callback_values = []

        def on_change(value):
            callback_values.append(value)

        toggle = MenuToggle("test", "Test", value=False, on_change=on_change)
        toggle.value = True
        assert callback_values == [True]

    def test_toggle_render(self):
        toggle = MenuToggle("test", "Test Label", value=True)
        render_data = toggle.render()
        assert render_data["type"] == "toggle"
        assert render_data["id"] == "test"
        assert render_data["label"] == "Test Label"
        assert render_data["value"] is True


class TestMenuSlider:
    """Tests for MenuSlider class."""

    def test_slider_creation(self):
        slider = MenuSlider(
            item_id="volume",
            label="Volume",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
        )
        assert slider.id == "volume"
        assert slider.value == 0.5
        assert slider.get_type() == MenuItemType.SLIDER

    def test_slider_value_clamping(self):
        slider = MenuSlider("test", "Test", value=0.5, min_value=0.0, max_value=1.0)
        slider.value = 2.0
        assert slider.value == 1.0
        slider.value = -1.0
        assert slider.value == 0.0

    def test_slider_increment_decrement(self):
        slider = MenuSlider("test", "Test", value=5, min_value=0, max_value=10, step=1)
        slider.increment()
        assert slider.value == 6
        slider.decrement()
        assert slider.value == 5

    def test_slider_callback(self):
        values = []

        def on_change(value):
            values.append(value)

        slider = MenuSlider("test", "Test", value=0, min_value=0, max_value=10, on_change=on_change)
        slider.value = 5
        assert values == [5]

    def test_slider_formatted_value(self):
        slider = MenuSlider(
            "test",
            "Test",
            value=50.5,
            min_value=0,
            max_value=100,
            format_string="{value:.1f}%",
        )
        assert slider.get_formatted_value() == "50.5%"

    def test_slider_render(self):
        slider = MenuSlider("test", "Test", value=5, min_value=0, max_value=10)
        render_data = slider.render()
        assert render_data["type"] == "slider"
        assert render_data["value"] == 5
        assert render_data["min_value"] == 0
        assert render_data["max_value"] == 10


class TestMenuAction:
    """Tests for MenuAction class."""

    def test_action_creation(self):
        action = MenuAction(
            item_id="reset",
            label="Reset Settings",
            callback=lambda: "reset",
        )
        assert action.id == "reset"
        assert action.get_type() == MenuItemType.ACTION

    def test_action_execute(self):
        executed = [False]

        def callback():
            executed[0] = True
            return "done"

        action = MenuAction("test", "Test", callback=callback)
        result = action.execute()
        assert executed[0] is True
        assert result == "done"

    def test_action_execute_disabled(self):
        executed = [False]
        action = MenuAction("test", "Test", callback=lambda: executed.__setitem__(0, True))
        action.disable()
        action.execute()
        assert executed[0] is False

    def test_action_render(self):
        action = MenuAction("test", "Test", callback=lambda: None)
        render_data = action.render()
        assert render_data["type"] == "action"
        assert render_data["label"] == "Test"


class TestMenuText:
    """Tests for MenuText class."""

    def test_text_creation(self):
        text = MenuText(item_id="info", text="Information")
        assert text.text == "Information"
        assert text.get_type() == MenuItemType.TEXT

    def test_text_update(self):
        text = MenuText("info", "Original")
        text.text = "Updated"
        assert text.text == "Updated"

    def test_text_render(self):
        text = MenuText("info", "Some text")
        render_data = text.render()
        assert render_data["type"] == "text"
        assert render_data["text"] == "Some text"


class TestMenuSeparator:
    """Tests for MenuSeparator class."""

    def test_separator_creation(self):
        sep = MenuSeparator()
        assert sep.get_type() == MenuItemType.SEPARATOR

    def test_separator_render(self):
        sep = MenuSeparator("sep1")
        render_data = sep.render()
        assert render_data["type"] == "separator"


class TestMenuDropdown:
    """Tests for MenuDropdown class."""

    def test_dropdown_creation(self):
        dropdown = MenuDropdown(
            item_id="quality",
            label="Quality",
            options=[("Low", 0), ("Medium", 1), ("High", 2)],
            selected_index=1,
        )
        assert dropdown.id == "quality"
        assert dropdown.selected_index == 1
        assert dropdown.selected_value == 1
        assert dropdown.selected_label == "Medium"
        assert dropdown.get_type() == MenuItemType.DROPDOWN

    def test_dropdown_select(self):
        dropdown = MenuDropdown(
            "test",
            "Test",
            options=[("A", "a"), ("B", "b"), ("C", "c")],
        )
        dropdown.selected_index = 2
        assert dropdown.selected_value == "c"
        assert dropdown.selected_label == "C"

    def test_dropdown_select_next_previous(self):
        dropdown = MenuDropdown(
            "test",
            "Test",
            options=[("A", 0), ("B", 1), ("C", 2)],
            selected_index=0,
        )
        dropdown.select_next()
        assert dropdown.selected_index == 1
        dropdown.select_previous()
        assert dropdown.selected_index == 0
        dropdown.select_previous()
        assert dropdown.selected_index == 2  # Wraps

    def test_dropdown_callback(self):
        values = []
        dropdown = MenuDropdown(
            "test",
            "Test",
            options=[("A", "a"), ("B", "b")],
            on_change=lambda v: values.append(v),
        )
        dropdown.selected_index = 1
        assert values == ["b"]

    def test_dropdown_render(self):
        dropdown = MenuDropdown(
            "test",
            "Test",
            options=[("A", 0), ("B", 1)],
            selected_index=0,
        )
        render_data = dropdown.render()
        assert render_data["type"] == "dropdown"
        assert render_data["options"] == ["A", "B"]
        assert render_data["selected_index"] == 0


class TestSubMenu:
    """Tests for SubMenu class."""

    def test_submenu_creation(self):
        submenu = SubMenu(
            item_id="rendering",
            label="Rendering",
        )
        assert submenu.id == "rendering"
        assert submenu.get_type() == MenuItemType.SUBMENU
        assert submenu.item_count == 0

    def test_submenu_add_item(self):
        submenu = SubMenu("test", "Test")
        toggle = MenuToggle("toggle", "Toggle", value=False)
        submenu.add_item(toggle)
        assert submenu.item_count == 1
        assert toggle.parent is submenu

    def test_submenu_remove_item(self):
        submenu = SubMenu("test", "Test")
        toggle = MenuToggle("toggle", "Toggle", value=False)
        submenu.add_item(toggle)
        removed = submenu.remove_item("toggle")
        assert removed is toggle
        assert submenu.item_count == 0
        assert toggle.parent is None

    def test_submenu_get_item(self):
        submenu = SubMenu("test", "Test")
        toggle = MenuToggle("toggle", "Toggle", value=False)
        submenu.add_item(toggle)
        found = submenu.get_item("toggle")
        assert found is toggle

    def test_submenu_get_item_nested(self):
        submenu = SubMenu("parent", "Parent")
        child = SubMenu("child", "Child")
        toggle = MenuToggle("nested_toggle", "Nested", value=False)
        child.add_item(toggle)
        submenu.add_item(child)

        found = submenu.get_item("nested_toggle")
        assert found is toggle

    def test_submenu_expand_collapse(self):
        submenu = SubMenu("test", "Test", expanded=False)
        assert not submenu.expanded
        submenu.expand()
        assert submenu.expanded
        submenu.collapse()
        assert not submenu.expanded
        submenu.toggle_expanded()
        assert submenu.expanded

    def test_submenu_clear(self):
        submenu = SubMenu("test", "Test")
        submenu.add_item(MenuToggle("t1", "T1", value=False))
        submenu.add_item(MenuToggle("t2", "T2", value=False))
        submenu.clear()
        assert submenu.item_count == 0

    def test_submenu_helper_methods(self):
        submenu = SubMenu("test", "Test")
        sep = submenu.add_separator()
        assert sep.get_type() == MenuItemType.SEPARATOR

        toggle = submenu.add_toggle("t", "Toggle")
        assert toggle.get_type() == MenuItemType.TOGGLE

        slider = submenu.add_slider("s", "Slider", 0, 0, 10)
        assert slider.get_type() == MenuItemType.SLIDER

        action = submenu.add_action("a", "Action", lambda: None)
        assert action.get_type() == MenuItemType.ACTION

        child = submenu.add_submenu("child", "Child")
        assert child.get_type() == MenuItemType.SUBMENU

    def test_submenu_render(self):
        submenu = SubMenu("test", "Test", expanded=True)
        submenu.add_toggle("t", "Toggle", value=True)

        render_data = submenu.render()
        assert render_data["type"] == "submenu"
        assert render_data["expanded"] is True
        assert len(render_data["items"]) == 1


class TestMenuItemBase:
    """Tests for base MenuItem functionality."""

    def test_item_enable_disable(self):
        toggle = MenuToggle("test", "Test", value=False)
        assert toggle.enabled is True
        toggle.disable()
        assert toggle.enabled is False
        toggle.enable()
        assert toggle.enabled is True

    def test_item_show_hide(self):
        toggle = MenuToggle("test", "Test", value=False)
        assert toggle.visible is True
        toggle.hide()
        assert toggle.visible is False
        toggle.show()
        assert toggle.visible is True

    def test_item_tooltip(self):
        toggle = MenuToggle("test", "Test", value=False, tooltip="Tooltip text")
        assert toggle.tooltip == "Tooltip text"
        toggle.tooltip = "New tooltip"
        assert toggle.tooltip == "New tooltip"

    def test_item_shortcut(self):
        action = MenuAction("test", "Test", callback=lambda: None, shortcut="Ctrl+T")
        assert action.shortcut == "Ctrl+T"

    def test_item_category(self):
        toggle = MenuToggle("test", "Test", value=False, category=MenuCategory.RENDERING)
        assert toggle.category == MenuCategory.RENDERING


class TestDebugMenu:
    """Tests for DebugMenu singleton."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        DebugMenu.reset_instance()
        yield
        DebugMenu.reset_instance()

    def test_singleton(self):
        m1 = DebugMenu.get_instance()
        m2 = DebugMenu.get_instance()
        assert m1 is m2

    def test_enable_disable(self):
        menu = DebugMenu.get_instance()
        menu.enable()
        assert menu.is_enabled
        menu.disable()
        assert not menu.is_enabled

    def test_show_hide(self):
        menu = DebugMenu.get_instance()
        menu.show()
        assert menu.is_visible
        menu.hide()
        assert not menu.is_visible

    def test_toggle(self):
        menu = DebugMenu.get_instance()
        menu.hide()
        result = menu.toggle()
        assert result is True
        result = menu.toggle()
        assert result is False

    def test_root_menu(self):
        menu = DebugMenu.get_instance()
        assert isinstance(menu.root, SubMenu)
        assert menu.root.id == "root"

    def test_default_menus(self):
        menu = DebugMenu.get_instance()
        # Should have default submenus
        rendering = menu.get_item("rendering")
        assert rendering is not None
        assert isinstance(rendering, SubMenu)

    def test_get_item(self):
        menu = DebugMenu.get_instance()
        # Get a default item
        wireframe = menu.get_item("wireframe")
        assert wireframe is not None

    def test_add_item(self):
        menu = DebugMenu.get_instance()
        toggle = MenuToggle("new_toggle", "New Toggle", value=False)
        result = menu.add_item(toggle)
        assert result is True
        assert menu.get_item("new_toggle") is toggle

    def test_add_item_to_submenu(self):
        menu = DebugMenu.get_instance()
        toggle = MenuToggle("render_toggle", "Render Toggle", value=False)
        result = menu.add_item(toggle, parent_id="rendering")
        assert result is True
        assert menu.get_item("render_toggle") is toggle

    def test_remove_item(self):
        menu = DebugMenu.get_instance()
        toggle = MenuToggle("temp", "Temp", value=False)
        menu.add_item(toggle)
        removed = menu.remove_item("temp")
        assert removed is toggle

    def test_style(self):
        menu = DebugMenu.get_instance()
        new_style = MenuStyle(font_size=20.0)
        menu.style = new_style
        assert menu.style.font_size == 20.0

    def test_register_shortcut(self):
        menu = DebugMenu.get_instance()
        toggle = MenuToggle("test_toggle", "Test", value=False)
        menu.add_item(toggle)
        menu.register_shortcut("Ctrl+T", "test_toggle")

        result = menu.handle_shortcut("Ctrl+T")
        assert result is True
        assert toggle.value is True  # Toggled

    def test_handle_shortcut_not_found(self):
        menu = DebugMenu.get_instance()
        result = menu.handle_shortcut("Ctrl+Unknown")
        assert result is False

    def test_select_item(self):
        menu = DebugMenu.get_instance()
        result = menu.select_item("wireframe")
        assert result is True
        assert menu.selected_item == "wireframe"

    def test_activate_selected_toggle(self):
        menu = DebugMenu.get_instance()
        toggle = menu.get_item("wireframe")
        original_value = toggle.value
        menu.select_item("wireframe")
        menu.activate_selected()
        assert toggle.value != original_value

    def test_render(self):
        menu = DebugMenu.get_instance()
        menu.show()
        render_data = menu.render()
        assert render_data["type"] == "debug_menu"
        assert render_data["visible"] is True
        assert "root" in render_data

    def test_render_hidden(self):
        menu = DebugMenu.get_instance()
        menu.hide()
        render_data = menu.render()
        assert render_data == {}

    def test_get_all_toggles(self):
        menu = DebugMenu.get_instance()
        toggles = menu.get_all_toggles()
        # Should have default toggles
        assert "wireframe" in toggles

    def test_set_toggle(self):
        menu = DebugMenu.get_instance()
        result = menu.set_toggle("wireframe", True)
        assert result is True
        toggle = menu.get_item("wireframe")
        assert toggle.value is True

    def test_set_slider(self):
        menu = DebugMenu.get_instance()
        # Add a slider first
        slider = MenuSlider("test_slider", "Test", value=0, min_value=0, max_value=100)
        menu.add_item(slider)

        result = menu.set_slider("test_slider", 50)
        assert result is True
        assert slider.value == 50
