"""
Tests for Wrist UI (wrist_ui.py).

Tests the WristUI component and related classes:
    WristUI, WristUIPosition, WristMenuItem, WristUIManager

Each test verifies:
1. Wrist UI creation and configuration
2. Menu item management
3. Visibility modes
4. Layout calculations
"""

import sys
from pathlib import Path

import pytest
import math

# Add engine to path for direct imports
engine_path = Path(__file__).parents[3]
if str(engine_path) not in sys.path:
    sys.path.insert(0, str(engine_path))

from engine.xr.ui.wrist_ui import (
    WristUI,
    WristUIPosition,
    WristUILayout,
    WristUIVisibilityMode,
    WristMenuItem,
    WristUIStyle,
    WristUIManager,
)


# =============================================================================
# WristMenuItem
# =============================================================================


class TestWristMenuItem:
    def test_default_creation(self):
        item = WristMenuItem(id="test")
        assert item.id == "test"
        assert item.label == ""
        assert item.icon == ""
        assert item.is_enabled is True
        assert item.badge_count == 0

    def test_with_all_params(self):
        item = WristMenuItem(
            id="settings",
            label="Settings",
            icon="gear",
            is_enabled=True,
            badge_count=5,
            color=(1.0, 0.0, 0.0, 1.0),
        )
        assert item.id == "settings"
        assert item.label == "Settings"
        assert item.icon == "gear"
        assert item.badge_count == 5
        assert item.color == (1.0, 0.0, 0.0, 1.0)

    def test_execute_with_action(self):
        executed = False

        def action():
            nonlocal executed
            executed = True

        item = WristMenuItem(id="test", action=action)
        result = item.execute()

        assert result is True
        assert executed is True

    def test_execute_without_action(self):
        item = WristMenuItem(id="test")
        result = item.execute()
        assert result is False

    def test_execute_when_disabled(self):
        executed = False

        def action():
            nonlocal executed
            executed = True

        item = WristMenuItem(id="test", action=action, is_enabled=False)
        result = item.execute()

        assert result is False
        assert executed is False


# =============================================================================
# WristUIStyle
# =============================================================================


class TestWristUIStyle:
    def test_default_values(self):
        style = WristUIStyle()
        assert len(style.background_color) == 4
        assert len(style.item_color) == 4
        assert style.item_size > 0
        assert style.item_spacing >= 0

    def test_custom_values(self):
        style = WristUIStyle(
            background_color=(0.5, 0.5, 0.5, 0.8),
            item_size=0.03,
        )
        assert style.background_color == (0.5, 0.5, 0.5, 0.8)
        assert style.item_size == 0.03


# =============================================================================
# WristUI
# =============================================================================


class TestWristUI:
    def test_default_creation(self):
        ui = WristUI()
        assert ui.position == WristUIPosition.LEFT
        assert ui.layout == WristUILayout.CIRCULAR
        assert ui.visibility_mode == WristUIVisibilityMode.LOOK_AT
        assert ui.is_visible is False
        assert ui.is_active is True

    def test_left_wrist(self):
        ui = WristUI(position=WristUIPosition.LEFT)
        assert ui.position == WristUIPosition.LEFT

    def test_right_wrist(self):
        ui = WristUI(position=WristUIPosition.RIGHT)
        assert ui.position == WristUIPosition.RIGHT

    def test_all_layouts(self):
        for layout in WristUILayout:
            ui = WristUI(layout=layout)
            assert ui.layout == layout

    def test_all_visibility_modes(self):
        for mode in WristUIVisibilityMode:
            ui = WristUI(visibility_mode=mode)
            assert ui.visibility_mode == mode

    def test_custom_size(self):
        ui = WristUI(size=0.1)
        assert ui.size == 0.1

    def test_add_item(self):
        ui = WristUI()
        item = WristMenuItem(id="test", label="Test")

        result = ui.add_item(item)

        assert result is True
        assert ui.item_count == 1
        assert item._index == 0

    def test_add_too_many_items(self):
        ui = WristUI(layout=WristUILayout.CIRCULAR)  # max 8 items

        # Add max items
        for i in range(ui.max_items):
            result = ui.add_item(WristMenuItem(id=f"item{i}"))
            assert result is True

        # Try to add one more
        result = ui.add_item(WristMenuItem(id="overflow"))
        assert result is False

    def test_remove_item(self):
        ui = WristUI()
        ui.add_item(WristMenuItem(id="test"))

        result = ui.remove_item("test")

        assert result is True
        assert ui.item_count == 0

    def test_remove_nonexistent_item(self):
        ui = WristUI()
        result = ui.remove_item("nonexistent")
        assert result is False

    def test_get_item(self):
        ui = WristUI()
        item = WristMenuItem(id="test", label="Test")
        ui.add_item(item)

        retrieved = ui.get_item("test")
        assert retrieved == item

    def test_get_nonexistent_item(self):
        ui = WristUI()
        retrieved = ui.get_item("nonexistent")
        assert retrieved is None

    def test_clear_items(self):
        ui = WristUI()
        ui.add_item(WristMenuItem(id="item1"))
        ui.add_item(WristMenuItem(id="item2"))

        ui.clear_items()

        assert ui.item_count == 0

    def test_show_hide(self):
        ui = WristUI()

        ui.show()
        assert ui.is_visible is True

        ui.hide()
        assert ui.is_visible is False

    def test_toggle(self):
        ui = WristUI()

        ui.toggle()
        assert ui.is_visible is True

        ui.toggle()
        assert ui.is_visible is False

    def test_activate_deactivate(self):
        ui = WristUI()

        ui.deactivate()
        assert ui.is_active is False

        ui.activate()
        assert ui.is_active is True

    def test_hover_item(self):
        ui = WristUI()
        item = WristMenuItem(id="test")
        ui.add_item(item)

        hovered = ui.hover_item(0)

        assert hovered == item
        assert item.is_hovered is True

    def test_hover_item_clears_previous(self):
        ui = WristUI()
        item1 = WristMenuItem(id="test1")
        item2 = WristMenuItem(id="test2")
        ui.add_item(item1)
        ui.add_item(item2)

        ui.hover_item(0)
        ui.hover_item(1)

        assert item1.is_hovered is False
        assert item2.is_hovered is True

    def test_hover_disabled_item(self):
        ui = WristUI()
        item = WristMenuItem(id="test", is_enabled=False)
        ui.add_item(item)

        hovered = ui.hover_item(0)

        assert hovered is None
        assert item.is_hovered is False

    def test_select_hovered(self):
        ui = WristUI()
        executed = False

        def action():
            nonlocal executed
            executed = True

        item = WristMenuItem(id="test", action=action)
        ui.add_item(item)
        ui.hover_item(0)

        result = ui.select_hovered()

        assert result is True
        assert executed is True

    def test_select_item_by_id(self):
        ui = WristUI()
        executed = False

        def action():
            nonlocal executed
            executed = True

        ui.add_item(WristMenuItem(id="test", action=action))

        result = ui.select_item("test")

        assert result is True
        assert executed is True

    def test_visibility_callback(self):
        ui = WristUI()
        visibility_changes = []

        def on_change(visible):
            visibility_changes.append(visible)

        ui.on_visibility_changed(on_change)

        ui.show()
        ui.hide()

        assert visibility_changes == [True, False]

    def test_item_selected_callback(self):
        ui = WristUI()
        selected_items = []

        def on_select(item):
            selected_items.append(item.id)

        ui.on_item_selected(on_select)
        ui.add_item(WristMenuItem(id="test", action=lambda: None))
        ui.select_item("test")

        assert selected_items == ["test"]

    def test_update_always_visible(self):
        ui = WristUI(visibility_mode=WristUIVisibilityMode.ALWAYS)
        ui.is_active = True

        ui.update(0.016)

        assert ui.is_visible is True

    def test_update_manual_visibility(self):
        ui = WristUI(visibility_mode=WristUIVisibilityMode.MANUAL)
        ui.is_active = True

        ui.update(0.016)
        assert ui.is_visible is False  # Should not auto-show

        ui.show()
        ui.update(0.016)
        assert ui.is_visible is True  # Should stay visible

    def test_update_wrist_tracking(self):
        ui = WristUI()
        ui.update_wrist_tracking(
            position=(0.0, 1.0, -0.3),
            orientation=(0.0, 0.0, 0.0, 1.0)
        )

        assert ui._wrist_position == (0.0, 1.0, -0.3)
        assert ui._wrist_orientation == (0.0, 0.0, 0.0, 1.0)

    def test_update_head_tracking(self):
        ui = WristUI()
        ui.update_head_tracking(
            position=(0.0, 1.7, 0.0),
            forward=(0.0, 0.0, -1.0)
        )

        assert ui._head_position == (0.0, 1.7, 0.0)
        assert ui._head_forward == (0.0, 0.0, -1.0)

    def test_get_item_position_circular(self):
        ui = WristUI(layout=WristUILayout.CIRCULAR, size=0.08)
        ui.add_item(WristMenuItem(id="test"))

        x, y = ui.get_item_position(0)

        # First item position depends on angle calculation
        # With single item at index 0, angle = -pi/2 (pointing down by default)
        # The position should be non-zero (on the circle edge)
        radius = ui.size / 2 - ui.style.item_size / 2
        distance = (x * x + y * y) ** 0.5
        assert abs(distance - radius) < 0.01  # Should be on the circle edge

    def test_get_item_position_rectangular(self):
        ui = WristUI(layout=WristUILayout.RECTANGULAR)
        for i in range(4):
            ui.add_item(WristMenuItem(id=f"item{i}"))

        # Check that items have different positions
        positions = [ui.get_item_position(i) for i in range(4)]

        # All positions should be unique
        unique_positions = set(positions)
        assert len(unique_positions) == 4

        # First and second items should have different x or y
        x0, y0 = positions[0]
        x1, y1 = positions[1]
        assert x0 != x1 or y0 != y1

    def test_max_items_by_layout(self):
        circular = WristUI(layout=WristUILayout.CIRCULAR)
        assert circular.max_items == 8

        radial = WristUI(layout=WristUILayout.RADIAL)
        assert radial.max_items == 8

        rectangular = WristUI(layout=WristUILayout.RECTANGULAR)
        assert rectangular.max_items == 12


# =============================================================================
# WristUIManager
# =============================================================================


class TestWristUIManager:
    def test_creation(self):
        manager = WristUIManager()
        assert manager.left is None
        assert manager.right is None

    def test_create_left(self):
        manager = WristUIManager()
        ui = manager.create_left()

        assert ui.position == WristUIPosition.LEFT
        assert manager.left == ui

    def test_create_right(self):
        manager = WristUIManager()
        ui = manager.create_right()

        assert ui.position == WristUIPosition.RIGHT
        assert manager.right == ui

    def test_get_by_position(self):
        manager = WristUIManager()
        left = manager.create_left()
        right = manager.create_right()

        assert manager.get(WristUIPosition.LEFT) == left
        assert manager.get(WristUIPosition.RIGHT) == right

    def test_update(self):
        manager = WristUIManager()
        manager.create_left()
        manager.create_right()

        # Should not raise
        manager.update(
            delta_time=0.016,
            left_wrist_pos=(0.0, 1.0, 0.0),
            left_wrist_rot=(0.0, 0.0, 0.0, 1.0),
            right_wrist_pos=(0.0, 1.0, 0.0),
            right_wrist_rot=(0.0, 0.0, 0.0, 1.0),
            head_pos=(0.0, 1.7, 0.0),
            head_forward=(0.0, 0.0, -1.0),
        )

    def test_show_hide_all(self):
        manager = WristUIManager()
        left = manager.create_left()
        right = manager.create_right()

        manager.show_all()
        assert left.is_visible is True
        assert right.is_visible is True

        manager.hide_all()
        assert left.is_visible is False
        assert right.is_visible is False
