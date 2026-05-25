"""
Comprehensive tests for InventorySlot widget.

Tests cover:
- Initialization and defaults
- Item management (set, clear, swap)
- Stack operations (add, remove, split)
- Cooldown system
- Visual states
- Input handling (click, hover)
- Drag and drop operations
- Drop validation and results
- Callbacks
- Rendering helpers
- Rarity colors
"""

import pytest
import sys
from pathlib import Path

# Add engine to path
engine_path = Path(__file__).parent.parent.parent.parent.parent / "engine"
sys.path.insert(0, str(engine_path.parent))

from engine.ui.widgets.game.inventory_slot import (
    InventorySlot,
    ItemData,
    ItemRarity,
    SlotState,
    DragPayload,
    DropResult,
)


class TestInventorySlotInitialization:
    """Test InventorySlot initialization and defaults."""

    def test_default_initialization(self):
        """Test slot initializes with correct defaults."""
        slot = InventorySlot()
        assert slot.item is None
        assert slot.is_empty is True
        assert slot.state == SlotState.NORMAL

    def test_custom_position(self):
        """Test initialization with custom position."""
        slot = InventorySlot(x=100.0, y=50.0)
        assert slot.x == 100.0
        assert slot.y == 50.0

    def test_custom_size(self):
        """Test initialization with custom size."""
        slot = InventorySlot(size=48.0)
        assert slot.size == 48.0
        assert slot.width == 48.0
        assert slot.height == 48.0

    def test_custom_slot_index(self):
        """Test initialization with slot index."""
        slot = InventorySlot(slot_index=5)
        assert slot.slot_index == 5

    def test_custom_container_id(self):
        """Test initialization with container ID."""
        slot = InventorySlot(container_id="player_inventory")
        assert slot.container_id == "player_inventory"

    def test_initialization_with_item(self):
        """Test initialization with item."""
        item = ItemData(id="sword", name="Iron Sword")
        slot = InventorySlot(item=item)
        assert slot.item is not None
        assert slot.item.name == "Iron Sword"

    def test_unique_id(self):
        """Test each slot gets unique ID."""
        slot1 = InventorySlot()
        slot2 = InventorySlot()
        assert slot1.id != slot2.id


class TestItemData:
    """Test ItemData class."""

    def test_item_creation(self):
        """Test creating item data."""
        item = ItemData(id="potion", name="Health Potion")
        assert item.id == "potion"
        assert item.name == "Health Potion"

    def test_item_default_stack(self):
        """Test default stack count."""
        item = ItemData(id="gold", name="Gold Coin")
        assert item.stack_count == 1

    def test_item_default_rarity(self):
        """Test default rarity."""
        item = ItemData(id="stick", name="Stick")
        assert item.rarity == ItemRarity.COMMON

    def test_item_can_stack_same_id(self):
        """Test stacking same item ID."""
        item1 = ItemData(id="coin", name="Coin", is_stackable=True)
        item2 = ItemData(id="coin", name="Coin", is_stackable=True)
        assert item1.can_stack_with(item2) is True

    def test_item_cannot_stack_different_id(self):
        """Test cannot stack different items."""
        item1 = ItemData(id="coin", name="Coin", is_stackable=True)
        item2 = ItemData(id="gem", name="Gem", is_stackable=True)
        assert item1.can_stack_with(item2) is False

    def test_item_cannot_stack_unstackable(self):
        """Test cannot stack unstackable items."""
        item1 = ItemData(id="sword", name="Sword", is_stackable=False)
        item2 = ItemData(id="sword", name="Sword", is_stackable=False)
        assert item1.can_stack_with(item2) is False

    def test_remaining_stack_space(self):
        """Test remaining stack space calculation."""
        item = ItemData(
            id="potion", name="Potion",
            stack_count=50, max_stack=99,
            is_stackable=True,
        )
        assert item.get_remaining_stack_space() == 49


class TestItemRarity:
    """Test ItemRarity enum."""

    def test_rarity_colors(self):
        """Test rarity colors are defined."""
        assert ItemRarity.COMMON.get_color() is not None
        assert ItemRarity.UNCOMMON.get_color() is not None
        assert ItemRarity.RARE.get_color() is not None
        assert ItemRarity.EPIC.get_color() is not None
        assert ItemRarity.LEGENDARY.get_color() is not None
        assert ItemRarity.MYTHIC.get_color() is not None

    def test_rarity_color_differences(self):
        """Test rarity colors are different."""
        colors = [r.get_color() for r in ItemRarity]
        # Not all should be the same
        assert len(set(colors)) > 1


class TestInventorySlotItemManagement:
    """Test InventorySlot item management."""

    def test_set_item(self):
        """Test setting item in slot."""
        slot = InventorySlot()
        item = ItemData(id="sword", name="Sword")
        old_item = slot.set_item(item)
        assert old_item is None
        assert slot.item is item
        assert slot.has_item is True

    def test_set_item_returns_old(self):
        """Test set_item returns old item."""
        slot = InventorySlot()
        item1 = ItemData(id="sword", name="Sword")
        item2 = ItemData(id="axe", name="Axe")
        slot.set_item(item1)
        old_item = slot.set_item(item2)
        assert old_item is item1

    def test_clear_item(self):
        """Test clearing item from slot."""
        slot = InventorySlot()
        item = ItemData(id="shield", name="Shield")
        slot.set_item(item)
        removed = slot.clear()
        assert removed is item
        assert slot.item is None
        assert slot.is_empty is True

    def test_swap_with(self):
        """Test swapping items between slots."""
        slot1 = InventorySlot()
        slot2 = InventorySlot()
        item1 = ItemData(id="sword", name="Sword")
        item2 = ItemData(id="shield", name="Shield")
        slot1.set_item(item1)
        slot2.set_item(item2)
        slot1.swap_with(slot2)
        assert slot1.item.id == "shield"
        assert slot2.item.id == "sword"


class TestInventorySlotStackOperations:
    """Test InventorySlot stack operations."""

    def test_add_to_stack(self):
        """Test adding to item stack."""
        item = ItemData(
            id="coin", name="Coin",
            stack_count=10, max_stack=99,
            is_stackable=True,
        )
        slot = InventorySlot(item=item)
        added = slot.add_to_stack(5)
        assert added == 5
        assert slot.item.stack_count == 15

    def test_add_to_stack_capped(self):
        """Test adding to stack is capped at max."""
        item = ItemData(
            id="coin", name="Coin",
            stack_count=95, max_stack=99,
            is_stackable=True,
        )
        slot = InventorySlot(item=item)
        added = slot.add_to_stack(10)
        assert added == 4  # Only 4 could fit
        assert slot.item.stack_count == 99

    def test_add_to_stack_empty_slot(self):
        """Test adding to stack with empty slot."""
        slot = InventorySlot()
        added = slot.add_to_stack(5)
        assert added == 0

    def test_add_to_stack_unstackable(self):
        """Test adding to stack with unstackable item."""
        item = ItemData(
            id="sword", name="Sword",
            is_stackable=False,
        )
        slot = InventorySlot(item=item)
        added = slot.add_to_stack(5)
        assert added == 0

    def test_remove_from_stack(self):
        """Test removing from item stack."""
        item = ItemData(
            id="coin", name="Coin",
            stack_count=20,
            is_stackable=True,
        )
        slot = InventorySlot(item=item)
        removed = slot.remove_from_stack(5)
        assert removed == 5
        assert slot.item.stack_count == 15

    def test_remove_from_stack_clears_slot(self):
        """Test removing all items clears slot."""
        item = ItemData(
            id="coin", name="Coin",
            stack_count=5,
            is_stackable=True,
        )
        slot = InventorySlot(item=item)
        removed = slot.remove_from_stack(5)
        assert removed == 5
        assert slot.is_empty is True

    def test_split_stack(self):
        """Test splitting stack."""
        item = ItemData(
            id="coin", name="Coin",
            stack_count=20, max_stack=99,
            is_stackable=True,
        )
        slot = InventorySlot(item=item)
        split = slot.split_stack(8)
        assert split is not None
        assert split.stack_count == 8
        assert slot.item.stack_count == 12

    def test_split_stack_cannot_split_all(self):
        """Test cannot split entire stack."""
        item = ItemData(
            id="coin", name="Coin",
            stack_count=10,
            is_stackable=True,
        )
        slot = InventorySlot(item=item)
        split = slot.split_stack(10)
        assert split is None

    def test_split_stack_unstackable(self):
        """Test cannot split unstackable item."""
        item = ItemData(id="sword", name="Sword", is_stackable=False)
        slot = InventorySlot(item=item)
        split = slot.split_stack(1)
        assert split is None


class TestInventorySlotCooldown:
    """Test InventorySlot cooldown system."""

    def test_start_cooldown(self):
        """Test starting cooldown."""
        slot = InventorySlot()
        slot.start_cooldown(5.0)
        assert slot.is_on_cooldown is True

    def test_cooldown_percent(self):
        """Test cooldown percentage."""
        slot = InventorySlot()
        slot.start_cooldown(10.0)
        slot.update(5.0)  # Half the cooldown
        assert slot.cooldown_percent == 0.5

    def test_cooldown_complete(self):
        """Test cooldown completion."""
        slot = InventorySlot()
        slot.start_cooldown(1.0)
        slot.update(2.0)  # Past cooldown
        assert slot.is_on_cooldown is False
        assert slot.cooldown_percent == 1.0

    def test_reset_cooldown(self):
        """Test resetting cooldown."""
        slot = InventorySlot()
        slot.start_cooldown(5.0)
        slot.reset_cooldown()
        assert slot.is_on_cooldown is False


class TestInventorySlotVisualStates:
    """Test InventorySlot visual states."""

    def test_normal_state(self):
        """Test normal state."""
        slot = InventorySlot()
        assert slot.state == SlotState.NORMAL

    def test_hovered_state(self):
        """Test hovered state."""
        slot = InventorySlot()
        slot.handle_hover_enter()
        assert slot.is_hovered is True
        assert slot.state == SlotState.HOVERED

    def test_selected_state(self):
        """Test selected state."""
        slot = InventorySlot()
        slot.is_selected = True
        assert slot.state == SlotState.SELECTED

    def test_disabled_state(self):
        """Test disabled state."""
        slot = InventorySlot()
        slot.is_enabled = False
        assert slot.state == SlotState.DISABLED

    def test_locked_state(self):
        """Test locked state."""
        slot = InventorySlot()
        slot.is_locked = True
        assert slot.state == SlotState.LOCKED


class TestInventorySlotInputHandling:
    """Test InventorySlot input handling."""

    def test_handle_hover_enter(self):
        """Test handling hover enter."""
        slot = InventorySlot()
        slot.handle_hover_enter()
        assert slot.is_hovered is True

    def test_handle_hover_exit(self):
        """Test handling hover exit."""
        slot = InventorySlot()
        slot.handle_hover_enter()
        slot.handle_hover_exit()
        assert slot.is_hovered is False

    def test_handle_click(self):
        """Test handling click."""
        slot = InventorySlot()
        result = slot.handle_click()
        assert result is True

    def test_handle_click_disabled(self):
        """Test click on disabled slot."""
        slot = InventorySlot()
        slot.is_enabled = False
        result = slot.handle_click()
        assert result is False

    def test_handle_click_locked(self):
        """Test click on locked slot."""
        slot = InventorySlot()
        slot.is_locked = True
        result = slot.handle_click()
        assert result is False

    def test_handle_right_click(self):
        """Test handling right click."""
        slot = InventorySlot()
        result = slot.handle_right_click()
        assert result is True

    def test_handle_double_click(self):
        """Test handling double click."""
        slot = InventorySlot()
        result = slot.handle_double_click()
        assert result is True


class TestInventorySlotDragAndDrop:
    """Test InventorySlot drag and drop."""

    def test_begin_drag(self):
        """Test beginning drag."""
        item = ItemData(id="sword", name="Sword")
        slot = InventorySlot(item=item)
        payload = slot.begin_drag(50.0, 50.0)
        assert payload is not None
        assert payload.item is item
        assert slot.is_dragging is True

    def test_begin_drag_empty_slot(self):
        """Test beginning drag on empty slot."""
        slot = InventorySlot()
        payload = slot.begin_drag(50.0, 50.0)
        assert payload is None

    def test_begin_drag_disabled(self):
        """Test beginning drag on disabled slot."""
        item = ItemData(id="sword", name="Sword")
        slot = InventorySlot(item=item)
        slot.is_enabled = False
        payload = slot.begin_drag(50.0, 50.0)
        assert payload is None

    def test_begin_drag_locked(self):
        """Test beginning drag on locked slot."""
        item = ItemData(id="sword", name="Sword")
        slot = InventorySlot(item=item)
        slot.is_locked = True
        payload = slot.begin_drag(50.0, 50.0)
        assert payload is None

    def test_begin_drag_partial_stack(self):
        """Test beginning drag with partial stack."""
        item = ItemData(
            id="coin", name="Coin",
            stack_count=20,
            is_stackable=True,
        )
        slot = InventorySlot(item=item)
        payload = slot.begin_drag(50.0, 50.0, count=10)
        assert payload is not None
        assert payload.drag_count == 10

    def test_end_drag_success(self):
        """Test ending successful drag."""
        item = ItemData(id="sword", name="Sword")
        slot = InventorySlot(item=item)
        slot.begin_drag(50.0, 50.0)
        slot.end_drag(success=True)
        assert slot.is_dragging is False
        assert slot.is_empty is True  # Item was removed

    def test_end_drag_failure(self):
        """Test ending failed drag."""
        item = ItemData(id="sword", name="Sword")
        slot = InventorySlot(item=item)
        slot.begin_drag(50.0, 50.0)
        slot.end_drag(success=False)
        assert slot.is_dragging is False
        assert slot.has_item is True  # Item remains


class TestInventorySlotDropValidation:
    """Test InventorySlot drop validation."""

    def test_can_accept_empty_slot(self):
        """Test empty slot can accept drop."""
        source = InventorySlot(item=ItemData(id="sword", name="Sword"))
        target = InventorySlot()
        payload = source.begin_drag(50.0, 50.0)
        assert target.can_accept(payload) is True

    def test_can_accept_same_slot(self):
        """Test cannot drop on same slot."""
        slot = InventorySlot(item=ItemData(id="sword", name="Sword"))
        payload = slot.begin_drag(50.0, 50.0)
        assert slot.can_accept(payload) is False

    def test_can_accept_stackable(self):
        """Test can accept stackable items."""
        item1 = ItemData(id="coin", name="Coin", stack_count=50, is_stackable=True)
        item2 = ItemData(id="coin", name="Coin", stack_count=20, is_stackable=True)
        source = InventorySlot(item=item2)
        target = InventorySlot(item=item1)
        payload = source.begin_drag(50.0, 50.0)
        assert target.can_accept(payload) is True

    def test_can_accept_disabled(self):
        """Test disabled slot cannot accept."""
        source = InventorySlot(item=ItemData(id="sword", name="Sword"))
        target = InventorySlot()
        target.is_enabled = False
        payload = source.begin_drag(50.0, 50.0)
        assert target.can_accept(payload) is False


class TestInventorySlotDropResults:
    """Test InventorySlot drop operations."""

    def test_handle_drop_success(self):
        """Test successful drop on empty slot."""
        item = ItemData(id="sword", name="Sword")
        source = InventorySlot(item=item)
        target = InventorySlot()
        payload = source.begin_drag(50.0, 50.0)
        result = target.handle_drop(payload)
        assert result == DropResult.SUCCESS
        assert target.has_item is True

    def test_handle_drop_stacked(self):
        """Test drop that stacks items."""
        item1 = ItemData(id="coin", name="Coin", stack_count=50, max_stack=99, is_stackable=True)
        item2 = ItemData(id="coin", name="Coin", stack_count=20, max_stack=99, is_stackable=True)
        source = InventorySlot(item=item2)
        target = InventorySlot(item=item1)
        payload = source.begin_drag(50.0, 50.0)
        result = target.handle_drop(payload)
        assert result == DropResult.STACKED

    def test_handle_drop_swapped(self):
        """Test drop that swaps items."""
        item1 = ItemData(id="sword", name="Sword")
        item2 = ItemData(id="axe", name="Axe")
        source = InventorySlot(item=item1)
        target = InventorySlot(item=item2)
        payload = source.begin_drag(50.0, 50.0)
        result = target.handle_drop(payload)
        assert result == DropResult.SWAPPED

    def test_handle_drop_invalid(self):
        """Test invalid drop."""
        source = InventorySlot(item=ItemData(id="sword", name="Sword"))
        target = InventorySlot()
        target.is_enabled = False
        payload = source.begin_drag(50.0, 50.0)
        result = target.handle_drop(payload)
        assert result == DropResult.INVALID_TARGET


class TestInventorySlotCallbacks:
    """Test InventorySlot callbacks."""

    def test_on_click_callback(self):
        """Test click callback."""
        slot = InventorySlot()
        clicks = []

        def callback(s):
            clicks.append(s)

        slot.on_click(callback)
        slot.handle_click()
        assert len(clicks) == 1

    def test_on_hover_start_callback(self):
        """Test hover start callback."""
        slot = InventorySlot()
        hovers = []

        def callback(s):
            hovers.append(s)

        slot.on_hover_start(callback)
        slot.handle_hover_enter()
        assert len(hovers) == 1

    def test_on_hover_end_callback(self):
        """Test hover end callback."""
        slot = InventorySlot()
        ends = []

        def callback(s):
            ends.append(s)

        slot.on_hover_end(callback)
        slot.handle_hover_enter()
        slot.handle_hover_exit()
        assert len(ends) == 1

    def test_on_drag_start_callback(self):
        """Test drag start callback."""
        slot = InventorySlot(item=ItemData(id="sword", name="Sword"))
        drags = []

        def callback(payload):
            drags.append(payload)
            return True  # Allow drag

        slot.on_drag_start(callback)
        slot.begin_drag(50.0, 50.0)
        assert len(drags) == 1

    def test_on_drag_start_cancel(self):
        """Test drag start callback can cancel."""
        slot = InventorySlot(item=ItemData(id="sword", name="Sword"))

        def callback(payload):
            return False  # Cancel drag

        slot.on_drag_start(callback)
        payload = slot.begin_drag(50.0, 50.0)
        assert payload is None


class TestInventorySlotTooltip:
    """Test InventorySlot tooltip behavior."""

    def test_tooltip_delay(self):
        """Test tooltip delay."""
        slot = InventorySlot(item=ItemData(id="sword", name="Sword"))
        slot.handle_hover_enter()
        assert slot.should_show_tooltip is False  # Not yet
        slot.update(1.0)  # Past delay
        assert slot.should_show_tooltip is True

    def test_tooltip_hides_on_exit(self):
        """Test tooltip hides on hover exit."""
        slot = InventorySlot(item=ItemData(id="sword", name="Sword"))
        slot.handle_hover_enter()
        slot.update(1.0)
        slot.handle_hover_exit()
        assert slot.should_show_tooltip is False

    def test_tooltip_empty_slot(self):
        """Test no tooltip for empty slot."""
        slot = InventorySlot()
        slot.handle_hover_enter()
        slot.update(1.0)
        assert slot.should_show_tooltip is False


class TestInventorySlotRenderingHelpers:
    """Test InventorySlot rendering helpers."""

    def test_get_icon_rect(self):
        """Test get_icon_rect method."""
        slot = InventorySlot(x=0.0, y=0.0, size=64.0)
        rect = slot.get_icon_rect()
        assert len(rect) == 4
        x, y, w, h = rect
        assert w < 64.0  # Accounts for padding

    def test_get_cooldown_overlay_rect(self):
        """Test get_cooldown_overlay_rect method."""
        slot = InventorySlot()
        slot.start_cooldown(2.0)
        slot.update(1.0)  # 50% cooldown
        rect = slot.get_cooldown_overlay_rect()
        assert rect is not None

    def test_get_cooldown_overlay_no_cooldown(self):
        """Test no cooldown overlay when not on cooldown."""
        slot = InventorySlot()
        rect = slot.get_cooldown_overlay_rect()
        assert rect is None

    def test_get_stack_count_position(self):
        """Test get_stack_count_position method."""
        slot = InventorySlot(size=64.0)
        pos = slot.get_stack_count_position()
        assert len(pos) == 2

    def test_point_in_bounds(self):
        """Test point_in_bounds method."""
        slot = InventorySlot(x=0.0, y=0.0, size=64.0)
        assert slot.point_in_bounds(32.0, 32.0) is True
        assert slot.point_in_bounds(100.0, 100.0) is False

    def test_border_color_with_rarity(self):
        """Test border color reflects rarity."""
        item = ItemData(id="sword", name="Legendary Sword", rarity=ItemRarity.LEGENDARY)
        slot = InventorySlot(item=item)
        assert slot.border_color == ItemRarity.LEGENDARY.get_color()


class TestInventorySlotRepr:
    """Test InventorySlot string representation."""

    def test_repr_empty(self):
        """Test repr for empty slot."""
        slot = InventorySlot(slot_index=3)
        repr_str = repr(slot)
        assert "InventorySlot" in repr_str
        assert "empty" in repr_str

    def test_repr_with_item(self):
        """Test repr with item."""
        slot = InventorySlot(item=ItemData(id="sword", name="Sword"))
        repr_str = repr(slot)
        assert "Sword" in repr_str
