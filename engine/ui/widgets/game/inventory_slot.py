"""
Inventory slot widget with drag-and-drop support.

Provides an inventory slot widget supporting:
- Item icon display
- Stack count
- Rarity border coloring
- Drag and drop interactions
- Tooltip on hover
- Cooldown overlay
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Any


class ItemRarity(Enum):
    """Item rarity levels."""
    COMMON = auto()
    UNCOMMON = auto()
    RARE = auto()
    EPIC = auto()
    LEGENDARY = auto()
    MYTHIC = auto()

    def get_color(self) -> str:
        """Get the color associated with this rarity."""
        colors = {
            ItemRarity.COMMON: "#9ca3af",      # Gray
            ItemRarity.UNCOMMON: "#22c55e",    # Green
            ItemRarity.RARE: "#3b82f6",        # Blue
            ItemRarity.EPIC: "#a855f7",        # Purple
            ItemRarity.LEGENDARY: "#f59e0b",   # Orange
            ItemRarity.MYTHIC: "#ef4444",      # Red
        }
        return colors.get(self, "#9ca3af")


class SlotState(Enum):
    """Visual states for inventory slot."""
    NORMAL = auto()
    HOVERED = auto()
    SELECTED = auto()
    DRAGGING = auto()
    DROP_TARGET = auto()
    DROP_INVALID = auto()
    DISABLED = auto()
    LOCKED = auto()


@dataclass(slots=True)
class ItemData:
    """Data for an item in a slot."""
    id: str
    name: str
    icon: Optional[str] = None  # Icon path or identifier
    rarity: ItemRarity = ItemRarity.COMMON
    stack_count: int = 1
    max_stack: int = 99
    is_stackable: bool = True
    is_usable: bool = False
    is_equippable: bool = False
    is_consumable: bool = False
    cooldown_duration: float = 0.0
    tooltip_title: Optional[str] = None
    tooltip_description: Optional[str] = None
    tooltip_stats: dict = field(default_factory=dict)
    custom_data: dict = field(default_factory=dict)

    def can_stack_with(self, other: "ItemData") -> bool:
        """Check if this item can stack with another."""
        if not self.is_stackable or not other.is_stackable:
            return False
        return self.id == other.id

    def get_remaining_stack_space(self) -> int:
        """Get remaining stack space."""
        return max(0, self.max_stack - self.stack_count)


@dataclass(slots=True)
class DragPayload:
    """Payload for drag operations."""
    source_slot: "InventorySlot"
    item: ItemData
    drag_count: int  # Number of items being dragged
    original_position: tuple[float, float]
    offset_x: float = 0.0
    offset_y: float = 0.0


class DropResult(Enum):
    """Result of a drop operation."""
    SUCCESS = auto()
    SWAPPED = auto()
    STACKED = auto()
    PARTIAL_STACK = auto()
    REJECTED = auto()
    INVALID_TARGET = auto()


class InventorySlot:
    """Inventory slot widget with drag-and-drop support.

    Features:
    - Item icon display with optional stack count
    - Rarity-colored border
    - Full drag and drop support
    - Tooltip integration
    - Cooldown overlay animation
    - Multiple visual states
    """

    __slots__ = (
        '_id', '_x', '_y', '_size',
        '_item', '_state',
        '_slot_index', '_container_id',
        '_is_visible', '_is_enabled', '_is_locked',
        '_is_hovered', '_is_selected', '_is_dragging',
        '_cooldown_remaining', '_cooldown_total',
        '_background_color', '_border_color', '_border_width',
        '_corner_radius', '_icon_padding',
        '_show_stack_count', '_show_rarity_border',
        '_on_click', '_on_right_click', '_on_double_click',
        '_on_hover_start', '_on_hover_end',
        '_on_drag_start', '_on_drag_end', '_on_drop',
        '_can_accept_drop',
        '_drag_payload', '_drag_preview_offset',
        '_parent', '_children',
        '_tooltip_delay', '_tooltip_timer', '_show_tooltip',
    )

    _next_id: int = 0

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        size: float = 64.0,
        slot_index: int = 0,
        container_id: Optional[str] = None,
        item: Optional[ItemData] = None,
    ):
        """Initialize the inventory slot.

        Args:
            x: X position
            y: Y position
            size: Slot size (square)
            slot_index: Index within container
            container_id: Parent container identifier
            item: Initial item data
        """
        self._id = InventorySlot._next_id
        InventorySlot._next_id += 1

        self._x = x
        self._y = y
        self._size = max(1.0, size)

        self._item = item
        self._state = SlotState.NORMAL

        self._slot_index = slot_index
        self._container_id = container_id

        self._is_visible = True
        self._is_enabled = True
        self._is_locked = False
        self._is_hovered = False
        self._is_selected = False
        self._is_dragging = False

        self._cooldown_remaining = 0.0
        self._cooldown_total = 0.0

        # Styling
        self._background_color = "#374151"
        self._border_color = "#4b5563"
        self._border_width = 2.0
        self._corner_radius = 4.0
        self._icon_padding = 4.0

        self._show_stack_count = True
        self._show_rarity_border = True

        # Callbacks
        self._on_click: Optional[Callable[["InventorySlot"], None]] = None
        self._on_right_click: Optional[Callable[["InventorySlot"], None]] = None
        self._on_double_click: Optional[Callable[["InventorySlot"], None]] = None
        self._on_hover_start: Optional[Callable[["InventorySlot"], None]] = None
        self._on_hover_end: Optional[Callable[["InventorySlot"], None]] = None
        self._on_drag_start: Optional[Callable[[DragPayload], bool]] = None
        self._on_drag_end: Optional[Callable[[DragPayload, bool], None]] = None
        self._on_drop: Optional[Callable[["InventorySlot", DragPayload], DropResult]] = None
        self._can_accept_drop: Optional[Callable[[DragPayload], bool]] = None

        self._drag_payload: Optional[DragPayload] = None
        self._drag_preview_offset = (0.0, 0.0)

        self._parent = None
        self._children: list = []

        self._tooltip_delay = 0.5
        self._tooltip_timer = 0.0
        self._show_tooltip = False

    # Properties
    @property
    def id(self) -> int:
        """Get widget ID."""
        return self._id

    @property
    def x(self) -> float:
        """Get X position."""
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        """Set X position."""
        self._x = value

    @property
    def y(self) -> float:
        """Get Y position."""
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        """Set Y position."""
        self._y = value

    @property
    def size(self) -> float:
        """Get slot size."""
        return self._size

    @size.setter
    def size(self, value: float) -> None:
        """Set slot size."""
        self._size = max(1.0, value)

    @property
    def width(self) -> float:
        """Get width (alias for size)."""
        return self._size

    @property
    def height(self) -> float:
        """Get height (alias for size)."""
        return self._size

    @property
    def item(self) -> Optional[ItemData]:
        """Get item in slot."""
        return self._item

    @property
    def has_item(self) -> bool:
        """Check if slot has an item."""
        return self._item is not None

    @property
    def is_empty(self) -> bool:
        """Check if slot is empty."""
        return self._item is None

    @property
    def state(self) -> SlotState:
        """Get current visual state."""
        return self._state

    @property
    def slot_index(self) -> int:
        """Get slot index."""
        return self._slot_index

    @property
    def container_id(self) -> Optional[str]:
        """Get container ID."""
        return self._container_id

    @property
    def is_visible(self) -> bool:
        """Check if slot is visible."""
        return self._is_visible

    @is_visible.setter
    def is_visible(self, value: bool) -> None:
        """Set visibility."""
        self._is_visible = value

    @property
    def is_enabled(self) -> bool:
        """Check if slot is enabled."""
        return self._is_enabled

    @is_enabled.setter
    def is_enabled(self, value: bool) -> None:
        """Set enabled state."""
        self._is_enabled = value
        self._update_state()

    @property
    def is_locked(self) -> bool:
        """Check if slot is locked."""
        return self._is_locked

    @is_locked.setter
    def is_locked(self, value: bool) -> None:
        """Set locked state."""
        self._is_locked = value
        self._update_state()

    @property
    def is_hovered(self) -> bool:
        """Check if slot is hovered."""
        return self._is_hovered

    @property
    def is_selected(self) -> bool:
        """Check if slot is selected."""
        return self._is_selected

    @is_selected.setter
    def is_selected(self, value: bool) -> None:
        """Set selected state."""
        self._is_selected = value
        self._update_state()

    @property
    def is_dragging(self) -> bool:
        """Check if slot is being dragged from."""
        return self._is_dragging

    @property
    def is_on_cooldown(self) -> bool:
        """Check if item is on cooldown."""
        return self._cooldown_remaining > 0

    @property
    def cooldown_percent(self) -> float:
        """Get cooldown progress (0.0-1.0, 1.0 = ready)."""
        if self._cooldown_total <= 0:
            return 1.0
        return 1.0 - (self._cooldown_remaining / self._cooldown_total)

    @property
    def should_show_tooltip(self) -> bool:
        """Check if tooltip should be shown."""
        return self._show_tooltip and self._item is not None

    @property
    def border_color(self) -> str:
        """Get effective border color."""
        if self._show_rarity_border and self._item:
            return self._item.rarity.get_color()
        return self._border_color

    # Item management
    def set_item(self, item: Optional[ItemData]) -> Optional[ItemData]:
        """Set item in slot.

        Args:
            item: Item to place

        Returns:
            Previous item if any
        """
        old_item = self._item
        self._item = item

        # Reset cooldown when item changes
        if item is None or (old_item and item.id != old_item.id):
            self._cooldown_remaining = 0.0
            self._cooldown_total = 0.0

        return old_item

    def clear(self) -> Optional[ItemData]:
        """Clear item from slot.

        Returns:
            Removed item if any
        """
        return self.set_item(None)

    def swap_with(self, other: "InventorySlot") -> None:
        """Swap items with another slot.

        Args:
            other: Other slot to swap with
        """
        self._item, other._item = other._item, self._item

    def add_to_stack(self, count: int) -> int:
        """Add to item stack.

        Args:
            count: Amount to add

        Returns:
            Amount actually added
        """
        if not self._item or not self._item.is_stackable:
            return 0

        space = self._item.get_remaining_stack_space()
        added = min(count, space)
        self._item.stack_count += added
        return added

    def remove_from_stack(self, count: int) -> int:
        """Remove from item stack.

        Args:
            count: Amount to remove

        Returns:
            Amount actually removed
        """
        if not self._item:
            return 0

        removed = min(count, self._item.stack_count)
        self._item.stack_count -= removed

        if self._item.stack_count <= 0:
            self.clear()

        return removed

    def split_stack(self, count: int) -> Optional[ItemData]:
        """Split stack and return split portion.

        Args:
            count: Amount to split off

        Returns:
            New item with split amount, or None
        """
        if not self._item or not self._item.is_stackable:
            return None

        if count >= self._item.stack_count:
            return None  # Can't split all items

        # Create new item with split count
        split_item = ItemData(
            id=self._item.id,
            name=self._item.name,
            icon=self._item.icon,
            rarity=self._item.rarity,
            stack_count=count,
            max_stack=self._item.max_stack,
            is_stackable=True,
            tooltip_title=self._item.tooltip_title,
            tooltip_description=self._item.tooltip_description,
            tooltip_stats=self._item.tooltip_stats.copy(),
            custom_data=self._item.custom_data.copy(),
        )

        self._item.stack_count -= count
        return split_item

    # Cooldown
    def start_cooldown(self, duration: float) -> None:
        """Start item cooldown.

        Args:
            duration: Cooldown duration in seconds
        """
        self._cooldown_total = max(0.0, duration)
        self._cooldown_remaining = self._cooldown_total

    def reset_cooldown(self) -> None:
        """Reset cooldown to zero."""
        self._cooldown_remaining = 0.0
        self._cooldown_total = 0.0

    # Input handling
    def handle_hover_enter(self) -> None:
        """Handle mouse entering slot."""
        if not self._is_enabled:
            return

        self._is_hovered = True
        self._tooltip_timer = 0.0
        self._update_state()

        if self._on_hover_start:
            self._on_hover_start(self)

    def handle_hover_exit(self) -> None:
        """Handle mouse leaving slot."""
        self._is_hovered = False
        self._tooltip_timer = 0.0
        self._show_tooltip = False
        self._update_state()

        if self._on_hover_end:
            self._on_hover_end(self)

    def handle_click(self) -> bool:
        """Handle click on slot.

        Returns:
            True if click was handled
        """
        if not self._is_enabled or self._is_locked:
            return False

        if self._on_click:
            self._on_click(self)

        return True

    def handle_right_click(self) -> bool:
        """Handle right-click on slot.

        Returns:
            True if click was handled
        """
        if not self._is_enabled or self._is_locked:
            return False

        if self._on_right_click:
            self._on_right_click(self)

        return True

    def handle_double_click(self) -> bool:
        """Handle double-click on slot.

        Returns:
            True if click was handled
        """
        if not self._is_enabled or self._is_locked:
            return False

        if self._on_double_click:
            self._on_double_click(self)

        return True

    # Drag and drop
    def begin_drag(self, mouse_x: float, mouse_y: float, count: Optional[int] = None) -> Optional[DragPayload]:
        """Begin dragging from this slot.

        Args:
            mouse_x: Mouse X position
            mouse_y: Mouse Y position
            count: Number of items to drag (None = all)

        Returns:
            Drag payload if drag started
        """
        if not self._item or not self._is_enabled or self._is_locked:
            return None

        drag_count = count if count is not None else self._item.stack_count
        drag_count = min(drag_count, self._item.stack_count)

        if drag_count <= 0:
            return None

        payload = DragPayload(
            source_slot=self,
            item=self._item,
            drag_count=drag_count,
            original_position=(self._x, self._y),
            offset_x=mouse_x - self._x,
            offset_y=mouse_y - self._y,
        )

        # Check if drag is allowed
        if self._on_drag_start and not self._on_drag_start(payload):
            return None

        self._is_dragging = True
        self._drag_payload = payload
        self._update_state()

        return payload

    def end_drag(self, success: bool) -> None:
        """End drag operation.

        Args:
            success: Whether drag was successful
        """
        if not self._is_dragging:
            return

        if self._on_drag_end:
            self._on_drag_end(self._drag_payload, success)

        if success and self._drag_payload:
            # Remove dragged items from slot
            self.remove_from_stack(self._drag_payload.drag_count)

        self._is_dragging = False
        self._drag_payload = None
        self._update_state()

    def can_accept(self, payload: DragPayload) -> bool:
        """Check if slot can accept a drop.

        Args:
            payload: Drag payload

        Returns:
            True if drop is valid
        """
        if not self._is_enabled or self._is_locked:
            return False

        if payload.source_slot is self:
            return False  # Can't drop on self

        if self._can_accept_drop:
            return self._can_accept_drop(payload)

        # Default: accept if empty or can stack
        if self.is_empty:
            return True

        if self._item.can_stack_with(payload.item):
            return self._item.get_remaining_stack_space() > 0

        return True  # Allow swap

    def handle_drop(self, payload: DragPayload) -> DropResult:
        """Handle drop on this slot.

        Args:
            payload: Drag payload

        Returns:
            Drop result
        """
        if not self.can_accept(payload):
            return DropResult.INVALID_TARGET

        if self._on_drop:
            return self._on_drop(self, payload)

        # Default drop behavior
        if self.is_empty:
            # Place item in empty slot
            self.set_item(ItemData(
                id=payload.item.id,
                name=payload.item.name,
                icon=payload.item.icon,
                rarity=payload.item.rarity,
                stack_count=payload.drag_count,
                max_stack=payload.item.max_stack,
                is_stackable=payload.item.is_stackable,
                tooltip_title=payload.item.tooltip_title,
                tooltip_description=payload.item.tooltip_description,
                tooltip_stats=payload.item.tooltip_stats.copy(),
                custom_data=payload.item.custom_data.copy(),
            ))
            return DropResult.SUCCESS

        if self._item.can_stack_with(payload.item):
            # Stack items
            added = self.add_to_stack(payload.drag_count)
            if added == payload.drag_count:
                return DropResult.STACKED
            return DropResult.PARTIAL_STACK

        # Swap items
        self.swap_with(payload.source_slot)
        return DropResult.SWAPPED

    def handle_drag_enter(self, payload: DragPayload) -> None:
        """Handle drag entering slot.

        Args:
            payload: Current drag payload
        """
        if self.can_accept(payload):
            self._state = SlotState.DROP_TARGET
        else:
            self._state = SlotState.DROP_INVALID

    def handle_drag_exit(self) -> None:
        """Handle drag leaving slot."""
        self._update_state()

    # Callbacks
    def on_click(self, callback: Callable[["InventorySlot"], None]) -> None:
        """Set click callback."""
        self._on_click = callback

    def on_right_click(self, callback: Callable[["InventorySlot"], None]) -> None:
        """Set right-click callback."""
        self._on_right_click = callback

    def on_double_click(self, callback: Callable[["InventorySlot"], None]) -> None:
        """Set double-click callback."""
        self._on_double_click = callback

    def on_hover_start(self, callback: Callable[["InventorySlot"], None]) -> None:
        """Set hover start callback."""
        self._on_hover_start = callback

    def on_hover_end(self, callback: Callable[["InventorySlot"], None]) -> None:
        """Set hover end callback."""
        self._on_hover_end = callback

    def on_drag_start(self, callback: Callable[[DragPayload], bool]) -> None:
        """Set drag start callback (return False to cancel)."""
        self._on_drag_start = callback

    def on_drag_end(self, callback: Callable[[DragPayload, bool], None]) -> None:
        """Set drag end callback."""
        self._on_drag_end = callback

    def on_drop(self, callback: Callable[["InventorySlot", DragPayload], DropResult]) -> None:
        """Set drop callback."""
        self._on_drop = callback

    def set_drop_validator(self, callback: Callable[[DragPayload], bool]) -> None:
        """Set drop validation callback."""
        self._can_accept_drop = callback

    # Update
    def update(self, delta_time: float) -> None:
        """Update slot state.

        Args:
            delta_time: Time since last update
        """
        # Update cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining = max(0.0, self._cooldown_remaining - delta_time)

        # Update tooltip timer
        if self._is_hovered and self._item:
            self._tooltip_timer += delta_time
            if self._tooltip_timer >= self._tooltip_delay:
                self._show_tooltip = True

    # Rendering helpers
    def get_icon_rect(self) -> tuple[float, float, float, float]:
        """Get icon display rectangle.

        Returns:
            (x, y, width, height)
        """
        padding = self._icon_padding
        return (
            self._x + padding,
            self._y + padding,
            self._size - padding * 2,
            self._size - padding * 2,
        )

    def get_cooldown_overlay_rect(self) -> Optional[tuple[float, float, float, float]]:
        """Get cooldown overlay rectangle.

        Returns:
            (x, y, width, height) or None if no cooldown
        """
        if not self.is_on_cooldown:
            return None

        # Overlay fills from top based on cooldown progress
        overlay_height = self._size * (1.0 - self.cooldown_percent)
        return (self._x, self._y, self._size, overlay_height)

    def get_stack_count_position(self) -> tuple[float, float]:
        """Get position for stack count text.

        Returns:
            (x, y) for bottom-right corner
        """
        return (
            self._x + self._size - 4,
            self._y + self._size - 4,
        )

    def point_in_bounds(self, px: float, py: float) -> bool:
        """Check if point is within slot bounds.

        Args:
            px: Point X
            py: Point Y

        Returns:
            True if point is inside slot
        """
        return (
            self._x <= px <= self._x + self._size and
            self._y <= py <= self._y + self._size
        )

    # Private methods
    def _update_state(self) -> None:
        """Update visual state based on current flags."""
        if not self._is_enabled:
            self._state = SlotState.DISABLED
        elif self._is_locked:
            self._state = SlotState.LOCKED
        elif self._is_dragging:
            self._state = SlotState.DRAGGING
        elif self._is_selected:
            self._state = SlotState.SELECTED
        elif self._is_hovered:
            self._state = SlotState.HOVERED
        else:
            self._state = SlotState.NORMAL

    def __repr__(self) -> str:
        item_str = f"'{self._item.name}'" if self._item else "empty"
        return (
            f"InventorySlot(id={self._id}, "
            f"index={self._slot_index}, "
            f"item={item_str})"
        )
