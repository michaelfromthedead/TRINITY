"""
Game-specific UI widgets for HUD, inventory, and combat feedback.

This module provides specialized widgets for game UI:
- HealthBar: Health/resource bars with animation and damage preview
- Minimap: World overview with entity markers
- InventorySlot: Drag-and-drop item slots
- DamageNumbers: Floating combat text
- Tooltip: Contextual information display
"""

from .health_bar import (
    HealthBar,
    HealthBarStyle,
    HealthBarSegment,
    ResourceType,
)
from .minimap import (
    Minimap,
    MinimapMarker,
    MarkerType,
    MinimapConfig,
)
from .inventory_slot import (
    InventorySlot,
    ItemData,
    ItemRarity,
    DragPayload,
    DropResult,
    SlotState,
)
from .damage_numbers import (
    DamageNumber,
    DamageNumberManager,
    DamageType,
    DamageNumberConfig,
)
from .tooltip import (
    Tooltip,
    TooltipManager,
    TooltipContent,
    TooltipPosition,
    TooltipAnimation,
    TooltipStyle,
    RichTooltip,
)

__all__ = [
    # Health Bar
    "HealthBar",
    "HealthBarStyle",
    "HealthBarSegment",
    "ResourceType",
    # Minimap
    "Minimap",
    "MinimapMarker",
    "MarkerType",
    "MinimapConfig",
    # Inventory Slot
    "InventorySlot",
    "ItemData",
    "ItemRarity",
    "DragPayload",
    "DropResult",
    "SlotState",
    # Damage Numbers
    "DamageNumber",
    "DamageNumberManager",
    "DamageType",
    "DamageNumberConfig",
    # Tooltip
    "Tooltip",
    "TooltipManager",
    "TooltipContent",
    "TooltipPosition",
    "TooltipAnimation",
    "TooltipStyle",
    "RichTooltip",
]
