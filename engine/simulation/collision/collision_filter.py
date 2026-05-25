"""
Collision Filtering System.

This module implements collision layers and masks for controlling
which objects can collide with each other. Features include:
- 32 collision layers
- Bitwise collision masks
- Predefined layer constants
- Filter groups and categories
"""

from dataclasses import dataclass, field
from enum import IntFlag, auto
from typing import Callable


# =============================================================================
# Collision Layer Constants
# =============================================================================


class CollisionLayer(IntFlag):
    """
    Predefined collision layers (32 total available).

    Each layer is a bit position in a 32-bit mask.
    Custom layers can use CUSTOM_1 through CUSTOM_16.
    """

    NONE = 0

    # Core layers (bits 0-7)
    DEFAULT = 1 << 0      # Default layer for all objects
    STATIC = 1 << 1       # Static world geometry
    DYNAMIC = 1 << 2      # Dynamic physics objects
    KINEMATIC = 1 << 3    # Kinematic objects (player-controlled)
    TRIGGER = 1 << 4      # Trigger volumes (no physical response)
    PROJECTILE = 1 << 5   # Projectiles and bullets
    DEBRIS = 1 << 6       # Debris and small objects
    SENSOR = 1 << 7       # Sensor objects (detect but don't respond)

    # Character layers (bits 8-11)
    PLAYER = 1 << 8       # Player characters
    NPC = 1 << 9          # Non-player characters
    ENEMY = 1 << 10       # Enemy characters
    VEHICLE = 1 << 11     # Vehicles

    # Environment layers (bits 12-15)
    TERRAIN = 1 << 12     # Ground/terrain
    WATER = 1 << 13       # Water volumes
    CLIMBABLE = 1 << 14   # Climbable surfaces
    DESTRUCTIBLE = 1 << 15  # Destructible objects

    # Custom layers (bits 16-31)
    CUSTOM_1 = 1 << 16
    CUSTOM_2 = 1 << 17
    CUSTOM_3 = 1 << 18
    CUSTOM_4 = 1 << 19
    CUSTOM_5 = 1 << 20
    CUSTOM_6 = 1 << 21
    CUSTOM_7 = 1 << 22
    CUSTOM_8 = 1 << 23
    CUSTOM_9 = 1 << 24
    CUSTOM_10 = 1 << 25
    CUSTOM_11 = 1 << 26
    CUSTOM_12 = 1 << 27
    CUSTOM_13 = 1 << 28
    CUSTOM_14 = 1 << 29
    CUSTOM_15 = 1 << 30
    CUSTOM_16 = 1 << 31

    # Common combinations
    ALL = 0xFFFFFFFF
    ALL_STATIC = STATIC | TERRAIN
    ALL_DYNAMIC = DYNAMIC | PLAYER | NPC | ENEMY | VEHICLE
    ALL_CHARACTERS = PLAYER | NPC | ENEMY


# =============================================================================
# Collision Mask
# =============================================================================


@dataclass
class CollisionMask:
    """
    Defines which layers an object collides with.

    The mask is a 32-bit integer where each bit corresponds
    to a collision layer.
    """

    value: int = CollisionLayer.ALL

    @classmethod
    def from_layers(cls, *layers: CollisionLayer) -> "CollisionMask":
        """Create mask from multiple layers."""
        mask = 0
        for layer in layers:
            mask |= layer
        return cls(mask)

    @classmethod
    def all_except(cls, *layers: CollisionLayer) -> "CollisionMask":
        """Create mask with all layers except specified."""
        mask = CollisionLayer.ALL
        for layer in layers:
            mask &= ~layer
        return cls(mask)

    @classmethod
    def none(cls) -> "CollisionMask":
        """Create empty mask (collides with nothing)."""
        return cls(CollisionLayer.NONE)

    def includes(self, layer: CollisionLayer) -> bool:
        """Check if mask includes a layer."""
        return bool(self.value & layer)

    def add(self, layer: CollisionLayer) -> "CollisionMask":
        """Return new mask with layer added."""
        return CollisionMask(self.value | layer)

    def remove(self, layer: CollisionLayer) -> "CollisionMask":
        """Return new mask with layer removed."""
        return CollisionMask(self.value & ~layer)

    def toggle(self, layer: CollisionLayer) -> "CollisionMask":
        """Return new mask with layer toggled."""
        return CollisionMask(self.value ^ layer)

    def intersects(self, other: "CollisionMask") -> bool:
        """Check if masks have any common layers."""
        return bool(self.value & other.value)

    def __and__(self, other: "CollisionMask") -> "CollisionMask":
        return CollisionMask(self.value & other.value)

    def __or__(self, other: "CollisionMask") -> "CollisionMask":
        return CollisionMask(self.value | other.value)

    def __invert__(self) -> "CollisionMask":
        return CollisionMask(~self.value & CollisionLayer.ALL)


# =============================================================================
# Collision Filter
# =============================================================================


@dataclass
class CollisionFilter:
    """
    Complete collision filter for an object.

    Combines category (what layer the object is in) with
    mask (what layers it collides with).
    """

    # Category: which layer(s) this object belongs to
    category: CollisionLayer = CollisionLayer.DEFAULT

    # Mask: which layers this object collides with
    mask: CollisionMask = field(default_factory=lambda: CollisionMask(CollisionLayer.ALL))

    # Group: objects in same group don't collide (0 = no group)
    group: int = 0

    def __post_init__(self):
        # Ensure mask is CollisionMask type
        if isinstance(self.mask, int):
            self.mask = CollisionMask(self.mask)

    @classmethod
    def static(cls) -> "CollisionFilter":
        """Create filter for static objects."""
        return cls(
            category=CollisionLayer.STATIC,
            mask=CollisionMask.from_layers(
                CollisionLayer.DYNAMIC,
                CollisionLayer.KINEMATIC,
                CollisionLayer.PLAYER,
                CollisionLayer.NPC,
                CollisionLayer.ENEMY,
                CollisionLayer.VEHICLE,
                CollisionLayer.PROJECTILE,
            ),
        )

    @classmethod
    def dynamic(cls) -> "CollisionFilter":
        """Create filter for dynamic objects."""
        return cls(
            category=CollisionLayer.DYNAMIC,
            mask=CollisionMask(CollisionLayer.ALL),
        )

    @classmethod
    def kinematic(cls) -> "CollisionFilter":
        """Create filter for kinematic objects."""
        return cls(
            category=CollisionLayer.KINEMATIC,
            mask=CollisionMask.from_layers(
                CollisionLayer.STATIC,
                CollisionLayer.TERRAIN,
                CollisionLayer.DYNAMIC,
            ),
        )

    @classmethod
    def trigger(cls) -> "CollisionFilter":
        """Create filter for trigger volumes."""
        return cls(
            category=CollisionLayer.TRIGGER,
            mask=CollisionMask.from_layers(
                CollisionLayer.PLAYER,
                CollisionLayer.NPC,
                CollisionLayer.ENEMY,
                CollisionLayer.VEHICLE,
            ),
        )

    @classmethod
    def projectile(cls) -> "CollisionFilter":
        """Create filter for projectiles."""
        return cls(
            category=CollisionLayer.PROJECTILE,
            mask=CollisionMask.from_layers(
                CollisionLayer.STATIC,
                CollisionLayer.TERRAIN,
                CollisionLayer.DYNAMIC,
                CollisionLayer.PLAYER,
                CollisionLayer.NPC,
                CollisionLayer.ENEMY,
                CollisionLayer.DESTRUCTIBLE,
            ),
        )

    @classmethod
    def player(cls, group: int = 0) -> "CollisionFilter":
        """Create filter for player characters."""
        return cls(
            category=CollisionLayer.PLAYER,
            mask=CollisionMask.all_except(CollisionLayer.PLAYER),
            group=group,
        )

    @classmethod
    def npc(cls, group: int = 0) -> "CollisionFilter":
        """Create filter for NPCs."""
        return cls(
            category=CollisionLayer.NPC,
            mask=CollisionMask(CollisionLayer.ALL),
            group=group,
        )

    @classmethod
    def enemy(cls, group: int = 0) -> "CollisionFilter":
        """Create filter for enemies."""
        return cls(
            category=CollisionLayer.ENEMY,
            mask=CollisionMask.all_except(CollisionLayer.ENEMY),
            group=group,
        )

    @classmethod
    def debris(cls) -> "CollisionFilter":
        """Create filter for debris (collides with world only)."""
        return cls(
            category=CollisionLayer.DEBRIS,
            mask=CollisionMask.from_layers(
                CollisionLayer.STATIC,
                CollisionLayer.TERRAIN,
            ),
        )

    @classmethod
    def sensor(cls, layers_to_detect: CollisionLayer = CollisionLayer.ALL_DYNAMIC) -> "CollisionFilter":
        """Create filter for sensor objects."""
        return cls(
            category=CollisionLayer.SENSOR,
            mask=CollisionMask(layers_to_detect),
        )


# =============================================================================
# Filter Functions
# =============================================================================


def should_collide(filter_a: CollisionFilter, filter_b: CollisionFilter) -> bool:
    """
    Determine if two objects should collide based on their filters.

    Collision occurs if:
    1. Neither is in a group, OR groups are different
    2. A's mask includes B's category AND B's mask includes A's category

    Args:
        filter_a: Filter for first object
        filter_b: Filter for second object

    Returns:
        True if objects should collide
    """
    # Group filtering: same non-zero group means no collision
    if filter_a.group != 0 and filter_a.group == filter_b.group:
        return False

    # Category/mask filtering: both must accept each other
    a_accepts_b = filter_a.mask.includes(filter_b.category)
    b_accepts_a = filter_b.mask.includes(filter_a.category)

    return a_accepts_b and b_accepts_a


def create_layer_matrix() -> list[list[bool]]:
    """
    Create a 32x32 collision matrix for all layer pairs.

    Returns:
        2D list where matrix[i][j] indicates if layers i and j collide
    """
    matrix: list[list[bool]] = []
    for i in range(32):
        row: list[bool] = []
        for j in range(32):
            # Default: all layers collide with each other
            row.append(True)
        matrix.append(row)
    return matrix


# =============================================================================
# Collision Filter Manager
# =============================================================================


class CollisionFilterManager:
    """
    Manages collision filtering for a physics world.

    Supports both filter-based and callback-based filtering.
    """

    def __init__(self):
        self._filters: dict[int, CollisionFilter] = {}
        self._collision_matrix: list[list[bool]] = create_layer_matrix()
        self._custom_callbacks: list[Callable[[int, int], bool]] = []

        # Initialize default layer interactions
        self._setup_default_interactions()

    def _setup_default_interactions(self) -> None:
        """Set up default layer collision interactions."""
        # Triggers don't physically collide with anything
        trigger_bit = 4  # CollisionLayer.TRIGGER
        for i in range(32):
            self._collision_matrix[trigger_bit][i] = False
            self._collision_matrix[i][trigger_bit] = False

        # Sensors detect but don't respond
        sensor_bit = 7  # CollisionLayer.SENSOR
        for i in range(32):
            self._collision_matrix[sensor_bit][i] = False
            self._collision_matrix[i][sensor_bit] = False

        # Debris only collides with static/terrain
        debris_bit = 6  # CollisionLayer.DEBRIS
        for i in range(32):
            if i not in (1, 12):  # STATIC, TERRAIN
                self._collision_matrix[debris_bit][i] = False
                self._collision_matrix[i][debris_bit] = False

    def set_filter(self, object_id: int, filter_: CollisionFilter) -> None:
        """
        Set collision filter for an object.

        Args:
            object_id: Object identifier
            filter_: Collision filter to apply
        """
        self._filters[object_id] = filter_

    def get_filter(self, object_id: int) -> CollisionFilter:
        """
        Get collision filter for an object.

        Args:
            object_id: Object identifier

        Returns:
            Collision filter (default if not set)
        """
        return self._filters.get(object_id, CollisionFilter())

    def remove_filter(self, object_id: int) -> bool:
        """
        Remove collision filter for an object.

        Args:
            object_id: Object identifier

        Returns:
            True if filter was removed
        """
        if object_id in self._filters:
            del self._filters[object_id]
            return True
        return False

    def set_layer_collision(
        self,
        layer_a: CollisionLayer,
        layer_b: CollisionLayer,
        collide: bool,
    ) -> None:
        """
        Set collision between two layers.

        Args:
            layer_a: First layer
            layer_b: Second layer
            collide: Whether layers should collide
        """
        bit_a = (layer_a.bit_length() - 1) if layer_a else 0
        bit_b = (layer_b.bit_length() - 1) if layer_b else 0

        if 0 <= bit_a < 32 and 0 <= bit_b < 32:
            self._collision_matrix[bit_a][bit_b] = collide
            self._collision_matrix[bit_b][bit_a] = collide

    def get_layer_collision(
        self,
        layer_a: CollisionLayer,
        layer_b: CollisionLayer,
    ) -> bool:
        """
        Check if two layers collide.

        Args:
            layer_a: First layer
            layer_b: Second layer

        Returns:
            True if layers collide
        """
        bit_a = (layer_a.bit_length() - 1) if layer_a else 0
        bit_b = (layer_b.bit_length() - 1) if layer_b else 0

        if 0 <= bit_a < 32 and 0 <= bit_b < 32:
            return self._collision_matrix[bit_a][bit_b]
        return False

    def add_callback(self, callback: Callable[[int, int], bool]) -> None:
        """
        Add custom collision callback.

        Callback receives (id_a, id_b) and returns True to allow collision.

        Args:
            callback: Filter callback function
        """
        self._custom_callbacks.append(callback)

    def remove_callback(self, callback: Callable[[int, int], bool]) -> bool:
        """
        Remove custom collision callback.

        Args:
            callback: Callback to remove

        Returns:
            True if callback was removed
        """
        if callback in self._custom_callbacks:
            self._custom_callbacks.remove(callback)
            return True
        return False

    def should_collide(self, id_a: int, id_b: int) -> bool:
        """
        Check if two objects should collide.

        Considers filters, layer matrix, and custom callbacks.

        Args:
            id_a: First object ID
            id_b: Second object ID

        Returns:
            True if objects should collide
        """
        filter_a = self.get_filter(id_a)
        filter_b = self.get_filter(id_b)

        # Check filter-based collision
        if not should_collide(filter_a, filter_b):
            return False

        # Check layer matrix
        for layer_a in CollisionLayer:
            if layer_a & filter_a.category:
                for layer_b in CollisionLayer:
                    if layer_b & filter_b.category:
                        bit_a = (layer_a.bit_length() - 1) if layer_a else 0
                        bit_b = (layer_b.bit_length() - 1) if layer_b else 0
                        if 0 <= bit_a < 32 and 0 <= bit_b < 32:
                            if not self._collision_matrix[bit_a][bit_b]:
                                return False

        # Check custom callbacks
        for callback in self._custom_callbacks:
            if not callback(id_a, id_b):
                return False

        return True

    def clear(self) -> None:
        """Clear all filters and reset matrix."""
        self._filters.clear()
        self._collision_matrix = create_layer_matrix()
        self._custom_callbacks.clear()
        self._setup_default_interactions()


# =============================================================================
# Filter Presets
# =============================================================================


class FilterPresets:
    """Common filter configurations."""

    @staticmethod
    def fps_game() -> dict[str, CollisionFilter]:
        """Filters for FPS game."""
        return {
            "world": CollisionFilter.static(),
            "player": CollisionFilter.player(group=1),
            "enemy": CollisionFilter.enemy(group=2),
            "bullet": CollisionFilter.projectile(),
            "pickup": CollisionFilter.trigger(),
        }

    @staticmethod
    def platformer() -> dict[str, CollisionFilter]:
        """Filters for platformer game."""
        return {
            "terrain": CollisionFilter(
                category=CollisionLayer.TERRAIN,
                mask=CollisionMask(CollisionLayer.ALL),
            ),
            "player": CollisionFilter(
                category=CollisionLayer.PLAYER,
                mask=CollisionMask.all_except(CollisionLayer.DEBRIS),
            ),
            "platform": CollisionFilter(
                category=CollisionLayer.STATIC,
                mask=CollisionMask(CollisionLayer.ALL_CHARACTERS),
            ),
            "collectible": CollisionFilter.trigger(),
        }

    @staticmethod
    def racing() -> dict[str, CollisionFilter]:
        """Filters for racing game."""
        return {
            "track": CollisionFilter.static(),
            "vehicle": CollisionFilter(
                category=CollisionLayer.VEHICLE,
                mask=CollisionMask(CollisionLayer.ALL),
                group=0,  # Vehicles collide with each other
            ),
            "barrier": CollisionFilter(
                category=CollisionLayer.STATIC,
                mask=CollisionMask.from_layers(CollisionLayer.VEHICLE),
            ),
            "checkpoint": CollisionFilter.trigger(),
        }
