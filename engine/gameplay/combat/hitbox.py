"""
Combat System - Hitbox Module

Provides hitbox and hurtbox collision detection for combat:
- Hitbox definition (attacking/damaging boxes)
- Hurtbox definition (vulnerable areas)
- Collision detection between hitboxes and hurtboxes
- Hitbox groups and hierarchies
- Priority system for overlapping hits
- Activation/deactivation for animation integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import time
import math

from .constants import (
    HitboxZone,
    HITBOX_DAMAGE_MULTIPLIERS,
    CRITICAL_HIT_ZONES,
    COUNTER_HIT_DAMAGE_MULTIPLIER,
)


# =============================================================================
# ENUMS
# =============================================================================


class HitboxType(Enum):
    """Types of hitboxes."""

    ATTACK = auto()  # Deals damage
    PROJECTILE = auto()  # Projectile hitbox
    GRAB = auto()  # Grab/throw hitbox
    PARRY = auto()  # Parry/counter window
    BLOCK = auto()  # Blocking area
    INVINCIBLE = auto()  # Invincibility frames


class HurtboxType(Enum):
    """Types of hurtboxes."""

    NORMAL = auto()  # Standard vulnerable area
    COUNTER = auto()  # Counter-hit state
    ARMORED = auto()  # Super armor/hyper armor
    INTANGIBLE = auto()  # Invincible


class HitboxShape(Enum):
    """Shapes for collision detection."""

    SPHERE = auto()
    BOX = auto()
    CAPSULE = auto()
    CYLINDER = auto()


class CollisionResult(Enum):
    """Result of a collision check."""

    NO_COLLISION = auto()
    HIT = auto()
    BLOCKED = auto()
    PARRIED = auto()
    COUNTER_HIT = auto()
    ARMORED = auto()
    INVINCIBLE = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Vector3:
    """Simple 3D vector."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance_to(self, other: "Vector3") -> float:
        """Calculate distance to another point."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""

    min_point: Vector3 = field(default_factory=Vector3)
    max_point: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))

    @property
    def center(self) -> Vector3:
        """Get center point."""
        return Vector3(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2,
            (self.min_point.z + self.max_point.z) / 2,
        )

    @property
    def size(self) -> Vector3:
        """Get size/dimensions."""
        return Vector3(
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z,
        )

    @property
    def half_extents(self) -> Vector3:
        """Get half extents."""
        s = self.size
        return Vector3(s.x / 2, s.y / 2, s.z / 2)

    def contains_point(self, point: Vector3) -> bool:
        """Check if point is inside the box."""
        return (
            self.min_point.x <= point.x <= self.max_point.x and
            self.min_point.y <= point.y <= self.max_point.y and
            self.min_point.z <= point.z <= self.max_point.z
        )

    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this box intersects another."""
        return (
            self.min_point.x <= other.max_point.x and
            self.max_point.x >= other.min_point.x and
            self.min_point.y <= other.max_point.y and
            self.max_point.y >= other.min_point.y and
            self.min_point.z <= other.max_point.z and
            self.max_point.z >= other.min_point.z
        )


@dataclass
class Hitbox:
    """
    An attacking/damaging collision volume.

    Attributes:
        hitbox_id: Unique identifier
        owner_id: Entity that owns this hitbox
        hitbox_type: Type of hitbox
        zone: Body zone this hitbox represents
        shape: Collision shape
        position: World position (center)
        size: Size/dimensions
        damage: Base damage dealt
        priority: Higher priority hits override lower
        groups: Set of group names this hitbox belongs to
        ignore_groups: Groups to ignore in collision
        active: Whether hitbox is currently active
        hit_entities: Entities already hit (for multi-hit prevention)
        max_hits: Maximum entities this can hit (-1 for unlimited)
        lifetime: Time until auto-deactivation (None for manual)
    """

    hitbox_id: str
    owner_id: int
    hitbox_type: HitboxType = HitboxType.ATTACK
    zone: HitboxZone = HitboxZone.GENERIC
    shape: HitboxShape = HitboxShape.BOX
    position: Vector3 = field(default_factory=Vector3)
    size: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    damage: float = 10.0
    priority: int = 0
    groups: Set[str] = field(default_factory=set)
    ignore_groups: Set[str] = field(default_factory=set)
    active: bool = False
    hit_entities: Set[int] = field(default_factory=set)
    max_hits: int = -1
    lifetime: Optional[float] = None

    # Animation integration
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    bone_name: Optional[str] = None
    offset: Vector3 = field(default_factory=Vector3)

    # State
    _activation_time: float = 0.0
    _hit_count: int = 0

    @property
    def bounding_box(self) -> BoundingBox:
        """Get axis-aligned bounding box."""
        half = Vector3(self.size.x / 2, self.size.y / 2, self.size.z / 2)
        return BoundingBox(
            min_point=self.position - half,
            max_point=self.position + half,
        )

    @property
    def damage_multiplier(self) -> float:
        """Get damage multiplier for this hitbox zone."""
        return HITBOX_DAMAGE_MULTIPLIERS.get(self.zone, 1.0)

    @property
    def is_critical_zone(self) -> bool:
        """Check if this hitbox can cause critical hits."""
        return self.zone in CRITICAL_HIT_ZONES

    @property
    def time_active(self) -> float:
        """Time since activation."""
        if not self.active:
            return 0.0
        return time.time() - self._activation_time

    @property
    def is_expired(self) -> bool:
        """Check if hitbox has exceeded its lifetime."""
        if not self.active or self.lifetime is None:
            return False
        return self.time_active >= self.lifetime

    @property
    def can_hit_more(self) -> bool:
        """Check if hitbox can hit more entities."""
        if self.max_hits < 0:
            return True
        return self._hit_count < self.max_hits

    def activate(self, lifetime: Optional[float] = None) -> None:
        """Activate the hitbox."""
        self.active = True
        self._activation_time = time.time()
        self.hit_entities.clear()
        self._hit_count = 0
        if lifetime is not None:
            self.lifetime = lifetime

    def deactivate(self) -> None:
        """Deactivate the hitbox."""
        self.active = False

    def record_hit(self, entity_id: int) -> bool:
        """
        Record a hit on an entity.

        Args:
            entity_id: Entity that was hit

        Returns:
            True if hit was recorded (False if already hit)
        """
        if entity_id in self.hit_entities:
            return False

        self.hit_entities.add(entity_id)
        self._hit_count += 1

        # Auto-deactivate if max hits reached
        if not self.can_hit_more:
            self.deactivate()

        return True

    def can_hit_entity(self, entity_id: int) -> bool:
        """Check if this hitbox can hit the specified entity."""
        if entity_id == self.owner_id:
            return False
        if entity_id in self.hit_entities:
            return False
        if not self.can_hit_more:
            return False
        return True

    def set_position(self, position: Tuple[float, float, float]) -> None:
        """Set position from tuple."""
        self.position = Vector3(position[0], position[1], position[2])


@dataclass
class Hurtbox:
    """
    A vulnerable collision volume that can receive damage.

    Attributes:
        hurtbox_id: Unique identifier
        owner_id: Entity that owns this hurtbox
        hurtbox_type: Type of hurtbox (normal, counter, armored)
        zone: Body zone this represents
        shape: Collision shape
        position: World position (center)
        size: Size/dimensions
        groups: Set of group names
        active: Whether hurtbox is currently active
        armor_value: Super armor hits (0 = no armor)
        damage_multiplier: Damage modifier for this hurtbox
    """

    hurtbox_id: str
    owner_id: int
    hurtbox_type: HurtboxType = HurtboxType.NORMAL
    zone: HitboxZone = HitboxZone.GENERIC
    shape: HitboxShape = HitboxShape.BOX
    position: Vector3 = field(default_factory=Vector3)
    size: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    groups: Set[str] = field(default_factory=set)
    active: bool = True
    armor_value: int = 0
    damage_multiplier: float = 1.0

    # Animation integration
    bone_name: Optional[str] = None
    offset: Vector3 = field(default_factory=Vector3)

    # State
    _armor_remaining: int = 0

    def __post_init__(self) -> None:
        """Initialize armor state."""
        self._armor_remaining = self.armor_value

    @property
    def bounding_box(self) -> BoundingBox:
        """Get axis-aligned bounding box."""
        half = Vector3(self.size.x / 2, self.size.y / 2, self.size.z / 2)
        return BoundingBox(
            min_point=self.position - half,
            max_point=self.position + half,
        )

    @property
    def zone_multiplier(self) -> float:
        """Get damage multiplier for this zone."""
        return HITBOX_DAMAGE_MULTIPLIERS.get(self.zone, 1.0) * self.damage_multiplier

    @property
    def is_invincible(self) -> bool:
        """Check if hurtbox is intangible/invincible."""
        return self.hurtbox_type == HurtboxType.INTANGIBLE

    @property
    def is_counter_state(self) -> bool:
        """Check if in counter-hit state."""
        return self.hurtbox_type == HurtboxType.COUNTER

    @property
    def has_armor(self) -> bool:
        """Check if has active super armor."""
        return self.hurtbox_type == HurtboxType.ARMORED and self._armor_remaining > 0

    def absorb_armor_hit(self) -> bool:
        """
        Absorb a hit with armor.

        Returns:
            True if armor absorbed the hit
        """
        if self._armor_remaining > 0:
            self._armor_remaining -= 1
            return True
        return False

    def reset_armor(self) -> None:
        """Reset armor to full value."""
        self._armor_remaining = self.armor_value

    def set_position(self, position: Tuple[float, float, float]) -> None:
        """Set position from tuple."""
        self.position = Vector3(position[0], position[1], position[2])


@dataclass
class CollisionInfo:
    """Information about a hitbox/hurtbox collision."""

    hitbox: Hitbox
    hurtbox: Hurtbox
    result: CollisionResult
    point: Vector3  # Contact point
    damage: float  # Final damage after modifiers
    timestamp: float = field(default_factory=time.time)

    @property
    def is_hit(self) -> bool:
        """Check if this was a successful hit."""
        return self.result in (CollisionResult.HIT, CollisionResult.COUNTER_HIT)

    @property
    def is_counter_hit(self) -> bool:
        """Check if this was a counter hit."""
        return self.result == CollisionResult.COUNTER_HIT


# =============================================================================
# HITBOX SYSTEM
# =============================================================================


class HitboxSystem:
    """
    System for managing hitboxes and hurtboxes.

    Features:
    - Registration and tracking of hitboxes/hurtboxes
    - Collision detection between hitboxes and hurtboxes
    - Group-based filtering
    - Priority system for overlapping hits
    - Event callbacks for collisions
    """

    def __init__(self) -> None:
        """Initialize the hitbox system."""
        # Hitbox/hurtbox registries
        self._hitboxes: Dict[str, Hitbox] = {}
        self._hurtboxes: Dict[str, Hurtbox] = {}

        # Entity tracking
        self._entity_hitboxes: Dict[int, List[str]] = {}
        self._entity_hurtboxes: Dict[int, List[str]] = {}

        # Group indices
        self._hitbox_groups: Dict[str, Set[str]] = {}
        self._hurtbox_groups: Dict[str, Set[str]] = {}

        # Collision tracking
        self._recent_collisions: List[CollisionInfo] = []
        self._max_collision_history: int = 1000

        # Event handlers
        self._on_hit: List[Callable[[CollisionInfo], None]] = []
        self._on_blocked: List[Callable[[CollisionInfo], None]] = []
        self._on_parried: List[Callable[[CollisionInfo], None]] = []

    # =========================================================================
    # HITBOX MANAGEMENT
    # =========================================================================

    def create_hitbox(
        self,
        hitbox_id: str,
        owner_id: int,
        position: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (1, 1, 1),
        damage: float = 10.0,
        hitbox_type: HitboxType = HitboxType.ATTACK,
        zone: HitboxZone = HitboxZone.GENERIC,
        priority: int = 0,
        groups: Optional[Set[str]] = None,
        max_hits: int = -1,
        lifetime: Optional[float] = None,
    ) -> Hitbox:
        """
        Create and register a new hitbox.

        Args:
            hitbox_id: Unique identifier
            owner_id: Owning entity ID
            position: World position
            size: Box dimensions
            damage: Base damage
            hitbox_type: Type of hitbox
            zone: Body zone
            priority: Hit priority
            groups: Group memberships
            max_hits: Maximum hits (-1 for unlimited)
            lifetime: Auto-deactivation time

        Returns:
            Created Hitbox
        """
        hitbox = Hitbox(
            hitbox_id=hitbox_id,
            owner_id=owner_id,
            hitbox_type=hitbox_type,
            zone=zone,
            position=Vector3(position[0], position[1], position[2]),
            size=Vector3(size[0], size[1], size[2]),
            damage=damage,
            priority=priority,
            groups=groups or set(),
            max_hits=max_hits,
            lifetime=lifetime,
        )

        self._register_hitbox(hitbox)
        return hitbox

    def _register_hitbox(self, hitbox: Hitbox) -> None:
        """Register a hitbox."""
        self._hitboxes[hitbox.hitbox_id] = hitbox

        # Track by entity
        if hitbox.owner_id not in self._entity_hitboxes:
            self._entity_hitboxes[hitbox.owner_id] = []
        self._entity_hitboxes[hitbox.owner_id].append(hitbox.hitbox_id)

        # Index by groups
        for group in hitbox.groups:
            if group not in self._hitbox_groups:
                self._hitbox_groups[group] = set()
            self._hitbox_groups[group].add(hitbox.hitbox_id)

    def remove_hitbox(self, hitbox_id: str) -> bool:
        """Remove a hitbox."""
        hitbox = self._hitboxes.pop(hitbox_id, None)
        if not hitbox:
            return False

        # Remove from entity tracking
        if hitbox.owner_id in self._entity_hitboxes:
            try:
                self._entity_hitboxes[hitbox.owner_id].remove(hitbox_id)
            except ValueError:
                pass

        # Remove from groups
        for group in hitbox.groups:
            if group in self._hitbox_groups:
                self._hitbox_groups[group].discard(hitbox_id)

        return True

    def get_hitbox(self, hitbox_id: str) -> Optional[Hitbox]:
        """Get a hitbox by ID."""
        return self._hitboxes.get(hitbox_id)

    def get_entity_hitboxes(self, entity_id: int) -> List[Hitbox]:
        """Get all hitboxes for an entity."""
        hitbox_ids = self._entity_hitboxes.get(entity_id, [])
        return [self._hitboxes[hid] for hid in hitbox_ids if hid in self._hitboxes]

    def get_active_hitboxes(self) -> List[Hitbox]:
        """Get all active hitboxes."""
        return [h for h in self._hitboxes.values() if h.active]

    # =========================================================================
    # HURTBOX MANAGEMENT
    # =========================================================================

    def create_hurtbox(
        self,
        hurtbox_id: str,
        owner_id: int,
        position: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (1, 1, 1),
        hurtbox_type: HurtboxType = HurtboxType.NORMAL,
        zone: HitboxZone = HitboxZone.GENERIC,
        groups: Optional[Set[str]] = None,
        armor_value: int = 0,
        damage_multiplier: float = 1.0,
    ) -> Hurtbox:
        """
        Create and register a new hurtbox.

        Args:
            hurtbox_id: Unique identifier
            owner_id: Owning entity ID
            position: World position
            size: Box dimensions
            hurtbox_type: Type of hurtbox
            zone: Body zone
            groups: Group memberships
            armor_value: Super armor hits
            damage_multiplier: Damage modifier

        Returns:
            Created Hurtbox
        """
        hurtbox = Hurtbox(
            hurtbox_id=hurtbox_id,
            owner_id=owner_id,
            hurtbox_type=hurtbox_type,
            zone=zone,
            position=Vector3(position[0], position[1], position[2]),
            size=Vector3(size[0], size[1], size[2]),
            groups=groups or set(),
            armor_value=armor_value,
            damage_multiplier=damage_multiplier,
        )

        self._register_hurtbox(hurtbox)
        return hurtbox

    def _register_hurtbox(self, hurtbox: Hurtbox) -> None:
        """Register a hurtbox."""
        self._hurtboxes[hurtbox.hurtbox_id] = hurtbox

        # Track by entity
        if hurtbox.owner_id not in self._entity_hurtboxes:
            self._entity_hurtboxes[hurtbox.owner_id] = []
        self._entity_hurtboxes[hurtbox.owner_id].append(hurtbox.hurtbox_id)

        # Index by groups
        for group in hurtbox.groups:
            if group not in self._hurtbox_groups:
                self._hurtbox_groups[group] = set()
            self._hurtbox_groups[group].add(hurtbox.hurtbox_id)

    def remove_hurtbox(self, hurtbox_id: str) -> bool:
        """Remove a hurtbox."""
        hurtbox = self._hurtboxes.pop(hurtbox_id, None)
        if not hurtbox:
            return False

        # Remove from entity tracking
        if hurtbox.owner_id in self._entity_hurtboxes:
            try:
                self._entity_hurtboxes[hurtbox.owner_id].remove(hurtbox_id)
            except ValueError:
                pass

        # Remove from groups
        for group in hurtbox.groups:
            if group in self._hurtbox_groups:
                self._hurtbox_groups[group].discard(hurtbox_id)

        return True

    def get_hurtbox(self, hurtbox_id: str) -> Optional[Hurtbox]:
        """Get a hurtbox by ID."""
        return self._hurtboxes.get(hurtbox_id)

    def get_entity_hurtboxes(self, entity_id: int) -> List[Hurtbox]:
        """Get all hurtboxes for an entity."""
        hurtbox_ids = self._entity_hurtboxes.get(entity_id, [])
        return [self._hurtboxes[hid] for hid in hurtbox_ids if hid in self._hurtboxes]

    def get_active_hurtboxes(self) -> List[Hurtbox]:
        """Get all active hurtboxes."""
        return [h for h in self._hurtboxes.values() if h.active]

    # =========================================================================
    # COLLISION DETECTION
    # =========================================================================

    def check_collision(
        self,
        hitbox: Hitbox,
        hurtbox: Hurtbox,
    ) -> Optional[CollisionInfo]:
        """
        Check collision between a hitbox and hurtbox.

        Args:
            hitbox: The attacking hitbox
            hurtbox: The vulnerable hurtbox

        Returns:
            CollisionInfo if collision occurred, None otherwise
        """
        # Skip if same owner
        if hitbox.owner_id == hurtbox.owner_id:
            return None

        # Skip if hitbox already hit this entity
        if not hitbox.can_hit_entity(hurtbox.owner_id):
            return None

        # Skip inactive boxes
        if not hitbox.active or not hurtbox.active:
            return None

        # Check group exclusions
        if hitbox.ignore_groups & hurtbox.groups:
            return None

        # Check bounding box intersection
        if not hitbox.bounding_box.intersects(hurtbox.bounding_box):
            return None

        # Collision detected - determine result
        result = self._determine_collision_result(hitbox, hurtbox)
        if result == CollisionResult.NO_COLLISION:
            return None

        # Calculate damage
        damage = hitbox.damage * hitbox.damage_multiplier * hurtbox.zone_multiplier

        # Apply counter-hit bonus
        if result == CollisionResult.COUNTER_HIT:
            damage *= COUNTER_HIT_DAMAGE_MULTIPLIER

        # Create collision info
        contact_point = Vector3(
            (hitbox.position.x + hurtbox.position.x) / 2,
            (hitbox.position.y + hurtbox.position.y) / 2,
            (hitbox.position.z + hurtbox.position.z) / 2,
        )

        collision = CollisionInfo(
            hitbox=hitbox,
            hurtbox=hurtbox,
            result=result,
            point=contact_point,
            damage=damage,
        )

        return collision

    def _determine_collision_result(
        self,
        hitbox: Hitbox,
        hurtbox: Hurtbox,
    ) -> CollisionResult:
        """Determine the result of a collision."""
        # Check hurtbox type
        if hurtbox.is_invincible:
            return CollisionResult.INVINCIBLE

        # Check for parry
        if hitbox.hitbox_type == HitboxType.ATTACK:
            # Check if target has active parry hitbox
            parry_hitboxes = [
                h for h in self.get_entity_hitboxes(hurtbox.owner_id)
                if h.active and h.hitbox_type == HitboxType.PARRY
            ]
            for parry in parry_hitboxes:
                if hitbox.bounding_box.intersects(parry.bounding_box):
                    return CollisionResult.PARRIED

        # Check for block
        if hitbox.hitbox_type == HitboxType.ATTACK:
            block_hitboxes = [
                h for h in self.get_entity_hitboxes(hurtbox.owner_id)
                if h.active and h.hitbox_type == HitboxType.BLOCK
            ]
            for block in block_hitboxes:
                if hitbox.bounding_box.intersects(block.bounding_box):
                    return CollisionResult.BLOCKED

        # Check for armor
        if hurtbox.has_armor:
            if hurtbox.absorb_armor_hit():
                return CollisionResult.ARMORED

        # Check for counter-hit
        if hurtbox.is_counter_state:
            return CollisionResult.COUNTER_HIT

        return CollisionResult.HIT

    def process_collisions(self) -> List[CollisionInfo]:
        """
        Process all active hitbox/hurtbox collisions.

        Returns:
            List of collision results
        """
        collisions: List[CollisionInfo] = []
        active_hitboxes = self.get_active_hitboxes()
        active_hurtboxes = self.get_active_hurtboxes()

        # Group collisions by target to handle priority
        target_collisions: Dict[int, List[CollisionInfo]] = {}

        for hitbox in active_hitboxes:
            for hurtbox in active_hurtboxes:
                collision = self.check_collision(hitbox, hurtbox)
                if collision:
                    target_id = hurtbox.owner_id
                    if target_id not in target_collisions:
                        target_collisions[target_id] = []
                    target_collisions[target_id].append(collision)

        # Apply priority - only highest priority hit per target
        for target_id, target_hits in target_collisions.items():
            if not target_hits:
                continue

            # Sort by priority (highest first)
            target_hits.sort(key=lambda c: c.hitbox.priority, reverse=True)

            # Get highest priority
            best_priority = target_hits[0].hitbox.priority
            best_hits = [c for c in target_hits if c.hitbox.priority == best_priority]

            # Process best hits
            for collision in best_hits:
                # Record hit
                collision.hitbox.record_hit(target_id)

                # Record collision
                self._recent_collisions.append(collision)

                # Emit events
                if collision.is_hit:
                    for handler in self._on_hit:
                        try:
                            handler(collision)
                        except Exception:
                            pass
                elif collision.result == CollisionResult.BLOCKED:
                    for handler in self._on_blocked:
                        try:
                            handler(collision)
                        except Exception:
                            pass
                elif collision.result == CollisionResult.PARRIED:
                    for handler in self._on_parried:
                        try:
                            handler(collision)
                        except Exception:
                            pass

                collisions.append(collision)

        # Trim collision history
        if len(self._recent_collisions) > self._max_collision_history:
            self._recent_collisions = self._recent_collisions[-self._max_collision_history:]

        return collisions

    # =========================================================================
    # ACTIVATION MANAGEMENT
    # =========================================================================

    def activate_hitbox(
        self,
        hitbox_id: str,
        lifetime: Optional[float] = None,
    ) -> bool:
        """Activate a hitbox."""
        hitbox = self._hitboxes.get(hitbox_id)
        if not hitbox:
            return False
        hitbox.activate(lifetime)
        return True

    def deactivate_hitbox(self, hitbox_id: str) -> bool:
        """Deactivate a hitbox."""
        hitbox = self._hitboxes.get(hitbox_id)
        if not hitbox:
            return False
        hitbox.deactivate()
        return True

    def activate_entity_hitboxes(
        self,
        entity_id: int,
        lifetime: Optional[float] = None,
    ) -> int:
        """Activate all hitboxes for an entity. Returns count activated."""
        count = 0
        for hitbox in self.get_entity_hitboxes(entity_id):
            hitbox.activate(lifetime)
            count += 1
        return count

    def deactivate_entity_hitboxes(self, entity_id: int) -> int:
        """Deactivate all hitboxes for an entity. Returns count deactivated."""
        count = 0
        for hitbox in self.get_entity_hitboxes(entity_id):
            hitbox.deactivate()
            count += 1
        return count

    def activate_hurtbox(self, hurtbox_id: str) -> bool:
        """Activate a hurtbox."""
        hurtbox = self._hurtboxes.get(hurtbox_id)
        if not hurtbox:
            return False
        hurtbox.active = True
        return True

    def deactivate_hurtbox(self, hurtbox_id: str) -> bool:
        """Deactivate a hurtbox."""
        hurtbox = self._hurtboxes.get(hurtbox_id)
        if not hurtbox:
            return False
        hurtbox.active = False
        return True

    def set_hurtbox_type(
        self,
        hurtbox_id: str,
        hurtbox_type: HurtboxType,
    ) -> bool:
        """Set hurtbox type (for state changes like counter/armor)."""
        hurtbox = self._hurtboxes.get(hurtbox_id)
        if not hurtbox:
            return False
        hurtbox.hurtbox_type = hurtbox_type
        if hurtbox_type == HurtboxType.ARMORED:
            hurtbox.reset_armor()
        return True

    # =========================================================================
    # UPDATE
    # =========================================================================

    def update(self, delta_time: float) -> List[CollisionInfo]:
        """
        Update hitbox system and process collisions.

        Args:
            delta_time: Time since last update

        Returns:
            List of collisions that occurred
        """
        # Deactivate expired hitboxes
        for hitbox in list(self._hitboxes.values()):
            if hitbox.is_expired:
                hitbox.deactivate()

        # Process collisions
        return self.process_collisions()

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def on_hit(self, handler: Callable[[CollisionInfo], None]) -> None:
        """Register handler for successful hits."""
        self._on_hit.append(handler)

    def on_blocked(self, handler: Callable[[CollisionInfo], None]) -> None:
        """Register handler for blocked hits."""
        self._on_blocked.append(handler)

    def on_parried(self, handler: Callable[[CollisionInfo], None]) -> None:
        """Register handler for parried hits."""
        self._on_parried.append(handler)

    # =========================================================================
    # ENTITY MANAGEMENT
    # =========================================================================

    def remove_entity(self, entity_id: int) -> None:
        """Remove all hitboxes and hurtboxes for an entity."""
        # Remove hitboxes
        hitbox_ids = list(self._entity_hitboxes.get(entity_id, []))
        for hitbox_id in hitbox_ids:
            self.remove_hitbox(hitbox_id)
        self._entity_hitboxes.pop(entity_id, None)

        # Remove hurtboxes
        hurtbox_ids = list(self._entity_hurtboxes.get(entity_id, []))
        for hurtbox_id in hurtbox_ids:
            self.remove_hurtbox(hurtbox_id)
        self._entity_hurtboxes.pop(entity_id, None)

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_collisions_for_entity(
        self,
        entity_id: int,
        as_attacker: bool = True,
        as_target: bool = True,
        limit: int = 100,
    ) -> List[CollisionInfo]:
        """Get recent collisions involving an entity."""
        collisions = []
        for collision in reversed(self._recent_collisions):
            if len(collisions) >= limit:
                break
            if as_attacker and collision.hitbox.owner_id == entity_id:
                collisions.append(collision)
            elif as_target and collision.hurtbox.owner_id == entity_id:
                collisions.append(collision)
        return list(reversed(collisions))

    def get_stats(self) -> Dict[str, Any]:
        """Get hitbox system statistics."""
        return {
            "total_hitboxes": len(self._hitboxes),
            "active_hitboxes": len(self.get_active_hitboxes()),
            "total_hurtboxes": len(self._hurtboxes),
            "active_hurtboxes": len(self.get_active_hurtboxes()),
            "recent_collisions": len(self._recent_collisions),
            "entities_tracked": len(
                set(self._entity_hitboxes.keys()) |
                set(self._entity_hurtboxes.keys())
            ),
        }

    # =========================================================================
    # UTILITY
    # =========================================================================

    def clear(self) -> None:
        """Clear all hitboxes and hurtboxes."""
        self._hitboxes.clear()
        self._hurtboxes.clear()
        self._entity_hitboxes.clear()
        self._entity_hurtboxes.clear()
        self._hitbox_groups.clear()
        self._hurtbox_groups.clear()
        self._recent_collisions.clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "HitboxType",
    "HurtboxType",
    "HitboxShape",
    "CollisionResult",
    # Data classes
    "Vector3",
    "BoundingBox",
    "Hitbox",
    "Hurtbox",
    "CollisionInfo",
    # System
    "HitboxSystem",
]
