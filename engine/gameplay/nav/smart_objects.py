"""
Smart objects for interactive navigation.

Provides support for objects that agents can interact with during navigation,
such as doors, buttons, cover points, and interaction stations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple

from .constants import (
    DEFAULT_INTERACTION_ANGLE,
    DEFAULT_INTERACTION_RADIUS,
    DEFAULT_MAX_QUEUE_SIZE,
    DEFAULT_RESERVATION_TIMEOUT,
    DEFAULT_SPATIAL_CELL_SIZE,
    SlotState,
)
from .navmesh import Vector3


# =============================================================================
# Enums
# =============================================================================


class SmartObjectCategory(Enum):
    """Categories of smart objects."""
    GENERIC = auto()
    COVER = auto()           # Cover point for combat
    DOOR = auto()            # Interactive door
    BUTTON = auto()          # Button/switch
    TERMINAL = auto()        # Computer terminal
    SEAT = auto()            # Chair/bench
    BED = auto()             # Sleeping spot
    WORKSTATION = auto()     # Crafting/work area
    VEHICLE = auto()         # Vehicle entry point
    CONTAINER = auto()       # Loot container
    CONVERSATION = auto()    # NPC conversation point


class InteractionType(Enum):
    """Types of interactions with smart objects."""
    INSTANT = auto()         # Immediate interaction
    TIMED = auto()           # Takes time to complete
    CONTINUOUS = auto()      # Ongoing while held
    TOGGLE = auto()          # On/off state


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class SmartObjectSlot:
    """Single interaction slot on a smart object."""
    id: int
    local_position: Vector3 = field(default_factory=Vector3)
    local_rotation: float = 0.0  # Facing direction (radians)
    state: SlotState = SlotState.AVAILABLE
    reserved_by: Optional[int] = None  # Agent ID
    occupied_by: Optional[int] = None  # Agent ID
    reservation_time: float = 0.0
    tags: Set[str] = field(default_factory=set)

    def is_available(self) -> bool:
        """Check if slot is available for reservation."""
        return self.state == SlotState.AVAILABLE

    def is_reserved(self) -> bool:
        """Check if slot is reserved."""
        return self.state == SlotState.RESERVED

    def is_occupied(self) -> bool:
        """Check if slot is occupied."""
        return self.state == SlotState.OCCUPIED

    def is_disabled(self) -> bool:
        """Check if slot is disabled."""
        return self.state == SlotState.DISABLED


@dataclass
class SmartObjectParams:
    """Parameters for smart object configuration."""
    # Interaction parameters
    interaction_radius: float = DEFAULT_INTERACTION_RADIUS
    interaction_angle: float = DEFAULT_INTERACTION_ANGLE  # Degrees
    interaction_duration: float = 1.0
    interaction_type: InteractionType = InteractionType.TIMED

    # Queue parameters
    max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE
    reservation_timeout: float = DEFAULT_RESERVATION_TIMEOUT

    # Requirements
    required_tags: Set[str] = field(default_factory=set)
    required_flags: int = 0

    # Animation
    approach_animation: Optional[str] = None
    interaction_animation: Optional[str] = None
    exit_animation: Optional[str] = None

    # Cost modifiers
    navigation_cost: float = 1.0
    priority: float = 1.0


@dataclass
class SmartObject:
    """Interactive object for navigation."""
    id: int
    name: str
    category: SmartObjectCategory = SmartObjectCategory.GENERIC
    position: Vector3 = field(default_factory=Vector3)
    rotation: float = 0.0  # Yaw in radians
    params: SmartObjectParams = field(default_factory=SmartObjectParams)
    slots: List[SmartObjectSlot] = field(default_factory=list)
    enabled: bool = True
    user_data: Optional[object] = None
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Create default slot if none provided."""
        if not self.slots:
            self.slots.append(SmartObjectSlot(id=0))

    @property
    def slot_count(self) -> int:
        """Get number of slots."""
        return len(self.slots)

    @property
    def available_slot_count(self) -> int:
        """Get number of available slots."""
        return sum(1 for slot in self.slots if slot.is_available())

    def get_slot(self, slot_id: int) -> Optional[SmartObjectSlot]:
        """Get slot by ID."""
        for slot in self.slots:
            if slot.id == slot_id:
                return slot
        return None

    def get_first_available_slot(self) -> Optional[SmartObjectSlot]:
        """Get first available slot."""
        for slot in self.slots:
            if slot.is_available():
                return slot
        return None

    def get_slot_world_position(self, slot: SmartObjectSlot) -> Vector3:
        """Get world position of a slot."""
        # Rotate local position by object rotation
        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)

        local = slot.local_position
        world_x = self.position.x + local.x * cos_r - local.z * sin_r
        world_y = self.position.y + local.y
        world_z = self.position.z + local.x * sin_r + local.z * cos_r

        return Vector3(world_x, world_y, world_z)

    def get_slot_world_rotation(self, slot: SmartObjectSlot) -> float:
        """Get world rotation of a slot."""
        return self.rotation + slot.local_rotation

    def is_position_in_range(self, position: Vector3) -> bool:
        """Check if position is within interaction range."""
        # Check distance
        dist_sq = self.position.distance_squared_to(position)
        if dist_sq > self.params.interaction_radius * self.params.interaction_radius:
            return False

        # Check angle if required
        if self.params.interaction_angle < 360:
            to_position = position - self.position
            if to_position.length_squared() < 0.001:
                return True

            # Calculate angle to position
            angle = math.atan2(to_position.x, to_position.z)
            angle_diff = abs(angle - self.rotation)

            # Normalize to 0-180
            while angle_diff > math.pi:
                angle_diff = abs(angle_diff - 2 * math.pi)

            max_angle = math.radians(self.params.interaction_angle / 2)
            if angle_diff > max_angle:
                return False

        return True

    def get_approach_position(self, slot: SmartObjectSlot) -> Vector3:
        """Get position to navigate to for approaching this slot."""
        slot_pos = self.get_slot_world_position(slot)
        slot_rot = self.get_slot_world_rotation(slot)

        # Position slightly in front of the slot
        approach_dist = self.params.interaction_radius * 0.5
        approach_x = slot_pos.x - math.sin(slot_rot) * approach_dist
        approach_z = slot_pos.z - math.cos(slot_rot) * approach_dist

        return Vector3(approach_x, slot_pos.y, approach_z)


@dataclass
class QueueEntry:
    """Entry in a smart object queue."""
    agent_id: int
    priority: float
    request_time: float
    slot_preference: Optional[int] = None


@dataclass
class InteractionResult:
    """Result of an interaction attempt."""
    success: bool = False
    slot_id: Optional[int] = None
    message: str = ""
    duration: float = 0.0


# =============================================================================
# Cover Point (Specialized Smart Object)
# =============================================================================


@dataclass
class CoverParams:
    """Parameters specific to cover points."""
    cover_type: str = "half"  # "half", "full", "corner_left", "corner_right"
    stand_to_crouch: bool = True  # Can transition between stances
    lean_left: bool = False
    lean_right: bool = False
    vault_over: bool = False
    cover_quality: float = 1.0  # 0-1, how good the cover is
    exposed_directions: Set[str] = field(default_factory=set)  # "front", "left", "right", "back"


class CoverPoint:
    """Cover point for tactical navigation."""

    def __init__(
        self, smart_object: SmartObject,
        cover_params: Optional[CoverParams] = None
    ) -> None:
        """Initialize cover point."""
        self._smart_object = smart_object
        self._cover_params = cover_params or CoverParams()
        smart_object.category = SmartObjectCategory.COVER

    @property
    def smart_object(self) -> SmartObject:
        """Get underlying smart object."""
        return self._smart_object

    @property
    def cover_params(self) -> CoverParams:
        """Get cover parameters."""
        return self._cover_params

    @property
    def position(self) -> Vector3:
        """Get cover position."""
        return self._smart_object.position

    @property
    def rotation(self) -> float:
        """Get cover facing direction."""
        return self._smart_object.rotation

    def is_safe_from(self, threat_position: Vector3) -> bool:
        """Check if this cover provides protection from a threat."""
        # Calculate direction to threat
        to_threat = threat_position - self.position
        if to_threat.length_squared() < 0.001:
            return False

        threat_angle = math.atan2(to_threat.x, to_threat.z)

        # Check if threat is from an exposed direction
        relative_angle = threat_angle - self.rotation

        # Normalize to -pi to pi
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi

        # Determine direction
        if abs(relative_angle) < math.pi / 4:
            direction = "front"
        elif relative_angle > math.pi * 3 / 4 or relative_angle < -math.pi * 3 / 4:
            direction = "back"
        elif relative_angle > 0:
            direction = "right"
        else:
            direction = "left"

        return direction not in self._cover_params.exposed_directions

    def get_fire_position(self, stance: str = "crouch") -> Vector3:
        """Get position for firing from cover."""
        pos = self.position

        # Offset based on cover type and stance
        if self._cover_params.cover_type == "corner_left":
            offset = Vector3(-0.3, 0, 0.3)
        elif self._cover_params.cover_type == "corner_right":
            offset = Vector3(0.3, 0, 0.3)
        else:
            offset = Vector3(0, 0, 0.3)

        # Rotate offset by cover rotation
        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)

        world_offset = Vector3(
            offset.x * cos_r - offset.z * sin_r,
            offset.y,
            offset.x * sin_r + offset.z * cos_r
        )

        return pos + world_offset


# =============================================================================
# Smart Object Manager
# =============================================================================


class SmartObjectManager:
    """
    Manager for all smart objects in a level.

    Handles registration, queries, reservations, and interactions.
    """

    def __init__(self) -> None:
        """Initialize smart object manager."""
        self._objects: Dict[int, SmartObject] = {}
        self._cover_points: Dict[int, CoverPoint] = {}
        self._queues: Dict[int, List[QueueEntry]] = {}
        self._active_interactions: Dict[int, Tuple[int, int, float]] = {}  # agent -> (object, slot, start_time)
        self._next_id = 0

        # Spatial index
        self._spatial_cells: Dict[Tuple[int, int, int], List[int]] = {}
        self._cell_size = DEFAULT_SPATIAL_CELL_SIZE

        # Category index
        self._by_category: Dict[SmartObjectCategory, Set[int]] = {}

    @property
    def object_count(self) -> int:
        """Get total number of smart objects."""
        return len(self._objects)

    @property
    def cover_point_count(self) -> int:
        """Get number of cover points."""
        return len(self._cover_points)

    def _get_cell_key(self, position: Vector3) -> Tuple[int, int, int]:
        """Get spatial cell key for position."""
        return (
            int(position.x / self._cell_size),
            int(position.y / self._cell_size),
            int(position.z / self._cell_size)
        )

    def _add_to_spatial_index(self, obj: SmartObject) -> None:
        """Add object to spatial index."""
        cell = self._get_cell_key(obj.position)
        if cell not in self._spatial_cells:
            self._spatial_cells[cell] = []
        if obj.id not in self._spatial_cells[cell]:
            self._spatial_cells[cell].append(obj.id)

    def _remove_from_spatial_index(self, obj: SmartObject) -> None:
        """Remove object from spatial index."""
        cell = self._get_cell_key(obj.position)
        if cell in self._spatial_cells and obj.id in self._spatial_cells[cell]:
            self._spatial_cells[cell].remove(obj.id)

    def _add_to_category_index(self, obj: SmartObject) -> None:
        """Add object to category index."""
        if obj.category not in self._by_category:
            self._by_category[obj.category] = set()
        self._by_category[obj.category].add(obj.id)

    def _remove_from_category_index(self, obj: SmartObject) -> None:
        """Remove object from category index."""
        if obj.category in self._by_category:
            self._by_category[obj.category].discard(obj.id)

    def register(self, obj: SmartObject) -> int:
        """Register a smart object."""
        if obj.id == 0:
            self._next_id += 1
            obj.id = self._next_id

        self._objects[obj.id] = obj
        self._queues[obj.id] = []
        self._add_to_spatial_index(obj)
        self._add_to_category_index(obj)

        return obj.id

    def create_object(
        self, name: str,
        position: Vector3,
        category: SmartObjectCategory = SmartObjectCategory.GENERIC,
        rotation: float = 0.0,
        params: Optional[SmartObjectParams] = None,
        num_slots: int = 1
    ) -> int:
        """Create and register a new smart object."""
        self._next_id += 1

        slots = [SmartObjectSlot(id=i) for i in range(num_slots)]

        obj = SmartObject(
            id=self._next_id,
            name=name,
            category=category,
            position=position,
            rotation=rotation,
            params=params or SmartObjectParams(),
            slots=slots
        )

        return self.register(obj)

    def create_cover_point(
        self, position: Vector3,
        rotation: float = 0.0,
        cover_params: Optional[CoverParams] = None,
        name: str = "Cover"
    ) -> int:
        """Create a cover point."""
        obj_id = self.create_object(
            name=name,
            position=position,
            category=SmartObjectCategory.COVER,
            rotation=rotation
        )

        obj = self._objects[obj_id]
        cover = CoverPoint(obj, cover_params)
        self._cover_points[obj_id] = cover

        return obj_id

    def unregister(self, obj_id: int) -> bool:
        """Unregister a smart object."""
        obj = self._objects.get(obj_id)
        if obj is None:
            return False

        self._remove_from_spatial_index(obj)
        self._remove_from_category_index(obj)
        del self._objects[obj_id]
        self._queues.pop(obj_id, None)
        self._cover_points.pop(obj_id, None)

        return True

    def get_object(self, obj_id: int) -> Optional[SmartObject]:
        """Get smart object by ID."""
        return self._objects.get(obj_id)

    def get_cover_point(self, obj_id: int) -> Optional[CoverPoint]:
        """Get cover point by ID."""
        return self._cover_points.get(obj_id)

    def find_objects_in_radius(
        self, position: Vector3, radius: float,
        category: Optional[SmartObjectCategory] = None
    ) -> List[SmartObject]:
        """Find all objects within radius of position."""
        result = []
        radius_sq = radius * radius

        # Determine cells to check
        cell = self._get_cell_key(position)
        cell_range = int(math.ceil(radius / self._cell_size)) + 1

        cells_to_check = [
            (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            for dx in range(-cell_range, cell_range + 1)
            for dy in range(-cell_range, cell_range + 1)
            for dz in range(-cell_range, cell_range + 1)
        ]

        checked_ids: Set[int] = set()

        for check_cell in cells_to_check:
            if check_cell not in self._spatial_cells:
                continue

            for obj_id in self._spatial_cells[check_cell]:
                if obj_id in checked_ids:
                    continue
                checked_ids.add(obj_id)

                obj = self._objects.get(obj_id)
                if obj is None or not obj.enabled:
                    continue

                if category is not None and obj.category != category:
                    continue

                dist_sq = position.distance_squared_to(obj.position)
                if dist_sq <= radius_sq:
                    result.append(obj)

        return result

    def find_objects_by_category(
        self, category: SmartObjectCategory
    ) -> List[SmartObject]:
        """Find all objects of a category."""
        if category not in self._by_category:
            return []

        return [
            self._objects[obj_id]
            for obj_id in self._by_category[category]
            if obj_id in self._objects and self._objects[obj_id].enabled
        ]

    def find_nearest_object(
        self, position: Vector3,
        category: Optional[SmartObjectCategory] = None,
        max_distance: float = float('inf'),
        require_available: bool = False
    ) -> Optional[SmartObject]:
        """Find nearest object, optionally filtered."""
        best_obj = None
        best_dist_sq = max_distance * max_distance

        for obj in self._objects.values():
            if not obj.enabled:
                continue

            if category is not None and obj.category != category:
                continue

            if require_available and obj.available_slot_count == 0:
                continue

            dist_sq = position.distance_squared_to(obj.position)
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_obj = obj

        return best_obj

    def find_cover_from_threat(
        self, agent_position: Vector3,
        threat_position: Vector3,
        max_distance: float = 20.0
    ) -> List[CoverPoint]:
        """Find cover points that provide protection from threat."""
        result = []

        for cover in self._cover_points.values():
            # Check distance from agent
            dist = agent_position.distance_to(cover.position)
            if dist > max_distance:
                continue

            # Check if cover is safe from threat
            if cover.is_safe_from(threat_position):
                # Check availability
                if cover.smart_object.available_slot_count > 0:
                    result.append(cover)

        # Sort by distance
        result.sort(key=lambda c: agent_position.distance_to(c.position))

        return result

    def reserve_slot(
        self, obj_id: int, agent_id: int,
        slot_id: Optional[int] = None,
        priority: float = 1.0
    ) -> InteractionResult:
        """Reserve a slot on a smart object."""
        result = InteractionResult()

        obj = self._objects.get(obj_id)
        if obj is None:
            result.message = "Object not found"
            return result

        if not obj.enabled:
            result.message = "Object is disabled"
            return result

        # Find slot to reserve
        if slot_id is not None:
            slot = obj.get_slot(slot_id)
            if slot is None:
                result.message = "Slot not found"
                return result
            if not slot.is_available():
                result.message = "Slot not available"
                return result
        else:
            slot = obj.get_first_available_slot()
            if slot is None:
                # Add to queue
                if len(self._queues[obj_id]) < obj.params.max_queue_size:
                    entry = QueueEntry(
                        agent_id=agent_id,
                        priority=priority,
                        request_time=0  # Should be current time
                    )
                    self._queues[obj_id].append(entry)
                    self._queues[obj_id].sort(key=lambda e: -e.priority)
                    result.message = "Added to queue"
                else:
                    result.message = "Queue is full"
                return result

        # Reserve the slot
        slot.state = SlotState.RESERVED
        slot.reserved_by = agent_id
        slot.reservation_time = 0  # Should be current time

        result.success = True
        result.slot_id = slot.id
        result.message = "Slot reserved"

        return result

    def occupy_slot(
        self, obj_id: int, agent_id: int, slot_id: int
    ) -> InteractionResult:
        """Occupy a previously reserved slot."""
        result = InteractionResult()

        obj = self._objects.get(obj_id)
        if obj is None:
            result.message = "Object not found"
            return result

        slot = obj.get_slot(slot_id)
        if slot is None:
            result.message = "Slot not found"
            return result

        if not slot.is_reserved() or slot.reserved_by != agent_id:
            result.message = "Slot not reserved by this agent"
            return result

        slot.state = SlotState.OCCUPIED
        slot.occupied_by = agent_id

        # Track active interaction
        self._active_interactions[agent_id] = (obj_id, slot_id, 0)

        result.success = True
        result.slot_id = slot_id
        result.duration = obj.params.interaction_duration
        result.message = "Slot occupied"

        return result

    def release_slot(
        self, obj_id: int, agent_id: int, slot_id: int
    ) -> bool:
        """Release a reserved or occupied slot."""
        obj = self._objects.get(obj_id)
        if obj is None:
            return False

        slot = obj.get_slot(slot_id)
        if slot is None:
            return False

        if slot.reserved_by != agent_id and slot.occupied_by != agent_id:
            return False

        slot.state = SlotState.AVAILABLE
        slot.reserved_by = None
        slot.occupied_by = None

        # Remove from active interactions
        self._active_interactions.pop(agent_id, None)

        # Check queue
        self._process_queue(obj_id)

        return True

    def cancel_reservation(self, obj_id: int, agent_id: int) -> bool:
        """Cancel a reservation (including queue position)."""
        obj = self._objects.get(obj_id)
        if obj is None:
            return False

        # Check slots
        for slot in obj.slots:
            if slot.reserved_by == agent_id:
                slot.state = SlotState.AVAILABLE
                slot.reserved_by = None
                slot.reservation_time = 0
                self._process_queue(obj_id)
                return True

        # Check queue
        queue = self._queues.get(obj_id, [])
        for i, entry in enumerate(queue):
            if entry.agent_id == agent_id:
                queue.pop(i)
                return True

        return False

    def _process_queue(self, obj_id: int) -> None:
        """Process queue when a slot becomes available."""
        queue = self._queues.get(obj_id, [])
        if not queue:
            return

        obj = self._objects.get(obj_id)
        if obj is None:
            return

        slot = obj.get_first_available_slot()
        if slot is None:
            return

        # Get next from queue
        entry = queue.pop(0)

        # Reserve slot for them
        slot.state = SlotState.RESERVED
        slot.reserved_by = entry.agent_id

    def update(self, dt: float, current_time: float) -> None:
        """
        Update all smart objects.

        Handles reservation timeouts and queue processing.
        """
        for obj_id, obj in self._objects.items():
            # Check reservation timeouts
            for slot in obj.slots:
                if slot.is_reserved():
                    slot.reservation_time += dt
                    if slot.reservation_time > obj.params.reservation_timeout:
                        # Timeout - release reservation
                        slot.state = SlotState.AVAILABLE
                        slot.reserved_by = None
                        slot.reservation_time = 0
                        self._process_queue(obj_id)

    def get_queue_position(self, obj_id: int, agent_id: int) -> int:
        """Get agent's position in queue (-1 if not in queue)."""
        queue = self._queues.get(obj_id, [])
        for i, entry in enumerate(queue):
            if entry.agent_id == agent_id:
                return i
        return -1

    def get_queue_length(self, obj_id: int) -> int:
        """Get current queue length for an object."""
        return len(self._queues.get(obj_id, []))

    def enable_object(self, obj_id: int, enabled: bool = True) -> bool:
        """Enable or disable a smart object."""
        obj = self._objects.get(obj_id)
        if obj is None:
            return False
        obj.enabled = enabled
        return True

    def enable_slot(self, obj_id: int, slot_id: int, enabled: bool = True) -> bool:
        """Enable or disable a specific slot."""
        obj = self._objects.get(obj_id)
        if obj is None:
            return False

        slot = obj.get_slot(slot_id)
        if slot is None:
            return False

        if enabled:
            if slot.state == SlotState.DISABLED:
                slot.state = SlotState.AVAILABLE
        else:
            slot.state = SlotState.DISABLED
            slot.reserved_by = None
            slot.occupied_by = None

        return True

    def is_agent_interacting(self, agent_id: int) -> bool:
        """Check if agent is currently interacting with any object."""
        return agent_id in self._active_interactions

    def get_agent_interaction(self, agent_id: int) -> Optional[Tuple[int, int]]:
        """Get object and slot agent is interacting with."""
        interaction = self._active_interactions.get(agent_id)
        if interaction is None:
            return None
        return (interaction[0], interaction[1])
