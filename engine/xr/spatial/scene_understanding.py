"""Semantic scene understanding for AR.

Provides scene labeling, room classification, object segmentation,
and high-level understanding of the physical environment.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat


class SemanticLabel(Enum):
    """Semantic labels for scene regions."""
    UNKNOWN = auto()
    FLOOR = auto()
    CEILING = auto()
    WALL = auto()
    TABLE = auto()
    CHAIR = auto()
    COUCH = auto()
    BED = auto()
    DOOR = auto()
    WINDOW = auto()
    SCREEN = auto()
    LAMP = auto()
    PLANT = auto()
    STORAGE = auto()
    APPLIANCE = auto()
    PLATFORM = auto()
    OBSTACLE = auto()
    PERSON = auto()
    PET = auto()


class RoomType(Enum):
    """Classification of room types."""
    UNKNOWN = auto()
    LIVING_ROOM = auto()
    BEDROOM = auto()
    KITCHEN = auto()
    BATHROOM = auto()
    OFFICE = auto()
    DINING_ROOM = auto()
    HALLWAY = auto()
    GARAGE = auto()
    OUTDOOR = auto()


class OcclusionMode(Enum):
    """Mode for AR occlusion rendering."""
    NONE = auto()          # No occlusion
    DEPTH_BASED = auto()   # Use depth buffer
    MESH_BASED = auto()    # Use spatial mesh
    HUMAN_ONLY = auto()    # Only occlude behind humans
    FULL = auto()          # Full scene occlusion


@dataclass(slots=True)
class SemanticRegion:
    """A region with semantic meaning in the scene."""
    region_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: SemanticLabel = SemanticLabel.UNKNOWN
    confidence: float = 0.0
    center: Vec3 = field(default_factory=Vec3.zero)
    bounds_min: Vec3 = field(default_factory=Vec3.zero)
    bounds_max: Vec3 = field(default_factory=Vec3.zero)
    normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    area: float = 0.0
    is_dynamic: bool = False
    last_updated: float = 0.0

    @property
    def size(self) -> Vec3:
        """Get the size of the region."""
        return self.bounds_max - self.bounds_min

    def contains_point(self, point: Vec3) -> bool:
        """Check if a point is inside this region.

        Args:
            point: Point to test

        Returns:
            True if point is inside
        """
        return (
            self.bounds_min.x <= point.x <= self.bounds_max.x and
            self.bounds_min.y <= point.y <= self.bounds_max.y and
            self.bounds_min.z <= point.z <= self.bounds_max.z
        )


@dataclass(slots=True)
class RoomBounds:
    """Bounds and properties of a detected room."""
    room_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    room_type: RoomType = RoomType.UNKNOWN
    confidence: float = 0.0
    center: Vec3 = field(default_factory=Vec3.zero)
    dimensions: Vec3 = field(default_factory=Vec3.zero)  # Width, Height, Depth
    floor_level: float = 0.0
    ceiling_level: float = 2.5
    wall_count: int = 0
    door_count: int = 0
    window_count: int = 0

    @property
    def floor_area(self) -> float:
        """Get the floor area in square meters."""
        return self.dimensions.x * self.dimensions.z

    @property
    def volume(self) -> float:
        """Get the room volume in cubic meters."""
        return self.dimensions.x * self.dimensions.y * self.dimensions.z


@dataclass(slots=True)
class SceneObject:
    """A detected object in the scene."""
    object_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: SemanticLabel = SemanticLabel.UNKNOWN
    confidence: float = 0.0
    position: Vec3 = field(default_factory=Vec3.zero)
    orientation: Quat = field(default_factory=Quat.identity)
    bounds_min: Vec3 = field(default_factory=Vec3.zero)
    bounds_max: Vec3 = field(default_factory=Vec3.zero)
    is_movable: bool = True
    is_interactable: bool = False
    last_seen: float = 0.0

    @property
    def size(self) -> Vec3:
        """Get the object size."""
        return self.bounds_max - self.bounds_min

    @property
    def center(self) -> Vec3:
        """Get the center of the object."""
        return (self.bounds_min + self.bounds_max) * 0.5


@dataclass(slots=True)
class HumanSegment:
    """Segmentation data for a detected human."""
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    position: Vec3 = field(default_factory=Vec3.zero)
    bounds_min: Vec3 = field(default_factory=Vec3.zero)
    bounds_max: Vec3 = field(default_factory=Vec3.zero)
    confidence: float = 0.0
    is_user: bool = False  # Is this the device user
    depth_available: bool = True
    last_updated: float = 0.0


@dataclass(slots=True)
class LightEstimate:
    """Estimated ambient lighting conditions."""
    ambient_intensity: float = 1.0
    ambient_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    main_light_direction: Vec3 = field(default_factory=lambda: Vec3(0, -1, 0))
    main_light_intensity: float = 1.0
    main_light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    spherical_harmonics: Optional[list[float]] = None
    environment_texture_available: bool = False
    timestamp: float = 0.0


@dataclass(slots=True)
class SceneUnderstandingConfig:
    """Configuration for scene understanding."""
    semantic_labeling_enabled: bool = True
    room_classification_enabled: bool = True
    object_detection_enabled: bool = True
    human_segmentation_enabled: bool = True
    light_estimation_enabled: bool = True
    occlusion_mode: OcclusionMode = OcclusionMode.DEPTH_BASED
    update_rate: float = 10.0  # Hz
    min_region_area: float = 0.1  # m2
    min_object_confidence: float = 0.5


class SceneUnderstanding:
    """Semantic scene understanding system.

    Provides high-level understanding of the physical environment
    including labeling, room detection, and object recognition.

    Attributes:
        config: Understanding configuration
        regions: Detected semantic regions
        room: Current room bounds
        objects: Detected objects
    """
    __slots__ = (
        '_config',
        '_regions',
        '_room',
        '_objects',
        '_humans',
        '_light_estimate',
        '_is_running',
        '_is_ready',
        '_last_update',
        '_callbacks',
    )

    def __init__(self, config: Optional[SceneUnderstandingConfig] = None) -> None:
        """Initialize scene understanding.

        Args:
            config: Understanding configuration
        """
        self._config: SceneUnderstandingConfig = config or SceneUnderstandingConfig()
        self._regions: dict[str, SemanticRegion] = {}
        self._room: Optional[RoomBounds] = None
        self._objects: dict[str, SceneObject] = {}
        self._humans: dict[str, HumanSegment] = {}
        self._light_estimate: LightEstimate = LightEstimate()
        self._is_running: bool = False
        self._is_ready: bool = False
        self._last_update: float = 0.0
        self._callbacks: dict[str, list[Callable]] = {
            "scene_updated": [],
            "room_detected": [],
            "object_detected": [],
            "object_lost": [],
            "human_detected": [],
            "light_updated": [],
        }

    @property
    def config(self) -> SceneUnderstandingConfig:
        """Get the configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if understanding is active."""
        return self._is_running

    @property
    def is_ready(self) -> bool:
        """Check if scene data is ready."""
        return self._is_ready

    @property
    def room(self) -> Optional[RoomBounds]:
        """Get the current room bounds."""
        return self._room

    @property
    def light_estimate(self) -> LightEstimate:
        """Get the current light estimate."""
        return self._light_estimate

    @property
    def occlusion_mode(self) -> OcclusionMode:
        """Get the current occlusion mode."""
        return self._config.occlusion_mode

    @occlusion_mode.setter
    def occlusion_mode(self, mode: OcclusionMode) -> None:
        """Set the occlusion mode."""
        self._config.occlusion_mode = mode

    def start(self) -> bool:
        """Start scene understanding.

        Returns:
            True if started successfully
        """
        if self._is_running:
            return False
        self._is_running = True
        return True

    def stop(self) -> bool:
        """Stop scene understanding.

        Returns:
            True if stopped successfully
        """
        if not self._is_running:
            return False
        self._is_running = False
        return True

    def update(self, timestamp: float) -> None:
        """Update scene understanding.

        Args:
            timestamp: Current time
        """
        if not self._is_running:
            return

        self._last_update = timestamp

        # In a real implementation, this would process sensor data
        # and update regions, objects, etc.

    # Semantic region methods

    def get_region(self, region_id: str) -> Optional[SemanticRegion]:
        """Get a semantic region by ID.

        Args:
            region_id: Region identifier

        Returns:
            Region if found, None otherwise
        """
        return self._regions.get(region_id)

    def get_all_regions(self) -> list[SemanticRegion]:
        """Get all semantic regions.

        Returns:
            List of all regions
        """
        return list(self._regions.values())

    def get_regions_by_label(self, label: SemanticLabel) -> list[SemanticRegion]:
        """Get regions with a specific label.

        Args:
            label: Semantic label to filter by

        Returns:
            List of matching regions
        """
        return [r for r in self._regions.values() if r.label == label]

    def get_floor_regions(self) -> list[SemanticRegion]:
        """Get all floor regions.

        Returns:
            List of floor regions
        """
        return self.get_regions_by_label(SemanticLabel.FLOOR)

    def get_wall_regions(self) -> list[SemanticRegion]:
        """Get all wall regions.

        Returns:
            List of wall regions
        """
        return self.get_regions_by_label(SemanticLabel.WALL)

    def get_region_at_point(self, point: Vec3) -> Optional[SemanticRegion]:
        """Get the region containing a point.

        Args:
            point: Point to test

        Returns:
            Region containing point, or None
        """
        for region in self._regions.values():
            if region.contains_point(point):
                return region
        return None

    def add_region(self, region: SemanticRegion) -> None:
        """Add a semantic region.

        Args:
            region: Region to add
        """
        self._regions[region.region_id] = region
        self._notify_callbacks("scene_updated")

    def remove_region(self, region_id: str) -> bool:
        """Remove a semantic region.

        Args:
            region_id: Region to remove

        Returns:
            True if removed
        """
        if region_id in self._regions:
            del self._regions[region_id]
            return True
        return False

    # Room methods

    def set_room(self, room: RoomBounds) -> None:
        """Set the current room bounds.

        Args:
            room: Room bounds data
        """
        self._room = room
        self._notify_callbacks("room_detected")

    def classify_room(self) -> RoomType:
        """Classify the current room type.

        Returns:
            Detected room type
        """
        if not self._room:
            return RoomType.UNKNOWN

        # Simple heuristic classification
        # In a real implementation, this would use ML

        floor_area = self._room.floor_area

        # Check for specific objects
        has_bed = any(
            o.label == SemanticLabel.BED
            for o in self._objects.values()
        )
        has_couch = any(
            o.label == SemanticLabel.COUCH
            for o in self._objects.values()
        )
        has_table = any(
            o.label == SemanticLabel.TABLE
            for o in self._objects.values()
        )

        if has_bed:
            return RoomType.BEDROOM
        elif has_couch:
            return RoomType.LIVING_ROOM
        elif floor_area < 5.0:
            return RoomType.BATHROOM
        elif has_table and floor_area > 10.0:
            return RoomType.DINING_ROOM
        else:
            return RoomType.UNKNOWN

    # Object methods

    def get_object(self, object_id: str) -> Optional[SceneObject]:
        """Get a scene object by ID.

        Args:
            object_id: Object identifier

        Returns:
            Object if found, None otherwise
        """
        return self._objects.get(object_id)

    def get_all_objects(self) -> list[SceneObject]:
        """Get all detected objects.

        Returns:
            List of all objects
        """
        return list(self._objects.values())

    def get_objects_by_label(self, label: SemanticLabel) -> list[SceneObject]:
        """Get objects with a specific label.

        Args:
            label: Semantic label to filter by

        Returns:
            List of matching objects
        """
        return [o for o in self._objects.values() if o.label == label]

    def get_objects_near(
        self,
        position: Vec3,
        radius: float,
    ) -> list[SceneObject]:
        """Get objects near a position.

        Args:
            position: Query position
            radius: Search radius

        Returns:
            List of nearby objects
        """
        results = []
        for obj in self._objects.values():
            if obj.position.distance(position) <= radius:
                results.append(obj)
        return results

    def get_interactable_objects(self) -> list[SceneObject]:
        """Get all interactable objects.

        Returns:
            List of interactable objects
        """
        return [o for o in self._objects.values() if o.is_interactable]

    def add_object(self, obj: SceneObject) -> None:
        """Add a detected object.

        Args:
            obj: Object to add
        """
        self._objects[obj.object_id] = obj
        self._notify_callbacks("object_detected", obj)

    def remove_object(self, object_id: str) -> bool:
        """Remove an object.

        Args:
            object_id: Object to remove

        Returns:
            True if removed
        """
        obj = self._objects.pop(object_id, None)
        if obj:
            self._notify_callbacks("object_lost", obj)
            return True
        return False

    # Human segmentation methods

    def get_humans(self) -> list[HumanSegment]:
        """Get all detected humans.

        Returns:
            List of human segments
        """
        return list(self._humans.values())

    def get_user_segment(self) -> Optional[HumanSegment]:
        """Get the device user's segment.

        Returns:
            User segment if detected
        """
        for human in self._humans.values():
            if human.is_user:
                return human
        return None

    def add_human(self, human: HumanSegment) -> None:
        """Add a detected human.

        Args:
            human: Human segment to add
        """
        self._humans[human.segment_id] = human
        self._notify_callbacks("human_detected", human)

    def remove_human(self, segment_id: str) -> bool:
        """Remove a human segment.

        Args:
            segment_id: Segment to remove

        Returns:
            True if removed
        """
        return self._humans.pop(segment_id, None) is not None

    def is_point_occluded_by_human(self, point: Vec3) -> bool:
        """Check if a point is behind a detected human.

        Args:
            point: Point to test

        Returns:
            True if point is occluded by a human
        """
        for human in self._humans.values():
            if human.bounds_min.x <= point.x <= human.bounds_max.x and \
               human.bounds_min.y <= point.y <= human.bounds_max.y:
                # Check depth (z)
                if point.z > human.position.z:
                    return True
        return False

    # Light estimation methods

    def update_light_estimate(self, estimate: LightEstimate) -> None:
        """Update the light estimate.

        Args:
            estimate: New light estimate
        """
        self._light_estimate = estimate
        self._notify_callbacks("light_updated")

    def get_ambient_light(self) -> tuple[float, tuple[float, float, float]]:
        """Get ambient light intensity and color.

        Returns:
            Tuple of (intensity, (r, g, b))
        """
        return (
            self._light_estimate.ambient_intensity,
            self._light_estimate.ambient_color,
        )

    def get_main_light(
        self,
    ) -> tuple[Vec3, float, tuple[float, float, float]]:
        """Get main directional light.

        Returns:
            Tuple of (direction, intensity, (r, g, b))
        """
        return (
            self._light_estimate.main_light_direction,
            self._light_estimate.main_light_intensity,
            self._light_estimate.main_light_color,
        )

    # Placement helpers

    def find_floor_position(
        self,
        near_position: Vec3,
        search_radius: float = 2.0,
    ) -> Optional[Vec3]:
        """Find a floor position for placement.

        Args:
            near_position: Preferred position
            search_radius: Search radius

        Returns:
            Floor position if found
        """
        floor_regions = self.get_floor_regions()

        best_position: Optional[Vec3] = None
        best_distance = search_radius

        for region in floor_regions:
            # Project to floor level
            floor_pos = Vec3(near_position.x, region.center.y, near_position.z)

            if region.contains_point(floor_pos):
                distance = near_position.distance(floor_pos)
                if distance < best_distance:
                    best_distance = distance
                    best_position = floor_pos

        return best_position

    def find_wall_position(
        self,
        near_position: Vec3,
        search_radius: float = 2.0,
    ) -> Optional[tuple[Vec3, Vec3]]:
        """Find a wall position for placement.

        Args:
            near_position: Preferred position
            search_radius: Search radius

        Returns:
            Tuple of (position, normal) if found
        """
        wall_regions = self.get_wall_regions()

        best_position: Optional[Vec3] = None
        best_normal: Optional[Vec3] = None
        best_distance = search_radius

        for region in wall_regions:
            # Project to wall plane
            to_point = near_position - region.center
            distance = to_point.dot(region.normal)
            wall_pos = near_position - region.normal * distance

            if abs(distance) < best_distance:
                best_distance = abs(distance)
                best_position = wall_pos
                best_normal = region.normal

        if best_position and best_normal:
            return (best_position, best_normal)
        return None

    def find_table_position(
        self,
        near_position: Vec3,
        search_radius: float = 2.0,
    ) -> Optional[Vec3]:
        """Find a table/surface position for placement.

        Args:
            near_position: Preferred position
            search_radius: Search radius

        Returns:
            Table surface position if found
        """
        table_regions = self.get_regions_by_label(SemanticLabel.TABLE)

        best_position: Optional[Vec3] = None
        best_distance = search_radius

        for region in table_regions:
            if region.contains_point(near_position):
                surface_pos = Vec3(near_position.x, region.center.y, near_position.z)
                distance = near_position.distance(surface_pos)
                if distance < best_distance:
                    best_distance = distance
                    best_position = surface_pos

        return best_position

    # Occlusion methods

    def is_point_occluded(self, point: Vec3, camera_position: Vec3) -> bool:
        """Check if a point should be occluded.

        Args:
            point: Point in world space
            camera_position: Camera/viewer position

        Returns:
            True if point should be occluded
        """
        mode = self._config.occlusion_mode

        if mode == OcclusionMode.NONE:
            return False
        elif mode == OcclusionMode.HUMAN_ONLY:
            return self.is_point_occluded_by_human(point)
        elif mode in (OcclusionMode.DEPTH_BASED, OcclusionMode.MESH_BASED, OcclusionMode.FULL):
            # Would use depth buffer or mesh in real implementation
            return False

        return False

    # Callback methods

    def add_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for scene events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def remove_callback(self, event: str, callback: Callable) -> None:
        """Remove a registered callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def _notify_callbacks(self, event: str, data: object = None) -> None:
        """Notify callbacks for an event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                if data is not None:
                    callback(data)
                else:
                    callback()

    def clear(self) -> None:
        """Clear all scene understanding data."""
        self._regions.clear()
        self._room = None
        self._objects.clear()
        self._humans.clear()
        self._light_estimate = LightEstimate()
        self._is_ready = False
