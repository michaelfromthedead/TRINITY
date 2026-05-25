"""AR 3D object tracking.

Provides detection and tracking of 3D objects in the real world
for advanced AR experiences and object recognition.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat


class ObjectTrackingState(Enum):
    """Tracking state of a 3D object."""
    NONE = auto()           # Not detected
    DETECTING = auto()      # Detection in progress
    TRACKING = auto()       # Actively tracked
    LIMITED = auto()        # Tracked with reduced quality
    EXTENDED = auto()       # Extended tracking
    LOST = auto()           # Tracking lost


class ObjectTrackingQuality(Enum):
    """Quality level of object tracking."""
    LOW = auto()       # Basic tracking
    MEDIUM = auto()    # Standard tracking
    HIGH = auto()      # High-quality tracking
    ULTRA = auto()     # Maximum quality


@dataclass(slots=True)
class ObjectBounds:
    """Bounding box for a 3D object."""
    center: Vec3 = field(default_factory=Vec3.zero)
    size: Vec3 = field(default_factory=lambda: Vec3(1, 1, 1))
    orientation: Quat = field(default_factory=Quat.identity)

    @property
    def extents(self) -> Vec3:
        """Get the half-size of the bounds."""
        return self.size * 0.5

    @property
    def min_point(self) -> Vec3:
        """Get the minimum corner in local space."""
        return self.center - self.extents

    @property
    def max_point(self) -> Vec3:
        """Get the maximum corner in local space."""
        return self.center + self.extents

    def contains_point(self, point: Vec3) -> bool:
        """Check if a point is inside the bounds.

        Args:
            point: Point to test (in local space)

        Returns:
            True if point is inside
        """
        e = self.extents
        rel = point - self.center
        return (
            abs(rel.x) <= e.x and
            abs(rel.y) <= e.y and
            abs(rel.z) <= e.z
        )


@dataclass(slots=True)
class ObjectReference:
    """Reference 3D object in the tracking database."""
    reference_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    bounds: ObjectBounds = field(default_factory=ObjectBounds)
    model_path: str = ""
    feature_data: Optional[bytes] = None
    feature_count: int = 0
    is_enabled: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class ObjectPose:
    """Pose data for a tracked object."""
    position: Vec3 = field(default_factory=Vec3.zero)
    orientation: Quat = field(default_factory=Quat.identity)
    scale: Vec3 = field(default_factory=lambda: Vec3(1, 1, 1))
    timestamp: float = 0.0

    def transform_point(self, local_point: Vec3) -> Vec3:
        """Transform a local point to world space.

        Args:
            local_point: Point in object local space

        Returns:
            Point in world space
        """
        # Apply scale
        scaled = Vec3(
            local_point.x * self.scale.x,
            local_point.y * self.scale.y,
            local_point.z * self.scale.z,
        )
        # Apply rotation and translation
        rotated = self.orientation.rotate_vector(scaled)
        return rotated + self.position

    def inverse_transform_point(self, world_point: Vec3) -> Vec3:
        """Transform a world point to local space.

        Args:
            world_point: Point in world space

        Returns:
            Point in object local space
        """
        relative = world_point - self.position
        inv_rot = self.orientation.inverse()
        rotated = inv_rot.rotate_vector(relative)
        # Remove scale
        return Vec3(
            rotated.x / self.scale.x if self.scale.x != 0 else 0,
            rotated.y / self.scale.y if self.scale.y != 0 else 0,
            rotated.z / self.scale.z if self.scale.z != 0 else 0,
        )


class TrackedObject:
    """A tracked 3D object in the scene.

    Represents a reference 3D object that can be detected and tracked
    in the real world for AR content alignment.

    Attributes:
        object_id: Unique identifier
        reference: Reference object data
        tracking_state: Current tracking status
        pose: World position, orientation, and scale
    """
    __slots__ = (
        '_object_id',
        '_reference',
        '_tracking_state',
        '_tracking_quality',
        '_pose',
        '_confidence',
        '_detected_bounds',
        '_is_active',
        '_native_handle',
        '_last_seen',
        '_first_detected',
        '_occlusion_enabled',
        '_callbacks',
    )

    def __init__(
        self,
        reference: ObjectReference,
        tracking_quality: ObjectTrackingQuality = ObjectTrackingQuality.MEDIUM,
    ) -> None:
        """Initialize a tracked object.

        Args:
            reference: Reference object data
            tracking_quality: Desired tracking quality
        """
        self._object_id: str = str(uuid.uuid4())
        self._reference: ObjectReference = reference
        self._tracking_state: ObjectTrackingState = ObjectTrackingState.NONE
        self._tracking_quality: ObjectTrackingQuality = tracking_quality
        self._pose: ObjectPose = ObjectPose()
        self._confidence: float = 0.0
        self._detected_bounds: ObjectBounds = ObjectBounds()
        self._is_active: bool = False
        self._native_handle: Optional[int] = None
        self._last_seen: float = 0.0
        self._first_detected: float = 0.0
        self._occlusion_enabled: bool = True
        self._callbacks: dict[str, list[Callable]] = {
            "detected": [],
            "tracking": [],
            "lost": [],
            "pose_updated": [],
        }

    @property
    def object_id(self) -> str:
        """Get the unique object identifier."""
        return self._object_id

    @property
    def reference_id(self) -> str:
        """Get the reference object ID."""
        return self._reference.reference_id

    @property
    def reference(self) -> ObjectReference:
        """Get the reference object data."""
        return self._reference

    @property
    def name(self) -> str:
        """Get the object name."""
        return self._reference.name

    @property
    def tracking_state(self) -> ObjectTrackingState:
        """Get the current tracking state."""
        return self._tracking_state

    @property
    def tracking_quality(self) -> ObjectTrackingQuality:
        """Get the tracking quality setting."""
        return self._tracking_quality

    @property
    def is_tracked(self) -> bool:
        """Check if the object is actively tracked."""
        return self._tracking_state == ObjectTrackingState.TRACKING

    @property
    def is_visible(self) -> bool:
        """Check if the object is currently visible."""
        return self._tracking_state in (
            ObjectTrackingState.TRACKING,
            ObjectTrackingState.LIMITED,
        )

    @property
    def position(self) -> Vec3:
        """Get the world position."""
        return self._pose.position

    @property
    def orientation(self) -> Quat:
        """Get the world orientation."""
        return self._pose.orientation

    @property
    def scale(self) -> Vec3:
        """Get the world scale."""
        return self._pose.scale

    @property
    def pose(self) -> ObjectPose:
        """Get the full pose data."""
        return self._pose

    @property
    def confidence(self) -> float:
        """Get tracking confidence (0.0 to 1.0)."""
        return self._confidence

    @property
    def bounds(self) -> ObjectBounds:
        """Get the detected world-space bounds."""
        return self._detected_bounds

    @property
    def is_active(self) -> bool:
        """Check if the object is active for tracking."""
        return self._is_active

    @property
    def last_seen_time(self) -> float:
        """Get when the object was last seen."""
        return self._last_seen

    @property
    def occlusion_enabled(self) -> bool:
        """Check if occlusion is enabled for this object."""
        return self._occlusion_enabled

    @occlusion_enabled.setter
    def occlusion_enabled(self, value: bool) -> None:
        """Enable or disable occlusion."""
        self._occlusion_enabled = value

    def activate(self) -> bool:
        """Activate this object for tracking.

        Returns:
            True if activated successfully
        """
        if self._is_active:
            return False
        self._is_active = True
        self._tracking_state = ObjectTrackingState.DETECTING
        return True

    def deactivate(self) -> bool:
        """Deactivate this object.

        Returns:
            True if deactivated successfully
        """
        if not self._is_active:
            return False
        self._is_active = False
        self._tracking_state = ObjectTrackingState.NONE
        return True

    def update_pose(
        self,
        position: Vec3,
        orientation: Quat,
        scale: Vec3,
        confidence: float,
        timestamp: float,
    ) -> None:
        """Update the object pose from tracking.

        Args:
            position: World position
            orientation: World orientation
            scale: World scale
            confidence: Tracking confidence
            timestamp: Update timestamp
        """
        old_state = self._tracking_state

        self._pose.position = position
        self._pose.orientation = orientation
        self._pose.scale = scale
        self._pose.timestamp = timestamp
        self._confidence = max(0.0, min(1.0, confidence))
        self._last_seen = timestamp

        # Update detected bounds
        ref_bounds = self._reference.bounds
        self._detected_bounds = ObjectBounds(
            center=position,
            size=Vec3(
                ref_bounds.size.x * scale.x,
                ref_bounds.size.y * scale.y,
                ref_bounds.size.z * scale.z,
            ),
            orientation=orientation,
        )

        if old_state == ObjectTrackingState.NONE or old_state == ObjectTrackingState.DETECTING:
            self._first_detected = timestamp
            self._tracking_state = ObjectTrackingState.TRACKING
            self._notify_callbacks("detected")
        else:
            self._tracking_state = ObjectTrackingState.TRACKING

        self._notify_callbacks("pose_updated")

    def update_tracking_state(
        self,
        state: ObjectTrackingState,
        confidence: float = 0.0,
    ) -> None:
        """Update the tracking state.

        Args:
            state: New tracking state
            confidence: Optional confidence update
        """
        old_state = self._tracking_state
        self._tracking_state = state

        if confidence > 0.0:
            self._confidence = max(0.0, min(1.0, confidence))

        if old_state != state:
            if state == ObjectTrackingState.TRACKING:
                self._notify_callbacks("tracking")
            elif state == ObjectTrackingState.LOST:
                self._notify_callbacks("lost")

    def get_world_bounds(self) -> ObjectBounds:
        """Get the current world-space bounds.

        Returns:
            Object bounds in world space
        """
        return self._detected_bounds

    def get_corner_positions(self) -> list[Vec3]:
        """Get world positions of the bounding box corners.

        Returns:
            List of 8 corner positions
        """
        e = self._detected_bounds.extents
        corners_local = [
            Vec3(-e.x, -e.y, -e.z),
            Vec3(e.x, -e.y, -e.z),
            Vec3(e.x, -e.y, e.z),
            Vec3(-e.x, -e.y, e.z),
            Vec3(-e.x, e.y, -e.z),
            Vec3(e.x, e.y, -e.z),
            Vec3(e.x, e.y, e.z),
            Vec3(-e.x, e.y, e.z),
        ]
        return [self._pose.transform_point(c) for c in corners_local]

    def local_to_world(self, local_point: Vec3) -> Vec3:
        """Convert a local point to world space.

        Args:
            local_point: Point in object local space

        Returns:
            Point in world space
        """
        return self._pose.transform_point(local_point)

    def world_to_local(self, world_point: Vec3) -> Vec3:
        """Convert a world point to local space.

        Args:
            world_point: Point in world space

        Returns:
            Point in object local space
        """
        return self._pose.inverse_transform_point(world_point)

    def contains_point(self, world_point: Vec3) -> bool:
        """Check if a world point is inside the object bounds.

        Args:
            world_point: Point to test

        Returns:
            True if inside bounds
        """
        local = self.world_to_local(world_point)
        return self._reference.bounds.contains_point(local)

    def add_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for object events.

        Args:
            event: Event name
            callback: Function to call
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def remove_callback(self, event: str, callback: Callable) -> None:
        """Remove a registered callback.

        Args:
            event: Event name
            callback: Callback to remove
        """
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def _notify_callbacks(self, event: str) -> None:
        """Notify callbacks for an event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                callback(self)

    def __repr__(self) -> str:
        return (
            f"TrackedObject(id={self._object_id[:8]}..., "
            f"name='{self._reference.name}', "
            f"state={self._tracking_state.name}, "
            f"confidence={self._confidence:.2f})"
        )


@dataclass(slots=True)
class ObjectTrackerConfig:
    """Configuration for 3D object tracking."""
    max_tracked_objects: int = 3
    tracking_quality: ObjectTrackingQuality = ObjectTrackingQuality.MEDIUM
    enable_extended_tracking: bool = True
    tracking_timeout: float = 3.0
    enable_occlusion: bool = True
    min_confidence_threshold: float = 0.4


class ObjectTracker:
    """Manages 3D object tracking.

    Handles detection, tracking, and lifecycle of 3D object targets
    in the AR scene.

    Attributes:
        config: Tracker configuration
        database: Reference object database
        objects: Active tracking objects
    """
    __slots__ = (
        '_config',
        '_database',
        '_objects',
        '_is_running',
        '_callbacks',
        '_last_update',
    )

    def __init__(self, config: Optional[ObjectTrackerConfig] = None) -> None:
        """Initialize the object tracker.

        Args:
            config: Tracker configuration
        """
        self._config: ObjectTrackerConfig = config or ObjectTrackerConfig()
        self._database: dict[str, ObjectReference] = {}
        self._objects: dict[str, TrackedObject] = {}
        self._is_running: bool = False
        self._callbacks: dict[str, list[Callable]] = {
            "object_detected": [],
            "object_lost": [],
            "tracking_started": [],
            "tracking_stopped": [],
        }
        self._last_update: float = 0.0

    @property
    def config(self) -> ObjectTrackerConfig:
        """Get the tracker configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if tracking is active."""
        return self._is_running

    @property
    def tracked_count(self) -> int:
        """Get the number of currently tracked objects."""
        return len([o for o in self._objects.values() if o.is_tracked])

    @property
    def database_size(self) -> int:
        """Get the number of reference objects."""
        return len(self._database)

    # Database methods

    def add_reference_object(
        self,
        name: str,
        bounds: ObjectBounds,
        model_path: str = "",
        feature_data: Optional[bytes] = None,
        metadata: Optional[dict] = None,
    ) -> ObjectReference:
        """Add a reference object to the database.

        Args:
            name: Object name
            bounds: Object bounding box
            model_path: Path to 3D model
            feature_data: Feature extraction data
            metadata: Optional metadata

        Returns:
            Created reference object
        """
        reference = ObjectReference(
            name=name,
            bounds=bounds,
            model_path=model_path,
            feature_data=feature_data,
            metadata=metadata or {},
        )
        self._database[reference.reference_id] = reference
        return reference

    def remove_reference_object(self, reference_id: str) -> bool:
        """Remove a reference object from the database.

        Args:
            reference_id: Reference to remove

        Returns:
            True if removed
        """
        if reference_id in self._database:
            # Remove any objects using this reference
            objects_to_remove = [
                oid for oid, o in self._objects.items()
                if o.reference_id == reference_id
            ]
            for oid in objects_to_remove:
                self.remove_object(oid)

            del self._database[reference_id]
            return True
        return False

    def get_reference_object(self, reference_id: str) -> Optional[ObjectReference]:
        """Get a reference object by ID.

        Args:
            reference_id: Reference identifier

        Returns:
            Reference if found
        """
        return self._database.get(reference_id)

    def get_all_references(self) -> list[ObjectReference]:
        """Get all reference objects.

        Returns:
            List of all references
        """
        return list(self._database.values())

    # Object methods

    def create_object(
        self,
        reference_id: str,
        tracking_quality: Optional[ObjectTrackingQuality] = None,
    ) -> Optional[TrackedObject]:
        """Create a tracking object from a reference.

        Args:
            reference_id: Reference object ID
            tracking_quality: Optional quality override

        Returns:
            Created object or None if failed
        """
        reference = self._database.get(reference_id)
        if not reference:
            return None

        if not reference.is_enabled:
            return None

        obj = TrackedObject(
            reference=reference,
            tracking_quality=tracking_quality or self._config.tracking_quality,
        )
        self._objects[obj.object_id] = obj
        return obj

    def remove_object(self, object_id: str) -> bool:
        """Remove a tracking object.

        Args:
            object_id: Object to remove

        Returns:
            True if removed
        """
        obj = self._objects.pop(object_id, None)
        if obj:
            obj.deactivate()
            return True
        return False

    def get_object(self, object_id: str) -> Optional[TrackedObject]:
        """Get an object by ID.

        Args:
            object_id: Object identifier

        Returns:
            Object if found
        """
        return self._objects.get(object_id)

    def get_all_objects(self) -> list[TrackedObject]:
        """Get all tracking objects.

        Returns:
            List of all objects
        """
        return list(self._objects.values())

    def get_tracked_objects(self) -> list[TrackedObject]:
        """Get currently tracked objects.

        Returns:
            List of actively tracked objects
        """
        return [o for o in self._objects.values() if o.is_tracked]

    def get_visible_objects(self) -> list[TrackedObject]:
        """Get currently visible objects.

        Returns:
            List of visible objects
        """
        return [o for o in self._objects.values() if o.is_visible]

    def find_object_by_reference(self, reference_id: str) -> Optional[TrackedObject]:
        """Find an object by its reference ID.

        Args:
            reference_id: Reference object ID

        Returns:
            Object if found
        """
        for obj in self._objects.values():
            if obj.reference_id == reference_id:
                return obj
        return None

    def find_objects_at_point(self, point: Vec3) -> list[TrackedObject]:
        """Find objects containing a point.

        Args:
            point: World-space point

        Returns:
            List of objects containing the point
        """
        results = []
        for obj in self._objects.values():
            if obj.is_visible and obj.contains_point(point):
                results.append(obj)
        return results

    # Tracking control

    def start(self) -> bool:
        """Start object tracking.

        Returns:
            True if started successfully
        """
        if self._is_running:
            return False

        self._is_running = True

        for obj in self._objects.values():
            if obj.reference.is_enabled:
                obj.activate()

        self._notify_callbacks("tracking_started")
        return True

    def stop(self) -> bool:
        """Stop object tracking.

        Returns:
            True if stopped successfully
        """
        if not self._is_running:
            return False

        self._is_running = False

        for obj in self._objects.values():
            obj.deactivate()

        self._notify_callbacks("tracking_stopped")
        return True

    def update(self, timestamp: float) -> None:
        """Update object tracking.

        Args:
            timestamp: Current time
        """
        if not self._is_running:
            return

        self._last_update = timestamp

        # Check for tracking timeouts
        for obj in self._objects.values():
            if not obj.is_active:
                continue

            if obj.tracking_state == ObjectTrackingState.EXTENDED:
                if timestamp - obj.last_seen_time > self._config.tracking_timeout:
                    obj.update_tracking_state(ObjectTrackingState.LOST)
                    self._notify_callbacks("object_lost", obj)

    # Callbacks

    def add_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for tracker events."""
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
        """Clear all objects and references."""
        self.stop()
        self._objects.clear()
        self._database.clear()
