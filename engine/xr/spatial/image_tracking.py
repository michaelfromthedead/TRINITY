"""AR image target tracking.

Provides detection and tracking of 2D images in the real world
for AR marker-based experiences.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from engine.core.math.vec import Vec2, Vec3
from engine.core.math.quat import Quat


class ImageTrackingState(Enum):
    """Tracking state of an image target."""
    NONE = auto()           # Not detected
    DETECTING = auto()      # Detection in progress
    TRACKING = auto()       # Actively tracked
    LIMITED = auto()        # Tracked with reduced quality
    EXTENDED = auto()       # Extended tracking (image not visible)
    LOST = auto()           # Tracking lost


class TrackingMode(Enum):
    """Mode for image tracking."""
    CONTINUOUS = auto()     # Track continuously
    ONCE = auto()           # Track once, then stop
    ADAPTIVE = auto()       # Adapt based on movement


def ar_trackable(
    trackable_type: str = "image",
    reference_id: str = "",
) -> Callable:
    """Decorator to mark a class as AR trackable.

    Args:
        trackable_type: Type of trackable ('image' or 'object')
        reference_id: Reference identifier in the tracking database

    Returns:
        Decorator function
    """
    valid_types = {"image", "object"}
    if trackable_type not in valid_types:
        raise ValueError(f"Invalid trackable_type '{trackable_type}', must be one of {valid_types}")

    def decorator(cls):
        cls._ar_trackable = True
        cls._trackable_type = trackable_type
        cls._reference_id = reference_id

        # Set up tags for decorator introspection
        if not hasattr(cls, "_tags"):
            cls._tags = {}
        cls._tags["ar_trackable"] = True
        cls._tags["trackable_type"] = trackable_type
        cls._tags["reference_id"] = reference_id

        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = set()
        cls._applied_decorators.add("ar_trackable")

        return cls

    return decorator


@dataclass(slots=True)
class ImageReference:
    """Reference image in the tracking database."""
    reference_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    physical_width: float = 0.1  # Width in meters
    physical_height: float = 0.0  # 0 = auto from aspect
    image_data: Optional[bytes] = None
    image_path: str = ""
    feature_count: int = 0
    is_enabled: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class ImageTargetPose:
    """Pose data for a tracked image."""
    position: Vec3 = field(default_factory=Vec3.zero)
    orientation: Quat = field(default_factory=Quat.identity)
    timestamp: float = 0.0

    def transform_point(self, local_point: Vec3) -> Vec3:
        """Transform a local point to world space.

        Args:
            local_point: Point in image local space

        Returns:
            Point in world space
        """
        rotated = self.orientation.rotate_vector(local_point)
        return rotated + self.position


class ImageTarget:
    """A tracked 2D image target.

    Represents a reference image that can be detected and tracked
    in the camera feed for AR content placement.

    Attributes:
        target_id: Unique identifier
        reference: Reference image data
        tracking_state: Current tracking status
        pose: World position and orientation
    """
    __slots__ = (
        '_target_id',
        '_reference',
        '_tracking_state',
        '_tracking_mode',
        '_pose',
        '_confidence',
        '_tracked_size',
        '_is_active',
        '_native_handle',
        '_last_seen',
        '_first_detected',
        '_total_tracking_time',
        '_callbacks',
    )

    def __init__(
        self,
        reference: ImageReference,
        tracking_mode: TrackingMode = TrackingMode.CONTINUOUS,
    ) -> None:
        """Initialize an image target.

        Args:
            reference: Reference image data
            tracking_mode: How to track this target
        """
        self._target_id: str = str(uuid.uuid4())
        self._reference: ImageReference = reference
        self._tracking_state: ImageTrackingState = ImageTrackingState.NONE
        self._tracking_mode: TrackingMode = tracking_mode
        self._pose: ImageTargetPose = ImageTargetPose()
        self._confidence: float = 0.0
        self._tracked_size: Vec2 = Vec2(reference.physical_width, reference.physical_height)
        self._is_active: bool = False
        self._native_handle: Optional[int] = None
        self._last_seen: float = 0.0
        self._first_detected: float = 0.0
        self._total_tracking_time: float = 0.0
        self._callbacks: dict[str, list[Callable]] = {
            "detected": [],
            "tracking": [],
            "lost": [],
            "pose_updated": [],
        }

    @property
    def target_id(self) -> str:
        """Get the unique target identifier."""
        return self._target_id

    @property
    def reference_id(self) -> str:
        """Get the reference image ID."""
        return self._reference.reference_id

    @property
    def reference(self) -> ImageReference:
        """Get the reference image data."""
        return self._reference

    @property
    def name(self) -> str:
        """Get the target name."""
        return self._reference.name

    @property
    def tracking_state(self) -> ImageTrackingState:
        """Get the current tracking state."""
        return self._tracking_state

    @property
    def tracking_mode(self) -> TrackingMode:
        """Get the tracking mode."""
        return self._tracking_mode

    @property
    def is_tracked(self) -> bool:
        """Check if the target is actively tracked."""
        return self._tracking_state == ImageTrackingState.TRACKING

    @property
    def is_visible(self) -> bool:
        """Check if the target is currently visible."""
        return self._tracking_state in (
            ImageTrackingState.TRACKING,
            ImageTrackingState.LIMITED,
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
    def pose(self) -> ImageTargetPose:
        """Get the full pose data."""
        return self._pose

    @property
    def confidence(self) -> float:
        """Get tracking confidence (0.0 to 1.0)."""
        return self._confidence

    @property
    def tracked_size(self) -> Vec2:
        """Get the tracked size in world units."""
        return self._tracked_size

    @property
    def physical_size(self) -> Vec2:
        """Get the physical size from reference."""
        return Vec2(self._reference.physical_width, self._reference.physical_height)

    @property
    def is_active(self) -> bool:
        """Check if the target is active for tracking."""
        return self._is_active

    @property
    def last_seen_time(self) -> float:
        """Get when the target was last seen."""
        return self._last_seen

    def activate(self) -> bool:
        """Activate this target for tracking.

        Returns:
            True if activated successfully
        """
        if self._is_active:
            return False
        self._is_active = True
        self._tracking_state = ImageTrackingState.DETECTING
        return True

    def deactivate(self) -> bool:
        """Deactivate this target.

        Returns:
            True if deactivated successfully
        """
        if not self._is_active:
            return False
        self._is_active = False
        self._tracking_state = ImageTrackingState.NONE
        return True

    def update_pose(
        self,
        position: Vec3,
        orientation: Quat,
        confidence: float,
        tracked_size: Vec2,
        timestamp: float,
    ) -> None:
        """Update the target pose from tracking.

        Args:
            position: World position
            orientation: World orientation
            confidence: Tracking confidence
            tracked_size: Detected size in world units
            timestamp: Update timestamp
        """
        old_state = self._tracking_state

        self._pose.position = position
        self._pose.orientation = orientation
        self._pose.timestamp = timestamp
        self._confidence = max(0.0, min(1.0, confidence))
        self._tracked_size = tracked_size
        self._last_seen = timestamp

        if old_state == ImageTrackingState.NONE or old_state == ImageTrackingState.DETECTING:
            self._first_detected = timestamp
            self._tracking_state = ImageTrackingState.TRACKING
            self._notify_callbacks("detected")
        else:
            self._tracking_state = ImageTrackingState.TRACKING

        self._notify_callbacks("pose_updated")

    def update_tracking_state(
        self,
        state: ImageTrackingState,
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
            if state == ImageTrackingState.TRACKING:
                self._notify_callbacks("tracking")
            elif state == ImageTrackingState.LOST:
                self._notify_callbacks("lost")

    def get_corner_positions(self) -> list[Vec3]:
        """Get world positions of the image corners.

        Returns:
            List of 4 corner positions [TL, TR, BR, BL]
        """
        half_w = self._tracked_size.x / 2.0
        half_h = self._tracked_size.y / 2.0

        # Local corners (centered at origin)
        corners_local = [
            Vec3(-half_w, 0, -half_h),  # Top-left
            Vec3(half_w, 0, -half_h),   # Top-right
            Vec3(half_w, 0, half_h),    # Bottom-right
            Vec3(-half_w, 0, half_h),   # Bottom-left
        ]

        # Transform to world space
        return [self._pose.transform_point(c) for c in corners_local]

    def get_center_position(self) -> Vec3:
        """Get the world position of the image center.

        Returns:
            Center position in world space
        """
        return self._pose.position

    def get_forward_direction(self) -> Vec3:
        """Get the forward direction (image normal).

        Returns:
            Forward direction vector
        """
        return self._pose.orientation.forward()

    def get_up_direction(self) -> Vec3:
        """Get the up direction on the image.

        Returns:
            Up direction vector
        """
        return self._pose.orientation.up()

    def local_to_world(self, local_point: Vec3) -> Vec3:
        """Convert a local point to world space.

        Args:
            local_point: Point in image local space

        Returns:
            Point in world space
        """
        return self._pose.transform_point(local_point)

    def world_to_local(self, world_point: Vec3) -> Vec3:
        """Convert a world point to local space.

        Args:
            world_point: Point in world space

        Returns:
            Point in image local space
        """
        relative = world_point - self._pose.position
        inv_rot = self._pose.orientation.inverse()
        return inv_rot.rotate_vector(relative)

    def add_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for target events.

        Args:
            event: Event name ('detected', 'tracking', 'lost', 'pose_updated')
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
            f"ImageTarget(id={self._target_id[:8]}..., "
            f"name='{self._reference.name}', "
            f"state={self._tracking_state.name}, "
            f"confidence={self._confidence:.2f})"
        )


@dataclass(slots=True)
class ImageTrackerConfig:
    """Configuration for image tracking."""
    max_tracked_images: int = 4
    tracking_mode: TrackingMode = TrackingMode.CONTINUOUS
    enable_extended_tracking: bool = True
    tracking_timeout: float = 2.0  # Seconds before extended -> lost
    min_confidence_threshold: float = 0.3


class ImageTracker:
    """Manages image target tracking.

    Handles detection, tracking, and lifecycle of image targets
    in the AR camera feed.

    Attributes:
        config: Tracker configuration
        database: Reference image database
        targets: Active tracking targets
    """
    __slots__ = (
        '_config',
        '_database',
        '_targets',
        '_active_targets',
        '_is_running',
        '_callbacks',
        '_last_update',
    )

    def __init__(self, config: Optional[ImageTrackerConfig] = None) -> None:
        """Initialize the image tracker.

        Args:
            config: Tracker configuration
        """
        self._config: ImageTrackerConfig = config or ImageTrackerConfig()
        self._database: dict[str, ImageReference] = {}
        self._targets: dict[str, ImageTarget] = {}
        self._active_targets: list[str] = []
        self._is_running: bool = False
        self._callbacks: dict[str, list[Callable]] = {
            "target_detected": [],
            "target_lost": [],
            "tracking_started": [],
            "tracking_stopped": [],
        }
        self._last_update: float = 0.0

    @property
    def config(self) -> ImageTrackerConfig:
        """Get the tracker configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if tracking is active."""
        return self._is_running

    @property
    def tracked_count(self) -> int:
        """Get the number of currently tracked images."""
        return len([t for t in self._targets.values() if t.is_tracked])

    @property
    def database_size(self) -> int:
        """Get the number of reference images."""
        return len(self._database)

    # Database methods

    def add_reference_image(
        self,
        name: str,
        physical_width: float,
        image_path: str = "",
        image_data: Optional[bytes] = None,
        metadata: Optional[dict] = None,
    ) -> ImageReference:
        """Add a reference image to the database.

        Args:
            name: Image name
            physical_width: Physical width in meters
            image_path: Path to image file
            image_data: Raw image bytes (alternative to path)
            metadata: Optional metadata

        Returns:
            Created reference image
        """
        reference = ImageReference(
            name=name,
            physical_width=physical_width,
            image_path=image_path,
            image_data=image_data,
            metadata=metadata or {},
        )
        self._database[reference.reference_id] = reference
        return reference

    def remove_reference_image(self, reference_id: str) -> bool:
        """Remove a reference image from the database.

        Args:
            reference_id: Reference to remove

        Returns:
            True if removed
        """
        if reference_id in self._database:
            # Also remove any targets using this reference
            targets_to_remove = [
                tid for tid, t in self._targets.items()
                if t.reference_id == reference_id
            ]
            for tid in targets_to_remove:
                self.remove_target(tid)

            del self._database[reference_id]
            return True
        return False

    def get_reference_image(self, reference_id: str) -> Optional[ImageReference]:
        """Get a reference image by ID.

        Args:
            reference_id: Reference identifier

        Returns:
            Reference if found
        """
        return self._database.get(reference_id)

    def get_all_references(self) -> list[ImageReference]:
        """Get all reference images.

        Returns:
            List of all references
        """
        return list(self._database.values())

    def set_reference_enabled(self, reference_id: str, enabled: bool) -> bool:
        """Enable or disable a reference image.

        Args:
            reference_id: Reference to modify
            enabled: Whether to enable

        Returns:
            True if modified
        """
        ref = self._database.get(reference_id)
        if ref:
            ref.is_enabled = enabled
            return True
        return False

    # Target methods

    def create_target(
        self,
        reference_id: str,
        tracking_mode: Optional[TrackingMode] = None,
    ) -> Optional[ImageTarget]:
        """Create a tracking target from a reference.

        Args:
            reference_id: Reference image ID
            tracking_mode: Optional tracking mode override

        Returns:
            Created target or None if failed
        """
        reference = self._database.get(reference_id)
        if not reference:
            return None

        if not reference.is_enabled:
            return None

        target = ImageTarget(
            reference=reference,
            tracking_mode=tracking_mode or self._config.tracking_mode,
        )
        self._targets[target.target_id] = target
        return target

    def remove_target(self, target_id: str) -> bool:
        """Remove a tracking target.

        Args:
            target_id: Target to remove

        Returns:
            True if removed
        """
        target = self._targets.pop(target_id, None)
        if target:
            target.deactivate()
            if target_id in self._active_targets:
                self._active_targets.remove(target_id)
            return True
        return False

    def get_target(self, target_id: str) -> Optional[ImageTarget]:
        """Get a target by ID.

        Args:
            target_id: Target identifier

        Returns:
            Target if found
        """
        return self._targets.get(target_id)

    def get_all_targets(self) -> list[ImageTarget]:
        """Get all tracking targets.

        Returns:
            List of all targets
        """
        return list(self._targets.values())

    def get_tracked_targets(self) -> list[ImageTarget]:
        """Get currently tracked targets.

        Returns:
            List of actively tracked targets
        """
        return [t for t in self._targets.values() if t.is_tracked]

    def get_visible_targets(self) -> list[ImageTarget]:
        """Get currently visible targets.

        Returns:
            List of visible targets
        """
        return [t for t in self._targets.values() if t.is_visible]

    def find_target_by_reference(self, reference_id: str) -> Optional[ImageTarget]:
        """Find a target by its reference ID.

        Args:
            reference_id: Reference image ID

        Returns:
            Target if found
        """
        for target in self._targets.values():
            if target.reference_id == reference_id:
                return target
        return None

    # Tracking control

    def start(self) -> bool:
        """Start image tracking.

        Returns:
            True if started successfully
        """
        if self._is_running:
            return False

        self._is_running = True

        # Activate all targets
        for target in self._targets.values():
            if target.reference.is_enabled:
                target.activate()

        self._notify_callbacks("tracking_started")
        return True

    def stop(self) -> bool:
        """Stop image tracking.

        Returns:
            True if stopped successfully
        """
        if not self._is_running:
            return False

        self._is_running = False

        # Deactivate all targets
        for target in self._targets.values():
            target.deactivate()

        self._active_targets.clear()
        self._notify_callbacks("tracking_stopped")
        return True

    def update(self, timestamp: float) -> None:
        """Update image tracking.

        Args:
            timestamp: Current time
        """
        if not self._is_running:
            return

        self._last_update = timestamp

        # Check for tracking timeouts
        for target in self._targets.values():
            if not target.is_active:
                continue

            if target.tracking_state == ImageTrackingState.EXTENDED:
                if timestamp - target.last_seen_time > self._config.tracking_timeout:
                    target.update_tracking_state(ImageTrackingState.LOST)
                    self._notify_callbacks("target_lost", target)

    # Callbacks

    def add_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for tracker events.

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

    def _notify_callbacks(self, event: str, data: object = None) -> None:
        """Notify callbacks for an event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                if data is not None:
                    callback(data)
                else:
                    callback()

    def clear(self) -> None:
        """Clear all targets and references."""
        self.stop()
        self._targets.clear()
        self._database.clear()
        self._active_targets.clear()
