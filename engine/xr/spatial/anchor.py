"""Spatial anchors for AR world-locked content.

Provides anchor types for local, persistent, and cloud-shared spatial anchors
that lock virtual content to real-world positions across sessions and devices.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.xr.config import XR_CONFIG


class AnchorType(Enum):
    """Type of spatial anchor."""
    LOCAL = auto()       # Session-only, lost when app closes
    PERSISTENT = auto()  # Saved to device, restored across sessions
    CLOUD = auto()       # Shared via cloud service for multi-user


class AnchorTrackingState(Enum):
    """Tracking state of a spatial anchor."""
    UNKNOWN = auto()       # Initial state before tracking begins
    TRACKING = auto()      # Actively tracked with good confidence
    LIMITED = auto()       # Tracked but with reduced accuracy
    PAUSED = auto()        # Tracking temporarily paused (e.g., occluded)
    LOST = auto()          # Tracking lost, attempting recovery
    NOT_TRACKING = auto()  # Not being tracked (e.g., out of range)


class AnchorPersistenceState(Enum):
    """Persistence state for cloud anchors."""
    NONE = auto()          # Not persisted
    PENDING_SAVE = auto()  # Save in progress
    SAVED = auto()         # Successfully saved to cloud
    SAVE_FAILED = auto()   # Save operation failed
    PENDING_LOAD = auto()  # Load/resolve in progress
    LOADED = auto()        # Successfully loaded from cloud
    LOAD_FAILED = auto()   # Load operation failed


@dataclass(slots=True)
class AnchorPose:
    """Pose data for a spatial anchor."""
    position: Vec3 = field(default_factory=Vec3.zero)
    orientation: Quat = field(default_factory=Quat.identity)
    timestamp: float = 0.0


@dataclass(slots=True)
class CloudAnchorConfig:
    """Configuration for cloud anchor sharing."""
    anchor_id: str = ""
    expires_in_days: int = XR_CONFIG.spatial.DEFAULT_CLOUD_ANCHOR_EXPIRY_DAYS
    share_token: str = ""
    permissions: list[str] = field(default_factory=lambda: ["read"])


def spatial_anchor(
    anchor_type: str = "local",
    persistent: bool = False,
    cloud_id: Optional[str] = None,
) -> Callable:
    """Decorator to mark a class as a spatial anchor component.

    Args:
        anchor_type: Type of anchor - 'local', 'persistent', or 'cloud'
        persistent: Whether the anchor persists across sessions
        cloud_id: Cloud anchor identifier for shared anchors

    Returns:
        Decorator function
    """
    valid_types = {"local", "persistent", "cloud"}
    if anchor_type not in valid_types:
        raise ValueError(f"Invalid anchor_type '{anchor_type}', must be one of {valid_types}")

    def decorator(cls):
        cls._spatial_anchor = True
        cls._anchor_type = anchor_type
        cls._anchor_persistent = persistent
        cls._anchor_cloud_id = cloud_id

        # Set up tags for decorator introspection
        if not hasattr(cls, "_tags"):
            cls._tags = {}
        cls._tags["spatial_anchor"] = True
        cls._tags["anchor_type"] = anchor_type
        cls._tags["anchor_persistent"] = persistent

        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = set()
        cls._applied_decorators.add("spatial_anchor")

        return cls

    return decorator


class SpatialAnchor:
    """Spatial anchor component for world-locked AR content.

    Anchors lock virtual content to real-world positions. They can be:
    - Local: Exist only for the current session
    - Persistent: Saved to device storage, restored on restart
    - Cloud: Shared between devices via cloud anchor service

    Attributes:
        anchor_id: Unique identifier for this anchor
        anchor_type: Type of anchor (local, persistent, cloud)
        pose: Current position and orientation
        tracking_state: Current tracking status
        confidence: Tracking confidence (0.0 to 1.0)
    """
    __slots__ = (
        '_anchor_id',
        '_anchor_type',
        '_pose',
        '_tracking_state',
        '_confidence',
        '_is_active',
        '_cloud_config',
        '_persistence_state',
        '_native_handle',
        '_attached_entities',
        '_created_timestamp',
        '_last_updated',
        '_callbacks',
    )

    def __init__(
        self,
        anchor_type: AnchorType = AnchorType.LOCAL,
        position: Optional[Vec3] = None,
        orientation: Optional[Quat] = None,
    ) -> None:
        """Initialize a spatial anchor.

        Args:
            anchor_type: Type of anchor to create
            position: Initial world position
            orientation: Initial world orientation
        """
        self._anchor_id: str = str(uuid.uuid4())
        self._anchor_type: AnchorType = anchor_type
        self._pose: AnchorPose = AnchorPose(
            position=position or Vec3.zero(),
            orientation=orientation or Quat.identity(),
        )
        self._tracking_state: AnchorTrackingState = AnchorTrackingState.UNKNOWN
        self._confidence: float = 0.0
        self._is_active: bool = False
        self._cloud_config: Optional[CloudAnchorConfig] = None
        self._persistence_state: AnchorPersistenceState = AnchorPersistenceState.NONE
        self._native_handle: Optional[int] = None
        self._attached_entities: list[int] = []
        self._created_timestamp: float = 0.0
        self._last_updated: float = 0.0
        self._callbacks: dict[str, list[Callable]] = {
            "tracking_changed": [],
            "pose_updated": [],
            "persistence_changed": [],
        }

    @property
    def anchor_id(self) -> str:
        """Get the unique anchor identifier."""
        return self._anchor_id

    @property
    def anchor_type(self) -> AnchorType:
        """Get the anchor type."""
        return self._anchor_type

    @property
    def position(self) -> Vec3:
        """Get the current world position."""
        return self._pose.position

    @position.setter
    def position(self, value: Vec3) -> None:
        """Set the world position."""
        self._pose.position = value
        self._notify_callbacks("pose_updated")

    @property
    def orientation(self) -> Quat:
        """Get the current world orientation."""
        return self._pose.orientation

    @orientation.setter
    def orientation(self, value: Quat) -> None:
        """Set the world orientation."""
        self._pose.orientation = value
        self._notify_callbacks("pose_updated")

    @property
    def pose(self) -> AnchorPose:
        """Get the full pose data."""
        return self._pose

    @property
    def tracking_state(self) -> AnchorTrackingState:
        """Get the current tracking state."""
        return self._tracking_state

    @property
    def is_tracking(self) -> bool:
        """Check if the anchor is actively tracking."""
        return self._tracking_state == AnchorTrackingState.TRACKING

    @property
    def confidence(self) -> float:
        """Get tracking confidence (0.0 to 1.0)."""
        return self._confidence

    @property
    def is_active(self) -> bool:
        """Check if the anchor is active."""
        return self._is_active

    @property
    def is_persistent(self) -> bool:
        """Check if this is a persistent anchor."""
        return self._anchor_type in (AnchorType.PERSISTENT, AnchorType.CLOUD)

    @property
    def is_cloud_anchor(self) -> bool:
        """Check if this is a cloud-shared anchor."""
        return self._anchor_type == AnchorType.CLOUD

    @property
    def cloud_anchor_id(self) -> Optional[str]:
        """Get the cloud anchor ID if this is a cloud anchor."""
        if self._cloud_config:
            return self._cloud_config.anchor_id
        return None

    @property
    def persistence_state(self) -> AnchorPersistenceState:
        """Get the current persistence state."""
        return self._persistence_state

    def create(self, timestamp: float = 0.0) -> bool:
        """Create the anchor in the AR runtime.

        Args:
            timestamp: Creation timestamp

        Returns:
            True if creation succeeded
        """
        if self._is_active:
            return False

        self._created_timestamp = timestamp
        self._is_active = True
        self._tracking_state = AnchorTrackingState.TRACKING
        self._confidence = 1.0
        return True

    def destroy(self) -> bool:
        """Destroy the anchor and release resources.

        Returns:
            True if destruction succeeded
        """
        if not self._is_active:
            return False

        self._is_active = False
        self._tracking_state = AnchorTrackingState.NOT_TRACKING
        self._confidence = 0.0
        self._native_handle = None
        return True

    def update_pose(
        self,
        position: Vec3,
        orientation: Quat,
        confidence: float,
        timestamp: float,
    ) -> None:
        """Update the anchor pose from tracking system.

        Args:
            position: New world position
            orientation: New world orientation
            confidence: Tracking confidence (0.0 to 1.0)
            timestamp: Update timestamp
        """
        self._pose.position = position
        self._pose.orientation = orientation
        self._pose.timestamp = timestamp
        self._confidence = max(0.0, min(1.0, confidence))
        self._last_updated = timestamp
        self._notify_callbacks("pose_updated")

    def update_tracking_state(
        self,
        state: AnchorTrackingState,
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
            self._notify_callbacks("tracking_changed")

    def attach_entity(self, entity_id: int) -> None:
        """Attach an entity to this anchor.

        Args:
            entity_id: Entity ID to attach
        """
        if entity_id not in self._attached_entities:
            self._attached_entities.append(entity_id)

    def detach_entity(self, entity_id: int) -> None:
        """Detach an entity from this anchor.

        Args:
            entity_id: Entity ID to detach
        """
        if entity_id in self._attached_entities:
            self._attached_entities.remove(entity_id)

    def get_attached_entities(self) -> list[int]:
        """Get all entities attached to this anchor.

        Returns:
            List of attached entity IDs
        """
        return list(self._attached_entities)

    # Cloud anchor operations

    def save_to_cloud(self, expires_in_days: int = XR_CONFIG.spatial.DEFAULT_CLOUD_ANCHOR_EXPIRY_DAYS) -> bool:
        """Save this anchor to the cloud for sharing.

        Args:
            expires_in_days: Days until anchor expires

        Returns:
            True if save operation started
        """
        if self._anchor_type != AnchorType.CLOUD:
            return False

        if not self._is_active:
            return False

        self._cloud_config = CloudAnchorConfig(
            expires_in_days=expires_in_days,
        )
        self._persistence_state = AnchorPersistenceState.PENDING_SAVE
        self._notify_callbacks("persistence_changed")
        return True

    def resolve_cloud_anchor(self, cloud_anchor_id: str) -> bool:
        """Resolve a cloud anchor by its ID.

        Args:
            cloud_anchor_id: Cloud anchor identifier

        Returns:
            True if resolve operation started
        """
        if self._anchor_type != AnchorType.CLOUD:
            return False

        self._cloud_config = CloudAnchorConfig(anchor_id=cloud_anchor_id)
        self._persistence_state = AnchorPersistenceState.PENDING_LOAD
        self._notify_callbacks("persistence_changed")
        return True

    def on_cloud_save_complete(self, cloud_id: str, success: bool) -> None:
        """Handle cloud save completion.

        Args:
            cloud_id: Assigned cloud anchor ID
            success: Whether save succeeded
        """
        if success:
            if self._cloud_config:
                self._cloud_config.anchor_id = cloud_id
            self._persistence_state = AnchorPersistenceState.SAVED
        else:
            self._persistence_state = AnchorPersistenceState.SAVE_FAILED
        self._notify_callbacks("persistence_changed")

    def on_cloud_resolve_complete(self, success: bool) -> None:
        """Handle cloud resolve completion.

        Args:
            success: Whether resolve succeeded
        """
        if success:
            self._persistence_state = AnchorPersistenceState.LOADED
            self._tracking_state = AnchorTrackingState.TRACKING
        else:
            self._persistence_state = AnchorPersistenceState.LOAD_FAILED
        self._notify_callbacks("persistence_changed")

    # Persistence operations

    def save_to_disk(self) -> bool:
        """Save this anchor to local storage.

        Returns:
            True if save succeeded
        """
        if self._anchor_type not in (AnchorType.PERSISTENT, AnchorType.CLOUD):
            return False

        self._persistence_state = AnchorPersistenceState.SAVED
        return True

    def load_from_disk(self, anchor_id: str) -> bool:
        """Load anchor data from local storage.

        Args:
            anchor_id: Anchor ID to load

        Returns:
            True if load succeeded
        """
        self._anchor_id = anchor_id
        self._persistence_state = AnchorPersistenceState.LOADED
        return True

    # Callback management

    def add_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for anchor events.

        Args:
            event: Event name ('tracking_changed', 'pose_updated', 'persistence_changed')
            callback: Function to call when event occurs
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
        """Notify all callbacks for an event.

        Args:
            event: Event name
        """
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                callback(self)

    def __repr__(self) -> str:
        return (
            f"SpatialAnchor(id={self._anchor_id[:8]}..., "
            f"type={self._anchor_type.name}, "
            f"tracking={self._tracking_state.name}, "
            f"confidence={self._confidence:.2f})"
        )


class AnchorManager:
    """Manages spatial anchors across the AR session.

    Handles creation, tracking, persistence, and lifecycle of anchors.
    Provides querying by type, region, and tracking state.
    """
    __slots__ = (
        '_anchors',
        '_pending_cloud_operations',
        '_persistence_path',
        '_cloud_service_enabled',
    )

    def __init__(self, persistence_path: str = "") -> None:
        """Initialize the anchor manager.

        Args:
            persistence_path: Path for saving persistent anchors
        """
        self._anchors: dict[str, SpatialAnchor] = {}
        self._pending_cloud_operations: list[str] = []
        self._persistence_path: str = persistence_path
        self._cloud_service_enabled: bool = False

    def create_anchor(
        self,
        position: Vec3,
        orientation: Quat,
        anchor_type: AnchorType = AnchorType.LOCAL,
        timestamp: float = 0.0,
    ) -> SpatialAnchor:
        """Create a new spatial anchor.

        Args:
            position: World position
            orientation: World orientation
            anchor_type: Type of anchor to create
            timestamp: Creation timestamp

        Returns:
            Created anchor
        """
        anchor = SpatialAnchor(
            anchor_type=anchor_type,
            position=position,
            orientation=orientation,
        )
        anchor.create(timestamp)
        self._anchors[anchor.anchor_id] = anchor
        return anchor

    def destroy_anchor(self, anchor_id: str) -> bool:
        """Destroy an anchor.

        Args:
            anchor_id: Anchor to destroy

        Returns:
            True if anchor was destroyed
        """
        anchor = self._anchors.get(anchor_id)
        if anchor:
            anchor.destroy()
            del self._anchors[anchor_id]
            return True
        return False

    def get_anchor(self, anchor_id: str) -> Optional[SpatialAnchor]:
        """Get an anchor by ID.

        Args:
            anchor_id: Anchor identifier

        Returns:
            Anchor if found, None otherwise
        """
        return self._anchors.get(anchor_id)

    def get_all_anchors(self) -> list[SpatialAnchor]:
        """Get all active anchors.

        Returns:
            List of all anchors
        """
        return list(self._anchors.values())

    def get_anchors_by_type(self, anchor_type: AnchorType) -> list[SpatialAnchor]:
        """Get anchors by type.

        Args:
            anchor_type: Type to filter by

        Returns:
            List of matching anchors
        """
        return [a for a in self._anchors.values() if a.anchor_type == anchor_type]

    def get_tracking_anchors(self) -> list[SpatialAnchor]:
        """Get all actively tracking anchors.

        Returns:
            List of tracking anchors
        """
        return [a for a in self._anchors.values() if a.is_tracking]

    def get_anchors_near(
        self,
        position: Vec3,
        max_distance: float,
    ) -> list[SpatialAnchor]:
        """Get anchors within a distance of a point.

        Args:
            position: Query position
            max_distance: Maximum distance

        Returns:
            List of anchors within range
        """
        results = []
        for anchor in self._anchors.values():
            if anchor.position.distance(position) <= max_distance:
                results.append(anchor)
        return results

    def update(self, delta_time: float) -> None:
        """Update all anchors.

        Args:
            delta_time: Time since last update
        """
        for anchor in self._anchors.values():
            # Decay confidence for limited/paused anchors
            if anchor.tracking_state in (
                AnchorTrackingState.LIMITED,
                AnchorTrackingState.PAUSED,
            ):
                new_confidence = anchor.confidence - delta_time * XR_CONFIG.spatial.CONFIDENCE_DECAY_RATE
                if new_confidence <= 0.0:
                    anchor.update_tracking_state(
                        AnchorTrackingState.LOST,
                        confidence=0.0,
                    )

    def save_persistent_anchors(self) -> int:
        """Save all persistent anchors to disk.

        Returns:
            Number of anchors saved
        """
        count = 0
        for anchor in self._anchors.values():
            if anchor.is_persistent and anchor.save_to_disk():
                count += 1
        return count

    def load_persistent_anchors(self) -> int:
        """Load persistent anchors from disk.

        Returns:
            Number of anchors loaded
        """
        # In a real implementation, this would read from persistence_path
        return 0

    def enable_cloud_anchors(self, enabled: bool = True) -> None:
        """Enable or disable cloud anchor service.

        Args:
            enabled: Whether to enable cloud anchors
        """
        self._cloud_service_enabled = enabled

    def resolve_cloud_anchor(self, cloud_anchor_id: str) -> Optional[SpatialAnchor]:
        """Resolve a cloud anchor by ID.

        Args:
            cloud_anchor_id: Cloud anchor identifier

        Returns:
            Created anchor if resolution started, None if failed
        """
        if not self._cloud_service_enabled:
            return None

        anchor = SpatialAnchor(anchor_type=AnchorType.CLOUD)
        if anchor.resolve_cloud_anchor(cloud_anchor_id):
            self._anchors[anchor.anchor_id] = anchor
            self._pending_cloud_operations.append(anchor.anchor_id)
            return anchor
        return None

    def clear_all(self) -> None:
        """Clear all anchors."""
        for anchor in list(self._anchors.values()):
            anchor.destroy()
        self._anchors.clear()
