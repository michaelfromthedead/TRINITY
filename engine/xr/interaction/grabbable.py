"""XR Grabbable component for objects that can be picked up and manipulated.

This module extends XRInteractable to provide full grab functionality
with support for physics-based and kinematic attachment modes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, Optional, TypeVar

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform, RigidTransform
from engine.xr.utils.math_utils import rotation_from_axes

from .interactable import (
    XRInteractable,
    InteractionEvent,
    InteractionType,
    InteractorType,
    xr_interactable,
)


class GrabType(Enum):
    """Method of grabbing an object."""
    DIRECT = auto()      # Hand directly touches object
    RAY = auto()         # Ray-based distance grab
    SOCKET = auto()      # Grabbed from a socket


class AttachmentMode(Enum):
    """How the grabbed object attaches to the interactor."""
    FIXED = auto()       # Object rigidly follows interactor
    PHYSICS = auto()     # Object connected via physics joint
    CUSTOM = auto()      # Custom attachment behavior


class HandPoseMode(Enum):
    """How the hand pose is determined when grabbing."""
    AUTO = auto()        # Automatically determine based on object shape
    PRESET = auto()      # Use a predefined hand pose
    BLENDING = auto()    # Blend between poses based on grip


@dataclass(slots=True)
class GrabAttachPoint:
    """Defines where and how an object attaches when grabbed."""
    local_position: Vec3 = field(default_factory=Vec3.zero)
    local_rotation: Quat = field(default_factory=Quat.identity)
    hand_pose_id: Optional[str] = None
    is_two_handed: bool = False
    secondary_attach_point: Optional['GrabAttachPoint'] = None


@dataclass(slots=True)
class ThrowData:
    """Data for calculating throw physics."""
    linear_velocity: Vec3
    angular_velocity: Vec3
    release_position: Vec3
    release_rotation: Quat
    velocity_scale: float = 1.0
    angular_velocity_scale: float = 1.0


@dataclass(slots=True)
class GrabState:
    """Current state of a grab interaction."""
    interactor_id: int
    grab_type: GrabType
    attachment_mode: AttachmentMode
    attach_point: GrabAttachPoint
    grab_start_time: float
    interactor_transform_at_grab: RigidTransform
    object_transform_at_grab: RigidTransform
    is_two_handed: bool = False
    secondary_interactor_id: Optional[int] = None


# Type variable for decorator
T = TypeVar('T', bound=type)


def xr_grabbable(
    grab_type: GrabType = GrabType.DIRECT,
    attachment_mode: AttachmentMode = AttachmentMode.FIXED,
    throwable: bool = True,
    two_handed: bool = False,
    hand_pose_mode: HandPoseMode = HandPoseMode.AUTO,
    interaction_layers: list[str] | None = None,
    priority: int = 0
) -> Callable[[T], T]:
    """Decorator to mark a class as XR grabbable.

    Args:
        grab_type: How the object can be grabbed
        attachment_mode: How object attaches to interactor
        throwable: Whether object can be thrown
        two_handed: Whether object supports two-handed grab
        hand_pose_mode: How hand pose is determined
        interaction_layers: Interaction layer names
        priority: Selection priority

    Returns:
        Decorated class with XR grabbable metadata

    Example:
        @xr_grabbable(throwable=True, two_handed=True)
        class Sword(XRGrabbable):
            pass
    """
    def decorator(cls: T) -> T:
        # Apply base interactable decorator
        cls = xr_interactable(interaction_layers, priority)(cls)

        # Store grabbable metadata
        cls._xr_grabbable = True
        cls._grab_type = grab_type
        cls._attachment_mode = attachment_mode
        cls._throwable = throwable
        cls._two_handed = two_handed
        cls._hand_pose_mode = hand_pose_mode

        # Track applied decorators
        cls._applied_decorators.add('xr_grabbable')

        # Store tags (class-level)
        cls._class_tags['xr_grabbable'] = True
        cls._class_tags['grab_type'] = grab_type.name
        cls._class_tags['attachment_mode'] = attachment_mode.name
        cls._class_tags['throwable'] = throwable
        cls._class_tags['two_handed'] = two_handed

        return cls
    return decorator


class XRGrabbable(XRInteractable):
    """Grabbable object with physics/kinematic attachment support.

    Extends XRInteractable with full grab functionality including:
    - Multiple grab types (direct, ray, socket)
    - Physics and kinematic attachment modes
    - Throwing with velocity transfer
    - Two-handed grab support
    - Custom hand poses

    Attributes:
        grab_type: How object can be grabbed
        attachment_mode: How object attaches to interactor
        throwable: Whether object can be thrown
        supports_two_handed: Whether two-handed grab is supported
    """
    __slots__ = (
        '_grab_type', '_attachment_mode', '_throwable', '_two_handed',
        '_hand_pose_mode', '_attach_points', '_grab_state',
        '_velocity_tracker', '_on_grab_callbacks', '_on_throw_callbacks',
        '_grab_filter', '_throw_velocity_scale', '_throw_angular_scale',
        '_secondary_grab_state'
    )

    def __init__(
        self,
        entity_id: int = 0,
        grab_type: GrabType = GrabType.DIRECT,
        attachment_mode: AttachmentMode = AttachmentMode.FIXED,
        throwable: bool = True,
        two_handed: bool = False,
        hand_pose_mode: HandPoseMode = HandPoseMode.AUTO,
        interaction_layers: list[str] | None = None,
        priority: int = 0
    ):
        """Initialize the grabbable component.

        Args:
            entity_id: Entity this component is attached to
            grab_type: Method of grabbing
            attachment_mode: Attachment behavior
            throwable: Allow throwing
            two_handed: Support two-handed grabs
            hand_pose_mode: Hand pose determination
            interaction_layers: Layer filter list
            priority: Selection priority
        """
        super().__init__(entity_id, interaction_layers, priority)

        self._grab_type = grab_type
        self._attachment_mode = attachment_mode
        self._throwable = throwable
        self._two_handed = two_handed
        self._hand_pose_mode = hand_pose_mode
        self._attach_points: list[GrabAttachPoint] = [GrabAttachPoint()]
        self._grab_state: Optional[GrabState] = None
        # Use deque with maxlen for efficient velocity tracking without reallocations
        # At 90Hz, 10 samples covers ~110ms which is more than the 100ms we need
        self._velocity_tracker: Deque[tuple[float, Vec3, Vec3]] = deque(maxlen=10)
        self._on_grab_callbacks: list[Callable[[GrabState], None]] = []
        self._on_throw_callbacks: list[Callable[[ThrowData], None]] = []
        self._grab_filter: Optional[Callable[[int, GrabType], bool]] = None
        self._throw_velocity_scale = 1.0
        self._throw_angular_scale = 1.0
        self._secondary_grab_state: Optional[GrabState] = None

    @property
    def grab_type(self) -> GrabType:
        """Get the grab type."""
        return self._grab_type

    @property
    def attachment_mode(self) -> AttachmentMode:
        """Get the attachment mode."""
        return self._attachment_mode

    @property
    def throwable(self) -> bool:
        """Check if object can be thrown."""
        return self._throwable

    @property
    def supports_two_handed(self) -> bool:
        """Check if two-handed grab is supported."""
        return self._two_handed

    @property
    def hand_pose_mode(self) -> HandPoseMode:
        """Get the hand pose mode."""
        return self._hand_pose_mode

    @property
    def is_two_hand_grabbed(self) -> bool:
        """Check if currently grabbed with two hands."""
        return self._grab_state is not None and self._grab_state.is_two_handed

    @property
    def grab_state(self) -> Optional[GrabState]:
        """Get the current grab state."""
        return self._grab_state

    def add_attach_point(self, attach_point: GrabAttachPoint) -> None:
        """Add a grab attach point.

        Args:
            attach_point: The attach point to add
        """
        self._attach_points.append(attach_point)

    def get_attach_points(self) -> list[GrabAttachPoint]:
        """Get all attach points.

        Returns:
            List of attach points
        """
        return self._attach_points.copy()

    def get_nearest_attach_point(
        self,
        grab_position: Vec3,
        object_transform: Transform
    ) -> GrabAttachPoint:
        """Find the nearest attach point to a grab position.

        Args:
            grab_position: World position of grab attempt
            object_transform: Current object transform

        Returns:
            The nearest attach point
        """
        if len(self._attach_points) == 1:
            return self._attach_points[0]

        best_point = self._attach_points[0]
        best_distance = float('inf')

        for point in self._attach_points:
            world_pos = object_transform.transform_point(point.local_position)
            distance = (grab_position - world_pos).length_squared()
            if distance < best_distance:
                best_distance = distance
                best_point = point

        return best_point

    def set_grab_filter(
        self,
        filter_func: Optional[Callable[[int, GrabType], bool]]
    ) -> None:
        """Set a filter function to allow/deny grab attempts.

        Args:
            filter_func: Function(interactor_id, grab_type) -> bool
        """
        self._grab_filter = filter_func

    def set_throw_scales(
        self,
        velocity_scale: float = 1.0,
        angular_scale: float = 1.0
    ) -> None:
        """Set velocity scales for throwing.

        Args:
            velocity_scale: Linear velocity multiplier
            angular_scale: Angular velocity multiplier
        """
        self._throw_velocity_scale = velocity_scale
        self._throw_angular_scale = angular_scale

    def can_be_grabbed(
        self,
        interactor_id: int,
        grab_type: GrabType
    ) -> bool:
        """Check if object can be grabbed by specified interactor.

        Args:
            interactor_id: ID of attempting interactor
            grab_type: Type of grab being attempted

        Returns:
            True if grab is allowed
        """
        if not self._enabled:
            return False

        if self._grab_interactor is not None and not self._two_handed:
            return False

        if self._grab_filter and not self._grab_filter(interactor_id, grab_type):
            return False

        return True

    def try_grab(
        self,
        interactor_id: int,
        grab_type: GrabType,
        interactor_transform: RigidTransform,
        object_transform: Transform,
        event: InteractionEvent
    ) -> bool:
        """Attempt to grab the object.

        Args:
            interactor_id: ID of grabbing interactor
            grab_type: Type of grab
            interactor_transform: Transform of interactor
            object_transform: Current object transform
            event: The interaction event

        Returns:
            True if grab succeeded
        """
        if not self.can_be_grabbed(interactor_id, grab_type):
            return False

        # Handle two-handed grab
        if self._grab_interactor is not None and self._two_handed:
            return self._try_secondary_grab(
                interactor_id, grab_type, interactor_transform, object_transform, event
            )

        # Find best attach point
        grab_pos = Vec3(
            event.position.x, event.position.y, event.position.z
        )
        attach_point = self.get_nearest_attach_point(grab_pos, object_transform)

        # Create grab state
        self._grab_state = GrabState(
            interactor_id=interactor_id,
            grab_type=grab_type,
            attachment_mode=self._attachment_mode,
            attach_point=attach_point,
            grab_start_time=event.timestamp,
            interactor_transform_at_grab=RigidTransform(
                interactor_transform.translation,
                interactor_transform.rotation
            ),
            object_transform_at_grab=RigidTransform(
                object_transform.translation,
                object_transform.rotation
            ),
            is_two_handed=False
        )

        # Clear velocity history for throw calculation
        self._velocity_tracker.clear()

        # Notify via base class
        self.on_grab_enter(interactor_id, event)

        # Notify grab callbacks
        for callback in self._on_grab_callbacks:
            try:
                callback(self._grab_state)
            except Exception:
                pass

        return True

    def _try_secondary_grab(
        self,
        interactor_id: int,
        grab_type: GrabType,
        interactor_transform: RigidTransform,
        object_transform: Transform,
        event: InteractionEvent
    ) -> bool:
        """Handle secondary grab for two-handed interaction.

        Args:
            interactor_id: ID of secondary grabbing interactor
            grab_type: Type of grab
            interactor_transform: Transform of interactor
            object_transform: Current object transform
            event: The interaction event

        Returns:
            True if secondary grab succeeded
        """
        if self._grab_state is None:
            return False

        # Find secondary attach point
        primary_attach = self._grab_state.attach_point
        secondary_attach = primary_attach.secondary_attach_point

        if secondary_attach is None:
            # Create default secondary point
            secondary_attach = GrabAttachPoint(
                local_position=Vec3(0, 0, 0.2),  # Default offset
                local_rotation=Quat.identity()
            )

        self._secondary_grab_state = GrabState(
            interactor_id=interactor_id,
            grab_type=grab_type,
            attachment_mode=self._attachment_mode,
            attach_point=secondary_attach,
            grab_start_time=event.timestamp,
            interactor_transform_at_grab=RigidTransform(
                interactor_transform.translation,
                interactor_transform.rotation
            ),
            object_transform_at_grab=RigidTransform(
                object_transform.translation,
                object_transform.rotation
            )
        )

        self._grab_state.is_two_handed = True
        self._grab_state.secondary_interactor_id = interactor_id

        return True

    def release(
        self,
        interactor_id: int,
        event: InteractionEvent,
        current_velocity: Optional[Vec3] = None,
        current_angular_velocity: Optional[Vec3] = None
    ) -> Optional[ThrowData]:
        """Release the grabbed object.

        Args:
            interactor_id: ID of releasing interactor
            event: The interaction event
            current_velocity: Optional current linear velocity
            current_angular_velocity: Optional current angular velocity

        Returns:
            ThrowData if object was thrown, None otherwise
        """
        if self._grab_state is None:
            return None

        # Handle secondary hand release
        if self._grab_state.is_two_handed:
            if self._secondary_grab_state and \
               self._secondary_grab_state.interactor_id == interactor_id:
                self._secondary_grab_state = None
                self._grab_state.is_two_handed = False
                self._grab_state.secondary_interactor_id = None
                return None

        # Only primary grabber can fully release
        if self._grab_state.interactor_id != interactor_id:
            return None

        throw_data = None

        if self._throwable:
            throw_data = self._calculate_throw(
                event, current_velocity, current_angular_velocity
            )

            for callback in self._on_throw_callbacks:
                try:
                    callback(throw_data)
                except Exception:
                    pass

        # Clear grab state
        self._grab_state = None
        self._secondary_grab_state = None

        # Notify via base class
        self.on_grab_exit(interactor_id, event)

        return throw_data

    def track_velocity(
        self,
        timestamp: float,
        position: Vec3,
        rotation: Quat
    ) -> None:
        """Track position/rotation for throw velocity calculation.

        Args:
            timestamp: Current time
            position: Current world position
            rotation: Current world rotation
        """
        # Convert rotation to angular velocity approximation
        euler = rotation.to_euler()
        angular = Vec3(euler[0], euler[1], euler[2])

        # deque with maxlen automatically drops oldest entries, no reallocation needed
        self._velocity_tracker.append((timestamp, position, angular))

        # Remove samples older than 100ms from the front (deque popleft is O(1))
        cutoff = timestamp - 0.1
        while self._velocity_tracker and self._velocity_tracker[0][0] < cutoff:
            self._velocity_tracker.popleft()

    def _calculate_throw(
        self,
        event: InteractionEvent,
        velocity_override: Optional[Vec3],
        angular_override: Optional[Vec3]
    ) -> ThrowData:
        """Calculate throw velocity from tracking data.

        Args:
            event: The release event
            velocity_override: Override linear velocity
            angular_override: Override angular velocity

        Returns:
            Calculated throw data
        """
        linear_vel = velocity_override or Vec3.zero()
        angular_vel = angular_override or Vec3.zero()

        if velocity_override is None and len(self._velocity_tracker) >= 2:
            # Calculate average velocity from recent samples
            samples = self._velocity_tracker[-5:]  # Use last 5 samples

            if len(samples) >= 2:
                dt = samples[-1][0] - samples[0][0]
                if dt > 0.001:  # Avoid division by zero
                    dp = samples[-1][1] - samples[0][1]
                    linear_vel = dp / dt

        if angular_override is None and len(self._velocity_tracker) >= 2:
            samples = self._velocity_tracker[-5:]

            if len(samples) >= 2:
                dt = samples[-1][0] - samples[0][0]
                if dt > 0.001:
                    da = samples[-1][2] - samples[0][2]
                    angular_vel = da / dt

        return ThrowData(
            linear_velocity=linear_vel * self._throw_velocity_scale,
            angular_velocity=angular_vel * self._throw_angular_scale,
            release_position=event.position,
            release_rotation=event.rotation,
            velocity_scale=self._throw_velocity_scale,
            angular_velocity_scale=self._throw_angular_scale
        )

    def compute_attached_transform(
        self,
        interactor_transform: RigidTransform,
        secondary_transform: Optional[RigidTransform] = None
    ) -> RigidTransform:
        """Compute object transform based on interactor(s).

        Args:
            interactor_transform: Primary interactor transform
            secondary_transform: Optional secondary interactor transform

        Returns:
            Computed object transform
        """
        if self._grab_state is None:
            return RigidTransform()

        attach = self._grab_state.attach_point

        if self._grab_state.is_two_handed and secondary_transform:
            return self._compute_two_handed_transform(
                interactor_transform, secondary_transform
            )

        # Single-handed: offset from interactor by attach point
        offset_rotated = interactor_transform.rotation.rotate_vector(
            -attach.local_position
        )

        position = interactor_transform.translation + offset_rotated
        rotation = interactor_transform.rotation * attach.local_rotation.inverse()

        return RigidTransform(position, rotation)

    def _compute_two_handed_transform(
        self,
        primary_transform: RigidTransform,
        secondary_transform: RigidTransform
    ) -> RigidTransform:
        """Compute transform for two-handed grab.

        Uses the line between both hands to orient the object.

        Args:
            primary_transform: Primary hand transform
            secondary_transform: Secondary hand transform

        Returns:
            Computed object transform
        """
        # Position is midpoint between hands
        position = (primary_transform.translation + secondary_transform.translation) * 0.5

        # Forward direction is from primary to secondary
        forward = (secondary_transform.translation - primary_transform.translation).normalized()

        # Approximate up from average of both hand ups
        up1 = primary_transform.rotation.up()
        up2 = secondary_transform.rotation.up()
        up = (up1 + up2).normalized()

        # Compute right and fix up
        right = forward.cross(up).normalized()
        up = right.cross(forward).normalized()

        # Build rotation from axes using shared utility
        rotation = rotation_from_axes(forward, up, right)

        return RigidTransform(position, rotation)

    def add_grab_callback(
        self,
        callback: Callable[[GrabState], None]
    ) -> None:
        """Add a callback for grab events.

        Args:
            callback: Function to call when grabbed
        """
        self._on_grab_callbacks.append(callback)

    def add_throw_callback(
        self,
        callback: Callable[[ThrowData], None]
    ) -> None:
        """Add a callback for throw events.

        Args:
            callback: Function to call when thrown
        """
        self._on_throw_callbacks.append(callback)

    def _on_grab_started(self, event: InteractionEvent) -> None:
        """Override: Called when grab starts."""
        pass

    def _on_grab_ended(self, event: InteractionEvent) -> None:
        """Override: Called when grab ends."""
        self._velocity_tracker.clear()
