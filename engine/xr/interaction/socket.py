"""XR Socket system for snapping objects to predefined positions.

This module provides snap socket functionality for XR interactions,
allowing objects to be placed in specific locations with optional
filtering by tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, TypeVar

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform, RigidTransform

from .interactable import (
    XRInteractable,
    InteractionEvent,
    InteractionType,
    InteractorType,
    xr_interactable,
)
from .grabbable import XRGrabbable, GrabType


class SnapBehavior(Enum):
    """How objects snap to the socket."""
    INSTANT = auto()      # Immediately snap to position
    LERP = auto()         # Smoothly interpolate to position
    PHYSICS = auto()      # Use physics forces to guide


class EjectBehavior(Enum):
    """How objects are ejected from the socket."""
    INSTANT = auto()      # Immediately release
    FORCE = auto()        # Apply ejection force
    ANIMATED = auto()     # Play animation


@dataclass(slots=True)
class SocketState:
    """Current state of a socket."""
    is_occupied: bool = False
    attached_entity_id: Optional[int] = None
    attached_grabbable: Optional[XRGrabbable] = None
    attach_time: float = 0.0


@dataclass(slots=True)
class SocketAttachEvent:
    """Event data when object attaches to socket."""
    socket: 'XRSocket'
    grabbable: XRGrabbable
    timestamp: float
    was_thrown: bool = False


@dataclass(slots=True)
class SocketDetachEvent:
    """Event data when object detaches from socket."""
    socket: 'XRSocket'
    grabbable: XRGrabbable
    timestamp: float
    grabbed_by_interactor: bool = False
    interactor_id: Optional[int] = None


# Type variable for decorator
T = TypeVar('T', bound=type)


def xr_socket(
    accepted_tags: list[str] | None = None,
    snap_distance: float = 0.1,
    snap_behavior: SnapBehavior = SnapBehavior.INSTANT,
    eject_behavior: EjectBehavior = EjectBehavior.INSTANT,
    hover_highlight: bool = True,
    interaction_layers: list[str] | None = None,
    priority: int = 0
) -> Callable[[T], T]:
    """Decorator to mark a class as XR socket.

    Args:
        accepted_tags: List of tags that can attach to this socket
        snap_distance: Distance threshold for snapping
        snap_behavior: How objects snap to socket
        eject_behavior: How objects are ejected
        hover_highlight: Whether to highlight on hover
        interaction_layers: Interaction layer names
        priority: Selection priority

    Returns:
        Decorated class with XR socket metadata

    Example:
        @xr_socket(accepted_tags=["weapon", "tool"], snap_distance=0.15)
        class WeaponHolster(XRSocket):
            pass
    """
    def decorator(cls: T) -> T:
        # Apply base interactable decorator
        cls = xr_interactable(interaction_layers, priority)(cls)

        # Store socket metadata
        cls._xr_socket = True
        cls._accepted_tags = accepted_tags
        cls._snap_distance = snap_distance
        cls._snap_behavior = snap_behavior
        cls._eject_behavior = eject_behavior
        cls._hover_highlight = hover_highlight

        # Track applied decorators
        cls._applied_decorators.add('xr_socket')

        # Store tags (class-level)
        cls._class_tags['xr_socket'] = True
        cls._class_tags['accepted_tags'] = accepted_tags
        cls._class_tags['snap_distance'] = snap_distance
        cls._class_tags['snap_behavior'] = snap_behavior.name

        return cls
    return decorator


class XRSocket(XRInteractable):
    """Socket component for snapping grabbable objects.

    Provides snap-to-place functionality for XR grabbables with:
    - Tag-based filtering of acceptable objects
    - Distance-based snap detection
    - Multiple snap/eject behaviors
    - Hover preview support

    Attributes:
        accepted_tags: Tags of objects that can attach
        snap_distance: Distance threshold for snapping
        is_occupied: Whether socket has an attached object
        attached_object: Currently attached grabbable
    """
    __slots__ = (
        '_accepted_tags', '_snap_distance', '_snap_behavior', '_eject_behavior',
        '_hover_highlight', '_state', '_snap_lerp_speed', '_eject_force',
        '_attach_transform', '_on_attach_callbacks', '_on_detach_callbacks',
        '_hover_preview_entity', '_custom_filter', '_recycle_on_grab'
    )

    def __init__(
        self,
        entity_id: int = 0,
        accepted_tags: list[str] | None = None,
        snap_distance: float = 0.1,
        snap_behavior: SnapBehavior = SnapBehavior.INSTANT,
        eject_behavior: EjectBehavior = EjectBehavior.INSTANT,
        hover_highlight: bool = True,
        interaction_layers: list[str] | None = None,
        priority: int = 0
    ):
        """Initialize the socket component.

        Args:
            entity_id: Entity this component is attached to
            accepted_tags: Tags that can attach (None = all)
            snap_distance: Distance for snap detection
            snap_behavior: Snapping behavior
            eject_behavior: Ejection behavior
            hover_highlight: Show highlight on hover
            interaction_layers: Layer filter list
            priority: Selection priority
        """
        super().__init__(entity_id, interaction_layers, priority)

        self._accepted_tags = accepted_tags
        self._snap_distance = snap_distance
        self._snap_behavior = snap_behavior
        self._eject_behavior = eject_behavior
        self._hover_highlight = hover_highlight
        self._state = SocketState()
        self._snap_lerp_speed = 10.0
        self._eject_force = 1.0
        self._attach_transform = RigidTransform()
        self._on_attach_callbacks: list[Callable[[SocketAttachEvent], None]] = []
        self._on_detach_callbacks: list[Callable[[SocketDetachEvent], None]] = []
        self._hover_preview_entity: Optional[int] = None
        self._custom_filter: Optional[Callable[[XRGrabbable], bool]] = None
        self._recycle_on_grab = True

    @property
    def accepted_tags(self) -> list[str] | None:
        """Get accepted tags (None means all accepted)."""
        return self._accepted_tags.copy() if self._accepted_tags else None

    @property
    def snap_distance(self) -> float:
        """Get the snap distance threshold."""
        return self._snap_distance

    @property
    def is_occupied(self) -> bool:
        """Check if socket has an attached object."""
        return self._state.is_occupied

    @property
    def attached_object(self) -> Optional[XRGrabbable]:
        """Get the currently attached grabbable."""
        return self._state.attached_grabbable

    @property
    def snap_behavior(self) -> SnapBehavior:
        """Get the snap behavior."""
        return self._snap_behavior

    @property
    def eject_behavior(self) -> EjectBehavior:
        """Get the eject behavior."""
        return self._eject_behavior

    @property
    def attach_transform(self) -> RigidTransform:
        """Get the attachment transform."""
        return self._attach_transform

    def set_attach_transform(self, transform: RigidTransform) -> None:
        """Set the attachment transform for snapped objects.

        Args:
            transform: Local transform offset for attached objects
        """
        self._attach_transform = transform

    def set_custom_filter(
        self,
        filter_func: Optional[Callable[[XRGrabbable], bool]]
    ) -> None:
        """Set a custom filter for accepting objects.

        Args:
            filter_func: Function(grabbable) -> bool, or None to disable
        """
        self._custom_filter = filter_func

    def set_snap_lerp_speed(self, speed: float) -> None:
        """Set the lerp speed for LERP snap behavior.

        Args:
            speed: Interpolation speed (units per second)
        """
        self._snap_lerp_speed = max(0.1, speed)

    def set_eject_force(self, force: float) -> None:
        """Set the ejection force for FORCE eject behavior.

        Args:
            force: Force magnitude for ejection
        """
        self._eject_force = max(0.0, force)

    def set_recycle_on_grab(self, enabled: bool) -> None:
        """Set whether object is detached when grabbed from socket.

        Args:
            enabled: If True, grabbing detaches object from socket
        """
        self._recycle_on_grab = enabled

    def accepts(self, grabbable: XRGrabbable) -> bool:
        """Check if this socket accepts a specific grabbable.

        Args:
            grabbable: The grabbable to check

        Returns:
            True if grabbable can attach to this socket
        """
        if self._state.is_occupied:
            return False

        # Apply custom filter first
        if self._custom_filter and not self._custom_filter(grabbable):
            return False

        # If no tag restriction, accept all
        if self._accepted_tags is None:
            return True

        # Check if grabbable has any accepted tag
        grabbable_tags = getattr(grabbable, '_tags', {})
        obj_tags = grabbable_tags.get('socket_tags', [])

        if isinstance(obj_tags, str):
            obj_tags = [obj_tags]

        return any(tag in self._accepted_tags for tag in obj_tags)

    def is_within_snap_range(
        self,
        socket_transform: Transform,
        object_position: Vec3
    ) -> bool:
        """Check if an object is within snap range.

        Args:
            socket_transform: World transform of socket
            object_position: World position of object

        Returns:
            True if within snap distance
        """
        socket_pos = socket_transform.translation
        distance = (object_position - socket_pos).length()
        return distance <= self._snap_distance

    def try_attach(
        self,
        grabbable: XRGrabbable,
        timestamp: float,
        was_thrown: bool = False
    ) -> bool:
        """Attempt to attach a grabbable to this socket.

        Args:
            grabbable: The grabbable to attach
            timestamp: Current time
            was_thrown: Whether object was thrown into socket

        Returns:
            True if attachment succeeded
        """
        if not self.accepts(grabbable):
            return False

        # Update state
        self._state.is_occupied = True
        self._state.attached_entity_id = grabbable.entity_id
        self._state.attached_grabbable = grabbable
        self._state.attach_time = timestamp

        # Emit attach event
        event = SocketAttachEvent(
            socket=self,
            grabbable=grabbable,
            timestamp=timestamp,
            was_thrown=was_thrown
        )

        for callback in self._on_attach_callbacks:
            try:
                callback(event)
            except Exception:
                pass

        return True

    def detach(
        self,
        timestamp: float,
        grabbed_by_interactor: bool = False,
        interactor_id: Optional[int] = None
    ) -> Optional[XRGrabbable]:
        """Detach the current object from socket.

        Args:
            timestamp: Current time
            grabbed_by_interactor: Whether detached via grab
            interactor_id: ID of grabbing interactor if applicable

        Returns:
            The detached grabbable, or None if socket was empty
        """
        if not self._state.is_occupied:
            return None

        grabbable = self._state.attached_grabbable

        # Emit detach event
        event = SocketDetachEvent(
            socket=self,
            grabbable=grabbable,
            timestamp=timestamp,
            grabbed_by_interactor=grabbed_by_interactor,
            interactor_id=interactor_id
        )

        for callback in self._on_detach_callbacks:
            try:
                callback(event)
            except Exception:
                pass

        # Clear state
        self._state.is_occupied = False
        self._state.attached_entity_id = None
        self._state.attached_grabbable = None

        return grabbable

    def force_eject(self, timestamp: float, direction: Optional[Vec3] = None) -> Optional[XRGrabbable]:
        """Force eject the attached object.

        Args:
            timestamp: Current time
            direction: Optional ejection direction (default: up)

        Returns:
            The ejected grabbable, or None if socket was empty
        """
        if not self._state.is_occupied:
            return None

        # Get direction (default up)
        eject_dir = direction or Vec3.up()

        grabbable = self.detach(timestamp)

        # Apply ejection force if physics mode
        if grabbable and self._eject_behavior == EjectBehavior.FORCE:
            # Store ejection velocity for physics system to use
            if hasattr(grabbable, '_ejection_velocity'):
                grabbable._ejection_velocity = eject_dir * self._eject_force

        return grabbable

    def compute_snap_transform(
        self,
        socket_transform: Transform
    ) -> RigidTransform:
        """Compute the world transform for a snapped object.

        Args:
            socket_transform: World transform of socket

        Returns:
            World transform for snapped object
        """
        # Apply attach transform offset
        offset_pos = socket_transform.rotation.rotate_vector(
            self._attach_transform.translation
        )

        position = socket_transform.translation + offset_pos
        rotation = socket_transform.rotation * self._attach_transform.rotation

        return RigidTransform(position, rotation)

    def compute_lerp_transform(
        self,
        current_transform: RigidTransform,
        target_transform: RigidTransform,
        delta_time: float
    ) -> RigidTransform:
        """Compute interpolated transform for LERP snap behavior.

        Args:
            current_transform: Current object transform
            target_transform: Target snapped transform
            delta_time: Time since last update

        Returns:
            Interpolated transform
        """
        t = min(1.0, self._snap_lerp_speed * delta_time)

        position = current_transform.translation.lerp(target_transform.translation, t)
        rotation = current_transform.rotation.slerp(target_transform.rotation, t)

        return RigidTransform(position, rotation)

    def on_hover_enter_with_grabbable(
        self,
        grabbable: XRGrabbable,
        event: InteractionEvent
    ) -> None:
        """Called when a held grabbable hovers over socket.

        Args:
            grabbable: The hovering grabbable
            event: The interaction event
        """
        if self._hover_highlight and self.accepts(grabbable):
            self._on_valid_hover_enter(grabbable, event)

    def on_hover_exit_with_grabbable(
        self,
        grabbable: XRGrabbable,
        event: InteractionEvent
    ) -> None:
        """Called when a held grabbable stops hovering.

        Args:
            grabbable: The grabbable that stopped hovering
            event: The interaction event
        """
        if self._hover_highlight:
            self._on_hover_exit_highlight(grabbable, event)

    def _on_valid_hover_enter(
        self,
        grabbable: XRGrabbable,
        event: InteractionEvent
    ) -> None:
        """Override: Called when valid object hovers. Subclass for highlight."""
        pass

    def _on_hover_exit_highlight(
        self,
        grabbable: XRGrabbable,
        event: InteractionEvent
    ) -> None:
        """Override: Called when hover ends. Subclass for highlight removal."""
        pass

    def add_attach_callback(
        self,
        callback: Callable[[SocketAttachEvent], None]
    ) -> None:
        """Add callback for attach events.

        Args:
            callback: Function to call on attach
        """
        self._on_attach_callbacks.append(callback)

    def add_detach_callback(
        self,
        callback: Callable[[SocketDetachEvent], None]
    ) -> None:
        """Add callback for detach events.

        Args:
            callback: Function to call on detach
        """
        self._on_detach_callbacks.append(callback)

    def remove_attach_callback(
        self,
        callback: Callable[[SocketAttachEvent], None]
    ) -> None:
        """Remove an attach callback.

        Args:
            callback: Callback to remove
        """
        try:
            self._on_attach_callbacks.remove(callback)
        except ValueError:
            pass

    def remove_detach_callback(
        self,
        callback: Callable[[SocketDetachEvent], None]
    ) -> None:
        """Remove a detach callback.

        Args:
            callback: Callback to remove
        """
        try:
            self._on_detach_callbacks.remove(callback)
        except ValueError:
            pass


class SocketManager:
    """Manages all sockets in the scene for efficient lookup."""
    __slots__ = ('_sockets', '_by_tag', '_next_id')

    def __init__(self):
        """Initialize the socket manager."""
        self._sockets: dict[int, XRSocket] = {}
        self._by_tag: dict[str, set[int]] = {}
        self._next_id = 0

    def register(self, socket: XRSocket) -> int:
        """Register a socket and return its ID.

        Args:
            socket: The socket to register

        Returns:
            Assigned socket ID
        """
        socket_id = self._next_id
        self._next_id += 1

        self._sockets[socket_id] = socket

        if socket.accepted_tags:
            for tag in socket.accepted_tags:
                if tag not in self._by_tag:
                    self._by_tag[tag] = set()
                self._by_tag[tag].add(socket_id)
        else:
            # Socket accepts all - add to special "any" group
            if '_any_' not in self._by_tag:
                self._by_tag['_any_'] = set()
            self._by_tag['_any_'].add(socket_id)

        return socket_id

    def unregister(self, socket_id: int) -> None:
        """Unregister a socket.

        Args:
            socket_id: ID of socket to remove
        """
        socket = self._sockets.pop(socket_id, None)
        if socket:
            if socket.accepted_tags:
                for tag in socket.accepted_tags:
                    if tag in self._by_tag:
                        self._by_tag[tag].discard(socket_id)
            else:
                if '_any_' in self._by_tag:
                    self._by_tag['_any_'].discard(socket_id)

    def get(self, socket_id: int) -> Optional[XRSocket]:
        """Get a socket by ID.

        Args:
            socket_id: The socket ID

        Returns:
            Socket if found, None otherwise
        """
        return self._sockets.get(socket_id)

    def find_accepting_sockets(
        self,
        grabbable: XRGrabbable,
        position: Vec3,
        max_distance: Optional[float] = None
    ) -> list[tuple[int, XRSocket, float]]:
        """Find sockets that accept a grabbable within range.

        Args:
            grabbable: The grabbable to match
            position: Current position of grabbable
            max_distance: Maximum search distance (None = use socket snap_distance)

        Returns:
            List of (socket_id, socket, distance) tuples, sorted by distance
        """
        results = []

        for socket_id, socket in self._sockets.items():
            if not socket.accepts(grabbable):
                continue

            # Get socket world position (simplified - assumes entity has position)
            socket_pos = Vec3.zero()  # Would get from entity system
            if hasattr(socket, '_world_position'):
                socket_pos = socket._world_position

            distance = (position - socket_pos).length()
            threshold = max_distance if max_distance else socket.snap_distance

            if distance <= threshold:
                results.append((socket_id, socket, distance))

        return sorted(results, key=lambda x: x[2])

    def get_nearest_available_socket(
        self,
        grabbable: XRGrabbable,
        position: Vec3,
        max_distance: Optional[float] = None
    ) -> Optional[tuple[int, XRSocket]]:
        """Get the nearest available socket for a grabbable.

        Args:
            grabbable: The grabbable to match
            position: Current position
            max_distance: Maximum search distance

        Returns:
            (socket_id, socket) tuple if found, None otherwise
        """
        sockets = self.find_accepting_sockets(grabbable, position, max_distance)

        for socket_id, socket, _ in sockets:
            if not socket.is_occupied:
                return (socket_id, socket)

        return None
