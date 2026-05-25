"""Direct/poke XR interactor for touch-based interactions.

This module provides direct interaction through physical touch,
supporting poke, touch, and direct grab functionality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform, RigidTransform

from .interactable import (
    XRInteractable,
    InteractionEvent,
    InteractionType,
    InteractorType,
    InteractionHit,
    InteractableManager,
)
from .grabbable import XRGrabbable, GrabType


class PokeMode(Enum):
    """Modes for poke interaction."""
    FINGERTIP = auto()     # Single point at fingertip
    SPHERE = auto()        # Sphere collision
    MULTI_POINT = auto()   # Multiple contact points


class GrabDetection(Enum):
    """Methods for detecting grab gestures."""
    PINCH = auto()         # Thumb-index pinch
    GRIP = auto()          # Full hand grip
    PROXIMITY = auto()     # Object proximity + button
    CUSTOM = auto()        # Custom gesture


@dataclass(slots=True)
class DirectConfig:
    """Configuration for direct interactor behavior."""
    poke_mode: PokeMode = PokeMode.FINGERTIP
    grab_detection: GrabDetection = GrabDetection.PINCH
    interaction_radius: float = 0.02       # Touch detection radius
    poke_depth_threshold: float = 0.01     # Depth to trigger poke
    grab_threshold: float = 0.7            # Pinch/grip threshold
    hover_distance: float = 0.05           # Hover trigger distance
    sticky_grab: bool = True               # Keep grab if hand moves away


@dataclass(slots=True)
class ContactPoint:
    """A contact point for direct interaction."""
    position: Vec3
    normal: Vec3
    penetration_depth: float
    collider_id: Optional[int] = None


@dataclass(slots=True)
class DirectState:
    """Current state of direct interactor."""
    position: Vec3 = field(default_factory=Vec3.zero)
    rotation: Quat = field(default_factory=Quat.identity)
    velocity: Vec3 = field(default_factory=Vec3.zero)
    contact_points: list[ContactPoint] = field(default_factory=list)
    hovered_interactable: Optional[XRInteractable] = None
    poked_interactable: Optional[XRInteractable] = None
    grabbed_object: Optional[XRGrabbable] = None
    pinch_strength: float = 0.0
    grip_strength: float = 0.0
    is_poking: bool = False
    is_grabbing: bool = False
    poke_depth: float = 0.0


class DirectInteractor:
    """Direct/poke interactor for physical touch interactions.

    Enables direct manipulation of XR objects through physical
    touch, supporting:
    - Poke interactions (UI buttons, etc.)
    - Direct grab (physically touching objects)
    - Touch feedback

    Attributes:
        interactor_id: Unique identifier for this interactor
        config: Interactor configuration
        is_active: Whether interactor is active
    """
    __slots__ = (
        '_interactor_id', '_config', '_state', '_is_active', '_layer_mask',
        '_collision_callback', '_poke_callbacks', '_grab_callbacks',
        '_hover_callbacks', '_interactable_manager', '_timestamp',
        '_previous_position', '_poke_start_position'
    )

    def __init__(
        self,
        interactor_id: int,
        config: Optional[DirectConfig] = None,
        interactable_manager: Optional[InteractableManager] = None
    ):
        """Initialize the direct interactor.

        Args:
            interactor_id: Unique ID for this interactor
            config: Interactor configuration (uses defaults if None)
            interactable_manager: Manager for finding interactables
        """
        self._interactor_id = interactor_id
        self._config = config or DirectConfig()
        self._state = DirectState()
        self._is_active = True
        self._layer_mask: list[str] | None = None
        self._collision_callback: Optional[Callable[[Vec3, float], list[InteractionHit]]] = None
        self._poke_callbacks: list[Callable[[InteractionEvent], None]] = []
        self._grab_callbacks: list[Callable[[InteractionEvent], None]] = []
        self._hover_callbacks: list[Callable[[InteractionEvent], None]] = []
        self._interactable_manager = interactable_manager
        self._timestamp = 0.0
        self._previous_position = Vec3.zero()
        self._poke_start_position: Optional[Vec3] = None

    @property
    def interactor_id(self) -> int:
        """Get the interactor ID."""
        return self._interactor_id

    @property
    def config(self) -> DirectConfig:
        """Get the configuration."""
        return self._config

    @property
    def is_active(self) -> bool:
        """Check if interactor is active."""
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        """Enable or disable the interactor."""
        if not value and self._is_active:
            self._release_all()
        self._is_active = value

    @property
    def state(self) -> DirectState:
        """Get the current state."""
        return self._state

    @property
    def is_hovering(self) -> bool:
        """Check if hovering over an interactable."""
        return self._state.hovered_interactable is not None

    @property
    def is_poking(self) -> bool:
        """Check if currently poking."""
        return self._state.is_poking

    @property
    def is_grabbing(self) -> bool:
        """Check if currently grabbing."""
        return self._state.is_grabbing

    def set_layer_mask(self, layers: list[str] | None) -> None:
        """Set the layer mask for filtering interactions.

        Args:
            layers: List of layer names, or None for all
        """
        self._layer_mask = layers

    def set_collision_callback(
        self,
        callback: Callable[[Vec3, float], list[InteractionHit]]
    ) -> None:
        """Set the collision detection callback.

        Args:
            callback: Function(position, radius) -> [InteractionHit]
        """
        self._collision_callback = callback

    def update(
        self,
        position: Vec3,
        rotation: Quat,
        pinch_strength: float,
        grip_strength: float,
        timestamp: float
    ) -> None:
        """Update the direct interactor state.

        Args:
            position: Fingertip/interaction point world position
            rotation: Hand rotation
            pinch_strength: Pinch gesture strength 0.0-1.0
            grip_strength: Grip gesture strength 0.0-1.0
            timestamp: Current frame timestamp
        """
        if not self._is_active:
            return

        # Calculate velocity
        dt = timestamp - self._timestamp if self._timestamp > 0 else 0.016
        if dt > 0:
            self._state.velocity = (position - self._previous_position) / dt

        self._previous_position = self._state.position
        self._timestamp = timestamp
        self._state.position = position
        self._state.rotation = rotation
        self._state.pinch_strength = pinch_strength
        self._state.grip_strength = grip_strength

        # Perform collision detection
        hits = self._detect_collisions()

        # Process hover
        self._process_hover(hits)

        # Process poke
        self._process_poke(hits)

        # Process grab
        self._process_grab(hits, pinch_strength, grip_strength)

        # Update grabbed object
        if self._state.grabbed_object:
            self._update_grabbed_object()

    def _detect_collisions(self) -> list[InteractionHit]:
        """Detect collisions with interactables.

        Returns:
            List of interaction hits
        """
        if not self._collision_callback:
            return []

        hits = self._collision_callback(
            self._state.position,
            self._config.interaction_radius
        )

        # Filter by layer mask
        if self._layer_mask:
            hits = [h for h in hits if any(
                h.interactable.is_in_layer(l) for l in self._layer_mask
            )]

        # Sort by distance
        hits.sort(key=lambda h: h.distance)

        return hits

    def _process_hover(self, hits: list[InteractionHit]) -> None:
        """Process hover state.

        Args:
            hits: Current collision hits
        """
        # Find nearest interactable within hover distance
        new_hover = None
        for hit in hits:
            if hit.distance <= self._config.hover_distance:
                new_hover = hit.interactable
                break

        if new_hover != self._state.hovered_interactable:
            # Exit previous hover
            if self._state.hovered_interactable:
                event = self._create_event(InteractionType.HOVER)
                self._state.hovered_interactable.on_hover_exit(
                    self._interactor_id, event
                )
                self._emit_hover_callbacks(event)

            # Enter new hover
            if new_hover:
                event = self._create_event(InteractionType.HOVER)
                new_hover.on_hover_enter(self._interactor_id, event)
                self._emit_hover_callbacks(event)

            self._state.hovered_interactable = new_hover

    def _process_poke(self, hits: list[InteractionHit]) -> None:
        """Process poke interactions.

        Args:
            hits: Current collision hits
        """
        # Find contact with sufficient depth
        poke_hit = None
        poke_depth = 0.0

        for hit in hits:
            if hit.distance < 0:  # Penetrating
                depth = abs(hit.distance)
                if depth > poke_depth:
                    poke_depth = depth
                    poke_hit = hit

        # Check if poke depth exceeds threshold
        if poke_hit and poke_depth >= self._config.poke_depth_threshold:
            if not self._state.is_poking:
                # Start poke
                self._start_poke(poke_hit.interactable, poke_depth)
            else:
                # Update poke depth
                self._state.poke_depth = poke_depth
        elif self._state.is_poking:
            # End poke
            self._end_poke()

    def _start_poke(self, interactable: XRInteractable, depth: float) -> None:
        """Start a poke interaction.

        Args:
            interactable: The poked interactable
            depth: Poke depth
        """
        self._state.is_poking = True
        self._state.poked_interactable = interactable
        self._state.poke_depth = depth
        self._poke_start_position = self._state.position

        # Send select enter (poke acts as select)
        event = self._create_event(InteractionType.SELECT)
        event.data['poke_depth'] = depth
        interactable.on_select_enter(self._interactor_id, event)
        self._emit_poke_callbacks(event)

    def _end_poke(self) -> None:
        """End the current poke interaction."""
        if self._state.poked_interactable:
            event = self._create_event(InteractionType.SELECT)
            event.data['poke_depth'] = self._state.poke_depth
            self._state.poked_interactable.on_select_exit(
                self._interactor_id, event
            )

            # Activate if poke was deep enough
            if self._state.poke_depth >= self._config.poke_depth_threshold * 1.5:
                self._state.poked_interactable.on_activate(event)

            self._emit_poke_callbacks(event)

        self._state.is_poking = False
        self._state.poked_interactable = None
        self._state.poke_depth = 0.0
        self._poke_start_position = None

    def _process_grab(
        self,
        hits: list[InteractionHit],
        pinch: float,
        grip: float
    ) -> None:
        """Process grab interactions.

        Args:
            hits: Current collision hits
            pinch: Pinch strength
            grip: Grip strength
        """
        # Determine if grab gesture is active
        grab_active = self._is_grab_gesture_active(pinch, grip)

        if grab_active and not self._state.is_grabbing:
            # Try to grab
            self._try_grab(hits)
        elif not grab_active and self._state.is_grabbing:
            # Release grab
            if not self._config.sticky_grab:
                self._release_grab()

    def _is_grab_gesture_active(self, pinch: float, grip: float) -> bool:
        """Check if a grab gesture is being performed.

        Args:
            pinch: Pinch strength
            grip: Grip strength

        Returns:
            True if grab gesture is active
        """
        threshold = self._config.grab_threshold

        if self._config.grab_detection == GrabDetection.PINCH:
            return pinch >= threshold
        elif self._config.grab_detection == GrabDetection.GRIP:
            return grip >= threshold
        elif self._config.grab_detection == GrabDetection.PROXIMITY:
            # Proximity mode - any touch + some grip
            return grip >= threshold * 0.5
        else:
            # Custom - default to pinch
            return pinch >= threshold

    def _try_grab(self, hits: list[InteractionHit]) -> None:
        """Attempt to grab from collision hits.

        Args:
            hits: Current collision hits
        """
        for hit in hits:
            if not isinstance(hit.interactable, XRGrabbable):
                continue

            grabbable = hit.interactable

            if not grabbable.can_be_grabbed(self._interactor_id, GrabType.DIRECT):
                continue

            # Get interactor transform
            interactor_transform = RigidTransform(
                self._state.position,
                self._state.rotation
            )

            # Get object transform from hit
            object_transform = Transform(translation=hit.hit_point)

            event = self._create_event(InteractionType.GRAB)

            if grabbable.try_grab(
                self._interactor_id,
                GrabType.DIRECT,
                interactor_transform,
                object_transform,
                event
            ):
                self._state.grabbed_object = grabbable
                self._state.is_grabbing = True
                self._emit_grab_callbacks(event)
                return

    def _release_grab(self) -> None:
        """Release the currently grabbed object."""
        if not self._state.grabbed_object:
            return

        event = self._create_event(InteractionType.GRAB)

        # Release with current velocity
        self._state.grabbed_object.release(
            self._interactor_id,
            event,
            self._state.velocity
        )

        self._emit_grab_callbacks(event)

        self._state.grabbed_object = None
        self._state.is_grabbing = False

    def _update_grabbed_object(self) -> None:
        """Update grabbed object position tracking."""
        if not self._state.grabbed_object:
            return

        self._state.grabbed_object.track_velocity(
            self._timestamp,
            self._state.position,
            self._state.rotation
        )

    def _release_all(self) -> None:
        """Release all interactions."""
        if self._state.is_grabbing:
            self._release_grab()
        if self._state.is_poking:
            self._end_poke()
        if self._state.hovered_interactable:
            event = self._create_event(InteractionType.HOVER)
            self._state.hovered_interactable.on_hover_exit(
                self._interactor_id, event
            )
            self._state.hovered_interactable = None

    def _create_event(self, interaction_type: InteractionType) -> InteractionEvent:
        """Create an interaction event.

        Args:
            interaction_type: Type of interaction

        Returns:
            New interaction event
        """
        return InteractionEvent(
            interactor_type=InteractorType.DIRECT,
            interactor_id=self._interactor_id,
            interaction_type=interaction_type,
            position=self._state.position,
            rotation=self._state.rotation,
            timestamp=self._timestamp,
            data={
                'velocity': self._state.velocity,
                'pinch_strength': self._state.pinch_strength,
                'grip_strength': self._state.grip_strength
            }
        )

    def _emit_hover_callbacks(self, event: InteractionEvent) -> None:
        """Emit hover callbacks."""
        for callback in self._hover_callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def _emit_poke_callbacks(self, event: InteractionEvent) -> None:
        """Emit poke callbacks."""
        for callback in self._poke_callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def _emit_grab_callbacks(self, event: InteractionEvent) -> None:
        """Emit grab callbacks."""
        for callback in self._grab_callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def add_hover_callback(
        self,
        callback: Callable[[InteractionEvent], None]
    ) -> None:
        """Add callback for hover events."""
        self._hover_callbacks.append(callback)

    def add_poke_callback(
        self,
        callback: Callable[[InteractionEvent], None]
    ) -> None:
        """Add callback for poke events."""
        self._poke_callbacks.append(callback)

    def add_grab_callback(
        self,
        callback: Callable[[InteractionEvent], None]
    ) -> None:
        """Add callback for grab events."""
        self._grab_callbacks.append(callback)

    def force_release(self) -> None:
        """Force release of current grab (for external systems)."""
        self._release_grab()

    def get_poke_progress(self) -> float:
        """Get current poke progress as 0.0-1.0.

        Returns:
            Poke progress, 0.0 if not poking
        """
        if not self._state.is_poking:
            return 0.0

        max_depth = self._config.poke_depth_threshold * 2.0
        return min(1.0, self._state.poke_depth / max_depth)


class MultiPointDirectInteractor:
    """Direct interactor supporting multiple simultaneous contact points.

    Enables multi-finger interaction for complex gestures and
    manipulation scenarios.
    """
    __slots__ = (
        '_interactor_id', '_contact_interactors', '_max_contacts',
        '_interactable_manager'
    )

    def __init__(
        self,
        interactor_id: int,
        max_contacts: int = 5,
        interactable_manager: Optional[InteractableManager] = None
    ):
        """Initialize multi-point direct interactor.

        Args:
            interactor_id: Base ID for this interactor
            max_contacts: Maximum simultaneous contact points
            interactable_manager: Manager for interactables
        """
        self._interactor_id = interactor_id
        self._max_contacts = max_contacts
        self._contact_interactors: dict[int, DirectInteractor] = {}
        self._interactable_manager = interactable_manager

    def update_contact(
        self,
        contact_index: int,
        position: Vec3,
        rotation: Quat,
        pinch_strength: float,
        grip_strength: float,
        timestamp: float
    ) -> None:
        """Update a specific contact point.

        Args:
            contact_index: Index of contact point (0-max_contacts)
            position: Contact position
            rotation: Contact rotation
            pinch_strength: Pinch strength at contact
            grip_strength: Grip strength at contact
            timestamp: Current timestamp
        """
        if contact_index >= self._max_contacts:
            return

        # Create interactor if needed
        if contact_index not in self._contact_interactors:
            self._contact_interactors[contact_index] = DirectInteractor(
                self._interactor_id * 100 + contact_index,
                interactable_manager=self._interactable_manager
            )

        self._contact_interactors[contact_index].update(
            position, rotation, pinch_strength, grip_strength, timestamp
        )

    def remove_contact(self, contact_index: int) -> None:
        """Remove a contact point.

        Args:
            contact_index: Index of contact to remove
        """
        if contact_index in self._contact_interactors:
            self._contact_interactors[contact_index].is_active = False
            del self._contact_interactors[contact_index]

    def get_active_contacts(self) -> list[DirectInteractor]:
        """Get all active contact interactors.

        Returns:
            List of active contact interactors
        """
        return list(self._contact_interactors.values())

    def release_all(self) -> None:
        """Release all contacts."""
        for interactor in self._contact_interactors.values():
            interactor.is_active = False
        self._contact_interactors.clear()
