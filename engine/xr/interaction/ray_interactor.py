"""Ray-based XR interactor for laser pointer style interactions.

This module provides ray casting interaction for VR/AR,
supporting hover, select, and grab at a distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform, RigidTransform
from engine.xr.utils.math_utils import rotation_from_direction

from .interactable import (
    XRInteractable,
    InteractionEvent,
    InteractionType,
    InteractorType,
    InteractionHit,
    InteractableManager,
)
from .grabbable import XRGrabbable, GrabType
from engine.xr.config import XR_CONFIG


class RayVisualMode(Enum):
    """Visual representation modes for the ray."""
    LINE = auto()         # Simple line
    CURVED = auto()       # Curved/bezier line
    DASHED = auto()       # Dashed line
    GRADIENT = auto()     # Gradient opacity line
    HIDDEN = auto()       # No visual


class RayHitIndicator(Enum):
    """Visual indicator at ray hit point."""
    NONE = auto()
    CIRCLE = auto()
    RETICLE = auto()
    CUSTOM = auto()


@dataclass(slots=True)
class RayConfig:
    """Configuration for ray interactor behavior."""
    max_distance: float = XR_CONFIG.interaction.RAY_MAX_LENGTH
    ray_width: float = XR_CONFIG.interaction.RAY_WIDTH
    visual_mode: RayVisualMode = RayVisualMode.LINE
    hit_indicator: RayHitIndicator = RayHitIndicator.RETICLE
    curve_points: int = 20
    select_threshold: float = XR_CONFIG.interaction.SMOOTHING_FACTOR    # Trigger threshold for select
    grab_threshold: float = XR_CONFIG.interaction.GRAB_ACTIVATION_THRESHOLD      # Trigger threshold for grab
    sticky_hover: bool = False       # Keep hovering even when ray moves off
    auto_select_on_hover: bool = False


@dataclass(slots=True)
class RayState:
    """Current state of the ray interactor."""
    origin: Vec3 = field(default_factory=Vec3.zero)
    direction: Vec3 = field(default_factory=lambda: Vec3(0, 0, -1))
    current_hit: Optional[InteractionHit] = None
    hovered_interactable: Optional[XRInteractable] = None
    selected_interactable: Optional[XRInteractable] = None
    grabbed_object: Optional[XRGrabbable] = None
    trigger_value: float = 0.0
    is_selecting: bool = False
    is_grabbing: bool = False


@dataclass(slots=True)
class RayCastResult:
    """Result of a ray cast operation."""
    hit: bool
    point: Vec3 = field(default_factory=Vec3.zero)
    normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    distance: float = float('inf')
    entity_id: Optional[int] = None
    interactable: Optional[XRInteractable] = None


class RayInteractor:
    """Ray-based interactor for XR laser pointer interactions.

    Provides ray casting interaction for hovering, selecting, and
    grabbing objects at a distance. Commonly used for UI interaction
    and distant object manipulation.

    Attributes:
        interactor_id: Unique identifier for this interactor
        config: Ray configuration settings
        is_active: Whether interactor is currently active
    """
    __slots__ = (
        '_interactor_id', '_config', '_state', '_is_active', '_layer_mask',
        '_raycast_callback', '_visual_callback', '_select_callbacks',
        '_hover_callbacks', '_grab_callbacks', '_interactable_manager',
        '_timestamp'
    )

    def __init__(
        self,
        interactor_id: int,
        config: Optional[RayConfig] = None,
        interactable_manager: Optional[InteractableManager] = None
    ):
        """Initialize the ray interactor.

        Args:
            interactor_id: Unique ID for this interactor
            config: Ray configuration (uses defaults if None)
            interactable_manager: Manager for finding interactables
        """
        self._interactor_id = interactor_id
        self._config = config or RayConfig()
        self._state = RayState()
        self._is_active = True
        self._layer_mask: list[str] | None = None
        self._raycast_callback: Optional[Callable[[Vec3, Vec3, float], list[RayCastResult]]] = None
        self._visual_callback: Optional[Callable[[RayState], None]] = None
        self._select_callbacks: list[Callable[[InteractionEvent], None]] = []
        self._hover_callbacks: list[Callable[[InteractionEvent], None]] = []
        self._grab_callbacks: list[Callable[[InteractionEvent], None]] = []
        self._interactable_manager = interactable_manager
        self._timestamp = 0.0

    @property
    def interactor_id(self) -> int:
        """Get the interactor ID."""
        return self._interactor_id

    @property
    def config(self) -> RayConfig:
        """Get the ray configuration."""
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
    def state(self) -> RayState:
        """Get the current ray state."""
        return self._state

    @property
    def is_hovering(self) -> bool:
        """Check if ray is hovering over an interactable."""
        return self._state.hovered_interactable is not None

    @property
    def is_selecting(self) -> bool:
        """Check if currently selecting."""
        return self._state.is_selecting

    @property
    def is_grabbing(self) -> bool:
        """Check if currently grabbing."""
        return self._state.is_grabbing

    @property
    def current_hit_point(self) -> Optional[Vec3]:
        """Get the current ray hit point."""
        if self._state.current_hit:
            return self._state.current_hit.hit_point
        return None

    def set_layer_mask(self, layers: list[str] | None) -> None:
        """Set the layer mask for filtering interactions.

        Args:
            layers: List of layer names, or None for all layers
        """
        self._layer_mask = layers

    def set_raycast_callback(
        self,
        callback: Callable[[Vec3, Vec3, float], list[RayCastResult]]
    ) -> None:
        """Set the physics raycast callback.

        Args:
            callback: Function(origin, direction, max_distance) -> [RayCastResult]
        """
        self._raycast_callback = callback

    def set_visual_callback(
        self,
        callback: Callable[[RayState], None]
    ) -> None:
        """Set callback for updating ray visual.

        Args:
            callback: Function(state) for visual updates
        """
        self._visual_callback = callback

    def update(
        self,
        origin: Vec3,
        direction: Vec3,
        trigger_value: float,
        timestamp: float
    ) -> None:
        """Update the ray interactor state.

        Args:
            origin: Ray origin in world space
            direction: Ray direction (normalized)
            trigger_value: Trigger/button value 0.0-1.0
            timestamp: Current frame timestamp
        """
        if not self._is_active:
            return

        self._timestamp = timestamp
        self._state.origin = origin
        self._state.direction = direction.normalized()
        self._state.trigger_value = trigger_value

        # Perform raycast
        hits = self._perform_raycast()

        # Process hover
        self._process_hover(hits)

        # Process select/grab based on trigger
        self._process_trigger(trigger_value)

        # Update grabbed object if any
        if self._state.grabbed_object:
            self._update_grabbed_object()

        # Update visual
        if self._visual_callback:
            self._visual_callback(self._state)

    def _perform_raycast(self) -> list[InteractionHit]:
        """Perform physics raycast and convert to interaction hits.

        Returns:
            List of interaction hits sorted by distance
        """
        if not self._raycast_callback:
            return []

        raw_results = self._raycast_callback(
            self._state.origin,
            self._state.direction,
            self._config.max_distance
        )

        hits = []
        for result in raw_results:
            if not result.hit:
                continue

            interactable = result.interactable
            if interactable is None and self._interactable_manager:
                # Try to find interactable by entity
                if result.entity_id is not None:
                    interactable = self._interactable_manager.get(result.entity_id)

            if interactable and interactable.enabled:
                hits.append(InteractionHit(
                    interactable=interactable,
                    hit_point=result.point,
                    hit_normal=result.normal,
                    distance=result.distance,
                    collider_id=result.entity_id
                ))

        # Filter by layer mask
        if self._layer_mask:
            hits = [h for h in hits if any(
                h.interactable.is_in_layer(l) for l in self._layer_mask
            )]

        # Sort by priority, then distance
        hits.sort(key=lambda h: (-h.interactable.priority, h.distance))

        return hits

    def _process_hover(self, hits: list[InteractionHit]) -> None:
        """Process hover state changes.

        Args:
            hits: Current interaction hits
        """
        new_hover = hits[0].interactable if hits else None

        # Check sticky hover
        if self._config.sticky_hover and self._state.is_selecting:
            new_hover = self._state.hovered_interactable

        # Update current hit
        self._state.current_hit = hits[0] if hits else None

        # Handle hover change
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

                # Auto-select if configured
                if self._config.auto_select_on_hover:
                    self._start_select(new_hover)

            self._state.hovered_interactable = new_hover

    def _process_trigger(self, trigger_value: float) -> None:
        """Process trigger input for select/grab.

        Args:
            trigger_value: Current trigger value 0.0-1.0
        """
        select_pressed = trigger_value >= self._config.select_threshold
        grab_pressed = trigger_value >= self._config.grab_threshold

        # Handle grab state
        if grab_pressed and not self._state.is_grabbing:
            self._try_grab()
        elif not grab_pressed and self._state.is_grabbing:
            self._release_grab()

        # Handle select state (only if not grabbing)
        if not self._state.is_grabbing:
            if select_pressed and not self._state.is_selecting:
                if self._state.hovered_interactable:
                    self._start_select(self._state.hovered_interactable)
            elif not select_pressed and self._state.is_selecting:
                self._end_select()

    def _try_grab(self) -> None:
        """Attempt to grab the hovered interactable."""
        if not self._state.hovered_interactable:
            return

        # Check if it's grabbable
        if not isinstance(self._state.hovered_interactable, XRGrabbable):
            return

        grabbable = self._state.hovered_interactable

        if not grabbable.can_be_grabbed(self._interactor_id, GrabType.RAY):
            return

        # Get interactor transform
        interactor_transform = RigidTransform(
            self._state.origin,
            rotation_from_direction(self._state.direction)
        )

        # Get object transform (simplified)
        hit_point = self._state.current_hit.hit_point if self._state.current_hit else self._state.origin
        object_transform = Transform(translation=hit_point)

        event = self._create_event(InteractionType.GRAB)

        if grabbable.try_grab(
            self._interactor_id,
            GrabType.RAY,
            interactor_transform,
            object_transform,
            event
        ):
            self._state.grabbed_object = grabbable
            self._state.is_grabbing = True
            self._emit_grab_callbacks(event)

    def _release_grab(self) -> None:
        """Release the currently grabbed object."""
        if not self._state.grabbed_object:
            return

        event = self._create_event(InteractionType.GRAB)

        # Calculate throw velocity from ray movement
        throw_data = self._state.grabbed_object.release(
            self._interactor_id,
            event
        )

        self._emit_grab_callbacks(event)

        self._state.grabbed_object = None
        self._state.is_grabbing = False

    def _start_select(self, interactable: XRInteractable) -> None:
        """Start selecting an interactable.

        Args:
            interactable: The interactable to select
        """
        event = self._create_event(InteractionType.SELECT)
        interactable.on_select_enter(self._interactor_id, event)
        self._state.selected_interactable = interactable
        self._state.is_selecting = True
        self._emit_select_callbacks(event)

    def _end_select(self) -> None:
        """End the current selection."""
        if self._state.selected_interactable:
            event = self._create_event(InteractionType.SELECT)
            self._state.selected_interactable.on_select_exit(
                self._interactor_id, event
            )
            self._emit_select_callbacks(event)

        self._state.selected_interactable = None
        self._state.is_selecting = False

    def _update_grabbed_object(self) -> None:
        """Update the position of grabbed object along ray."""
        if not self._state.grabbed_object or not self._state.current_hit:
            return

        # Track velocity for throwing
        hit = self._state.current_hit
        rotation = rotation_from_direction(self._state.direction)

        self._state.grabbed_object.track_velocity(
            self._timestamp,
            hit.hit_point,
            rotation
        )

    def _release_all(self) -> None:
        """Release all interactions."""
        if self._state.is_grabbing:
            self._release_grab()
        if self._state.is_selecting:
            self._end_select()
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
        hit_point = self._state.origin
        if self._state.current_hit:
            hit_point = self._state.current_hit.hit_point

        return InteractionEvent(
            interactor_type=InteractorType.RAY,
            interactor_id=self._interactor_id,
            interaction_type=interaction_type,
            position=hit_point,
            rotation=rotation_from_direction(self._state.direction),
            timestamp=self._timestamp,
            data={
                'ray_origin': self._state.origin,
                'ray_direction': self._state.direction,
                'trigger_value': self._state.trigger_value
            }
        )

    def _emit_hover_callbacks(self, event: InteractionEvent) -> None:
        """Emit hover callbacks."""
        for callback in self._hover_callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def _emit_select_callbacks(self, event: InteractionEvent) -> None:
        """Emit select callbacks."""
        for callback in self._select_callbacks:
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

    def add_select_callback(
        self,
        callback: Callable[[InteractionEvent], None]
    ) -> None:
        """Add callback for select events."""
        self._select_callbacks.append(callback)

    def add_grab_callback(
        self,
        callback: Callable[[InteractionEvent], None]
    ) -> None:
        """Add callback for grab events."""
        self._grab_callbacks.append(callback)

    def get_ray_points(self) -> list[Vec3]:
        """Get points for rendering the ray visual.

        Returns:
            List of points along the ray
        """
        if self._config.visual_mode == RayVisualMode.HIDDEN:
            return []

        end_distance = self._config.max_distance
        if self._state.current_hit:
            end_distance = self._state.current_hit.distance

        end_point = self._state.origin + self._state.direction * end_distance

        if self._config.visual_mode == RayVisualMode.CURVED:
            return self._generate_curved_points(end_point)

        return [self._state.origin, end_point]

    def _generate_curved_points(self, end_point: Vec3) -> list[Vec3]:
        """Generate points for a curved ray (bezier).

        Args:
            end_point: Target end point

        Returns:
            List of points along curve
        """
        points = []
        num_points = self._config.curve_points

        # Control point for curve (slightly below midpoint)
        mid = (self._state.origin + end_point) * 0.5
        control = Vec3(mid.x, mid.y - 0.2, mid.z)

        for i in range(num_points):
            t = i / (num_points - 1)
            t2 = t * t
            mt = 1 - t
            mt2 = mt * mt

            # Quadratic bezier
            x = mt2 * self._state.origin.x + 2 * mt * t * control.x + t2 * end_point.x
            y = mt2 * self._state.origin.y + 2 * mt * t * control.y + t2 * end_point.y
            z = mt2 * self._state.origin.z + 2 * mt * t * control.z + t2 * end_point.z

            points.append(Vec3(x, y, z))

        return points

    def activate(self, interactable: XRInteractable) -> None:
        """Send an activation event to an interactable.

        Args:
            interactable: The interactable to activate
        """
        if not interactable.enabled:
            return

        event = self._create_event(InteractionType.ACTIVATE)
        interactable.on_activate(event)
