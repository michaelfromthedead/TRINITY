"""Gaze-based XR interactor for eye tracking and head-gaze interactions.

This module provides gaze interaction for VR/AR, supporting hover
selection with dwell time activation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, Optional

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import RigidTransform
from engine.xr.utils.math_utils import rotation_from_direction

from .interactable import (
    XRInteractable,
    InteractionEvent,
    InteractionType,
    InteractorType,
    InteractionHit,
    InteractableManager,
)


class GazeSource(Enum):
    """Source of gaze direction."""
    HEAD = auto()          # Head-based gaze (HMD forward)
    EYE_CENTER = auto()    # Combined eye tracking
    EYE_LEFT = auto()      # Left eye only
    EYE_RIGHT = auto()     # Right eye only


class ActivationMode(Enum):
    """How gaze interaction is activated."""
    DWELL = auto()         # Hold gaze for duration
    BLINK = auto()         # Eye blink triggers
    BUTTON = auto()        # External button trigger
    PINCH = auto()         # Hand pinch gesture


class DwellIndicator(Enum):
    """Visual indicator for dwell progress."""
    NONE = auto()
    RADIAL = auto()        # Radial fill around reticle
    LINEAR = auto()        # Linear progress bar
    SHRINK = auto()        # Shrinking circle


@dataclass(slots=True)
class GazeConfig:
    """Configuration for gaze interactor."""
    gaze_source: GazeSource = GazeSource.HEAD
    activation_mode: ActivationMode = ActivationMode.DWELL
    dwell_time: float = 1.0            # Seconds to dwell for activation
    dwell_indicator: DwellIndicator = DwellIndicator.RADIAL
    max_distance: float = 20.0         # Maximum gaze interaction distance
    reticle_size: float = 0.01         # Base reticle size
    sticky_target: bool = True         # Keep targeting during dwell
    require_stable_gaze: bool = True   # Require stable gaze for dwell
    stability_threshold: float = 0.02  # Movement threshold for stability


@dataclass(slots=True)
class EyeTrackingData:
    """Eye tracking data for gaze calculation."""
    left_eye_origin: Vec3 = field(default_factory=Vec3.zero)
    left_eye_direction: Vec3 = field(default_factory=lambda: Vec3(0, 0, -1))
    right_eye_origin: Vec3 = field(default_factory=Vec3.zero)
    right_eye_direction: Vec3 = field(default_factory=lambda: Vec3(0, 0, -1))
    convergence_distance: float = 1.0
    left_openness: float = 1.0         # 0.0 = closed, 1.0 = open
    right_openness: float = 1.0
    is_tracking: bool = False


@dataclass(slots=True)
class GazeState:
    """Current state of gaze interactor."""
    origin: Vec3 = field(default_factory=Vec3.zero)
    direction: Vec3 = field(default_factory=lambda: Vec3(0, 0, -1))
    hit_point: Vec3 = field(default_factory=Vec3.zero)
    hit_distance: float = float('inf')
    hovered_interactable: Optional[XRInteractable] = None
    dwell_target: Optional[XRInteractable] = None
    dwell_start_time: float = 0.0
    dwell_progress: float = 0.0
    is_dwelling: bool = False
    last_activation_time: float = 0.0
    gaze_stable: bool = True
    eye_data: Optional[EyeTrackingData] = None


class GazeInteractor:
    """Gaze-based interactor for eye/head tracking interactions.

    Provides gaze-based interaction with dwell-time activation,
    commonly used for hands-free interaction or as a fallback
    when controllers aren't available.

    Attributes:
        interactor_id: Unique identifier for this interactor
        config: Gaze configuration settings
        is_active: Whether interactor is active
    """
    __slots__ = (
        '_interactor_id', '_config', '_state', '_is_active', '_layer_mask',
        '_raycast_callback', '_visual_callback', '_dwell_callbacks',
        '_activation_callbacks', '_hover_callbacks', '_interactable_manager',
        '_timestamp', '_gaze_history', '_external_trigger'
    )

    def __init__(
        self,
        interactor_id: int,
        config: Optional[GazeConfig] = None,
        interactable_manager: Optional[InteractableManager] = None
    ):
        """Initialize the gaze interactor.

        Args:
            interactor_id: Unique ID for this interactor
            config: Gaze configuration (uses defaults if None)
            interactable_manager: Manager for finding interactables
        """
        self._interactor_id = interactor_id
        self._config = config or GazeConfig()
        self._state = GazeState()
        self._is_active = True
        self._layer_mask: list[str] | None = None
        self._raycast_callback: Optional[Callable[[Vec3, Vec3, float], list[InteractionHit]]] = None
        self._visual_callback: Optional[Callable[[GazeState], None]] = None
        self._dwell_callbacks: list[Callable[[float], None]] = []
        self._activation_callbacks: list[Callable[[InteractionEvent], None]] = []
        self._hover_callbacks: list[Callable[[InteractionEvent], None]] = []
        self._interactable_manager = interactable_manager
        self._timestamp = 0.0
        # Use deque with maxlen for efficient gaze history without reallocations
        # At 90Hz, 45 samples covers ~500ms which is what we track
        self._gaze_history: Deque[tuple[float, Vec3]] = deque(maxlen=45)
        self._external_trigger = False

    @property
    def interactor_id(self) -> int:
        """Get the interactor ID."""
        return self._interactor_id

    @property
    def config(self) -> GazeConfig:
        """Get the gaze configuration."""
        return self._config

    @property
    def is_active(self) -> bool:
        """Check if interactor is active."""
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        """Enable or disable the interactor."""
        if not value and self._is_active:
            self._reset_state()
        self._is_active = value

    @property
    def state(self) -> GazeState:
        """Get the current state."""
        return self._state

    @property
    def is_hovering(self) -> bool:
        """Check if gaze is hovering over an interactable."""
        return self._state.hovered_interactable is not None

    @property
    def is_dwelling(self) -> bool:
        """Check if currently dwelling on a target."""
        return self._state.is_dwelling

    @property
    def dwell_progress(self) -> float:
        """Get current dwell progress 0.0-1.0."""
        return self._state.dwell_progress

    @property
    def current_target(self) -> Optional[XRInteractable]:
        """Get the current gaze target."""
        return self._state.hovered_interactable

    def set_layer_mask(self, layers: list[str] | None) -> None:
        """Set the layer mask for filtering interactions.

        Args:
            layers: List of layer names, or None for all
        """
        self._layer_mask = layers

    def set_raycast_callback(
        self,
        callback: Callable[[Vec3, Vec3, float], list[InteractionHit]]
    ) -> None:
        """Set the physics raycast callback.

        Args:
            callback: Function(origin, direction, max_distance) -> [InteractionHit]
        """
        self._raycast_callback = callback

    def set_visual_callback(
        self,
        callback: Callable[[GazeState], None]
    ) -> None:
        """Set callback for visual updates (reticle, indicators).

        Args:
            callback: Function(state) for visual updates
        """
        self._visual_callback = callback

    def set_external_trigger(self, triggered: bool) -> None:
        """Set external trigger state for BUTTON/PINCH activation modes.

        Args:
            triggered: Whether external trigger is active
        """
        self._external_trigger = triggered

    def update_head_gaze(
        self,
        head_position: Vec3,
        head_forward: Vec3,
        timestamp: float
    ) -> None:
        """Update gaze from head tracking.

        Args:
            head_position: HMD world position
            head_forward: HMD forward direction
            timestamp: Current frame timestamp
        """
        if not self._is_active:
            return

        if self._config.gaze_source != GazeSource.HEAD:
            return

        self._update_gaze(head_position, head_forward.normalized(), timestamp)

    def update_eye_tracking(
        self,
        eye_data: EyeTrackingData,
        timestamp: float
    ) -> None:
        """Update gaze from eye tracking data.

        Args:
            eye_data: Eye tracking data
            timestamp: Current frame timestamp
        """
        if not self._is_active:
            return

        if self._config.gaze_source == GazeSource.HEAD:
            return

        if not eye_data.is_tracking:
            return

        self._state.eye_data = eye_data

        # Calculate gaze based on source
        origin, direction = self._calculate_eye_gaze(eye_data)
        self._update_gaze(origin, direction, timestamp)

        # Check for blink activation
        if self._config.activation_mode == ActivationMode.BLINK:
            self._check_blink_activation(eye_data, timestamp)

    def _calculate_eye_gaze(
        self,
        eye_data: EyeTrackingData
    ) -> tuple[Vec3, Vec3]:
        """Calculate gaze origin and direction from eye data.

        Args:
            eye_data: Eye tracking data

        Returns:
            (origin, direction) tuple
        """
        if self._config.gaze_source == GazeSource.EYE_LEFT:
            return (eye_data.left_eye_origin, eye_data.left_eye_direction)
        elif self._config.gaze_source == GazeSource.EYE_RIGHT:
            return (eye_data.right_eye_origin, eye_data.right_eye_direction)
        else:  # EYE_CENTER
            # Average of both eyes
            origin = (eye_data.left_eye_origin + eye_data.right_eye_origin) * 0.5
            direction = (eye_data.left_eye_direction + eye_data.right_eye_direction).normalized()
            return (origin, direction)

    def _update_gaze(
        self,
        origin: Vec3,
        direction: Vec3,
        timestamp: float
    ) -> None:
        """Update gaze state and process interactions.

        Args:
            origin: Gaze ray origin
            direction: Gaze ray direction
            timestamp: Current timestamp
        """
        self._timestamp = timestamp
        self._state.origin = origin
        self._state.direction = direction

        # Update gaze history for stability check
        self._update_gaze_history(timestamp, direction)

        # Check gaze stability
        self._state.gaze_stable = self._check_gaze_stability()

        # Perform raycast
        hits = self._perform_raycast()

        # Update hit info
        if hits:
            self._state.hit_point = hits[0].hit_point
            self._state.hit_distance = hits[0].distance
        else:
            self._state.hit_point = origin + direction * self._config.max_distance
            self._state.hit_distance = self._config.max_distance

        # Process hover
        self._process_hover(hits)

        # Process dwell/activation
        self._process_activation(timestamp)

        # Update visual
        if self._visual_callback:
            self._visual_callback(self._state)

    def _update_gaze_history(self, timestamp: float, direction: Vec3) -> None:
        """Update gaze direction history for stability tracking.

        Args:
            timestamp: Current timestamp
            direction: Current gaze direction
        """
        # deque with maxlen automatically drops oldest entries, no reallocation needed
        self._gaze_history.append((timestamp, direction))

        # Remove samples older than 500ms from the front (deque popleft is O(1))
        cutoff = timestamp - 0.5
        while self._gaze_history and self._gaze_history[0][0] < cutoff:
            self._gaze_history.popleft()

    def _check_gaze_stability(self) -> bool:
        """Check if gaze has been stable (not moving much).

        Returns:
            True if gaze is stable
        """
        if not self._config.require_stable_gaze:
            return True

        if len(self._gaze_history) < 3:
            return True

        # Calculate variance in gaze direction
        recent = self._gaze_history[-10:] if len(self._gaze_history) >= 10 else self._gaze_history

        avg_dir = Vec3.zero()
        for _, d in recent:
            avg_dir = avg_dir + d
        avg_dir = avg_dir / len(recent)

        max_deviation = 0.0
        for _, d in recent:
            deviation = (d - avg_dir).length()
            max_deviation = max(max_deviation, deviation)

        return max_deviation < self._config.stability_threshold

    def _perform_raycast(self) -> list[InteractionHit]:
        """Perform gaze raycast.

        Returns:
            List of interaction hits
        """
        if not self._raycast_callback:
            return []

        hits = self._raycast_callback(
            self._state.origin,
            self._state.direction,
            self._config.max_distance
        )

        # Filter by layer
        if self._layer_mask:
            hits = [h for h in hits if any(
                h.interactable.is_in_layer(l) for l in self._layer_mask
            )]

        hits.sort(key=lambda h: h.distance)
        return hits

    def _process_hover(self, hits: list[InteractionHit]) -> None:
        """Process hover state changes.

        Args:
            hits: Current raycast hits
        """
        new_hover = hits[0].interactable if hits else None

        # Sticky target during dwell
        if self._config.sticky_target and self._state.is_dwelling:
            new_hover = self._state.dwell_target

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

    def _process_activation(self, timestamp: float) -> None:
        """Process dwell/activation based on mode.

        Args:
            timestamp: Current timestamp
        """
        if self._config.activation_mode == ActivationMode.DWELL:
            self._process_dwell(timestamp)
        elif self._config.activation_mode == ActivationMode.BUTTON:
            self._process_button_activation(timestamp)
        elif self._config.activation_mode == ActivationMode.PINCH:
            self._process_button_activation(timestamp)
        # BLINK is handled in update_eye_tracking

    def _process_dwell(self, timestamp: float) -> None:
        """Process dwell-based activation.

        Args:
            timestamp: Current timestamp
        """
        target = self._state.hovered_interactable

        # Check if target changed or gaze unstable
        if target != self._state.dwell_target or not self._state.gaze_stable:
            # Reset dwell
            self._state.dwell_target = target
            self._state.dwell_start_time = timestamp
            self._state.dwell_progress = 0.0
            self._state.is_dwelling = False

            if target:
                self._state.is_dwelling = True

            return

        # Update dwell progress
        if self._state.is_dwelling and target:
            elapsed = timestamp - self._state.dwell_start_time
            self._state.dwell_progress = min(1.0, elapsed / self._config.dwell_time)

            # Emit progress callbacks
            for callback in self._dwell_callbacks:
                try:
                    callback(self._state.dwell_progress)
                except Exception:
                    pass

            # Check for activation
            if self._state.dwell_progress >= 1.0:
                self._activate_target(target, timestamp)
                self._reset_dwell(timestamp)

    def _process_button_activation(self, timestamp: float) -> None:
        """Process button/pinch triggered activation.

        Args:
            timestamp: Current timestamp
        """
        if self._external_trigger and self._state.hovered_interactable:
            # Prevent rapid re-activation
            if timestamp - self._state.last_activation_time > 0.3:
                self._activate_target(
                    self._state.hovered_interactable,
                    timestamp
                )

    def _check_blink_activation(
        self,
        eye_data: EyeTrackingData,
        timestamp: float
    ) -> None:
        """Check for blink-based activation.

        Args:
            eye_data: Current eye tracking data
            timestamp: Current timestamp
        """
        # Detect intentional blink (both eyes closed briefly)
        both_closed = eye_data.left_openness < 0.2 and eye_data.right_openness < 0.2

        if both_closed and self._state.hovered_interactable:
            if timestamp - self._state.last_activation_time > 0.5:
                self._activate_target(
                    self._state.hovered_interactable,
                    timestamp
                )

    def _activate_target(
        self,
        target: XRInteractable,
        timestamp: float
    ) -> None:
        """Activate the target interactable.

        Args:
            target: Interactable to activate
            timestamp: Current timestamp
        """
        self._state.last_activation_time = timestamp

        # Send select events
        select_event = self._create_event(InteractionType.SELECT)
        target.on_select_enter(self._interactor_id, select_event)
        target.on_select_exit(self._interactor_id, select_event)

        # Send activate event
        activate_event = self._create_event(InteractionType.ACTIVATE)
        target.on_activate(activate_event)

        self._emit_activation_callbacks(activate_event)

    def _reset_dwell(self, timestamp: float) -> None:
        """Reset dwell state after activation.

        Args:
            timestamp: Current timestamp
        """
        self._state.dwell_start_time = timestamp
        self._state.dwell_progress = 0.0
        # Keep dwelling on same target but reset progress

    def _reset_state(self) -> None:
        """Reset all interaction state."""
        if self._state.hovered_interactable:
            event = self._create_event(InteractionType.HOVER)
            self._state.hovered_interactable.on_hover_exit(
                self._interactor_id, event
            )

        self._state = GazeState()
        self._gaze_history.clear()

    def _create_event(self, interaction_type: InteractionType) -> InteractionEvent:
        """Create an interaction event.

        Args:
            interaction_type: Type of interaction

        Returns:
            New interaction event
        """
        return InteractionEvent(
            interactor_type=InteractorType.GAZE,
            interactor_id=self._interactor_id,
            interaction_type=interaction_type,
            position=self._state.hit_point,
            rotation=rotation_from_direction(self._state.direction),
            timestamp=self._timestamp,
            data={
                'gaze_origin': self._state.origin,
                'gaze_direction': self._state.direction,
                'dwell_progress': self._state.dwell_progress,
                'gaze_stable': self._state.gaze_stable
            }
        )

    def _emit_hover_callbacks(self, event: InteractionEvent) -> None:
        """Emit hover callbacks."""
        for callback in self._hover_callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def _emit_activation_callbacks(self, event: InteractionEvent) -> None:
        """Emit activation callbacks."""
        for callback in self._activation_callbacks:
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

    def add_dwell_callback(
        self,
        callback: Callable[[float], None]
    ) -> None:
        """Add callback for dwell progress updates."""
        self._dwell_callbacks.append(callback)

    def add_activation_callback(
        self,
        callback: Callable[[InteractionEvent], None]
    ) -> None:
        """Add callback for activation events."""
        self._activation_callbacks.append(callback)

    def get_reticle_info(self) -> dict:
        """Get info for rendering gaze reticle.

        Returns:
            Dictionary with reticle rendering info
        """
        return {
            'position': self._state.hit_point,
            'distance': self._state.hit_distance,
            'size': self._config.reticle_size * (1.0 + self._state.hit_distance * 0.1),
            'dwell_progress': self._state.dwell_progress,
            'indicator_type': self._config.dwell_indicator,
            'is_hovering': self.is_hovering,
            'is_dwelling': self._state.is_dwelling
        }
