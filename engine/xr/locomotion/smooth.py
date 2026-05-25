"""
Smooth locomotion for XR environments.

Provides continuous movement with various input modes and comfort features
including snap turn, smooth turn, and vignette support.

Components:
    SmoothLocomotion  - Main smooth locomotion component
    TurnSettings      - Turn behavior configuration

Decorators:
    @xr_locomotion    - Configure locomotion behavior

Based on XR_CONTEXT.md specifications:
    - Smooth locomotion: Thumbstick, Arm swing, Head-directed, Hand-directed
    - Turn types: Snap turn, Smooth turn
    - Comfort settings: snap_turn_enabled, snap_turn_angle (15-90), smooth_turn_speed
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Callable, Optional

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry
from engine.xr.utils.markers import Tracked, Range, Observable, Transient, Immutable

if TYPE_CHECKING:
    pass


# =============================================================================
# ENUMS
# =============================================================================


class MovementMode(Enum):
    """Movement input modes for smooth locomotion."""

    THUMBSTICK = "thumbstick"  # Standard thumbstick/joystick input
    ARM_SWING = "arm_swing"  # Arm swing detection
    HEAD_DIRECTED = "head_directed"  # Movement relative to head direction
    HAND_DIRECTED = "hand_directed"  # Movement relative to hand direction


class TurnType(Enum):
    """Turning behavior types."""

    SNAP = "snap"  # Discrete snap turns
    SMOOTH = "smooth"  # Continuous smooth turning
    DISABLED = "disabled"  # No turning via locomotion


class MovementState(Enum):
    """Smooth locomotion state machine states."""

    IDLE = "idle"  # Not moving
    MOVING = "moving"  # Actively moving
    TURNING = "turning"  # Performing snap turn
    DISABLED = "disabled"  # Locomotion disabled


class StrafeBehavior(Enum):
    """Strafe movement behavior."""

    NORMAL = "normal"  # Standard strafe speed
    REDUCED = "reduced"  # Reduced strafe speed for comfort
    DISABLED = "disabled"  # No strafing allowed


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MovementInput:
    """Input state for smooth locomotion."""

    forward: float = 0.0  # -1 to 1, forward/backward
    strafe: float = 0.0  # -1 to 1, left/right
    turn: float = 0.0  # -1 to 1, turn left/right
    sprint: bool = False  # Sprint modifier


@dataclass
class MovementResult:
    """Result of movement calculation for a frame."""

    velocity: tuple[float, float, float]
    rotation_delta: float
    is_moving: bool
    is_turning: bool
    vignette_intensity: float = 0.0


@dataclass
class ArmSwingData:
    """Data for arm swing movement detection."""

    left_hand_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_hand_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    swing_threshold: float = 0.5  # Minimum velocity to register swing
    swing_multiplier: float = 2.0  # Movement speed multiplier


# =============================================================================
# TURN SETTINGS
# =============================================================================


@dataclass
class TurnSettings:
    """
    Configuration for turning behavior.

    Attributes:
        turn_type: Snap or smooth turning
        snap_angle: Degrees per snap turn (15-90)
        smooth_speed: Degrees per second for smooth turn
        snap_cooldown: Cooldown between snap turns
        dead_zone: Input dead zone for turn stick
    """

    turn_type: Annotated[TurnType, Tracked] = TurnType.SNAP
    snap_angle: Annotated[float, Tracked, Range(15.0, 90.0)] = 45.0
    smooth_speed: Annotated[float, Tracked, Range(30.0, 180.0)] = 90.0
    snap_cooldown: Annotated[float, Tracked, Range(0.0, 0.5)] = 0.15
    dead_zone: Annotated[float, Tracked, Range(0.0, 0.5)] = 0.2

    # Internal state
    _snap_cooldown_remaining: Annotated[float, Transient] = 0.0
    _last_snap_direction: Annotated[int, Transient] = 0  # -1, 0, or 1

    def can_snap_turn(self, direction: int) -> bool:
        """Check if snap turn is available."""
        if self.turn_type != TurnType.SNAP:
            return False
        if self._snap_cooldown_remaining > 0:
            return False
        return True

    def execute_snap_turn(self, direction: int) -> float:
        """
        Execute a snap turn.

        Args:
            direction: -1 for left, 1 for right

        Returns:
            Rotation delta in radians
        """
        if not self.can_snap_turn(direction):
            return 0.0

        self._snap_cooldown_remaining = self.snap_cooldown
        self._last_snap_direction = direction

        return math.radians(self.snap_angle * direction)

    def calculate_smooth_turn(self, input_value: float, delta_time: float) -> float:
        """
        Calculate smooth turn rotation.

        Args:
            input_value: Turn input (-1 to 1)
            delta_time: Time since last frame

        Returns:
            Rotation delta in radians
        """
        if self.turn_type != TurnType.SMOOTH:
            return 0.0

        # Apply dead zone
        if abs(input_value) < self.dead_zone:
            return 0.0

        # Normalize input after dead zone
        sign = 1 if input_value > 0 else -1
        normalized = (abs(input_value) - self.dead_zone) / (1.0 - self.dead_zone)
        normalized *= sign

        return math.radians(self.smooth_speed * normalized * delta_time)

    def update(self, delta_time: float) -> None:
        """Update turn settings state."""
        if self._snap_cooldown_remaining > 0:
            self._snap_cooldown_remaining -= delta_time
            if self._snap_cooldown_remaining < 0:
                self._snap_cooldown_remaining = 0.0


# =============================================================================
# SMOOTH LOCOMOTION COMPONENT
# =============================================================================


@dataclass
class SmoothLocomotion:
    """
    Smooth continuous locomotion with comfort features.

    Provides continuous movement controlled by thumbstick, arm swing,
    or other input methods with configurable comfort settings.

    Attributes:
        mode: Movement input mode
        move_speed: Forward/backward movement speed (m/s)
        strafe_speed: Left/right strafe speed (m/s)
        sprint_multiplier: Speed multiplier when sprinting
        turn_settings: Turn behavior configuration
        vignette_enabled: Enable comfort vignette during movement
        vignette_intensity: Base vignette intensity
        vignette_velocity_threshold: Velocity threshold for vignette activation
    """

    # Mode settings
    mode: Annotated[MovementMode, Tracked] = MovementMode.THUMBSTICK
    state: Annotated[MovementState, Tracked, Observable] = MovementState.IDLE

    # Movement speeds
    move_speed: Annotated[float, Tracked, Range(0.5, 10.0)] = 3.0
    strafe_speed: Annotated[float, Tracked, Range(0.5, 10.0)] = 2.0
    backward_speed_multiplier: Annotated[float, Tracked, Range(0.25, 1.0)] = 0.7
    sprint_multiplier: Annotated[float, Tracked, Range(1.0, 3.0)] = 1.5

    # Strafe behavior
    strafe_behavior: Annotated[StrafeBehavior, Tracked] = StrafeBehavior.NORMAL
    strafe_reduction_factor: Annotated[float, Tracked, Range(0.25, 1.0)] = 0.5

    # Turn settings (embedded)
    turn_type: Annotated[TurnType, Tracked] = TurnType.SNAP
    snap_angle: Annotated[float, Tracked, Range(15.0, 90.0)] = 45.0
    smooth_turn_speed: Annotated[float, Tracked, Range(30.0, 180.0)] = 90.0

    # Input settings
    dead_zone: Annotated[float, Tracked, Range(0.0, 0.5)] = 0.1
    input_curve: Annotated[float, Tracked, Range(1.0, 3.0)] = 1.5  # Input response curve

    # Comfort: Vignette
    vignette_enabled: Annotated[bool, Tracked] = True
    vignette_intensity: Annotated[float, Tracked, Range(0.0, 1.0)] = 0.5
    vignette_velocity_threshold: Annotated[float, Tracked, Range(0.0, 5.0)] = 0.5
    vignette_angular_velocity_threshold: Annotated[float, Tracked, Range(0.0, 90.0)] = 30.0

    # Direction reference
    direction_source: Annotated[str, Tracked] = "head"  # head, left_hand, right_hand

    # Gravity/falling
    gravity_enabled: Annotated[bool, Tracked] = True
    gravity: Annotated[float, Tracked] = -9.8
    is_grounded: Annotated[bool, Tracked] = True

    # Internal state
    _turn_settings: Annotated[Optional[TurnSettings], Transient] = None
    _current_velocity: Annotated[tuple[float, float, float], Transient] = (0.0, 0.0, 0.0)
    _current_vignette: Annotated[float, Transient] = 0.0
    _arm_swing_data: Annotated[Optional[ArmSwingData], Transient] = None

    # Callbacks
    _on_movement_start: Annotated[Optional[Callable[[], None]], Transient] = None
    _on_movement_stop: Annotated[Optional[Callable[[], None]], Transient] = None

    def __post_init__(self) -> None:
        """Initialize internal state after dataclass init."""
        self._turn_settings = TurnSettings(
            turn_type=self.turn_type,
            snap_angle=self.snap_angle,
            smooth_speed=self.smooth_turn_speed,
            dead_zone=self.dead_zone,
        )
        self._arm_swing_data = ArmSwingData()

    def _apply_input_curve(self, value: float) -> float:
        """Apply input response curve."""
        sign = 1 if value >= 0 else -1
        return sign * (abs(value) ** self.input_curve)

    def _apply_dead_zone(self, value: float) -> float:
        """Apply dead zone to input value."""
        if abs(value) < self.dead_zone:
            return 0.0
        sign = 1 if value > 0 else -1
        normalized = (abs(value) - self.dead_zone) / (1.0 - self.dead_zone)
        return sign * normalized

    def calculate_movement(
        self,
        input_state: MovementInput,
        forward_direction: tuple[float, float, float],
        delta_time: float,
    ) -> MovementResult:
        """
        Calculate movement for the current frame.

        Args:
            input_state: Current input state
            forward_direction: Direction to move forward (normalized)
            delta_time: Time since last frame

        Returns:
            MovementResult with velocity, rotation, and vignette data
        """
        # Ensure turn settings are initialized
        if self._turn_settings is None:
            self._turn_settings = TurnSettings(
                turn_type=self.turn_type,
                snap_angle=self.snap_angle,
                smooth_speed=self.smooth_turn_speed,
                dead_zone=self.dead_zone,
            )

        # Apply dead zone and input curve
        forward_input = self._apply_input_curve(
            self._apply_dead_zone(input_state.forward)
        )
        strafe_input = self._apply_input_curve(
            self._apply_dead_zone(input_state.strafe)
        )
        turn_input = self._apply_dead_zone(input_state.turn)

        # Calculate speeds
        forward_speed = self.move_speed
        if forward_input < 0:
            forward_speed *= self.backward_speed_multiplier
        if input_state.sprint and forward_input > 0:
            forward_speed *= self.sprint_multiplier

        strafe_final_speed = self.strafe_speed
        if self.strafe_behavior == StrafeBehavior.REDUCED:
            strafe_final_speed *= self.strafe_reduction_factor
        elif self.strafe_behavior == StrafeBehavior.DISABLED:
            strafe_final_speed = 0.0
            strafe_input = 0.0

        # Calculate velocity components
        forward_velocity = forward_input * forward_speed
        strafe_velocity = strafe_input * strafe_final_speed

        # Build velocity vector
        # Forward direction is (fx, fy, fz), right is perpendicular
        fx, fy, fz = forward_direction
        # Right vector (perpendicular to forward in XZ plane)
        right_x = fz
        right_z = -fx

        vx = fx * forward_velocity + right_x * strafe_velocity
        vy = 0.0 if self.is_grounded else self.gravity * delta_time
        vz = fz * forward_velocity + right_z * strafe_velocity

        velocity = (vx, vy, vz)
        self._current_velocity = velocity

        # Calculate rotation
        rotation_delta = 0.0
        is_turning = False

        if self.turn_type == TurnType.SNAP:
            if abs(turn_input) > 0.5:  # Threshold for snap turn
                direction = 1 if turn_input > 0 else -1
                rotation_delta = self._turn_settings.execute_snap_turn(direction)
                if rotation_delta != 0:
                    is_turning = True
        elif self.turn_type == TurnType.SMOOTH:
            rotation_delta = self._turn_settings.calculate_smooth_turn(
                turn_input, delta_time
            )
            is_turning = abs(rotation_delta) > 0.001

        # Calculate vignette intensity
        vignette = 0.0
        if self.vignette_enabled:
            # Linear velocity contribution
            speed = math.sqrt(vx * vx + vz * vz)
            if speed > self.vignette_velocity_threshold:
                linear_factor = min(
                    (speed - self.vignette_velocity_threshold) / 2.0, 1.0
                )
                vignette = max(vignette, linear_factor * self.vignette_intensity)

            # Angular velocity contribution
            MIN_DELTA_TIME = 1e-6
            angular_speed = abs(math.degrees(rotation_delta / max(delta_time, MIN_DELTA_TIME)))
            if angular_speed > self.vignette_angular_velocity_threshold:
                angular_factor = min(
                    (angular_speed - self.vignette_angular_velocity_threshold) / 60.0,
                    1.0,
                )
                vignette = max(vignette, angular_factor * self.vignette_intensity)

        self._current_vignette = vignette

        # Determine if moving
        is_moving = abs(forward_input) > 0.01 or abs(strafe_input) > 0.01

        # Update state
        was_moving = self.state == MovementState.MOVING
        if is_moving and not was_moving:
            self.state = MovementState.MOVING
            if self._on_movement_start:
                self._on_movement_start()
        elif not is_moving and was_moving:
            self.state = MovementState.IDLE
            if self._on_movement_stop:
                self._on_movement_stop()

        return MovementResult(
            velocity=velocity,
            rotation_delta=rotation_delta,
            is_moving=is_moving,
            is_turning=is_turning,
            vignette_intensity=vignette,
        )

    def calculate_arm_swing_movement(
        self,
        left_hand_velocity: tuple[float, float, float],
        right_hand_velocity: tuple[float, float, float],
        forward_direction: tuple[float, float, float],
        delta_time: float,
    ) -> MovementResult:
        """
        Calculate movement from arm swing input.

        Args:
            left_hand_velocity: Left hand velocity vector
            right_hand_velocity: Right hand velocity vector
            forward_direction: Forward direction for movement
            delta_time: Time since last frame

        Returns:
            MovementResult based on arm swing
        """
        if self._arm_swing_data is None:
            self._arm_swing_data = ArmSwingData()

        # Calculate swing magnitude (primarily vertical component of hand movement)
        left_swing = abs(left_hand_velocity[1])
        right_swing = abs(right_hand_velocity[1])
        avg_swing = (left_swing + right_swing) / 2.0

        # Convert to forward input if above threshold
        forward_input = 0.0
        if avg_swing > self._arm_swing_data.swing_threshold:
            normalized_swing = (
                avg_swing - self._arm_swing_data.swing_threshold
            ) / (2.0 - self._arm_swing_data.swing_threshold)
            forward_input = min(normalized_swing * self._arm_swing_data.swing_multiplier, 1.0)

        # Create input state and delegate to standard calculation
        input_state = MovementInput(forward=forward_input, strafe=0.0, turn=0.0)
        return self.calculate_movement(input_state, forward_direction, delta_time)

    def update(self, delta_time: float) -> None:
        """
        Update internal state.

        Args:
            delta_time: Time since last frame
        """
        if self._turn_settings is not None:
            self._turn_settings.update(delta_time)

    def set_movement_callbacks(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Set movement event callbacks.

        Args:
            on_start: Called when movement begins
            on_stop: Called when movement stops
        """
        self._on_movement_start = on_start
        self._on_movement_stop = on_stop

    def get_current_vignette_intensity(self) -> float:
        """Get current vignette intensity based on movement."""
        return self._current_vignette

    def set_grounded(self, grounded: bool) -> None:
        """Set whether the player is on the ground."""
        self.is_grounded = grounded


# =============================================================================
# DECORATOR: @xr_locomotion
# =============================================================================


def _validate_xr_locomotion(
    locomotion_type: str = "smooth", speed: float = 3.0, **_: Any
) -> None:
    """Validate xr_locomotion decorator parameters."""
    valid_types = {"smooth", "teleport", "climbing", "hybrid"}
    if locomotion_type not in valid_types:
        raise ValueError(
            f"@xr_locomotion: 'locomotion_type' must be one of {valid_types}, "
            f"got '{locomotion_type}'"
        )
    if speed <= 0:
        raise ValueError(
            f"@xr_locomotion: 'speed' must be positive, got {speed}"
        )


def _xr_locomotion_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for xr_locomotion decorator."""
    locomotion_type = params.get("locomotion_type", "smooth")
    speed = params.get("speed", 3.0)
    return [
        Step(Op.TAG, {"key": "xr_locomotion", "value": True}),
        Step(Op.TAG, {"key": "locomotion_type", "value": locomotion_type}),
        Step(Op.TAG, {"key": "locomotion_speed", "value": speed}),
        Step(Op.REGISTER, {"registry": "xr"}),
    ]


def _after_xr_locomotion(target: Any, params: dict[str, Any]) -> Any:
    """Post-processing for xr_locomotion decorator."""
    target._xr_locomotion = True
    target._locomotion_type = params.get("locomotion_type", "smooth")
    target._locomotion_speed = params.get("speed", 3.0)
    return None


xr_locomotion = make_decorator(
    name="xr_locomotion",
    steps=_xr_locomotion_steps,
    doc="Configure XR locomotion behavior for a component.",
    validate=_validate_xr_locomotion,
    after_steps=_after_xr_locomotion,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("xr_locomotion", xr_locomotion, ("class", "function")),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.GAMEPLAY,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.GAMEPLAY].append(_spec)


# =============================================================================
# PROVIDER INTERFACE
# =============================================================================


class SmoothLocomotionProvider:
    """
    Abstract provider interface for smooth locomotion.

    Subclass this to integrate with specific XR runtimes or input systems.
    """

    def __init__(self, locomotion: SmoothLocomotion):
        """
        Initialize provider with locomotion component.

        Args:
            locomotion: SmoothLocomotion component to control
        """
        self.locomotion = locomotion

    def on_input(
        self,
        forward: float,
        strafe: float,
        turn: float,
        sprint: bool = False,
    ) -> None:
        """
        Process input and update movement state.

        Args:
            forward: Forward/backward input (-1 to 1)
            strafe: Left/right strafe input (-1 to 1)
            turn: Turn input (-1 to 1)
            sprint: Sprint modifier active
        """
        # Input is processed in calculate_movement
        pass

    def update(
        self,
        input_state: MovementInput,
        forward_direction: tuple[float, float, float],
        delta_time: float,
    ) -> MovementResult:
        """
        Update locomotion and get movement result.

        Args:
            input_state: Current input state
            forward_direction: Direction for forward movement
            delta_time: Time since last frame

        Returns:
            MovementResult with velocity and rotation
        """
        self.locomotion.update(delta_time)
        return self.locomotion.calculate_movement(
            input_state, forward_direction, delta_time
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "MovementMode",
    "TurnType",
    "MovementState",
    "StrafeBehavior",
    # Data classes
    "MovementInput",
    "MovementResult",
    "ArmSwingData",
    "TurnSettings",
    # Components
    "SmoothLocomotion",
    # Provider
    "SmoothLocomotionProvider",
    # Decorator
    "xr_locomotion",
    # Type markers
    "Tracked",
    "Range",
    "Observable",
    "Transient",
    "Immutable",
]
