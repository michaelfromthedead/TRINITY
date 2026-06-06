"""
Movement Component - Velocity, speed, and movement mode management.

Provides movement state management for entities including velocity tracking,
movement modes, and physics integration points.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from engine.core.math.vec import Vec3

from trinity.descriptors import (
    TrackedDescriptor,
    clear_dirty,
    is_dirty,
)

from engine.gameplay.components.constants import MovementConstants

if TYPE_CHECKING:
    from foundation import to_dict, from_dict


class MovementMode(Enum):
    """Movement modes that affect how movement is processed."""
    WALKING = auto()      # Ground-based walking
    RUNNING = auto()      # Faster ground movement
    SPRINTING = auto()    # Maximum speed ground movement
    CROUCHING = auto()    # Slow, reduced height
    SWIMMING = auto()     # Water-based movement
    FLYING = auto()       # Full 3D air movement
    FALLING = auto()      # Airborne, gravity-affected
    CLIMBING = auto()     # Surface-attached climbing
    SLIDING = auto()      # Reduced friction slide
    CUSTOM = auto()       # Game-specific mode


class MovementState(Enum):
    """Current movement state."""
    IDLE = auto()         # Not moving
    MOVING = auto()       # Active movement
    JUMPING = auto()      # Jump in progress
    AIRBORNE = auto()     # In the air (not jumping)
    LANDED = auto()       # Just landed
    DISABLED = auto()     # Movement disabled


@dataclass
class MovementSettings:
    """Configuration for a movement mode."""
    max_speed: float = 5.0
    acceleration: float = 20.0
    deceleration: float = 15.0
    turn_rate: float = 360.0  # Degrees per second
    gravity_scale: float = 1.0
    can_jump: bool = True
    jump_velocity: float = 8.0
    air_control: float = 0.3
    height_scale: float = 1.0  # For crouching


@dataclass
class MovementSnapshot:
    """Snapshot of movement state for networking/replay."""
    position: Vec3
    velocity: Vec3
    mode: MovementMode
    state: MovementState
    timestamp: float = 0.0


class MovementComponent:
    """
    Movement component with velocity, speed, and mode management.

    Features:
    - Multiple movement modes with different characteristics
    - Velocity tracking with acceleration/deceleration
    - Jump mechanics with coyote time and jump buffering
    - Ground detection integration
    - Input direction handling
    - Network interpolation support

    Attributes:
        velocity: Current velocity vector
        input_direction: Desired movement direction (normalized)
        movement_mode: Current movement mode
        movement_state: Current movement state
    """

    # Tracked descriptors
    velocity = TrackedDescriptor(field_type=Vec3, use_bitmask=True, field_offset=0)
    input_direction = TrackedDescriptor(field_type=Vec3, use_bitmask=True, field_offset=1)

    __slots__ = (
        "__dict__",
        "__weakref__",
        "_movement_mode",
        "_movement_state",
        "_mode_settings",
        "_is_grounded",
        "_ground_normal",
        "_last_grounded_time",
        "_jump_requested",
        "_jump_request_time",
        "_jumps_remaining",
        "_max_jumps",
        "_coyote_time",
        "_jump_buffer_time",
        "_facing_direction",
        "_movement_enabled",
        "_speed_multiplier",
        "_on_mode_changed",
        "_on_state_changed",
        "_on_landed",
        "_on_jumped",
        "_entity_id",
    )

    # Default settings for each movement mode
    DEFAULT_MODE_SETTINGS: Dict[MovementMode, MovementSettings] = {
        MovementMode.WALKING: MovementSettings(max_speed=4.0, acceleration=15.0),
        MovementMode.RUNNING: MovementSettings(max_speed=7.0, acceleration=20.0),
        MovementMode.SPRINTING: MovementSettings(max_speed=10.0, acceleration=25.0, turn_rate=180.0),
        MovementMode.CROUCHING: MovementSettings(max_speed=2.0, acceleration=10.0, height_scale=0.5, can_jump=False),
        MovementMode.SWIMMING: MovementSettings(max_speed=3.0, acceleration=8.0, gravity_scale=0.1, jump_velocity=4.0),
        MovementMode.FLYING: MovementSettings(max_speed=8.0, acceleration=12.0, gravity_scale=0.0, air_control=1.0),
        MovementMode.FALLING: MovementSettings(max_speed=50.0, acceleration=0.0, air_control=0.2),
        MovementMode.CLIMBING: MovementSettings(max_speed=2.0, acceleration=10.0, gravity_scale=0.0),
        MovementMode.SLIDING: MovementSettings(max_speed=12.0, acceleration=5.0, deceleration=3.0),
        MovementMode.CUSTOM: MovementSettings(),
    }

    def __init__(
        self,
        max_speed: float = 5.0,
        movement_mode: MovementMode = MovementMode.WALKING,
        entity_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the movement component.

        Args:
            max_speed: Default maximum movement speed
            movement_mode: Initial movement mode
            entity_id: Optional entity ID for tracking
        """
        self._movement_mode = movement_mode
        self._movement_state = MovementState.IDLE
        self._mode_settings: Dict[MovementMode, MovementSettings] = {
            mode: MovementSettings(
                max_speed=settings.max_speed,
                acceleration=settings.acceleration,
                deceleration=settings.deceleration,
                turn_rate=settings.turn_rate,
                gravity_scale=settings.gravity_scale,
                can_jump=settings.can_jump,
                jump_velocity=settings.jump_velocity,
                air_control=settings.air_control,
                height_scale=settings.height_scale,
            )
            for mode, settings in self.DEFAULT_MODE_SETTINGS.items()
        }

        # Ground state
        self._is_grounded = True
        self._ground_normal = Vec3.up()
        self._last_grounded_time = 0.0

        # Jump state
        self._jump_requested = False
        self._jump_request_time = 0.0
        self._jumps_remaining = 1
        self._max_jumps = MovementConstants.DEFAULT_MAX_JUMPS
        self._coyote_time = MovementConstants.DEFAULT_COYOTE_TIME
        self._jump_buffer_time = MovementConstants.DEFAULT_JUMP_BUFFER_TIME

        # Direction and control
        self._facing_direction = Vec3.forward()
        self._movement_enabled = True
        self._speed_multiplier = 1.0
        self._entity_id = entity_id

        # Callbacks
        self._on_mode_changed: List[Callable[[MovementMode, MovementMode], None]] = []
        self._on_state_changed: List[Callable[[MovementState, MovementState], None]] = []
        self._on_landed: List[Callable[[MovementComponent], None]] = []
        self._on_jumped: List[Callable[[MovementComponent], None]] = []

        # Set tracked values
        self.velocity = Vec3.zero()
        self.input_direction = Vec3.zero()

        # Override walking speed with provided value
        self._mode_settings[MovementMode.WALKING].max_speed = max_speed

        clear_dirty(self)

    # =========================================================================
    # MOVEMENT MODE
    # =========================================================================

    @property
    def movement_mode(self) -> MovementMode:
        """Get current movement mode."""
        return self._movement_mode

    @movement_mode.setter
    def movement_mode(self, value: MovementMode) -> None:
        """Set movement mode."""
        if value == self._movement_mode:
            return
        old_mode = self._movement_mode
        self._movement_mode = value
        for callback in self._on_mode_changed:
            callback(old_mode, value)

    @property
    def movement_state(self) -> MovementState:
        """Get current movement state."""
        return self._movement_state

    @movement_state.setter
    def movement_state(self, value: MovementState) -> None:
        """Set movement state."""
        if value == self._movement_state:
            return
        old_state = self._movement_state
        self._movement_state = value
        for callback in self._on_state_changed:
            callback(old_state, value)

    @property
    def current_settings(self) -> MovementSettings:
        """Get settings for current movement mode."""
        return self._mode_settings[self._movement_mode]

    def get_mode_settings(self, mode: MovementMode) -> MovementSettings:
        """Get settings for a specific mode."""
        return self._mode_settings[mode]

    def set_mode_settings(self, mode: MovementMode, settings: MovementSettings) -> None:
        """Set settings for a specific mode."""
        self._mode_settings[mode] = settings

    # =========================================================================
    # SPEED AND VELOCITY
    # =========================================================================

    @property
    def speed(self) -> float:
        """Get current speed (velocity magnitude on XZ plane)."""
        return Vec3(self.velocity.x, 0, self.velocity.z).length()

    @property
    def speed_3d(self) -> float:
        """Get current 3D speed."""
        return self.velocity.length()

    @property
    def max_speed(self) -> float:
        """Get maximum speed for current mode."""
        return self.current_settings.max_speed * self._speed_multiplier

    @property
    def speed_percentage(self) -> float:
        """Get current speed as percentage of max (0.0 to 1.0+)."""
        max_spd = self.max_speed
        if max_spd <= 0:
            return 0.0
        return self.speed / max_spd

    @property
    def is_moving(self) -> bool:
        """Check if currently moving."""
        return self.speed > MovementConstants.IS_MOVING_SPEED_THRESHOLD

    @property
    def horizontal_velocity(self) -> Vec3:
        """Get velocity on the horizontal plane (XZ)."""
        return Vec3(self.velocity.x, 0, self.velocity.z)

    @property
    def vertical_velocity(self) -> float:
        """Get vertical velocity component."""
        return self.velocity.y

    def set_velocity(self, velocity: Vec3) -> None:
        """Set velocity directly."""
        self.velocity = velocity

    def add_velocity(self, velocity: Vec3) -> None:
        """Add to current velocity."""
        self.velocity = self.velocity + velocity

    def add_impulse(self, impulse: Vec3) -> None:
        """Add an instant velocity change (impulse)."""
        self.velocity = self.velocity + impulse

    # =========================================================================
    # GROUND STATE
    # =========================================================================

    @property
    def is_grounded(self) -> bool:
        """Check if on ground."""
        return self._is_grounded

    @property
    def ground_normal(self) -> Vec3:
        """Get normal of the ground surface."""
        return self._ground_normal

    def set_grounded(self, grounded: bool, normal: Optional[Vec3] = None, current_time: float = 0.0) -> None:
        """
        Update grounded state.

        Args:
            grounded: Whether on ground
            normal: Ground surface normal
            current_time: Current game time for coyote time tracking
        """
        was_grounded = self._is_grounded
        self._is_grounded = grounded

        if normal is not None:
            self._ground_normal = normal
        elif grounded:
            self._ground_normal = Vec3.up()

        if grounded:
            self._last_grounded_time = current_time
            self._jumps_remaining = self._max_jumps

            # Handle landing
            if not was_grounded:
                self.movement_state = MovementState.LANDED
                for callback in self._on_landed:
                    callback(self)

                # Check for buffered jump
                if self._jump_requested and (current_time - self._jump_request_time) < self._jump_buffer_time:
                    self._execute_jump()
        else:
            if was_grounded:
                # Just left ground - record time for coyote time
                self._last_grounded_time = current_time
                if self.movement_state != MovementState.JUMPING:
                    self.movement_state = MovementState.AIRBORNE

    def can_use_coyote_time(self, current_time: float) -> bool:
        """Check if coyote time allows jumping."""
        return (current_time - self._last_grounded_time) < self._coyote_time

    # =========================================================================
    # JUMPING
    # =========================================================================

    @property
    def can_jump(self) -> bool:
        """Check if jumping is currently possible."""
        if not self._movement_enabled:
            return False
        if not self.current_settings.can_jump:
            return False
        return self._jumps_remaining > 0

    @property
    def jumps_remaining(self) -> int:
        """Get remaining jump count."""
        return self._jumps_remaining

    @property
    def max_jumps(self) -> int:
        """Get maximum jump count (for multi-jump)."""
        return self._max_jumps

    @max_jumps.setter
    def max_jumps(self, value: int) -> None:
        """Set maximum jump count."""
        self._max_jumps = max(0, value)
        self._jumps_remaining = min(self._jumps_remaining, self._max_jumps)

    def request_jump(self, current_time: float = 0.0) -> bool:
        """
        Request a jump. May be buffered if not currently possible.

        Args:
            current_time: Current game time for buffering

        Returns:
            True if jump was executed immediately
        """
        self._jump_requested = True
        self._jump_request_time = current_time

        # Check if can jump now (including coyote time)
        can_jump_now = self._jumps_remaining > 0 and (
            self._is_grounded or self.can_use_coyote_time(current_time)
        )

        if can_jump_now and self.current_settings.can_jump:
            self._execute_jump()
            return True

        return False

    def _execute_jump(self) -> None:
        """Execute a jump."""
        if self._jumps_remaining <= 0:
            return

        self._jumps_remaining -= 1
        self._jump_requested = False

        # Apply jump velocity
        jump_vel = self.current_settings.jump_velocity
        self.velocity = Vec3(self.velocity.x, jump_vel, self.velocity.z)

        self._is_grounded = False
        self.movement_state = MovementState.JUMPING

        for callback in self._on_jumped:
            callback(self)

    def cancel_jump(self) -> None:
        """Cancel a jump in progress (for variable jump height)."""
        if self.movement_state == MovementState.JUMPING and self.velocity.y > 0:
            # Reduce upward velocity
            self.velocity = Vec3(self.velocity.x, self.velocity.y * MovementConstants.CANCEL_JUMP_VELOCITY_FACTOR, self.velocity.z)

    # =========================================================================
    # INPUT PROCESSING
    # =========================================================================

    def set_input_direction(self, direction: Vec3) -> None:
        """Set the desired movement direction (should be normalized or zero)."""
        if direction.length_squared() > 1.0:
            direction = direction.normalized()
        self.input_direction = direction

    def clear_input(self) -> None:
        """Clear movement input."""
        self.input_direction = Vec3.zero()

    @property
    def has_input(self) -> bool:
        """Check if there's active movement input."""
        return self.input_direction.length_squared() > MovementConstants.HAS_INPUT_THRESHOLD

    @property
    def facing_direction(self) -> Vec3:
        """Get the direction the entity is facing."""
        return self._facing_direction

    @facing_direction.setter
    def facing_direction(self, value: Vec3) -> None:
        """Set facing direction (will be normalized)."""
        if value.length_squared() > MovementConstants.FACING_DIRECTION_THRESHOLD:
            self._facing_direction = value.normalized()

    # =========================================================================
    # MOVEMENT CONTROL
    # =========================================================================

    @property
    def movement_enabled(self) -> bool:
        """Check if movement is enabled."""
        return self._movement_enabled

    @movement_enabled.setter
    def movement_enabled(self, value: bool) -> None:
        """Enable or disable movement."""
        self._movement_enabled = value
        if not value:
            self.movement_state = MovementState.DISABLED

    @property
    def speed_multiplier(self) -> float:
        """Get speed multiplier."""
        return self._speed_multiplier

    @speed_multiplier.setter
    def speed_multiplier(self, value: float) -> None:
        """Set speed multiplier."""
        self._speed_multiplier = max(0.0, value)

    def stop(self) -> None:
        """Immediately stop all movement."""
        self.velocity = Vec3.zero()
        self.input_direction = Vec3.zero()
        self.movement_state = MovementState.IDLE

    def freeze(self) -> None:
        """Freeze all movement (disable and stop)."""
        self.movement_enabled = False
        self.stop()

    def unfreeze(self) -> None:
        """Unfreeze movement."""
        self.movement_enabled = True
        self.movement_state = MovementState.IDLE

    # =========================================================================
    # UPDATE
    # =========================================================================

    def update(self, delta_time: float, current_time: float = 0.0) -> None:
        """
        Update movement state (should be called each frame).

        Args:
            delta_time: Time since last update
            current_time: Current game time
        """
        if not self._movement_enabled:
            return

        settings = self.current_settings

        # Update movement state based on input
        if self._is_grounded:
            if self.has_input:
                self.movement_state = MovementState.MOVING
            elif self.speed < MovementConstants.IDLE_SPEED_THRESHOLD:
                self.movement_state = MovementState.IDLE
        else:
            if self.movement_state not in (MovementState.JUMPING,):
                self.movement_state = MovementState.AIRBORNE

        # Determine control factor (reduced in air)
        control = 1.0 if self._is_grounded else settings.air_control

        # Calculate target velocity from input
        if self.has_input:
            target_velocity = self.input_direction * self.max_speed
        else:
            target_velocity = Vec3.zero()

        # Accelerate/decelerate toward target
        current_horizontal = Vec3(self.velocity.x, 0, self.velocity.z)
        target_horizontal = Vec3(target_velocity.x, 0, target_velocity.z)

        diff = target_horizontal - current_horizontal

        if diff.length_squared() > MovementConstants.VELOCITY_DIFF_THRESHOLD:
            # Determine acceleration rate
            if target_horizontal.length_squared() > current_horizontal.length_squared():
                accel_rate = settings.acceleration
            else:
                accel_rate = settings.deceleration

            accel_rate *= control

            # Apply acceleration
            accel = diff.normalized() * accel_rate * delta_time
            if accel.length_squared() > diff.length_squared():
                accel = diff  # Don't overshoot

            new_horizontal = current_horizontal + accel
            self.velocity = Vec3(new_horizontal.x, self.velocity.y, new_horizontal.z)

        # Update facing direction based on movement
        if self.is_moving and self.has_input:
            self.facing_direction = self.horizontal_velocity

        # Post-update state check: transition to IDLE if speed dropped below threshold
        if self._is_grounded and not self.has_input and self.speed < MovementConstants.IDLE_SPEED_THRESHOLD:
            self.movement_state = MovementState.IDLE

    def apply_gravity(self, gravity: float, delta_time: float) -> None:
        """
        Apply gravity to velocity.

        Args:
            gravity: Gravity acceleration (positive = downward)
            delta_time: Time since last update
        """
        if not self._is_grounded:
            scaled_gravity = gravity * self.current_settings.gravity_scale
            self.velocity = Vec3(
                self.velocity.x,
                self.velocity.y - scaled_gravity * delta_time,
                self.velocity.z,
            )

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_mode_changed(self, callback: Callable[[MovementMode, MovementMode], None]) -> None:
        """Register callback for mode changes (old_mode, new_mode)."""
        self._on_mode_changed.append(callback)

    def on_state_changed(self, callback: Callable[[MovementState, MovementState], None]) -> None:
        """Register callback for state changes (old_state, new_state)."""
        self._on_state_changed.append(callback)

    def on_landed(self, callback: Callable[[MovementComponent], None]) -> None:
        """Register callback for landing."""
        self._on_landed.append(callback)

    def on_jumped(self, callback: Callable[[MovementComponent], None]) -> None:
        """Register callback for jumping."""
        self._on_jumped.append(callback)

    # =========================================================================
    # SNAPSHOTS
    # =========================================================================

    def create_snapshot(self, position: Vec3, timestamp: float = 0.0) -> MovementSnapshot:
        """Create a snapshot of current movement state."""
        return MovementSnapshot(
            position=position,
            velocity=Vec3(self.velocity.x, self.velocity.y, self.velocity.z),
            mode=self._movement_mode,
            state=self._movement_state,
            timestamp=timestamp,
        )

    def apply_snapshot(self, snapshot: MovementSnapshot) -> None:
        """Apply a snapshot to restore movement state."""
        self.velocity = snapshot.velocity
        self._movement_mode = snapshot.mode
        self._movement_state = snapshot.state

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize movement component to dictionary."""
        return {
            "velocity": [self.velocity.x, self.velocity.y, self.velocity.z],
            "input_direction": [self.input_direction.x, self.input_direction.y, self.input_direction.z],
            "movement_mode": self._movement_mode.name,
            "movement_state": self._movement_state.name,
            "is_grounded": self._is_grounded,
            "jumps_remaining": self._jumps_remaining,
            "max_jumps": self._max_jumps,
            "speed_multiplier": self._speed_multiplier,
            "movement_enabled": self._movement_enabled,
            "facing_direction": [self._facing_direction.x, self._facing_direction.y, self._facing_direction.z],
            "entity_id": self._entity_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MovementComponent:
        """Deserialize movement component from dictionary."""
        mode = MovementMode[data.get("movement_mode", "WALKING")]
        component = cls(movement_mode=mode, entity_id=data.get("entity_id"))

        component.velocity = Vec3(*data.get("velocity", [0, 0, 0]))
        component.input_direction = Vec3(*data.get("input_direction", [0, 0, 0]))
        component._movement_state = MovementState[data.get("movement_state", "IDLE")]
        component._is_grounded = data.get("is_grounded", True)
        component._jumps_remaining = data.get("jumps_remaining", 1)
        component._max_jumps = data.get("max_jumps", 1)
        component._speed_multiplier = data.get("speed_multiplier", 1.0)
        component._movement_enabled = data.get("movement_enabled", True)

        if "facing_direction" in data:
            component._facing_direction = Vec3(*data["facing_direction"])

        return component

    def __repr__(self) -> str:
        return (
            f"MovementComponent(mode={self._movement_mode.name}, "
            f"state={self._movement_state.name}, speed={self.speed:.1f})"
        )


# Descriptor setup
MovementComponent.velocity.__set_name__(MovementComponent, "velocity")
MovementComponent.input_direction.__set_name__(MovementComponent, "input_direction")


__all__ = [
    "MovementComponent",
    "MovementMode",
    "MovementState",
    "MovementSettings",
    "MovementSnapshot",
]
