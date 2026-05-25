"""
Movement Mode System.

Defines different movement states (walking, running, crouching, etc.)
and handles transitions between them with appropriate speed and acceleration
parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from .character_controller import Vector3
from .config import (
    AIR_ACCELERATION,
    AIR_DECELERATION,
    CLIMBING_SPEED,
    CROUCHING_SPEED,
    DEFAULT_CROUCHED_HEIGHT,
    DEFAULT_PRONE_HEIGHT,
    FLYING_SPEED,
    GROUND_ACCELERATION,
    GROUND_DECELERATION,
    PRONE_SPEED,
    RUNNING_SPEED,
    SPRINTING_SPEED,
    SWIMMING_SPEED,
    TURN_ACCELERATION,
    WALKING_SPEED,
)


# =============================================================================
# Movement Modes
# =============================================================================

class MovementMode(Enum):
    """Available movement modes for characters."""
    WALKING = auto()
    RUNNING = auto()
    SPRINTING = auto()
    CROUCHING = auto()
    PRONE = auto()
    SWIMMING = auto()
    CLIMBING = auto()
    FLYING = auto()
    FALLING = auto()
    LADDERING = auto()
    SLIDING = auto()
    VAULTING = auto()
    HANGING = auto()
    CUSTOM = auto()


class MovementContext(Enum):
    """Environmental context affecting movement."""
    GROUND = auto()
    AIR = auto()
    WATER = auto()
    LADDER = auto()
    ROPE = auto()
    LEDGE = auto()


# =============================================================================
# Mode Parameters
# =============================================================================

@dataclass
class MovementModeParams:
    """
    Parameters for a specific movement mode.

    Attributes:
        max_speed: Maximum speed in this mode (units/sec)
        acceleration: Acceleration rate (units/sec^2)
        deceleration: Deceleration rate (units/sec^2)
        turn_speed: Turning speed multiplier
        gravity_scale: Gravity multiplier
        can_jump: Whether jumping is allowed
        height_modifier: Capsule height modifier
        friction_modifier: Surface friction modifier
        stamina_cost: Stamina cost per second
    """
    max_speed: float = 5.0
    acceleration: float = 20.0
    deceleration: float = 15.0
    turn_speed: float = 1.0
    gravity_scale: float = 1.0
    can_jump: bool = True
    height_modifier: float = 1.0
    friction_modifier: float = 1.0
    stamina_cost: float = 0.0
    air_control: float = 0.3


# Default parameters for each mode
DEFAULT_MODE_PARAMS: dict[MovementMode, MovementModeParams] = {
    MovementMode.WALKING: MovementModeParams(
        max_speed=WALKING_SPEED,
        acceleration=GROUND_ACCELERATION,
        deceleration=GROUND_DECELERATION,
        can_jump=True,
        stamina_cost=0.0,
    ),
    MovementMode.RUNNING: MovementModeParams(
        max_speed=RUNNING_SPEED,
        acceleration=GROUND_ACCELERATION * 1.2,
        deceleration=GROUND_DECELERATION,
        can_jump=True,
        stamina_cost=5.0,
    ),
    MovementMode.SPRINTING: MovementModeParams(
        max_speed=SPRINTING_SPEED,
        acceleration=GROUND_ACCELERATION * 1.5,
        deceleration=GROUND_DECELERATION * 0.8,
        can_jump=True,
        stamina_cost=15.0,
    ),
    MovementMode.CROUCHING: MovementModeParams(
        max_speed=CROUCHING_SPEED,
        acceleration=GROUND_ACCELERATION * 0.8,
        deceleration=GROUND_DECELERATION * 1.2,
        can_jump=True,
        height_modifier=DEFAULT_CROUCHED_HEIGHT / 1.8,
        stamina_cost=0.0,
    ),
    MovementMode.PRONE: MovementModeParams(
        max_speed=PRONE_SPEED,
        acceleration=GROUND_ACCELERATION * 0.5,
        deceleration=GROUND_DECELERATION * 2.0,
        can_jump=False,
        height_modifier=DEFAULT_PRONE_HEIGHT / 1.8,
        stamina_cost=0.0,
    ),
    MovementMode.SWIMMING: MovementModeParams(
        max_speed=SWIMMING_SPEED,
        acceleration=8.0,
        deceleration=10.0,
        gravity_scale=0.1,
        can_jump=False,
        friction_modifier=0.3,
        stamina_cost=10.0,
        air_control=1.0,
    ),
    MovementMode.CLIMBING: MovementModeParams(
        max_speed=CLIMBING_SPEED,
        acceleration=10.0,
        deceleration=20.0,
        gravity_scale=0.0,
        can_jump=True,
        stamina_cost=8.0,
    ),
    MovementMode.FLYING: MovementModeParams(
        max_speed=FLYING_SPEED,
        acceleration=15.0,
        deceleration=10.0,
        gravity_scale=0.0,
        can_jump=False,
        air_control=1.0,
        stamina_cost=5.0,
    ),
    MovementMode.FALLING: MovementModeParams(
        max_speed=50.0,  # Terminal velocity
        acceleration=AIR_ACCELERATION,
        deceleration=AIR_DECELERATION,
        gravity_scale=1.0,
        can_jump=False,
        air_control=0.3,
    ),
    MovementMode.LADDERING: MovementModeParams(
        max_speed=2.5,
        acceleration=15.0,
        deceleration=25.0,
        gravity_scale=0.0,
        can_jump=True,
        stamina_cost=3.0,
    ),
    MovementMode.SLIDING: MovementModeParams(
        max_speed=12.0,
        acceleration=5.0,
        deceleration=3.0,
        gravity_scale=1.5,
        can_jump=True,
        height_modifier=0.5,
        friction_modifier=0.2,
    ),
    MovementMode.VAULTING: MovementModeParams(
        max_speed=6.0,
        acceleration=30.0,
        deceleration=30.0,
        gravity_scale=0.0,
        can_jump=False,
    ),
    MovementMode.HANGING: MovementModeParams(
        max_speed=2.0,
        acceleration=10.0,
        deceleration=20.0,
        gravity_scale=0.0,
        can_jump=True,
        stamina_cost=5.0,
    ),
    MovementMode.CUSTOM: MovementModeParams(),
}


# =============================================================================
# Movement State
# =============================================================================

@dataclass
class MovementState:
    """
    Current movement state including mode, velocity, and transition data.

    Attributes:
        mode: Current movement mode
        context: Environmental context
        current_velocity: Current velocity vector
        target_velocity: Target velocity vector
        current_speed: Current scalar speed
        is_transitioning: Whether a mode transition is in progress
        transition_progress: Progress of current transition (0-1)
        previous_mode: Mode before transition
        time_in_mode: Time spent in current mode (seconds)
    """
    mode: MovementMode = MovementMode.WALKING
    context: MovementContext = MovementContext.GROUND
    current_velocity: Vector3 = field(default_factory=Vector3.zero)
    target_velocity: Vector3 = field(default_factory=Vector3.zero)
    current_speed: float = 0.0
    is_transitioning: bool = False
    transition_progress: float = 1.0
    previous_mode: MovementMode = MovementMode.WALKING
    time_in_mode: float = 0.0
    stamina: float = 100.0
    max_stamina: float = 100.0

    def get_params(self) -> MovementModeParams:
        """Get parameters for current mode."""
        return DEFAULT_MODE_PARAMS.get(self.mode, MovementModeParams())


# =============================================================================
# Transition Rules
# =============================================================================

@dataclass
class TransitionRule:
    """
    Rule for transitioning between movement modes.

    Attributes:
        from_mode: Source movement mode
        to_mode: Target movement mode
        duration: Transition duration in seconds
        requires_grounded: Whether ground contact is required
        min_stamina: Minimum stamina required
        condition: Optional custom condition function
    """
    from_mode: MovementMode
    to_mode: MovementMode
    duration: float = 0.0
    requires_grounded: bool = False
    min_stamina: float = 0.0
    condition: Optional[Callable[[MovementState], bool]] = None


# Default transition rules
TRANSITION_RULES: list[TransitionRule] = [
    # Walking transitions
    TransitionRule(MovementMode.WALKING, MovementMode.RUNNING),
    TransitionRule(MovementMode.WALKING, MovementMode.CROUCHING, duration=0.2, requires_grounded=True),
    TransitionRule(MovementMode.WALKING, MovementMode.PRONE, duration=0.5, requires_grounded=True),

    # Running transitions
    TransitionRule(MovementMode.RUNNING, MovementMode.WALKING),
    TransitionRule(MovementMode.RUNNING, MovementMode.SPRINTING, min_stamina=20.0),
    TransitionRule(MovementMode.RUNNING, MovementMode.SLIDING, duration=0.3, min_stamina=10.0),

    # Sprinting transitions
    TransitionRule(MovementMode.SPRINTING, MovementMode.RUNNING),
    TransitionRule(MovementMode.SPRINTING, MovementMode.SLIDING, duration=0.3),

    # Crouching transitions
    TransitionRule(MovementMode.CROUCHING, MovementMode.WALKING, duration=0.2),
    TransitionRule(MovementMode.CROUCHING, MovementMode.PRONE, duration=0.3),

    # Prone transitions
    TransitionRule(MovementMode.PRONE, MovementMode.CROUCHING, duration=0.3),
    TransitionRule(MovementMode.PRONE, MovementMode.WALKING, duration=0.5),

    # Falling transitions
    TransitionRule(MovementMode.FALLING, MovementMode.WALKING, requires_grounded=True),
    TransitionRule(MovementMode.WALKING, MovementMode.FALLING),
    TransitionRule(MovementMode.RUNNING, MovementMode.FALLING),

    # Swimming transitions
    TransitionRule(MovementMode.WALKING, MovementMode.SWIMMING, duration=0.2),
    TransitionRule(MovementMode.SWIMMING, MovementMode.WALKING, duration=0.2, requires_grounded=True),

    # Climbing transitions
    TransitionRule(MovementMode.WALKING, MovementMode.CLIMBING),
    TransitionRule(MovementMode.CLIMBING, MovementMode.FALLING),
    TransitionRule(MovementMode.CLIMBING, MovementMode.HANGING),

    # Hanging transitions
    TransitionRule(MovementMode.HANGING, MovementMode.CLIMBING),
    TransitionRule(MovementMode.HANGING, MovementMode.FALLING),

    # Ladder transitions
    TransitionRule(MovementMode.WALKING, MovementMode.LADDERING),
    TransitionRule(MovementMode.LADDERING, MovementMode.WALKING),
    TransitionRule(MovementMode.LADDERING, MovementMode.FALLING),

    # Sliding transitions
    TransitionRule(MovementMode.SLIDING, MovementMode.CROUCHING, duration=0.2),
    TransitionRule(MovementMode.SLIDING, MovementMode.WALKING, duration=0.3),

    # Vaulting transitions
    TransitionRule(MovementMode.RUNNING, MovementMode.VAULTING, duration=0.1),
    TransitionRule(MovementMode.VAULTING, MovementMode.WALKING, duration=0.1),
    TransitionRule(MovementMode.VAULTING, MovementMode.FALLING),
]


# =============================================================================
# Movement Mode Manager
# =============================================================================

class MovementModeManager:
    """
    Manages movement modes and transitions.

    Handles:
    - Mode transitions with rules and validation
    - Velocity calculation based on mode parameters
    - Stamina management
    - Movement context tracking
    """

    def __init__(self):
        self._state = MovementState()
        self._custom_params: dict[MovementMode, MovementModeParams] = {}
        self._transition_rules = list(TRANSITION_RULES)
        self._blocked_modes: set[MovementMode] = set()

        # Callbacks
        self._on_mode_change: Optional[Callable[[MovementMode, MovementMode], None]] = None
        self._on_transition_complete: Optional[Callable[[MovementMode], None]] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> MovementState:
        """Current movement state."""
        return self._state

    @property
    def current_mode(self) -> MovementMode:
        """Current movement mode."""
        return self._state.mode

    @property
    def is_transitioning(self) -> bool:
        """Whether a transition is in progress."""
        return self._state.is_transitioning

    @property
    def max_speed(self) -> float:
        """Maximum speed for current mode."""
        params = self._get_params(self._state.mode)
        return params.max_speed

    @property
    def can_jump(self) -> bool:
        """Whether jumping is allowed in current mode."""
        params = self._get_params(self._state.mode)
        return params.can_jump

    @property
    def height_modifier(self) -> float:
        """Capsule height modifier for current mode."""
        params = self._get_params(self._state.mode)
        return params.height_modifier

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_mode_change_callback(
        self, callback: Optional[Callable[[MovementMode, MovementMode], None]]
    ) -> None:
        """Set callback for mode changes."""
        self._on_mode_change = callback

    def set_transition_complete_callback(
        self, callback: Optional[Callable[[MovementMode], None]]
    ) -> None:
        """Set callback for transition completion."""
        self._on_transition_complete = callback

    # -------------------------------------------------------------------------
    # Mode Management
    # -------------------------------------------------------------------------

    def transition_to_mode(
        self,
        target_mode: MovementMode,
        force: bool = False,
    ) -> bool:
        """
        Attempt to transition to a new movement mode.

        Args:
            target_mode: Target movement mode
            force: Bypass transition rules if True

        Returns:
            True if transition was started or completed
        """
        if target_mode == self._state.mode:
            # Already in this mode - return True but don't trigger callbacks
            # since no actual transition occurred
            return True

        if target_mode in self._blocked_modes:
            return False

        if not force:
            rule = self._find_transition_rule(self._state.mode, target_mode)
            if rule is None:
                return False

            # Check rule conditions
            if rule.requires_grounded and self._state.context != MovementContext.GROUND:
                return False
            if rule.min_stamina > 0 and self._state.stamina < rule.min_stamina:
                return False
            if rule.condition and not rule.condition(self._state):
                return False

            # Start transition
            if rule.duration > 0:
                self._state.is_transitioning = True
                self._state.transition_progress = 0.0
                self._state.previous_mode = self._state.mode
            else:
                self._complete_transition(target_mode)
        else:
            self._complete_transition(target_mode)

        return True

    def _complete_transition(self, target_mode: MovementMode) -> None:
        """Complete transition to target mode."""
        old_mode = self._state.mode
        self._state.mode = target_mode
        self._state.is_transitioning = False
        self._state.transition_progress = 1.0
        self._state.time_in_mode = 0.0

        if self._on_mode_change:
            self._on_mode_change(old_mode, target_mode)
        if self._on_transition_complete:
            self._on_transition_complete(target_mode)

    def _find_transition_rule(
        self, from_mode: MovementMode, to_mode: MovementMode
    ) -> Optional[TransitionRule]:
        """Find transition rule between modes."""
        for rule in self._transition_rules:
            if rule.from_mode == from_mode and rule.to_mode == to_mode:
                return rule
        return None

    def block_mode(self, mode: MovementMode) -> None:
        """Block a movement mode from being used."""
        self._blocked_modes.add(mode)

    def unblock_mode(self, mode: MovementMode) -> None:
        """Unblock a movement mode."""
        self._blocked_modes.discard(mode)

    def set_custom_params(
        self, mode: MovementMode, params: MovementModeParams
    ) -> None:
        """Set custom parameters for a mode."""
        self._custom_params[mode] = params

    def _get_params(self, mode: MovementMode) -> MovementModeParams:
        """Get parameters for a mode, including custom overrides."""
        if mode in self._custom_params:
            return self._custom_params[mode]
        return DEFAULT_MODE_PARAMS.get(mode, MovementModeParams())

    # -------------------------------------------------------------------------
    # Movement Application
    # -------------------------------------------------------------------------

    def apply_movement(
        self,
        input_direction: Vector3,
        dt: float,
        is_grounded: bool,
    ) -> Vector3:
        """
        Calculate and apply movement based on current mode.

        Args:
            input_direction: Normalized input direction
            dt: Delta time in seconds
            is_grounded: Whether character is on ground

        Returns:
            Movement vector to apply
        """
        params = self._get_params(self._state.mode)

        # Update transition
        if self._state.is_transitioning:
            rule = self._find_transition_rule(
                self._state.previous_mode, self._state.mode
            )
            if rule and rule.duration > 0:
                self._state.transition_progress += dt / rule.duration
                if self._state.transition_progress >= 1.0:
                    self._complete_transition(self._state.mode)

        # Update time in mode
        self._state.time_in_mode += dt

        # Update stamina
        self._update_stamina(params.stamina_cost, dt)

        # Calculate target velocity
        target_speed = params.max_speed * input_direction.magnitude()
        direction = input_direction.normalized() if input_direction.magnitude() > 0.01 else Vector3.zero()

        self._state.target_velocity = direction * target_speed

        # Apply acceleration/deceleration
        if is_grounded or self._state.context != MovementContext.GROUND:
            accel = params.acceleration if target_speed > 0.01 else params.deceleration
        else:
            accel = params.acceleration * params.air_control

        # Smooth velocity change
        velocity_diff = self._state.target_velocity - self._state.current_velocity.horizontal()
        change = velocity_diff.normalized() * accel * dt

        if change.magnitude() > velocity_diff.magnitude():
            horizontal = self._state.target_velocity.horizontal()
        else:
            horizontal = self._state.current_velocity.horizontal() + change

        # Clamp to max speed
        if horizontal.magnitude() > params.max_speed:
            horizontal = horizontal.normalized() * params.max_speed

        self._state.current_velocity = Vector3(
            horizontal.x,
            self._state.current_velocity.y,
            horizontal.z,
        )
        self._state.current_speed = horizontal.magnitude()

        # Return movement for this frame
        return self._state.current_velocity * dt

    def set_vertical_velocity(self, velocity_y: float) -> None:
        """Set vertical velocity component."""
        self._state.current_velocity = Vector3(
            self._state.current_velocity.x,
            velocity_y,
            self._state.current_velocity.z,
        )

    def add_vertical_velocity(self, delta_y: float) -> None:
        """Add to vertical velocity."""
        self._state.current_velocity.y += delta_y

    def set_context(self, context: MovementContext) -> None:
        """Set the movement context."""
        self._state.context = context

    # -------------------------------------------------------------------------
    # Stamina
    # -------------------------------------------------------------------------

    # Stamina regeneration rate (per second) when not consuming stamina
    STAMINA_REGEN_RATE: float = 10.0

    def _update_stamina(self, cost_per_second: float, dt: float) -> None:
        """Update stamina based on mode cost."""
        if cost_per_second > 0:
            self._state.stamina -= cost_per_second * dt
            self._state.stamina = max(0.0, self._state.stamina)
        else:
            # Regenerate stamina using class constant
            self._state.stamina += self.STAMINA_REGEN_RATE * dt
            self._state.stamina = min(self._state.max_stamina, self._state.stamina)

    def has_stamina(self, amount: float) -> bool:
        """Check if enough stamina is available."""
        return self._state.stamina >= amount

    def consume_stamina(self, amount: float) -> bool:
        """
        Consume stamina if available.

        Returns:
            True if stamina was consumed
        """
        if self._state.stamina >= amount:
            self._state.stamina -= amount
            return True
        return False

    def restore_stamina(self, amount: float) -> None:
        """Restore stamina."""
        self._state.stamina = min(
            self._state.max_stamina, self._state.stamina + amount
        )

    # -------------------------------------------------------------------------
    # State Queries
    # -------------------------------------------------------------------------

    def is_moving(self) -> bool:
        """Check if character is actively moving."""
        return self._state.current_speed > 0.1

    def is_sprinting(self) -> bool:
        """Check if in sprinting mode."""
        return self._state.mode == MovementMode.SPRINTING

    def is_crouching(self) -> bool:
        """Check if in crouching or prone mode."""
        return self._state.mode in (MovementMode.CROUCHING, MovementMode.PRONE)

    def is_airborne(self) -> bool:
        """Check if in an airborne state."""
        return self._state.mode == MovementMode.FALLING or self._state.mode == MovementMode.FLYING

    def is_swimming(self) -> bool:
        """Check if in swimming mode."""
        return self._state.mode == MovementMode.SWIMMING

    def get_gravity_scale(self) -> float:
        """Get gravity scale for current mode."""
        params = self._get_params(self._state.mode)
        return params.gravity_scale

    def get_friction_modifier(self) -> float:
        """Get friction modifier for current mode."""
        params = self._get_params(self._state.mode)
        return params.friction_modifier

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_state_dict(self) -> dict[str, Any]:
        """Get state as dictionary for serialization."""
        return {
            "mode": self._state.mode.name,
            "context": self._state.context.name,
            "current_speed": self._state.current_speed,
            "stamina": self._state.stamina,
            "time_in_mode": self._state.time_in_mode,
            "velocity": (
                self._state.current_velocity.x,
                self._state.current_velocity.y,
                self._state.current_velocity.z,
            ),
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        """Load state from dictionary."""
        self._state.mode = MovementMode[data.get("mode", "WALKING")]
        self._state.context = MovementContext[data.get("context", "GROUND")]
        self._state.current_speed = data.get("current_speed", 0.0)
        self._state.stamina = data.get("stamina", 100.0)
        self._state.time_in_mode = data.get("time_in_mode", 0.0)
        vel = data.get("velocity", (0.0, 0.0, 0.0))
        self._state.current_velocity = Vector3(vel[0], vel[1], vel[2])
