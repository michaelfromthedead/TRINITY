"""
Climbing locomotion for XR environments.

Provides climbing-based movement allowing players to grab and climb surfaces,
ladders, and climbable objects in VR.

Components:
    ClimbingLocomotion  - Main climbing locomotion component
    ClimbableVolume     - Marker for climbable surfaces
    GrabPoint           - Individual grab point on climbable surface

Based on XR_CONTEXT.md specifications:
    - Physical: Room-scale movement, climbing mechanics
    - Integration with grab/interaction system
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


class ClimbingState(Enum):
    """Climbing locomotion state machine states."""

    IDLE = "idle"  # Not climbing, normal locomotion
    GRABBING = "grabbing"  # One hand grabbing
    CLIMBING = "climbing"  # Actively climbing with movement
    MANTLING = "mantling"  # Pulling up over ledge
    FALLING = "falling"  # Released grip, in air


class GrabHandState(Enum):
    """State of a single grabbing hand."""

    FREE = "free"  # Not grabbing
    GRABBING = "grabbing"  # Currently holding
    RELEASING = "releasing"  # In process of releasing


class ClimbableType(Enum):
    """Type of climbable surface."""

    SURFACE = "surface"  # General climbable surface (wall, rock)
    LADDER = "ladder"  # Ladder with defined rungs
    ROPE = "rope"  # Rope or chain
    LEDGE = "ledge"  # Ledge for mantling
    HOLDS = "holds"  # Discrete climbing holds


class MantleType(Enum):
    """Type of mantle/vault action."""

    PULL_UP = "pull_up"  # Pull body up and over
    VAULT = "vault"  # Quick vault over obstacle
    CLIMB_OVER = "climb_over"  # Slow climb over


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class GrabState:
    """State of a single hand grab."""

    hand: str  # "left" or "right"
    state: GrabHandState = GrabHandState.FREE
    grab_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    surface_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    grab_strength: float = 1.0  # 0-1, for grip strength mechanics
    climbable_id: Optional[int] = None  # Entity ID of grabbed climbable


@dataclass
class ClimbingInput:
    """Input state for climbing locomotion."""

    left_grip: float = 0.0  # 0-1 grip button
    right_grip: float = 0.0
    left_hand_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_hand_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_hand_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_hand_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ClimbingMovement:
    """Result of climbing movement calculation."""

    player_velocity: tuple[float, float, float]
    is_climbing: bool
    is_mantling: bool
    stamina_drain: float = 0.0
    mantle_progress: float = 0.0  # 0-1 for mantle animation


@dataclass
class GrabPoint:
    """
    Individual grab point on a climbable surface.

    Used for surfaces with discrete holds rather than continuous gripping.
    """

    position: tuple[float, float, float]
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    grip_type: str = "hold"  # hold, ledge, rung
    grip_radius: float = 0.1
    is_available: bool = True


# =============================================================================
# CLIMBABLE VOLUME COMPONENT
# =============================================================================


@dataclass
class ClimbableVolume:
    """
    Marker component for climbable surfaces and volumes.

    Attach to geometry that players can climb on.

    Attributes:
        climbable_type: Type of climbable (surface, ladder, rope, etc)
        surface_normal: Primary surface normal for flat surfaces
        grip_strength_required: Minimum grip strength to hold (0-1)
        stamina_drain_rate: Stamina drain per second while climbing
        allow_two_handed: Whether two-handed climbing is required
        grab_points: Discrete grab points (for holds-type climbables)
    """

    # Type and identity
    climbable_type: Annotated[ClimbableType, Immutable] = ClimbableType.SURFACE

    # Surface properties
    surface_normal: Annotated[tuple[float, float, float], Tracked] = (0.0, 0.0, 1.0)
    surface_friction: Annotated[float, Tracked, Range(0.0, 1.0)] = 0.8

    # Grip requirements
    grip_strength_required: Annotated[float, Tracked, Range(0.0, 1.0)] = 0.3
    grip_threshold: Annotated[float, Tracked, Range(0.0, 1.0)] = 0.5

    # Stamina
    stamina_drain_rate: Annotated[float, Tracked, Range(0.0, 50.0)] = 5.0
    allow_stamina_recovery: Annotated[bool, Tracked] = False

    # Two-handed requirements
    allow_one_handed: Annotated[bool, Tracked] = True
    require_alternating: Annotated[bool, Tracked] = False  # Must alternate hands

    # Grab points (for discrete holds)
    grab_points: Annotated[list[GrabPoint], Tracked] = field(default_factory=list)
    auto_generate_points: Annotated[bool, Tracked] = True
    point_spacing: Annotated[float, Tracked] = 0.3  # For auto-generation

    # Mantle support
    allow_mantle: Annotated[bool, Tracked] = True
    mantle_height_threshold: Annotated[float, Tracked] = 0.3  # Height from top to trigger

    # Bounds
    bounds_min: Annotated[tuple[float, float, float], Tracked] = (0.0, 0.0, 0.0)
    bounds_max: Annotated[tuple[float, float, float], Tracked] = (1.0, 2.0, 0.1)

    # State
    is_active: Annotated[bool, Tracked] = True
    current_climbers: Annotated[int, Tracked] = 0
    max_climbers: Annotated[int, Tracked] = 4

    def is_point_inside(self, point: tuple[float, float, float]) -> bool:
        """Check if a point is within the climbable bounds."""
        return (
            self.bounds_min[0] <= point[0] <= self.bounds_max[0]
            and self.bounds_min[1] <= point[1] <= self.bounds_max[1]
            and self.bounds_min[2] <= point[2] <= self.bounds_max[2]
        )

    def find_nearest_grab_point(
        self, position: tuple[float, float, float], max_distance: float = 0.5
    ) -> Optional[GrabPoint]:
        """Find the nearest available grab point to a position."""
        nearest: Optional[GrabPoint] = None
        nearest_dist = max_distance

        for point in self.grab_points:
            if not point.is_available:
                continue

            dx = point.position[0] - position[0]
            dy = point.position[1] - position[1]
            dz = point.position[2] - position[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < nearest_dist:
                nearest = point
                nearest_dist = dist

        return nearest


# =============================================================================
# CLIMBING LOCOMOTION COMPONENT
# =============================================================================


@dataclass
class ClimbingLocomotion:
    """
    Climbing-based locomotion for XR.

    Allows players to grab and climb surfaces by moving their hands
    while gripping climbable objects.

    Attributes:
        state: Current climbing state
        left_grab: Left hand grab state
        right_grab: Right hand grab state
        climb_speed_multiplier: Multiplier for climbing speed
        stamina: Current stamina (0-100)
        stamina_max: Maximum stamina
        gravity_while_climbing: Gravity applied while climbing
    """

    # State
    state: Annotated[ClimbingState, Tracked, Observable] = ClimbingState.IDLE
    is_climbing: Annotated[bool, Tracked, Observable] = False

    # Hand states
    left_grab: Annotated[GrabState, Tracked] = field(
        default_factory=lambda: GrabState(hand="left")
    )
    right_grab: Annotated[GrabState, Tracked] = field(
        default_factory=lambda: GrabState(hand="right")
    )

    # Movement settings
    climb_speed_multiplier: Annotated[float, Tracked, Range(0.5, 2.0)] = 1.0
    movement_smoothing: Annotated[float, Tracked, Range(0.0, 1.0)] = 0.3

    # Grip settings
    grip_threshold: Annotated[float, Tracked, Range(0.0, 1.0)] = 0.5
    release_threshold: Annotated[float, Tracked, Range(0.0, 1.0)] = 0.3
    auto_grip_distance: Annotated[float, Tracked, Range(0.0, 0.3)] = 0.15

    # Stamina system
    stamina_enabled: Annotated[bool, Tracked] = True
    stamina: Annotated[float, Tracked, Range(0.0, 100.0)] = 100.0
    stamina_max: Annotated[float, Tracked] = 100.0
    stamina_recovery_rate: Annotated[float, Tracked] = 10.0  # Per second when not climbing
    stamina_recovery_delay: Annotated[float, Tracked] = 1.0  # Seconds before recovery

    # Physics
    gravity_while_climbing: Annotated[float, Tracked] = -2.0  # Reduced while grabbing
    fall_gravity: Annotated[float, Tracked] = -9.8
    slip_velocity: Annotated[float, Tracked] = 0.5  # Downward velocity when stamina empty

    # Mantle settings
    mantle_enabled: Annotated[bool, Tracked] = True
    mantle_speed: Annotated[float, Tracked, Range(0.5, 3.0)] = 1.5
    mantle_height: Annotated[float, Tracked] = 0.3  # Minimum height to trigger

    # Haptic feedback
    haptic_on_grab: Annotated[bool, Tracked] = True
    haptic_on_slip: Annotated[bool, Tracked] = True
    haptic_intensity: Annotated[float, Tracked, Range(0.0, 1.0)] = 0.5

    # Internal state
    _stamina_recovery_timer: Annotated[float, Transient] = 0.0
    _mantle_progress: Annotated[float, Transient] = 0.0
    _mantle_target: Annotated[Optional[tuple[float, float, float]], Transient] = None
    _smoothed_velocity: Annotated[tuple[float, float, float], Transient] = (0.0, 0.0, 0.0)

    # Callbacks
    _on_grab: Annotated[Optional[Callable[[str, GrabState], None]], Transient] = None
    _on_release: Annotated[Optional[Callable[[str], None]], Transient] = None
    _on_stamina_empty: Annotated[Optional[Callable[[], None]], Transient] = None
    _on_mantle_start: Annotated[Optional[Callable[[], None]], Transient] = None
    _on_mantle_complete: Annotated[Optional[Callable[[], None]], Transient] = None

    def _get_grab_state(self, hand: str) -> GrabState:
        """Get grab state for specified hand."""
        return self.left_grab if hand == "left" else self.right_grab

    def _set_grab_state(self, hand: str, state: GrabState) -> None:
        """Set grab state for specified hand."""
        if hand == "left":
            self.left_grab = state
        else:
            self.right_grab = state

    def try_grab(
        self,
        hand: str,
        hand_position: tuple[float, float, float],
        grip_strength: float,
        climbable: Optional[ClimbableVolume] = None,
    ) -> bool:
        """
        Attempt to grab with specified hand.

        Args:
            hand: "left" or "right"
            hand_position: Current hand position
            grip_strength: Current grip input (0-1)
            climbable: Climbable volume to grab (if detected)

        Returns:
            True if grab was successful
        """
        if grip_strength < self.grip_threshold:
            return False

        grab_state = self._get_grab_state(hand)

        if grab_state.state == GrabHandState.GRABBING:
            return True  # Already grabbing

        # Check if near climbable
        if climbable is None:
            return False

        if not climbable.is_active:
            return False

        # Find grab point or use surface grab
        grab_point = climbable.find_nearest_grab_point(
            hand_position, self.auto_grip_distance
        )

        if grab_point is not None:
            grab_position = grab_point.position
            surface_normal = grab_point.normal
        elif climbable.is_point_inside(hand_position):
            grab_position = hand_position
            surface_normal = climbable.surface_normal
        else:
            return False

        # Check grip strength requirement
        if grip_strength < climbable.grip_strength_required:
            return False

        # Execute grab
        new_state = GrabState(
            hand=hand,
            state=GrabHandState.GRABBING,
            grab_position=grab_position,
            surface_normal=surface_normal,
            grab_strength=grip_strength,
        )
        self._set_grab_state(hand, new_state)

        # Update climbing state
        self._update_climbing_state()

        if self._on_grab:
            self._on_grab(hand, new_state)

        return True

    def release_grab(self, hand: str) -> None:
        """
        Release grab with specified hand.

        Args:
            hand: "left" or "right"
        """
        grab_state = self._get_grab_state(hand)

        if grab_state.state != GrabHandState.GRABBING:
            return

        new_state = GrabState(hand=hand, state=GrabHandState.FREE)
        self._set_grab_state(hand, new_state)

        # Update climbing state
        self._update_climbing_state()

        if self._on_release:
            self._on_release(hand)

    def _update_climbing_state(self) -> None:
        """Update overall climbing state based on hand states."""
        left_grabbing = self.left_grab.state == GrabHandState.GRABBING
        right_grabbing = self.right_grab.state == GrabHandState.GRABBING

        if left_grabbing or right_grabbing:
            self.is_climbing = True
            if left_grabbing and right_grabbing:
                self.state = ClimbingState.CLIMBING
            else:
                self.state = ClimbingState.GRABBING
        else:
            self.is_climbing = False
            if self.state in (ClimbingState.GRABBING, ClimbingState.CLIMBING):
                self.state = ClimbingState.FALLING
            else:
                self.state = ClimbingState.IDLE

    def calculate_climbing_movement(
        self,
        input_state: ClimbingInput,
        delta_time: float,
    ) -> ClimbingMovement:
        """
        Calculate climbing movement for the current frame.

        Args:
            input_state: Current climbing input state
            delta_time: Time since last frame

        Returns:
            ClimbingMovement with player velocity and state
        """
        velocity = (0.0, 0.0, 0.0)
        stamina_drain = 0.0
        is_mantling = self.state == ClimbingState.MANTLING

        # Handle mantle in progress
        if is_mantling and self._mantle_target is not None:
            self._mantle_progress += delta_time * self.mantle_speed
            if self._mantle_progress >= 1.0:
                self._mantle_progress = 0.0
                self.state = ClimbingState.IDLE
                self.is_climbing = False
                if self._on_mantle_complete:
                    self._on_mantle_complete()

            return ClimbingMovement(
                player_velocity=(0.0, 0.0, 0.0),
                is_climbing=True,
                is_mantling=True,
                stamina_drain=0.0,
                mantle_progress=self._mantle_progress,
            )

        # Check grip state changes
        if input_state.left_grip >= self.grip_threshold:
            if self.left_grab.state != GrabHandState.GRABBING:
                # Try to grab - climbable detection done externally
                pass
        elif input_state.left_grip < self.release_threshold:
            if self.left_grab.state == GrabHandState.GRABBING:
                self.release_grab("left")

        if input_state.right_grip >= self.grip_threshold:
            if self.right_grab.state != GrabHandState.GRABBING:
                pass
        elif input_state.right_grip < self.release_threshold:
            if self.right_grab.state == GrabHandState.GRABBING:
                self.release_grab("right")

        # Calculate movement from hand motion while grabbing
        if self.is_climbing:
            vx, vy, vz = 0.0, 0.0, 0.0

            # Movement is inverse of hand movement when grabbing
            if self.left_grab.state == GrabHandState.GRABBING:
                # Player moves opposite to hand movement
                vx -= input_state.left_hand_velocity[0] * self.climb_speed_multiplier
                vy -= input_state.left_hand_velocity[1] * self.climb_speed_multiplier
                vz -= input_state.left_hand_velocity[2] * self.climb_speed_multiplier

            if self.right_grab.state == GrabHandState.GRABBING:
                vx -= input_state.right_hand_velocity[0] * self.climb_speed_multiplier
                vy -= input_state.right_hand_velocity[1] * self.climb_speed_multiplier
                vz -= input_state.right_hand_velocity[2] * self.climb_speed_multiplier

            # Average if both hands grabbing
            if (
                self.left_grab.state == GrabHandState.GRABBING
                and self.right_grab.state == GrabHandState.GRABBING
            ):
                vx /= 2.0
                vy /= 2.0
                vz /= 2.0

            # Apply reduced gravity while climbing
            vy += self.gravity_while_climbing * delta_time

            # Apply smoothing
            self._smoothed_velocity = (
                self._smoothed_velocity[0] * self.movement_smoothing
                + vx * (1.0 - self.movement_smoothing),
                self._smoothed_velocity[1] * self.movement_smoothing
                + vy * (1.0 - self.movement_smoothing),
                self._smoothed_velocity[2] * self.movement_smoothing
                + vz * (1.0 - self.movement_smoothing),
            )

            velocity = self._smoothed_velocity

            # Drain stamina
            if self.stamina_enabled:
                stamina_drain = 5.0 * delta_time  # Base drain rate
                # Add drain based on movement
                move_speed = math.sqrt(vx * vx + vy * vy + vz * vz)
                stamina_drain += move_speed * 2.0 * delta_time

        elif self.state == ClimbingState.FALLING:
            # Apply full gravity
            vy = self.fall_gravity * delta_time
            velocity = (0.0, vy, 0.0)

        return ClimbingMovement(
            player_velocity=velocity,
            is_climbing=self.is_climbing,
            is_mantling=is_mantling,
            stamina_drain=stamina_drain,
            mantle_progress=self._mantle_progress,
        )

    def try_mantle(self, ledge_position: tuple[float, float, float]) -> bool:
        """
        Attempt to mantle up to a ledge.

        Args:
            ledge_position: Position of the ledge top

        Returns:
            True if mantle started
        """
        if not self.mantle_enabled:
            return False

        if self.state not in (ClimbingState.GRABBING, ClimbingState.CLIMBING):
            return False

        self.state = ClimbingState.MANTLING
        self._mantle_target = ledge_position
        self._mantle_progress = 0.0

        if self._on_mantle_start:
            self._on_mantle_start()

        return True

    def update(self, delta_time: float) -> None:
        """
        Update climbing state.

        Args:
            delta_time: Time since last frame
        """
        # Update stamina
        if self.stamina_enabled:
            if self.is_climbing:
                self._stamina_recovery_timer = self.stamina_recovery_delay
            else:
                self._stamina_recovery_timer -= delta_time
                if self._stamina_recovery_timer <= 0:
                    self.stamina = min(
                        self.stamina + self.stamina_recovery_rate * delta_time,
                        self.stamina_max,
                    )

            # Check for stamina depletion
            if self.stamina <= 0 and self.is_climbing:
                # Force release
                self.release_grab("left")
                self.release_grab("right")
                if self._on_stamina_empty:
                    self._on_stamina_empty()

        # Transition from falling to idle when grounded
        if self.state == ClimbingState.FALLING:
            # Grounded check done externally
            pass

    def drain_stamina(self, amount: float) -> None:
        """
        Drain stamina by specified amount.

        Args:
            amount: Amount to drain
        """
        if self.stamina_enabled:
            self.stamina = max(0.0, self.stamina - amount)

    def set_grounded(self) -> None:
        """Mark player as grounded (called when touching ground)."""
        if self.state == ClimbingState.FALLING:
            self.state = ClimbingState.IDLE

    def set_callbacks(
        self,
        on_grab: Optional[Callable[[str, GrabState], None]] = None,
        on_release: Optional[Callable[[str], None]] = None,
        on_stamina_empty: Optional[Callable[[], None]] = None,
        on_mantle_start: Optional[Callable[[], None]] = None,
        on_mantle_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        """Set climbing event callbacks."""
        self._on_grab = on_grab
        self._on_release = on_release
        self._on_stamina_empty = on_stamina_empty
        self._on_mantle_start = on_mantle_start
        self._on_mantle_complete = on_mantle_complete


# =============================================================================
# DECORATOR: @xr_climbable
# =============================================================================


def _validate_xr_climbable(
    climbable_type: str = "surface", **_: Any
) -> None:
    """Validate xr_climbable decorator parameters."""
    valid_types = {"surface", "ladder", "rope", "ledge", "holds"}
    if climbable_type not in valid_types:
        raise ValueError(
            f"@xr_climbable: 'climbable_type' must be one of {valid_types}, "
            f"got '{climbable_type}'"
        )


def _xr_climbable_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for xr_climbable decorator."""
    climbable_type = params.get("climbable_type", "surface")
    return [
        Step(Op.TAG, {"key": "xr_climbable", "value": True}),
        Step(Op.TAG, {"key": "climbable_type", "value": climbable_type}),
        Step(Op.REGISTER, {"registry": "xr"}),
    ]


def _after_xr_climbable(target: Any, params: dict[str, Any]) -> Any:
    """Post-processing for xr_climbable decorator."""
    target._xr_climbable = True
    target._climbable_type = params.get("climbable_type", "surface")
    return None


xr_climbable = make_decorator(
    name="xr_climbable",
    steps=_xr_climbable_steps,
    doc="Mark an object or surface as climbable in XR.",
    validate=_validate_xr_climbable,
    after_steps=_after_xr_climbable,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("xr_climbable", xr_climbable, ("class", "function")),
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


class ClimbingLocomotionProvider:
    """
    Abstract provider interface for climbing locomotion.

    Subclass this to integrate with specific XR runtimes.
    """

    def __init__(self, locomotion: ClimbingLocomotion):
        """
        Initialize provider with locomotion component.

        Args:
            locomotion: ClimbingLocomotion component to control
        """
        self.locomotion = locomotion

    def update(
        self,
        input_state: ClimbingInput,
        delta_time: float,
    ) -> ClimbingMovement:
        """
        Update climbing locomotion.

        Args:
            input_state: Current climbing input
            delta_time: Time since last frame

        Returns:
            ClimbingMovement result
        """
        self.locomotion.update(delta_time)
        return self.locomotion.calculate_climbing_movement(input_state, delta_time)

    def on_climbable_detected(
        self,
        hand: str,
        hand_position: tuple[float, float, float],
        grip_strength: float,
        climbable: ClimbableVolume,
    ) -> bool:
        """
        Handle climbable surface detected near hand.

        Args:
            hand: "left" or "right"
            hand_position: Current hand position
            grip_strength: Current grip input
            climbable: Detected climbable volume

        Returns:
            True if grab was initiated
        """
        return self.locomotion.try_grab(hand, hand_position, grip_strength, climbable)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ClimbingState",
    "GrabHandState",
    "ClimbableType",
    "MantleType",
    # Data classes
    "GrabState",
    "ClimbingInput",
    "ClimbingMovement",
    "GrabPoint",
    # Components
    "ClimbableVolume",
    "ClimbingLocomotion",
    # Provider
    "ClimbingLocomotionProvider",
    # Decorator
    "xr_climbable",
    # Type markers
    "Tracked",
    "Range",
    "Observable",
    "Transient",
    "Immutable",
]
