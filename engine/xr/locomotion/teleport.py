"""
Teleport locomotion for XR environments.

Provides teleportation-based movement with arc visualization, fade transitions,
and multiple teleport styles (instant, fade, dash).

Components:
    TeleportLocomotion  - Main teleport locomotion component
    TeleportTarget      - Valid teleportation target marker
    TeleportArc         - Arc trajectory visualization

Decorators:
    @xr_teleport_area   - Mark an object as a valid teleport destination

Based on XR_CONTEXT.md specifications:
    - Teleportation: Instant, Fade, Dash, Arc visualizer
    - Comfort options: fade_enabled, fade_duration (0-0.5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Callable, Optional

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry
from engine.xr.utils.markers import Tracked, Range, Observable, Transient, Immutable
from engine.xr.config import XR_CONFIG

if TYPE_CHECKING:
    from typing import Protocol

    class TrackedProtocol(Protocol):
        """Protocol for tracked descriptor marker."""

        pass


# =============================================================================
# ENUMS
# =============================================================================


class TeleportStyle(Enum):
    """Teleportation transition styles."""

    INSTANT = "instant"  # Immediate teleport, no transition
    FADE = "fade"  # Fade to black and back
    DASH = "dash"  # Quick linear movement to destination
    BLINK = "blink"  # Quick fade with eye blink effect


class TeleportState(Enum):
    """Teleport locomotion state machine states."""

    IDLE = "idle"  # Not teleporting
    AIMING = "aiming"  # Showing arc, selecting target
    VALIDATING = "validating"  # Checking if target is valid
    TRANSITIONING = "transitioning"  # Performing teleport transition
    COOLDOWN = "cooldown"  # Brief cooldown after teleport


class ArcSegmentType(Enum):
    """Type of arc segment for visualization."""

    VALID = "valid"  # Over valid teleport surface
    INVALID = "invalid"  # Over invalid surface
    OUT_OF_RANGE = "out_of_range"  # Beyond max distance


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ArcPoint:
    """Single point along the teleport arc trajectory."""

    position: tuple[float, float, float]
    segment_type: ArcSegmentType = ArcSegmentType.VALID
    time: float = 0.0  # Time along arc from start


@dataclass
class TeleportResult:
    """Result of a teleport operation."""

    success: bool
    start_position: tuple[float, float, float]
    end_position: tuple[float, float, float]
    rotation: float = 0.0
    duration: float = 0.0
    style: TeleportStyle = TeleportStyle.INSTANT


# =============================================================================
# TELEPORT ARC CALCULATOR
# =============================================================================


class TeleportArcCalculator:
    """
    Calculates parabolic arc trajectory for teleport visualization.

    Uses projectile motion physics with configurable gravity and initial velocity.
    """

    def __init__(
        self,
        gravity: float = -9.8,
        initial_velocity: float = 8.0,
        max_distance: float = 10.0,
        arc_resolution: int = 32,
    ):
        """
        Initialize arc calculator.

        Args:
            gravity: Gravity acceleration (negative for downward)
            initial_velocity: Initial launch velocity
            max_distance: Maximum teleport distance
            arc_resolution: Number of points in the arc
        """
        self.gravity = gravity
        self.initial_velocity = initial_velocity
        self.max_distance = max_distance
        self.arc_resolution = arc_resolution

    def calculate_arc(
        self,
        start_position: tuple[float, float, float],
        direction: tuple[float, float, float],
        launch_angle: float = 45.0,
    ) -> list[ArcPoint]:
        """
        Calculate arc trajectory points.

        Args:
            start_position: Starting position (x, y, z)
            direction: Normalized forward direction (x, y, z)
            launch_angle: Launch angle in degrees above horizontal

        Returns:
            List of ArcPoint objects forming the trajectory
        """
        points: list[ArcPoint] = []

        # Convert angle to radians
        angle_rad = math.radians(launch_angle)

        # Calculate initial velocity components
        v_horizontal = self.initial_velocity * math.cos(angle_rad)
        v_vertical = self.initial_velocity * math.sin(angle_rad)

        # Time step for arc resolution
        # Calculate total flight time to reach ground (y=0)
        # Using quadratic formula: 0 = start_y + v_vertical*t + 0.5*gravity*t^2
        # Solving: t = (-v_vertical ± sqrt(v_vertical^2 - 2*gravity*start_y)) / gravity
        if self.gravity != 0:
            start_y = start_position[1]
            discriminant = v_vertical**2 - 2 * self.gravity * start_y
            if discriminant >= 0:
                # Take the positive root (time when arc hits ground)
                total_time = (-v_vertical - math.sqrt(discriminant)) / self.gravity
            else:
                # Fallback to symmetric arc time if no ground intersection
                total_time = (2.0 * v_vertical) / abs(self.gravity)
        else:
            total_time = self.max_distance / max(v_horizontal, 0.001)

        time_step = total_time / self.arc_resolution

        # Normalize direction
        dir_len = math.sqrt(direction[0] ** 2 + direction[2] ** 2)
        if dir_len > 0:
            dir_x = direction[0] / dir_len
            dir_z = direction[2] / dir_len
        else:
            dir_x, dir_z = 0.0, 1.0

        accumulated_distance = 0.0

        for i in range(self.arc_resolution + 1):
            t = i * time_step

            # Calculate position at time t
            horizontal_dist = v_horizontal * t
            vertical_dist = v_vertical * t + 0.5 * self.gravity * t * t

            x = start_position[0] + dir_x * horizontal_dist
            y = start_position[1] + vertical_dist
            z = start_position[2] + dir_z * horizontal_dist

            # Check if below ground (y < 0 approximation)
            if y < 0:
                y = 0

            # Calculate distance from start
            dx = x - start_position[0]
            dz = z - start_position[2]
            accumulated_distance = math.sqrt(dx * dx + dz * dz)

            # Determine segment type based on distance
            if accumulated_distance > self.max_distance:
                segment_type = ArcSegmentType.OUT_OF_RANGE
            else:
                segment_type = ArcSegmentType.VALID

            points.append(ArcPoint(position=(x, y, z), segment_type=segment_type, time=t))

            # Stop if we've hit the ground and gone past peak
            if y <= 0 and t > total_time * 0.5:
                break

        return points

    def find_landing_point(
        self,
        arc_points: list[ArcPoint],
        ground_height: float = 0.0,
    ) -> Optional[tuple[float, float, float]]:
        """
        Find where the arc intersects with ground level.

        Args:
            arc_points: Calculated arc points
            ground_height: Height of the ground plane

        Returns:
            Landing position or None if no valid landing
        """
        if len(arc_points) < 2:
            return None

        GROUND_EPSILON = 1e-6  # Tolerance for ground level comparison
        for i in range(1, len(arc_points)):
            prev = arc_points[i - 1]
            curr = arc_points[i]

            # Check if arc crosses ground level (with floating point tolerance)
            if prev.position[1] >= ground_height - GROUND_EPSILON and curr.position[1] <= ground_height + GROUND_EPSILON:
                # Linear interpolation to find exact crossing point
                EPSILON = 1e-6
                denominator = curr.position[1] - prev.position[1]
                if abs(denominator) < EPSILON:
                    continue
                t = (ground_height - prev.position[1]) / denominator
                x = prev.position[0] + t * (curr.position[0] - prev.position[0])
                z = prev.position[2] + t * (curr.position[2] - prev.position[2])
                return (x, ground_height, z)

        return None


# =============================================================================
# TELEPORT LOCOMOTION COMPONENT
# =============================================================================


@dataclass
class TeleportLocomotion:
    """
    Teleport-based locomotion with comfort features.

    Provides teleportation movement with arc visualization, multiple transition
    styles, and configurable comfort settings.

    Attributes:
        is_aiming: Currently showing teleport arc
        aim_valid: Target position is valid for teleport
        target_position: Destination position
        target_rotation: Rotation at destination (radians)
        arc_points: Calculated arc trajectory points
        style: Teleportation transition style
        max_distance: Maximum teleport distance in meters
        arc_gravity: Gravity for arc calculation
        fade_enabled: Whether to use fade transition
        fade_duration: Duration of fade transition (0-0.5s)
        cooldown_duration: Cooldown between teleports
    """

    # State
    is_aiming: Annotated[bool, Tracked] = False
    aim_valid: Annotated[bool, Tracked] = False
    state: Annotated[TeleportState, Tracked, Observable] = TeleportState.IDLE

    # Target
    target_position: Annotated[tuple[float, float, float], Tracked] = (0.0, 0.0, 0.0)
    target_rotation: Annotated[float, Tracked] = 0.0
    target_normal: Annotated[tuple[float, float, float], Tracked] = (0.0, 1.0, 0.0)

    # Arc visualization
    arc_points: Annotated[list[ArcPoint], Tracked] = field(default_factory=list)
    arc_visible: Annotated[bool, Tracked] = False

    # Settings
    style: Annotated[TeleportStyle, Tracked] = TeleportStyle.FADE
    max_distance: Annotated[float, Tracked, Range(1.0, 50.0)] = XR_CONFIG.locomotion.TELEPORT_MAX_DISTANCE_M
    arc_gravity: Annotated[float, Tracked] = XR_CONFIG.locomotion.TELEPORT_ARC_GRAVITY
    arc_initial_velocity: Annotated[float, Tracked] = XR_CONFIG.locomotion.TELEPORT_ARC_VELOCITY
    arc_launch_angle: Annotated[float, Tracked, Range(15.0, 75.0)] = 45.0
    arc_resolution: Annotated[int, Tracked, Range(8, 64)] = 32

    # Comfort settings
    fade_enabled: Annotated[bool, Tracked] = True
    fade_duration: Annotated[float, Tracked, Range(0.0, 0.5)] = XR_CONFIG.locomotion.TELEPORT_FADE_DURATION_SECONDS
    fade_color: Annotated[tuple[float, float, float], Tracked] = (0.0, 0.0, 0.0)

    # Dash settings (for dash style)
    dash_speed: Annotated[float, Tracked, Range(5.0, 50.0)] = 20.0

    # Cooldown
    cooldown_duration: Annotated[float, Tracked, Range(0.0, 2.0)] = XR_CONFIG.locomotion.TELEPORT_COOLDOWN_SECONDS
    _cooldown_remaining: Annotated[float, Transient] = 0.0

    # Rotation settings
    rotation_enabled: Annotated[bool, Tracked] = True
    rotation_snap_angle: Annotated[float, Tracked, Range(0.0, 90.0)] = XR_CONFIG.locomotion.SNAP_TURN_ANGLE_DEGREES

    # Valid surface tags
    valid_surface_tags: Annotated[list[str], Tracked] = field(
        default_factory=lambda: ["teleport_surface", "floor", "ground"]
    )

    # Callbacks (transient, not serialized)
    _on_teleport_start: Annotated[Optional[Callable[[], None]], Transient] = None
    _on_teleport_end: Annotated[Optional[Callable[[TeleportResult], None]], Transient] = None

    # Internal state
    _arc_calculator: Annotated[Optional[TeleportArcCalculator], Transient] = None
    _transition_progress: Annotated[float, Transient] = 0.0
    _start_position: Annotated[tuple[float, float, float], Transient] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """Initialize arc calculator after dataclass init."""
        self._arc_calculator = TeleportArcCalculator(
            gravity=self.arc_gravity,
            initial_velocity=self.arc_initial_velocity,
            max_distance=self.max_distance,
            arc_resolution=self.arc_resolution,
        )

    def begin_aim(self, start_position: tuple[float, float, float]) -> bool:
        """
        Begin teleport aiming from the given position.

        Args:
            start_position: Current position to teleport from

        Returns:
            True if aiming started, False if on cooldown
        """
        if self.state != TeleportState.IDLE:
            return False

        if self._cooldown_remaining > 0:
            return False

        self._start_position = start_position
        self.is_aiming = True
        self.arc_visible = True
        self.state = TeleportState.AIMING
        return True

    def update_aim(
        self,
        direction: tuple[float, float, float],
        rotation_offset: float = 0.0,
    ) -> None:
        """
        Update teleport arc based on aim direction.

        Args:
            direction: Normalized aim direction
            rotation_offset: Rotation offset for destination orientation
        """
        if self.state != TeleportState.AIMING:
            return

        if self._arc_calculator is None:
            self._arc_calculator = TeleportArcCalculator(
                gravity=self.arc_gravity,
                initial_velocity=self.arc_initial_velocity,
                max_distance=self.max_distance,
                arc_resolution=self.arc_resolution,
            )

        # Calculate arc trajectory
        self.arc_points = self._arc_calculator.calculate_arc(
            self._start_position,
            direction,
            self.arc_launch_angle,
        )

        # Find landing point
        landing = self._arc_calculator.find_landing_point(self.arc_points)
        if landing is not None:
            self.target_position = landing
            self.aim_valid = True

            # Calculate rotation
            if self.rotation_enabled:
                # Snap rotation to configured angle
                if self.rotation_snap_angle > 0:
                    snapped = (
                        round(rotation_offset / math.radians(self.rotation_snap_angle))
                        * math.radians(self.rotation_snap_angle)
                    )
                    self.target_rotation = snapped
                else:
                    self.target_rotation = rotation_offset
        else:
            self.aim_valid = False

    def cancel_aim(self) -> None:
        """Cancel current teleport aiming."""
        self.is_aiming = False
        self.arc_visible = False
        self.aim_valid = False
        self.arc_points = []
        self.state = TeleportState.IDLE

    def execute_teleport(self) -> Optional[TeleportResult]:
        """
        Execute teleport to current target position.

        Returns:
            TeleportResult if successful, None if invalid
        """
        if self.state != TeleportState.AIMING or not self.aim_valid:
            return None

        self.state = TeleportState.TRANSITIONING
        self._transition_progress = 0.0

        if self._on_teleport_start:
            self._on_teleport_start()

        # Calculate transition duration based on style
        if self.style == TeleportStyle.INSTANT:
            duration = 0.0
        elif self.style == TeleportStyle.FADE:
            duration = self.fade_duration * 2  # Fade out + fade in
        elif self.style == TeleportStyle.DASH:
            distance = math.sqrt(
                sum(
                    (a - b) ** 2
                    for a, b in zip(self._start_position, self.target_position)
                )
            )
            duration = distance / self.dash_speed
        elif self.style == TeleportStyle.BLINK:
            duration = 0.15  # Quick blink effect
        else:
            duration = 0.0

        result = TeleportResult(
            success=True,
            start_position=self._start_position,
            end_position=self.target_position,
            rotation=self.target_rotation,
            duration=duration,
            style=self.style,
        )

        return result

    def complete_teleport(self) -> None:
        """Mark teleport transition as complete."""
        self.is_aiming = False
        self.arc_visible = False
        self.arc_points = []
        self.state = TeleportState.COOLDOWN
        self._cooldown_remaining = self.cooldown_duration

        if self._on_teleport_end:
            result = TeleportResult(
                success=True,
                start_position=self._start_position,
                end_position=self.target_position,
                rotation=self.target_rotation,
                duration=0.0,
                style=self.style,
            )
            self._on_teleport_end(result)

    def update(self, delta_time: float) -> None:
        """
        Update teleport state.

        Args:
            delta_time: Time since last update in seconds
        """
        if self.state == TeleportState.COOLDOWN:
            self._cooldown_remaining -= delta_time
            if self._cooldown_remaining <= 0:
                self._cooldown_remaining = 0.0
                self.state = TeleportState.IDLE

        elif self.state == TeleportState.TRANSITIONING:
            self._transition_progress += delta_time
            # Transition completion is handled externally based on style

    def get_fade_alpha(self) -> float:
        """
        Get current fade alpha for fade-style teleport.

        Returns:
            Alpha value 0-1 for fade effect
        """
        if self.state != TeleportState.TRANSITIONING:
            return 0.0

        if self.style != TeleportStyle.FADE:
            return 0.0

        half_duration = self.fade_duration
        if self._transition_progress < half_duration:
            # Fading out
            return self._transition_progress / half_duration
        else:
            # Fading in
            return 1.0 - (self._transition_progress - half_duration) / half_duration

    def set_teleport_callback(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_end: Optional[Callable[[TeleportResult], None]] = None,
    ) -> None:
        """
        Set teleport event callbacks.

        Args:
            on_start: Called when teleport begins
            on_end: Called when teleport completes
        """
        self._on_teleport_start = on_start
        self._on_teleport_end = on_end


# =============================================================================
# TELEPORT TARGET COMPONENT
# =============================================================================


@dataclass
class TeleportTarget:
    """
    Marker component for valid teleportation destinations.

    Place on surfaces or areas where teleportation is allowed.

    Attributes:
        teleport_type: Type of teleport allowed (instant, fade, etc)
        surface_normal: Normal direction of the surface
        landing_offset: Offset from surface for player placement
        rotation_hint: Suggested rotation when landing
        priority: Selection priority when multiple targets overlap
    """

    # Configuration
    teleport_type: Annotated[str, Immutable] = "any"  # any, instant, fade, dash
    surface_normal: Annotated[tuple[float, float, float], Tracked] = (0.0, 1.0, 0.0)
    landing_offset: Annotated[float, Tracked] = 0.0
    rotation_hint: Annotated[Optional[float], Tracked] = None  # None = preserve rotation

    # State
    is_valid: Annotated[bool, Tracked] = True
    is_highlighted: Annotated[bool, Tracked, Observable] = False
    priority: Annotated[int, Tracked] = 0

    # Bounds (for area targets)
    is_area: Annotated[bool, Immutable] = False
    area_bounds: Annotated[list[tuple[float, float, float]], Tracked] = field(
        default_factory=list
    )

    # Visual settings
    indicator_color: Annotated[tuple[float, float, float], Tracked] = (0.0, 1.0, 0.5)
    indicator_radius: Annotated[float, Tracked] = 0.5


# =============================================================================
# DECORATOR: @xr_teleport_area
# =============================================================================


def _validate_xr_teleport_area(teleport_type: str = "any", **_: Any) -> None:
    """Validate xr_teleport_area decorator parameters."""
    valid_types = {"any", "instant", "fade", "dash", "blink"}
    if teleport_type not in valid_types:
        raise ValueError(
            f"@xr_teleport_area: 'teleport_type' must be one of {valid_types}, got '{teleport_type}'"
        )


def _xr_teleport_area_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for xr_teleport_area decorator."""
    teleport_type = params.get("teleport_type", "any")
    priority = params.get("priority", 0)
    return [
        Step(Op.TAG, {"key": "xr_teleport_area", "value": True}),
        Step(Op.TAG, {"key": "teleport_type", "value": teleport_type}),
        Step(Op.TAG, {"key": "teleport_priority", "value": priority}),
        Step(Op.REGISTER, {"registry": "xr"}),
    ]


def _after_xr_teleport_area(target: Any, params: dict[str, Any]) -> Any:
    """Post-processing for xr_teleport_area decorator."""
    target._xr_teleport_area = True
    target._teleport_type = params.get("teleport_type", "any")
    target._teleport_priority = params.get("priority", 0)
    return None


xr_teleport_area = make_decorator(
    name="xr_teleport_area",
    steps=_xr_teleport_area_steps,
    doc="Mark an object or surface as a valid teleportation destination.",
    validate=_validate_xr_teleport_area,
    after_steps=_after_xr_teleport_area,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("xr_teleport_area", xr_teleport_area, ("class", "function")),
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


class TeleportLocomotionProvider:
    """
    Abstract provider interface for teleport locomotion.

    Subclass this to integrate with specific XR runtimes or input systems.
    """

    def __init__(self, locomotion: TeleportLocomotion):
        """
        Initialize provider with locomotion component.

        Args:
            locomotion: TeleportLocomotion component to control
        """
        self.locomotion = locomotion

    def on_aim_start(self, position: tuple[float, float, float]) -> None:
        """Called when user starts aiming (e.g., thumbstick press)."""
        self.locomotion.begin_aim(position)

    def on_aim_update(
        self, direction: tuple[float, float, float], rotation: float = 0.0
    ) -> None:
        """Called each frame while aiming."""
        self.locomotion.update_aim(direction, rotation)

    def on_aim_cancel(self) -> None:
        """Called when user cancels aiming."""
        self.locomotion.cancel_aim()

    def on_teleport_confirm(self) -> Optional[TeleportResult]:
        """Called when user confirms teleport (e.g., thumbstick release)."""
        return self.locomotion.execute_teleport()

    def update(self, delta_time: float) -> None:
        """Called each frame to update state."""
        self.locomotion.update(delta_time)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "TeleportStyle",
    "TeleportState",
    "ArcSegmentType",
    # Data classes
    "ArcPoint",
    "TeleportResult",
    # Components
    "TeleportLocomotion",
    "TeleportTarget",
    # Calculator
    "TeleportArcCalculator",
    # Provider
    "TeleportLocomotionProvider",
    # Decorator
    "xr_teleport_area",
    # Type markers
    "Tracked",
    "Range",
    "Observable",
    "Transient",
    "Immutable",
]
