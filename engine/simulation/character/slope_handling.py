"""
Slope Physics Handling.

Provides functionality for handling slopes, steps, and terrain navigation
including walkability checks, velocity modifiers, and step up/down mechanics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .character_controller import (
    CollisionHit,
    PhysicsWorldInterface,
    SweepResult,
    Vector3,
)
from .config import (
    DEFAULT_GRAVITY,
    DEFAULT_STEP_HEIGHT,
    GROUND_PROBE_DISTANCE,
    MASK_GROUND_DETECTION,
    MAX_FALL_VELOCITY,
    MAX_SLOPE_ANGLE,
    MIN_SLOPE_ANGLE,
    SKIN_WIDTH,
    STEEP_SLOPE_ANGLE,
)


# =============================================================================
# Slope Information
# =============================================================================

@dataclass
class SlopeInfo:
    """
    Detailed information about a slope surface.

    Attributes:
        angle: Slope angle in degrees
        direction: Downhill direction on the slope
        normal: Surface normal
        is_walkable: Whether the slope can be walked on
        is_steep: Whether the slope causes sliding
        velocity_modifier: Speed modifier for the slope
        friction: Slope friction coefficient
    """
    angle: float = 0.0
    direction: Vector3 = field(default_factory=Vector3.zero)
    normal: Vector3 = field(default_factory=Vector3.up)
    is_walkable: bool = True
    is_steep: bool = False
    velocity_modifier: float = 1.0
    friction: float = 0.6


@dataclass
class StepInfo:
    """
    Information about a step attempt.

    Attributes:
        can_step: Whether the step is possible
        step_height: Height of the step
        surface_normal: Normal of the step surface
        landing_position: Position after completing step
    """
    can_step: bool = False
    step_height: float = 0.0
    surface_normal: Vector3 = field(default_factory=Vector3.up)
    landing_position: Vector3 = field(default_factory=Vector3.zero)


# =============================================================================
# Slope Handler
# =============================================================================

class SlopeHandler:
    """
    Handles slope-related physics calculations.

    Provides:
    - Walkability determination
    - Velocity modifiers for slopes
    - Sliding physics on steep slopes
    - Step up/down mechanics for stairs
    - Ground snapping
    """

    def __init__(
        self,
        physics_world: PhysicsWorldInterface,
        slope_limit: float = MAX_SLOPE_ANGLE,
        step_height: float = DEFAULT_STEP_HEIGHT,
    ):
        self._physics = physics_world
        self._slope_limit = slope_limit
        self._step_height = step_height
        self._collision_mask = MASK_GROUND_DETECTION

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_slope_limit(self, angle_degrees: float) -> None:
        """Set maximum walkable slope angle in degrees."""
        self._slope_limit = max(0.0, min(90.0, angle_degrees))

    def set_step_height(self, height: float) -> None:
        """Set maximum step height."""
        self._step_height = max(0.0, height)

    @property
    def slope_limit(self) -> float:
        """Maximum walkable slope angle."""
        return self._slope_limit

    @property
    def step_height(self) -> float:
        """Maximum step height."""
        return self._step_height

    # -------------------------------------------------------------------------
    # Slope Checks
    # -------------------------------------------------------------------------

    def is_walkable_slope(self, normal: Vector3) -> bool:
        """
        Check if a surface is walkable based on its normal.

        Args:
            normal: Surface normal vector

        Returns:
            True if the slope is walkable
        """
        angle = self.compute_slope_angle(normal)
        # Use small epsilon to handle floating point comparison at exact boundary
        return angle <= self._slope_limit + 0.001

    def is_steep_slope(self, normal: Vector3) -> bool:
        """
        Check if a surface is too steep to stand on (causes sliding).

        Args:
            normal: Surface normal vector

        Returns:
            True if the slope causes sliding
        """
        angle = self.compute_slope_angle(normal)
        return angle > self._slope_limit and angle < 90.0

    def is_wall(self, normal: Vector3) -> bool:
        """
        Check if a surface is essentially a wall.

        Args:
            normal: Surface normal vector

        Returns:
            True if the surface is too steep to be a slope
        """
        angle = self.compute_slope_angle(normal)
        return angle >= 85.0

    def compute_slope_angle(self, normal: Vector3) -> float:
        """
        Compute the slope angle from a surface normal.

        Args:
            normal: Surface normal vector

        Returns:
            Slope angle in degrees (0 = flat, 90 = vertical)
        """
        dot = max(-1.0, min(1.0, normal.dot(Vector3.up())))
        angle_rad = math.acos(dot)
        return math.degrees(angle_rad)

    def get_slope_info(self, normal: Vector3, friction: float = 0.6) -> SlopeInfo:
        """
        Get complete slope information from a surface normal.

        Args:
            normal: Surface normal vector
            friction: Surface friction coefficient

        Returns:
            Complete slope information
        """
        angle = self.compute_slope_angle(normal)

        # Calculate downhill direction
        # Project 'down' onto the slope plane
        down = Vector3.down()
        normal_component = normal * down.dot(normal)
        slope_direction = (down - normal_component).normalized()

        # Determine walkability - use epsilon for consistent boundary handling
        is_walkable = angle <= self._slope_limit + 0.001
        is_steep = angle > self._slope_limit + 0.001 and angle < 85.0

        # Calculate velocity modifier
        velocity_modifier = self.compute_slope_velocity_modifier(angle, is_uphill=False)

        return SlopeInfo(
            angle=angle,
            direction=slope_direction,
            normal=normal,
            is_walkable=is_walkable,
            is_steep=is_steep,
            velocity_modifier=velocity_modifier,
            friction=friction,
        )

    # -------------------------------------------------------------------------
    # Velocity Modifiers
    # -------------------------------------------------------------------------

    def compute_slope_velocity_modifier(
        self,
        slope_angle: float,
        is_uphill: bool,
        base_modifier: float = 1.0,
    ) -> float:
        """
        Compute velocity modifier based on slope angle.

        Args:
            slope_angle: Slope angle in degrees
            is_uphill: Whether moving uphill
            base_modifier: Base multiplier to apply

        Returns:
            Velocity multiplier (1.0 = no change)
        """
        if slope_angle < MIN_SLOPE_ANGLE:
            return base_modifier

        # Normalize angle to 0-1 range (relative to slope limit)
        normalized = min(slope_angle / self._slope_limit, 1.5)

        if is_uphill:
            # Slower going uphill
            # At slope limit, speed is reduced to ~50%
            modifier = 1.0 - (normalized * 0.5)
        else:
            # Faster going downhill
            # At slope limit, speed is increased by ~30%
            modifier = 1.0 + (normalized * 0.3)

        return max(0.1, modifier * base_modifier)

    def compute_directional_modifier(
        self,
        slope_normal: Vector3,
        movement_direction: Vector3,
    ) -> float:
        """
        Compute velocity modifier based on movement direction on slope.

        Args:
            slope_normal: Surface normal of the slope
            movement_direction: Direction of movement (normalized)

        Returns:
            Velocity multiplier
        """
        if movement_direction.magnitude() < 0.01:
            return 1.0

        slope_info = self.get_slope_info(slope_normal)

        if slope_info.angle < MIN_SLOPE_ANGLE:
            return 1.0

        # Calculate how much we're moving uphill vs downhill
        # Dot product with downhill direction: positive = downhill, negative = uphill
        dir_horizontal = movement_direction.horizontal().normalized()
        slope_dir = slope_info.direction.horizontal().normalized()

        alignment = dir_horizontal.dot(slope_dir)

        # alignment: 1.0 = downhill, -1.0 = uphill, 0.0 = across
        is_uphill = alignment < 0

        return self.compute_slope_velocity_modifier(
            slope_info.angle, is_uphill, base_modifier=1.0
        )

    # -------------------------------------------------------------------------
    # Sliding Physics
    # -------------------------------------------------------------------------

    def slide_down_steep_slope(
        self,
        position: Vector3,
        velocity: Vector3,
        slope_normal: Vector3,
        dt: float,
        friction: float = 0.3,
    ) -> tuple[Vector3, Vector3]:
        """
        Calculate sliding movement on a steep slope.

        Args:
            position: Current position
            velocity: Current velocity
            slope_normal: Surface normal of the slope
            dt: Delta time
            friction: Surface friction

        Returns:
            Tuple of (new_position, new_velocity)
        """
        slope_info = self.get_slope_info(slope_normal, friction)

        if not slope_info.is_steep:
            return position, velocity

        # Gravity component along slope - use config constant
        gravity = abs(DEFAULT_GRAVITY)
        slope_gravity = gravity * math.sin(math.radians(slope_info.angle))

        # Apply friction
        friction_force = slope_gravity * friction

        # Net acceleration down the slope
        net_acceleration = slope_gravity - friction_force

        # Update velocity
        slide_direction = slope_info.direction
        slide_velocity = slide_direction * (velocity.dot(slide_direction) + net_acceleration * dt)

        # Clamp slide speed using config constant for max fall velocity
        max_slide_speed = MAX_FALL_VELOCITY * 0.4  # Sliding is slower than freefall
        if slide_velocity.magnitude() > max_slide_speed:
            slide_velocity = slide_velocity.normalized() * max_slide_speed

        # Calculate new position
        new_position = position + slide_velocity * dt

        # Combine with existing velocity perpendicular to slope
        perpendicular = velocity - slide_direction * velocity.dot(slide_direction)
        new_velocity = slide_velocity + perpendicular

        return new_position, new_velocity

    def get_slide_direction(self, slope_normal: Vector3) -> Vector3:
        """
        Get the direction to slide on a slope.

        Args:
            slope_normal: Surface normal

        Returns:
            Downhill slide direction
        """
        slope_info = self.get_slope_info(slope_normal)
        return slope_info.direction

    # -------------------------------------------------------------------------
    # Step Up/Down
    # -------------------------------------------------------------------------

    def step_up(
        self,
        position: Vector3,
        movement: Vector3,
        capsule_radius: float,
        capsule_height: float,
    ) -> Optional[StepInfo]:
        """
        Try to step up over an obstacle.

        Args:
            position: Current position (foot level)
            movement: Desired movement direction
            capsule_radius: Character capsule radius
            capsule_height: Character capsule height

        Returns:
            StepInfo if step is possible, None otherwise
        """
        if movement.magnitude() < 0.001:
            return None

        # Phase 1: Cast upward to check clearance
        up_end = position + Vector3(0.0, self._step_height, 0.0)
        up_sweep = self._physics.capsule_sweep(
            start=position,
            end=up_end,
            radius=capsule_radius + SKIN_WIDTH,
            height=capsule_height,
            mask=self._collision_mask,
        )

        if up_sweep.blocked and up_sweep.safe_fraction < 0.5:
            return None

        step_up_height = self._step_height * up_sweep.safe_fraction
        elevated_pos = position + Vector3(0.0, step_up_height, 0.0)

        # Phase 2: Cast forward from elevated position
        forward = movement.horizontal().normalized()
        forward_end = elevated_pos + forward * (capsule_radius * 2.0 + movement.magnitude())

        forward_sweep = self._physics.capsule_sweep(
            start=elevated_pos,
            end=forward_end,
            radius=capsule_radius + SKIN_WIDTH,
            height=capsule_height,
            mask=self._collision_mask,
        )

        if forward_sweep.blocked and forward_sweep.safe_fraction < 0.3:
            return None

        forward_pos = elevated_pos + forward * (forward_sweep.safe_fraction * forward.magnitude())

        # Phase 3: Cast downward to find landing
        down_end = forward_pos + Vector3(0.0, -self._step_height * 2.0, 0.0)
        down_sweep = self._physics.capsule_sweep(
            start=forward_pos,
            end=down_end,
            radius=capsule_radius + SKIN_WIDTH,
            height=capsule_height,
            mask=self._collision_mask,
        )

        if not down_sweep.hit or down_sweep.first_hit is None:
            return None

        hit = down_sweep.first_hit

        # Check if landing surface is walkable
        if not self.is_walkable_slope(hit.normal):
            return None

        # Calculate landing position
        step_down = self._step_height * 2.0 * down_sweep.safe_fraction
        landing_pos = forward_pos + Vector3(0.0, -step_down + SKIN_WIDTH, 0.0)

        # Calculate actual step height
        actual_step = landing_pos.y - position.y

        # Only count as step if there's meaningful height change
        if abs(actual_step) < 0.05:
            return None

        return StepInfo(
            can_step=True,
            step_height=actual_step,
            surface_normal=hit.normal,
            landing_position=landing_pos,
        )

    def step_down(
        self,
        position: Vector3,
        capsule_radius: float,
        capsule_height: float,
        max_step_down: Optional[float] = None,
    ) -> Optional[StepInfo]:
        """
        Try to step down (snap to ground) when going down stairs or slopes.

        Args:
            position: Current position (foot level)
            capsule_radius: Character capsule radius
            capsule_height: Character capsule height
            max_step_down: Maximum step down distance

        Returns:
            StepInfo if step down found, None otherwise
        """
        step_down = max_step_down if max_step_down is not None else self._step_height

        # Cast downward
        down_end = position + Vector3(0.0, -step_down, 0.0)
        down_sweep = self._physics.capsule_sweep(
            start=position,
            end=down_end,
            radius=capsule_radius + SKIN_WIDTH,
            height=capsule_height,
            mask=self._collision_mask,
        )

        if not down_sweep.hit or down_sweep.first_hit is None:
            return None

        hit = down_sweep.first_hit

        # Check if surface is walkable
        if not self.is_walkable_slope(hit.normal):
            return None

        # Calculate landing position
        actual_step = step_down * down_sweep.safe_fraction
        landing_pos = position + Vector3(0.0, -actual_step + SKIN_WIDTH, 0.0)

        return StepInfo(
            can_step=True,
            step_height=-actual_step,
            surface_normal=hit.normal,
            landing_position=landing_pos,
        )

    def should_step_down(
        self,
        velocity: Vector3,
        is_grounded: bool,
        was_grounded: bool,
    ) -> bool:
        """
        Determine if step down should be applied.

        Args:
            velocity: Current velocity
            is_grounded: Current grounded state
            was_grounded: Previous grounded state

        Returns:
            True if step down should be attempted
        """
        # Only step down if we were grounded and moving horizontally
        if not was_grounded:
            return False

        # Don't step down if jumping
        if velocity.y > 0.5:
            return False

        # Don't step down if moving too fast downward
        if velocity.y < -5.0:
            return False

        return True

    # -------------------------------------------------------------------------
    # Ground Snapping
    # -------------------------------------------------------------------------

    def snap_to_ground(
        self,
        position: Vector3,
        capsule_radius: float,
        snap_distance: float = 0.2,
    ) -> tuple[Vector3, bool]:
        """
        Snap position to ground if close enough.

        Args:
            position: Current position
            capsule_radius: Character capsule radius
            snap_distance: Maximum distance to snap

        Returns:
            Tuple of (snapped_position, was_snapped)
        """
        # Raycast downward
        start = Vector3(position.x, position.y + 0.1, position.z)
        hit = self._physics.raycast(
            start=start,
            direction=Vector3.down(),
            distance=snap_distance + 0.1,
            mask=self._collision_mask,
        )

        if hit is None:
            return position, False

        if not self.is_walkable_slope(hit.normal):
            return position, False

        # Calculate snapped position
        snapped_y = hit.point.y + SKIN_WIDTH
        snapped_pos = Vector3(position.x, snapped_y, position.z)

        return snapped_pos, True

    # -------------------------------------------------------------------------
    # Slope Movement Projection
    # -------------------------------------------------------------------------

    def project_on_slope(
        self,
        movement: Vector3,
        slope_normal: Vector3,
    ) -> Vector3:
        """
        Project movement vector onto a slope surface.

        Args:
            movement: Desired movement vector
            slope_normal: Surface normal of the slope

        Returns:
            Movement projected onto the slope plane
        """
        # Remove component perpendicular to slope
        dot = movement.dot(slope_normal)
        return movement - slope_normal * dot

    def get_uphill_direction(self, slope_normal: Vector3) -> Vector3:
        """
        Get the uphill direction on a slope.

        Args:
            slope_normal: Surface normal

        Returns:
            Direction pointing uphill
        """
        slope_info = self.get_slope_info(slope_normal)
        return -slope_info.direction

    def calculate_slope_effect(
        self,
        velocity: Vector3,
        slope_normal: Vector3,
        dt: float,
    ) -> Vector3:
        """
        Calculate the effect of a slope on velocity.

        Args:
            velocity: Current velocity
            slope_normal: Surface normal
            dt: Delta time

        Returns:
            Velocity delta from slope effect
        """
        slope_info = self.get_slope_info(slope_normal)

        if slope_info.angle < MIN_SLOPE_ANGLE:
            return Vector3.zero()

        # Calculate gravitational component along slope using config constant
        gravity = abs(DEFAULT_GRAVITY)
        gravity_effect = slope_info.direction * (gravity * math.sin(math.radians(slope_info.angle)) * dt)

        # Apply friction
        friction_effect = gravity_effect * slope_info.friction

        return gravity_effect - friction_effect
