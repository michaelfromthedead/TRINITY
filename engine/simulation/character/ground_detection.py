"""
Ground Detection System.

Provides comprehensive ground sensing capabilities including raycasting,
sphere sweeps, slope detection, ledge detection, and coyote time support.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .character_controller import (
    CollisionHit,
    PhysicsWorldInterface,
    SweepResult,
    Vector3,
)
from .config import (
    COYOTE_TIME_MS,
    GROUND_PROBE_DISTANCE,
    GROUND_SPHERE_PROBE_RADIUS,
    JUMP_BUFFER_TIME_MS,
    LEDGE_DETECTION_HEIGHT,
    LEDGE_GRAB_DISTANCE,
    MASK_GROUND_DETECTION,
    MAX_SLOPE_ANGLE,
    MIN_SLOPE_ANGLE,
    STEEP_SLOPE_ANGLE,
)


# =============================================================================
# Ground Detection Data Structures
# =============================================================================

class GroundType(str, Enum):
    """Types of ground surfaces."""
    SOLID = "solid"
    SLOPE = "slope"
    STEEP = "steep"
    LEDGE = "ledge"
    WATER = "water"
    PLATFORM = "platform"
    NONE = "none"


@dataclass
class GroundInfo:
    """
    Complete information about the ground beneath a character.

    Attributes:
        is_grounded: Whether the character is on solid ground
        normal: Surface normal of the ground
        material: Physical material type of the surface
        distance: Distance to ground surface
        ground_type: Type of ground surface
        slope_angle: Angle of slope in degrees
        is_walkable: Whether the surface is walkable
        platform_id: ID of platform if on moving platform
        friction: Surface friction coefficient
    """
    is_grounded: bool = False
    normal: Vector3 = field(default_factory=Vector3.up)
    material: str = "default"
    distance: float = float("inf")
    ground_type: GroundType = GroundType.NONE
    slope_angle: float = 0.0
    is_walkable: bool = False
    platform_id: Optional[int] = None
    friction: float = 0.6
    hit_point: Vector3 = field(default_factory=Vector3.zero)
    collider_id: int = 0


@dataclass
class LedgeInfo:
    """
    Information about a detected ledge.

    Attributes:
        has_ledge: Whether a ledge was detected
        ledge_position: World position of the ledge
        ledge_normal: Normal pointing away from the wall
        climb_height: Height needed to climb
        grab_point: Optimal grab position
        surface_normal: Normal of the ledge surface (top)
        is_climbable: Whether the ledge can be climbed
    """
    has_ledge: bool = False
    ledge_position: Vector3 = field(default_factory=Vector3.zero)
    ledge_normal: Vector3 = field(default_factory=Vector3.forward)
    climb_height: float = 0.0
    grab_point: Vector3 = field(default_factory=Vector3.zero)
    surface_normal: Vector3 = field(default_factory=Vector3.up)
    is_climbable: bool = False


# =============================================================================
# Ground Detector
# =============================================================================

class GroundDetector:
    """
    Comprehensive ground detection system.

    Provides multiple methods for detecting ground contact including:
    - Raycasting for fast, simple checks
    - Sphere sweeps for more accurate detection
    - Slope angle computation
    - Ledge detection for climbing
    - Coyote time support for platformer mechanics
    """

    def __init__(
        self,
        physics_world: PhysicsWorldInterface,
        character_radius: float = 0.35,
        character_height: float = 1.8,
    ):
        self._physics = physics_world
        self._radius = character_radius
        self._height = character_height

        # Coyote time tracking
        self._last_grounded_time: float = 0.0
        self._was_grounded: bool = False
        self._coyote_time_ms = COYOTE_TIME_MS

        # Jump buffer tracking
        self._last_jump_input_time: float = 0.0
        self._jump_buffer_time_ms = JUMP_BUFFER_TIME_MS

        # Configuration
        self._probe_distance = GROUND_PROBE_DISTANCE
        self._slope_limit = MAX_SLOPE_ANGLE
        self._collision_mask = MASK_GROUND_DETECTION

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_probe_distance(self, distance: float) -> None:
        """Set the ground probe distance."""
        self._probe_distance = max(0.01, distance)

    def set_slope_limit(self, angle_degrees: float) -> None:
        """Set the maximum walkable slope angle in degrees."""
        self._slope_limit = max(0.0, min(90.0, angle_degrees))

    def set_coyote_time(self, time_ms: float) -> None:
        """Set the coyote time in milliseconds."""
        self._coyote_time_ms = max(0.0, time_ms)

    def set_jump_buffer_time(self, time_ms: float) -> None:
        """Set the jump buffer time in milliseconds."""
        self._jump_buffer_time_ms = max(0.0, time_ms)

    # -------------------------------------------------------------------------
    # Ground Detection Methods
    # -------------------------------------------------------------------------

    def detect_ground(self, position: Vector3) -> GroundInfo:
        """
        Perform full ground detection from a position.

        Uses sphere sweep for accurate detection.

        Args:
            position: Character position (foot level)

        Returns:
            Complete ground information
        """
        # Perform sphere sweep
        sweep_info = self.sphere_sweep_ground(position)

        if not sweep_info.is_grounded:
            # Try raycast as fallback
            ray_info = self.raycast_ground(position)
            if ray_info.is_grounded and ray_info.distance <= self._probe_distance * 1.5:
                sweep_info = ray_info

        # Update coyote time tracking
        current_time = time.time() * 1000.0
        if sweep_info.is_grounded:
            self._last_grounded_time = current_time
            self._was_grounded = True
        elif self._was_grounded:
            self._was_grounded = False

        return sweep_info

    def raycast_ground(
        self,
        position: Vector3,
        probe_distance: Optional[float] = None,
    ) -> GroundInfo:
        """
        Perform a simple raycast for ground detection.

        Args:
            position: Character position (foot level)
            probe_distance: Optional override for probe distance

        Returns:
            Ground information from raycast
        """
        distance = probe_distance if probe_distance is not None else self._probe_distance

        # Cast from slightly above foot position
        start = Vector3(position.x, position.y + 0.05, position.z)
        hit = self._physics.raycast(
            start=start,
            direction=Vector3.down(),
            distance=distance + 0.05,
            mask=self._collision_mask,
        )

        if hit is None:
            return GroundInfo()

        # Calculate slope angle
        slope_angle = self.detect_slope_angle(hit.normal)
        is_walkable = slope_angle <= self._slope_limit
        ground_type = self._determine_ground_type(slope_angle, hit.material)

        return GroundInfo(
            is_grounded=True,
            normal=hit.normal,
            material=hit.material,
            distance=hit.distance - 0.05,
            ground_type=ground_type,
            slope_angle=slope_angle,
            is_walkable=is_walkable,
            friction=self._get_friction(hit.material),
            hit_point=hit.point,
            collider_id=hit.collider_id,
        )

    def sphere_sweep_ground(
        self,
        position: Vector3,
        probe_distance: Optional[float] = None,
    ) -> GroundInfo:
        """
        Perform a sphere sweep for more accurate ground detection.

        Args:
            position: Character position (foot level)
            probe_distance: Optional override for probe distance

        Returns:
            Ground information from sphere sweep
        """
        distance = probe_distance if probe_distance is not None else self._probe_distance

        # Sweep from character center downward
        start = Vector3(position.x, position.y + self._radius + 0.01, position.z)
        end = Vector3(position.x, position.y - distance, position.z)

        sweep = self._physics.sphere_sweep(
            start=start,
            end=end,
            radius=GROUND_SPHERE_PROBE_RADIUS,
            mask=self._collision_mask,
        )

        if not sweep.hit or sweep.first_hit is None:
            return GroundInfo()

        hit = sweep.first_hit
        slope_angle = self.detect_slope_angle(hit.normal)
        is_walkable = slope_angle <= self._slope_limit
        ground_type = self._determine_ground_type(slope_angle, hit.material)

        return GroundInfo(
            is_grounded=True,
            normal=hit.normal,
            material=hit.material,
            distance=hit.distance,
            ground_type=ground_type,
            slope_angle=slope_angle,
            is_walkable=is_walkable,
            friction=self._get_friction(hit.material),
            hit_point=hit.point,
            collider_id=hit.collider_id,
        )

    def detect_slope_angle(self, normal: Vector3) -> float:
        """
        Calculate the slope angle from a surface normal.

        Args:
            normal: Surface normal vector

        Returns:
            Slope angle in degrees (0 = flat, 90 = vertical)
        """
        # Dot product with up vector gives cosine of angle
        dot = max(-1.0, min(1.0, normal.dot(Vector3.up())))
        angle_rad = math.acos(dot)
        return math.degrees(angle_rad)

    def is_flat_ground(self, normal: Vector3) -> bool:
        """Check if surface is essentially flat."""
        return self.detect_slope_angle(normal) < MIN_SLOPE_ANGLE

    def is_walkable_slope(self, normal: Vector3) -> bool:
        """Check if slope is walkable."""
        return self.detect_slope_angle(normal) <= self._slope_limit

    def is_steep_slope(self, normal: Vector3) -> bool:
        """Check if slope is too steep to stand on."""
        angle = self.detect_slope_angle(normal)
        # Use epsilon to be consistent with is_walkable_slope boundary
        return angle > self._slope_limit + 0.001 and angle < 90.0

    # -------------------------------------------------------------------------
    # Ledge Detection
    # -------------------------------------------------------------------------

    def detect_ledge(
        self,
        position: Vector3,
        forward: Vector3,
        max_height: Optional[float] = None,
    ) -> LedgeInfo:
        """
        Detect climbable ledges in front of the character.

        Args:
            position: Character position (foot level)
            forward: Forward direction of the character
            max_height: Maximum ledge height to detect

        Returns:
            Ledge information if found
        """
        ledge_height = max_height if max_height is not None else LEDGE_DETECTION_HEIGHT

        # Normalize forward to horizontal
        forward_h = forward.horizontal().normalized()

        # Cast forward at head height to find wall
        head_pos = Vector3(position.x, position.y + self._height * 0.8, position.z)
        wall_hit = self._physics.raycast(
            start=head_pos,
            direction=forward_h,
            distance=LEDGE_GRAB_DISTANCE,
            mask=self._collision_mask,
        )

        if wall_hit is None:
            return LedgeInfo()

        # Cast downward from above the wall to find ledge surface
        ledge_check_pos = Vector3(
            wall_hit.point.x + forward_h.x * 0.1,
            position.y + ledge_height,
            wall_hit.point.z + forward_h.z * 0.1,
        )

        ledge_hit = self._physics.raycast(
            start=ledge_check_pos,
            direction=Vector3.down(),
            distance=ledge_height,
            mask=self._collision_mask,
        )

        if ledge_hit is None:
            return LedgeInfo()

        # Check if ledge surface is walkable
        ledge_angle = self.detect_slope_angle(ledge_hit.normal)
        if ledge_angle > self._slope_limit:
            return LedgeInfo()

        # Calculate climb height
        climb_height = ledge_hit.point.y - position.y

        # Determine grab point
        grab_point = Vector3(
            wall_hit.point.x,
            ledge_hit.point.y,
            wall_hit.point.z,
        )

        return LedgeInfo(
            has_ledge=True,
            ledge_position=ledge_hit.point,
            ledge_normal=-forward_h,  # Normal points away from wall
            climb_height=climb_height,
            grab_point=grab_point,
            surface_normal=ledge_hit.normal,
            is_climbable=climb_height > 0 and climb_height <= ledge_height,
        )

    # -------------------------------------------------------------------------
    # Coyote Time
    # -------------------------------------------------------------------------

    def is_in_coyote_time(self) -> bool:
        """
        Check if within coyote time window.

        Coyote time allows jumps shortly after leaving a platform.

        Returns:
            True if within coyote time window
        """
        if self._coyote_time_ms <= 0:
            return False

        current_time = time.time() * 1000.0
        elapsed = current_time - self._last_grounded_time
        return elapsed <= self._coyote_time_ms

    def register_jump_input(self) -> None:
        """Register a jump input for jump buffering."""
        self._last_jump_input_time = time.time() * 1000.0

    def is_jump_buffered(self) -> bool:
        """
        Check if there's a buffered jump input.

        Returns:
            True if jump was input within buffer window
        """
        if self._jump_buffer_time_ms <= 0:
            return False

        current_time = time.time() * 1000.0
        elapsed = current_time - self._last_jump_input_time
        return elapsed <= self._jump_buffer_time_ms

    def clear_jump_buffer(self) -> None:
        """Clear the jump buffer."""
        self._last_jump_input_time = 0.0

    def can_jump(self, is_grounded: bool) -> bool:
        """
        Check if character can jump considering coyote time.

        Args:
            is_grounded: Current grounded state

        Returns:
            True if jump is allowed
        """
        return is_grounded or self.is_in_coyote_time()

    # -------------------------------------------------------------------------
    # Multi-Point Detection
    # -------------------------------------------------------------------------

    def detect_ground_multi_point(
        self,
        position: Vector3,
        forward: Vector3,
        num_points: int = 4,
    ) -> tuple[GroundInfo, list[GroundInfo]]:
        """
        Perform multi-point ground detection for better accuracy.

        Casts multiple rays around the character's feet.

        Args:
            position: Character position
            forward: Forward direction
            num_points: Number of probe points (minimum 4)

        Returns:
            Tuple of (combined result, individual results)
        """
        num_points = max(4, num_points)
        probes: list[GroundInfo] = []

        # Center probe
        center_info = self.raycast_ground(position)
        probes.append(center_info)

        # Radial probes
        angle_step = 2.0 * math.pi / (num_points - 1)
        probe_radius = self._radius * 0.8

        for i in range(num_points - 1):
            angle = i * angle_step
            offset = Vector3(
                math.cos(angle) * probe_radius,
                0.0,
                math.sin(angle) * probe_radius,
            )
            probe_pos = position + offset
            info = self.raycast_ground(probe_pos)
            probes.append(info)

        # Combine results
        grounded_probes = [p for p in probes if p.is_grounded]

        if not grounded_probes:
            return GroundInfo(), probes

        # Average normal
        avg_normal = Vector3.zero()
        min_distance = float("inf")
        avg_slope = 0.0
        primary_material = "default"

        for probe in grounded_probes:
            avg_normal = avg_normal + probe.normal
            min_distance = min(min_distance, probe.distance)
            avg_slope += probe.slope_angle

        avg_normal = avg_normal.normalized()
        avg_slope /= len(grounded_probes)

        # Use most common material
        material_counts: dict[str, int] = {}
        for probe in grounded_probes:
            material_counts[probe.material] = material_counts.get(probe.material, 0) + 1
        primary_material = max(material_counts.keys(), key=lambda m: material_counts[m])

        combined = GroundInfo(
            is_grounded=True,
            normal=avg_normal,
            material=primary_material,
            distance=min_distance,
            ground_type=self._determine_ground_type(avg_slope, primary_material),
            slope_angle=avg_slope,
            is_walkable=avg_slope <= self._slope_limit,
            friction=self._get_friction(primary_material),
        )

        return combined, probes

    # -------------------------------------------------------------------------
    # Edge Detection
    # -------------------------------------------------------------------------

    def detect_edge(
        self,
        position: Vector3,
        direction: Vector3,
        check_distance: float = 1.0,
    ) -> tuple[bool, float]:
        """
        Detect if there's an edge/drop-off ahead.

        Args:
            position: Current position
            direction: Direction to check
            check_distance: How far ahead to check

        Returns:
            Tuple of (is_edge, drop_height)
        """
        direction_h = direction.horizontal().normalized()

        # Check current ground
        current_ground = self.raycast_ground(position)
        if not current_ground.is_grounded:
            return False, 0.0

        # Check ahead position
        ahead_pos = position + direction_h * check_distance
        ahead_ground = self.raycast_ground(ahead_pos, probe_distance=self._height)

        if not ahead_ground.is_grounded:
            # Deep drop or no ground ahead
            return True, float("inf")

        drop_height = ahead_pos.y - ahead_ground.hit_point.y - (
            position.y - current_ground.hit_point.y
        )

        # Significant drop
        if drop_height > 0.5:
            return True, drop_height

        return False, drop_height

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _determine_ground_type(self, slope_angle: float, material: str) -> GroundType:
        """Determine the ground type from slope and material."""
        if material == "water":
            return GroundType.WATER

        if slope_angle < MIN_SLOPE_ANGLE:
            return GroundType.SOLID
        elif slope_angle <= self._slope_limit:
            return GroundType.SLOPE
        elif slope_angle < STEEP_SLOPE_ANGLE:
            return GroundType.STEEP
        else:
            return GroundType.NONE

    def _get_friction(self, material: str) -> float:
        """Get friction coefficient for a material."""
        friction_map = {
            "default": 0.6,
            "concrete": 0.8,
            "metal": 0.4,
            "wood": 0.5,
            "grass": 0.5,
            "sand": 0.4,
            "ice": 0.05,
            "mud": 0.3,
            "water": 0.0,
        }
        return friction_map.get(material, 0.6)
