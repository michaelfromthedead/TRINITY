"""
Character Controller Implementation.

Provides the main CharacterController class for handling character movement,
collision detection, and physics interaction. Supports kinematic, dynamic,
and hybrid controller modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .config import (
    AIR_CONTROL,
    DEFAULT_CAPSULE_HEIGHT,
    DEFAULT_CAPSULE_RADIUS,
    DEFAULT_GRAVITY,
    DEFAULT_JUMP_VELOCITY,
    DEFAULT_STEP_HEIGHT,
    GROUND_PROBE_DISTANCE,
    MASK_CHARACTER_MOVEMENT,
    MAX_COLLISION_ITERATIONS,
    MAX_DEPENETRATION_VELOCITY,
    MAX_FALL_VELOCITY,
    MAX_SLOPE_ANGLE,
    MIN_MOVE_DISTANCE,
    SKIN_WIDTH,
)


# =============================================================================
# Vector3 and Transform Types (simplified for standalone use)
# =============================================================================

@dataclass
class Vector3:
    """3D vector for positions and directions."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3:
        return self.__mul__(scalar)

    def __neg__(self) -> Vector3:
        return Vector3(-self.x, -self.y, -self.z)

    def __truediv__(self, scalar: float) -> Vector3:
        if abs(scalar) < 1e-10:
            return Vector3(0.0, 0.0, 0.0)
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: Vector3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def magnitude_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> Vector3:
        mag = self.magnitude()
        if mag < 1e-10:
            return Vector3(0.0, 0.0, 0.0)
        return self / mag

    def horizontal(self) -> Vector3:
        """Return horizontal component (XZ plane)."""
        return Vector3(self.x, 0.0, self.z)

    @staticmethod
    def up() -> Vector3:
        return Vector3(0.0, 1.0, 0.0)

    @staticmethod
    def down() -> Vector3:
        return Vector3(0.0, -1.0, 0.0)

    @staticmethod
    def forward() -> Vector3:
        return Vector3(0.0, 0.0, 1.0)

    @staticmethod
    def right() -> Vector3:
        return Vector3(1.0, 0.0, 0.0)

    @staticmethod
    def zero() -> Vector3:
        return Vector3(0.0, 0.0, 0.0)

    @staticmethod
    def one() -> Vector3:
        return Vector3(1.0, 1.0, 1.0)

    @staticmethod
    def lerp(a: Vector3, b: Vector3, t: float) -> Vector3:
        t = max(0.0, min(1.0, t))
        return a + (b - a) * t


@dataclass
class Quaternion:
    """Quaternion for rotations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def rotate_vector(self, v: Vector3) -> Vector3:
        """Rotate a vector by this quaternion."""
        q_vec = Vector3(self.x, self.y, self.z)
        uv = q_vec.cross(v)
        uuv = q_vec.cross(uv)
        return v + (uv * self.w + uuv) * 2.0

    @staticmethod
    def identity() -> Quaternion:
        return Quaternion(0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def from_euler(pitch: float, yaw: float, roll: float) -> Quaternion:
        """Create quaternion from Euler angles (radians)."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return Quaternion(
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
            w=cr * cp * cy + sr * sp * sy,
        )


@dataclass
class Transform:
    """Transform with position, rotation, and scale."""
    position: Vector3 = field(default_factory=Vector3.zero)
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    scale: Vector3 = field(default_factory=Vector3.one)

    def transform_point(self, point: Vector3) -> Vector3:
        """Transform a point from local to world space."""
        scaled = Vector3(
            point.x * self.scale.x,
            point.y * self.scale.y,
            point.z * self.scale.z,
        )
        rotated = self.rotation.rotate_vector(scaled)
        return rotated + self.position

    def inverse_transform_point(self, point: Vector3) -> Vector3:
        """Transform a point from world to local space."""
        local = point - self.position
        # Simplified inverse rotation (assumes unit quaternion)
        inv_rot = Quaternion(-self.rotation.x, -self.rotation.y,
                            -self.rotation.z, self.rotation.w)
        rotated = inv_rot.rotate_vector(local)
        return Vector3(
            rotated.x / self.scale.x if abs(self.scale.x) > 1e-10 else 0.0,
            rotated.y / self.scale.y if abs(self.scale.y) > 1e-10 else 0.0,
            rotated.z / self.scale.z if abs(self.scale.z) > 1e-10 else 0.0,
        )


# =============================================================================
# Controller Types
# =============================================================================

class ControllerType(str, Enum):
    """Types of character controller implementations."""
    KINEMATIC = "kinematic"   # Directly controlled, no physics simulation
    DYNAMIC = "dynamic"       # Fully physics-driven
    HYBRID = "hybrid"         # Kinematic with physics interactions


# =============================================================================
# Collision Data Structures
# =============================================================================

@dataclass
class CollisionHit:
    """Information about a collision hit."""
    point: Vector3 = field(default_factory=Vector3.zero)
    normal: Vector3 = field(default_factory=Vector3.up)
    distance: float = 0.0
    penetration: float = 0.0
    collider_id: int = 0
    material: str = "default"
    is_trigger: bool = False


@dataclass
class SweepResult:
    """Result of a shape sweep test."""
    hit: bool = False
    hits: list[CollisionHit] = field(default_factory=list)
    blocked: bool = False
    start_penetrating: bool = False
    safe_fraction: float = 1.0

    @property
    def first_hit(self) -> Optional[CollisionHit]:
        """Get the first/closest hit."""
        return self.hits[0] if self.hits else None


@dataclass
class ControllerCollision:
    """Collision data for the character controller."""
    hit: CollisionHit
    move_direction: Vector3 = field(default_factory=Vector3.zero)
    move_length: float = 0.0
    controller_velocity: Vector3 = field(default_factory=Vector3.zero)
    other_velocity: Vector3 = field(default_factory=Vector3.zero)


# =============================================================================
# Physics World Interface (Abstract)
# =============================================================================

class PhysicsWorldInterface:
    """Interface for physics world operations. Override in actual implementation."""

    def capsule_sweep(
        self,
        start: Vector3,
        end: Vector3,
        radius: float,
        height: float,
        mask: int = MASK_CHARACTER_MOVEMENT,
    ) -> SweepResult:
        """Perform a capsule sweep test."""
        return SweepResult()

    def raycast(
        self,
        start: Vector3,
        direction: Vector3,
        distance: float,
        mask: int = MASK_CHARACTER_MOVEMENT,
    ) -> Optional[CollisionHit]:
        """Perform a raycast."""
        return None

    def sphere_sweep(
        self,
        start: Vector3,
        end: Vector3,
        radius: float,
        mask: int = MASK_CHARACTER_MOVEMENT,
    ) -> SweepResult:
        """Perform a sphere sweep test."""
        return SweepResult()

    def overlap_capsule(
        self,
        position: Vector3,
        radius: float,
        height: float,
        mask: int = MASK_CHARACTER_MOVEMENT,
    ) -> list[CollisionHit]:
        """Check for overlapping colliders."""
        return []

    def get_collider_velocity(self, collider_id: int) -> Vector3:
        """Get velocity of a collider (for moving platforms)."""
        return Vector3.zero()


# =============================================================================
# Character Controller
# =============================================================================

@dataclass
class CharacterControllerConfig:
    """Configuration for character controller."""
    radius: float = DEFAULT_CAPSULE_RADIUS
    height: float = DEFAULT_CAPSULE_HEIGHT
    step_height: float = DEFAULT_STEP_HEIGHT
    slope_limit: float = MAX_SLOPE_ANGLE
    skin_width: float = SKIN_WIDTH
    min_move_distance: float = MIN_MOVE_DISTANCE
    controller_type: ControllerType = ControllerType.KINEMATIC
    gravity: float = DEFAULT_GRAVITY
    jump_velocity: float = DEFAULT_JUMP_VELOCITY
    air_control: float = AIR_CONTROL
    collision_mask: int = MASK_CHARACTER_MOVEMENT


class CharacterController:
    """
    Main character controller for movement and collision.

    Handles:
    - Move and slide collision resolution
    - Step up/down for stairs and small obstacles
    - Slope handling
    - Ground detection
    - Jump mechanics
    - Platform attachment
    """

    def __init__(
        self,
        physics_world: PhysicsWorldInterface,
        config: Optional[CharacterControllerConfig] = None,
    ):
        self._physics = physics_world
        self._config = config or CharacterControllerConfig()

        # State
        self._position = Vector3.zero()
        self._rotation = Quaternion.identity()
        self._velocity = Vector3.zero()
        self._external_velocity = Vector3.zero()

        # Ground state
        self._is_grounded = False
        self._ground_normal = Vector3.up()
        self._ground_distance = float("inf")
        self._ground_material = "default"
        self._ground_collider_id = 0

        # Movement state
        self._last_move_direction = Vector3.zero()
        self._collision_flags = 0

        # Callbacks
        self._on_collision: Optional[Callable[[ControllerCollision], None]] = None
        self._on_ground_change: Optional[Callable[[bool], None]] = None

        # Platform tracking
        self._attached_platform_id: Optional[int] = None
        self._platform_offset = Vector3.zero()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def position(self) -> Vector3:
        """Current position of the controller."""
        return self._position

    @position.setter
    def position(self, value: Vector3) -> None:
        self._position = value

    @property
    def rotation(self) -> Quaternion:
        """Current rotation of the controller."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: Quaternion) -> None:
        self._rotation = value

    @property
    def velocity(self) -> Vector3:
        """Current velocity of the controller."""
        return self._velocity

    @velocity.setter
    def velocity(self, value: Vector3) -> None:
        self._velocity = value

    @property
    def is_grounded(self) -> bool:
        """Whether the controller is on the ground."""
        return self._is_grounded

    @property
    def ground_normal(self) -> Vector3:
        """Normal of the ground surface."""
        return self._ground_normal

    @property
    def ground_material(self) -> str:
        """Material type of the ground surface."""
        return self._ground_material

    @property
    def controller_type(self) -> ControllerType:
        """Type of controller (kinematic, dynamic, hybrid)."""
        return self._config.controller_type

    @property
    def config(self) -> CharacterControllerConfig:
        """Controller configuration."""
        return self._config

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_collision_callback(
        self, callback: Optional[Callable[[ControllerCollision], None]]
    ) -> None:
        """Set callback for collision events."""
        self._on_collision = callback

    def set_ground_change_callback(
        self, callback: Optional[Callable[[bool], None]]
    ) -> None:
        """Set callback for ground state changes."""
        self._on_ground_change = callback

    # -------------------------------------------------------------------------
    # Movement
    # -------------------------------------------------------------------------

    def move(self, direction: Vector3, dt: float) -> Vector3:
        """
        Move the character controller.

        Args:
            direction: Movement direction (should be normalized for consistent speed)
            dt: Delta time in seconds

        Returns:
            The actual displacement that occurred.
        """
        if dt <= 0:
            return Vector3.zero()

        # Store original position
        original_position = Vector3(
            self._position.x, self._position.y, self._position.z
        )

        # Apply gravity
        if not self._is_grounded:
            self._velocity.y += self._config.gravity * dt
            self._velocity.y = max(self._velocity.y, -MAX_FALL_VELOCITY)

        # Calculate desired velocity
        horizontal_dir = direction.horizontal().normalized()

        if self._is_grounded:
            # Full control on ground
            desired_horizontal = horizontal_dir * direction.magnitude()
        else:
            # Limited air control
            air_control = self._config.air_control
            desired_horizontal = horizontal_dir * direction.magnitude() * air_control

        # Calculate total movement
        movement = (
            desired_horizontal * dt +
            Vector3(0.0, self._velocity.y * dt, 0.0) +
            self._external_velocity * dt
        )

        # Clear external velocity after use
        self._external_velocity = Vector3.zero()

        # Perform move and slide
        actual_movement = self._move_and_slide(movement, dt)

        # Update ground detection
        self._detect_ground()

        # Store last move direction
        if movement.magnitude_squared() > 0.001:
            self._last_move_direction = movement.normalized()

        return self._position - original_position

    def _move_and_slide(self, movement: Vector3, dt: float) -> Vector3:
        """
        Perform move-and-slide collision resolution.

        Args:
            movement: Desired movement vector
            dt: Delta time

        Returns:
            Actual movement achieved
        """
        if movement.magnitude() < self._config.min_move_distance:
            return Vector3.zero()

        remaining = Vector3(movement.x, movement.y, movement.z)
        total_moved = Vector3.zero()
        self._collision_flags = 0

        for iteration in range(MAX_COLLISION_ITERATIONS):
            if remaining.magnitude() < self._config.min_move_distance:
                break

            # Perform capsule sweep
            sweep = self._physics.capsule_sweep(
                start=self._position,
                end=self._position + remaining,
                radius=self._config.radius + self._config.skin_width,
                height=self._config.height,
                mask=self._config.collision_mask,
            )

            if not sweep.hit:
                # No collision, move full distance
                self._position = self._position + remaining
                total_moved = total_moved + remaining
                break

            # Handle penetration
            if sweep.start_penetrating:
                self._resolve_penetration(sweep)
                continue

            # Move to safe position
            safe_move = remaining * sweep.safe_fraction
            if safe_move.magnitude() >= self._config.min_move_distance:
                self._position = self._position + safe_move
                total_moved = total_moved + safe_move

            hit = sweep.first_hit
            if hit is None:
                break

            # Fire collision callback
            if self._on_collision:
                collision = ControllerCollision(
                    hit=hit,
                    move_direction=remaining.normalized(),
                    move_length=remaining.magnitude(),
                    controller_velocity=self._velocity,
                    other_velocity=self._physics.get_collider_velocity(hit.collider_id),
                )
                self._on_collision(collision)

            # Calculate remaining movement after slide
            remaining = remaining * (1.0 - sweep.safe_fraction)
            remaining = self._slide_along_surface(remaining, hit.normal)

            # Try step up for horizontal collisions
            if abs(hit.normal.y) < 0.1 and remaining.horizontal().magnitude() > 0.01:
                stepped = self._try_step_up(remaining, hit)
                if stepped:
                    break

        return total_moved

    def _slide_along_surface(
        self, velocity: Vector3, normal: Vector3
    ) -> Vector3:
        """
        Calculate slide velocity along a surface.

        Args:
            velocity: Input velocity
            normal: Surface normal

        Returns:
            Velocity after sliding along surface
        """
        # Project velocity onto surface plane
        dot = velocity.dot(normal)
        return velocity - normal * dot

    def _resolve_penetration(self, sweep: SweepResult) -> None:
        """Resolve penetration when starting inside geometry."""
        total_push = Vector3.zero()
        for hit in sweep.hits:
            if hit.penetration > 0:
                # Push out along normal with per-frame limit
                push_amount = min(
                    hit.penetration + self._config.skin_width,
                    MAX_DEPENETRATION_VELOCITY * 0.016,  # Limit per frame
                )
                push = hit.normal * push_amount
                total_push = total_push + push

        # Apply combined push, clamped to max depenetration
        push_magnitude = total_push.magnitude()
        if push_magnitude > MAX_DEPENETRATION_VELOCITY * 0.016:
            total_push = total_push.normalized() * (MAX_DEPENETRATION_VELOCITY * 0.016)
        self._position = self._position + total_push

    def _try_step_up(self, movement: Vector3, obstacle_hit: CollisionHit) -> bool:
        """
        Try to step up over a small obstacle.

        Args:
            movement: Remaining movement
            obstacle_hit: Hit information for the obstacle

        Returns:
            True if step up was successful
        """
        step_height = self._config.step_height

        # Cast upward to check clearance
        up_sweep = self._physics.capsule_sweep(
            start=self._position,
            end=self._position + Vector3(0.0, step_height, 0.0),
            radius=self._config.radius + self._config.skin_width,
            height=self._config.height,
            mask=self._config.collision_mask,
        )

        if up_sweep.blocked:
            return False

        # Move up
        up_position = self._position + Vector3(0.0, step_height * up_sweep.safe_fraction, 0.0)

        # Try to move forward
        horizontal = movement.horizontal()
        forward_sweep = self._physics.capsule_sweep(
            start=up_position,
            end=up_position + horizontal,
            radius=self._config.radius + self._config.skin_width,
            height=self._config.height,
            mask=self._config.collision_mask,
        )

        if forward_sweep.blocked:
            return False

        forward_position = up_position + horizontal * forward_sweep.safe_fraction

        # Cast down to find step surface
        down_sweep = self._physics.capsule_sweep(
            start=forward_position,
            end=forward_position + Vector3(0.0, -step_height * 2.0, 0.0),
            radius=self._config.radius + self._config.skin_width,
            height=self._config.height,
            mask=self._config.collision_mask,
        )

        if not down_sweep.hit or down_sweep.first_hit is None:
            return False

        # Check if the step is walkable
        step_normal = down_sweep.first_hit.normal
        slope_angle = math.degrees(math.acos(max(0.0, min(1.0, step_normal.y))))
        if slope_angle > self._config.slope_limit:
            return False

        # Apply the step
        step_down = step_height * 2.0 * down_sweep.safe_fraction
        self._position = forward_position + Vector3(0.0, -step_down, 0.0)
        return True

    # -------------------------------------------------------------------------
    # Ground Detection
    # -------------------------------------------------------------------------

    def _detect_ground(self) -> None:
        """Detect and update ground state."""
        was_grounded = self._is_grounded

        # Sphere sweep downward
        probe_start = Vector3(
            self._position.x,
            self._position.y + self._config.radius,
            self._position.z,
        )
        probe_end = Vector3(
            self._position.x,
            self._position.y - GROUND_PROBE_DISTANCE,
            self._position.z,
        )

        sweep = self._physics.sphere_sweep(
            start=probe_start,
            end=probe_end,
            radius=self._config.radius * 0.9,  # Slightly smaller to avoid edge cases
            mask=self._config.collision_mask,
        )

        if sweep.hit and sweep.first_hit is not None:
            hit = sweep.first_hit

            # Check slope angle
            slope_angle = math.degrees(math.acos(max(0.0, min(1.0, hit.normal.y))))
            if slope_angle <= self._config.slope_limit:
                self._is_grounded = True
                self._ground_normal = hit.normal
                self._ground_distance = hit.distance
                self._ground_material = hit.material
                self._ground_collider_id = hit.collider_id

                # Reset vertical velocity when landing
                if not was_grounded and self._velocity.y < 0:
                    self._velocity.y = 0.0
            else:
                self._is_grounded = False
                self._ground_normal = Vector3.up()
        else:
            self._is_grounded = False
            self._ground_normal = Vector3.up()
            self._ground_distance = float("inf")

        # Fire callback on state change
        if was_grounded != self._is_grounded and self._on_ground_change:
            self._on_ground_change(self._is_grounded)

    # -------------------------------------------------------------------------
    # Jump
    # -------------------------------------------------------------------------

    def jump(self, velocity: Optional[float] = None) -> bool:
        """
        Make the character jump.

        Args:
            velocity: Optional jump velocity override

        Returns:
            True if jump was initiated
        """
        if not self._is_grounded:
            return False

        jump_vel = velocity if velocity is not None else self._config.jump_velocity
        self._velocity.y = jump_vel
        self._is_grounded = False

        # Fire ground change callback
        if self._on_ground_change:
            self._on_ground_change(False)

        return True

    def add_impulse(self, impulse: Vector3) -> None:
        """
        Add an instantaneous impulse to the character.

        Args:
            impulse: Impulse vector
        """
        self._velocity = self._velocity + impulse

    def add_force(self, force: Vector3, dt: float) -> None:
        """
        Add a force to the character over time.

        Args:
            force: Force vector
            dt: Delta time
        """
        self._velocity = self._velocity + force * dt

    def set_external_velocity(self, velocity: Vector3) -> None:
        """
        Set external velocity (e.g., from moving platform).

        Args:
            velocity: External velocity to add to movement
        """
        self._external_velocity = velocity

    # -------------------------------------------------------------------------
    # Platform Attachment
    # -------------------------------------------------------------------------

    def attach_to_platform(self, platform_id: int, offset: Vector3) -> None:
        """
        Attach to a moving platform.

        Args:
            platform_id: ID of the platform collider
            offset: Local offset on the platform
        """
        self._attached_platform_id = platform_id
        self._platform_offset = offset

    def detach_from_platform(self) -> None:
        """Detach from current platform."""
        self._attached_platform_id = None
        self._platform_offset = Vector3.zero()

    @property
    def attached_platform_id(self) -> Optional[int]:
        """ID of the currently attached platform."""
        return self._attached_platform_id

    # -------------------------------------------------------------------------
    # Shape Modification
    # -------------------------------------------------------------------------

    def resize(self, height: float, radius: Optional[float] = None) -> bool:
        """
        Resize the controller capsule.

        Args:
            height: New height
            radius: New radius (optional)

        Returns:
            True if resize was successful (no collision at new size)
        """
        new_radius = radius if radius is not None else self._config.radius

        # Check if new size overlaps geometry
        overlaps = self._physics.overlap_capsule(
            position=self._position,
            radius=new_radius + self._config.skin_width,
            height=height,
            mask=self._config.collision_mask,
        )

        if overlaps:
            return False

        self._config.height = height
        self._config.radius = new_radius
        return True

    # -------------------------------------------------------------------------
    # Teleportation
    # -------------------------------------------------------------------------

    def teleport(self, position: Vector3, rotation: Optional[Quaternion] = None) -> None:
        """
        Teleport the controller to a new position.

        Args:
            position: New position
            rotation: New rotation (optional)
        """
        self._position = position
        if rotation is not None:
            self._rotation = rotation
        self._velocity = Vector3.zero()
        self._external_velocity = Vector3.zero()
        self._detect_ground()

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def get_debug_info(self) -> dict[str, Any]:
        """Get debug information about controller state."""
        return {
            "position": (self._position.x, self._position.y, self._position.z),
            "velocity": (self._velocity.x, self._velocity.y, self._velocity.z),
            "is_grounded": self._is_grounded,
            "ground_normal": (
                self._ground_normal.x,
                self._ground_normal.y,
                self._ground_normal.z,
            ),
            "ground_material": self._ground_material,
            "collision_flags": self._collision_flags,
            "attached_platform": self._attached_platform_id,
            "config": {
                "radius": self._config.radius,
                "height": self._config.height,
                "step_height": self._config.step_height,
                "slope_limit": self._config.slope_limit,
            },
        }
