"""
Spring/Jiggle Physics for Bones.

Implements damped spring physics for secondary bone motion such as hair,
tails, cloth, and accessories. Uses Verlet integration for numerical stability.

Physics Formula: F = -kx - cv (spring force with damping)
Verlet Integration: x_new = 2*x - x_old + a*dt^2

Usage:
    spring = SpringBone(bone_index=5, stiffness=50.0, damping=0.3)
    chain = SpringChain(root_bone=5, bone_indices=[5, 6, 7, 8])
    modified_pose = chain.simulate(pose, dt=1/60)

Note on Numerical Stability:
    Large timesteps (dt > 0.033s) can cause spring physics instability.
    The implementation clamps dt to MAX_PHYSICS_DT for stability.
    For best results, use fixed timesteps around 1/60 second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Protocol, Union

from engine.animation.procedural.config import ProceduralConfig

# Type aliases for clarity
Vec3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # (x, y, z, w)


class Pose(Protocol):
    """Protocol for pose data - maps bone indices to transforms."""

    def get_bone_position(self, bone_index: int) -> Vec3:
        """Get world position of a bone."""
        ...

    def set_bone_position(self, bone_index: int, position: Vec3) -> None:
        """Set world position of a bone."""
        ...

    def get_bone_rotation(self, bone_index: int) -> Quaternion:
        """Get world rotation of a bone."""
        ...

    def set_bone_rotation(self, bone_index: int, rotation: Quaternion) -> None:
        """Set world rotation of a bone."""
        ...

    def get_parent_index(self, bone_index: int) -> int:
        """Get parent bone index, -1 for root."""
        ...

    def copy(self) -> "Pose":
        """Create a copy of this pose."""
        ...


def vec3_add(a: Vec3, b: Vec3) -> Vec3:
    """Add two vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_sub(a: Vec3, b: Vec3) -> Vec3:
    """Subtract two vectors."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_scale(v: Vec3, s: float) -> Vec3:
    """Scale a vector."""
    return (v[0] * s, v[1] * s, v[2] * s)


def vec3_length(v: Vec3) -> float:
    """Get vector length."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec3_normalize(v: Vec3) -> Vec3:
    """Normalize a vector."""
    length = vec3_length(v)
    if length < 1e-10:
        return (0.0, 0.0, 0.0)
    inv_length = 1.0 / length
    return (v[0] * inv_length, v[1] * inv_length, v[2] * inv_length)


def vec3_dot(a: Vec3, b: Vec3) -> float:
    """Dot product of two vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    """Linear interpolation between two vectors."""
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def vec3_distance(a: Vec3, b: Vec3) -> float:
    """Distance between two points."""
    return vec3_length(vec3_sub(b, a))


@dataclass
class CollisionSphere:
    """Spherical collision primitive for spring bones."""

    center: Vec3
    radius: float
    bone_index: int = -1  # If attached to a bone, updates with skeleton

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("Collision sphere radius must be > 0")

    def get_center(self, pose: Optional[Pose] = None) -> Vec3:
        """Get world-space center, optionally following a bone."""
        if self.bone_index >= 0 and pose is not None:
            bone_pos = pose.get_bone_position(self.bone_index)
            return vec3_add(bone_pos, self.center)
        return self.center

    def resolve_collision(
        self, position: Vec3, pose: Optional[Pose] = None
    ) -> Tuple[Vec3, bool]:
        """
        Resolve collision with a point.
        Returns: (new_position, did_collide)
        """
        center = self.get_center(pose)
        to_point = vec3_sub(position, center)
        distance = vec3_length(to_point)

        if distance < self.radius:
            if distance < 1e-10:
                # Point at center, push along arbitrary axis
                return (vec3_add(center, (self.radius, 0.0, 0.0)), True)

            direction = vec3_normalize(to_point)
            new_pos = vec3_add(center, vec3_scale(direction, self.radius))
            return (new_pos, True)

        return (position, False)


@dataclass
class CollisionCapsule:
    """Capsule collision primitive for spring bones."""

    start: Vec3
    end: Vec3
    radius: float
    bone_index: int = -1  # If attached to a bone, updates with skeleton

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("Collision capsule radius must be > 0")

    def get_endpoints(self, pose: Optional[Pose] = None) -> Tuple[Vec3, Vec3]:
        """Get world-space endpoints, optionally following a bone."""
        if self.bone_index >= 0 and pose is not None:
            bone_pos = pose.get_bone_position(self.bone_index)
            return (
                vec3_add(bone_pos, self.start),
                vec3_add(bone_pos, self.end),
            )
        return (self.start, self.end)

    def _closest_point_on_segment(
        self, point: Vec3, seg_start: Vec3, seg_end: Vec3
    ) -> Vec3:
        """Find closest point on line segment to a point."""
        segment = vec3_sub(seg_end, seg_start)
        segment_length_sq = vec3_dot(segment, segment)

        if segment_length_sq < 1e-10:
            return seg_start

        to_point = vec3_sub(point, seg_start)
        t = vec3_dot(to_point, segment) / segment_length_sq
        t = max(0.0, min(1.0, t))

        return vec3_add(seg_start, vec3_scale(segment, t))

    def resolve_collision(
        self, position: Vec3, pose: Optional[Pose] = None
    ) -> Tuple[Vec3, bool]:
        """
        Resolve collision with a point.
        Returns: (new_position, did_collide)
        """
        start, end = self.get_endpoints(pose)
        closest = self._closest_point_on_segment(position, start, end)
        to_point = vec3_sub(position, closest)
        distance = vec3_length(to_point)

        if distance < self.radius:
            if distance < 1e-10:
                # Point on axis, push along perpendicular
                segment = vec3_sub(end, start)
                perp = (1.0, 0.0, 0.0) if abs(segment[0]) < 0.9 else (0.0, 1.0, 0.0)
                return (vec3_add(closest, vec3_scale(perp, self.radius)), True)

            direction = vec3_normalize(to_point)
            new_pos = vec3_add(closest, vec3_scale(direction, self.radius))
            return (new_pos, True)

        return (position, False)


@dataclass
class WindForce:
    """Wind influence on spring bones."""

    direction: Vec3 = (1.0, 0.0, 0.0)
    strength: float = 1.0
    turbulence: float = 0.0  # 0-1, adds noise to wind
    frequency: float = 1.0  # Turbulence oscillation frequency
    _time: float = field(default=0.0, repr=False)

    def __post_init__(self):
        if self.strength < 0:
            raise ValueError("Wind strength must be >= 0")
        if not (0.0 <= self.turbulence <= 1.0):
            raise ValueError("Wind turbulence must be in [0, 1]")

    def get_force(self, position: Vec3, dt: float) -> Vec3:
        """Calculate wind force at a position with optional turbulence."""
        self._time += dt

        base_force = vec3_scale(vec3_normalize(self.direction), self.strength)

        if self.turbulence > 0:
            # Simple sine-based turbulence
            phase = self._time * self.frequency * 2 * math.pi
            noise_x = math.sin(phase + position[0]) * self.turbulence
            noise_y = math.sin(phase * 1.3 + position[1]) * self.turbulence * 0.5
            noise_z = math.sin(phase * 0.7 + position[2]) * self.turbulence

            noise_force = vec3_scale((noise_x, noise_y, noise_z), self.strength)
            base_force = vec3_add(base_force, noise_force)

        return base_force

    def reset(self) -> None:
        """Reset internal time accumulator."""
        self._time = 0.0


@dataclass
class SpringBone:
    """
    Single spring bone with physics simulation.

    Uses Verlet integration for stability:
    - position_new = 2 * position - position_old + acceleration * dt^2

    Spring force: F = -k*x - c*v
    where k is stiffness, c is damping, x is displacement, v is velocity
    """

    bone_index: int
    stiffness: float = 50.0  # Spring constant (k)
    damping: float = 0.3  # Damping coefficient (0-1)
    gravity: Vec3 = (0.0, -9.81, 0.0)
    mass: float = 1.0

    # Internal state for Verlet integration
    _position: Vec3 = field(default=(0.0, 0.0, 0.0), repr=False)
    _previous_position: Vec3 = field(default=(0.0, 0.0, 0.0), repr=False)
    _rest_position: Vec3 = field(default=(0.0, 0.0, 0.0), repr=False)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.bone_index < 0:
            raise ValueError("bone_index must be >= 0")
        if self.stiffness < 0:
            raise ValueError("stiffness must be >= 0")
        if not (0.0 <= self.damping <= 1.0):
            raise ValueError("damping must be in [0, 1]")
        if self.mass <= 0:
            raise ValueError("mass must be > 0")

    def initialize(self, pose: Pose) -> None:
        """Initialize spring bone from current pose."""
        self._position = pose.get_bone_position(self.bone_index)
        self._previous_position = self._position
        self._rest_position = self._position
        self._initialized = True

    def simulate(
        self,
        pose: Pose,
        dt: float,
        colliders: Optional[List[Union[CollisionSphere, CollisionCapsule]]] = None,
        wind: Optional[WindForce] = None,
    ) -> Pose:
        """
        Simulate spring physics and update pose.

        Args:
            pose: Current animation pose
            dt: Time step in seconds (clamped to MAX_PHYSICS_DT for stability)
            colliders: Optional collision primitives
            wind: Optional wind force

        Returns:
            Modified pose with spring physics applied

        Note:
            Large dt values can cause numerical instability in spring physics.
            This method clamps dt to ProceduralConfig.MAX_PHYSICS_DT (0.033s).
            For smooth simulation with large gaps, call multiple times with
            smaller substeps.
        """
        if dt <= 0:
            return pose

        # Clamp dt for numerical stability - large dt causes spring explosion
        dt = min(dt, ProceduralConfig.MAX_PHYSICS_DT)

        if not self._initialized:
            self.initialize(pose)
            return pose

        # Get rest position from animation
        rest_pos = pose.get_bone_position(self.bone_index)

        # Calculate spring force: F = -k * (current - rest)
        displacement = vec3_sub(self._position, rest_pos)
        spring_force = vec3_scale(displacement, -self.stiffness)

        # Calculate velocity from Verlet (implicit)
        velocity = vec3_scale(
            vec3_sub(self._position, self._previous_position),
            1.0 / dt if dt > 1e-10 else 0.0
        )

        # Damping force: F = -c * v
        damping_force = vec3_scale(velocity, -self.damping * self.stiffness)

        # Gravity force
        gravity_force = vec3_scale(self.gravity, self.mass)

        # Wind force
        wind_force = (0.0, 0.0, 0.0)
        if wind is not None:
            wind_force = wind.get_force(self._position, dt)

        # Total force and acceleration
        total_force = vec3_add(
            vec3_add(spring_force, damping_force),
            vec3_add(gravity_force, wind_force)
        )
        acceleration = vec3_scale(total_force, 1.0 / self.mass)

        # Verlet integration: x_new = 2*x - x_old + a*dt^2
        dt_sq = dt * dt
        new_position = vec3_add(
            vec3_sub(vec3_scale(self._position, 2.0), self._previous_position),
            vec3_scale(acceleration, dt_sq)
        )

        # Apply collision detection
        if colliders:
            for collider in colliders:
                new_position, _ = collider.resolve_collision(new_position, pose)

        # Update state
        self._previous_position = self._position
        self._position = new_position

        # Update pose
        result = pose.copy()
        result.set_bone_position(self.bone_index, new_position)

        return result

    def reset(self, pose: Pose) -> None:
        """Reset spring to current animation pose."""
        self._initialized = False
        self.initialize(pose)

    def get_position(self) -> Vec3:
        """Get current simulated position."""
        return self._position

    def get_velocity(self, dt: float) -> Vec3:
        """Get current velocity estimate."""
        if dt <= 1e-10:
            return (0.0, 0.0, 0.0)
        return vec3_scale(
            vec3_sub(self._position, self._previous_position),
            1.0 / dt
        )


@dataclass
class SpringChain:
    """
    Chain of spring bones for tails, hair, cloth simulation.

    Bones are connected in sequence, with constraints maintaining
    their original distances.
    """

    root_bone: int
    bone_indices: List[int]
    stiffness: float = 50.0
    damping: float = 0.3
    gravity: Vec3 = (0.0, -9.81, 0.0)
    mass_per_bone: float = 1.0
    constraint_iterations: int = 3  # Distance constraint solver iterations

    _springs: List[SpringBone] = field(default_factory=list, repr=False)
    _rest_lengths: List[float] = field(default_factory=list, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.root_bone < 0:
            raise ValueError("root_bone must be >= 0")
        if not self.bone_indices:
            raise ValueError("bone_indices must not be empty")
        if self.stiffness < 0:
            raise ValueError("stiffness must be >= 0")
        if not (0.0 <= self.damping <= 1.0):
            raise ValueError("damping must be in [0, 1]")
        if self.constraint_iterations < 1:
            raise ValueError("constraint_iterations must be >= 1")

        # Create spring bones (but don't initialize until we have a pose)
        self._springs = []
        for i, bone_idx in enumerate(self.bone_indices):
            spring = SpringBone(
                bone_index=bone_idx,
                stiffness=self.stiffness,
                damping=self.damping,
                gravity=self.gravity,
                mass=self.mass_per_bone,
            )
            self._springs.append(spring)

    def initialize(self, pose: Pose) -> None:
        """Initialize chain from current pose."""
        # Initialize all springs
        for spring in self._springs:
            spring.initialize(pose)

        # Calculate rest lengths between consecutive bones
        self._rest_lengths = []

        # First segment: root to first bone
        root_pos = pose.get_bone_position(self.root_bone)
        first_pos = pose.get_bone_position(self.bone_indices[0])
        self._rest_lengths.append(vec3_distance(root_pos, first_pos))

        # Remaining segments
        for i in range(1, len(self.bone_indices)):
            prev_pos = pose.get_bone_position(self.bone_indices[i - 1])
            curr_pos = pose.get_bone_position(self.bone_indices[i])
            self._rest_lengths.append(vec3_distance(prev_pos, curr_pos))

        self._initialized = True

    def _apply_distance_constraints(self, pose: Pose) -> None:
        """Apply distance constraints to maintain bone lengths."""
        for _ in range(self.constraint_iterations):
            # Constrain first bone to root
            root_pos = pose.get_bone_position(self.root_bone)
            first_spring = self._springs[0]

            direction = vec3_sub(first_spring._position, root_pos)
            distance = vec3_length(direction)

            if distance > 1e-10:
                direction = vec3_normalize(direction)
                target = vec3_add(root_pos, vec3_scale(direction, self._rest_lengths[0]))

                # Move towards constraint (soft constraint)
                first_spring._position = vec3_lerp(
                    first_spring._position, target, 0.5
                )

            # Constrain consecutive bones
            for i in range(1, len(self._springs)):
                prev_spring = self._springs[i - 1]
                curr_spring = self._springs[i]

                direction = vec3_sub(curr_spring._position, prev_spring._position)
                distance = vec3_length(direction)

                if distance > 1e-10:
                    direction = vec3_normalize(direction)
                    rest_length = self._rest_lengths[i]

                    # Calculate correction
                    correction = (distance - rest_length) / 2.0

                    # Apply correction (both bones move)
                    prev_spring._position = vec3_add(
                        prev_spring._position,
                        vec3_scale(direction, correction)
                    )
                    curr_spring._position = vec3_sub(
                        curr_spring._position,
                        vec3_scale(direction, correction)
                    )

    def simulate(
        self,
        pose: Pose,
        dt: float,
        colliders: Optional[List[Union[CollisionSphere, CollisionCapsule]]] = None,
        wind: Optional[WindForce] = None,
    ) -> Pose:
        """
        Simulate the entire spring chain.

        Args:
            pose: Current animation pose
            dt: Time step in seconds (clamped to MAX_PHYSICS_DT for stability)
            colliders: Optional collision primitives
            wind: Optional wind force

        Returns:
            Modified pose with spring chain physics applied

        Note:
            Large dt values can cause numerical instability in spring physics.
            This method clamps dt to ProceduralConfig.MAX_PHYSICS_DT (0.033s).
        """
        if dt <= 0:
            return pose

        # Clamp dt for numerical stability - large dt causes spring explosion
        dt = min(dt, ProceduralConfig.MAX_PHYSICS_DT)

        if not self._initialized:
            self.initialize(pose)
            return pose

        result = pose.copy()

        # Simulate each spring bone
        for spring in self._springs:
            # Get rest position from animation for this bone
            rest_pos = pose.get_bone_position(spring.bone_index)
            spring._rest_position = rest_pos

            # Apply forces (without updating pose yet)
            displacement = vec3_sub(spring._position, rest_pos)
            spring_force = vec3_scale(displacement, -spring.stiffness)

            velocity = vec3_scale(
                vec3_sub(spring._position, spring._previous_position),
                1.0 / dt if dt > 1e-10 else 0.0
            )
            damping_force = vec3_scale(velocity, -spring.damping * spring.stiffness)
            gravity_force = vec3_scale(spring.gravity, spring.mass)

            wind_force = (0.0, 0.0, 0.0)
            if wind is not None:
                wind_force = wind.get_force(spring._position, dt)

            total_force = vec3_add(
                vec3_add(spring_force, damping_force),
                vec3_add(gravity_force, wind_force)
            )
            acceleration = vec3_scale(total_force, 1.0 / spring.mass)

            # Verlet integration
            dt_sq = dt * dt
            new_position = vec3_add(
                vec3_sub(vec3_scale(spring._position, 2.0), spring._previous_position),
                vec3_scale(acceleration, dt_sq)
            )

            spring._previous_position = spring._position
            spring._position = new_position

        # Apply distance constraints
        self._apply_distance_constraints(pose)

        # Apply collision detection
        if colliders:
            for spring in self._springs:
                for collider in colliders:
                    spring._position, _ = collider.resolve_collision(spring._position, pose)

        # Update pose with final positions
        for spring in self._springs:
            result.set_bone_position(spring.bone_index, spring._position)

        return result

    def reset(self, pose: Pose) -> None:
        """Reset all springs in the chain."""
        self._initialized = False
        self.initialize(pose)

    def get_bone_count(self) -> int:
        """Get number of bones in the chain."""
        return len(self.bone_indices)

    def get_spring(self, index: int) -> SpringBone:
        """Get spring bone by index in chain."""
        return self._springs[index]


def procedural_bone(type: str):
    """
    Decorator to mark a bone for procedural animation.

    Args:
        type: Procedural behavior type ("jiggle", "spring", "lookat", "aim", "twist")

    Usage:
        @procedural_bone(type="spring")
        class HairBone:
            stiffness: float = 50.0
            damping: float = 0.3
    """
    VALID_TYPES = {"jiggle", "spring", "lookat", "aim", "twist"}

    if type not in VALID_TYPES:
        raise ValueError(f"Invalid type '{type}', must be one of: {VALID_TYPES}")

    def decorator(cls):
        cls._procedural_bone = True
        cls._procedural_bone_type = type

        # Initialize tags dict if not present
        if not hasattr(cls, "_tags"):
            cls._tags = {}
        cls._tags["procedural_bone"] = True
        cls._tags["procedural_bone_type"] = type

        # Track applied decorators
        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = set()
        cls._applied_decorators.add("procedural_bone")

        return cls

    return decorator
