"""
Modular Particle Behavior System.

Provides composable modules for particle spawn, update, and render phases.
Each module encapsulates a specific behavior that can be combined into particle systems.

Module Categories:
    SpawnModules - Control where/how particles are born
    UpdateModules - Apply forces, modify attributes over time
    RenderModules - Control how particles are drawn

Supports @particle_module decorator with stage and lod_range configuration.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, Union

from engine.rendering.particles.particle_system import (
    Particle,
    ParticleState,
    Vec3,
    Vec4,
)
from engine.rendering.particles.constants import PARTICLE_CONSTANTS

# Default LOD range from constants
DEFAULT_LOD_RANGE = (PARTICLE_CONSTANTS.DEFAULT_LOD_MIN, PARTICLE_CONSTANTS.DEFAULT_LOD_MAX)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class ModuleStage(Enum):
    """Particle module execution stage."""

    SPAWN = auto()  # Called when particle is created
    UPDATE = auto()  # Called every frame for alive particles
    RENDER = auto()  # Called during rendering preparation


class EmitterShape(Enum):
    """Shape for particle emission."""

    POINT = auto()  # Single point in space
    SPHERE = auto()  # Sphere surface or volume
    BOX = auto()  # Axis-aligned box
    CONE = auto()  # Cone surface or volume
    CIRCLE = auto()  # Circle (2D disk)
    EDGE = auto()  # Line segment
    MESH_SURFACE = auto()  # Mesh surface triangles
    MESH_VOLUME = auto()  # Mesh interior volume


class CollisionMode(Enum):
    """Particle collision handling mode."""

    NONE = auto()  # No collision
    DEPTH_BUFFER = auto()  # Collision against depth buffer
    SDF = auto()  # Signed distance field collision
    PRIMITIVE = auto()  # Analytical primitive collision


class BlendMode(Enum):
    """Particle color blending mode."""

    REPLACE = auto()  # Replace color
    MULTIPLY = auto()  # Multiply colors
    ADD = auto()  # Additive blend
    SUBTRACT = auto()  # Subtractive blend


# =============================================================================
# MODULE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ModuleConfig:
    """
    Configuration for a particle module from @particle_module decorator.

    Attributes:
        stage: When the module executes (spawn/update/render)
        lod_range: LOD levels where module is active (min_lod, max_lod)
    """

    stage: ModuleStage = ModuleStage.UPDATE
    lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE

    @classmethod
    def from_decorator_params(
        cls,
        stage: str,
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
        **kwargs: Any,
    ) -> "ModuleConfig":
        """Create config from @particle_module decorator parameters."""
        stage_map = {
            "spawn": ModuleStage.SPAWN,
            "update": ModuleStage.UPDATE,
            "render": ModuleStage.RENDER,
        }
        return cls(
            stage=stage_map.get(stage.lower(), ModuleStage.UPDATE),
            lod_range=lod_range,
        )


# =============================================================================
# BASE MODULE CLASS
# =============================================================================


class ParticleModule(ABC):
    """
    Base class for all particle modules.

    Modules are composable behaviors that can be applied during
    spawn, update, or render phases of the particle lifecycle.
    """

    def __init__(
        self,
        stage: ModuleStage = ModuleStage.UPDATE,
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
        enabled: bool = True,
    ) -> None:
        self._stage = stage
        self._lod_range = lod_range
        self._enabled = enabled

    @property
    def stage(self) -> ModuleStage:
        return self._stage

    @property
    def lod_range(self) -> Tuple[int, int]:
        return self._lod_range

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def is_active_for_lod(self, lod_level: int) -> bool:
        """Check if module is active for the given LOD level."""
        return (
            self._enabled
            and self._lod_range[0] <= lod_level <= self._lod_range[1]
        )

    def get_spawn_count(self, dt: float) -> int:
        """
        Get number of particles to spawn this frame.

        Only meaningful for spawn-stage modules that control emission rate.
        """
        return 0

    @abstractmethod
    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        """Apply module behavior to a particle."""
        pass


# =============================================================================
# SPAWN MODULES
# =============================================================================


class ShapeEmitter(ParticleModule):
    """
    Emit particles from a geometric shape.

    Supports point, sphere, box, cone, circle, and edge shapes.
    """

    def __init__(
        self,
        shape: EmitterShape = EmitterShape.POINT,
        position: Vec3 = None,
        size: Vec3 = None,
        radius: float = 1.0,
        angle: float = 30.0,  # Cone angle in degrees
        emit_from_surface: bool = True,
        randomize_direction: bool = True,
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.SPAWN, lod_range)
        self._shape = shape
        self._position = position or Vec3()
        self._size = size or Vec3(1, 1, 1)
        self._radius = radius
        self._angle = math.radians(angle)
        self._emit_from_surface = emit_from_surface
        self._randomize_direction = randomize_direction

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        """Set particle position and initial velocity based on shape."""
        pos, vel = self._sample_shape()
        particle.position = self._position + pos
        if self._randomize_direction:
            particle.velocity = vel

    def _sample_shape(self) -> Tuple[Vec3, Vec3]:
        """Sample a position and direction from the shape."""
        if self._shape == EmitterShape.POINT:
            return Vec3(), Vec3(0, 1, 0)

        elif self._shape == EmitterShape.SPHERE:
            return self._sample_sphere()

        elif self._shape == EmitterShape.BOX:
            return self._sample_box()

        elif self._shape == EmitterShape.CONE:
            return self._sample_cone()

        elif self._shape == EmitterShape.CIRCLE:
            return self._sample_circle()

        elif self._shape == EmitterShape.EDGE:
            return self._sample_edge()

        return Vec3(), Vec3(0, 1, 0)

    def _sample_sphere(self) -> Tuple[Vec3, Vec3]:
        """Sample from sphere surface or volume."""
        # Random direction
        theta = random.uniform(0, 2 * math.pi)
        phi = math.acos(random.uniform(-1, 1))

        dir_x = math.sin(phi) * math.cos(theta)
        dir_y = math.sin(phi) * math.sin(theta)
        dir_z = math.cos(phi)
        direction = Vec3(dir_x, dir_y, dir_z)

        if self._emit_from_surface:
            r = self._radius
        else:
            r = self._radius * (random.random() ** (1 / 3))

        position = direction * r
        return position, direction

    def _sample_box(self) -> Tuple[Vec3, Vec3]:
        """Sample from box surface or volume."""
        if self._emit_from_surface:
            # Choose a random face
            face = random.randint(0, 5)
            x = random.uniform(-self._size.x / 2, self._size.x / 2)
            y = random.uniform(-self._size.y / 2, self._size.y / 2)
            z = random.uniform(-self._size.z / 2, self._size.z / 2)

            if face == 0:  # +X
                x = self._size.x / 2
                direction = Vec3(1, 0, 0)
            elif face == 1:  # -X
                x = -self._size.x / 2
                direction = Vec3(-1, 0, 0)
            elif face == 2:  # +Y
                y = self._size.y / 2
                direction = Vec3(0, 1, 0)
            elif face == 3:  # -Y
                y = -self._size.y / 2
                direction = Vec3(0, -1, 0)
            elif face == 4:  # +Z
                z = self._size.z / 2
                direction = Vec3(0, 0, 1)
            else:  # -Z
                z = -self._size.z / 2
                direction = Vec3(0, 0, -1)

            return Vec3(x, y, z), direction
        else:
            x = random.uniform(-self._size.x / 2, self._size.x / 2)
            y = random.uniform(-self._size.y / 2, self._size.y / 2)
            z = random.uniform(-self._size.z / 2, self._size.z / 2)
            return Vec3(x, y, z), Vec3(0, 1, 0)

    def _sample_cone(self) -> Tuple[Vec3, Vec3]:
        """Sample from cone."""
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, self._angle)

        dir_x = math.sin(phi) * math.cos(theta)
        dir_y = math.cos(phi)
        dir_z = math.sin(phi) * math.sin(theta)
        direction = Vec3(dir_x, dir_y, dir_z)

        return Vec3(), direction

    def _sample_circle(self) -> Tuple[Vec3, Vec3]:
        """Sample from circle (2D disk)."""
        theta = random.uniform(0, 2 * math.pi)

        if self._emit_from_surface:
            r = self._radius
        else:
            r = self._radius * math.sqrt(random.random())

        x = r * math.cos(theta)
        z = r * math.sin(theta)
        return Vec3(x, 0, z), Vec3(0, 1, 0)

    def _sample_edge(self) -> Tuple[Vec3, Vec3]:
        """Sample from line edge."""
        t = random.random()
        x = self._size.x * (t - 0.5)
        return Vec3(x, 0, 0), Vec3(0, 1, 0)


class BurstEmitter(ParticleModule):
    """
    Emit a burst of particles instantly.

    Spawns a fixed count of particles in a single frame.
    """

    def __init__(
        self,
        count: int = 10,
        repeat_interval: float = 0.0,  # 0 = single burst
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.SPAWN, lod_range)
        self._count = count
        self._repeat_interval = repeat_interval
        self._pending_count = count
        self._time_since_burst = 0.0

    def get_spawn_count(self, dt: float) -> int:
        """Return burst count when triggered."""
        if self._pending_count > 0:
            count = self._pending_count
            self._pending_count = 0
            return count

        if self._repeat_interval > 0:
            self._time_since_burst += dt
            if self._time_since_burst >= self._repeat_interval:
                self._time_since_burst = 0.0
                return self._count

        return 0

    def trigger(self) -> None:
        """Manually trigger a burst."""
        self._pending_count = self._count

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        """Burst emitter doesn't modify particles directly."""
        pass


class RateEmitter(ParticleModule):
    """
    Emit particles at a constant rate (particles per second).
    """

    def __init__(
        self,
        rate: float = 100.0,  # Particles per second
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.SPAWN, lod_range)
        self._rate = rate
        self._accumulator = 0.0

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        self._rate = max(0, value)

    def get_spawn_count(self, dt: float) -> int:
        """Calculate particles to spawn this frame based on rate."""
        self._accumulator += self._rate * dt
        count = int(self._accumulator)
        self._accumulator -= count
        return count

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        """Rate emitter doesn't modify particles directly."""
        pass


# =============================================================================
# UPDATE MODULES - FORCES
# =============================================================================


class GravityModule(ParticleModule):
    """Apply gravity acceleration to particles."""

    def __init__(
        self,
        gravity: Vec3 = None,
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.UPDATE, lod_range)
        self._gravity = gravity or Vec3(0, PARTICLE_CONSTANTS.DEFAULT_GRAVITY_Y, 0)

    @property
    def gravity(self) -> Vec3:
        return self._gravity

    @gravity.setter
    def gravity(self, value: Vec3) -> None:
        self._gravity = value

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        particle.acceleration = particle.acceleration + self._gravity


class WindModule(ParticleModule):
    """Apply wind force to particles."""

    def __init__(
        self,
        direction: Vec3 = None,
        strength: float = 1.0,
        turbulence: float = 0.0,
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.UPDATE, lod_range)
        self._direction = (direction or Vec3(1, 0, 0)).normalized()
        self._strength = strength
        self._turbulence = turbulence

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        # Base wind force
        force = self._direction * self._strength

        # Add turbulence
        if self._turbulence > 0:
            turb = Vec3(
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1),
            )
            force = force + turb * self._turbulence

        particle.acceleration = particle.acceleration + force


class TurbulenceModule(ParticleModule):
    """Apply noise-based turbulence force."""

    def __init__(
        self,
        strength: float = 1.0,
        frequency: float = 1.0,
        octaves: int = 3,
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.UPDATE, lod_range)
        self._strength = strength
        self._frequency = frequency
        self._octaves = octaves

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        # Simple pseudo-noise based on position
        # In real implementation, use Perlin/Simplex noise
        t = particle.age * self._frequency
        px = particle.position.x * self._frequency
        py = particle.position.y * self._frequency
        pz = particle.position.z * self._frequency

        fx = math.sin(t + px) * math.cos(py * 0.7) * self._strength
        fy = math.sin(t + py * 1.3) * math.cos(pz * 0.5) * self._strength
        fz = math.sin(t + pz * 0.9) * math.cos(px * 1.1) * self._strength

        particle.acceleration = particle.acceleration + Vec3(fx, fy, fz)


class VortexModule(ParticleModule):
    """Apply vortex/swirl force around an axis."""

    def __init__(
        self,
        center: Vec3 = None,
        axis: Vec3 = None,
        strength: float = 1.0,
        pull_strength: float = 0.0,  # Inward pull
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.UPDATE, lod_range)
        self._center = center or Vec3()
        self._axis = (axis or Vec3(0, 1, 0)).normalized()
        self._strength = strength
        self._pull_strength = pull_strength

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        # Vector from center to particle
        to_particle = particle.position - self._center

        # Project onto plane perpendicular to axis
        axis_component = self._axis * to_particle.dot(self._axis)
        radial = to_particle - axis_component

        distance = radial.length()
        if distance < 0.001:
            return

        # Tangent direction (perpendicular to radial, in the plane)
        tangent = self._axis.cross(radial.normalized())

        # Apply tangential force (swirl)
        force = tangent * self._strength

        # Apply radial force (pull toward center)
        if self._pull_strength != 0:
            force = force - radial.normalized() * self._pull_strength

        particle.acceleration = particle.acceleration + force


class AttractionModule(ParticleModule):
    """Attract/repel particles toward a point."""

    def __init__(
        self,
        target: Vec3 = None,
        strength: float = 1.0,
        radius: float = 10.0,
        falloff: str = "linear",  # "linear", "quadratic", "none"
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.UPDATE, lod_range)
        self._target = target or Vec3()
        self._strength = strength
        self._radius = radius
        self._falloff = falloff

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        to_target = self._target - particle.position
        distance = to_target.length()

        if distance < 0.001 or distance > self._radius:
            return

        direction = to_target.normalized()

        # Calculate falloff
        if self._falloff == "linear":
            factor = 1.0 - (distance / self._radius)
        elif self._falloff == "quadratic":
            factor = 1.0 - (distance / self._radius) ** 2
        else:
            factor = 1.0

        force = direction * self._strength * factor
        particle.acceleration = particle.acceleration + force


class VectorFieldModule(ParticleModule):
    """
    Apply force from a 3D vector field (force volume).

    The field is sampled at particle positions to get force direction.
    """

    def __init__(
        self,
        bounds_min: Vec3 = None,
        bounds_max: Vec3 = None,
        resolution: Tuple[int, int, int] = (16, 16, 16),
        strength: float = 1.0,
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.UPDATE, lod_range)
        self._bounds_min = bounds_min or Vec3(-10, -10, -10)
        self._bounds_max = bounds_max or Vec3(10, 10, 10)
        self._resolution = resolution
        self._strength = strength

        # Initialize empty field (would be loaded or generated)
        self._field: Optional[list[list[list[Vec3]]]] = None

    def set_field_data(self, field: list[list[list[Vec3]]]) -> None:
        """Set the vector field data."""
        self._field = field

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        if self._field is None:
            return

        # Check bounds
        p = particle.position
        if (
            p.x < self._bounds_min.x
            or p.x > self._bounds_max.x
            or p.y < self._bounds_min.y
            or p.y > self._bounds_max.y
            or p.z < self._bounds_min.z
            or p.z > self._bounds_max.z
        ):
            return

        # Calculate field coordinates
        size = Vec3(
            self._bounds_max.x - self._bounds_min.x,
            self._bounds_max.y - self._bounds_min.y,
            self._bounds_max.z - self._bounds_min.z,
        )

        fx = ((p.x - self._bounds_min.x) / size.x) * (self._resolution[0] - 1)
        fy = ((p.y - self._bounds_min.y) / size.y) * (self._resolution[1] - 1)
        fz = ((p.z - self._bounds_min.z) / size.z) * (self._resolution[2] - 1)

        # Nearest neighbor sampling (trilinear would be better)
        ix = min(max(int(fx), 0), self._resolution[0] - 1)
        iy = min(max(int(fy), 0), self._resolution[1] - 1)
        iz = min(max(int(fz), 0), self._resolution[2] - 1)

        force = self._field[ix][iy][iz] * self._strength
        particle.acceleration = particle.acceleration + force


class CollisionModule(ParticleModule):
    """
    Handle particle collisions with surfaces.

    Supports depth buffer, SDF, and primitive collision modes.
    Uses spatial hashing for efficient collision detection when enabled.
    """

    def __init__(
        self,
        mode: CollisionMode = CollisionMode.PRIMITIVE,
        bounce: float = 0.5,
        friction: float = 0.5,
        kill_on_collision: bool = False,
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
        use_spatial_hash: bool = False,
        cell_size: float = 1.0,
    ) -> None:
        super().__init__(ModuleStage.UPDATE, lod_range)
        self._mode = mode
        self._bounce = bounce
        self._friction = friction
        self._kill_on_collision = kill_on_collision
        self._use_spatial_hash = use_spatial_hash
        self._cell_size = cell_size

        # Primitive collision data (ground plane default)
        self._plane_normal = Vec3(0, 1, 0)
        self._plane_distance = 0.0

        # Spatial hash for particle-particle collision (optional)
        self._spatial_hash: dict[tuple[int, int, int], list[Particle]] = {}

    def _get_cell(self, position: Vec3) -> tuple[int, int, int]:
        """Get spatial hash cell for a position."""
        return (
            int(position.x // self._cell_size),
            int(position.y // self._cell_size),
            int(position.z // self._cell_size),
        )

    def clear_spatial_hash(self) -> None:
        """Clear the spatial hash for a new frame."""
        self._spatial_hash.clear()

    def add_to_spatial_hash(self, particle: Particle) -> None:
        """Add a particle to the spatial hash."""
        cell = self._get_cell(particle.position)
        if cell not in self._spatial_hash:
            self._spatial_hash[cell] = []
        self._spatial_hash[cell].append(particle)

    def get_nearby_particles(self, particle: Particle) -> list[Particle]:
        """Get particles in nearby cells for collision checking."""
        cell = self._get_cell(particle.position)
        nearby = []
        # Check 3x3x3 neighborhood
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    if neighbor in self._spatial_hash:
                        nearby.extend(self._spatial_hash[neighbor])
        return nearby

    def set_ground_plane(self, height: float = 0.0) -> None:
        """Configure ground plane collision."""
        self._plane_normal = Vec3(0, 1, 0)
        self._plane_distance = height

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        if self._mode == CollisionMode.NONE:
            return

        if self._mode == CollisionMode.PRIMITIVE:
            self._handle_primitive_collision(particle)
        elif self._mode == CollisionMode.DEPTH_BUFFER:
            self._handle_depth_collision(particle)
        elif self._mode == CollisionMode.SDF:
            self._handle_sdf_collision(particle)

    def _handle_primitive_collision(self, particle: Particle) -> None:
        """Handle collision with primitive (ground plane)."""
        # Distance to plane
        dist = particle.position.dot(self._plane_normal) - self._plane_distance

        if dist < 0:
            if self._kill_on_collision:
                particle.state = ParticleState.DYING
                return

            # Push particle above plane
            particle.position = (
                particle.position - self._plane_normal * dist
            )

            # Reflect velocity
            vn = self._plane_normal * particle.velocity.dot(self._plane_normal)
            vt = particle.velocity - vn

            particle.velocity = vt * (1.0 - self._friction) - vn * self._bounce

    def _handle_depth_collision(self, particle: Particle) -> None:
        """Handle collision against depth buffer (requires GPU)."""
        # Would sample depth buffer and compare with particle depth
        pass

    def _handle_sdf_collision(self, particle: Particle) -> None:
        """Handle collision against signed distance field."""
        # Would sample SDF and use gradient for normal
        pass


# =============================================================================
# UPDATE MODULES - ATTRIBUTE MODIFICATION
# =============================================================================


class SizeOverLifeModule(ParticleModule):
    """Modify particle size based on lifetime."""

    def __init__(
        self,
        start_size: float = 1.0,
        end_size: float = 0.0,
        curve: str = "linear",  # "linear", "ease_in", "ease_out", "ease_in_out"
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.UPDATE, lod_range)
        self._start_size = start_size
        self._end_size = end_size
        self._curve = curve

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        t = particle.normalized_age
        t = self._apply_curve(t)
        particle.size = self._start_size + (self._end_size - self._start_size) * t

    def _apply_curve(self, t: float) -> float:
        """Apply easing curve to t."""
        if self._curve == "ease_in":
            return t * t
        elif self._curve == "ease_out":
            return 1 - (1 - t) * (1 - t)
        elif self._curve == "ease_in_out":
            if t < 0.5:
                return 2 * t * t
            return 1 - 2 * (1 - t) * (1 - t)
        return t  # linear


class ColorOverLifeModule(ParticleModule):
    """Modify particle color based on lifetime."""

    def __init__(
        self,
        start_color: Vec4 = None,
        end_color: Vec4 = None,
        gradient: Optional[list[Tuple[float, Vec4]]] = None,
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.UPDATE, lod_range)
        self._start_color = start_color or Vec4(1, 1, 1, 1)
        self._end_color = end_color or Vec4(1, 1, 1, 0)
        self._gradient = gradient  # Optional: [(time, color), ...]

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        t = particle.normalized_age

        if self._gradient:
            particle.color = self._sample_gradient(t)
        else:
            particle.color = self._start_color.lerp(self._end_color, t)

    def _sample_gradient(self, t: float) -> Vec4:
        """Sample color from gradient at time t."""
        if not self._gradient:
            return Vec4(1, 1, 1, 1)

        # Find surrounding gradient keys
        for i in range(len(self._gradient) - 1):
            t0, c0 = self._gradient[i]
            t1, c1 = self._gradient[i + 1]
            if t0 <= t <= t1:
                local_t = (t - t0) / (t1 - t0) if t1 > t0 else 0
                return c0.lerp(c1, local_t)

        # Return last color if past end
        return self._gradient[-1][1]


class RotationModule(ParticleModule):
    """Apply rotation and angular velocity to particles."""

    def __init__(
        self,
        initial_rotation: Tuple[float, float] = (0, 360),  # (min, max) degrees
        angular_velocity: Tuple[float, float] = (-90, 90),  # (min, max) deg/sec
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.SPAWN, lod_range)  # Applied at spawn
        self._initial_rotation = initial_rotation
        self._angular_velocity = angular_velocity

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        # Set initial rotation
        particle.rotation = math.radians(
            random.uniform(self._initial_rotation[0], self._initial_rotation[1])
        )
        # Set angular velocity
        particle.angular_velocity = math.radians(
            random.uniform(self._angular_velocity[0], self._angular_velocity[1])
        )


class LifetimeModule(ParticleModule):
    """Set particle lifetime at spawn."""

    def __init__(
        self,
        lifetime: Tuple[float, float] = (1.0, 2.0),  # (min, max) seconds
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.SPAWN, lod_range)
        self._lifetime_range = lifetime

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        particle.lifetime = random.uniform(
            self._lifetime_range[0], self._lifetime_range[1]
        )


class VelocityModule(ParticleModule):
    """Set initial velocity at spawn."""

    def __init__(
        self,
        velocity: Vec3 = None,
        velocity_spread: Vec3 = None,
        inherit_velocity: float = 0.0,  # Inherit emitter velocity
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.SPAWN, lod_range)
        self._velocity = velocity or Vec3(0, 5, 0)
        self._spread = velocity_spread or Vec3(1, 1, 1)
        self._inherit = inherit_velocity

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        vel = Vec3(
            self._velocity.x + random.uniform(-self._spread.x, self._spread.x),
            self._velocity.y + random.uniform(-self._spread.y, self._spread.y),
            self._velocity.z + random.uniform(-self._spread.z, self._spread.z),
        )
        particle.velocity = vel


# =============================================================================
# RENDER MODULES
# =============================================================================


class BillboardRenderer(ParticleModule):
    """
    Render particles as camera-facing quads (billboards).

    Prepares rendering data for billboard particles.
    """

    def __init__(
        self,
        alignment: str = "view",  # "view", "velocity", "custom"
        stretch: float = 0.0,  # Stretch along velocity
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.RENDER, lod_range)
        self._alignment = alignment
        self._stretch = stretch
        self._camera_position = Vec3()
        self._camera_up = Vec3(0, 1, 0)

    def set_camera(self, position: Vec3, up: Vec3) -> None:
        """Set camera vectors for billboard orientation."""
        self._camera_position = position
        self._camera_up = up

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        # Calculate billboard orientation (stored in custom_data for rendering)
        if self._alignment == "view":
            # Face camera
            to_camera = (self._camera_position - particle.position).normalized()
            right = self._camera_up.cross(to_camera).normalized()
            up = to_camera.cross(right)
        elif self._alignment == "velocity":
            # Align with velocity
            vel_len = particle.velocity.length()
            if vel_len > 0.001:
                forward = particle.velocity.normalized()
                right = self._camera_up.cross(forward).normalized()
                up = forward.cross(right)
            else:
                right = Vec3(1, 0, 0)
                up = Vec3(0, 1, 0)
        else:
            right = Vec3(1, 0, 0)
            up = Vec3(0, 1, 0)

        particle.custom_data["billboard_right"] = right
        particle.custom_data["billboard_up"] = up

        # Apply velocity stretch
        if self._stretch > 0:
            vel_len = particle.velocity.length()
            particle.custom_data["stretch"] = 1.0 + vel_len * self._stretch


class MeshParticleRenderer(ParticleModule):
    """
    Render particles as instanced meshes.

    Each particle becomes an instance of a mesh with transform from particle data.
    """

    def __init__(
        self,
        mesh_path: Optional[str] = None,
        align_to_velocity: bool = False,
        scale_with_size: bool = True,
        lod_range: Tuple[int, int] = DEFAULT_LOD_RANGE,
    ) -> None:
        super().__init__(ModuleStage.RENDER, lod_range)
        self._mesh_path = mesh_path
        self._align_to_velocity = align_to_velocity
        self._scale_with_size = scale_with_size

    def apply_to_particle(self, particle: Particle, dt: float) -> None:
        # Prepare instance transform data
        scale = particle.size if self._scale_with_size else 1.0
        particle.custom_data["instance_scale"] = scale

        if self._align_to_velocity:
            vel_len = particle.velocity.length()
            if vel_len > 0.001:
                forward = particle.velocity.normalized()
                # Calculate rotation quaternion/matrix from forward direction
                particle.custom_data["instance_forward"] = forward


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Enums
    "ModuleStage",
    "EmitterShape",
    "CollisionMode",
    "BlendMode",
    # Config
    "ModuleConfig",
    # Base class
    "ParticleModule",
    # Spawn modules
    "ShapeEmitter",
    "BurstEmitter",
    "RateEmitter",
    # Force modules
    "GravityModule",
    "WindModule",
    "TurbulenceModule",
    "VortexModule",
    "AttractionModule",
    "VectorFieldModule",
    "CollisionModule",
    # Attribute modules
    "SizeOverLifeModule",
    "ColorOverLifeModule",
    "RotationModule",
    "LifetimeModule",
    "VelocityModule",
    # Render modules
    "BillboardRenderer",
    "MeshParticleRenderer",
]
