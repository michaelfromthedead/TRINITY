"""ECS system for crowd animation and rendering (T-AN-9.9).

Integrates crowd simulation, LOD selection, frustum culling, and GPU rendering.

Key Features:
- @system(phase="animation", order=5) annotation for ECS scheduling
- RVO/ORCA agent steering update for collision avoidance
- Animation texture baking for GPU-based skeletal animation
- LOD selection per agent based on camera distance
- Frustum culling per agent (skip invisible agents)
- Per-agent animation phase offset for variation
- Instance data buffer output for efficient GPU instanced rendering
- Support for large crowds (1000+ agents)

Dependencies:
- engine.animation.crowds.animation_texture: AnimationTextureAtlas (T-AN-8.1)
- engine.animation.crowds.crowd_renderer: CrowdRenderer (T-AN-8.2)
- engine.animation.crowds.crowd_lod: CrowdLOD (T-AN-8.5)
- engine.animation.crowds.crowd_behavior: CrowdSimulator, CrowdAgent
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional, Protocol, Sequence, Tuple, TYPE_CHECKING

from engine.core.math import Vec3, Vec4, Quat, Mat4, Transform
from engine.core.ecs import Entity, World

from ..crowds.animation_texture import (
    AnimationTexture,
    AnimationTextureAtlas,
    Skeleton,
    AnimationClip,
    bake_clip_to_texture,
    TextureFormat,
)
from ..crowds.crowd_renderer import (
    CrowdRenderer,
    CrowdInstance,
    CrowdRenderBatch,
    InstanceBuffer,
    RenderPriority,
)
from ..crowds.crowd_lod import CrowdLOD, LODLevel, LODTransition, LODTransitionMode
from ..crowds.crowd_behavior import (
    CrowdSimulator,
    CrowdAgent,
    AgentState,
    AnimationBlend,
    BehaviorContext,
)
from engine.animation.config import (
    CROWD_SYSTEM_CONFIG,
    CROWD_BEHAVIOR_CONFIG,
    CROWD_LOD_CONFIG,
)

if TYPE_CHECKING:
    from engine.core.ecs import Entity, World


# =============================================================================
# SYSTEM DECORATOR
# =============================================================================


def system(
    phase: str = "update",
    order: int = 0,
    reads: Optional[Tuple[str, ...]] = None,
    writes: Optional[Tuple[str, ...]] = None,
) -> Callable:
    """Decorator to mark a class as an ECS system with phase scheduling.

    Args:
        phase: Frame phase for execution ("animation", "update", "render", etc.)
        order: Execution order within phase (lower = earlier)
        reads: Component types this system reads from
        writes: Component types this system writes to

    Returns:
        Decorated class with system metadata.
    """
    def decorator(cls: type) -> type:
        cls._system_phase = phase
        cls._system_order = order
        cls._system_reads = reads or ()
        cls._system_writes = writes or ()
        return cls
    return decorator


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class SteeringMode(Enum):
    """Steering algorithm for crowd agent avoidance."""
    SIMPLE = auto()       # Basic separation steering
    RVO = auto()          # Reciprocal Velocity Obstacles
    ORCA = auto()         # Optimal Reciprocal Collision Avoidance


class CullingMode(Enum):
    """Frustum culling mode for crowd agents."""
    NONE = auto()         # No culling
    SPHERE = auto()       # Sphere-based frustum test
    BOX = auto()          # AABB-based frustum test


class AnimationBakeMode(Enum):
    """Animation texture baking strategy."""
    ON_DEMAND = auto()    # Bake when animation changes
    PRE_BAKED = auto()    # All animations pre-baked to atlas
    HYBRID = auto()       # Pre-bake common, on-demand for rare


# Default configuration constants
DEFAULT_NEIGHBOR_RADIUS = 5.0          # Radius to consider neighbors for steering
DEFAULT_MAX_NEIGHBORS = 10             # Maximum neighbors for RVO/ORCA
DEFAULT_TIME_HORIZON = 2.0             # Time horizon for velocity obstacles
DEFAULT_TIME_HORIZON_OBSTACLE = 1.0    # Time horizon for static obstacles
DEFAULT_PHASE_OFFSET_RANGE = 1.0       # Max random phase offset (seconds)


# =============================================================================
# FRUSTUM FOR CULLING
# =============================================================================


@dataclass
class Plane:
    """A plane defined by normal and distance from origin."""
    normal: Vec3 = field(default_factory=Vec3.zero)
    distance: float = 0.0

    def distance_to_point(self, point: Vec3) -> float:
        """Signed distance from point to plane."""
        return self.normal.dot(point) + self.distance


@dataclass
class Frustum:
    """View frustum for culling, defined by 6 planes.

    Planes are ordered: left, right, bottom, top, near, far.
    """
    planes: list[Plane] = field(default_factory=lambda: [Plane() for _ in range(6)])

    @staticmethod
    def from_view_projection(vp_matrix: Mat4) -> Frustum:
        """Extract frustum planes from view-projection matrix.

        Uses Gribb-Hartmann method for plane extraction.

        Args:
            vp_matrix: Combined view-projection matrix

        Returns:
            Frustum with 6 planes
        """
        m = vp_matrix.m
        planes = []

        # Left plane: row3 + row0
        planes.append(Plane(
            normal=Vec3(m[3] + m[0], m[7] + m[4], m[11] + m[8]).normalized(),
            distance=m[15] + m[12],
        ))
        # Right plane: row3 - row0
        planes.append(Plane(
            normal=Vec3(m[3] - m[0], m[7] - m[4], m[11] - m[8]).normalized(),
            distance=m[15] - m[12],
        ))
        # Bottom plane: row3 + row1
        planes.append(Plane(
            normal=Vec3(m[3] + m[1], m[7] + m[5], m[11] + m[9]).normalized(),
            distance=m[15] + m[13],
        ))
        # Top plane: row3 - row1
        planes.append(Plane(
            normal=Vec3(m[3] - m[1], m[7] - m[5], m[11] - m[9]).normalized(),
            distance=m[15] - m[13],
        ))
        # Near plane: row3 + row2
        planes.append(Plane(
            normal=Vec3(m[3] + m[2], m[7] + m[6], m[11] + m[10]).normalized(),
            distance=m[15] + m[14],
        ))
        # Far plane: row3 - row2
        planes.append(Plane(
            normal=Vec3(m[3] - m[2], m[7] - m[6], m[11] - m[10]).normalized(),
            distance=m[15] - m[14],
        ))

        # Normalize planes
        for plane in planes:
            length = plane.normal.length()
            if length > 0.0001:
                plane.normal = plane.normal * (1.0 / length)
                plane.distance /= length

        return Frustum(planes=planes)

    def is_sphere_visible(self, center: Vec3, radius: float) -> bool:
        """Test if sphere intersects frustum.

        Args:
            center: Sphere center
            radius: Sphere radius

        Returns:
            True if sphere is at least partially inside frustum
        """
        for plane in self.planes:
            if plane.distance_to_point(center) < -radius:
                return False
        return True

    def is_point_visible(self, point: Vec3) -> bool:
        """Test if point is inside frustum."""
        return self.is_sphere_visible(point, 0.0)


# =============================================================================
# RVO/ORCA STEERING
# =============================================================================


@dataclass
class VelocityObstacle:
    """Velocity obstacle for RVO/ORCA steering.

    Represents the set of velocities that will cause collision with
    another agent within a given time horizon.
    """
    apex: Vec3 = field(default_factory=Vec3.zero)
    left_leg: Vec3 = field(default_factory=Vec3.zero)
    right_leg: Vec3 = field(default_factory=Vec3.zero)


@dataclass
class ORCALine:
    """ORCA half-plane constraint.

    Defines a linear constraint on velocity space.
    """
    point: Vec3 = field(default_factory=Vec3.zero)
    direction: Vec3 = field(default_factory=Vec3.zero)


@dataclass
class RVOConfig:
    """Configuration for RVO/ORCA steering."""
    neighbor_radius: float = DEFAULT_NEIGHBOR_RADIUS
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS
    time_horizon: float = DEFAULT_TIME_HORIZON
    time_horizon_obstacle: float = DEFAULT_TIME_HORIZON_OBSTACLE
    max_speed: float = CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED * 1.5
    radius: float = CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_RADIUS


class RVOSteering:
    """Reciprocal Velocity Obstacles steering algorithm.

    Implements RVO for crowd agents to achieve collision-free velocities
    in dense scenarios. Supports both RVO and ORCA modes.
    """

    def __init__(self, config: RVOConfig | None = None):
        self._config = config or RVOConfig()
        self._velocity_obstacles: list[VelocityObstacle] = []
        self._orca_lines: list[ORCALine] = []

    @property
    def config(self) -> RVOConfig:
        return self._config

    def compute_new_velocity(
        self,
        agent: CrowdAgent,
        neighbors: list[CrowdAgent],
        obstacles: list[tuple[Vec3, float]],
        preferred_velocity: Vec3,
        mode: SteeringMode = SteeringMode.RVO,
    ) -> Vec3:
        """Compute collision-free velocity for agent.

        Args:
            agent: Agent to compute velocity for
            neighbors: Nearby agents to consider
            obstacles: Static obstacles (position, radius)
            preferred_velocity: Desired velocity without avoidance
            mode: Steering algorithm to use

        Returns:
            New velocity that avoids collisions
        """
        if mode == SteeringMode.SIMPLE:
            return self._compute_simple_avoidance(agent, neighbors, obstacles, preferred_velocity)
        elif mode == SteeringMode.RVO:
            return self._compute_rvo_velocity(agent, neighbors, obstacles, preferred_velocity)
        else:  # ORCA
            return self._compute_orca_velocity(agent, neighbors, obstacles, preferred_velocity)

    def _compute_simple_avoidance(
        self,
        agent: CrowdAgent,
        neighbors: list[CrowdAgent],
        obstacles: list[tuple[Vec3, float]],
        preferred_velocity: Vec3,
    ) -> Vec3:
        """Simple separation-based avoidance."""
        avoidance = Vec3.zero()

        for neighbor in neighbors:
            to_agent = agent.position - neighbor.position
            dist = to_agent.length()
            if dist < self._config.neighbor_radius and dist > 0.01:
                strength = 1.0 - dist / self._config.neighbor_radius
                avoidance = avoidance + to_agent.normalized() * strength

        for obs_pos, obs_radius in obstacles:
            to_agent = agent.position - obs_pos
            dist = to_agent.length()
            combined_radius = obs_radius + agent.radius
            if dist < combined_radius + self._config.neighbor_radius and dist > 0.01:
                strength = 1.0 - (dist - combined_radius) / self._config.neighbor_radius
                strength = max(0.0, strength)
                avoidance = avoidance + to_agent.normalized() * strength * 2.0

        result = preferred_velocity + avoidance
        if result.length() > self._config.max_speed:
            result = result.normalized() * self._config.max_speed
        return result

    def _compute_rvo_velocity(
        self,
        agent: CrowdAgent,
        neighbors: list[CrowdAgent],
        obstacles: list[tuple[Vec3, float]],
        preferred_velocity: Vec3,
    ) -> Vec3:
        """Compute velocity using RVO (Reciprocal Velocity Obstacles)."""
        self._velocity_obstacles.clear()

        # Build velocity obstacles for each neighbor
        for neighbor in neighbors:
            vo = self._build_velocity_obstacle(agent, neighbor)
            if vo is not None:
                self._velocity_obstacles.append(vo)

        # Add obstacles as velocity obstacles
        for obs_pos, obs_radius in obstacles:
            vo = self._build_static_obstacle_vo(agent, obs_pos, obs_radius)
            if vo is not None:
                self._velocity_obstacles.append(vo)

        # If no obstacles, return preferred velocity
        if not self._velocity_obstacles:
            if preferred_velocity.length() > self._config.max_speed:
                return preferred_velocity.normalized() * self._config.max_speed
            return preferred_velocity

        # Sample velocities and find best collision-free one
        return self._sample_rvo_velocity(agent, preferred_velocity)

    def _build_velocity_obstacle(
        self,
        agent: CrowdAgent,
        neighbor: CrowdAgent,
    ) -> VelocityObstacle | None:
        """Build velocity obstacle for neighbor agent."""
        relative_pos = neighbor.position - agent.position
        dist = relative_pos.length()

        combined_radius = agent.radius + neighbor.radius
        if dist < combined_radius:
            # Already colliding - push apart direction
            if dist < 0.001:
                return None
            return VelocityObstacle(
                apex=Vec3.zero(),
                left_leg=Vec3(-relative_pos.z, 0, relative_pos.x).normalized(),
                right_leg=Vec3(relative_pos.z, 0, -relative_pos.x).normalized(),
            )

        # Calculate tangent directions
        theta = math.asin(min(1.0, combined_radius / dist))
        direction = relative_pos.normalized()

        # Rotate direction by theta for left and right legs
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        left_leg = Vec3(
            direction.x * cos_t - direction.z * sin_t,
            0.0,
            direction.x * sin_t + direction.z * cos_t,
        )
        right_leg = Vec3(
            direction.x * cos_t + direction.z * sin_t,
            0.0,
            -direction.x * sin_t + direction.z * cos_t,
        )

        # RVO: apex at average velocity
        avg_velocity = (agent.velocity + neighbor.velocity) * 0.5
        apex = avg_velocity

        return VelocityObstacle(apex=apex, left_leg=left_leg, right_leg=right_leg)

    def _build_static_obstacle_vo(
        self,
        agent: CrowdAgent,
        obs_pos: Vec3,
        obs_radius: float,
    ) -> VelocityObstacle | None:
        """Build velocity obstacle for static obstacle."""
        relative_pos = obs_pos - agent.position
        dist = relative_pos.length()

        combined_radius = agent.radius + obs_radius
        if dist < combined_radius:
            if dist < 0.001:
                return None
            return VelocityObstacle(
                apex=Vec3.zero(),
                left_leg=Vec3(-relative_pos.z, 0, relative_pos.x).normalized() * 2.0,
                right_leg=Vec3(relative_pos.z, 0, -relative_pos.x).normalized() * 2.0,
            )

        if dist >= combined_radius:
            theta = math.asin(min(1.0, combined_radius / dist))
            direction = relative_pos.normalized()

            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            left_leg = Vec3(
                direction.x * cos_t - direction.z * sin_t,
                0.0,
                direction.x * sin_t + direction.z * cos_t,
            )
            right_leg = Vec3(
                direction.x * cos_t + direction.z * sin_t,
                0.0,
                -direction.x * sin_t + direction.z * cos_t,
            )

            return VelocityObstacle(apex=Vec3.zero(), left_leg=left_leg, right_leg=right_leg)

        return None

    def _sample_rvo_velocity(
        self,
        agent: CrowdAgent,
        preferred_velocity: Vec3,
    ) -> Vec3:
        """Sample velocity space to find collision-free velocity."""
        best_velocity = preferred_velocity
        best_cost = float('inf')

        # Check preferred velocity first
        if not self._is_in_any_vo(preferred_velocity):
            return preferred_velocity if preferred_velocity.length() <= self._config.max_speed else preferred_velocity.normalized() * self._config.max_speed

        # Sample velocities in concentric rings
        for speed_factor in [1.0, 0.75, 0.5, 0.25, 0.0]:
            speed = self._config.max_speed * speed_factor
            for angle_idx in range(16):
                angle = angle_idx * math.pi * 2 / 16
                sample_vel = Vec3(
                    math.cos(angle) * speed,
                    0.0,
                    math.sin(angle) * speed,
                )

                if not self._is_in_any_vo(sample_vel):
                    cost = (sample_vel - preferred_velocity).length_squared()
                    if cost < best_cost:
                        best_cost = cost
                        best_velocity = sample_vel

        return best_velocity

    def _is_in_any_vo(self, velocity: Vec3) -> bool:
        """Check if velocity is inside any velocity obstacle."""
        for vo in self._velocity_obstacles:
            if self._is_in_vo(velocity, vo):
                return True
        return False

    def _is_in_vo(self, velocity: Vec3, vo: VelocityObstacle) -> bool:
        """Check if velocity is inside a velocity obstacle cone."""
        relative_vel = velocity - vo.apex

        # Cross products to check which side of legs
        cross_left = relative_vel.x * vo.left_leg.z - relative_vel.z * vo.left_leg.x
        cross_right = relative_vel.x * vo.right_leg.z - relative_vel.z * vo.right_leg.x

        # Inside if between both legs (signs differ based on orientation)
        return cross_left <= 0 and cross_right >= 0

    def _compute_orca_velocity(
        self,
        agent: CrowdAgent,
        neighbors: list[CrowdAgent],
        obstacles: list[tuple[Vec3, float]],
        preferred_velocity: Vec3,
    ) -> Vec3:
        """Compute velocity using ORCA (Optimal Reciprocal Collision Avoidance)."""
        self._orca_lines.clear()

        # Build ORCA half-planes for each neighbor
        for neighbor in neighbors:
            line = self._build_orca_line(agent, neighbor)
            if line is not None:
                self._orca_lines.append(line)

        # Add lines for static obstacles
        for obs_pos, obs_radius in obstacles:
            line = self._build_obstacle_orca_line(agent, obs_pos, obs_radius)
            if line is not None:
                self._orca_lines.append(line)

        # If no constraints, return preferred velocity
        if not self._orca_lines:
            if preferred_velocity.length() > self._config.max_speed:
                return preferred_velocity.normalized() * self._config.max_speed
            return preferred_velocity

        # Linear program to find closest velocity to preferred
        return self._solve_orca_linear_program(preferred_velocity)

    def _build_orca_line(
        self,
        agent: CrowdAgent,
        neighbor: CrowdAgent,
    ) -> ORCALine | None:
        """Build ORCA half-plane constraint for neighbor."""
        relative_pos = neighbor.position - agent.position
        relative_vel = agent.velocity - neighbor.velocity
        dist_sq = relative_pos.length_squared()

        combined_radius = agent.radius + neighbor.radius
        combined_radius_sq = combined_radius * combined_radius

        if dist_sq > combined_radius_sq:
            # No collision yet
            w = relative_vel - relative_pos * (1.0 / self._config.time_horizon)

            w_length_sq = w.length_squared()
            dot_product1 = w.dot(relative_pos)

            if dot_product1 < 0.0 and dot_product1 * dot_product1 > combined_radius_sq * w_length_sq:
                # Project on cut-off circle
                w_length = math.sqrt(w_length_sq)
                if w_length < 0.0001:
                    return None
                unit_w = w * (1.0 / w_length)
                direction = Vec3(unit_w.z, 0.0, -unit_w.x)
                u = (combined_radius / self._config.time_horizon - w_length) * unit_w
            else:
                # Project on legs
                dist = math.sqrt(dist_sq)
                leg = math.sqrt(dist_sq - combined_radius_sq)

                if relative_pos.x * relative_vel.z - relative_pos.z * relative_vel.x > 0.0:
                    # Left leg
                    direction = Vec3(
                        relative_pos.x * leg - relative_pos.z * combined_radius,
                        0.0,
                        relative_pos.x * combined_radius + relative_pos.z * leg,
                    ) * (1.0 / dist_sq)
                else:
                    # Right leg
                    direction = Vec3(
                        relative_pos.x * leg + relative_pos.z * combined_radius,
                        0.0,
                        -relative_pos.x * combined_radius + relative_pos.z * leg,
                    ) * (-1.0 / dist_sq)

                dot_product2 = relative_vel.dot(direction)
                u = direction * dot_product2 - relative_vel

            return ORCALine(
                point=agent.velocity + u * 0.5,
                direction=direction.normalized() if direction.length() > 0.0001 else Vec3.forward(),
            )
        else:
            # Already colliding - push apart immediately
            dist = math.sqrt(dist_sq)
            if dist < 0.0001:
                return None

            inv_time_step = 1.0 / 0.016  # Assume 60fps
            w = relative_vel - relative_pos * inv_time_step
            w_length = w.length()
            if w_length < 0.0001:
                return None

            unit_w = w * (1.0 / w_length)
            direction = Vec3(unit_w.z, 0.0, -unit_w.x)
            u = (combined_radius * inv_time_step - w_length) * unit_w

            return ORCALine(
                point=agent.velocity + u * 0.5,
                direction=direction,
            )

    def _build_obstacle_orca_line(
        self,
        agent: CrowdAgent,
        obs_pos: Vec3,
        obs_radius: float,
    ) -> ORCALine | None:
        """Build ORCA constraint for static obstacle."""
        relative_pos = obs_pos - agent.position
        dist = relative_pos.length()
        combined_radius = agent.radius + obs_radius

        if dist < 0.0001:
            return None

        if dist < combined_radius:
            # Inside obstacle - push out
            direction = Vec3(-relative_pos.z, 0.0, relative_pos.x).normalized()
            point = agent.velocity + relative_pos.normalized() * self._config.max_speed
            return ORCALine(point=point, direction=direction)

        return None

    def _solve_orca_linear_program(self, preferred_velocity: Vec3) -> Vec3:
        """Solve linear program to find velocity closest to preferred."""
        result = preferred_velocity

        for line in self._orca_lines:
            # Check if current result satisfies constraint
            if self._det(line.direction, line.point - result) > 0.0:
                # Does not satisfy - project onto line
                t = self._det(line.direction, result - line.point)
                result = line.point + line.direction * t

        # Clamp to max speed
        if result.length() > self._config.max_speed:
            result = result.normalized() * self._config.max_speed

        return result

    def _det(self, v1: Vec3, v2: Vec3) -> float:
        """2D determinant (cross product Z component)."""
        return v1.x * v2.z - v1.z * v2.x


# =============================================================================
# INSTANCE DATA BUFFER
# =============================================================================


@dataclass
class CrowdInstanceData:
    """Per-instance data packed for GPU rendering.

    Contains all per-agent data needed by GPU shaders:
    - Transform (position, rotation, scale)
    - Animation (texture row, time offset, LOD)
    - Visual (tint color, visibility flags)
    """
    transform: Mat4 = field(default_factory=Mat4.identity)
    animation_texture_row: int = 0
    animation_time: float = 0.0
    animation_phase_offset: float = 0.0
    lod_level: int = 0
    tint_color: Vec4 = field(default_factory=lambda: Vec4(1.0, 1.0, 1.0, 1.0))
    visible: bool = True
    agent_id: int = 0


@dataclass
class CrowdInstanceBuffer:
    """GPU-ready buffer of instance data.

    Packs instance data in a format suitable for GPU instanced rendering.
    Uses Structure of Arrays (SoA) layout for better GPU cache utilization.
    """
    # SoA layout for GPU efficiency
    transforms: list[float] = field(default_factory=list)      # 16 floats per instance (4x4 matrix)
    animations: list[float] = field(default_factory=list)      # 4 floats per instance (row, time, offset, lod)
    colors: list[float] = field(default_factory=list)          # 4 floats per instance (RGBA)
    agent_ids: list[int] = field(default_factory=list)         # Instance to agent mapping

    instance_count: int = 0
    visible_count: int = 0
    capacity: int = 0
    dirty: bool = True

    def clear(self) -> None:
        """Clear all instance data."""
        self.transforms.clear()
        self.animations.clear()
        self.colors.clear()
        self.agent_ids.clear()
        self.instance_count = 0
        self.visible_count = 0
        self.dirty = True

    def reserve(self, count: int) -> None:
        """Reserve capacity for instances."""
        self.capacity = count
        self.transforms = [0.0] * (count * 16)
        self.animations = [0.0] * (count * 4)
        self.colors = [0.0] * (count * 4)
        self.agent_ids = [0] * count

    def add_instance(self, data: CrowdInstanceData) -> int:
        """Add instance to buffer, returns index."""
        if self.instance_count >= self.capacity:
            self._grow()

        idx = self.instance_count
        self.instance_count += 1

        if data.visible:
            self.visible_count += 1

        # Pack transform matrix
        mat_offset = idx * 16
        self.transforms[mat_offset:mat_offset + 16] = data.transform.m

        # Pack animation data
        anim_offset = idx * 4
        self.animations[anim_offset:anim_offset + 4] = [
            float(data.animation_texture_row),
            data.animation_time + data.animation_phase_offset,
            data.animation_phase_offset,
            float(data.lod_level),
        ]

        # Pack color data
        color_offset = idx * 4
        self.colors[color_offset:color_offset + 4] = [
            data.tint_color.x,
            data.tint_color.y,
            data.tint_color.z,
            data.tint_color.w if data.visible else 0.0,  # Alpha 0 for invisible
        ]

        self.agent_ids[idx] = data.agent_id
        self.dirty = True
        return idx

    def update_instance(self, index: int, data: CrowdInstanceData) -> None:
        """Update existing instance data."""
        if index < 0 or index >= self.instance_count:
            return

        mat_offset = index * 16
        self.transforms[mat_offset:mat_offset + 16] = data.transform.m

        anim_offset = index * 4
        self.animations[anim_offset:anim_offset + 4] = [
            float(data.animation_texture_row),
            data.animation_time + data.animation_phase_offset,
            data.animation_phase_offset,
            float(data.lod_level),
        ]

        color_offset = index * 4
        self.colors[color_offset:color_offset + 4] = [
            data.tint_color.x,
            data.tint_color.y,
            data.tint_color.z,
            data.tint_color.w if data.visible else 0.0,
        ]

        self.dirty = True

    def _grow(self) -> None:
        """Grow buffer capacity."""
        new_capacity = max(self.capacity * 2, 64)
        growth = new_capacity - self.capacity

        self.transforms.extend([0.0] * (growth * 16))
        self.animations.extend([0.0] * (growth * 4))
        self.colors.extend([0.0] * (growth * 4))
        self.agent_ids.extend([0] * growth)
        self.capacity = new_capacity

    def get_memory_size_bytes(self) -> int:
        """Calculate memory size in bytes."""
        float_bytes = (len(self.transforms) + len(self.animations) + len(self.colors)) * 4
        int_bytes = len(self.agent_ids) * 4
        return float_bytes + int_bytes


# =============================================================================
# CROWD COMPONENT
# =============================================================================


@dataclass
class CrowdComponent:
    """Component for entities managing crowds.

    Attributes:
        simulator: Crowd behavior simulator
        renderer: GPU crowd renderer
        lod: LOD manager
        enabled: Whether crowd is enabled
        update_rate: Simulation update rate (per second)
        max_visible: Maximum visible instances
        steering_mode: Steering algorithm to use
        culling_mode: Frustum culling mode
        rvo_config: Configuration for RVO/ORCA steering
    """
    simulator: CrowdSimulator = field(default_factory=CrowdSimulator)
    renderer: CrowdRenderer = field(default_factory=CrowdRenderer)
    lod: CrowdLOD = field(default_factory=CrowdLOD)
    enabled: bool = True
    update_rate: float = CROWD_SYSTEM_CONFIG.DEFAULT_UPDATE_RATE
    max_visible: int = CROWD_SYSTEM_CONFIG.DEFAULT_MAX_VISIBLE

    # Steering configuration
    steering_mode: SteeringMode = SteeringMode.RVO
    rvo_config: RVOConfig = field(default_factory=RVOConfig)

    # Culling configuration
    culling_mode: CullingMode = CullingMode.SPHERE
    culling_radius: float = 0.5  # Agent bounding sphere radius

    # Animation texture atlas
    animation_atlas: AnimationTextureAtlas | None = None
    animation_names: dict[int, str] = field(default_factory=dict)  # anim_index -> clip name

    # Camera reference for LOD and culling
    camera_position: Vec3 = field(default_factory=Vec3.zero)
    camera_forward: Vec3 = field(default_factory=Vec3.forward)
    view_projection_matrix: Mat4 | None = None

    # Per-agent phase offsets for animation variation
    _phase_offsets: dict[int, float] = field(default_factory=dict)

    # State
    _accumulated_time: float = 0.0
    _synced_instances: dict[int, int] = field(default_factory=dict)  # agent_id -> instance_id
    _instance_buffer: CrowdInstanceBuffer = field(default_factory=CrowdInstanceBuffer)
    _frustum: Frustum | None = None

    def add_agent(
        self,
        position: Vec3,
        mesh_id: int = 0,
        material_id: int = 0,
        initial_state: AgentState = AgentState.IDLE,
        phase_offset: float | None = None,
    ) -> int:
        """Add a new crowd agent.

        Args:
            position: World position
            mesh_id: Mesh identifier for rendering
            material_id: Material identifier
            initial_state: Initial behavior state
            phase_offset: Animation phase offset (random if None)

        Returns:
            Agent ID
        """
        agent = CrowdAgent(
            position=position,
            current_state=initial_state,
        )
        agent_id = self.simulator.add_agent(agent)

        instance = CrowdInstance(
            position=position,
            rotation=agent.get_rotation(),
        )
        instance_id = self.renderer.add_instance(instance, mesh_id, material_id)

        self._synced_instances[agent_id] = instance_id

        # Set random phase offset for animation variation
        if phase_offset is None:
            phase_offset = random.uniform(0, DEFAULT_PHASE_OFFSET_RANGE)
        self._phase_offsets[agent_id] = phase_offset

        return agent_id

    def remove_agent(self, agent_id: int) -> bool:
        """Remove agent and its render instance."""
        if agent_id not in self._synced_instances:
            return False

        instance_id = self._synced_instances.pop(agent_id)
        self._phase_offsets.pop(agent_id, None)
        self.simulator.remove_agent(agent_id)
        self.renderer.remove_instance(instance_id)
        return True

    def get_agent(self, agent_id: int) -> CrowdAgent | None:
        """Get agent by ID."""
        return self.simulator.get_agent(agent_id)

    def set_agent_target(self, agent_id: int, target: Vec3) -> bool:
        """Set movement target for agent."""
        agent = self.simulator.get_agent(agent_id)
        if agent:
            agent.target_position = target
            self.simulator.transition_agent(agent, AgentState.WALKING)
            return True
        return False

    def get_agent_count(self) -> int:
        """Get total agent count."""
        return self.simulator.agent_count

    def get_visible_count(self) -> int:
        """Get visible instance count."""
        return self._instance_buffer.visible_count

    def get_phase_offset(self, agent_id: int) -> float:
        """Get animation phase offset for agent."""
        return self._phase_offsets.get(agent_id, 0.0)

    def set_phase_offset(self, agent_id: int, offset: float) -> None:
        """Set animation phase offset for agent."""
        self._phase_offsets[agent_id] = offset

    def get_instance_buffer(self) -> CrowdInstanceBuffer:
        """Get the GPU-ready instance buffer."""
        return self._instance_buffer

    def set_view_projection_matrix(self, matrix: Mat4) -> None:
        """Set view-projection matrix for frustum culling."""
        self.view_projection_matrix = matrix
        self._frustum = Frustum.from_view_projection(matrix)

    def set_animation_atlas(self, atlas: AnimationTextureAtlas, clip_names: dict[int, str]) -> None:
        """Set animation texture atlas and clip name mapping.

        Args:
            atlas: Pre-baked animation texture atlas
            clip_names: Mapping from animation index to clip name in atlas
        """
        self.animation_atlas = atlas
        self.animation_names = clip_names


# =============================================================================
# CROWD SYSTEM
# =============================================================================


@system(phase="animation", order=5, reads=("CrowdComponent",), writes=("CrowdInstance",))
class CrowdSystem:
    """ECS system for crowd simulation and rendering.

    Manages crowd behavior simulation, RVO/ORCA steering, LOD updates,
    frustum culling, and GPU render data preparation.

    Update order within animation phase:
    0: Motion Matching
    1: IK
    2: Procedural
    3: Skinning
    5: Crowd (this system)
    """

    def __init__(self):
        self._lod_distances: list[float] = list(CROWD_SYSTEM_CONFIG.DEFAULT_LOD_DISTANCES)
        self._cull_distance: float = CROWD_SYSTEM_CONFIG.DEFAULT_CULL_DISTANCE
        self._rvo_steering = RVOSteering()
        self._total_agents_processed = 0
        self._total_agents_visible = 0
        self._total_agents_culled = 0

    def set_lod_distances(self, distances: list[float]) -> None:
        """Set LOD distance thresholds."""
        self._lod_distances = sorted(distances)

    def set_cull_distance(self, distance: float) -> None:
        """Set maximum render distance."""
        self._cull_distance = distance

    def update(
        self,
        world: World,
        dt: float,
        entity_components: list[tuple[Entity, CrowdComponent]],
    ) -> None:
        """Update all crowd components.

        For each CrowdAgent entity:
        1. Update steering via RVO/ORCA local avoidance
        2. Select animation based on velocity/state
        3. Bake bone transforms to animation texture
        4. Select LOD level based on camera distance
        5. Perform frustum culling (skip invisible agents)
        6. Output instance data buffer for GPU rendering

        Args:
            world: ECS world
            dt: Delta time
            entity_components: List of (entity, component) tuples
        """
        self._total_agents_processed = 0
        self._total_agents_visible = 0
        self._total_agents_culled = 0

        for entity, component in entity_components:
            if not component.enabled:
                continue

            self._update_component(component, dt)

    def _update_component(self, component: CrowdComponent, dt: float) -> None:
        """Update single crowd component."""
        # Accumulate time for fixed-rate updates
        component._accumulated_time += dt
        update_interval = 1.0 / component.update_rate

        # Run simulation updates at fixed rate
        while component._accumulated_time >= update_interval:
            component._accumulated_time -= update_interval
            self._update_simulation(component, update_interval)

        # Sync agent states to render instances and build instance buffer
        self._build_instance_buffer(component)

        # Update renderer
        component.renderer.update(dt)

        # Advance LOD frame counter
        component.lod.advance_frame()

    def _update_simulation(self, component: CrowdComponent, dt: float) -> None:
        """Update crowd simulation with RVO/ORCA steering."""
        # Get all agents and context
        agents: list[CrowdAgent] = []
        for i in range(component.simulator.agent_count):
            agent_id = list(component._synced_instances.keys())[i] if i < len(component._synced_instances) else None
            if agent_id:
                agent = component.simulator.get_agent(agent_id)
                if agent:
                    agents.append(agent)

        # Build context for behavior updates
        context = BehaviorContext(
            all_agents=agents,
            obstacles=[],  # Could be populated from world
            time=component.simulator.time,
        )

        # Update RVO steering config
        self._rvo_steering._config = component.rvo_config

        # Update each agent with steering
        for agent in agents:
            if agent.current_state in (AgentState.WALKING, AgentState.FLEEING):
                # Get nearby neighbors for RVO
                neighbors = context.get_nearby_agents(agent, component.rvo_config.neighbor_radius)
                neighbors = neighbors[:component.rvo_config.max_neighbors]

                # Compute RVO/ORCA velocity
                new_velocity = self._rvo_steering.compute_new_velocity(
                    agent=agent,
                    neighbors=neighbors,
                    obstacles=context.obstacles,
                    preferred_velocity=agent.target_velocity,
                    mode=component.steering_mode,
                )

                # Apply steering result
                agent.velocity = new_velocity
                agent.position = agent.position + new_velocity * dt

                # Update facing
                if new_velocity.length_squared() > 0.01:
                    agent.facing = math.atan2(new_velocity.x, new_velocity.z)

        # Run standard behavior simulation
        component.simulator.update(dt)

    def _build_instance_buffer(self, component: CrowdComponent) -> None:
        """Build GPU instance buffer from agent states."""
        component._instance_buffer.clear()
        component._instance_buffer.reserve(component.get_agent_count())

        for agent_id, instance_id in component._synced_instances.items():
            agent = component.simulator.get_agent(agent_id)
            if not agent:
                continue

            self._total_agents_processed += 1

            # Calculate distance to camera
            distance = agent.position.distance(component.camera_position)

            # Distance culling
            if distance > self._cull_distance:
                self._total_agents_culled += 1
                continue

            # Frustum culling
            if component.culling_mode != CullingMode.NONE and component._frustum:
                is_visible = component._frustum.is_sphere_visible(
                    agent.position,
                    component.culling_radius,
                )
                if not is_visible:
                    self._total_agents_culled += 1
                    continue

            self._total_agents_visible += 1

            # Select LOD based on distance
            lod_level = component.lod.get_lod_for_distance(distance, 0)

            # Get animation data
            anim_blend = agent.animation_blend
            primary_anim = anim_blend.get_primary_animation()

            # Get animation texture row from atlas
            texture_row = 0
            if component.animation_atlas and primary_anim in component.animation_names:
                clip_name = component.animation_names[primary_anim]
                clip_info = component.animation_atlas.get_clip_info(clip_name)
                if clip_info:
                    texture_row = clip_info[0]  # start_row

            # Get phase offset for animation variation
            phase_offset = component.get_phase_offset(agent_id)

            # Build transform
            transform = Transform(
                translation=agent.position,
                rotation=agent.get_rotation(),
                scale=Vec3(1.0, 1.0, 1.0),
            )

            # Create instance data
            instance_data = CrowdInstanceData(
                transform=transform.to_matrix(),
                animation_texture_row=texture_row,
                animation_time=agent.state_time,
                animation_phase_offset=phase_offset,
                lod_level=lod_level,
                tint_color=Vec4(1.0, 1.0, 1.0, 1.0),
                visible=True,
                agent_id=agent_id,
            )

            component._instance_buffer.add_instance(instance_data)

            # Also sync to renderer's instances for compatibility
            for batch in component.renderer.get_batches():
                for instance in batch.instances:
                    if instance.instance_id == instance_id:
                        instance.position = agent.position
                        instance.rotation = agent.get_rotation()
                        instance.animation_index = primary_anim
                        instance.lod_level = lod_level
                        instance.visible = True
                        break

    def prepare_render_data(
        self,
        entity_components: list[tuple[Entity, CrowdComponent]],
    ) -> list[tuple[Entity, CrowdInstanceBuffer]]:
        """Prepare instance buffers for GPU rendering.

        Returns:
            List of (entity, instance_buffer) tuples ready for GPU upload
        """
        result = []
        for entity, component in entity_components:
            if not component.enabled:
                continue
            result.append((entity, component._instance_buffer))
        return result

    def get_legacy_render_data(
        self,
        entity_components: list[tuple[Entity, CrowdComponent]],
    ) -> list[tuple[Entity, Any]]:
        """Prepare render data using legacy CrowdRenderer interface.

        For backward compatibility with existing rendering code.

        Returns:
            List of (entity, render_data) tuples
        """
        result = []
        for entity, component in entity_components:
            if not component.enabled:
                continue
            render_data = component.renderer.prepare_render_data()
            result.append((entity, render_data))
        return result

    def trigger_flee_event(
        self,
        component: CrowdComponent,
        threat_position: Vec3,
        radius: float,
    ) -> int:
        """Trigger flee event for agents near threat.

        Args:
            component: Crowd component
            threat_position: Position of threat
            radius: Affect radius

        Returns:
            Number of agents affected
        """
        return component.simulator.trigger_flee(threat_position, radius)

    def spawn_crowd_formation(
        self,
        component: CrowdComponent,
        center: Vec3,
        count: int,
        radius: float,
        mesh_id: int = 0,
        material_id: int = 0,
        formation: str = "circle",
        randomize_phase: bool = True,
    ) -> list[int]:
        """Spawn agents in a formation.

        Args:
            component: Crowd component
            center: Formation center
            count: Number of agents
            radius: Formation radius
            mesh_id: Mesh ID for rendering
            material_id: Material ID
            formation: Formation type (circle, grid, random)
            randomize_phase: Whether to randomize animation phase offsets

        Returns:
            List of agent IDs
        """
        agent_ids = []

        if formation == "circle":
            for i in range(count):
                angle = (i / count) * math.pi * 2
                pos = Vec3(
                    center.x + math.cos(angle) * radius,
                    center.y,
                    center.z + math.sin(angle) * radius,
                )
                phase = random.uniform(0, DEFAULT_PHASE_OFFSET_RANGE) if randomize_phase else 0.0
                agent_id = component.add_agent(pos, mesh_id, material_id, phase_offset=phase)
                agent_ids.append(agent_id)

        elif formation == "grid":
            side = int(math.ceil(math.sqrt(count)))
            spacing = radius * 2 / max(1, side - 1) if side > 1 else 0
            idx = 0
            for row in range(side):
                for col in range(side):
                    if idx >= count:
                        break
                    pos = Vec3(
                        center.x - radius + col * spacing,
                        center.y,
                        center.z - radius + row * spacing,
                    )
                    phase = random.uniform(0, DEFAULT_PHASE_OFFSET_RANGE) if randomize_phase else 0.0
                    agent_id = component.add_agent(pos, mesh_id, material_id, phase_offset=phase)
                    agent_ids.append(agent_id)
                    idx += 1

        elif formation == "random":
            for _ in range(count):
                angle = random.random() * math.pi * 2
                dist = random.random() * radius
                pos = Vec3(
                    center.x + math.cos(angle) * dist,
                    center.y,
                    center.z + math.sin(angle) * dist,
                )
                phase = random.uniform(0, DEFAULT_PHASE_OFFSET_RANGE) if randomize_phase else 0.0
                agent_id = component.add_agent(pos, mesh_id, material_id, phase_offset=phase)
                agent_ids.append(agent_id)

        return agent_ids

    def bake_animation_to_atlas(
        self,
        component: CrowdComponent,
        skeleton: Skeleton,
        clips: list[AnimationClip],
        texture_format: TextureFormat = TextureFormat.FLOAT32,
    ) -> AnimationTextureAtlas:
        """Bake animation clips to texture atlas.

        Args:
            component: Crowd component to associate atlas with
            skeleton: Skeleton for the animations
            clips: Animation clips to bake
            texture_format: Texture format for baking

        Returns:
            Baked animation texture atlas
        """
        atlas = AnimationTextureAtlas()
        anim_names: dict[int, str] = {}

        for i, clip in enumerate(clips):
            texture = bake_clip_to_texture(clip, skeleton, texture_format)
            if atlas.add_clip(clip.name, texture):
                anim_names[i] = clip.name

        component.set_animation_atlas(atlas, anim_names)
        return atlas

    def get_stats(
        self,
        entity_components: list[tuple[Entity, CrowdComponent]],
    ) -> dict[str, Any]:
        """Get aggregate statistics for all crowds.

        Returns:
            Dictionary of statistics
        """
        total_agents = 0
        total_visible = 0
        total_batches = 0
        total_memory = 0

        for _, component in entity_components:
            total_agents += component.get_agent_count()
            total_visible += component.get_visible_count()
            total_batches += component.renderer.batch_count
            total_memory += component._instance_buffer.get_memory_size_bytes()

        return {
            "total_agents": total_agents,
            "total_visible": total_visible,
            "total_culled": self._total_agents_culled,
            "total_batches": total_batches,
            "total_memory_bytes": total_memory,
            "cull_distance": self._cull_distance,
            "lod_distances": self._lod_distances,
            "agents_processed_last_frame": self._total_agents_processed,
        }
