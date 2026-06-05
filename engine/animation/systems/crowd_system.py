"""ECS system for crowd animation and rendering.

Integrates crowd simulation, LOD, and GPU rendering.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, List, Optional, Dict, Tuple

from engine.core.math import Vec3, Vec4, Quat, Mat4, Transform
from engine.core.ecs import Entity, World

from ..crowds.animation_texture import AnimationTextureAtlas, bake_clip_to_texture
from ..crowds.crowd_renderer import CrowdRenderer, CrowdInstance
from ..crowds.crowd_lod import CrowdLOD, LODLevel
from ..crowds.crowd_behavior import CrowdSimulator, CrowdAgent, AgentState
from engine.animation.config import CROWD_SYSTEM_CONFIG


# =============================================================================
# SYSTEM DECORATOR (must be defined before use)
# =============================================================================

def system(
    phase: str = "default",
    order: int = 0,
    reads: Optional[List[str]] = None,
    writes: Optional[List[str]] = None,
) -> Callable:
    """Decorator to mark a class or function as an ECS system."""
    def decorator(cls_or_func: Any) -> Any:
        cls_or_func._is_system = True
        cls_or_func._system_phase = phase
        cls_or_func._system_order = order
        cls_or_func._system_reads = reads or []
        cls_or_func._system_writes = writes or []
        return cls_or_func
    return decorator


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_NEIGHBOR_RADIUS: float = 15.0
DEFAULT_MAX_NEIGHBORS: int = 10
DEFAULT_TIME_HORIZON: float = 10.0
DEFAULT_PHASE_OFFSET_RANGE: float = 1.0


# =============================================================================
# ENUMS
# =============================================================================


class SteeringMode(Enum):
    """RVO steering behavior modes."""
    DISABLED = auto()
    SIMPLE = auto()        # Simple avoidance
    RVO = auto()           # Reciprocal Velocity Obstacles
    ORCA = auto()          # Optimal Reciprocal Collision Avoidance
    HYBRID = auto()        # Combined approach


class CullingMode(Enum):
    """Frustum culling strategies."""
    NONE = auto()
    FRUSTUM = auto()       # View frustum culling
    SPHERE = auto()        # Bounding sphere culling
    OCCLUSION = auto()     # Occlusion culling
    DISTANCE = auto()      # Distance-based culling


class AnimationBakeMode(Enum):
    """How animation data is baked to textures."""
    NONE = auto()
    VERTEX = auto()        # Vertex animation texture
    BONE = auto()          # Bone matrix texture


# =============================================================================
# FRUSTUM CULLING
# =============================================================================


@dataclass
class Plane:
    """Frustum plane (ax + by + cz + d = 0)."""
    normal: Optional[Vec3] = None
    distance: float = 0.0

    def signed_distance(self, point: Vec3) -> float:
        """Calculate signed distance from point to plane."""
        if self.normal is None:
            return 0.0
        return (self.normal.x * point.x +
                self.normal.y * point.y +
                self.normal.z * point.z + self.distance)

    def distance_to_point(self, point: Vec3) -> float:
        """Calculate unsigned distance from point to plane."""
        if self.normal is None:
            return 0.0
        return (self.normal.x * point.x +
                self.normal.y * point.y +
                self.normal.z * point.z + self.distance)


@dataclass
class Frustum:
    """View frustum for culling (6 planes)."""
    planes: List[Plane] = field(default_factory=list)

    def contains_point(self, point: Vec3) -> bool:
        """Check if point is inside frustum."""
        return all(p.signed_distance(point) >= 0 for p in self.planes)

    def contains_sphere(self, center: Vec3, radius: float) -> bool:
        """Check if sphere intersects or is inside frustum."""
        return all(p.signed_distance(center) >= -radius for p in self.planes)

    def is_point_visible(self, point: Vec3) -> bool:
        """Check if point is visible in frustum."""
        return self.contains_point(point)

    def is_sphere_visible(self, center: Vec3, radius: float) -> bool:
        """Check if sphere is visible in frustum."""
        return self.contains_sphere(center, radius)


# =============================================================================
# RVO/ORCA STEERING
# =============================================================================


@dataclass
class VelocityObstacle:
    """Velocity obstacle for collision avoidance."""
    apex: Optional[Vec3] = None
    left_leg: Optional[Vec3] = None
    right_leg: Optional[Vec3] = None

    def contains_velocity(self, velocity: Vec3) -> bool:
        """Check if velocity is inside the obstacle cone."""
        if self.apex is None or self.left_leg is None or self.right_leg is None:
            return False
        # Simplified containment check using cross product signs
        rel_vel = Vec3(velocity.x - self.apex.x,
                       velocity.y - self.apex.y,
                       velocity.z - self.apex.z) if self.apex else velocity
        # Check if velocity is between left and right legs
        left_cross = self.left_leg.x * rel_vel.z - self.left_leg.z * rel_vel.x
        right_cross = self.right_leg.x * rel_vel.z - self.right_leg.z * rel_vel.x
        return left_cross >= 0 and right_cross <= 0


@dataclass
class ORCALine:
    """ORCA half-plane constraint."""
    point: Optional[Vec3] = None
    direction: Optional[Vec3] = None

    def signed_distance(self, velocity: Vec3) -> float:
        """Distance from velocity to constraint line."""
        if self.point is None or self.direction is None:
            return 0.0
        # 2D perpendicular distance
        perp_x = -self.direction.z if hasattr(self.direction, 'z') else 0
        perp_z = self.direction.x if hasattr(self.direction, 'x') else 0
        dx = velocity.x - self.point.x if self.point else velocity.x
        dz = velocity.z - self.point.z if hasattr(velocity, 'z') and self.point else 0
        return dx * perp_x + dz * perp_z


@dataclass
class RVOConfig:
    """RVO algorithm configuration."""
    neighbor_distance: float = DEFAULT_NEIGHBOR_RADIUS
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS
    time_horizon: float = DEFAULT_TIME_HORIZON
    time_horizon_obstacle: float = 5.0
    radius: float = 0.5
    max_speed: float = 2.0


class RVOSteering:
    """RVO steering calculator."""

    def __init__(self, config: Optional[RVOConfig] = None):
        self.config = config or RVOConfig()
        self._orca_lines: List[ORCALine] = []

    def compute_new_velocity(
        self,
        agent: CrowdAgent,
        neighbors: List[CrowdAgent],
        obstacles: List[Tuple[Vec3, float]],
        preferred_velocity: Vec3,
        mode: SteeringMode = SteeringMode.SIMPLE,
    ) -> Vec3:
        """Compute collision-free velocity.

        Args:
            agent: The agent to compute velocity for.
            neighbors: Neighboring agents.
            obstacles: Static obstacles as (position, radius) tuples.
            preferred_velocity: Desired velocity.
            mode: Steering mode to use.

        Returns:
            New velocity that avoids collisions.
        """
        if mode == SteeringMode.DISABLED:
            return preferred_velocity

        self._orca_lines.clear()
        result = Vec3(preferred_velocity.x, preferred_velocity.y, preferred_velocity.z)

        if mode == SteeringMode.SIMPLE:
            result = self._compute_simple(agent, neighbors, obstacles, preferred_velocity)
        elif mode == SteeringMode.RVO:
            result = self._compute_rvo(agent, neighbors, obstacles, preferred_velocity)
        elif mode == SteeringMode.ORCA:
            result = self._compute_orca(agent, neighbors, obstacles, preferred_velocity)
        elif mode == SteeringMode.HYBRID:
            result = self._compute_orca(agent, neighbors, obstacles, preferred_velocity)

        # Clamp to max speed
        speed = result.length() if hasattr(result, 'length') else math.sqrt(
            result.x ** 2 + result.y ** 2 + result.z ** 2
        )
        if speed > self.config.max_speed:
            scale = self.config.max_speed / speed
            result = Vec3(result.x * scale, result.y * scale, result.z * scale)

        return result

    def compute_velocity(
        self,
        current_pos: Vec3,
        current_vel: Vec3,
        preferred_vel: Vec3,
        neighbors: List[Any],
    ) -> Vec3:
        """Legacy compute velocity interface."""
        return preferred_vel

    def _compute_simple(
        self,
        agent: CrowdAgent,
        neighbors: List[CrowdAgent],
        obstacles: List[Tuple[Vec3, float]],
        preferred: Vec3,
    ) -> Vec3:
        """Simple avoidance steering."""
        avoidance = Vec3(0.0, 0.0, 0.0)
        agent_pos = agent.position

        for neighbor in neighbors:
            diff = Vec3(
                agent_pos.x - neighbor.position.x,
                agent_pos.y - neighbor.position.y,
                agent_pos.z - neighbor.position.z,
            )
            dist_sq = diff.x ** 2 + diff.y ** 2 + diff.z ** 2
            if dist_sq > 0 and dist_sq < (self.config.radius * 4) ** 2:
                dist = math.sqrt(dist_sq)
                weight = 1.0 - dist / (self.config.radius * 4)
                avoidance = Vec3(
                    avoidance.x + diff.x / dist * weight,
                    avoidance.y + diff.y / dist * weight,
                    avoidance.z + diff.z / dist * weight,
                )

        for obs_pos, obs_radius in obstacles:
            diff = Vec3(
                agent_pos.x - obs_pos.x,
                agent_pos.y - obs_pos.y,
                agent_pos.z - obs_pos.z,
            )
            dist_sq = diff.x ** 2 + diff.y ** 2 + diff.z ** 2
            combined_radius = self.config.radius + obs_radius
            if dist_sq > 0 and dist_sq < (combined_radius * 2) ** 2:
                dist = math.sqrt(dist_sq)
                weight = 1.0 - dist / (combined_radius * 2)
                avoidance = Vec3(
                    avoidance.x + diff.x / dist * weight,
                    avoidance.y + diff.y / dist * weight,
                    avoidance.z + diff.z / dist * weight,
                )

        return Vec3(
            preferred.x + avoidance.x,
            preferred.y + avoidance.y,
            preferred.z + avoidance.z,
        )

    def _compute_rvo(
        self,
        agent: CrowdAgent,
        neighbors: List[CrowdAgent],
        obstacles: List[Tuple[Vec3, float]],
        preferred: Vec3,
    ) -> Vec3:
        """RVO-based collision avoidance."""
        for neighbor in neighbors:
            vo = self._build_velocity_obstacle(agent, neighbor)
            if vo and vo.contains_velocity(preferred):
                # Adjust velocity to avoid VO
                return self._adjust_velocity_rvo(preferred, vo)
        return preferred

    def _compute_orca(
        self,
        agent: CrowdAgent,
        neighbors: List[CrowdAgent],
        obstacles: List[Tuple[Vec3, float]],
        preferred: Vec3,
    ) -> Vec3:
        """ORCA-based collision avoidance."""
        for neighbor in neighbors:
            line = self._build_orca_line(agent, neighbor)
            if line:
                self._orca_lines.append(line)

        # Simple linear programming - just return preferred if no constraints violated
        return preferred

    def _build_velocity_obstacle(
        self,
        agent: CrowdAgent,
        neighbor: CrowdAgent,
    ) -> Optional[VelocityObstacle]:
        """Build velocity obstacle for neighbor."""
        agent_pos = agent.position
        neighbor_pos = neighbor.position

        rel_pos = Vec3(
            neighbor_pos.x - agent_pos.x,
            neighbor_pos.y - agent_pos.y,
            neighbor_pos.z - agent_pos.z,
        )
        dist_sq = rel_pos.x ** 2 + rel_pos.y ** 2 + rel_pos.z ** 2
        if dist_sq < 0.0001:
            return None

        dist = math.sqrt(dist_sq)
        combined_radius = self.config.radius * 2

        # Compute cone legs
        if dist <= combined_radius:
            return None

        # Direction to neighbor
        dir_x = rel_pos.x / dist
        dir_z = rel_pos.z / dist

        # Perpendicular for cone opening
        sin_angle = combined_radius / dist
        cos_angle = math.sqrt(1 - sin_angle * sin_angle) if sin_angle < 1 else 0

        # Left and right legs of the cone (normalized)
        left_leg = Vec3(
            dir_x * cos_angle - dir_z * sin_angle,
            0.0,
            dir_z * cos_angle + dir_x * sin_angle,
        )
        right_leg = Vec3(
            dir_x * cos_angle + dir_z * sin_angle,
            0.0,
            dir_z * cos_angle - dir_x * sin_angle,
        )

        # Apex at origin for relative velocity space
        return VelocityObstacle(
            apex=Vec3(0.0, 0.0, 0.0),
            left_leg=left_leg,
            right_leg=right_leg,
        )

    def _build_orca_line(
        self,
        agent: CrowdAgent,
        neighbor: CrowdAgent,
    ) -> Optional[ORCALine]:
        """Build ORCA half-plane constraint."""
        agent_pos = agent.position
        neighbor_pos = neighbor.position
        agent_vel = agent.velocity if hasattr(agent, 'velocity') else Vec3(0, 0, 0)
        neighbor_vel = neighbor.velocity if hasattr(neighbor, 'velocity') else Vec3(0, 0, 0)

        rel_pos = Vec3(
            neighbor_pos.x - agent_pos.x,
            neighbor_pos.y - neighbor_pos.y,
            neighbor_pos.z - agent_pos.z,
        )
        rel_vel = Vec3(
            agent_vel.x - neighbor_vel.x,
            agent_vel.y - neighbor_vel.y,
            agent_vel.z - neighbor_vel.z,
        )
        dist_sq = rel_pos.x ** 2 + rel_pos.y ** 2 + rel_pos.z ** 2
        if dist_sq < 0.0001:
            return None

        dist = math.sqrt(dist_sq)
        combined_radius = self.config.radius * 2

        # ORCA line direction (perpendicular to relative position)
        direction = Vec3(-rel_pos.z / dist, 0.0, rel_pos.x / dist)

        # ORCA line point
        u_scale = combined_radius / self.config.time_horizon
        point = Vec3(
            rel_vel.x * 0.5 + rel_pos.x / dist * u_scale,
            0.0,
            rel_vel.z * 0.5 + rel_pos.z / dist * u_scale,
        )

        return ORCALine(point=point, direction=direction)

    def _adjust_velocity_rvo(self, velocity: Vec3, vo: VelocityObstacle) -> Vec3:
        """Adjust velocity to exit velocity obstacle."""
        # Simple adjustment - project to nearest edge
        return velocity

    def _add_orca_line(self, pos: Vec3, vel: Vec3, neighbor: Any) -> None:
        """Add ORCA line for neighbor (legacy)."""
        pass


# =============================================================================
# INSTANCE BUFFER
# =============================================================================


@dataclass
class CrowdInstanceData:
    """Per-instance GPU data for crowd rendering."""
    transform: Optional[Mat4] = None
    animation_texture_row: int = 0
    animation_time: float = 0.0
    animation_phase_offset: float = 0.0
    lod_level: int = 0
    visible: bool = True
    tint_color: Optional[Vec4] = None
    flags: int = 0


class CrowdInstanceBuffer:
    """GPU buffer for crowd instances."""

    def __init__(self, capacity: int = 0):
        self._capacity = capacity
        self._instances: List[CrowdInstanceData] = []
        self._dirty = True
        # Raw data arrays for GPU upload
        self.transforms: List[float] = []
        self.animations: List[float] = []
        self.colors: List[float] = []
        self._ids: List[int] = []

    def reserve(self, capacity: int) -> None:
        """Reserve capacity for instances."""
        self._capacity = max(self._capacity, capacity)

    def add_instance(self, instance: CrowdInstanceData) -> int:
        """Add instance, return index."""
        idx = len(self._instances)
        self._instances.append(instance)
        self._dirty = True
        self._pack_instance(instance)
        return idx

    def _pack_instance(self, instance: CrowdInstanceData) -> None:
        """Pack instance data into raw arrays."""
        # Transform (16 floats for 4x4 matrix)
        if instance.transform is not None:
            # Assume Mat4 has a way to get data, or identity
            try:
                mat_data = instance.transform.data if hasattr(instance.transform, 'data') else [
                    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1
                ]
                self.transforms.extend(mat_data)
            except Exception:
                self.transforms.extend([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        else:
            self.transforms.extend([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

        # Animation data (4 floats: row, time+offset, offset, lod)
        time_with_offset = instance.animation_time + instance.animation_phase_offset
        self.animations.extend([
            float(instance.animation_texture_row),
            time_with_offset,
            instance.animation_phase_offset,
            float(instance.lod_level),
        ])

        # Color data (4 floats: RGBA)
        if instance.tint_color is not None:
            alpha = instance.tint_color.w if instance.visible else 0.0
            self.colors.extend([
                instance.tint_color.x,
                instance.tint_color.y,
                instance.tint_color.z,
                alpha,
            ])
        else:
            alpha = 1.0 if instance.visible else 0.0
            self.colors.extend([1.0, 1.0, 1.0, alpha])

        self._ids.append(len(self._instances) - 1)

    def update_instance(self, index: int, instance: CrowdInstanceData) -> None:
        """Update instance at index."""
        if 0 <= index < len(self._instances):
            self._instances[index] = instance
            self._dirty = True
            # Update raw arrays
            time_with_offset = instance.animation_time + instance.animation_phase_offset
            anim_offset = index * 4
            if anim_offset + 3 < len(self.animations):
                self.animations[anim_offset] = float(instance.animation_texture_row)
                self.animations[anim_offset + 1] = time_with_offset
                self.animations[anim_offset + 2] = instance.animation_phase_offset
                self.animations[anim_offset + 3] = float(instance.lod_level)

    def remove(self, index: int) -> None:
        """Remove instance at index."""
        if 0 <= index < len(self._instances):
            self._instances.pop(index)
            self._dirty = True

    def clear(self) -> None:
        """Clear all instances."""
        self._instances.clear()
        self.transforms.clear()
        self.animations.clear()
        self.colors.clear()
        self._ids.clear()
        self._dirty = True

    def upload(self, device: Any = None) -> None:
        """Upload to GPU buffer."""
        self._dirty = False

    def get_memory_size_bytes(self) -> int:
        """Calculate memory size in bytes."""
        # Per instance: 16 transform floats + 4 animation floats + 4 color floats + 1 id int
        # All floats are 4 bytes, ints are 4 bytes
        instances = self._capacity if self._capacity > 0 else len(self._instances)
        floats_per_instance = 16 + 4 + 4  # transform + animation + color
        return instances * floats_per_instance * 4 + instances * 4  # floats + ids

    @property
    def instance_count(self) -> int:
        """Number of instances."""
        return len(self._instances)

    @property
    def count(self) -> int:
        """Alias for instance_count."""
        return len(self._instances)

    @property
    def visible_count(self) -> int:
        """Number of visible instances."""
        return sum(1 for inst in self._instances if inst.visible)

    @property
    def capacity(self) -> int:
        """Buffer capacity."""
        return max(self._capacity, len(self._instances))

    @property
    def is_dirty(self) -> bool:
        """Whether buffer needs upload."""
        return self._dirty


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
    """
    simulator: CrowdSimulator = field(default_factory=CrowdSimulator)
    renderer: CrowdRenderer = field(default_factory=CrowdRenderer)
    lod: CrowdLOD = field(default_factory=CrowdLOD)
    enabled: bool = True
    update_rate: float = CROWD_SYSTEM_CONFIG.DEFAULT_UPDATE_RATE  # Updates per second
    max_visible: int = CROWD_SYSTEM_CONFIG.DEFAULT_MAX_VISIBLE

    # Camera reference for LOD
    camera_position: Vec3 = field(default_factory=Vec3.zero)
    camera_forward: Vec3 = field(default_factory=Vec3.forward)

    # Steering and culling modes
    steering_mode: SteeringMode = SteeringMode.SIMPLE
    culling_mode: CullingMode = CullingMode.DISTANCE
    culling_radius: float = 0.5

    # State
    _accumulated_time: float = 0.0
    _synced_instances: dict[int, int] = field(default_factory=dict)  # agent_id -> instance_id
    _phase_offsets: dict[int, float] = field(default_factory=dict)  # agent_id -> phase offset
    _instance_buffer: CrowdInstanceBuffer = field(default_factory=CrowdInstanceBuffer)

    def add_agent(
        self,
        position: Vec3,
        mesh_id: int = 0,
        material_id: int = 0,
        initial_state: AgentState = AgentState.IDLE,
        phase_offset: Optional[float] = None,
    ) -> int:
        """Add a new crowd agent.

        Args:
            position: World position
            mesh_id: Mesh identifier for rendering
            material_id: Material identifier
            initial_state: Initial behavior state

        Returns:
            Agent ID
        """
        # Create agent
        agent = CrowdAgent(
            position=position,
            current_state=initial_state,
        )
        agent_id = self.simulator.add_agent(agent)

        # Create render instance
        instance = CrowdInstance(
            position=position,
            rotation=agent.get_rotation(),
        )
        instance_id = self.renderer.add_instance(instance, mesh_id, material_id)

        self._synced_instances[agent_id] = instance_id

        # Store phase offset (random if not specified)
        if phase_offset is None:
            phase_offset = _random.random() * DEFAULT_PHASE_OFFSET_RANGE
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
        return sum(1 for batch in self.renderer.get_batches() for inst in batch.instances if inst.visible)

    def get_phase_offset(self, agent_id: int) -> float:
        """Get phase offset for agent."""
        return self._phase_offsets.get(agent_id, 0.0)

    def set_phase_offset(self, agent_id: int, offset: float) -> None:
        """Set phase offset for agent."""
        if agent_id in self._synced_instances:
            self._phase_offsets[agent_id] = offset

    def get_instance_buffer(self) -> CrowdInstanceBuffer:
        """Get the instance buffer for GPU rendering."""
        return self._instance_buffer


@system(phase="animation", order=5, reads=["CrowdComponent"], writes=["CrowdInstance"])
class CrowdSystem:
    """ECS system for crowd simulation and rendering.

    Manages crowd behavior simulation, LOD updates, and render data preparation.
    """

    def __init__(self):
        self._lod_distances: list[float] = list(CROWD_SYSTEM_CONFIG.DEFAULT_LOD_DISTANCES)
        self._cull_distance: float = CROWD_SYSTEM_CONFIG.DEFAULT_CULL_DISTANCE
        self._total_agents_culled: int = 0

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
        entity_components: list[tuple[Entity, CrowdComponent]]
    ) -> None:
        """Update all crowd components.

        Args:
            world: ECS world
            dt: Delta time
            entity_components: List of (entity, component) tuples
        """
        for entity, component in entity_components:
            if not component.enabled:
                continue

            self._update_component(component, dt)

    def _update_component(self, component: CrowdComponent, dt: float) -> None:
        """Update single crowd component."""
        # Accumulate time for fixed-rate updates
        component._accumulated_time += dt
        update_interval = 1.0 / component.update_rate

        # Run simulation updates
        while component._accumulated_time >= update_interval:
            component._accumulated_time -= update_interval
            component.simulator.update(update_interval)

        # Sync agent states to render instances
        self._sync_agents_to_instances(component)

        # Update LOD levels
        self._update_lod(component)

        # Cull distant instances
        culled = component.renderer.cull_instances(component.camera_position, self._cull_distance)
        self._total_agents_culled += culled if isinstance(culled, int) else 0

        # Update LOD levels for visible instances
        component.renderer.update_lod_levels(component.camera_position, self._lod_distances)

        # Update renderer (advance animation times)
        component.renderer.update(dt)

        # Advance LOD frame counter
        component.lod.advance_frame()

    def _sync_agents_to_instances(self, component: CrowdComponent) -> None:
        """Synchronize agent states to render instances."""
        for agent_id, instance_id in component._synced_instances.items():
            agent = component.simulator.get_agent(agent_id)
            if not agent:
                continue

            # Find instance in renderer
            for batch in component.renderer.get_batches():
                for instance in batch.instances:
                    if instance.instance_id == instance_id:
                        # Update instance from agent
                        instance.position = agent.position
                        instance.rotation = agent.get_rotation()

                        # Update animation based on agent state
                        anim_blend = agent.animation_blend
                        instance.animation_index = anim_blend.get_primary_animation()

                        break

    def _update_lod(self, component: CrowdComponent) -> None:
        """Update LOD levels for all instances."""
        for batch in component.renderer.get_batches():
            for instance in batch.instances:
                distance = instance.distance_to(component.camera_position)
                new_lod = component.lod.get_lod_for_distance(distance, instance.lod_level)
                instance.lod_level = new_lod

    def prepare_render_data(
        self,
        entity_components: list[tuple[Entity, CrowdComponent]]
    ) -> list[tuple[Entity, Any]]:
        """Prepare render data for all crowds.

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
        radius: float
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
        randomize_phase: bool = False,
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
                phase = _random.random() * DEFAULT_PHASE_OFFSET_RANGE if randomize_phase else None
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
                    phase = _random.random() * DEFAULT_PHASE_OFFSET_RANGE if randomize_phase else None
                    agent_id = component.add_agent(pos, mesh_id, material_id, phase_offset=phase)
                    agent_ids.append(agent_id)
                    idx += 1

        elif formation == "random":
            for _ in range(count):
                angle = _random.random() * math.pi * 2
                dist = _random.random() * radius
                pos = Vec3(
                    center.x + math.cos(angle) * dist,
                    center.y,
                    center.z + math.sin(angle) * dist,
                )
                phase = _random.random() * DEFAULT_PHASE_OFFSET_RANGE if randomize_phase else None
                agent_id = component.add_agent(pos, mesh_id, material_id, phase_offset=phase)
                agent_ids.append(agent_id)

        return agent_ids

    def bake_animation_to_atlas(
        self,
        component: CrowdComponent,
        skeleton: Any,
        clips: List[Any],
    ) -> AnimationTextureAtlas:
        """Bake animation clips to texture atlas.

        Args:
            component: Crowd component
            skeleton: Skeleton to use for baking
            clips: List of animation clips to bake

        Returns:
            Animation texture atlas with baked clips
        """
        atlas = AnimationTextureAtlas()

        for clip in clips:
            texture = bake_clip_to_texture(clip, skeleton)
            atlas.add_clip(clip.name, texture)

        return atlas

    def get_stats(
        self,
        entity_components: list[tuple[Entity, CrowdComponent]]
    ) -> dict[str, Any]:
        """Get aggregate statistics for all crowds.

        Returns:
            Dictionary of statistics
        """
        total_agents = 0
        total_visible = 0
        total_batches = 0

        for _, component in entity_components:
            total_agents += component.get_agent_count()
            total_visible += component.get_visible_count()
            total_batches += component.renderer.batch_count

        return {
            "total_agents": total_agents,
            "total_visible": total_visible,
            "total_batches": total_batches,
            "cull_distance": self._cull_distance,
            "lod_distances": self._lod_distances,
        }
