"""
RVO/ORCA collision avoidance for navigation agents.

Implements Reciprocal Velocity Obstacles (RVO) and Optimal Reciprocal
Collision Avoidance (ORCA) for real-time local avoidance between agents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .constants import (
    AvoidanceMode,
    DEFAULT_AVOIDANCE_DISTANCE,
    DEFAULT_AVOIDANCE_FORCE,
    DEFAULT_RVO_MAX_NEIGHBORS,
    DEFAULT_RVO_NEIGHBOR_DISTANCE,
    DEFAULT_RVO_TIME_HORIZON,
    DEFAULT_RVO_TIME_HORIZON_OBSTACLES,
    RVO_SPEED_SAMPLE_FACTORS,
    RVO_VELOCITY_SAMPLES,
)
from .navmesh import Vector3


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class AvoidanceAgent:
    """Agent participating in collision avoidance."""
    id: int
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    preferred_velocity: Vector3 = field(default_factory=Vector3)
    radius: float = 0.5
    max_speed: float = 5.0
    priority: float = 1.0  # Higher priority agents have right of way
    group_id: int = 0  # Agents in same group avoid less aggressively
    enabled: bool = True

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AvoidanceAgent):
            return NotImplemented
        return self.id == other.id


@dataclass
class AvoidanceObstacle:
    """Static obstacle for avoidance."""
    id: int
    vertices: List[Vector3] = field(default_factory=list)  # For line/polygon
    position: Vector3 = field(default_factory=Vector3)  # For circle
    radius: float = 0.0  # For circular obstacle
    is_convex: bool = True
    enabled: bool = True


@dataclass
class VelocityObstacle:
    """Velocity obstacle representation."""
    apex: Vector3
    left_leg: Vector3  # Direction of left leg
    right_leg: Vector3  # Direction of right leg
    is_collision: bool = False  # True if agents are already colliding


@dataclass
class HalfPlane:
    """Half-plane constraint for ORCA."""
    point: Vector3  # Point on the boundary
    normal: Vector3  # Normal pointing into the valid half-plane

    def contains(self, velocity: Vector3) -> bool:
        """Check if velocity is in the valid half-plane."""
        relative = velocity - self.point
        return relative.dot(self.normal) >= 0


@dataclass
class AvoidanceResult:
    """Result of avoidance calculation."""
    velocity: Vector3 = field(default_factory=Vector3)
    success: bool = True
    constraints_violated: int = 0
    nearby_agents: int = 0
    nearby_obstacles: int = 0


# =============================================================================
# RVO Implementation
# =============================================================================


class RVOAvoidance:
    """
    Reciprocal Velocity Obstacle avoidance.

    Implements the RVO algorithm for local collision avoidance
    between multiple agents.
    """

    def __init__(
        self,
        time_horizon: float = DEFAULT_RVO_TIME_HORIZON,
        neighbor_distance: float = DEFAULT_RVO_NEIGHBOR_DISTANCE,
        max_neighbors: int = DEFAULT_RVO_MAX_NEIGHBORS
    ) -> None:
        """Initialize RVO avoidance system."""
        self._time_horizon = time_horizon
        self._neighbor_distance = neighbor_distance
        self._max_neighbors = max_neighbors
        self._agents: Dict[int, AvoidanceAgent] = {}
        self._obstacles: Dict[int, AvoidanceObstacle] = {}
        self._next_agent_id = 0
        self._next_obstacle_id = 0

    @property
    def time_horizon(self) -> float:
        """Get time horizon for avoidance."""
        return self._time_horizon

    @time_horizon.setter
    def time_horizon(self, value: float) -> None:
        """Set time horizon for avoidance."""
        self._time_horizon = max(0.1, value)

    @property
    def neighbor_distance(self) -> float:
        """Get neighbor search distance."""
        return self._neighbor_distance

    @property
    def max_neighbors(self) -> int:
        """Get maximum number of neighbors to consider."""
        return self._max_neighbors

    @property
    def agent_count(self) -> int:
        """Get number of agents."""
        return len(self._agents)

    @property
    def obstacle_count(self) -> int:
        """Get number of obstacles."""
        return len(self._obstacles)

    def add_agent(self, agent: AvoidanceAgent) -> int:
        """Add agent to avoidance system."""
        if agent.id == 0:
            self._next_agent_id += 1
            agent.id = self._next_agent_id
        self._agents[agent.id] = agent
        return agent.id

    def remove_agent(self, agent_id: int) -> bool:
        """Remove agent from avoidance system."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    def get_agent(self, agent_id: int) -> Optional[AvoidanceAgent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def update_agent(
        self, agent_id: int,
        position: Optional[Vector3] = None,
        velocity: Optional[Vector3] = None,
        preferred_velocity: Optional[Vector3] = None
    ) -> bool:
        """Update agent state."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return False

        if position is not None:
            agent.position = position
        if velocity is not None:
            agent.velocity = velocity
        if preferred_velocity is not None:
            agent.preferred_velocity = preferred_velocity

        return True

    def add_obstacle(self, obstacle: AvoidanceObstacle) -> int:
        """Add static obstacle."""
        if obstacle.id == 0:
            self._next_obstacle_id += 1
            obstacle.id = self._next_obstacle_id
        self._obstacles[obstacle.id] = obstacle
        return obstacle.id

    def remove_obstacle(self, obstacle_id: int) -> bool:
        """Remove static obstacle."""
        if obstacle_id in self._obstacles:
            del self._obstacles[obstacle_id]
            return True
        return False

    def compute_new_velocity(self, agent_id: int) -> AvoidanceResult:
        """
        Compute collision-free velocity for agent.

        Args:
            agent_id: ID of the agent

        Returns:
            AvoidanceResult with new velocity
        """
        result = AvoidanceResult()

        agent = self._agents.get(agent_id)
        if agent is None or not agent.enabled:
            return result

        # Get nearby agents
        neighbors = self._get_neighbors(agent)
        result.nearby_agents = len(neighbors)

        # Get nearby obstacles
        nearby_obstacles = self._get_nearby_obstacles(agent)
        result.nearby_obstacles = len(nearby_obstacles)

        if not neighbors and not nearby_obstacles:
            result.velocity = agent.preferred_velocity
            return result

        # Compute velocity obstacles
        velocity_obstacles = []
        for neighbor in neighbors:
            vo = self._compute_velocity_obstacle(agent, neighbor)
            if vo is not None:
                velocity_obstacles.append(vo)

        # Sample velocities to find best one
        best_velocity = self._sample_velocities(
            agent, velocity_obstacles, nearby_obstacles
        )

        result.velocity = best_velocity
        return result

    def _get_neighbors(self, agent: AvoidanceAgent) -> List[AvoidanceAgent]:
        """Get neighboring agents within range."""
        neighbors = []
        range_sq = self._neighbor_distance * self._neighbor_distance

        for other in self._agents.values():
            if other.id == agent.id or not other.enabled:
                continue

            dist_sq = agent.position.distance_squared_to(other.position)
            if dist_sq < range_sq:
                neighbors.append(other)

        # Sort by distance and limit
        neighbors.sort(
            key=lambda n: agent.position.distance_squared_to(n.position)
        )
        return neighbors[:self._max_neighbors]

    def _get_nearby_obstacles(
        self, agent: AvoidanceAgent
    ) -> List[AvoidanceObstacle]:
        """Get obstacles within range."""
        result = []
        range_sq = self._neighbor_distance * self._neighbor_distance

        for obstacle in self._obstacles.values():
            if not obstacle.enabled:
                continue

            if obstacle.radius > 0:
                # Circular obstacle
                dist_sq = agent.position.distance_squared_to(obstacle.position)
                if dist_sq < range_sq:
                    result.append(obstacle)
            elif obstacle.vertices:
                # Line/polygon obstacle
                for vertex in obstacle.vertices:
                    if agent.position.distance_squared_to(vertex) < range_sq:
                        result.append(obstacle)
                        break

        return result

    def _compute_velocity_obstacle(
        self, agent: AvoidanceAgent, other: AvoidanceAgent
    ) -> Optional[VelocityObstacle]:
        """Compute velocity obstacle created by other agent."""
        relative_pos = other.position - agent.position
        dist = relative_pos.length()
        combined_radius = agent.radius + other.radius

        if dist < combined_radius:
            # Already colliding
            return VelocityObstacle(
                apex=Vector3(),
                left_leg=Vector3(-relative_pos.z, 0, relative_pos.x).normalized(),
                right_leg=Vector3(relative_pos.z, 0, -relative_pos.x).normalized(),
                is_collision=True
            )

        # Calculate VO cone
        # Apex is at the relative velocity of the other agent
        apex = other.velocity

        # Calculate leg directions
        leg_length = math.sqrt(dist * dist - combined_radius * combined_radius)
        sin_angle = combined_radius / dist
        cos_angle = leg_length / dist

        # Rotate relative position to get leg directions
        rel_normalized = relative_pos.normalized()

        left_leg = Vector3(
            rel_normalized.x * cos_angle - rel_normalized.z * sin_angle,
            0,
            rel_normalized.x * sin_angle + rel_normalized.z * cos_angle
        )

        right_leg = Vector3(
            rel_normalized.x * cos_angle + rel_normalized.z * sin_angle,
            0,
            -rel_normalized.x * sin_angle + rel_normalized.z * cos_angle
        )

        return VelocityObstacle(
            apex=apex,
            left_leg=left_leg,
            right_leg=right_leg,
            is_collision=False
        )

    def _sample_velocities(
        self, agent: AvoidanceAgent,
        velocity_obstacles: List[VelocityObstacle],
        obstacles: List[AvoidanceObstacle]
    ) -> Vector3:
        """Sample velocities to find best collision-free option."""
        preferred = agent.preferred_velocity

        # First check if preferred velocity is valid
        if self._is_velocity_valid(agent, preferred, velocity_obstacles, obstacles):
            return preferred

        # Sample velocities in a grid pattern
        best_velocity = Vector3()
        best_penalty = float('inf')

        for i in range(RVO_VELOCITY_SAMPLES):
            angle = 2 * math.pi * i / RVO_VELOCITY_SAMPLES
            for speed_factor in RVO_SPEED_SAMPLE_FACTORS:
                speed = agent.max_speed * speed_factor
                sample = Vector3(
                    math.cos(angle) * speed,
                    0,
                    math.sin(angle) * speed
                )

                if self._is_velocity_valid(agent, sample, velocity_obstacles, obstacles):
                    # Calculate penalty (distance from preferred)
                    penalty = (sample - preferred).length_squared()
                    if penalty < best_penalty:
                        best_penalty = penalty
                        best_velocity = sample

        return best_velocity

    def _is_velocity_valid(
        self, agent: AvoidanceAgent, velocity: Vector3,
        velocity_obstacles: List[VelocityObstacle],
        obstacles: List[AvoidanceObstacle]
    ) -> bool:
        """Check if velocity is collision-free."""
        for vo in velocity_obstacles:
            if self._is_in_velocity_obstacle(velocity, vo):
                return False

        # Check static obstacles
        for obstacle in obstacles:
            if self._velocity_hits_obstacle(agent, velocity, obstacle):
                return False

        return True

    def _is_in_velocity_obstacle(
        self, velocity: Vector3, vo: VelocityObstacle
    ) -> bool:
        """Check if velocity is inside velocity obstacle."""
        if vo.is_collision:
            # During collision, any velocity toward the obstacle is bad
            return velocity.dot(vo.left_leg.cross(vo.right_leg)) < 0

        # Check if velocity is inside the cone
        relative_vel = velocity - vo.apex

        left_cross = vo.left_leg.x * relative_vel.z - vo.left_leg.z * relative_vel.x
        right_cross = vo.right_leg.x * relative_vel.z - vo.right_leg.z * relative_vel.x

        # Inside if to the right of left leg and to the left of right leg
        return left_cross < 0 and right_cross > 0

    def _velocity_hits_obstacle(
        self, agent: AvoidanceAgent, velocity: Vector3,
        obstacle: AvoidanceObstacle
    ) -> bool:
        """Check if velocity will cause collision with obstacle."""
        if obstacle.radius > 0:
            # Circular obstacle
            future_pos = agent.position + velocity * self._time_horizon
            dist = future_pos.distance_to(obstacle.position)
            return dist < agent.radius + obstacle.radius

        # Line obstacle - simplified check
        return False

    def step(self, dt: float) -> None:
        """Advance simulation by timestep."""
        # Compute new velocities for all agents
        new_velocities: Dict[int, Vector3] = {}

        for agent_id, agent in self._agents.items():
            if not agent.enabled:
                continue
            result = self.compute_new_velocity(agent_id)
            new_velocities[agent_id] = result.velocity

        # Update all agents
        for agent_id, velocity in new_velocities.items():
            agent = self._agents[agent_id]
            agent.velocity = velocity
            agent.position = agent.position + velocity * dt


# =============================================================================
# ORCA Implementation
# =============================================================================


class ORCAAvoidance:
    """
    Optimal Reciprocal Collision Avoidance.

    Implements the ORCA algorithm which provides guaranteed
    collision-free velocities using half-plane constraints.
    """

    def __init__(
        self,
        time_horizon: float = DEFAULT_RVO_TIME_HORIZON,
        time_horizon_obstacles: float = DEFAULT_RVO_TIME_HORIZON_OBSTACLES,
        neighbor_distance: float = DEFAULT_RVO_NEIGHBOR_DISTANCE,
        max_neighbors: int = DEFAULT_RVO_MAX_NEIGHBORS
    ) -> None:
        """Initialize ORCA avoidance system."""
        self._time_horizon = time_horizon
        self._time_horizon_obstacles = time_horizon_obstacles
        self._neighbor_distance = neighbor_distance
        self._max_neighbors = max_neighbors
        self._agents: Dict[int, AvoidanceAgent] = {}
        self._obstacles: Dict[int, AvoidanceObstacle] = {}
        self._next_agent_id = 0
        self._next_obstacle_id = 0

    @property
    def time_horizon(self) -> float:
        """Get time horizon for agent avoidance."""
        return self._time_horizon

    @property
    def time_horizon_obstacles(self) -> float:
        """Get time horizon for obstacle avoidance."""
        return self._time_horizon_obstacles

    @property
    def agent_count(self) -> int:
        """Get number of agents."""
        return len(self._agents)

    def add_agent(self, agent: AvoidanceAgent) -> int:
        """Add agent to avoidance system."""
        if agent.id == 0:
            self._next_agent_id += 1
            agent.id = self._next_agent_id
        self._agents[agent.id] = agent
        return agent.id

    def remove_agent(self, agent_id: int) -> bool:
        """Remove agent from avoidance system."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    def get_agent(self, agent_id: int) -> Optional[AvoidanceAgent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def update_agent(
        self, agent_id: int,
        position: Optional[Vector3] = None,
        velocity: Optional[Vector3] = None,
        preferred_velocity: Optional[Vector3] = None
    ) -> bool:
        """Update agent state."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return False

        if position is not None:
            agent.position = position
        if velocity is not None:
            agent.velocity = velocity
        if preferred_velocity is not None:
            agent.preferred_velocity = preferred_velocity

        return True

    def add_obstacle(self, obstacle: AvoidanceObstacle) -> int:
        """Add static obstacle."""
        if obstacle.id == 0:
            self._next_obstacle_id += 1
            obstacle.id = self._next_obstacle_id
        self._obstacles[obstacle.id] = obstacle
        return obstacle.id

    def remove_obstacle(self, obstacle_id: int) -> bool:
        """Remove static obstacle."""
        if obstacle_id in self._obstacles:
            del self._obstacles[obstacle_id]
            return True
        return False

    def compute_new_velocity(self, agent_id: int) -> AvoidanceResult:
        """
        Compute collision-free velocity using ORCA.

        Args:
            agent_id: ID of the agent

        Returns:
            AvoidanceResult with new velocity
        """
        result = AvoidanceResult()

        agent = self._agents.get(agent_id)
        if agent is None or not agent.enabled:
            return result

        # Get nearby agents
        neighbors = self._get_neighbors(agent)
        result.nearby_agents = len(neighbors)

        if not neighbors:
            result.velocity = agent.preferred_velocity
            return result

        # Compute ORCA half-planes
        constraints: List[HalfPlane] = []

        for neighbor in neighbors:
            constraint = self._compute_orca_constraint(agent, neighbor)
            if constraint is not None:
                constraints.append(constraint)

        # Solve linear program to find best velocity
        result.velocity = self._solve_linear_program(
            agent, constraints
        )
        result.constraints_violated = sum(
            1 for c in constraints if not c.contains(result.velocity)
        )

        return result

    def _get_neighbors(self, agent: AvoidanceAgent) -> List[AvoidanceAgent]:
        """Get neighboring agents within range."""
        neighbors = []
        range_sq = self._neighbor_distance * self._neighbor_distance

        for other in self._agents.values():
            if other.id == agent.id or not other.enabled:
                continue

            dist_sq = agent.position.distance_squared_to(other.position)
            if dist_sq < range_sq:
                neighbors.append(other)

        neighbors.sort(
            key=lambda n: agent.position.distance_squared_to(n.position)
        )
        return neighbors[:self._max_neighbors]

    def _compute_orca_constraint(
        self, agent: AvoidanceAgent, other: AvoidanceAgent
    ) -> Optional[HalfPlane]:
        """Compute ORCA half-plane constraint for avoiding other agent."""
        relative_pos = other.position - agent.position
        relative_vel = agent.velocity - other.velocity
        dist_sq = relative_pos.length_squared()
        combined_radius = agent.radius + other.radius
        combined_radius_sq = combined_radius * combined_radius

        if dist_sq < combined_radius_sq:
            # Agents are colliding - generate emergency constraint
            dist = math.sqrt(dist_sq)
            if dist > 0.001:
                normal = (agent.position - other.position).normalized()
            else:
                normal = Vector3(1, 0, 0)

            return HalfPlane(
                point=agent.velocity,
                normal=normal
            )

        # Calculate the truncated VO apex
        inv_time_horizon = 1.0 / self._time_horizon
        relative_pos_scaled = relative_pos * inv_time_horizon

        # Calculate legs of the truncated VO
        dist = math.sqrt(dist_sq)
        leg_length = math.sqrt(dist_sq - combined_radius_sq)

        # Calculate the closest point on the VO boundary
        # to the relative velocity
        if relative_pos_scaled.length_squared() > 0:
            w = relative_vel - relative_pos_scaled
            w_length_sq = w.length_squared()

            dot_product1 = w.dot(relative_pos)

            if dot_product1 < 0 and dot_product1 * dot_product1 > combined_radius_sq * w_length_sq:
                # Project onto cut-off circle
                w_length = math.sqrt(w_length_sq)
                unit_w = w / w_length if w_length > 0 else Vector3()

                normal = unit_w
                u = unit_w * (combined_radius * inv_time_horizon - w_length)
            else:
                # Project onto legs
                leg = math.sqrt(dist_sq - combined_radius_sq)

                if (relative_pos.x * w.z - relative_pos.z * w.x) < 0:
                    # Left leg
                    direction = Vector3(
                        relative_pos.x * leg - relative_pos.z * combined_radius,
                        0,
                        relative_pos.x * combined_radius + relative_pos.z * leg
                    ) / dist_sq
                else:
                    # Right leg
                    direction = Vector3(
                        relative_pos.x * leg + relative_pos.z * combined_radius,
                        0,
                        -relative_pos.x * combined_radius + relative_pos.z * leg
                    ) / dist_sq

                dot_product2 = relative_vel.dot(direction)
                u = direction * dot_product2 - relative_vel
                normal = Vector3(-direction.z, 0, direction.x)
        else:
            normal = relative_pos.normalized()
            u = normal * (combined_radius * inv_time_horizon)

        # ORCA constraint: v + 0.5*u should be in the half-plane
        # with normal pointing away from the VO
        half_u = u * 0.5
        point = agent.velocity + half_u

        # Adjust by priority
        priority_factor = other.priority / (agent.priority + other.priority)
        point = agent.velocity + u * priority_factor

        return HalfPlane(point=point, normal=normal)

    def _solve_linear_program(
        self, agent: AvoidanceAgent, constraints: List[HalfPlane]
    ) -> Vector3:
        """Solve 2D linear program to find best velocity."""
        # Start with preferred velocity
        result = agent.preferred_velocity

        # Iteratively project onto constraints
        for constraint in constraints:
            if not constraint.contains(result):
                # Project result onto constraint boundary
                result = self._project_to_half_plane(result, constraint)

        # Clamp to max speed
        if result.length() > agent.max_speed:
            result = result.normalized() * agent.max_speed

        return result

    def _project_to_half_plane(
        self, velocity: Vector3, constraint: HalfPlane
    ) -> Vector3:
        """Project velocity onto the boundary of a half-plane."""
        relative = velocity - constraint.point
        dot = relative.dot(constraint.normal)

        if dot >= 0:
            return velocity  # Already in valid region

        # Project onto boundary
        return velocity - constraint.normal * dot

    def step(self, dt: float) -> None:
        """Advance simulation by timestep."""
        new_velocities: Dict[int, Vector3] = {}

        for agent_id, agent in self._agents.items():
            if not agent.enabled:
                continue
            result = self.compute_new_velocity(agent_id)
            new_velocities[agent_id] = result.velocity

        for agent_id, velocity in new_velocities.items():
            agent = self._agents[agent_id]
            agent.velocity = velocity
            agent.position = agent.position + velocity * dt


# =============================================================================
# Force-Based Avoidance
# =============================================================================


class ForceBasedAvoidance:
    """
    Simple force-based collision avoidance.

    Uses repulsion forces to push agents away from each other
    and obstacles. Simpler but less optimal than ORCA.
    """

    def __init__(
        self,
        avoidance_force: float = DEFAULT_AVOIDANCE_FORCE,
        avoidance_distance: float = DEFAULT_AVOIDANCE_DISTANCE,
        max_neighbors: int = DEFAULT_RVO_MAX_NEIGHBORS
    ) -> None:
        """Initialize force-based avoidance."""
        self._avoidance_force = avoidance_force
        self._avoidance_distance = avoidance_distance
        self._max_neighbors = max_neighbors
        self._agents: Dict[int, AvoidanceAgent] = {}
        self._obstacles: Dict[int, AvoidanceObstacle] = {}
        self._next_agent_id = 0
        self._next_obstacle_id = 0

    @property
    def avoidance_force(self) -> float:
        """Get avoidance force magnitude."""
        return self._avoidance_force

    @property
    def avoidance_distance(self) -> float:
        """Get avoidance trigger distance."""
        return self._avoidance_distance

    @property
    def agent_count(self) -> int:
        """Get number of agents."""
        return len(self._agents)

    def add_agent(self, agent: AvoidanceAgent) -> int:
        """Add agent to avoidance system."""
        if agent.id == 0:
            self._next_agent_id += 1
            agent.id = self._next_agent_id
        self._agents[agent.id] = agent
        return agent.id

    def remove_agent(self, agent_id: int) -> bool:
        """Remove agent from avoidance system."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    def get_agent(self, agent_id: int) -> Optional[AvoidanceAgent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def update_agent(
        self, agent_id: int,
        position: Optional[Vector3] = None,
        velocity: Optional[Vector3] = None,
        preferred_velocity: Optional[Vector3] = None
    ) -> bool:
        """Update agent state."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return False

        if position is not None:
            agent.position = position
        if velocity is not None:
            agent.velocity = velocity
        if preferred_velocity is not None:
            agent.preferred_velocity = preferred_velocity

        return True

    def add_obstacle(self, obstacle: AvoidanceObstacle) -> int:
        """Add static obstacle."""
        if obstacle.id == 0:
            self._next_obstacle_id += 1
            obstacle.id = self._next_obstacle_id
        self._obstacles[obstacle.id] = obstacle
        return obstacle.id

    def remove_obstacle(self, obstacle_id: int) -> bool:
        """Remove static obstacle."""
        if obstacle_id in self._obstacles:
            del self._obstacles[obstacle_id]
            return True
        return False

    def compute_new_velocity(self, agent_id: int) -> AvoidanceResult:
        """Compute collision-avoiding velocity using forces."""
        result = AvoidanceResult()

        agent = self._agents.get(agent_id)
        if agent is None or not agent.enabled:
            return result

        # Start with preferred velocity
        velocity = agent.preferred_velocity

        # Calculate repulsion forces from nearby agents
        agent_force = Vector3()
        neighbor_count = 0

        for other in self._agents.values():
            if other.id == agent.id or not other.enabled:
                continue

            to_agent = agent.position - other.position
            dist = to_agent.length()
            combined_radius = agent.radius + other.radius

            if dist < self._avoidance_distance and dist > 0.001:
                # Calculate repulsion force (stronger when closer)
                strength = 1.0 - (dist / self._avoidance_distance)
                strength = strength * strength  # Quadratic falloff

                force_dir = to_agent.normalized()
                agent_force = agent_force + force_dir * (strength * self._avoidance_force)
                neighbor_count += 1

        result.nearby_agents = neighbor_count

        # Calculate repulsion from obstacles
        obstacle_force = Vector3()
        obstacle_count = 0

        for obstacle in self._obstacles.values():
            if not obstacle.enabled:
                continue

            if obstacle.radius > 0:
                # Circular obstacle
                to_agent = agent.position - obstacle.position
                dist = to_agent.length() - obstacle.radius

                if dist < self._avoidance_distance and dist > 0.001:
                    strength = 1.0 - (dist / self._avoidance_distance)
                    force_dir = to_agent.normalized()
                    obstacle_force = obstacle_force + force_dir * (strength * self._avoidance_force)
                    obstacle_count += 1

        result.nearby_obstacles = obstacle_count

        # Combine forces
        total_force = agent_force + obstacle_force
        velocity = velocity + total_force

        # Clamp to max speed
        if velocity.length() > agent.max_speed:
            velocity = velocity.normalized() * agent.max_speed

        result.velocity = velocity
        return result

    def step(self, dt: float) -> None:
        """Advance simulation by timestep."""
        new_velocities: Dict[int, Vector3] = {}

        for agent_id, agent in self._agents.items():
            if not agent.enabled:
                continue
            result = self.compute_new_velocity(agent_id)
            new_velocities[agent_id] = result.velocity

        for agent_id, velocity in new_velocities.items():
            agent = self._agents[agent_id]
            agent.velocity = velocity
            agent.position = agent.position + velocity * dt


# =============================================================================
# Unified Avoidance System
# =============================================================================


class AvoidanceSystem:
    """
    Unified avoidance system supporting multiple algorithms.

    Provides a common interface for RVO, ORCA, and force-based avoidance.
    """

    def __init__(
        self,
        mode: AvoidanceMode = AvoidanceMode.ORCA,
        time_horizon: float = DEFAULT_RVO_TIME_HORIZON,
        neighbor_distance: float = DEFAULT_RVO_NEIGHBOR_DISTANCE,
        max_neighbors: int = DEFAULT_RVO_MAX_NEIGHBORS
    ) -> None:
        """Initialize avoidance system with specified mode."""
        self._mode = mode

        if mode == AvoidanceMode.RVO:
            self._impl: RVOAvoidance | ORCAAvoidance | ForceBasedAvoidance = RVOAvoidance(
                time_horizon, neighbor_distance, max_neighbors
            )
        elif mode == AvoidanceMode.ORCA:
            self._impl = ORCAAvoidance(
                time_horizon, DEFAULT_RVO_TIME_HORIZON_OBSTACLES,
                neighbor_distance, max_neighbors
            )
        elif mode == AvoidanceMode.FORCE_BASED:
            self._impl = ForceBasedAvoidance(
                DEFAULT_AVOIDANCE_FORCE, DEFAULT_AVOIDANCE_DISTANCE,
                max_neighbors
            )
        else:
            # NONE mode - use a dummy implementation
            self._impl = ForceBasedAvoidance(0, 0, 0)

    @property
    def mode(self) -> AvoidanceMode:
        """Get current avoidance mode."""
        return self._mode

    @property
    def agent_count(self) -> int:
        """Get number of agents."""
        return self._impl.agent_count

    def add_agent(self, agent: AvoidanceAgent) -> int:
        """Add agent to avoidance system."""
        return self._impl.add_agent(agent)

    def remove_agent(self, agent_id: int) -> bool:
        """Remove agent from avoidance system."""
        return self._impl.remove_agent(agent_id)

    def get_agent(self, agent_id: int) -> Optional[AvoidanceAgent]:
        """Get agent by ID."""
        return self._impl.get_agent(agent_id)

    def update_agent(
        self, agent_id: int,
        position: Optional[Vector3] = None,
        velocity: Optional[Vector3] = None,
        preferred_velocity: Optional[Vector3] = None
    ) -> bool:
        """Update agent state."""
        return self._impl.update_agent(
            agent_id, position, velocity, preferred_velocity
        )

    def add_obstacle(self, obstacle: AvoidanceObstacle) -> int:
        """Add static obstacle."""
        return self._impl.add_obstacle(obstacle)

    def remove_obstacle(self, obstacle_id: int) -> bool:
        """Remove static obstacle."""
        return self._impl.remove_obstacle(obstacle_id)

    def compute_new_velocity(self, agent_id: int) -> AvoidanceResult:
        """Compute collision-free velocity for agent."""
        if self._mode == AvoidanceMode.NONE:
            agent = self._impl.get_agent(agent_id)
            if agent:
                return AvoidanceResult(velocity=agent.preferred_velocity)
            return AvoidanceResult()

        return self._impl.compute_new_velocity(agent_id)

    def step(self, dt: float) -> None:
        """Advance simulation by timestep."""
        if self._mode != AvoidanceMode.NONE:
            self._impl.step(dt)
