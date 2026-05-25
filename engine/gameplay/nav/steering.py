"""
Steering behaviors for autonomous agents.

Implements classic steering behaviors including Seek, Flee, Arrive,
Pursue, Evade, Wander, and flocking behaviors (Separation, Alignment, Cohesion).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .constants import (
    DEFAULT_ALIGNMENT_WEIGHT,
    DEFAULT_ARRIVE_SLOW_RADIUS,
    DEFAULT_ARRIVE_STOP_RADIUS,
    DEFAULT_ARRIVE_WEIGHT,
    DEFAULT_COHESION_WEIGHT,
    DEFAULT_EVADE_WEIGHT,
    DEFAULT_FLEE_WEIGHT,
    DEFAULT_FLOCKING_WEIGHT,
    DEFAULT_MAX_ACCELERATION,
    DEFAULT_MAX_FORCE,
    DEFAULT_MAX_SPEED,
    DEFAULT_NEIGHBOR_DISTANCE,
    DEFAULT_OBSTACLE_AVOIDANCE_WEIGHT,
    DEFAULT_PATH_FOLLOWING_WEIGHT,
    DEFAULT_PURSUE_WEIGHT,
    DEFAULT_SEEK_WEIGHT,
    DEFAULT_SEPARATION_DISTANCE,
    DEFAULT_SEPARATION_WEIGHT,
    DEFAULT_WALL_FOLLOWING_WEIGHT,
    DEFAULT_WANDER_DISTANCE,
    DEFAULT_WANDER_JITTER,
    DEFAULT_WANDER_RADIUS,
    DEFAULT_WANDER_WEIGHT,
    OBSTACLE_AVOIDANCE_BRAKING_WEIGHT,
    ZERO_LENGTH_THRESHOLD,
    SteeringBehavior,
)
from .navmesh import Vector3


# =============================================================================
# Steering Agent
# =============================================================================


@dataclass
class SteeringAgent:
    """
    Agent with steering capabilities.

    Maintains position, velocity, and other kinematic properties
    needed for steering behaviors.
    """
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    heading: Vector3 = field(default_factory=lambda: Vector3(0, 0, 1))
    side: Vector3 = field(default_factory=lambda: Vector3(1, 0, 0))

    mass: float = 1.0
    max_speed: float = DEFAULT_MAX_SPEED
    max_force: float = DEFAULT_MAX_FORCE
    max_turn_rate: float = math.pi  # radians per second

    # Agent dimensions for avoidance
    radius: float = 0.5
    height: float = 2.0

    # ID for identification in groups
    id: int = 0

    def __post_init__(self) -> None:
        """Ensure heading and side are normalized."""
        if self.heading.length_squared() > 0:
            self.heading = self.heading.normalized()
        if self.side.length_squared() > 0:
            self.side = self.side.normalized()

    def speed(self) -> float:
        """Get current speed."""
        return self.velocity.length()

    def update(self, steering_force: Vector3, dt: float) -> None:
        """
        Update agent based on steering force.

        Args:
            steering_force: Force to apply
            dt: Delta time in seconds
        """
        # Limit force
        if steering_force.length() > self.max_force:
            steering_force = steering_force.normalized() * self.max_force

        # Calculate acceleration
        acceleration = steering_force / self.mass

        # Update velocity
        self.velocity = self.velocity + acceleration * dt

        # Limit speed
        if self.velocity.length() > self.max_speed:
            self.velocity = self.velocity.normalized() * self.max_speed

        # Update position
        self.position = self.position + self.velocity * dt

        # Update heading if moving
        if self.velocity.length_squared() > ZERO_LENGTH_THRESHOLD:
            self.heading = self.velocity.normalized()
            # Side is perpendicular to heading in XZ plane
            self.side = Vector3(-self.heading.z, 0, self.heading.x)

    def local_to_world(self, local: Vector3) -> Vector3:
        """Convert local coordinates to world coordinates."""
        return Vector3(
            self.heading.x * local.z + self.side.x * local.x + self.position.x,
            self.position.y + local.y,
            self.heading.z * local.z + self.side.z * local.x + self.position.z
        )

    def world_to_local(self, world: Vector3) -> Vector3:
        """Convert world coordinates to local coordinates."""
        tx = world.x - self.position.x
        ty = world.y - self.position.y
        tz = world.z - self.position.z

        return Vector3(
            self.side.x * tx + self.side.z * tz,
            ty,
            self.heading.x * tx + self.heading.z * tz
        )


# =============================================================================
# Steering Behaviors
# =============================================================================


def seek(agent: SteeringAgent, target: Vector3) -> Vector3:
    """
    Seek behavior - move toward target at maximum speed.

    Args:
        agent: The steering agent
        target: Target position to seek

    Returns:
        Steering force toward target
    """
    desired_velocity = (target - agent.position).normalized() * agent.max_speed
    return desired_velocity - agent.velocity


def flee(agent: SteeringAgent, target: Vector3) -> Vector3:
    """
    Flee behavior - move away from target at maximum speed.

    Args:
        agent: The steering agent
        target: Position to flee from

    Returns:
        Steering force away from target
    """
    desired_velocity = (agent.position - target).normalized() * agent.max_speed
    return desired_velocity - agent.velocity


def arrive(
    agent: SteeringAgent, target: Vector3,
    slow_radius: float = DEFAULT_ARRIVE_SLOW_RADIUS,
    stop_radius: float = DEFAULT_ARRIVE_STOP_RADIUS
) -> Vector3:
    """
    Arrive behavior - move toward target and slow down as we approach.

    Args:
        agent: The steering agent
        target: Target position to arrive at
        slow_radius: Distance at which to start slowing down
        stop_radius: Distance at which to stop

    Returns:
        Steering force toward target with deceleration
    """
    to_target = target - agent.position
    distance = to_target.length()

    if distance < stop_radius:
        return -agent.velocity  # Stop

    if distance < slow_radius:
        # Scale speed based on distance
        target_speed = agent.max_speed * (distance - stop_radius) / (slow_radius - stop_radius)
    else:
        target_speed = agent.max_speed

    if distance < ZERO_LENGTH_THRESHOLD:
        return Vector3()

    desired_velocity = to_target.normalized() * target_speed
    return desired_velocity - agent.velocity


def pursue(
    agent: SteeringAgent, target: SteeringAgent,
    max_prediction_time: float = 1.0
) -> Vector3:
    """
    Pursue behavior - predict target's future position and seek it.

    Args:
        agent: The steering agent
        target: Target agent to pursue
        max_prediction_time: Maximum time to predict ahead

    Returns:
        Steering force toward predicted target position
    """
    to_target = target.position - agent.position
    distance = to_target.length()

    # Calculate look-ahead time based on distance and speeds
    relative_heading = agent.heading.dot(target.heading)

    # If target is ahead and facing us, use seek
    if relative_heading < -0.95 and to_target.dot(agent.heading) > 0:
        return seek(agent, target.position)

    # Calculate prediction time
    combined_speed = agent.speed() + target.speed()
    if combined_speed < ZERO_LENGTH_THRESHOLD:
        prediction_time = max_prediction_time
    else:
        prediction_time = min(distance / combined_speed, max_prediction_time)

    # Predict future position
    predicted_position = target.position + target.velocity * prediction_time

    return seek(agent, predicted_position)


def evade(
    agent: SteeringAgent, target: SteeringAgent,
    max_prediction_time: float = 1.0
) -> Vector3:
    """
    Evade behavior - predict target's future position and flee from it.

    Args:
        agent: The steering agent
        target: Target agent to evade
        max_prediction_time: Maximum time to predict ahead

    Returns:
        Steering force away from predicted target position
    """
    to_target = target.position - agent.position
    distance = to_target.length()

    # Calculate prediction time
    combined_speed = agent.speed() + target.speed()
    if combined_speed < ZERO_LENGTH_THRESHOLD:
        prediction_time = max_prediction_time
    else:
        prediction_time = min(distance / combined_speed, max_prediction_time)

    # Predict future position
    predicted_position = target.position + target.velocity * prediction_time

    return flee(agent, predicted_position)


@dataclass
class WanderState:
    """State for wander behavior."""
    wander_target: Vector3 = field(default_factory=lambda: Vector3(0, 0, 1))


def wander(
    agent: SteeringAgent, state: WanderState,
    radius: float = DEFAULT_WANDER_RADIUS,
    distance: float = DEFAULT_WANDER_DISTANCE,
    jitter: float = DEFAULT_WANDER_JITTER,
    dt: float = 0.016
) -> Vector3:
    """
    Wander behavior - random steering for natural-looking movement.

    Args:
        agent: The steering agent
        state: WanderState containing persistent wander target
        radius: Radius of the wander circle
        distance: Distance of wander circle from agent
        jitter: Amount of random displacement per second
        dt: Delta time for jitter calculation

    Returns:
        Steering force for wandering movement
    """
    # Add random displacement to wander target
    jitter_amount = jitter * dt
    state.wander_target = state.wander_target + Vector3(
        random.uniform(-1, 1) * jitter_amount,
        0,
        random.uniform(-1, 1) * jitter_amount
    )

    # Project onto circle
    state.wander_target = state.wander_target.normalized() * radius

    # Project circle in front of agent
    target_local = Vector3(
        state.wander_target.x,
        0,
        state.wander_target.z + distance
    )

    # Convert to world coordinates
    target_world = agent.local_to_world(target_local)

    return target_world - agent.position


def separation(
    agent: SteeringAgent, neighbors: List[SteeringAgent],
    separation_distance: float = DEFAULT_SEPARATION_DISTANCE
) -> Vector3:
    """
    Separation behavior - steer away from nearby neighbors.

    Args:
        agent: The steering agent
        neighbors: List of neighboring agents
        separation_distance: Distance within which to separate

    Returns:
        Steering force away from neighbors
    """
    steering = Vector3()
    count = 0

    for neighbor in neighbors:
        if neighbor.id == agent.id:
            continue

        to_agent = agent.position - neighbor.position
        distance = to_agent.length()

        if 0 < distance < separation_distance:
            # Weight by inverse distance
            steering = steering + to_agent.normalized() / distance
            count += 1

    if count > 0:
        steering = steering / count

    return steering


def alignment(
    agent: SteeringAgent, neighbors: List[SteeringAgent],
    neighbor_distance: float = DEFAULT_NEIGHBOR_DISTANCE
) -> Vector3:
    """
    Alignment behavior - steer to match neighbors' heading.

    Args:
        agent: The steering agent
        neighbors: List of neighboring agents
        neighbor_distance: Distance within which to align

    Returns:
        Steering force toward average neighbor heading
    """
    average_heading = Vector3()
    count = 0

    for neighbor in neighbors:
        if neighbor.id == agent.id:
            continue

        distance = agent.position.distance_to(neighbor.position)
        if distance < neighbor_distance:
            average_heading = average_heading + neighbor.heading
            count += 1

    if count > 0:
        average_heading = average_heading / count
        return average_heading - agent.heading

    return Vector3()


def cohesion(
    agent: SteeringAgent, neighbors: List[SteeringAgent],
    neighbor_distance: float = DEFAULT_NEIGHBOR_DISTANCE
) -> Vector3:
    """
    Cohesion behavior - steer toward center of mass of neighbors.

    Args:
        agent: The steering agent
        neighbors: List of neighboring agents
        neighbor_distance: Distance within which to cohere

    Returns:
        Steering force toward neighbor center of mass
    """
    center_of_mass = Vector3()
    count = 0

    for neighbor in neighbors:
        if neighbor.id == agent.id:
            continue

        distance = agent.position.distance_to(neighbor.position)
        if distance < neighbor_distance:
            center_of_mass = center_of_mass + neighbor.position
            count += 1

    if count > 0:
        center_of_mass = center_of_mass / count
        return seek(agent, center_of_mass)

    return Vector3()


def flocking(
    agent: SteeringAgent, neighbors: List[SteeringAgent],
    separation_weight: float = DEFAULT_SEPARATION_WEIGHT,
    alignment_weight: float = DEFAULT_ALIGNMENT_WEIGHT,
    cohesion_weight: float = DEFAULT_COHESION_WEIGHT,
    separation_distance: float = DEFAULT_SEPARATION_DISTANCE,
    neighbor_distance: float = DEFAULT_NEIGHBOR_DISTANCE
) -> Vector3:
    """
    Flocking behavior - combination of separation, alignment, and cohesion.

    Args:
        agent: The steering agent
        neighbors: List of neighboring agents
        separation_weight: Weight for separation behavior
        alignment_weight: Weight for alignment behavior
        cohesion_weight: Weight for cohesion behavior
        separation_distance: Distance for separation calculation
        neighbor_distance: Distance for alignment/cohesion calculation

    Returns:
        Combined steering force for flocking
    """
    sep = separation(agent, neighbors, separation_distance) * separation_weight
    ali = alignment(agent, neighbors, neighbor_distance) * alignment_weight
    coh = cohesion(agent, neighbors, neighbor_distance) * cohesion_weight

    return sep + ali + coh


def obstacle_avoidance(
    agent: SteeringAgent, obstacles: List[Tuple[Vector3, float]],
    detection_length: float = 5.0
) -> Vector3:
    """
    Obstacle avoidance behavior - steer away from nearby obstacles.

    Args:
        agent: The steering agent
        obstacles: List of (position, radius) tuples
        detection_length: How far ahead to detect obstacles

    Returns:
        Steering force to avoid obstacles
    """
    # Detection box length based on speed
    box_length = agent.radius + detection_length * (agent.speed() / agent.max_speed)

    closest_obstacle: Optional[Tuple[Vector3, float]] = None
    closest_dist = float('inf')
    closest_local = Vector3()

    for obs_pos, obs_radius in obstacles:
        # Transform to local space
        local_pos = agent.world_to_local(obs_pos)

        # Check if obstacle is in front
        if local_pos.z <= 0:
            continue

        # Expand radius by agent radius
        expanded_radius = obs_radius + agent.radius

        # Check if within detection box width
        if abs(local_pos.x) >= expanded_radius:
            continue

        # Check if obstacle overlaps detection box
        sqrt_part = expanded_radius * expanded_radius - local_pos.x * local_pos.x
        if sqrt_part < 0:
            continue

        intersection_z = local_pos.z - math.sqrt(sqrt_part)

        if 0 < intersection_z < box_length and intersection_z < closest_dist:
            closest_dist = intersection_z
            closest_obstacle = (obs_pos, obs_radius)
            closest_local = local_pos

    if closest_obstacle is None:
        return Vector3()

    # Calculate avoidance force
    multiplier = 1.0 + (box_length - closest_local.z) / box_length

    # Lateral force
    lateral_force = (closest_obstacle[1] + agent.radius - closest_local.x) * multiplier

    # Braking force
    braking_force = (closest_obstacle[1] - closest_local.z) * OBSTACLE_AVOIDANCE_BRAKING_WEIGHT

    # Convert to world coordinates
    return Vector3(
        agent.side.x * lateral_force + agent.heading.x * braking_force,
        0,
        agent.side.z * lateral_force + agent.heading.z * braking_force
    )


def wall_following(
    agent: SteeringAgent, walls: List[Tuple[Vector3, Vector3]],
    detection_distance: float = 3.0
) -> Vector3:
    """
    Wall following behavior - steer parallel to nearby walls.

    Args:
        agent: The steering agent
        walls: List of (start, end) wall segments
        detection_distance: How far to detect walls

    Returns:
        Steering force to follow walls
    """
    # Cast feelers
    feelers = [
        agent.position + agent.heading * detection_distance,
        agent.position + (agent.heading + agent.side * 0.5).normalized() * detection_distance * 0.5,
        agent.position + (agent.heading - agent.side * 0.5).normalized() * detection_distance * 0.5
    ]

    closest_dist = float('inf')
    closest_wall: Optional[Tuple[Vector3, Vector3]] = None
    closest_point = Vector3()
    closest_feeler_idx = 0

    for feeler_idx, feeler in enumerate(feelers):
        for wall_start, wall_end in walls:
            # Check intersection
            intersection = _line_intersection(agent.position, feeler, wall_start, wall_end)
            if intersection is not None:
                dist = agent.position.distance_to(intersection)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_wall = (wall_start, wall_end)
                    closest_point = intersection
                    closest_feeler_idx = feeler_idx

    if closest_wall is None:
        return Vector3()

    # Calculate wall normal
    wall_vec = closest_wall[1] - closest_wall[0]
    wall_normal = Vector3(-wall_vec.z, 0, wall_vec.x).normalized()

    # Ensure normal points toward agent
    if wall_normal.dot(agent.position - closest_point) < 0:
        wall_normal = -wall_normal

    # Calculate overshoot
    overshoot = feelers[closest_feeler_idx] - closest_point

    return wall_normal * overshoot.length()


def _line_intersection(
    p1: Vector3, p2: Vector3, p3: Vector3, p4: Vector3
) -> Optional[Vector3]:
    """Calculate intersection point of two line segments in XZ plane."""
    d1 = p2 - p1
    d2 = p4 - p3
    d3 = p1 - p3

    denom = d1.x * d2.z - d1.z * d2.x
    if abs(denom) < ZERO_LENGTH_THRESHOLD:
        return None

    t = (d2.x * d3.z - d2.z * d3.x) / denom
    u = (d1.x * d3.z - d1.z * d3.x) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        return p1 + d1 * t

    return None


def path_following(
    agent: SteeringAgent, path: List[Vector3],
    prediction_distance: float = 1.0,
    path_offset: float = 0.5
) -> Vector3:
    """
    Path following behavior - follow a predefined path.

    Args:
        agent: The steering agent
        path: List of waypoints to follow
        prediction_distance: How far ahead to predict position
        path_offset: Distance from path to trigger steering

    Returns:
        Steering force to follow the path
    """
    if len(path) < 2:
        return Vector3()

    # Predict future position
    future_pos = agent.position + agent.velocity.normalized() * prediction_distance

    # Find closest point on path
    closest_point = path[0]
    closest_dist = float('inf')
    closest_segment = 0

    for i in range(len(path) - 1):
        segment_start = path[i]
        segment_end = path[i + 1]

        point = _closest_point_on_segment(future_pos, segment_start, segment_end)
        dist = future_pos.distance_to(point)

        if dist < closest_dist:
            closest_dist = dist
            closest_point = point
            closest_segment = i

    # If too far from path, seek toward it
    if closest_dist > path_offset:
        return seek(agent, closest_point)

    # Otherwise, seek toward a point ahead on the path
    target_idx = min(closest_segment + 1, len(path) - 1)
    return seek(agent, path[target_idx])


def _closest_point_on_segment(point: Vector3, seg_start: Vector3, seg_end: Vector3) -> Vector3:
    """Find closest point on line segment to given point."""
    segment = seg_end - seg_start
    seg_len_sq = segment.length_squared()

    if seg_len_sq < ZERO_LENGTH_THRESHOLD:
        return seg_start

    t = max(0, min(1, (point - seg_start).dot(segment) / seg_len_sq))
    return seg_start + segment * t


# =============================================================================
# Steering Manager
# =============================================================================


@dataclass
class SteeringWeight:
    """Weight configuration for a steering behavior."""
    behavior: SteeringBehavior
    weight: float
    enabled: bool = True


class SteeringManager:
    """
    Manager for combining multiple steering behaviors.

    Allows configuration of behavior weights and provides
    methods for calculating combined steering forces.
    """

    def __init__(self) -> None:
        """Initialize steering manager with default weights."""
        self._weights: Dict[SteeringBehavior, SteeringWeight] = {}
        self._wander_states: Dict[int, WanderState] = {}

        # Set default weights
        self._weights[SteeringBehavior.SEEK] = SteeringWeight(
            SteeringBehavior.SEEK, DEFAULT_SEEK_WEIGHT
        )
        self._weights[SteeringBehavior.FLEE] = SteeringWeight(
            SteeringBehavior.FLEE, DEFAULT_FLEE_WEIGHT
        )
        self._weights[SteeringBehavior.ARRIVE] = SteeringWeight(
            SteeringBehavior.ARRIVE, DEFAULT_ARRIVE_WEIGHT
        )
        self._weights[SteeringBehavior.PURSUE] = SteeringWeight(
            SteeringBehavior.PURSUE, DEFAULT_PURSUE_WEIGHT
        )
        self._weights[SteeringBehavior.EVADE] = SteeringWeight(
            SteeringBehavior.EVADE, DEFAULT_EVADE_WEIGHT
        )
        self._weights[SteeringBehavior.WANDER] = SteeringWeight(
            SteeringBehavior.WANDER, DEFAULT_WANDER_WEIGHT
        )
        self._weights[SteeringBehavior.SEPARATION] = SteeringWeight(
            SteeringBehavior.SEPARATION, DEFAULT_SEPARATION_WEIGHT
        )
        self._weights[SteeringBehavior.ALIGNMENT] = SteeringWeight(
            SteeringBehavior.ALIGNMENT, DEFAULT_ALIGNMENT_WEIGHT
        )
        self._weights[SteeringBehavior.COHESION] = SteeringWeight(
            SteeringBehavior.COHESION, DEFAULT_COHESION_WEIGHT
        )
        self._weights[SteeringBehavior.OBSTACLE_AVOIDANCE] = SteeringWeight(
            SteeringBehavior.OBSTACLE_AVOIDANCE, DEFAULT_OBSTACLE_AVOIDANCE_WEIGHT
        )
        self._weights[SteeringBehavior.WALL_FOLLOWING] = SteeringWeight(
            SteeringBehavior.WALL_FOLLOWING, DEFAULT_WALL_FOLLOWING_WEIGHT
        )
        self._weights[SteeringBehavior.PATH_FOLLOWING] = SteeringWeight(
            SteeringBehavior.PATH_FOLLOWING, DEFAULT_PATH_FOLLOWING_WEIGHT
        )
        self._weights[SteeringBehavior.FLOCKING] = SteeringWeight(
            SteeringBehavior.FLOCKING, DEFAULT_FLOCKING_WEIGHT
        )

    def set_weight(self, behavior: SteeringBehavior, weight: float) -> None:
        """Set weight for a behavior."""
        if behavior in self._weights:
            self._weights[behavior].weight = weight
        else:
            self._weights[behavior] = SteeringWeight(behavior, weight)

    def get_weight(self, behavior: SteeringBehavior) -> float:
        """Get weight for a behavior."""
        if behavior in self._weights:
            return self._weights[behavior].weight
        return 0.0

    def enable_behavior(self, behavior: SteeringBehavior) -> None:
        """Enable a behavior."""
        if behavior in self._weights:
            self._weights[behavior].enabled = True

    def disable_behavior(self, behavior: SteeringBehavior) -> None:
        """Disable a behavior."""
        if behavior in self._weights:
            self._weights[behavior].enabled = False

    def is_enabled(self, behavior: SteeringBehavior) -> bool:
        """Check if behavior is enabled."""
        if behavior in self._weights:
            return self._weights[behavior].enabled
        return False

    def get_wander_state(self, agent_id: int) -> WanderState:
        """Get or create wander state for agent."""
        if agent_id not in self._wander_states:
            self._wander_states[agent_id] = WanderState()
        return self._wander_states[agent_id]

    def calculate_weighted_sum(
        self, agent: SteeringAgent,
        seek_target: Optional[Vector3] = None,
        flee_target: Optional[Vector3] = None,
        arrive_target: Optional[Vector3] = None,
        pursue_target: Optional[SteeringAgent] = None,
        evade_target: Optional[SteeringAgent] = None,
        neighbors: Optional[List[SteeringAgent]] = None,
        obstacles: Optional[List[Tuple[Vector3, float]]] = None,
        walls: Optional[List[Tuple[Vector3, Vector3]]] = None,
        path: Optional[List[Vector3]] = None,
        dt: float = 0.016
    ) -> Vector3:
        """
        Calculate combined steering force using weighted sum.

        Args:
            agent: The steering agent
            seek_target: Target for seek behavior
            flee_target: Target for flee behavior
            arrive_target: Target for arrive behavior
            pursue_target: Target agent for pursue behavior
            evade_target: Target agent for evade behavior
            neighbors: Neighboring agents for flocking
            obstacles: Obstacles for avoidance
            walls: Walls for wall following
            path: Path for path following
            dt: Delta time

        Returns:
            Combined steering force
        """
        force = Vector3()
        neighbors = neighbors or []
        obstacles = obstacles or []
        walls = walls or []

        # Seek
        if seek_target and self._is_active(SteeringBehavior.SEEK):
            force = force + seek(agent, seek_target) * self._weights[SteeringBehavior.SEEK].weight

        # Flee
        if flee_target and self._is_active(SteeringBehavior.FLEE):
            force = force + flee(agent, flee_target) * self._weights[SteeringBehavior.FLEE].weight

        # Arrive
        if arrive_target and self._is_active(SteeringBehavior.ARRIVE):
            force = force + arrive(agent, arrive_target) * self._weights[SteeringBehavior.ARRIVE].weight

        # Pursue
        if pursue_target and self._is_active(SteeringBehavior.PURSUE):
            force = force + pursue(agent, pursue_target) * self._weights[SteeringBehavior.PURSUE].weight

        # Evade
        if evade_target and self._is_active(SteeringBehavior.EVADE):
            force = force + evade(agent, evade_target) * self._weights[SteeringBehavior.EVADE].weight

        # Wander
        if self._is_active(SteeringBehavior.WANDER):
            state = self.get_wander_state(agent.id)
            force = force + wander(agent, state, dt=dt) * self._weights[SteeringBehavior.WANDER].weight

        # Separation
        if neighbors and self._is_active(SteeringBehavior.SEPARATION):
            force = force + separation(agent, neighbors) * self._weights[SteeringBehavior.SEPARATION].weight

        # Alignment
        if neighbors and self._is_active(SteeringBehavior.ALIGNMENT):
            force = force + alignment(agent, neighbors) * self._weights[SteeringBehavior.ALIGNMENT].weight

        # Cohesion
        if neighbors and self._is_active(SteeringBehavior.COHESION):
            force = force + cohesion(agent, neighbors) * self._weights[SteeringBehavior.COHESION].weight

        # Obstacle avoidance
        if obstacles and self._is_active(SteeringBehavior.OBSTACLE_AVOIDANCE):
            force = force + obstacle_avoidance(agent, obstacles) * self._weights[SteeringBehavior.OBSTACLE_AVOIDANCE].weight

        # Wall following
        if walls and self._is_active(SteeringBehavior.WALL_FOLLOWING):
            force = force + wall_following(agent, walls) * self._weights[SteeringBehavior.WALL_FOLLOWING].weight

        # Path following
        if path and self._is_active(SteeringBehavior.PATH_FOLLOWING):
            force = force + path_following(agent, path) * self._weights[SteeringBehavior.PATH_FOLLOWING].weight

        # Flocking (combined)
        if neighbors and self._is_active(SteeringBehavior.FLOCKING):
            force = force + flocking(
                agent, neighbors,
                self._weights[SteeringBehavior.SEPARATION].weight,
                self._weights[SteeringBehavior.ALIGNMENT].weight,
                self._weights[SteeringBehavior.COHESION].weight
            ) * self._weights[SteeringBehavior.FLOCKING].weight

        return force

    def calculate_priority(
        self, agent: SteeringAgent,
        seek_target: Optional[Vector3] = None,
        flee_target: Optional[Vector3] = None,
        arrive_target: Optional[Vector3] = None,
        pursue_target: Optional[SteeringAgent] = None,
        evade_target: Optional[SteeringAgent] = None,
        neighbors: Optional[List[SteeringAgent]] = None,
        obstacles: Optional[List[Tuple[Vector3, float]]] = None,
        walls: Optional[List[Tuple[Vector3, Vector3]]] = None,
        path: Optional[List[Vector3]] = None,
        dt: float = 0.016
    ) -> Vector3:
        """
        Calculate combined steering force using prioritized dithering.

        Higher priority behaviors are checked first. Returns immediately
        when max force is reached.

        Args:
            (Same as calculate_weighted_sum)

        Returns:
            Combined steering force
        """
        force = Vector3()
        remaining_force = agent.max_force
        neighbors = neighbors or []
        obstacles = obstacles or []
        walls = walls or []

        # Priority order (highest first)
        behaviors = [
            (SteeringBehavior.OBSTACLE_AVOIDANCE, lambda: obstacle_avoidance(agent, obstacles) if obstacles else Vector3()),
            (SteeringBehavior.WALL_FOLLOWING, lambda: wall_following(agent, walls) if walls else Vector3()),
            (SteeringBehavior.EVADE, lambda: evade(agent, evade_target) if evade_target else Vector3()),
            (SteeringBehavior.FLEE, lambda: flee(agent, flee_target) if flee_target else Vector3()),
            (SteeringBehavior.SEPARATION, lambda: separation(agent, neighbors) if neighbors else Vector3()),
            (SteeringBehavior.ALIGNMENT, lambda: alignment(agent, neighbors) if neighbors else Vector3()),
            (SteeringBehavior.COHESION, lambda: cohesion(agent, neighbors) if neighbors else Vector3()),
            (SteeringBehavior.PATH_FOLLOWING, lambda: path_following(agent, path) if path else Vector3()),
            (SteeringBehavior.ARRIVE, lambda: arrive(agent, arrive_target) if arrive_target else Vector3()),
            (SteeringBehavior.SEEK, lambda: seek(agent, seek_target) if seek_target else Vector3()),
            (SteeringBehavior.PURSUE, lambda: pursue(agent, pursue_target) if pursue_target else Vector3()),
            (SteeringBehavior.WANDER, lambda: wander(agent, self.get_wander_state(agent.id), dt=dt)),
        ]

        for behavior, calc_func in behaviors:
            if not self._is_active(behavior):
                continue

            behavior_force = calc_func() * self._weights[behavior].weight

            force_magnitude = behavior_force.length()
            if force_magnitude > remaining_force:
                # Truncate to remaining force
                behavior_force = behavior_force.normalized() * remaining_force
                force = force + behavior_force
                break
            else:
                force = force + behavior_force
                remaining_force -= force_magnitude

            if remaining_force <= 0:
                break

        return force

    def _is_active(self, behavior: SteeringBehavior) -> bool:
        """Check if behavior is active (exists and enabled with weight > 0)."""
        if behavior not in self._weights:
            return False
        weight = self._weights[behavior]
        return weight.enabled and weight.weight > 0
