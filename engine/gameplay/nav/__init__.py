"""Navigation subsystem: NavMesh, pathfinding, steering, avoidance, nav links, and smart objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, auto
from heapq import heappush, heappop
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)
import math

from engine.gameplay.constants import (
    NavLinkType,
    PathfindAlgorithm,
    AvoidanceType,
    NAV_AGENT_RADIUS_DEFAULT,
    NAV_AGENT_HEIGHT_DEFAULT,
    NAV_STEP_HEIGHT_DEFAULT,
    NAV_MAX_SLOPE_DEFAULT,
    PATH_MAX_NODES,
    PATH_SMOOTH_ITERATIONS,
    STEERING_ARRIVE_SLOW_RADIUS,
    STEERING_ARRIVE_TARGET_RADIUS,
    STEERING_SEPARATION_RADIUS,
    STEERING_COHESION_RADIUS,
    STEERING_ALIGNMENT_RADIUS,
    AVOIDANCE_TIME_HORIZON,
    AVOIDANCE_MAX_NEIGHBORS,
)

if TYPE_CHECKING:
    from engine.gameplay.entity import Actor


# === Basic Types ===

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]


def vec3_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def vec3_length(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec3_normalize(v: Vec3) -> Vec3:
    length = vec3_length(v)
    if length < 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def vec3_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_distance(a: Vec3, b: Vec3) -> float:
    return vec3_length(vec3_sub(a, b))


# === NavMesh ===

@dataclass
class NavMeshConfig:
    """Configuration for NavMesh generation."""

    agent_radius: float = NAV_AGENT_RADIUS_DEFAULT
    agent_height: float = NAV_AGENT_HEIGHT_DEFAULT
    step_height: float = NAV_STEP_HEIGHT_DEFAULT
    max_slope: float = NAV_MAX_SLOPE_DEFAULT
    cell_size: float = 0.3
    cell_height: float = 0.2
    region_min_size: int = 8
    region_merge_size: int = 20
    edge_max_length: float = 12.0
    edge_max_error: float = 1.3


@dataclass
class NavMeshPoly:
    """Single polygon in NavMesh."""

    poly_id: int
    vertices: List[Vec3]
    center: Vec3
    neighbors: List[int] = field(default_factory=list)
    area_type: int = 0
    flags: int = 0


@dataclass
class NavMeshTile:
    """Tile of NavMesh for streaming."""

    tile_x: int
    tile_z: int
    polygons: List[NavMeshPoly] = field(default_factory=list)
    bounds_min: Vec3 = (0.0, 0.0, 0.0)
    bounds_max: Vec3 = (0.0, 0.0, 0.0)


class NavMesh:
    """Navigation mesh for pathfinding."""

    def __init__(self, config: Optional[NavMeshConfig] = None) -> None:
        self._config = config or NavMeshConfig()
        self._tiles: Dict[Tuple[int, int], NavMeshTile] = {}
        self._polygons: Dict[int, NavMeshPoly] = {}
        self._poly_counter: int = 0
        self._bounds_min: Vec3 = (0.0, 0.0, 0.0)
        self._bounds_max: Vec3 = (0.0, 0.0, 0.0)

    @property
    def config(self) -> NavMeshConfig:
        return self._config

    @property
    def polygon_count(self) -> int:
        return len(self._polygons)

    def add_polygon(self, vertices: List[Vec3], area_type: int = 0) -> int:
        """Add polygon to NavMesh."""
        self._poly_counter += 1
        poly_id = self._poly_counter

        # Calculate center
        center_x = sum(v[0] for v in vertices) / len(vertices)
        center_y = sum(v[1] for v in vertices) / len(vertices)
        center_z = sum(v[2] for v in vertices) / len(vertices)

        poly = NavMeshPoly(
            poly_id=poly_id,
            vertices=vertices,
            center=(center_x, center_y, center_z),
            area_type=area_type,
        )
        self._polygons[poly_id] = poly
        return poly_id

    def connect_polygons(self, poly_a: int, poly_b: int) -> None:
        """Connect two adjacent polygons."""
        if poly_a in self._polygons and poly_b in self._polygons:
            if poly_b not in self._polygons[poly_a].neighbors:
                self._polygons[poly_a].neighbors.append(poly_b)
            if poly_a not in self._polygons[poly_b].neighbors:
                self._polygons[poly_b].neighbors.append(poly_a)

    def get_polygon(self, poly_id: int) -> Optional[NavMeshPoly]:
        """Get polygon by ID."""
        return self._polygons.get(poly_id)

    def find_nearest_polygon(self, position: Vec3) -> Optional[int]:
        """Find nearest polygon to position."""
        nearest_id = None
        nearest_dist = float("inf")

        for poly_id, poly in self._polygons.items():
            dist = vec3_distance(position, poly.center)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = poly_id

        return nearest_id

    def is_position_on_navmesh(self, position: Vec3, tolerance: float = 0.5) -> bool:
        """Check if position is on NavMesh."""
        nearest = self.find_nearest_polygon(position)
        if nearest is None:
            return False

        poly = self._polygons[nearest]
        return vec3_distance(position, poly.center) < tolerance

    def get_random_point(self) -> Optional[Vec3]:
        """Get random navigable point."""
        import random
        if not self._polygons:
            return None
        poly = random.choice(list(self._polygons.values()))
        return poly.center

    def raycast(
        self,
        start: Vec3,
        end: Vec3,
    ) -> Tuple[bool, Vec3, Optional[int]]:
        """Raycast on NavMesh. Returns (hit, hit_point, poly_id)."""
        # Simplified: check if both points are on navmesh
        start_poly = self.find_nearest_polygon(start)
        end_poly = self.find_nearest_polygon(end)

        if start_poly is None or end_poly is None:
            return (True, start, None)

        return (False, end, end_poly)


# === Pathfinding ===

@dataclass
class PathNode:
    """Node in pathfinding search."""

    poly_id: int
    position: Vec3
    g_cost: float
    h_cost: float
    parent: Optional[PathNode] = None

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def __lt__(self, other: PathNode) -> bool:
        return self.f_cost < other.f_cost


@dataclass
class NavPath:
    """Navigation path result."""

    points: List[Vec3] = field(default_factory=list)
    length: float = 0.0
    is_complete: bool = False
    is_partial: bool = False

    @property
    def point_count(self) -> int:
        return len(self.points)

    def get_point(self, index: int) -> Optional[Vec3]:
        if 0 <= index < len(self.points):
            return self.points[index]
        return None


class Pathfinding:
    """Pathfinding system using A* on NavMesh."""

    def __init__(
        self,
        navmesh: NavMesh,
        algorithm: PathfindAlgorithm = PathfindAlgorithm.ASTAR,
    ) -> None:
        self._navmesh = navmesh
        self._algorithm = algorithm

    @property
    def navmesh(self) -> NavMesh:
        return self._navmesh

    def find_path(
        self,
        start: Vec3,
        goal: Vec3,
        max_nodes: int = PATH_MAX_NODES,
    ) -> NavPath:
        """Find path from start to goal."""
        path = NavPath()

        # Find start and goal polygons
        start_poly = self._navmesh.find_nearest_polygon(start)
        goal_poly = self._navmesh.find_nearest_polygon(goal)

        if start_poly is None or goal_poly is None:
            return path

        if start_poly == goal_poly:
            path.points = [start, goal]
            path.length = vec3_distance(start, goal)
            path.is_complete = True
            return path

        # A* search
        open_set: List[PathNode] = []
        closed_set: Set[int] = set()

        start_node = PathNode(
            poly_id=start_poly,
            position=start,
            g_cost=0.0,
            h_cost=vec3_distance(start, goal),
        )
        heappush(open_set, start_node)

        goal_pos = goal
        nodes_processed = 0

        while open_set and nodes_processed < max_nodes:
            nodes_processed += 1

            current = heappop(open_set)

            if current.poly_id in closed_set:
                continue
            closed_set.add(current.poly_id)

            # Check if goal reached
            if current.poly_id == goal_poly:
                return self._reconstruct_path(current, start, goal)

            # Expand neighbors
            poly = self._navmesh.get_polygon(current.poly_id)
            if poly is None:
                continue

            for neighbor_id in poly.neighbors:
                if neighbor_id in closed_set:
                    continue

                neighbor_poly = self._navmesh.get_polygon(neighbor_id)
                if neighbor_poly is None:
                    continue

                g_cost = current.g_cost + vec3_distance(current.position, neighbor_poly.center)
                h_cost = vec3_distance(neighbor_poly.center, goal_pos)

                neighbor_node = PathNode(
                    poly_id=neighbor_id,
                    position=neighbor_poly.center,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    parent=current,
                )
                heappush(open_set, neighbor_node)

        # No path found
        path.is_partial = True
        return path

    def _reconstruct_path(
        self,
        node: PathNode,
        start: Vec3,
        goal: Vec3,
    ) -> NavPath:
        """Reconstruct path from node chain."""
        path = NavPath()
        points = [goal]

        current = node
        while current.parent:
            points.append(current.position)
            current = current.parent

        points.append(start)
        points.reverse()

        path.points = points
        path.is_complete = True

        # Calculate length
        for i in range(1, len(points)):
            path.length += vec3_distance(points[i - 1], points[i])

        # Smooth path
        path.points = self._smooth_path(path.points)

        return path

    def _smooth_path(self, points: List[Vec3]) -> List[Vec3]:
        """Smooth path using string pulling."""
        if len(points) <= 2:
            return points

        smoothed = [points[0]]

        for _ in range(PATH_SMOOTH_ITERATIONS):
            i = 0
            while i < len(points) - 1:
                # Try to skip intermediate points
                j = len(points) - 1
                while j > i + 1:
                    # Check if direct path is clear
                    hit, _, _ = self._navmesh.raycast(points[i], points[j])
                    if not hit:
                        break
                    j -= 1

                smoothed.append(points[j])
                i = j

            points = smoothed
            smoothed = [points[0]]

        return points


# === Steering Behaviors ===

@dataclass
class SteeringOutput:
    """Output from steering behavior."""

    linear: Vec3 = (0.0, 0.0, 0.0)
    angular: float = 0.0

    def __add__(self, other: SteeringOutput) -> SteeringOutput:
        return SteeringOutput(
            linear=vec3_add(self.linear, other.linear),
            angular=self.angular + other.angular,
        )


class SteeringBehavior(ABC):
    """Base class for steering behaviors."""

    def __init__(self, weight: float = 1.0) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

    @abstractmethod
    def calculate(
        self,
        position: Vec3,
        velocity: Vec3,
        max_speed: float,
    ) -> SteeringOutput:
        """Calculate steering output."""
        pass


class SeekSteering(SteeringBehavior):
    """Seek toward target position."""

    def __init__(self, target: Vec3, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._target = target

    @property
    def target(self) -> Vec3:
        return self._target

    @target.setter
    def target(self, value: Vec3) -> None:
        self._target = value

    def calculate(
        self,
        position: Vec3,
        velocity: Vec3,
        max_speed: float,
    ) -> SteeringOutput:
        desired = vec3_sub(self._target, position)
        desired = vec3_normalize(desired)
        desired = vec3_scale(desired, max_speed)

        steering = vec3_sub(desired, velocity)
        steering = vec3_scale(steering, self._weight)

        return SteeringOutput(linear=steering)


class FleeSteering(SteeringBehavior):
    """Flee from target position."""

    def __init__(self, target: Vec3, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._target = target

    @property
    def target(self) -> Vec3:
        return self._target

    @target.setter
    def target(self, value: Vec3) -> None:
        self._target = value

    def calculate(
        self,
        position: Vec3,
        velocity: Vec3,
        max_speed: float,
    ) -> SteeringOutput:
        desired = vec3_sub(position, self._target)
        desired = vec3_normalize(desired)
        desired = vec3_scale(desired, max_speed)

        steering = vec3_sub(desired, velocity)
        steering = vec3_scale(steering, self._weight)

        return SteeringOutput(linear=steering)


class ArriveSteering(SteeringBehavior):
    """Arrive at target with deceleration."""

    def __init__(
        self,
        target: Vec3,
        slow_radius: float = STEERING_ARRIVE_SLOW_RADIUS,
        target_radius: float = STEERING_ARRIVE_TARGET_RADIUS,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self._target = target
        self._slow_radius = slow_radius
        self._target_radius = target_radius

    @property
    def target(self) -> Vec3:
        return self._target

    @target.setter
    def target(self, value: Vec3) -> None:
        self._target = value

    def calculate(
        self,
        position: Vec3,
        velocity: Vec3,
        max_speed: float,
    ) -> SteeringOutput:
        to_target = vec3_sub(self._target, position)
        distance = vec3_length(to_target)

        if distance < self._target_radius:
            return SteeringOutput()

        if distance < self._slow_radius:
            target_speed = max_speed * (distance / self._slow_radius)
        else:
            target_speed = max_speed

        desired = vec3_normalize(to_target)
        desired = vec3_scale(desired, target_speed)

        steering = vec3_sub(desired, velocity)
        steering = vec3_scale(steering, self._weight)

        return SteeringOutput(linear=steering)


class PursueSteering(SteeringBehavior):
    """Pursue moving target by predicting position."""

    def __init__(
        self,
        target_position: Vec3,
        target_velocity: Vec3,
        max_prediction: float = 2.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self._target_position = target_position
        self._target_velocity = target_velocity
        self._max_prediction = max_prediction

    def set_target(self, position: Vec3, velocity: Vec3) -> None:
        """Update target position and velocity."""
        self._target_position = position
        self._target_velocity = velocity

    def calculate(
        self,
        position: Vec3,
        velocity: Vec3,
        max_speed: float,
    ) -> SteeringOutput:
        to_target = vec3_sub(self._target_position, position)
        distance = vec3_length(to_target)

        speed = vec3_length(velocity)
        if speed > 0.001:
            prediction = min(distance / speed, self._max_prediction)
        else:
            prediction = self._max_prediction

        predicted = vec3_add(
            self._target_position,
            vec3_scale(self._target_velocity, prediction),
        )

        seek = SeekSteering(predicted, self._weight)
        return seek.calculate(position, velocity, max_speed)


class SeparationSteering(SteeringBehavior):
    """Separate from nearby agents."""

    def __init__(
        self,
        neighbors: List[Vec3],
        radius: float = STEERING_SEPARATION_RADIUS,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self._neighbors = neighbors
        self._radius = radius

    def set_neighbors(self, neighbors: List[Vec3]) -> None:
        """Update neighbor positions."""
        self._neighbors = neighbors

    def calculate(
        self,
        position: Vec3,
        velocity: Vec3,
        max_speed: float,
    ) -> SteeringOutput:
        steering = (0.0, 0.0, 0.0)
        count = 0

        for neighbor in self._neighbors:
            to_self = vec3_sub(position, neighbor)
            distance = vec3_length(to_self)

            if 0 < distance < self._radius:
                # Weight by inverse distance
                push = vec3_normalize(to_self)
                push = vec3_scale(push, (self._radius - distance) / self._radius)
                steering = vec3_add(steering, push)
                count += 1

        if count > 0:
            steering = vec3_scale(steering, 1.0 / count)
            steering = vec3_normalize(steering)
            steering = vec3_scale(steering, max_speed * self._weight)

        return SteeringOutput(linear=steering)


class CohesionSteering(SteeringBehavior):
    """Move toward center of nearby agents."""

    def __init__(
        self,
        neighbors: List[Vec3],
        radius: float = STEERING_COHESION_RADIUS,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self._neighbors = neighbors
        self._radius = radius

    def set_neighbors(self, neighbors: List[Vec3]) -> None:
        """Update neighbor positions."""
        self._neighbors = neighbors

    def calculate(
        self,
        position: Vec3,
        velocity: Vec3,
        max_speed: float,
    ) -> SteeringOutput:
        if not self._neighbors:
            return SteeringOutput()

        center = (0.0, 0.0, 0.0)
        count = 0

        for neighbor in self._neighbors:
            distance = vec3_distance(position, neighbor)
            if distance < self._radius:
                center = vec3_add(center, neighbor)
                count += 1

        if count > 0:
            center = vec3_scale(center, 1.0 / count)
            seek = SeekSteering(center, self._weight)
            return seek.calculate(position, velocity, max_speed)

        return SteeringOutput()


class AlignmentSteering(SteeringBehavior):
    """Align velocity with nearby agents."""

    def __init__(
        self,
        neighbor_velocities: List[Vec3],
        radius: float = STEERING_ALIGNMENT_RADIUS,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self._neighbor_velocities = neighbor_velocities
        self._radius = radius

    def set_neighbors(self, velocities: List[Vec3]) -> None:
        """Update neighbor velocities."""
        self._neighbor_velocities = velocities

    def calculate(
        self,
        position: Vec3,
        velocity: Vec3,
        max_speed: float,
    ) -> SteeringOutput:
        if not self._neighbor_velocities:
            return SteeringOutput()

        average = (0.0, 0.0, 0.0)
        for vel in self._neighbor_velocities:
            average = vec3_add(average, vel)

        average = vec3_scale(average, 1.0 / len(self._neighbor_velocities))
        average = vec3_normalize(average)
        average = vec3_scale(average, max_speed)

        steering = vec3_sub(average, velocity)
        steering = vec3_scale(steering, self._weight)

        return SteeringOutput(linear=steering)


class Steering:
    """Combined steering behavior manager."""

    def __init__(self, max_speed: float = 5.0, max_force: float = 10.0) -> None:
        self._behaviors: List[SteeringBehavior] = []
        self._max_speed = max_speed
        self._max_force = max_force

    @property
    def max_speed(self) -> float:
        return self._max_speed

    @property
    def max_force(self) -> float:
        return self._max_force

    def add_behavior(self, behavior: SteeringBehavior) -> None:
        """Add steering behavior."""
        self._behaviors.append(behavior)

    def remove_behavior(self, behavior: SteeringBehavior) -> None:
        """Remove steering behavior."""
        if behavior in self._behaviors:
            self._behaviors.remove(behavior)

    def clear_behaviors(self) -> None:
        """Clear all behaviors."""
        self._behaviors.clear()

    def calculate(self, position: Vec3, velocity: Vec3) -> SteeringOutput:
        """Calculate combined steering output."""
        result = SteeringOutput()

        for behavior in self._behaviors:
            output = behavior.calculate(position, velocity, self._max_speed)
            result = result + output

        # Limit force
        force_mag = vec3_length(result.linear)
        if force_mag > self._max_force:
            result.linear = vec3_scale(
                vec3_normalize(result.linear),
                self._max_force,
            )

        return result


# === Collision Avoidance ===

@dataclass
class AvoidanceAgent:
    """Agent for collision avoidance."""

    agent_id: int
    position: Vec3
    velocity: Vec3
    radius: float
    max_speed: float


class Avoidance:
    """Collision avoidance system (simplified RVO)."""

    def __init__(
        self,
        avoidance_type: AvoidanceType = AvoidanceType.RVO,
        time_horizon: float = AVOIDANCE_TIME_HORIZON,
        max_neighbors: int = AVOIDANCE_MAX_NEIGHBORS,
    ) -> None:
        self._avoidance_type = avoidance_type
        self._time_horizon = time_horizon
        self._max_neighbors = max_neighbors
        self._agents: Dict[int, AvoidanceAgent] = {}

    def add_agent(self, agent: AvoidanceAgent) -> None:
        """Add agent to avoidance system."""
        self._agents[agent.agent_id] = agent

    def remove_agent(self, agent_id: int) -> None:
        """Remove agent from avoidance system."""
        if agent_id in self._agents:
            del self._agents[agent_id]

    def update_agent(
        self,
        agent_id: int,
        position: Vec3,
        velocity: Vec3,
    ) -> None:
        """Update agent position and velocity."""
        if agent_id in self._agents:
            self._agents[agent_id].position = position
            self._agents[agent_id].velocity = velocity

    def compute_velocity(
        self,
        agent_id: int,
        preferred_velocity: Vec3,
    ) -> Vec3:
        """Compute collision-free velocity for agent."""
        if agent_id not in self._agents:
            return preferred_velocity

        agent = self._agents[agent_id]

        # Find nearby agents
        neighbors = self._get_neighbors(agent)
        if not neighbors:
            return preferred_velocity

        # Simple RVO: adjust velocity to avoid collisions
        avoidance = (0.0, 0.0, 0.0)

        for other in neighbors:
            relative_pos = vec3_sub(other.position, agent.position)
            distance = vec3_length(relative_pos)
            combined_radius = agent.radius + other.radius

            if distance < combined_radius * 2:
                # Collision imminent
                push_dir = vec3_normalize(vec3_sub(agent.position, other.position))
                urgency = 1.0 - (distance / (combined_radius * 2))
                push = vec3_scale(push_dir, urgency * agent.max_speed)
                avoidance = vec3_add(avoidance, push)

        # Blend preferred velocity with avoidance
        result = vec3_add(preferred_velocity, avoidance)
        speed = vec3_length(result)

        if speed > agent.max_speed:
            result = vec3_scale(vec3_normalize(result), agent.max_speed)

        return result

    def _get_neighbors(self, agent: AvoidanceAgent) -> List[AvoidanceAgent]:
        """Get neighboring agents within range."""
        neighbors = []
        range_sq = (self._time_horizon * agent.max_speed) ** 2

        for other_id, other in self._agents.items():
            if other_id == agent.agent_id:
                continue

            dist_sq = (
                (other.position[0] - agent.position[0]) ** 2 +
                (other.position[1] - agent.position[1]) ** 2 +
                (other.position[2] - agent.position[2]) ** 2
            )

            if dist_sq < range_sq:
                neighbors.append(other)
                if len(neighbors) >= self._max_neighbors:
                    break

        return neighbors


# === Nav Links ===

@dataclass
class NavLink:
    """Navigation link between disconnected areas."""

    link_id: int
    link_type: NavLinkType
    start_position: Vec3
    end_position: Vec3
    start_poly: int
    end_poly: int
    bidirectional: bool = True
    cost: float = 1.0
    enabled: bool = True
    required_ability: Optional[str] = None


class NavLinks:
    """Manager for navigation links."""

    def __init__(self, navmesh: NavMesh) -> None:
        self._navmesh = navmesh
        self._links: Dict[int, NavLink] = {}
        self._link_counter: int = 0

    def add_link(
        self,
        link_type: NavLinkType,
        start: Vec3,
        end: Vec3,
        bidirectional: bool = True,
        cost: float = 1.0,
    ) -> int:
        """Add navigation link."""
        self._link_counter += 1
        link_id = self._link_counter

        start_poly = self._navmesh.find_nearest_polygon(start) or 0
        end_poly = self._navmesh.find_nearest_polygon(end) or 0

        link = NavLink(
            link_id=link_id,
            link_type=link_type,
            start_position=start,
            end_position=end,
            start_poly=start_poly,
            end_poly=end_poly,
            bidirectional=bidirectional,
            cost=cost,
        )
        self._links[link_id] = link
        return link_id

    def remove_link(self, link_id: int) -> bool:
        """Remove navigation link."""
        if link_id in self._links:
            del self._links[link_id]
            return True
        return False

    def get_link(self, link_id: int) -> Optional[NavLink]:
        """Get link by ID."""
        return self._links.get(link_id)

    def enable_link(self, link_id: int, enabled: bool = True) -> None:
        """Enable or disable a link."""
        if link_id in self._links:
            self._links[link_id].enabled = enabled

    def get_links_at_position(
        self,
        position: Vec3,
        radius: float = 1.0,
    ) -> List[NavLink]:
        """Get all links near a position."""
        result = []
        for link in self._links.values():
            if not link.enabled:
                continue
            if vec3_distance(position, link.start_position) < radius:
                result.append(link)
            elif link.bidirectional and vec3_distance(position, link.end_position) < radius:
                result.append(link)
        return result

    def get_links_by_type(self, link_type: NavLinkType) -> List[NavLink]:
        """Get all links of a specific type."""
        return [l for l in self._links.values() if l.link_type == link_type]


# === Smart Objects ===

class SmartObjectSlot(IntEnum):
    """Slot states for smart objects."""
    AVAILABLE = auto()
    RESERVED = auto()
    OCCUPIED = auto()


@dataclass
class SmartObjectDefinition:
    """Definition of a smart object interaction."""

    object_id: str
    slots: int = 1
    interaction_range: float = 2.0
    interaction_time: float = 1.0
    required_tags: Set[str] = field(default_factory=set)
    animation: Optional[str] = None


@dataclass
class SmartObjectInstance:
    """Instance of a smart object in the world."""

    instance_id: int
    definition: SmartObjectDefinition
    position: Vec3
    rotation: float = 0.0
    slot_states: List[SmartObjectSlot] = field(default_factory=list)
    reserved_by: List[Optional[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.slot_states = [SmartObjectSlot.AVAILABLE] * self.definition.slots
        self.reserved_by = [None] * self.definition.slots


class SmartObjects:
    """Manager for smart objects."""

    def __init__(self) -> None:
        self._definitions: Dict[str, SmartObjectDefinition] = {}
        self._instances: Dict[int, SmartObjectInstance] = {}
        self._instance_counter: int = 0

    def register_definition(self, definition: SmartObjectDefinition) -> None:
        """Register smart object definition."""
        self._definitions[definition.object_id] = definition

    def spawn_instance(
        self,
        object_id: str,
        position: Vec3,
        rotation: float = 0.0,
    ) -> Optional[int]:
        """Spawn smart object instance."""
        if object_id not in self._definitions:
            return None

        self._instance_counter += 1
        instance_id = self._instance_counter

        instance = SmartObjectInstance(
            instance_id=instance_id,
            definition=self._definitions[object_id],
            position=position,
            rotation=rotation,
        )
        self._instances[instance_id] = instance
        return instance_id

    def remove_instance(self, instance_id: int) -> bool:
        """Remove smart object instance."""
        if instance_id in self._instances:
            del self._instances[instance_id]
            return True
        return False

    def get_instance(self, instance_id: int) -> Optional[SmartObjectInstance]:
        """Get smart object instance."""
        return self._instances.get(instance_id)

    def reserve_slot(
        self,
        instance_id: int,
        agent_id: int,
    ) -> Optional[int]:
        """Reserve a slot. Returns slot index or None."""
        instance = self._instances.get(instance_id)
        if not instance:
            return None

        for i, state in enumerate(instance.slot_states):
            if state == SmartObjectSlot.AVAILABLE:
                instance.slot_states[i] = SmartObjectSlot.RESERVED
                instance.reserved_by[i] = agent_id
                return i

        return None

    def occupy_slot(self, instance_id: int, slot: int, agent_id: int) -> bool:
        """Occupy a reserved slot."""
        instance = self._instances.get(instance_id)
        if not instance or slot >= len(instance.slot_states):
            return False

        if (instance.slot_states[slot] == SmartObjectSlot.RESERVED and
            instance.reserved_by[slot] == agent_id):
            instance.slot_states[slot] = SmartObjectSlot.OCCUPIED
            return True

        return False

    def release_slot(self, instance_id: int, slot: int) -> bool:
        """Release a slot."""
        instance = self._instances.get(instance_id)
        if not instance or slot >= len(instance.slot_states):
            return False

        instance.slot_states[slot] = SmartObjectSlot.AVAILABLE
        instance.reserved_by[slot] = None
        return True

    def find_nearest_available(
        self,
        position: Vec3,
        object_id: str,
    ) -> Optional[Tuple[int, int]]:
        """Find nearest available smart object. Returns (instance_id, slot)."""
        best_instance = None
        best_slot = None
        best_dist = float("inf")

        for instance in self._instances.values():
            if instance.definition.object_id != object_id:
                continue

            for i, state in enumerate(instance.slot_states):
                if state == SmartObjectSlot.AVAILABLE:
                    dist = vec3_distance(position, instance.position)
                    if dist < best_dist:
                        best_dist = dist
                        best_instance = instance.instance_id
                        best_slot = i

        if best_instance is not None and best_slot is not None:
            return (best_instance, best_slot)
        return None


__all__ = [
    # NavMesh
    "NavMeshConfig",
    "NavMeshPoly",
    "NavMeshTile",
    "NavMesh",
    # Pathfinding
    "PathNode",
    "NavPath",
    "Pathfinding",
    # Steering
    "SteeringOutput",
    "SteeringBehavior",
    "SeekSteering",
    "FleeSteering",
    "ArriveSteering",
    "PursueSteering",
    "SeparationSteering",
    "CohesionSteering",
    "AlignmentSteering",
    "Steering",
    # Avoidance
    "AvoidanceAgent",
    "Avoidance",
    # Nav Links
    "NavLink",
    "NavLinks",
    # Smart Objects
    "SmartObjectSlot",
    "SmartObjectDefinition",
    "SmartObjectInstance",
    "SmartObjects",
]
