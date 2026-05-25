"""Simple crowd AI for animation selection.

Provides behavior patterns for crowd agents that drive animation selection
and basic movement/avoidance.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Sequence

from engine.core.math import Vec2, Vec3, Quat
from engine.animation.config import CROWD_BEHAVIOR_CONFIG


class AgentState(Enum):
    """Crowd agent behavioral state."""
    IDLE = auto()
    WALKING = auto()
    WAITING = auto()
    FLEEING = auto()
    FORMATION = auto()
    CUSTOM = auto()


@dataclass
class AnimationBlend:
    """Blend between animations."""
    animation_indices: list[int] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    @staticmethod
    def single(index: int) -> AnimationBlend:
        """Create blend with single animation."""
        return AnimationBlend(animation_indices=[index], weights=[1.0])

    @staticmethod
    def blend(index_a: int, index_b: int, factor: float) -> AnimationBlend:
        """Create blend between two animations."""
        return AnimationBlend(
            animation_indices=[index_a, index_b],
            weights=[1.0 - factor, factor],
        )

    def get_primary_animation(self) -> int:
        """Get animation with highest weight."""
        if not self.animation_indices:
            return 0
        max_idx = 0
        max_weight = self.weights[0] if self.weights else 0.0
        for i, weight in enumerate(self.weights):
            if weight > max_weight:
                max_weight = weight
                max_idx = i
        return self.animation_indices[max_idx]


@dataclass
class CrowdAgent:
    """Individual crowd agent.

    Attributes:
        position: World position (Vec3 for 3D, using XZ plane for movement)
        velocity: Current velocity
        target_velocity: Desired velocity
        current_state: Current behavioral state
        animation_blend: Current animation blend state
        agent_id: Unique identifier
        group_id: Group identifier for formations
        speed: Maximum movement speed
        turn_speed: Maximum turn speed (radians/sec)
        facing: Forward direction (2D angle)
        radius: Agent collision radius
        priority: Priority for avoidance (higher = more dominant)
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    velocity: Vec3 = field(default_factory=Vec3.zero)
    target_velocity: Vec3 = field(default_factory=Vec3.zero)
    current_state: AgentState = AgentState.IDLE
    animation_blend: AnimationBlend = field(default_factory=AnimationBlend)
    agent_id: int = 0
    group_id: int = 0
    speed: float = CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED  # Average walking speed
    turn_speed: float = CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_TURN_SPEED  # ~180 degrees per second
    facing: float = 0.0  # Radians
    radius: float = CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_RADIUS
    priority: int = 0

    # Animation mapping
    idle_animation: int = 0
    walk_animation: int = 1
    run_animation: int = 2

    # State data
    state_time: float = 0.0
    idle_variation_time: float = 0.0
    target_position: Vec3 | None = None
    flee_source: Vec3 | None = None
    formation_offset: Vec3 | None = None

    _next_id: int = 0

    def __post_init__(self):
        if self.agent_id == 0:
            CrowdAgent._next_id += 1
            self.agent_id = CrowdAgent._next_id
        if not self.animation_blend.animation_indices:
            self.animation_blend = AnimationBlend.single(self.idle_animation)

    def get_forward(self) -> Vec3:
        """Get forward direction vector."""
        return Vec3(math.sin(self.facing), 0.0, math.cos(self.facing))

    def set_facing_from_direction(self, direction: Vec3) -> None:
        """Set facing from direction vector."""
        if direction.x != 0.0 or direction.z != 0.0:
            self.facing = math.atan2(direction.x, direction.z)

    def get_rotation(self) -> Quat:
        """Get rotation quaternion for rendering."""
        return Quat.from_axis_angle(Vec3.up(), self.facing)

    def distance_to(self, other: CrowdAgent | Vec3) -> float:
        """Calculate distance to another agent or point."""
        if isinstance(other, CrowdAgent):
            return self.position.distance(other.position)
        return self.position.distance(other)

    def is_moving(self) -> bool:
        """Check if agent is currently moving."""
        return self.velocity.length_squared() > 0.01


class CrowdBehavior(ABC):
    """Base class for crowd behaviors.

    Behaviors control agent state, movement, and animation selection.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Behavior name."""
        pass

    @abstractmethod
    def update(self, agent: CrowdAgent, dt: float, context: BehaviorContext) -> None:
        """Update agent based on this behavior.

        Args:
            agent: Agent to update
            dt: Delta time in seconds
            context: Simulation context
        """
        pass

    def can_transition_to(self, agent: CrowdAgent, target_state: AgentState) -> bool:
        """Check if transition to target state is allowed."""
        return True

    def on_enter(self, agent: CrowdAgent) -> None:
        """Called when agent enters this behavior state."""
        agent.state_time = 0.0

    def on_exit(self, agent: CrowdAgent) -> None:
        """Called when agent exits this behavior state."""
        pass


@dataclass
class BehaviorContext:
    """Context information for behavior updates."""
    all_agents: list[CrowdAgent] = field(default_factory=list)
    obstacles: list[tuple[Vec3, float]] = field(default_factory=list)  # (position, radius)
    navigation_points: list[Vec3] = field(default_factory=list)
    time: float = 0.0

    def get_nearby_agents(self, agent: CrowdAgent, radius: float) -> list[CrowdAgent]:
        """Get agents within radius of given agent."""
        return [
            other for other in self.all_agents
            if other.agent_id != agent.agent_id
            and agent.distance_to(other) <= radius
        ]


class IdleBehavior(CrowdBehavior):
    """Idle behavior with occasional variations.

    Agent stands in place with occasional idle animation variations.
    """

    def __init__(
        self,
        variation_interval: tuple[float, float] = (
            CROWD_BEHAVIOR_CONFIG.IDLE_VARIATION_MIN,
            CROWD_BEHAVIOR_CONFIG.IDLE_VARIATION_MAX
        ),
        idle_animations: list[int] | None = None,
    ):
        self._variation_min, self._variation_max = variation_interval
        self._idle_animations = idle_animations or [0]

    @property
    def name(self) -> str:
        return "idle"

    def update(self, agent: CrowdAgent, dt: float, context: BehaviorContext) -> None:
        agent.state_time += dt
        agent.idle_variation_time -= dt

        # Gradually stop movement
        agent.target_velocity = Vec3.zero()
        agent.velocity = agent.velocity * max(0.0, 1.0 - dt * 5.0)

        # Check for idle variation
        if agent.idle_variation_time <= 0:
            # Pick new variation time
            agent.idle_variation_time = random.uniform(self._variation_min, self._variation_max)

            # Pick random idle animation
            anim_idx = random.choice(self._idle_animations)
            agent.animation_blend = AnimationBlend.single(anim_idx)

    def on_enter(self, agent: CrowdAgent) -> None:
        super().on_enter(agent)
        agent.current_state = AgentState.IDLE
        agent.animation_blend = AnimationBlend.single(agent.idle_animation)
        agent.idle_variation_time = random.uniform(self._variation_min, self._variation_max)


class WalkingBehavior(CrowdBehavior):
    """Walking behavior with simple avoidance.

    Agent walks toward target with basic obstacle/crowd avoidance.
    """

    def __init__(
        self,
        avoidance_radius: float = CROWD_BEHAVIOR_CONFIG.DEFAULT_AVOIDANCE_RADIUS,
        avoidance_strength: float = CROWD_BEHAVIOR_CONFIG.DEFAULT_AVOIDANCE_STRENGTH,
        arrival_threshold: float = CROWD_BEHAVIOR_CONFIG.ARRIVAL_THRESHOLD,
    ):
        self._avoidance_radius = avoidance_radius
        self._avoidance_strength = avoidance_strength
        self._arrival_threshold = arrival_threshold

    @property
    def name(self) -> str:
        return "walking"

    def update(self, agent: CrowdAgent, dt: float, context: BehaviorContext) -> None:
        agent.state_time += dt

        # Calculate desired direction
        if agent.target_position is None:
            agent.target_velocity = Vec3.zero()
        else:
            to_target = agent.target_position - agent.position
            distance = to_target.length()

            if distance < self._arrival_threshold:
                agent.target_velocity = Vec3.zero()
                agent.target_position = None
            else:
                direction = to_target.normalized()

                # Apply avoidance
                avoidance = self._calculate_avoidance(agent, context)
                direction = (direction + avoidance).normalized()

                # Slow down when approaching target
                speed_factor = min(1.0, distance / 2.0)
                agent.target_velocity = direction * agent.speed * speed_factor

        # Smooth velocity transition
        velocity_diff = agent.target_velocity - agent.velocity
        agent.velocity = agent.velocity + velocity_diff * min(1.0, dt * 4.0)

        # Update position
        agent.position = agent.position + agent.velocity * dt

        # Update facing
        if agent.velocity.length_squared() > 0.01:
            target_facing = math.atan2(agent.velocity.x, agent.velocity.z)
            facing_diff = target_facing - agent.facing

            # Normalize to [-pi, pi]
            while facing_diff > math.pi:
                facing_diff -= 2 * math.pi
            while facing_diff < -math.pi:
                facing_diff += 2 * math.pi

            max_turn = agent.turn_speed * dt
            agent.facing += max(-max_turn, min(max_turn, facing_diff))

        # Update animation blend
        self._update_animation(agent)

    def _calculate_avoidance(self, agent: CrowdAgent, context: BehaviorContext) -> Vec3:
        """Calculate avoidance vector from nearby agents and obstacles.

        Uses configurable epsilon to avoid division by zero for coincident agents.
        """
        avoidance = Vec3.zero()

        # Avoid other agents
        for other in context.get_nearby_agents(agent, self._avoidance_radius):
            to_agent = agent.position - other.position
            dist = to_agent.length()
            if dist < CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON:
                # Agents are coincident - push in random direction
                angle = random.uniform(0, 2 * math.pi)
                avoidance = avoidance + Vec3(math.sin(angle), 0, math.cos(angle)) * self._avoidance_strength
                continue

            # Stronger avoidance when closer
            strength = (1.0 - dist / self._avoidance_radius) * self._avoidance_strength

            # Consider priority
            if other.priority > agent.priority:
                strength *= CROWD_BEHAVIOR_CONFIG.AVOIDANCE_PRIORITY_MULTIPLIER

            avoidance = avoidance + to_agent.normalized() * strength

        # Avoid obstacles
        for obs_pos, obs_radius in context.obstacles:
            to_agent = agent.position - obs_pos
            dist = to_agent.length()
            combined_radius = obs_radius + agent.radius + self._avoidance_radius

            if dist < combined_radius:
                if dist < CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON:
                    # Agent at obstacle center - push away in random direction
                    angle = random.uniform(0, 2 * math.pi)
                    avoidance = avoidance + Vec3(math.sin(angle), 0, math.cos(angle)) * self._avoidance_strength * 2.0
                else:
                    strength = (1.0 - dist / combined_radius) * self._avoidance_strength * 2.0
                    avoidance = avoidance + to_agent.normalized() * strength

        return avoidance

    def _update_animation(self, agent: CrowdAgent) -> None:
        """Update animation based on movement speed."""
        speed = agent.velocity.length()
        walk_threshold = agent.speed * 0.1
        run_threshold = agent.speed * 0.8

        if speed < walk_threshold:
            agent.animation_blend = AnimationBlend.single(agent.idle_animation)
        elif speed < run_threshold:
            # Blend between walk and run
            blend_factor = (speed - walk_threshold) / (run_threshold - walk_threshold)
            agent.animation_blend = AnimationBlend.blend(
                agent.walk_animation, agent.run_animation, blend_factor
            )
        else:
            agent.animation_blend = AnimationBlend.single(agent.run_animation)

    def on_enter(self, agent: CrowdAgent) -> None:
        super().on_enter(agent)
        agent.current_state = AgentState.WALKING


class WaitingBehavior(CrowdBehavior):
    """Waiting in place behavior.

    Agent waits in position, occasionally shifting weight or looking around.
    """

    def __init__(
        self,
        fidget_interval: tuple[float, float] = (5.0, 15.0),
        wait_animations: list[int] | None = None,
    ):
        self._fidget_min, self._fidget_max = fidget_interval
        self._wait_animations = wait_animations or [0]
        self._fidget_timer = 0.0

    @property
    def name(self) -> str:
        return "waiting"

    def update(self, agent: CrowdAgent, dt: float, context: BehaviorContext) -> None:
        agent.state_time += dt
        self._fidget_timer -= dt

        # Stay in place
        agent.target_velocity = Vec3.zero()
        agent.velocity = agent.velocity * max(0.0, 1.0 - dt * 5.0)

        # Occasional fidget
        if self._fidget_timer <= 0:
            self._fidget_timer = random.uniform(self._fidget_min, self._fidget_max)
            anim_idx = random.choice(self._wait_animations)
            agent.animation_blend = AnimationBlend.single(anim_idx)

            # Small random facing adjustment
            agent.facing += random.uniform(-0.3, 0.3)

    def on_enter(self, agent: CrowdAgent) -> None:
        super().on_enter(agent)
        agent.current_state = AgentState.WAITING
        self._fidget_timer = random.uniform(self._fidget_min, self._fidget_max)


class FleeingBehavior(CrowdBehavior):
    """Fleeing from a threat.

    Agent moves away from flee source with increased speed.
    """

    def __init__(
        self,
        flee_speed_multiplier: float = CROWD_BEHAVIOR_CONFIG.FLEE_SPEED_MULTIPLIER,
        safe_distance: float = CROWD_BEHAVIOR_CONFIG.FLEE_SAFE_DISTANCE,
    ):
        self._speed_multiplier = flee_speed_multiplier
        self._safe_distance = safe_distance

    @property
    def name(self) -> str:
        return "fleeing"

    def update(self, agent: CrowdAgent, dt: float, context: BehaviorContext) -> None:
        agent.state_time += dt

        if agent.flee_source is None:
            agent.target_velocity = Vec3.zero()
            return

        # Calculate flee direction
        from_threat = agent.position - agent.flee_source
        distance = from_threat.length()

        if distance < CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON:
            # Pick random direction if exactly at threat
            angle = random.uniform(0, 2 * math.pi)
            flee_dir = Vec3(math.sin(angle), 0, math.cos(angle))
        else:
            flee_dir = from_threat.normalized()

        # Apply avoidance from other agents
        for other in context.get_nearby_agents(agent, 2.0):
            to_agent = agent.position - other.position
            if to_agent.length() > 0.01:
                flee_dir = (flee_dir + to_agent.normalized() * 0.5).normalized()

        # Set velocity
        flee_speed = agent.speed * self._speed_multiplier
        agent.target_velocity = flee_dir * flee_speed

        # Quick acceleration when fleeing (using config for flee acceleration)
        velocity_diff = agent.target_velocity - agent.velocity
        agent.velocity = agent.velocity + velocity_diff * min(1.0, dt * CROWD_BEHAVIOR_CONFIG.FLEE_ACCELERATION)

        # Update position
        agent.position = agent.position + agent.velocity * dt

        # Update facing
        if agent.velocity.length_squared() > 0.01:
            agent.facing = math.atan2(agent.velocity.x, agent.velocity.z)

        # Always use run animation when fleeing
        agent.animation_blend = AnimationBlend.single(agent.run_animation)

    def on_enter(self, agent: CrowdAgent) -> None:
        super().on_enter(agent)
        agent.current_state = AgentState.FLEEING


class FormationBehavior(CrowdBehavior):
    """Formation movement behavior.

    Agent maintains position relative to formation leader/center.
    """

    def __init__(
        self,
        formation_tightness: float = 0.5,
    ):
        self._tightness = formation_tightness

    @property
    def name(self) -> str:
        return "formation"

    def update(self, agent: CrowdAgent, dt: float, context: BehaviorContext) -> None:
        agent.state_time += dt

        if agent.formation_offset is None:
            agent.target_velocity = Vec3.zero()
            return

        # Find formation leader (lowest ID in same group)
        leader = self._find_leader(agent, context)
        if leader is None:
            agent.target_velocity = Vec3.zero()
            return

        # Calculate target position relative to leader
        target = leader.position + agent.formation_offset
        to_target = target - agent.position
        distance = to_target.length()

        if distance < 0.2:
            agent.target_velocity = Vec3.zero()
        else:
            # Move toward formation position
            speed_factor = min(1.0, distance * self._tightness)
            agent.target_velocity = to_target.normalized() * agent.speed * speed_factor

        # Smooth velocity
        velocity_diff = agent.target_velocity - agent.velocity
        agent.velocity = agent.velocity + velocity_diff * min(1.0, dt * 4.0)

        # Update position
        agent.position = agent.position + agent.velocity * dt

        # Face same direction as leader
        facing_diff = leader.facing - agent.facing
        while facing_diff > math.pi:
            facing_diff -= 2 * math.pi
        while facing_diff < -math.pi:
            facing_diff += 2 * math.pi
        agent.facing += facing_diff * min(1.0, dt * 2.0)

        # Update animation
        speed = agent.velocity.length()
        if speed < agent.speed * 0.1:
            agent.animation_blend = AnimationBlend.single(agent.idle_animation)
        else:
            agent.animation_blend = AnimationBlend.single(agent.walk_animation)

    def _find_leader(self, agent: CrowdAgent, context: BehaviorContext) -> CrowdAgent | None:
        """Find formation leader (lowest ID in group)."""
        leader = None
        for other in context.all_agents:
            if other.group_id == agent.group_id:
                if leader is None or other.agent_id < leader.agent_id:
                    leader = other
        return leader if leader and leader.agent_id != agent.agent_id else None

    def on_enter(self, agent: CrowdAgent) -> None:
        super().on_enter(agent)
        agent.current_state = AgentState.FORMATION


class CrowdSimulator:
    """Crowd simulation manager.

    Manages agents and behaviors, runs simulation updates.
    """

    def __init__(self):
        self._agents: list[CrowdAgent] = []
        self._behaviors: dict[AgentState, CrowdBehavior] = {}
        self._context = BehaviorContext()
        self._time = 0.0

        # Register default behaviors
        self.register_behavior(AgentState.IDLE, IdleBehavior())
        self.register_behavior(AgentState.WALKING, WalkingBehavior())
        self.register_behavior(AgentState.WAITING, WaitingBehavior())
        self.register_behavior(AgentState.FLEEING, FleeingBehavior())
        self.register_behavior(AgentState.FORMATION, FormationBehavior())

    @property
    def agent_count(self) -> int:
        """Number of agents in simulation."""
        return len(self._agents)

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._time

    def register_behavior(self, state: AgentState, behavior: CrowdBehavior) -> None:
        """Register a behavior for a state."""
        self._behaviors[state] = behavior

    def add_agent(self, agent: CrowdAgent) -> int:
        """Add agent to simulation.

        Returns:
            Agent ID
        """
        self._agents.append(agent)
        behavior = self._behaviors.get(agent.current_state)
        if behavior:
            behavior.on_enter(agent)
        return agent.agent_id

    def remove_agent(self, agent_id: int) -> bool:
        """Remove agent from simulation."""
        for i, agent in enumerate(self._agents):
            if agent.agent_id == agent_id:
                behavior = self._behaviors.get(agent.current_state)
                if behavior:
                    behavior.on_exit(agent)
                self._agents.pop(i)
                return True
        return False

    def get_agent(self, agent_id: int) -> CrowdAgent | None:
        """Get agent by ID."""
        for agent in self._agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def transition_agent(self, agent: CrowdAgent, new_state: AgentState) -> bool:
        """Transition agent to new state.

        Returns:
            True if transition succeeded
        """
        if agent.current_state == new_state:
            return True

        old_behavior = self._behaviors.get(agent.current_state)
        new_behavior = self._behaviors.get(new_state)

        # Check if transition is allowed
        if old_behavior and not old_behavior.can_transition_to(agent, new_state):
            return False

        # Perform transition
        if old_behavior:
            old_behavior.on_exit(agent)

        agent.current_state = new_state

        if new_behavior:
            new_behavior.on_enter(agent)

        return True

    def set_obstacles(self, obstacles: list[tuple[Vec3, float]]) -> None:
        """Set obstacles for avoidance."""
        self._context.obstacles = obstacles

    def add_obstacle(self, position: Vec3, radius: float) -> None:
        """Add a single obstacle."""
        self._context.obstacles.append((position, radius))

    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self._context.obstacles.clear()

    def update(self, dt: float) -> None:
        """Update all agents.

        Args:
            dt: Delta time in seconds
        """
        self._time += dt

        # Update context
        self._context.all_agents = self._agents
        self._context.time = self._time

        # Update each agent with appropriate behavior
        for agent in self._agents:
            behavior = self._behaviors.get(agent.current_state)
            if behavior:
                behavior.update(agent, dt, self._context)

    def get_agents_in_state(self, state: AgentState) -> list[CrowdAgent]:
        """Get all agents in given state."""
        return [a for a in self._agents if a.current_state == state]

    def get_agents_in_radius(self, center: Vec3, radius: float) -> list[CrowdAgent]:
        """Get all agents within radius of center point."""
        return [a for a in self._agents if a.position.distance(center) <= radius]

    def trigger_flee(self, threat_position: Vec3, radius: float) -> int:
        """Make agents within radius flee from threat.

        Returns:
            Number of agents affected
        """
        affected = 0
        for agent in self.get_agents_in_radius(threat_position, radius):
            agent.flee_source = threat_position
            if self.transition_agent(agent, AgentState.FLEEING):
                affected += 1
        return affected

    def clear(self) -> None:
        """Remove all agents."""
        self._agents.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get simulation statistics."""
        state_counts: dict[str, int] = {}
        for agent in self._agents:
            state_name = agent.current_state.name
            state_counts[state_name] = state_counts.get(state_name, 0) + 1

        return {
            "agent_count": len(self._agents),
            "simulation_time": self._time,
            "state_counts": state_counts,
            "obstacle_count": len(self._context.obstacles),
        }
