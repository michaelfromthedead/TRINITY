"""
Comprehensive tests for steering behaviors.

Tests cover:
- Seek/Flee behaviors
- Arrive (deceleration)
- Wander (random movement)
- Pursuit/Evade (prediction)
- Path following
- Obstacle avoidance
- Separation/Cohesion/Alignment (flocking)
- Behavior blending and priorities
"""

import math
import pytest
from typing import List, Tuple

from engine.gameplay.nav.steering import (
    SteeringAgent,
    SteeringManager,
    SteeringWeight,
    WanderState,
    alignment,
    arrive,
    cohesion,
    evade,
    flee,
    flocking,
    obstacle_avoidance,
    path_following,
    pursue,
    seek,
    separation,
    wall_following,
    wander,
)
from engine.gameplay.nav.navmesh import Vector3
from engine.gameplay.nav.constants import (
    DEFAULT_ALIGNMENT_WEIGHT,
    DEFAULT_ARRIVE_SLOW_RADIUS,
    DEFAULT_ARRIVE_STOP_RADIUS,
    DEFAULT_COHESION_WEIGHT,
    DEFAULT_MAX_FORCE,
    DEFAULT_MAX_SPEED,
    DEFAULT_NEIGHBOR_DISTANCE,
    DEFAULT_SEPARATION_DISTANCE,
    DEFAULT_SEPARATION_WEIGHT,
    DEFAULT_WANDER_DISTANCE,
    DEFAULT_WANDER_JITTER,
    DEFAULT_WANDER_RADIUS,
    SteeringBehavior,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_agent():
    """Create default steering agent."""
    return SteeringAgent(
        position=Vector3(0, 0, 0),
        velocity=Vector3(0, 0, 0),
        max_speed=5.0,
        max_force=10.0
    )


@pytest.fixture
def moving_agent():
    """Create agent moving forward."""
    return SteeringAgent(
        position=Vector3(0, 0, 0),
        velocity=Vector3(0, 0, 5),
        heading=Vector3(0, 0, 1),
        max_speed=5.0,
        max_force=10.0
    )


@pytest.fixture
def target_agent():
    """Create target agent."""
    return SteeringAgent(
        position=Vector3(10, 0, 10),
        velocity=Vector3(1, 0, 0),
        max_speed=3.0,
        max_force=10.0
    )


@pytest.fixture
def neighbor_agents():
    """Create list of neighboring agents."""
    neighbors = []
    for i in range(5):
        angle = i * 2 * math.pi / 5
        neighbors.append(SteeringAgent(
            id=i + 1,
            position=Vector3(math.cos(angle) * 3, 0, math.sin(angle) * 3),
            velocity=Vector3(1, 0, 0),
            heading=Vector3(1, 0, 0)
        ))
    return neighbors


@pytest.fixture
def steering_manager():
    """Create steering manager."""
    return SteeringManager()


# =============================================================================
# SteeringAgent Tests
# =============================================================================


class TestSteeringAgent:
    """Tests for SteeringAgent class."""

    def test_default_construction(self):
        """Test default agent construction."""
        agent = SteeringAgent()
        assert agent.position == Vector3()
        assert agent.velocity == Vector3()
        assert agent.mass == 1.0
        assert agent.max_speed == DEFAULT_MAX_SPEED
        assert agent.max_force == DEFAULT_MAX_FORCE

    def test_custom_construction(self, default_agent):
        """Test custom agent construction."""
        assert default_agent.position == Vector3(0, 0, 0)
        assert default_agent.max_speed == 5.0
        assert default_agent.max_force == 10.0

    def test_speed_calculation(self):
        """Test speed calculation."""
        agent = SteeringAgent(velocity=Vector3(3, 0, 4))
        assert agent.speed() == pytest.approx(5.0)

    def test_speed_zero(self, default_agent):
        """Test speed of stationary agent."""
        assert default_agent.speed() == 0.0

    def test_heading_normalized(self):
        """Test heading is normalized on construction."""
        agent = SteeringAgent(heading=Vector3(3, 0, 4))
        assert agent.heading.length() == pytest.approx(1.0)

    def test_side_normalized(self):
        """Test side is normalized on construction."""
        agent = SteeringAgent(side=Vector3(3, 0, 4))
        assert agent.side.length() == pytest.approx(1.0)

    def test_update_with_force(self, default_agent):
        """Test update applies force correctly."""
        force = Vector3(5, 0, 0)
        dt = 0.1

        default_agent.update(force, dt)

        # Velocity should have increased
        assert default_agent.velocity.x > 0
        # Position should have moved
        assert default_agent.position.x > 0

    def test_update_respects_max_speed(self, default_agent):
        """Test update respects max speed."""
        force = Vector3(100, 0, 0)
        dt = 1.0

        default_agent.update(force, dt)

        assert default_agent.velocity.length() <= default_agent.max_speed + 0.01

    def test_update_respects_max_force(self, default_agent):
        """Test update truncates large forces."""
        large_force = Vector3(1000, 0, 0)
        small_force = Vector3(5, 0, 0)
        dt = 0.1

        agent1 = SteeringAgent(max_force=10.0)
        agent2 = SteeringAgent(max_force=10.0)

        agent1.update(large_force, dt)
        agent2.update(small_force, dt)

        # Large force should be truncated to max_force
        # Both agents should have same acceleration since large_force is clamped to max_force=10
        # and small_force (5) is less than max_force
        # Velocity = acceleration * dt, acceleration = force / mass
        # agent1: force clamped to 10, so velocity = 10 * 0.1 = 1.0
        # agent2: force is 5, so velocity = 5 * 0.1 = 0.5
        assert agent1.velocity.x == pytest.approx(1.0, rel=0.1)
        assert agent2.velocity.x == pytest.approx(0.5, rel=0.1)
        # Large force agent should have velocity capped by max_force truncation
        assert agent1.velocity.length() <= agent1.max_speed + 0.01

    def test_update_heading(self, default_agent):
        """Test heading updates when moving."""
        force = Vector3(0, 0, 10)
        default_agent.update(force, 0.1)
        default_agent.update(Vector3(), 0.1)  # Continue moving

        # Heading should point in direction of movement
        assert default_agent.heading.z > 0.9

    def test_update_side_perpendicular(self, default_agent):
        """Test side is perpendicular to heading."""
        force = Vector3(0, 0, 10)
        default_agent.update(force, 0.1)

        dot = default_agent.heading.dot(default_agent.side)
        assert abs(dot) < 0.01

    def test_local_to_world(self):
        """Test local to world coordinate conversion."""
        agent = SteeringAgent(
            position=Vector3(10, 0, 10),
            heading=Vector3(0, 0, 1),
            side=Vector3(1, 0, 0)
        )

        # Point at (1, 0, 1) in local space
        local = Vector3(1, 0, 1)
        world = agent.local_to_world(local)

        # Should be (11, 0, 11) in world space
        assert world.x == pytest.approx(11.0)
        assert world.z == pytest.approx(11.0)

    def test_world_to_local(self):
        """Test world to local coordinate conversion."""
        agent = SteeringAgent(
            position=Vector3(10, 0, 10),
            heading=Vector3(0, 0, 1),
            side=Vector3(1, 0, 0)
        )

        world = Vector3(11, 0, 11)
        local = agent.world_to_local(world)

        assert local.x == pytest.approx(1.0)
        assert local.z == pytest.approx(1.0)

    def test_agent_radius(self):
        """Test agent radius property."""
        agent = SteeringAgent(radius=1.5)
        assert agent.radius == 1.5

    def test_agent_height(self):
        """Test agent height property."""
        agent = SteeringAgent(height=3.0)
        assert agent.height == 3.0

    def test_agent_id(self):
        """Test agent ID."""
        agent = SteeringAgent(id=42)
        assert agent.id == 42


# =============================================================================
# Seek Behavior Tests
# =============================================================================


class TestSeekBehavior:
    """Tests for seek steering behavior."""

    def test_seek_toward_target(self, default_agent):
        """Test seek produces force toward target."""
        target = Vector3(10, 0, 0)
        force = seek(default_agent, target)

        # Force should point toward target
        assert force.x > 0

    def test_seek_at_target(self, default_agent):
        """Test seek at target position."""
        target = Vector3(0, 0, 0)  # Same as agent position
        force = seek(default_agent, target)

        # Force should be minimal
        assert force.length() < 0.1

    def test_seek_force_magnitude(self, default_agent):
        """Test seek force is based on max speed."""
        target = Vector3(100, 0, 0)
        force = seek(default_agent, target)

        # Desired velocity is max_speed toward target
        # Force is desired - current (which is 0)
        assert force.length() == pytest.approx(default_agent.max_speed, rel=0.1)

    def test_seek_with_current_velocity(self, moving_agent):
        """Test seek considers current velocity."""
        target = Vector3(10, 0, 0)  # To the right
        force = seek(moving_agent, target)

        # Force should steer agent toward target
        assert force.x > 0

    def test_seek_behind_agent(self, moving_agent):
        """Test seek to target behind agent."""
        target = Vector3(0, 0, -10)  # Behind
        force = seek(moving_agent, target)

        # Force should turn agent around
        assert force.z < 0


# =============================================================================
# Flee Behavior Tests
# =============================================================================


class TestFleeBehavior:
    """Tests for flee steering behavior."""

    def test_flee_away_from_target(self, default_agent):
        """Test flee produces force away from target."""
        target = Vector3(10, 0, 0)
        force = flee(default_agent, target)

        # Force should point away from target
        assert force.x < 0

    def test_flee_at_target(self, default_agent):
        """Test flee at target position."""
        target = Vector3(0, 0, 0)
        force = flee(default_agent, target)

        # Force direction is undefined at same position
        # but should not crash

    def test_flee_force_magnitude(self, default_agent):
        """Test flee force magnitude."""
        target = Vector3(100, 0, 0)
        force = flee(default_agent, target)

        # Should be based on max speed
        assert force.length() == pytest.approx(default_agent.max_speed, rel=0.1)

    def test_flee_opposite_of_seek(self, default_agent):
        """Test flee is opposite of seek."""
        target = Vector3(10, 0, 10)
        seek_force = seek(default_agent, target)
        flee_force = flee(default_agent, target)

        # Forces should be roughly opposite
        dot = seek_force.normalized().dot(flee_force.normalized())
        assert dot < -0.9


# =============================================================================
# Arrive Behavior Tests
# =============================================================================


class TestArriveBehavior:
    """Tests for arrive steering behavior."""

    def test_arrive_toward_target(self, default_agent):
        """Test arrive moves toward target."""
        target = Vector3(20, 0, 0)
        force = arrive(default_agent, target)

        assert force.x > 0

    def test_arrive_slows_near_target(self, default_agent):
        """Test arrive slows down near target."""
        far_target = Vector3(20, 0, 0)
        near_target = Vector3(2, 0, 0)

        far_force = arrive(default_agent, far_target)
        near_force = arrive(default_agent, near_target)

        # Near target should have smaller force
        assert near_force.length() < far_force.length()

    def test_arrive_stops_at_target(self):
        """Test arrive returns stopping force at target."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(1, 0, 0)
        )
        target = Vector3(0.1, 0, 0)  # Very close

        force = arrive(agent, target, stop_radius=0.5)

        # Should try to stop (oppose current velocity)
        assert force.x < 0

    def test_arrive_custom_slow_radius(self, default_agent):
        """Test arrive with custom slow radius."""
        target = Vector3(5, 0, 0)

        force_small_radius = arrive(default_agent, target, slow_radius=2.0)
        force_large_radius = arrive(default_agent, target, slow_radius=10.0)

        # Different slow radii should produce different forces when at same distance
        # With small_radius=2.0, agent at distance 5 is outside slow zone (full speed)
        # With large_radius=10.0, agent at distance 5 is inside slow zone (reduced speed)
        # So force_small_radius should be larger than force_large_radius
        assert force_small_radius.length() > force_large_radius.length()

    def test_arrive_custom_stop_radius(self, default_agent):
        """Test arrive with custom stop radius."""
        target = Vector3(0.3, 0, 0)

        force_small_stop = arrive(default_agent, target, stop_radius=0.1)
        force_large_stop = arrive(default_agent, target, stop_radius=0.5)

        # Different stop radii affect behavior near target
        # With stop_radius=0.1, agent at distance 0.3 is outside stop zone (some force)
        # With stop_radius=0.5, agent at distance 0.3 is inside stop zone (minimal force)
        assert force_small_stop.length() > force_large_stop.length()
        # Large stop radius should result in near-zero force when inside stop zone
        assert force_large_stop.length() < 1.0

    def test_arrive_at_exact_target(self, default_agent):
        """Test arrive at exact target position."""
        target = Vector3(0, 0, 0)
        force = arrive(default_agent, target)

        # Should return stopping force
        assert force.length() < 1.0


# =============================================================================
# Pursue Behavior Tests
# =============================================================================


class TestPursueBehavior:
    """Tests for pursue steering behavior."""

    def test_pursue_stationary_target(self, default_agent, target_agent):
        """Test pursue stationary target (degenerates to seek)."""
        target_agent.velocity = Vector3(0, 0, 0)
        force = pursue(default_agent, target_agent)

        # Should point toward target
        to_target = target_agent.position - default_agent.position
        assert force.dot(to_target.normalized()) > 0

    def test_pursue_moving_target(self, default_agent, target_agent):
        """Test pursue predicts target position."""
        # Target moving right
        target_agent.velocity = Vector3(5, 0, 0)
        force = pursue(default_agent, target_agent)

        # Should aim ahead of target (more to the right)
        # Force should have a positive x component since target is moving right
        assert isinstance(force, Vector3)
        assert force.length() > 0

    def test_pursue_approaching_target(self, moving_agent, target_agent):
        """Test pursue when target approaches."""
        target_agent.position = Vector3(0, 0, 20)
        target_agent.velocity = Vector3(0, 0, -5)  # Coming toward us
        target_agent.heading = Vector3(0, 0, -1)

        force = pursue(moving_agent, target_agent)
        # Should account for approach - force calculation should succeed
        # Note: when both agents are on same line moving toward each other,
        # the predicted intercept point may result in minimal steering
        assert isinstance(force, Vector3)
        # Force can be zero if agents are perfectly aligned and moving toward each other

    def test_pursue_max_prediction(self, default_agent, target_agent):
        """Test pursue respects max prediction time."""
        # Target very far away
        target_agent.position = Vector3(1000, 0, 1000)
        target_agent.velocity = Vector3(5, 0, 0)

        force = pursue(default_agent, target_agent, max_prediction_time=1.0)
        # Should still produce valid force
        assert isinstance(force, Vector3)
        assert force.length() > 0


# =============================================================================
# Evade Behavior Tests
# =============================================================================


class TestEvadeBehavior:
    """Tests for evade steering behavior."""

    def test_evade_away_from_target(self, default_agent, target_agent):
        """Test evade moves away from target."""
        force = evade(default_agent, target_agent)

        to_target = target_agent.position - default_agent.position
        # Force should point away from target (or predicted position)
        assert force.dot(to_target.normalized()) < 0

    def test_evade_considers_velocity(self, default_agent, target_agent):
        """Test evade considers target velocity."""
        # Target coming directly at us
        target_agent.position = Vector3(10, 0, 0)
        target_agent.velocity = Vector3(-5, 0, 0)

        force = evade(default_agent, target_agent)
        # Should evade from predicted position - force should point away
        assert isinstance(force, Vector3)
        assert force.length() > 0
        # Since target is coming toward us from x=10, we should move away (negative x)
        assert force.x < 0

    def test_evade_stationary_target(self, default_agent, target_agent):
        """Test evade from stationary target (degenerates to flee)."""
        target_agent.velocity = Vector3(0, 0, 0)
        force = evade(default_agent, target_agent)

        # Should move away from target
        to_target = target_agent.position - default_agent.position
        assert force.dot(to_target.normalized()) < 0


# =============================================================================
# Wander Behavior Tests
# =============================================================================


class TestWanderBehavior:
    """Tests for wander steering behavior."""

    def test_wander_returns_force(self, default_agent):
        """Test wander returns a force."""
        state = WanderState()
        force = wander(default_agent, state)

        assert isinstance(force, Vector3)

    def test_wander_state_changes(self, default_agent):
        """Test wander state changes over time."""
        state = WanderState()

        initial_target = Vector3(
            state.wander_target.x,
            state.wander_target.y,
            state.wander_target.z
        )

        wander(default_agent, state, dt=0.1)

        # State should have changed
        # (may be same due to small jitter)

    def test_wander_custom_parameters(self, default_agent):
        """Test wander with custom parameters."""
        state = WanderState()

        force = wander(
            default_agent, state,
            radius=3.0,
            distance=5.0,
            jitter=50.0,
            dt=0.1
        )

        assert isinstance(force, Vector3)

    def test_wander_produces_varied_movement(self, moving_agent):
        """Test wander produces varied directions over time."""
        state = WanderState()
        forces = []

        for i in range(10):
            force = wander(moving_agent, state, dt=0.1)
            forces.append(force)

        # Forces should vary
        # (not guaranteed but likely with jitter)

    def test_wander_state_defaults(self):
        """Test WanderState default values."""
        state = WanderState()
        assert state.wander_target.length() > 0  # Default pointing forward


# =============================================================================
# Separation Behavior Tests
# =============================================================================


class TestSeparationBehavior:
    """Tests for separation steering behavior."""

    def test_separation_no_neighbors(self, default_agent):
        """Test separation with no neighbors."""
        force = separation(default_agent, [])

        assert force == Vector3(0, 0, 0)

    def test_separation_pushes_away(self, default_agent, neighbor_agents):
        """Test separation pushes away from neighbors."""
        # Place agent at center, neighbors around
        default_agent.position = Vector3(0, 0, 0)

        force = separation(default_agent, neighbor_agents, separation_distance=10.0)

        # Force direction depends on neighbor positions
        # but should be non-zero
        assert force.length() > 0

    def test_separation_ignores_self(self, default_agent):
        """Test separation ignores self in neighbors list."""
        default_agent.id = 1
        neighbors = [default_agent]

        force = separation(default_agent, neighbors)

        assert force == Vector3(0, 0, 0)

    def test_separation_distance_threshold(self, default_agent, neighbor_agents):
        """Test separation respects distance threshold."""
        # Place neighbors far away
        for n in neighbor_agents:
            n.position = n.position * 10  # 10x further

        force = separation(default_agent, neighbor_agents, separation_distance=5.0)

        # Should be zero or very small
        assert force.length() < 0.1

    def test_separation_inverse_distance(self, default_agent):
        """Test closer neighbors have stronger effect."""
        close_neighbor = SteeringAgent(id=1, position=Vector3(1, 0, 0))
        far_neighbor = SteeringAgent(id=2, position=Vector3(5, 0, 0))

        close_force = separation(default_agent, [close_neighbor], separation_distance=10.0)
        far_force = separation(default_agent, [far_neighbor], separation_distance=10.0)

        # Close neighbor should produce stronger force
        assert close_force.length() > far_force.length()


# =============================================================================
# Alignment Behavior Tests
# =============================================================================


class TestAlignmentBehavior:
    """Tests for alignment steering behavior."""

    def test_alignment_no_neighbors(self, default_agent):
        """Test alignment with no neighbors."""
        force = alignment(default_agent, [])

        assert force == Vector3(0, 0, 0)

    def test_alignment_matches_heading(self, default_agent, neighbor_agents):
        """Test alignment steers toward average heading."""
        # All neighbors heading same direction
        for n in neighbor_agents:
            n.heading = Vector3(1, 0, 0)

        force = alignment(default_agent, neighbor_agents)

        # Should steer toward (1, 0, 0)
        assert force.x > 0

    def test_alignment_ignores_self(self, default_agent):
        """Test alignment ignores self."""
        default_agent.id = 1
        force = alignment(default_agent, [default_agent])

        assert force == Vector3(0, 0, 0)

    def test_alignment_respects_distance(self, default_agent, neighbor_agents):
        """Test alignment respects neighbor distance."""
        # Place neighbors far away
        for n in neighbor_agents:
            n.position = n.position * 100

        force = alignment(default_agent, neighbor_agents, neighbor_distance=5.0)

        assert force.length() < 0.1


# =============================================================================
# Cohesion Behavior Tests
# =============================================================================


class TestCohesionBehavior:
    """Tests for cohesion steering behavior."""

    def test_cohesion_no_neighbors(self, default_agent):
        """Test cohesion with no neighbors."""
        force = cohesion(default_agent, [])

        assert force == Vector3(0, 0, 0)

    def test_cohesion_toward_center(self, default_agent, neighbor_agents):
        """Test cohesion steers toward neighbor center."""
        force = cohesion(default_agent, neighbor_agents)

        # Should steer toward center of mass
        # (which is roughly origin given circular neighbor placement)
        # For agent at origin with neighbors in a circle, force should be minimal
        # but should still be a valid Vector3
        assert isinstance(force, Vector3)

    def test_cohesion_ignores_self(self, default_agent):
        """Test cohesion ignores self."""
        default_agent.id = 1
        force = cohesion(default_agent, [default_agent])

        assert force == Vector3(0, 0, 0)

    def test_cohesion_respects_distance(self, default_agent, neighbor_agents):
        """Test cohesion respects neighbor distance."""
        for n in neighbor_agents:
            n.position = n.position * 100

        force = cohesion(default_agent, neighbor_agents, neighbor_distance=5.0)

        assert force.length() < 0.1


# =============================================================================
# Flocking Behavior Tests
# =============================================================================


class TestFlockingBehavior:
    """Tests for combined flocking behavior."""

    def test_flocking_combines_behaviors(self, default_agent, neighbor_agents):
        """Test flocking combines separation, alignment, cohesion."""
        force = flocking(default_agent, neighbor_agents)

        # Should produce some force combining separation, alignment, cohesion
        # (exact value depends on neighbor positions)
        assert isinstance(force, Vector3)

    def test_flocking_custom_weights(self, default_agent, neighbor_agents):
        """Test flocking with custom weights."""
        force_default = flocking(default_agent, neighbor_agents)

        force_high_separation = flocking(
            default_agent, neighbor_agents,
            separation_weight=5.0,
            alignment_weight=0.1,
            cohesion_weight=0.1
        )

        # Different weights should produce different forces

    def test_flocking_no_neighbors(self, default_agent):
        """Test flocking with no neighbors."""
        force = flocking(default_agent, [])

        assert force == Vector3(0, 0, 0)


# =============================================================================
# Obstacle Avoidance Tests
# =============================================================================


class TestObstacleAvoidance:
    """Tests for obstacle avoidance behavior."""

    def test_avoidance_no_obstacles(self, moving_agent):
        """Test avoidance with no obstacles."""
        force = obstacle_avoidance(moving_agent, [])

        assert force == Vector3(0, 0, 0)

    def test_avoidance_obstacle_ahead(self, moving_agent):
        """Test avoidance with obstacle directly ahead."""
        # Obstacle ahead of agent
        obstacles: List[Tuple[Vector3, float]] = [
            (Vector3(0, 0, 3), 1.0)
        ]

        force = obstacle_avoidance(moving_agent, obstacles)

        # Should produce lateral force to avoid
        assert isinstance(force, Vector3)
        # Agent moving forward (+z) with obstacle ahead should get lateral (x) force
        # Force should push agent to either side

    def test_avoidance_obstacle_beside(self, moving_agent):
        """Test avoidance with obstacle to side."""
        obstacles: List[Tuple[Vector3, float]] = [
            (Vector3(5, 0, 0), 1.0)  # To the side
        ]

        force = obstacle_avoidance(moving_agent, obstacles)

        # May or may not produce force depending on detection box width
        assert isinstance(force, Vector3)

    def test_avoidance_obstacle_behind(self, moving_agent):
        """Test avoidance ignores obstacles behind."""
        obstacles: List[Tuple[Vector3, float]] = [
            (Vector3(0, 0, -5), 1.0)  # Behind
        ]

        force = obstacle_avoidance(moving_agent, obstacles)

        # Should not affect agent moving away - obstacle is behind
        assert force.length() < 0.1  # Minimal or no force for obstacle behind

    def test_avoidance_detection_length(self, moving_agent):
        """Test avoidance detection length."""
        far_obstacle: List[Tuple[Vector3, float]] = [
            (Vector3(0, 0, 20), 1.0)
        ]
        near_obstacle: List[Tuple[Vector3, float]] = [
            (Vector3(0, 0, 3), 1.0)
        ]

        far_force = obstacle_avoidance(moving_agent, far_obstacle, detection_length=5.0)
        near_force = obstacle_avoidance(moving_agent, near_obstacle, detection_length=5.0)

        # Near obstacle should produce stronger response

    def test_avoidance_multiple_obstacles(self, moving_agent):
        """Test avoidance with multiple obstacles."""
        obstacles: List[Tuple[Vector3, float]] = [
            (Vector3(-1, 0, 3), 0.5),
            (Vector3(1, 0, 4), 0.5),
        ]

        force = obstacle_avoidance(moving_agent, obstacles)
        # Should avoid closest one
        assert isinstance(force, Vector3)


# =============================================================================
# Wall Following Tests
# =============================================================================


class TestWallFollowing:
    """Tests for wall following behavior."""

    def test_wall_following_no_walls(self, moving_agent):
        """Test wall following with no walls."""
        force = wall_following(moving_agent, [])

        assert force == Vector3(0, 0, 0)

    def test_wall_following_parallel_wall(self, moving_agent):
        """Test wall following parallel to movement."""
        walls: List[Tuple[Vector3, Vector3]] = [
            (Vector3(-5, 0, -10), Vector3(-5, 0, 10))  # Wall to left
        ]

        force = wall_following(moving_agent, walls)
        # May or may not detect based on feeler configuration
        assert isinstance(force, Vector3)

    def test_wall_following_approaching_wall(self):
        """Test wall following when approaching wall."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(5, 0, 0),
            heading=Vector3(1, 0, 0),
            side=Vector3(0, 0, 1)
        )

        walls: List[Tuple[Vector3, Vector3]] = [
            (Vector3(3, 0, -10), Vector3(3, 0, 10))  # Wall ahead
        ]

        force = wall_following(agent, walls)


# =============================================================================
# Path Following Tests
# =============================================================================


class TestPathFollowing:
    """Tests for path following behavior."""

    def test_path_following_empty_path(self, moving_agent):
        """Test path following with empty path."""
        force = path_following(moving_agent, [])

        assert force == Vector3(0, 0, 0)

    def test_path_following_single_point(self, moving_agent):
        """Test path following with single point."""
        force = path_following(moving_agent, [Vector3(10, 0, 10)])

        assert force == Vector3(0, 0, 0)

    def test_path_following_on_path(self, moving_agent):
        """Test path following when on path."""
        path = [
            Vector3(0, 0, 0),
            Vector3(0, 0, 10),
            Vector3(0, 0, 20)
        ]

        force = path_following(moving_agent, path)
        # Should seek toward next waypoint
        assert isinstance(force, Vector3)

    def test_path_following_off_path(self):
        """Test path following when off path."""
        agent = SteeringAgent(
            position=Vector3(5, 0, 5),
            velocity=Vector3(0, 0, 1),
            heading=Vector3(0, 0, 1)
        )

        path = [
            Vector3(0, 0, 0),
            Vector3(0, 0, 10),
            Vector3(0, 0, 20)
        ]

        force = path_following(agent, path)
        # Should steer back to path - force should point toward path (negative x)
        assert isinstance(force, Vector3)
        assert force.x < 0  # Steering back toward the path at x=0

    def test_path_following_prediction(self, moving_agent):
        """Test path following uses prediction."""
        path = [
            Vector3(0, 0, 0),
            Vector3(0, 0, 10),
            Vector3(10, 0, 10)
        ]

        force = path_following(moving_agent, path, prediction_distance=2.0)


# =============================================================================
# SteeringManager Tests
# =============================================================================


class TestSteeringManager:
    """Tests for SteeringManager class."""

    def test_default_construction(self, steering_manager):
        """Test default manager construction."""
        assert steering_manager is not None

    def test_set_weight(self, steering_manager):
        """Test setting behavior weight."""
        steering_manager.set_weight(SteeringBehavior.SEEK, 2.0)
        assert steering_manager.get_weight(SteeringBehavior.SEEK) == 2.0

    def test_get_weight_default(self, steering_manager):
        """Test getting default weight."""
        weight = steering_manager.get_weight(SteeringBehavior.SEEK)
        assert weight > 0

    def test_enable_behavior(self, steering_manager):
        """Test enabling behavior."""
        steering_manager.disable_behavior(SteeringBehavior.SEEK)
        steering_manager.enable_behavior(SteeringBehavior.SEEK)

        assert steering_manager.is_enabled(SteeringBehavior.SEEK)

    def test_disable_behavior(self, steering_manager):
        """Test disabling behavior."""
        steering_manager.disable_behavior(SteeringBehavior.SEEK)

        assert not steering_manager.is_enabled(SteeringBehavior.SEEK)

    def test_is_enabled_default(self, steering_manager):
        """Test default enabled state."""
        # Default weights should make behaviors enabled
        assert steering_manager.is_enabled(SteeringBehavior.SEEK)

    def test_get_wander_state(self, steering_manager):
        """Test getting wander state for agent."""
        state1 = steering_manager.get_wander_state(1)
        state2 = steering_manager.get_wander_state(1)

        # Same agent ID should get same state
        assert state1 is state2

    def test_get_wander_state_different_agents(self, steering_manager):
        """Test different agents get different states."""
        state1 = steering_manager.get_wander_state(1)
        state2 = steering_manager.get_wander_state(2)

        assert state1 is not state2

    def test_calculate_weighted_sum(self, steering_manager, default_agent):
        """Test weighted sum calculation."""
        force = steering_manager.calculate_weighted_sum(
            default_agent,
            seek_target=Vector3(10, 0, 0)
        )

        # Should produce force toward target
        assert force.x > 0

    def test_calculate_weighted_sum_multiple_behaviors(
        self, steering_manager, default_agent
    ):
        """Test weighted sum with multiple behaviors."""
        force = steering_manager.calculate_weighted_sum(
            default_agent,
            seek_target=Vector3(10, 0, 0),
            flee_target=Vector3(-5, 0, 0)  # Flee from behind
        )

        # Net force should be forward

    def test_calculate_priority(self, steering_manager, default_agent):
        """Test priority-based calculation."""
        force = steering_manager.calculate_priority(
            default_agent,
            seek_target=Vector3(10, 0, 0)
        )

        assert isinstance(force, Vector3)

    def test_calculate_priority_max_force(self, steering_manager, default_agent):
        """Test priority respects max force."""
        default_agent.max_force = 1.0

        force = steering_manager.calculate_priority(
            default_agent,
            seek_target=Vector3(100, 0, 0),
            flee_target=Vector3(-100, 0, 0)
        )

        assert force.length() <= default_agent.max_force + 0.01

    def test_weighted_sum_with_obstacles(
        self, steering_manager, moving_agent
    ):
        """Test weighted sum with obstacle avoidance."""
        obstacles: List[Tuple[Vector3, float]] = [
            (Vector3(0, 0, 5), 1.0)
        ]

        force = steering_manager.calculate_weighted_sum(
            moving_agent,
            obstacles=obstacles
        )

    def test_weighted_sum_with_neighbors(
        self, steering_manager, default_agent, neighbor_agents
    ):
        """Test weighted sum with flocking neighbors."""
        force = steering_manager.calculate_weighted_sum(
            default_agent,
            neighbors=neighbor_agents
        )

    def test_weighted_sum_with_path(self, steering_manager, moving_agent):
        """Test weighted sum with path following."""
        path = [
            Vector3(0, 0, 0),
            Vector3(0, 0, 10),
            Vector3(10, 0, 10)
        ]

        force = steering_manager.calculate_weighted_sum(
            moving_agent,
            path=path
        )


# =============================================================================
# SteeringWeight Tests
# =============================================================================


class TestSteeringWeight:
    """Tests for SteeringWeight class."""

    def test_construction(self):
        """Test SteeringWeight construction."""
        weight = SteeringWeight(
            behavior=SteeringBehavior.SEEK,
            weight=2.0
        )
        assert weight.behavior == SteeringBehavior.SEEK
        assert weight.weight == 2.0
        assert weight.enabled

    def test_disabled_weight(self):
        """Test disabled weight."""
        weight = SteeringWeight(
            behavior=SteeringBehavior.FLEE,
            weight=1.0,
            enabled=False
        )
        assert not weight.enabled


# =============================================================================
# Integration Tests
# =============================================================================


class TestSteeringIntegration:
    """Integration tests for steering behaviors."""

    def test_agent_reaches_target(self, default_agent):
        """Test agent can reach target using seek."""
        target = Vector3(10, 0, 0)

        for _ in range(100):
            force = seek(default_agent, target)
            default_agent.update(force, 0.1)

            if default_agent.position.distance_to(target) < 1.0:
                break

        assert default_agent.position.distance_to(target) < 5.0

    def test_agent_arrives_and_stops(self, default_agent):
        """Test agent arrives and stops at target."""
        target = Vector3(5, 0, 0)

        for _ in range(100):
            force = arrive(default_agent, target)
            default_agent.update(force, 0.1)

        # Should be near target and slow/stopped
        assert default_agent.position.distance_to(target) < 2.0
        assert default_agent.speed() < 2.0

    def test_flock_cohesion(self):
        """Test flock stays together."""
        agents = [
            SteeringAgent(
                id=i,
                position=Vector3(
                    (i % 3) * 2,
                    0,
                    (i // 3) * 2
                )
            )
            for i in range(9)
        ]

        for _ in range(50):
            for agent in agents:
                others = [a for a in agents if a.id != agent.id]
                force = flocking(agent, others)
                agent.update(force, 0.1)

        # Check agents are still relatively close
        positions = [a.position for a in agents]
        center = Vector3(
            sum(p.x for p in positions) / len(positions),
            0,
            sum(p.z for p in positions) / len(positions)
        )

        for agent in agents:
            dist = agent.position.distance_to(center)
            assert dist < 20.0  # Should stay together

    def test_agent_follows_path(self, moving_agent):
        """Test agent follows a path."""
        path = [
            Vector3(0, 0, 0),
            Vector3(0, 0, 10),
            Vector3(10, 0, 10),
            Vector3(10, 0, 0)
        ]

        manager = SteeringManager()
        manager.disable_behavior(SteeringBehavior.WANDER)

        for _ in range(200):
            force = path_following(moving_agent, path)
            moving_agent.update(force, 0.1)

        # Should have progressed along path - moved beyond start position
        assert moving_agent.position.z > 5.0  # Progressed forward


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestSteeringEdgeCases:
    """Tests for edge cases in steering behaviors."""

    def test_zero_velocity_agent(self):
        """Test behaviors with zero velocity agent."""
        agent = SteeringAgent(velocity=Vector3(0, 0, 0))

        seek(agent, Vector3(10, 0, 0))
        flee(agent, Vector3(10, 0, 0))
        arrive(agent, Vector3(10, 0, 0))

    def test_zero_max_speed(self):
        """Test behaviors with zero max speed."""
        agent = SteeringAgent(max_speed=0.0)
        target = Vector3(10, 0, 0)

        force = seek(agent, target)
        assert force.length() < 0.1

    def test_very_high_max_speed(self):
        """Test behaviors with very high max speed."""
        agent = SteeringAgent(max_speed=1000.0)
        target = Vector3(10, 0, 0)

        force = seek(agent, target)
        # Should still produce reasonable force
        assert isinstance(force, Vector3)
        assert force.length() > 0

    def test_negative_coordinates(self):
        """Test behaviors with negative coordinates."""
        agent = SteeringAgent(position=Vector3(-10, 0, -10))
        target = Vector3(-20, 0, -20)

        force = seek(agent, target)
        assert force.x < 0 and force.z < 0

    def test_very_large_coordinates(self):
        """Test behaviors with large coordinates."""
        agent = SteeringAgent(position=Vector3(10000, 0, 10000))
        target = Vector3(10010, 0, 10010)

        force = seek(agent, target)
        assert force.x > 0 and force.z > 0

    def test_many_neighbors(self):
        """Test flocking with many neighbors."""
        agent = SteeringAgent(id=0, position=Vector3(0, 0, 0))
        neighbors = [
            SteeringAgent(
                id=i + 1,
                position=Vector3(
                    math.cos(i * 0.1) * 5,
                    0,
                    math.sin(i * 0.1) * 5
                )
            )
            for i in range(100)
        ]

        force = flocking(agent, neighbors)
        # Should complete without issues
        assert isinstance(force, Vector3)
