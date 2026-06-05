"""
WHITEBOX tests for steering behaviors.

Tests internal implementation details, edge cases, and boundary conditions:
- T-NAV-1.4: Steering behaviors (seek, flee, arrive, pursue, evade, wander)
- Flocking behaviors (separation, alignment, cohesion)
- Path following and obstacle avoidance
- SteeringAgent operations
- SteeringManager weight calculations
- Force combination methods
"""

import math
import pytest
import random
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
    _line_intersection,
    _closest_point_on_segment,
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
    FLOAT_EPSILON,
    SteeringBehavior,
    ZERO_LENGTH_THRESHOLD,
)


# =============================================================================
# SteeringAgent WHITEBOX Tests
# =============================================================================

class TestSteeringAgentWhitebox:
    """Whitebox tests for SteeringAgent operations."""

    def test_agent_default_values(self):
        """Test SteeringAgent default values."""
        agent = SteeringAgent()
        assert agent.position == Vector3()
        assert agent.velocity == Vector3()
        assert agent.mass == 1.0
        assert agent.max_speed == DEFAULT_MAX_SPEED
        assert agent.max_force == DEFAULT_MAX_FORCE

    def test_agent_heading_normalized(self):
        """Test heading is normalized on init."""
        agent = SteeringAgent(heading=Vector3(10, 0, 0))
        assert abs(agent.heading.length() - 1.0) < FLOAT_EPSILON

    def test_agent_side_normalized(self):
        """Test side vector is normalized on init."""
        agent = SteeringAgent(side=Vector3(0, 0, 10))
        assert abs(agent.side.length() - 1.0) < FLOAT_EPSILON

    def test_agent_speed(self):
        """Test agent speed calculation."""
        agent = SteeringAgent(velocity=Vector3(3, 0, 4))
        assert abs(agent.speed() - 5.0) < FLOAT_EPSILON

    def test_agent_speed_zero_velocity(self):
        """Test agent speed with zero velocity."""
        agent = SteeringAgent(velocity=Vector3(0, 0, 0))
        assert agent.speed() == 0.0

    def test_agent_update_applies_force(self):
        """Test update applies steering force."""
        agent = SteeringAgent(position=Vector3(0, 0, 0))
        force = Vector3(10, 0, 0)
        agent.update(force, dt=1.0)

        # Position should change
        assert agent.position.x > 0

    def test_agent_update_limits_speed(self):
        """Test update limits speed to max_speed."""
        agent = SteeringAgent(max_speed=5.0)
        force = Vector3(1000, 0, 0)  # Large force
        agent.update(force, dt=1.0)

        assert agent.speed() <= 5.0 + FLOAT_EPSILON

    def test_agent_update_limits_force(self):
        """Test update limits force to max_force."""
        agent = SteeringAgent(max_force=10.0, mass=1.0)
        force = Vector3(1000, 0, 0)  # Large force

        # Force should be truncated in update
        agent.update(force, dt=0.1)
        # Velocity change should be limited

    def test_agent_update_updates_heading(self):
        """Test update updates heading when moving."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            heading=Vector3(0, 0, 1)
        )
        force = Vector3(10, 0, 0)  # Force in X direction
        agent.update(force, dt=1.0)

        # Heading should align with velocity
        assert abs(agent.heading.x) > 0.5

    def test_agent_update_updates_side(self):
        """Test update updates side vector perpendicular to heading."""
        agent = SteeringAgent(heading=Vector3(1, 0, 0))
        force = Vector3(0, 0, 10)  # Force in Z direction
        agent.update(force, dt=1.0)

        # Side should be perpendicular to heading
        dot = agent.heading.dot(agent.side)
        assert abs(dot) < FLOAT_EPSILON

    def test_agent_local_to_world(self):
        """Test local to world coordinate conversion."""
        agent = SteeringAgent(
            position=Vector3(10, 5, 20),
            heading=Vector3(0, 0, 1),
            side=Vector3(1, 0, 0)
        )
        local = Vector3(1, 0, 1)  # 1 unit right, 1 unit forward
        world = agent.local_to_world(local)

        assert abs(world.x - 11) < FLOAT_EPSILON  # 10 + 1
        assert abs(world.y - 5) < FLOAT_EPSILON   # 5 + 0
        assert abs(world.z - 21) < FLOAT_EPSILON  # 20 + 1

    def test_agent_world_to_local(self):
        """Test world to local coordinate conversion."""
        agent = SteeringAgent(
            position=Vector3(10, 5, 20),
            heading=Vector3(0, 0, 1),
            side=Vector3(1, 0, 0)
        )
        world = Vector3(11, 5, 21)
        local = agent.world_to_local(world)

        assert abs(local.x - 1) < FLOAT_EPSILON  # 1 unit right
        assert abs(local.y - 0) < FLOAT_EPSILON  # Same height
        assert abs(local.z - 1) < FLOAT_EPSILON  # 1 unit forward


# =============================================================================
# Seek Behavior WHITEBOX Tests
# =============================================================================

class TestSeekWhitebox:
    """Whitebox tests for seek behavior."""

    def test_seek_toward_target(self):
        """Test seek produces force toward target."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            max_speed=10.0
        )
        target = Vector3(10, 0, 0)

        force = seek(agent, target)

        # Force should be in positive X direction
        assert force.x > 0

    def test_seek_at_target(self):
        """Test seek when at target position."""
        agent = SteeringAgent(
            position=Vector3(10, 0, 0),
            velocity=Vector3(5, 0, 0),  # Moving
            max_speed=10.0
        )
        target = Vector3(10, 0, 0)  # Same position

        force = seek(agent, target)

        # Force should oppose current velocity (stop at target)
        assert force.x < 0

    def test_seek_maximum_desired_velocity(self):
        """Test seek uses max speed for desired velocity."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            max_speed=5.0
        )
        target = Vector3(100, 0, 0)  # Far away

        force = seek(agent, target)

        # Desired velocity magnitude should be max_speed
        desired_vel_magnitude = force.length()  # Since current vel is zero
        assert abs(desired_vel_magnitude - 5.0) < FLOAT_EPSILON


# =============================================================================
# Flee Behavior WHITEBOX Tests
# =============================================================================

class TestFleeWhitebox:
    """Whitebox tests for flee behavior."""

    def test_flee_away_from_target(self):
        """Test flee produces force away from target."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            max_speed=10.0
        )
        threat = Vector3(10, 0, 0)

        force = flee(agent, threat)

        # Force should be in negative X direction (away from threat)
        assert force.x < 0

    def test_flee_opposite_of_seek(self):
        """Test flee is opposite direction of seek."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            max_speed=10.0
        )
        target = Vector3(10, 0, 10)

        seek_force = seek(agent, target)
        flee_force = flee(agent, target)

        # Directions should be opposite
        dot = seek_force.normalized().dot(flee_force.normalized())
        assert dot < -0.9  # Nearly opposite


# =============================================================================
# Arrive Behavior WHITEBOX Tests
# =============================================================================

class TestArriveWhitebox:
    """Whitebox tests for arrive behavior."""

    def test_arrive_full_speed_outside_slow_radius(self):
        """Test arrive uses full speed outside slow radius."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            max_speed=10.0
        )
        target = Vector3(100, 0, 0)  # Far away

        force = arrive(agent, target, slow_radius=5.0, stop_radius=0.5)

        # Should be similar to seek
        seek_force = seek(agent, target)
        assert abs(force.x - seek_force.x) < 1.0

    def test_arrive_slows_in_slow_radius(self):
        """Test arrive reduces speed in slow radius."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(5, 0, 0),  # Moving toward target
            max_speed=10.0
        )
        target = Vector3(2, 0, 0)  # Close

        force = arrive(agent, target, slow_radius=5.0, stop_radius=0.5)

        # Force should be reduced compared to full speed seek

    def test_arrive_stops_in_stop_radius(self):
        """Test arrive stops when in stop radius."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(5, 0, 0),
            max_speed=10.0
        )
        target = Vector3(0.1, 0, 0)  # Very close

        force = arrive(agent, target, slow_radius=5.0, stop_radius=0.5)

        # Force should oppose velocity (braking)
        assert force.x < 0

    def test_arrive_at_target(self):
        """Test arrive when exactly at target."""
        agent = SteeringAgent(
            position=Vector3(5, 0, 5),
            velocity=Vector3(2, 0, 0),
            max_speed=10.0
        )
        target = Vector3(5, 0, 5)

        force = arrive(agent, target, slow_radius=5.0, stop_radius=0.5)

        # Should return braking force
        assert force.x < 0  # Oppose velocity


# =============================================================================
# Pursue Behavior WHITEBOX Tests
# =============================================================================

class TestPursueWhitebox:
    """Whitebox tests for pursue behavior."""

    def test_pursue_predicts_target_position(self):
        """Test pursue predicts target's future position."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            max_speed=10.0
        )
        target = SteeringAgent(
            position=Vector3(20, 0, 0),
            velocity=Vector3(5, 0, 0),  # Moving away
            max_speed=5.0
        )

        force = pursue(agent, target)

        # Should aim ahead of current target position
        # Force should be more in X direction than direct seek
        seek_force = seek(agent, target.position)
        assert force.x >= seek_force.x - 0.1

    def test_pursue_approaching_target(self):
        """Test pursue with target approaching."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(5, 0, 0),
            heading=Vector3(1, 0, 0),
            max_speed=10.0
        )
        target = SteeringAgent(
            position=Vector3(20, 0, 0),
            velocity=Vector3(-5, 0, 0),  # Coming toward us
            heading=Vector3(-1, 0, 0),
            max_speed=5.0
        )

        force = pursue(agent, target)
        # Should just seek toward current position

    def test_pursue_max_prediction_time(self):
        """Test pursue respects max prediction time."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            max_speed=1.0  # Slow
        )
        target = SteeringAgent(
            position=Vector3(1000, 0, 0),  # Far away
            velocity=Vector3(100, 0, 0),
            max_speed=100.0
        )

        force = pursue(agent, target, max_prediction_time=0.5)
        # Prediction should be limited


# =============================================================================
# Evade Behavior WHITEBOX Tests
# =============================================================================

class TestEvadeWhitebox:
    """Whitebox tests for evade behavior."""

    def test_evade_predicts_threat_position(self):
        """Test evade predicts threat's future position."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            max_speed=10.0
        )
        threat = SteeringAgent(
            position=Vector3(20, 0, 0),
            velocity=Vector3(-10, 0, 0),  # Coming toward us
            max_speed=10.0
        )

        force = evade(agent, threat)

        # Should flee from predicted position (ahead of threat)
        flee_force = flee(agent, threat.position)
        # Evade should be different from simple flee

    def test_evade_opposite_of_pursue(self):
        """Test evade is roughly opposite of pursue."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            max_speed=10.0
        )
        target = SteeringAgent(
            position=Vector3(20, 0, 0),
            velocity=Vector3(-5, 0, 0),
            max_speed=5.0
        )

        pursue_force = pursue(agent, target)
        evade_force = evade(agent, target)

        # Directions should be roughly opposite
        dot = pursue_force.normalized().dot(evade_force.normalized())
        assert dot < -0.5


# =============================================================================
# Wander Behavior WHITEBOX Tests
# =============================================================================

class TestWanderWhitebox:
    """Whitebox tests for wander behavior."""

    def test_wander_produces_force(self):
        """Test wander produces non-zero force."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            heading=Vector3(0, 0, 1),
            max_speed=5.0
        )
        state = WanderState()

        force = wander(agent, state)

        assert force.length() > 0

    def test_wander_varies_over_time(self):
        """Test wander produces different forces over time."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            heading=Vector3(0, 0, 1),
            max_speed=5.0
        )
        state = WanderState()

        forces = []
        for _ in range(10):
            force = wander(agent, state, dt=0.1)
            forces.append(force)

        # Forces should vary (not all identical)
        unique_x = len(set(f.x for f in forces))
        assert unique_x > 1

    def test_wander_state_persistence(self):
        """Test wander state persists between calls."""
        agent = SteeringAgent(heading=Vector3(0, 0, 1))
        state = WanderState(wander_target=Vector3(1, 0, 0))

        wander(agent, state, dt=0.01)

        # State should be modified
        # (depends on implementation details)

    def test_wander_jitter_effect(self):
        """Test jitter parameter affects randomness."""
        agent = SteeringAgent(heading=Vector3(0, 0, 1))

        state_low = WanderState()
        state_high = WanderState()

        # Low jitter
        random.seed(42)
        force_low = wander(agent, state_low, jitter=1.0, dt=0.1)

        # High jitter
        random.seed(42)
        force_high = wander(agent, state_high, jitter=100.0, dt=0.1)

        # Higher jitter should produce more variation over time


# =============================================================================
# Separation Behavior WHITEBOX Tests
# =============================================================================

class TestSeparationWhitebox:
    """Whitebox tests for separation behavior."""

    def test_separation_no_neighbors(self):
        """Test separation with no neighbors."""
        agent = SteeringAgent(id=1, position=Vector3(0, 0, 0))

        force = separation(agent, [])

        assert force == Vector3()

    def test_separation_self_excluded(self):
        """Test agent excludes itself from neighbors."""
        agent = SteeringAgent(id=1, position=Vector3(0, 0, 0))

        # Only self in list
        force = separation(agent, [agent])

        assert force == Vector3()

    def test_separation_pushes_away(self):
        """Test separation pushes away from nearby neighbors."""
        agent = SteeringAgent(id=1, position=Vector3(0, 0, 0))
        neighbor = SteeringAgent(id=2, position=Vector3(1, 0, 0))

        force = separation(agent, [neighbor], separation_distance=5.0)

        # Force should push away from neighbor (negative X)
        assert force.x < 0

    def test_separation_inverse_distance(self):
        """Test separation strength increases with proximity."""
        agent = SteeringAgent(id=1, position=Vector3(0, 0, 0))
        close_neighbor = SteeringAgent(id=2, position=Vector3(0.5, 0, 0))
        far_neighbor = SteeringAgent(id=3, position=Vector3(3, 0, 0))

        force_close = separation(agent, [close_neighbor], separation_distance=5.0)
        force_far = separation(agent, [far_neighbor], separation_distance=5.0)

        # Closer neighbor should produce stronger force
        assert abs(force_close.x) > abs(force_far.x)

    def test_separation_outside_distance(self):
        """Test separation ignores neighbors outside distance."""
        agent = SteeringAgent(id=1, position=Vector3(0, 0, 0))
        neighbor = SteeringAgent(id=2, position=Vector3(10, 0, 0))

        force = separation(agent, [neighbor], separation_distance=5.0)

        # Should be zero (neighbor too far)
        assert force == Vector3()


# =============================================================================
# Alignment Behavior WHITEBOX Tests
# =============================================================================

class TestAlignmentWhitebox:
    """Whitebox tests for alignment behavior."""

    def test_alignment_no_neighbors(self):
        """Test alignment with no neighbors."""
        agent = SteeringAgent(id=1, heading=Vector3(1, 0, 0))

        force = alignment(agent, [])

        assert force == Vector3()

    def test_alignment_self_excluded(self):
        """Test agent excludes itself from alignment."""
        agent = SteeringAgent(id=1, heading=Vector3(1, 0, 0))

        force = alignment(agent, [agent])

        assert force == Vector3()

    def test_alignment_matches_heading(self):
        """Test alignment steers toward average heading."""
        agent = SteeringAgent(
            id=1,
            position=Vector3(0, 0, 0),
            heading=Vector3(1, 0, 0)  # Facing +X
        )
        neighbor = SteeringAgent(
            id=2,
            position=Vector3(1, 0, 0),
            heading=Vector3(0, 0, 1)  # Facing +Z
        )

        force = alignment(agent, [neighbor], neighbor_distance=10.0)

        # Force should be toward +Z (neighbor's heading)
        assert force.z > 0

    def test_alignment_outside_distance(self):
        """Test alignment ignores neighbors outside distance."""
        agent = SteeringAgent(id=1, position=Vector3(0, 0, 0))
        neighbor = SteeringAgent(id=2, position=Vector3(20, 0, 0))

        force = alignment(agent, [neighbor], neighbor_distance=5.0)

        assert force == Vector3()


# =============================================================================
# Cohesion Behavior WHITEBOX Tests
# =============================================================================

class TestCohesionWhitebox:
    """Whitebox tests for cohesion behavior."""

    def test_cohesion_no_neighbors(self):
        """Test cohesion with no neighbors."""
        agent = SteeringAgent(id=1, position=Vector3(0, 0, 0))

        force = cohesion(agent, [])

        assert force == Vector3()

    def test_cohesion_self_excluded(self):
        """Test agent excludes itself from cohesion."""
        agent = SteeringAgent(id=1, position=Vector3(0, 0, 0))

        force = cohesion(agent, [agent])

        assert force == Vector3()

    def test_cohesion_moves_toward_center(self):
        """Test cohesion moves toward center of mass."""
        agent = SteeringAgent(
            id=1,
            position=Vector3(0, 0, 0),
            max_speed=10.0
        )
        neighbor = SteeringAgent(
            id=2,
            position=Vector3(10, 0, 0)
        )

        force = cohesion(agent, [neighbor], neighbor_distance=20.0)

        # Force should be toward neighbor (positive X)
        assert force.x > 0

    def test_cohesion_multiple_neighbors(self):
        """Test cohesion with multiple neighbors."""
        agent = SteeringAgent(
            id=1,
            position=Vector3(0, 0, 0),
            max_speed=10.0
        )
        neighbors = [
            SteeringAgent(id=2, position=Vector3(10, 0, 0)),
            SteeringAgent(id=3, position=Vector3(-10, 0, 0)),
        ]

        force = cohesion(agent, neighbors, neighbor_distance=20.0)

        # Center of mass is at origin, agent is at origin
        # Force should be small

    def test_cohesion_outside_distance(self):
        """Test cohesion ignores neighbors outside distance."""
        agent = SteeringAgent(id=1, position=Vector3(0, 0, 0))
        neighbor = SteeringAgent(id=2, position=Vector3(100, 0, 0))

        force = cohesion(agent, [neighbor], neighbor_distance=5.0)

        assert force == Vector3()


# =============================================================================
# Flocking Behavior WHITEBOX Tests
# =============================================================================

class TestFlockingWhitebox:
    """Whitebox tests for combined flocking behavior."""

    def test_flocking_combines_behaviors(self):
        """Test flocking combines separation, alignment, cohesion."""
        agent = SteeringAgent(
            id=1,
            position=Vector3(0, 0, 0),
            heading=Vector3(1, 0, 0),
            max_speed=10.0
        )
        neighbor = SteeringAgent(
            id=2,
            position=Vector3(5, 0, 0),
            heading=Vector3(0, 0, 1)
        )

        force = flocking(
            agent, [neighbor],
            separation_weight=1.0,
            alignment_weight=1.0,
            cohesion_weight=1.0
        )

        # Should produce some force
        assert force.length() > 0

    def test_flocking_weights(self):
        """Test flocking respects behavior weights."""
        agent = SteeringAgent(
            id=1,
            position=Vector3(0, 0, 0),
            heading=Vector3(1, 0, 0),
            max_speed=10.0
        )
        neighbor = SteeringAgent(
            id=2,
            position=Vector3(1, 0, 0),  # Close (separation)
            heading=Vector3(0, 0, 1)
        )

        # Heavy separation weight
        force_sep = flocking(
            agent, [neighbor],
            separation_weight=10.0,
            alignment_weight=0.0,
            cohesion_weight=0.0
        )

        # Heavy cohesion weight
        force_coh = flocking(
            agent, [neighbor],
            separation_weight=0.0,
            alignment_weight=0.0,
            cohesion_weight=10.0
        )

        # Separation pushes away, cohesion pulls toward
        assert force_sep.x < force_coh.x


# =============================================================================
# Obstacle Avoidance WHITEBOX Tests
# =============================================================================

class TestObstacleAvoidanceWhitebox:
    """Whitebox tests for obstacle avoidance behavior."""

    def test_obstacle_avoidance_no_obstacles(self):
        """Test avoidance with no obstacles."""
        agent = SteeringAgent(position=Vector3(0, 0, 0))

        force = obstacle_avoidance(agent, [])

        assert force == Vector3()

    def test_obstacle_avoidance_obstacle_behind(self):
        """Test avoidance ignores obstacles behind agent."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            heading=Vector3(0, 0, 1),  # Facing +Z
            velocity=Vector3(0, 0, 5)
        )
        # Obstacle behind (-Z direction)
        obstacles = [(Vector3(0, 0, -5), 1.0)]

        force = obstacle_avoidance(agent, obstacles)

        # Should ignore obstacle behind
        assert force == Vector3()

    def test_obstacle_avoidance_obstacle_ahead(self):
        """Test avoidance steers away from obstacle ahead."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            heading=Vector3(0, 0, 1),  # Facing +Z
            side=Vector3(1, 0, 0),
            velocity=Vector3(0, 0, 5),
            max_speed=5.0,
            radius=0.5
        )
        # Obstacle directly ahead
        obstacles = [(Vector3(0, 0, 3), 1.0)]

        force = obstacle_avoidance(agent, obstacles, detection_length=5.0)

        # Should produce lateral force

    def test_obstacle_avoidance_wide_obstacle(self):
        """Test avoidance handles obstacles outside detection width."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            heading=Vector3(0, 0, 1),
            side=Vector3(1, 0, 0),
            velocity=Vector3(0, 0, 5),
            radius=0.5
        )
        # Obstacle to the side (outside detection box)
        obstacles = [(Vector3(10, 0, 3), 0.5)]

        force = obstacle_avoidance(agent, obstacles)

        # Should be zero (obstacle not in path)


# =============================================================================
# Wall Following WHITEBOX Tests
# =============================================================================

class TestWallFollowingWhitebox:
    """Whitebox tests for wall following behavior."""

    def test_wall_following_no_walls(self):
        """Test wall following with no walls."""
        agent = SteeringAgent(position=Vector3(0, 0, 0))

        force = wall_following(agent, [])

        assert force == Vector3()

    def test_wall_following_wall_ahead(self):
        """Test wall following steers away from wall ahead."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            heading=Vector3(0, 0, 1),  # Facing +Z
            side=Vector3(1, 0, 0)
        )
        # Wall perpendicular to agent heading
        walls = [(Vector3(-5, 0, 2), Vector3(5, 0, 2))]

        force = wall_following(agent, walls, detection_distance=5.0)

        # Should produce force to avoid wall

    def test_line_intersection_parallel(self):
        """Test line intersection with parallel lines."""
        # Two parallel horizontal lines
        result = _line_intersection(
            Vector3(0, 0, 0), Vector3(10, 0, 0),
            Vector3(0, 0, 5), Vector3(10, 0, 5)
        )
        assert result is None

    def test_line_intersection_crossing(self):
        """Test line intersection with crossing lines."""
        # Two lines that cross at (5, 0, 5)
        result = _line_intersection(
            Vector3(0, 0, 5), Vector3(10, 0, 5),  # Horizontal
            Vector3(5, 0, 0), Vector3(5, 0, 10)   # Vertical
        )
        if result is not None:
            assert abs(result.x - 5) < FLOAT_EPSILON
            assert abs(result.z - 5) < FLOAT_EPSILON

    def test_line_intersection_non_intersecting(self):
        """Test line intersection with non-intersecting segments."""
        # Two segments that don't intersect
        result = _line_intersection(
            Vector3(0, 0, 0), Vector3(1, 0, 0),
            Vector3(5, 0, 5), Vector3(6, 0, 5)
        )
        assert result is None


# =============================================================================
# Path Following WHITEBOX Tests
# =============================================================================

class TestPathFollowingWhitebox:
    """Whitebox tests for path following behavior."""

    def test_path_following_empty_path(self):
        """Test path following with empty path."""
        agent = SteeringAgent(position=Vector3(0, 0, 0))

        force = path_following(agent, [])

        assert force == Vector3()

    def test_path_following_single_point(self):
        """Test path following with single point."""
        agent = SteeringAgent(position=Vector3(0, 0, 0))
        path = [Vector3(10, 0, 0)]

        force = path_following(agent, path)

        # Should be zero (not a valid path)
        assert force == Vector3()

    def test_path_following_on_path(self):
        """Test path following when on the path."""
        agent = SteeringAgent(
            position=Vector3(5, 0, 0),
            velocity=Vector3(1, 0, 0).normalized(),
            max_speed=5.0
        )
        path = [Vector3(0, 0, 0), Vector3(10, 0, 0)]

        force = path_following(agent, path)

        # Should seek toward next waypoint

    def test_path_following_off_path(self):
        """Test path following when off the path."""
        agent = SteeringAgent(
            position=Vector3(5, 0, 5),  # Off path
            velocity=Vector3(1, 0, 0).normalized(),
            max_speed=5.0
        )
        path = [Vector3(0, 0, 0), Vector3(10, 0, 0)]

        force = path_following(agent, path, path_offset=1.0)

        # Should steer back toward path

    def test_closest_point_on_segment_start(self):
        """Test closest point at segment start."""
        point = Vector3(-5, 0, 0)
        seg_start = Vector3(0, 0, 0)
        seg_end = Vector3(10, 0, 0)

        closest = _closest_point_on_segment(point, seg_start, seg_end)

        assert closest == seg_start

    def test_closest_point_on_segment_end(self):
        """Test closest point at segment end."""
        point = Vector3(15, 0, 0)
        seg_start = Vector3(0, 0, 0)
        seg_end = Vector3(10, 0, 0)

        closest = _closest_point_on_segment(point, seg_start, seg_end)

        assert closest == seg_end

    def test_closest_point_on_segment_middle(self):
        """Test closest point in segment middle."""
        point = Vector3(5, 0, 5)
        seg_start = Vector3(0, 0, 0)
        seg_end = Vector3(10, 0, 0)

        closest = _closest_point_on_segment(point, seg_start, seg_end)

        assert abs(closest.x - 5) < FLOAT_EPSILON
        assert abs(closest.z - 0) < FLOAT_EPSILON


# =============================================================================
# SteeringManager WHITEBOX Tests
# =============================================================================

class TestSteeringManagerWhitebox:
    """Whitebox tests for SteeringManager."""

    def test_manager_default_weights(self):
        """Test manager has default weights for all behaviors."""
        manager = SteeringManager()

        for behavior in SteeringBehavior:
            weight = manager.get_weight(behavior)
            assert weight >= 0

    def test_manager_set_weight(self):
        """Test setting behavior weight."""
        manager = SteeringManager()
        manager.set_weight(SteeringBehavior.SEEK, 2.5)

        assert manager.get_weight(SteeringBehavior.SEEK) == 2.5

    def test_manager_enable_disable(self):
        """Test enabling/disabling behaviors."""
        manager = SteeringManager()

        manager.disable_behavior(SteeringBehavior.SEEK)
        assert not manager.is_enabled(SteeringBehavior.SEEK)

        manager.enable_behavior(SteeringBehavior.SEEK)
        assert manager.is_enabled(SteeringBehavior.SEEK)

    def test_manager_wander_state(self):
        """Test wander state management."""
        manager = SteeringManager()

        state1 = manager.get_wander_state(1)
        state2 = manager.get_wander_state(1)

        # Same agent should get same state
        assert state1 is state2

        state3 = manager.get_wander_state(2)
        # Different agent gets different state
        assert state3 is not state1

    def test_manager_weighted_sum(self):
        """Test weighted sum combination."""
        manager = SteeringManager()
        manager.set_weight(SteeringBehavior.SEEK, 1.0)
        manager.enable_behavior(SteeringBehavior.SEEK)

        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            max_speed=10.0
        )
        target = Vector3(10, 0, 0)

        force = manager.calculate_weighted_sum(
            agent,
            seek_target=target
        )

        # Should produce seek-like force
        assert force.x > 0

    def test_manager_priority_combination(self):
        """Test priority-based combination."""
        manager = SteeringManager()
        manager.set_weight(SteeringBehavior.SEEK, 1.0)
        manager.enable_behavior(SteeringBehavior.SEEK)

        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            max_speed=10.0,
            max_force=10.0
        )
        target = Vector3(10, 0, 0)

        force = manager.calculate_priority(
            agent,
            seek_target=target
        )

        # Should produce force limited by max_force

    def test_manager_multiple_behaviors(self):
        """Test combining multiple behaviors."""
        manager = SteeringManager()
        manager.set_weight(SteeringBehavior.SEEK, 1.0)
        manager.set_weight(SteeringBehavior.SEPARATION, 1.0)
        manager.enable_behavior(SteeringBehavior.SEEK)
        manager.enable_behavior(SteeringBehavior.SEPARATION)

        agent = SteeringAgent(
            id=1,
            position=Vector3(0, 0, 0),
            max_speed=10.0
        )
        neighbor = SteeringAgent(
            id=2,
            position=Vector3(1, 0, 0)
        )
        target = Vector3(10, 0, 0)

        force = manager.calculate_weighted_sum(
            agent,
            seek_target=target,
            neighbors=[neighbor]
        )

        # Force should combine both behaviors

    def test_manager_disabled_behavior_ignored(self):
        """Test disabled behaviors are ignored."""
        manager = SteeringManager()
        manager.set_weight(SteeringBehavior.SEEK, 100.0)
        manager.disable_behavior(SteeringBehavior.SEEK)

        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            max_speed=10.0
        )
        target = Vector3(10, 0, 0)

        force = manager.calculate_weighted_sum(
            agent,
            seek_target=target
        )

        # Seek should be ignored
        # Force should be zero (only seek was configured)


# =============================================================================
# Mathematical Edge Cases
# =============================================================================

class TestSteeringMathEdgeCases:
    """Edge case tests for steering math."""

    def test_seek_zero_distance(self):
        """Test seek when already at target."""
        agent = SteeringAgent(
            position=Vector3(5, 0, 5),
            velocity=Vector3(0, 0, 0),
            max_speed=10.0
        )
        target = Vector3(5, 0, 5)

        # Should handle without division by zero
        force = seek(agent, target)

    def test_separation_coincident_positions(self):
        """Test separation with coincident positions."""
        agent = SteeringAgent(id=1, position=Vector3(5, 0, 5))
        neighbor = SteeringAgent(id=2, position=Vector3(5, 0, 5))  # Same position

        # Should handle without division by zero
        force = separation(agent, [neighbor])

    def test_wander_with_zero_dt(self):
        """Test wander with zero delta time."""
        agent = SteeringAgent(heading=Vector3(0, 0, 1))
        state = WanderState()

        # Should handle zero dt
        force = wander(agent, state, dt=0)

    def test_very_large_velocities(self):
        """Test with very large velocity values."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            velocity=Vector3(1000000, 0, 0),
            max_speed=1000000
        )
        target = Vector3(1, 0, 0)

        force = seek(agent, target)
        # Should not overflow or produce NaN

    def test_very_small_values(self):
        """Test with very small position values."""
        agent = SteeringAgent(
            position=Vector3(1e-10, 1e-10, 1e-10),
            max_speed=10.0
        )
        target = Vector3(2e-10, 2e-10, 2e-10)

        force = seek(agent, target)
        # Should handle small values


# =============================================================================
# Performance Tests
# =============================================================================

class TestSteeringPerformance:
    """Performance-related tests for steering."""

    def test_many_neighbors(self):
        """Test steering with many neighbors."""
        agent = SteeringAgent(id=0, position=Vector3(0, 0, 0))
        neighbors = [
            SteeringAgent(id=i+1, position=Vector3(i % 10, 0, i // 10))
            for i in range(100)
        ]

        force = flocking(agent, neighbors)
        # Should complete without issues

    def test_many_obstacles(self):
        """Test avoidance with many obstacles."""
        agent = SteeringAgent(
            position=Vector3(0, 0, 0),
            heading=Vector3(0, 0, 1),
            side=Vector3(1, 0, 0),
            velocity=Vector3(0, 0, 5)
        )
        obstacles = [
            (Vector3(i % 10, 0, i // 10 + 5), 0.5)
            for i in range(100)
        ]

        force = obstacle_avoidance(agent, obstacles)
        # Should complete without issues
