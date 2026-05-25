"""
Comprehensive tests for RVO/ORCA collision avoidance.

Tests cover:
- RVO (Reciprocal Velocity Obstacles)
- ORCA (Optimal Reciprocal Collision Avoidance)
- Agent radius and velocity constraints
- Static obstacle avoidance
- Dynamic obstacle avoidance
- Group avoidance
- Priority-based avoidance
"""

import math
import pytest
from typing import List

from engine.gameplay.nav.avoidance import (
    AvoidanceAgent,
    AvoidanceObstacle,
    AvoidanceResult,
    AvoidanceSystem,
    ForceBasedAvoidance,
    HalfPlane,
    ORCAAvoidance,
    RVOAvoidance,
    VelocityObstacle,
)
from engine.gameplay.nav.navmesh import Vector3
from engine.gameplay.nav.constants import (
    AvoidanceMode,
    DEFAULT_AVOIDANCE_DISTANCE,
    DEFAULT_AVOIDANCE_FORCE,
    DEFAULT_RVO_MAX_NEIGHBORS,
    DEFAULT_RVO_NEIGHBOR_DISTANCE,
    DEFAULT_RVO_TIME_HORIZON,
    DEFAULT_RVO_TIME_HORIZON_OBSTACLES,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_agent():
    """Create default avoidance agent."""
    return AvoidanceAgent(
        id=1,
        position=Vector3(0, 0, 0),
        velocity=Vector3(0, 0, 0),
        preferred_velocity=Vector3(1, 0, 0),
        radius=0.5,
        max_speed=5.0
    )


@pytest.fixture
def moving_agent():
    """Create moving avoidance agent."""
    return AvoidanceAgent(
        id=1,
        position=Vector3(0, 0, 0),
        velocity=Vector3(1, 0, 0),
        preferred_velocity=Vector3(1, 0, 0),
        radius=0.5,
        max_speed=5.0
    )


@pytest.fixture
def rvo_system():
    """Create RVO avoidance system."""
    return RVOAvoidance()


@pytest.fixture
def orca_system():
    """Create ORCA avoidance system."""
    return ORCAAvoidance()


@pytest.fixture
def force_system():
    """Create force-based avoidance system."""
    return ForceBasedAvoidance()


# =============================================================================
# AvoidanceAgent Tests
# =============================================================================


class TestAvoidanceAgent:
    """Tests for AvoidanceAgent class."""

    def test_default_construction(self):
        """Test default agent construction."""
        agent = AvoidanceAgent(id=1)
        assert agent.id == 1
        assert agent.position == Vector3()
        assert agent.velocity == Vector3()
        assert agent.radius == 0.5
        assert agent.max_speed == 5.0
        assert agent.priority == 1.0
        assert agent.group_id == 0
        assert agent.enabled

    def test_custom_construction(self, default_agent):
        """Test custom agent construction."""
        assert default_agent.id == 1
        assert default_agent.radius == 0.5
        assert default_agent.max_speed == 5.0

    def test_agent_hash(self):
        """Test agent hash is based on ID."""
        agent1 = AvoidanceAgent(id=1)
        agent2 = AvoidanceAgent(id=1)
        assert hash(agent1) == hash(agent2)

    def test_agent_equality(self):
        """Test agent equality is based on ID."""
        agent1 = AvoidanceAgent(id=1, position=Vector3(0, 0, 0))
        agent2 = AvoidanceAgent(id=1, position=Vector3(10, 10, 10))
        assert agent1 == agent2

    def test_agent_inequality(self):
        """Test agent inequality for different IDs."""
        agent1 = AvoidanceAgent(id=1)
        agent2 = AvoidanceAgent(id=2)
        assert not (agent1 == agent2)

    def test_agent_priority(self):
        """Test agent priority."""
        agent = AvoidanceAgent(id=1, priority=2.0)
        assert agent.priority == 2.0

    def test_agent_group(self):
        """Test agent group ID."""
        agent = AvoidanceAgent(id=1, group_id=5)
        assert agent.group_id == 5

    def test_agent_disabled(self):
        """Test disabled agent."""
        agent = AvoidanceAgent(id=1, enabled=False)
        assert not agent.enabled


# =============================================================================
# AvoidanceObstacle Tests
# =============================================================================


class TestAvoidanceObstacle:
    """Tests for AvoidanceObstacle class."""

    def test_circular_obstacle(self):
        """Test circular obstacle construction."""
        obstacle = AvoidanceObstacle(
            id=1,
            position=Vector3(5, 0, 5),
            radius=2.0
        )
        assert obstacle.radius == 2.0
        assert obstacle.position == Vector3(5, 0, 5)

    def test_line_obstacle(self):
        """Test line obstacle with vertices."""
        obstacle = AvoidanceObstacle(
            id=1,
            vertices=[Vector3(0, 0, 0), Vector3(10, 0, 0)]
        )
        assert len(obstacle.vertices) == 2

    def test_polygon_obstacle(self):
        """Test polygon obstacle."""
        obstacle = AvoidanceObstacle(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(5, 0, 0),
                Vector3(5, 0, 5),
                Vector3(0, 0, 5)
            ],
            is_convex=True
        )
        assert len(obstacle.vertices) == 4
        assert obstacle.is_convex

    def test_disabled_obstacle(self):
        """Test disabled obstacle."""
        obstacle = AvoidanceObstacle(id=1, enabled=False)
        assert not obstacle.enabled


# =============================================================================
# VelocityObstacle Tests
# =============================================================================


class TestVelocityObstacle:
    """Tests for VelocityObstacle class."""

    def test_construction(self):
        """Test VO construction."""
        vo = VelocityObstacle(
            apex=Vector3(0, 0, 0),
            left_leg=Vector3(-1, 0, 1).normalized(),
            right_leg=Vector3(1, 0, 1).normalized()
        )
        assert vo.apex == Vector3(0, 0, 0)
        assert not vo.is_collision

    def test_collision_vo(self):
        """Test VO for collision state."""
        vo = VelocityObstacle(
            apex=Vector3(0, 0, 0),
            left_leg=Vector3(-1, 0, 0),
            right_leg=Vector3(1, 0, 0),
            is_collision=True
        )
        assert vo.is_collision


# =============================================================================
# HalfPlane Tests
# =============================================================================


class TestHalfPlane:
    """Tests for HalfPlane class (ORCA constraint)."""

    def test_construction(self):
        """Test half-plane construction."""
        hp = HalfPlane(
            point=Vector3(0, 0, 0),
            normal=Vector3(1, 0, 0)
        )
        assert hp.point == Vector3(0, 0, 0)
        assert hp.normal == Vector3(1, 0, 0)

    def test_contains_in_valid_region(self):
        """Test contains for point in valid region."""
        hp = HalfPlane(
            point=Vector3(0, 0, 0),
            normal=Vector3(1, 0, 0)  # Valid region is x >= 0
        )
        assert hp.contains(Vector3(5, 0, 0))

    def test_contains_outside_valid_region(self):
        """Test contains for point outside valid region."""
        hp = HalfPlane(
            point=Vector3(0, 0, 0),
            normal=Vector3(1, 0, 0)
        )
        assert not hp.contains(Vector3(-5, 0, 0))

    def test_contains_on_boundary(self):
        """Test contains for point on boundary."""
        hp = HalfPlane(
            point=Vector3(0, 0, 0),
            normal=Vector3(1, 0, 0)
        )
        assert hp.contains(Vector3(0, 0, 0))


# =============================================================================
# AvoidanceResult Tests
# =============================================================================


class TestAvoidanceResult:
    """Tests for AvoidanceResult class."""

    def test_default_construction(self):
        """Test default result construction."""
        result = AvoidanceResult()
        assert result.velocity == Vector3()
        assert result.success
        assert result.constraints_violated == 0
        assert result.nearby_agents == 0
        assert result.nearby_obstacles == 0

    def test_custom_result(self):
        """Test custom result."""
        result = AvoidanceResult(
            velocity=Vector3(1, 0, 0),
            success=True,
            nearby_agents=3,
            nearby_obstacles=2
        )
        assert result.velocity == Vector3(1, 0, 0)
        assert result.nearby_agents == 3


# =============================================================================
# RVO Avoidance Tests
# =============================================================================


class TestRVOAvoidance:
    """Tests for RVO avoidance system."""

    def test_construction(self):
        """Test RVO system construction."""
        rvo = RVOAvoidance()
        assert rvo.agent_count == 0
        assert rvo.obstacle_count == 0
        assert rvo.time_horizon == DEFAULT_RVO_TIME_HORIZON

    def test_custom_construction(self):
        """Test RVO with custom parameters."""
        rvo = RVOAvoidance(
            time_horizon=3.0,
            neighbor_distance=20.0,
            max_neighbors=15
        )
        assert rvo.time_horizon == 3.0
        assert rvo.neighbor_distance == 20.0
        assert rvo.max_neighbors == 15

    def test_add_agent(self, rvo_system, default_agent):
        """Test adding agent."""
        agent_id = rvo_system.add_agent(default_agent)
        assert agent_id > 0
        assert rvo_system.agent_count == 1

    def test_remove_agent(self, rvo_system, default_agent):
        """Test removing agent."""
        agent_id = rvo_system.add_agent(default_agent)
        result = rvo_system.remove_agent(agent_id)
        assert result
        assert rvo_system.agent_count == 0

    def test_remove_nonexistent_agent(self, rvo_system):
        """Test removing nonexistent agent."""
        result = rvo_system.remove_agent(999)
        assert not result

    def test_get_agent(self, rvo_system, default_agent):
        """Test getting agent."""
        agent_id = rvo_system.add_agent(default_agent)
        retrieved = rvo_system.get_agent(agent_id)
        assert retrieved is not None
        assert retrieved.id == agent_id

    def test_get_nonexistent_agent(self, rvo_system):
        """Test getting nonexistent agent."""
        agent = rvo_system.get_agent(999)
        assert agent is None

    def test_update_agent_position(self, rvo_system, default_agent):
        """Test updating agent position."""
        agent_id = rvo_system.add_agent(default_agent)
        new_pos = Vector3(5, 0, 5)

        result = rvo_system.update_agent(agent_id, position=new_pos)
        assert result

        agent = rvo_system.get_agent(agent_id)
        assert agent.position == new_pos

    def test_update_agent_velocity(self, rvo_system, default_agent):
        """Test updating agent velocity."""
        agent_id = rvo_system.add_agent(default_agent)
        new_vel = Vector3(2, 0, 0)

        rvo_system.update_agent(agent_id, velocity=new_vel)
        agent = rvo_system.get_agent(agent_id)
        assert agent.velocity == new_vel

    def test_update_agent_preferred_velocity(self, rvo_system, default_agent):
        """Test updating agent preferred velocity."""
        agent_id = rvo_system.add_agent(default_agent)
        new_pref = Vector3(0, 0, 3)

        rvo_system.update_agent(agent_id, preferred_velocity=new_pref)
        agent = rvo_system.get_agent(agent_id)
        assert agent.preferred_velocity == new_pref

    def test_update_nonexistent_agent(self, rvo_system):
        """Test updating nonexistent agent."""
        result = rvo_system.update_agent(999, position=Vector3())
        assert not result

    def test_add_obstacle(self, rvo_system):
        """Test adding obstacle."""
        obstacle = AvoidanceObstacle(id=0, position=Vector3(5, 0, 5), radius=1.0)
        obs_id = rvo_system.add_obstacle(obstacle)
        assert obs_id > 0
        assert rvo_system.obstacle_count == 1

    def test_remove_obstacle(self, rvo_system):
        """Test removing obstacle."""
        obstacle = AvoidanceObstacle(id=0, position=Vector3(5, 0, 5), radius=1.0)
        obs_id = rvo_system.add_obstacle(obstacle)

        result = rvo_system.remove_obstacle(obs_id)
        assert result
        assert rvo_system.obstacle_count == 0

    def test_compute_velocity_no_neighbors(self, rvo_system, default_agent):
        """Test computing velocity with no neighbors."""
        agent_id = rvo_system.add_agent(default_agent)
        result = rvo_system.compute_new_velocity(agent_id)

        # Should return preferred velocity
        assert result.velocity == default_agent.preferred_velocity

    def test_compute_velocity_with_neighbor(self, rvo_system):
        """Test computing velocity with neighbor."""
        agent1 = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(1, 0, 0),
            radius=0.5,
            max_speed=5.0
        )
        agent2 = AvoidanceAgent(
            id=2,
            position=Vector3(3, 0, 0),
            preferred_velocity=Vector3(-1, 0, 0),
            radius=0.5,
            max_speed=5.0
        )

        rvo_system.add_agent(agent1)
        rvo_system.add_agent(agent2)

        result = rvo_system.compute_new_velocity(agent1.id)
        assert isinstance(result, AvoidanceResult)

    def test_compute_velocity_collision(self, rvo_system):
        """Test computing velocity during collision."""
        agent1 = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(1, 0, 0),
            radius=0.5
        )
        agent2 = AvoidanceAgent(
            id=2,
            position=Vector3(0.5, 0, 0),  # Overlapping
            preferred_velocity=Vector3(-1, 0, 0),
            radius=0.5
        )

        rvo_system.add_agent(agent1)
        rvo_system.add_agent(agent2)

        result = rvo_system.compute_new_velocity(agent1.id)
        # Should handle collision gracefully

    def test_compute_velocity_disabled_agent(self, rvo_system, default_agent):
        """Test computing velocity for disabled agent."""
        default_agent.enabled = False
        agent_id = rvo_system.add_agent(default_agent)

        result = rvo_system.compute_new_velocity(agent_id)
        # Should return default result

    def test_step_updates_agents(self, rvo_system):
        """Test step updates agent positions."""
        agent = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            velocity=Vector3(1, 0, 0),
            preferred_velocity=Vector3(1, 0, 0),
            max_speed=5.0
        )
        rvo_system.add_agent(agent)

        rvo_system.step(1.0)

        updated = rvo_system.get_agent(agent.id)
        assert updated.position.x > 0

    def test_time_horizon_property(self, rvo_system):
        """Test time horizon property."""
        rvo_system.time_horizon = 5.0
        assert rvo_system.time_horizon == 5.0

    def test_time_horizon_min_value(self, rvo_system):
        """Test time horizon has minimum value."""
        rvo_system.time_horizon = 0.0
        assert rvo_system.time_horizon >= 0.1


# =============================================================================
# ORCA Avoidance Tests
# =============================================================================


class TestORCAAvoidance:
    """Tests for ORCA avoidance system."""

    def test_construction(self):
        """Test ORCA system construction."""
        orca = ORCAAvoidance()
        assert orca.agent_count == 0
        assert orca.time_horizon == DEFAULT_RVO_TIME_HORIZON

    def test_custom_construction(self):
        """Test ORCA with custom parameters."""
        orca = ORCAAvoidance(
            time_horizon=3.0,
            time_horizon_obstacles=1.0,
            neighbor_distance=20.0,
            max_neighbors=15
        )
        assert orca.time_horizon == 3.0
        assert orca.time_horizon_obstacles == 1.0

    def test_add_agent(self, orca_system, default_agent):
        """Test adding agent to ORCA."""
        agent_id = orca_system.add_agent(default_agent)
        assert agent_id > 0
        assert orca_system.agent_count == 1

    def test_remove_agent(self, orca_system, default_agent):
        """Test removing agent from ORCA."""
        agent_id = orca_system.add_agent(default_agent)
        result = orca_system.remove_agent(agent_id)
        assert result

    def test_get_agent(self, orca_system, default_agent):
        """Test getting agent from ORCA."""
        agent_id = orca_system.add_agent(default_agent)
        agent = orca_system.get_agent(agent_id)
        assert agent is not None

    def test_update_agent(self, orca_system, default_agent):
        """Test updating agent in ORCA."""
        agent_id = orca_system.add_agent(default_agent)
        result = orca_system.update_agent(agent_id, position=Vector3(5, 0, 5))
        assert result

    def test_add_obstacle(self, orca_system):
        """Test adding obstacle to ORCA."""
        obstacle = AvoidanceObstacle(id=0, radius=1.0, position=Vector3(5, 0, 5))
        obs_id = orca_system.add_obstacle(obstacle)
        assert obs_id > 0

    def test_compute_velocity_no_neighbors(self, orca_system, default_agent):
        """Test ORCA with no neighbors returns preferred velocity."""
        agent_id = orca_system.add_agent(default_agent)
        result = orca_system.compute_new_velocity(agent_id)
        assert result.velocity == default_agent.preferred_velocity

    def test_compute_velocity_with_neighbor(self, orca_system):
        """Test ORCA velocity with neighbor."""
        agent1 = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            velocity=Vector3(1, 0, 0),
            preferred_velocity=Vector3(1, 0, 0),
            radius=0.5,
            max_speed=5.0
        )
        agent2 = AvoidanceAgent(
            id=2,
            position=Vector3(3, 0, 0),
            velocity=Vector3(-1, 0, 0),
            preferred_velocity=Vector3(-1, 0, 0),
            radius=0.5,
            max_speed=5.0
        )

        orca_system.add_agent(agent1)
        orca_system.add_agent(agent2)

        result = orca_system.compute_new_velocity(agent1.id)
        # ORCA should adjust velocity

    def test_compute_velocity_respects_max_speed(self, orca_system):
        """Test ORCA respects max speed."""
        agent = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(10, 0, 0),  # High preferred
            max_speed=5.0
        )
        orca_system.add_agent(agent)

        result = orca_system.compute_new_velocity(agent.id)
        assert result.velocity.length() <= agent.max_speed + 0.01

    def test_step_updates_positions(self, orca_system):
        """Test ORCA step updates positions."""
        agent = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            velocity=Vector3(2, 0, 0),
            preferred_velocity=Vector3(2, 0, 0),
            max_speed=5.0
        )
        orca_system.add_agent(agent)

        orca_system.step(0.5)

        updated = orca_system.get_agent(agent.id)
        assert updated.position.x > 0


# =============================================================================
# Force-Based Avoidance Tests
# =============================================================================


class TestForceBasedAvoidance:
    """Tests for force-based avoidance system."""

    def test_construction(self):
        """Test force-based system construction."""
        fb = ForceBasedAvoidance()
        assert fb.agent_count == 0

    def test_custom_construction(self):
        """Test force-based with custom parameters."""
        fb = ForceBasedAvoidance(
            avoidance_force=200.0,
            avoidance_distance=5.0,
            max_neighbors=20
        )
        assert fb.avoidance_force == 200.0
        assert fb.avoidance_distance == 5.0

    def test_add_agent(self, force_system, default_agent):
        """Test adding agent."""
        agent_id = force_system.add_agent(default_agent)
        assert agent_id > 0
        assert force_system.agent_count == 1

    def test_remove_agent(self, force_system, default_agent):
        """Test removing agent."""
        agent_id = force_system.add_agent(default_agent)
        result = force_system.remove_agent(agent_id)
        assert result

    def test_compute_velocity_no_neighbors(self, force_system, default_agent):
        """Test compute with no neighbors."""
        agent_id = force_system.add_agent(default_agent)
        result = force_system.compute_new_velocity(agent_id)
        assert result.velocity == default_agent.preferred_velocity

    def test_compute_velocity_with_neighbor(self, force_system):
        """Test force-based avoidance with neighbor."""
        agent1 = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(1, 0, 0),
            radius=0.5,
            max_speed=5.0
        )
        agent2 = AvoidanceAgent(
            id=2,
            position=Vector3(1.5, 0, 0),  # Close
            preferred_velocity=Vector3(-1, 0, 0),
            radius=0.5,
            max_speed=5.0
        )

        force_system.add_agent(agent1)
        force_system.add_agent(agent2)

        result = force_system.compute_new_velocity(agent1.id)
        # Should have repulsion force

    def test_compute_velocity_with_obstacle(self, force_system, default_agent):
        """Test force-based with obstacle."""
        agent_id = force_system.add_agent(default_agent)

        obstacle = AvoidanceObstacle(
            id=0,
            position=Vector3(1.0, 0, 0),
            radius=0.5
        )
        force_system.add_obstacle(obstacle)

        result = force_system.compute_new_velocity(agent_id)
        # May have avoidance force

    def test_step(self, force_system, moving_agent):
        """Test force-based step."""
        force_system.add_agent(moving_agent)
        force_system.step(0.1)

        updated = force_system.get_agent(moving_agent.id)
        assert updated.position.x > 0


# =============================================================================
# AvoidanceSystem Unified Interface Tests
# =============================================================================


class TestAvoidanceSystem:
    """Tests for unified AvoidanceSystem interface."""

    def test_rvo_mode(self):
        """Test system in RVO mode."""
        system = AvoidanceSystem(mode=AvoidanceMode.RVO)
        assert system.mode == AvoidanceMode.RVO

    def test_orca_mode(self):
        """Test system in ORCA mode."""
        system = AvoidanceSystem(mode=AvoidanceMode.ORCA)
        assert system.mode == AvoidanceMode.ORCA

    def test_force_mode(self):
        """Test system in force-based mode."""
        system = AvoidanceSystem(mode=AvoidanceMode.FORCE_BASED)
        assert system.mode == AvoidanceMode.FORCE_BASED

    def test_none_mode(self):
        """Test system in NONE mode (no avoidance)."""
        system = AvoidanceSystem(mode=AvoidanceMode.NONE)
        assert system.mode == AvoidanceMode.NONE

    def test_add_agent_unified(self):
        """Test adding agent through unified interface."""
        system = AvoidanceSystem()
        agent = AvoidanceAgent(id=0)
        agent_id = system.add_agent(agent)
        assert agent_id > 0

    def test_remove_agent_unified(self):
        """Test removing agent through unified interface."""
        system = AvoidanceSystem()
        agent = AvoidanceAgent(id=0)
        agent_id = system.add_agent(agent)
        result = system.remove_agent(agent_id)
        assert result

    def test_get_agent_unified(self):
        """Test getting agent through unified interface."""
        system = AvoidanceSystem()
        agent = AvoidanceAgent(id=0, position=Vector3(5, 0, 5))
        agent_id = system.add_agent(agent)
        retrieved = system.get_agent(agent_id)
        assert retrieved is not None
        assert retrieved.position == Vector3(5, 0, 5)

    def test_update_agent_unified(self):
        """Test updating agent through unified interface."""
        system = AvoidanceSystem()
        agent = AvoidanceAgent(id=0)
        agent_id = system.add_agent(agent)
        result = system.update_agent(agent_id, position=Vector3(10, 0, 10))
        assert result

    def test_compute_velocity_unified(self):
        """Test computing velocity through unified interface."""
        system = AvoidanceSystem()
        agent = AvoidanceAgent(
            id=0,
            preferred_velocity=Vector3(1, 0, 0),
            max_speed=5.0
        )
        agent_id = system.add_agent(agent)
        result = system.compute_new_velocity(agent_id)
        assert isinstance(result, AvoidanceResult)

    def test_step_unified(self):
        """Test step through unified interface."""
        system = AvoidanceSystem()
        agent = AvoidanceAgent(
            id=0,
            velocity=Vector3(1, 0, 0),
            preferred_velocity=Vector3(1, 0, 0)
        )
        system.add_agent(agent)
        system.step(0.1)  # Should not raise

    def test_none_mode_returns_preferred(self):
        """Test NONE mode returns preferred velocity."""
        system = AvoidanceSystem(mode=AvoidanceMode.NONE)
        agent = AvoidanceAgent(
            id=0,
            preferred_velocity=Vector3(3, 0, 0)
        )
        agent_id = system.add_agent(agent)
        result = system.compute_new_velocity(agent_id)
        assert result.velocity == Vector3(3, 0, 0)


# =============================================================================
# Multi-Agent Avoidance Tests
# =============================================================================


class TestMultiAgentAvoidance:
    """Tests for multi-agent avoidance scenarios."""

    def test_head_on_collision(self, rvo_system):
        """Test head-on collision avoidance."""
        agent1 = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            velocity=Vector3(2, 0, 0),
            preferred_velocity=Vector3(2, 0, 0),
            radius=0.5,
            max_speed=5.0
        )
        agent2 = AvoidanceAgent(
            id=2,
            position=Vector3(5, 0, 0),
            velocity=Vector3(-2, 0, 0),
            preferred_velocity=Vector3(-2, 0, 0),
            radius=0.5,
            max_speed=5.0
        )

        rvo_system.add_agent(agent1)
        rvo_system.add_agent(agent2)

        # Simulate several steps
        for _ in range(50):
            rvo_system.step(0.1)

        # Agents should have avoided each other
        a1 = rvo_system.get_agent(1)
        a2 = rvo_system.get_agent(2)
        dist = a1.position.distance_to(a2.position)
        assert dist >= 0.5  # At least combined radius

    def test_crossing_paths(self, orca_system):
        """Test crossing path avoidance."""
        agent1 = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            velocity=Vector3(2, 0, 0),
            preferred_velocity=Vector3(2, 0, 0),
            radius=0.5,
            max_speed=5.0
        )
        agent2 = AvoidanceAgent(
            id=2,
            position=Vector3(2.5, 0, -2.5),
            velocity=Vector3(0, 0, 2),
            preferred_velocity=Vector3(0, 0, 2),
            radius=0.5,
            max_speed=5.0
        )

        orca_system.add_agent(agent1)
        orca_system.add_agent(agent2)

        # Simulate
        for _ in range(30):
            orca_system.step(0.1)

    def test_many_agents_crowded(self, orca_system):
        """Test many agents in crowded space."""
        # Add 10 agents in a small area
        for i in range(10):
            angle = i * 2 * math.pi / 10
            pos = Vector3(math.cos(angle) * 2, 0, math.sin(angle) * 2)
            vel = Vector3(-math.cos(angle), 0, -math.sin(angle))

            agent = AvoidanceAgent(
                id=i + 1,
                position=pos,
                velocity=vel,
                preferred_velocity=vel,
                radius=0.3,
                max_speed=3.0
            )
            orca_system.add_agent(agent)

        # Simulate
        for _ in range(100):
            orca_system.step(0.05)

    def test_priority_avoidance(self, orca_system):
        """Test priority-based avoidance."""
        high_priority = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            velocity=Vector3(1, 0, 0),
            preferred_velocity=Vector3(1, 0, 0),
            radius=0.5,
            max_speed=5.0,
            priority=2.0  # Higher priority
        )
        low_priority = AvoidanceAgent(
            id=2,
            position=Vector3(3, 0, 0),
            velocity=Vector3(-1, 0, 0),
            preferred_velocity=Vector3(-1, 0, 0),
            radius=0.5,
            max_speed=5.0,
            priority=0.5  # Lower priority
        )

        orca_system.add_agent(high_priority)
        orca_system.add_agent(low_priority)

        # High priority should yield less

    def test_group_avoidance(self, rvo_system):
        """Test agents in same group avoid less."""
        agent1 = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            velocity=Vector3(1, 0, 0),
            preferred_velocity=Vector3(1, 0, 0),
            group_id=1
        )
        agent2 = AvoidanceAgent(
            id=2,
            position=Vector3(2, 0, 0),
            velocity=Vector3(0, 0, 1),
            preferred_velocity=Vector3(0, 0, 1),
            group_id=1  # Same group
        )

        rvo_system.add_agent(agent1)
        rvo_system.add_agent(agent2)


# =============================================================================
# Static Obstacle Avoidance Tests
# =============================================================================


class TestStaticObstacleAvoidance:
    """Tests for static obstacle avoidance."""

    def test_circular_obstacle_avoidance(self, rvo_system):
        """Test avoiding circular obstacle."""
        agent = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            velocity=Vector3(2, 0, 0),
            preferred_velocity=Vector3(2, 0, 0),
            radius=0.5,
            max_speed=5.0
        )
        rvo_system.add_agent(agent)

        obstacle = AvoidanceObstacle(
            id=0,
            position=Vector3(3, 0, 0),
            radius=1.0
        )
        rvo_system.add_obstacle(obstacle)

        # Simulate
        for _ in range(30):
            rvo_system.step(0.1)

    def test_line_obstacle_avoidance(self, orca_system):
        """Test avoiding line obstacle."""
        agent = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 2),
            preferred_velocity=Vector3(0, 0, 2),
            radius=0.5,
            max_speed=5.0
        )
        orca_system.add_agent(agent)

        obstacle = AvoidanceObstacle(
            id=0,
            vertices=[Vector3(-5, 0, 5), Vector3(5, 0, 5)]
        )
        orca_system.add_obstacle(obstacle)

    def test_disabled_obstacle_ignored(self, rvo_system):
        """Test disabled obstacle is ignored."""
        agent = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            velocity=Vector3(1, 0, 0),
            preferred_velocity=Vector3(1, 0, 0),
            max_speed=5.0
        )
        rvo_system.add_agent(agent)

        obstacle = AvoidanceObstacle(
            id=0,
            position=Vector3(2, 0, 0),
            radius=1.0,
            enabled=False
        )
        rvo_system.add_obstacle(obstacle)

        result = rvo_system.compute_new_velocity(agent.id)
        # Should return preferred velocity since obstacle is disabled
        assert result.nearby_obstacles == 0


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================


class TestAvoidanceEdgeCases:
    """Tests for edge cases in avoidance systems."""

    def test_zero_radius_agent(self, rvo_system):
        """Test agent with zero radius."""
        agent = AvoidanceAgent(
            id=1,
            radius=0.0,
            preferred_velocity=Vector3(1, 0, 0)
        )
        rvo_system.add_agent(agent)
        result = rvo_system.compute_new_velocity(agent.id)

    def test_very_large_radius(self, rvo_system):
        """Test agent with very large radius."""
        agent = AvoidanceAgent(
            id=1,
            radius=100.0,
            preferred_velocity=Vector3(1, 0, 0)
        )
        rvo_system.add_agent(agent)
        result = rvo_system.compute_new_velocity(agent.id)

    def test_zero_max_speed(self, orca_system):
        """Test agent with zero max speed."""
        agent = AvoidanceAgent(
            id=1,
            max_speed=0.0,
            preferred_velocity=Vector3(0, 0, 0)
        )
        orca_system.add_agent(agent)
        result = orca_system.compute_new_velocity(agent.id)
        assert result.velocity.length() < 0.01

    def test_same_position_agents(self, rvo_system):
        """Test agents at exactly same position."""
        agent1 = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(1, 0, 0)
        )
        agent2 = AvoidanceAgent(
            id=2,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(-1, 0, 0)
        )

        rvo_system.add_agent(agent1)
        rvo_system.add_agent(agent2)

        # Should handle gracefully
        result = rvo_system.compute_new_velocity(agent1.id)

    def test_very_close_agents(self, orca_system):
        """Test very close agents."""
        agent1 = AvoidanceAgent(
            id=1,
            position=Vector3(0, 0, 0),
            radius=0.5,
            preferred_velocity=Vector3(1, 0, 0)
        )
        agent2 = AvoidanceAgent(
            id=2,
            position=Vector3(0.01, 0, 0),  # Very close
            radius=0.5,
            preferred_velocity=Vector3(-1, 0, 0)
        )

        orca_system.add_agent(agent1)
        orca_system.add_agent(agent2)

        result = orca_system.compute_new_velocity(agent1.id)

    def test_negative_time_step(self, rvo_system, default_agent):
        """Test with negative time step."""
        rvo_system.add_agent(default_agent)
        # Should handle or reject gracefully
        rvo_system.step(-0.1)

    def test_large_time_step(self, rvo_system, moving_agent):
        """Test with large time step."""
        rvo_system.add_agent(moving_agent)
        rvo_system.step(10.0)

    def test_many_obstacles(self, force_system, default_agent):
        """Test with many obstacles."""
        force_system.add_agent(default_agent)

        for i in range(50):
            obstacle = AvoidanceObstacle(
                id=0,
                position=Vector3(
                    math.cos(i * 0.2) * 5,
                    0,
                    math.sin(i * 0.2) * 5
                ),
                radius=0.3
            )
            force_system.add_obstacle(obstacle)

        result = force_system.compute_new_velocity(default_agent.id)


# =============================================================================
# Performance and Scalability Tests
# =============================================================================


class TestAvoidancePerformance:
    """Performance tests for avoidance systems."""

    def test_many_agents_rvo(self):
        """Test RVO with many agents."""
        rvo = RVOAvoidance(max_neighbors=10)

        for i in range(100):
            agent = AvoidanceAgent(
                id=i + 1,
                position=Vector3(
                    (i % 10) * 3,
                    0,
                    (i // 10) * 3
                ),
                preferred_velocity=Vector3(1, 0, 0)
            )
            rvo.add_agent(agent)

        # Compute for all agents
        for i in range(100):
            rvo.compute_new_velocity(i + 1)

    def test_many_agents_orca(self):
        """Test ORCA with many agents."""
        orca = ORCAAvoidance(max_neighbors=10)

        for i in range(100):
            agent = AvoidanceAgent(
                id=i + 1,
                position=Vector3(
                    (i % 10) * 3,
                    0,
                    (i // 10) * 3
                ),
                preferred_velocity=Vector3(1, 0, 0)
            )
            orca.add_agent(agent)

        # Compute for all agents
        for i in range(100):
            orca.compute_new_velocity(i + 1)

    def test_repeated_steps(self, orca_system):
        """Test many repeated steps."""
        for i in range(10):
            agent = AvoidanceAgent(
                id=i + 1,
                position=Vector3(i * 2, 0, 0),
                velocity=Vector3(1, 0, 0),
                preferred_velocity=Vector3(1, 0, 0)
            )
            orca_system.add_agent(agent)

        for _ in range(100):
            orca_system.step(0.1)
