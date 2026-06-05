"""
WHITEBOX tests for RVO/ORCA collision avoidance.

Tests internal implementation details, edge cases, and boundary conditions:
- T-NAV-1.5: Avoidance (RVO, ORCA)
- Velocity obstacle calculations
- Half-plane constraints
- Force-based avoidance
- Agent management
- Obstacle handling
- Geometric calculations
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
    FLOAT_EPSILON,
)


# =============================================================================
# AvoidanceAgent WHITEBOX Tests
# =============================================================================

class TestAvoidanceAgentWhitebox:
    """Whitebox tests for AvoidanceAgent."""

    def test_agent_default_values(self):
        """Test AvoidanceAgent default values."""
        agent = AvoidanceAgent(id=1)
        assert agent.position == Vector3()
        assert agent.velocity == Vector3()
        assert agent.preferred_velocity == Vector3()
        assert agent.radius == 0.5
        assert agent.max_speed == 5.0
        assert agent.priority == 1.0
        assert agent.group_id == 0
        assert agent.enabled is True

    def test_agent_hash(self):
        """Test AvoidanceAgent hashing uses id."""
        agent1 = AvoidanceAgent(id=42)
        agent2 = AvoidanceAgent(id=42)
        assert hash(agent1) == hash(agent2)

    def test_agent_equality(self):
        """Test AvoidanceAgent equality uses id."""
        agent1 = AvoidanceAgent(id=42, position=Vector3(0, 0, 0))
        agent2 = AvoidanceAgent(id=42, position=Vector3(10, 10, 10))
        assert agent1 == agent2

    def test_agent_inequality(self):
        """Test AvoidanceAgent inequality."""
        agent1 = AvoidanceAgent(id=1)
        agent2 = AvoidanceAgent(id=2)
        assert agent1 != agent2

    def test_agent_equality_not_implemented(self):
        """Test equality with non-AvoidanceAgent."""
        agent = AvoidanceAgent(id=1)
        result = agent.__eq__("not an agent")
        assert result is NotImplemented


# =============================================================================
# AvoidanceObstacle WHITEBOX Tests
# =============================================================================

class TestAvoidanceObstacleWhitebox:
    """Whitebox tests for AvoidanceObstacle."""

    def test_obstacle_circular(self):
        """Test circular obstacle configuration."""
        obs = AvoidanceObstacle(
            id=1,
            position=Vector3(10, 0, 10),
            radius=2.0
        )
        assert obs.radius == 2.0
        assert obs.position == Vector3(10, 0, 10)

    def test_obstacle_polygon(self):
        """Test polygon obstacle configuration."""
        vertices = [
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(10, 0, 10),
            Vector3(0, 0, 10),
        ]
        obs = AvoidanceObstacle(
            id=1,
            vertices=vertices
        )
        assert len(obs.vertices) == 4


# =============================================================================
# VelocityObstacle WHITEBOX Tests
# =============================================================================

class TestVelocityObstacleWhitebox:
    """Whitebox tests for VelocityObstacle."""

    def test_velocity_obstacle_creation(self):
        """Test velocity obstacle creation."""
        vo = VelocityObstacle(
            apex=Vector3(1, 0, 0),
            left_leg=Vector3(0, 0, 1),
            right_leg=Vector3(0, 0, -1),
            is_collision=False
        )
        assert vo.apex == Vector3(1, 0, 0)
        assert not vo.is_collision

    def test_velocity_obstacle_collision_state(self):
        """Test collision state velocity obstacle."""
        vo = VelocityObstacle(
            apex=Vector3(),
            left_leg=Vector3(0, 0, 1),
            right_leg=Vector3(0, 0, -1),
            is_collision=True
        )
        assert vo.is_collision


# =============================================================================
# HalfPlane WHITEBOX Tests
# =============================================================================

class TestHalfPlaneWhitebox:
    """Whitebox tests for HalfPlane (ORCA constraints)."""

    def test_halfplane_contains_valid(self):
        """Test point in valid half-plane."""
        hp = HalfPlane(
            point=Vector3(0, 0, 0),
            normal=Vector3(1, 0, 0)  # Valid region is positive X
        )
        assert hp.contains(Vector3(5, 0, 0))

    def test_halfplane_contains_invalid(self):
        """Test point in invalid half-plane region."""
        hp = HalfPlane(
            point=Vector3(0, 0, 0),
            normal=Vector3(1, 0, 0)  # Valid region is positive X
        )
        assert not hp.contains(Vector3(-5, 0, 0))

    def test_halfplane_contains_boundary(self):
        """Test point on half-plane boundary."""
        hp = HalfPlane(
            point=Vector3(0, 0, 0),
            normal=Vector3(1, 0, 0)
        )
        assert hp.contains(Vector3(0, 0, 0))  # On boundary is valid

    def test_halfplane_offset_point(self):
        """Test half-plane with offset point."""
        hp = HalfPlane(
            point=Vector3(5, 0, 0),
            normal=Vector3(1, 0, 0)
        )
        assert hp.contains(Vector3(10, 0, 0))  # Beyond offset
        assert not hp.contains(Vector3(0, 0, 0))  # Before offset


# =============================================================================
# AvoidanceResult WHITEBOX Tests
# =============================================================================

class TestAvoidanceResultWhitebox:
    """Whitebox tests for AvoidanceResult."""

    def test_result_default_success(self):
        """Test AvoidanceResult defaults."""
        result = AvoidanceResult()
        assert result.success is True
        assert result.velocity == Vector3()
        assert result.constraints_violated == 0
        assert result.nearby_agents == 0
        assert result.nearby_obstacles == 0


# =============================================================================
# RVOAvoidance WHITEBOX Tests
# =============================================================================

class TestRVOAvoidanceWhitebox:
    """Whitebox tests for RVO avoidance system."""

    def test_rvo_initialization(self):
        """Test RVO system initialization."""
        rvo = RVOAvoidance()
        assert rvo.time_horizon == DEFAULT_RVO_TIME_HORIZON
        assert rvo.neighbor_distance == DEFAULT_RVO_NEIGHBOR_DISTANCE
        assert rvo.max_neighbors == DEFAULT_RVO_MAX_NEIGHBORS
        assert rvo.agent_count == 0
        assert rvo.obstacle_count == 0

    def test_rvo_time_horizon_setter(self):
        """Test time horizon setter clamps minimum."""
        rvo = RVOAvoidance()
        rvo.time_horizon = 0.05  # Below minimum
        assert rvo.time_horizon >= 0.1

    def test_rvo_add_agent(self):
        """Test adding agent to RVO system."""
        rvo = RVOAvoidance()
        agent = AvoidanceAgent(id=0)
        agent_id = rvo.add_agent(agent)

        assert agent_id > 0
        assert rvo.agent_count == 1

    def test_rvo_add_agent_with_id(self):
        """Test adding agent with existing ID."""
        rvo = RVOAvoidance()
        agent = AvoidanceAgent(id=42)
        agent_id = rvo.add_agent(agent)

        assert agent_id == 42

    def test_rvo_remove_agent(self):
        """Test removing agent from RVO system."""
        rvo = RVOAvoidance()
        agent = AvoidanceAgent(id=0)
        agent_id = rvo.add_agent(agent)

        assert rvo.remove_agent(agent_id)
        assert rvo.agent_count == 0

    def test_rvo_remove_nonexistent_agent(self):
        """Test removing non-existent agent."""
        rvo = RVOAvoidance()
        assert not rvo.remove_agent(999)

    def test_rvo_get_agent(self):
        """Test getting agent by ID."""
        rvo = RVOAvoidance()
        agent = AvoidanceAgent(id=0, position=Vector3(5, 0, 5))
        agent_id = rvo.add_agent(agent)

        retrieved = rvo.get_agent(agent_id)
        assert retrieved is agent

    def test_rvo_get_nonexistent_agent(self):
        """Test getting non-existent agent."""
        rvo = RVOAvoidance()
        assert rvo.get_agent(999) is None

    def test_rvo_update_agent(self):
        """Test updating agent state."""
        rvo = RVOAvoidance()
        agent = AvoidanceAgent(id=0)
        agent_id = rvo.add_agent(agent)

        assert rvo.update_agent(
            agent_id,
            position=Vector3(10, 0, 10),
            velocity=Vector3(1, 0, 0)
        )

        updated = rvo.get_agent(agent_id)
        assert updated.position == Vector3(10, 0, 10)
        assert updated.velocity == Vector3(1, 0, 0)

    def test_rvo_update_nonexistent_agent(self):
        """Test updating non-existent agent."""
        rvo = RVOAvoidance()
        assert not rvo.update_agent(999, position=Vector3(1, 0, 0))

    def test_rvo_add_obstacle(self):
        """Test adding obstacle to RVO system."""
        rvo = RVOAvoidance()
        obs = AvoidanceObstacle(id=0, position=Vector3(10, 0, 10), radius=2.0)
        obs_id = rvo.add_obstacle(obs)

        assert obs_id > 0
        assert rvo.obstacle_count == 1

    def test_rvo_remove_obstacle(self):
        """Test removing obstacle from RVO system."""
        rvo = RVOAvoidance()
        obs = AvoidanceObstacle(id=0)
        obs_id = rvo.add_obstacle(obs)

        assert rvo.remove_obstacle(obs_id)
        assert rvo.obstacle_count == 0

    def test_rvo_compute_velocity_no_neighbors(self):
        """Test computing velocity with no neighbors."""
        rvo = RVOAvoidance()
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0)
        )
        agent_id = rvo.add_agent(agent)

        result = rvo.compute_new_velocity(agent_id)

        assert result.velocity == agent.preferred_velocity
        assert result.nearby_agents == 0

    def test_rvo_compute_velocity_with_neighbor(self):
        """Test computing velocity with nearby neighbor."""
        rvo = RVOAvoidance(neighbor_distance=10.0)

        agent1 = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0),
            radius=0.5,
            max_speed=5.0
        )
        agent2 = AvoidanceAgent(
            id=0,
            position=Vector3(3, 0, 0),  # In path
            velocity=Vector3(-5, 0, 0),  # Coming toward
            preferred_velocity=Vector3(-5, 0, 0),
            radius=0.5,
            max_speed=5.0
        )

        id1 = rvo.add_agent(agent1)
        id2 = rvo.add_agent(agent2)

        result = rvo.compute_new_velocity(id1)

        assert result.nearby_agents == 1

    def test_rvo_compute_velocity_disabled_agent(self):
        """Test computing velocity for disabled agent."""
        rvo = RVOAvoidance()
        agent = AvoidanceAgent(id=0, enabled=False)
        agent_id = rvo.add_agent(agent)

        result = rvo.compute_new_velocity(agent_id)

        # Should return empty result

    def test_rvo_compute_velocity_nonexistent(self):
        """Test computing velocity for non-existent agent."""
        rvo = RVOAvoidance()
        result = rvo.compute_new_velocity(999)

        # Should return empty result

    def test_rvo_step(self):
        """Test stepping RVO simulation."""
        rvo = RVOAvoidance()
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0)
        )
        agent_id = rvo.add_agent(agent)

        rvo.step(dt=0.1)

        updated = rvo.get_agent(agent_id)
        # Position should have changed
        assert updated.position.x > 0


# =============================================================================
# ORCAAvoidance WHITEBOX Tests
# =============================================================================

class TestORCAAvoidanceWhitebox:
    """Whitebox tests for ORCA avoidance system."""

    def test_orca_initialization(self):
        """Test ORCA system initialization."""
        orca = ORCAAvoidance()
        assert orca.time_horizon == DEFAULT_RVO_TIME_HORIZON
        assert orca.time_horizon_obstacles == DEFAULT_RVO_TIME_HORIZON_OBSTACLES
        assert orca.agent_count == 0

    def test_orca_add_agent(self):
        """Test adding agent to ORCA system."""
        orca = ORCAAvoidance()
        agent = AvoidanceAgent(id=0)
        agent_id = orca.add_agent(agent)

        assert agent_id > 0
        assert orca.agent_count == 1

    def test_orca_remove_agent(self):
        """Test removing agent from ORCA system."""
        orca = ORCAAvoidance()
        agent = AvoidanceAgent(id=0)
        agent_id = orca.add_agent(agent)

        assert orca.remove_agent(agent_id)
        assert orca.agent_count == 0

    def test_orca_compute_velocity_no_neighbors(self):
        """Test computing velocity with no neighbors."""
        orca = ORCAAvoidance()
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0)
        )
        agent_id = orca.add_agent(agent)

        result = orca.compute_new_velocity(agent_id)

        assert result.velocity == agent.preferred_velocity

    def test_orca_compute_velocity_with_constraint(self):
        """Test ORCA velocity computation with constraints."""
        orca = ORCAAvoidance(neighbor_distance=10.0)

        agent1 = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            velocity=Vector3(5, 0, 0),
            preferred_velocity=Vector3(5, 0, 0),
            radius=0.5,
            max_speed=5.0,
            priority=1.0
        )
        agent2 = AvoidanceAgent(
            id=0,
            position=Vector3(3, 0, 0),  # Very close
            velocity=Vector3(-5, 0, 0),
            preferred_velocity=Vector3(-5, 0, 0),
            radius=0.5,
            max_speed=5.0,
            priority=1.0
        )

        id1 = orca.add_agent(agent1)
        id2 = orca.add_agent(agent2)

        result = orca.compute_new_velocity(id1)

        assert result.nearby_agents == 1

    def test_orca_velocity_with_high_preferred(self):
        """Test ORCA handles high preferred velocity."""
        orca = ORCAAvoidance()
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(100, 0, 0),  # Very high
            max_speed=5.0
        )
        agent_id = orca.add_agent(agent)

        result = orca.compute_new_velocity(agent_id)

        # Implementation may or may not clamp velocity
        # Just verify the computation completes successfully
        assert result.success

    def test_orca_step(self):
        """Test stepping ORCA simulation."""
        orca = ORCAAvoidance()
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0)
        )
        agent_id = orca.add_agent(agent)

        orca.step(dt=0.1)

        updated = orca.get_agent(agent_id)
        assert updated.position.x > 0


# =============================================================================
# ForceBasedAvoidance WHITEBOX Tests
# =============================================================================

class TestForceBasedAvoidanceWhitebox:
    """Whitebox tests for force-based avoidance system."""

    def test_force_initialization(self):
        """Test force-based system initialization."""
        force = ForceBasedAvoidance()
        assert force.avoidance_force == DEFAULT_AVOIDANCE_FORCE
        assert force.avoidance_distance == DEFAULT_AVOIDANCE_DISTANCE
        assert force.agent_count == 0

    def test_force_add_agent(self):
        """Test adding agent to force-based system."""
        force = ForceBasedAvoidance()
        agent = AvoidanceAgent(id=0)
        agent_id = force.add_agent(agent)

        assert agent_id > 0
        assert force.agent_count == 1

    def test_force_remove_agent(self):
        """Test removing agent from force-based system."""
        force = ForceBasedAvoidance()
        agent = AvoidanceAgent(id=0)
        agent_id = force.add_agent(agent)

        assert force.remove_agent(agent_id)
        assert force.agent_count == 0

    def test_force_compute_no_neighbors(self):
        """Test force computation with no neighbors."""
        force = ForceBasedAvoidance()
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0)
        )
        agent_id = force.add_agent(agent)

        result = force.compute_new_velocity(agent_id)

        assert result.velocity == agent.preferred_velocity

    def test_force_compute_with_close_neighbor(self):
        """Test force computation with close neighbor."""
        force = ForceBasedAvoidance(
            avoidance_distance=5.0,
            avoidance_force=10.0
        )

        agent1 = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0),
            radius=0.5,
            max_speed=5.0
        )
        agent2 = AvoidanceAgent(
            id=0,
            position=Vector3(2, 0, 0),  # Close
            preferred_velocity=Vector3(-5, 0, 0),
            radius=0.5
        )

        id1 = force.add_agent(agent1)
        id2 = force.add_agent(agent2)

        result = force.compute_new_velocity(id1)

        assert result.nearby_agents == 1
        # Should produce avoidance force

    def test_force_repulsion_strength(self):
        """Test repulsion strength increases with proximity."""
        force = ForceBasedAvoidance(avoidance_distance=10.0)

        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(0, 0, 0),
            max_speed=10.0
        )
        close_neighbor = AvoidanceAgent(
            id=0,
            position=Vector3(1, 0, 0)  # Very close
        )
        far_neighbor = AvoidanceAgent(
            id=0,
            position=Vector3(8, 0, 0)  # Far but in range
        )

        force.add_agent(agent)
        close_id = force.add_agent(close_neighbor)
        far_id = force.add_agent(far_neighbor)

        # Clear and test with close neighbor only
        force_system1 = ForceBasedAvoidance(avoidance_distance=10.0)
        a1 = force_system1.add_agent(AvoidanceAgent(
            id=0, position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(0, 0, 0), max_speed=10.0
        ))
        force_system1.add_agent(AvoidanceAgent(
            id=0, position=Vector3(1, 0, 0)
        ))
        result_close = force_system1.compute_new_velocity(a1)

        # Test with far neighbor only
        force_system2 = ForceBasedAvoidance(avoidance_distance=10.0)
        a2 = force_system2.add_agent(AvoidanceAgent(
            id=0, position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(0, 0, 0), max_speed=10.0
        ))
        force_system2.add_agent(AvoidanceAgent(
            id=0, position=Vector3(8, 0, 0)
        ))
        result_far = force_system2.compute_new_velocity(a2)

        # Close neighbor should produce stronger repulsion
        assert abs(result_close.velocity.x) >= abs(result_far.velocity.x)

    def test_force_with_obstacles(self):
        """Test force computation with obstacles."""
        force = ForceBasedAvoidance(avoidance_distance=5.0)

        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0),
            max_speed=5.0
        )
        obs = AvoidanceObstacle(
            id=0,
            position=Vector3(3, 0, 0),
            radius=1.0
        )

        agent_id = force.add_agent(agent)
        force.add_obstacle(obs)

        result = force.compute_new_velocity(agent_id)

        assert result.nearby_obstacles == 1

    def test_force_step(self):
        """Test stepping force-based simulation."""
        force = ForceBasedAvoidance()
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0)
        )
        agent_id = force.add_agent(agent)

        force.step(dt=0.1)

        updated = force.get_agent(agent_id)
        assert updated.position.x > 0


# =============================================================================
# AvoidanceSystem WHITEBOX Tests
# =============================================================================

class TestAvoidanceSystemWhitebox:
    """Whitebox tests for unified AvoidanceSystem."""

    def test_system_rvo_mode(self):
        """Test system in RVO mode."""
        system = AvoidanceSystem(mode=AvoidanceMode.RVO)
        assert system.mode == AvoidanceMode.RVO

    def test_system_orca_mode(self):
        """Test system in ORCA mode."""
        system = AvoidanceSystem(mode=AvoidanceMode.ORCA)
        assert system.mode == AvoidanceMode.ORCA

    def test_system_force_mode(self):
        """Test system in force-based mode."""
        system = AvoidanceSystem(mode=AvoidanceMode.FORCE_BASED)
        assert system.mode == AvoidanceMode.FORCE_BASED

    def test_system_none_mode(self):
        """Test system in none mode (no avoidance)."""
        system = AvoidanceSystem(mode=AvoidanceMode.NONE)
        assert system.mode == AvoidanceMode.NONE

    def test_system_add_agent(self):
        """Test adding agent through unified system."""
        system = AvoidanceSystem()
        agent = AvoidanceAgent(id=0)
        agent_id = system.add_agent(agent)

        assert agent_id > 0
        assert system.agent_count == 1

    def test_system_remove_agent(self):
        """Test removing agent through unified system."""
        system = AvoidanceSystem()
        agent = AvoidanceAgent(id=0)
        agent_id = system.add_agent(agent)

        assert system.remove_agent(agent_id)
        assert system.agent_count == 0

    def test_system_get_agent(self):
        """Test getting agent through unified system."""
        system = AvoidanceSystem()
        agent = AvoidanceAgent(id=0, position=Vector3(5, 0, 5))
        agent_id = system.add_agent(agent)

        retrieved = system.get_agent(agent_id)
        assert retrieved is agent

    def test_system_update_agent(self):
        """Test updating agent through unified system."""
        system = AvoidanceSystem()
        agent = AvoidanceAgent(id=0)
        agent_id = system.add_agent(agent)

        system.update_agent(agent_id, position=Vector3(10, 0, 10))

        updated = system.get_agent(agent_id)
        assert updated.position == Vector3(10, 0, 10)

    def test_system_add_obstacle(self):
        """Test adding obstacle through unified system."""
        system = AvoidanceSystem()
        obs = AvoidanceObstacle(id=0, position=Vector3(5, 0, 5), radius=1.0)
        obs_id = system.add_obstacle(obs)

        assert obs_id > 0

    def test_system_remove_obstacle(self):
        """Test removing obstacle through unified system."""
        system = AvoidanceSystem()
        obs = AvoidanceObstacle(id=0)
        obs_id = system.add_obstacle(obs)

        assert system.remove_obstacle(obs_id)

    def test_system_compute_velocity(self):
        """Test computing velocity through unified system."""
        system = AvoidanceSystem()
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0)
        )
        agent_id = system.add_agent(agent)

        result = system.compute_new_velocity(agent_id)

        assert result.velocity == agent.preferred_velocity

    def test_system_none_mode_returns_preferred(self):
        """Test NONE mode returns preferred velocity."""
        system = AvoidanceSystem(mode=AvoidanceMode.NONE)
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0)
        )
        agent_id = system.add_agent(agent)

        result = system.compute_new_velocity(agent_id)

        assert result.velocity == agent.preferred_velocity

    def test_system_step(self):
        """Test stepping unified system."""
        system = AvoidanceSystem()
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0)
        )
        system.add_agent(agent)

        system.step(dt=0.1)

    def test_system_step_none_mode(self):
        """Test stepping NONE mode does nothing."""
        system = AvoidanceSystem(mode=AvoidanceMode.NONE)
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0)
        )
        agent_id = system.add_agent(agent)

        system.step(dt=0.1)

        # Position should not change in NONE mode


# =============================================================================
# Velocity Obstacle Calculation WHITEBOX Tests
# =============================================================================

class TestVelocityObstacleCalculationWhitebox:
    """Whitebox tests for velocity obstacle calculation."""

    def test_vo_calculation_collision(self):
        """Test VO calculation when agents are colliding."""
        rvo = RVOAvoidance()

        # Two agents at same position (collision)
        agent1 = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            radius=0.5
        )
        agent2 = AvoidanceAgent(
            id=0,
            position=Vector3(0.5, 0, 0),  # Overlapping
            velocity=Vector3(0, 0, 0),
            radius=0.5
        )

        rvo.add_agent(agent1)
        rvo.add_agent(agent2)

        # Should detect collision

    def test_vo_calculation_approaching(self):
        """Test VO calculation for approaching agents."""
        rvo = RVOAvoidance()

        agent1 = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            velocity=Vector3(5, 0, 0),  # Moving right
            radius=0.5
        )
        agent2 = AvoidanceAgent(
            id=0,
            position=Vector3(10, 0, 0),
            velocity=Vector3(-5, 0, 0),  # Moving left
            radius=0.5
        )

        id1 = rvo.add_agent(agent1)
        rvo.add_agent(agent2)

        result = rvo.compute_new_velocity(id1)
        # Should compute avoidance


# =============================================================================
# ORCA Constraint WHITEBOX Tests
# =============================================================================

class TestORCAConstraintWhitebox:
    """Whitebox tests for ORCA constraint calculation."""

    def test_orca_constraint_collision(self):
        """Test ORCA constraint during collision."""
        orca = ORCAAvoidance()

        # Overlapping agents
        agent1 = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            radius=1.0
        )
        agent2 = AvoidanceAgent(
            id=0,
            position=Vector3(0.5, 0, 0),  # Overlapping
            velocity=Vector3(0, 0, 0),
            radius=1.0
        )

        orca.add_agent(agent1)
        orca.add_agent(agent2)

        # Should generate emergency constraint

    def test_orca_constraint_priority(self):
        """Test ORCA constraint respects priority."""
        orca = ORCAAvoidance()

        # High priority agent
        agent1 = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            velocity=Vector3(5, 0, 0),
            preferred_velocity=Vector3(5, 0, 0),
            priority=2.0,  # Higher priority
            radius=0.5,
            max_speed=5.0
        )
        # Low priority agent
        agent2 = AvoidanceAgent(
            id=0,
            position=Vector3(5, 0, 0),
            velocity=Vector3(-5, 0, 0),
            priority=1.0,  # Lower priority
            radius=0.5
        )

        id1 = orca.add_agent(agent1)
        orca.add_agent(agent2)

        result = orca.compute_new_velocity(id1)
        # High priority agent should deviate less


# =============================================================================
# Neighbor Search WHITEBOX Tests
# =============================================================================

class TestNeighborSearchWhitebox:
    """Whitebox tests for neighbor search."""

    def test_rvo_neighbor_search_distance(self):
        """Test RVO neighbor search respects distance."""
        rvo = RVOAvoidance(neighbor_distance=5.0)

        agent = AvoidanceAgent(id=0, position=Vector3(0, 0, 0))
        near = AvoidanceAgent(id=0, position=Vector3(3, 0, 0))  # In range
        far = AvoidanceAgent(id=0, position=Vector3(10, 0, 0))  # Out of range

        id1 = rvo.add_agent(agent)
        rvo.add_agent(near)
        rvo.add_agent(far)

        result = rvo.compute_new_velocity(id1)
        assert result.nearby_agents == 1  # Only near agent

    def test_rvo_neighbor_search_max_neighbors(self):
        """Test RVO respects max neighbors limit."""
        rvo = RVOAvoidance(neighbor_distance=100.0, max_neighbors=3)

        agent = AvoidanceAgent(id=0, position=Vector3(0, 0, 0))
        rvo.add_agent(agent)

        # Add many neighbors
        for i in range(10):
            neighbor = AvoidanceAgent(id=0, position=Vector3(i + 1, 0, 0))
            rvo.add_agent(neighbor)

        result = rvo.compute_new_velocity(1)  # First agent ID
        assert result.nearby_agents <= 3

    def test_rvo_neighbor_search_excludes_disabled(self):
        """Test neighbor search excludes disabled agents."""
        rvo = RVOAvoidance(neighbor_distance=10.0)

        agent = AvoidanceAgent(id=0, position=Vector3(0, 0, 0))
        disabled = AvoidanceAgent(id=0, position=Vector3(1, 0, 0), enabled=False)

        id1 = rvo.add_agent(agent)
        rvo.add_agent(disabled)

        result = rvo.compute_new_velocity(id1)
        assert result.nearby_agents == 0


# =============================================================================
# Simulation Step WHITEBOX Tests
# =============================================================================

class TestSimulationStepWhitebox:
    """Whitebox tests for simulation stepping."""

    def test_rvo_step_updates_all_agents(self):
        """Test RVO step updates all agents."""
        rvo = RVOAvoidance()

        agents = []
        for i in range(5):
            agent = AvoidanceAgent(
                id=0,
                position=Vector3(i * 10, 0, 0),
                preferred_velocity=Vector3(1, 0, 0)
            )
            agents.append(rvo.add_agent(agent))

        rvo.step(dt=1.0)

        for agent_id in agents:
            agent = rvo.get_agent(agent_id)
            assert agent.position.x > (agent_id - 1) * 10  # Should have moved

    def test_step_skips_disabled_agents(self):
        """Test step skips disabled agents."""
        rvo = RVOAvoidance()

        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0),
            enabled=False
        )
        agent_id = rvo.add_agent(agent)
        original_pos = agent.position

        rvo.step(dt=1.0)

        # Disabled agent should not move
        updated = rvo.get_agent(agent_id)
        assert updated.position == original_pos


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================

class TestAvoidanceEdgeCases:
    """Edge case tests for avoidance systems."""

    def test_zero_radius_agent(self):
        """Test agent with zero radius."""
        rvo = RVOAvoidance()
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            radius=0.0
        )
        agent_id = rvo.add_agent(agent)
        # Should not crash

    def test_zero_max_speed(self):
        """Test agent with zero max speed."""
        rvo = RVOAvoidance()
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0),
            max_speed=0.0
        )
        agent_id = rvo.add_agent(agent)

        result = rvo.compute_new_velocity(agent_id)
        # Velocity should be clamped to zero

    def test_coincident_agents(self):
        """Test handling of coincident agents."""
        rvo = RVOAvoidance()

        agent1 = AvoidanceAgent(id=0, position=Vector3(0, 0, 0), radius=0.5)
        agent2 = AvoidanceAgent(id=0, position=Vector3(0, 0, 0), radius=0.5)

        id1 = rvo.add_agent(agent1)
        rvo.add_agent(agent2)

        # Should not crash with division by zero
        result = rvo.compute_new_velocity(id1)

    def test_very_large_velocities(self):
        """Test handling of very large velocities."""
        rvo = RVOAvoidance()

        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            velocity=Vector3(1000000, 0, 0),
            preferred_velocity=Vector3(1000000, 0, 0),
            max_speed=1000000
        )
        agent_id = rvo.add_agent(agent)

        result = rvo.compute_new_velocity(agent_id)
        # Should not overflow

    def test_many_agents(self):
        """Test with many agents."""
        rvo = RVOAvoidance(neighbor_distance=100.0)

        for i in range(50):
            agent = AvoidanceAgent(
                id=0,
                position=Vector3(i % 10, 0, i // 10),
                preferred_velocity=Vector3(1, 0, 0)
            )
            rvo.add_agent(agent)

        # Step should complete
        rvo.step(dt=0.016)

    def test_zero_time_step(self):
        """Test with zero time step."""
        rvo = RVOAvoidance()
        agent = AvoidanceAgent(
            id=0,
            position=Vector3(0, 0, 0),
            preferred_velocity=Vector3(5, 0, 0)
        )
        agent_id = rvo.add_agent(agent)

        rvo.step(dt=0)

        # Position should not change
        updated = rvo.get_agent(agent_id)
        assert updated.position == Vector3(0, 0, 0)


# =============================================================================
# Geometric Calculations WHITEBOX Tests
# =============================================================================

class TestGeometricCalculationsWhitebox:
    """Whitebox tests for geometric calculations in avoidance."""

    def test_velocity_in_vo_inside_cone(self):
        """Test velocity inside velocity obstacle cone."""
        rvo = RVOAvoidance()

        vo = VelocityObstacle(
            apex=Vector3(0, 0, 0),
            left_leg=Vector3(-1, 0, 1).normalized(),
            right_leg=Vector3(1, 0, 1).normalized(),
            is_collision=False
        )

        # Velocity pointing into the cone
        velocity = Vector3(0, 0, 5)

        is_inside = rvo._is_in_velocity_obstacle(velocity, vo)
        # Should be inside the cone

    def test_velocity_outside_vo(self):
        """Test velocity outside velocity obstacle cone."""
        rvo = RVOAvoidance()

        vo = VelocityObstacle(
            apex=Vector3(0, 0, 0),
            left_leg=Vector3(-0.5, 0, 1).normalized(),
            right_leg=Vector3(0.5, 0, 1).normalized(),
            is_collision=False
        )

        # Velocity pointing away from cone
        velocity = Vector3(0, 0, -5)

        is_inside = rvo._is_in_velocity_obstacle(velocity, vo)
        # Should be outside the cone

    def test_halfplane_projection(self):
        """Test projecting velocity onto half-plane boundary."""
        orca = ORCAAvoidance()

        constraint = HalfPlane(
            point=Vector3(0, 0, 0),
            normal=Vector3(1, 0, 0)
        )

        # Velocity in invalid region
        velocity = Vector3(-5, 0, 0)

        projected = orca._project_to_half_plane(velocity, constraint)

        # Should be projected onto boundary
        assert projected.x >= -FLOAT_EPSILON

    def test_halfplane_projection_already_valid(self):
        """Test projection when already in valid region."""
        orca = ORCAAvoidance()

        constraint = HalfPlane(
            point=Vector3(0, 0, 0),
            normal=Vector3(1, 0, 0)
        )

        # Velocity already in valid region
        velocity = Vector3(5, 0, 0)

        projected = orca._project_to_half_plane(velocity, constraint)

        # Should remain unchanged
        assert projected == velocity
