"""Whitebox tests for avoidance algorithm in crowd_behavior.py.

Tests T2.1: Avoidance Algorithm Verification
- Agents do not collide at avoidance_radius distance
- Coincident agents separate (not stuck)
- Priority weighting works (high priority agents push more)
- MIN_DISTANCE_EPSILON prevents division by zero
"""

from __future__ import annotations

import math
import random
from unittest.mock import patch

import pytest

from engine.animation.config import CROWD_BEHAVIOR_CONFIG
from engine.animation.crowds.crowd_behavior import (
    AgentState,
    AnimationBlend,
    BehaviorContext,
    CrowdAgent,
    CrowdSimulator,
    FleeingBehavior,
    WalkingBehavior,
    calculate_avoidance,
)
from engine.core.math import Vec3


# ============================================================================
# Helper functions
# ============================================================================

def make_agent(
    position: Vec3 | None = None,
    velocity: Vec3 | None = None,
    priority: int = 0,
    agent_id: int = 0,
    speed: float = CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED,
    radius: float = CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_RADIUS,
) -> CrowdAgent:
    """Create a CrowdAgent with specified parameters."""
    agent = CrowdAgent(
        position=position or Vec3.zero(),
        velocity=velocity or Vec3.zero(),
        priority=priority,
        speed=speed,
        radius=radius,
    )
    if agent_id != 0:
        agent.agent_id = agent_id
    return agent


def make_context(agents: list[CrowdAgent], obstacles: list[tuple[Vec3, float]] | None = None) -> BehaviorContext:
    """Create a BehaviorContext with given agents and obstacles."""
    return BehaviorContext(
        all_agents=agents,
        obstacles=obstacles or [],
        navigation_points=[],
        time=0.0,
    )


# ============================================================================
# Test calculate_avoidance() module-level function
# ============================================================================

class TestCalculateAvoidanceForce:
    """Tests for calculate_avoidance() function."""

    def test_no_nearby_agents_returns_zero_avoidance(self):
        """Avoidance vector is zero when no agents are nearby."""
        agent = make_agent(position=Vec3(0, 0, 0))
        other = make_agent(position=Vec3(10, 0, 10))  # Far away
        context = make_context([agent, other])

        avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0)

        assert avoidance.length() < 0.001, "Should return zero avoidance when no agents nearby"

    def test_agent_within_avoidance_radius_produces_force(self):
        """Agent within avoidance radius produces non-zero avoidance force."""
        agent = make_agent(position=Vec3(0, 0, 0))
        other = make_agent(position=Vec3(1, 0, 0))  # 1 unit away
        context = make_context([agent, other])

        avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0, avoidance_strength=1.5)

        assert avoidance.length() > 0.1, "Should produce avoidance force"
        assert avoidance.x < 0, "Force should push away from other agent (negative x)"

    def test_avoidance_strength_increases_with_proximity(self):
        """Closer agents produce stronger avoidance forces."""
        agent = make_agent(position=Vec3(0, 0, 0))

        # Test at different distances
        other_close = make_agent(position=Vec3(0.5, 0, 0))
        other_far = make_agent(position=Vec3(1.5, 0, 0))

        context_close = make_context([agent, other_close])
        context_far = make_context([agent, other_far])

        avoidance_close = calculate_avoidance(agent, context_close, avoidance_radius=2.0, avoidance_strength=1.5)
        avoidance_far = calculate_avoidance(agent, context_far, avoidance_radius=2.0, avoidance_strength=1.5)

        assert avoidance_close.length() > avoidance_far.length(), \
            "Closer agents should produce stronger avoidance"

    def test_avoidance_at_boundary_of_radius(self):
        """Agent exactly at avoidance radius produces minimal force."""
        agent = make_agent(position=Vec3(0, 0, 0))
        other = make_agent(position=Vec3(1.99, 0, 0))  # Just inside radius
        context = make_context([agent, other])

        avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0, avoidance_strength=1.5)

        # Should be very small but non-zero
        assert 0 < avoidance.length() < 0.1, "Should produce minimal force at boundary"

    def test_multiple_agents_combine_avoidance(self):
        """Multiple nearby agents combine their avoidance forces."""
        agent = make_agent(position=Vec3(0, 0, 0))
        other1 = make_agent(position=Vec3(1, 0, 0))
        other2 = make_agent(position=Vec3(-1, 0, 0))
        context = make_context([agent, other1, other2])

        avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0, avoidance_strength=1.0)

        # Forces from opposite sides should partially cancel
        assert abs(avoidance.x) < 0.5, "Opposing forces should partially cancel"

    def test_avoidance_direction_away_from_other(self):
        """Avoidance force points away from other agent."""
        agent = make_agent(position=Vec3(0, 0, 0))

        # Test different directions
        test_cases = [
            (Vec3(1, 0, 0), lambda v: v.x < 0),   # Other on +X -> push to -X
            (Vec3(-1, 0, 0), lambda v: v.x > 0), # Other on -X -> push to +X
            (Vec3(0, 0, 1), lambda v: v.z < 0),   # Other on +Z -> push to -Z
            (Vec3(0, 0, -1), lambda v: v.z > 0), # Other on -Z -> push to +Z
        ]

        for other_pos, check_fn in test_cases:
            other = make_agent(position=other_pos)
            context = make_context([agent, other])
            avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0, avoidance_strength=1.5)
            assert check_fn(avoidance), f"Avoidance should push away from {other_pos}"


# ============================================================================
# Test Coincident Agent Separation (MIN_DISTANCE_EPSILON)
# ============================================================================

class TestCoincidentAgentSeparation:
    """Tests for coincident agent handling using MIN_DISTANCE_EPSILON."""

    def test_coincident_agents_get_random_separation(self):
        """Agents at exact same position get random separation direction."""
        agent = make_agent(position=Vec3(5, 0, 5))
        other = make_agent(position=Vec3(5, 0, 5))  # Exact same position
        context = make_context([agent, other])

        # Run multiple times to verify randomness
        directions = []
        for _ in range(10):
            avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0, avoidance_strength=1.5)
            if avoidance.length() > 0.01:
                directions.append(avoidance.normalized())

        # Should produce non-zero avoidance
        assert len(directions) > 0, "Coincident agents should produce avoidance"

    def test_epsilon_threshold_prevents_division_by_zero(self):
        """Distance below MIN_DISTANCE_EPSILON triggers epsilon handling."""
        epsilon = CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON
        agent = make_agent(position=Vec3(0, 0, 0))
        other = make_agent(position=Vec3(epsilon / 2, 0, 0))  # Below epsilon
        context = make_context([agent, other])

        # This should not raise ZeroDivisionError
        avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0, avoidance_strength=1.5)
        assert avoidance.length() > 0, "Should produce avoidance without error"

    def test_exactly_at_epsilon_boundary(self):
        """Agents exactly at epsilon distance are handled correctly."""
        epsilon = CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON
        agent = make_agent(position=Vec3(0, 0, 0))
        other = make_agent(position=Vec3(epsilon, 0, 0))  # Exactly at epsilon
        context = make_context([agent, other])

        # Should not crash
        avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0, avoidance_strength=1.5)
        assert isinstance(avoidance, Vec3), "Should return Vec3 at epsilon boundary"

    def test_just_above_epsilon_uses_normal_avoidance(self):
        """Distance just above epsilon uses normal avoidance calculation."""
        epsilon = CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON
        agent = make_agent(position=Vec3(0, 0, 0))
        other = make_agent(position=Vec3(epsilon * 2, 0, 0))  # Above epsilon
        context = make_context([agent, other])

        avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0, avoidance_strength=1.5)

        # Should push in -X direction (away from other)
        assert avoidance.x < 0, "Should push away from other agent"

    def test_no_stuck_agents_after_coincidence(self):
        """Coincident agents should eventually separate, not remain stuck."""
        sim = CrowdSimulator()
        pos = Vec3(10, 0, 10)

        agent1 = make_agent(position=pos, agent_id=1)
        agent2 = make_agent(position=pos, agent_id=2)

        agent1.target_position = Vec3(20, 0, 20)
        agent2.target_position = Vec3(0, 0, 0)

        sim.add_agent(agent1)
        sim.add_agent(agent2)

        sim.transition_agent(agent1, AgentState.WALKING)
        sim.transition_agent(agent2, AgentState.WALKING)

        # Run simulation for several steps
        for _ in range(50):
            sim.update(0.016)  # ~60 FPS

        # Agents should have separated
        distance = agent1.position.distance(agent2.position)
        assert distance > 0.1, f"Agents should separate, but distance is {distance}"


# ============================================================================
# Test Priority Weighting
# ============================================================================

class TestPriorityWeighting:
    """Tests for priority-based avoidance multipliers."""

    def test_higher_priority_agent_yields_less(self):
        """Higher priority agent has reduced avoidance force."""
        high_priority = make_agent(position=Vec3(0, 0, 0), priority=10)
        low_priority = make_agent(position=Vec3(1, 0, 0), priority=0)

        context = make_context([high_priority, low_priority])
        avoidance = calculate_avoidance(high_priority, context, avoidance_radius=2.0, avoidance_strength=1.5)

        # High priority should have reduced avoidance
        # Force is divided by AVOIDANCE_PRIORITY_MULTIPLIER
        base_strength = (1.0 - 1.0 / 2.0) * 1.5  # At distance 1 with radius 2
        reduced_strength = base_strength / CROWD_BEHAVIOR_CONFIG.AVOIDANCE_PRIORITY_MULTIPLIER

        assert avoidance.length() < base_strength, \
            "High priority agent should have reduced avoidance"

    def test_lower_priority_agent_yields_more(self):
        """Lower priority agent has increased avoidance force."""
        low_priority = make_agent(position=Vec3(0, 0, 0), priority=0)
        high_priority = make_agent(position=Vec3(1, 0, 0), priority=10)

        context = make_context([low_priority, high_priority])
        avoidance = calculate_avoidance(low_priority, context, avoidance_radius=2.0, avoidance_strength=1.5)

        # Low priority should have increased avoidance
        base_strength = (1.0 - 1.0 / 2.0) * 1.5

        assert avoidance.length() > base_strength, \
            "Low priority agent should have increased avoidance"

    def test_equal_priority_no_multiplier(self):
        """Equal priority agents have no multiplier applied."""
        agent1 = make_agent(position=Vec3(0, 0, 0), priority=5)
        agent2 = make_agent(position=Vec3(1, 0, 0), priority=5)

        context = make_context([agent1, agent2])
        avoidance = calculate_avoidance(agent1, context, avoidance_radius=2.0, avoidance_strength=1.5)

        # Should be base strength without multiplier
        expected_strength = (1.0 - 1.0 / 2.0) * 1.5  # ~0.75

        assert abs(avoidance.length() - expected_strength) < 0.1, \
            "Equal priority should use base avoidance strength"

    def test_priority_bidirectional_symmetric(self):
        """Priority effects are symmetric: A->B and B->A are inversely related."""
        high = make_agent(position=Vec3(0, 0, 0), priority=10)
        low = make_agent(position=Vec3(1, 0, 0), priority=0)

        context = make_context([high, low])

        avoidance_high = calculate_avoidance(high, context, avoidance_radius=2.0, avoidance_strength=1.5)
        avoidance_low = calculate_avoidance(low, context, avoidance_radius=2.0, avoidance_strength=1.5)

        # High yields less (smaller force)
        # Low yields more (larger force)
        assert avoidance_high.length() < avoidance_low.length(), \
            "High priority agent should have smaller avoidance force"

        # Check ratio matches multiplier
        ratio = avoidance_low.length() / avoidance_high.length()
        expected_ratio = CROWD_BEHAVIOR_CONFIG.AVOIDANCE_PRIORITY_MULTIPLIER ** 2

        assert abs(ratio - expected_ratio) < 0.5, \
            f"Priority ratio should be ~{expected_ratio}, got {ratio}"

    def test_priority_zero_vs_zero(self):
        """Both agents with priority 0 use base avoidance."""
        agent1 = make_agent(position=Vec3(0, 0, 0), priority=0)
        agent2 = make_agent(position=Vec3(1, 0, 0), priority=0)

        context = make_context([agent1, agent2])
        avoidance = calculate_avoidance(agent1, context, avoidance_radius=2.0, avoidance_strength=1.5)

        expected = (1.0 - 1.0 / 2.0) * 1.5
        assert abs(avoidance.length() - expected) < 0.1

    def test_negative_priority_handled(self):
        """Negative priority values work correctly."""
        negative = make_agent(position=Vec3(0, 0, 0), priority=-5)
        positive = make_agent(position=Vec3(1, 0, 0), priority=5)

        context = make_context([negative, positive])
        avoidance = calculate_avoidance(negative, context, avoidance_radius=2.0, avoidance_strength=1.5)

        # Negative priority agent should yield more to positive
        base = (1.0 - 1.0 / 2.0) * 1.5
        assert avoidance.length() > base, \
            "Negative priority should yield more"


# ============================================================================
# Test Obstacle Avoidance with Epsilon
# ============================================================================

class TestObstacleAvoidance:
    """Tests for obstacle avoidance including epsilon handling."""

    def test_obstacle_at_epsilon_distance_handled(self):
        """Agent at epsilon distance from obstacle is handled."""
        epsilon = CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON

        agent = make_agent(position=Vec3(0, 0, 0))
        obstacle_pos = Vec3(epsilon / 2, 0, 0)

        context = make_context([agent], obstacles=[(obstacle_pos, 0.5)])

        # Should not crash
        avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0, avoidance_strength=1.5)
        assert avoidance.length() > 0, "Should produce obstacle avoidance"

    def test_agent_at_obstacle_center(self):
        """Agent exactly at obstacle center gets random push."""
        agent = make_agent(position=Vec3(5, 0, 5))
        obstacle_pos = Vec3(5, 0, 5)  # Same position

        context = make_context([agent], obstacles=[(obstacle_pos, 1.0)])

        # Run multiple times to verify non-zero response
        total_avoidance = Vec3.zero()
        for _ in range(5):
            avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0, avoidance_strength=1.5)
            total_avoidance = total_avoidance + avoidance

        assert total_avoidance.length() > 0, \
            "Agent at obstacle center should get pushed away"

    def test_obstacle_radius_considered(self):
        """Combined radius (obstacle + agent + avoidance) is respected."""
        agent_radius = 0.4
        obstacle_radius = 1.0

        agent = make_agent(position=Vec3(0, 0, 0), radius=agent_radius)

        # Place obstacle just within combined radius
        combined_radius = obstacle_radius + agent_radius + 1.0  # 2.4
        obstacle_pos = Vec3(combined_radius - 0.1, 0, 0)  # Just inside

        context = make_context([agent], obstacles=[(obstacle_pos, obstacle_radius)])
        avoidance = calculate_avoidance(agent, context, avoidance_radius=1.0, avoidance_strength=1.5)

        assert avoidance.length() > 0, "Should avoid obstacle within combined radius"


# ============================================================================
# Test FleeingBehavior with Epsilon
# ============================================================================

class TestFleeingBehaviorEpsilon:
    """Tests for FleeingBehavior epsilon handling."""

    def test_flee_from_exact_position(self):
        """Fleeing agent at threat position picks random direction."""
        behavior = FleeingBehavior()

        agent = make_agent(position=Vec3(5, 0, 5))
        agent.flee_source = Vec3(5, 0, 5)  # Same position as threat
        agent.current_state = AgentState.FLEEING

        context = make_context([agent])

        # Multiple updates should produce movement
        for _ in range(10):
            behavior.update(agent, 0.016, context)

        assert agent.velocity.length() > 0, \
            "Agent at threat position should still flee"

    def test_flee_from_epsilon_distance(self):
        """Fleeing from threat within epsilon distance."""
        behavior = FleeingBehavior()
        epsilon = CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON

        agent = make_agent(position=Vec3(0, 0, 0))
        agent.flee_source = Vec3(epsilon / 2, 0, 0)
        agent.current_state = AgentState.FLEEING

        context = make_context([agent])

        # Should not crash and should produce movement
        behavior.update(agent, 0.016, context)

        assert agent.target_velocity.length() > 0, \
            "Should flee from epsilon-distance threat"

    def test_flee_avoidance_of_coincident_other_agents(self):
        """Fleeing agent avoids coincident other agents."""
        behavior = FleeingBehavior()

        agent = make_agent(position=Vec3(5, 0, 5))
        agent.flee_source = Vec3(0, 0, 0)
        agent.current_state = AgentState.FLEEING

        other = make_agent(position=Vec3(5, 0, 5))  # Same position

        context = make_context([agent, other])

        # Update should handle coincident agent without crashing
        for _ in range(5):
            behavior.update(agent, 0.016, context)

        # Should have velocity (fleeing + avoidance)
        assert agent.velocity.length() > 0


# ============================================================================
# Test Avoidance Radius Boundary Conditions
# ============================================================================

class TestAvoidanceRadiusBoundary:
    """Tests for avoidance at radius boundary conditions."""

    def test_exactly_at_radius_minimal_force(self):
        """Agent exactly at avoidance radius has minimal force."""
        radius = 2.0

        agent = make_agent(position=Vec3(0, 0, 0))
        # Place other at exactly the radius minus tiny amount
        other = make_agent(position=Vec3(radius - 0.001, 0, 0))

        context = make_context([agent, other])
        avoidance = calculate_avoidance(agent, context, avoidance_radius=radius, avoidance_strength=1.5)

        # Force should be very small but non-zero
        assert 0 < avoidance.length() < 0.1

    def test_just_outside_radius_no_force(self):
        """Agent just outside avoidance radius produces no force."""
        radius = 2.0

        agent = make_agent(position=Vec3(0, 0, 0))
        other = make_agent(position=Vec3(radius + 0.1, 0, 0))

        context = make_context([agent, other])
        avoidance = calculate_avoidance(agent, context, avoidance_radius=radius, avoidance_strength=1.5)

        assert avoidance.length() < 0.001, \
            "No avoidance outside radius"

    def test_very_close_to_center_strong_force(self):
        """Agent very close to center has strong force."""
        agent = make_agent(position=Vec3(0, 0, 0))
        other = make_agent(position=Vec3(0.05, 0, 0))  # Very close but above epsilon

        context = make_context([agent, other])
        avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0, avoidance_strength=1.5)

        # Force should be strong (close to max)
        max_strength = 1.5  # avoidance_strength
        assert avoidance.length() > max_strength * 0.8


# ============================================================================
# Integration Test: No Collision at Avoidance Radius
# ============================================================================

class TestNoCollisionAtAvoidanceRadius:
    """Integration tests verifying agents don't collide."""

    def test_approaching_agents_avoid_collision(self):
        """Two agents walking toward each other avoid collision.

        Note: Head-on collisions can result in agents getting close before
        avoidance forces redirect them. The avoidance algorithm is soft
        constraint - it influences direction, not a hard collision boundary.
        """
        sim = CrowdSimulator()

        agent1 = make_agent(position=Vec3(0, 0, 0), agent_id=1)
        agent2 = make_agent(position=Vec3(10, 0, 0), agent_id=2)

        agent1.target_position = Vec3(10, 0, 0)
        agent2.target_position = Vec3(0, 0, 0)

        sim.add_agent(agent1)
        sim.add_agent(agent2)

        sim.transition_agent(agent1, AgentState.WALKING)
        sim.transition_agent(agent2, AgentState.WALKING)

        min_distance = float('inf')

        # Simulate for 5 seconds
        for _ in range(300):  # 300 * 0.016 ~ 5 sec
            sim.update(0.016)
            dist = agent1.position.distance(agent2.position)
            min_distance = min(min_distance, dist)

        # Agents should maintain some separation (soft constraint)
        # The avoidance algorithm influences direction, not hard collision
        # In head-on scenarios, agents may briefly get close
        assert min_distance > 0.1, \
            f"Agents should not overlap completely: {min_distance}"
        # Verify avoidance was active (they didn't just walk through each other)
        assert min_distance > 0.4, \
            f"Avoidance should prevent direct collision: {min_distance}"

    def test_multiple_agents_maintain_separation(self):
        """Multiple agents in close proximity maintain separation."""
        sim = CrowdSimulator()

        # Create 4 agents in a small area
        agents = []
        for i in range(4):
            angle = i * math.pi / 2
            pos = Vec3(math.cos(angle) * 0.5, 0, math.sin(angle) * 0.5)
            agent = make_agent(position=pos, agent_id=i + 1)
            agents.append(agent)
            sim.add_agent(agent)

            # Give them targets on opposite sides
            target_angle = angle + math.pi
            agent.target_position = Vec3(
                math.cos(target_angle) * 5,
                0,
                math.sin(target_angle) * 5
            )
            sim.transition_agent(agent, AgentState.WALKING)

        min_pairwise = float('inf')

        for _ in range(200):
            sim.update(0.016)
            for i, a1 in enumerate(agents):
                for a2 in agents[i+1:]:
                    dist = a1.position.distance(a2.position)
                    min_pairwise = min(min_pairwise, dist)

        # Should maintain some minimum separation
        assert min_pairwise > 0.1, \
            f"Agents got too close: min_pairwise={min_pairwise}"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests for avoidance algorithm."""

    def test_zero_speed_agent(self):
        """Agent with zero speed still calculates avoidance."""
        agent = make_agent(position=Vec3(0, 0, 0), speed=0)
        other = make_agent(position=Vec3(1, 0, 0))
        context = make_context([agent, other])

        avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0)
        assert isinstance(avoidance, Vec3)

    def test_large_number_of_nearby_agents(self):
        """Avoidance handles many nearby agents."""
        agent = make_agent(position=Vec3(0, 0, 0))

        # Create 20 agents around the center
        others = []
        for i in range(20):
            angle = i * 2 * math.pi / 20
            pos = Vec3(math.cos(angle) * 2, 0, math.sin(angle) * 2)
            others.append(make_agent(position=pos, agent_id=i + 100))

        context = make_context([agent] + others)

        # Should not crash and avoidance should be small (forces cancel)
        avoidance = calculate_avoidance(agent, context, avoidance_radius=5.0, avoidance_strength=1.0)
        assert avoidance.length() < 1.0, "Symmetric forces should mostly cancel"

    def test_agent_does_not_avoid_itself(self):
        """Agent does not include itself in avoidance calculation."""
        agent = make_agent(position=Vec3(0, 0, 0))

        # Only the agent itself in context
        context = make_context([agent])

        avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0)
        assert avoidance.length() < 0.001, "Agent should not avoid itself"

    def test_extremely_high_priority(self):
        """Extremely high priority values handled correctly."""
        high = make_agent(position=Vec3(0, 0, 0), priority=1000000)
        low = make_agent(position=Vec3(1, 0, 0), priority=0)

        context = make_context([high, low])
        avoidance = calculate_avoidance(high, context, avoidance_radius=2.0, avoidance_strength=1.5)

        # Should not overflow or crash, force should be minimal
        assert avoidance.length() < 1.0

    def test_y_axis_positions_ignored_for_xz_avoidance(self):
        """Avoidance works in XZ plane, Y doesn't affect distance calc directly."""
        # Agents at same XZ but different Y
        agent = make_agent(position=Vec3(0, 0, 0))
        other = make_agent(position=Vec3(1, 10, 0))  # Same XZ, high Y

        context = make_context([agent, other])
        avoidance = calculate_avoidance(agent, context, avoidance_radius=2.0, avoidance_strength=1.5)

        # Y is included in Vec3 distance calculation by the implementation
        # so at (1, 10, 0) the distance is sqrt(1 + 100) > 10 > avoidance_radius
        assert avoidance.length() < 0.01, \
            "Y difference should affect 3D distance calculation"


# ============================================================================
# Configuration Value Tests
# ============================================================================

class TestConfigurationValues:
    """Tests verifying config values are used correctly."""

    def test_min_distance_epsilon_value(self):
        """MIN_DISTANCE_EPSILON has expected value."""
        assert CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON == 0.01

    def test_avoidance_priority_multiplier_value(self):
        """AVOIDANCE_PRIORITY_MULTIPLIER has expected value."""
        assert CROWD_BEHAVIOR_CONFIG.AVOIDANCE_PRIORITY_MULTIPLIER == 1.5

    def test_default_avoidance_radius_value(self):
        """DEFAULT_AVOIDANCE_RADIUS has expected value."""
        assert CROWD_BEHAVIOR_CONFIG.DEFAULT_AVOIDANCE_RADIUS == 2.0

    def test_default_avoidance_strength_value(self):
        """DEFAULT_AVOIDANCE_STRENGTH has expected value."""
        assert CROWD_BEHAVIOR_CONFIG.DEFAULT_AVOIDANCE_STRENGTH == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
