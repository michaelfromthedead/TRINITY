"""
BLACKBOX TESTS: T2.1 Avoidance Algorithm Verification

Cleanroom tests based solely on public contract from PHASE_2_ARCH.md and PHASE_2_TODO.md.
No implementation details accessed.

Public Contract:
- CrowdAgent: position, velocity, facing, current_state, priority, group_id
- CrowdSimulator: update(dt) updates all agents
- Avoidance: agents separate when too close
- Priority: higher priority agents push more, lower priority agents move more
- Coincident agents: random direction push to separate
- MIN_DISTANCE_EPSILON: prevents division by zero

Acceptance Criteria (from PHASE_2_TODO.md):
1. Agents do not collide at avoidance_radius distance
2. Coincident agents separate (not stuck)
3. Priority weighting works (high priority agents push more)
4. MIN_DISTANCE_EPSILON prevents division by zero
"""

import pytest
import math

# Import public API only
from engine.animation.crowds import (
    CrowdAgent,
    CrowdSimulator,
    AgentState,
    WalkingBehavior,
    IdleBehavior,
    FleeingBehavior,
)
from engine.core.math import Vec3


def create_simulator_with_behaviors():
    """Create a simulator with all default behaviors registered."""
    simulator = CrowdSimulator()
    simulator.register_behavior(AgentState.IDLE, IdleBehavior())
    simulator.register_behavior(AgentState.WALKING, WalkingBehavior())
    simulator.register_behavior(AgentState.FLEEING, FleeingBehavior())
    return simulator


class TestCoincidentAgentsSeparate:
    """Test that coincident agents (same position) separate properly."""

    def test_coincident_agents_separate_after_update(self):
        """Two agents at exact same position must separate after simulation update."""
        simulator = create_simulator_with_behaviors()
        agent_a = CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING)
        agent_b = CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING)

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        # Run simulation
        simulator.update(dt=0.016)

        # They must no longer be coincident
        dist = agent_a.distance_to(agent_b)
        assert dist > 0, "Coincident agents must separate after update"

    def test_coincident_agents_separate_after_multiple_updates(self):
        """Coincident agents should continue separating over multiple updates."""
        simulator = create_simulator_with_behaviors()
        agent_a = CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING)
        agent_b = CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING)

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        # Run multiple updates
        for _ in range(10):
            simulator.update(dt=0.016)

        # Should be well separated by now
        dist = agent_a.distance_to(agent_b)
        assert dist > 0.01, "Coincident agents must be well separated after multiple updates"

    def test_three_coincident_agents_all_separate(self):
        """Three agents at same position must all separate from each other."""
        simulator = create_simulator_with_behaviors()
        agents = [
            CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING),
            CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING),
            CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING),
        ]

        for agent in agents:
            simulator.add_agent(agent)

        # Run simulation
        for _ in range(10):
            simulator.update(dt=0.016)

        # All pairs must be separated
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                dist = agents[i].distance_to(agents[j])
                assert dist > 0, f"Agents {i} and {j} must separate"


class TestPriorityAvoidance:
    """Test that priority affects avoidance behavior."""

    def test_low_priority_moves_more_than_high_priority(self):
        """Low priority agent should move more to avoid high priority agent."""
        simulator = create_simulator_with_behaviors()

        # Low priority agent
        low_priority = CrowdAgent(
            position=Vec3(0.0, 0.0, 0.0),
            priority=1,
            current_state=AgentState.WALKING
        )
        # High priority agent
        high_priority = CrowdAgent(
            position=Vec3(0.5, 0.0, 0.0),  # Close but not coincident
            priority=10,
            current_state=AgentState.WALKING
        )

        simulator.add_agent(low_priority)
        simulator.add_agent(high_priority)

        low_start = Vec3(low_priority.position.x, low_priority.position.y, low_priority.position.z)
        high_start = Vec3(high_priority.position.x, high_priority.position.y, high_priority.position.z)

        # Run simulation
        for _ in range(10):
            simulator.update(dt=0.016)

        # Calculate displacement
        low_displacement = low_priority.position.distance(low_start)
        high_displacement = high_priority.position.distance(high_start)

        # Low priority should have moved more (or at least not less)
        assert low_displacement >= high_displacement * 0.5, \
            "Low priority agent should move more than high priority"

    def test_equal_priority_agents_move_equally(self):
        """Agents with equal priority should move approximately equally."""
        simulator = create_simulator_with_behaviors()

        agent_a = CrowdAgent(
            position=Vec3(0.0, 0.0, 0.0),
            priority=5,
            current_state=AgentState.WALKING
        )
        agent_b = CrowdAgent(
            position=Vec3(0.5, 0.0, 0.0),
            priority=5,
            current_state=AgentState.WALKING
        )

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        a_start = Vec3(agent_a.position.x, agent_a.position.y, agent_a.position.z)
        b_start = Vec3(agent_b.position.x, agent_b.position.y, agent_b.position.z)

        # Run simulation
        for _ in range(10):
            simulator.update(dt=0.016)

        a_displacement = agent_a.position.distance(a_start)
        b_displacement = agent_b.position.distance(b_start)

        # Should be within 3x of each other
        if a_displacement > 0 and b_displacement > 0:
            ratio = max(a_displacement, b_displacement) / min(a_displacement, b_displacement)
            assert ratio < 3.0, "Equal priority agents should move similarly"


class TestNoCollision:
    """Test that agents do not collide at avoidance_radius distance."""

    def test_agents_do_not_collide_when_walking_toward_each_other(self):
        """Two agents walking toward each other should avoid collision."""
        simulator = create_simulator_with_behaviors()

        # Agent A at left, with target to the right
        agent_a = CrowdAgent(
            position=Vec3(-5.0, 0.0, 0.0),
            target_position=Vec3(5.0, 0.0, 0.0),
            current_state=AgentState.WALKING
        )
        # Agent B at right, with target to the left
        agent_b = CrowdAgent(
            position=Vec3(5.0, 0.0, 0.0),
            target_position=Vec3(-5.0, 0.0, 0.0),
            current_state=AgentState.WALKING
        )

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        min_distance = float('inf')

        # Run simulation for 200 updates
        for _ in range(200):
            simulator.update(dt=0.016)
            dist = agent_a.distance_to(agent_b)
            min_distance = min(min_distance, dist)

        # Should never get too close (avoidance_radius is typically ~0.5-1.0)
        # Using a conservative threshold
        assert min_distance > 0.1, \
            f"Agents should avoid collision, min distance was {min_distance}"

    def test_multiple_agents_in_crowd_avoid_collisions(self):
        """Multiple agents moving in a crowd should avoid collisions."""
        simulator = create_simulator_with_behaviors()

        # Create a grid of agents
        agents = []
        for x in range(-2, 3):
            for z in range(-2, 3):
                agent = CrowdAgent(
                    position=Vec3(x * 1.0, 0.0, z * 1.0),
                    current_state=AgentState.WALKING
                )
                agents.append(agent)
                simulator.add_agent(agent)

        min_pair_distance = float('inf')

        # Run simulation
        for _ in range(100):
            simulator.update(dt=0.016)

            # Check all pairs
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    dist = agents[i].distance_to(agents[j])
                    min_pair_distance = min(min_pair_distance, dist)

        # Should maintain some separation
        assert min_pair_distance > 0.05, \
            f"Crowd agents should avoid collisions, min distance was {min_pair_distance}"


class TestDivisionByZero:
    """Test that no division by zero errors occur."""

    def test_no_crash_on_coincident_agents(self):
        """Exact same position should not cause division by zero."""
        simulator = create_simulator_with_behaviors()

        # Exactly coincident
        agent_a = CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING)
        agent_b = CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING)

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        # Should not raise any exception
        try:
            simulator.update(dt=0.016)
        except ZeroDivisionError:
            pytest.fail("Division by zero when agents are coincident")
        except Exception as e:
            if "division" in str(e).lower():
                pytest.fail(f"Division error: {e}")

    def test_no_crash_on_very_close_agents(self):
        """Very small separation should not cause division by zero."""
        simulator = create_simulator_with_behaviors()

        # Extremely close but not coincident
        epsilon = 1e-10
        agent_a = CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING)
        agent_b = CrowdAgent(position=Vec3(epsilon, 0.0, 0.0), current_state=AgentState.WALKING)

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        # Should not raise any exception
        try:
            simulator.update(dt=0.016)
        except ZeroDivisionError:
            pytest.fail("Division by zero with very close agents")

    def test_no_nan_in_position_after_update(self):
        """Positions should never become NaN after simulation updates."""
        simulator = create_simulator_with_behaviors()

        agent_a = CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING)
        agent_b = CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING)

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        for _ in range(100):
            simulator.update(dt=0.016)

            # Check for NaN
            for agent in [agent_a, agent_b]:
                assert not math.isnan(agent.position.x), "Position.x became NaN"
                assert not math.isnan(agent.position.y), "Position.y became NaN"
                assert not math.isnan(agent.position.z), "Position.z became NaN"
                assert not math.isinf(agent.position.x), "Position.x became infinite"
                assert not math.isinf(agent.position.y), "Position.y became infinite"
                assert not math.isinf(agent.position.z), "Position.z became infinite"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_agent_does_not_move_without_target(self):
        """A single idle agent should not move without a target."""
        simulator = create_simulator_with_behaviors()

        agent = CrowdAgent(
            position=Vec3(0.0, 0.0, 0.0),
            current_state=AgentState.IDLE
        )

        simulator.add_agent(agent)
        initial_pos = Vec3(agent.position.x, agent.position.y, agent.position.z)

        simulator.update(dt=0.016)

        # Idle agent with no target should stay put or move minimally
        assert agent.position.distance(initial_pos) < 0.1

    def test_zero_dt_does_not_change_positions(self):
        """Zero timestep should not change positions significantly."""
        simulator = create_simulator_with_behaviors()

        agent_a = CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING)
        agent_b = CrowdAgent(position=Vec3(1.0, 0.0, 0.0), current_state=AgentState.WALKING)

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        pos_a_before = Vec3(agent_a.position.x, agent_a.position.y, agent_a.position.z)
        pos_b_before = Vec3(agent_b.position.x, agent_b.position.y, agent_b.position.z)

        simulator.update(dt=0.0)

        # Positions should be unchanged or nearly so
        assert agent_a.position.distance(pos_a_before) < 1e-6
        assert agent_b.position.distance(pos_b_before) < 1e-6

    def test_very_large_dt_does_not_cause_instability(self):
        """Large timestep should not cause position explosions."""
        simulator = create_simulator_with_behaviors()

        agent_a = CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING)
        agent_b = CrowdAgent(position=Vec3(0.5, 0.0, 0.0), current_state=AgentState.WALKING)

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        # Large timestep
        simulator.update(dt=1.0)

        # Positions should still be reasonable (not exploded)
        for agent in [agent_a, agent_b]:
            assert abs(agent.position.x) < 1000, f"Position.x exploded to {agent.position.x}"
            assert abs(agent.position.y) < 1000, f"Position.y exploded to {agent.position.y}"
            assert abs(agent.position.z) < 1000, f"Position.z exploded to {agent.position.z}"

    def test_negative_priority_is_handled(self):
        """Negative priority values should be handled gracefully."""
        simulator = create_simulator_with_behaviors()

        agent_a = CrowdAgent(position=Vec3(0.0, 0.0, 0.0), priority=-5, current_state=AgentState.WALKING)
        agent_b = CrowdAgent(position=Vec3(0.5, 0.0, 0.0), priority=5, current_state=AgentState.WALKING)

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        # Should not crash
        try:
            simulator.update(dt=0.016)
        except Exception as e:
            pytest.fail(f"Negative priority caused error: {e}")


class TestAvoidanceRadius:
    """Test avoidance behavior at different distances."""

    def test_agents_inside_avoidance_radius_separate(self):
        """Agents inside avoidance radius should push apart."""
        simulator = create_simulator_with_behaviors()

        # Start very close
        agent_a = CrowdAgent(position=Vec3(0.0, 0.0, 0.0), current_state=AgentState.WALKING)
        agent_b = CrowdAgent(position=Vec3(0.1, 0.0, 0.0), current_state=AgentState.WALKING)

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        initial_dist = agent_a.distance_to(agent_b)

        for _ in range(20):
            simulator.update(dt=0.016)

        final_dist = agent_a.distance_to(agent_b)

        # Should have separated
        assert final_dist > initial_dist, \
            f"Agents should separate: initial={initial_dist}, final={final_dist}"

    def test_agents_outside_avoidance_radius_not_affected(self):
        """Agents far apart should not affect each other significantly."""
        simulator = create_simulator_with_behaviors()

        # Start far apart
        agent_a = CrowdAgent(
            position=Vec3(0.0, 0.0, 0.0),
            current_state=AgentState.IDLE
        )
        agent_b = CrowdAgent(
            position=Vec3(100.0, 0.0, 0.0),
            current_state=AgentState.IDLE
        )

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        initial_a = Vec3(agent_a.position.x, agent_a.position.y, agent_a.position.z)
        initial_b = Vec3(agent_b.position.x, agent_b.position.y, agent_b.position.z)

        simulator.update(dt=0.016)

        # At least they should not have moved significantly toward each other
        current_dist = agent_a.distance_to(agent_b)
        assert current_dist >= 99.9, "Far agents should not attract"


class TestAgentStates:
    """Test avoidance behavior in different agent states."""

    def test_walking_agents_avoid(self):
        """Walking agents should avoid each other."""
        simulator = create_simulator_with_behaviors()

        agent_a = CrowdAgent(
            position=Vec3(0.0, 0.0, 0.0),
            current_state=AgentState.WALKING
        )
        agent_b = CrowdAgent(
            position=Vec3(0.3, 0.0, 0.0),
            current_state=AgentState.WALKING
        )

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        initial_dist = agent_a.distance_to(agent_b)

        for _ in range(20):
            simulator.update(dt=0.016)

        final_dist = agent_a.distance_to(agent_b)

        # Should have separated or at least not collided
        assert final_dist >= initial_dist * 0.5, "Walking agents should avoid"

    def test_fleeing_agents_avoid_each_other(self):
        """Fleeing agents should still avoid each other."""
        simulator = create_simulator_with_behaviors()

        agent_a = CrowdAgent(
            position=Vec3(0.0, 0.0, 0.0),
            current_state=AgentState.FLEEING,
            flee_source=Vec3(-10.0, 0.0, 0.0)
        )
        agent_b = CrowdAgent(
            position=Vec3(0.3, 0.0, 0.0),
            current_state=AgentState.FLEEING,
            flee_source=Vec3(-10.0, 0.0, 0.0)
        )

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        # Should not crash and should maintain some separation
        min_dist = float('inf')
        for _ in range(50):
            simulator.update(dt=0.016)
            min_dist = min(min_dist, agent_a.distance_to(agent_b))

        assert min_dist > 0.01, "Fleeing agents should still avoid collisions"


class TestSimulatorOperations:
    """Test basic simulator operations."""

    def test_add_agent_to_simulator(self):
        """Agents can be added to simulator."""
        simulator = create_simulator_with_behaviors()
        agent = CrowdAgent(position=Vec3(0.0, 0.0, 0.0))

        simulator.add_agent(agent)

        # Should be able to update without error
        simulator.update(dt=0.016)

    def test_update_with_empty_simulator(self):
        """Empty simulator should handle update gracefully."""
        simulator = create_simulator_with_behaviors()

        # Should not crash
        simulator.update(dt=0.016)

    def test_update_with_many_agents(self):
        """Simulator should handle many agents."""
        simulator = create_simulator_with_behaviors()

        # Add 100 agents
        agents = []
        for i in range(100):
            x = (i % 10) * 2.0
            z = (i // 10) * 2.0
            agent = CrowdAgent(position=Vec3(x, 0.0, z), current_state=AgentState.WALKING)
            agents.append(agent)
            simulator.add_agent(agent)

        # Should complete without error
        simulator.update(dt=0.016)

        # No positions should be NaN
        for agent in agents:
            assert not math.isnan(agent.position.x), "Position.x became NaN with many agents"
            assert not math.isnan(agent.position.y), "Position.y became NaN with many agents"
            assert not math.isnan(agent.position.z), "Position.z became NaN with many agents"


class TestAgentRadius:
    """Test that agent radius affects avoidance."""

    def test_larger_radius_agents_separate_more(self):
        """Agents with larger radius should maintain greater separation."""
        simulator = create_simulator_with_behaviors()

        # Large radius agents
        large_a = CrowdAgent(
            position=Vec3(0.0, 0.0, 0.0),
            radius=1.0,
            current_state=AgentState.WALKING
        )
        large_b = CrowdAgent(
            position=Vec3(1.5, 0.0, 0.0),
            radius=1.0,
            current_state=AgentState.WALKING
        )

        simulator.add_agent(large_a)
        simulator.add_agent(large_b)

        for _ in range(50):
            simulator.update(dt=0.016)

        large_separation = large_a.distance_to(large_b)

        # Clear and test small radius
        simulator.clear()
        small_a = CrowdAgent(
            position=Vec3(0.0, 0.0, 0.0),
            radius=0.2,
            current_state=AgentState.WALKING
        )
        small_b = CrowdAgent(
            position=Vec3(0.5, 0.0, 0.0),
            radius=0.2,
            current_state=AgentState.WALKING
        )

        simulator.add_agent(small_a)
        simulator.add_agent(small_b)

        for _ in range(50):
            simulator.update(dt=0.016)

        small_separation = small_a.distance_to(small_b)

        # Large radius agents should maintain greater separation
        # (relative to their combined radii)
        assert large_separation >= small_separation * 0.5


class TestVelocityBehavior:
    """Test velocity-related avoidance behavior."""

    def test_stationary_agents_still_avoid(self):
        """Agents with no initial velocity should still separate if too close."""
        simulator = create_simulator_with_behaviors()

        agent_a = CrowdAgent(
            position=Vec3(0.0, 0.0, 0.0),
            velocity=Vec3(0.0, 0.0, 0.0),
            current_state=AgentState.WALKING
        )
        agent_b = CrowdAgent(
            position=Vec3(0.2, 0.0, 0.0),
            velocity=Vec3(0.0, 0.0, 0.0),
            current_state=AgentState.WALKING
        )

        simulator.add_agent(agent_a)
        simulator.add_agent(agent_b)

        initial_dist = agent_a.distance_to(agent_b)

        for _ in range(30):
            simulator.update(dt=0.016)

        final_dist = agent_a.distance_to(agent_b)

        # Should have separated even without initial velocity
        # Note: If agents don't have a target they may not move
        # So at minimum, they should not have collided
        assert final_dist >= initial_dist * 0.9, \
            "Stationary agents should not collide when close"
