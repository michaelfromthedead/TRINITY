"""
Tests for test_assertions.py - Custom assertions for game testing.
"""

import math

import pytest

from engine.tooling.testing.test_assertions import (
    AssertionError as GameAssertionError,
    GameAssertions,
    assert_entity_count,
    assert_entity_has_component,
    assert_event_fired,
    assert_frame_time,
    assert_no_memory_leaks,
    assert_quaternion_equal,
    assert_system_executed,
    assert_transform_equal,
    assert_vector_equal,
    assert_vector_near,
)


class TestVectorAssertions:
    """Tests for vector assertion functions."""

    def test_vector_equal_pass(self):
        assert_vector_equal((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))

    def test_vector_equal_fail(self):
        with pytest.raises(GameAssertionError):
            assert_vector_equal((1.0, 2.0, 3.0), (1.0, 2.0, 4.0))

    def test_vector_equal_dimension_mismatch(self):
        with pytest.raises(GameAssertionError):
            assert_vector_equal((1.0, 2.0), (1.0, 2.0, 3.0))

    def test_vector_near_pass(self):
        assert_vector_near((1.0, 2.0, 3.0), (1.0001, 2.0001, 3.0001), tolerance=0.001)

    def test_vector_near_fail(self):
        with pytest.raises(GameAssertionError):
            assert_vector_near((1.0, 2.0, 3.0), (1.1, 2.0, 3.0), tolerance=0.01)

    def test_vector_near_custom_message(self):
        try:
            assert_vector_near((0.0,), (1.0,), tolerance=0.1, message="Custom error")
        except GameAssertionError as e:
            assert "Custom error" in str(e)


class TestQuaternionAssertions:
    """Tests for quaternion assertion functions."""

    def test_quaternion_equal_same(self):
        q = (0.0, 0.0, 0.0, 1.0)  # Identity quaternion
        assert_quaternion_equal(q, q)

    def test_quaternion_equal_negated(self):
        # q and -q represent the same rotation
        q1 = (0.0, 0.707, 0.0, 0.707)
        q2 = (0.0, -0.707, 0.0, -0.707)
        assert_quaternion_equal(q1, q2)

    def test_quaternion_equal_fail(self):
        q1 = (0.0, 0.0, 0.0, 1.0)
        q2 = (0.5, 0.5, 0.5, 0.5)  # Different rotation
        with pytest.raises(GameAssertionError):
            assert_quaternion_equal(q1, q2, tolerance=0.01)

    def test_quaternion_normalized(self):
        # Should normalize before comparing
        q1 = (0.0, 0.0, 0.0, 2.0)  # Not normalized
        q2 = (0.0, 0.0, 0.0, 1.0)  # Normalized identity
        assert_quaternion_equal(q1, q2)


class TestTransformAssertions:
    """Tests for transform assertion functions."""

    def test_transform_equal_position(self):
        t1 = {"position": (1.0, 2.0, 3.0)}
        t2 = {"position": (1.0, 2.0, 3.0)}
        assert_transform_equal(t1, t2)

    def test_transform_equal_all_components(self):
        t1 = {
            "position": (0.0, 0.0, 0.0),
            "rotation": (0.0, 0.0, 0.0, 1.0),
            "scale": (1.0, 1.0, 1.0),
        }
        t2 = {
            "position": (0.0, 0.0, 0.0),
            "rotation": (0.0, 0.0, 0.0, 1.0),
            "scale": (1.0, 1.0, 1.0),
        }
        assert_transform_equal(t1, t2)

    def test_transform_equal_fail(self):
        t1 = {"position": (0.0, 0.0, 0.0)}
        t2 = {"position": (1.0, 0.0, 0.0)}
        with pytest.raises(GameAssertionError):
            assert_transform_equal(t1, t2)


class TestEntityAssertions:
    """Tests for entity assertion functions."""

    def test_entity_has_component_pass(self):
        class MockWorld:
            def has_component(self, entity_id, comp_type):
                return comp_type == "Position"

        world = MockWorld()
        assert_entity_has_component(world, 1, "Position")

    def test_entity_has_component_fail(self):
        class MockWorld:
            def has_component(self, entity_id, comp_type):
                return False

        world = MockWorld()
        with pytest.raises(GameAssertionError):
            assert_entity_has_component(world, 1, "Position")

    def test_entity_has_component_dict_world(self):
        class MockWorld:
            components = {1: {"Position": {"x": 0, "y": 0}}}

        world = MockWorld()
        assert_entity_has_component(world, 1, "Position")


class TestEntityCountAssertions:
    """Tests for entity count assertions."""

    def test_entity_count_pass(self):
        class MockWorld:
            entities = {1: {}, 2: {}, 3: {}}

        world = MockWorld()
        assert_entity_count(world, 3)

    def test_entity_count_fail(self):
        class MockWorld:
            entities = {1: {}}

        world = MockWorld()
        with pytest.raises(GameAssertionError):
            assert_entity_count(world, 5)

    def test_entity_count_with_filter(self):
        class MockWorld:
            components = {
                1: {"Position": {}},
                2: {"Position": {}, "Velocity": {}},
                3: {"Health": {}},
            }

        world = MockWorld()
        assert_entity_count(world, 2, component_filter="Position")


class TestSystemAssertions:
    """Tests for system execution assertions."""

    def test_system_executed_pass(self):
        class MockSystem:
            _execution_count = 5

        system = MockSystem()
        assert_system_executed(system)

    def test_system_executed_fail(self):
        class MockSystem:
            _execution_count = 0

        system = MockSystem()
        with pytest.raises(GameAssertionError):
            assert_system_executed(system)

    def test_system_executed_count(self):
        class MockSystem:
            _execution_count = 3

        system = MockSystem()
        assert_system_executed(system, times=3)

    def test_system_executed_count_mismatch(self):
        class MockSystem:
            _execution_count = 2

        system = MockSystem()
        with pytest.raises(GameAssertionError):
            assert_system_executed(system, times=5)


class TestEventAssertions:
    """Tests for event firing assertions."""

    def test_event_fired_pass(self):
        class MockWorld:
            def get_events(self):
                return [{"type": "DamageEvent", "amount": 10}]

        world = MockWorld()
        assert_event_fired(world, "DamageEvent")

    def test_event_fired_fail(self):
        class MockWorld:
            def get_events(self):
                return []

        world = MockWorld()
        with pytest.raises(GameAssertionError):
            assert_event_fired(world, "DamageEvent")

    def test_event_fired_count(self):
        class MockWorld:
            def get_events(self):
                return [
                    {"type": "HitEvent"},
                    {"type": "HitEvent"},
                    {"type": "HitEvent"},
                ]

        world = MockWorld()
        assert_event_fired(world, "HitEvent", count=3)


class TestPerformanceAssertions:
    """Tests for performance-related assertions."""

    def test_no_memory_leaks_pass(self):
        def no_leak():
            x = 1 + 1
            return x

        assert_no_memory_leaks(no_leak, max_growth_bytes=1024*1024, iterations=10)

    def test_frame_time_pass(self):
        def fast_func():
            pass

        assert_frame_time(fast_func, max_ms=100.0, samples=10)

    def test_frame_time_fail(self):
        import time

        def slow_func():
            time.sleep(0.1)

        with pytest.raises(GameAssertionError):
            assert_frame_time(slow_func, max_ms=1.0, samples=5)


class TestGameAssertionsMixin:
    """Tests for GameAssertions mixin class."""

    def test_assert_position_near(self):
        class MockEntity:
            position = (1.0, 2.0, 3.0)

        class TestWithAssertions(GameAssertions):
            pass

        tester = TestWithAssertions()
        entity = MockEntity()
        tester.assert_position_near(entity, (1.0, 2.0, 3.0))

    def test_assert_is_alive(self):
        class MockEntity:
            health = 100

        class TestWithAssertions(GameAssertions):
            pass

        tester = TestWithAssertions()
        entity = MockEntity()
        tester.assert_is_alive(entity)

    def test_assert_is_dead(self):
        class MockEntity:
            health = 0

        class TestWithAssertions(GameAssertions):
            pass

        tester = TestWithAssertions()
        entity = MockEntity()
        tester.assert_is_dead(entity)

    def test_assert_in_bounds(self):
        class MockEntity:
            position = (5.0, 5.0, 5.0)

        class TestWithAssertions(GameAssertions):
            pass

        tester = TestWithAssertions()
        entity = MockEntity()
        tester.assert_in_bounds(entity, (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))

    def test_assert_collision(self):
        class MockEntity:
            def __init__(self, pos, size):
                self.position = pos
                self.size = size

        class TestWithAssertions(GameAssertions):
            pass

        tester = TestWithAssertions()
        e1 = MockEntity((0, 0, 0), (2, 2, 2))
        e2 = MockEntity((1, 0, 0), (2, 2, 2))

        tester.assert_collision(e1, e2)

    def test_assert_no_collision(self):
        class MockEntity:
            def __init__(self, pos, size):
                self.position = pos
                self.size = size

        class TestWithAssertions(GameAssertions):
            pass

        tester = TestWithAssertions()
        e1 = MockEntity((0, 0, 0), (1, 1, 1))
        e2 = MockEntity((100, 0, 0), (1, 1, 1))

        tester.assert_no_collision(e1, e2)
