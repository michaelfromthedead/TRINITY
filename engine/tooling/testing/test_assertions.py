"""
Custom assertions for game testing.

Provides game-specific assertion functions for testing vectors,
transforms, entities, systems, and other game engine concepts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union


class AssertionError(Exception):
    """Custom assertion error with additional context."""

    def __init__(self, message: str, expected: Any = None, actual: Any = None):
        super().__init__(message)
        self.expected = expected
        self.actual = actual


def assert_vector_equal(
    actual: Tuple[float, ...],
    expected: Tuple[float, ...],
    message: str = "",
) -> None:
    """
    Assert that two vectors are exactly equal.

    Args:
        actual: The actual vector
        expected: The expected vector
        message: Optional message on failure
    """
    if len(actual) != len(expected):
        raise AssertionError(
            f"Vector dimension mismatch: {len(actual)} vs {len(expected)}. {message}",
            expected=expected,
            actual=actual,
        )

    for i, (a, e) in enumerate(zip(actual, expected)):
        if a != e:
            raise AssertionError(
                f"Vector component {i} mismatch: {a} != {e}. {message}",
                expected=expected,
                actual=actual,
            )


def assert_vector_near(
    actual: Tuple[float, ...],
    expected: Tuple[float, ...],
    tolerance: float = 1e-6,
    message: str = "",
) -> None:
    """
    Assert that two vectors are approximately equal.

    Args:
        actual: The actual vector
        expected: The expected vector
        tolerance: Maximum allowed difference per component
        message: Optional message on failure
    """
    if len(actual) != len(expected):
        raise AssertionError(
            f"Vector dimension mismatch: {len(actual)} vs {len(expected)}. {message}",
            expected=expected,
            actual=actual,
        )

    for i, (a, e) in enumerate(zip(actual, expected)):
        if abs(a - e) > tolerance:
            raise AssertionError(
                f"Vector component {i} not within tolerance: {a} vs {e} "
                f"(diff: {abs(a - e)}, tolerance: {tolerance}). {message}",
                expected=expected,
                actual=actual,
            )


def assert_quaternion_equal(
    actual: Tuple[float, float, float, float],
    expected: Tuple[float, float, float, float],
    tolerance: float = 1e-6,
    message: str = "",
) -> None:
    """
    Assert that two quaternions represent the same rotation.

    Note: q and -q represent the same rotation, so we check both.

    Args:
        actual: The actual quaternion (x, y, z, w)
        expected: The expected quaternion (x, y, z, w)
        tolerance: Maximum allowed difference
        message: Optional message on failure
    """
    # Normalize both quaternions
    def normalize(q):
        length = math.sqrt(sum(c * c for c in q))
        if length < 1e-10:
            return q
        return tuple(c / length for c in q)

    actual = normalize(actual)
    expected = normalize(expected)

    # Check if they're equal or negated
    diff_pos = sum(abs(a - e) for a, e in zip(actual, expected))
    diff_neg = sum(abs(a + e) for a, e in zip(actual, expected))

    if min(diff_pos, diff_neg) > tolerance * 4:  # 4 components
        raise AssertionError(
            f"Quaternion mismatch: {actual} vs {expected}. {message}",
            expected=expected,
            actual=actual,
        )


def assert_transform_equal(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
    position_tolerance: float = 1e-6,
    rotation_tolerance: float = 1e-6,
    scale_tolerance: float = 1e-6,
    message: str = "",
) -> None:
    """
    Assert that two transforms are approximately equal.

    Args:
        actual: Transform with position, rotation, scale
        expected: Expected transform
        position_tolerance: Tolerance for position
        rotation_tolerance: Tolerance for rotation
        scale_tolerance: Tolerance for scale
        message: Optional message on failure
    """
    if "position" in expected:
        actual_pos = actual.get("position", (0, 0, 0))
        expected_pos = expected["position"]
        assert_vector_near(
            actual_pos,
            expected_pos,
            position_tolerance,
            f"Position mismatch. {message}",
        )

    if "rotation" in expected:
        actual_rot = actual.get("rotation", (0, 0, 0, 1))
        expected_rot = expected["rotation"]
        assert_quaternion_equal(
            actual_rot,
            expected_rot,
            rotation_tolerance,
            f"Rotation mismatch. {message}",
        )

    if "scale" in expected:
        actual_scale = actual.get("scale", (1, 1, 1))
        expected_scale = expected["scale"]
        assert_vector_near(
            actual_scale,
            expected_scale,
            scale_tolerance,
            f"Scale mismatch. {message}",
        )


def assert_entity_has_component(
    world: Any,
    entity_id: int,
    component_type: Union[str, Type],
    message: str = "",
) -> None:
    """
    Assert that an entity has a specific component.

    Args:
        world: The game world
        entity_id: Entity ID to check
        component_type: Component type to look for
        message: Optional message on failure
    """
    type_name = component_type if isinstance(component_type, str) else component_type.__name__

    # Support different world interfaces
    has_component = False

    if hasattr(world, "has_component"):
        has_component = world.has_component(entity_id, component_type)
    elif hasattr(world, "get_component"):
        has_component = world.get_component(entity_id, component_type) is not None
    elif hasattr(world, "components"):
        has_component = type_name in world.components.get(entity_id, {})

    if not has_component:
        raise AssertionError(
            f"Entity {entity_id} does not have component '{type_name}'. {message}",
            expected=type_name,
            actual=None,
        )


def assert_entity_count(
    world: Any,
    expected_count: int,
    component_filter: Optional[Union[str, Type]] = None,
    message: str = "",
) -> None:
    """
    Assert the number of entities in the world.

    Args:
        world: The game world
        expected_count: Expected number of entities
        component_filter: Optional component to filter by
        message: Optional message on failure
    """
    # Get entity count based on world interface
    if component_filter:
        type_name = (
            component_filter
            if isinstance(component_filter, str)
            else component_filter.__name__
        )

        if hasattr(world, "query"):
            actual_count = len(list(world.query(component_filter)))
        elif hasattr(world, "components"):
            actual_count = sum(
                1 for eid, comps in world.components.items()
                if type_name in comps
            )
        else:
            actual_count = 0
    else:
        if hasattr(world, "entity_count"):
            actual_count = world.entity_count()
        elif hasattr(world, "entities"):
            actual_count = len(world.entities)
        else:
            actual_count = 0

    if actual_count != expected_count:
        filter_str = f" with '{component_filter}'" if component_filter else ""
        raise AssertionError(
            f"Entity count mismatch{filter_str}: expected {expected_count}, "
            f"got {actual_count}. {message}",
            expected=expected_count,
            actual=actual_count,
        )


def assert_system_executed(
    system: Any,
    times: Optional[int] = None,
    message: str = "",
) -> None:
    """
    Assert that a system was executed.

    Args:
        system: The system to check
        times: Expected execution count (None = at least once)
        message: Optional message on failure
    """
    exec_count = getattr(system, "_execution_count", 0)

    if times is None:
        if exec_count == 0:
            raise AssertionError(
                f"System '{system.__class__.__name__}' was not executed. {message}",
                expected="at least 1",
                actual=0,
            )
    elif exec_count != times:
        raise AssertionError(
            f"System '{system.__class__.__name__}' execution count mismatch: "
            f"expected {times}, got {exec_count}. {message}",
            expected=times,
            actual=exec_count,
        )


def assert_event_fired(
    world: Any,
    event_type: Union[str, Type],
    count: Optional[int] = None,
    message: str = "",
) -> None:
    """
    Assert that an event was fired.

    Args:
        world: The game world
        event_type: Type of event to check
        count: Expected number of times (None = at least once)
        message: Optional message on failure
    """
    type_name = event_type if isinstance(event_type, str) else event_type.__name__

    # Get events from world
    events = []
    if hasattr(world, "get_events"):
        events = world.get_events()
    elif hasattr(world, "_events"):
        events = world._events

    # Filter by type
    matching = [
        e for e in events
        if (isinstance(e, dict) and e.get("type") == type_name) or
           (hasattr(e, "__class__") and e.__class__.__name__ == type_name) or
           (isinstance(e, str) and e == type_name)
    ]

    if count is None:
        if len(matching) == 0:
            raise AssertionError(
                f"Event '{type_name}' was not fired. {message}",
                expected="at least 1",
                actual=0,
            )
    elif len(matching) != count:
        raise AssertionError(
            f"Event '{type_name}' fire count mismatch: "
            f"expected {count}, got {len(matching)}. {message}",
            expected=count,
            actual=len(matching),
        )


def assert_no_memory_leaks(
    func: Callable,
    max_growth_bytes: int = 1024 * 1024,  # 1MB
    iterations: int = 10,
    message: str = "",
) -> None:
    """
    Assert that a function doesn't leak memory.

    Args:
        func: Function to test
        max_growth_bytes: Maximum allowed memory growth
        iterations: Number of iterations to run
        message: Optional message on failure
    """
    import gc
    import tracemalloc

    gc.collect()
    tracemalloc.start()

    initial = tracemalloc.get_traced_memory()[0]

    for _ in range(iterations):
        func()
        gc.collect()

    final = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    growth = final - initial

    if growth > max_growth_bytes:
        raise AssertionError(
            f"Memory leak detected: grew by {growth} bytes over {iterations} iterations "
            f"(max allowed: {max_growth_bytes}). {message}",
            expected=f"<= {max_growth_bytes}",
            actual=growth,
        )


def assert_frame_time(
    func: Callable,
    max_ms: float = 16.67,  # 60 FPS
    warmup: int = 5,
    samples: int = 100,
    percentile: float = 95.0,
    message: str = "",
) -> None:
    """
    Assert that a function completes within a frame time budget.

    Args:
        func: Function to test
        max_ms: Maximum allowed time in milliseconds
        warmup: Number of warmup iterations
        samples: Number of samples to collect
        percentile: Percentile to check (default 95th)
        message: Optional message on failure
    """
    import time

    # Warmup
    for _ in range(warmup):
        func()

    # Collect samples
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

    times.sort()
    idx = int(len(times) * percentile / 100)
    p_value = times[min(idx, len(times) - 1)]

    if p_value > max_ms:
        raise AssertionError(
            f"Frame time exceeded: {percentile}th percentile was {p_value:.2f}ms "
            f"(max: {max_ms}ms). {message}",
            expected=f"<= {max_ms}ms",
            actual=f"{p_value:.2f}ms",
        )


class GameAssertions:
    """
    Collection of game-specific assertions as a mixin.

    Example:
        class MyTest(TestCase, GameAssertions):
            def test_player_position(self):
                self.assert_position_near(player, (0, 0, 0))
    """

    def assert_position_near(
        self,
        entity: Any,
        expected: Tuple[float, ...],
        tolerance: float = 0.01,
    ) -> None:
        """Assert entity position is near expected."""
        position = self._get_position(entity)
        assert_vector_near(position, expected, tolerance)

    def assert_velocity_near(
        self,
        entity: Any,
        expected: Tuple[float, ...],
        tolerance: float = 0.01,
    ) -> None:
        """Assert entity velocity is near expected."""
        velocity = self._get_velocity(entity)
        assert_vector_near(velocity, expected, tolerance)

    def assert_rotation_near(
        self,
        entity: Any,
        expected: Tuple[float, float, float, float],
        tolerance: float = 0.001,
    ) -> None:
        """Assert entity rotation is near expected."""
        rotation = self._get_rotation(entity)
        assert_quaternion_equal(rotation, expected, tolerance)

    def assert_health_equal(
        self,
        entity: Any,
        expected: float,
        message: str = "",
    ) -> None:
        """Assert entity health equals expected."""
        health = self._get_health(entity)
        if health != expected:
            raise AssertionError(
                f"Health mismatch: expected {expected}, got {health}. {message}",
                expected=expected,
                actual=health,
            )

    def assert_is_alive(self, entity: Any) -> None:
        """Assert entity is alive."""
        if not self._is_alive(entity):
            raise AssertionError(f"Entity is not alive")

    def assert_is_dead(self, entity: Any) -> None:
        """Assert entity is dead."""
        if self._is_alive(entity):
            raise AssertionError(f"Entity is not dead")

    def assert_collision(
        self,
        entity_a: Any,
        entity_b: Any,
        message: str = "",
    ) -> None:
        """Assert two entities are colliding."""
        if not self._check_collision(entity_a, entity_b):
            raise AssertionError(f"Entities are not colliding. {message}")

    def assert_no_collision(
        self,
        entity_a: Any,
        entity_b: Any,
        message: str = "",
    ) -> None:
        """Assert two entities are not colliding."""
        if self._check_collision(entity_a, entity_b):
            raise AssertionError(f"Entities are colliding. {message}")

    def assert_in_bounds(
        self,
        entity: Any,
        min_bounds: Tuple[float, ...],
        max_bounds: Tuple[float, ...],
        message: str = "",
    ) -> None:
        """Assert entity is within bounds."""
        position = self._get_position(entity)
        for i, (pos, min_b, max_b) in enumerate(zip(position, min_bounds, max_bounds)):
            if not (min_b <= pos <= max_b):
                raise AssertionError(
                    f"Entity out of bounds at axis {i}: {pos} not in [{min_b}, {max_b}]. {message}"
                )

    # Helper methods (override in subclasses for specific entity types)

    def _get_position(self, entity: Any) -> Tuple[float, ...]:
        """Get entity position."""
        if hasattr(entity, "position"):
            return entity.position
        if hasattr(entity, "get_position"):
            return entity.get_position()
        if isinstance(entity, dict):
            return entity.get("position", (0, 0, 0))
        return (0, 0, 0)

    def _get_velocity(self, entity: Any) -> Tuple[float, ...]:
        """Get entity velocity."""
        if hasattr(entity, "velocity"):
            return entity.velocity
        if hasattr(entity, "get_velocity"):
            return entity.get_velocity()
        if isinstance(entity, dict):
            return entity.get("velocity", (0, 0, 0))
        return (0, 0, 0)

    def _get_rotation(self, entity: Any) -> Tuple[float, float, float, float]:
        """Get entity rotation as quaternion."""
        if hasattr(entity, "rotation"):
            return entity.rotation
        if hasattr(entity, "get_rotation"):
            return entity.get_rotation()
        if isinstance(entity, dict):
            return entity.get("rotation", (0, 0, 0, 1))
        return (0, 0, 0, 1)

    def _get_health(self, entity: Any) -> float:
        """Get entity health."""
        if hasattr(entity, "health"):
            return entity.health
        if hasattr(entity, "get_health"):
            return entity.get_health()
        if isinstance(entity, dict):
            return entity.get("health", 0)
        return 0

    def _is_alive(self, entity: Any) -> bool:
        """Check if entity is alive."""
        if hasattr(entity, "is_alive"):
            return entity.is_alive()
        if hasattr(entity, "alive"):
            return entity.alive
        health = self._get_health(entity)
        return health > 0

    def _check_collision(self, entity_a: Any, entity_b: Any) -> bool:
        """Check if two entities are colliding."""
        # Simple AABB collision check
        pos_a = self._get_position(entity_a)
        pos_b = self._get_position(entity_b)

        # Get sizes (default to point)
        size_a = getattr(entity_a, "size", (0, 0, 0))
        size_b = getattr(entity_b, "size", (0, 0, 0))

        if isinstance(entity_a, dict):
            size_a = entity_a.get("size", (0, 0, 0))
        if isinstance(entity_b, dict):
            size_b = entity_b.get("size", (0, 0, 0))

        for i in range(min(len(pos_a), len(pos_b))):
            half_a = size_a[i] / 2 if i < len(size_a) else 0
            half_b = size_b[i] / 2 if i < len(size_b) else 0

            if abs(pos_a[i] - pos_b[i]) > half_a + half_b:
                return False

        return True
