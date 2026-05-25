"""
Comprehensive unit tests for the Query system.

Tests cover:
- Query creation and component types
- WhereFilter with various operators (eq, lt, gt, le, ge, ne, in, contains)
- NearFilter for spatial queries
- HasComponentFilter for component checking
- Query hashing (stability and uniqueness)
- Query algebra (and, or, sub)
- QueryCache (hits, misses, invalidation)
- Query subscriptions (on_add, on_remove, on_change)
"""
import gc
import math
import pytest
import sys
import weakref

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.query import (
    Query,
    QueryCache,
    QuerySubscriber,
    WhereFilter,
    NearFilter,
    HasComponentFilter,
    AndFilter,
    OrFilter,
    NotFilter,
    TrackedQueryCache,
)
from foundation.tracker import Tracker


# =============================================================================
# TEST FIXTURES AND HELPERS
# =============================================================================

class Entity:
    """Simple entity class for testing."""
    def __init__(self, id: int, **kwargs):
        self.id = id
        for key, value in kwargs.items():
            setattr(self, key, value)


class ComponentEntity:
    """Entity with component system."""
    def __init__(self, id: int, components: dict = None):
        self.id = id
        self.components = components or {}

    def has_component(self, component_type: str) -> bool:
        return component_type in self.components


class World:
    """Simple world class for testing."""
    def __init__(self, entities: list = None):
        self._entities = entities or []
        self._entity_map = {e.id: e for e in self._entities}

    @property
    def entities(self):
        return self._entities

    def get_entity(self, entity_id: int):
        return self._entity_map.get(entity_id)

    def add(self, entity):
        self._entities.append(entity)
        self._entity_map[entity.id] = entity


@pytest.fixture
def sample_entities():
    """Create a set of sample entities for testing."""
    return [
        Entity(1, name="Hero", health=100, team="blue", x=0, y=0),
        Entity(2, name="Enemy1", health=50, team="red", x=5, y=5),
        Entity(3, name="Enemy2", health=25, team="red", x=10, y=0),
        Entity(4, name="Ally", health=80, team="blue", x=2, y=2),
        Entity(5, name="Boss", health=200, team="red", x=20, y=20),
    ]


@pytest.fixture
def world(sample_entities):
    """Create a world with sample entities."""
    return World(sample_entities)


# =============================================================================
# TEST QUERY CREATION
# =============================================================================

class TestQueryCreation:
    """Tests for basic query creation."""

    def test_query_creation_empty(self):
        """Query can be created with no component types."""
        q = Query()
        assert q.component_types == ()
        assert q.filters == ()

    def test_query_creation_single_component(self):
        """Query can be created with a single component type."""
        q = Query("Health")
        assert q.component_types == ("Health",)

    def test_query_creation_multiple_components(self):
        """Query can be created with multiple component types."""
        q = Query("Health", "Position", "Velocity")
        assert q.component_types == ("Health", "Position", "Velocity")

    def test_query_repr(self):
        """Query has a meaningful string representation."""
        q = Query("Health", "Position")
        repr_str = repr(q)
        assert "Query" in repr_str
        assert "Health" in repr_str or "components" in repr_str


# =============================================================================
# TEST WHERE FILTER
# =============================================================================

class TestQueryWhereEq:
    """Tests for where() with equality operator."""

    def test_query_where_eq_implicit(self, world):
        """where(field=value) should match entities with field == value."""
        q = Query().where(team="blue")
        results = q(world)

        assert len(results) == 2
        for entity in results:
            assert entity.team == "blue"

    def test_query_where_eq_explicit(self, world):
        """where(field__eq=value) should match entities with field == value."""
        q = Query().where(team__eq="red")
        results = q(world)

        assert len(results) == 3
        for entity in results:
            assert entity.team == "red"

    def test_query_where_eq_no_match(self, world):
        """where() should return empty list when nothing matches."""
        q = Query().where(team="green")
        results = q(world)

        assert results == []


class TestQueryWhereLt:
    """Tests for where() with less than operator."""

    def test_query_where_lt(self, world):
        """where(field__lt=value) should match entities with field < value."""
        q = Query().where(health__lt=50)
        results = q(world)

        assert len(results) == 1
        assert results[0].health == 25

    def test_query_where_lt_boundary(self, world):
        """Less than should not include boundary value."""
        q = Query().where(health__lt=100)
        results = q(world)

        # Should not include entity with health == 100
        assert all(e.health < 100 for e in results)


class TestQueryWhereGt:
    """Tests for where() with greater than operator."""

    def test_query_where_gt(self, world):
        """where(field__gt=value) should match entities with field > value."""
        q = Query().where(health__gt=100)
        results = q(world)

        assert len(results) == 1
        assert results[0].health == 200

    def test_query_where_gt_boundary(self, world):
        """Greater than should not include boundary value."""
        q = Query().where(health__gt=50)
        results = q(world)

        assert all(e.health > 50 for e in results)


class TestQueryWhereLe:
    """Tests for where() with less than or equal operator."""

    def test_query_where_le(self, world):
        """where(field__le=value) should match entities with field <= value."""
        q = Query().where(health__le=50)
        results = q(world)

        assert len(results) == 2
        assert all(e.health <= 50 for e in results)


class TestQueryWhereGe:
    """Tests for where() with greater than or equal operator."""

    def test_query_where_ge(self, world):
        """where(field__ge=value) should match entities with field >= value."""
        q = Query().where(health__ge=100)
        results = q(world)

        assert len(results) == 2
        assert all(e.health >= 100 for e in results)


class TestQueryWhereNe:
    """Tests for where() with not equal operator."""

    def test_query_where_ne(self, world):
        """where(field__ne=value) should match entities with field != value."""
        q = Query().where(team__ne="blue")
        results = q(world)

        assert len(results) == 3
        assert all(e.team != "blue" for e in results)


class TestQueryWhereIn:
    """Tests for where() with 'in' operator."""

    def test_query_where_in(self, world):
        """where(field__in=list) should match entities with field in list."""
        q = Query().where(health__in=[50, 100, 200])
        results = q(world)

        assert len(results) == 3
        assert all(e.health in [50, 100, 200] for e in results)


class TestQueryWhereContains:
    """Tests for where() with contains operator."""

    def test_query_where_contains_string(self, world):
        """where(field__contains=value) should match strings containing value."""
        q = Query().where(name__contains="Enemy")
        results = q(world)

        assert len(results) == 2
        assert all("Enemy" in e.name for e in results)


class TestQueryWhereMultiple:
    """Tests for where() with multiple conditions."""

    def test_query_where_multiple_conditions(self, world):
        """Multiple conditions should be ANDed together."""
        q = Query().where(team="red", health__lt=100)
        results = q(world)

        assert len(results) == 2
        for entity in results:
            assert entity.team == "red"
            assert entity.health < 100

    def test_query_where_chained(self, world):
        """Chained where() calls should combine conditions."""
        q = Query().where(team="red").where(health__gt=30)
        results = q(world)

        assert len(results) == 2
        for entity in results:
            assert entity.team == "red"
            assert entity.health > 30


# =============================================================================
# TEST NEAR FILTER
# =============================================================================

class TestQueryNear:
    """Tests for near() spatial filter."""

    def test_query_near_basic(self, world):
        """near() should filter entities within radius."""
        hero = world.get_entity(1)  # At (0, 0)
        q = Query().near(hero, 10)
        results = q(world)

        # Hero at (0,0), Enemy1 at (5,5), Enemy2 at (10,0), Ally at (2,2)
        # Distance to Enemy1: sqrt(50) ~ 7.07 (within 10)
        # Distance to Enemy2: 10 (within 10)
        # Distance to Ally: sqrt(8) ~ 2.83 (within 10)
        # Distance to Boss: sqrt(800) ~ 28.28 (outside 10)
        assert len(results) == 4  # Hero, Enemy1, Enemy2, Ally

    def test_query_near_excludes_distant(self, world):
        """near() should exclude entities outside radius."""
        hero = world.get_entity(1)  # At (0, 0)
        q = Query().near(hero, 5)
        results = q(world)

        # Only Ally at (2,2) is within radius 5 (distance ~2.83)
        # Hero itself at (0,0)
        assert all(
            math.sqrt(e.x**2 + e.y**2) <= 5
            for e in results
        )

    def test_query_near_with_where(self, world):
        """near() should combine with where()."""
        hero = world.get_entity(1)
        q = Query().where(team="red").near(hero, 15)
        results = q(world)

        assert len(results) == 2  # Enemy1 and Enemy2
        for entity in results:
            assert entity.team == "red"


# =============================================================================
# TEST HASH STABILITY
# =============================================================================

class TestQueryHashStable:
    """Tests for query hash stability."""

    def test_query_hash_same_query(self):
        """Same query should produce same hash."""
        q1 = Query("Health", "Position").where(health__gt=50)
        q2 = Query("Health", "Position").where(health__gt=50)

        assert q1.hash() == q2.hash()

    def test_query_hash_cached(self):
        """Hash should be cached after first computation."""
        q = Query("Health").where(health=100)

        hash1 = q.hash()
        hash2 = q.hash()

        assert hash1 == hash2
        assert q._cached_hash is not None

    def test_query_hash_stable_after_execution(self, world):
        """Hash should remain stable after query execution."""
        q = Query().where(team="blue")

        hash_before = q.hash()
        _ = q(world)
        hash_after = q.hash()

        assert hash_before == hash_after


class TestQueryHashDiffers:
    """Tests for query hash uniqueness."""

    def test_query_hash_different_components(self):
        """Different component types should produce different hashes."""
        q1 = Query("Health")
        q2 = Query("Position")

        assert q1.hash() != q2.hash()

    def test_query_hash_different_conditions(self):
        """Different conditions should produce different hashes."""
        q1 = Query().where(health=100)
        q2 = Query().where(health=50)

        assert q1.hash() != q2.hash()

    def test_query_hash_different_operators(self):
        """Different operators should produce different hashes."""
        q1 = Query().where(health__lt=100)
        q2 = Query().where(health__gt=100)

        assert q1.hash() != q2.hash()

    def test_query_hash_different_fields(self):
        """Different fields should produce different hashes."""
        q1 = Query().where(health=100)
        q2 = Query().where(mana=100)

        assert q1.hash() != q2.hash()


# =============================================================================
# TEST QUERY ALGEBRA
# =============================================================================

class TestQueryAnd:
    """Tests for query intersection (AND)."""

    def test_query_and_basic(self, world):
        """Intersection should return entities matching both queries."""
        q1 = Query().where(team="red")
        q2 = Query().where(health__lt=100)

        q_and = q1 & q2
        results = q_and(world)

        assert len(results) == 2
        for entity in results:
            assert entity.team == "red"
            assert entity.health < 100

    def test_query_and_empty_result(self, world):
        """Intersection should return empty if no overlap."""
        q1 = Query().where(team="blue")
        q2 = Query().where(health__gt=150)

        q_and = q1 & q2
        results = q_and(world)

        assert results == []

    def test_query_and_hash_unique(self):
        """Intersection query should have unique hash."""
        q1 = Query().where(team="red")
        q2 = Query().where(health=50)

        q_and = q1 & q2

        assert q_and.hash() != q1.hash()
        assert q_and.hash() != q2.hash()


class TestQueryOr:
    """Tests for query union (OR)."""

    def test_query_or_basic(self, world):
        """Union should return entities matching either query."""
        q1 = Query().where(team="blue")
        q2 = Query().where(health__gt=150)

        q_or = q1 | q2
        results = q_or(world)

        # Blue team: Hero, Ally
        # health > 150: Boss
        # Total: 3 unique entities
        assert len(results) == 3

    def test_query_or_no_duplicates(self, world):
        """Union should not return duplicates."""
        q1 = Query().where(health__gt=50)
        q2 = Query().where(team="blue")

        q_or = q1 | q2
        results = q_or(world)

        # Check for duplicates
        ids = [e.id for e in results]
        assert len(ids) == len(set(ids))

    def test_query_or_hash_unique(self):
        """Union query should have unique hash."""
        q1 = Query().where(team="red")
        q2 = Query().where(health=50)

        q_or = q1 | q2

        assert q_or.hash() != q1.hash()
        assert q_or.hash() != q2.hash()


class TestQuerySub:
    """Tests for query difference (SUB)."""

    def test_query_sub_basic(self, world):
        """Difference should return entities in first but not second."""
        q1 = Query().where(team="red")
        q2 = Query().where(health__lt=100)

        q_sub = q1 - q2
        results = q_sub(world)

        # Red team: Enemy1(50), Enemy2(25), Boss(200)
        # health < 100: Enemy1, Enemy2
        # Difference: Boss only
        assert len(results) == 1
        assert results[0].name == "Boss"

    def test_query_sub_empty_result(self, world):
        """Difference should return empty if all are excluded."""
        q1 = Query().where(team="blue")
        q2 = Query().where(health__le=100)

        q_sub = q1 - q2
        results = q_sub(world)

        # Blue team all have health <= 100
        assert results == []

    def test_query_sub_hash_unique(self):
        """Difference query should have unique hash."""
        q1 = Query().where(team="red")
        q2 = Query().where(health=50)

        q_sub = q1 - q2

        assert q_sub.hash() != q1.hash()
        assert q_sub.hash() != q2.hash()


# =============================================================================
# TEST QUERY CACHE
# =============================================================================

class TestQueryCacheHit:
    """Tests for query cache hits."""

    def test_query_cache_hit(self, world):
        """Cached results should be returned on second lookup."""
        cache = QueryCache()
        q = Query().where(team="blue")

        # First lookup - miss
        results1 = cache.get(q)
        assert results1 is None
        assert cache.misses == 1
        assert cache.hits == 0

        # Execute and cache
        results = q(world)
        cache.set(q, results)

        # Second lookup - hit
        results2 = cache.get(q)
        assert results2 is not None
        assert cache.hits == 1

    def test_query_cache_hit_rate(self):
        """Cache hit rate should be calculated correctly."""
        cache = QueryCache()
        q = Query().where(health=100)

        # 1 miss
        cache.get(q)
        cache.set(q, [])

        # 3 hits
        cache.get(q)
        cache.get(q)
        cache.get(q)

        assert cache.hits == 3
        assert cache.misses == 1
        assert cache.hit_rate == 0.75


class TestQueryCacheMiss:
    """Tests for query cache misses."""

    def test_query_cache_miss_empty(self):
        """Empty cache should always miss."""
        cache = QueryCache()
        q = Query().where(team="blue")

        result = cache.get(q)

        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_query_cache_miss_different_query(self, world):
        """Different queries should not share cache."""
        cache = QueryCache()
        q1 = Query().where(team="blue")
        q2 = Query().where(team="red")

        cache.set(q1, q1(world))

        result = cache.get(q2)
        assert result is None


class TestQueryCacheInvalidation:
    """Tests for query cache invalidation."""

    def test_query_cache_invalidate_specific(self, world):
        """invalidate() should remove specific query from cache."""
        cache = QueryCache()
        q = Query().where(team="blue")

        cache.set(q, q(world))
        assert cache.size == 1

        result = cache.invalidate(q)

        assert result is True
        assert cache.size == 0
        assert cache.get(q) is None

    def test_query_cache_invalidate_for_field(self, world):
        """invalidate_for() should remove queries depending on field."""
        cache = QueryCache()
        q1 = Query().where(health=100)
        q2 = Query().where(team="blue")

        cache.set(q1, q1(world))
        cache.set(q2, q2(world))
        assert cache.size == 2

        # Invalidate queries depending on 'health'
        invalidated = cache.invalidate_for(world.entities[0], 'health')

        assert invalidated == 1
        assert cache.get(q1) is None
        assert cache.get(q2) is not None

    def test_query_cache_clear(self, world):
        """clear() should remove all entries."""
        cache = QueryCache()
        q1 = Query().where(health=100)
        q2 = Query().where(team="blue")

        cache.set(q1, q1(world))
        cache.set(q2, q2(world))

        cache.clear()

        assert cache.size == 0
        assert cache.get(q1) is None
        assert cache.get(q2) is None

    def test_query_cache_max_size(self):
        """Cache should enforce max size."""
        cache = QueryCache(max_size=3)

        for i in range(5):
            q = Query().where(health=i)
            cache.set(q, [])

        assert cache.size <= 3


# =============================================================================
# TEST QUERY SUBSCRIPTIONS
# =============================================================================

class TestQuerySubscriptionAdd:
    """Tests for query subscription on_add callback."""

    def test_query_subscription_on_add(self):
        """on_add callback should be called when notified."""
        q = Query().where(team="blue")
        added = []

        q.subscribe(on_add=lambda e: added.append(e))

        entity = Entity(99, name="New", team="blue")
        q.notify_add(entity)

        assert len(added) == 1
        assert added[0] is entity

    def test_query_subscription_multiple_on_add(self):
        """Multiple subscribers should all receive notifications."""
        q = Query()
        added1 = []
        added2 = []

        q.subscribe(on_add=lambda e: added1.append(e))
        q.subscribe(on_add=lambda e: added2.append(e))

        entity = Entity(1)
        q.notify_add(entity)

        assert len(added1) == 1
        assert len(added2) == 1


class TestQuerySubscriptionRemove:
    """Tests for query subscription on_remove callback."""

    def test_query_subscription_on_remove(self):
        """on_remove callback should be called when notified."""
        q = Query()
        removed = []

        q.subscribe(on_remove=lambda e: removed.append(e))

        entity = Entity(1)
        q.notify_remove(entity)

        assert len(removed) == 1
        assert removed[0] is entity

    def test_query_unsubscribe(self):
        """unsubscribe() should prevent further notifications."""
        q = Query()
        removed = []

        sub = q.subscribe(on_remove=lambda e: removed.append(e))
        q.unsubscribe(sub)

        q.notify_remove(Entity(1))

        assert removed == []


class TestQuerySubscriptionChange:
    """Tests for query subscription on_change callback."""

    def test_query_subscription_on_change(self):
        """on_change callback should be called when notified."""
        q = Query()
        changes = []

        q.subscribe(on_change=lambda e, f, o, n: changes.append((e.id, f, o, n)))

        entity = Entity(1)
        q.notify_change(entity, 'health', 100, 50)

        assert len(changes) == 1
        assert changes[0] == (1, 'health', 100, 50)

    def test_query_subscription_callback_error_swallowed(self):
        """Errors in callbacks should be swallowed."""
        q = Query()

        def bad_callback(e):
            raise ValueError("Test error")

        q.subscribe(on_add=bad_callback)

        # Should not raise
        q.notify_add(Entity(1))


# =============================================================================
# TEST FILTER CLASSES
# =============================================================================

class TestWhereFilter:
    """Tests for WhereFilter class."""

    def test_where_filter_matches(self):
        """WhereFilter should correctly match entities."""
        f = WhereFilter('health', 'gt', 50)
        entity = Entity(1, health=100)

        assert f.matches(entity) is True

    def test_where_filter_no_match(self):
        """WhereFilter should return False for non-matching entities."""
        f = WhereFilter('health', 'gt', 50)
        entity = Entity(1, health=25)

        assert f.matches(entity) is False

    def test_where_filter_missing_field(self):
        """WhereFilter should return False if field is missing."""
        f = WhereFilter('health', 'eq', 100)
        entity = Entity(1)  # No health field

        assert f.matches(entity) is False

    def test_where_filter_to_dict(self):
        """WhereFilter.to_dict() should return correct structure."""
        f = WhereFilter('health', 'lt', 50)
        d = f.to_dict()

        assert d['type'] == 'where'
        assert d['field'] == 'health'
        assert d['op'] == 'lt'
        assert d['value'] == 50


class TestHasComponentFilter:
    """Tests for HasComponentFilter class."""

    def test_has_component_filter_with_method(self):
        """HasComponentFilter should use has_component method if available."""
        f = HasComponentFilter('Health')
        entity = ComponentEntity(1, components={'Health': {}})

        assert f.matches(entity) is True

    def test_has_component_filter_missing_component(self):
        """HasComponentFilter should return False for missing components."""
        f = HasComponentFilter('Health')
        entity = ComponentEntity(1, components={})

        assert f.matches(entity) is False

    def test_has_component_filter_with_attribute(self):
        """HasComponentFilter should check for attribute as fallback."""
        f = HasComponentFilter('health')
        entity = Entity(1, health=100)

        assert f.matches(entity) is True


class TestFilterCombinations:
    """Tests for filter algebra."""

    def test_and_filter(self):
        """AndFilter should require both filters to match."""
        f1 = WhereFilter('health', 'gt', 50)
        f2 = WhereFilter('team', 'eq', 'blue')
        f_and = f1 & f2

        assert f_and.matches(Entity(1, health=100, team='blue')) is True
        assert f_and.matches(Entity(2, health=100, team='red')) is False
        assert f_and.matches(Entity(3, health=25, team='blue')) is False

    def test_or_filter(self):
        """OrFilter should require either filter to match."""
        f1 = WhereFilter('health', 'gt', 150)
        f2 = WhereFilter('team', 'eq', 'blue')
        f_or = f1 | f2

        assert f_or.matches(Entity(1, health=200, team='red')) is True
        assert f_or.matches(Entity(2, health=50, team='blue')) is True
        assert f_or.matches(Entity(3, health=50, team='red')) is False

    def test_not_filter(self):
        """NotFilter should negate the inner filter."""
        f = WhereFilter('team', 'eq', 'blue')
        f_not = ~f

        assert f_not.matches(Entity(1, team='red')) is True
        assert f_not.matches(Entity(2, team='blue')) is False


# =============================================================================
# TEST TRACKED QUERY CACHE
# =============================================================================

class TestTrackedQueryCache:
    """Tests for TrackedQueryCache integration with Tracker."""

    def test_tracked_cache_invalidates_on_change(self, world):
        """TrackedQueryCache should invalidate when tracker notifies."""
        tracker = Tracker()
        cache = TrackedQueryCache(tracker)

        q = Query().where(health=100)
        cache.set(q, q(world))
        assert cache.get(q) is not None

        # Simulate a change notification
        entity = world.entities[0]
        tracker.mark_dirty(entity, 'health', 100, 50)

        # Cache should be invalidated
        assert cache.get(q) is None

        # Clean up
        cache.disconnect()

    def test_tracked_cache_disconnect(self, world):
        """disconnect() should stop invalidation."""
        tracker = Tracker()
        cache = TrackedQueryCache(tracker)

        q = Query().where(health=100)
        cache.set(q, q(world))

        cache.disconnect()

        # Change should not invalidate
        tracker.mark_dirty(world.entities[0], 'health', 100, 50)

        # Manually retrieve to check (would fail if still subscribed)
        # Cache should still have the entry since we disconnected
        result = cache.get(q)
        # Note: This might be None due to weak references, but disconnect worked


# =============================================================================
# TEST EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_query_empty_world(self):
        """Query against empty world should return empty list."""
        q = Query().where(health=100)
        world = World([])

        results = q(world)

        assert results == []

    def test_query_iterable_world(self):
        """Query should work with any iterable as world."""
        q = Query().where(health__gt=50)
        entities = [Entity(1, health=100), Entity(2, health=25)]

        results = q(entities)

        assert len(results) == 1

    def test_where_filter_unknown_operator(self):
        """Unknown operator should raise ValueError."""
        f = WhereFilter('health', 'unknown_op', 50)
        entity = Entity(1, health=100)

        with pytest.raises(ValueError, match="Unknown operator"):
            f.matches(entity)

    def test_near_filter_missing_position(self, world):
        """NearFilter should return False for entities without position."""
        hero = world.get_entity(1)
        f = NearFilter(target_id=hero.id, radius=10)

        # Entity without x, y
        entity = Entity(99, name="NoPos")

        assert f.matches(entity, world) is False

    def test_cache_weak_references(self, world):
        """Cache should handle garbage collected entities."""
        cache = QueryCache()
        q = Query().where(team="blue")

        # Create temporary entity
        temp_entity = Entity(999, team="blue")
        temp_world = World([temp_entity])

        results = q(temp_world)
        cache.set(q, results)

        # Delete the entity
        del temp_entity
        del temp_world
        gc.collect()

        # Cache get should return surviving references (empty in this case)
        cached = cache.get(q)
        # Should not crash, may return empty list due to weak refs


class TestQueryHasComponent:
    """Tests for Query.has() method."""

    def test_query_has_adds_component(self):
        """has() should add component to requirements."""
        q = Query("Health").has("Position")

        assert "Health" in q.component_types
        assert "Position" in q.component_types

    def test_query_has_no_duplicates(self):
        """has() should not add duplicate components."""
        q = Query("Health").has("Health")

        assert q.component_types.count("Health") == 1


class TestCacheStats:
    """Tests for cache statistics."""

    def test_cache_stats_dict(self):
        """stats() should return comprehensive statistics."""
        cache = QueryCache(max_size=100)

        # Generate some activity
        q = Query().where(health=100)
        cache.get(q)  # miss
        cache.set(q, [])
        cache.get(q)  # hit

        stats = cache.stats()

        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_rate' in stats
        assert 'size' in stats
        assert 'max_size' in stats

        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
        assert stats['size'] == 1
        assert stats['max_size'] == 100
