"""
Tests for QueryCacheMirror - meta-level inspection of query cache state.

Tests cover:
- QueryInfo creation and attributes
- QueryCacheMirror creation from empty and populated caches
- Hit/miss statistics reflection
- Hit rate calculation including edge cases
- Convenience function
"""
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.query import Query, QueryCache
from foundation.query_cache_mirror import QueryInfo, QueryCacheMirror, mirror_query_cache


# =============================================================================
# TEST FIXTURES AND HELPERS
# =============================================================================

class Entity:
    """Simple entity class for testing."""
    def __init__(self, id: int, **kwargs):
        self.id = id
        for key, value in kwargs.items():
            setattr(self, key, value)


class World:
    """Simple world class for testing."""
    def __init__(self, entities: list = None):
        self._entities = entities or []

    @property
    def entities(self):
        return self._entities


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        Entity(1, name="Hero", health=100, team="blue"),
        Entity(2, name="Enemy1", health=50, team="red"),
        Entity(3, name="Enemy2", health=25, team="red"),
    ]


@pytest.fixture
def world(sample_entities):
    """Create a world with sample entities."""
    return World(sample_entities)


# =============================================================================
# TEST QUERY INFO
# =============================================================================

class TestQueryInfo:
    """Tests for QueryInfo dataclass."""

    def test_query_info_creation(self):
        """QueryInfo stores query metadata."""
        info = QueryInfo(
            hash="abc123",
            components=["Enemy", "Health"],
            cached_count=5,
            filters=["health__lt=50"]
        )
        assert info.hash == "abc123"
        assert info.components == ["Enemy", "Health"]
        assert info.cached_count == 5
        assert info.filters == ["health__lt=50"]

    def test_query_info_default_filters(self):
        """QueryInfo filters defaults to empty list."""
        info = QueryInfo(
            hash="def456",
            components=["Position"],
            cached_count=3
        )
        assert info.filters == []

    def test_query_info_empty_components(self):
        """QueryInfo can have empty components list."""
        info = QueryInfo(
            hash="ghi789",
            components=[],
            cached_count=0
        )
        assert info.components == []
        assert info.cached_count == 0


# =============================================================================
# TEST QUERY CACHE MIRROR - EMPTY CACHE
# =============================================================================

class TestQueryCacheMirrorEmpty:
    """Tests for QueryCacheMirror with empty cache."""

    def test_mirror_empty_cache(self):
        """Mirror reflects empty cache."""
        cache = QueryCache()
        mirror = QueryCacheMirror.from_cache(cache)

        assert mirror.registered_queries == 0
        assert mirror.cache_hits == 0
        assert mirror.cache_misses == 0
        assert mirror.queries == []

    def test_mirror_empty_cache_hit_rate(self):
        """Empty cache has zero hit rate."""
        cache = QueryCache()
        mirror = QueryCacheMirror.from_cache(cache)

        assert mirror.hit_rate == 0.0

    def test_mirror_empty_cache_total_entities(self):
        """Empty cache has zero cached entities."""
        cache = QueryCache()
        mirror = QueryCacheMirror.from_cache(cache)

        assert mirror.total_cached_entities == 0


# =============================================================================
# TEST QUERY CACHE MIRROR - WITH QUERIES
# =============================================================================

class TestQueryCacheMirrorWithQueries:
    """Tests for QueryCacheMirror with cached queries."""

    def test_mirror_with_queries(self, world):
        """Mirror reflects cached queries."""
        cache = QueryCache()
        q = Query().where(team="red")

        # Execute and cache
        results = q(world)
        cache.set(q, results)

        mirror = QueryCacheMirror.from_cache(cache)

        assert mirror.registered_queries == 1
        assert len(mirror.queries) == 1
        assert mirror.queries[0].cached_count == 2  # Enemy1 and Enemy2

    def test_mirror_multiple_queries(self, world):
        """Mirror reflects multiple cached queries."""
        cache = QueryCache()
        q1 = Query().where(team="red")
        q2 = Query().where(team="blue")

        cache.set(q1, q1(world))
        cache.set(q2, q2(world))

        mirror = QueryCacheMirror.from_cache(cache)

        assert mirror.registered_queries == 2
        assert len(mirror.queries) == 2

    def test_mirror_total_cached_entities(self, world):
        """Mirror calculates total cached entities."""
        cache = QueryCache()
        q1 = Query().where(team="red")
        q2 = Query().where(team="blue")

        cache.set(q1, q1(world))  # 2 entities
        cache.set(q2, q2(world))  # 1 entity

        mirror = QueryCacheMirror.from_cache(cache)

        assert mirror.total_cached_entities == 3

    def test_mirror_query_hash(self, world):
        """Mirror captures query hashes."""
        cache = QueryCache()
        q = Query().where(health=100)

        cache.set(q, q(world))

        mirror = QueryCacheMirror.from_cache(cache)

        assert len(mirror.queries) == 1
        assert mirror.queries[0].hash == q.hash()


# =============================================================================
# TEST HIT/MISS STATISTICS
# =============================================================================

class TestQueryCacheMirrorStats:
    """Tests for hit/miss statistics in mirror."""

    def test_mirror_hit_miss_stats(self, world):
        """Mirror reflects hit/miss statistics."""
        cache = QueryCache()
        q = Query().where(team="red")

        # Miss then hit
        cache.get(q)  # miss
        cache.set(q, q(world))
        cache.get(q)  # hit

        mirror = QueryCacheMirror.from_cache(cache)

        assert mirror.cache_hits == 1
        assert mirror.cache_misses == 1

    def test_mirror_multiple_hits(self, world):
        """Mirror reflects multiple cache hits."""
        cache = QueryCache()
        q = Query().where(team="blue")

        cache.get(q)  # miss
        cache.set(q, q(world))
        cache.get(q)  # hit
        cache.get(q)  # hit
        cache.get(q)  # hit

        mirror = QueryCacheMirror.from_cache(cache)

        assert mirror.cache_hits == 3
        assert mirror.cache_misses == 1

    def test_mirror_only_misses(self):
        """Mirror reflects cache with only misses."""
        cache = QueryCache()
        q1 = Query().where(health=100)
        q2 = Query().where(health=50)

        cache.get(q1)  # miss
        cache.get(q2)  # miss

        mirror = QueryCacheMirror.from_cache(cache)

        assert mirror.cache_hits == 0
        assert mirror.cache_misses == 2


# =============================================================================
# TEST HIT RATE CALCULATION
# =============================================================================

class TestHitRate:
    """Tests for hit rate calculation."""

    def test_hit_rate(self):
        """hit_rate property calculates correctly."""
        mirror = QueryCacheMirror(
            registered_queries=1,
            cache_hits=3,
            cache_misses=1,
            queries=[]
        )
        assert mirror.hit_rate == 0.75

    def test_hit_rate_zero_total(self):
        """hit_rate handles zero total."""
        mirror = QueryCacheMirror(0, 0, 0, [])
        assert mirror.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """hit_rate handles all hits."""
        mirror = QueryCacheMirror(1, 10, 0, [])
        assert mirror.hit_rate == 1.0

    def test_hit_rate_all_misses(self):
        """hit_rate handles all misses."""
        mirror = QueryCacheMirror(0, 0, 10, [])
        assert mirror.hit_rate == 0.0

    def test_hit_rate_50_percent(self):
        """hit_rate handles 50% hit rate."""
        mirror = QueryCacheMirror(1, 5, 5, [])
        assert mirror.hit_rate == 0.5


# =============================================================================
# TEST CONVENIENCE FUNCTION
# =============================================================================

class TestConvenienceFunction:
    """Tests for mirror_query_cache convenience function."""

    def test_convenience_function(self):
        """mirror_query_cache creates mirror."""
        cache = QueryCache()
        mirror = mirror_query_cache(cache)

        assert isinstance(mirror, QueryCacheMirror)

    def test_convenience_function_with_data(self, world):
        """mirror_query_cache captures cache state."""
        cache = QueryCache()
        q = Query().where(team="red")
        cache.set(q, q(world))

        mirror = mirror_query_cache(cache)

        assert mirror.registered_queries == 1
        assert len(mirror.queries) == 1


# =============================================================================
# TEST REPR
# =============================================================================

class TestRepr:
    """Tests for string representation."""

    def test_mirror_repr(self):
        """Mirror has informative repr."""
        mirror = QueryCacheMirror(
            registered_queries=3,
            cache_hits=10,
            cache_misses=5,
            queries=[]
        )

        repr_str = repr(mirror)

        assert "QueryCacheMirror" in repr_str
        assert "queries=3" in repr_str
        assert "hits=10" in repr_str
        assert "misses=5" in repr_str

    def test_mirror_repr_empty(self):
        """Empty mirror has valid repr."""
        mirror = QueryCacheMirror(0, 0, 0, [])
        repr_str = repr(mirror)

        assert "QueryCacheMirror" in repr_str


# =============================================================================
# TEST FILTER TRACKING
# =============================================================================

class TestFilterTracking:
    """Tests for filter/field tracking in mirror."""

    def test_mirror_tracks_query_fields(self, world):
        """Mirror captures tracked fields from queries."""
        cache = QueryCache()
        q = Query().where(health=100, team="blue")
        cache.set(q, q(world))

        mirror = QueryCacheMirror.from_cache(cache)

        assert len(mirror.queries) == 1
        # The filters list should contain the tracked field names
        assert 'health' in mirror.queries[0].filters
        assert 'team' in mirror.queries[0].filters
