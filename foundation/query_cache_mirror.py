"""
QueryCacheMirror - Mirror for inspecting query cache state.
Part of Core Foundation meta-level inspection.

Provides read-only introspection of QueryCache state for debugging,
profiling, and monitoring cache effectiveness.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from foundation.query import QueryCache


@dataclass
class QueryInfo:
    """
    Information about a cached query.

    Attributes:
        hash: The query's content-based hash
        components: List of component type names the query filters on
        cached_count: Number of entities in cached results
        filters: List of filter descriptions
    """
    hash: str
    components: list[str]
    cached_count: int
    filters: list[str] = field(default_factory=list)


@dataclass
class QueryCacheMirror:
    """
    Mirror for inspecting QueryCache state.

    Provides a read-only snapshot of cache state for debugging and monitoring.
    The mirror captures the state at creation time and does not update
    automatically when the cache changes.

    Usage:
        from foundation.query import QueryCache
        from foundation.query_cache_mirror import QueryCacheMirror

        cache = QueryCache()
        # ... use cache ...
        mirror = QueryCacheMirror.from_cache(cache)
        print(f"Hits: {mirror.cache_hits}, Misses: {mirror.cache_misses}")
        print(f"Hit rate: {mirror.hit_rate:.1%}")
        for q in mirror.queries:
            print(f"Query {q.hash[:8]}: {q.cached_count} entities")

    Attributes:
        registered_queries: Number of queries in cache
        cache_hits: Total cache hits
        cache_misses: Total cache misses
        queries: List of QueryInfo for each cached query
    """
    registered_queries: int
    cache_hits: int
    cache_misses: int
    queries: list[QueryInfo] = field(default_factory=list)

    @classmethod
    def from_cache(cls, cache: 'QueryCache') -> 'QueryCacheMirror':
        """
        Create a mirror from a QueryCache.

        Args:
            cache: The QueryCache to inspect

        Returns:
            QueryCacheMirror with current cache state
        """
        queries = []

        # Use public API if available (preferred), otherwise fall back to internals
        if hasattr(cache, 'get_cache_snapshot'):
            snapshot = cache.get_cache_snapshot()
            for query_data in snapshot['queries']:
                query_info = QueryInfo(
                    hash=query_data['hash'],
                    components=[],  # Components require the original Query object
                    cached_count=query_data['cached_count'],
                    filters=query_data['tracked_fields'],
                )
                queries.append(query_info)
            return cls(
                registered_queries=len(queries),
                cache_hits=snapshot['hits'],
                cache_misses=snapshot['misses'],
                queries=queries,
            )

        # Legacy fallback: Access cache internals directly
        # This path is for backwards compatibility with older QueryCache versions
        if hasattr(cache, '_cache'):
            for query_hash, refs in cache._cache.items():
                # Count live references
                cached_count = 0
                for ref in refs:
                    if ref() is not None:
                        cached_count += 1

                # Get tracked fields for this query if available
                tracked_fields = []
                if hasattr(cache, '_query_fields'):
                    fields_set = cache._query_fields.get(query_hash, set())
                    tracked_fields = list(fields_set)

                query_info = QueryInfo(
                    hash=query_hash,
                    components=[],
                    cached_count=cached_count,
                    filters=tracked_fields,
                )
                queries.append(query_info)

        return cls(
            registered_queries=len(queries),
            cache_hits=getattr(cache, '_hits', 0),
            cache_misses=getattr(cache, '_misses', 0),
            queries=queries,
        )

    @property
    def hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Hit rate as a float from 0.0 to 1.0
        """
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def total_cached_entities(self) -> int:
        """
        Get total number of cached entity references.

        Returns:
            Sum of cached_count across all queries
        """
        return sum(q.cached_count for q in self.queries)

    def __repr__(self) -> str:
        """String representation of the mirror."""
        return (
            f"QueryCacheMirror(queries={self.registered_queries}, "
            f"hits={self.cache_hits}, misses={self.cache_misses}, "
            f"hit_rate={self.hit_rate:.1%})"
        )


def mirror_query_cache(cache: 'QueryCache') -> QueryCacheMirror:
    """
    Convenience function to create a mirror from a cache.

    Args:
        cache: The QueryCache to inspect

    Returns:
        QueryCacheMirror with current cache state

    Usage:
        mirror = mirror_query_cache(cache)
    """
    return QueryCacheMirror.from_cache(cache)


__all__ = [
    'QueryInfo',
    'QueryCacheMirror',
    'mirror_query_cache',
]
