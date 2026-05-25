"""
Query - First-class query objects with identity, caching, and subscription.
Part of Core Foundation Layer 2 (Reactive).

Provides a declarative query system for filtering and retrieving entities,
with automatic caching, invalidation, and reactive subscriptions.

Features:
    - Declarative query construction with fluent API
    - Content-based identity hashing for caching
    - Query algebra (union, intersection, difference)
    - Reactive subscriptions for result set changes
    - Integration with Tracker for automatic invalidation
"""
from __future__ import annotations
import hashlib
import json
import logging
import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterator, Optional, TypeVar, Union

from foundation.constants import HASH_LENGTH, DEFAULT_QUERY_CACHE_SIZE

T = TypeVar('T')

# Logger for subscription callback errors
_logger = logging.getLogger(__name__)


# =============================================================================
# FILTER CLASSES
# =============================================================================

class Filter(ABC):
    """
    Base class for query filters.

    Filters are predicates that can be combined to form complex queries.
    Each filter must implement matches() for evaluation and to_dict() for
    content-based hashing.
    """

    @abstractmethod
    def matches(self, entity: Any, world: Any = None) -> bool:
        """
        Check if an entity matches this filter.

        Args:
            entity: The entity to check.
            world: Optional world context (needed for spatial queries).

        Returns:
            True if the entity matches, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Convert filter to a dictionary for hashing.

        Returns:
            Dictionary representation of the filter.
        """
        raise NotImplementedError

    def __and__(self, other: Filter) -> AndFilter:
        """Combine filters with AND logic."""
        return AndFilter(self, other)

    def __or__(self, other: Filter) -> OrFilter:
        """Combine filters with OR logic."""
        return OrFilter(self, other)

    def __invert__(self) -> NotFilter:
        """Negate this filter."""
        return NotFilter(self)


@dataclass(frozen=True)
class WhereFilter(Filter):
    """
    Filter by field condition.

    Supports various comparison operators:
        - 'eq': Equal to
        - 'ne': Not equal to
        - 'lt': Less than
        - 'le': Less than or equal
        - 'gt': Greater than
        - 'ge': Greater than or equal
        - 'in': Value is in collection
        - 'contains': Field contains value (for collections/strings)

    Attributes:
        field: Name of the field to check.
        op: Comparison operator.
        value: Value to compare against.
    """
    field: str
    op: str
    value: Any

    _OPERATORS = {
        'eq': lambda a, b: a == b,
        'ne': lambda a, b: a != b,
        'lt': lambda a, b: a < b,
        'le': lambda a, b: a <= b,
        'gt': lambda a, b: a > b,
        'ge': lambda a, b: a >= b,
        'in': lambda a, b: a in b,
        'contains': lambda a, b: b in a,
    }

    def matches(self, entity: Any, world: Any = None) -> bool:
        """Check if entity field matches the condition."""
        if not hasattr(entity, self.field):
            return False
        field_value = getattr(entity, self.field)
        op_func = self._OPERATORS.get(self.op)
        if op_func is None:
            raise ValueError(f"Unknown operator: {self.op}")
        try:
            return op_func(field_value, self.value)
        except TypeError:
            return False

    def to_dict(self) -> dict:
        """Convert to dictionary for hashing."""
        return {
            'type': 'where',
            'field': self.field,
            'op': self.op,
            'value': _serialize_value(self.value),
        }


@dataclass(frozen=True)
class NearFilter(Filter):
    """
    Spatial proximity filter.

    Filters entities within a given radius of a target entity.
    Requires both entities to have x, y attributes (and optionally z).

    Attributes:
        target_id: ID of the target entity.
        radius: Maximum distance from target.
    """
    target_id: int
    radius: float

    def matches(self, entity: Any, world: Any = None) -> bool:
        """Check if entity is within radius of target."""
        if world is None:
            return False

        # Get target entity from world
        target = None
        if hasattr(world, 'get_entity'):
            target = world.get_entity(self.target_id)
        elif hasattr(world, 'entities'):
            for e in world.entities:
                if getattr(e, 'id', None) == self.target_id:
                    target = e
                    break

        if target is None:
            return False

        # Check both have position
        if not (hasattr(entity, 'x') and hasattr(entity, 'y')):
            return False
        if not (hasattr(target, 'x') and hasattr(target, 'y')):
            return False

        # Calculate distance
        dx = getattr(entity, 'x', 0) - getattr(target, 'x', 0)
        dy = getattr(entity, 'y', 0) - getattr(target, 'y', 0)
        dz = 0
        if hasattr(entity, 'z') and hasattr(target, 'z'):
            dz = getattr(entity, 'z', 0) - getattr(target, 'z', 0)

        dist_sq = dx * dx + dy * dy + dz * dz
        return dist_sq <= self.radius * self.radius

    def to_dict(self) -> dict:
        """Convert to dictionary for hashing."""
        return {
            'type': 'near',
            'target_id': self.target_id,
            'radius': self.radius,
        }


@dataclass(frozen=True)
class HasComponentFilter(Filter):
    """
    Filter by component type.

    Filters entities that have a specific component attached.

    Attributes:
        component_type: Name of the component type to check for.
    """
    component_type: str

    def matches(self, entity: Any, world: Any = None) -> bool:
        """Check if entity has the component."""
        # Check for has_component method (ECS pattern)
        if hasattr(entity, 'has_component'):
            return entity.has_component(self.component_type)
        # Check for components dict
        if hasattr(entity, 'components'):
            components = getattr(entity, 'components')
            if isinstance(components, dict):
                return self.component_type in components
        # Check for attribute with component type name
        return hasattr(entity, self.component_type)

    def to_dict(self) -> dict:
        """Convert to dictionary for hashing."""
        return {
            'type': 'has_component',
            'component_type': self.component_type,
        }


@dataclass(frozen=True)
class AndFilter(Filter):
    """Combines two filters with AND logic."""
    left: Filter
    right: Filter

    def matches(self, entity: Any, world: Any = None) -> bool:
        """Return True only if both filters match."""
        return self.left.matches(entity, world) and self.right.matches(entity, world)

    def to_dict(self) -> dict:
        """Convert to dictionary for hashing."""
        return {
            'type': 'and',
            'left': self.left.to_dict(),
            'right': self.right.to_dict(),
        }


@dataclass(frozen=True)
class OrFilter(Filter):
    """Combines two filters with OR logic."""
    left: Filter
    right: Filter

    def matches(self, entity: Any, world: Any = None) -> bool:
        """Return True if either filter matches."""
        return self.left.matches(entity, world) or self.right.matches(entity, world)

    def to_dict(self) -> dict:
        """Convert to dictionary for hashing."""
        return {
            'type': 'or',
            'left': self.left.to_dict(),
            'right': self.right.to_dict(),
        }


@dataclass(frozen=True)
class NotFilter(Filter):
    """Negates a filter."""
    inner: Filter

    def matches(self, entity: Any, world: Any = None) -> bool:
        """Return True if inner filter does NOT match."""
        return not self.inner.matches(entity, world)

    def to_dict(self) -> dict:
        """Convert to dictionary for hashing."""
        return {
            'type': 'not',
            'inner': self.inner.to_dict(),
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _serialize_value(value: Any) -> Any:
    """Serialize a value for hashing purposes."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, set):
        return sorted([_serialize_value(v) for v in value], key=str)
    # For other types, use repr
    return repr(value)


def _parse_condition(key: str, value: Any) -> WhereFilter:
    """
    Parse a condition from field__op=value syntax.

    Examples:
        health=100 -> WhereFilter('health', 'eq', 100)
        health__lt=50 -> WhereFilter('health', 'lt', 50)
        name__contains='hero' -> WhereFilter('name', 'contains', 'hero')
    """
    if '__' in key:
        parts = key.rsplit('__', 1)
        field_name = parts[0]
        op = parts[1]
    else:
        field_name = key
        op = 'eq'

    return WhereFilter(field=field_name, op=op, value=value)


# =============================================================================
# QUERY SUBSCRIBER
# =============================================================================

@dataclass
class QuerySubscriber:
    """
    Subscriber for query result changes.

    Callbacks are invoked when entities enter or leave the query result set,
    or when matched entities are modified.

    Attributes:
        on_add: Called when an entity enters the result set.
        on_remove: Called when an entity leaves the result set.
        on_change: Called when a matched entity is modified.
    """
    on_add: Optional[Callable[[Any], None]] = None
    on_remove: Optional[Callable[[Any], None]] = None
    on_change: Optional[Callable[[Any, str, Any, Any], None]] = None


# =============================================================================
# QUERY CLASS
# =============================================================================

class Query:
    """
    First-class query object with identity and caching.

    Queries are declarative descriptions of entity sets that can be:
    - Executed against a world to get matching entities
    - Cached and invalidated automatically
    - Subscribed to for reactive updates
    - Combined with query algebra (union, intersection, difference)

    Usage:
        >>> q = Query("Enemy", "Health").where(health__lt=50).near(player, 10)
        >>> results = q(world)  # Execute query
        >>> q.subscribe(on_add=handle_add, on_remove=handle_remove)

    The Query class supports a fluent API for building complex queries:
        >>> Query("Unit").where(team="red", health__gt=0).near(base, 100)
    """

    def __init__(self, *component_types: str):
        """
        Create a new query.

        Args:
            *component_types: Component types that entities must have.
        """
        self._component_types: list[str] = list(component_types)
        self._filters: list[Filter] = []
        self._subscribers: list[QuerySubscriber] = []
        self._cached_hash: Optional[str] = None
        self._lock = threading.RLock()

    @property
    def component_types(self) -> tuple[str, ...]:
        """Get required component types."""
        return tuple(self._component_types)

    @property
    def filters(self) -> tuple[Filter, ...]:
        """Get applied filters."""
        return tuple(self._filters)

    def where(self, **conditions) -> Query:
        """
        Add filter conditions using field__op=value syntax.

        Supported operators:
            - eq (default): Equal to
            - ne: Not equal
            - lt: Less than
            - le: Less than or equal
            - gt: Greater than
            - ge: Greater than or equal
            - in: Value in collection
            - contains: Field contains value

        Args:
            **conditions: Field conditions as keyword arguments.

        Returns:
            New Query with conditions added (immutable pattern).

        Examples:
            >>> q.where(health=100)  # health == 100
            >>> q.where(health__lt=50)  # health < 50
            >>> q.where(team__in=['red', 'blue'])  # team in list
        """
        new_query = Query(*self._component_types)
        new_query._filters = list(self._filters)

        for key, value in conditions.items():
            new_query._filters.append(_parse_condition(key, value))

        return new_query

    def near(self, entity: Any, radius: float) -> Query:
        """
        Add spatial proximity filter.

        Args:
            entity: Target entity (must have 'id' attribute).
            radius: Maximum distance from target.

        Returns:
            New Query with proximity filter added.
        """
        entity_id = getattr(entity, 'id', id(entity))

        new_query = Query(*self._component_types)
        new_query._filters = list(self._filters)
        new_query._filters.append(NearFilter(target_id=entity_id, radius=radius))

        return new_query

    def has(self, component_type: str) -> Query:
        """
        Add component requirement.

        Args:
            component_type: Name of required component.

        Returns:
            New Query with component requirement added.
        """
        new_query = Query(*self._component_types)
        new_query._filters = list(self._filters)

        if component_type not in new_query._component_types:
            new_query._component_types.append(component_type)

        return new_query

    def hash(self) -> str:
        """
        Generate content-based identity hash.

        The hash is stable for identical queries, allowing them
        to share cache entries.

        Returns:
            Hexadecimal hash string.
        """
        with self._lock:
            if self._cached_hash is not None:
                return self._cached_hash

            canonical = {
                'component_types': sorted(self._component_types),
                'filters': [f.to_dict() for f in self._filters],
            }

            canonical_json = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
            self._cached_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()[:HASH_LENGTH]
            return self._cached_hash

    def _matches(self, entity: Any, world: Any = None) -> bool:
        """Check if an entity matches all query criteria."""
        # Check component types
        for comp_type in self._component_types:
            filter_check = HasComponentFilter(comp_type)
            if not filter_check.matches(entity, world):
                return False

        # Check all filters
        for f in self._filters:
            if not f.matches(entity, world):
                return False

        return True

    def __call__(self, world: Any) -> list:
        """
        Execute query against a world.

        Args:
            world: World object containing entities.

        Returns:
            List of matching entities.
        """
        results = []

        # Get entities from world
        entities: list = []
        if hasattr(world, 'entities'):
            entities = list(world.entities) if hasattr(world.entities, '__iter__') else []
        elif hasattr(world, 'get_all_entities'):
            entities = world.get_all_entities()
        elif hasattr(world, '__iter__'):
            entities = list(world)

        for entity in entities:
            if self._matches(entity, world):
                results.append(entity)

        return results

    def subscribe(
        self,
        on_add: Optional[Callable[[Any], None]] = None,
        on_remove: Optional[Callable[[Any], None]] = None,
        on_change: Optional[Callable[[Any, str, Any, Any], None]] = None,
    ) -> QuerySubscriber:
        """
        Subscribe to query result changes.

        Args:
            on_add: Callback when entity enters result set.
            on_remove: Callback when entity leaves result set.
            on_change: Callback when matched entity changes.

        Returns:
            The created QuerySubscriber for later unsubscription.
        """
        subscriber = QuerySubscriber(on_add=on_add, on_remove=on_remove, on_change=on_change)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: QuerySubscriber) -> bool:
        """
        Remove a subscriber.

        Args:
            subscriber: The subscriber to remove.

        Returns:
            True if subscriber was found and removed.
        """
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
                return True
            return False

    def notify_add(self, entity: Any) -> None:
        """Notify subscribers that an entity was added to results."""
        with self._lock:
            subscribers = list(self._subscribers)
        for sub in subscribers:
            if sub.on_add is not None:
                try:
                    sub.on_add(entity)
                except Exception as e:
                    _logger.exception("Error in on_add callback: %s", e)

    def notify_remove(self, entity: Any) -> None:
        """Notify subscribers that an entity was removed from results."""
        with self._lock:
            subscribers = list(self._subscribers)
        for sub in subscribers:
            if sub.on_remove is not None:
                try:
                    sub.on_remove(entity)
                except Exception as e:
                    _logger.exception("Error in on_remove callback: %s", e)

    def notify_change(self, entity: Any, field: str, old_value: Any, new_value: Any) -> None:
        """Notify subscribers that a matched entity changed."""
        with self._lock:
            subscribers = list(self._subscribers)
        for sub in subscribers:
            if sub.on_change is not None:
                try:
                    sub.on_change(entity, field, old_value, new_value)
                except Exception as e:
                    _logger.exception("Error in on_change callback: %s", e)

    # =========================================================================
    # QUERY ALGEBRA
    # =========================================================================

    def __and__(self, other: Query) -> Query:
        """
        Intersection: entities must match BOTH queries.

        Args:
            other: Query to intersect with.

        Returns:
            New Query representing intersection.
        """
        return IntersectionQuery(self, other)

    def __or__(self, other: Query) -> Query:
        """
        Union: entities must match EITHER query.

        Args:
            other: Query to union with.

        Returns:
            New Query representing union.
        """
        return UnionQuery(self, other)

    def __sub__(self, other: Query) -> Query:
        """
        Difference: entities in self but NOT in other.

        Args:
            other: Query to subtract.

        Returns:
            New Query representing difference.
        """
        return DifferenceQuery(self, other)

    def __repr__(self) -> str:
        """String representation of the query."""
        parts = []
        if self._component_types:
            parts.append(f"components={self._component_types}")
        if self._filters:
            parts.append(f"filters={len(self._filters)}")
        return f"Query({', '.join(parts)})"


# =============================================================================
# COMPOSITE QUERIES
# =============================================================================

class IntersectionQuery(Query):
    """Query representing the intersection of two queries."""

    def __init__(self, left: Query, right: Query):
        super().__init__()
        self._left = left
        self._right = right

    def _matches(self, entity: Any, world: Any = None) -> bool:
        """Entity must match both queries."""
        return self._left._matches(entity, world) and self._right._matches(entity, world)

    def hash(self) -> str:
        """Generate hash for intersection query."""
        with self._lock:
            if self._cached_hash is not None:
                return self._cached_hash

            canonical = {
                'type': 'intersection',
                'left': self._left.hash(),
                'right': self._right.hash(),
            }
            canonical_json = json.dumps(canonical, sort_keys=True)
            self._cached_hash = hashlib.sha256(canonical_json.encode()).hexdigest()[:HASH_LENGTH]
            return self._cached_hash


class UnionQuery(Query):
    """Query representing the union of two queries."""

    def __init__(self, left: Query, right: Query):
        super().__init__()
        self._left = left
        self._right = right

    def _matches(self, entity: Any, world: Any = None) -> bool:
        """Entity must match either query."""
        return self._left._matches(entity, world) or self._right._matches(entity, world)

    def hash(self) -> str:
        """Generate hash for union query."""
        with self._lock:
            if self._cached_hash is not None:
                return self._cached_hash

            canonical = {
                'type': 'union',
                'left': self._left.hash(),
                'right': self._right.hash(),
            }
            canonical_json = json.dumps(canonical, sort_keys=True)
            self._cached_hash = hashlib.sha256(canonical_json.encode()).hexdigest()[:HASH_LENGTH]
            return self._cached_hash


class DifferenceQuery(Query):
    """Query representing the difference of two queries."""

    def __init__(self, left: Query, right: Query):
        super().__init__()
        self._left = left
        self._right = right

    def _matches(self, entity: Any, world: Any = None) -> bool:
        """Entity must match left but NOT right."""
        return self._left._matches(entity, world) and not self._right._matches(entity, world)

    def hash(self) -> str:
        """Generate hash for difference query."""
        with self._lock:
            if self._cached_hash is not None:
                return self._cached_hash

            canonical = {
                'type': 'difference',
                'left': self._left.hash(),
                'right': self._right.hash(),
            }
            canonical_json = json.dumps(canonical, sort_keys=True)
            self._cached_hash = hashlib.sha256(canonical_json.encode()).hexdigest()[:HASH_LENGTH]
            return self._cached_hash


# =============================================================================
# QUERY CACHE
# =============================================================================

class QueryCache:
    """
    Cache for query results with automatic invalidation.

    The cache stores query results indexed by query hash. When entities
    change, affected queries can be invalidated.

    Usage:
        >>> cache = QueryCache()
        >>> results = cache.get(query)
        >>> if results is None:
        ...     results = query(world)
        ...     cache.set(query, results)
    """

    def __init__(self, max_size: int = DEFAULT_QUERY_CACHE_SIZE):
        """
        Create a new query cache.

        Args:
            max_size: Maximum number of cached queries.
        """
        self._cache: dict[str, list[weakref.ref]] = {}
        self._query_fields: dict[str, set[str]] = {}  # query_hash -> fields it depends on
        self._hits: int = 0
        self._misses: int = 0
        self._max_size: int = max_size
        self._lock = threading.RLock()

    def get(self, query: Query) -> Optional[list]:
        """
        Get cached results for a query.

        Args:
            query: The query to look up.

        Returns:
            Cached results list, or None if not cached.
        """
        query_hash = query.hash()

        with self._lock:
            if query_hash not in self._cache:
                self._misses += 1
                return None

            # Resolve weak references
            refs = self._cache[query_hash]
            results = []
            live_refs = []

            for ref in refs:
                obj = ref()
                if obj is not None:
                    results.append(obj)
                    live_refs.append(ref)

            # Update cache with only live references
            if len(live_refs) < len(refs):
                self._cache[query_hash] = live_refs

            self._hits += 1
            return results

    def set(self, query: Query, results: list) -> None:
        """
        Cache results for a query.

        Args:
            query: The query to cache results for.
            results: The list of matching entities.
        """
        query_hash = query.hash()

        with self._lock:
            # Enforce max size (simple LRU eviction)
            if len(self._cache) >= self._max_size:
                # Remove oldest entry (first key)
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                self._query_fields.pop(oldest, None)

            # Store weak references to results
            self._cache[query_hash] = [weakref.ref(entity) for entity in results]

            # Track which fields this query depends on
            fields: set[str] = set()
            for f in query.filters:
                if isinstance(f, WhereFilter):
                    fields.add(f.field)
            self._query_fields[query_hash] = fields

    def invalidate(self, query: Query) -> bool:
        """
        Invalidate a specific query's cache.

        Args:
            query: The query to invalidate.

        Returns:
            True if query was cached and is now invalidated.
        """
        query_hash = query.hash()

        with self._lock:
            if query_hash in self._cache:
                del self._cache[query_hash]
                self._query_fields.pop(query_hash, None)
                return True
            return False

    def invalidate_for(self, entity: Any, field: str) -> int:
        """
        Invalidate queries that might be affected by a field change.

        Args:
            entity: The entity that changed.
            field: The field that changed.

        Returns:
            Number of queries invalidated.
        """
        invalidated = 0

        with self._lock:
            # Find queries that depend on this field
            to_invalidate = []
            for query_hash, fields in self._query_fields.items():
                if field in fields or not fields:  # Empty fields means query might depend on anything
                    to_invalidate.append(query_hash)

            # Invalidate them
            for query_hash in to_invalidate:
                if query_hash in self._cache:
                    del self._cache[query_hash]
                    self._query_fields.pop(query_hash, None)
                    invalidated += 1

        return invalidated

    def invalidate_all_for_entity(self, entity: Any) -> int:
        """
        Invalidate all queries that might contain an entity.

        Args:
            entity: The entity to invalidate for.

        Returns:
            Number of queries invalidated.
        """
        entity_id = id(entity)
        invalidated = 0

        with self._lock:
            to_invalidate = []
            for query_hash, refs in self._cache.items():
                for ref in refs:
                    obj = ref()
                    if obj is not None and id(obj) == entity_id:
                        to_invalidate.append(query_hash)
                        break

            for query_hash in to_invalidate:
                del self._cache[query_hash]
                self._query_fields.pop(query_hash, None)
                invalidated += 1

        return invalidated

    def clear(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
            self._query_fields.clear()

    @property
    def hits(self) -> int:
        """Number of cache hits."""
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        """Number of cache misses."""
        with self._lock:
            return self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        """Number of cached queries."""
        with self._lock:
            return len(self._cache)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': self.hit_rate,
                'size': len(self._cache),
                'max_size': self._max_size,
            }

    def get_cache_snapshot(self) -> dict[str, Any]:
        """
        Get a snapshot of cache state for introspection (used by QueryCacheMirror).

        Returns:
            Dictionary with cache state including:
            - queries: List of dicts with hash, cached_count, tracked_fields
            - hits: Total cache hits
            - misses: Total cache misses
        """
        with self._lock:
            queries = []
            for query_hash, refs in self._cache.items():
                # Count live references
                cached_count = sum(1 for ref in refs if ref() is not None)
                tracked_fields = list(self._query_fields.get(query_hash, set()))
                queries.append({
                    'hash': query_hash,
                    'cached_count': cached_count,
                    'tracked_fields': tracked_fields,
                })
            return {
                'queries': queries,
                'hits': self._hits,
                'misses': self._misses,
            }


# =============================================================================
# TRACKED QUERY CACHE (integrates with Tracker)
# =============================================================================

class TrackedQueryCache(QueryCache):
    """
    Query cache that automatically invalidates when tracked objects change.

    Integrates with the Tracker system to automatically invalidate cached
    queries when their dependent entities are modified.

    Usage:
        >>> from foundation import tracker
        >>> cache = TrackedQueryCache(tracker)
        >>> # Cache automatically invalidates when entities change
    """

    def __init__(self, tracker_instance: Any, max_size: int = 1000):
        """
        Create a tracked query cache.

        Args:
            tracker_instance: The Tracker instance to listen to.
            max_size: Maximum number of cached queries.
        """
        super().__init__(max_size=max_size)
        self._tracker = tracker_instance

        # Subscribe to all changes
        self._tracker.on_change(self._on_change)

    def _on_change(self, obj: Any, field: str, old_value: Any, new_value: Any) -> None:
        """Handle change notifications from tracker."""
        self.invalidate_for(obj, field)

    def disconnect(self) -> None:
        """Disconnect from tracker notifications."""
        self._tracker.off_change(self._on_change)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Filter classes
    'Filter',
    'WhereFilter',
    'NearFilter',
    'HasComponentFilter',
    'AndFilter',
    'OrFilter',
    'NotFilter',
    # Query classes
    'Query',
    'QuerySubscriber',
    'IntersectionQuery',
    'UnionQuery',
    'DifferenceQuery',
    # Cache classes
    'QueryCache',
    'TrackedQueryCache',
]
