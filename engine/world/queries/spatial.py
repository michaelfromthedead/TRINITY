"""
Spatial query system for the game engine world layer.

Provides raycast, sweep, and overlap queries with efficient spatial indexing.
Uses the Trinity pattern with @cached for query result caching.

Query Types:
    - Raycast: Line trace returning first/all hits
    - Sweep: Shape trace along a line
    - Overlap: Shape intersection test

Collision Channels:
    - DEFAULT, STATIC, DYNAMIC, PAWN, VEHICLE, PROJECTILE, TRIGGER

Known Limitations:
    - Capsule sweep is approximated as sphere sweep (capsule treated as sphere)
    - Sweep queries may miss very thin geometry at grazing angles
    - Box sweep uses discrete stepping and may tunnel through thin obstacles
    - Spatial index update frequency affects query accuracy for moving objects

Example:
    >>> query_system = SpatialQuerySystem(spatial_index)
    >>> ray = Ray(origin=(0, 0, 0), direction=(0, 0, 1), max_distance=100.0)
    >>> hit = query_system.execute_raycast(ray, QueryFilter())
    >>> if hit.hit:
    ...     print(f"Hit at {hit.position}, distance {hit.distance}")
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

from engine.world.queries.constants import (
    EPSILON_NORMALIZE,
    EPSILON_HIT_DETECTION,
    DEFAULT_RAY_DIRECTION,
    DEFAULT_RAY_MAX_DISTANCE,
    DEFAULT_MAX_HITS,
    DEFAULT_CLOSEST_POINT_DISTANCE,
    SWEEP_STEP_MULTIPLIER_SPHERE,
    SWEEP_MAX_ADAPTIVE_MULTIPLIER,
    DEFAULT_SPHERE_RADIUS,
    DEFAULT_BOX_HALF_EXTENT,
    DEFAULT_CAPSULE_RADIUS,
    DEFAULT_CAPSULE_HALF_HEIGHT,
    DEFAULT_OVERLAP_RADIUS,
)


# =============================================================================
# TYPE ALIASES
# =============================================================================

Vector3 = Tuple[float, float, float]
T = TypeVar("T")


# =============================================================================
# ENUMS
# =============================================================================


class QueryType(Enum):
    """Types of spatial queries supported by the engine."""

    RAYCAST = auto()
    SPHERE_SWEEP = auto()
    BOX_SWEEP = auto()
    CAPSULE_SWEEP = auto()
    SPHERE_OVERLAP = auto()
    BOX_OVERLAP = auto()


class CollisionChannel(Enum):
    """Collision channels for filtering query results."""

    DEFAULT = auto()
    STATIC = auto()
    DYNAMIC = auto()
    PAWN = auto()
    VEHICLE = auto()
    PROJECTILE = auto()
    TRIGGER = auto()


# =============================================================================
# QUERY FILTER
# =============================================================================


@dataclass
class QueryFilter:
    """
    Filter for spatial queries controlling what to include/exclude.

    Attributes:
        channels: Collision channels to query against.
        ignore_actors: Actor IDs to skip during query.
        tags_required: Objects must have ALL of these tags.
        tags_excluded: Objects must have NONE of these tags.
    """

    channels: Set[CollisionChannel] = field(
        default_factory=lambda: {CollisionChannel.DEFAULT}
    )
    ignore_actors: Set[int] = field(default_factory=set)
    tags_required: Set[str] = field(default_factory=set)
    tags_excluded: Set[str] = field(default_factory=set)

    def matches(
        self,
        actor_id: int,
        actor_channel: CollisionChannel,
        actor_tags: Set[str],
    ) -> bool:
        """
        Check if an actor passes this filter.

        Args:
            actor_id: The actor's unique identifier.
            actor_channel: The actor's collision channel.
            actor_tags: Tags associated with the actor.

        Returns:
            True if the actor passes all filter conditions.
        """
        # Check ignore list
        if actor_id in self.ignore_actors:
            return False

        # Check channel
        if actor_channel not in self.channels:
            return False

        # Check required tags (ALL must be present)
        if self.tags_required and not self.tags_required.issubset(actor_tags):
            return False

        # Check excluded tags (NONE must be present)
        if self.tags_excluded and self.tags_excluded.intersection(actor_tags):
            return False

        return True

    def with_channel(self, channel: CollisionChannel) -> "QueryFilter":
        """Return a new filter with an additional channel."""
        new_channels = self.channels.copy()
        new_channels.add(channel)
        return QueryFilter(
            channels=new_channels,
            ignore_actors=self.ignore_actors.copy(),
            tags_required=self.tags_required.copy(),
            tags_excluded=self.tags_excluded.copy(),
        )

    def without_actor(self, actor_id: int) -> "QueryFilter":
        """Return a new filter that ignores an additional actor."""
        new_ignore = self.ignore_actors.copy()
        new_ignore.add(actor_id)
        return QueryFilter(
            channels=self.channels.copy(),
            ignore_actors=new_ignore,
            tags_required=self.tags_required.copy(),
            tags_excluded=self.tags_excluded.copy(),
        )


# =============================================================================
# HIT RESULT
# =============================================================================


@dataclass
class HitResult:
    """
    Result of a spatial query hit.

    Attributes:
        hit: Whether a hit occurred.
        position: World position of the hit point.
        normal: Surface normal at the hit point.
        distance: Distance from query origin to hit.
        actor_id: ID of the actor that was hit.
        component_id: ID of the specific component hit.
        physical_material: Name of the physical material at hit point.
        bone_name: Name of the bone if hitting a skeletal mesh.
    """

    hit: bool = False
    position: Vector3 = (0.0, 0.0, 0.0)
    normal: Vector3 = (0.0, 1.0, 0.0)
    distance: float = 0.0
    actor_id: Optional[int] = None
    component_id: Optional[int] = None
    physical_material: str = "default"
    bone_name: Optional[str] = None

    @staticmethod
    def no_hit() -> "HitResult":
        """Create a no-hit result."""
        return HitResult(hit=False)

    def __lt__(self, other: "HitResult") -> bool:
        """Compare by distance for sorting."""
        return self.distance < other.distance


# =============================================================================
# RAY
# =============================================================================


@dataclass
class Ray:
    """
    A ray for raycasting queries.

    Attributes:
        origin: Starting point of the ray.
        direction: Normalized direction vector.
        max_distance: Maximum distance to trace.
    """

    origin: Vector3
    direction: Vector3
    max_distance: float = DEFAULT_RAY_MAX_DISTANCE

    def __post_init__(self) -> None:
        """Normalize the direction vector."""
        self.direction = self._normalize(self.direction)

    @staticmethod
    def _normalize(v: Vector3) -> Vector3:
        """Normalize a 3D vector."""
        length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if length < EPSILON_NORMALIZE:
            return DEFAULT_RAY_DIRECTION
        return (v[0] / length, v[1] / length, v[2] / length)

    def point_at(self, t: float) -> Vector3:
        """Get a point along the ray at distance t."""
        return (
            self.origin[0] + self.direction[0] * t,
            self.origin[1] + self.direction[1] * t,
            self.origin[2] + self.direction[2] * t,
        )

    def is_within_range(self, distance: float) -> bool:
        """Check if distance is within the ray's max range."""
        return 0 <= distance <= self.max_distance


# =============================================================================
# SPATIAL INDEX PROTOCOL
# =============================================================================


@runtime_checkable
class SpatialIndex(Protocol):
    """Protocol for spatial indexing structures."""

    def query_ray(
        self,
        origin: Vector3,
        direction: Vector3,
        max_distance: float,
    ) -> List[Tuple[int, float, Vector3, Vector3]]:
        """
        Query for objects intersecting a ray.

        Returns list of (actor_id, distance, position, normal) tuples.
        """
        ...

    def query_sphere(
        self, center: Vector3, radius: float
    ) -> List[int]:
        """Query for objects overlapping a sphere."""
        ...

    def query_box(
        self, min_point: Vector3, max_point: Vector3
    ) -> List[int]:
        """Query for objects overlapping an AABB."""
        ...

    def get_actor_channel(self, actor_id: int) -> CollisionChannel:
        """Get the collision channel for an actor."""
        ...

    def get_actor_tags(self, actor_id: int) -> Set[str]:
        """Get the tags for an actor."""
        ...

    def get_closest_point(
        self, actor_id: int, point: Vector3
    ) -> Vector3:
        """Get the closest point on an actor's geometry to a given point."""
        ...


# =============================================================================
# QUERY BASE CLASS
# =============================================================================


class SpatialQuery(ABC):
    """
    Base class for all spatial queries.

    Subclasses implement specific query types (raycast, sweep, overlap).
    """

    def __init__(self, filter: Optional[QueryFilter] = None) -> None:
        """
        Initialize the query with a filter.

        Args:
            filter: Optional filter for the query. Defaults to empty filter.
        """
        self.filter = filter or QueryFilter()

    @abstractmethod
    def execute(self, spatial_index: SpatialIndex) -> Any:
        """
        Execute the query against a spatial index.

        Args:
            spatial_index: The spatial index to query.

        Returns:
            Query-specific result type.
        """
        pass


# =============================================================================
# RAYCAST QUERIES
# =============================================================================


class RaycastQuery(SpatialQuery):
    """
    Single raycast query returning the first hit.

    Example:
        >>> query = RaycastQuery(
        ...     ray=Ray(origin=(0, 0, 0), direction=(0, 0, 1)),
        ...     filter=QueryFilter(channels={CollisionChannel.STATIC})
        ... )
        >>> result = query.execute(spatial_index)
    """

    def __init__(
        self,
        ray: Ray,
        filter: Optional[QueryFilter] = None,
    ) -> None:
        super().__init__(filter)
        self.ray = ray

    def execute(self, spatial_index: SpatialIndex) -> HitResult:
        """Execute raycast and return the closest hit."""
        raw_hits = spatial_index.query_ray(
            self.ray.origin,
            self.ray.direction,
            self.ray.max_distance,
        )

        closest: Optional[HitResult] = None
        closest_distance = float("inf")

        for actor_id, distance, position, normal in raw_hits:
            # Check range
            if not self.ray.is_within_range(distance):
                continue

            # Apply filter
            channel = spatial_index.get_actor_channel(actor_id)
            tags = spatial_index.get_actor_tags(actor_id)
            if not self.filter.matches(actor_id, channel, tags):
                continue

            # Track closest
            if distance < closest_distance:
                closest_distance = distance
                closest = HitResult(
                    hit=True,
                    position=position,
                    normal=normal,
                    distance=distance,
                    actor_id=actor_id,
                )

        return closest if closest else HitResult.no_hit()


class RaycastMultiQuery(SpatialQuery):
    """
    Raycast query returning multiple hits sorted by distance.

    Example:
        >>> query = RaycastMultiQuery(
        ...     ray=Ray(origin=(0, 0, 0), direction=(0, 0, 1)),
        ...     max_hits=10
        ... )
        >>> results = query.execute(spatial_index)
        >>> for hit in results:
        ...     print(f"Hit actor {hit.actor_id} at {hit.distance}")
    """

    def __init__(
        self,
        ray: Ray,
        max_hits: int = DEFAULT_MAX_HITS,
        filter: Optional[QueryFilter] = None,
    ) -> None:
        super().__init__(filter)
        self.ray = ray
        self.max_hits = max_hits

    def execute(self, spatial_index: SpatialIndex) -> List[HitResult]:
        """Execute raycast and return all hits up to max_hits."""
        raw_hits = spatial_index.query_ray(
            self.ray.origin,
            self.ray.direction,
            self.ray.max_distance,
        )

        results: List[HitResult] = []

        for actor_id, distance, position, normal in raw_hits:
            # Check range
            if not self.ray.is_within_range(distance):
                continue

            # Apply filter
            channel = spatial_index.get_actor_channel(actor_id)
            tags = spatial_index.get_actor_tags(actor_id)
            if not self.filter.matches(actor_id, channel, tags):
                continue

            results.append(
                HitResult(
                    hit=True,
                    position=position,
                    normal=normal,
                    distance=distance,
                    actor_id=actor_id,
                )
            )

        # Sort by distance and limit
        results.sort(key=lambda h: h.distance)
        return results[: self.max_hits]


# =============================================================================
# SWEEP QUERIES
# =============================================================================


@dataclass
class SweepShape:
    """Parameters for sweep shapes."""

    shape_type: str  # "sphere", "box", "capsule"
    params: Dict[str, float]

    @staticmethod
    def sphere(radius: float) -> "SweepShape":
        """Create a sphere sweep shape.

        Args:
            radius: Sphere radius. Must be positive.

        Returns:
            SweepShape configured for sphere sweep.

        Raises:
            ValueError: If radius is not positive.
        """
        if radius <= 0:
            raise ValueError(f"Sphere radius must be positive, got {radius}")
        return SweepShape(shape_type="sphere", params={"radius": radius})

    @staticmethod
    def box(half_extents: Vector3) -> "SweepShape":
        """Create a box sweep shape.

        Args:
            half_extents: Half extents (x, y, z). All must be positive.

        Returns:
            SweepShape configured for box sweep.

        Raises:
            ValueError: If any half extent is not positive.
        """
        if any(e <= 0 for e in half_extents):
            raise ValueError(f"Box half extents must be positive, got {half_extents}")
        return SweepShape(
            shape_type="box",
            params={
                "half_x": half_extents[0],
                "half_y": half_extents[1],
                "half_z": half_extents[2],
            },
        )

    @staticmethod
    def capsule(radius: float, half_height: float) -> "SweepShape":
        """Create a capsule sweep shape.

        Args:
            radius: Capsule radius. Must be positive.
            half_height: Half height of the capsule cylinder. Must be positive.

        Returns:
            SweepShape configured for capsule sweep.

        Raises:
            ValueError: If radius or half_height is not positive.
        """
        if radius <= 0:
            raise ValueError(f"Capsule radius must be positive, got {radius}")
        if half_height <= 0:
            raise ValueError(f"Capsule half_height must be positive, got {half_height}")
        return SweepShape(
            shape_type="capsule",
            params={"radius": radius, "half_height": half_height},
        )


class SweepQuery(SpatialQuery):
    """
    Sweep a shape along a line and find the first blocking hit.

    The shape is moved from start to end position, detecting collisions.

    Example:
        >>> query = SweepQuery(
        ...     shape=SweepShape.sphere(radius=1.0),
        ...     start=(0, 0, 0),
        ...     end=(10, 0, 0),
        ... )
        >>> result = query.execute(spatial_index)
    """

    def __init__(
        self,
        shape: SweepShape,
        start: Vector3,
        end: Vector3,
        filter: Optional[QueryFilter] = None,
    ) -> None:
        super().__init__(filter)
        self.shape = shape
        self.start = start
        self.end = end

    def execute(self, spatial_index: SpatialIndex) -> HitResult:
        """Execute sweep and return the first blocking hit."""
        # Calculate sweep direction and distance
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        dz = self.end[2] - self.start[2]
        total_distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if total_distance < EPSILON_HIT_DETECTION:
            return HitResult.no_hit()

        direction = (dx / total_distance, dy / total_distance, dz / total_distance)

        # For sphere sweep, expand the ray query by the radius
        if self.shape.shape_type == "sphere":
            radius = self.shape.params.get("radius", DEFAULT_SPHERE_RADIUS)
            if radius <= 0:
                return HitResult.no_hit()
            return self._sweep_sphere(spatial_index, direction, total_distance, radius)
        elif self.shape.shape_type == "box":
            return self._sweep_box(spatial_index, direction, total_distance)
        elif self.shape.shape_type == "capsule":
            return self._sweep_capsule(spatial_index, direction, total_distance)
        else:
            return HitResult.no_hit()

    def _sweep_sphere(
        self,
        spatial_index: SpatialIndex,
        direction: Vector3,
        distance: float,
        radius: float,
    ) -> HitResult:
        """Perform sphere sweep using expanded ray query."""
        # Simple implementation: sample along the sweep path
        step_count = max(1, int(distance / (radius * SWEEP_STEP_MULTIPLIER_SPHERE)))
        step_size = distance / step_count

        for i in range(step_count + 1):
            t = i * step_size
            sample_point = (
                self.start[0] + direction[0] * t,
                self.start[1] + direction[1] * t,
                self.start[2] + direction[2] * t,
            )

            # Check for overlaps at this position
            overlaps = spatial_index.query_sphere(sample_point, radius)

            for actor_id in overlaps:
                # Apply filter
                channel = spatial_index.get_actor_channel(actor_id)
                tags = spatial_index.get_actor_tags(actor_id)
                if not self.filter.matches(actor_id, channel, tags):
                    continue

                # Found a blocking hit
                return HitResult(
                    hit=True,
                    position=sample_point,
                    normal=(0.0, 1.0, 0.0),  # Simplified normal
                    distance=t,
                    actor_id=actor_id,
                )

        return HitResult.no_hit()

    def _sweep_box(
        self,
        spatial_index: SpatialIndex,
        direction: Vector3,
        distance: float,
    ) -> HitResult:
        """Perform box sweep."""
        half_x = self.shape.params.get("half_x", DEFAULT_BOX_HALF_EXTENT)
        half_y = self.shape.params.get("half_y", DEFAULT_BOX_HALF_EXTENT)
        half_z = self.shape.params.get("half_z", DEFAULT_BOX_HALF_EXTENT)

        # Validate dimensions - all must be positive
        if half_x <= 0 or half_y <= 0 or half_z <= 0:
            return HitResult.no_hit()

        max_half = max(half_x, half_y, half_z)

        step_count = max(1, int(distance / max_half))
        step_size = distance / step_count

        for i in range(step_count + 1):
            t = i * step_size
            center = (
                self.start[0] + direction[0] * t,
                self.start[1] + direction[1] * t,
                self.start[2] + direction[2] * t,
            )

            min_pt = (center[0] - half_x, center[1] - half_y, center[2] - half_z)
            max_pt = (center[0] + half_x, center[1] + half_y, center[2] + half_z)

            overlaps = spatial_index.query_box(min_pt, max_pt)

            for actor_id in overlaps:
                channel = spatial_index.get_actor_channel(actor_id)
                tags = spatial_index.get_actor_tags(actor_id)
                if not self.filter.matches(actor_id, channel, tags):
                    continue

                return HitResult(
                    hit=True,
                    position=center,
                    normal=(0.0, 1.0, 0.0),
                    distance=t,
                    actor_id=actor_id,
                )

        return HitResult.no_hit()

    def _sweep_capsule(
        self,
        spatial_index: SpatialIndex,
        direction: Vector3,
        distance: float,
    ) -> HitResult:
        """Perform capsule sweep (approximate with sphere at center)."""
        radius = self.shape.params.get("radius", DEFAULT_CAPSULE_RADIUS)
        half_height = self.shape.params.get("half_height", DEFAULT_CAPSULE_HALF_HEIGHT)

        # Validate dimensions
        if radius <= 0 or half_height <= 0:
            return HitResult.no_hit()

        # Use the larger dimension for stepping
        effective_radius = max(radius, half_height)

        step_count = max(1, int(distance / (effective_radius * SWEEP_STEP_MULTIPLIER_SPHERE)))
        step_size = distance / step_count

        for i in range(step_count + 1):
            t = i * step_size
            center = (
                self.start[0] + direction[0] * t,
                self.start[1] + direction[1] * t,
                self.start[2] + direction[2] * t,
            )

            # Approximate capsule as sphere with effective radius
            overlaps = spatial_index.query_sphere(center, effective_radius)

            for actor_id in overlaps:
                channel = spatial_index.get_actor_channel(actor_id)
                tags = spatial_index.get_actor_tags(actor_id)
                if not self.filter.matches(actor_id, channel, tags):
                    continue

                return HitResult(
                    hit=True,
                    position=center,
                    normal=(0.0, 1.0, 0.0),
                    distance=t,
                    actor_id=actor_id,
                )

        return HitResult.no_hit()


# =============================================================================
# OVERLAP QUERIES
# =============================================================================


class OverlapQuery(SpatialQuery):
    """
    Query for all actors overlapping a shape at a position.

    Example:
        >>> query = OverlapQuery(
        ...     shape="sphere",
        ...     shape_params={"radius": 5.0},
        ...     position=(0, 0, 0),
        ... )
        >>> actor_ids = query.execute(spatial_index)
    """

    def __init__(
        self,
        shape: str,
        shape_params: Dict[str, float],
        position: Vector3,
        filter: Optional[QueryFilter] = None,
    ) -> None:
        super().__init__(filter)
        self.shape = shape
        self.shape_params = shape_params
        self.position = position

    def execute(self, spatial_index: SpatialIndex) -> List[int]:
        """Execute overlap query and return list of overlapping actor IDs."""
        if self.shape == "sphere":
            radius = self.shape_params.get("radius", DEFAULT_OVERLAP_RADIUS)
            if radius <= 0:
                return []
            raw_overlaps = spatial_index.query_sphere(self.position, radius)
        elif self.shape == "box":
            half_x = self.shape_params.get("half_x", DEFAULT_OVERLAP_RADIUS)
            half_y = self.shape_params.get("half_y", DEFAULT_OVERLAP_RADIUS)
            half_z = self.shape_params.get("half_z", DEFAULT_OVERLAP_RADIUS)
            # Validate dimensions
            if half_x <= 0 or half_y <= 0 or half_z <= 0:
                return []
            min_pt = (
                self.position[0] - half_x,
                self.position[1] - half_y,
                self.position[2] - half_z,
            )
            max_pt = (
                self.position[0] + half_x,
                self.position[1] + half_y,
                self.position[2] + half_z,
            )
            raw_overlaps = spatial_index.query_box(min_pt, max_pt)
        else:
            return []

        # Apply filter
        results: List[int] = []
        for actor_id in raw_overlaps:
            channel = spatial_index.get_actor_channel(actor_id)
            tags = spatial_index.get_actor_tags(actor_id)
            if self.filter.matches(actor_id, channel, tags):
                results.append(actor_id)

        return results


# =============================================================================
# CLOSEST POINT QUERY
# =============================================================================


class ClosestPointQuery(SpatialQuery):
    """
    Find the closest point on world geometry to a given position.

    Example:
        >>> query = ClosestPointQuery(
        ...     position=(5, 10, 5),
        ...     max_distance=100.0,
        ... )
        >>> closest = query.execute(spatial_index)
    """

    def __init__(
        self,
        position: Vector3,
        max_distance: float = DEFAULT_CLOSEST_POINT_DISTANCE,
        filter: Optional[QueryFilter] = None,
    ) -> None:
        super().__init__(filter)
        self.position = position
        self.max_distance = max_distance

    def execute(self, spatial_index: SpatialIndex) -> Optional[Vector3]:
        """Find the closest point on any geometry within range."""
        # Query sphere at position with max_distance
        candidates = spatial_index.query_sphere(self.position, self.max_distance)

        closest_point: Optional[Vector3] = None
        closest_dist_sq = float("inf")

        for actor_id in candidates:
            # Apply filter
            channel = spatial_index.get_actor_channel(actor_id)
            tags = spatial_index.get_actor_tags(actor_id)
            if not self.filter.matches(actor_id, channel, tags):
                continue

            # Get closest point on this actor
            point = spatial_index.get_closest_point(actor_id, self.position)

            # Calculate distance squared
            dx = point[0] - self.position[0]
            dy = point[1] - self.position[1]
            dz = point[2] - self.position[2]
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq < closest_dist_sq:
                closest_dist_sq = dist_sq
                closest_point = point

        return closest_point


# =============================================================================
# SPATIAL QUERY SYSTEM
# =============================================================================


class SpatialQuerySystem:
    """
    Main system for executing spatial queries.

    Provides caching and optimized query execution.

    Example:
        >>> system = SpatialQuerySystem(spatial_index)
        >>> ray = Ray(origin=(0, 0, 0), direction=(0, 0, 1), max_distance=100)
        >>> hit = system.execute_raycast(ray, QueryFilter())
        >>> overlaps = system.execute_overlap("sphere", {"radius": 5}, (0, 0, 0))
    """

    def __init__(self, spatial_index: SpatialIndex) -> None:
        """
        Initialize the query system.

        Args:
            spatial_index: The spatial index to query against.
        """
        self._spatial_index = spatial_index
        self._cache_enabled = True
        self._cache: Dict[int, Any] = {}

    @property
    def spatial_index(self) -> SpatialIndex:
        """Get the spatial index."""
        return self._spatial_index

    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable query caching."""
        self._cache_enabled = enabled
        if not enabled:
            self._cache.clear()

    def invalidate_cache(self) -> None:
        """Clear all cached query results."""
        self._cache.clear()

    def execute_raycast(
        self,
        ray: Ray,
        filter: Optional[QueryFilter] = None,
    ) -> HitResult:
        """
        Execute a single raycast query.

        Args:
            ray: The ray to cast.
            filter: Optional query filter.

        Returns:
            HitResult with the closest hit, or no_hit if nothing was hit.
        """
        query = RaycastQuery(ray=ray, filter=filter)
        return query.execute(self._spatial_index)

    def execute_raycast_multi(
        self,
        ray: Ray,
        filter: Optional[QueryFilter] = None,
        max_hits: int = DEFAULT_MAX_HITS,
    ) -> List[HitResult]:
        """
        Execute a raycast returning multiple hits.

        Args:
            ray: The ray to cast.
            filter: Optional query filter.
            max_hits: Maximum number of hits to return.

        Returns:
            List of HitResults sorted by distance.
        """
        query = RaycastMultiQuery(ray=ray, max_hits=max_hits, filter=filter)
        return query.execute(self._spatial_index)

    def execute_sweep(
        self,
        shape: Union[str, SweepShape],
        start: Vector3,
        end: Vector3,
        filter: Optional[QueryFilter] = None,
        shape_params: Optional[Dict[str, float]] = None,
    ) -> HitResult:
        """
        Execute a sweep query.

        Args:
            shape: Shape type or SweepShape instance.
            start: Starting position.
            end: Ending position.
            filter: Optional query filter.
            shape_params: Shape parameters if shape is a string.

        Returns:
            HitResult with the first blocking hit.
        """
        if isinstance(shape, str):
            if shape == "sphere":
                radius = shape_params.get("radius", DEFAULT_SPHERE_RADIUS) if shape_params else DEFAULT_SPHERE_RADIUS
                if radius <= 0:
                    return HitResult.no_hit()
                sweep_shape = SweepShape.sphere(radius)
            elif shape == "box":
                params = shape_params or {}
                half_extents = (
                    params.get("half_x", DEFAULT_BOX_HALF_EXTENT),
                    params.get("half_y", DEFAULT_BOX_HALF_EXTENT),
                    params.get("half_z", DEFAULT_BOX_HALF_EXTENT),
                )
                if any(e <= 0 for e in half_extents):
                    return HitResult.no_hit()
                sweep_shape = SweepShape.box(half_extents)
            elif shape == "capsule":
                params = shape_params or {}
                radius = params.get("radius", DEFAULT_CAPSULE_RADIUS)
                half_height = params.get("half_height", DEFAULT_CAPSULE_HALF_HEIGHT)
                if radius <= 0 or half_height <= 0:
                    return HitResult.no_hit()
                sweep_shape = SweepShape.capsule(
                    radius=radius,
                    half_height=half_height,
                )
            else:
                return HitResult.no_hit()
        else:
            sweep_shape = shape

        query = SweepQuery(shape=sweep_shape, start=start, end=end, filter=filter)
        return query.execute(self._spatial_index)

    def execute_overlap(
        self,
        shape: str,
        position: Vector3,
        filter: Optional[QueryFilter] = None,
        shape_params: Optional[Dict[str, float]] = None,
    ) -> List[int]:
        """
        Execute an overlap query.

        Args:
            shape: Shape type ("sphere" or "box").
            position: Center position for the overlap test.
            filter: Optional query filter.
            shape_params: Shape parameters.

        Returns:
            List of actor IDs overlapping the shape.
        """
        params = shape_params or {"radius": DEFAULT_OVERLAP_RADIUS}
        query = OverlapQuery(
            shape=shape,
            shape_params=params,
            position=position,
            filter=filter,
        )
        return query.execute(self._spatial_index)

    def find_closest_point(
        self,
        position: Vector3,
        max_distance: float = DEFAULT_CLOSEST_POINT_DISTANCE,
        filter: Optional[QueryFilter] = None,
    ) -> Optional[Vector3]:
        """
        Find the closest point on world geometry.

        Args:
            position: Reference position.
            max_distance: Maximum search distance.
            filter: Optional query filter.

        Returns:
            Closest point on geometry, or None if nothing found.
        """
        query = ClosestPointQuery(
            position=position,
            max_distance=max_distance,
            filter=filter,
        )
        return query.execute(self._spatial_index)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "QueryType",
    "CollisionChannel",
    # Data classes
    "QueryFilter",
    "HitResult",
    "Ray",
    "SweepShape",
    # Protocols
    "SpatialIndex",
    # Query classes
    "SpatialQuery",
    "RaycastQuery",
    "RaycastMultiQuery",
    "SweepQuery",
    "OverlapQuery",
    "ClosestPointQuery",
    # Systems
    "SpatialQuerySystem",
]
