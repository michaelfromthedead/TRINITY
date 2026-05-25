"""
World Queries module for the game engine world layer.

Provides spatial, terrain, and navigation queries with efficient indexing
and caching support.

Submodules:
    - spatial: Raycast, sweep, and overlap queries
    - terrain: Height, normal, and terrain-specific queries
    - navigation: Pathfinding and reachability queries
    - constants: Configuration constants and magic number definitions

Example:
    >>> from engine.world.queries import (
    ...     SpatialQuerySystem,
    ...     TerrainQuerySystem,
    ...     NavigationQuerySystem,
    ... )
    >>>
    >>> # Spatial queries
    >>> spatial = SpatialQuerySystem(spatial_index)
    >>> hit = spatial.execute_raycast(ray, filter)
    >>>
    >>> # Terrain queries
    >>> terrain = TerrainQuerySystem(terrain_system)
    >>> height = terrain.query.get_height_at(100, 100)
    >>>
    >>> # Navigation queries
    >>> nav = NavigationQuerySystem(navmesh)
    >>> path = nav.query_path(start, end)
"""

from engine.world.queries.spatial import (
    # Enums
    QueryType,
    CollisionChannel,
    # Data classes
    QueryFilter,
    HitResult,
    Ray,
    SweepShape,
    # Protocols
    SpatialIndex,
    # Query classes
    SpatialQuery,
    RaycastQuery,
    RaycastMultiQuery,
    SweepQuery,
    OverlapQuery,
    ClosestPointQuery,
    # Systems
    SpatialQuerySystem,
)

from engine.world.queries.terrain import (
    # Data classes
    TerrainHitResult,
    # Protocols
    TerrainSystem,
    TerrainHoleManager,
    # Query classes
    TerrainQuery,
    TerrainRaycast,
    TerrainLineTrace,
    TerrainAreaQuery,
    TerrainVisibility,
    # Systems
    TerrainQuerySystem,
)

from engine.world.queries.navigation import (
    # Enums
    NavQueryResult,
    # Data classes
    NavPoint,
    NavPath,
    NavAreaCost,
    PathConfig,
    # Protocols
    NavMesh,
    # Query classes
    NavigationQuery,
    PathQuery,
    ReachabilityQuery,
    NavigationRaycast,
    NavModifierQuery,
    # Systems
    NavigationQuerySystem,
    # Testing
    StubNavMesh,
)

# Import constants module for direct access
from engine.world.queries import constants


__all__ = [
    # Spatial queries
    "QueryType",
    "CollisionChannel",
    "QueryFilter",
    "HitResult",
    "Ray",
    "SweepShape",
    "SpatialIndex",
    "SpatialQuery",
    "RaycastQuery",
    "RaycastMultiQuery",
    "SweepQuery",
    "OverlapQuery",
    "ClosestPointQuery",
    "SpatialQuerySystem",
    # Terrain queries
    "TerrainHitResult",
    "TerrainSystem",
    "TerrainHoleManager",
    "TerrainQuery",
    "TerrainRaycast",
    "TerrainLineTrace",
    "TerrainAreaQuery",
    "TerrainVisibility",
    "TerrainQuerySystem",
    # Navigation queries
    "NavQueryResult",
    "NavPoint",
    "NavPath",
    "NavAreaCost",
    "PathConfig",
    "NavMesh",
    "NavigationQuery",
    "PathQuery",
    "ReachabilityQuery",
    "NavigationRaycast",
    "NavModifierQuery",
    "NavigationQuerySystem",
    "StubNavMesh",
    # Constants module
    "constants",
]
