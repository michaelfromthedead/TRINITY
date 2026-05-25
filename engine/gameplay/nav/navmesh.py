"""
NavMesh generation and queries.

Provides functionality for generating navigation meshes from geometry,
including voxelization, region building, contour tracing, and mesh building.
Also provides query operations for pathfinding and spatial queries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

from .constants import (
    DEFAULT_AGENT_HEIGHT,
    DEFAULT_AGENT_RADIUS,
    DEFAULT_CELL_HEIGHT,
    DEFAULT_CELL_SIZE,
    DEFAULT_CORRIDOR_WIDTH,
    DEFAULT_FUNNEL_EPSILON,
    DEFAULT_MAX_CONTOUR_ERROR,
    DEFAULT_MAX_EDGE_LENGTH,
    DEFAULT_MAX_SLOPE,
    DEFAULT_MAX_VERTICES_PER_POLY,
    DEFAULT_MERGE_REGION_AREA,
    DEFAULT_MIN_REGION_AREA,
    DEFAULT_SMOOTH_FACTOR,
    DEFAULT_SMOOTH_ITERATIONS,
    DEFAULT_STEP_HEIGHT,
    DEFAULT_TILE_SIZE,
    FLOAT_EPSILON,
    MAX_AGENT_HEIGHT,
    MAX_AGENT_RADIUS,
    MAX_CELL_SIZE,
    MAX_MAX_SLOPE,
    MAX_TILE_SIZE,
    MAX_VERTICES_PER_POLY,
    MIN_AGENT_HEIGHT,
    MIN_AGENT_RADIUS,
    MIN_CELL_SIZE,
    MIN_MAX_SLOPE,
    MIN_TILE_SIZE,
    MIN_VERTICES_PER_POLY,
    ZERO_LENGTH_THRESHOLD,
    NavMeshBuildMode,
    ObstacleType,
    QueryType,
)


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class Vector3:
    """3D vector for positions and directions."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> Vector3:
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> Vector3:
        return Vector3(-self.x, -self.y, -self.z)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3):
            return NotImplemented
        return (
            abs(self.x - other.x) < FLOAT_EPSILON and
            abs(self.y - other.y) < FLOAT_EPSILON and
            abs(self.z - other.z) < FLOAT_EPSILON
        )

    def __hash__(self) -> int:
        return hash((round(self.x, 6), round(self.y, 6), round(self.z, 6)))

    def dot(self, other: Vector3) -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        """Cross product."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self) -> float:
        """Vector magnitude."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_squared(self) -> float:
        """Squared vector magnitude (faster than length)."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> Vector3:
        """Return normalized vector."""
        mag = self.length()
        if mag < ZERO_LENGTH_THRESHOLD:
            return Vector3(0, 0, 0)
        return self / mag

    def distance_to(self, other: Vector3) -> float:
        """Distance to another point."""
        return (self - other).length()

    def distance_squared_to(self, other: Vector3) -> float:
        """Squared distance to another point (faster than distance_to)."""
        return (self - other).length_squared()

    def lerp(self, other: Vector3, t: float) -> Vector3:
        """Linear interpolation."""
        return self + (other - self) * t

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)

    @staticmethod
    def from_tuple(t: Tuple[float, float, float]) -> Vector3:
        """Create from tuple."""
        return Vector3(t[0], t[1], t[2])


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min_point: Vector3 = field(default_factory=Vector3)
    max_point: Vector3 = field(default_factory=Vector3)

    def contains(self, point: Vector3) -> bool:
        """Check if point is inside bounding box."""
        return (
            self.min_point.x <= point.x <= self.max_point.x and
            self.min_point.y <= point.y <= self.max_point.y and
            self.min_point.z <= point.z <= self.max_point.z
        )

    def intersects(self, other: BoundingBox) -> bool:
        """Check if this box intersects another."""
        return (
            self.min_point.x <= other.max_point.x and
            self.max_point.x >= other.min_point.x and
            self.min_point.y <= other.max_point.y and
            self.max_point.y >= other.min_point.y and
            self.min_point.z <= other.max_point.z and
            self.max_point.z >= other.min_point.z
        )

    def expand(self, amount: float) -> BoundingBox:
        """Return expanded bounding box."""
        return BoundingBox(
            Vector3(
                self.min_point.x - amount,
                self.min_point.y - amount,
                self.min_point.z - amount
            ),
            Vector3(
                self.max_point.x + amount,
                self.max_point.y + amount,
                self.max_point.z + amount
            )
        )

    def center(self) -> Vector3:
        """Return center point."""
        return (self.min_point + self.max_point) / 2

    def size(self) -> Vector3:
        """Return size in each dimension."""
        return self.max_point - self.min_point


@dataclass
class Triangle:
    """Triangle defined by three vertices."""
    v0: Vector3
    v1: Vector3
    v2: Vector3

    def normal(self) -> Vector3:
        """Calculate face normal."""
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        return edge1.cross(edge2).normalized()

    def area(self) -> float:
        """Calculate triangle area."""
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        return edge1.cross(edge2).length() / 2

    def centroid(self) -> Vector3:
        """Calculate centroid."""
        return (self.v0 + self.v1 + self.v2) / 3

    def bounding_box(self) -> BoundingBox:
        """Calculate bounding box."""
        return BoundingBox(
            Vector3(
                min(self.v0.x, self.v1.x, self.v2.x),
                min(self.v0.y, self.v1.y, self.v2.y),
                min(self.v0.z, self.v1.z, self.v2.z)
            ),
            Vector3(
                max(self.v0.x, self.v1.x, self.v2.x),
                max(self.v0.y, self.v1.y, self.v2.y),
                max(self.v0.z, self.v1.z, self.v2.z)
            )
        )


# =============================================================================
# NavMesh Components
# =============================================================================


@dataclass
class NavMeshParams:
    """Configuration parameters for NavMesh generation."""
    # Agent parameters
    agent_radius: float = DEFAULT_AGENT_RADIUS
    agent_height: float = DEFAULT_AGENT_HEIGHT
    step_height: float = DEFAULT_STEP_HEIGHT
    max_slope: float = DEFAULT_MAX_SLOPE

    # Voxelization
    cell_size: float = DEFAULT_CELL_SIZE
    cell_height: float = DEFAULT_CELL_HEIGHT

    # Region building
    min_region_area: float = DEFAULT_MIN_REGION_AREA
    merge_region_area: float = DEFAULT_MERGE_REGION_AREA

    # Contour tracing
    max_contour_error: float = DEFAULT_MAX_CONTOUR_ERROR
    max_edge_length: float = DEFAULT_MAX_EDGE_LENGTH

    # Mesh building
    max_vertices_per_poly: int = DEFAULT_MAX_VERTICES_PER_POLY

    # Tiling
    tile_size: float = DEFAULT_TILE_SIZE

    # Build mode
    build_mode: NavMeshBuildMode = NavMeshBuildMode.STATIC

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate all parameters are within acceptable ranges."""
        if not MIN_AGENT_RADIUS <= self.agent_radius <= MAX_AGENT_RADIUS:
            raise ValueError(
                f"agent_radius must be between {MIN_AGENT_RADIUS} and {MAX_AGENT_RADIUS}"
            )
        if not MIN_AGENT_HEIGHT <= self.agent_height <= MAX_AGENT_HEIGHT:
            raise ValueError(
                f"agent_height must be between {MIN_AGENT_HEIGHT} and {MAX_AGENT_HEIGHT}"
            )
        if self.step_height < 0:
            raise ValueError("step_height must be >= 0")
        if not MIN_MAX_SLOPE <= self.max_slope <= MAX_MAX_SLOPE:
            raise ValueError(
                f"max_slope must be between {MIN_MAX_SLOPE} and {MAX_MAX_SLOPE}"
            )
        if not MIN_CELL_SIZE <= self.cell_size <= MAX_CELL_SIZE:
            raise ValueError(
                f"cell_size must be between {MIN_CELL_SIZE} and {MAX_CELL_SIZE}"
            )
        if self.cell_height <= 0:
            raise ValueError("cell_height must be > 0")
        if self.min_region_area < 0:
            raise ValueError("min_region_area must be >= 0")
        if self.merge_region_area < 0:
            raise ValueError("merge_region_area must be >= 0")
        if self.max_contour_error <= 0:
            raise ValueError("max_contour_error must be > 0")
        if self.max_edge_length <= 0:
            raise ValueError("max_edge_length must be > 0")
        if not MIN_VERTICES_PER_POLY <= self.max_vertices_per_poly <= MAX_VERTICES_PER_POLY:
            raise ValueError(
                f"max_vertices_per_poly must be between {MIN_VERTICES_PER_POLY} and {MAX_VERTICES_PER_POLY}"
            )
        if not MIN_TILE_SIZE <= self.tile_size <= MAX_TILE_SIZE:
            raise ValueError(
                f"tile_size must be between {MIN_TILE_SIZE} and {MAX_TILE_SIZE}"
            )


@dataclass
class Voxel:
    """Single voxel in the heightfield."""
    x: int
    y: int
    z: int
    span_min: int = 0
    span_max: int = 0
    area: int = 0  # 0 = unwalkable, 1+ = walkable with area type
    region_id: int = 0


@dataclass
class HeightfieldSpan:
    """Vertical span in a heightfield column."""
    min_height: int
    max_height: int
    area: int = 0
    region_id: int = 0
    next: Optional[HeightfieldSpan] = None


@dataclass
class Heightfield:
    """Voxelized heightfield for NavMesh generation."""
    width: int
    depth: int
    cell_size: float
    cell_height: float
    bounds: BoundingBox
    spans: List[List[Optional[HeightfieldSpan]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize span grid if empty."""
        if not self.spans:
            self.spans = [
                [None for _ in range(self.depth)]
                for _ in range(self.width)
            ]

    def add_span(self, x: int, z: int, min_h: int, max_h: int, area: int = 1) -> None:
        """Add a span to the heightfield."""
        if not (0 <= x < self.width and 0 <= z < self.depth):
            return

        new_span = HeightfieldSpan(min_h, max_h, area)

        if self.spans[x][z] is None:
            self.spans[x][z] = new_span
            return

        # Insert in sorted order by min_height
        current = self.spans[x][z]
        prev: Optional[HeightfieldSpan] = None

        while current is not None and current.min_height < min_h:
            prev = current
            current = current.next

        if prev is None:
            new_span.next = self.spans[x][z]
            self.spans[x][z] = new_span
        else:
            new_span.next = prev.next
            prev.next = new_span

    def get_spans(self, x: int, z: int) -> List[HeightfieldSpan]:
        """Get all spans in a column."""
        result = []
        span = self.spans[x][z] if 0 <= x < self.width and 0 <= z < self.depth else None
        while span is not None:
            result.append(span)
            span = span.next
        return result


@dataclass
class Contour:
    """Contour representing the outline of a region."""
    vertices: List[Vector3] = field(default_factory=list)
    region_id: int = 0
    area: int = 0
    raw_vertices: List[Vector3] = field(default_factory=list)

    def vertex_count(self) -> int:
        """Return number of vertices."""
        return len(self.vertices)

    def is_valid(self) -> bool:
        """Check if contour is valid (has at least 3 vertices)."""
        return len(self.vertices) >= 3


@dataclass
class ContourSet:
    """Set of contours for all regions."""
    contours: List[Contour] = field(default_factory=list)
    bounds: BoundingBox = field(default_factory=BoundingBox)
    cell_size: float = DEFAULT_CELL_SIZE
    cell_height: float = DEFAULT_CELL_HEIGHT

    def add_contour(self, contour: Contour) -> None:
        """Add a contour to the set."""
        self.contours.append(contour)

    def contour_count(self) -> int:
        """Return number of contours."""
        return len(self.contours)


@dataclass
class NavMeshPolygon:
    """Single polygon in the NavMesh."""
    id: int
    vertices: List[Vector3] = field(default_factory=list)
    neighbors: List[int] = field(default_factory=list)  # Polygon IDs of neighbors
    area_type: int = 0
    flags: int = 0
    center: Vector3 = field(default_factory=Vector3)

    def __post_init__(self) -> None:
        """Calculate center if vertices exist."""
        if self.vertices and self.center == Vector3():
            self._calculate_center()

    def _calculate_center(self) -> None:
        """Calculate polygon center."""
        if not self.vertices:
            return
        total = Vector3()
        for v in self.vertices:
            total = total + v
        self.center = total / len(self.vertices)

    def contains_point_2d(self, point: Vector3) -> bool:
        """Check if point is inside polygon (2D test in XZ plane)."""
        n = len(self.vertices)
        if n < 3:
            return False

        inside = False
        j = n - 1
        for i in range(n):
            vi = self.vertices[i]
            vj = self.vertices[j]

            if ((vi.z > point.z) != (vj.z > point.z) and
                point.x < (vj.x - vi.x) * (point.z - vi.z) / (vj.z - vi.z) + vi.x):
                inside = not inside
            j = i

        return inside

    def get_edge(self, index: int) -> Tuple[Vector3, Vector3]:
        """Get edge at index."""
        n = len(self.vertices)
        return (self.vertices[index], self.vertices[(index + 1) % n])


@dataclass
class NavMeshTile:
    """Single tile in a tiled NavMesh."""
    id: int
    x: int
    z: int
    polygons: List[NavMeshPolygon] = field(default_factory=list)
    bounds: BoundingBox = field(default_factory=BoundingBox)
    off_mesh_connections: List[int] = field(default_factory=list)
    is_loaded: bool = True

    def polygon_count(self) -> int:
        """Return number of polygons in this tile."""
        return len(self.polygons)

    def get_polygon(self, poly_id: int) -> Optional[NavMeshPolygon]:
        """Get polygon by ID."""
        for poly in self.polygons:
            if poly.id == poly_id:
                return poly
        return None


# =============================================================================
# NavMesh Query Results
# =============================================================================


@dataclass
class NavMeshQueryResult:
    """Result of a NavMesh query."""
    success: bool = False
    query_type: QueryType = QueryType.NEAREST_POINT
    position: Optional[Vector3] = None
    polygon_id: Optional[int] = None
    distance: float = float('inf')
    hit_normal: Optional[Vector3] = None
    path: List[Vector3] = field(default_factory=list)
    visited_polygons: List[int] = field(default_factory=list)


@dataclass
class RaycastResult:
    """Result of a raycast query."""
    hit: bool = False
    position: Vector3 = field(default_factory=Vector3)
    normal: Vector3 = field(default_factory=Vector3)
    distance: float = float('inf')
    polygon_id: int = -1


# =============================================================================
# Dynamic Obstacles
# =============================================================================


@dataclass
class NavMeshObstacle:
    """Dynamic obstacle that affects the NavMesh."""
    id: int
    obstacle_type: ObstacleType = ObstacleType.CYLINDER
    position: Vector3 = field(default_factory=Vector3)
    rotation: float = 0.0  # Yaw rotation in radians
    radius: float = 1.0
    height: float = 2.0
    half_extents: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    vertices: List[Vector3] = field(default_factory=list)  # For convex obstacles
    carve: bool = True  # Whether this obstacle carves into NavMesh
    enabled: bool = True

    def get_bounds(self) -> BoundingBox:
        """Get bounding box of obstacle."""
        if self.obstacle_type == ObstacleType.CYLINDER:
            return BoundingBox(
                Vector3(
                    self.position.x - self.radius,
                    self.position.y,
                    self.position.z - self.radius
                ),
                Vector3(
                    self.position.x + self.radius,
                    self.position.y + self.height,
                    self.position.z + self.radius
                )
            )
        elif self.obstacle_type == ObstacleType.BOX:
            return BoundingBox(
                self.position - self.half_extents,
                self.position + self.half_extents
            )
        else:
            # Calculate from vertices
            if not self.vertices:
                return BoundingBox(self.position, self.position)
            min_p = Vector3(float('inf'), float('inf'), float('inf'))
            max_p = Vector3(float('-inf'), float('-inf'), float('-inf'))
            for v in self.vertices:
                min_p = Vector3(
                    min(min_p.x, v.x), min(min_p.y, v.y), min(min_p.z, v.z)
                )
                max_p = Vector3(
                    max(max_p.x, v.x), max(max_p.y, v.y), max(max_p.z, v.z)
                )
            return BoundingBox(min_p, max_p)


# =============================================================================
# NavMesh Class
# =============================================================================


class NavMesh:
    """
    Navigation mesh for pathfinding and spatial queries.

    Supports static, dynamic, and tiled NavMesh modes with
    optional obstacle carving.
    """

    def __init__(self, params: Optional[NavMeshParams] = None) -> None:
        """Initialize NavMesh with parameters."""
        self.params = params or NavMeshParams()
        self._polygons: Dict[int, NavMeshPolygon] = {}
        self._tiles: Dict[Tuple[int, int], NavMeshTile] = {}
        self._obstacles: Dict[int, NavMeshObstacle] = {}
        self._bounds = BoundingBox()
        self._is_built = False
        self._next_polygon_id = 0
        self._next_obstacle_id = 0
        self._dirty_tiles: Set[Tuple[int, int]] = set()
        self._heightfield: Optional[Heightfield] = None
        self._contour_set: Optional[ContourSet] = None

    # =========================================================================
    # Building Pipeline
    # =========================================================================

    def build(self, triangles: List[Triangle]) -> bool:
        """
        Build NavMesh from input geometry.

        Args:
            triangles: List of input triangles (world geometry)

        Returns:
            True if build succeeded
        """
        if not triangles:
            return False

        # Calculate bounds
        self._calculate_bounds(triangles)

        # Step 1: Voxelization
        self._heightfield = self._voxelize(triangles)
        if self._heightfield is None:
            return False

        # Step 2: Region building
        if not self._build_regions():
            return False

        # Step 3: Contour tracing
        self._contour_set = self._trace_contours()
        if self._contour_set is None:
            return False

        # Step 4: Mesh building
        if not self._build_mesh():
            return False

        self._is_built = True
        return True

    def _calculate_bounds(self, triangles: List[Triangle]) -> None:
        """Calculate bounding box from triangles."""
        min_p = Vector3(float('inf'), float('inf'), float('inf'))
        max_p = Vector3(float('-inf'), float('-inf'), float('-inf'))

        for tri in triangles:
            for v in [tri.v0, tri.v1, tri.v2]:
                min_p = Vector3(
                    min(min_p.x, v.x), min(min_p.y, v.y), min(min_p.z, v.z)
                )
                max_p = Vector3(
                    max(max_p.x, v.x), max(max_p.y, v.y), max(max_p.z, v.z)
                )

        self._bounds = BoundingBox(min_p, max_p)

    def _voxelize(self, triangles: List[Triangle]) -> Optional[Heightfield]:
        """
        Convert triangles to voxelized heightfield.

        Args:
            triangles: Input geometry

        Returns:
            Heightfield containing voxel data
        """
        size = self._bounds.size()
        width = max(1, int(math.ceil(size.x / self.params.cell_size)))
        depth = max(1, int(math.ceil(size.z / self.params.cell_size)))

        hf = Heightfield(
            width=width,
            depth=depth,
            cell_size=self.params.cell_size,
            cell_height=self.params.cell_height,
            bounds=self._bounds
        )

        # Rasterize each triangle
        for tri in triangles:
            self._rasterize_triangle(hf, tri)

        return hf

    def _rasterize_triangle(self, hf: Heightfield, tri: Triangle) -> None:
        """Rasterize a single triangle into the heightfield."""
        # Calculate triangle's slope
        normal = tri.normal()
        slope_angle = math.degrees(math.acos(abs(normal.y)))

        if slope_angle > self.params.max_slope:
            area = 0  # Unwalkable
        else:
            area = 1  # Walkable

        # Get triangle bounding box in cell coordinates
        tri_bounds = tri.bounding_box()
        min_x = max(0, int((tri_bounds.min_point.x - self._bounds.min_point.x) / self.params.cell_size))
        max_x = min(hf.width - 1, int((tri_bounds.max_point.x - self._bounds.min_point.x) / self.params.cell_size))
        min_z = max(0, int((tri_bounds.min_point.z - self._bounds.min_point.z) / self.params.cell_size))
        max_z = min(hf.depth - 1, int((tri_bounds.max_point.z - self._bounds.min_point.z) / self.params.cell_size))

        # Rasterize cells
        for x in range(min_x, max_x + 1):
            for z in range(min_z, max_z + 1):
                # Calculate cell center
                cx = self._bounds.min_point.x + (x + 0.5) * self.params.cell_size
                cz = self._bounds.min_point.z + (z + 0.5) * self.params.cell_size

                # Check if cell overlaps triangle (simplified)
                cell_center = Vector3(cx, 0, cz)
                if self._point_in_triangle_2d(cell_center, tri):
                    # Calculate height at this position
                    height = self._get_triangle_height_at(tri, cx, cz)
                    if height is not None:
                        min_h = int((height - self._bounds.min_point.y) / self.params.cell_height)
                        max_h = min_h + 1
                        hf.add_span(x, z, min_h, max_h, area)

    def _point_in_triangle_2d(self, p: Vector3, tri: Triangle) -> bool:
        """Check if point is inside triangle in XZ plane."""
        def sign(p1: Vector3, p2: Vector3, p3: Vector3) -> float:
            return (p1.x - p3.x) * (p2.z - p3.z) - (p2.x - p3.x) * (p1.z - p3.z)

        d1 = sign(p, tri.v0, tri.v1)
        d2 = sign(p, tri.v1, tri.v2)
        d3 = sign(p, tri.v2, tri.v0)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def _get_triangle_height_at(self, tri: Triangle, x: float, z: float) -> Optional[float]:
        """Get height of triangle at XZ position using barycentric interpolation."""
        # Calculate barycentric coordinates
        v0 = tri.v0
        v1 = tri.v1
        v2 = tri.v2

        denom = (v1.z - v2.z) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.z - v2.z)
        if abs(denom) < FLOAT_EPSILON:
            return None

        a = ((v1.z - v2.z) * (x - v2.x) + (v2.x - v1.x) * (z - v2.z)) / denom
        b = ((v2.z - v0.z) * (x - v2.x) + (v0.x - v2.x) * (z - v2.z)) / denom
        c = 1 - a - b

        if a < 0 or b < 0 or c < 0:
            return None

        return a * v0.y + b * v1.y + c * v2.y

    def _build_regions(self) -> bool:
        """Build walkable regions from heightfield."""
        if self._heightfield is None:
            return False

        # Simple region building - assign region IDs to connected walkable spans
        region_id = 1
        for x in range(self._heightfield.width):
            for z in range(self._heightfield.depth):
                spans = self._heightfield.get_spans(x, z)
                for span in spans:
                    if span.area > 0 and span.region_id == 0:
                        self._flood_fill_region(x, z, span, region_id)
                        region_id += 1

        return True

    def _flood_fill_region(
        self, start_x: int, start_z: int, start_span: HeightfieldSpan, region_id: int
    ) -> None:
        """Flood fill to assign region ID to connected spans."""
        if self._heightfield is None:
            return

        stack = [(start_x, start_z, start_span)]

        while stack:
            x, z, span = stack.pop()
            if span.region_id != 0:
                continue

            span.region_id = region_id

            # Check 4-connected neighbors
            for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, nz = x + dx, z + dz
                if 0 <= nx < self._heightfield.width and 0 <= nz < self._heightfield.depth:
                    for neighbor_span in self._heightfield.get_spans(nx, nz):
                        if (neighbor_span.area > 0 and
                            neighbor_span.region_id == 0 and
                            abs(neighbor_span.min_height - span.min_height) <=
                            int(self.params.step_height / self.params.cell_height)):
                            stack.append((nx, nz, neighbor_span))

    def _trace_contours(self) -> Optional[ContourSet]:
        """Trace contours around regions."""
        if self._heightfield is None:
            return None

        contour_set = ContourSet(
            bounds=self._bounds,
            cell_size=self.params.cell_size,
            cell_height=self.params.cell_height
        )

        # Find unique regions
        regions: Dict[int, List[Tuple[int, int, HeightfieldSpan]]] = {}
        for x in range(self._heightfield.width):
            for z in range(self._heightfield.depth):
                for span in self._heightfield.get_spans(x, z):
                    if span.region_id > 0:
                        if span.region_id not in regions:
                            regions[span.region_id] = []
                        regions[span.region_id].append((x, z, span))

        # Create contour for each region
        for region_id, cells in regions.items():
            contour = self._build_contour_for_region(region_id, cells)
            if contour and contour.is_valid():
                contour_set.add_contour(contour)

        return contour_set

    def _build_contour_for_region(
        self, region_id: int, cells: List[Tuple[int, int, HeightfieldSpan]]
    ) -> Optional[Contour]:
        """Build contour for a single region."""
        if not cells:
            return None

        # Simple convex hull approach for contour vertices
        points = []
        for x, z, span in cells:
            world_x = self._bounds.min_point.x + (x + 0.5) * self.params.cell_size
            world_y = self._bounds.min_point.y + span.min_height * self.params.cell_height
            world_z = self._bounds.min_point.z + (z + 0.5) * self.params.cell_size
            points.append(Vector3(world_x, world_y, world_z))

        if len(points) < 3:
            return None

        # Calculate convex hull (simplified 2D in XZ plane)
        hull = self._convex_hull_2d(points)

        return Contour(
            vertices=hull,
            region_id=region_id,
            area=1,
            raw_vertices=points
        )

    def _convex_hull_2d(self, points: List[Vector3]) -> List[Vector3]:
        """Calculate 2D convex hull of points in XZ plane (Graham scan)."""
        if len(points) < 3:
            return points

        # Find lowest point (smallest z, then smallest x)
        start = min(points, key=lambda p: (p.z, p.x))

        # Sort by polar angle
        def polar_angle(p: Vector3) -> float:
            return math.atan2(p.x - start.x, p.z - start.z)

        sorted_points = sorted(points, key=polar_angle)

        # Graham scan
        hull: List[Vector3] = []
        for p in sorted_points:
            while len(hull) >= 2:
                # Cross product to check turn direction
                o = hull[-2]
                a = hull[-1]
                cross = (a.x - o.x) * (p.z - o.z) - (a.z - o.z) * (p.x - o.x)
                if cross <= 0:
                    hull.pop()
                else:
                    break
            hull.append(p)

        return hull

    def _build_mesh(self) -> bool:
        """Build final NavMesh polygons from contours."""
        if self._contour_set is None:
            return False

        for contour in self._contour_set.contours:
            if not contour.is_valid():
                continue

            # Create polygon from contour
            polygon = NavMeshPolygon(
                id=self._next_polygon_id,
                vertices=contour.vertices.copy(),
                area_type=contour.area
            )
            self._polygons[polygon.id] = polygon
            self._next_polygon_id += 1

        # Build neighbor connections
        self._build_adjacency()

        return True

    def _build_adjacency(self) -> None:
        """Build neighbor connections between polygons."""
        polygon_list = list(self._polygons.values())

        for i, poly_a in enumerate(polygon_list):
            for j, poly_b in enumerate(polygon_list):
                if i >= j:
                    continue

                # Check if polygons share an edge
                if self._polygons_share_edge(poly_a, poly_b):
                    poly_a.neighbors.append(poly_b.id)
                    poly_b.neighbors.append(poly_a.id)

    def _polygons_share_edge(self, a: NavMeshPolygon, b: NavMeshPolygon) -> bool:
        """Check if two polygons share an edge."""
        threshold = self.params.cell_size * 0.5

        for i in range(len(a.vertices)):
            edge_a = a.get_edge(i)
            for j in range(len(b.vertices)):
                edge_b = b.get_edge(j)

                # Check if edges overlap (in either direction)
                if (edge_a[0].distance_to(edge_b[1]) < threshold and
                    edge_a[1].distance_to(edge_b[0]) < threshold):
                    return True
                if (edge_a[0].distance_to(edge_b[0]) < threshold and
                    edge_a[1].distance_to(edge_b[1]) < threshold):
                    return True

        return False

    # =========================================================================
    # Runtime Operations
    # =========================================================================

    def add_polygon(self, vertices: List[Vector3], area_type: int = 1) -> int:
        """Add a polygon manually (for dynamic NavMesh)."""
        polygon = NavMeshPolygon(
            id=self._next_polygon_id,
            vertices=vertices,
            area_type=area_type
        )
        self._polygons[polygon.id] = polygon
        self._next_polygon_id += 1
        return polygon.id

    def remove_polygon(self, polygon_id: int) -> bool:
        """Remove a polygon from the NavMesh."""
        if polygon_id not in self._polygons:
            return False

        # Remove from neighbors
        polygon = self._polygons[polygon_id]
        for neighbor_id in polygon.neighbors:
            if neighbor_id in self._polygons:
                neighbors = self._polygons[neighbor_id].neighbors
                if polygon_id in neighbors:
                    neighbors.remove(polygon_id)

        del self._polygons[polygon_id]
        return True

    def add_obstacle(self, obstacle: NavMeshObstacle) -> int:
        """Add a dynamic obstacle."""
        obstacle.id = self._next_obstacle_id
        self._obstacles[obstacle.id] = obstacle
        self._next_obstacle_id += 1

        if obstacle.carve and self.params.build_mode in (
            NavMeshBuildMode.DYNAMIC, NavMeshBuildMode.HYBRID
        ):
            self._mark_dirty_tiles_for_obstacle(obstacle)

        return obstacle.id

    def remove_obstacle(self, obstacle_id: int) -> bool:
        """Remove a dynamic obstacle."""
        if obstacle_id not in self._obstacles:
            return False

        obstacle = self._obstacles[obstacle_id]
        if obstacle.carve and self.params.build_mode in (
            NavMeshBuildMode.DYNAMIC, NavMeshBuildMode.HYBRID
        ):
            self._mark_dirty_tiles_for_obstacle(obstacle)

        del self._obstacles[obstacle_id]
        return True

    def update_obstacle(
        self, obstacle_id: int, position: Optional[Vector3] = None,
        rotation: Optional[float] = None
    ) -> bool:
        """Update an obstacle's position or rotation."""
        if obstacle_id not in self._obstacles:
            return False

        obstacle = self._obstacles[obstacle_id]
        if obstacle.carve:
            self._mark_dirty_tiles_for_obstacle(obstacle)

        if position is not None:
            obstacle.position = position
        if rotation is not None:
            obstacle.rotation = rotation

        if obstacle.carve:
            self._mark_dirty_tiles_for_obstacle(obstacle)

        return True

    def _mark_dirty_tiles_for_obstacle(self, obstacle: NavMeshObstacle) -> None:
        """Mark tiles affected by obstacle as dirty."""
        bounds = obstacle.get_bounds()
        min_tx = int((bounds.min_point.x - self._bounds.min_point.x) / self.params.tile_size)
        max_tx = int((bounds.max_point.x - self._bounds.min_point.x) / self.params.tile_size)
        min_tz = int((bounds.min_point.z - self._bounds.min_point.z) / self.params.tile_size)
        max_tz = int((bounds.max_point.z - self._bounds.min_point.z) / self.params.tile_size)

        for tx in range(min_tx, max_tx + 1):
            for tz in range(min_tz, max_tz + 1):
                self._dirty_tiles.add((tx, tz))

    def update(self) -> None:
        """Update NavMesh (rebuild dirty tiles)."""
        if not self._dirty_tiles:
            return

        for tile_coord in list(self._dirty_tiles):
            self._rebuild_tile(tile_coord)

        self._dirty_tiles.clear()

    def _rebuild_tile(self, tile_coord: Tuple[int, int]) -> None:
        """
        Rebuild a single tile.

        This method regenerates the navmesh polygons for a specific tile,
        applying any obstacle carvings and reconnecting adjacency.
        """
        tile = self._tiles.get(tile_coord)
        if tile is None:
            return

        # Get tile bounds
        tile_min_x = tile_coord[0] * self._config.tile_size
        tile_min_z = tile_coord[1] * self._config.tile_size
        tile_max_x = tile_min_x + self._config.tile_size
        tile_max_z = tile_min_z + self._config.tile_size

        # Remove old polygons in this tile
        polys_to_remove = []
        for poly_id, polygon in self._polygons.items():
            center = polygon.center
            if (tile_min_x <= center.x < tile_max_x and
                    tile_min_z <= center.z < tile_max_z):
                polys_to_remove.append(poly_id)

        for poly_id in polys_to_remove:
            # Remove adjacency references
            polygon = self._polygons[poly_id]
            for adj_id in polygon.adjacent:
                adj_poly = self._polygons.get(adj_id)
                if adj_poly and poly_id in adj_poly.adjacent:
                    adj_poly.adjacent.remove(poly_id)
            del self._polygons[poly_id]

        # Get obstacles affecting this tile
        tile_obstacles = []
        for obs in self._obstacles.values():
            if obs.bounds.intersects(BoundingBox(
                min_point=Vector3(tile_min_x, float('-inf'), tile_min_z),
                max_point=Vector3(tile_max_x, float('inf'), tile_max_z)
            )):
                tile_obstacles.append(obs)

        # Regenerate polygons for tile (simplified: create grid-based polygons)
        # Full implementation would use voxelization and contour generation
        if not tile_obstacles:
            # No obstacles - create simple quad covering the tile
            cell_size = self._config.cell_size
            for x in range(int((tile_max_x - tile_min_x) / cell_size)):
                for z in range(int((tile_max_z - tile_min_z) / cell_size)):
                    self._next_polygon_id += 1
                    poly_min_x = tile_min_x + x * cell_size
                    poly_min_z = tile_min_z + z * cell_size
                    vertices = [
                        Vector3(poly_min_x, 0, poly_min_z),
                        Vector3(poly_min_x + cell_size, 0, poly_min_z),
                        Vector3(poly_min_x + cell_size, 0, poly_min_z + cell_size),
                        Vector3(poly_min_x, 0, poly_min_z + cell_size),
                    ]
                    polygon = NavMeshPolygon(
                        id=self._next_polygon_id,
                        vertices=vertices,
                        center=Vector3(
                            poly_min_x + cell_size / 2,
                            0,
                            poly_min_z + cell_size / 2
                        ),
                        flags=0
                    )
                    self._polygons[polygon.id] = polygon

        # Rebuild adjacency for new polygons
        self._rebuild_adjacency_for_tile(tile_coord)

    def _rebuild_adjacency_for_tile(self, tile_coord: Tuple[int, int]) -> None:
        """
        Rebuild adjacency connections for polygons in a tile.

        Checks for shared edges between polygons in this tile and neighboring tiles.
        """
        tile_min_x = tile_coord[0] * self._config.tile_size
        tile_min_z = tile_coord[1] * self._config.tile_size
        tile_max_x = tile_min_x + self._config.tile_size
        tile_max_z = tile_min_z + self._config.tile_size

        # Get polygons in this tile
        tile_polys = []
        for poly_id, polygon in self._polygons.items():
            center = polygon.center
            if (tile_min_x <= center.x < tile_max_x and
                    tile_min_z <= center.z < tile_max_z):
                tile_polys.append(polygon)

        # Check adjacency with all other polygons
        edge_threshold = self._config.cell_size * 0.1

        for tile_poly in tile_polys:
            for poly_id, other_poly in self._polygons.items():
                if tile_poly.id == poly_id:
                    continue
                if poly_id in tile_poly.adjacent:
                    continue

                # Check if polygons share an edge
                if self._polygons_share_edge(tile_poly, other_poly, edge_threshold):
                    tile_poly.adjacent.append(poly_id)
                    other_poly.adjacent.append(tile_poly.id)

    def _polygons_share_edge(
        self, poly_a: NavMeshPolygon, poly_b: NavMeshPolygon, threshold: float
    ) -> bool:
        """Check if two polygons share an edge within threshold distance."""
        shared_count = 0
        for va in poly_a.vertices:
            for vb in poly_b.vertices:
                if va.distance_to(vb) < threshold:
                    shared_count += 1
                    if shared_count >= 2:
                        return True
                    break
        return False

    # =========================================================================
    # Queries
    # =========================================================================

    def find_nearest_point(self, position: Vector3, search_radius: float = 10.0) -> NavMeshQueryResult:
        """Find nearest point on NavMesh to given position."""
        result = NavMeshQueryResult(query_type=QueryType.NEAREST_POINT)

        nearest_dist_sq = search_radius * search_radius
        nearest_point: Optional[Vector3] = None
        nearest_poly_id: Optional[int] = None

        for poly_id, polygon in self._polygons.items():
            # Quick bounds check
            center_dist_sq = polygon.center.distance_squared_to(position)
            if center_dist_sq > nearest_dist_sq * 4:  # Conservative early out
                continue

            # Check if point is inside polygon
            if polygon.contains_point_2d(position):
                # Project to polygon surface
                projected = self._project_to_polygon(position, polygon)
                dist_sq = projected.distance_squared_to(position)
                if dist_sq < nearest_dist_sq:
                    nearest_dist_sq = dist_sq
                    nearest_point = projected
                    nearest_poly_id = poly_id
            else:
                # Find nearest point on polygon edges
                edge_point = self._nearest_point_on_polygon_edge(position, polygon)
                dist_sq = edge_point.distance_squared_to(position)
                if dist_sq < nearest_dist_sq:
                    nearest_dist_sq = dist_sq
                    nearest_point = edge_point
                    nearest_poly_id = poly_id

        if nearest_point is not None:
            result.success = True
            result.position = nearest_point
            result.polygon_id = nearest_poly_id
            result.distance = math.sqrt(nearest_dist_sq)

        return result

    def _project_to_polygon(self, point: Vector3, polygon: NavMeshPolygon) -> Vector3:
        """Project point onto polygon surface."""
        if not polygon.vertices:
            return point

        # Simple projection - use average height of polygon vertices
        avg_y = sum(v.y for v in polygon.vertices) / len(polygon.vertices)
        return Vector3(point.x, avg_y, point.z)

    def _nearest_point_on_polygon_edge(self, point: Vector3, polygon: NavMeshPolygon) -> Vector3:
        """Find nearest point on polygon edges."""
        nearest_point = polygon.center
        nearest_dist_sq = float('inf')

        for i in range(len(polygon.vertices)):
            edge_start, edge_end = polygon.get_edge(i)
            edge_point = self._nearest_point_on_segment(point, edge_start, edge_end)
            dist_sq = edge_point.distance_squared_to(point)
            if dist_sq < nearest_dist_sq:
                nearest_dist_sq = dist_sq
                nearest_point = edge_point

        return nearest_point

    def _nearest_point_on_segment(
        self, point: Vector3, seg_start: Vector3, seg_end: Vector3
    ) -> Vector3:
        """Find nearest point on line segment."""
        segment = seg_end - seg_start
        seg_len_sq = segment.length_squared()

        if seg_len_sq < ZERO_LENGTH_THRESHOLD:
            return seg_start

        t = max(0, min(1, (point - seg_start).dot(segment) / seg_len_sq))
        return seg_start + segment * t

    def raycast(
        self, start: Vector3, end: Vector3, filter_flags: int = 0
    ) -> RaycastResult:
        """Cast ray through NavMesh."""
        result = RaycastResult()
        direction = end - start
        max_distance = direction.length()

        if max_distance < ZERO_LENGTH_THRESHOLD:
            return result

        direction = direction.normalized()

        for polygon in self._polygons.values():
            if filter_flags != 0 and (polygon.flags & filter_flags) == 0:
                continue

            hit_result = self._raycast_polygon(start, direction, max_distance, polygon)
            if hit_result.hit and hit_result.distance < result.distance:
                result = hit_result

        return result

    def _raycast_polygon(
        self, origin: Vector3, direction: Vector3, max_dist: float, polygon: NavMeshPolygon
    ) -> RaycastResult:
        """Raycast against a single polygon."""
        result = RaycastResult()

        if len(polygon.vertices) < 3:
            return result

        # Calculate polygon plane
        v0, v1, v2 = polygon.vertices[0], polygon.vertices[1], polygon.vertices[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = edge1.cross(edge2).normalized()

        # Check if ray is parallel to plane
        denom = normal.dot(direction)
        if abs(denom) < FLOAT_EPSILON:
            return result

        # Calculate intersection distance
        t = (v0 - origin).dot(normal) / denom
        if t < 0 or t > max_dist:
            return result

        # Calculate intersection point
        hit_point = origin + direction * t

        # Check if point is inside polygon
        if polygon.contains_point_2d(hit_point):
            result.hit = True
            result.position = hit_point
            result.normal = normal if denom < 0 else -normal
            result.distance = t
            result.polygon_id = polygon.id

        return result

    def find_polygon_at(self, position: Vector3) -> Optional[int]:
        """Find polygon containing position."""
        for poly_id, polygon in self._polygons.items():
            if polygon.contains_point_2d(position):
                # Verify height
                projected = self._project_to_polygon(position, polygon)
                if abs(projected.y - position.y) < self.params.agent_height:
                    return poly_id
        return None

    def get_random_point(self) -> NavMeshQueryResult:
        """Get random point on NavMesh."""
        result = NavMeshQueryResult(query_type=QueryType.RANDOM_POINT)

        if not self._polygons:
            return result

        # Choose random polygon weighted by area (simplified)
        import random
        polygon = random.choice(list(self._polygons.values()))

        if len(polygon.vertices) >= 3:
            # Random point inside polygon (barycentric)
            r1 = random.random()
            r2 = random.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2

            v0 = polygon.vertices[0]
            v1 = polygon.vertices[1]
            v2 = polygon.vertices[2]

            point = v0 + (v1 - v0) * r1 + (v2 - v0) * r2
            result.success = True
            result.position = point
            result.polygon_id = polygon.id

        return result

    def get_random_point_in_radius(
        self, center: Vector3, radius: float
    ) -> NavMeshQueryResult:
        """Get random point on NavMesh within radius of center."""
        result = NavMeshQueryResult(query_type=QueryType.RANDOM_POINT)

        # Gather polygons within radius
        candidates = []
        for poly_id, polygon in self._polygons.items():
            if polygon.center.distance_to(center) < radius + self._get_polygon_radius(polygon):
                candidates.append(polygon)

        if not candidates:
            return result

        import random

        # Try several times to find valid point
        for _ in range(10):
            polygon = random.choice(candidates)
            random_result = self.get_random_point()

            if random_result.success and random_result.position:
                if random_result.position.distance_to(center) <= radius:
                    return random_result

        return result

    def _get_polygon_radius(self, polygon: NavMeshPolygon) -> float:
        """Get approximate radius of polygon from center."""
        if not polygon.vertices:
            return 0
        return max(v.distance_to(polygon.center) for v in polygon.vertices)

    def polygon_query(
        self, center: Vector3, half_extents: Vector3, filter_flags: int = 0
    ) -> List[int]:
        """Find polygons overlapping with box."""
        result = []
        query_box = BoundingBox(center - half_extents, center + half_extents)

        for poly_id, polygon in self._polygons.items():
            if filter_flags != 0 and (polygon.flags & filter_flags) == 0:
                continue

            # Check if polygon bounds intersect query box
            poly_bounds = self._get_polygon_bounds(polygon)
            if poly_bounds.intersects(query_box):
                result.append(poly_id)

        return result

    def _get_polygon_bounds(self, polygon: NavMeshPolygon) -> BoundingBox:
        """Get bounding box of polygon."""
        if not polygon.vertices:
            return BoundingBox()

        min_p = Vector3(float('inf'), float('inf'), float('inf'))
        max_p = Vector3(float('-inf'), float('-inf'), float('-inf'))

        for v in polygon.vertices:
            min_p = Vector3(min(min_p.x, v.x), min(min_p.y, v.y), min(min_p.z, v.z))
            max_p = Vector3(max(max_p.x, v.x), max(max_p.y, v.y), max(max_p.z, v.z))

        return BoundingBox(min_p, max_p)

    # =========================================================================
    # Path Modification
    # =========================================================================

    def smooth_path(
        self, path: List[Vector3],
        iterations: int = DEFAULT_SMOOTH_ITERATIONS,
        factor: float = DEFAULT_SMOOTH_FACTOR
    ) -> List[Vector3]:
        """Apply path smoothing using Chaikin's algorithm variant."""
        if len(path) < 3:
            return path

        result = path.copy()

        for _ in range(iterations):
            new_path = [result[0]]

            for i in range(len(result) - 1):
                p0 = result[i]
                p1 = result[i + 1]

                # Add two intermediate points
                q = p0.lerp(p1, 0.25 * factor)
                r = p0.lerp(p1, 0.75 * factor)

                new_path.append(q)
                new_path.append(r)

            new_path.append(result[-1])
            result = new_path

        return result

    def funnel_path(
        self, path: List[Vector3], portal_left: List[Vector3],
        portal_right: List[Vector3], epsilon: float = DEFAULT_FUNNEL_EPSILON
    ) -> List[Vector3]:
        """Apply funnel algorithm (string-pulling) to path."""
        if len(path) < 2:
            return path

        result = [path[0]]
        apex = path[0]
        apex_index = 0

        left = portal_left[0] if portal_left else path[0]
        right = portal_right[0] if portal_right else path[0]
        left_index = 0
        right_index = 0

        for i in range(1, len(path)):
            new_left = portal_left[i] if i < len(portal_left) else path[i]
            new_right = portal_right[i] if i < len(portal_right) else path[i]

            # Update right vertex
            if self._triangle_area_2d(apex, right, new_right) <= 0:
                if apex == right or self._triangle_area_2d(apex, left, new_right) > 0:
                    right = new_right
                    right_index = i
                else:
                    result.append(left)
                    apex = left
                    apex_index = left_index
                    left = apex
                    right = apex
                    left_index = apex_index
                    right_index = apex_index
                    i = apex_index
                    continue

            # Update left vertex
            if self._triangle_area_2d(apex, left, new_left) >= 0:
                if apex == left or self._triangle_area_2d(apex, right, new_left) < 0:
                    left = new_left
                    left_index = i
                else:
                    result.append(right)
                    apex = right
                    apex_index = right_index
                    left = apex
                    right = apex
                    left_index = apex_index
                    right_index = apex_index
                    i = apex_index
                    continue

        result.append(path[-1])
        return result

    def _triangle_area_2d(self, a: Vector3, b: Vector3, c: Vector3) -> float:
        """Calculate signed area of triangle in XZ plane."""
        return (c.x - a.x) * (b.z - a.z) - (b.x - a.x) * (c.z - a.z)

    def adjust_corridor_width(
        self, path: List[Vector3], width: float = DEFAULT_CORRIDOR_WIDTH
    ) -> List[Vector3]:
        """Adjust path to maintain minimum corridor width."""
        if len(path) < 2 or width <= 0:
            return path

        result = [path[0]]

        for i in range(1, len(path) - 1):
            prev_point = path[i - 1]
            curr_point = path[i]
            next_point = path[i + 1]

            # Calculate direction
            dir_prev = (curr_point - prev_point).normalized()
            dir_next = (next_point - curr_point).normalized()

            # Calculate perpendicular offset for corridor width
            perp = Vector3(-dir_prev.z, 0, dir_prev.x)

            # Check if there's enough clearance
            left_clear = self._check_clearance(curr_point, perp, width / 2)
            right_clear = self._check_clearance(curr_point, -perp, width / 2)

            if left_clear and right_clear:
                result.append(curr_point)
            else:
                # Adjust point toward clearer side
                if left_clear:
                    result.append(curr_point + perp * (width / 4))
                elif right_clear:
                    result.append(curr_point - perp * (width / 4))
                else:
                    result.append(curr_point)

        result.append(path[-1])
        return result

    def _check_clearance(self, point: Vector3, direction: Vector3, distance: float) -> bool:
        """Check if there's clearance in given direction."""
        test_point = point + direction * distance
        poly_id = self.find_polygon_at(test_point)
        return poly_id is not None

    # =========================================================================
    # Accessors
    # =========================================================================

    @property
    def is_built(self) -> bool:
        """Check if NavMesh has been built."""
        return self._is_built

    @property
    def polygon_count(self) -> int:
        """Get total number of polygons."""
        return len(self._polygons)

    @property
    def obstacle_count(self) -> int:
        """Get total number of obstacles."""
        return len(self._obstacles)

    @property
    def bounds(self) -> BoundingBox:
        """Get NavMesh bounding box."""
        return self._bounds

    def get_polygon(self, polygon_id: int) -> Optional[NavMeshPolygon]:
        """Get polygon by ID."""
        return self._polygons.get(polygon_id)

    def get_polygons(self) -> Iterator[NavMeshPolygon]:
        """Iterate over all polygons."""
        return iter(self._polygons.values())

    def get_obstacle(self, obstacle_id: int) -> Optional[NavMeshObstacle]:
        """Get obstacle by ID."""
        return self._obstacles.get(obstacle_id)

    def get_obstacles(self) -> Iterator[NavMeshObstacle]:
        """Iterate over all obstacles."""
        return iter(self._obstacles.values())

    def get_neighbors(self, polygon_id: int) -> List[int]:
        """Get neighbor polygon IDs."""
        polygon = self._polygons.get(polygon_id)
        if polygon is None:
            return []
        return polygon.neighbors.copy()
