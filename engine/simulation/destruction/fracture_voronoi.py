"""
Voronoi Fracture System.

Implements Voronoi cell-based mesh fracturing for realistic destruction effects.
Generates natural-looking fracture patterns by partitioning mesh volume into
cells defined by randomly placed Voronoi sites.
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Callable
from enum import IntEnum

from .config import (
    DEFAULT_FRACTURE_SEED,
    MIN_CHUNK_VOLUME,
    MAX_CHUNKS_PER_OBJECT,
    MIN_VORONOI_SITES,
    MAX_VORONOI_SITES,
    DEFAULT_VORONOI_SITES,
    SURFACE_SAMPLE_ITERATIONS,
    DEGENERATE_TRIANGLE_AREA_THRESHOLD,
)


# Type aliases
Vec3 = Tuple[float, float, float]
Triangle = Tuple[int, int, int]


def vec3_add(a: Vec3, b: Vec3) -> Vec3:
    """Add two vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_sub(a: Vec3, b: Vec3) -> Vec3:
    """Subtract two vectors."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_mul(a: Vec3, s: float) -> Vec3:
    """Multiply vector by scalar."""
    return (a[0] * s, a[1] * s, a[2] * s)


def vec3_dot(a: Vec3, b: Vec3) -> float:
    """Dot product of two vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product of two vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )


def vec3_length(v: Vec3) -> float:
    """Length of a vector."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec3_normalize(v: Vec3) -> Vec3:
    """Normalize a vector."""
    length = vec3_length(v)
    if length < 1e-10:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def vec3_distance(a: Vec3, b: Vec3) -> float:
    """Distance between two points."""
    return vec3_length(vec3_sub(a, b))


def vec3_lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    """Linear interpolation between two vectors."""
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t
    )


def triangle_area(v0: Vec3, v1: Vec3, v2: Vec3) -> float:
    """
    Calculate the area of a triangle from three vertices.

    Uses the cross product method: area = 0.5 * |v1-v0 x v2-v0|

    Args:
        v0, v1, v2: Triangle vertices.

    Returns:
        Area of the triangle.
    """
    edge1 = vec3_sub(v1, v0)
    edge2 = vec3_sub(v2, v0)
    cross = vec3_cross(edge1, edge2)
    return 0.5 * vec3_length(cross)


def is_degenerate_triangle(v0: Vec3, v1: Vec3, v2: Vec3, threshold: float = DEGENERATE_TRIANGLE_AREA_THRESHOLD) -> bool:
    """
    Check if a triangle is degenerate (has near-zero area).

    Args:
        v0, v1, v2: Triangle vertices.
        threshold: Minimum area threshold.

    Returns:
        True if triangle is degenerate.
    """
    return triangle_area(v0, v1, v2) < threshold


@dataclass(slots=True)
class Plane:
    """
    3D plane defined by point and normal.

    Attributes:
        point: A point on the plane.
        normal: Unit normal vector of the plane.
    """
    point: Vec3
    normal: Vec3

    def __post_init__(self) -> None:
        self.normal = vec3_normalize(self.normal)

    def signed_distance(self, p: Vec3) -> float:
        """
        Calculate signed distance from point to plane.

        Positive = in front of plane (normal side)
        Negative = behind plane
        Zero = on plane
        """
        return vec3_dot(vec3_sub(p, self.point), self.normal)

    def classify_point(self, p: Vec3, epsilon: float = 1e-6) -> int:
        """
        Classify point relative to plane.

        Returns:
            1 = front, -1 = back, 0 = on plane
        """
        d = self.signed_distance(p)
        if d > epsilon:
            return 1
        elif d < -epsilon:
            return -1
        return 0

    @classmethod
    def from_three_points(cls, p0: Vec3, p1: Vec3, p2: Vec3) -> Plane:
        """Create plane from three non-collinear points."""
        v1 = vec3_sub(p1, p0)
        v2 = vec3_sub(p2, p0)
        normal = vec3_normalize(vec3_cross(v1, v2))
        return cls(point=p0, normal=normal)

    @classmethod
    def bisector(cls, p1: Vec3, p2: Vec3) -> Plane:
        """Create bisector plane between two points."""
        midpoint = vec3_lerp(p1, p2, 0.5)
        normal = vec3_normalize(vec3_sub(p2, p1))
        return cls(point=midpoint, normal=normal)


@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned bounding box."""
    min_point: Vec3
    max_point: Vec3

    @property
    def center(self) -> Vec3:
        """Center of the bounding box."""
        return vec3_lerp(self.min_point, self.max_point, 0.5)

    @property
    def size(self) -> Vec3:
        """Size (dimensions) of the bounding box."""
        return vec3_sub(self.max_point, self.min_point)

    @property
    def volume(self) -> float:
        """Volume of the bounding box."""
        s = self.size
        return s[0] * s[1] * s[2]

    def contains(self, point: Vec3) -> bool:
        """Check if point is inside the bounding box."""
        return (
            self.min_point[0] <= point[0] <= self.max_point[0] and
            self.min_point[1] <= point[1] <= self.max_point[1] and
            self.min_point[2] <= point[2] <= self.max_point[2]
        )

    def random_point(self, rng: random.Random) -> Vec3:
        """Generate a random point inside the bounding box."""
        return (
            rng.uniform(self.min_point[0], self.max_point[0]),
            rng.uniform(self.min_point[1], self.max_point[1]),
            rng.uniform(self.min_point[2], self.max_point[2])
        )

    def expand(self, amount: float) -> BoundingBox:
        """Return expanded bounding box."""
        return BoundingBox(
            min_point=vec3_sub(self.min_point, (amount, amount, amount)),
            max_point=vec3_add(self.max_point, (amount, amount, amount))
        )

    @classmethod
    def from_points(cls, points: List[Vec3]) -> BoundingBox:
        """Create bounding box from list of points."""
        if not points:
            return cls(min_point=(0, 0, 0), max_point=(0, 0, 0))

        min_p = list(points[0])
        max_p = list(points[0])

        for p in points[1:]:
            for i in range(3):
                min_p[i] = min(min_p[i], p[i])
                max_p[i] = max(max_p[i], p[i])

        return cls(min_point=tuple(min_p), max_point=tuple(max_p))


@dataclass(slots=True)
class Chunk:
    """
    Represents a fractured chunk of mesh.

    Attributes:
        vertices: List of vertex positions.
        triangles: List of triangle indices.
        volume: Computed volume of the chunk.
        centroid: Center of mass of the chunk.
        is_interior: Whether chunk contains interior (cut) faces.
        cell_index: Index of the Voronoi cell this chunk belongs to.
        adjacent_chunks: Set of adjacent chunk indices.
    """
    vertices: List[Vec3]
    triangles: List[Triangle]
    volume: float = 0.0
    centroid: Vec3 = (0.0, 0.0, 0.0)
    is_interior: bool = False
    cell_index: int = -1
    adjacent_chunks: Set[int] = field(default_factory=set)

    def compute_volume(self) -> float:
        """
        Compute the volume of the chunk using signed tetrahedron method.

        Returns:
            Volume of the chunk (always positive).
        """
        if len(self.vertices) < 4 or len(self.triangles) < 1:
            self.volume = 0.0
            return 0.0

        total_volume = 0.0

        for tri in self.triangles:
            v0 = self.vertices[tri[0]]
            v1 = self.vertices[tri[1]]
            v2 = self.vertices[tri[2]]

            # Signed volume of tetrahedron with origin
            volume = vec3_dot(v0, vec3_cross(v1, v2)) / 6.0
            total_volume += volume

        self.volume = abs(total_volume)
        return self.volume

    def compute_centroid(self) -> Vec3:
        """
        Compute the centroid (center of mass) of the chunk.

        Returns:
            Centroid position.
        """
        if not self.vertices:
            self.centroid = (0.0, 0.0, 0.0)
            return self.centroid

        cx = sum(v[0] for v in self.vertices) / len(self.vertices)
        cy = sum(v[1] for v in self.vertices) / len(self.vertices)
        cz = sum(v[2] for v in self.vertices) / len(self.vertices)

        self.centroid = (cx, cy, cz)
        return self.centroid

    def get_bounds(self) -> BoundingBox:
        """Get bounding box of the chunk."""
        return BoundingBox.from_points(self.vertices)

    def is_valid(self, min_volume: float = MIN_CHUNK_VOLUME) -> bool:
        """Check if chunk is valid (has sufficient volume)."""
        if self.volume <= 0:
            self.compute_volume()
        return self.volume >= min_volume and len(self.triangles) >= 4


@dataclass(slots=True)
class VoronoiCell:
    """
    Represents a single Voronoi cell.

    Attributes:
        site: The Voronoi site (seed point) for this cell.
        index: Index of this cell.
        planes: Bounding planes that define the cell.
        neighbors: Set of neighboring cell indices.
    """
    site: Vec3
    index: int
    planes: List[Plane] = field(default_factory=list)
    neighbors: Set[int] = field(default_factory=set)

    def contains_point(self, point: Vec3) -> bool:
        """Check if a point is inside this Voronoi cell."""
        for plane in self.planes:
            if plane.signed_distance(point) > 1e-6:
                return False
        return True


class SiteDistribution(IntEnum):
    """Distribution patterns for Voronoi sites."""
    UNIFORM = 0
    CLUSTERED = 1
    SURFACE_BIASED = 2
    IMPACT_CENTERED = 3


class VoronoiFracture:
    """
    Voronoi-based mesh fracturing system.

    Generates fracture patterns by computing Voronoi cells within a mesh
    and clipping the mesh geometry to each cell.
    """

    __slots__ = (
        '_seed', '_rng', '_num_sites', '_sites', '_cells',
        '_bounds', '_min_chunk_volume', '_max_chunks'
    )

    def __init__(
        self,
        seed: int = DEFAULT_FRACTURE_SEED,
        num_sites: int = DEFAULT_VORONOI_SITES,
        min_chunk_volume: float = MIN_CHUNK_VOLUME,
        max_chunks: int = MAX_CHUNKS_PER_OBJECT
    ) -> None:
        """
        Initialize Voronoi fracture generator.

        Args:
            seed: Random seed for deterministic generation.
            num_sites: Number of Voronoi sites to generate.
            min_chunk_volume: Minimum volume for valid chunks.
            max_chunks: Maximum number of chunks to generate.
        """
        self._seed = seed
        self._rng = random.Random(seed)
        self._num_sites = max(MIN_VORONOI_SITES, min(num_sites, MAX_VORONOI_SITES))
        self._sites: List[Vec3] = []
        self._cells: List[VoronoiCell] = []
        self._bounds: Optional[BoundingBox] = None
        self._min_chunk_volume = min_chunk_volume
        self._max_chunks = max_chunks

    @property
    def seed(self) -> int:
        """Current random seed."""
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        """Set random seed and reset RNG."""
        self._seed = value
        self._rng = random.Random(value)

    @property
    def num_sites(self) -> int:
        """Number of Voronoi sites."""
        return self._num_sites

    @property
    def sites(self) -> List[Vec3]:
        """List of Voronoi sites."""
        return self._sites.copy()

    @property
    def cells(self) -> List[VoronoiCell]:
        """List of computed Voronoi cells."""
        return self._cells

    def generate_voronoi_sites(
        self,
        bounds: BoundingBox,
        distribution: SiteDistribution = SiteDistribution.UNIFORM,
        impact_point: Optional[Vec3] = None,
        surface_points: Optional[List[Vec3]] = None
    ) -> List[Vec3]:
        """
        Generate Voronoi sites within the given bounds.

        Args:
            bounds: Bounding box to generate sites within.
            distribution: Site distribution pattern.
            impact_point: Optional impact point for IMPACT_CENTERED distribution.
            surface_points: Optional surface points for SURFACE_BIASED distribution.

        Returns:
            List of generated site positions.
        """
        self._bounds = bounds
        self._sites = []

        if distribution == SiteDistribution.UNIFORM:
            self._sites = self._generate_uniform_sites(bounds)
        elif distribution == SiteDistribution.CLUSTERED:
            self._sites = self._generate_clustered_sites(bounds)
        elif distribution == SiteDistribution.SURFACE_BIASED:
            self._sites = self._generate_surface_biased_sites(bounds, surface_points)
        elif distribution == SiteDistribution.IMPACT_CENTERED:
            self._sites = self._generate_impact_centered_sites(bounds, impact_point)

        return self._sites

    def _generate_uniform_sites(self, bounds: BoundingBox) -> List[Vec3]:
        """Generate uniformly distributed sites."""
        sites = []
        for _ in range(self._num_sites):
            sites.append(bounds.random_point(self._rng))
        return sites

    def _generate_clustered_sites(self, bounds: BoundingBox) -> List[Vec3]:
        """Generate sites clustered around random centers."""
        sites = []
        num_clusters = max(2, self._num_sites // 4)
        cluster_centers = [bounds.random_point(self._rng) for _ in range(num_clusters)]

        size = bounds.size
        cluster_radius = min(size) * 0.2

        for _ in range(self._num_sites):
            # Pick a random cluster center
            center = self._rng.choice(cluster_centers)

            # Generate point near cluster center
            offset = (
                self._rng.gauss(0, cluster_radius),
                self._rng.gauss(0, cluster_radius),
                self._rng.gauss(0, cluster_radius)
            )
            point = vec3_add(center, offset)

            # Clamp to bounds
            point = (
                max(bounds.min_point[0], min(bounds.max_point[0], point[0])),
                max(bounds.min_point[1], min(bounds.max_point[1], point[1])),
                max(bounds.min_point[2], min(bounds.max_point[2], point[2]))
            )
            sites.append(point)

        return sites

    def _generate_surface_biased_sites(
        self,
        bounds: BoundingBox,
        surface_points: Optional[List[Vec3]] = None
    ) -> List[Vec3]:
        """Generate sites biased toward surface of the mesh."""
        sites = []

        # If no surface points provided, use bounds corners and edges
        if surface_points is None:
            surface_points = self._generate_bounds_surface_points(bounds)

        num_surface = self._num_sites * 2 // 3
        num_interior = self._num_sites - num_surface

        # Surface-biased sites
        for _ in range(num_surface):
            if surface_points:
                base = self._rng.choice(surface_points)
                # Add small random offset
                offset = (
                    self._rng.gauss(0, 0.1),
                    self._rng.gauss(0, 0.1),
                    self._rng.gauss(0, 0.1)
                )
                point = vec3_add(base, offset)
                # Clamp to bounds
                point = (
                    max(bounds.min_point[0], min(bounds.max_point[0], point[0])),
                    max(bounds.min_point[1], min(bounds.max_point[1], point[1])),
                    max(bounds.min_point[2], min(bounds.max_point[2], point[2]))
                )
                sites.append(point)
            else:
                sites.append(bounds.random_point(self._rng))

        # Interior sites
        for _ in range(num_interior):
            sites.append(bounds.random_point(self._rng))

        return sites

    def _generate_impact_centered_sites(
        self,
        bounds: BoundingBox,
        impact_point: Optional[Vec3] = None
    ) -> List[Vec3]:
        """Generate sites concentrated around an impact point."""
        sites = []

        # Use center if no impact point provided
        if impact_point is None:
            impact_point = bounds.center

        size = bounds.size
        max_radius = vec3_length(size) / 2

        # Generate sites with distance-based density
        for _ in range(self._num_sites):
            # Exponential distribution - more sites near impact
            distance = self._rng.expovariate(3.0) * max_radius

            # Random direction
            theta = self._rng.uniform(0, 2 * math.pi)
            phi = self._rng.uniform(0, math.pi)

            direction = (
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi)
            )

            point = vec3_add(impact_point, vec3_mul(direction, distance))

            # Clamp to bounds
            if bounds.contains(point):
                sites.append(point)
            else:
                # Project back to bounds
                point = (
                    max(bounds.min_point[0], min(bounds.max_point[0], point[0])),
                    max(bounds.min_point[1], min(bounds.max_point[1], point[1])),
                    max(bounds.min_point[2], min(bounds.max_point[2], point[2]))
                )
                sites.append(point)

        return sites

    def _generate_bounds_surface_points(self, bounds: BoundingBox) -> List[Vec3]:
        """Generate surface sample points from bounding box."""
        points = []
        min_p = bounds.min_point
        max_p = bounds.max_point

        # Sample points on each face
        for _ in range(SURFACE_SAMPLE_ITERATIONS):
            # X faces
            points.append((min_p[0], self._rng.uniform(min_p[1], max_p[1]),
                          self._rng.uniform(min_p[2], max_p[2])))
            points.append((max_p[0], self._rng.uniform(min_p[1], max_p[1]),
                          self._rng.uniform(min_p[2], max_p[2])))
            # Y faces
            points.append((self._rng.uniform(min_p[0], max_p[0]), min_p[1],
                          self._rng.uniform(min_p[2], max_p[2])))
            points.append((self._rng.uniform(min_p[0], max_p[0]), max_p[1],
                          self._rng.uniform(min_p[2], max_p[2])))
            # Z faces
            points.append((self._rng.uniform(min_p[0], max_p[0]),
                          self._rng.uniform(min_p[1], max_p[1]), min_p[2]))
            points.append((self._rng.uniform(min_p[0], max_p[0]),
                          self._rng.uniform(min_p[1], max_p[1]), max_p[2]))

        return points

    def compute_voronoi_cells(self, sites: Optional[List[Vec3]] = None) -> List[VoronoiCell]:
        """
        Compute Voronoi cells from sites.

        Uses the dual of Delaunay triangulation approach, computing cells
        as intersections of half-spaces defined by bisector planes.

        Args:
            sites: Optional list of sites. Uses stored sites if not provided.

        Returns:
            List of computed Voronoi cells.
        """
        if sites is not None:
            self._sites = sites

        if not self._sites:
            return []

        self._cells = []

        # For each site, compute its Voronoi cell
        for i, site in enumerate(self._sites):
            cell = VoronoiCell(site=site, index=i)

            # Create bisector planes with all other sites
            for j, other_site in enumerate(self._sites):
                if i == j:
                    continue

                # Bisector plane between this site and other site
                plane = Plane.bisector(site, other_site)
                cell.planes.append(plane)
                cell.neighbors.add(j)

            # Add bounding box planes if bounds exist
            if self._bounds:
                min_p = self._bounds.min_point
                max_p = self._bounds.max_point

                cell.planes.extend([
                    Plane(point=(min_p[0], 0, 0), normal=(-1, 0, 0)),
                    Plane(point=(max_p[0], 0, 0), normal=(1, 0, 0)),
                    Plane(point=(0, min_p[1], 0), normal=(0, -1, 0)),
                    Plane(point=(0, max_p[1], 0), normal=(0, 1, 0)),
                    Plane(point=(0, 0, min_p[2]), normal=(0, 0, -1)),
                    Plane(point=(0, 0, max_p[2]), normal=(0, 0, 1)),
                ])

            self._cells.append(cell)

        return self._cells

    def mesh_to_chunks(
        self,
        vertices: List[Vec3],
        triangles: List[Triangle],
        cells: Optional[List[VoronoiCell]] = None
    ) -> List[Chunk]:
        """
        Convert a mesh to chunks by clipping against Voronoi cells.

        Args:
            vertices: Mesh vertex positions.
            triangles: Mesh triangle indices.
            cells: Optional Voronoi cells. Uses stored cells if not provided.

        Returns:
            List of mesh chunks.
        """
        if cells is not None:
            self._cells = cells

        if not self._cells:
            return []

        chunks = []

        for cell in self._cells:
            chunk = self._clip_mesh_to_cell(vertices, triangles, cell)

            if chunk is not None:
                chunk.compute_volume()
                chunk.compute_centroid()

                if chunk.is_valid(self._min_chunk_volume):
                    chunks.append(chunk)

                    if len(chunks) >= self._max_chunks:
                        break

        # Compute adjacency between chunks
        self._compute_chunk_adjacency(chunks)

        return chunks

    def _clip_mesh_to_cell(
        self,
        vertices: List[Vec3],
        triangles: List[Triangle],
        cell: VoronoiCell
    ) -> Optional[Chunk]:
        """
        Clip mesh geometry to a single Voronoi cell.

        Uses Sutherland-Hodgman algorithm extended to 3D.
        """
        # Start with all triangles
        current_tris = [(vertices[t[0]], vertices[t[1]], vertices[t[2]]) for t in triangles]

        # Clip against each plane
        for plane in cell.planes:
            if not current_tris:
                break

            next_tris = []

            for tri in current_tris:
                clipped = self._clip_triangle_to_plane(tri, plane)
                next_tris.extend(clipped)

            current_tris = next_tris

        if not current_tris:
            return None

        # Convert back to indexed mesh
        chunk_vertices = []
        chunk_triangles = []
        vertex_map: Dict[Vec3, int] = {}

        for tri in current_tris:
            indices = []
            for v in tri:
                # Quantize for comparison
                v_key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))

                if v_key not in vertex_map:
                    vertex_map[v_key] = len(chunk_vertices)
                    chunk_vertices.append(v)

                indices.append(vertex_map[v_key])

            # Validate triangle: unique indices and non-degenerate geometry
            if len(set(indices)) == 3:
                tri_verts = [chunk_vertices[idx] for idx in indices]
                if not is_degenerate_triangle(tri_verts[0], tri_verts[1], tri_verts[2]):
                    chunk_triangles.append(tuple(indices))

        if not chunk_triangles:
            return None

        return Chunk(
            vertices=chunk_vertices,
            triangles=chunk_triangles,
            cell_index=cell.index,
            is_interior=True
        )

    def _clip_triangle_to_plane(
        self,
        tri: Tuple[Vec3, Vec3, Vec3],
        plane: Plane
    ) -> List[Tuple[Vec3, Vec3, Vec3]]:
        """
        Clip a triangle against a plane.

        Returns list of triangles that are on the negative side of the plane.
        Handles edge cases including degenerate triangles and edges on the plane.
        """
        v0, v1, v2 = tri

        # Skip degenerate input triangles
        if is_degenerate_triangle(v0, v1, v2):
            return []

        d0 = plane.signed_distance(v0)
        d1 = plane.signed_distance(v1)
        d2 = plane.signed_distance(v2)

        epsilon = 1e-6

        # Classify vertices
        pos_count = sum(1 for d in [d0, d1, d2] if d > epsilon)
        neg_count = sum(1 for d in [d0, d1, d2] if d < -epsilon)

        # All behind plane - keep triangle
        if pos_count == 0:
            return [tri]

        # All in front - discard triangle
        if neg_count == 0:
            return []

        # Mixed case - clip triangle
        result = []
        points = [v0, v1, v2]
        distances = [d0, d1, d2]

        # Collect output polygon vertices
        output = []

        for i in range(3):
            curr = points[i]
            next_p = points[(i + 1) % 3]
            curr_d = distances[i]
            next_d = distances[(i + 1) % 3]

            if curr_d <= epsilon:  # Current is behind or on plane
                output.append(curr)

            # Check for edge crossing
            if (curr_d > epsilon) != (next_d > epsilon):
                # Guard against division by zero when edge lies on plane
                denominator = curr_d - next_d
                if abs(denominator) > epsilon:
                    t = curr_d / denominator
                    # Clamp t to [0, 1] for numerical stability
                    t = max(0.0, min(1.0, t))
                    intersection = vec3_lerp(curr, next_p, t)
                    output.append(intersection)

        # Triangulate output polygon, filtering degenerate results
        if len(output) >= 3:
            for i in range(1, len(output) - 1):
                new_tri = (output[0], output[i], output[i + 1])
                # Only add non-degenerate triangles
                if not is_degenerate_triangle(new_tri[0], new_tri[1], new_tri[2]):
                    result.append(new_tri)

        return result

    def _compute_chunk_adjacency(self, chunks: List[Chunk]) -> None:
        """Compute which chunks are adjacent to each other."""
        for i, chunk_i in enumerate(chunks):
            for j, chunk_j in enumerate(chunks):
                if i >= j:
                    continue

                # Check if cells were neighbors
                cell_i = self._cells[chunk_i.cell_index] if chunk_i.cell_index < len(self._cells) else None
                cell_j = self._cells[chunk_j.cell_index] if chunk_j.cell_index < len(self._cells) else None

                if cell_i and cell_j:
                    if chunk_j.cell_index in cell_i.neighbors:
                        chunk_i.adjacent_chunks.add(j)
                        chunk_j.adjacent_chunks.add(i)

    def fracture(
        self,
        vertices: List[Vec3],
        triangles: List[Triangle],
        impact_point: Optional[Vec3] = None,
        distribution: SiteDistribution = SiteDistribution.UNIFORM
    ) -> List[Chunk]:
        """
        Perform complete fracture operation.

        Args:
            vertices: Mesh vertex positions.
            triangles: Mesh triangle indices.
            impact_point: Optional impact point for site distribution.
            distribution: Site distribution pattern.

        Returns:
            List of fractured chunks.
        """
        bounds = BoundingBox.from_points(vertices)
        self.generate_voronoi_sites(bounds, distribution, impact_point)
        self.compute_voronoi_cells()
        return self.mesh_to_chunks(vertices, triangles)


class TetrahedralVoronoiFracture(VoronoiFracture):
    """
    Voronoi fracture with tetrahedral mesh support.

    Extends base VoronoiFracture to handle volumetric tetrahedral meshes
    for more accurate internal fracture representation.
    """

    def fracture_tetrahedral(
        self,
        vertices: List[Vec3],
        tetrahedra: List[Tuple[int, int, int, int]],
        impact_point: Optional[Vec3] = None,
        distribution: SiteDistribution = SiteDistribution.UNIFORM
    ) -> List[Chunk]:
        """
        Fracture a tetrahedral mesh.

        Args:
            vertices: Mesh vertex positions.
            tetrahedra: List of tetrahedron vertex indices.
            impact_point: Optional impact point.
            distribution: Site distribution pattern.

        Returns:
            List of fractured chunks.
        """
        # Convert tetrahedra to surface triangles
        surface_tris = []
        face_count: Dict[Tuple[int, int, int], int] = {}

        for tet in tetrahedra:
            # Each tetrahedron has 4 faces
            faces = [
                tuple(sorted([tet[0], tet[1], tet[2]])),
                tuple(sorted([tet[0], tet[1], tet[3]])),
                tuple(sorted([tet[0], tet[2], tet[3]])),
                tuple(sorted([tet[1], tet[2], tet[3]])),
            ]

            for face in faces:
                face_count[face] = face_count.get(face, 0) + 1

        # Surface faces are those that appear only once
        for face, count in face_count.items():
            if count == 1:
                surface_tris.append(face)

        # Use base fracture with surface mesh
        return self.fracture(vertices, surface_tris, impact_point, distribution)

    def compute_tetrahedral_volume(
        self,
        vertices: List[Vec3],
        tetrahedra: List[Tuple[int, int, int, int]]
    ) -> float:
        """Compute total volume of tetrahedral mesh."""
        total_volume = 0.0

        for tet in tetrahedra:
            v0 = vertices[tet[0]]
            v1 = vertices[tet[1]]
            v2 = vertices[tet[2]]
            v3 = vertices[tet[3]]

            # Signed volume of tetrahedron
            d1 = vec3_sub(v1, v0)
            d2 = vec3_sub(v2, v0)
            d3 = vec3_sub(v3, v0)

            volume = abs(vec3_dot(d1, vec3_cross(d2, d3))) / 6.0
            total_volume += volume

        return total_volume
