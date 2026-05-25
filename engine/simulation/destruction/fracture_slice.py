"""
Planar Slice Fracture System.

Implements mesh slicing using arbitrary planes for controlled fracturing.
Supports single cuts, multiple parallel slices, and random slice plane generation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set

from .config import (
    DEFAULT_FRACTURE_SEED,
    MIN_CHUNK_VOLUME,
    MAX_CHUNKS_PER_OBJECT,
    SLICE_MAX_PLANES,
)
from .fracture_voronoi import (
    Vec3,
    Triangle,
    Plane,
    BoundingBox,
    Chunk,
    vec3_add,
    vec3_sub,
    vec3_mul,
    vec3_dot,
    vec3_cross,
    vec3_length,
    vec3_normalize,
    vec3_lerp,
    vec3_distance,
)


@dataclass(slots=True)
class SliceResult:
    """
    Result of a single slice operation.

    Attributes:
        front_chunk: Chunk on the positive side of the slice plane.
        back_chunk: Chunk on the negative side of the slice plane.
        plane: The slicing plane.
        cut_vertices: Vertices created along the cut.
    """
    front_chunk: Optional[Chunk]
    back_chunk: Optional[Chunk]
    plane: Plane
    cut_vertices: List[Vec3] = field(default_factory=list)


@dataclass(slots=True)
class CappedMesh:
    """
    Mesh data with capping information.

    Attributes:
        vertices: Vertex positions.
        triangles: Triangle indices.
        cap_vertices: Indices of vertices on the cap.
        cap_triangles: Indices of triangles forming the cap.
        is_front: Whether this is the front (positive side) mesh.
    """
    vertices: List[Vec3]
    triangles: List[Triangle]
    cap_vertices: List[int] = field(default_factory=list)
    cap_triangles: List[int] = field(default_factory=list)
    is_front: bool = True


class SliceFracture:
    """
    Planar slice-based mesh fracturing system.

    Cuts meshes along planes and generates capped surfaces for
    clean-looking fracture results.
    """

    __slots__ = (
        '_seed', '_rng', '_min_chunk_volume', '_max_chunks',
        '_generate_caps', '_cap_inset', '_planes'
    )

    def __init__(
        self,
        seed: int = DEFAULT_FRACTURE_SEED,
        min_chunk_volume: float = MIN_CHUNK_VOLUME,
        max_chunks: int = MAX_CHUNKS_PER_OBJECT,
        generate_caps: bool = True,
        cap_inset: float = 0.0
    ) -> None:
        """
        Initialize slice fracture generator.

        Args:
            seed: Random seed for deterministic generation.
            min_chunk_volume: Minimum volume for valid chunks.
            max_chunks: Maximum number of chunks to generate.
            generate_caps: Whether to generate cap surfaces.
            cap_inset: Inset amount for cap edges.
        """
        self._seed = seed
        self._rng = random.Random(seed)
        self._min_chunk_volume = min_chunk_volume
        self._max_chunks = max_chunks
        self._generate_caps = generate_caps
        self._cap_inset = cap_inset
        self._planes: List[Plane] = []

    @property
    def seed(self) -> int:
        """Current random seed."""
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        """Set random seed and reset RNG."""
        self._seed = value
        self._rng = random.Random(value)

    def slice_mesh(
        self,
        vertices: List[Vec3],
        triangles: List[Triangle],
        plane: Plane
    ) -> SliceResult:
        """
        Slice a mesh with a single plane.

        Args:
            vertices: Mesh vertex positions.
            triangles: Mesh triangle indices.
            plane: Slicing plane.

        Returns:
            SliceResult containing front and back chunks.
        """
        front_tris = []
        back_tris = []
        cut_edges: Dict[Tuple[int, int], Vec3] = {}

        # Process each triangle
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

            result = self._classify_and_clip_triangle(
                (v0, v1, v2),
                (tri[0], tri[1], tri[2]),
                plane,
                cut_edges
            )

            front_tris.extend(result[0])
            back_tris.extend(result[1])

        # Build chunks
        front_chunk = self._build_chunk(front_tris, cut_edges, plane, True)
        back_chunk = self._build_chunk(back_tris, cut_edges, plane, False)

        return SliceResult(
            front_chunk=front_chunk,
            back_chunk=back_chunk,
            plane=plane,
            cut_vertices=list(cut_edges.values())
        )

    def _is_degenerate_triangle(
        self,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        threshold: float = 1e-10
    ) -> bool:
        """Check if a triangle is degenerate (has near-zero area)."""
        edge1 = vec3_sub(v1, v0)
        edge2 = vec3_sub(v2, v0)
        cross = vec3_cross(edge1, edge2)
        area = 0.5 * vec3_length(cross)
        return area < threshold

    def _classify_and_clip_triangle(
        self,
        vertices: Tuple[Vec3, Vec3, Vec3],
        indices: Tuple[int, int, int],
        plane: Plane,
        cut_edges: Dict[Tuple[int, int], Vec3]
    ) -> Tuple[List[Tuple[Vec3, Vec3, Vec3]], List[Tuple[Vec3, Vec3, Vec3]]]:
        """
        Classify and clip a triangle against a plane.

        Returns tuple of (front_triangles, back_triangles).
        Handles degenerate input triangles and division by zero edge cases.
        """
        v0, v1, v2 = vertices

        # Skip degenerate input triangles
        if self._is_degenerate_triangle(v0, v1, v2):
            return [], []

        d0 = plane.signed_distance(v0)
        d1 = plane.signed_distance(v1)
        d2 = plane.signed_distance(v2)

        epsilon = 1e-6

        # Classify vertices
        c0 = 1 if d0 > epsilon else (-1 if d0 < -epsilon else 0)
        c1 = 1 if d1 > epsilon else (-1 if d1 < -epsilon else 0)
        c2 = 1 if d2 > epsilon else (-1 if d2 < -epsilon else 0)

        # All on one side
        if c0 >= 0 and c1 >= 0 and c2 >= 0:
            return [(v0, v1, v2)], []
        if c0 <= 0 and c1 <= 0 and c2 <= 0:
            return [], [(v0, v1, v2)]

        # Mixed - need to clip
        front = []
        back = []

        points = [v0, v1, v2]
        distances = [d0, d1, d2]
        idx = [indices[0], indices[1], indices[2]]

        front_verts = []
        back_verts = []

        for i in range(3):
            curr = points[i]
            curr_d = distances[i]
            curr_i = idx[i]

            next_idx = (i + 1) % 3
            next_v = points[next_idx]
            next_d = distances[next_idx]
            next_i = idx[next_idx]

            if curr_d >= -epsilon:
                front_verts.append(curr)
            if curr_d <= epsilon:
                back_verts.append(curr)

            # Check for intersection
            if (curr_d > epsilon) != (next_d > epsilon):
                # Guard against division by zero when edge lies on plane
                denominator = curr_d - next_d
                if abs(denominator) > epsilon:
                    t = curr_d / denominator
                    # Clamp t to [0, 1] for numerical stability
                    t = max(0.0, min(1.0, t))
                    intersection = vec3_lerp(curr, next_v, t)

                    front_verts.append(intersection)
                    back_verts.append(intersection)

                    # Store cut edge
                    edge_key = tuple(sorted([curr_i, next_i]))
                    cut_edges[edge_key] = intersection

        # Triangulate front and back polygons, filtering degenerate results
        if len(front_verts) >= 3:
            for i in range(1, len(front_verts) - 1):
                new_tri = (front_verts[0], front_verts[i], front_verts[i + 1])
                if not self._is_degenerate_triangle(new_tri[0], new_tri[1], new_tri[2]):
                    front.append(new_tri)

        if len(back_verts) >= 3:
            for i in range(1, len(back_verts) - 1):
                new_tri = (back_verts[0], back_verts[i], back_verts[i + 1])
                if not self._is_degenerate_triangle(new_tri[0], new_tri[1], new_tri[2]):
                    back.append(new_tri)

        return front, back

    def _build_chunk(
        self,
        triangles: List[Tuple[Vec3, Vec3, Vec3]],
        cut_edges: Dict[Tuple[int, int], Vec3],
        plane: Plane,
        is_front: bool
    ) -> Optional[Chunk]:
        """Build a chunk from triangles, optionally adding a cap."""
        if not triangles:
            return None

        # Convert to indexed mesh
        vertices = []
        indexed_tris = []
        vertex_map: Dict[Tuple, int] = {}

        for tri in triangles:
            indices = []
            for v in tri:
                v_key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))

                if v_key not in vertex_map:
                    vertex_map[v_key] = len(vertices)
                    vertices.append(v)

                indices.append(vertex_map[v_key])

            # Validate triangle: unique indices and non-degenerate geometry
            if len(set(indices)) == 3:
                tri_verts = [vertices[idx] for idx in indices]
                if not self._is_degenerate_triangle(tri_verts[0], tri_verts[1], tri_verts[2]):
                    indexed_tris.append(tuple(indices))

        if not indexed_tris:
            return None

        # Generate cap if requested
        if self._generate_caps and cut_edges:
            cap_tris = self._generate_cap(
                list(cut_edges.values()),
                plane,
                is_front,
                vertices,
                vertex_map
            )
            indexed_tris.extend(cap_tris)

        chunk = Chunk(
            vertices=vertices,
            triangles=indexed_tris,
            is_interior=True
        )
        chunk.compute_volume()
        chunk.compute_centroid()

        return chunk

    def _generate_cap(
        self,
        cut_vertices: List[Vec3],
        plane: Plane,
        is_front: bool,
        mesh_vertices: List[Vec3],
        vertex_map: Dict[Tuple, int]
    ) -> List[Triangle]:
        """
        Generate a cap surface from cut vertices.

        Uses ear clipping for 2D triangulation of the cut polygon.
        """
        if len(cut_vertices) < 3:
            return []

        # Project cut vertices to 2D on the plane
        tangent, bitangent = self._get_plane_basis(plane)

        projected = []
        for v in cut_vertices:
            local = vec3_sub(v, plane.point)
            u = vec3_dot(local, tangent)
            w = vec3_dot(local, bitangent)
            projected.append((u, w))

        # Sort vertices by angle around centroid
        cx = sum(p[0] for p in projected) / len(projected)
        cy = sum(p[1] for p in projected) / len(projected)

        angles = []
        for i, (u, w) in enumerate(projected):
            angle = math.atan2(w - cy, u - cx)
            angles.append((angle, i))

        angles.sort()
        sorted_indices = [i for _, i in angles]

        # Get or add vertices to mesh
        cap_indices = []
        for idx in sorted_indices:
            v = cut_vertices[idx]
            v_key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))

            if v_key not in vertex_map:
                vertex_map[v_key] = len(mesh_vertices)
                mesh_vertices.append(v)

            cap_indices.append(vertex_map[v_key])

        # Triangulate using fan triangulation (simple for convex polygons)
        cap_tris = []
        for i in range(1, len(cap_indices) - 1):
            if is_front:
                cap_tris.append((cap_indices[0], cap_indices[i], cap_indices[i + 1]))
            else:
                cap_tris.append((cap_indices[0], cap_indices[i + 1], cap_indices[i]))

        return cap_tris

    def _get_plane_basis(self, plane: Plane) -> Tuple[Vec3, Vec3]:
        """Get orthonormal basis vectors on the plane."""
        if abs(plane.normal[0]) < 0.9:
            up = (1.0, 0.0, 0.0)
        else:
            up = (0.0, 1.0, 0.0)

        tangent = vec3_normalize(vec3_cross(plane.normal, up))
        bitangent = vec3_normalize(vec3_cross(plane.normal, tangent))

        return tangent, bitangent

    def multi_slice(
        self,
        vertices: List[Vec3],
        triangles: List[Triangle],
        planes: List[Plane]
    ) -> List[Chunk]:
        """
        Slice mesh with multiple planes.

        Args:
            vertices: Mesh vertex positions.
            triangles: Mesh triangle indices.
            planes: List of slicing planes.

        Returns:
            List of resulting chunks.
        """
        self._planes = planes[:SLICE_MAX_PLANES]

        # Start with the original mesh as a single chunk
        current_chunks = [Chunk(
            vertices=list(vertices),
            triangles=list(triangles)
        )]

        for plane in self._planes:
            next_chunks = []

            for chunk in current_chunks:
                if len(next_chunks) >= self._max_chunks:
                    next_chunks.append(chunk)
                    continue

                result = self.slice_mesh(chunk.vertices, chunk.triangles, plane)

                if result.front_chunk and result.front_chunk.is_valid(self._min_chunk_volume):
                    next_chunks.append(result.front_chunk)
                if result.back_chunk and result.back_chunk.is_valid(self._min_chunk_volume):
                    next_chunks.append(result.back_chunk)

            current_chunks = next_chunks

            if len(current_chunks) >= self._max_chunks:
                break

        # Finalize chunks
        for i, chunk in enumerate(current_chunks):
            chunk.compute_volume()
            chunk.compute_centroid()
            chunk.cell_index = i

        return current_chunks

    def parallel_slices(
        self,
        vertices: List[Vec3],
        triangles: List[Triangle],
        direction: Vec3,
        num_slices: int,
        spacing: Optional[float] = None
    ) -> List[Chunk]:
        """
        Create multiple parallel slices.

        Args:
            vertices: Mesh vertex positions.
            triangles: Mesh triangle indices.
            direction: Normal direction for all slices.
            num_slices: Number of parallel slices.
            spacing: Distance between slices (auto-calculated if None).

        Returns:
            List of resulting chunks.
        """
        bounds = BoundingBox.from_points(vertices)
        direction = vec3_normalize(direction)

        # Calculate slice positions along the direction
        # Project bounds onto direction to find extent
        min_proj = float('inf')
        max_proj = float('-inf')

        corners = [
            bounds.min_point,
            bounds.max_point,
            (bounds.min_point[0], bounds.min_point[1], bounds.max_point[2]),
            (bounds.min_point[0], bounds.max_point[1], bounds.min_point[2]),
            (bounds.max_point[0], bounds.min_point[1], bounds.min_point[2]),
            (bounds.min_point[0], bounds.max_point[1], bounds.max_point[2]),
            (bounds.max_point[0], bounds.min_point[1], bounds.max_point[2]),
            (bounds.max_point[0], bounds.max_point[1], bounds.min_point[2]),
        ]

        for corner in corners:
            proj = vec3_dot(corner, direction)
            min_proj = min(min_proj, proj)
            max_proj = max(max_proj, proj)

        extent = max_proj - min_proj

        if spacing is None:
            spacing = extent / (num_slices + 1)

        # Create planes
        planes = []
        for i in range(1, num_slices + 1):
            t = i / (num_slices + 1)
            proj = min_proj + t * extent
            point = vec3_mul(direction, proj)
            planes.append(Plane(point=point, normal=direction))

        return self.multi_slice(vertices, triangles, planes)

    def random_slice_planes(
        self,
        bounds: BoundingBox,
        num_planes: int,
        bias_direction: Optional[Vec3] = None,
        bias_strength: float = 0.0
    ) -> List[Plane]:
        """
        Generate random slice planes within bounds.

        Args:
            bounds: Bounding box for plane generation.
            num_planes: Number of planes to generate.
            bias_direction: Optional direction to bias normals toward.
            bias_strength: Strength of directional bias (0-1).

        Returns:
            List of random planes.
        """
        planes = []

        for _ in range(min(num_planes, SLICE_MAX_PLANES)):
            # Random point within bounds
            point = bounds.random_point(self._rng)

            # Random normal
            theta = self._rng.uniform(0, 2 * math.pi)
            phi = self._rng.uniform(0, math.pi)

            normal = (
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi)
            )

            # Apply bias if specified
            if bias_direction and bias_strength > 0:
                bias_dir = vec3_normalize(bias_direction)
                normal = vec3_lerp(normal, bias_dir, bias_strength)
                normal = vec3_normalize(normal)

            planes.append(Plane(point=point, normal=normal))

        return planes

    def grid_slice(
        self,
        vertices: List[Vec3],
        triangles: List[Triangle],
        grid_size: Tuple[int, int, int]
    ) -> List[Chunk]:
        """
        Slice mesh into a regular grid of chunks.

        Args:
            vertices: Mesh vertex positions.
            triangles: Mesh triangle indices.
            grid_size: Number of divisions in (x, y, z).

        Returns:
            List of resulting chunks.
        """
        bounds = BoundingBox.from_points(vertices)

        planes = []

        # X slices
        for i in range(1, grid_size[0]):
            t = i / grid_size[0]
            x = bounds.min_point[0] + t * (bounds.max_point[0] - bounds.min_point[0])
            planes.append(Plane(point=(x, 0, 0), normal=(1, 0, 0)))

        # Y slices
        for i in range(1, grid_size[1]):
            t = i / grid_size[1]
            y = bounds.min_point[1] + t * (bounds.max_point[1] - bounds.min_point[1])
            planes.append(Plane(point=(0, y, 0), normal=(0, 1, 0)))

        # Z slices
        for i in range(1, grid_size[2]):
            t = i / grid_size[2]
            z = bounds.min_point[2] + t * (bounds.max_point[2] - bounds.min_point[2])
            planes.append(Plane(point=(0, 0, z), normal=(0, 0, 1)))

        return self.multi_slice(vertices, triangles, planes)

    def fracture_along_edge(
        self,
        vertices: List[Vec3],
        triangles: List[Triangle],
        edge_start: Vec3,
        edge_end: Vec3,
        num_slices: int = 1
    ) -> List[Chunk]:
        """
        Create slices along an edge (for blade/cutting effects).

        Args:
            vertices: Mesh vertex positions.
            triangles: Mesh triangle indices.
            edge_start: Start point of the cutting edge.
            edge_end: End point of the cutting edge.
            num_slices: Number of parallel slices along the edge.

        Returns:
            List of resulting chunks.
        """
        # Edge direction
        edge_dir = vec3_normalize(vec3_sub(edge_end, edge_start))

        # Create cutting planes perpendicular to the edge
        planes = []

        for i in range(num_slices):
            t = (i + 1) / (num_slices + 1)
            point = vec3_lerp(edge_start, edge_end, t)

            # Plane normal perpendicular to edge
            # Use any perpendicular direction
            if abs(edge_dir[0]) < 0.9:
                up = (1.0, 0.0, 0.0)
            else:
                up = (0.0, 1.0, 0.0)

            normal = vec3_normalize(vec3_cross(edge_dir, up))
            planes.append(Plane(point=point, normal=normal))

        return self.multi_slice(vertices, triangles, planes)


class AdaptiveSliceFracture(SliceFracture):
    """
    Adaptive slice fracture that generates more slices in impact areas.
    """

    def __init__(
        self,
        seed: int = DEFAULT_FRACTURE_SEED,
        min_chunk_volume: float = MIN_CHUNK_VOLUME,
        max_chunks: int = MAX_CHUNKS_PER_OBJECT,
        generate_caps: bool = True,
        base_slices: int = 4,
        max_slices: int = 16
    ) -> None:
        """
        Initialize adaptive slice fracture.

        Args:
            seed: Random seed.
            min_chunk_volume: Minimum chunk volume.
            max_chunks: Maximum chunks.
            generate_caps: Whether to generate caps.
            base_slices: Base number of slices.
            max_slices: Maximum number of slices.
        """
        super().__init__(seed, min_chunk_volume, max_chunks, generate_caps)
        self._base_slices = base_slices
        self._max_slices = max_slices

    def fracture_adaptive(
        self,
        vertices: List[Vec3],
        triangles: List[Triangle],
        impact_point: Vec3,
        impact_intensity: float = 1.0
    ) -> List[Chunk]:
        """
        Create adaptive fracture with more detail near impact.

        Args:
            vertices: Mesh vertex positions.
            triangles: Mesh triangle indices.
            impact_point: Point of impact.
            impact_intensity: Intensity (0-1) affecting number of slices.

        Returns:
            List of resulting chunks.
        """
        bounds = BoundingBox.from_points(vertices)

        # Calculate number of slices based on intensity
        num_slices = int(
            self._base_slices + (self._max_slices - self._base_slices) * impact_intensity
        )

        # Generate planes with density based on distance from impact
        planes = []
        max_dist = vec3_length(bounds.size)

        for _ in range(num_slices):
            # Random direction
            theta = self._rng.uniform(0, 2 * math.pi)
            phi = self._rng.uniform(0, math.pi)

            normal = (
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi)
            )

            # Distance from impact point (exponential distribution = more near impact)
            dist = self._rng.expovariate(3.0) * max_dist * 0.3
            point = vec3_add(impact_point, vec3_mul(normal, dist))

            # Clamp to bounds
            if bounds.contains(point):
                planes.append(Plane(point=point, normal=normal))

        return self.multi_slice(vertices, triangles, planes)


class HierarchicalSliceFracture(SliceFracture):
    """
    Hierarchical slice fracture that creates progressively smaller pieces.
    """

    def __init__(
        self,
        seed: int = DEFAULT_FRACTURE_SEED,
        min_chunk_volume: float = MIN_CHUNK_VOLUME,
        max_chunks: int = MAX_CHUNKS_PER_OBJECT,
        generate_caps: bool = True,
        max_depth: int = 3,
        split_threshold: float = 0.1
    ) -> None:
        """
        Initialize hierarchical slice fracture.

        Args:
            seed: Random seed.
            min_chunk_volume: Minimum chunk volume.
            max_chunks: Maximum chunks.
            generate_caps: Whether to generate caps.
            max_depth: Maximum recursion depth.
            split_threshold: Volume threshold for splitting (relative to original).
        """
        super().__init__(seed, min_chunk_volume, max_chunks, generate_caps)
        self._max_depth = max_depth
        self._split_threshold = split_threshold

    def fracture_hierarchical(
        self,
        vertices: List[Vec3],
        triangles: List[Triangle],
        depth: int = 0
    ) -> List[Chunk]:
        """
        Recursively fracture mesh into progressively smaller pieces.

        Args:
            vertices: Mesh vertex positions.
            triangles: Mesh triangle indices.
            depth: Current recursion depth.

        Returns:
            List of resulting chunks.
        """
        if depth >= self._max_depth:
            chunk = Chunk(vertices=list(vertices), triangles=list(triangles))
            chunk.compute_volume()
            chunk.compute_centroid()
            return [chunk] if chunk.is_valid(self._min_chunk_volume) else []

        bounds = BoundingBox.from_points(vertices)

        # Find longest axis
        size = bounds.size
        if size[0] >= size[1] and size[0] >= size[2]:
            axis = 0
            normal = (1, 0, 0)
        elif size[1] >= size[2]:
            axis = 1
            normal = (0, 1, 0)
        else:
            axis = 2
            normal = (0, 0, 1)

        # Create splitting plane at midpoint with jitter
        mid = bounds.center[axis]
        jitter = self._rng.uniform(-0.2, 0.2) * size[axis]
        point = list(bounds.center)
        point[axis] = mid + jitter
        plane = Plane(point=tuple(point), normal=normal)

        # Split
        result = self.slice_mesh(vertices, triangles, plane)

        chunks = []

        # Recursively process each half
        for chunk in [result.front_chunk, result.back_chunk]:
            if chunk is None:
                continue

            if chunk.volume > self._min_chunk_volume * self._split_threshold:
                sub_chunks = self.fracture_hierarchical(
                    chunk.vertices, chunk.triangles, depth + 1
                )
                chunks.extend(sub_chunks)
            else:
                if chunk.is_valid(self._min_chunk_volume):
                    chunks.append(chunk)

            if len(chunks) >= self._max_chunks:
                break

        return chunks
