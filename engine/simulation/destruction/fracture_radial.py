"""
Radial Fracture System.

Implements radial fracture patterns for impact-based destruction effects.
Creates fractures radiating outward from an impact point with optional
concentric rings for more complex fragmentation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict

from .config import (
    DEFAULT_FRACTURE_SEED,
    MIN_CHUNK_VOLUME,
    MAX_CHUNKS_PER_OBJECT,
    RADIAL_MIN_SLICES,
    RADIAL_MAX_SLICES,
    RADIAL_DEFAULT_SLICES,
    RADIAL_MIN_RINGS,
    RADIAL_MAX_RINGS,
    RADIAL_DEFAULT_RINGS,
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
    vec3_distance,
    vec3_lerp,
)


@dataclass(slots=True)
class RadialSlice:
    """
    Represents a radial slice (wedge) in the fracture pattern.

    Attributes:
        index: Index of this slice.
        angle_start: Starting angle in radians.
        angle_end: Ending angle in radians.
        plane: The cutting plane for this slice.
    """
    index: int
    angle_start: float
    angle_end: float
    plane: Plane


@dataclass(slots=True)
class ConcentricRing:
    """
    Represents a concentric ring in the fracture pattern.

    Attributes:
        index: Index of this ring (0 = innermost).
        radius_inner: Inner radius of the ring.
        radius_outer: Outer radius of the ring.
    """
    index: int
    radius_inner: float
    radius_outer: float


@dataclass(slots=True)
class RadialChunk(Chunk):
    """
    Extended chunk for radial fracture results.

    Attributes:
        slice_index: Index of the radial slice this chunk belongs to.
        ring_index: Index of the concentric ring this chunk belongs to.
    """
    slice_index: int = -1
    ring_index: int = -1


class RadialFracture:
    """
    Radial fracture pattern generator.

    Creates fractures radiating outward from an impact point, optionally
    with concentric rings for more complex fragmentation patterns.
    """

    __slots__ = (
        '_seed', '_rng', '_num_slices', '_num_rings', '_slices', '_rings',
        '_center', '_direction', '_radius', '_min_chunk_volume', '_max_chunks',
        '_jitter_amount'
    )

    def __init__(
        self,
        seed: int = DEFAULT_FRACTURE_SEED,
        num_slices: int = RADIAL_DEFAULT_SLICES,
        num_rings: int = RADIAL_DEFAULT_RINGS,
        min_chunk_volume: float = MIN_CHUNK_VOLUME,
        max_chunks: int = MAX_CHUNKS_PER_OBJECT,
        jitter_amount: float = 0.1
    ) -> None:
        """
        Initialize radial fracture generator.

        Args:
            seed: Random seed for deterministic generation.
            num_slices: Number of radial slices (wedges).
            num_rings: Number of concentric rings.
            min_chunk_volume: Minimum volume for valid chunks.
            max_chunks: Maximum number of chunks to generate.
            jitter_amount: Amount of randomization for slice angles (0-1).
        """
        self._seed = seed
        self._rng = random.Random(seed)
        self._num_slices = max(RADIAL_MIN_SLICES, min(num_slices, RADIAL_MAX_SLICES))
        self._num_rings = max(RADIAL_MIN_RINGS, min(num_rings, RADIAL_MAX_RINGS))
        self._slices: List[RadialSlice] = []
        self._rings: List[ConcentricRing] = []
        self._center: Vec3 = (0.0, 0.0, 0.0)
        self._direction: Vec3 = (0.0, 0.0, 1.0)
        self._radius: float = 1.0
        self._min_chunk_volume = min_chunk_volume
        self._max_chunks = max_chunks
        self._jitter_amount = max(0.0, min(1.0, jitter_amount))

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
    def num_slices(self) -> int:
        """Number of radial slices."""
        return self._num_slices

    @property
    def num_rings(self) -> int:
        """Number of concentric rings."""
        return self._num_rings

    @property
    def slices(self) -> List[RadialSlice]:
        """List of radial slices."""
        return self._slices.copy()

    @property
    def rings(self) -> List[ConcentricRing]:
        """List of concentric rings."""
        return self._rings.copy()

    def generate_radial_pattern(
        self,
        center: Vec3,
        radius: float,
        direction: Optional[Vec3] = None,
        num_slices: Optional[int] = None,
        num_rings: Optional[int] = None
    ) -> Tuple[List[RadialSlice], List[ConcentricRing]]:
        """
        Generate the radial fracture pattern.

        Args:
            center: Center point of the radial pattern.
            radius: Maximum radius of the pattern.
            direction: Normal direction of the radial plane.
            num_slices: Override number of slices.
            num_rings: Override number of rings.

        Returns:
            Tuple of (slices, rings).
        """
        self._center = center
        self._radius = radius
        self._direction = vec3_normalize(direction) if direction else (0.0, 0.0, 1.0)

        if num_slices is not None:
            self._num_slices = max(RADIAL_MIN_SLICES, min(num_slices, RADIAL_MAX_SLICES))
        if num_rings is not None:
            self._num_rings = max(RADIAL_MIN_RINGS, min(num_rings, RADIAL_MAX_RINGS))

        self._generate_slices()
        self._generate_rings()

        return self._slices, self._rings

    def _generate_slices(self) -> None:
        """Generate radial slices with optional jitter."""
        self._slices = []

        # Base angle increment
        base_angle = 2.0 * math.pi / self._num_slices

        # Generate angles with jitter
        angles = []
        for i in range(self._num_slices):
            angle = i * base_angle

            # Add jitter
            if self._jitter_amount > 0 and i > 0:
                jitter = self._rng.uniform(-1, 1) * base_angle * self._jitter_amount * 0.5
                angle += jitter

            angles.append(angle)

        # Sort and wrap
        angles.sort()

        # Create slice objects with cutting planes
        for i in range(self._num_slices):
            angle_start = angles[i]
            angle_end = angles[(i + 1) % self._num_slices]

            if angle_end < angle_start:
                angle_end += 2.0 * math.pi

            # Create cutting plane for this slice boundary
            # The plane normal is perpendicular to both the direction and the radial direction
            plane = self._create_radial_plane(angles[i])

            self._slices.append(RadialSlice(
                index=i,
                angle_start=angle_start,
                angle_end=angle_end,
                plane=plane
            ))

    def _generate_rings(self) -> None:
        """Generate concentric rings."""
        self._rings = []

        # Use quadratic spacing for rings (smaller pieces near center)
        for i in range(self._num_rings):
            t_inner = (i / self._num_rings) ** 1.5
            t_outer = ((i + 1) / self._num_rings) ** 1.5

            radius_inner = t_inner * self._radius
            radius_outer = t_outer * self._radius

            self._rings.append(ConcentricRing(
                index=i,
                radius_inner=radius_inner,
                radius_outer=radius_outer
            ))

    def _create_radial_plane(self, angle: float) -> Plane:
        """Create a cutting plane at the given angle from center."""
        # Get orthonormal basis for the radial plane
        tangent, bitangent = self._get_orthonormal_basis()

        # Direction in the radial plane at this angle
        radial_dir = vec3_add(
            vec3_mul(tangent, math.cos(angle)),
            vec3_mul(bitangent, math.sin(angle))
        )

        # Plane normal is perpendicular to radial direction in the plane
        plane_normal = vec3_cross(radial_dir, self._direction)
        plane_normal = vec3_normalize(plane_normal)

        return Plane(point=self._center, normal=plane_normal)

    def _get_orthonormal_basis(self) -> Tuple[Vec3, Vec3]:
        """Get orthonormal basis vectors perpendicular to direction."""
        # Find a vector not parallel to direction
        if abs(self._direction[0]) < 0.9:
            up = (1.0, 0.0, 0.0)
        else:
            up = (0.0, 1.0, 0.0)

        tangent = vec3_normalize(vec3_cross(self._direction, up))
        bitangent = vec3_normalize(vec3_cross(self._direction, tangent))

        return tangent, bitangent

    def generate_impact_directed(
        self,
        center: Vec3,
        impact_direction: Vec3,
        radius: float,
        intensity: float = 1.0
    ) -> Tuple[List[RadialSlice], List[ConcentricRing]]:
        """
        Generate impact-directed radial pattern.

        The pattern is oriented based on the impact direction, with more
        fragmentation in the direction of impact.

        Args:
            center: Impact point.
            impact_direction: Direction of the impact force.
            radius: Maximum radius of fracture.
            intensity: Impact intensity (affects number of fragments).

        Returns:
            Tuple of (slices, rings).
        """
        # Adjust pattern based on intensity
        num_slices = int(self._num_slices * (0.5 + intensity * 0.5))
        num_rings = int(self._num_rings * (0.5 + intensity * 0.5))

        num_slices = max(RADIAL_MIN_SLICES, min(num_slices, RADIAL_MAX_SLICES))
        num_rings = max(RADIAL_MIN_RINGS, min(num_rings, RADIAL_MAX_RINGS))

        # Generate pattern with impact direction as the normal
        return self.generate_radial_pattern(
            center=center,
            radius=radius,
            direction=vec3_normalize(impact_direction),
            num_slices=num_slices,
            num_rings=num_rings
        )

    def fracture_mesh(
        self,
        vertices: List[Vec3],
        triangles: List[Triangle],
        center: Optional[Vec3] = None,
        radius: Optional[float] = None,
        direction: Optional[Vec3] = None
    ) -> List[RadialChunk]:
        """
        Fracture a mesh using radial pattern.

        Args:
            vertices: Mesh vertex positions.
            triangles: Mesh triangle indices.
            center: Fracture center (default: mesh centroid).
            radius: Fracture radius (default: auto from bounds).
            direction: Fracture normal direction.

        Returns:
            List of fractured chunks.
        """
        bounds = BoundingBox.from_points(vertices)

        if center is None:
            center = bounds.center

        if radius is None:
            radius = vec3_length(bounds.size) / 2.0

        self.generate_radial_pattern(center, radius, direction)

        chunks = []

        # For each cell (slice x ring), clip the mesh
        for slice_obj in self._slices:
            for ring in self._rings:
                chunk = self._clip_mesh_to_cell(
                    vertices, triangles,
                    slice_obj, ring
                )

                if chunk is not None and chunk.is_valid(self._min_chunk_volume):
                    chunk.slice_index = slice_obj.index
                    chunk.ring_index = ring.index
                    chunks.append(chunk)

                    if len(chunks) >= self._max_chunks:
                        return chunks

        self._compute_adjacency(chunks)
        return chunks

    def _clip_mesh_to_cell(
        self,
        vertices: List[Vec3],
        triangles: List[Triangle],
        slice_obj: RadialSlice,
        ring: ConcentricRing
    ) -> Optional[RadialChunk]:
        """Clip mesh to a single radial cell (slice x ring)."""
        # Collect all cutting planes for this cell
        planes = []

        # Radial slice planes
        plane1 = slice_obj.plane

        # Get the next slice's plane (with opposite normal)
        next_slice_idx = (slice_obj.index + 1) % len(self._slices)
        if next_slice_idx < len(self._slices):
            plane2 = self._slices[next_slice_idx].plane
            plane2 = Plane(point=plane2.point, normal=vec3_mul(plane2.normal, -1))
            planes.append(plane2)

        planes.append(plane1)

        # Ring boundaries (as spherical constraints converted to planes)
        # We approximate with multiple tangent planes

        # Filter triangles by rough distance check first
        filtered_tris = []
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            centroid = (
                (v0[0] + v1[0] + v2[0]) / 3,
                (v0[1] + v1[1] + v2[1]) / 3,
                (v0[2] + v1[2] + v2[2]) / 3
            )

            dist = vec3_distance(centroid, self._center)
            if ring.radius_inner <= dist <= ring.radius_outer * 1.5:
                filtered_tris.append((v0, v1, v2))

        if not filtered_tris:
            return None

        # Clip against planes
        current_tris = filtered_tris

        for plane in planes:
            if not current_tris:
                break

            next_tris = []
            for tri in current_tris:
                clipped = self._clip_triangle_to_plane(tri, plane)
                next_tris.extend(clipped)

            current_tris = next_tris

        # Filter by ring radius
        ring_filtered = []
        for tri in current_tris:
            # Check if triangle is within ring
            centroid = (
                (tri[0][0] + tri[1][0] + tri[2][0]) / 3,
                (tri[0][1] + tri[1][1] + tri[2][1]) / 3,
                (tri[0][2] + tri[1][2] + tri[2][2]) / 3
            )

            dist = vec3_distance(centroid, self._center)
            if ring.radius_inner <= dist <= ring.radius_outer:
                ring_filtered.append(tri)

        if not ring_filtered:
            return None

        # Convert to indexed mesh
        chunk_vertices = []
        chunk_triangles = []
        vertex_map: Dict[Tuple, int] = {}

        for tri in ring_filtered:
            indices = []
            for v in tri:
                v_key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))

                if v_key not in vertex_map:
                    vertex_map[v_key] = len(chunk_vertices)
                    chunk_vertices.append(v)

                indices.append(vertex_map[v_key])

            if len(set(indices)) == 3:
                chunk_triangles.append(tuple(indices))

        if not chunk_triangles:
            return None

        chunk = RadialChunk(
            vertices=chunk_vertices,
            triangles=chunk_triangles,
            is_interior=True
        )
        chunk.compute_volume()
        chunk.compute_centroid()

        return chunk

    def _clip_triangle_to_plane(
        self,
        tri: Tuple[Vec3, Vec3, Vec3],
        plane: Plane
    ) -> List[Tuple[Vec3, Vec3, Vec3]]:
        """Clip a triangle against a plane.

        Handles edge cases including degenerate triangles and edges on the plane.
        """
        v0, v1, v2 = tri

        # Skip degenerate input triangles
        if self._is_degenerate_triangle(v0, v1, v2):
            return []

        d0 = plane.signed_distance(v0)
        d1 = plane.signed_distance(v1)
        d2 = plane.signed_distance(v2)

        epsilon = 1e-6

        pos_count = sum(1 for d in [d0, d1, d2] if d > epsilon)
        neg_count = sum(1 for d in [d0, d1, d2] if d < -epsilon)

        if pos_count == 0:
            return [tri]

        if neg_count == 0:
            return []

        result = []
        points = [v0, v1, v2]
        distances = [d0, d1, d2]

        output = []

        for i in range(3):
            curr = points[i]
            next_p = points[(i + 1) % 3]
            curr_d = distances[i]
            next_d = distances[(i + 1) % 3]

            if curr_d <= epsilon:
                output.append(curr)

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
                if not self._is_degenerate_triangle(new_tri[0], new_tri[1], new_tri[2]):
                    result.append(new_tri)

        return result

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

    def _compute_adjacency(self, chunks: List[RadialChunk]) -> None:
        """Compute adjacency between radial chunks."""
        for i, chunk_i in enumerate(chunks):
            for j, chunk_j in enumerate(chunks):
                if i >= j:
                    continue

                # Adjacent if same slice and neighboring rings
                if chunk_i.slice_index == chunk_j.slice_index:
                    if abs(chunk_i.ring_index - chunk_j.ring_index) == 1:
                        chunk_i.adjacent_chunks.add(j)
                        chunk_j.adjacent_chunks.add(i)
                        continue

                # Adjacent if same ring and neighboring slices
                if chunk_i.ring_index == chunk_j.ring_index:
                    slice_diff = abs(chunk_i.slice_index - chunk_j.slice_index)
                    if slice_diff == 1 or slice_diff == self._num_slices - 1:
                        chunk_i.adjacent_chunks.add(j)
                        chunk_j.adjacent_chunks.add(i)

    def get_cut_planes(self) -> List[Plane]:
        """
        Get all cutting planes for the radial pattern.

        Useful for debris effects or visual representation.
        """
        planes = []

        for slice_obj in self._slices:
            planes.append(slice_obj.plane)

        return planes

    def get_ring_boundaries(self) -> List[float]:
        """Get list of ring boundary radii."""
        boundaries = [0.0]
        for ring in self._rings:
            boundaries.append(ring.radius_outer)
        return boundaries


class ConcentricRadialFracture(RadialFracture):
    """
    Enhanced radial fracture with emphasis on concentric rings.

    Creates patterns typical of impact craters or bullet holes.
    """

    def __init__(
        self,
        seed: int = DEFAULT_FRACTURE_SEED,
        num_slices: int = RADIAL_DEFAULT_SLICES,
        num_rings: int = RADIAL_DEFAULT_RINGS + 2,  # More rings by default
        min_chunk_volume: float = MIN_CHUNK_VOLUME,
        max_chunks: int = MAX_CHUNKS_PER_OBJECT,
        ring_spacing: str = "quadratic"  # "linear", "quadratic", "logarithmic"
    ) -> None:
        """
        Initialize concentric radial fracture generator.

        Args:
            seed: Random seed.
            num_slices: Number of radial slices.
            num_rings: Number of concentric rings.
            min_chunk_volume: Minimum chunk volume.
            max_chunks: Maximum chunks.
            ring_spacing: Ring spacing function.
        """
        super().__init__(
            seed, num_slices, num_rings,
            min_chunk_volume, max_chunks
        )
        self._ring_spacing = ring_spacing

    def _generate_rings(self) -> None:
        """Generate rings with configurable spacing."""
        self._rings = []

        for i in range(self._num_rings):
            if self._ring_spacing == "linear":
                t_inner = i / self._num_rings
                t_outer = (i + 1) / self._num_rings
            elif self._ring_spacing == "logarithmic":
                t_inner = math.log(i + 1) / math.log(self._num_rings + 1)
                t_outer = math.log(i + 2) / math.log(self._num_rings + 1)
            else:  # quadratic (default)
                t_inner = (i / self._num_rings) ** 1.5
                t_outer = ((i + 1) / self._num_rings) ** 1.5

            radius_inner = t_inner * self._radius
            radius_outer = t_outer * self._radius

            self._rings.append(ConcentricRing(
                index=i,
                radius_inner=radius_inner,
                radius_outer=radius_outer
            ))


class SpiderWebFracture(RadialFracture):
    """
    Spider web fracture pattern.

    Creates a pattern resembling cracked glass with radial and
    circular crack lines.
    """

    def __init__(
        self,
        seed: int = DEFAULT_FRACTURE_SEED,
        num_radial: int = 8,
        num_circular: int = 4,
        irregularity: float = 0.3
    ) -> None:
        """
        Initialize spider web fracture generator.

        Args:
            seed: Random seed.
            num_radial: Number of radial crack lines.
            num_circular: Number of circular crack rings.
            irregularity: Amount of irregularity (0-1).
        """
        super().__init__(
            seed=seed,
            num_slices=num_radial,
            num_rings=num_circular,
            jitter_amount=irregularity
        )
        self._irregularity = irregularity

    def _generate_rings(self) -> None:
        """Generate irregular circular rings."""
        self._rings = []

        # Random ring radii with sorting
        ring_radii = sorted([
            self._rng.uniform(0.1, 1.0) * self._radius
            for _ in range(self._num_rings)
        ])

        for i in range(self._num_rings):
            radius_inner = 0.0 if i == 0 else ring_radii[i - 1]
            radius_outer = ring_radii[i]

            # Add irregularity
            if self._irregularity > 0:
                radius_inner *= 1.0 + self._rng.uniform(-1, 1) * self._irregularity * 0.2
                radius_outer *= 1.0 + self._rng.uniform(-1, 1) * self._irregularity * 0.2

            radius_inner = max(0, radius_inner)
            radius_outer = max(radius_inner + 0.01, radius_outer)

            self._rings.append(ConcentricRing(
                index=i,
                radius_inner=radius_inner,
                radius_outer=radius_outer
            ))
