"""
Meshlet/Cluster System for GPU-driven rendering.

Meshlets are small groups of triangles (~64 vertices, ~124 triangles) that
enable fine-grained GPU culling and efficient mesh shader pipelines.

References:
- RENDERING_CONTEXT.md Section 6.7 Geometry Systems
- Nanite-style meshlet clustering for GPU-driven rendering
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Iterator, Optional, Sequence

from engine.rendering.gpu_driven.culling import AABB, BoundingSphere, Vec3


# =============================================================================
# MESHLET CONSTANTS
# =============================================================================


class MeshletConstants(IntEnum):
    """Standard meshlet size constants."""
    MAX_VERTICES = 64  # Maximum vertices per meshlet
    MAX_TRIANGLES = 124  # Maximum triangles per meshlet (126 for AMD, 124 for NVIDIA)
    MAX_INDICES = MAX_TRIANGLES * 3  # Maximum indices per meshlet


# =============================================================================
# MESHLET DATA STRUCTURES
# =============================================================================


@dataclass(slots=True)
class MeshletBounds:
    """
    Bounding information for meshlet culling.

    Contains a bounding sphere for frustum/occlusion culling and
    a normal cone for backface cluster culling.
    """
    # Bounding sphere for frustum and occlusion culling
    bounding_sphere: BoundingSphere = field(default_factory=BoundingSphere)

    # Normal cone for backface culling
    # Cone axis points in average normal direction
    cone_axis: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 1.0))
    # Cosine of half-angle (dot(view_dir, -cone_axis) > cone_cutoff means visible)
    cone_cutoff: float = 1.0  # 1.0 = degenerate cone, always visible

    # Apex offset for better cone fitting (for curved surfaces)
    cone_apex_offset: float = 0.0

    @property
    def is_cone_degenerate(self) -> bool:
        """Check if the normal cone is degenerate (all triangles must be tested)."""
        return self.cone_cutoff >= 1.0 or self.cone_cutoff < -1.0


@dataclass
class Meshlet:
    """
    A meshlet represents a small cluster of triangles from a mesh.

    Meshlets enable:
    - Fine-grained GPU culling (per-cluster instead of per-object)
    - Efficient mesh shader workload distribution
    - Better cache utilization
    """
    # Unique meshlet ID within the mesh
    meshlet_id: int = 0

    # Vertex indices local to this meshlet (into global vertex buffer)
    vertex_indices: list[int] = field(default_factory=list)

    # Triangle indices as offsets into vertex_indices (not global indices)
    # Each consecutive 3 values form a triangle
    local_indices: list[int] = field(default_factory=list)

    # Bounding information for culling
    bounds: MeshletBounds = field(default_factory=MeshletBounds)

    # LOD level this meshlet belongs to
    lod_level: int = 0

    # Parent meshlet ID for hierarchical LOD (Nanite-style)
    parent_meshlet_id: Optional[int] = None

    @property
    def vertex_count(self) -> int:
        return len(self.vertex_indices)

    @property
    def triangle_count(self) -> int:
        return len(self.local_indices) // 3

    @property
    def index_count(self) -> int:
        return len(self.local_indices)

    def get_triangle(self, triangle_index: int) -> tuple[int, int, int]:
        """Get global vertex indices for a triangle."""
        base = triangle_index * 3
        local_a = self.local_indices[base]
        local_b = self.local_indices[base + 1]
        local_c = self.local_indices[base + 2]
        return (
            self.vertex_indices[local_a],
            self.vertex_indices[local_b],
            self.vertex_indices[local_c],
        )

    def iterate_triangles(self) -> Iterator[tuple[int, int, int]]:
        """Iterate over all triangles, yielding global vertex indices."""
        for i in range(self.triangle_count):
            yield self.get_triangle(i)


# =============================================================================
# VERTEX DATA
# =============================================================================


@dataclass(slots=True)
class Vertex:
    """Vertex with position and normal for meshlet building."""
    position: Vec3 = field(default_factory=Vec3)
    normal: Vec3 = field(default_factory=lambda: Vec3(0.0, 1.0, 0.0))


# =============================================================================
# MESHLET BUILDER
# =============================================================================


class MeshletBuildConfig:
    """Configuration for meshlet building."""

    def __init__(
        self,
        max_vertices: int = MeshletConstants.MAX_VERTICES,
        max_triangles: int = MeshletConstants.MAX_TRIANGLES,
        cone_weight: float = 0.5,  # Weight for normal cone optimization (0-1)
        locality_weight: float = 0.5,  # Weight for spatial locality (0-1)
    ) -> None:
        self.max_vertices = min(max_vertices, MeshletConstants.MAX_VERTICES)
        self.max_triangles = min(max_triangles, MeshletConstants.MAX_TRIANGLES)
        self.cone_weight = max(0.0, min(1.0, cone_weight))
        self.locality_weight = max(0.0, min(1.0, locality_weight))


class MeshletBuilder:
    """
    Builds meshlets from mesh data.

    Algorithm:
    1. Start with a seed triangle
    2. Greedily add adjacent triangles that:
       - Share vertices with existing triangles (cache friendly)
       - Don't exceed vertex/triangle limits
       - Maintain good normal cone fit
    3. When full, start a new meshlet
    """

    def __init__(self, config: Optional[MeshletBuildConfig] = None) -> None:
        self._config = config or MeshletBuildConfig()
        self._vertices: list[Vertex] = []
        self._indices: list[int] = []

    @property
    def config(self) -> MeshletBuildConfig:
        return self._config

    def build(
        self,
        vertices: Sequence[Vertex],
        indices: Sequence[int],
    ) -> list[Meshlet]:
        """
        Build meshlets from mesh data.

        Args:
            vertices: List of vertices with positions and normals
            indices: Triangle indices (must be multiple of 3)

        Returns:
            List of meshlets covering all triangles
        """
        if len(indices) % 3 != 0:
            raise ValueError("Index count must be a multiple of 3")

        self._vertices = list(vertices)
        self._indices = list(indices)

        num_triangles = len(indices) // 3

        # Track which triangles have been assigned to meshlets
        triangle_used = [False] * num_triangles

        # Build adjacency information
        adjacency = self._build_adjacency(num_triangles)

        meshlets: list[Meshlet] = []
        meshlet_id = 0

        # Process all triangles
        for seed_tri in range(num_triangles):
            if triangle_used[seed_tri]:
                continue

            # Start a new meshlet
            meshlet = self._build_meshlet(
                seed_tri,
                triangle_used,
                adjacency,
                meshlet_id,
            )
            meshlets.append(meshlet)
            meshlet_id += 1

        return meshlets

    def _build_adjacency(self, num_triangles: int) -> dict[int, set[int]]:
        """Build triangle adjacency map (triangles sharing edges/vertices)."""
        # Map vertex index to triangles containing it
        vertex_to_triangles: dict[int, list[int]] = {}
        for tri_idx in range(num_triangles):
            for j in range(3):
                vert_idx = self._indices[tri_idx * 3 + j]
                if vert_idx not in vertex_to_triangles:
                    vertex_to_triangles[vert_idx] = []
                vertex_to_triangles[vert_idx].append(tri_idx)

        # Build adjacency (triangles sharing at least one vertex)
        adjacency: dict[int, set[int]] = {i: set() for i in range(num_triangles)}
        for tri_idx in range(num_triangles):
            for j in range(3):
                vert_idx = self._indices[tri_idx * 3 + j]
                for neighbor in vertex_to_triangles[vert_idx]:
                    if neighbor != tri_idx:
                        adjacency[tri_idx].add(neighbor)

        return adjacency

    def _build_meshlet(
        self,
        seed_triangle: int,
        triangle_used: list[bool],
        adjacency: dict[int, set[int]],
        meshlet_id: int,
    ) -> Meshlet:
        """Build a single meshlet starting from a seed triangle."""
        meshlet = Meshlet(meshlet_id=meshlet_id)

        # Map global vertex index to local index within meshlet
        global_to_local: dict[int, int] = {}

        # Candidate triangles (adjacent to current meshlet)
        candidates: set[int] = set()

        # Add seed triangle
        self._add_triangle_to_meshlet(
            seed_triangle,
            meshlet,
            global_to_local,
            triangle_used,
            adjacency,
            candidates,
        )

        # Greedily add triangles
        while candidates and meshlet.triangle_count < self._config.max_triangles:
            best_tri = self._select_best_candidate(
                candidates,
                meshlet,
                global_to_local,
            )

            if best_tri is None:
                break

            # Check if adding this triangle would exceed limits
            new_verts = self._count_new_vertices(best_tri, global_to_local)
            if meshlet.vertex_count + new_verts > self._config.max_vertices:
                candidates.discard(best_tri)
                continue

            self._add_triangle_to_meshlet(
                best_tri,
                meshlet,
                global_to_local,
                triangle_used,
                adjacency,
                candidates,
            )

        # Compute bounds
        meshlet.bounds = self._compute_meshlet_bounds(meshlet)

        return meshlet

    def _add_triangle_to_meshlet(
        self,
        triangle_idx: int,
        meshlet: Meshlet,
        global_to_local: dict[int, int],
        triangle_used: list[bool],
        adjacency: dict[int, set[int]],
        candidates: set[int],
    ) -> None:
        """Add a triangle to the meshlet."""
        triangle_used[triangle_idx] = True
        candidates.discard(triangle_idx)

        # Add vertices and local indices
        for j in range(3):
            global_idx = self._indices[triangle_idx * 3 + j]

            if global_idx not in global_to_local:
                local_idx = len(meshlet.vertex_indices)
                global_to_local[global_idx] = local_idx
                meshlet.vertex_indices.append(global_idx)

            meshlet.local_indices.append(global_to_local[global_idx])

        # Add adjacent triangles to candidates
        for neighbor in adjacency[triangle_idx]:
            if not triangle_used[neighbor]:
                candidates.add(neighbor)

    def _count_new_vertices(
        self,
        triangle_idx: int,
        global_to_local: dict[int, int],
    ) -> int:
        """Count how many new vertices a triangle would add."""
        count = 0
        for j in range(3):
            global_idx = self._indices[triangle_idx * 3 + j]
            if global_idx not in global_to_local:
                count += 1
        return count

    def _select_best_candidate(
        self,
        candidates: set[int],
        meshlet: Meshlet,
        global_to_local: dict[int, int],
    ) -> Optional[int]:
        """Select the best candidate triangle to add (fewest new vertices)."""
        best_tri = None
        best_score = float("inf")

        for tri_idx in candidates:
            # Score: fewer new vertices = better (cache efficiency)
            new_verts = self._count_new_vertices(tri_idx, global_to_local)
            score = new_verts

            if score < best_score:
                best_score = score
                best_tri = tri_idx

        return best_tri

    def _compute_meshlet_bounds(self, meshlet: Meshlet) -> MeshletBounds:
        """Compute bounding sphere and normal cone for meshlet."""
        bounds = MeshletBounds()

        if not meshlet.vertex_indices:
            return bounds

        # Compute AABB first
        aabb = AABB()
        for global_idx in meshlet.vertex_indices:
            aabb.expand(self._vertices[global_idx].position)

        # Bounding sphere from AABB
        bounds.bounding_sphere = BoundingSphere.from_aabb(aabb)

        # Refine bounding sphere with Ritter's algorithm
        self._refine_bounding_sphere(meshlet, bounds.bounding_sphere)

        # Compute normal cone
        self._compute_normal_cone(meshlet, bounds)

        return bounds

    def _refine_bounding_sphere(
        self,
        meshlet: Meshlet,
        sphere: BoundingSphere,
    ) -> None:
        """Refine bounding sphere using Ritter's algorithm."""
        for global_idx in meshlet.vertex_indices:
            pos = self._vertices[global_idx].position
            diff = pos - sphere.center
            dist_sq = diff.length_squared()

            if dist_sq > sphere.radius * sphere.radius:
                # Point is outside sphere, expand
                dist = math.sqrt(dist_sq)
                new_radius = (sphere.radius + dist) * 0.5
                move_dist = dist - sphere.radius
                sphere.center = sphere.center + diff * (move_dist / (2.0 * dist))
                sphere.radius = new_radius

    def _compute_normal_cone(
        self,
        meshlet: Meshlet,
        bounds: MeshletBounds,
    ) -> None:
        """
        Compute normal cone for backface cluster culling.

        The normal cone represents the spread of triangle normals in the meshlet.
        If all normals are within a cone, we can cull the entire meshlet when
        viewing from certain directions.
        """
        if meshlet.triangle_count == 0:
            return

        # Compute average normal
        avg_normal = Vec3(0.0, 0.0, 0.0)
        normals: list[Vec3] = []

        for i in range(meshlet.triangle_count):
            tri = meshlet.get_triangle(i)
            p0 = self._vertices[tri[0]].position
            p1 = self._vertices[tri[1]].position
            p2 = self._vertices[tri[2]].position

            # Triangle normal (unnormalized for area weighting)
            e1 = p1 - p0
            e2 = p2 - p0
            normal = Vec3(
                e1.y * e2.z - e1.z * e2.y,
                e1.z * e2.x - e1.x * e2.z,
                e1.x * e2.y - e1.y * e2.x,
            )
            normals.append(normal.normalized())
            avg_normal = avg_normal + normal

        avg_normal = avg_normal.normalized()

        if avg_normal.length_squared() < 1e-8:
            # Degenerate case
            bounds.cone_axis = Vec3(0.0, 0.0, 1.0)
            bounds.cone_cutoff = 1.0
            return

        bounds.cone_axis = avg_normal

        # Find maximum deviation from average normal
        min_dot = 1.0
        for normal in normals:
            dot = normal.dot(avg_normal)
            min_dot = min(min_dot, dot)

        # cone_cutoff is the cosine of the half-angle
        # A triangle is potentially visible if dot(view_dir, -cone_axis) > cone_cutoff
        bounds.cone_cutoff = min_dot


# =============================================================================
# MESHLET CULLING
# =============================================================================


class MeshletCuller:
    """
    Performs per-meshlet culling including backface cluster culling.

    Culling tests:
    1. Frustum culling (bounding sphere vs frustum planes)
    2. Backface culling (normal cone test)
    3. Occlusion culling (optional, HZB-based)
    """

    def __init__(self) -> None:
        self._camera_position: Vec3 = Vec3()

    def update(
        self,
        camera_position: Optional[Vec3] = None,
    ) -> None:
        """Update culler state."""
        if camera_position is not None:
            self._camera_position = camera_position

    def is_backface_culled(
        self,
        meshlet: Meshlet,
        object_position: Optional[Vec3] = None,
    ) -> bool:
        """
        Test if meshlet can be backface culled based on normal cone.

        Returns True if the meshlet is entirely backfacing (can be culled).
        """
        if object_position is None:
            object_position = Vec3()

        bounds = meshlet.bounds

        if bounds.is_cone_degenerate:
            return False  # Cone is degenerate, can't cull

        # View direction from meshlet center to camera
        meshlet_center = bounds.bounding_sphere.center + object_position
        view_dir = (self._camera_position - meshlet_center).normalized()

        # If view direction is outside the normal cone, all triangles are backfacing
        # Cone axis points in average normal direction
        # Triangle is frontfacing if dot(view_dir, normal) > 0
        # All triangles in cone are backfacing if dot(view_dir, -cone_axis) <= -cone_cutoff
        # Which is equivalent to: dot(view_dir, cone_axis) >= cone_cutoff

        dot = view_dir.dot(bounds.cone_axis)

        # Note: cone_cutoff is cos(half_angle)
        # If dot >= cone_cutoff, view is within the cone of normals, some faces are front
        # If dot < cone_cutoff, view is outside cone, but we need to account for the sign
        # For backface culling: cull if dot(view_dir, all_normals) < 0
        # This happens when dot(view_dir, cone_axis) < -cone_cutoff (for negative cone)

        # Simplified: cull if all normals point away
        return dot < -bounds.cone_cutoff

    def cull_meshlets(
        self,
        meshlets: Sequence[Meshlet],
        object_position: Optional[Vec3] = None,
        enable_backface_cull: bool = True,
    ) -> list[int]:
        """
        Cull meshlets and return indices of visible ones.

        Args:
            meshlets: List of meshlets to cull
            object_position: World position of the object
            enable_backface_cull: Enable normal cone backface culling

        Returns:
            Indices of visible meshlets
        """
        if object_position is None:
            object_position = Vec3()

        visible: list[int] = []

        for i, meshlet in enumerate(meshlets):
            if enable_backface_cull and self.is_backface_culled(meshlet, object_position):
                continue
            visible.append(i)

        return visible


# =============================================================================
# MESHLET LOD SYSTEM
# =============================================================================


@dataclass
class MeshletLODLevel:
    """A single LOD level containing meshlets."""
    level: int
    meshlets: list[Meshlet] = field(default_factory=list)
    screen_size_threshold: float = 0.0  # Minimum screen size to use this LOD

    @property
    def triangle_count(self) -> int:
        return sum(m.triangle_count for m in self.meshlets)

    @property
    def meshlet_count(self) -> int:
        return len(self.meshlets)


class MeshletLODChain:
    """
    Chain of LOD levels for a mesh, each containing meshlets.

    Lower LOD levels have fewer/simpler meshlets.
    """

    def __init__(self) -> None:
        self._lod_levels: list[MeshletLODLevel] = []

    @property
    def lod_count(self) -> int:
        return len(self._lod_levels)

    def add_lod_level(
        self,
        meshlets: list[Meshlet],
        screen_size_threshold: float = 0.0,
    ) -> None:
        """Add a LOD level with its meshlets."""
        level = len(self._lod_levels)
        for meshlet in meshlets:
            meshlet.lod_level = level

        lod = MeshletLODLevel(
            level=level,
            meshlets=meshlets,
            screen_size_threshold=screen_size_threshold,
        )
        self._lod_levels.append(lod)

    def get_lod_level(self, level: int) -> Optional[MeshletLODLevel]:
        """Get a specific LOD level."""
        if 0 <= level < len(self._lod_levels):
            return self._lod_levels[level]
        return None

    def select_lod(self, screen_size: float) -> int:
        """Select appropriate LOD level based on screen size."""
        for i, lod in enumerate(self._lod_levels):
            if screen_size >= lod.screen_size_threshold:
                return i
        return len(self._lod_levels) - 1 if self._lod_levels else 0

    def get_all_meshlets(self) -> list[Meshlet]:
        """Get all meshlets from all LOD levels."""
        result: list[Meshlet] = []
        for lod in self._lod_levels:
            result.extend(lod.meshlets)
        return result


# =============================================================================
# MESHLET MESH
# =============================================================================


@dataclass
class MeshletMesh:
    """
    A mesh represented as a collection of meshlets.

    Contains the original vertex data and meshlet definitions.
    """
    # Original mesh data
    vertices: list[Vertex] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)

    # Meshlet data
    meshlets: list[Meshlet] = field(default_factory=list)

    # LOD chain (optional)
    lod_chain: Optional[MeshletLODChain] = None

    # Overall bounds
    bounds: BoundingSphere = field(default_factory=BoundingSphere)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3

    @property
    def meshlet_count(self) -> int:
        return len(self.meshlets)

    @classmethod
    def from_mesh_data(
        cls,
        vertices: Sequence[Vertex],
        indices: Sequence[int],
        config: Optional[MeshletBuildConfig] = None,
    ) -> "MeshletMesh":
        """
        Create a MeshletMesh from raw mesh data.

        Args:
            vertices: List of vertices
            indices: Triangle indices
            config: Meshlet building configuration

        Returns:
            New MeshletMesh with generated meshlets
        """
        mesh = cls(
            vertices=list(vertices),
            indices=list(indices),
        )

        # Build meshlets
        builder = MeshletBuilder(config)
        mesh.meshlets = builder.build(vertices, indices)

        # Compute overall bounds
        aabb = AABB()
        for vertex in vertices:
            aabb.expand(vertex.position)
        mesh.bounds = BoundingSphere.from_aabb(aabb)

        return mesh


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Constants
    "MeshletConstants",
    # Data structures
    "MeshletBounds",
    "Meshlet",
    "Vertex",
    # Builder
    "MeshletBuildConfig",
    "MeshletBuilder",
    # Culling
    "MeshletCuller",
    # LOD
    "MeshletLODLevel",
    "MeshletLODChain",
    # Mesh
    "MeshletMesh",
]
