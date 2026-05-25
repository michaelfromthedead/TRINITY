"""Deformable mesh handling for rendering.

This module provides utilities for connecting soft body simulation
to rendering, including:
- Embedded surface extraction from tetrahedral mesh
- Normal recomputation after deformation
- Skinning weights from tetrahedra
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence, Dict

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Type Aliases
# =============================================================================

Vector3 = NDArray[np.float64]  # Shape: (3,)
Matrix3x3 = NDArray[np.float64]  # Shape: (3, 3)
Matrix4x4 = NDArray[np.float64]  # Shape: (4, 4)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EmbeddedVertex:
    """A surface vertex embedded in a tetrahedron.

    Attributes:
        tet_index: Index of containing tetrahedron
        barycentric: Barycentric coordinates (4 values summing to 1)
        surface_index: Index in the surface mesh
    """
    tet_index: int
    barycentric: NDArray[np.float64]  # Shape: (4,)
    surface_index: int


@dataclass
class EmbeddedSurface:
    """A surface mesh embedded in a tetrahedral volume.

    The surface mesh can be higher resolution than the simulation mesh.
    Vertex positions are interpolated from the containing tetrahedra.

    Attributes:
        surface_vertices: Surface mesh vertices (N, 3)
        surface_triangles: Surface triangle indices (M, 3)
        surface_normals: Per-vertex normals (N, 3)
        embeddings: Embedding info for each surface vertex
        tet_mesh_vertices: Reference to tet mesh vertices
        tet_mesh_tetrahedra: Reference to tet mesh tetrahedra
    """
    surface_vertices: NDArray[np.float64]
    surface_triangles: NDArray[np.int32]
    surface_normals: NDArray[np.float64]
    embeddings: List[EmbeddedVertex]
    tet_mesh_vertices: Optional[NDArray[np.float64]] = None
    tet_mesh_tetrahedra: Optional[NDArray[np.int32]] = None

    @property
    def num_vertices(self) -> int:
        return len(self.surface_vertices)

    @property
    def num_triangles(self) -> int:
        return len(self.surface_triangles)


@dataclass
class TetSkinning:
    """Skinning data for deforming a surface from tet mesh.

    Attributes:
        surface_to_tet: Mapping from surface vertex to tet vertex
        weights: Skinning weights for each surface vertex
        influence_tets: Tetrahedra influencing each surface vertex
    """
    surface_to_tet: NDArray[np.int32]
    weights: NDArray[np.float64]
    influence_tets: List[List[int]]


# =============================================================================
# Helper Functions
# =============================================================================

def compute_barycentric_coords(
    point: Vector3,
    v0: Vector3,
    v1: Vector3,
    v2: Vector3,
    v3: Vector3
) -> NDArray[np.float64]:
    """Compute barycentric coordinates of a point in a tetrahedron.

    Args:
        point: Query point
        v0, v1, v2, v3: Tetrahedron vertices

    Returns:
        Barycentric coordinates (4,), sum to 1 if inside
    """
    # Vectors from v0
    vp = point - v0
    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0

    # 6x volume of full tet
    vol6 = np.dot(v10, np.cross(v20, v30))

    if abs(vol6) < 1e-10:
        # Degenerate tetrahedron
        return np.array([0.25, 0.25, 0.25, 0.25])

    # Barycentric coords via sub-tetrahedron volumes
    b1 = np.dot(vp, np.cross(v20, v30)) / vol6
    b2 = np.dot(v10, np.cross(vp, v30)) / vol6
    b3 = np.dot(v10, np.cross(v20, vp)) / vol6
    b0 = 1.0 - b1 - b2 - b3

    return np.array([b0, b1, b2, b3])


def point_in_tetrahedron(
    point: Vector3,
    v0: Vector3,
    v1: Vector3,
    v2: Vector3,
    v3: Vector3,
    epsilon: float = 1e-6
) -> bool:
    """Check if a point is inside a tetrahedron.

    Args:
        point: Query point
        v0, v1, v2, v3: Tetrahedron vertices
        epsilon: Tolerance for boundary cases

    Returns:
        True if point is inside (or on boundary within epsilon)
    """
    bary = compute_barycentric_coords(point, v0, v1, v2, v3)
    return np.all(bary >= -epsilon) and np.all(bary <= 1.0 + epsilon)


def compute_triangle_normal(
    v0: Vector3,
    v1: Vector3,
    v2: Vector3
) -> Vector3:
    """Compute normal of a triangle.

    Args:
        v0, v1, v2: Triangle vertices

    Returns:
        Normalized normal vector
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    length = np.linalg.norm(normal)

    if length < 1e-10:
        return np.array([0.0, 1.0, 0.0])

    return normal / length


def interpolate_position(
    tet_vertices: NDArray[np.float64],  # (4, 3)
    barycentric: NDArray[np.float64]  # (4,)
) -> Vector3:
    """Interpolate position using barycentric coordinates.

    Args:
        tet_vertices: Four tetrahedron vertices
        barycentric: Barycentric coordinates

    Returns:
        Interpolated position
    """
    return (
        barycentric[0] * tet_vertices[0] +
        barycentric[1] * tet_vertices[1] +
        barycentric[2] * tet_vertices[2] +
        barycentric[3] * tet_vertices[3]
    )


# =============================================================================
# Deformable Mesh Class
# =============================================================================

class DeformableMesh:
    """Deformable mesh for rendering soft body simulation.

    Manages a high-resolution surface mesh that is embedded in and
    deformed by a lower-resolution tetrahedral simulation mesh.

    Attributes:
        tet_vertices: Tetrahedral mesh vertices (simulation)
        tet_tetrahedra: Tetrahedron indices
        surface: Embedded surface mesh (rendering)
        normals_dirty: Whether normals need recomputation
    """

    def __init__(
        self,
        tet_vertices: NDArray[np.float64],
        tet_tetrahedra: NDArray[np.int32],
        surface_vertices: Optional[NDArray[np.float64]] = None,
        surface_triangles: Optional[NDArray[np.int32]] = None
    ):
        """Initialize deformable mesh.

        Args:
            tet_vertices: Tetrahedral mesh vertices, shape (N, 3)
            tet_tetrahedra: Tetrahedron indices, shape (M, 4)
            surface_vertices: Surface mesh vertices (optional)
            surface_triangles: Surface triangle indices (optional)
        """
        self.tet_vertices = tet_vertices.copy().astype(np.float64)
        self.tet_tetrahedra = tet_tetrahedra.astype(np.int32)

        # Build AABB tree for tet lookup (simplified: just store bounds)
        self._tet_bounds = self._compute_tet_bounds()

        # If no surface provided, extract from tet mesh
        if surface_vertices is None or surface_triangles is None:
            surface_vertices, surface_triangles = self._extract_surface()

        # Create embedded surface
        self.surface = self._embed_surface(surface_vertices, surface_triangles)
        self.normals_dirty = True

        # Update surface to initial positions
        self.update_surface_positions()
        self.update_normals()

    def _compute_tet_bounds(self) -> NDArray[np.float64]:
        """Compute axis-aligned bounds for each tetrahedron.

        Returns:
            Array of shape (M, 6) with [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        n_tets = len(self.tet_tetrahedra)
        bounds = np.zeros((n_tets, 6), dtype=np.float64)

        for i, tet in enumerate(self.tet_tetrahedra):
            verts = self.tet_vertices[tet]
            bounds[i, :3] = np.min(verts, axis=0)
            bounds[i, 3:] = np.max(verts, axis=0)

        return bounds

    def _extract_surface(self) -> Tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Extract surface triangles from tetrahedral mesh.

        Surface faces are those that appear in only one tetrahedron.

        Returns:
            Tuple of (vertices, triangles)
        """
        # Count face occurrences
        face_count: Dict[Tuple[int, ...], int] = {}
        face_tets: Dict[Tuple[int, ...], List[int]] = {}

        for ti, tet in enumerate(self.tet_tetrahedra):
            # Four faces per tetrahedron
            faces = [
                tuple(sorted([tet[0], tet[1], tet[2]])),
                tuple(sorted([tet[0], tet[1], tet[3]])),
                tuple(sorted([tet[0], tet[2], tet[3]])),
                tuple(sorted([tet[1], tet[2], tet[3]]))
            ]

            for face in faces:
                if face not in face_count:
                    face_count[face] = 0
                    face_tets[face] = []
                face_count[face] += 1
                face_tets[face].append(ti)

        # Extract surface faces (count == 1)
        surface_faces = [face for face, count in face_count.items() if count == 1]

        # Build surface triangles
        triangles = np.array(surface_faces, dtype=np.int32)

        # For now, surface vertices are just the tet vertices
        # (could deduplicate/reindex if needed)
        vertices = self.tet_vertices.copy()

        return vertices, triangles

    def _embed_surface(
        self,
        surface_vertices: NDArray[np.float64],
        surface_triangles: NDArray[np.int32]
    ) -> EmbeddedSurface:
        """Embed surface vertices in tetrahedral mesh.

        For each surface vertex, find containing tetrahedron and
        compute barycentric coordinates.

        Args:
            surface_vertices: Surface mesh vertices
            surface_triangles: Surface triangle indices

        Returns:
            EmbeddedSurface with embedding info
        """
        n_surface = len(surface_vertices)
        embeddings: List[EmbeddedVertex] = []

        for si in range(n_surface):
            point = surface_vertices[si]

            # Find containing tetrahedron
            tet_index, bary = self._find_containing_tet(point)

            embeddings.append(EmbeddedVertex(
                tet_index=tet_index,
                barycentric=bary,
                surface_index=si
            ))

        # Initialize normals
        normals = np.zeros_like(surface_vertices)

        return EmbeddedSurface(
            surface_vertices=surface_vertices.copy(),
            surface_triangles=surface_triangles.copy(),
            surface_normals=normals,
            embeddings=embeddings,
            tet_mesh_vertices=self.tet_vertices,
            tet_mesh_tetrahedra=self.tet_tetrahedra
        )

    def _find_containing_tet(
        self,
        point: Vector3
    ) -> Tuple[int, NDArray[np.float64]]:
        """Find tetrahedron containing a point.

        Args:
            point: Query point

        Returns:
            Tuple of (tet_index, barycentric_coords)
            Returns nearest tet if point is outside mesh
        """
        best_tet = 0
        best_distance = float('inf')
        best_bary = np.array([0.25, 0.25, 0.25, 0.25])

        for ti in range(len(self.tet_tetrahedra)):
            # Quick AABB check
            bounds = self._tet_bounds[ti]
            if not (bounds[0] <= point[0] <= bounds[3] and
                    bounds[1] <= point[1] <= bounds[4] and
                    bounds[2] <= point[2] <= bounds[5]):
                # Point outside tet bounds, compute distance to center
                center = (bounds[:3] + bounds[3:]) / 2
                dist = np.linalg.norm(point - center)
                if dist < best_distance:
                    best_distance = dist
                    best_tet = ti
                continue

            # Detailed check
            tet = self.tet_tetrahedra[ti]
            v0, v1, v2, v3 = self.tet_vertices[tet]

            bary = compute_barycentric_coords(point, v0, v1, v2, v3)

            # Check if inside
            if np.all(bary >= -1e-6) and np.all(bary <= 1.0 + 1e-6):
                # Clamp barycentric coords
                bary = np.clip(bary, 0.0, 1.0)
                bary /= np.sum(bary)  # Normalize
                return ti, bary

            # Track closest
            center = (v0 + v1 + v2 + v3) / 4
            dist = np.linalg.norm(point - center)
            if dist < best_distance:
                best_distance = dist
                best_tet = ti
                best_bary = bary

        # Point outside mesh, return nearest tet
        # Clamp barycentric to valid range
        best_bary = np.clip(best_bary, 0.0, 1.0)
        total = np.sum(best_bary)
        if total > 1e-10:
            best_bary /= total
        else:
            best_bary = np.array([0.25, 0.25, 0.25, 0.25])

        return best_tet, best_bary

    def embedded_surface(self) -> EmbeddedSurface:
        """Get the embedded surface mesh.

        Returns:
            EmbeddedSurface containing rendering data
        """
        return self.surface

    def update_surface_positions(self) -> None:
        """Update surface vertex positions from tet mesh.

        Call this after simulation updates tet_vertices.
        """
        for emb in self.surface.embeddings:
            tet = self.tet_tetrahedra[emb.tet_index]
            tet_verts = self.tet_vertices[tet]

            self.surface.surface_vertices[emb.surface_index] = interpolate_position(
                tet_verts, emb.barycentric
            )

        self.normals_dirty = True

    def update_normals(self) -> None:
        """Recompute surface normals.

        Uses angle-weighted vertex normals for smooth shading.
        """
        n_verts = self.surface.num_vertices
        normals = np.zeros((n_verts, 3), dtype=np.float64)

        for tri in self.surface.surface_triangles:
            v0 = self.surface.surface_vertices[tri[0]]
            v1 = self.surface.surface_vertices[tri[1]]
            v2 = self.surface.surface_vertices[tri[2]]

            # Face normal
            face_normal = compute_triangle_normal(v0, v1, v2)

            # Compute angles at each vertex
            e01 = v1 - v0
            e02 = v2 - v0
            e12 = v2 - v1
            e10 = -e01
            e20 = -e02
            e21 = -e12

            def angle_between(a: Vector3, b: Vector3) -> float:
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na < 1e-10 or nb < 1e-10:
                    return 0.0
                cos_angle = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
                return math.acos(cos_angle)

            angle0 = angle_between(e01, e02)
            angle1 = angle_between(e10, e12)
            angle2 = angle_between(e20, e21)

            # Add weighted contribution
            normals[tri[0]] += face_normal * angle0
            normals[tri[1]] += face_normal * angle1
            normals[tri[2]] += face_normal * angle2

        # Normalize
        for i in range(n_verts):
            length = np.linalg.norm(normals[i])
            if length > 1e-10:
                normals[i] /= length
            else:
                normals[i] = np.array([0.0, 1.0, 0.0])

        self.surface.surface_normals = normals
        self.normals_dirty = False

    def skinning_from_tets(self) -> TetSkinning:
        """Generate skinning weights from tet embeddings.

        Creates linear blend skinning weights for GPU skinning.

        Returns:
            TetSkinning with weights and influence info
        """
        n_surface = self.surface.num_vertices

        # Each surface vertex is influenced by 4 tet vertices
        surface_to_tet = np.zeros((n_surface, 4), dtype=np.int32)
        weights = np.zeros((n_surface, 4), dtype=np.float64)
        influence_tets: List[List[int]] = []

        for emb in self.surface.embeddings:
            tet = self.tet_tetrahedra[emb.tet_index]
            si = emb.surface_index

            surface_to_tet[si] = tet
            weights[si] = emb.barycentric
            influence_tets.append([emb.tet_index])

        return TetSkinning(
            surface_to_tet=surface_to_tet,
            weights=weights,
            influence_tets=influence_tets
        )

    def set_tet_vertices(self, vertices: NDArray[np.float64]) -> None:
        """Update tetrahedral mesh vertices.

        Args:
            vertices: New vertex positions
        """
        self.tet_vertices = vertices.copy()
        self._tet_bounds = self._compute_tet_bounds()
        self.update_surface_positions()

    def get_surface_vertices(self) -> NDArray[np.float64]:
        """Get current surface vertex positions.

        Returns:
            Surface vertices array
        """
        return self.surface.surface_vertices

    def get_surface_normals(self) -> NDArray[np.float64]:
        """Get current surface normals.

        Returns:
            Surface normals array
        """
        if self.normals_dirty:
            self.update_normals()
        return self.surface.surface_normals

    def get_surface_triangles(self) -> NDArray[np.int32]:
        """Get surface triangle indices.

        Returns:
            Triangle indices array
        """
        return self.surface.surface_triangles

    def compute_surface_area(self) -> float:
        """Compute total surface area.

        Returns:
            Total surface area
        """
        total = 0.0

        for tri in self.surface.surface_triangles:
            v0 = self.surface.surface_vertices[tri[0]]
            v1 = self.surface.surface_vertices[tri[1]]
            v2 = self.surface.surface_vertices[tri[2]]

            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            total += np.linalg.norm(cross) / 2.0

        return total

    def compute_bounding_box(self) -> Tuple[Vector3, Vector3]:
        """Compute axis-aligned bounding box of surface.

        Returns:
            Tuple of (min_corner, max_corner)
        """
        min_corner = np.min(self.surface.surface_vertices, axis=0)
        max_corner = np.max(self.surface.surface_vertices, axis=0)
        return min_corner, max_corner
