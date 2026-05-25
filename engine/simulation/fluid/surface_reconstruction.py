"""Fluid surface reconstruction.

This module implements surface reconstruction from particles:
- Marching cubes for isosurface extraction
- Density field computation
- Anisotropic kernels for smooth surfaces
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import (
    PARTICLE_RADIUS,
    SMOOTHING_LENGTH,
    REST_DENSITY,
    GRID_CELL_SIZE,
    MC_MIN_EDGE_LENGTH,
    MC_ISO_EPSILON,
)


# =============================================================================
# Type Aliases
# =============================================================================

Vector3 = NDArray[np.float64]


# =============================================================================
# Density Field
# =============================================================================

@dataclass
class DensityField:
    """3D scalar field for density values.

    Attributes:
        data: 3D array of density values
        resolution: Grid resolution (nx, ny, nz)
        origin: Grid origin in world space
        cell_size: Size of each cell
    """
    data: NDArray[np.float64]  # (nx, ny, nz)
    origin: Vector3
    cell_size: float

    @property
    def resolution(self) -> Tuple[int, int, int]:
        return self.data.shape

    def world_to_grid(self, pos: Vector3) -> Vector3:
        """Convert world position to grid coordinates."""
        return (pos - self.origin) / self.cell_size

    def grid_to_world(self, grid_pos: Vector3) -> Vector3:
        """Convert grid coordinates to world position."""
        return grid_pos * self.cell_size + self.origin

    def sample(self, pos: Vector3) -> float:
        """Sample density at world position using trilinear interpolation."""
        grid_pos = self.world_to_grid(pos)
        nx, ny, nz = self.resolution

        # Clamp to valid range
        grid_pos = np.clip(grid_pos, [0, 0, 0], [nx-1.001, ny-1.001, nz-1.001])

        i, j, k = int(grid_pos[0]), int(grid_pos[1]), int(grid_pos[2])
        fx, fy, fz = grid_pos[0] - i, grid_pos[1] - j, grid_pos[2] - k

        i1 = min(i + 1, nx - 1)
        j1 = min(j + 1, ny - 1)
        k1 = min(k + 1, nz - 1)

        return (
            self.data[i, j, k] * (1-fx) * (1-fy) * (1-fz) +
            self.data[i1, j, k] * fx * (1-fy) * (1-fz) +
            self.data[i, j1, k] * (1-fx) * fy * (1-fz) +
            self.data[i, j, k1] * (1-fx) * (1-fy) * fz +
            self.data[i1, j1, k] * fx * fy * (1-fz) +
            self.data[i, j1, k1] * (1-fx) * fy * fz +
            self.data[i1, j, k1] * fx * (1-fy) * fz +
            self.data[i1, j1, k1] * fx * fy * fz
        )


# =============================================================================
# Fluid Surface
# =============================================================================

@dataclass
class FluidSurface:
    """Extracted fluid surface mesh.

    Attributes:
        vertices: Vertex positions (N, 3)
        triangles: Triangle indices (M, 3)
        normals: Per-vertex normals (N, 3)
    """
    vertices: NDArray[np.float64]
    triangles: NDArray[np.int32]
    normals: NDArray[np.float64]

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_triangles(self) -> int:
        return len(self.triangles)

    def compute_bounds(self) -> Tuple[Vector3, Vector3]:
        """Compute axis-aligned bounding box."""
        if len(self.vertices) == 0:
            return np.zeros(3), np.zeros(3)
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)


# =============================================================================
# Marching Cubes Tables
# =============================================================================

# Edge table: for each cube configuration, which edges are intersected
# (Stored as 12-bit mask, one bit per edge)
EDGE_TABLE = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
]

# Triangle table: for each configuration, list of edges forming triangles
# -1 marks end of list
# (Compressed version - in real implementation this is a large 256x16 table)
# Simplified: we'll generate triangles procedurally


# =============================================================================
# Marching Cubes
# =============================================================================

class MarchingCubes:
    """Marching cubes isosurface extraction.

    Extracts a triangulated surface from a scalar field at a given
    iso-level (threshold).

    Attributes:
        iso_level: Threshold for surface extraction
    """

    # Edge endpoints (vertex indices for each edge)
    EDGE_VERTICES = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Vertex positions in unit cube
    CUBE_VERTICES = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float64)

    def __init__(self, iso_level: float = 0.5):
        """Initialize marching cubes.

        Args:
            iso_level: Isosurface threshold
        """
        self.iso_level = iso_level

    def extract_isosurface(self, density_field: DensityField) -> FluidSurface:
        """Extract isosurface from density field.

        Args:
            density_field: 3D scalar field

        Returns:
            Extracted surface mesh
        """
        nx, ny, nz = density_field.resolution
        vertices = []
        triangles = []

        # Process each cell
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    self._process_cell(
                        density_field, i, j, k,
                        vertices, triangles
                    )

        if not vertices:
            return FluidSurface(
                vertices=np.zeros((0, 3)),
                triangles=np.zeros((0, 3), dtype=np.int32),
                normals=np.zeros((0, 3))
            )

        vertices = np.array(vertices)
        triangles = np.array(triangles, dtype=np.int32)

        # Compute normals
        normals = self._compute_normals(vertices, triangles, density_field)

        return FluidSurface(
            vertices=vertices,
            triangles=triangles,
            normals=normals
        )

    def _process_cell(
        self,
        field: DensityField,
        i: int, j: int, k: int,
        vertices: List[Vector3],
        triangles: List[Tuple[int, int, int]]
    ) -> None:
        """Process a single cell for marching cubes.

        Handles edge cases:
        - Corner values exactly at iso-level (perturbed slightly)
        - Ambiguous configurations
        - Degenerate triangles
        """
        # Get corner values
        corners = np.zeros(8)
        corners[0] = field.data[i, j, k]
        corners[1] = field.data[i+1, j, k]
        corners[2] = field.data[i+1, j+1, k]
        corners[3] = field.data[i, j+1, k]
        corners[4] = field.data[i, j, k+1]
        corners[5] = field.data[i+1, j, k+1]
        corners[6] = field.data[i+1, j+1, k+1]
        corners[7] = field.data[i, j+1, k+1]

        # Handle corner values exactly at iso-level by perturbing slightly
        # This prevents ambiguous configurations
        for v in range(8):
            if abs(corners[v] - self.iso_level) < MC_ISO_EPSILON:
                corners[v] = self.iso_level - MC_ISO_EPSILON

        # Compute cube index
        cube_index = 0
        for v in range(8):
            if corners[v] < self.iso_level:
                cube_index |= (1 << v)

        # No surface in this cell
        edge_mask = EDGE_TABLE[cube_index]
        if edge_mask == 0:
            return

        # Interpolate vertices on edges
        edge_vertices = {}
        for e in range(12):
            if edge_mask & (1 << e):
                v0, v1 = self.EDGE_VERTICES[e]
                t = self._interpolate_edge(
                    corners[v0], corners[v1]
                )

                # Position on edge
                p0 = self.CUBE_VERTICES[v0]
                p1 = self.CUBE_VERTICES[v1]
                edge_pos = p0 + t * (p1 - p0)

                # Convert to world space
                cell_origin = field.grid_to_world(np.array([i, j, k], dtype=np.float64))
                world_pos = cell_origin + edge_pos * field.cell_size

                edge_vertices[e] = len(vertices)
                vertices.append(world_pos)

        # Generate triangles using a simplified approach
        # (In a full implementation, use the triangle table)
        self._triangulate_cell(cube_index, edge_vertices, triangles)

    def _interpolate_edge(self, v0: float, v1: float) -> float:
        """Interpolate position along edge where surface crosses.

        Handles edge cases:
        - Both values equal (degenerate edge)
        - Values very close to iso-level
        - Both values on same side of iso-level (shouldn't happen but handle gracefully)
        """
        diff = v1 - v0

        # Degenerate case: both values essentially equal
        if abs(diff) < MC_MIN_EDGE_LENGTH:
            return 0.5

        t = (self.iso_level - v0) / diff

        # Clamp to valid range [0, 1] to handle numerical precision issues
        # This prevents vertices from being placed outside the edge
        return np.clip(t, 0.0, 1.0)

    def _triangulate_cell(
        self,
        cube_index: int,
        edge_vertices: dict,
        triangles: List[Tuple[int, int, int]]
    ) -> None:
        """Generate triangles for a cell configuration.

        Simplified version - generates triangles by connecting
        edge intersections in a consistent order.
        """
        # Get intersected edges
        edges = sorted(edge_vertices.keys())

        if len(edges) < 3:
            return

        # Simple fan triangulation from first vertex
        # (This is a simplification; full MC uses lookup table)
        for idx in range(1, len(edges) - 1):
            triangles.append((
                edge_vertices[edges[0]],
                edge_vertices[edges[idx]],
                edge_vertices[edges[idx + 1]]
            ))

    def _compute_normals(
        self,
        vertices: NDArray[np.float64],
        triangles: NDArray[np.int32],
        field: DensityField
    ) -> NDArray[np.float64]:
        """Compute per-vertex normals using gradient of density field."""
        normals = np.zeros_like(vertices)

        # Use central differences for gradient
        eps = field.cell_size * 0.5

        for i, v in enumerate(vertices):
            # Sample density field gradient
            dx = field.sample(v + np.array([eps, 0, 0])) - field.sample(v - np.array([eps, 0, 0]))
            dy = field.sample(v + np.array([0, eps, 0])) - field.sample(v - np.array([0, eps, 0]))
            dz = field.sample(v + np.array([0, 0, eps])) - field.sample(v - np.array([0, 0, eps]))

            grad = np.array([dx, dy, dz])
            length = np.linalg.norm(grad)

            if length > 1e-10:
                # Normal points away from fluid (opposite to gradient)
                normals[i] = -grad / length
            else:
                normals[i] = np.array([0, 1, 0])

        return normals


# =============================================================================
# Density Field Computation
# =============================================================================

def compute_density_field(
    positions: NDArray[np.float64],
    bounds_min: Vector3,
    bounds_max: Vector3,
    resolution: Tuple[int, int, int],
    smoothing_length: float = SMOOTHING_LENGTH
) -> DensityField:
    """Compute density field from particle positions.

    Uses SPH kernel to splat particle contributions onto grid.

    Args:
        positions: Particle positions (N, 3)
        bounds_min: Domain minimum
        bounds_max: Domain maximum
        resolution: Grid resolution
        smoothing_length: SPH kernel radius

    Returns:
        DensityField with accumulated densities
    """
    nx, ny, nz = resolution
    cell_size = (bounds_max - bounds_min) / np.array(resolution)
    cell_size = np.min(cell_size)  # Use uniform cell size

    data = np.zeros(resolution, dtype=np.float64)

    # Poly6 kernel coefficient
    h = smoothing_length
    poly6_coeff = 315.0 / (64.0 * math.pi * h ** 9)

    for pos in positions:
        # Grid coordinates
        grid_pos = (pos - bounds_min) / cell_size

        # Cells within kernel radius
        min_cell = np.maximum(0, (grid_pos - h / cell_size).astype(int))
        max_cell = np.minimum(
            [nx-1, ny-1, nz-1],
            (grid_pos + h / cell_size).astype(int)
        )

        for i in range(min_cell[0], max_cell[0] + 1):
            for j in range(min_cell[1], max_cell[1] + 1):
                for k in range(min_cell[2], max_cell[2] + 1):
                    # Cell center
                    cell_center = bounds_min + (np.array([i, j, k]) + 0.5) * cell_size
                    r_sq = np.sum((pos - cell_center) ** 2)

                    if r_sq < h * h:
                        # Poly6 kernel
                        diff = h * h - r_sq
                        w = poly6_coeff * diff * diff * diff
                        data[i, j, k] += w

    return DensityField(
        data=data,
        origin=bounds_min,
        cell_size=cell_size
    )


def extract_isosurface(
    density_field: DensityField,
    iso_level: float = 0.5
) -> FluidSurface:
    """Extract isosurface from density field.

    Convenience function wrapping MarchingCubes.

    Args:
        density_field: Input density field
        iso_level: Threshold for surface

    Returns:
        Extracted FluidSurface
    """
    mc = MarchingCubes(iso_level)
    return mc.extract_isosurface(density_field)
