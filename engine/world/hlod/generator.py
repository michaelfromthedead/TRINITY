"""
HLOD (Hierarchical Level of Detail) Mesh Generation.

Provides mesh generation methods for HLOD proxies including mesh merging,
simplification using edge collapse, impostor/billboard generation, and
proxy mesh generation.

Pipeline: Source Meshes -> Select Method -> Generate HLOD Mesh -> Output

Known Limitations:
- Impostor capture uses simplified CPU rasterization (not GPU-accelerated)
- Edge collapse simplification does not preserve UV seams perfectly
- Very thin or degenerate triangles may cause numerical instability
- Billboard impostors have limited angular fidelity (depends on view count)

References:
- WORLD_CONTEXT.md Section 7 HLOD System
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import heapq


# =============================================================================
# HLOD CONSTANTS
# =============================================================================

# Import centralized constants
from .constants import (
    FloatingPointConstants,
    SimplificationConstants,
    MergeConstants,
    ImpostorConstants,
    MethodSelectionConstants,
)


class HLODConstants:
    """Constants used throughout the HLOD system."""
    # Floating point comparison epsilon
    EPSILON: float = FloatingPointConstants.EPSILON

    # Default simplification settings
    DEFAULT_TARGET_RATIO: float = SimplificationConstants.DEFAULT_TARGET_RATIO
    DEFAULT_MAX_ERROR: float = SimplificationConstants.DEFAULT_MAX_ERROR

    # Default impostor settings
    DEFAULT_IMPOSTOR_RESOLUTION: int = ImpostorConstants.DEFAULT_RESOLUTION
    DEFAULT_IMPOSTOR_VIEW_COUNT: int = ImpostorConstants.DEFAULT_VIEW_COUNT

    # Mesh merging thresholds
    DEFAULT_MERGE_DISTANCE: float = MergeConstants.DEFAULT_MERGE_DISTANCE
    DEFAULT_UV_TOLERANCE: float = MergeConstants.DEFAULT_UV_TOLERANCE

    # Edge collapse parameters
    EDGE_COLLAPSE_WEIGHT_POSITION: float = SimplificationConstants.EDGE_COLLAPSE_WEIGHT_POSITION
    EDGE_COLLAPSE_WEIGHT_NORMAL: float = SimplificationConstants.EDGE_COLLAPSE_WEIGHT_NORMAL
    EDGE_COLLAPSE_WEIGHT_UV: float = SimplificationConstants.EDGE_COLLAPSE_WEIGHT_UV

    # Interior face detection (extracted magic numbers)
    OPPOSING_NORMAL_THRESHOLD: float = MergeConstants.OPPOSING_NORMAL_THRESHOLD
    INTERIOR_FACE_DISTANCE_MULTIPLIER: float = MergeConstants.INTERIOR_FACE_DISTANCE_MULTIPLIER

    # Method selection thresholds
    SMALL_MESH_TRIANGLE_THRESHOLD: int = MethodSelectionConstants.SMALL_MESH_TRIANGLE_THRESHOLD
    MEDIUM_MESH_TRIANGLE_THRESHOLD: int = MethodSelectionConstants.MEDIUM_MESH_TRIANGLE_THRESHOLD
    LARGE_MESH_TRIANGLE_THRESHOLD: int = MethodSelectionConstants.LARGE_MESH_TRIANGLE_THRESHOLD
    MANY_MESHES_THRESHOLD: int = MethodSelectionConstants.MANY_MESHES_THRESHOLD

    # Impostor generation
    HEMI_OCTAHEDRON_Y_ELEVATION: float = ImpostorConstants.HEMI_OCTAHEDRON_Y_ELEVATION

    # Simplification limits
    MIN_TRIANGLES: int = SimplificationConstants.MIN_TRIANGLES
    MIN_TRIANGLE_AREA: float = SimplificationConstants.MIN_TRIANGLE_AREA


# =============================================================================
# HLOD GENERATION METHOD ENUM
# =============================================================================


class HLODGenerationMethod(Enum):
    """Methods for generating HLOD meshes."""
    MESH_MERGING = auto()       # Combine static meshes
    SIMPLIFICATION = auto()     # Reduce polygon count via edge collapse
    IMPOSTOR = auto()           # Billboard capture from multiple views
    PROXY_MESH = auto()         # Simplified bounding geometry


# =============================================================================
# SETTINGS DATA CLASSES
# =============================================================================


@dataclass
class SimplificationSettings:
    """Settings for mesh simplification."""
    target_ratio: float = HLODConstants.DEFAULT_TARGET_RATIO  # Target triangle ratio
    max_error: float = HLODConstants.DEFAULT_MAX_ERROR        # Maximum geometric error
    preserve_borders: bool = True                              # Preserve mesh borders
    preserve_uvs: bool = True                                  # Preserve UV seams
    lock_vertices: List[int] = field(default_factory=list)    # Locked vertex indices

    def __post_init__(self) -> None:
        """Validate settings."""
        if not 0.0 < self.target_ratio <= 1.0:
            raise ValueError("target_ratio must be in (0, 1]")
        if self.max_error < 0.0:
            raise ValueError("max_error must be non-negative")


@dataclass
class ImpostorSettings:
    """Settings for impostor/billboard generation."""
    resolution: int = HLODConstants.DEFAULT_IMPOSTOR_RESOLUTION  # Texture resolution
    view_count: int = HLODConstants.DEFAULT_IMPOSTOR_VIEW_COUNT  # Number of views
    capture_normals: bool = True                                  # Capture normal map
    capture_depth: bool = True                                    # Capture depth map
    hemi_octahedron: bool = False                                # Use hemi-octahedron map

    def __post_init__(self) -> None:
        """Validate settings."""
        if self.resolution < 1:
            raise ValueError("resolution must be positive")
        if self.view_count < 1:
            raise ValueError("view_count must be positive")


@dataclass
class MergeSettings:
    """Settings for mesh merging."""
    remove_interior_faces: bool = True       # Remove faces inside merged mesh
    merge_distance: float = HLODConstants.DEFAULT_MERGE_DISTANCE  # Vertex merge distance
    preserve_materials: bool = True          # Keep material assignments

    def __post_init__(self) -> None:
        """Validate settings."""
        if self.merge_distance < 0.0:
            raise ValueError("merge_distance must be non-negative")


# =============================================================================
# MATH TYPES
# =============================================================================


@dataclass(slots=True)
class Vec3:
    """3D vector for positions and directions."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vec3":
        if abs(scalar) < HLODConstants.EPSILON:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec3):
            return False
        return (
            abs(self.x - other.x) < HLODConstants.EPSILON
            and abs(self.y - other.y) < HLODConstants.EPSILON
            and abs(self.z - other.z) < HLODConstants.EPSILON
        )

    def __hash__(self) -> int:
        return hash((round(self.x, 6), round(self.y, 6), round(self.z, 6)))

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def length_squared(self) -> float:
        return self.dot(self)

    def normalized(self) -> "Vec3":
        length = self.length()
        if length < HLODConstants.EPSILON:
            return Vec3(0.0, 0.0, 0.0)
        return self / length

    def distance_to(self, other: "Vec3") -> float:
        return (self - other).length()

    def lerp(self, other: "Vec3", t: float) -> "Vec3":
        """Linear interpolation between self and other."""
        return self * (1.0 - t) + other * t

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(slots=True)
class Vec2:
    """2D vector for UV coordinates."""
    u: float = 0.0
    v: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.u + other.u, self.v + other.v)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.u - other.u, self.v - other.v)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.u * scalar, self.v * scalar)

    def __truediv__(self, scalar: float) -> "Vec2":
        if abs(scalar) < HLODConstants.EPSILON:
            return Vec2(0.0, 0.0)
        return Vec2(self.u / scalar, self.v / scalar)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec2):
            return False
        return (
            abs(self.u - other.u) < HLODConstants.EPSILON
            and abs(self.v - other.v) < HLODConstants.EPSILON
        )

    def __hash__(self) -> int:
        return hash((round(self.u, 6), round(self.v, 6)))

    def lerp(self, other: "Vec2", t: float) -> "Vec2":
        """Linear interpolation between self and other."""
        return self * (1.0 - t) + other * t

    def to_tuple(self) -> Tuple[float, float]:
        return (self.u, self.v)


@dataclass
class AABB:
    """Axis-Aligned Bounding Box."""
    min_point: Vec3 = field(default_factory=lambda: Vec3(float("inf"), float("inf"), float("inf")))
    max_point: Vec3 = field(default_factory=lambda: Vec3(float("-inf"), float("-inf"), float("-inf")))

    @property
    def center(self) -> Vec3:
        return Vec3(
            (self.min_point.x + self.max_point.x) * 0.5,
            (self.min_point.y + self.max_point.y) * 0.5,
            (self.min_point.z + self.max_point.z) * 0.5,
        )

    @property
    def extents(self) -> Vec3:
        """Half-extents from center to corner."""
        return Vec3(
            (self.max_point.x - self.min_point.x) * 0.5,
            (self.max_point.y - self.min_point.y) * 0.5,
            (self.max_point.z - self.min_point.z) * 0.5,
        )

    @property
    def size(self) -> Vec3:
        """Full size of the bounding box."""
        return Vec3(
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z,
        )

    def expand(self, point: Vec3) -> None:
        """Expand AABB to include the given point."""
        self.min_point = Vec3(
            min(self.min_point.x, point.x),
            min(self.min_point.y, point.y),
            min(self.min_point.z, point.z),
        )
        self.max_point = Vec3(
            max(self.max_point.x, point.x),
            max(self.max_point.y, point.y),
            max(self.max_point.z, point.z),
        )

    def merge(self, other: "AABB") -> "AABB":
        """Create new AABB containing both this and other."""
        return AABB(
            min_point=Vec3(
                min(self.min_point.x, other.min_point.x),
                min(self.min_point.y, other.min_point.y),
                min(self.min_point.z, other.min_point.z),
            ),
            max_point=Vec3(
                max(self.max_point.x, other.max_point.x),
                max(self.max_point.y, other.max_point.y),
                max(self.max_point.z, other.max_point.z),
            ),
        )

    def is_valid(self) -> bool:
        """Check if AABB is valid (has been expanded at least once)."""
        return (
            self.min_point.x <= self.max_point.x
            and self.min_point.y <= self.max_point.y
            and self.min_point.z <= self.max_point.z
        )


# =============================================================================
# MESH DATA STRUCTURES
# =============================================================================


@dataclass
class MeshData:
    """Mesh data container for HLOD generation."""
    vertices: List[Vec3] = field(default_factory=list)
    normals: List[Vec3] = field(default_factory=list)
    uvs: List[Vec2] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    bounds: AABB = field(default_factory=AABB)
    material_id: int = 0

    def get_triangle_count(self) -> int:
        """Get the number of triangles in the mesh."""
        return len(self.indices) // 3

    def get_vertex_count(self) -> int:
        """Get the number of vertices in the mesh."""
        return len(self.vertices)

    def is_valid(self) -> bool:
        """Check if mesh data is valid."""
        if not self.indices:
            return True  # Empty mesh is valid

        # Index count must be multiple of 3
        if len(self.indices) % 3 != 0:
            return False

        # All indices must be in bounds
        vertex_count = len(self.vertices)
        for idx in self.indices:
            if idx < 0 or idx >= vertex_count:
                return False

        # Normals and UVs must match vertex count if present
        if self.normals and len(self.normals) != vertex_count:
            return False
        if self.uvs and len(self.uvs) != vertex_count:
            return False

        return True

    def compute_bounds(self) -> None:
        """Compute bounding box from vertices."""
        self.bounds = AABB()
        for vertex in self.vertices:
            self.bounds.expand(vertex)

    def merge_with(self, other: "MeshData") -> "MeshData":
        """Merge this mesh with another, returning a new combined mesh."""
        result = MeshData()

        # Copy vertices from self
        result.vertices = list(self.vertices)
        result.normals = list(self.normals)
        result.uvs = list(self.uvs)
        result.indices = list(self.indices)

        # Offset for other's indices
        offset = len(self.vertices)

        # Add vertices from other
        result.vertices.extend(other.vertices)
        result.normals.extend(other.normals)
        result.uvs.extend(other.uvs)

        # Add indices from other with offset
        for idx in other.indices:
            result.indices.append(idx + offset)

        # Merge bounds
        result.bounds = self.bounds.merge(other.bounds)

        return result

    def get_triangle(self, tri_index: int) -> Tuple[int, int, int]:
        """Get indices of a triangle by triangle index."""
        base = tri_index * 3
        return (self.indices[base], self.indices[base + 1], self.indices[base + 2])

    def compute_triangle_normal(self, tri_index: int) -> Vec3:
        """Compute the normal of a triangle."""
        i0, i1, i2 = self.get_triangle(tri_index)
        v0, v1, v2 = self.vertices[i0], self.vertices[i1], self.vertices[i2]

        edge1 = v1 - v0
        edge2 = v2 - v0

        normal = edge1.cross(edge2)
        return normal.normalized()

    def compute_triangle_area(self, tri_index: int) -> float:
        """Compute the area of a triangle."""
        i0, i1, i2 = self.get_triangle(tri_index)
        v0, v1, v2 = self.vertices[i0], self.vertices[i1], self.vertices[i2]

        edge1 = v1 - v0
        edge2 = v2 - v0

        cross = edge1.cross(edge2)
        return cross.length() * 0.5


@dataclass
class HLODMeshData:
    """Extended mesh data for HLOD with additional metadata."""
    mesh: MeshData = field(default_factory=MeshData)
    source_mesh_count: int = 0
    original_triangle_count: int = 0
    method_used: HLODGenerationMethod = HLODGenerationMethod.MESH_MERGING
    generation_time_ms: float = 0.0

    @property
    def reduction_ratio(self) -> float:
        """Calculate triangle reduction ratio."""
        if self.original_triangle_count == 0:
            return 0.0
        return 1.0 - (self.mesh.get_triangle_count() / self.original_triangle_count)


@dataclass
class ImpostorData:
    """Data for billboard impostor."""
    albedo_atlas: List[List[Tuple[float, float, float, float]]] = field(default_factory=list)
    normal_atlas: Optional[List[List[Tuple[float, float, float]]]] = None
    depth_atlas: Optional[List[List[float]]] = None
    view_directions: List[Vec3] = field(default_factory=list)
    resolution: int = 512
    bounds: AABB = field(default_factory=AABB)


# =============================================================================
# MESH MERGER
# =============================================================================


class MeshMerger:
    """Merges multiple meshes into a single mesh."""

    def __init__(self, settings: Optional[MergeSettings] = None) -> None:
        self._settings = settings or MergeSettings()

    @property
    def settings(self) -> MergeSettings:
        return self._settings

    def merge_meshes(self, meshes: List[MeshData]) -> MeshData:
        """
        Combine multiple meshes into one.

        Args:
            meshes: List of meshes to merge

        Returns:
            Combined mesh with remapped indices
        """
        if not meshes:
            return MeshData()

        if len(meshes) == 1:
            return self._copy_mesh(meshes[0])

        result = MeshData()

        for mesh in meshes:
            if not mesh.vertices:
                continue

            # Current offset for indices
            offset = len(result.vertices)

            # Add vertices
            result.vertices.extend(mesh.vertices)
            result.normals.extend(mesh.normals if mesh.normals else [Vec3(0, 1, 0)] * len(mesh.vertices))
            result.uvs.extend(mesh.uvs if mesh.uvs else [Vec2(0, 0)] * len(mesh.vertices))

            # Add indices with offset
            for idx in mesh.indices:
                result.indices.append(idx + offset)

        # Compute combined bounds
        result.compute_bounds()

        # Optionally weld vertices that are very close
        if self._settings.merge_distance > 0:
            result = self._weld_vertices(result)

        # Optionally remove interior faces
        if self._settings.remove_interior_faces:
            result = self._remove_interior_faces(result)

        return result

    def _copy_mesh(self, mesh: MeshData) -> MeshData:
        """Create a deep copy of a mesh."""
        return MeshData(
            vertices=list(mesh.vertices),
            normals=list(mesh.normals),
            uvs=list(mesh.uvs),
            indices=list(mesh.indices),
            bounds=AABB(
                min_point=Vec3(mesh.bounds.min_point.x, mesh.bounds.min_point.y, mesh.bounds.min_point.z),
                max_point=Vec3(mesh.bounds.max_point.x, mesh.bounds.max_point.y, mesh.bounds.max_point.z),
            ),
            material_id=mesh.material_id,
        )

    def _weld_vertices(self, mesh: MeshData) -> MeshData:
        """Weld vertices that are within merge_distance of each other."""
        if not mesh.vertices:
            return mesh

        distance = self._settings.merge_distance

        # Map from old index to new index
        index_map: Dict[int, int] = {}
        new_vertices: List[Vec3] = []
        new_normals: List[Vec3] = []
        new_uvs: List[Vec2] = []

        for i, vertex in enumerate(mesh.vertices):
            # Check if this vertex is close to an existing one
            found = False
            for j, new_vertex in enumerate(new_vertices):
                if vertex.distance_to(new_vertex) < distance:
                    index_map[i] = j
                    found = True
                    break

            if not found:
                index_map[i] = len(new_vertices)
                new_vertices.append(vertex)
                if mesh.normals:
                    new_normals.append(mesh.normals[i])
                if mesh.uvs:
                    new_uvs.append(mesh.uvs[i])

        # Remap indices
        new_indices: List[int] = []
        for idx in mesh.indices:
            new_indices.append(index_map[idx])

        result = MeshData(
            vertices=new_vertices,
            normals=new_normals,
            uvs=new_uvs,
            indices=new_indices,
            bounds=mesh.bounds,
            material_id=mesh.material_id,
        )

        return result

    def _remove_interior_faces(self, mesh: MeshData) -> MeshData:
        """Remove triangles that are completely inside the merged mesh."""
        if mesh.get_triangle_count() < 2:
            return mesh

        # Simple heuristic: remove triangles with opposing faces very close
        # This is a simplified implementation
        triangles_to_keep: List[int] = []

        for tri_idx in range(mesh.get_triangle_count()):
            normal = mesh.compute_triangle_normal(tri_idx)
            i0, i1, i2 = mesh.get_triangle(tri_idx)
            center = (mesh.vertices[i0] + mesh.vertices[i1] + mesh.vertices[i2]) / 3.0

            is_interior = False

            # Check if there's an opposing triangle nearby
            for other_idx in range(mesh.get_triangle_count()):
                if other_idx == tri_idx:
                    continue

                other_normal = mesh.compute_triangle_normal(other_idx)
                j0, j1, j2 = mesh.get_triangle(other_idx)
                other_center = (mesh.vertices[j0] + mesh.vertices[j1] + mesh.vertices[j2]) / 3.0

                # Check if normals are opposing and centers are very close
                dot = normal.dot(other_normal)
                dist = center.distance_to(other_center)

                if (dot < HLODConstants.OPPOSING_NORMAL_THRESHOLD and
                    dist < self._settings.merge_distance * HLODConstants.INTERIOR_FACE_DISTANCE_MULTIPLIER):
                    is_interior = True
                    break

            if not is_interior:
                triangles_to_keep.append(tri_idx)

        if len(triangles_to_keep) == mesh.get_triangle_count():
            return mesh

        # Build new mesh with kept triangles
        new_indices: List[int] = []
        for tri_idx in triangles_to_keep:
            i0, i1, i2 = mesh.get_triangle(tri_idx)
            new_indices.extend([i0, i1, i2])

        return MeshData(
            vertices=mesh.vertices,
            normals=mesh.normals,
            uvs=mesh.uvs,
            indices=new_indices,
            bounds=mesh.bounds,
            material_id=mesh.material_id,
        )


# =============================================================================
# MESH SIMPLIFIER (EDGE COLLAPSE)
# =============================================================================


@dataclass
class Edge:
    """Edge in the mesh for edge collapse algorithm."""
    v0: int
    v1: int
    cost: float = 0.0
    collapse_point: Vec3 = field(default_factory=Vec3)

    def __lt__(self, other: "Edge") -> bool:
        return self.cost < other.cost

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return False
        return (self.v0, self.v1) == (other.v0, other.v1) or (self.v0, self.v1) == (other.v1, other.v0)

    def __hash__(self) -> int:
        return hash((min(self.v0, self.v1), max(self.v0, self.v1)))


class MeshSimplifier:
    """
    Mesh simplification using edge collapse algorithm.

    Based on quadric error metrics (QEM) for determining edge collapse cost.
    """

    def __init__(self, settings: Optional[SimplificationSettings] = None) -> None:
        self._settings = settings or SimplificationSettings()
        self._mesh: Optional[MeshData] = None
        self._vertex_quadrics: List[List[List[float]]] = []
        self._edges: List[Edge] = []
        self._edge_heap: List[Edge] = []
        self._vertex_edges: List[List[Edge]] = []
        self._removed_vertices: List[bool] = []
        self._removed_triangles: List[bool] = []

    @property
    def settings(self) -> SimplificationSettings:
        return self._settings

    def simplify(self, mesh: MeshData) -> MeshData:
        """
        Simplify mesh using edge collapse algorithm.

        Args:
            mesh: Input mesh to simplify

        Returns:
            Simplified mesh
        """
        if mesh.get_triangle_count() == 0:
            return MeshData()

        # Copy mesh for modification
        self._mesh = MeshData(
            vertices=list(mesh.vertices),
            normals=list(mesh.normals) if mesh.normals else [],
            uvs=list(mesh.uvs) if mesh.uvs else [],
            indices=list(mesh.indices),
            bounds=mesh.bounds,
            material_id=mesh.material_id,
        )

        target_triangles = max(1, int(mesh.get_triangle_count() * self._settings.target_ratio))

        # Initialize data structures
        self._init_quadrics()
        self._init_edges()

        # Perform edge collapse until target reached
        current_triangles = mesh.get_triangle_count()

        while current_triangles > target_triangles and self._edge_heap:
            edge = heapq.heappop(self._edge_heap)

            # Skip if vertices already removed
            if self._removed_vertices[edge.v0] or self._removed_vertices[edge.v1]:
                continue

            # Check error threshold
            if edge.cost > self._settings.max_error:
                break

            # Check if vertex is locked
            if edge.v0 in self._settings.lock_vertices or edge.v1 in self._settings.lock_vertices:
                continue

            # Perform collapse
            triangles_removed = self._collapse_edge(edge)
            current_triangles -= triangles_removed

        # Build output mesh
        return self._build_output_mesh()

    def _init_quadrics(self) -> None:
        """Initialize quadric error matrices for all vertices."""
        if not self._mesh:
            return

        vertex_count = len(self._mesh.vertices)
        self._vertex_quadrics = [self._zero_quadric() for _ in range(vertex_count)]
        self._removed_vertices = [False] * vertex_count
        self._removed_triangles = [False] * self._mesh.get_triangle_count()

        # For each triangle, add its plane equation to vertex quadrics
        for tri_idx in range(self._mesh.get_triangle_count()):
            i0, i1, i2 = self._mesh.get_triangle(tri_idx)
            v0, v1, v2 = self._mesh.vertices[i0], self._mesh.vertices[i1], self._mesh.vertices[i2]

            # Compute plane equation ax + by + cz + d = 0
            normal = (v1 - v0).cross(v2 - v0).normalized()
            d = -normal.dot(v0)

            # Create plane quadric
            plane_quadric = self._plane_quadric(normal.x, normal.y, normal.z, d)

            # Add to vertex quadrics
            self._add_quadric(self._vertex_quadrics[i0], plane_quadric)
            self._add_quadric(self._vertex_quadrics[i1], plane_quadric)
            self._add_quadric(self._vertex_quadrics[i2], plane_quadric)

    def _init_edges(self) -> None:
        """Initialize edge list and heap."""
        if not self._mesh:
            return

        vertex_count = len(self._mesh.vertices)
        self._vertex_edges = [[] for _ in range(vertex_count)]
        edge_set: set = set()

        # Collect all unique edges
        for tri_idx in range(self._mesh.get_triangle_count()):
            i0, i1, i2 = self._mesh.get_triangle(tri_idx)

            for v0, v1 in [(i0, i1), (i1, i2), (i2, i0)]:
                edge_key = (min(v0, v1), max(v0, v1))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edge = Edge(v0=edge_key[0], v1=edge_key[1])
                    self._compute_edge_cost(edge)
                    self._edges.append(edge)
                    self._vertex_edges[v0].append(edge)
                    self._vertex_edges[v1].append(edge)

        # Build heap
        self._edge_heap = list(self._edges)
        heapq.heapify(self._edge_heap)

    def _compute_edge_cost(self, edge: Edge) -> None:
        """Compute the cost of collapsing an edge using quadric error metrics."""
        if not self._mesh:
            return

        q0 = self._vertex_quadrics[edge.v0]
        q1 = self._vertex_quadrics[edge.v1]

        # Combined quadric
        q_sum = self._zero_quadric()
        self._add_quadric(q_sum, q0)
        self._add_quadric(q_sum, q1)

        # Find optimal collapse point
        v0 = self._mesh.vertices[edge.v0]
        v1 = self._mesh.vertices[edge.v1]

        # Try midpoint as collapse position
        midpoint = v0.lerp(v1, 0.5)
        edge.collapse_point = midpoint

        # Compute error at collapse point
        edge.cost = self._evaluate_quadric(q_sum, midpoint)

    def _collapse_edge(self, edge: Edge) -> int:
        """
        Collapse an edge by merging v1 into v0.

        Returns number of triangles removed.
        """
        if not self._mesh:
            return 0

        v0, v1 = edge.v0, edge.v1

        # Validate vertex indices are in bounds
        vertex_count = len(self._mesh.vertices)
        if v0 < 0 or v0 >= vertex_count or v1 < 0 or v1 >= vertex_count:
            return 0

        # Skip if either vertex is already removed
        if self._removed_vertices[v0] or self._removed_vertices[v1]:
            return 0

        # Move v0 to collapse point
        self._mesh.vertices[v0] = edge.collapse_point

        # Update normal if available (average)
        if self._mesh.normals:
            n0 = self._mesh.normals[v0]
            n1 = self._mesh.normals[v1]
            self._mesh.normals[v0] = n0.lerp(n1, 0.5).normalized()

        # Update UV if available (average)
        if self._mesh.uvs:
            uv0 = self._mesh.uvs[v0]
            uv1 = self._mesh.uvs[v1]
            self._mesh.uvs[v0] = uv0.lerp(uv1, 0.5)

        # Mark v1 as removed
        self._removed_vertices[v1] = True

        # Update quadric for v0
        self._add_quadric(self._vertex_quadrics[v0], self._vertex_quadrics[v1])

        # Find and remove degenerate triangles
        triangles_removed = 0
        for tri_idx in range(self._mesh.get_triangle_count()):
            if self._removed_triangles[tri_idx]:
                continue

            i0, i1, i2 = self._mesh.get_triangle(tri_idx)

            # Check if triangle uses edge
            uses_edge = (v0 in (i0, i1, i2)) and (v1 in (i0, i1, i2))

            if uses_edge:
                # Triangle collapses to a line
                self._removed_triangles[tri_idx] = True
                triangles_removed += 1
            elif v1 in (i0, i1, i2):
                # Update triangle to use v0 instead of v1
                base = tri_idx * 3
                for k in range(3):
                    if self._mesh.indices[base + k] == v1:
                        self._mesh.indices[base + k] = v0

        # Update neighbor edges
        self._update_neighbors(v0)

        return triangles_removed

    def _update_neighbors(self, vertex: int) -> None:
        """Update edge costs for edges connected to a vertex."""
        for edge in self._vertex_edges[vertex]:
            if self._removed_vertices[edge.v0] or self._removed_vertices[edge.v1]:
                continue
            self._compute_edge_cost(edge)
            heapq.heappush(self._edge_heap, edge)

    def _build_output_mesh(self) -> MeshData:
        """Build the final simplified mesh."""
        if not self._mesh:
            return MeshData()

        # Map old indices to new indices
        index_map: Dict[int, int] = {}
        new_vertices: List[Vec3] = []
        new_normals: List[Vec3] = []
        new_uvs: List[Vec2] = []

        for i, removed in enumerate(self._removed_vertices):
            if not removed:
                index_map[i] = len(new_vertices)
                new_vertices.append(self._mesh.vertices[i])
                if self._mesh.normals:
                    new_normals.append(self._mesh.normals[i])
                if self._mesh.uvs:
                    new_uvs.append(self._mesh.uvs[i])

        # Build new indices with comprehensive validation
        new_indices: List[int] = []
        for tri_idx in range(self._mesh.get_triangle_count()):
            if self._removed_triangles[tri_idx]:
                continue

            i0, i1, i2 = self._mesh.get_triangle(tri_idx)

            # Skip triangles with removed vertices (safety check)
            if self._removed_vertices[i0] or self._removed_vertices[i1] or self._removed_vertices[i2]:
                continue

            # Check for degenerate triangle (same vertex indices)
            if i0 == i1 or i1 == i2 or i2 == i0:
                continue

            # Validate indices are in the map (defensive check)
            if i0 not in index_map or i1 not in index_map or i2 not in index_map:
                continue

            # Get remapped indices
            new_i0, new_i1, new_i2 = index_map[i0], index_map[i1], index_map[i2]

            # Skip if remapped indices create a degenerate triangle
            if new_i0 == new_i1 or new_i1 == new_i2 or new_i2 == new_i0:
                continue

            # Validate triangle has non-zero area
            if self._is_degenerate_triangle(
                new_vertices[new_i0],
                new_vertices[new_i1],
                new_vertices[new_i2]
            ):
                continue

            new_indices.extend([new_i0, new_i1, new_i2])

        result = MeshData(
            vertices=new_vertices,
            normals=new_normals,
            uvs=new_uvs,
            indices=new_indices,
            material_id=self._mesh.material_id,
        )
        result.compute_bounds()

        return result

    def _is_degenerate_triangle(self, v0: Vec3, v1: Vec3, v2: Vec3) -> bool:
        """Check if a triangle is degenerate (zero or near-zero area)."""
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = edge1.cross(edge2)
        area = cross.length() * 0.5
        return area < HLODConstants.MIN_TRIANGLE_AREA

    def _zero_quadric(self) -> List[List[float]]:
        """Create a zero 4x4 quadric matrix."""
        return [[0.0 for _ in range(4)] for _ in range(4)]

    def _plane_quadric(self, a: float, b: float, c: float, d: float) -> List[List[float]]:
        """Create quadric matrix from plane equation ax + by + cz + d = 0."""
        return [
            [a * a, a * b, a * c, a * d],
            [a * b, b * b, b * c, b * d],
            [a * c, b * c, c * c, c * d],
            [a * d, b * d, c * d, d * d],
        ]

    def _add_quadric(self, dest: List[List[float]], src: List[List[float]]) -> None:
        """Add src quadric to dest quadric in place."""
        for i in range(4):
            for j in range(4):
                dest[i][j] += src[i][j]

    def _evaluate_quadric(self, q: List[List[float]], v: Vec3) -> float:
        """Evaluate quadric error at a point."""
        x, y, z = v.x, v.y, v.z
        return (
            q[0][0] * x * x + 2 * q[0][1] * x * y + 2 * q[0][2] * x * z + 2 * q[0][3] * x
            + q[1][1] * y * y + 2 * q[1][2] * y * z + 2 * q[1][3] * y
            + q[2][2] * z * z + 2 * q[2][3] * z
            + q[3][3]
        )


# =============================================================================
# IMPOSTOR GENERATOR
# =============================================================================


class ImpostorGenerator:
    """Generates billboard impostors from meshes."""

    def __init__(self, settings: Optional[ImpostorSettings] = None) -> None:
        self._settings = settings or ImpostorSettings()

    @property
    def settings(self) -> ImpostorSettings:
        return self._settings

    def generate(self, mesh: MeshData, bounds: AABB) -> ImpostorData:
        """
        Generate impostor data from a mesh.

        Args:
            mesh: Source mesh
            bounds: Bounding box for capture

        Returns:
            Impostor data with captured views
        """
        result = ImpostorData(
            resolution=self._settings.resolution,
            bounds=bounds,
        )

        # Generate view directions
        result.view_directions = self._generate_view_directions()

        # Capture each view
        views: List[List[List[Tuple[float, float, float, float]]]] = []
        normal_views: List[List[List[Tuple[float, float, float]]]] = []
        depth_views: List[List[List[float]]] = []

        for view_dir in result.view_directions:
            view_data = self._capture_view(mesh, bounds, view_dir)
            views.append(view_data["albedo"])

            if self._settings.capture_normals:
                normal_views.append(view_data["normal"])
            if self._settings.capture_depth:
                depth_views.append(view_data["depth"])

        # Pack views into atlas
        result.albedo_atlas = self._pack_views(views)

        if self._settings.capture_normals:
            result.normal_atlas = self._pack_normal_views(normal_views)
        if self._settings.capture_depth:
            result.depth_atlas = self._pack_depth_views(depth_views)

        return result

    def _generate_view_directions(self) -> List[Vec3]:
        """Generate evenly distributed view directions."""
        directions: List[Vec3] = []

        view_count = self._settings.view_count

        if self._settings.hemi_octahedron:
            # Generate hemi-octahedron directions for upper hemisphere
            for i in range(view_count):
                angle = 2.0 * math.pi * i / view_count
                x = math.cos(angle)
                z = math.sin(angle)
                directions.append(Vec3(x, HLODConstants.HEMI_OCTAHEDRON_Y_ELEVATION, z).normalized())

            # Add top-down view
            directions.append(Vec3(0.0, 1.0, 0.0))
        else:
            # Simple equidistant views around Y axis
            for i in range(view_count):
                angle = 2.0 * math.pi * i / view_count
                x = math.cos(angle)
                z = math.sin(angle)
                directions.append(Vec3(x, 0.0, z))

        return directions

    def _capture_view(
        self,
        mesh: MeshData,
        bounds: AABB,
        view_direction: Vec3,
    ) -> Dict[str, Any]:
        """
        Capture a single view of the mesh.

        This is a simplified rasterization for demonstration.
        In production, this would use GPU rendering.
        """
        resolution = self._settings.resolution

        # Initialize buffers
        albedo: List[List[Tuple[float, float, float, float]]] = [
            [(0.0, 0.0, 0.0, 0.0) for _ in range(resolution)]
            for _ in range(resolution)
        ]
        normal: List[List[Tuple[float, float, float]]] = [
            [(0.0, 0.0, 1.0) for _ in range(resolution)]
            for _ in range(resolution)
        ]
        depth: List[List[float]] = [
            [float("inf") for _ in range(resolution)]
            for _ in range(resolution)
        ]

        # Compute view matrix (look at center from view direction)
        center = bounds.center
        eye = center + view_direction * bounds.extents.length() * 2.0

        # Simple orthographic projection based on bounds
        size = bounds.size
        max_extent = max(size.x, size.y, size.z)

        # For each triangle, rasterize to buffer
        # (Simplified - actual implementation would use proper rasterization)
        for tri_idx in range(mesh.get_triangle_count()):
            i0, i1, i2 = mesh.get_triangle(tri_idx)
            v0, v1, v2 = mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2]

            # Project to screen space (simplified orthographic)
            def project(v: Vec3) -> Tuple[float, float, float]:
                # Project onto plane perpendicular to view direction
                rel = v - center

                # Create orthogonal basis
                up = Vec3(0.0, 1.0, 0.0)
                if abs(view_direction.dot(up)) > 0.99:
                    up = Vec3(1.0, 0.0, 0.0)

                right = view_direction.cross(up).normalized()
                up_actual = right.cross(view_direction).normalized()

                x = rel.dot(right) / max_extent + 0.5
                y = rel.dot(up_actual) / max_extent + 0.5
                z = rel.dot(view_direction)

                return (x, y, z)

            p0 = project(v0)
            p1 = project(v1)
            p2 = project(v2)

            # Simple bounding box fill (very simplified)
            min_x = max(0, int(min(p0[0], p1[0], p2[0]) * resolution))
            max_x = min(resolution - 1, int(max(p0[0], p1[0], p2[0]) * resolution))
            min_y = max(0, int(min(p0[1], p1[1], p2[1]) * resolution))
            max_y = min(resolution - 1, int(max(p0[1], p1[1], p2[1]) * resolution))

            tri_normal = mesh.compute_triangle_normal(tri_idx)

            for py in range(min_y, max_y + 1):
                for px in range(min_x, max_x + 1):
                    avg_z = (p0[2] + p1[2] + p2[2]) / 3.0

                    if avg_z < depth[py][px]:
                        depth[py][px] = avg_z
                        albedo[py][px] = (0.7, 0.7, 0.7, 1.0)  # Default gray
                        normal[py][px] = (tri_normal.x, tri_normal.y, tri_normal.z)

        return {"albedo": albedo, "normal": normal, "depth": depth}

    def _pack_views(
        self,
        views: List[List[List[Tuple[float, float, float, float]]]],
    ) -> List[List[Tuple[float, float, float, float]]]:
        """Pack multiple views into a texture atlas."""
        if not views:
            return []

        view_count = len(views)
        view_size = self._settings.resolution

        # Calculate atlas dimensions
        atlas_cols = int(math.ceil(math.sqrt(view_count)))
        atlas_rows = int(math.ceil(view_count / atlas_cols))

        atlas_width = atlas_cols * view_size
        atlas_height = atlas_rows * view_size

        # Create atlas
        atlas: List[List[Tuple[float, float, float, float]]] = [
            [(0.0, 0.0, 0.0, 0.0) for _ in range(atlas_width)]
            for _ in range(atlas_height)
        ]

        # Copy views into atlas
        for view_idx, view in enumerate(views):
            col = view_idx % atlas_cols
            row = view_idx // atlas_cols

            offset_x = col * view_size
            offset_y = row * view_size

            for y in range(view_size):
                for x in range(view_size):
                    atlas[offset_y + y][offset_x + x] = view[y][x]

        return atlas

    def _pack_normal_views(
        self,
        views: List[List[List[Tuple[float, float, float]]]],
    ) -> List[List[Tuple[float, float, float]]]:
        """Pack normal views into atlas."""
        if not views:
            return []

        view_count = len(views)
        view_size = self._settings.resolution

        atlas_cols = int(math.ceil(math.sqrt(view_count)))
        atlas_rows = int(math.ceil(view_count / atlas_cols))

        atlas_width = atlas_cols * view_size
        atlas_height = atlas_rows * view_size

        atlas: List[List[Tuple[float, float, float]]] = [
            [(0.0, 0.0, 1.0) for _ in range(atlas_width)]
            for _ in range(atlas_height)
        ]

        for view_idx, view in enumerate(views):
            col = view_idx % atlas_cols
            row = view_idx // atlas_cols

            offset_x = col * view_size
            offset_y = row * view_size

            for y in range(view_size):
                for x in range(view_size):
                    atlas[offset_y + y][offset_x + x] = view[y][x]

        return atlas

    def _pack_depth_views(
        self,
        views: List[List[List[float]]],
    ) -> List[List[float]]:
        """Pack depth views into atlas."""
        if not views:
            return []

        view_count = len(views)
        view_size = self._settings.resolution

        atlas_cols = int(math.ceil(math.sqrt(view_count)))
        atlas_rows = int(math.ceil(view_count / atlas_cols))

        atlas_width = atlas_cols * view_size
        atlas_height = atlas_rows * view_size

        atlas: List[List[float]] = [
            [1.0 for _ in range(atlas_width)]
            for _ in range(atlas_height)
        ]

        for view_idx, view in enumerate(views):
            col = view_idx % atlas_cols
            row = view_idx // atlas_cols

            offset_x = col * view_size
            offset_y = row * view_size

            for y in range(view_size):
                for x in range(view_size):
                    atlas[offset_y + y][offset_x + x] = view[y][x]

        return atlas


# =============================================================================
# PROXY MESH GENERATOR
# =============================================================================


class ProxyMeshGenerator:
    """Generates simplified proxy meshes from bounding geometry."""

    def generate_box(self, bounds: AABB) -> MeshData:
        """
        Generate a box mesh from AABB bounds.

        Args:
            bounds: Axis-aligned bounding box

        Returns:
            Box mesh with 12 triangles
        """
        min_p = bounds.min_point
        max_p = bounds.max_point

        # 8 vertices of the box
        vertices = [
            Vec3(min_p.x, min_p.y, min_p.z),  # 0: ---
            Vec3(max_p.x, min_p.y, min_p.z),  # 1: +--
            Vec3(max_p.x, max_p.y, min_p.z),  # 2: ++-
            Vec3(min_p.x, max_p.y, min_p.z),  # 3: -+-
            Vec3(min_p.x, min_p.y, max_p.z),  # 4: --+
            Vec3(max_p.x, min_p.y, max_p.z),  # 5: +-+
            Vec3(max_p.x, max_p.y, max_p.z),  # 6: +++
            Vec3(min_p.x, max_p.y, max_p.z),  # 7: -++
        ]

        # 6 faces, 2 triangles each
        indices = [
            # Front (-Z)
            0, 1, 2, 0, 2, 3,
            # Back (+Z)
            5, 4, 7, 5, 7, 6,
            # Left (-X)
            4, 0, 3, 4, 3, 7,
            # Right (+X)
            1, 5, 6, 1, 6, 2,
            # Bottom (-Y)
            4, 5, 1, 4, 1, 0,
            # Top (+Y)
            3, 2, 6, 3, 6, 7,
        ]

        # Normals for each vertex (average of incident faces)
        normals = [
            Vec3(-1, -1, -1).normalized(),
            Vec3(1, -1, -1).normalized(),
            Vec3(1, 1, -1).normalized(),
            Vec3(-1, 1, -1).normalized(),
            Vec3(-1, -1, 1).normalized(),
            Vec3(1, -1, 1).normalized(),
            Vec3(1, 1, 1).normalized(),
            Vec3(-1, 1, 1).normalized(),
        ]

        # Simple UVs
        uvs = [
            Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1),
            Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1),
        ]

        return MeshData(
            vertices=vertices,
            normals=normals,
            uvs=uvs,
            indices=indices,
            bounds=bounds,
        )

    def generate_convex_hull(self, mesh: MeshData) -> MeshData:
        """
        Generate a convex hull mesh from input mesh.

        This is a simplified implementation using gift wrapping.

        Args:
            mesh: Input mesh

        Returns:
            Convex hull mesh
        """
        if not mesh.vertices or len(mesh.vertices) < 4:
            return self.generate_box(mesh.bounds)

        # Simple approach: use bounding box vertices plus extremes
        # For production, use proper convex hull algorithm (QuickHull, etc.)

        points = list(mesh.vertices)

        # Find extremes along each axis
        min_x = min(points, key=lambda v: v.x)
        max_x = max(points, key=lambda v: v.x)
        min_y = min(points, key=lambda v: v.y)
        max_y = max(points, key=lambda v: v.y)
        min_z = min(points, key=lambda v: v.z)
        max_z = max(points, key=lambda v: v.z)

        # Unique extreme points
        extremes = list({min_x, max_x, min_y, max_y, min_z, max_z})

        if len(extremes) < 4:
            return self.generate_box(mesh.bounds)

        # Simple tetrahedron from first 4 extremes
        hull_vertices = extremes[:4]

        # Generate tetrahedron faces
        indices = [
            0, 1, 2,
            0, 2, 3,
            0, 3, 1,
            1, 3, 2,
        ]

        # Compute normals
        normals = [v.normalized() for v in hull_vertices]
        uvs = [Vec2(0, 0)] * len(hull_vertices)

        result = MeshData(
            vertices=hull_vertices,
            normals=normals,
            uvs=uvs,
            indices=indices,
        )
        result.compute_bounds()

        return result

    def generate_simplified_bounds(self, meshes: List[MeshData]) -> MeshData:
        """
        Generate a simplified bounding mesh from multiple meshes.

        Args:
            meshes: List of source meshes

        Returns:
            Simplified proxy mesh
        """
        if not meshes:
            return MeshData()

        # Compute combined bounds
        combined_bounds = AABB()
        for mesh in meshes:
            if mesh.bounds.is_valid():
                combined_bounds = combined_bounds.merge(mesh.bounds)

        if not combined_bounds.is_valid():
            # Compute from vertices
            for mesh in meshes:
                for vertex in mesh.vertices:
                    combined_bounds.expand(vertex)

        # Generate box for combined bounds
        return self.generate_box(combined_bounds)


# =============================================================================
# HLOD GENERATOR (MAIN CLASS)
# =============================================================================


class HLODGenerator:
    """
    Main HLOD mesh generator that selects and applies appropriate method.
    """

    def __init__(
        self,
        method: HLODGenerationMethod = HLODGenerationMethod.SIMPLIFICATION,
    ) -> None:
        self._method = method
        self._mesh_merger = MeshMerger()
        self._mesh_simplifier = MeshSimplifier()
        self._impostor_generator = ImpostorGenerator()
        self._proxy_generator = ProxyMeshGenerator()

    @property
    def method(self) -> HLODGenerationMethod:
        return self._method

    @method.setter
    def method(self, value: HLODGenerationMethod) -> None:
        self._method = value

    def configure(
        self,
        merge_settings: Optional[MergeSettings] = None,
        simplification_settings: Optional[SimplificationSettings] = None,
        impostor_settings: Optional[ImpostorSettings] = None,
    ) -> None:
        """Configure generators with custom settings."""
        if merge_settings:
            self._mesh_merger = MeshMerger(merge_settings)
        if simplification_settings:
            self._mesh_simplifier = MeshSimplifier(simplification_settings)
        if impostor_settings:
            self._impostor_generator = ImpostorGenerator(impostor_settings)

    def generate(
        self,
        source_meshes: List[MeshData],
        bounds: AABB,
        method: Optional[HLODGenerationMethod] = None,
    ) -> HLODMeshData:
        """
        Generate HLOD mesh from source meshes.

        Args:
            source_meshes: List of source meshes to combine
            bounds: Bounding box for the HLOD
            method: Override method (uses instance method if None)

        Returns:
            Generated HLOD mesh data
        """
        import time
        start_time = time.time()

        use_method = method if method is not None else self._method

        # Calculate original triangle count
        original_triangles = sum(m.get_triangle_count() for m in source_meshes)

        # Generate based on method
        if use_method == HLODGenerationMethod.MESH_MERGING:
            result_mesh = self._generate_merged(source_meshes)
        elif use_method == HLODGenerationMethod.SIMPLIFICATION:
            result_mesh = self._generate_simplified(source_meshes)
        elif use_method == HLODGenerationMethod.IMPOSTOR:
            result_mesh = self._generate_impostor_quad(source_meshes, bounds)
        elif use_method == HLODGenerationMethod.PROXY_MESH:
            result_mesh = self._generate_proxy(source_meshes, bounds)
        else:
            result_mesh = self._generate_merged(source_meshes)

        elapsed_ms = (time.time() - start_time) * 1000.0

        return HLODMeshData(
            mesh=result_mesh,
            source_mesh_count=len(source_meshes),
            original_triangle_count=original_triangles,
            method_used=use_method,
            generation_time_ms=elapsed_ms,
        )

    def select_method(
        self,
        mesh_count: int,
        total_triangles: int,
    ) -> HLODGenerationMethod:
        """
        Automatically select best HLOD method based on input complexity.

        Args:
            mesh_count: Number of source meshes
            total_triangles: Total triangle count

        Returns:
            Recommended generation method
        """
        # Heuristics for method selection using centralized constants
        if total_triangles < HLODConstants.SMALL_MESH_TRIANGLE_THRESHOLD:
            # Small meshes: just merge them
            return HLODGenerationMethod.MESH_MERGING
        elif total_triangles < HLODConstants.MEDIUM_MESH_TRIANGLE_THRESHOLD:
            # Medium meshes: simplify
            return HLODGenerationMethod.SIMPLIFICATION
        elif (mesh_count > HLODConstants.MANY_MESHES_THRESHOLD or
              total_triangles > HLODConstants.LARGE_MESH_TRIANGLE_THRESHOLD):
            # Many meshes or high poly: use impostor
            return HLODGenerationMethod.IMPOSTOR
        else:
            # Default to simplification
            return HLODGenerationMethod.SIMPLIFICATION

    def _generate_merged(self, meshes: List[MeshData]) -> MeshData:
        """Generate HLOD by merging meshes."""
        return self._mesh_merger.merge_meshes(meshes)

    def _generate_simplified(self, meshes: List[MeshData]) -> MeshData:
        """Generate HLOD by merging and simplifying."""
        merged = self._mesh_merger.merge_meshes(meshes)
        return self._mesh_simplifier.simplify(merged)

    def _generate_impostor_quad(
        self,
        meshes: List[MeshData],
        bounds: AABB,
    ) -> MeshData:
        """Generate a simple billboard quad for impostor."""
        # Return a quad facing the camera
        center = bounds.center
        extents = bounds.extents
        half_size = max(extents.x, extents.y, extents.z)

        vertices = [
            Vec3(center.x - half_size, center.y - half_size, center.z),
            Vec3(center.x + half_size, center.y - half_size, center.z),
            Vec3(center.x + half_size, center.y + half_size, center.z),
            Vec3(center.x - half_size, center.y + half_size, center.z),
        ]

        normals = [Vec3(0, 0, 1)] * 4
        uvs = [Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)]
        indices = [0, 1, 2, 0, 2, 3]

        return MeshData(
            vertices=vertices,
            normals=normals,
            uvs=uvs,
            indices=indices,
            bounds=bounds,
        )

    def _generate_proxy(
        self,
        meshes: List[MeshData],
        bounds: AABB,
    ) -> MeshData:
        """Generate proxy mesh from bounds."""
        return self._proxy_generator.generate_simplified_bounds(meshes)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Constants
    "HLODConstants",
    # Enums
    "HLODGenerationMethod",
    # Settings
    "SimplificationSettings",
    "ImpostorSettings",
    "MergeSettings",
    # Math types
    "Vec3",
    "Vec2",
    "AABB",
    # Data structures
    "MeshData",
    "HLODMeshData",
    "ImpostorData",
    "Edge",
    # Generators
    "MeshMerger",
    "MeshSimplifier",
    "ImpostorGenerator",
    "ProxyMeshGenerator",
    "HLODGenerator",
]
