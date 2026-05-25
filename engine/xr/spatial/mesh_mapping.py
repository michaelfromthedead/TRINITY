"""Spatial mesh mapping for AR scene reconstruction.

Provides real-time mesh generation from depth sensors and cameras,
supporting LOD, cleanup, and optimization for AR occlusion and physics.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from engine.core.math.vec import Vec3


class MeshUpdateMode(Enum):
    """How mesh updates are processed."""
    NONE = auto()         # No updates
    FULL = auto()         # Full mesh replacement
    INCREMENTAL = auto()  # Incremental updates only
    ADAPTIVE = auto()     # Automatic based on changes


class MeshLODLevel(Enum):
    """Level of detail for spatial mesh."""
    LOW = auto()       # Coarse mesh, minimal triangles
    MEDIUM = auto()    # Balanced detail/performance
    HIGH = auto()      # High detail mesh
    ULTRA = auto()     # Maximum detail


class MeshClassification(Enum):
    """Classification of mesh regions."""
    NONE = auto()
    FLOOR = auto()
    CEILING = auto()
    WALL = auto()
    TABLE = auto()
    SEAT = auto()
    DOOR = auto()
    WINDOW = auto()
    UNKNOWN = auto()


@dataclass(slots=True)
class MeshVertex:
    """Vertex in a spatial mesh."""
    position: Vec3
    normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    classification: MeshClassification = MeshClassification.NONE
    confidence: float = 1.0


@dataclass(slots=True)
class MeshTriangle:
    """Triangle in a spatial mesh."""
    v0: int  # Vertex indices
    v1: int
    v2: int


@dataclass(slots=True)
class MeshBounds:
    """Axis-aligned bounding box for a mesh region."""
    min_point: Vec3 = field(default_factory=Vec3.zero)
    max_point: Vec3 = field(default_factory=Vec3.zero)

    @property
    def center(self) -> Vec3:
        """Get the center of the bounds."""
        return (self.min_point + self.max_point) * 0.5

    @property
    def size(self) -> Vec3:
        """Get the size of the bounds."""
        return self.max_point - self.min_point

    @property
    def extents(self) -> Vec3:
        """Get the half-size of the bounds."""
        return self.size * 0.5

    def contains_point(self, point: Vec3) -> bool:
        """Check if a point is inside the bounds.

        Args:
            point: Point to test

        Returns:
            True if point is inside
        """
        return (
            self.min_point.x <= point.x <= self.max_point.x and
            self.min_point.y <= point.y <= self.max_point.y and
            self.min_point.z <= point.z <= self.max_point.z
        )

    def intersects(self, other: MeshBounds) -> bool:
        """Check if this bounds intersects another.

        Args:
            other: Other bounds to test

        Returns:
            True if bounds intersect
        """
        return (
            self.min_point.x <= other.max_point.x and
            self.max_point.x >= other.min_point.x and
            self.min_point.y <= other.max_point.y and
            self.max_point.y >= other.min_point.y and
            self.min_point.z <= other.max_point.z and
            self.max_point.z >= other.min_point.z
        )

    def expand_to_include(self, point: Vec3) -> None:
        """Expand bounds to include a point.

        Args:
            point: Point to include
        """
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


@dataclass(slots=True)
class MeshBlock:
    """A block/chunk of spatial mesh data.

    Meshes are divided into blocks for efficient updates and culling.
    """
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vertices: list[MeshVertex] = field(default_factory=list)
    triangles: list[MeshTriangle] = field(default_factory=list)
    bounds: MeshBounds = field(default_factory=MeshBounds)
    lod_level: MeshLODLevel = MeshLODLevel.MEDIUM
    is_dirty: bool = True
    last_updated: float = 0.0
    version: int = 0

    @property
    def vertex_count(self) -> int:
        """Get the number of vertices."""
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        """Get the number of triangles."""
        return len(self.triangles)

    def compute_bounds(self) -> None:
        """Recompute the bounding box from vertices."""
        if not self.vertices:
            self.bounds = MeshBounds()
            return

        first = self.vertices[0].position
        self.bounds = MeshBounds(
            min_point=Vec3(first.x, first.y, first.z),
            max_point=Vec3(first.x, first.y, first.z),
        )

        for vertex in self.vertices[1:]:
            self.bounds.expand_to_include(vertex.position)

    def clear(self) -> None:
        """Clear all mesh data."""
        self.vertices.clear()
        self.triangles.clear()
        self.bounds = MeshBounds()
        self.is_dirty = True
        self.version += 1


class SpatialMesh:
    """Spatial mesh representing the scanned environment.

    Manages real-time mesh generation, LOD, and updates from
    depth sensor data.

    Attributes:
        mesh_id: Unique identifier
        blocks: Mesh blocks making up the full mesh
        lod_level: Current level of detail
    """
    __slots__ = (
        '_mesh_id',
        '_blocks',
        '_lod_level',
        '_bounds',
        '_update_mode',
        '_is_valid',
        '_vertex_count',
        '_triangle_count',
        '_last_updated',
        '_version',
        '_callbacks',
        '_classification_enabled',
    )

    def __init__(
        self,
        lod_level: MeshLODLevel = MeshLODLevel.MEDIUM,
        update_mode: MeshUpdateMode = MeshUpdateMode.INCREMENTAL,
    ) -> None:
        """Initialize a spatial mesh.

        Args:
            lod_level: Level of detail
            update_mode: How updates are processed
        """
        self._mesh_id: str = str(uuid.uuid4())
        self._blocks: dict[str, MeshBlock] = {}
        self._lod_level: MeshLODLevel = lod_level
        self._bounds: MeshBounds = MeshBounds()
        self._update_mode: MeshUpdateMode = update_mode
        self._is_valid: bool = False
        self._vertex_count: int = 0
        self._triangle_count: int = 0
        self._last_updated: float = 0.0
        self._version: int = 0
        self._callbacks: dict[str, list[Callable]] = {
            "mesh_updated": [],
            "block_added": [],
            "block_removed": [],
        }
        self._classification_enabled: bool = False

    @property
    def mesh_id(self) -> str:
        """Get the unique mesh identifier."""
        return self._mesh_id

    @property
    def lod_level(self) -> MeshLODLevel:
        """Get the current LOD level."""
        return self._lod_level

    @lod_level.setter
    def lod_level(self, value: MeshLODLevel) -> None:
        """Set the LOD level."""
        if value != self._lod_level:
            self._lod_level = value
            self._mark_all_dirty()

    @property
    def update_mode(self) -> MeshUpdateMode:
        """Get the update mode."""
        return self._update_mode

    @update_mode.setter
    def update_mode(self, value: MeshUpdateMode) -> None:
        """Set the update mode."""
        self._update_mode = value

    @property
    def bounds(self) -> MeshBounds:
        """Get the overall mesh bounds."""
        return self._bounds

    @property
    def is_valid(self) -> bool:
        """Check if the mesh data is valid."""
        return self._is_valid

    @property
    def block_count(self) -> int:
        """Get the number of mesh blocks."""
        return len(self._blocks)

    @property
    def vertex_count(self) -> int:
        """Get the total vertex count."""
        return self._vertex_count

    @property
    def triangle_count(self) -> int:
        """Get the total triangle count."""
        return self._triangle_count

    @property
    def version(self) -> int:
        """Get the mesh version (incremented on updates)."""
        return self._version

    @property
    def classification_enabled(self) -> bool:
        """Check if mesh classification is enabled."""
        return self._classification_enabled

    @classification_enabled.setter
    def classification_enabled(self, value: bool) -> None:
        """Enable or disable mesh classification."""
        self._classification_enabled = value

    def get_block(self, block_id: str) -> Optional[MeshBlock]:
        """Get a mesh block by ID.

        Args:
            block_id: Block identifier

        Returns:
            Block if found, None otherwise
        """
        return self._blocks.get(block_id)

    def get_all_blocks(self) -> list[MeshBlock]:
        """Get all mesh blocks.

        Returns:
            List of all blocks
        """
        return list(self._blocks.values())

    def get_dirty_blocks(self) -> list[MeshBlock]:
        """Get blocks that need updating.

        Returns:
            List of dirty blocks
        """
        return [b for b in self._blocks.values() if b.is_dirty]

    def get_blocks_in_bounds(self, bounds: MeshBounds) -> list[MeshBlock]:
        """Get blocks intersecting a bounding box.

        Args:
            bounds: Bounds to test against

        Returns:
            List of intersecting blocks
        """
        return [
            b for b in self._blocks.values()
            if b.bounds.intersects(bounds)
        ]

    def get_blocks_near(self, position: Vec3, radius: float) -> list[MeshBlock]:
        """Get blocks near a position.

        Args:
            position: Center position
            radius: Search radius

        Returns:
            List of nearby blocks
        """
        results = []
        for block in self._blocks.values():
            center = block.bounds.center
            if center.distance(position) <= radius:
                results.append(block)
        return results

    def add_block(self, block: MeshBlock) -> None:
        """Add a mesh block.

        Args:
            block: Block to add
        """
        self._blocks[block.block_id] = block
        self._update_counts()
        self._update_bounds()
        self._version += 1
        self._is_valid = True
        self._notify_callbacks("block_added", block)

    def update_block(
        self,
        block_id: str,
        vertices: list[MeshVertex],
        triangles: list[MeshTriangle],
        timestamp: float,
    ) -> bool:
        """Update a mesh block's data.

        Args:
            block_id: Block to update
            vertices: New vertex data
            triangles: New triangle data
            timestamp: Update timestamp

        Returns:
            True if block was updated
        """
        block = self._blocks.get(block_id)
        if not block:
            return False

        block.vertices = vertices
        block.triangles = triangles
        block.compute_bounds()
        block.is_dirty = False
        block.last_updated = timestamp
        block.version += 1

        self._update_counts()
        self._update_bounds()
        self._last_updated = timestamp
        self._version += 1
        self._notify_callbacks("mesh_updated", block)
        return True

    def remove_block(self, block_id: str) -> bool:
        """Remove a mesh block.

        Args:
            block_id: Block to remove

        Returns:
            True if block was removed
        """
        block = self._blocks.pop(block_id, None)
        if block:
            self._update_counts()
            self._update_bounds()
            self._version += 1
            self._notify_callbacks("block_removed", block)
            return True
        return False

    def raycast(
        self,
        origin: Vec3,
        direction: Vec3,
        max_distance: float = 100.0,
    ) -> Optional[tuple[Vec3, Vec3, float]]:
        """Cast a ray against the mesh.

        Args:
            origin: Ray origin
            direction: Ray direction (normalized)
            max_distance: Maximum ray distance

        Returns:
            Tuple of (hit_point, normal, distance) or None
        """
        closest_hit: Optional[tuple[Vec3, Vec3, float]] = None
        closest_distance = max_distance

        # Simple brute-force for now - could use spatial acceleration
        for block in self._blocks.values():
            # Quick bounds check
            if not self._ray_intersects_bounds(origin, direction, block.bounds, max_distance):
                continue

            # Check each triangle
            for tri in block.triangles:
                v0 = block.vertices[tri.v0].position
                v1 = block.vertices[tri.v1].position
                v2 = block.vertices[tri.v2].position

                hit = self._ray_triangle_intersection(
                    origin, direction, v0, v1, v2, closest_distance
                )
                if hit:
                    hit_point, t = hit
                    if t < closest_distance:
                        # Compute normal
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        normal = edge1.cross(edge2).normalized()
                        closest_hit = (hit_point, normal, t)
                        closest_distance = t

        return closest_hit

    def _ray_intersects_bounds(
        self,
        origin: Vec3,
        direction: Vec3,
        bounds: MeshBounds,
        max_distance: float,
    ) -> bool:
        """Quick ray-bounds intersection test."""
        RAY_EPSILON = 1e-6
        RAY_INV_EPSILON = 1e10
        inv_dir = Vec3(
            1.0 / direction.x if abs(direction.x) > RAY_EPSILON else RAY_INV_EPSILON,
            1.0 / direction.y if abs(direction.y) > RAY_EPSILON else RAY_INV_EPSILON,
            1.0 / direction.z if abs(direction.z) > RAY_EPSILON else RAY_INV_EPSILON,
        )

        t1 = (bounds.min_point.x - origin.x) * inv_dir.x
        t2 = (bounds.max_point.x - origin.x) * inv_dir.x
        t3 = (bounds.min_point.y - origin.y) * inv_dir.y
        t4 = (bounds.max_point.y - origin.y) * inv_dir.y
        t5 = (bounds.min_point.z - origin.z) * inv_dir.z
        t6 = (bounds.max_point.z - origin.z) * inv_dir.z

        tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
        tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))

        return tmax >= 0 and tmin <= tmax and tmin <= max_distance

    def _ray_triangle_intersection(
        self,
        origin: Vec3,
        direction: Vec3,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        max_distance: float,
    ) -> Optional[tuple[Vec3, float]]:
        """Moller-Trumbore ray-triangle intersection."""
        RAY_TRIANGLE_EPSILON = 1e-6

        edge1 = v1 - v0
        edge2 = v2 - v0
        h = direction.cross(edge2)
        a = edge1.dot(h)

        if abs(a) < RAY_TRIANGLE_EPSILON:
            return None

        f = 1.0 / a
        s = origin - v0
        u = f * s.dot(h)

        if u < 0.0 or u > 1.0:
            return None

        q = s.cross(edge1)
        v = f * direction.dot(q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * edge2.dot(q)

        if t > RAY_TRIANGLE_EPSILON and t < max_distance:
            hit_point = origin + direction * t
            return (hit_point, t)

        return None

    def cleanup_distant_blocks(self, reference: Vec3, max_distance: float) -> int:
        """Remove blocks far from a reference point.

        Args:
            reference: Reference position (e.g., camera)
            max_distance: Maximum allowed distance

        Returns:
            Number of blocks removed
        """
        to_remove = []
        for block_id, block in self._blocks.items():
            if block.bounds.center.distance(reference) > max_distance:
                to_remove.append(block_id)

        for block_id in to_remove:
            self.remove_block(block_id)

        return len(to_remove)

    def optimize(self) -> None:
        """Optimize the mesh by merging nearby vertices and removing degenerates."""
        for block in self._blocks.values():
            self._optimize_block(block)
        self._update_counts()

    def _optimize_block(self, block: MeshBlock) -> None:
        """Optimize a single block."""
        if not block.vertices:
            return

        # Remove degenerate triangles
        valid_triangles = []
        for tri in block.triangles:
            if tri.v0 != tri.v1 and tri.v1 != tri.v2 and tri.v0 != tri.v2:
                if tri.v0 < len(block.vertices) and \
                   tri.v1 < len(block.vertices) and \
                   tri.v2 < len(block.vertices):
                    valid_triangles.append(tri)
        block.triangles = valid_triangles
        block.version += 1

    def clear(self) -> None:
        """Clear all mesh data."""
        self._blocks.clear()
        self._bounds = MeshBounds()
        self._vertex_count = 0
        self._triangle_count = 0
        self._is_valid = False
        self._version += 1

    def _update_counts(self) -> None:
        """Update vertex and triangle counts."""
        self._vertex_count = sum(b.vertex_count for b in self._blocks.values())
        self._triangle_count = sum(b.triangle_count for b in self._blocks.values())

    def _update_bounds(self) -> None:
        """Update overall mesh bounds."""
        if not self._blocks:
            self._bounds = MeshBounds()
            return

        first = True
        for block in self._blocks.values():
            if first:
                self._bounds = MeshBounds(
                    min_point=Vec3(
                        block.bounds.min_point.x,
                        block.bounds.min_point.y,
                        block.bounds.min_point.z,
                    ),
                    max_point=Vec3(
                        block.bounds.max_point.x,
                        block.bounds.max_point.y,
                        block.bounds.max_point.z,
                    ),
                )
                first = False
            else:
                self._bounds.expand_to_include(block.bounds.min_point)
                self._bounds.expand_to_include(block.bounds.max_point)

    def _mark_all_dirty(self) -> None:
        """Mark all blocks as needing update."""
        for block in self._blocks.values():
            block.is_dirty = True

    def add_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for mesh events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def remove_callback(self, event: str, callback: Callable) -> None:
        """Remove a registered callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def _notify_callbacks(self, event: str, block: MeshBlock) -> None:
        """Notify callbacks for an event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                callback(self, block)


@dataclass(slots=True)
class MeshManagerConfig:
    """Configuration for spatial mesh management."""
    lod_level: MeshLODLevel = MeshLODLevel.MEDIUM
    update_mode: MeshUpdateMode = MeshUpdateMode.INCREMENTAL
    max_distance: float = 20.0
    cleanup_interval: float = 5.0
    classification_enabled: bool = True
    physics_mesh_enabled: bool = True
    occlusion_mesh_enabled: bool = True


class SpatialMeshManager:
    """Manages spatial mesh generation and updates.

    Handles the complete lifecycle of spatial mesh data including
    generation, LOD, cleanup, and integration with physics/occlusion.
    """
    __slots__ = (
        '_config',
        '_mesh',
        '_is_running',
        '_observer_position',
        '_last_cleanup',
        '_callbacks',
    )

    def __init__(self, config: Optional[MeshManagerConfig] = None) -> None:
        """Initialize the mesh manager.

        Args:
            config: Manager configuration
        """
        self._config: MeshManagerConfig = config or MeshManagerConfig()
        self._mesh: SpatialMesh = SpatialMesh(
            lod_level=self._config.lod_level,
            update_mode=self._config.update_mode,
        )
        self._mesh.classification_enabled = self._config.classification_enabled
        self._is_running: bool = False
        self._observer_position: Vec3 = Vec3.zero()
        self._last_cleanup: float = 0.0
        self._callbacks: dict[str, list[Callable]] = {
            "mesh_ready": [],
            "cleanup_complete": [],
        }

    @property
    def config(self) -> MeshManagerConfig:
        """Get the manager configuration."""
        return self._config

    @property
    def mesh(self) -> SpatialMesh:
        """Get the spatial mesh."""
        return self._mesh

    @property
    def is_running(self) -> bool:
        """Check if mesh generation is active."""
        return self._is_running

    def start(self) -> bool:
        """Start mesh generation.

        Returns:
            True if started successfully
        """
        if self._is_running:
            return False
        self._is_running = True
        return True

    def stop(self) -> bool:
        """Stop mesh generation.

        Returns:
            True if stopped successfully
        """
        if not self._is_running:
            return False
        self._is_running = False
        return True

    def update(self, observer_position: Vec3, timestamp: float) -> None:
        """Update mesh generation.

        Args:
            observer_position: Current camera/observer position
            timestamp: Current time
        """
        if not self._is_running:
            return

        self._observer_position = observer_position

        # Periodic cleanup of distant blocks
        if timestamp - self._last_cleanup > self._config.cleanup_interval:
            removed = self._mesh.cleanup_distant_blocks(
                observer_position,
                self._config.max_distance,
            )
            self._last_cleanup = timestamp
            if removed > 0:
                self._notify_callbacks("cleanup_complete")

    def set_lod_level(self, level: MeshLODLevel) -> None:
        """Set the mesh LOD level.

        Args:
            level: New LOD level
        """
        self._config.lod_level = level
        self._mesh.lod_level = level

    def set_max_distance(self, distance: float) -> None:
        """Set the maximum mesh distance.

        Args:
            distance: Maximum distance from observer
        """
        self._config.max_distance = max(1.0, distance)

    def raycast(
        self,
        origin: Vec3,
        direction: Vec3,
        max_distance: float = 100.0,
    ) -> Optional[tuple[Vec3, Vec3, float]]:
        """Cast a ray against the mesh.

        Args:
            origin: Ray origin
            direction: Ray direction
            max_distance: Maximum ray distance

        Returns:
            Tuple of (hit_point, normal, distance) or None
        """
        return self._mesh.raycast(origin, direction, max_distance)

    def get_mesh_for_physics(self) -> Optional[SpatialMesh]:
        """Get mesh data suitable for physics collision.

        Returns:
            Physics-ready mesh or None if disabled
        """
        if not self._config.physics_mesh_enabled:
            return None
        return self._mesh

    def get_mesh_for_occlusion(self) -> Optional[SpatialMesh]:
        """Get mesh data suitable for AR occlusion.

        Returns:
            Occlusion-ready mesh or None if disabled
        """
        if not self._config.occlusion_mesh_enabled:
            return None
        return self._mesh

    def clear(self) -> None:
        """Clear all mesh data."""
        self._mesh.clear()

    def add_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for manager events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _notify_callbacks(self, event: str) -> None:
        """Notify callbacks for an event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                callback(self)
