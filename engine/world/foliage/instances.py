"""
Foliage instance management using Hierarchical Instanced Static Mesh (HISM) pattern.

Provides efficient instanced foliage rendering with:
- Hierarchical spatial clustering
- Per-cluster frustum culling
- LOD management per instance
- Instance buffer generation for GPU
- BatchedDescriptor for bulk operations
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from engine.world.foliage.constants import (
    DEFAULT_LOD_DISTANCES,
    DEFAULT_CULL_DISTANCE,
)
from .placement import Bounds, PlacementResult
from .types import FoliageType


@dataclass
class FoliageInstance:
    """
    Single foliage instance.

    Represents one rendered instance of a foliage type with
    transform, visibility, and LOD state.

    Attributes:
        instance_id: Unique identifier within the HISM
        foliage_type_id: Associated foliage type
        position: World position (x, y, z)
        rotation: Euler rotation in degrees (pitch, yaw, roll)
        scale: Scale factors (x, y, z)
        visible: Whether instance is currently visible
        lod_level: Current LOD level (0 = highest detail)
    """

    instance_id: int = 0
    foliage_type_id: str = ""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    visible: bool = True
    lod_level: int = 0

    def distance_to(self, point: Tuple[float, float, float]) -> float:
        """
        Calculate distance to a point.

        Args:
            point: Point to measure distance to

        Returns:
            Euclidean distance
        """
        dx = self.position[0] - point[0]
        dy = self.position[1] - point[1]
        dz = self.position[2] - point[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def distance_squared_to(self, point: Tuple[float, float, float]) -> float:
        """
        Calculate squared distance to a point.

        Args:
            point: Point to measure distance to

        Returns:
            Squared Euclidean distance (faster than distance_to)
        """
        dx = self.position[0] - point[0]
        dy = self.position[1] - point[1]
        dz = self.position[2] - point[2]
        return dx * dx + dy * dy + dz * dz

    @classmethod
    def from_placement(
        cls, instance_id: int, placement: PlacementResult
    ) -> "FoliageInstance":
        """
        Create instance from placement result.

        Args:
            instance_id: Unique ID to assign
            placement: Placement data to use

        Returns:
            New FoliageInstance
        """
        return cls(
            instance_id=instance_id,
            foliage_type_id=placement.foliage_type_id,
            position=placement.position,
            rotation=placement.rotation,
            scale=placement.scale,
            visible=True,
            lod_level=0,
        )


@dataclass
class Frustum:
    """
    View frustum for culling.

    Simplified frustum representation using 6 planes.
    Each plane is (nx, ny, nz, d) where n is the normal
    and d is the distance from origin.
    """

    planes: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def contains_point(self, point: Tuple[float, float, float]) -> bool:
        """
        Check if point is inside frustum.

        Args:
            point: Point to test

        Returns:
            True if point is inside all planes
        """
        x, y, z = point
        for nx, ny, nz, d in self.planes:
            if nx * x + ny * y + nz * z + d < 0:
                return False
        return True

    def contains_sphere(
        self, center: Tuple[float, float, float], radius: float
    ) -> bool:
        """
        Check if sphere intersects frustum.

        Args:
            center: Sphere center
            radius: Sphere radius

        Returns:
            True if sphere intersects or is inside frustum
        """
        x, y, z = center
        for nx, ny, nz, d in self.planes:
            dist = nx * x + ny * y + nz * z + d
            if dist < -radius:
                return False
        return True

    def contains_bounds(self, bounds: Bounds, min_y: float, max_y: float) -> bool:
        """
        Check if axis-aligned bounds intersect frustum.

        Args:
            bounds: XZ bounds
            min_y: Minimum Y
            max_y: Maximum Y

        Returns:
            True if bounds intersect frustum
        """
        # Get 8 corners of bounding box
        corners = [
            (bounds.min_x, min_y, bounds.min_z),
            (bounds.max_x, min_y, bounds.min_z),
            (bounds.min_x, max_y, bounds.min_z),
            (bounds.max_x, max_y, bounds.min_z),
            (bounds.min_x, min_y, bounds.max_z),
            (bounds.max_x, min_y, bounds.max_z),
            (bounds.min_x, max_y, bounds.max_z),
            (bounds.max_x, max_y, bounds.max_z),
        ]

        # For each plane, check if all corners are outside
        for nx, ny, nz, d in self.planes:
            all_outside = True
            for x, y, z in corners:
                if nx * x + ny * y + nz * z + d >= 0:
                    all_outside = False
                    break
            if all_outside:
                return False
        return True

    @classmethod
    def from_view_projection(cls, matrix: List[List[float]]) -> "Frustum":
        """
        Extract frustum planes from view-projection matrix.

        Args:
            matrix: 4x4 view-projection matrix

        Returns:
            Frustum with 6 planes
        """
        m = matrix
        planes = []

        # Left plane
        planes.append(
            (
                m[0][3] + m[0][0],
                m[1][3] + m[1][0],
                m[2][3] + m[2][0],
                m[3][3] + m[3][0],
            )
        )
        # Right plane
        planes.append(
            (
                m[0][3] - m[0][0],
                m[1][3] - m[1][0],
                m[2][3] - m[2][0],
                m[3][3] - m[3][0],
            )
        )
        # Bottom plane
        planes.append(
            (
                m[0][3] + m[0][1],
                m[1][3] + m[1][1],
                m[2][3] + m[2][1],
                m[3][3] + m[3][1],
            )
        )
        # Top plane
        planes.append(
            (
                m[0][3] - m[0][1],
                m[1][3] - m[1][1],
                m[2][3] - m[2][1],
                m[3][3] - m[3][1],
            )
        )
        # Near plane
        planes.append(
            (
                m[0][3] + m[0][2],
                m[1][3] + m[1][2],
                m[2][3] + m[2][2],
                m[3][3] + m[3][2],
            )
        )
        # Far plane
        planes.append(
            (
                m[0][3] - m[0][2],
                m[1][3] - m[1][2],
                m[2][3] - m[2][2],
                m[3][3] - m[3][2],
            )
        )

        # Normalize planes
        normalized = []
        for nx, ny, nz, d in planes:
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if length > 0:
                normalized.append((nx / length, ny / length, nz / length, d / length))
            else:
                normalized.append((0.0, 0.0, 0.0, 0.0))

        return cls(planes=normalized)


class FoliageCluster:
    """
    Cluster of foliage instances for spatial organization.

    Groups nearby instances for efficient frustum culling
    and LOD transitions.
    """

    __slots__ = (
        "_bounds",
        "_instances",
        "_foliage_type_id",
        "_min_y",
        "_max_y",
        "_visible_count",
    )

    def __init__(
        self,
        bounds: Bounds,
        foliage_type_id: str,
    ) -> None:
        """
        Initialize cluster.

        Args:
            bounds: Spatial bounds of cluster
            foliage_type_id: Foliage type for this cluster
        """
        self._bounds = bounds
        self._foliage_type_id = foliage_type_id
        self._instances: List[FoliageInstance] = []
        self._min_y = float("inf")
        self._max_y = float("-inf")
        self._visible_count = 0

    @property
    def bounds(self) -> Bounds:
        """Get cluster bounds."""
        return self._bounds

    @property
    def foliage_type_id(self) -> str:
        """Get foliage type ID."""
        return self._foliage_type_id

    @property
    def instance_count(self) -> int:
        """Get total instance count."""
        return len(self._instances)

    @property
    def visible_count(self) -> int:
        """Get visible instance count."""
        return self._visible_count

    def add_instance(self, instance: FoliageInstance) -> None:
        """
        Add an instance to the cluster.

        Args:
            instance: Instance to add
        """
        self._instances.append(instance)
        y = instance.position[1]
        self._min_y = min(self._min_y, y)
        self._max_y = max(self._max_y, y)
        if instance.visible:
            self._visible_count += 1

    def remove_instance(self, instance_id: int) -> bool:
        """
        Remove an instance by ID.

        Args:
            instance_id: ID of instance to remove

        Returns:
            True if instance was removed
        """
        for i, inst in enumerate(self._instances):
            if inst.instance_id == instance_id:
                if inst.visible:
                    self._visible_count -= 1
                del self._instances[i]
                return True
        return False

    def get_instances(self) -> List[FoliageInstance]:
        """Get all instances."""
        return self._instances

    def get_visible_instances(self) -> List[FoliageInstance]:
        """Get only visible instances."""
        return [i for i in self._instances if i.visible]

    def cull(self, frustum: Frustum) -> int:
        """
        Cull instances against frustum.

        Args:
            frustum: View frustum

        Returns:
            Number of visible instances after culling
        """
        # First check if entire cluster is visible
        if not frustum.contains_bounds(self._bounds, self._min_y, self._max_y):
            # Entire cluster is culled
            for inst in self._instances:
                inst.visible = False
            self._visible_count = 0
            return 0

        # Check individual instances
        self._visible_count = 0
        for inst in self._instances:
            inst.visible = frustum.contains_point(inst.position)
            if inst.visible:
                self._visible_count += 1

        return self._visible_count

    def update_lod(
        self, camera_pos: Tuple[float, float, float], lod_distances: List[float]
    ) -> None:
        """
        Update LOD levels based on camera distance.

        Args:
            camera_pos: Camera position
            lod_distances: LOD transition distances
        """
        for inst in self._instances:
            if not inst.visible:
                continue

            dist = inst.distance_to(camera_pos)
            lod = 0
            for i, lod_dist in enumerate(lod_distances):
                if dist >= lod_dist:
                    lod = i + 1
            inst.lod_level = lod

    def clear(self) -> None:
        """Remove all instances."""
        self._instances.clear()
        self._min_y = float("inf")
        self._max_y = float("-inf")
        self._visible_count = 0


class HierarchicalInstancedMesh:
    """
    Hierarchical Instanced Static Mesh (HISM) for efficient foliage rendering.

    Organizes instances into spatial clusters for efficient culling
    and LOD management.
    """

    __slots__ = (
        "_foliage_type",
        "_clusters",
        "_cluster_size",
        "_next_instance_id",
        "_total_instances",
        "_visible_instances",
        "_instance_lookup",
    )

    def __init__(
        self,
        foliage_type: FoliageType,
        cluster_size: float = 50.0,
    ) -> None:
        """
        Initialize HISM.

        Args:
            foliage_type: Foliage type for this HISM
            cluster_size: Size of spatial clusters
        """
        self._foliage_type = foliage_type
        self._cluster_size = cluster_size
        self._clusters: Dict[Tuple[int, int], FoliageCluster] = {}
        self._next_instance_id = 0
        self._total_instances = 0
        self._visible_instances = 0
        self._instance_lookup: Dict[int, Tuple[int, int]] = {}

    @property
    def foliage_type(self) -> FoliageType:
        """Get foliage type."""
        return self._foliage_type

    @property
    def total_instances(self) -> int:
        """Get total instance count."""
        return self._total_instances

    @property
    def visible_instances(self) -> int:
        """Get visible instance count."""
        return self._visible_instances

    @property
    def cluster_count(self) -> int:
        """Get number of clusters."""
        return len(self._clusters)

    def _get_cluster_key(self, x: float, z: float) -> Tuple[int, int]:
        """Get cluster key for position."""
        return (
            int(math.floor(x / self._cluster_size)),
            int(math.floor(z / self._cluster_size)),
        )

    def _get_or_create_cluster(self, key: Tuple[int, int]) -> FoliageCluster:
        """Get or create cluster for key."""
        if key not in self._clusters:
            bounds = Bounds(
                min_x=key[0] * self._cluster_size,
                min_z=key[1] * self._cluster_size,
                max_x=(key[0] + 1) * self._cluster_size,
                max_z=(key[1] + 1) * self._cluster_size,
            )
            self._clusters[key] = FoliageCluster(bounds, self._foliage_type.type_id)
        return self._clusters[key]

    def add_instance(self, placement: PlacementResult) -> int:
        """
        Add a single instance.

        Args:
            placement: Placement data

        Returns:
            Instance ID
        """
        instance_id = self._next_instance_id
        self._next_instance_id += 1

        instance = FoliageInstance.from_placement(instance_id, placement)
        key = self._get_cluster_key(placement.position[0], placement.position[2])
        cluster = self._get_or_create_cluster(key)
        cluster.add_instance(instance)

        self._instance_lookup[instance_id] = key
        self._total_instances += 1
        self._visible_instances += 1

        return instance_id

    def add_instances(self, placements: List[PlacementResult]) -> List[int]:
        """
        Add multiple instances (BatchedDescriptor pattern).

        Args:
            placements: List of placement data

        Returns:
            List of instance IDs
        """
        ids = []
        for placement in placements:
            ids.append(self.add_instance(placement))
        return ids

    def remove_instance(self, instance_id: int) -> bool:
        """
        Remove a single instance.

        Args:
            instance_id: ID of instance to remove

        Returns:
            True if instance was removed
        """
        if instance_id not in self._instance_lookup:
            return False

        key = self._instance_lookup[instance_id]
        if key in self._clusters:
            if self._clusters[key].remove_instance(instance_id):
                del self._instance_lookup[instance_id]
                self._total_instances -= 1

                # Remove empty clusters
                if self._clusters[key].instance_count == 0:
                    del self._clusters[key]
                return True
        return False

    def remove_instances_in_bounds(self, bounds: Bounds) -> int:
        """
        Remove all instances within bounds.

        Args:
            bounds: Region to clear

        Returns:
            Number of instances removed
        """
        removed = 0
        to_remove: List[int] = []

        # Find affected clusters
        min_key = self._get_cluster_key(bounds.min_x, bounds.min_z)
        max_key = self._get_cluster_key(bounds.max_x, bounds.max_z)

        for cx in range(min_key[0], max_key[0] + 1):
            for cz in range(min_key[1], max_key[1] + 1):
                key = (cx, cz)
                if key not in self._clusters:
                    continue

                cluster = self._clusters[key]
                for inst in cluster.get_instances():
                    x, _, z = inst.position
                    if bounds.contains(x, z):
                        to_remove.append(inst.instance_id)

        for instance_id in to_remove:
            if self.remove_instance(instance_id):
                removed += 1

        return removed

    def update_visibility(
        self, camera_pos: Tuple[float, float, float], frustum: Frustum
    ) -> None:
        """
        Update instance visibility based on camera and frustum.

        Args:
            camera_pos: Camera world position
            frustum: View frustum for culling
        """
        self._visible_instances = 0

        for cluster in self._clusters.values():
            # Cull against frustum
            visible = cluster.cull(frustum)

            # Update LOD for visible instances
            if visible > 0:
                cluster.update_lod(camera_pos, self._foliage_type.lod_distances)

            self._visible_instances += visible

    def get_instance_buffer(self) -> List[Dict]:
        """
        Get instance data buffer for GPU rendering.

        Returns:
            List of instance data dictionaries
        """
        buffer = []
        for cluster in self._clusters.values():
            for inst in cluster.get_visible_instances():
                buffer.append(
                    {
                        "position": inst.position,
                        "rotation": inst.rotation,
                        "scale": inst.scale,
                        "lod_level": inst.lod_level,
                    }
                )
        return buffer

    def get_instance_buffer_by_lod(self) -> Dict[int, List[Dict]]:
        """
        Get instance data grouped by LOD level.

        Returns:
            Dictionary mapping LOD level to instance data
        """
        by_lod: Dict[int, List[Dict]] = {}
        for cluster in self._clusters.values():
            for inst in cluster.get_visible_instances():
                if inst.lod_level not in by_lod:
                    by_lod[inst.lod_level] = []
                by_lod[inst.lod_level].append(
                    {
                        "position": inst.position,
                        "rotation": inst.rotation,
                        "scale": inst.scale,
                    }
                )
        return by_lod

    def get_clusters_in_bounds(self, bounds: Bounds) -> List[FoliageCluster]:
        """
        Get clusters overlapping bounds.

        Args:
            bounds: Query region

        Returns:
            List of overlapping clusters
        """
        result = []
        min_key = self._get_cluster_key(bounds.min_x, bounds.min_z)
        max_key = self._get_cluster_key(bounds.max_x, bounds.max_z)

        for cx in range(min_key[0], max_key[0] + 1):
            for cz in range(min_key[1], max_key[1] + 1):
                key = (cx, cz)
                if key in self._clusters:
                    result.append(self._clusters[key])

        return result

    def clear(self) -> None:
        """Remove all instances and clusters."""
        self._clusters.clear()
        self._instance_lookup.clear()
        self._total_instances = 0
        self._visible_instances = 0


class FoliageManager:
    """
    Central manager for all foliage instances.

    Manages multiple HISMs for different foliage types and
    coordinates updates and rendering.
    """

    __slots__ = ("_hism_by_type", "_cluster_size")

    def __init__(self, cluster_size: float = 50.0) -> None:
        """
        Initialize foliage manager.

        Args:
            cluster_size: Default cluster size for HISMs
        """
        self._hism_by_type: Dict[str, HierarchicalInstancedMesh] = {}
        self._cluster_size = cluster_size

    def register_type(self, foliage_type: FoliageType) -> None:
        """
        Register a foliage type and create its HISM.

        Args:
            foliage_type: Foliage type to register
        """
        if foliage_type.type_id in self._hism_by_type:
            raise ValueError(
                f"Foliage type '{foliage_type.type_id}' already registered"
            )

        self._hism_by_type[foliage_type.type_id] = HierarchicalInstancedMesh(
            foliage_type, self._cluster_size
        )

    def unregister_type(self, type_id: str) -> bool:
        """
        Unregister a foliage type and remove its HISM.

        Args:
            type_id: Foliage type ID to unregister

        Returns:
            True if type was unregistered
        """
        if type_id in self._hism_by_type:
            del self._hism_by_type[type_id]
            return True
        return False

    def add_instances(self, type_id: str, placements: List[PlacementResult]) -> List[int]:
        """
        Add instances for a foliage type.

        Args:
            type_id: Foliage type ID
            placements: List of placements

        Returns:
            List of instance IDs

        Raises:
            KeyError: If type is not registered
        """
        if type_id not in self._hism_by_type:
            raise KeyError(f"Foliage type '{type_id}' not registered")

        return self._hism_by_type[type_id].add_instances(placements)

    def remove_instances(self, type_id: str, bounds: Bounds) -> int:
        """
        Remove instances within bounds for a type.

        Args:
            type_id: Foliage type ID
            bounds: Region to clear

        Returns:
            Number of instances removed

        Raises:
            KeyError: If type is not registered
        """
        if type_id not in self._hism_by_type:
            raise KeyError(f"Foliage type '{type_id}' not registered")

        return self._hism_by_type[type_id].remove_instances_in_bounds(bounds)

    def remove_all_instances(self, bounds: Bounds) -> int:
        """
        Remove instances within bounds for all types.

        Args:
            bounds: Region to clear

        Returns:
            Total number of instances removed
        """
        total = 0
        for hism in self._hism_by_type.values():
            total += hism.remove_instances_in_bounds(bounds)
        return total

    def update(
        self, camera_position: Tuple[float, float, float], frustum: Frustum
    ) -> None:
        """
        Update all foliage visibility and LOD.

        Args:
            camera_position: Camera world position
            frustum: View frustum for culling
        """
        for hism in self._hism_by_type.values():
            hism.update_visibility(camera_position, frustum)

    def get_render_data(self) -> List[Tuple[str, List[Dict]]]:
        """
        Get render data for all foliage types.

        Returns:
            List of (type_id, instance_buffer) tuples
        """
        result = []
        for type_id, hism in self._hism_by_type.items():
            buffer = hism.get_instance_buffer()
            if buffer:
                result.append((type_id, buffer))
        return result

    def get_render_data_by_lod(self) -> List[Tuple[str, Dict[int, List[Dict]]]]:
        """
        Get render data grouped by LOD for all types.

        Returns:
            List of (type_id, lod_buffer_dict) tuples
        """
        result = []
        for type_id, hism in self._hism_by_type.items():
            by_lod = hism.get_instance_buffer_by_lod()
            if by_lod:
                result.append((type_id, by_lod))
        return result

    def get_hism(self, type_id: str) -> Optional[HierarchicalInstancedMesh]:
        """
        Get HISM for a foliage type.

        Args:
            type_id: Foliage type ID

        Returns:
            HISM if type is registered, None otherwise
        """
        return self._hism_by_type.get(type_id)

    def get_total_instances(self) -> int:
        """Get total instance count across all types."""
        return sum(h.total_instances for h in self._hism_by_type.values())

    def get_visible_instances(self) -> int:
        """Get visible instance count across all types."""
        return sum(h.visible_instances for h in self._hism_by_type.values())

    def get_registered_types(self) -> List[str]:
        """Get list of registered type IDs."""
        return list(self._hism_by_type.keys())

    def clear(self) -> None:
        """Remove all HISMs and instances."""
        self._hism_by_type.clear()

    def clear_type(self, type_id: str) -> bool:
        """
        Clear all instances for a type.

        Args:
            type_id: Type to clear

        Returns:
            True if type exists
        """
        if type_id in self._hism_by_type:
            self._hism_by_type[type_id].clear()
            return True
        return False


class BatchedDescriptor:
    """
    Batched operation descriptor for bulk foliage operations.

    Collects operations and applies them in batch for efficiency.
    """

    __slots__ = ("_manager", "_pending_adds", "_pending_removes")

    def __init__(self, manager: FoliageManager) -> None:
        """
        Initialize batched descriptor.

        Args:
            manager: Foliage manager to operate on
        """
        self._manager = manager
        self._pending_adds: Dict[str, List[PlacementResult]] = {}
        self._pending_removes: List[Tuple[str, Bounds]] = []

    def add(self, type_id: str, placements: List[PlacementResult]) -> "BatchedDescriptor":
        """
        Queue placements for batch add.

        Args:
            type_id: Foliage type ID
            placements: Placements to add

        Returns:
            Self for chaining
        """
        if type_id not in self._pending_adds:
            self._pending_adds[type_id] = []
        self._pending_adds[type_id].extend(placements)
        return self

    def remove(self, type_id: str, bounds: Bounds) -> "BatchedDescriptor":
        """
        Queue region for batch removal.

        Args:
            type_id: Foliage type ID
            bounds: Region to clear

        Returns:
            Self for chaining
        """
        self._pending_removes.append((type_id, bounds))
        return self

    def execute(self) -> Tuple[int, int]:
        """
        Execute all pending operations.

        Returns:
            Tuple of (instances added, instances removed)
        """
        added = 0
        removed = 0

        # Process removes first
        for type_id, bounds in self._pending_removes:
            try:
                removed += self._manager.remove_instances(type_id, bounds)
            except KeyError:
                pass

        # Process adds
        for type_id, placements in self._pending_adds.items():
            try:
                ids = self._manager.add_instances(type_id, placements)
                added += len(ids)
            except KeyError:
                pass

        # Clear pending
        self._pending_adds.clear()
        self._pending_removes.clear()

        return added, removed

    def clear(self) -> None:
        """Clear all pending operations without executing."""
        self._pending_adds.clear()
        self._pending_removes.clear()
