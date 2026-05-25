"""
Broadphase Collision Detection Algorithms.

This module implements various spatial partitioning structures for efficient
broadphase collision detection:
- Sweep and Prune (SAP) - axis-aligned sorting
- Dynamic BVH - bounding volume hierarchy with incremental updates
- Spatial Hash Grid - uniform grid spatial hashing
- Octree - hierarchical grid subdivision
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Generic, Iterator, TypeVar, Optional
import math

from .config import (
    BROADPHASE_MARGIN,
    SPATIAL_HASH_CELL_SIZE,
    OCTREE_MAX_DEPTH,
    OCTREE_MAX_OBJECTS_PER_LEAF,
    SAP_PRIMARY_AXIS,
    INITIAL_PAIR_CAPACITY,
    BVH_REBALANCE_THRESHOLD,
    NUMERICAL_EPSILON,
)


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar("T")  # Generic type for user data


@dataclass
class Vec3:
    """Simple 3D vector for collision calculations."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        raise IndexError(f"Vec3 index {index} out of range")

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vec3":
        length = self.length()
        if length < NUMERICAL_EPSILON:
            return Vec3(0.0, 0.0, 0.0)
        return self * (1.0 / length)

    def min_components(self, other: "Vec3") -> "Vec3":
        return Vec3(min(self.x, other.x), min(self.y, other.y), min(self.z, other.z))

    def max_components(self, other: "Vec3") -> "Vec3":
        return Vec3(max(self.x, other.x), max(self.y, other.y), max(self.z, other.z))


@dataclass
class AABB:
    """Axis-Aligned Bounding Box."""

    min_point: Vec3 = field(default_factory=Vec3)
    max_point: Vec3 = field(default_factory=Vec3)

    @classmethod
    def from_center_extents(cls, center: Vec3, extents: Vec3) -> "AABB":
        """Create AABB from center and half-extents."""
        return cls(center - extents, center + extents)

    @classmethod
    def from_points(cls, points: list[Vec3]) -> "AABB":
        """Create AABB enclosing all points."""
        if not points:
            return cls()
        min_p = Vec3(float("inf"), float("inf"), float("inf"))
        max_p = Vec3(float("-inf"), float("-inf"), float("-inf"))
        for p in points:
            min_p = min_p.min_components(p)
            max_p = max_p.max_components(p)
        return cls(min_p, max_p)

    def center(self) -> Vec3:
        """Get center point of AABB."""
        return (self.min_point + self.max_point) * 0.5

    def extents(self) -> Vec3:
        """Get half-extents of AABB."""
        return (self.max_point - self.min_point) * 0.5

    def size(self) -> Vec3:
        """Get full size of AABB."""
        return self.max_point - self.min_point

    def surface_area(self) -> float:
        """Calculate surface area for SAH heuristic."""
        size = self.size()
        return 2.0 * (size.x * size.y + size.y * size.z + size.z * size.x)

    def volume(self) -> float:
        """Calculate volume."""
        size = self.size()
        return size.x * size.y * size.z

    def expanded(self, margin: float) -> "AABB":
        """Return expanded AABB by margin on all sides."""
        offset = Vec3(margin, margin, margin)
        return AABB(self.min_point - offset, self.max_point + offset)

    def contains_point(self, point: Vec3) -> bool:
        """Check if point is inside AABB."""
        return (
            self.min_point.x <= point.x <= self.max_point.x
            and self.min_point.y <= point.y <= self.max_point.y
            and self.min_point.z <= point.z <= self.max_point.z
        )

    def intersects(self, other: "AABB") -> bool:
        """Check if two AABBs intersect."""
        return (
            self.min_point.x <= other.max_point.x
            and self.max_point.x >= other.min_point.x
            and self.min_point.y <= other.max_point.y
            and self.max_point.y >= other.min_point.y
            and self.min_point.z <= other.max_point.z
            and self.max_point.z >= other.min_point.z
        )

    def merge(self, other: "AABB") -> "AABB":
        """Return AABB containing both AABBs."""
        return AABB(
            self.min_point.min_components(other.min_point),
            self.max_point.max_components(other.max_point),
        )

    def ray_intersect(
        self, origin: Vec3, direction: Vec3
    ) -> tuple[bool, float, float]:
        """
        Ray-AABB intersection test.

        Returns (hit, t_min, t_max) where hit is True if ray intersects.
        """
        t_min = float("-inf")
        t_max = float("inf")

        for i in range(3):
            if abs(direction[i]) < NUMERICAL_EPSILON:
                # Ray parallel to slab
                if origin[i] < self.min_point[i] or origin[i] > self.max_point[i]:
                    return (False, 0.0, 0.0)
            else:
                inv_d = 1.0 / direction[i]
                t1 = (self.min_point[i] - origin[i]) * inv_d
                t2 = (self.max_point[i] - origin[i]) * inv_d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return (False, 0.0, 0.0)

        return (True, t_min, t_max)


@dataclass
class Ray:
    """Ray for raycasting queries."""

    origin: Vec3 = field(default_factory=Vec3)
    direction: Vec3 = field(default_factory=lambda: Vec3(0, 0, 1))
    max_distance: float = float("inf")

    def point_at(self, t: float) -> Vec3:
        """Get point at parameter t along ray."""
        return self.origin + self.direction * t


@dataclass
class CollisionPair:
    """Represents a potential collision pair from broadphase."""

    id_a: int
    id_b: int
    aabb_a: AABB | None = None
    aabb_b: AABB | None = None

    def __hash__(self) -> int:
        # Ensure consistent ordering for hashing
        min_id = min(self.id_a, self.id_b)
        max_id = max(self.id_a, self.id_b)
        return hash((min_id, max_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CollisionPair):
            return False
        return (self.id_a == other.id_a and self.id_b == other.id_b) or (
            self.id_a == other.id_b and self.id_b == other.id_a
        )


@dataclass
class RaycastHit:
    """Result of a raycast query."""

    object_id: int
    distance: float
    point: Vec3
    normal: Vec3 | None = None


# =============================================================================
# Enums
# =============================================================================


class BroadphaseType(Enum):
    """Available broadphase algorithms."""

    SAP = auto()      # Sweep and Prune
    BVH = auto()      # Bounding Volume Hierarchy
    GRID = auto()     # Spatial Hash Grid
    OCTREE = auto()   # Octree


# =============================================================================
# Abstract Base Class
# =============================================================================


class Broadphase(ABC, Generic[T]):
    """
    Abstract base class for broadphase collision detection.

    Broadphase algorithms quickly cull pairs that cannot possibly collide
    by using spatial partitioning and bounding volume tests.
    """

    def __init__(self, margin: float = BROADPHASE_MARGIN):
        self._margin = margin
        self._objects: dict[int, tuple[AABB, T | None]] = {}
        self._next_id = 0
        self._dirty = False

    @property
    def margin(self) -> float:
        """Get broadphase margin."""
        return self._margin

    @property
    def object_count(self) -> int:
        """Get number of objects in broadphase."""
        return len(self._objects)

    def insert(self, aabb: AABB, user_data: T | None = None) -> int:
        """
        Insert an object into the broadphase.

        Args:
            aabb: Axis-aligned bounding box of the object
            user_data: Optional user data associated with object

        Returns:
            Unique identifier for the object
        """
        object_id = self._next_id
        self._next_id += 1
        expanded = aabb.expanded(self._margin)
        self._objects[object_id] = (expanded, user_data)
        self._insert_impl(object_id, expanded, user_data)
        self._dirty = True
        return object_id

    def remove(self, object_id: int) -> bool:
        """
        Remove an object from the broadphase.

        Args:
            object_id: ID of object to remove

        Returns:
            True if object was removed, False if not found
        """
        if object_id not in self._objects:
            return False
        aabb, user_data = self._objects.pop(object_id)
        self._remove_impl(object_id, aabb, user_data)
        self._dirty = True
        return True

    def update_aabb(self, object_id: int, new_aabb: AABB) -> bool:
        """
        Update the AABB of an existing object.

        Args:
            object_id: ID of object to update
            new_aabb: New bounding box

        Returns:
            True if object was updated, False if not found
        """
        if object_id not in self._objects:
            return False
        old_aabb, user_data = self._objects[object_id]
        expanded = new_aabb.expanded(self._margin)

        # Check if update is necessary (new AABB outside old fat AABB)
        if not self._needs_update(old_aabb, new_aabb):
            return True

        self._objects[object_id] = (expanded, user_data)
        self._update_impl(object_id, old_aabb, expanded, user_data)
        self._dirty = True
        return True

    def _needs_update(self, fat_aabb: AABB, tight_aabb: AABB) -> bool:
        """Check if object moved outside its fat AABB."""
        return (
            tight_aabb.min_point.x < fat_aabb.min_point.x
            or tight_aabb.min_point.y < fat_aabb.min_point.y
            or tight_aabb.min_point.z < fat_aabb.min_point.z
            or tight_aabb.max_point.x > fat_aabb.max_point.x
            or tight_aabb.max_point.y > fat_aabb.max_point.y
            or tight_aabb.max_point.z > fat_aabb.max_point.z
        )

    def query_overlaps(self) -> list[CollisionPair]:
        """
        Query all overlapping pairs.

        Returns:
            List of collision pairs from broadphase
        """
        if self._dirty:
            self._rebuild_if_needed()
            self._dirty = False
        return self._query_overlaps_impl()

    def query_aabb(self, aabb: AABB) -> list[int]:
        """
        Query all objects overlapping with given AABB.

        Args:
            aabb: Query bounding box

        Returns:
            List of object IDs overlapping the query AABB
        """
        return self._query_aabb_impl(aabb)

    def query_ray(
        self, ray: Ray, filter_fn: Callable[[int], bool] | None = None
    ) -> list[RaycastHit]:
        """
        Cast a ray and return all hits.

        Args:
            ray: Ray to cast
            filter_fn: Optional filter function for objects

        Returns:
            List of ray hits sorted by distance
        """
        hits = self._query_ray_impl(ray, filter_fn)
        hits.sort(key=lambda h: h.distance)
        return hits

    def get_aabb(self, object_id: int) -> AABB | None:
        """Get the AABB for an object."""
        if object_id in self._objects:
            return self._objects[object_id][0]
        return None

    def get_user_data(self, object_id: int) -> T | None:
        """Get user data for an object."""
        if object_id in self._objects:
            return self._objects[object_id][1]
        return None

    def clear(self) -> None:
        """Remove all objects from broadphase."""
        self._objects.clear()
        self._clear_impl()
        self._dirty = False

    @abstractmethod
    def _insert_impl(self, object_id: int, aabb: AABB, user_data: T | None) -> None:
        """Implementation-specific insert."""
        pass

    @abstractmethod
    def _remove_impl(self, object_id: int, aabb: AABB, user_data: T | None) -> None:
        """Implementation-specific remove."""
        pass

    @abstractmethod
    def _update_impl(
        self, object_id: int, old_aabb: AABB, new_aabb: AABB, user_data: T | None
    ) -> None:
        """Implementation-specific update."""
        pass

    @abstractmethod
    def _query_overlaps_impl(self) -> list[CollisionPair]:
        """Implementation-specific overlap query."""
        pass

    @abstractmethod
    def _query_aabb_impl(self, aabb: AABB) -> list[int]:
        """Implementation-specific AABB query."""
        pass

    @abstractmethod
    def _query_ray_impl(
        self, ray: Ray, filter_fn: Callable[[int], bool] | None
    ) -> list[RaycastHit]:
        """Implementation-specific ray query."""
        pass

    @abstractmethod
    def _clear_impl(self) -> None:
        """Implementation-specific clear."""
        pass

    def _rebuild_if_needed(self) -> None:
        """Optional rebuild for implementations that need it."""
        pass


# =============================================================================
# Sweep and Prune (SAP)
# =============================================================================


@dataclass
class SAPEndpoint:
    """Endpoint for SAP algorithm."""

    value: float
    object_id: int
    is_min: bool


class SweepAndPrune(Broadphase[T]):
    """
    Sweep and Prune broadphase using axis-aligned sorting.

    Efficient for scenes where objects move incrementally between frames.
    Time complexity: O(n log n) rebuild, O(n + k) for k overlaps.
    """

    def __init__(
        self, margin: float = BROADPHASE_MARGIN, primary_axis: int = SAP_PRIMARY_AXIS
    ):
        super().__init__(margin)
        self._primary_axis = primary_axis
        self._endpoints: list[SAPEndpoint] = []
        self._sorted_ids: list[int] = []

    def _insert_impl(self, object_id: int, aabb: AABB, user_data: T | None) -> None:
        # Endpoints will be rebuilt on next query
        pass

    def _remove_impl(self, object_id: int, aabb: AABB, user_data: T | None) -> None:
        # Endpoints will be rebuilt on next query
        pass

    def _update_impl(
        self, object_id: int, old_aabb: AABB, new_aabb: AABB, user_data: T | None
    ) -> None:
        # Endpoints will be rebuilt on next query
        pass

    def _rebuild_if_needed(self) -> None:
        """Rebuild sorted endpoint list."""
        self._endpoints.clear()
        for obj_id, (aabb, _) in self._objects.items():
            self._endpoints.append(
                SAPEndpoint(aabb.min_point[self._primary_axis], obj_id, True)
            )
            self._endpoints.append(
                SAPEndpoint(aabb.max_point[self._primary_axis], obj_id, False)
            )
        # Sort by value
        self._endpoints.sort(key=lambda e: (e.value, not e.is_min))

    def _query_overlaps_impl(self) -> list[CollisionPair]:
        """Find all overlapping pairs using SAP."""
        pairs: list[CollisionPair] = []
        active: set[int] = set()

        for endpoint in self._endpoints:
            if endpoint.is_min:
                # Starting a new interval - check against all active
                aabb_a = self._objects[endpoint.object_id][0]
                for other_id in active:
                    aabb_b = self._objects[other_id][0]
                    # Full AABB test on other axes
                    if aabb_a.intersects(aabb_b):
                        pairs.append(
                            CollisionPair(endpoint.object_id, other_id, aabb_a, aabb_b)
                        )
                active.add(endpoint.object_id)
            else:
                # Ending interval
                active.discard(endpoint.object_id)

        return pairs

    def _query_aabb_impl(self, query_aabb: AABB) -> list[int]:
        """Query objects overlapping AABB."""
        results: list[int] = []
        for obj_id, (aabb, _) in self._objects.items():
            if aabb.intersects(query_aabb):
                results.append(obj_id)
        return results

    def _query_ray_impl(
        self, ray: Ray, filter_fn: Callable[[int], bool] | None
    ) -> list[RaycastHit]:
        """Ray query through all objects."""
        hits: list[RaycastHit] = []
        for obj_id, (aabb, _) in self._objects.items():
            if filter_fn and not filter_fn(obj_id):
                continue
            hit, t_min, t_max = aabb.ray_intersect(ray.origin, ray.direction)
            if hit and t_min >= 0 and t_min <= ray.max_distance:
                point = ray.point_at(t_min)
                hits.append(RaycastHit(obj_id, t_min, point))
        return hits

    def _clear_impl(self) -> None:
        """Clear SAP data structures."""
        self._endpoints.clear()
        self._sorted_ids.clear()


# =============================================================================
# Dynamic BVH
# =============================================================================


@dataclass
class BVHNode:
    """Node in the BVH tree."""

    aabb: AABB
    parent: int = -1
    left: int = -1
    right: int = -1
    object_id: int = -1  # -1 for internal nodes, >= 0 for leaves
    height: int = 0

    def is_leaf(self) -> bool:
        return self.left == -1


class DynamicBVH(Broadphase[T]):
    """
    Dynamic Bounding Volume Hierarchy with incremental updates.

    Uses SAH (Surface Area Heuristic) for optimal tree construction.
    Supports insertion, removal, and refit operations.
    Time complexity: O(log n) insert/remove, O(n) overlap query.
    """

    def __init__(
        self, margin: float = BROADPHASE_MARGIN,
        rebalance_threshold: float = BVH_REBALANCE_THRESHOLD
    ):
        super().__init__(margin)
        self._nodes: list[BVHNode] = []
        self._root: int = -1
        self._free_list: list[int] = []
        self._object_to_node: dict[int, int] = {}
        self._insertions_since_rebuild = 0
        self._rebalance_threshold = rebalance_threshold

    def _allocate_node(self) -> int:
        """Allocate a new node or reuse from free list."""
        if self._free_list:
            return self._free_list.pop()
        node_id = len(self._nodes)
        self._nodes.append(BVHNode(AABB()))
        return node_id

    def _free_node(self, node_id: int) -> None:
        """Return node to free list."""
        self._nodes[node_id] = BVHNode(AABB())
        self._free_list.append(node_id)

    def _insert_leaf(self, leaf_id: int) -> None:
        """Insert a leaf node into the tree."""
        if self._root == -1:
            self._root = leaf_id
            self._nodes[leaf_id].parent = -1
            return

        # Find best sibling using SAH
        leaf_aabb = self._nodes[leaf_id].aabb
        sibling = self._find_best_sibling(leaf_aabb)

        # Create new parent
        old_parent = self._nodes[sibling].parent
        new_parent = self._allocate_node()
        self._nodes[new_parent].parent = old_parent
        self._nodes[new_parent].aabb = leaf_aabb.merge(self._nodes[sibling].aabb)
        self._nodes[new_parent].height = self._nodes[sibling].height + 1

        if old_parent != -1:
            # Sibling was not root
            if self._nodes[old_parent].left == sibling:
                self._nodes[old_parent].left = new_parent
            else:
                self._nodes[old_parent].right = new_parent
        else:
            # Sibling was root
            self._root = new_parent

        self._nodes[new_parent].left = sibling
        self._nodes[new_parent].right = leaf_id
        self._nodes[sibling].parent = new_parent
        self._nodes[leaf_id].parent = new_parent

        # Walk back up fixing heights and AABBs
        self._fix_upward(new_parent)

    def _find_best_sibling(self, leaf_aabb: AABB) -> int:
        """Find best sibling for insertion using SAH."""
        best_sibling = self._root
        best_cost = float("inf")

        # Simple implementation - traverse tree
        stack = [self._root]
        while stack:
            node_id = stack.pop()
            node = self._nodes[node_id]

            combined = node.aabb.merge(leaf_aabb)
            combined_cost = combined.surface_area()

            if node.is_leaf():
                cost = combined_cost
                if cost < best_cost:
                    best_cost = cost
                    best_sibling = node_id
            else:
                # Cost of creating new parent here
                inherited_cost = combined_cost - node.aabb.surface_area()
                cost = combined_cost

                if cost < best_cost:
                    best_cost = cost
                    best_sibling = node_id

                # Check if we should descend
                lower_bound = leaf_aabb.surface_area() + inherited_cost
                if lower_bound < best_cost:
                    stack.append(node.left)
                    stack.append(node.right)

        return best_sibling

    def _remove_leaf(self, leaf_id: int) -> None:
        """Remove a leaf from the tree."""
        if leaf_id == self._root:
            self._root = -1
            return

        parent = self._nodes[leaf_id].parent
        grandparent = self._nodes[parent].parent
        sibling = (
            self._nodes[parent].right
            if self._nodes[parent].left == leaf_id
            else self._nodes[parent].left
        )

        if grandparent != -1:
            # Replace parent with sibling
            if self._nodes[grandparent].left == parent:
                self._nodes[grandparent].left = sibling
            else:
                self._nodes[grandparent].right = sibling
            self._nodes[sibling].parent = grandparent
            self._fix_upward(grandparent)
        else:
            # Parent was root
            self._root = sibling
            self._nodes[sibling].parent = -1

        self._free_node(parent)

    def _fix_upward(self, node_id: int) -> None:
        """Walk up tree fixing heights and AABBs."""
        current = node_id
        while current != -1:
            current = self._balance(current)
            node = self._nodes[current]

            left = self._nodes[node.left]
            right = self._nodes[node.right]
            node.height = 1 + max(left.height, right.height)
            node.aabb = left.aabb.merge(right.aabb)

            current = node.parent

    def _balance(self, node_id: int) -> int:
        """Balance tree using rotations."""
        node = self._nodes[node_id]
        if node.is_leaf() or node.height < 2:
            return node_id

        left_id = node.left
        right_id = node.right
        left = self._nodes[left_id]
        right = self._nodes[right_id]

        balance = right.height - left.height

        # Rotate right branch up
        if balance > 1:
            right_left_id = right.left
            right_right_id = right.right
            right_left = self._nodes[right_left_id]
            right_right = self._nodes[right_right_id]

            # Swap node and right
            right.left = node_id
            right.parent = node.parent
            node.parent = right_id

            if right.parent != -1:
                if self._nodes[right.parent].left == node_id:
                    self._nodes[right.parent].left = right_id
                else:
                    self._nodes[right.parent].right = right_id
            else:
                self._root = right_id

            # Rotate
            if right_left.height > right_right.height:
                right.right = right_left_id
                node.right = right_right_id
                right_right.parent = node_id
                node.aabb = left.aabb.merge(right_right.aabb)
                right.aabb = node.aabb.merge(right_left.aabb)
                node.height = 1 + max(left.height, right_right.height)
                right.height = 1 + max(node.height, right_left.height)
            else:
                right.right = right_right_id
                node.right = right_left_id
                right_left.parent = node_id
                node.aabb = left.aabb.merge(right_left.aabb)
                right.aabb = node.aabb.merge(right_right.aabb)
                node.height = 1 + max(left.height, right_left.height)
                right.height = 1 + max(node.height, right_right.height)

            return right_id

        # Rotate left branch up
        if balance < -1:
            left_left_id = left.left
            left_right_id = left.right
            left_left = self._nodes[left_left_id]
            left_right = self._nodes[left_right_id]

            # Swap node and left
            left.left = node_id
            left.parent = node.parent
            node.parent = left_id

            if left.parent != -1:
                if self._nodes[left.parent].left == node_id:
                    self._nodes[left.parent].left = left_id
                else:
                    self._nodes[left.parent].right = left_id
            else:
                self._root = left_id

            # Rotate
            if left_left.height > left_right.height:
                left.right = left_left_id
                node.left = left_right_id
                left_right.parent = node_id
                node.aabb = right.aabb.merge(left_right.aabb)
                left.aabb = node.aabb.merge(left_left.aabb)
                node.height = 1 + max(right.height, left_right.height)
                left.height = 1 + max(node.height, left_left.height)
            else:
                left.right = left_right_id
                node.left = left_left_id
                left_left.parent = node_id
                node.aabb = right.aabb.merge(left_left.aabb)
                left.aabb = node.aabb.merge(left_right.aabb)
                node.height = 1 + max(right.height, left_left.height)
                left.height = 1 + max(node.height, left_right.height)

            return left_id

        return node_id

    def _insert_impl(self, object_id: int, aabb: AABB, user_data: T | None) -> None:
        """Insert object into BVH."""
        leaf_id = self._allocate_node()
        self._nodes[leaf_id].aabb = aabb
        self._nodes[leaf_id].object_id = object_id
        self._nodes[leaf_id].height = 0
        self._object_to_node[object_id] = leaf_id
        self._insert_leaf(leaf_id)
        self._insertions_since_rebuild += 1

    def _remove_impl(self, object_id: int, aabb: AABB, user_data: T | None) -> None:
        """Remove object from BVH."""
        if object_id not in self._object_to_node:
            return
        leaf_id = self._object_to_node.pop(object_id)
        self._remove_leaf(leaf_id)
        self._free_node(leaf_id)

    def _update_impl(
        self, object_id: int, old_aabb: AABB, new_aabb: AABB, user_data: T | None
    ) -> None:
        """Update object in BVH."""
        if object_id not in self._object_to_node:
            return
        leaf_id = self._object_to_node[object_id]
        self._remove_leaf(leaf_id)
        self._nodes[leaf_id].aabb = new_aabb
        self._insert_leaf(leaf_id)

    def _query_overlaps_impl(self) -> list[CollisionPair]:
        """Find all overlapping pairs in BVH."""
        pairs: list[CollisionPair] = []
        if self._root == -1:
            return pairs

        # Get all leaves
        leaves: list[int] = []
        stack = [self._root]
        while stack:
            node_id = stack.pop()
            node = self._nodes[node_id]
            if node.is_leaf():
                leaves.append(node_id)
            else:
                stack.append(node.left)
                stack.append(node.right)

        # Query each leaf against tree
        for leaf_id in leaves:
            leaf = self._nodes[leaf_id]
            self._query_aabb_recursive(
                self._root, leaf.aabb, leaf.object_id, pairs
            )

        return pairs

    def _query_aabb_recursive(
        self,
        node_id: int,
        query_aabb: AABB,
        exclude_id: int,
        pairs: list[CollisionPair],
    ) -> None:
        """Recursively query AABB against subtree."""
        if node_id == -1:
            return

        node = self._nodes[node_id]
        if not node.aabb.intersects(query_aabb):
            return

        if node.is_leaf():
            if node.object_id > exclude_id:  # Avoid duplicate pairs
                pairs.append(
                    CollisionPair(
                        exclude_id,
                        node.object_id,
                        query_aabb,
                        node.aabb,
                    )
                )
        else:
            self._query_aabb_recursive(node.left, query_aabb, exclude_id, pairs)
            self._query_aabb_recursive(node.right, query_aabb, exclude_id, pairs)

    def _query_aabb_impl(self, query_aabb: AABB) -> list[int]:
        """Query objects overlapping AABB."""
        results: list[int] = []
        if self._root == -1:
            return results

        stack = [self._root]
        while stack:
            node_id = stack.pop()
            node = self._nodes[node_id]

            if not node.aabb.intersects(query_aabb):
                continue

            if node.is_leaf():
                results.append(node.object_id)
            else:
                stack.append(node.left)
                stack.append(node.right)

        return results

    def _query_ray_impl(
        self, ray: Ray, filter_fn: Callable[[int], bool] | None
    ) -> list[RaycastHit]:
        """Cast ray through BVH."""
        hits: list[RaycastHit] = []
        if self._root == -1:
            return hits

        stack = [self._root]
        while stack:
            node_id = stack.pop()
            node = self._nodes[node_id]

            hit, t_min, t_max = node.aabb.ray_intersect(ray.origin, ray.direction)
            if not hit or t_min > ray.max_distance:
                continue

            if node.is_leaf():
                if filter_fn is None or filter_fn(node.object_id):
                    if t_min >= 0:
                        point = ray.point_at(t_min)
                        hits.append(RaycastHit(node.object_id, t_min, point))
            else:
                stack.append(node.left)
                stack.append(node.right)

        return hits

    def _clear_impl(self) -> None:
        """Clear BVH."""
        self._nodes.clear()
        self._free_list.clear()
        self._object_to_node.clear()
        self._root = -1
        self._insertions_since_rebuild = 0


# =============================================================================
# Spatial Hash Grid
# =============================================================================


class SpatialHashGrid(Broadphase[T]):
    """
    Uniform grid spatial hashing for broadphase.

    Best for scenes with uniformly distributed, similarly-sized objects.
    Time complexity: O(1) insert/remove, O(k) query where k is local density.
    """

    def __init__(
        self,
        margin: float = BROADPHASE_MARGIN,
        cell_size: float = SPATIAL_HASH_CELL_SIZE,
    ):
        super().__init__(margin)
        if cell_size <= NUMERICAL_EPSILON:
            raise ValueError(f"cell_size must be positive, got {cell_size}")
        self._cell_size = cell_size
        self._inv_cell_size = 1.0 / cell_size
        self._cells: dict[tuple[int, int, int], set[int]] = {}
        self._object_cells: dict[int, list[tuple[int, int, int]]] = {}

    def _hash_point(self, point: Vec3) -> tuple[int, int, int]:
        """Convert point to cell coordinates."""
        return (
            int(math.floor(point.x * self._inv_cell_size)),
            int(math.floor(point.y * self._inv_cell_size)),
            int(math.floor(point.z * self._inv_cell_size)),
        )

    def _get_cells_for_aabb(self, aabb: AABB) -> list[tuple[int, int, int]]:
        """Get all cells overlapped by AABB."""
        min_cell = self._hash_point(aabb.min_point)
        max_cell = self._hash_point(aabb.max_point)

        cells: list[tuple[int, int, int]] = []
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    cells.append((x, y, z))
        return cells

    def _insert_impl(self, object_id: int, aabb: AABB, user_data: T | None) -> None:
        """Insert object into hash grid."""
        cells = self._get_cells_for_aabb(aabb)
        self._object_cells[object_id] = cells
        for cell in cells:
            if cell not in self._cells:
                self._cells[cell] = set()
            self._cells[cell].add(object_id)

    def _remove_impl(self, object_id: int, aabb: AABB, user_data: T | None) -> None:
        """Remove object from hash grid."""
        if object_id not in self._object_cells:
            return
        cells = self._object_cells.pop(object_id)
        for cell in cells:
            if cell in self._cells:
                self._cells[cell].discard(object_id)
                if not self._cells[cell]:
                    del self._cells[cell]

    def _update_impl(
        self, object_id: int, old_aabb: AABB, new_aabb: AABB, user_data: T | None
    ) -> None:
        """Update object in hash grid."""
        # Remove from old cells
        if object_id in self._object_cells:
            old_cells = self._object_cells[object_id]
            for cell in old_cells:
                if cell in self._cells:
                    self._cells[cell].discard(object_id)
                    if not self._cells[cell]:
                        del self._cells[cell]

        # Add to new cells
        new_cells = self._get_cells_for_aabb(new_aabb)
        self._object_cells[object_id] = new_cells
        for cell in new_cells:
            if cell not in self._cells:
                self._cells[cell] = set()
            self._cells[cell].add(object_id)

    def _query_overlaps_impl(self) -> list[CollisionPair]:
        """Find all overlapping pairs."""
        pairs: set[CollisionPair] = set()
        checked: set[tuple[int, int]] = set()

        for cell, objects in self._cells.items():
            obj_list = list(objects)
            for i, obj_a in enumerate(obj_list):
                aabb_a = self._objects[obj_a][0]
                for obj_b in obj_list[i + 1 :]:
                    pair_key = (min(obj_a, obj_b), max(obj_a, obj_b))
                    if pair_key in checked:
                        continue
                    checked.add(pair_key)

                    aabb_b = self._objects[obj_b][0]
                    if aabb_a.intersects(aabb_b):
                        pairs.add(CollisionPair(obj_a, obj_b, aabb_a, aabb_b))

        return list(pairs)

    def _query_aabb_impl(self, query_aabb: AABB) -> list[int]:
        """Query objects overlapping AABB."""
        results: set[int] = set()
        cells = self._get_cells_for_aabb(query_aabb)

        for cell in cells:
            if cell in self._cells:
                for obj_id in self._cells[cell]:
                    if obj_id not in results:
                        aabb = self._objects[obj_id][0]
                        if aabb.intersects(query_aabb):
                            results.add(obj_id)

        return list(results)

    def _query_ray_impl(
        self, ray: Ray, filter_fn: Callable[[int], bool] | None
    ) -> list[RaycastHit]:
        """Cast ray through hash grid using 3D DDA."""
        hits: list[RaycastHit] = []
        checked: set[int] = set()

        # Start cell
        current = self._hash_point(ray.origin)

        # Step direction
        step = (
            1 if ray.direction.x >= 0 else -1,
            1 if ray.direction.y >= 0 else -1,
            1 if ray.direction.z >= 0 else -1,
        )

        # Calculate tMax and tDelta
        def safe_divide(a: float, b: float) -> float:
            return a / b if abs(b) > 1e-10 else float("inf")

        t_max = Vec3(
            safe_divide(
                (current[0] + (1 if step[0] > 0 else 0)) * self._cell_size
                - ray.origin.x,
                ray.direction.x,
            ),
            safe_divide(
                (current[1] + (1 if step[1] > 0 else 0)) * self._cell_size
                - ray.origin.y,
                ray.direction.y,
            ),
            safe_divide(
                (current[2] + (1 if step[2] > 0 else 0)) * self._cell_size
                - ray.origin.z,
                ray.direction.z,
            ),
        )

        t_delta = Vec3(
            safe_divide(self._cell_size * step[0], ray.direction.x),
            safe_divide(self._cell_size * step[1], ray.direction.y),
            safe_divide(self._cell_size * step[2], ray.direction.z),
        )

        t = 0.0
        while t < ray.max_distance:
            # Check current cell
            if current in self._cells:
                for obj_id in self._cells[current]:
                    if obj_id in checked:
                        continue
                    checked.add(obj_id)

                    if filter_fn and not filter_fn(obj_id):
                        continue

                    aabb = self._objects[obj_id][0]
                    hit, t_min, t_max_val = aabb.ray_intersect(
                        ray.origin, ray.direction
                    )
                    if hit and t_min >= 0 and t_min <= ray.max_distance:
                        point = ray.point_at(t_min)
                        hits.append(RaycastHit(obj_id, t_min, point))

            # Move to next cell
            if t_max.x < t_max.y and t_max.x < t_max.z:
                current = (current[0] + step[0], current[1], current[2])
                t = t_max.x
                t_max = Vec3(t_max.x + abs(t_delta.x), t_max.y, t_max.z)
            elif t_max.y < t_max.z:
                current = (current[0], current[1] + step[1], current[2])
                t = t_max.y
                t_max = Vec3(t_max.x, t_max.y + abs(t_delta.y), t_max.z)
            else:
                current = (current[0], current[1], current[2] + step[2])
                t = t_max.z
                t_max = Vec3(t_max.x, t_max.y, t_max.z + abs(t_delta.z))

        return hits

    def _clear_impl(self) -> None:
        """Clear hash grid."""
        self._cells.clear()
        self._object_cells.clear()


# =============================================================================
# Octree
# =============================================================================


@dataclass
class OctreeNode:
    """Node in the octree."""

    bounds: AABB
    objects: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)  # 8 children
    parent: int = -1
    depth: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class Octree(Broadphase[T]):
    """
    Octree spatial partitioning for broadphase.

    Best for scenes with varying object densities and sizes.
    Provides logarithmic query times in sparse regions.
    """

    def __init__(
        self,
        margin: float = BROADPHASE_MARGIN,
        max_depth: int = OCTREE_MAX_DEPTH,
        max_objects_per_leaf: int = OCTREE_MAX_OBJECTS_PER_LEAF,
        bounds: AABB | None = None,
    ):
        super().__init__(margin)
        self._max_depth = max_depth
        self._max_objects_per_leaf = max_objects_per_leaf
        self._bounds = bounds or AABB(
            Vec3(-100, -100, -100), Vec3(100, 100, 100)
        )
        self._nodes: list[OctreeNode] = [OctreeNode(self._bounds)]
        self._object_nodes: dict[int, list[int]] = {}
        self._free_nodes: list[int] = []

    def _allocate_node(self, bounds: AABB, parent: int, depth: int) -> int:
        """Allocate a new octree node."""
        if self._free_nodes:
            node_id = self._free_nodes.pop()
            self._nodes[node_id] = OctreeNode(bounds, parent=parent, depth=depth)
        else:
            node_id = len(self._nodes)
            self._nodes.append(OctreeNode(bounds, parent=parent, depth=depth))
        return node_id

    def _get_child_bounds(self, parent_bounds: AABB, octant: int) -> AABB:
        """Get bounds for child octant (0-7)."""
        center = parent_bounds.center()
        min_p = parent_bounds.min_point
        max_p = parent_bounds.max_point

        # Octant encoding: bit 0 = x, bit 1 = y, bit 2 = z
        child_min = Vec3(
            center.x if (octant & 1) else min_p.x,
            center.y if (octant & 2) else min_p.y,
            center.z if (octant & 4) else min_p.z,
        )
        child_max = Vec3(
            max_p.x if (octant & 1) else center.x,
            max_p.y if (octant & 2) else center.y,
            max_p.z if (octant & 4) else center.z,
        )
        return AABB(child_min, child_max)

    def _get_containing_octants(self, node_bounds: AABB, aabb: AABB) -> list[int]:
        """Get octants that contain or intersect the AABB."""
        center = node_bounds.center()
        octants: list[int] = []

        for i in range(8):
            child_bounds = self._get_child_bounds(node_bounds, i)
            if child_bounds.intersects(aabb):
                octants.append(i)

        return octants

    def _subdivide(self, node_id: int) -> None:
        """Subdivide a leaf node into 8 children."""
        node = self._nodes[node_id]
        if not node.is_leaf() or node.depth >= self._max_depth:
            return

        # Create 8 children
        for i in range(8):
            child_bounds = self._get_child_bounds(node.bounds, i)
            child_id = self._allocate_node(child_bounds, node_id, node.depth + 1)
            node.children.append(child_id)

        # Redistribute objects
        for obj_id in node.objects[:]:
            obj_aabb = self._objects[obj_id][0]
            for octant in self._get_containing_octants(node.bounds, obj_aabb):
                child_id = node.children[octant]
                self._nodes[child_id].objects.append(obj_id)

        node.objects.clear()

    def _insert_into_node(self, node_id: int, object_id: int, aabb: AABB) -> None:
        """Insert object into specific node."""
        node = self._nodes[node_id]

        if node.is_leaf():
            node.objects.append(object_id)
            if object_id not in self._object_nodes:
                self._object_nodes[object_id] = []
            self._object_nodes[object_id].append(node_id)

            # Check if subdivision needed
            if (
                len(node.objects) > self._max_objects_per_leaf
                and node.depth < self._max_depth
            ):
                self._subdivide(node_id)
        else:
            # Insert into appropriate children
            for octant in self._get_containing_octants(node.bounds, aabb):
                child_id = node.children[octant]
                self._insert_into_node(child_id, object_id, aabb)

    def _remove_from_node(self, node_id: int, object_id: int) -> None:
        """Remove object from node."""
        node = self._nodes[node_id]
        if object_id in node.objects:
            node.objects.remove(object_id)

    def _insert_impl(self, object_id: int, aabb: AABB, user_data: T | None) -> None:
        """Insert object into octree."""
        # Expand bounds if necessary
        if not self._nodes[0].bounds.intersects(aabb):
            # Object outside bounds - expand root
            new_bounds = self._nodes[0].bounds.merge(aabb)
            self._nodes[0].bounds = new_bounds

        self._insert_into_node(0, object_id, aabb)

    def _remove_impl(self, object_id: int, aabb: AABB, user_data: T | None) -> None:
        """Remove object from octree."""
        if object_id in self._object_nodes:
            for node_id in self._object_nodes[object_id]:
                self._remove_from_node(node_id, object_id)
            del self._object_nodes[object_id]

    def _update_impl(
        self, object_id: int, old_aabb: AABB, new_aabb: AABB, user_data: T | None
    ) -> None:
        """Update object in octree."""
        # Simple implementation: remove and re-insert
        self._remove_impl(object_id, old_aabb, user_data)
        self._insert_impl(object_id, new_aabb, user_data)

    def _query_overlaps_impl(self) -> list[CollisionPair]:
        """Find all overlapping pairs."""
        pairs: set[CollisionPair] = set()

        def query_node(node_id: int, ancestors: list[int]) -> None:
            node = self._nodes[node_id]

            if node.is_leaf():
                # Check pairs within this leaf
                for i, obj_a in enumerate(node.objects):
                    aabb_a = self._objects[obj_a][0]
                    # Check against other objects in this leaf
                    for obj_b in node.objects[i + 1 :]:
                        aabb_b = self._objects[obj_b][0]
                        if aabb_a.intersects(aabb_b):
                            pairs.add(CollisionPair(obj_a, obj_b, aabb_a, aabb_b))
            else:
                # Query children
                for child_id in node.children:
                    query_node(child_id, ancestors + [node_id])

        query_node(0, [])
        return list(pairs)

    def _query_aabb_impl(self, query_aabb: AABB) -> list[int]:
        """Query objects overlapping AABB."""
        results: set[int] = set()

        def query_node(node_id: int) -> None:
            node = self._nodes[node_id]
            if not node.bounds.intersects(query_aabb):
                return

            if node.is_leaf():
                for obj_id in node.objects:
                    aabb = self._objects[obj_id][0]
                    if aabb.intersects(query_aabb):
                        results.add(obj_id)
            else:
                for child_id in node.children:
                    query_node(child_id)

        query_node(0)
        return list(results)

    def _query_ray_impl(
        self, ray: Ray, filter_fn: Callable[[int], bool] | None
    ) -> list[RaycastHit]:
        """Cast ray through octree."""
        hits: list[RaycastHit] = []
        checked: set[int] = set()

        def query_node(node_id: int) -> None:
            node = self._nodes[node_id]
            hit, t_min, t_max = node.bounds.ray_intersect(ray.origin, ray.direction)
            if not hit or t_min > ray.max_distance:
                return

            if node.is_leaf():
                for obj_id in node.objects:
                    if obj_id in checked:
                        continue
                    checked.add(obj_id)

                    if filter_fn and not filter_fn(obj_id):
                        continue

                    aabb = self._objects[obj_id][0]
                    hit, t_min, t_max = aabb.ray_intersect(ray.origin, ray.direction)
                    if hit and t_min >= 0 and t_min <= ray.max_distance:
                        point = ray.point_at(t_min)
                        hits.append(RaycastHit(obj_id, t_min, point))
            else:
                for child_id in node.children:
                    query_node(child_id)

        query_node(0)
        return hits

    def _clear_impl(self) -> None:
        """Clear octree."""
        self._nodes = [OctreeNode(self._bounds)]
        self._object_nodes.clear()
        self._free_nodes.clear()


# =============================================================================
# Factory Function
# =============================================================================


def create_broadphase(
    broadphase_type: BroadphaseType,
    margin: float = BROADPHASE_MARGIN,
    **kwargs,
) -> Broadphase:
    """
    Factory function to create broadphase instances.

    Args:
        broadphase_type: Type of broadphase to create
        margin: AABB expansion margin
        **kwargs: Additional arguments for specific implementations

    Returns:
        Configured broadphase instance
    """
    if broadphase_type == BroadphaseType.SAP:
        return SweepAndPrune(
            margin=margin,
            primary_axis=kwargs.get("primary_axis", SAP_PRIMARY_AXIS),
        )
    elif broadphase_type == BroadphaseType.BVH:
        return DynamicBVH(
            margin=margin,
            rebalance_threshold=kwargs.get(
                "rebalance_threshold", BVH_REBALANCE_THRESHOLD
            ),
        )
    elif broadphase_type == BroadphaseType.GRID:
        return SpatialHashGrid(
            margin=margin,
            cell_size=kwargs.get("cell_size", SPATIAL_HASH_CELL_SIZE),
        )
    elif broadphase_type == BroadphaseType.OCTREE:
        return Octree(
            margin=margin,
            max_depth=kwargs.get("max_depth", OCTREE_MAX_DEPTH),
            max_objects_per_leaf=kwargs.get(
                "max_objects_per_leaf", OCTREE_MAX_OBJECTS_PER_LEAF
            ),
            bounds=kwargs.get("bounds"),
        )
    else:
        raise ValueError(f"Unknown broadphase type: {broadphase_type}")
