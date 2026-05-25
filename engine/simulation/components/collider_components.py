"""
Collider Components.

Provides collider components for physics collision shapes including
sphere, box, capsule, and mesh colliders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..character.character_controller import Quaternion, Transform, Vector3


class ColliderType(str, Enum):
    """Type of collision shape."""
    SPHERE = "sphere"
    BOX = "box"
    CAPSULE = "capsule"
    MESH = "mesh"
    CONVEX_HULL = "convex_hull"
    TERRAIN = "terrain"
    COMPOUND = "compound"


@dataclass
class PhysicsMaterial:
    """
    Physics material for collision response.

    Attributes:
        friction: Coefficient of friction
        restitution: Coefficient of restitution
        friction_combine: How to combine friction ("average", "minimum", "maximum", "multiply")
        restitution_combine: How to combine restitution
    """
    friction: float = 0.5
    restitution: float = 0.0
    friction_combine: str = "average"
    restitution_combine: str = "average"


class ColliderComponent(ABC):
    """
    Base class for all collider components.

    Provides common functionality for collision shapes.
    """

    def __init__(
        self,
        entity_id: int,
        is_trigger: bool = False,
        material: Optional[PhysicsMaterial] = None,
    ):
        self._entity_id = entity_id
        self._is_trigger = is_trigger
        self._material = material or PhysicsMaterial()

        # Transform offset from entity
        self._local_position = Vector3.zero()
        self._local_rotation = Quaternion.identity()

        # State
        self._collider_id: Optional[int] = None
        self._enabled = True

        # Collision filtering
        self._collision_layer = 0
        self._collision_mask = 0xFFFF

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def entity_id(self) -> int:
        """Entity this collider belongs to."""
        return self._entity_id

    @property
    def collider_id(self) -> Optional[int]:
        """Physics collider ID."""
        return self._collider_id

    @property
    @abstractmethod
    def collider_type(self) -> ColliderType:
        """Type of this collider."""
        pass

    @property
    def is_trigger(self) -> bool:
        """Whether this is a trigger collider."""
        return self._is_trigger

    @is_trigger.setter
    def is_trigger(self, value: bool) -> None:
        self._is_trigger = value

    @property
    def enabled(self) -> bool:
        """Whether collider is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def material(self) -> PhysicsMaterial:
        """Physics material."""
        return self._material

    @property
    def local_position(self) -> Vector3:
        """Local position offset."""
        return self._local_position

    @local_position.setter
    def local_position(self, value: Vector3) -> None:
        self._local_position = value

    @property
    def local_rotation(self) -> Quaternion:
        """Local rotation offset."""
        return self._local_rotation

    @local_rotation.setter
    def local_rotation(self, value: Quaternion) -> None:
        self._local_rotation = value

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_bounds(self, world_transform: Transform) -> tuple[Vector3, Vector3]:
        """Get world-space AABB bounds (min, max)."""
        pass

    @abstractmethod
    def get_volume(self) -> float:
        """Get volume of the collider."""
        pass

    @abstractmethod
    def get_inertia_tensor(self, mass: float) -> Vector3:
        """Get inertia tensor diagonal for given mass."""
        pass

    # -------------------------------------------------------------------------
    # Common Methods
    # -------------------------------------------------------------------------

    def set_collision_filter(self, layer: int, mask: int) -> None:
        """Set collision layer and mask."""
        self._collision_layer = layer
        self._collision_mask = mask

    def set_material(self, material: PhysicsMaterial) -> None:
        """Set physics material."""
        self._material = material

    def initialize(self, collider_id: int) -> None:
        """Initialize with physics collider ID."""
        self._collider_id = collider_id

    def cleanup(self) -> None:
        """Cleanup component."""
        self._collider_id = None

    def get_state(self) -> dict[str, Any]:
        """Get serializable state."""
        return {
            "entity_id": self._entity_id,
            "type": self.collider_type.value,
            "is_trigger": self._is_trigger,
            "enabled": self._enabled,
            "local_position": (
                self._local_position.x,
                self._local_position.y,
                self._local_position.z,
            ),
            "local_rotation": (
                self._local_rotation.x,
                self._local_rotation.y,
                self._local_rotation.z,
                self._local_rotation.w,
            ),
            "material": {
                "friction": self._material.friction,
                "restitution": self._material.restitution,
            },
        }


# =============================================================================
# Sphere Collider
# =============================================================================

class SphereCollider(ColliderComponent):
    """
    Spherical collision shape.

    Simple and efficient for many use cases.
    """

    def __init__(
        self,
        entity_id: int,
        radius: float = 0.5,
        is_trigger: bool = False,
        material: Optional[PhysicsMaterial] = None,
    ):
        super().__init__(entity_id, is_trigger, material)
        self._radius = radius

    @property
    def collider_type(self) -> ColliderType:
        return ColliderType.SPHERE

    @property
    def radius(self) -> float:
        """Sphere radius."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = max(0.001, value)

    def get_bounds(self, world_transform: Transform) -> tuple[Vector3, Vector3]:
        """Get world-space AABB."""
        center = world_transform.transform_point(self._local_position)
        extent = Vector3(self._radius, self._radius, self._radius)
        return center - extent, center + extent

    def get_volume(self) -> float:
        """Get sphere volume."""
        import math
        return (4.0 / 3.0) * math.pi * self._radius ** 3

    def get_inertia_tensor(self, mass: float) -> Vector3:
        """Get inertia tensor for solid sphere."""
        inertia = (2.0 / 5.0) * mass * self._radius ** 2
        return Vector3(inertia, inertia, inertia)

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["radius"] = self._radius
        return state


# =============================================================================
# Box Collider
# =============================================================================

class BoxCollider(ColliderComponent):
    """
    Axis-aligned box collision shape.

    Efficient and suitable for many objects.
    """

    def __init__(
        self,
        entity_id: int,
        half_extents: Optional[Vector3] = None,
        is_trigger: bool = False,
        material: Optional[PhysicsMaterial] = None,
    ):
        super().__init__(entity_id, is_trigger, material)
        self._half_extents = half_extents or Vector3(0.5, 0.5, 0.5)

    @property
    def collider_type(self) -> ColliderType:
        return ColliderType.BOX

    @property
    def half_extents(self) -> Vector3:
        """Half extents of the box."""
        return self._half_extents

    @half_extents.setter
    def half_extents(self, value: Vector3) -> None:
        self._half_extents = Vector3(
            max(0.001, value.x),
            max(0.001, value.y),
            max(0.001, value.z),
        )

    @property
    def size(self) -> Vector3:
        """Full size of the box."""
        return self._half_extents * 2.0

    @size.setter
    def size(self, value: Vector3) -> None:
        self._half_extents = value * 0.5

    def get_bounds(self, world_transform: Transform) -> tuple[Vector3, Vector3]:
        """Get world-space AABB."""
        center = world_transform.transform_point(self._local_position)

        # For rotated box, compute maximum extent
        # Simplified: assumes axis-aligned
        extent = self._half_extents
        return center - extent, center + extent

    def get_volume(self) -> float:
        """Get box volume."""
        return 8.0 * self._half_extents.x * self._half_extents.y * self._half_extents.z

    def get_inertia_tensor(self, mass: float) -> Vector3:
        """Get inertia tensor for solid box."""
        x2 = self._half_extents.x ** 2
        y2 = self._half_extents.y ** 2
        z2 = self._half_extents.z ** 2
        factor = mass / 3.0

        return Vector3(
            factor * (y2 + z2),
            factor * (x2 + z2),
            factor * (x2 + y2),
        )

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["half_extents"] = (
            self._half_extents.x,
            self._half_extents.y,
            self._half_extents.z,
        )
        return state


# =============================================================================
# Capsule Collider
# =============================================================================

class CapsuleCollider(ColliderComponent):
    """
    Capsule collision shape (cylinder with hemispherical caps).

    Ideal for character controllers.
    """

    def __init__(
        self,
        entity_id: int,
        radius: float = 0.35,
        height: float = 1.8,
        is_trigger: bool = False,
        material: Optional[PhysicsMaterial] = None,
    ):
        super().__init__(entity_id, is_trigger, material)
        self._radius = radius
        self._height = height  # Total height including caps

    @property
    def collider_type(self) -> ColliderType:
        return ColliderType.CAPSULE

    @property
    def radius(self) -> float:
        """Capsule radius."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = max(0.001, value)

    @property
    def height(self) -> float:
        """Total capsule height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        self._height = max(self._radius * 2.0 + 0.001, value)

    @property
    def cylinder_height(self) -> float:
        """Height of cylindrical portion (excluding caps)."""
        return max(0.0, self._height - 2.0 * self._radius)

    def get_bounds(self, world_transform: Transform) -> tuple[Vector3, Vector3]:
        """Get world-space AABB."""
        center = world_transform.transform_point(self._local_position)

        # Capsule is oriented along Y axis
        half_height = self._height * 0.5
        extent = Vector3(self._radius, half_height, self._radius)
        return center - extent, center + extent

    def get_volume(self) -> float:
        """Get capsule volume."""
        import math
        cylinder = math.pi * self._radius ** 2 * self.cylinder_height
        sphere = (4.0 / 3.0) * math.pi * self._radius ** 3
        return cylinder + sphere

    def get_inertia_tensor(self, mass: float) -> Vector3:
        """Get inertia tensor for solid capsule."""
        import math

        r = self._radius
        h = self.cylinder_height

        # Cylinder contribution
        cyl_mass = mass * (h / (h + 4.0 * r / 3.0))
        cyl_ixx = cyl_mass * (3.0 * r ** 2 + h ** 2) / 12.0
        cyl_iyy = cyl_mass * r ** 2 / 2.0

        # Hemisphere contributions (simplified)
        cap_mass = (mass - cyl_mass) / 2.0
        cap_ixx = cap_mass * (2.0 / 5.0) * r ** 2

        ixx = cyl_ixx + 2.0 * (cap_ixx + cap_mass * (h / 2.0 + 3.0 * r / 8.0) ** 2)
        iyy = cyl_iyy + 2.0 * cap_ixx

        return Vector3(ixx, iyy, ixx)

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["radius"] = self._radius
        state["height"] = self._height
        return state


# =============================================================================
# Mesh Collider
# =============================================================================

class MeshCollider(ColliderComponent):
    """
    Arbitrary mesh collision shape.

    For static geometry with complex shapes.
    """

    def __init__(
        self,
        entity_id: int,
        vertices: Optional[list[Vector3]] = None,
        indices: Optional[list[int]] = None,
        convex: bool = False,
        is_trigger: bool = False,
        material: Optional[PhysicsMaterial] = None,
    ):
        super().__init__(entity_id, is_trigger, material)
        self._vertices = vertices or []
        self._indices = indices or []
        self._convex = convex
        self._mesh_id: Optional[int] = None

        # Cached bounds
        self._bounds_min = Vector3.zero()
        self._bounds_max = Vector3.zero()
        self._compute_bounds()

    @property
    def collider_type(self) -> ColliderType:
        return ColliderType.CONVEX_HULL if self._convex else ColliderType.MESH

    @property
    def vertices(self) -> list[Vector3]:
        """Mesh vertices."""
        return self._vertices

    @property
    def indices(self) -> list[int]:
        """Triangle indices."""
        return self._indices

    @property
    def is_convex(self) -> bool:
        """Whether mesh is convex hull."""
        return self._convex

    @property
    def triangle_count(self) -> int:
        """Number of triangles."""
        return len(self._indices) // 3

    @property
    def vertex_count(self) -> int:
        """Number of vertices."""
        return len(self._vertices)

    def set_mesh(
        self,
        vertices: list[Vector3],
        indices: list[int],
        convex: bool = False,
    ) -> None:
        """Set mesh data."""
        self._vertices = vertices
        self._indices = indices
        self._convex = convex
        self._compute_bounds()

    def _compute_bounds(self) -> None:
        """Compute local AABB."""
        if not self._vertices:
            self._bounds_min = Vector3.zero()
            self._bounds_max = Vector3.zero()
            return

        min_x = min_y = min_z = float("inf")
        max_x = max_y = max_z = float("-inf")

        for v in self._vertices:
            min_x = min(min_x, v.x)
            min_y = min(min_y, v.y)
            min_z = min(min_z, v.z)
            max_x = max(max_x, v.x)
            max_y = max(max_y, v.y)
            max_z = max(max_z, v.z)

        self._bounds_min = Vector3(min_x, min_y, min_z)
        self._bounds_max = Vector3(max_x, max_y, max_z)

    def get_bounds(self, world_transform: Transform) -> tuple[Vector3, Vector3]:
        """Get world-space AABB."""
        # Transform bounds to world space (simplified - axis-aligned)
        min_world = world_transform.transform_point(
            self._bounds_min + self._local_position
        )
        max_world = world_transform.transform_point(
            self._bounds_max + self._local_position
        )

        return (
            Vector3(
                min(min_world.x, max_world.x),
                min(min_world.y, max_world.y),
                min(min_world.z, max_world.z),
            ),
            Vector3(
                max(min_world.x, max_world.x),
                max(min_world.y, max_world.y),
                max(min_world.z, max_world.z),
            ),
        )

    def get_volume(self) -> float:
        """Approximate volume using bounding box."""
        size = self._bounds_max - self._bounds_min
        return size.x * size.y * size.z

    def get_inertia_tensor(self, mass: float) -> Vector3:
        """Approximate inertia using bounding box."""
        size = self._bounds_max - self._bounds_min
        factor = mass / 12.0

        return Vector3(
            factor * (size.y ** 2 + size.z ** 2),
            factor * (size.x ** 2 + size.z ** 2),
            factor * (size.x ** 2 + size.y ** 2),
        )

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["convex"] = self._convex
        state["vertex_count"] = len(self._vertices)
        state["triangle_count"] = self.triangle_count
        state["mesh_id"] = self._mesh_id
        return state


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ColliderType",
    "PhysicsMaterial",
    "ColliderComponent",
    "SphereCollider",
    "BoxCollider",
    "CapsuleCollider",
    "MeshCollider",
]
