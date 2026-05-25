"""
Transform Component - Position, Rotation, Scale, and Hierarchy.

Provides spatial transformation for entities with support for hierarchical
parent-child relationships and dirty tracking for efficient updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, TYPE_CHECKING
from weakref import ref, ReferenceType

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4

from trinity.descriptors import (
    TrackedDescriptor,
    StorageDescriptor,
    clear_dirty,
    get_dirty_fields,
    is_dirty,
)

from engine.gameplay.components.constants import TransformConstants

if TYPE_CHECKING:
    from foundation import to_dict, from_dict


class TransformSpace(Enum):
    """Coordinate space for transform operations."""
    LOCAL = auto()   # Relative to parent
    WORLD = auto()   # Absolute world space
    SELF = auto()    # Object's local coordinate system


@dataclass
class TransformSnapshot:
    """Immutable snapshot of transform state for interpolation/replay."""
    position: Vec3
    rotation: Quat
    scale: Vec3
    timestamp: float = 0.0

    def lerp(self, other: TransformSnapshot, t: float) -> TransformSnapshot:
        """Interpolate between two snapshots."""
        return TransformSnapshot(
            position=self.position.lerp(other.position, t),
            rotation=self.rotation.slerp(other.rotation, t),
            scale=self.scale.lerp(other.scale, t),
            timestamp=self.timestamp + (other.timestamp - self.timestamp) * t,
        )


class TransformComponent:
    """
    Transform component with position, rotation, scale, and hierarchy support.

    Features:
    - Local and world space transforms
    - Parent-child hierarchy with automatic world transform updates
    - Dirty tracking for efficient caching
    - Serialization support
    - Transform snapshots for interpolation

    Attributes:
        position: Local position relative to parent (Vec3)
        rotation: Local rotation relative to parent (Quat)
        scale: Local scale (Vec3)
        parent: Parent transform (optional)
        children: List of child transforms
    """

    # Use tracked descriptors for dirty tracking
    position = TrackedDescriptor(field_type=Vec3, use_bitmask=True, field_offset=0)
    rotation = TrackedDescriptor(field_type=Quat, use_bitmask=True, field_offset=1)
    scale = TrackedDescriptor(field_type=Vec3, use_bitmask=True, field_offset=2)

    __slots__ = (
        "__dict__",
        "_parent_ref",
        "_children",
        "_world_matrix_cache",
        "_world_matrix_dirty",
        "_local_matrix_cache",
        "_local_matrix_dirty",
        "_entity_id",
        "_on_transform_changed",
    )

    def __init__(
        self,
        position: Optional[Vec3] = None,
        rotation: Optional[Quat] = None,
        scale: Optional[Vec3] = None,
        entity_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the transform component.

        Args:
            position: Initial local position (default: origin)
            rotation: Initial local rotation (default: identity)
            scale: Initial local scale (default: 1,1,1)
            entity_id: Optional entity ID for tracking
        """
        self._parent_ref: Optional[ReferenceType[TransformComponent]] = None
        self._children: List[ReferenceType[TransformComponent]] = []
        self._world_matrix_cache: Optional[Mat4] = None
        self._world_matrix_dirty: bool = True
        self._local_matrix_cache: Optional[Mat4] = None
        self._local_matrix_dirty: bool = True
        self._entity_id = entity_id
        self._on_transform_changed: List[Callable[[TransformComponent], None]] = []

        # Set initial values (triggers tracking)
        self.position = position if position is not None else Vec3.zero()
        self.rotation = rotation if rotation is not None else Quat.identity()
        self.scale = scale if scale is not None else Vec3.one()

        # Clear initial dirty state
        clear_dirty(self)

    # =========================================================================
    # HIERARCHY MANAGEMENT
    # =========================================================================

    @property
    def parent(self) -> Optional[TransformComponent]:
        """Get the parent transform, or None if this is a root."""
        if self._parent_ref is None:
            return None
        return self._parent_ref()

    @parent.setter
    def parent(self, new_parent: Optional[TransformComponent]) -> None:
        """Set the parent transform, handling hierarchy updates."""
        old_parent = self.parent

        if old_parent is new_parent:
            return

        # Remove from old parent's children
        if old_parent is not None:
            old_parent._remove_child(self)

        # Set new parent reference
        if new_parent is None:
            self._parent_ref = None
        else:
            self._parent_ref = ref(new_parent)
            new_parent._add_child(self)

        # Mark world transforms as dirty
        self._invalidate_world_matrix()

    def _add_child(self, child: TransformComponent) -> None:
        """Add a child to this transform's children list."""
        self._children.append(ref(child))

    def _remove_child(self, child: TransformComponent) -> None:
        """Remove a child from this transform's children list."""
        self._children = [
            c for c in self._children if c() is not None and c() is not child
        ]

    @property
    def children(self) -> List[TransformComponent]:
        """Get list of child transforms."""
        # Clean up dead references and return live children
        live_children = []
        new_refs = []
        for child_ref in self._children:
            child = child_ref()
            if child is not None:
                live_children.append(child)
                new_refs.append(child_ref)
        self._children = new_refs
        return live_children

    @property
    def has_parent(self) -> bool:
        """Check if this transform has a parent."""
        return self.parent is not None

    @property
    def is_root(self) -> bool:
        """Check if this transform is a root (no parent)."""
        return self.parent is None

    def get_root(self) -> TransformComponent:
        """Get the root transform of this hierarchy."""
        current = self
        while current.parent is not None:
            current = current.parent
        return current

    def iter_ancestors(self) -> Iterator[TransformComponent]:
        """Iterate through ancestors from immediate parent to root."""
        current = self.parent
        while current is not None:
            yield current
            current = current.parent

    def iter_descendants(self) -> Iterator[TransformComponent]:
        """Iterate through all descendants (depth-first)."""
        for child in self.children:
            yield child
            yield from child.iter_descendants()

    def detach_children(self) -> List[TransformComponent]:
        """Detach all children from this transform. Returns detached children."""
        children = self.children.copy()
        for child in children:
            child.parent = None
        return children

    def reparent(self, new_parent: Optional[TransformComponent], keep_world: bool = True) -> None:
        """
        Reparent this transform, optionally keeping world position.

        Args:
            new_parent: New parent transform, or None for root
            keep_world: If True, adjust local transform to maintain world position
        """
        if keep_world:
            world_pos = self.world_position
            world_rot = self.world_rotation
            world_scale = self.world_scale

            self.parent = new_parent

            # Convert world to new local space
            self.set_world_position(world_pos)
            self.set_world_rotation(world_rot)
            # Scale is more complex, just set local for now
            if new_parent is not None:
                parent_scale = new_parent.world_scale
                self.scale = Vec3(
                    world_scale.x / parent_scale.x if parent_scale.x != 0 else world_scale.x,
                    world_scale.y / parent_scale.y if parent_scale.y != 0 else world_scale.y,
                    world_scale.z / parent_scale.z if parent_scale.z != 0 else world_scale.z,
                )
            else:
                self.scale = world_scale
        else:
            self.parent = new_parent

    # =========================================================================
    # LOCAL TRANSFORM
    # =========================================================================

    @property
    def local_matrix(self) -> Mat4:
        """Get the local transformation matrix (position * rotation * scale)."""
        if self._local_matrix_dirty or self._local_matrix_cache is None:
            t = Mat4.translation(self.position)
            r = self.rotation.to_mat4()
            s = Mat4.scale(self.scale)
            self._local_matrix_cache = t @ r @ s
            self._local_matrix_dirty = False
        return self._local_matrix_cache

    def set_local_transform(self, position: Vec3, rotation: Quat, scale: Vec3) -> None:
        """Set all local transform values at once."""
        self.position = position
        self.rotation = rotation
        self.scale = scale

    # =========================================================================
    # WORLD TRANSFORM
    # =========================================================================

    @property
    def world_matrix(self) -> Mat4:
        """Get the world transformation matrix."""
        if self._world_matrix_dirty or self._world_matrix_cache is None:
            if self.parent is not None:
                self._world_matrix_cache = self.parent.world_matrix @ self.local_matrix
            else:
                self._world_matrix_cache = self.local_matrix
            self._world_matrix_dirty = False
        return self._world_matrix_cache

    @property
    def world_position(self) -> Vec3:
        """Get world-space position."""
        m = self.world_matrix.m
        return Vec3(m[12], m[13], m[14])

    @property
    def world_rotation(self) -> Quat:
        """Get world-space rotation."""
        if self.parent is None:
            return self.rotation
        return self.parent.world_rotation * self.rotation

    @property
    def world_scale(self) -> Vec3:
        """Get world-space scale (approximate for non-uniform parent scales)."""
        if self.parent is None:
            return self.scale
        ps = self.parent.world_scale
        return Vec3(
            self.scale.x * ps.x,
            self.scale.y * ps.y,
            self.scale.z * ps.z,
        )

    def set_world_position(self, position: Vec3) -> None:
        """Set position in world space."""
        if self.parent is None:
            self.position = position
        else:
            # Transform world position to local space
            parent_inv = self.parent.world_matrix.inverse()
            local_pos = parent_inv.transform_point(position)
            self.position = local_pos

    def set_world_rotation(self, rotation: Quat) -> None:
        """Set rotation in world space."""
        if self.parent is None:
            self.rotation = rotation
        else:
            # Convert to local rotation
            parent_rot_inv = self.parent.world_rotation.inverse()
            self.rotation = parent_rot_inv * rotation

    # =========================================================================
    # TRANSFORM OPERATIONS
    # =========================================================================

    def translate(self, offset: Vec3, space: TransformSpace = TransformSpace.LOCAL) -> None:
        """
        Translate the transform.

        Args:
            offset: Translation offset
            space: Coordinate space for the offset
        """
        if space == TransformSpace.WORLD:
            self.set_world_position(self.world_position + offset)
        elif space == TransformSpace.SELF:
            # Apply offset in object's local orientation
            rotated_offset = self.rotation.rotate_vector(offset)
            self.position = self.position + rotated_offset
        else:  # LOCAL
            self.position = self.position + offset

    def rotate(self, rotation: Quat, space: TransformSpace = TransformSpace.LOCAL) -> None:
        """
        Rotate the transform.

        Args:
            rotation: Rotation to apply
            space: Coordinate space for the rotation
        """
        if space == TransformSpace.WORLD:
            world_rot = rotation * self.world_rotation
            self.set_world_rotation(world_rot)
        else:  # LOCAL or SELF
            self.rotation = self.rotation * rotation

    def rotate_around(self, point: Vec3, axis: Vec3, angle: float) -> None:
        """
        Rotate around a point in world space.

        Args:
            point: Point to rotate around
            axis: Rotation axis (will be normalized)
            angle: Rotation angle in radians
        """
        rot = Quat.from_axis_angle(axis, angle)
        # Rotate position around point
        offset = self.world_position - point
        new_offset = rot.rotate_vector(offset)
        self.set_world_position(point + new_offset)
        # Apply rotation to orientation
        self.set_world_rotation(rot * self.world_rotation)

    def look_at(self, target: Vec3, up: Vec3 = Vec3.up()) -> None:
        """
        Rotate to look at a target point.

        Args:
            target: Point to look at
            up: Up vector for orientation
        """
        direction = (target - self.world_position).normalized()
        if direction.length_squared() < TransformConstants.LOOK_AT_DIRECTION_EPSILON:
            return

        # Build rotation from direction
        forward = direction
        right = up.cross(forward).normalized()
        if right.length_squared() < TransformConstants.LOOK_AT_RIGHT_EPSILON:
            right = Vec3.right()
        actual_up = forward.cross(right)

        # Convert basis to quaternion
        m00, m01, m02 = right.x, actual_up.x, forward.x
        m10, m11, m12 = right.y, actual_up.y, forward.y
        m20, m21, m22 = right.z, actual_up.z, forward.z

        trace = m00 + m11 + m22
        import math
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m21 - m12) * s
            y = (m02 - m20) * s
            z = (m10 - m01) * s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

        self.set_world_rotation(Quat(x, y, z, w).normalized())

    # =========================================================================
    # DIRECTION VECTORS
    # =========================================================================

    @property
    def forward(self) -> Vec3:
        """Get the forward direction vector in world space."""
        return self.world_rotation.rotate_vector(Vec3(0, 0, -1))

    @property
    def up(self) -> Vec3:
        """Get the up direction vector in world space."""
        return self.world_rotation.rotate_vector(Vec3(0, 1, 0))

    @property
    def right(self) -> Vec3:
        """Get the right direction vector in world space."""
        return self.world_rotation.rotate_vector(Vec3(1, 0, 0))

    # =========================================================================
    # DIRTY TRACKING
    # =========================================================================

    def _invalidate_world_matrix(self) -> None:
        """Mark world matrix as needing recalculation."""
        self._world_matrix_dirty = True
        self._local_matrix_dirty = True
        # Propagate to children
        for child in self.children:
            child._invalidate_world_matrix()
        # Notify listeners
        for callback in self._on_transform_changed:
            callback(self)

    def mark_dirty(self) -> None:
        """Manually mark transform as dirty."""
        self._invalidate_world_matrix()

    def is_transform_dirty(self) -> bool:
        """Check if any transform property has changed."""
        return is_dirty(self, "position") or is_dirty(self, "rotation") or is_dirty(self, "scale")

    def clear_dirty_flags(self) -> None:
        """Clear all dirty flags."""
        clear_dirty(self)

    def on_transform_changed(self, callback: Callable[[TransformComponent], None]) -> None:
        """Register a callback for transform changes."""
        self._on_transform_changed.append(callback)

    def off_transform_changed(self, callback: Callable[[TransformComponent], None]) -> None:
        """Unregister a transform change callback."""
        if callback in self._on_transform_changed:
            self._on_transform_changed.remove(callback)

    # =========================================================================
    # SNAPSHOTS AND INTERPOLATION
    # =========================================================================

    def create_snapshot(self, timestamp: float = 0.0) -> TransformSnapshot:
        """Create a snapshot of current transform state."""
        return TransformSnapshot(
            position=Vec3(self.position.x, self.position.y, self.position.z),
            rotation=Quat(self.rotation.x, self.rotation.y, self.rotation.z, self.rotation.w),
            scale=Vec3(self.scale.x, self.scale.y, self.scale.z),
            timestamp=timestamp,
        )

    def apply_snapshot(self, snapshot: TransformSnapshot) -> None:
        """Apply a snapshot to this transform."""
        self.position = snapshot.position
        self.rotation = snapshot.rotation
        self.scale = snapshot.scale

    # =========================================================================
    # COORDINATE TRANSFORMATION
    # =========================================================================

    def transform_point(self, point: Vec3) -> Vec3:
        """Transform a point from local to world space."""
        return self.world_matrix.transform_point(point)

    def inverse_transform_point(self, point: Vec3) -> Vec3:
        """Transform a point from world to local space."""
        return self.world_matrix.inverse().transform_point(point)

    def transform_direction(self, direction: Vec3) -> Vec3:
        """Transform a direction from local to world space (ignores position)."""
        return self.world_rotation.rotate_vector(direction)

    def inverse_transform_direction(self, direction: Vec3) -> Vec3:
        """Transform a direction from world to local space."""
        return self.world_rotation.inverse().rotate_vector(direction)

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize transform to dictionary."""
        return {
            "position": [self.position.x, self.position.y, self.position.z],
            "rotation": [self.rotation.x, self.rotation.y, self.rotation.z, self.rotation.w],
            "scale": [self.scale.x, self.scale.y, self.scale.z],
            "entity_id": self._entity_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TransformComponent:
        """Deserialize transform from dictionary."""
        position = Vec3(*data["position"]) if "position" in data else None
        rotation = Quat(*data["rotation"]) if "rotation" in data else None
        scale = Vec3(*data["scale"]) if "scale" in data else None
        return cls(
            position=position,
            rotation=rotation,
            scale=scale,
            entity_id=data.get("entity_id"),
        )

    def __repr__(self) -> str:
        return f"TransformComponent(pos={self.position}, rot={self.rotation}, scale={self.scale})"


# Descriptor setup
TransformComponent.position.__set_name__(TransformComponent, "position")
TransformComponent.rotation.__set_name__(TransformComponent, "rotation")
TransformComponent.scale.__set_name__(TransformComponent, "scale")


__all__ = [
    "TransformComponent",
    "TransformSnapshot",
    "TransformSpace",
]
