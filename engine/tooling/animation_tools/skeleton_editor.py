"""Skeleton hierarchy editing and retargeting setup.

Provides tools for editing skeleton hierarchies, socket attachments,
virtual bones, and retargeting configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from engine.core.math import Mat4, Quat, Transform, Vec3


# =============================================================================
# ENUMS
# =============================================================================


class BoneEditMode(Enum):
    """Bone editing modes."""

    SELECT = auto()
    TRANSLATE = auto()
    ROTATE = auto()
    SCALE = auto()
    ADD_BONE = auto()
    ADD_SOCKET = auto()


class VirtualBoneType(Enum):
    """Types of virtual bones."""

    CONSTRAINT = auto()  # Position/rotation constraint
    LOOK_AT = auto()     # Look-at constraint
    COPY = auto()        # Copy from another bone
    MIDPOINT = auto()    # Midpoint between two bones
    DISTANCE = auto()    # Fixed distance from bone


# =============================================================================
# BONE SELECTION
# =============================================================================


@dataclass
class BoneSelection:
    """Represents bone selection state.

    Attributes:
        selected_bones: Set of selected bone indices
        selected_sockets: Set of selected socket names
        selected_virtual_bones: Set of selected virtual bone names
        primary_bone: Primary selected bone (for operations)
    """

    selected_bones: Set[int] = field(default_factory=set)
    selected_sockets: Set[str] = field(default_factory=set)
    selected_virtual_bones: Set[str] = field(default_factory=set)
    primary_bone: int = -1

    @property
    def has_selection(self) -> bool:
        """Check if anything is selected."""
        return (
            len(self.selected_bones) > 0 or
            len(self.selected_sockets) > 0 or
            len(self.selected_virtual_bones) > 0
        )

    @property
    def bone_count(self) -> int:
        """Get number of selected bones."""
        return len(self.selected_bones)

    def clear(self) -> None:
        """Clear all selections."""
        self.selected_bones.clear()
        self.selected_sockets.clear()
        self.selected_virtual_bones.clear()
        self.primary_bone = -1

    def select_bone(self, index: int, add_to_selection: bool = False) -> None:
        """Select a bone."""
        if not add_to_selection:
            self.selected_bones.clear()
            self.selected_sockets.clear()
            self.selected_virtual_bones.clear()

        self.selected_bones.add(index)
        self.primary_bone = index

    def deselect_bone(self, index: int) -> None:
        """Deselect a bone."""
        self.selected_bones.discard(index)
        if self.primary_bone == index:
            self.primary_bone = next(iter(self.selected_bones), -1)

    def toggle_bone(self, index: int) -> None:
        """Toggle bone selection."""
        if index in self.selected_bones:
            self.deselect_bone(index)
        else:
            self.select_bone(index, add_to_selection=True)


# =============================================================================
# SOCKETS
# =============================================================================


@dataclass
class Socket:
    """A socket attachment point on a bone.

    Sockets are used to attach objects (weapons, effects, etc.) to bones.

    Attributes:
        name: Socket name
        bone_name: Parent bone name
        relative_transform: Transform relative to bone
        preview_mesh: Optional preview mesh path
    """

    name: str
    bone_name: str
    relative_transform: Transform = field(default_factory=Transform.identity)
    preview_mesh: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Socket name cannot be empty")
        if not self.bone_name:
            raise ValueError("Bone name cannot be empty")

    def get_world_transform(self, bone_world_transform: Transform) -> Transform:
        """Get socket world transform given bone transform."""
        bone_matrix = bone_world_transform.to_matrix()
        socket_matrix = self.relative_transform.to_matrix()
        world_matrix = bone_matrix @ socket_matrix
        return Transform.from_matrix(world_matrix)

    def copy(self) -> Socket:
        """Create a copy of this socket."""
        return Socket(
            name=self.name,
            bone_name=self.bone_name,
            relative_transform=Transform(
                translation=Vec3(
                    self.relative_transform.translation.x,
                    self.relative_transform.translation.y,
                    self.relative_transform.translation.z,
                ),
                rotation=Quat(
                    self.relative_transform.rotation.x,
                    self.relative_transform.rotation.y,
                    self.relative_transform.rotation.z,
                    self.relative_transform.rotation.w,
                ),
                scale=Vec3(
                    self.relative_transform.scale.x,
                    self.relative_transform.scale.y,
                    self.relative_transform.scale.z,
                ),
            ),
            preview_mesh=self.preview_mesh,
        )


@dataclass
class SocketAttachment:
    """An object attached to a socket.

    Attributes:
        socket_name: Name of the socket
        asset_path: Path to attached asset
        offset: Additional transform offset
    """

    socket_name: str
    asset_path: str
    offset: Transform = field(default_factory=Transform.identity)


# =============================================================================
# VIRTUAL BONES
# =============================================================================


@dataclass
class VirtualBone:
    """A virtual bone computed from other bones.

    Virtual bones don't exist in the actual skeleton but are computed
    at runtime for use by animations or attachments.

    Attributes:
        name: Virtual bone name
        source_bone: Primary source bone name
        target_bone: Optional target bone name (for two-bone operations)
        bone_type: Type of virtual bone
        blend_factor: Blend factor for interpolation
    """

    name: str
    source_bone: str
    target_bone: Optional[str] = None
    bone_type: VirtualBoneType = VirtualBoneType.MIDPOINT
    blend_factor: float = 0.5
    look_axis: Vec3 = field(default_factory=lambda: Vec3(0, 0, 1))
    up_axis: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Virtual bone name cannot be empty")
        if not self.source_bone:
            raise ValueError("Source bone cannot be empty")
        if self.bone_type == VirtualBoneType.MIDPOINT and not self.target_bone:
            raise ValueError("Midpoint virtual bone requires target bone")

    def compute_transform(
        self,
        source_transform: Transform,
        target_transform: Optional[Transform] = None,
    ) -> Transform:
        """Compute the virtual bone transform."""
        if self.bone_type == VirtualBoneType.COPY:
            return Transform(
                translation=source_transform.translation,
                rotation=source_transform.rotation,
                scale=source_transform.scale,
            )

        if self.bone_type == VirtualBoneType.MIDPOINT and target_transform:
            pos = source_transform.translation.lerp(
                target_transform.translation,
                self.blend_factor,
            )
            rot = source_transform.rotation.slerp(
                target_transform.rotation,
                self.blend_factor,
            )
            return Transform(translation=pos, rotation=rot)

        if self.bone_type == VirtualBoneType.LOOK_AT and target_transform:
            direction = (target_transform.translation - source_transform.translation).normalized()
            # Compute look-at rotation
            rot = self._compute_look_at_rotation(direction)
            return Transform(translation=source_transform.translation, rotation=rot)

        if self.bone_type == VirtualBoneType.DISTANCE:
            offset = self.look_axis * self.blend_factor
            return Transform(
                translation=source_transform.translation + source_transform.rotation.rotate_vector(offset),
                rotation=source_transform.rotation,
            )

        return source_transform

    def _compute_look_at_rotation(self, direction: Vec3) -> Quat:
        """Compute rotation to look at direction."""
        if direction.length_squared() < 1e-6:
            return Quat.identity()

        forward = direction.normalized()
        right = self.up_axis.cross(forward)

        if right.length_squared() < 1e-6:
            right = Vec3(1, 0, 0)
        else:
            right = right.normalized()

        up = forward.cross(right).normalized()

        # Build rotation matrix and convert to quaternion
        m00, m01, m02 = right.x, up.x, forward.x
        m10, m11, m12 = right.y, up.y, forward.y
        m20, m21, m22 = right.z, up.z, forward.z

        trace = m00 + m11 + m22

        if trace > 0:
            s = 0.5 / (trace + 1.0) ** 0.5
            w = 0.25 / s
            x = (m21 - m12) * s
            y = (m02 - m20) * s
            z = (m10 - m01) * s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * (1.0 + m00 - m11 - m22) ** 0.5
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * (1.0 + m11 - m00 - m22) ** 0.5
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * (1.0 + m22 - m00 - m11) ** 0.5
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

        return Quat(x, y, z, w).normalized()


# =============================================================================
# RETARGETING
# =============================================================================


@dataclass
class RetargetSource:
    """Source skeleton for retargeting.

    Attributes:
        name: Source skeleton name
        skeleton_path: Path to skeleton asset
        reference_pose: Reference pose transforms
    """

    name: str
    skeleton_path: str
    reference_pose: Dict[str, Transform] = field(default_factory=dict)


@dataclass
class BoneMirrorPair:
    """A pair of bones that mirror each other.

    Attributes:
        left_bone: Left side bone name
        right_bone: Right side bone name
        axis: Mirror axis
    """

    left_bone: str
    right_bone: str
    axis: str = "X"  # X, Y, or Z

    def get_mirror(self, bone_name: str) -> Optional[str]:
        """Get the mirror of a bone."""
        if bone_name == self.left_bone:
            return self.right_bone
        if bone_name == self.right_bone:
            return self.left_bone
        return None


@dataclass
class RetargetMapping:
    """Bone mapping for retargeting.

    Attributes:
        source_bone: Source skeleton bone name
        target_bone: Target skeleton bone name
        translation_mode: How to handle translation
        rotation_mode: How to handle rotation
        scale_mode: How to handle scale
    """

    source_bone: str
    target_bone: str
    translation_mode: str = "retarget"  # retarget, skeleton, animation
    rotation_mode: str = "retarget"
    scale_mode: str = "skeleton"
    translation_offset: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    rotation_offset: Quat = field(default_factory=Quat.identity)

    def apply(
        self,
        source_transform: Transform,
        source_ref: Transform,
        target_ref: Transform,
    ) -> Transform:
        """Apply retargeting to transform."""
        result = Transform.identity()

        # Translation
        if self.translation_mode == "retarget":
            # Scale translation difference by skeleton ratio
            source_delta = source_transform.translation - source_ref.translation
            result.translation = target_ref.translation + source_delta + self.translation_offset
        elif self.translation_mode == "skeleton":
            result.translation = target_ref.translation + self.translation_offset
        else:  # animation
            result.translation = source_transform.translation + self.translation_offset

        # Rotation
        if self.rotation_mode == "retarget":
            # Compute relative rotation from source ref and apply to target ref
            source_rel = source_ref.rotation.inverse() * source_transform.rotation
            result.rotation = target_ref.rotation * source_rel * self.rotation_offset
        elif self.rotation_mode == "skeleton":
            result.rotation = target_ref.rotation * self.rotation_offset
        else:  # animation
            result.rotation = source_transform.rotation * self.rotation_offset

        # Scale
        if self.scale_mode == "skeleton":
            result.scale = target_ref.scale
        else:
            result.scale = source_transform.scale

        return result


# =============================================================================
# SKELETON PREVIEW
# =============================================================================


class SkeletonPreview:
    """Preview state for skeleton visualization.

    Attributes:
        show_bones: Whether to show bones
        show_sockets: Whether to show sockets
        show_virtual_bones: Whether to show virtual bones
        show_bone_names: Whether to show bone names
        bone_scale: Scale of bone visualization
    """

    def __init__(self) -> None:
        self.show_bones = True
        self.show_sockets = True
        self.show_virtual_bones = True
        self.show_bone_names = False
        self.show_axes = False
        self.bone_scale = 1.0
        self.socket_scale = 0.5
        self.selected_color = (255, 200, 0)
        self.bone_color = (200, 200, 200)
        self.socket_color = (100, 200, 100)
        self.virtual_bone_color = (100, 100, 200)
        self._camera_distance = 3.0
        self._camera_rotation = Quat.identity()

    def set_camera_distance(self, distance: float) -> None:
        """Set camera distance from skeleton."""
        self._camera_distance = max(0.1, distance)

    def rotate_camera(self, delta_yaw: float, delta_pitch: float) -> None:
        """Rotate camera around skeleton."""
        yaw = Quat.from_axis_angle(Vec3(0, 1, 0), delta_yaw)
        pitch = Quat.from_axis_angle(Vec3(1, 0, 0), delta_pitch)
        self._camera_rotation = yaw * self._camera_rotation * pitch

    def reset_camera(self) -> None:
        """Reset camera to default position."""
        self._camera_distance = 3.0
        self._camera_rotation = Quat.identity()


# =============================================================================
# SKELETON EDITOR
# =============================================================================


class SkeletonEditor:
    """Editor for skeleton hierarchies.

    Provides functionality for editing bone hierarchies, sockets, virtual
    bones, and retargeting configuration.
    """

    def __init__(self) -> None:
        self._bone_names: List[str] = []
        self._bone_parents: List[int] = []  # -1 for root
        self._bone_transforms: List[Transform] = []
        self._sockets: Dict[str, Socket] = {}
        self._virtual_bones: Dict[str, VirtualBone] = {}
        self._mirror_pairs: List[BoneMirrorPair] = []
        self._retarget_sources: Dict[str, RetargetSource] = {}
        self._retarget_mappings: List[RetargetMapping] = []

        self._selection = BoneSelection()
        self._edit_mode = BoneEditMode.SELECT
        self._preview = SkeletonPreview()

        self._on_change_callbacks: List[Callable[[], None]] = []

    @property
    def bone_count(self) -> int:
        """Get number of bones."""
        return len(self._bone_names)

    @property
    def bone_names(self) -> List[str]:
        """Get all bone names."""
        return list(self._bone_names)

    @property
    def selection(self) -> BoneSelection:
        """Get current selection."""
        return self._selection

    @property
    def edit_mode(self) -> BoneEditMode:
        """Get current edit mode."""
        return self._edit_mode

    @edit_mode.setter
    def edit_mode(self, mode: BoneEditMode) -> None:
        """Set edit mode."""
        self._edit_mode = mode

    @property
    def preview(self) -> SkeletonPreview:
        """Get preview settings."""
        return self._preview

    def load_skeleton(self, bone_data: List[Dict[str, Any]]) -> None:
        """Load skeleton from bone data.

        Args:
            bone_data: List of bone dictionaries with name, parent_index, transform
        """
        self._bone_names.clear()
        self._bone_parents.clear()
        self._bone_transforms.clear()

        for bone in bone_data:
            self._bone_names.append(bone["name"])
            self._bone_parents.append(bone.get("parent_index", -1))
            transform = bone.get("transform", Transform.identity())
            self._bone_transforms.append(transform)

        self._notify_change()

    def get_bone_index(self, name: str) -> int:
        """Get bone index by name."""
        try:
            return self._bone_names.index(name)
        except ValueError:
            return -1

    def get_bone_name(self, index: int) -> Optional[str]:
        """Get bone name by index."""
        if 0 <= index < len(self._bone_names):
            return self._bone_names[index]
        return None

    def get_bone_parent(self, index: int) -> int:
        """Get parent bone index."""
        if 0 <= index < len(self._bone_parents):
            return self._bone_parents[index]
        return -1

    def get_bone_children(self, index: int) -> List[int]:
        """Get child bone indices."""
        return [i for i, p in enumerate(self._bone_parents) if p == index]

    def get_bone_transform(self, index: int) -> Optional[Transform]:
        """Get bone local transform."""
        if 0 <= index < len(self._bone_transforms):
            return self._bone_transforms[index]
        return None

    def set_bone_transform(self, index: int, transform: Transform) -> bool:
        """Set bone local transform."""
        if 0 <= index < len(self._bone_transforms):
            self._bone_transforms[index] = transform
            self._notify_change()
            return True
        return False

    def get_root_bones(self) -> List[int]:
        """Get root bone indices."""
        return [i for i, p in enumerate(self._bone_parents) if p == -1]

    def get_bone_depth(self, index: int) -> int:
        """Get depth of bone in hierarchy."""
        depth = 0
        current = index

        while current >= 0:
            parent = self._bone_parents[current]
            if parent < 0:
                break
            depth += 1
            current = parent

        return depth

    def get_bone_chain(self, start: int, end: int) -> List[int]:
        """Get chain of bones from start to end."""
        if start < 0 or start >= len(self._bone_names):
            return []
        if end < 0 or end >= len(self._bone_names):
            return []
        if start == end:
            return [start]

        # Find path from end up to start or common ancestor
        end_ancestors = []
        current = end
        while current >= 0:
            end_ancestors.append(current)
            if current == start:
                return list(reversed(end_ancestors))
            current = self._bone_parents[current]

        # Check if start is in end's ancestors
        start_ancestors = []
        current = start
        while current >= 0:
            start_ancestors.append(current)
            current = self._bone_parents[current]

        # Find common ancestor
        for ancestor in start_ancestors:
            if ancestor in end_ancestors:
                idx = end_ancestors.index(ancestor)
                path = list(reversed(start_ancestors[:start_ancestors.index(ancestor) + 1]))
                path.extend(end_ancestors[:idx])
                return path

        return []

    # Socket operations

    def add_socket(self, socket: Socket) -> bool:
        """Add a socket."""
        if socket.name in self._sockets:
            return False
        if self.get_bone_index(socket.bone_name) < 0:
            return False

        self._sockets[socket.name] = socket
        self._notify_change()
        return True

    def remove_socket(self, name: str) -> bool:
        """Remove a socket."""
        if name in self._sockets:
            del self._sockets[name]
            self._selection.selected_sockets.discard(name)
            self._notify_change()
            return True
        return False

    def get_socket(self, name: str) -> Optional[Socket]:
        """Get socket by name."""
        return self._sockets.get(name)

    def get_sockets(self) -> List[Socket]:
        """Get all sockets."""
        return list(self._sockets.values())

    def get_sockets_on_bone(self, bone_name: str) -> List[Socket]:
        """Get all sockets attached to a bone."""
        return [s for s in self._sockets.values() if s.bone_name == bone_name]

    def rename_socket(self, old_name: str, new_name: str) -> bool:
        """Rename a socket."""
        if old_name not in self._sockets:
            return False
        if new_name in self._sockets:
            return False

        socket = self._sockets.pop(old_name)
        socket.name = new_name
        self._sockets[new_name] = socket
        self._notify_change()
        return True

    # Virtual bone operations

    def add_virtual_bone(self, vbone: VirtualBone) -> bool:
        """Add a virtual bone."""
        if vbone.name in self._virtual_bones:
            return False
        if self.get_bone_index(vbone.source_bone) < 0:
            return False

        self._virtual_bones[vbone.name] = vbone
        self._notify_change()
        return True

    def remove_virtual_bone(self, name: str) -> bool:
        """Remove a virtual bone."""
        if name in self._virtual_bones:
            del self._virtual_bones[name]
            self._selection.selected_virtual_bones.discard(name)
            self._notify_change()
            return True
        return False

    def get_virtual_bone(self, name: str) -> Optional[VirtualBone]:
        """Get virtual bone by name."""
        return self._virtual_bones.get(name)

    def get_virtual_bones(self) -> List[VirtualBone]:
        """Get all virtual bones."""
        return list(self._virtual_bones.values())

    # Mirror operations

    def add_mirror_pair(self, pair: BoneMirrorPair) -> bool:
        """Add a bone mirror pair."""
        # Check bones exist
        if self.get_bone_index(pair.left_bone) < 0:
            return False
        if self.get_bone_index(pair.right_bone) < 0:
            return False

        # Check not already in a pair
        for existing in self._mirror_pairs:
            if pair.left_bone in (existing.left_bone, existing.right_bone):
                return False
            if pair.right_bone in (existing.left_bone, existing.right_bone):
                return False

        self._mirror_pairs.append(pair)
        self._notify_change()
        return True

    def remove_mirror_pair(self, left_or_right: str) -> bool:
        """Remove mirror pair containing bone."""
        for i, pair in enumerate(self._mirror_pairs):
            if pair.left_bone == left_or_right or pair.right_bone == left_or_right:
                self._mirror_pairs.pop(i)
                self._notify_change()
                return True
        return False

    def get_mirror_bone(self, bone_name: str) -> Optional[str]:
        """Get the mirror of a bone."""
        for pair in self._mirror_pairs:
            mirror = pair.get_mirror(bone_name)
            if mirror:
                return mirror
        return None

    def auto_detect_mirror_pairs(
        self,
        left_prefix: str = "_l",
        right_prefix: str = "_r",
    ) -> int:
        """Auto-detect mirror pairs based on naming convention."""
        detected = 0

        for name in self._bone_names:
            if left_prefix in name:
                mirror_name = name.replace(left_prefix, right_prefix)
                if mirror_name in self._bone_names:
                    pair = BoneMirrorPair(left_bone=name, right_bone=mirror_name)
                    if self.add_mirror_pair(pair):
                        detected += 1

        return detected

    # Retargeting

    def add_retarget_source(self, source: RetargetSource) -> bool:
        """Add a retarget source skeleton."""
        if source.name in self._retarget_sources:
            return False

        self._retarget_sources[source.name] = source
        self._notify_change()
        return True

    def remove_retarget_source(self, name: str) -> bool:
        """Remove a retarget source."""
        if name in self._retarget_sources:
            del self._retarget_sources[name]
            self._notify_change()
            return True
        return False

    def get_retarget_sources(self) -> List[RetargetSource]:
        """Get all retarget sources."""
        return list(self._retarget_sources.values())

    def add_retarget_mapping(self, mapping: RetargetMapping) -> None:
        """Add a retarget mapping."""
        # Remove existing mapping for target bone
        self._retarget_mappings = [
            m for m in self._retarget_mappings
            if m.target_bone != mapping.target_bone
        ]
        self._retarget_mappings.append(mapping)
        self._notify_change()

    def get_retarget_mappings(self) -> List[RetargetMapping]:
        """Get all retarget mappings."""
        return list(self._retarget_mappings)

    def auto_map_bones(self, source_name: str) -> int:
        """Auto-map bones by matching names."""
        source = self._retarget_sources.get(source_name)
        if source is None:
            return 0

        source_bones = set(source.reference_pose.keys())
        mapped = 0

        for target_bone in self._bone_names:
            if target_bone in source_bones:
                mapping = RetargetMapping(
                    source_bone=target_bone,
                    target_bone=target_bone,
                )
                self.add_retarget_mapping(mapping)
                mapped += 1

        return mapped

    # Selection operations

    def select_bone(self, index: int, add_to_selection: bool = False) -> None:
        """Select a bone."""
        self._selection.select_bone(index, add_to_selection)

    def select_bone_chain(self, start: int, end: int) -> None:
        """Select a chain of bones."""
        chain = self.get_bone_chain(start, end)
        self._selection.clear()
        for bone_idx in chain:
            self._selection.selected_bones.add(bone_idx)
        if chain:
            self._selection.primary_bone = chain[-1]

    def select_children(self, index: int) -> None:
        """Select bone and all descendants."""
        self._selection.clear()
        self._selection.select_bone(index)

        to_visit = self.get_bone_children(index)
        while to_visit:
            child = to_visit.pop(0)
            self._selection.selected_bones.add(child)
            to_visit.extend(self.get_bone_children(child))

    def clear_selection(self) -> None:
        """Clear selection."""
        self._selection.clear()

    # Callbacks

    def add_on_change(self, callback: Callable[[], None]) -> None:
        """Register change callback."""
        self._on_change_callbacks.append(callback)

    def remove_on_change(self, callback: Callable[[], None]) -> None:
        """Remove change callback."""
        if callback in self._on_change_callbacks:
            self._on_change_callbacks.remove(callback)

    def _notify_change(self) -> None:
        """Notify all change callbacks."""
        for callback in self._on_change_callbacks:
            callback()


__all__ = [
    "BoneEditMode",
    "VirtualBoneType",
    "BoneSelection",
    "Socket",
    "SocketAttachment",
    "VirtualBone",
    "RetargetSource",
    "BoneMirrorPair",
    "RetargetMapping",
    "SkeletonPreview",
    "SkeletonEditor",
]
