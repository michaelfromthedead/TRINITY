"""Pose library, pose blending, and additive poses.

Provides tools for creating, managing, and blending animation poses
including pose libraries and additive pose support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.core.math import Quat, Transform, Vec3


# =============================================================================
# ENUMS
# =============================================================================


class PoseType(Enum):
    """Types of poses."""

    FULL = auto()       # Full body pose
    PARTIAL = auto()    # Partial pose (bone mask)
    ADDITIVE = auto()   # Additive pose (delta from reference)


class PoseBlendMode(Enum):
    """Modes for blending poses."""

    OVERRIDE = auto()   # Replace existing pose
    BLEND = auto()      # Linear blend
    ADDITIVE = auto()   # Add to existing pose
    MULTIPLY = auto()   # Multiply with existing pose


class AdditiveType(Enum):
    """Types of additive poses."""

    LOCAL_SPACE = auto()    # Additive in local bone space
    MESH_SPACE = auto()     # Additive in mesh space


# =============================================================================
# ANIMATION POSE
# =============================================================================


class AnimPose:
    """A single animation pose.

    A pose contains transform data for a set of bones at a specific
    point in time.

    Attributes:
        name: Pose name
        pose_type: Type of pose
        bone_transforms: Dictionary of bone name to transform
    """

    def __init__(
        self,
        name: str,
        pose_type: PoseType = PoseType.FULL,
    ) -> None:
        if not name:
            raise ValueError("Pose name cannot be empty")

        self._name = name
        self._pose_type = pose_type
        self._bone_transforms: Dict[str, Transform] = {}
        self._bone_weights: Dict[str, float] = {}  # For partial poses

    @property
    def name(self) -> str:
        """Get pose name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set pose name."""
        if not value:
            raise ValueError("Pose name cannot be empty")
        self._name = value

    @property
    def pose_type(self) -> PoseType:
        """Get pose type."""
        return self._pose_type

    @property
    def bone_names(self) -> List[str]:
        """Get all bone names in pose."""
        return list(self._bone_transforms.keys())

    @property
    def bone_count(self) -> int:
        """Get number of bones in pose."""
        return len(self._bone_transforms)

    def set_bone_transform(
        self,
        bone_name: str,
        transform: Transform,
        weight: float = 1.0,
    ) -> None:
        """Set transform for a bone."""
        self._bone_transforms[bone_name] = transform
        self._bone_weights[bone_name] = weight

    def get_bone_transform(self, bone_name: str) -> Optional[Transform]:
        """Get transform for a bone."""
        return self._bone_transforms.get(bone_name)

    def get_bone_weight(self, bone_name: str) -> float:
        """Get weight for a bone."""
        return self._bone_weights.get(bone_name, 1.0)

    def remove_bone(self, bone_name: str) -> bool:
        """Remove a bone from the pose."""
        if bone_name in self._bone_transforms:
            del self._bone_transforms[bone_name]
            self._bone_weights.pop(bone_name, None)
            return True
        return False

    def has_bone(self, bone_name: str) -> bool:
        """Check if pose has data for a bone."""
        return bone_name in self._bone_transforms

    def copy(self) -> AnimPose:
        """Create a copy of this pose."""
        pose = AnimPose(self._name, self._pose_type)
        for bone_name, transform in self._bone_transforms.items():
            pose._bone_transforms[bone_name] = Transform(
                translation=Vec3(
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                ),
                rotation=Quat(
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ),
                scale=Vec3(
                    transform.scale.x,
                    transform.scale.y,
                    transform.scale.z,
                ),
            )
            pose._bone_weights[bone_name] = self._bone_weights.get(bone_name, 1.0)
        return pose

    def blend_with(
        self,
        other: AnimPose,
        alpha: float,
        mode: PoseBlendMode = PoseBlendMode.BLEND,
    ) -> AnimPose:
        """Blend this pose with another.

        Args:
            other: Other pose to blend with
            alpha: Blend factor (0 = this, 1 = other)
            mode: Blend mode

        Returns:
            New blended pose
        """
        result = AnimPose(f"{self._name}_blended", self._pose_type)
        all_bones = set(self._bone_transforms.keys()) | set(other._bone_transforms.keys())

        for bone_name in all_bones:
            self_transform = self._bone_transforms.get(bone_name)
            other_transform = other._bone_transforms.get(bone_name)

            if self_transform is None and other_transform is None:
                continue

            if self_transform is None:
                result.set_bone_transform(bone_name, other_transform)
                continue

            if other_transform is None:
                result.set_bone_transform(bone_name, self_transform)
                continue

            # Blend transforms
            if mode == PoseBlendMode.OVERRIDE:
                result.set_bone_transform(bone_name, other_transform if alpha > 0.5 else self_transform)

            elif mode == PoseBlendMode.BLEND:
                pos = self_transform.translation.lerp(other_transform.translation, alpha)
                rot = self_transform.rotation.slerp(other_transform.rotation, alpha)
                scale = self_transform.scale.lerp(other_transform.scale, alpha)
                result.set_bone_transform(bone_name, Transform(pos, rot, scale))

            elif mode == PoseBlendMode.ADDITIVE:
                # Add other as delta
                pos = self_transform.translation + (other_transform.translation * alpha)
                rot = self_transform.rotation * Quat.identity().slerp(other_transform.rotation, alpha)
                scale = Vec3(
                    self_transform.scale.x * (1 + (other_transform.scale.x - 1) * alpha),
                    self_transform.scale.y * (1 + (other_transform.scale.y - 1) * alpha),
                    self_transform.scale.z * (1 + (other_transform.scale.z - 1) * alpha),
                )
                result.set_bone_transform(bone_name, Transform(pos, rot, scale))

        return result


# =============================================================================
# ADDITIVE POSE
# =============================================================================


class AdditivePose:
    """An additive pose that modifies a base pose.

    Additive poses store the difference from a reference pose and
    can be applied to modify other poses.
    """

    def __init__(
        self,
        name: str,
        additive_type: AdditiveType = AdditiveType.LOCAL_SPACE,
    ) -> None:
        self._name = name
        self._additive_type = additive_type
        self._reference_pose: Optional[AnimPose] = None
        self._delta_pose: Optional[AnimPose] = None

    @property
    def name(self) -> str:
        """Get additive pose name."""
        return self._name

    @property
    def additive_type(self) -> AdditiveType:
        """Get additive type."""
        return self._additive_type

    @property
    def reference_pose(self) -> Optional[AnimPose]:
        """Get reference pose."""
        return self._reference_pose

    @property
    def delta_pose(self) -> Optional[AnimPose]:
        """Get delta pose."""
        return self._delta_pose

    def compute_from_poses(
        self,
        reference: AnimPose,
        target: AnimPose,
    ) -> None:
        """Compute additive from reference and target poses.

        Args:
            reference: Reference (base) pose
            target: Target pose to compute delta from
        """
        self._reference_pose = reference.copy()
        self._delta_pose = AnimPose(f"{self._name}_delta", PoseType.ADDITIVE)

        for bone_name in target.bone_names:
            ref_transform = reference.get_bone_transform(bone_name)
            tgt_transform = target.get_bone_transform(bone_name)

            if ref_transform is None or tgt_transform is None:
                continue

            # Compute delta
            if self._additive_type == AdditiveType.LOCAL_SPACE:
                delta_pos = tgt_transform.translation - ref_transform.translation
                delta_rot = ref_transform.rotation.inverse() * tgt_transform.rotation
                delta_scale = Vec3(
                    tgt_transform.scale.x / ref_transform.scale.x if ref_transform.scale.x != 0 else 1,
                    tgt_transform.scale.y / ref_transform.scale.y if ref_transform.scale.y != 0 else 1,
                    tgt_transform.scale.z / ref_transform.scale.z if ref_transform.scale.z != 0 else 1,
                )
            else:
                # Mesh space - just store the difference
                delta_pos = tgt_transform.translation - ref_transform.translation
                delta_rot = ref_transform.rotation.inverse() * tgt_transform.rotation
                delta_scale = Vec3(1, 1, 1)

            self._delta_pose.set_bone_transform(
                bone_name,
                Transform(delta_pos, delta_rot, delta_scale),
            )

    def apply_to(self, base_pose: AnimPose, alpha: float = 1.0) -> AnimPose:
        """Apply additive to a base pose.

        Args:
            base_pose: Pose to apply additive to
            alpha: Blend factor for additive

        Returns:
            New pose with additive applied
        """
        if self._delta_pose is None:
            return base_pose.copy()

        result = base_pose.copy()

        for bone_name in self._delta_pose.bone_names:
            base_transform = base_pose.get_bone_transform(bone_name)
            delta_transform = self._delta_pose.get_bone_transform(bone_name)

            if base_transform is None or delta_transform is None:
                continue

            # Apply delta
            new_pos = base_transform.translation + (delta_transform.translation * alpha)
            new_rot = base_transform.rotation * Quat.identity().slerp(delta_transform.rotation, alpha)
            new_scale = Vec3(
                base_transform.scale.x * (1 + (delta_transform.scale.x - 1) * alpha),
                base_transform.scale.y * (1 + (delta_transform.scale.y - 1) * alpha),
                base_transform.scale.z * (1 + (delta_transform.scale.z - 1) * alpha),
            )

            result.set_bone_transform(bone_name, Transform(new_pos, new_rot, new_scale))

        return result


# =============================================================================
# POSE CATEGORY
# =============================================================================


@dataclass
class PoseCategory:
    """A category for organizing poses.

    Attributes:
        name: Category name
        color: Display color
        poses: List of pose names in this category
    """

    name: str
    color: Tuple[int, int, int] = (128, 128, 128)
    poses: List[str] = field(default_factory=list)

    def add_pose(self, pose_name: str) -> bool:
        """Add a pose to this category."""
        if pose_name in self.poses:
            return False
        self.poses.append(pose_name)
        return True

    def remove_pose(self, pose_name: str) -> bool:
        """Remove a pose from this category."""
        if pose_name in self.poses:
            self.poses.remove(pose_name)
            return True
        return False


# =============================================================================
# POSE ASSET
# =============================================================================


@dataclass
class PoseAsset:
    """A pose asset with metadata.

    Attributes:
        pose: The animation pose
        source_animation: Source animation path
        source_frame: Frame number from source
        thumbnail: Optional thumbnail path
        tags: Tags for searching
    """

    pose: AnimPose
    source_animation: Optional[str] = None
    source_frame: int = 0
    thumbnail: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Get pose name."""
        return self.pose.name


# =============================================================================
# POSE LIBRARY
# =============================================================================


class PoseLibrary:
    """Library for storing and managing poses.

    The pose library provides organization, search, and management
    functionality for animation poses.
    """

    def __init__(self, name: str = "Pose Library") -> None:
        self._name = name
        self._poses: Dict[str, PoseAsset] = {}
        self._categories: Dict[str, PoseCategory] = {}
        self._default_category = "Uncategorized"

        # Add default category
        self._categories[self._default_category] = PoseCategory(
            name=self._default_category,
            color=(100, 100, 100),
        )

    @property
    def name(self) -> str:
        """Get library name."""
        return self._name

    @property
    def pose_count(self) -> int:
        """Get number of poses."""
        return len(self._poses)

    @property
    def category_count(self) -> int:
        """Get number of categories."""
        return len(self._categories)

    def add_pose(
        self,
        pose: AnimPose,
        category: Optional[str] = None,
        source_animation: Optional[str] = None,
        source_frame: int = 0,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Add a pose to the library."""
        if pose.name in self._poses:
            return False

        asset = PoseAsset(
            pose=pose,
            source_animation=source_animation,
            source_frame=source_frame,
            tags=tags or [],
        )
        self._poses[pose.name] = asset

        # Add to category
        cat_name = category or self._default_category
        if cat_name not in self._categories:
            self._categories[cat_name] = PoseCategory(name=cat_name)
        self._categories[cat_name].add_pose(pose.name)

        return True

    def remove_pose(self, name: str) -> bool:
        """Remove a pose from the library."""
        if name not in self._poses:
            return False

        del self._poses[name]

        # Remove from categories
        for category in self._categories.values():
            category.remove_pose(name)

        return True

    def get_pose(self, name: str) -> Optional[AnimPose]:
        """Get a pose by name."""
        asset = self._poses.get(name)
        return asset.pose if asset else None

    def get_pose_asset(self, name: str) -> Optional[PoseAsset]:
        """Get a pose asset by name."""
        return self._poses.get(name)

    def get_all_poses(self) -> List[AnimPose]:
        """Get all poses."""
        return [asset.pose for asset in self._poses.values()]

    def get_poses_in_category(self, category: str) -> List[AnimPose]:
        """Get all poses in a category."""
        cat = self._categories.get(category)
        if cat is None:
            return []
        return [self._poses[name].pose for name in cat.poses if name in self._poses]

    def add_category(self, name: str, color: Tuple[int, int, int] = (128, 128, 128)) -> bool:
        """Add a category."""
        if name in self._categories:
            return False
        self._categories[name] = PoseCategory(name=name, color=color)
        return True

    def remove_category(self, name: str, move_to: Optional[str] = None) -> bool:
        """Remove a category, optionally moving poses to another."""
        if name not in self._categories or name == self._default_category:
            return False

        cat = self._categories[name]

        # Move poses to another category
        target = move_to or self._default_category
        if target in self._categories:
            for pose_name in cat.poses:
                self._categories[target].add_pose(pose_name)

        del self._categories[name]
        return True

    def get_categories(self) -> List[str]:
        """Get all category names."""
        return list(self._categories.keys())

    def move_pose_to_category(self, pose_name: str, category: str) -> bool:
        """Move a pose to a category."""
        if pose_name not in self._poses:
            return False
        if category not in self._categories:
            return False

        # Remove from current category
        for cat in self._categories.values():
            cat.remove_pose(pose_name)

        # Add to new category
        self._categories[category].add_pose(pose_name)
        return True

    def search_poses(self, query: str) -> List[AnimPose]:
        """Search poses by name or tags."""
        query_lower = query.lower()
        results = []

        for asset in self._poses.values():
            if query_lower in asset.pose.name.lower():
                results.append(asset.pose)
            elif any(query_lower in tag.lower() for tag in asset.tags):
                results.append(asset.pose)

        return results

    def add_tag(self, pose_name: str, tag: str) -> bool:
        """Add a tag to a pose."""
        asset = self._poses.get(pose_name)
        if asset is None:
            return False
        if tag not in asset.tags:
            asset.tags.append(tag)
        return True

    def remove_tag(self, pose_name: str, tag: str) -> bool:
        """Remove a tag from a pose."""
        asset = self._poses.get(pose_name)
        if asset is None:
            return False
        if tag in asset.tags:
            asset.tags.remove(tag)
            return True
        return False


# =============================================================================
# POSE MIRROR SETTINGS
# =============================================================================


@dataclass
class PoseMirrorSettings:
    """Settings for pose mirroring.

    Attributes:
        mirror_axis: Axis to mirror across (X, Y, Z)
        bone_pairs: List of (left_bone, right_bone) pairs
        flip_rotation: Whether to flip rotation
    """

    mirror_axis: str = "X"
    bone_pairs: List[Tuple[str, str]] = field(default_factory=list)
    flip_rotation: bool = True

    def get_mirror_bone(self, bone_name: str) -> Optional[str]:
        """Get the mirror bone for a given bone."""
        for left, right in self.bone_pairs:
            if bone_name == left:
                return right
            if bone_name == right:
                return left
        return None

    def add_bone_pair(self, left: str, right: str) -> None:
        """Add a bone pair."""
        self.bone_pairs.append((left, right))


# =============================================================================
# POSE PREVIEW
# =============================================================================


class PosePreview:
    """Preview settings for pose visualization."""

    def __init__(self) -> None:
        self.show_skeleton = True
        self.show_mesh = True
        self.show_bone_names = False
        self.highlight_modified_bones = True
        self.comparison_pose: Optional[AnimPose] = None
        self.show_comparison = False
        self.comparison_opacity = 0.5


# =============================================================================
# POSE EDITOR
# =============================================================================


class PoseEditor:
    """Editor for animation poses.

    Provides functionality for creating, editing, and managing poses
    including blending and mirroring.
    """

    def __init__(self) -> None:
        self._current_pose: Optional[AnimPose] = None
        self._library: Optional[PoseLibrary] = None
        self._preview = PosePreview()
        self._mirror_settings = PoseMirrorSettings()
        self._selected_bones: List[str] = []
        self._on_change_callbacks: List[Callable[[], None]] = []

    @property
    def current_pose(self) -> Optional[AnimPose]:
        """Get current pose being edited."""
        return self._current_pose

    @property
    def library(self) -> Optional[PoseLibrary]:
        """Get associated pose library."""
        return self._library

    @property
    def preview(self) -> PosePreview:
        """Get preview settings."""
        return self._preview

    @property
    def mirror_settings(self) -> PoseMirrorSettings:
        """Get mirror settings."""
        return self._mirror_settings

    @property
    def selected_bones(self) -> List[str]:
        """Get selected bones."""
        return list(self._selected_bones)

    def set_library(self, library: PoseLibrary) -> None:
        """Set the pose library."""
        self._library = library

    def create_pose(self, name: str, pose_type: PoseType = PoseType.FULL) -> AnimPose:
        """Create a new pose."""
        self._current_pose = AnimPose(name, pose_type)
        self._notify_change()
        return self._current_pose

    def load_pose(self, pose: AnimPose) -> None:
        """Load a pose for editing."""
        self._current_pose = pose.copy()
        self._notify_change()

    def save_to_library(self, category: Optional[str] = None) -> bool:
        """Save current pose to library."""
        if self._current_pose is None or self._library is None:
            return False

        return self._library.add_pose(self._current_pose.copy(), category)

    def set_bone_transform(
        self,
        bone_name: str,
        transform: Transform,
    ) -> None:
        """Set transform for a bone."""
        if self._current_pose is None:
            return
        self._current_pose.set_bone_transform(bone_name, transform)
        self._notify_change()

    def set_bone_position(self, bone_name: str, position: Vec3) -> None:
        """Set position for a bone."""
        if self._current_pose is None:
            return

        current = self._current_pose.get_bone_transform(bone_name)
        if current is None:
            current = Transform.identity()

        self._current_pose.set_bone_transform(
            bone_name,
            Transform(position, current.rotation, current.scale),
        )
        self._notify_change()

    def set_bone_rotation(self, bone_name: str, rotation: Quat) -> None:
        """Set rotation for a bone."""
        if self._current_pose is None:
            return

        current = self._current_pose.get_bone_transform(bone_name)
        if current is None:
            current = Transform.identity()

        self._current_pose.set_bone_transform(
            bone_name,
            Transform(current.translation, rotation, current.scale),
        )
        self._notify_change()

    def set_bone_scale(self, bone_name: str, scale: Vec3) -> None:
        """Set scale for a bone."""
        if self._current_pose is None:
            return

        current = self._current_pose.get_bone_transform(bone_name)
        if current is None:
            current = Transform.identity()

        self._current_pose.set_bone_transform(
            bone_name,
            Transform(current.translation, current.rotation, scale),
        )
        self._notify_change()

    def select_bone(self, bone_name: str, add_to_selection: bool = False) -> None:
        """Select a bone."""
        if not add_to_selection:
            self._selected_bones.clear()
        if bone_name not in self._selected_bones:
            self._selected_bones.append(bone_name)

    def deselect_bone(self, bone_name: str) -> None:
        """Deselect a bone."""
        if bone_name in self._selected_bones:
            self._selected_bones.remove(bone_name)

    def clear_selection(self) -> None:
        """Clear bone selection."""
        self._selected_bones.clear()

    def copy_pose_from(self, source: AnimPose, bones: Optional[List[str]] = None) -> None:
        """Copy bone transforms from another pose."""
        if self._current_pose is None:
            return

        bones_to_copy = bones or source.bone_names

        for bone_name in bones_to_copy:
            transform = source.get_bone_transform(bone_name)
            if transform:
                self._current_pose.set_bone_transform(bone_name, transform)

        self._notify_change()

    def blend_pose(
        self,
        other: AnimPose,
        alpha: float,
        bones: Optional[List[str]] = None,
    ) -> None:
        """Blend current pose with another."""
        if self._current_pose is None:
            return

        bones_to_blend = bones or (
            set(self._current_pose.bone_names) | set(other.bone_names)
        )

        for bone_name in bones_to_blend:
            self_transform = self._current_pose.get_bone_transform(bone_name)
            other_transform = other.get_bone_transform(bone_name)

            if self_transform is None or other_transform is None:
                continue

            pos = self_transform.translation.lerp(other_transform.translation, alpha)
            rot = self_transform.rotation.slerp(other_transform.rotation, alpha)
            scale = self_transform.scale.lerp(other_transform.scale, alpha)

            self._current_pose.set_bone_transform(bone_name, Transform(pos, rot, scale))

        self._notify_change()

    def mirror_pose(self, bones: Optional[List[str]] = None) -> None:
        """Mirror pose using mirror settings."""
        if self._current_pose is None:
            return

        bones_to_mirror = bones or self._current_pose.bone_names
        axis = self._mirror_settings.mirror_axis

        for bone_name in bones_to_mirror:
            mirror_bone = self._mirror_settings.get_mirror_bone(bone_name)
            if mirror_bone is None:
                continue

            transform = self._current_pose.get_bone_transform(bone_name)
            if transform is None:
                continue

            # Mirror position
            pos = transform.translation
            if axis == "X":
                mirrored_pos = Vec3(-pos.x, pos.y, pos.z)
            elif axis == "Y":
                mirrored_pos = Vec3(pos.x, -pos.y, pos.z)
            else:
                mirrored_pos = Vec3(pos.x, pos.y, -pos.z)

            # Mirror rotation
            rot = transform.rotation
            if self._mirror_settings.flip_rotation:
                if axis == "X":
                    mirrored_rot = Quat(rot.x, -rot.y, -rot.z, rot.w)
                elif axis == "Y":
                    mirrored_rot = Quat(-rot.x, rot.y, -rot.z, rot.w)
                else:
                    mirrored_rot = Quat(-rot.x, -rot.y, rot.z, rot.w)
            else:
                mirrored_rot = rot

            self._current_pose.set_bone_transform(
                mirror_bone,
                Transform(mirrored_pos, mirrored_rot, transform.scale),
            )

        self._notify_change()

    def reset_bone(self, bone_name: str) -> None:
        """Reset bone to identity transform."""
        if self._current_pose is None:
            return
        self._current_pose.set_bone_transform(bone_name, Transform.identity())
        self._notify_change()

    def reset_all_bones(self) -> None:
        """Reset all bones to identity."""
        if self._current_pose is None:
            return
        for bone_name in self._current_pose.bone_names:
            self._current_pose.set_bone_transform(bone_name, Transform.identity())
        self._notify_change()

    def add_on_change(self, callback: Callable[[], None]) -> None:
        """Register change callback."""
        self._on_change_callbacks.append(callback)

    def remove_on_change(self, callback: Callable[[], None]) -> None:
        """Remove change callback."""
        if callback in self._on_change_callbacks:
            self._on_change_callbacks.remove(callback)

    def _notify_change(self) -> None:
        """Notify change callbacks."""
        for callback in self._on_change_callbacks:
            callback()


__all__ = [
    "PoseType",
    "PoseBlendMode",
    "AdditiveType",
    "AnimPose",
    "AdditivePose",
    "PoseCategory",
    "PoseAsset",
    "PoseLibrary",
    "PoseMirrorSettings",
    "PosePreview",
    "PoseEditor",
]
