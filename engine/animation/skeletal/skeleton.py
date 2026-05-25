"""Bone hierarchy and skeleton representation.

A Skeleton is a hierarchical collection of bones used for skeletal animation.
Each bone has a local bind pose, parent relationship, and inverse bind matrix
for skinning computations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from engine.core.math import Mat4, Quat, Transform, Vec3


# Animation data decorator (from ANIMATION_CONTEXT.md pattern)
def animation_data(cls):
    """Decorator for animation data classes. Registers with animation system."""
    cls._animation_data = True
    cls._animation_type = cls.__name__
    return cls


class AnimationMeta(type):
    """Metaclass for animation types with registration."""
    _registry: Dict[str, type] = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name not in ("Bone", "Skeleton"):  # Don't register base classes directly
            AnimationMeta._registry[name] = cls
        return cls


@animation_data
@dataclass
class Bone:
    """A single bone in a skeleton hierarchy.

    Attributes:
        index: Unique index of this bone in the skeleton.
        name: Human-readable bone name (e.g., "spine_01", "hand_l").
        parent_index: Index of parent bone, or -1 for root bones.
        local_bind_pose: Local transform in bind pose (relative to parent).
        inverse_bind_pose: Inverse of world bind pose matrix for skinning.
    """

    index: int
    name: str
    parent_index: int = -1
    local_bind_pose: Transform = field(default_factory=Transform.identity)
    inverse_bind_pose: Mat4 = field(default_factory=Mat4.identity)

    def __post_init__(self) -> None:
        """Validate bone data after initialization."""
        if self.index < 0:
            raise ValueError(f"Bone index must be >= 0, got {self.index}")
        if not self.name:
            raise ValueError("Bone name cannot be empty")
        if self.parent_index < -1:
            raise ValueError(f"Parent index must be >= -1, got {self.parent_index}")
        if self.parent_index == self.index:
            raise ValueError(f"Bone cannot be its own parent: index={self.index}")

    def is_root(self) -> bool:
        """Check if this bone is a root bone (no parent)."""
        return self.parent_index == -1

    def copy(self) -> Bone:
        """Create a deep copy of this bone."""
        return Bone(
            index=self.index,
            name=self.name,
            parent_index=self.parent_index,
            local_bind_pose=Transform(
                translation=Vec3(
                    self.local_bind_pose.translation.x,
                    self.local_bind_pose.translation.y,
                    self.local_bind_pose.translation.z,
                ),
                rotation=Quat(
                    self.local_bind_pose.rotation.x,
                    self.local_bind_pose.rotation.y,
                    self.local_bind_pose.rotation.z,
                    self.local_bind_pose.rotation.w,
                ),
                scale=Vec3(
                    self.local_bind_pose.scale.x,
                    self.local_bind_pose.scale.y,
                    self.local_bind_pose.scale.z,
                ),
            ),
            inverse_bind_pose=Mat4(list(self.inverse_bind_pose.m)),
        )

    def __repr__(self) -> str:
        parent_str = f"parent={self.parent_index}" if self.parent_index >= 0 else "root"
        return f"Bone({self.index}: '{self.name}', {parent_str})"


@animation_data
class Skeleton:
    """A complete skeleton with bone hierarchy.

    The skeleton maintains a list of bones with parent-child relationships,
    enabling efficient traversal and transform computation.

    Attributes:
        name: Name of the skeleton (e.g., "humanoid", "spider").
        bones: List of all bones in the skeleton.
        bone_name_to_index: Mapping from bone name to index for fast lookup.
        root_bone_indices: Indices of all root bones (bones with no parent).
    """

    def __init__(
        self,
        name: str = "skeleton",
        bones: Optional[List[Bone]] = None,
    ) -> None:
        """Initialize a skeleton.

        Args:
            name: Skeleton name.
            bones: List of bones. If None, creates an empty skeleton.
        """
        if not name:
            raise ValueError("Skeleton name cannot be empty")

        self.name = name
        self._bones: List[Bone] = []
        self._bone_name_to_index: Dict[str, int] = {}
        self._root_bone_indices: List[int] = []
        self._children_cache: Dict[int, List[int]] = {}

        if bones:
            for bone in bones:
                self.add_bone(bone)
            self._rebuild_caches()

    @property
    def bones(self) -> List[Bone]:
        """Get the list of bones (read-only view)."""
        return list(self._bones)

    @property
    def bone_name_to_index(self) -> Dict[str, int]:
        """Get the bone name to index mapping (read-only view)."""
        return dict(self._bone_name_to_index)

    @property
    def root_bone_indices(self) -> List[int]:
        """Get the list of root bone indices (read-only view)."""
        return list(self._root_bone_indices)

    @property
    def bone_count(self) -> int:
        """Get the number of bones in the skeleton."""
        return len(self._bones)

    def add_bone(self, bone: Bone) -> None:
        """Add a bone to the skeleton.

        Args:
            bone: The bone to add.

        Raises:
            ValueError: If bone index already exists or name is duplicate.
        """
        if bone.index != len(self._bones):
            raise ValueError(
                f"Bone index {bone.index} does not match expected index {len(self._bones)}"
            )
        if bone.name in self._bone_name_to_index:
            raise ValueError(f"Bone name '{bone.name}' already exists in skeleton")

        self._bones.append(bone)
        self._bone_name_to_index[bone.name] = bone.index

        if bone.is_root():
            self._root_bone_indices.append(bone.index)

    def _rebuild_caches(self) -> None:
        """Rebuild internal caches after bulk operations."""
        self._children_cache.clear()
        self._root_bone_indices.clear()

        for bone in self._bones:
            if bone.is_root():
                self._root_bone_indices.append(bone.index)

            if bone.parent_index >= 0:
                if bone.parent_index not in self._children_cache:
                    self._children_cache[bone.parent_index] = []
                self._children_cache[bone.parent_index].append(bone.index)

    def get_bone(self, index: int) -> Bone:
        """Get a bone by index.

        Args:
            index: Bone index.

        Returns:
            The bone at the given index.

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self._bones):
            raise IndexError(f"Bone index {index} out of range [0, {len(self._bones)})")
        return self._bones[index]

    def get_bone_by_name(self, name: str) -> Optional[Bone]:
        """Get a bone by name.

        Args:
            name: Bone name.

        Returns:
            The bone with the given name, or None if not found.
        """
        index = self._bone_name_to_index.get(name)
        if index is None:
            return None
        return self._bones[index]

    def get_bone_index(self, name: str) -> int:
        """Get bone index by name.

        Args:
            name: Bone name.

        Returns:
            Bone index, or -1 if not found.
        """
        return self._bone_name_to_index.get(name, -1)

    def get_bone_children(self, bone_index: int) -> List[int]:
        """Get indices of all direct children of a bone.

        Args:
            bone_index: Parent bone index.

        Returns:
            List of child bone indices.
        """
        if bone_index < 0 or bone_index >= len(self._bones):
            return []

        if not self._children_cache:
            self._rebuild_caches()

        return list(self._children_cache.get(bone_index, []))

    def get_bone_descendants(self, bone_index: int) -> List[int]:
        """Get indices of all descendants of a bone (children, grandchildren, etc).

        Args:
            bone_index: Ancestor bone index.

        Returns:
            List of all descendant bone indices in breadth-first order.
        """
        if bone_index < 0 or bone_index >= len(self._bones):
            return []

        descendants = []
        queue = self.get_bone_children(bone_index)

        while queue:
            child = queue.pop(0)
            descendants.append(child)
            queue.extend(self.get_bone_children(child))

        return descendants

    def get_bone_chain(self, start_index: int, end_index: int) -> List[int]:
        """Get the chain of bones from start to end (inclusive).

        The chain goes from start up through ancestors until reaching end,
        or from end down through descendants until reaching start.

        Args:
            start_index: Starting bone index.
            end_index: Ending bone index.

        Returns:
            List of bone indices forming the chain, or empty if no path exists.
        """
        if start_index < 0 or start_index >= len(self._bones):
            return []
        if end_index < 0 or end_index >= len(self._bones):
            return []
        if start_index == end_index:
            return [start_index]

        # Try path from start to end (going up through parents)
        path_up = self._find_path_to_ancestor(start_index, end_index)
        if path_up:
            return path_up

        # Try path from end to start (going up through parents), then reverse
        path_down = self._find_path_to_ancestor(end_index, start_index)
        if path_down:
            return list(reversed(path_down))

        # Try to find common ancestor and build path through it
        return self._find_path_through_common_ancestor(start_index, end_index)

    def _find_path_to_ancestor(self, start: int, end: int) -> List[int]:
        """Find path from start bone to end bone going through parents."""
        path = [start]
        current = start

        while current != end:
            parent = self._bones[current].parent_index
            if parent < 0:
                return []  # Reached root without finding end
            path.append(parent)
            current = parent

        return path

    def _find_path_through_common_ancestor(
        self, start: int, end: int
    ) -> List[int]:
        """Find path between two bones through their common ancestor."""
        # Build ancestor chains
        start_ancestors = self._get_ancestor_chain(start)
        end_ancestors = self._get_ancestor_chain(end)

        # Find lowest common ancestor
        start_set = set(start_ancestors)
        common = None
        for ancestor in end_ancestors:
            if ancestor in start_set:
                common = ancestor
                break

        if common is None:
            return []  # No common ancestor (different trees)

        # Build path: start -> common -> end
        path_to_common = []
        current = start
        while current != common:
            path_to_common.append(current)
            current = self._bones[current].parent_index

        path_from_common = []
        current = end
        while current != common:
            path_from_common.append(current)
            current = self._bones[current].parent_index

        return path_to_common + [common] + list(reversed(path_from_common))

    def _get_ancestor_chain(self, bone_index: int) -> List[int]:
        """Get chain of ancestors from bone to root."""
        chain = [bone_index]
        current = bone_index

        while True:
            parent = self._bones[current].parent_index
            if parent < 0:
                break
            chain.append(parent)
            current = parent

        return chain

    def get_bone_path(self, bone_index: int) -> str:
        """Get the full path from root to bone as a string.

        Args:
            bone_index: Target bone index.

        Returns:
            Path string like "root/spine/arm/hand".
        """
        if bone_index < 0 or bone_index >= len(self._bones):
            return ""

        path_parts = []
        current = bone_index

        while current >= 0:
            path_parts.append(self._bones[current].name)
            current = self._bones[current].parent_index

        return "/".join(reversed(path_parts))

    def find_bone_by_path(self, path: str) -> Optional[Bone]:
        """Find a bone by its path string.

        Args:
            path: Path like "root/spine/arm/hand".

        Returns:
            The bone at the path, or None if not found.
        """
        if not path:
            return None

        parts = path.split("/")
        return self.get_bone_by_name(parts[-1])

    def compute_world_transforms(
        self,
        local_transforms: Optional[List[Transform]] = None,
    ) -> List[Mat4]:
        """Compute world-space transform matrices for all bones.

        Args:
            local_transforms: Optional list of local transforms to use instead
                of bind poses. Must have same length as bone count.

        Returns:
            List of world-space 4x4 matrices, one per bone.
        """
        if local_transforms is None:
            local_transforms = [bone.local_bind_pose for bone in self._bones]

        if len(local_transforms) != len(self._bones):
            raise ValueError(
                f"Local transforms count ({len(local_transforms)}) must match "
                f"bone count ({len(self._bones)})"
            )

        world_matrices: List[Mat4] = [Mat4.identity() for _ in self._bones]

        # Process bones in order (parents before children due to index ordering)
        for bone in self._bones:
            local_matrix = local_transforms[bone.index].to_matrix()

            if bone.is_root():
                world_matrices[bone.index] = local_matrix
            else:
                parent_world = world_matrices[bone.parent_index]
                world_matrices[bone.index] = parent_world @ local_matrix

        return world_matrices

    def compute_skinning_matrices(
        self,
        world_transforms: List[Mat4],
    ) -> List[Mat4]:
        """Compute skinning matrices for vertex transformation.

        The skinning matrix transforms vertices from bind pose to current pose.

        Args:
            world_transforms: World-space transforms for current pose.

        Returns:
            List of skinning matrices (world * inverse_bind).
        """
        if len(world_transforms) != len(self._bones):
            raise ValueError(
                f"World transforms count ({len(world_transforms)}) must match "
                f"bone count ({len(self._bones)})"
            )

        return [
            world_transforms[i] @ self._bones[i].inverse_bind_pose
            for i in range(len(self._bones))
        ]

    def compute_inverse_bind_poses(self) -> None:
        """Compute and store inverse bind pose matrices for all bones.

        This should be called after all bones are added and before skinning.
        """
        world_matrices = self.compute_world_transforms()

        for i, bone in enumerate(self._bones):
            bone.inverse_bind_pose = world_matrices[i].inverse()

    def validate(self) -> List[str]:
        """Validate skeleton integrity.

        Returns:
            List of validation errors, empty if valid.
        """
        errors = []

        # Check indices are sequential
        for i, bone in enumerate(self._bones):
            if bone.index != i:
                errors.append(f"Bone at position {i} has index {bone.index}")

        # Check parent references are valid
        for bone in self._bones:
            if bone.parent_index >= 0:
                if bone.parent_index >= len(self._bones):
                    errors.append(
                        f"Bone '{bone.name}' has invalid parent index {bone.parent_index}"
                    )
                elif bone.parent_index >= bone.index:
                    errors.append(
                        f"Bone '{bone.name}' has parent index {bone.parent_index} "
                        f">= own index {bone.index}"
                    )

        # Check for duplicate names
        names = set()
        for bone in self._bones:
            if bone.name in names:
                errors.append(f"Duplicate bone name: '{bone.name}'")
            names.add(bone.name)

        # Check for at least one root
        if not self._root_bone_indices and self._bones:
            errors.append("Skeleton has no root bones")

        return errors

    def clone(self) -> Skeleton:
        """Create a deep copy of this skeleton.

        Returns:
            A new Skeleton with copied bone data.
        """
        cloned_bones = [bone.copy() for bone in self._bones]
        return Skeleton(name=self.name, bones=cloned_bones)

    def find_bones_by_pattern(
        self,
        pattern: Callable[[Bone], bool],
    ) -> List[Bone]:
        """Find all bones matching a predicate.

        Args:
            pattern: Function that returns True for matching bones.

        Returns:
            List of matching bones.
        """
        return [bone for bone in self._bones if pattern(bone)]

    def get_leaf_bones(self) -> List[int]:
        """Get indices of all leaf bones (bones with no children).

        Returns:
            List of leaf bone indices.
        """
        if not self._children_cache:
            self._rebuild_caches()

        all_parents = set(self._children_cache.keys())
        return [
            bone.index
            for bone in self._bones
            if bone.index not in all_parents
        ]

    def get_depth(self, bone_index: int) -> int:
        """Get the depth of a bone in the hierarchy (root = 0).

        Args:
            bone_index: Bone index.

        Returns:
            Depth of the bone, or -1 if invalid index.
        """
        if bone_index < 0 or bone_index >= len(self._bones):
            return -1

        depth = 0
        current = bone_index

        while self._bones[current].parent_index >= 0:
            depth += 1
            current = self._bones[current].parent_index

        return depth

    def get_max_depth(self) -> int:
        """Get the maximum depth of the skeleton hierarchy.

        Returns:
            Maximum depth (0 for single bone, higher for deeper hierarchies).
        """
        if not self._bones:
            return -1

        return max(self.get_depth(i) for i in range(len(self._bones)))

    def traverse_depth_first(
        self,
        callback: Callable[[Bone, int], None],
        start_index: Optional[int] = None,
    ) -> None:
        """Traverse bones in depth-first order.

        Args:
            callback: Function called for each bone with (bone, depth).
            start_index: Starting bone index, or None for all roots.
        """
        if not self._children_cache:
            self._rebuild_caches()

        def visit(index: int, depth: int) -> None:
            callback(self._bones[index], depth)
            for child in self._children_cache.get(index, []):
                visit(child, depth + 1)

        if start_index is not None:
            visit(start_index, 0)
        else:
            for root in self._root_bone_indices:
                visit(root, 0)

    def traverse_breadth_first(
        self,
        callback: Callable[[Bone, int], None],
        start_index: Optional[int] = None,
    ) -> None:
        """Traverse bones in breadth-first order.

        Args:
            callback: Function called for each bone with (bone, depth).
            start_index: Starting bone index, or None for all roots.
        """
        if not self._children_cache:
            self._rebuild_caches()

        queue: List[Tuple[int, int]] = []

        if start_index is not None:
            queue.append((start_index, 0))
        else:
            queue.extend((root, 0) for root in self._root_bone_indices)

        while queue:
            index, depth = queue.pop(0)
            callback(self._bones[index], depth)

            for child in self._children_cache.get(index, []):
                queue.append((child, depth + 1))

    def __repr__(self) -> str:
        return f"Skeleton('{self.name}', bones={len(self._bones)})"

    def __len__(self) -> int:
        return len(self._bones)

    def __iter__(self):
        return iter(self._bones)

    def __getitem__(self, index: int) -> Bone:
        return self.get_bone(index)


def create_humanoid_skeleton() -> Skeleton:
    """Create a standard humanoid skeleton for testing.

    Returns:
        A skeleton with common humanoid bones.
    """
    skeleton = Skeleton(name="humanoid")

    # Define bones: (name, parent_name)
    bone_defs = [
        ("root", None),
        ("pelvis", "root"),
        ("spine_01", "pelvis"),
        ("spine_02", "spine_01"),
        ("spine_03", "spine_02"),
        ("neck", "spine_03"),
        ("head", "neck"),
        ("clavicle_l", "spine_03"),
        ("upperarm_l", "clavicle_l"),
        ("lowerarm_l", "upperarm_l"),
        ("hand_l", "lowerarm_l"),
        ("clavicle_r", "spine_03"),
        ("upperarm_r", "clavicle_r"),
        ("lowerarm_r", "upperarm_r"),
        ("hand_r", "lowerarm_r"),
        ("thigh_l", "pelvis"),
        ("calf_l", "thigh_l"),
        ("foot_l", "calf_l"),
        ("thigh_r", "pelvis"),
        ("calf_r", "thigh_r"),
        ("foot_r", "calf_r"),
    ]

    for i, (name, parent_name) in enumerate(bone_defs):
        parent_index = -1 if parent_name is None else skeleton.get_bone_index(parent_name)
        bone = Bone(
            index=i,
            name=name,
            parent_index=parent_index,
            local_bind_pose=Transform.identity(),
        )
        skeleton.add_bone(bone)

    skeleton._rebuild_caches()
    skeleton.compute_inverse_bind_poses()

    return skeleton
