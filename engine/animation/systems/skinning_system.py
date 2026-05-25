"""ECS system for mesh skinning.

Computes skinning matrices and transforms vertices for rendering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence

from engine.core.math import Vec3, Vec4, Quat, Mat4, Transform
from engine.core.ecs import Entity, World
from engine.animation.config import SKINNING_CONFIG


class SkinningMethod(Enum):
    """Skinning computation method."""
    LINEAR = auto()  # Linear blend skinning (LBS)
    DUAL_QUATERNION = auto()  # Dual quaternion skinning
    GPU = auto()  # GPU-based skinning


@dataclass
class BoneInfluence:
    """Bone influence on a vertex.

    Attributes:
        bone_index: Index of influencing bone
        weight: Influence weight (0-1)
    """
    bone_index: int = 0
    weight: float = 0.0


@dataclass
class VertexSkinData:
    """Skinning data for a single vertex.

    Typically 4 bone influences per vertex for compatibility.
    """
    influences: list[BoneInfluence] = field(default_factory=list)

    @property
    def bone_count(self) -> int:
        return len(self.influences)

    def normalize(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(inf.weight for inf in self.influences)
        if total > 0:
            for inf in self.influences:
                inf.weight /= total


@dataclass
class SkinningData:
    """Complete skinning data for a mesh.

    Attributes:
        vertex_data: Per-vertex skinning information
        bind_poses: Bind pose (inverse of rest pose) per bone
        bone_names: Names of bones
        max_influences: Maximum influences per vertex
    """
    vertex_data: list[VertexSkinData] = field(default_factory=list)
    bind_poses: list[Mat4] = field(default_factory=list)
    bone_names: list[str] = field(default_factory=list)
    max_influences: int = SKINNING_CONFIG.DEFAULT_MAX_INFLUENCES

    @property
    def vertex_count(self) -> int:
        return len(self.vertex_data)

    @property
    def bone_count(self) -> int:
        return len(self.bind_poses)


@dataclass
class MeshData:
    """Simple mesh data representation."""
    positions: list[Vec3] = field(default_factory=list)
    normals: list[Vec3] = field(default_factory=list)
    tangents: list[Vec4] = field(default_factory=list)

    @property
    def vertex_count(self) -> int:
        return len(self.positions)


@dataclass
class SkinnedMeshComponent:
    """Component for skinned mesh entities.

    Attributes:
        mesh: Source mesh data
        skinning_data: Skinning weights and bind poses
        method: Skinning method to use
        enabled: Whether skinning is enabled
        skinned_positions: Output skinned positions
        skinned_normals: Output skinned normals
        skinning_matrices: Computed skinning matrices
    """
    mesh: MeshData | None = None
    skinning_data: SkinningData | None = None
    method: SkinningMethod = SkinningMethod.LINEAR
    enabled: bool = True

    # Output data
    skinned_positions: list[Vec3] = field(default_factory=list)
    skinned_normals: list[Vec3] = field(default_factory=list)
    skinning_matrices: list[Mat4] = field(default_factory=list)

    # For GPU skinning
    bone_matrices_buffer: list[float] = field(default_factory=list)  # Flattened 4x4 matrices

    def prepare_gpu_buffer(self) -> None:
        """Prepare flattened bone matrix buffer for GPU upload."""
        self.bone_matrices_buffer = []
        for mat in self.skinning_matrices:
            self.bone_matrices_buffer.extend(mat.m)

    def get_memory_size_bytes(self) -> int:
        """Calculate memory size of skinning data."""
        size = 0
        if self.mesh:
            size += self.mesh.vertex_count * (3 * 4 + 3 * 4)  # positions + normals
        if self.skinning_data:
            size += self.skinning_data.vertex_count * self.skinning_data.max_influences * (4 + 4)  # bone idx + weight
            size += self.skinning_data.bone_count * 16 * 4  # bind poses
        return size


class SkinningSystem:
    """ECS system for mesh skinning.

    Computes skinning matrices and optionally transforms vertices.
    """

    def __init__(self):
        self._use_dual_quaternion_threshold = SKINNING_CONFIG.DQ_BLEND_THRESHOLD  # Blend angle threshold

    def update(
        self,
        world: World,
        entity_components: list[tuple[Entity, SkinnedMeshComponent]],
        pose_data: dict[Entity, dict[int, Transform]]
    ) -> None:
        """Update skinning for all entities.

        Args:
            world: ECS world
            entity_components: List of (entity, component) tuples
            pose_data: Bone poses per entity
        """
        for entity, component in entity_components:
            if not component.enabled or not component.skinning_data:
                continue

            bone_transforms = pose_data.get(entity, {})

            # Compute skinning matrices
            self._compute_skinning_matrices(component, bone_transforms)

            # Transform vertices based on method
            if component.method == SkinningMethod.GPU:
                component.prepare_gpu_buffer()
            elif component.method == SkinningMethod.LINEAR:
                self._compute_linear_skinning(component)
            elif component.method == SkinningMethod.DUAL_QUATERNION:
                self._compute_dual_quaternion_skinning(component)

    def _compute_skinning_matrices(
        self,
        component: SkinnedMeshComponent,
        bone_transforms: dict[int, Transform]
    ) -> None:
        """Compute skinning matrices from bone transforms and bind poses."""
        skinning_data = component.skinning_data
        if not skinning_data:
            return

        component.skinning_matrices = []

        for i, bind_pose in enumerate(skinning_data.bind_poses):
            # Get current bone transform
            bone_transform = bone_transforms.get(i, Transform.identity())
            world_matrix = bone_transform.to_matrix()

            # Skinning matrix = world * bind_pose_inverse
            skinning_matrix = world_matrix @ bind_pose
            component.skinning_matrices.append(skinning_matrix)

    def _compute_linear_skinning(self, component: SkinnedMeshComponent) -> None:
        """Compute linear blend skinning on CPU."""
        mesh = component.mesh
        skinning_data = component.skinning_data

        if not mesh or not skinning_data:
            return

        component.skinned_positions = []
        component.skinned_normals = []

        for i, vertex_data in enumerate(skinning_data.vertex_data):
            if i >= mesh.vertex_count:
                break

            pos = mesh.positions[i]
            normal = mesh.normals[i] if i < len(mesh.normals) else Vec3.up()

            skinned_pos = Vec3.zero()
            skinned_normal = Vec3.zero()

            for influence in vertex_data.influences:
                if influence.weight <= 0:
                    continue

                bone_idx = influence.bone_index
                if bone_idx >= len(component.skinning_matrices):
                    continue

                mat = component.skinning_matrices[bone_idx]
                weight = influence.weight

                # Transform position
                transformed_pos = mat.transform_point(pos)
                skinned_pos = skinned_pos + transformed_pos * weight

                # Transform normal (using upper-left 3x3)
                transformed_normal = mat.transform_direction(normal)
                skinned_normal = skinned_normal + transformed_normal * weight

            component.skinned_positions.append(skinned_pos)
            component.skinned_normals.append(skinned_normal.normalized())

    def _compute_dual_quaternion_skinning(self, component: SkinnedMeshComponent) -> None:
        """Compute dual quaternion skinning on CPU.

        Reduces artifacts at joints compared to linear blend skinning.
        """
        mesh = component.mesh
        skinning_data = component.skinning_data

        if not mesh or not skinning_data:
            return

        # Convert skinning matrices to dual quaternions
        dual_quats = [self._mat4_to_dual_quat(mat) for mat in component.skinning_matrices]

        component.skinned_positions = []
        component.skinned_normals = []

        for i, vertex_data in enumerate(skinning_data.vertex_data):
            if i >= mesh.vertex_count:
                break

            pos = mesh.positions[i]
            normal = mesh.normals[i] if i < len(mesh.normals) else Vec3.up()

            # Blend dual quaternions
            blended_real = Quat(0, 0, 0, 0)
            blended_dual = Quat(0, 0, 0, 0)

            first_real = None
            for influence in vertex_data.influences:
                if influence.weight <= 0:
                    continue

                bone_idx = influence.bone_index
                if bone_idx >= len(dual_quats):
                    continue

                real, dual = dual_quats[bone_idx]
                weight = influence.weight

                # Ensure consistent hemisphere
                if first_real is None:
                    first_real = real
                elif real.dot(first_real) < 0:
                    real = Quat(-real.x, -real.y, -real.z, -real.w)
                    dual = Quat(-dual.x, -dual.y, -dual.z, -dual.w)

                blended_real = Quat(
                    blended_real.x + real.x * weight,
                    blended_real.y + real.y * weight,
                    blended_real.z + real.z * weight,
                    blended_real.w + real.w * weight,
                )
                blended_dual = Quat(
                    blended_dual.x + dual.x * weight,
                    blended_dual.y + dual.y * weight,
                    blended_dual.z + dual.z * weight,
                    blended_dual.w + dual.w * weight,
                )

            # Normalize
            length = blended_real.length()
            if length > SKINNING_CONFIG.MIN_QUATERNION_LENGTH:
                blended_real = Quat(
                    blended_real.x / length,
                    blended_real.y / length,
                    blended_real.z / length,
                    blended_real.w / length,
                )
                blended_dual = Quat(
                    blended_dual.x / length,
                    blended_dual.y / length,
                    blended_dual.z / length,
                    blended_dual.w / length,
                )

            # Transform position using dual quaternion
            skinned_pos = self._transform_point_dual_quat(pos, blended_real, blended_dual)

            # Transform normal using just rotation part
            skinned_normal = blended_real.rotate_vector(normal).normalized()

            component.skinned_positions.append(skinned_pos)
            component.skinned_normals.append(skinned_normal)

    def _mat4_to_dual_quat(self, mat: Mat4) -> tuple[Quat, Quat]:
        """Convert matrix to dual quaternion (rotation + translation)."""
        # Extract rotation
        transform = Transform.from_matrix(mat)
        rot = transform.rotation.normalized()

        # Dual part encodes translation
        t = transform.translation
        dual = Quat(
            0.5 * (t.x * rot.w + t.y * rot.z - t.z * rot.y),
            0.5 * (-t.x * rot.z + t.y * rot.w + t.z * rot.x),
            0.5 * (t.x * rot.y - t.y * rot.x + t.z * rot.w),
            -0.5 * (t.x * rot.x + t.y * rot.y + t.z * rot.z),
        )

        return (rot, dual)

    def _transform_point_dual_quat(
        self,
        point: Vec3,
        real: Quat,
        dual: Quat
    ) -> Vec3:
        """Transform point using dual quaternion."""
        # Rotate point
        rotated = real.rotate_vector(point)

        # Extract translation from dual quaternion
        t = Vec3(
            2.0 * (-dual.w * real.x + dual.x * real.w - dual.y * real.z + dual.z * real.y),
            2.0 * (-dual.w * real.y + dual.x * real.z + dual.y * real.w - dual.z * real.x),
            2.0 * (-dual.w * real.z - dual.x * real.y + dual.y * real.x + dual.z * real.w),
        )

        return rotated + t

    def compute_bounding_box(
        self,
        component: SkinnedMeshComponent
    ) -> tuple[Vec3, Vec3] | None:
        """Compute bounding box of skinned mesh.

        Returns:
            Tuple of (min, max) corners, or None if no data
        """
        positions = component.skinned_positions or (component.mesh.positions if component.mesh else None)
        if not positions:
            return None

        min_corner = Vec3(float('inf'), float('inf'), float('inf'))
        max_corner = Vec3(float('-inf'), float('-inf'), float('-inf'))

        for pos in positions:
            min_corner = Vec3(
                min(min_corner.x, pos.x),
                min(min_corner.y, pos.y),
                min(min_corner.z, pos.z),
            )
            max_corner = Vec3(
                max(max_corner.x, pos.x),
                max(max_corner.y, pos.y),
                max(max_corner.z, pos.z),
            )

        return (min_corner, max_corner)
