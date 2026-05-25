"""Mesh skinning implementations: LBS, DQS, and GPU preparation.

This module provides vertex skinning algorithms for skeletal animation:
- Linear Blend Skinning (LBS): Fast, standard approach with candy-wrapper artifacts
- Dual Quaternion Skinning (DQS): Volume-preserving with better joint behavior
- GPU skinning data preparation for compute shader upload
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple

from engine.core.math.vec import Vec3, Vec4
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON

if TYPE_CHECKING:
    from engine.animation.skeletal.skeleton import Skeleton
    from engine.animation.skeletal.pose import Pose


class SkinningMethod(Enum):
    """Skinning algorithm selection."""
    LBS = auto()  # Linear Blend Skinning - fast, standard
    DQS = auto()  # Dual Quaternion Skinning - volume preserving


@dataclass
class VertexWeight:
    """Per-vertex bone influences (up to 4 bones).

    Attributes:
        bone_indices: Tuple of up to 4 bone indices (padded with 0)
        weights: Tuple of corresponding weights (normalized, sum to 1.0)
    """
    bone_indices: Tuple[int, int, int, int] = (0, 0, 0, 0)
    weights: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        # Ensure we have exactly 4 elements
        if len(self.bone_indices) != 4:
            bi = list(self.bone_indices)
            while len(bi) < 4:
                bi.append(0)
            self.bone_indices = tuple(bi[:4])

        if len(self.weights) != 4:
            w = list(self.weights)
            while len(w) < 4:
                w.append(0.0)
            self.weights = tuple(w[:4])

    @staticmethod
    def from_dict(data: dict) -> VertexWeight:
        """Create from serialized dictionary."""
        return VertexWeight(
            bone_indices=tuple(data.get("bone_indices", [0, 0, 0, 0])),
            weights=tuple(data.get("weights", [1.0, 0.0, 0.0, 0.0]))
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "bone_indices": list(self.bone_indices),
            "weights": list(self.weights)
        }

    def normalize(self) -> VertexWeight:
        """Return a new VertexWeight with weights normalized to sum to 1.0."""
        total = sum(self.weights)
        if total < MATH_EPSILON:
            return VertexWeight(self.bone_indices, (1.0, 0.0, 0.0, 0.0))
        normalized = tuple(w / total for w in self.weights)
        return VertexWeight(self.bone_indices, normalized)

    def is_normalized(self, tolerance: float = 1e-6) -> bool:
        """Check if weights sum to approximately 1.0."""
        return abs(sum(self.weights) - 1.0) < tolerance

    @property
    def influence_count(self) -> int:
        """Count of non-zero weight influences."""
        return sum(1 for w in self.weights if w > MATH_EPSILON)


@dataclass
class SkinningData:
    """Complete skinning data for a mesh.

    Attributes:
        vertices: List of vertex positions in bind pose
        weights: Per-vertex bone weights
        bind_pose_matrices: Inverse bind pose matrices per bone
        normals: Optional vertex normals for normal transformation
        tangents: Optional vertex tangents for tangent transformation
    """
    vertices: List[Vec3] = field(default_factory=list)
    weights: List[VertexWeight] = field(default_factory=list)
    bind_pose_matrices: List[Mat4] = field(default_factory=list)
    normals: Optional[List[Vec3]] = None
    tangents: Optional[List[Vec4]] = None

    def __post_init__(self) -> None:
        self._cached_pose_hash: Optional[int] = None
        self._cached_skinning_matrices: Optional[List[Mat4]] = None

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def bone_count(self) -> int:
        return len(self.bind_pose_matrices)

    def validate(self) -> List[str]:
        """Validate skinning data consistency. Returns list of errors."""
        errors = []

        if len(self.vertices) != len(self.weights):
            errors.append(
                f"Vertex count ({len(self.vertices)}) != weight count ({len(self.weights)})"
            )

        if self.normals is not None and len(self.normals) != len(self.vertices):
            errors.append(
                f"Normal count ({len(self.normals)}) != vertex count ({len(self.vertices)})"
            )

        if self.tangents is not None and len(self.tangents) != len(self.vertices):
            errors.append(
                f"Tangent count ({len(self.tangents)}) != vertex count ({len(self.vertices)})"
            )

        for i, w in enumerate(self.weights):
            if not w.is_normalized():
                errors.append(f"Vertex {i} weights not normalized: sum={sum(w.weights)}")

            for bi in w.bone_indices:
                if bi < 0 or bi >= self.bone_count:
                    errors.append(f"Vertex {i} has invalid bone index {bi}")

        return errors


@dataclass
class DualQuaternion:
    """Dual quaternion for rigid transformation representation.

    A dual quaternion encodes rotation and translation in 8 numbers:
    - real: rotation quaternion
    - dual: translation encoded quaternion

    This representation avoids the candy-wrapper artifacts of LBS.
    """
    real: Quat = field(default_factory=Quat.identity)
    dual: Quat = field(default_factory=lambda: Quat(0, 0, 0, 0))

    @staticmethod
    def identity() -> DualQuaternion:
        """Return identity dual quaternion (no transformation)."""
        return DualQuaternion(Quat.identity(), Quat(0, 0, 0, 0))

    @staticmethod
    def from_transform(rotation: Quat, translation: Vec3) -> DualQuaternion:
        """Create dual quaternion from rotation and translation."""
        r = rotation.normalized()
        # dual = 0.5 * translation_quat * rotation
        # where translation_quat = Quat(t.x, t.y, t.z, 0)
        t = Quat(translation.x, translation.y, translation.z, 0)
        d = Quat(
            0.5 * (t.w * r.x + t.x * r.w + t.y * r.z - t.z * r.y),
            0.5 * (t.w * r.y - t.x * r.z + t.y * r.w + t.z * r.x),
            0.5 * (t.w * r.z + t.x * r.y - t.y * r.x + t.z * r.w),
            0.5 * (t.w * r.w - t.x * r.x - t.y * r.y - t.z * r.z),
        )
        return DualQuaternion(r, d)

    @staticmethod
    def from_matrix(m: Mat4) -> DualQuaternion:
        """Create dual quaternion from transformation matrix."""
        t = Transform.from_matrix(m)
        return DualQuaternion.from_transform(t.rotation, t.translation)

    def to_rotation(self) -> Quat:
        """Extract rotation quaternion."""
        return self.real.normalized()

    def to_translation(self) -> Vec3:
        """Extract translation vector."""
        # t = 2 * dual * conjugate(real)
        r = self.real
        d = self.dual
        rc = r.conjugate()

        # Multiply dual * conjugate(real)
        t = Quat(
            d.w * rc.x + d.x * rc.w + d.y * rc.z - d.z * rc.y,
            d.w * rc.y - d.x * rc.z + d.y * rc.w + d.z * rc.x,
            d.w * rc.z + d.x * rc.y - d.y * rc.x + d.z * rc.w,
            d.w * rc.w - d.x * rc.x - d.y * rc.y - d.z * rc.z,
        )
        return Vec3(2 * t.x, 2 * t.y, 2 * t.z)

    def to_matrix(self) -> Mat4:
        """Convert to transformation matrix."""
        rot = self.to_rotation().to_mat4()
        trans = self.to_translation()
        rot.m[12] = trans.x
        rot.m[13] = trans.y
        rot.m[14] = trans.z
        return rot

    def normalized(self) -> DualQuaternion:
        """Normalize the dual quaternion."""
        mag = self.real.length()
        if mag < MATH_EPSILON:
            return DualQuaternion.identity()
        return DualQuaternion(
            Quat(self.real.x / mag, self.real.y / mag,
                 self.real.z / mag, self.real.w / mag),
            Quat(self.dual.x / mag, self.dual.y / mag,
                 self.dual.z / mag, self.dual.w / mag)
        )

    def __add__(self, other: DualQuaternion) -> DualQuaternion:
        """Add two dual quaternions (for blending)."""
        return DualQuaternion(
            Quat(self.real.x + other.real.x, self.real.y + other.real.y,
                 self.real.z + other.real.z, self.real.w + other.real.w),
            Quat(self.dual.x + other.dual.x, self.dual.y + other.dual.y,
                 self.dual.z + other.dual.z, self.dual.w + other.dual.w)
        )

    def __mul__(self, scalar: float) -> DualQuaternion:
        """Scale dual quaternion by scalar (for weighted blending)."""
        return DualQuaternion(
            Quat(self.real.x * scalar, self.real.y * scalar,
                 self.real.z * scalar, self.real.w * scalar),
            Quat(self.dual.x * scalar, self.dual.y * scalar,
                 self.dual.z * scalar, self.dual.w * scalar)
        )

    def __rmul__(self, scalar: float) -> DualQuaternion:
        return self.__mul__(scalar)

    def transform_point(self, p: Vec3) -> Vec3:
        """Transform a point by this dual quaternion."""
        rot = self.to_rotation()
        trans = self.to_translation()
        return rot.rotate_vector(p) + trans

    def dot(self, other: DualQuaternion) -> float:
        """Dot product of real parts (for antipodality check)."""
        return self.real.dot(other.real)


class LinearBlendSkinning:
    """Linear Blend Skinning (LBS) implementation.

    LBS is the standard skinning technique that blends transformation
    matrices weighted by vertex influences. It's fast but can produce
    artifacts at extreme joint angles (candy-wrapper effect).
    """

    @staticmethod
    def compute_skinning_matrices(
        bone_world_transforms: List[Mat4],
        bind_pose_inverses: List[Mat4]
    ) -> List[Mat4]:
        """Compute skinning matrices from current pose.

        Args:
            bone_world_transforms: Current world-space bone transforms
            bind_pose_inverses: Inverse bind pose matrices

        Returns:
            List of skinning matrices (bone_world * bind_inverse)
        """
        if len(bone_world_transforms) != len(bind_pose_inverses):
            raise ValueError(
                f"Transform count ({len(bone_world_transforms)}) != "
                f"bind pose count ({len(bind_pose_inverses)})"
            )

        return [
            world @ bind_inv
            for world, bind_inv in zip(bone_world_transforms, bind_pose_inverses)
        ]

    @staticmethod
    def skin_vertex(
        vertex: Vec3,
        skinning_matrices: List[Mat4],
        weight: VertexWeight
    ) -> Vec3:
        """Skin a single vertex using LBS.

        Args:
            vertex: Original vertex position
            skinning_matrices: Pre-computed skinning matrices
            weight: Bone influences for this vertex

        Returns:
            Transformed vertex position
        """
        result = Vec3.zero()

        for i in range(4):
            w = weight.weights[i]
            if w < MATH_EPSILON:
                continue

            bone_idx = weight.bone_indices[i]
            if bone_idx < 0 or bone_idx >= len(skinning_matrices):
                continue

            transformed = skinning_matrices[bone_idx].transform_point(vertex)
            result = result + transformed * w

        return result

    @staticmethod
    def skin_vertices(
        vertices: List[Vec3],
        skinning_matrices: List[Mat4],
        weights: List[VertexWeight]
    ) -> List[Vec3]:
        """Skin all vertices using LBS.

        Args:
            vertices: Original vertex positions
            skinning_matrices: Pre-computed skinning matrices
            weights: Per-vertex bone influences

        Returns:
            List of transformed vertex positions
        """
        if len(vertices) != len(weights):
            raise ValueError(
                f"Vertex count ({len(vertices)}) != weight count ({len(weights)})"
            )

        return [
            LinearBlendSkinning.skin_vertex(v, skinning_matrices, w)
            for v, w in zip(vertices, weights)
        ]

    @staticmethod
    def skin_normal(
        normal: Vec3,
        skinning_matrices: List[Mat4],
        weight: VertexWeight
    ) -> Vec3:
        """Skin a normal vector (direction, not point)."""
        result = Vec3.zero()

        for i in range(4):
            w = weight.weights[i]
            if w < MATH_EPSILON:
                continue

            bone_idx = weight.bone_indices[i]
            if bone_idx < 0 or bone_idx >= len(skinning_matrices):
                continue

            transformed = skinning_matrices[bone_idx].transform_direction(normal)
            result = result + transformed * w

        return result.normalized()

    @staticmethod
    def skin_normals(
        normals: List[Vec3],
        skinning_matrices: List[Mat4],
        weights: List[VertexWeight]
    ) -> List[Vec3]:
        """Skin all normals using LBS."""
        return [
            LinearBlendSkinning.skin_normal(n, skinning_matrices, w)
            for n, w in zip(normals, weights)
        ]


class DualQuaternionSkinning:
    """Dual Quaternion Skinning (DQS) implementation.

    DQS uses dual quaternions instead of matrices for blending,
    which preserves volume and avoids the candy-wrapper artifact
    at the cost of slightly higher computation.
    """

    @staticmethod
    def matrix_to_dual_quaternion(m: Mat4) -> DualQuaternion:
        """Convert a transformation matrix to dual quaternion."""
        return DualQuaternion.from_matrix(m)

    @staticmethod
    def compute_skinning_dual_quaternions(
        bone_world_transforms: List[Mat4],
        bind_pose_inverses: List[Mat4]
    ) -> List[DualQuaternion]:
        """Compute dual quaternions from skinning matrices.

        Args:
            bone_world_transforms: Current world-space bone transforms
            bind_pose_inverses: Inverse bind pose matrices

        Returns:
            List of dual quaternions for skinning
        """
        skinning_matrices = LinearBlendSkinning.compute_skinning_matrices(
            bone_world_transforms, bind_pose_inverses
        )
        return [DualQuaternion.from_matrix(m) for m in skinning_matrices]

    @staticmethod
    def _handle_antipodality(
        dual_quats: List[DualQuaternion],
        weight: VertexWeight,
        base_idx: int
    ) -> List[DualQuaternion]:
        """Handle antipodality issues in dual quaternion blending.

        Dual quaternions q and -q represent the same transformation,
        but blending them produces incorrect results. We ensure all
        quaternions being blended are in the same hemisphere.

        Args:
            dual_quats: List of all dual quaternions
            weight: Vertex weight with bone indices
            base_idx: Index of the reference quaternion (highest weight)

        Returns:
            List of dual quaternions with antipodality handled
        """
        base_dq = dual_quats[weight.bone_indices[base_idx]]
        result = list(dual_quats)  # Copy to avoid mutation

        for i in range(4):
            if weight.weights[i] < MATH_EPSILON:
                continue
            if i == base_idx:
                continue

            bone_idx = weight.bone_indices[i]
            if bone_idx < 0 or bone_idx >= len(dual_quats):
                continue

            dq = result[bone_idx]
            # If dot product is negative, negate to bring to same hemisphere
            if base_dq.dot(dq) < 0:
                result[bone_idx] = DualQuaternion(
                    Quat(-dq.real.x, -dq.real.y, -dq.real.z, -dq.real.w),
                    Quat(-dq.dual.x, -dq.dual.y, -dq.dual.z, -dq.dual.w)
                )

        return result

    @staticmethod
    def skin_vertex(
        vertex: Vec3,
        dual_quaternions: List[DualQuaternion],
        weight: VertexWeight
    ) -> Vec3:
        """Skin a single vertex using DQS with DLB blending.

        DLB (Dual quaternion Linear Blending) directly blends dual
        quaternions then normalizes, providing smooth interpolation.

        Args:
            vertex: Original vertex position
            dual_quaternions: Dual quaternions per bone
            weight: Bone influences for this vertex

        Returns:
            Transformed vertex position
        """
        # Find the bone with highest weight for antipodality reference
        max_weight_idx = 0
        max_weight = weight.weights[0]
        for i in range(1, 4):
            if weight.weights[i] > max_weight:
                max_weight = weight.weights[i]
                max_weight_idx = i

        # Handle antipodality
        adjusted_dqs = DualQuaternionSkinning._handle_antipodality(
            dual_quaternions, weight, max_weight_idx
        )

        # Blend dual quaternions
        blended = DualQuaternion(
            Quat(0, 0, 0, 0),
            Quat(0, 0, 0, 0)
        )

        for i in range(4):
            w = weight.weights[i]
            if w < MATH_EPSILON:
                continue

            bone_idx = weight.bone_indices[i]
            if bone_idx < 0 or bone_idx >= len(adjusted_dqs):
                continue

            blended = blended + adjusted_dqs[bone_idx] * w

        # Normalize and transform
        blended = blended.normalized()
        return blended.transform_point(vertex)

    @staticmethod
    def skin_vertices(
        vertices: List[Vec3],
        dual_quaternions: List[DualQuaternion],
        weights: List[VertexWeight]
    ) -> List[Vec3]:
        """Skin all vertices using DQS."""
        return [
            DualQuaternionSkinning.skin_vertex(v, dual_quaternions, w)
            for v, w in zip(vertices, weights)
        ]

    @staticmethod
    def skin_normal(
        normal: Vec3,
        dual_quaternions: List[DualQuaternion],
        weight: VertexWeight
    ) -> Vec3:
        """Skin a normal vector using DQS."""
        # Find the bone with highest weight for antipodality reference
        max_weight_idx = 0
        max_weight = weight.weights[0]
        for i in range(1, 4):
            if weight.weights[i] > max_weight:
                max_weight = weight.weights[i]
                max_weight_idx = i

        adjusted_dqs = DualQuaternionSkinning._handle_antipodality(
            dual_quaternions, weight, max_weight_idx
        )

        # Blend only rotations for normals
        blended_rot = Quat(0, 0, 0, 0)
        for i in range(4):
            w = weight.weights[i]
            if w < MATH_EPSILON:
                continue

            bone_idx = weight.bone_indices[i]
            if bone_idx < 0 or bone_idx >= len(adjusted_dqs):
                continue

            r = adjusted_dqs[bone_idx].real
            blended_rot = Quat(
                blended_rot.x + r.x * w,
                blended_rot.y + r.y * w,
                blended_rot.z + r.z * w,
                blended_rot.w + r.w * w,
            )

        blended_rot = blended_rot.normalized()
        return blended_rot.rotate_vector(normal).normalized()


@dataclass
class GPUSkinningData:
    """Data prepared for GPU compute shader skinning.

    This class packages skinning data in a format suitable for
    upload to GPU buffers for compute shader skinning.
    """
    # Required fields first (no defaults) - must be ordered properly
    positions: List[float] = field(default_factory=list)  # Flat: x0,y0,z0,x1,y1,z1,...
    bone_indices: List[int] = field(default_factory=list)  # Flat: i0,i1,i2,i3 per vertex
    bone_weights: List[float] = field(default_factory=list)  # Flat: w0,w1,w2,w3 per vertex

    # Optional fields with defaults
    normals: Optional[List[float]] = None
    tangents: Optional[List[float]] = None
    skinning_matrices: List[float] = field(default_factory=list)
    skinning_dual_quaternions: Optional[List[float]] = None

    @property
    def vertex_count(self) -> int:
        return len(self.positions) // 3

    @property
    def bone_count(self) -> int:
        return len(self.skinning_matrices) // 16


def prepare_gpu_skinning_data(
    skinning_data: SkinningData,
    bone_world_transforms: List[Mat4],
    method: SkinningMethod = SkinningMethod.LBS
) -> GPUSkinningData:
    """Prepare skinning data for GPU upload.

    Args:
        skinning_data: CPU skinning data
        bone_world_transforms: Current bone transforms
        method: Skinning method to prepare for

    Returns:
        GPUSkinningData ready for buffer upload
    """
    # Flatten vertex positions
    positions = []
    for v in skinning_data.vertices:
        positions.extend([v.x, v.y, v.z])

    # Flatten normals if present
    normals = None
    if skinning_data.normals:
        normals = []
        for n in skinning_data.normals:
            normals.extend([n.x, n.y, n.z])

    # Flatten tangents if present
    tangents = None
    if skinning_data.tangents:
        tangents = []
        for t in skinning_data.tangents:
            tangents.extend([t.x, t.y, t.z, t.w])

    # Flatten bone indices and weights
    bone_indices = []
    bone_weights = []
    for w in skinning_data.weights:
        bone_indices.extend(w.bone_indices)
        bone_weights.extend(w.weights)

    # Compute and flatten skinning matrices
    skinning_matrices_list = LinearBlendSkinning.compute_skinning_matrices(
        bone_world_transforms, skinning_data.bind_pose_matrices
    )
    skinning_matrices = []
    for m in skinning_matrices_list:
        skinning_matrices.extend(m.m)

    # Optionally compute dual quaternions
    skinning_dqs = None
    if method == SkinningMethod.DQS:
        dqs = DualQuaternionSkinning.compute_skinning_dual_quaternions(
            bone_world_transforms, skinning_data.bind_pose_matrices
        )
        skinning_dqs = []
        for dq in dqs:
            skinning_dqs.extend([
                dq.real.x, dq.real.y, dq.real.z, dq.real.w,
                dq.dual.x, dq.dual.y, dq.dual.z, dq.dual.w
            ])

    return GPUSkinningData(
        positions=positions,
        normals=normals,
        tangents=tangents,
        bone_indices=bone_indices,
        bone_weights=bone_weights,
        skinning_matrices=skinning_matrices,
        skinning_dual_quaternions=skinning_dqs
    )


class SkinningCache:
    """Cache for skinning matrices when pose unchanged.

    Caches computed skinning matrices/dual quaternions to avoid
    redundant computation when the pose hasn't changed.
    """

    def __init__(self) -> None:
        self._pose_hash: Optional[int] = None
        self._cached_matrices: Optional[List[Mat4]] = None
        self._cached_dqs: Optional[List[DualQuaternion]] = None

    def _compute_pose_hash(self, bone_transforms: List[Mat4]) -> int:
        """Compute a hash of the current pose for cache validation."""
        # Use a simple hash based on matrix elements
        h = 0
        for m in bone_transforms:
            for v in m.m:
                h ^= hash(round(v, 6))
        return h

    def get_skinning_matrices(
        self,
        bone_world_transforms: List[Mat4],
        bind_pose_inverses: List[Mat4]
    ) -> List[Mat4]:
        """Get skinning matrices, using cache if pose unchanged."""
        pose_hash = self._compute_pose_hash(bone_world_transforms)

        if pose_hash == self._pose_hash and self._cached_matrices is not None:
            return self._cached_matrices

        self._cached_matrices = LinearBlendSkinning.compute_skinning_matrices(
            bone_world_transforms, bind_pose_inverses
        )
        self._cached_dqs = None  # Invalidate DQ cache
        self._pose_hash = pose_hash

        return self._cached_matrices

    def get_dual_quaternions(
        self,
        bone_world_transforms: List[Mat4],
        bind_pose_inverses: List[Mat4]
    ) -> List[DualQuaternion]:
        """Get dual quaternions, using cache if pose unchanged."""
        pose_hash = self._compute_pose_hash(bone_world_transforms)

        if pose_hash == self._pose_hash and self._cached_dqs is not None:
            return self._cached_dqs

        self._cached_dqs = DualQuaternionSkinning.compute_skinning_dual_quaternions(
            bone_world_transforms, bind_pose_inverses
        )
        self._cached_matrices = None  # We can compute these from DQs if needed
        self._pose_hash = pose_hash

        return self._cached_dqs

    def invalidate(self) -> None:
        """Force cache invalidation."""
        self._pose_hash = None
        self._cached_matrices = None
        self._cached_dqs = None


def skin_mesh(
    skinning_data: SkinningData,
    bone_world_transforms: List[Mat4],
    method: SkinningMethod = SkinningMethod.LBS,
    cache: Optional[SkinningCache] = None
) -> Tuple[List[Vec3], Optional[List[Vec3]]]:
    """Skin a mesh using the specified method.

    Args:
        skinning_data: Mesh skinning data
        bone_world_transforms: Current world-space bone transforms
        method: Skinning algorithm to use
        cache: Optional cache for skinning matrices

    Returns:
        Tuple of (skinned_vertices, skinned_normals or None)
    """
    if method == SkinningMethod.LBS:
        if cache:
            matrices = cache.get_skinning_matrices(
                bone_world_transforms, skinning_data.bind_pose_matrices
            )
        else:
            matrices = LinearBlendSkinning.compute_skinning_matrices(
                bone_world_transforms, skinning_data.bind_pose_matrices
            )

        vertices = LinearBlendSkinning.skin_vertices(
            skinning_data.vertices, matrices, skinning_data.weights
        )

        normals = None
        if skinning_data.normals:
            normals = LinearBlendSkinning.skin_normals(
                skinning_data.normals, matrices, skinning_data.weights
            )

        return vertices, normals

    else:  # DQS
        if cache:
            dqs = cache.get_dual_quaternions(
                bone_world_transforms, skinning_data.bind_pose_matrices
            )
        else:
            dqs = DualQuaternionSkinning.compute_skinning_dual_quaternions(
                bone_world_transforms, skinning_data.bind_pose_matrices
            )

        vertices = DualQuaternionSkinning.skin_vertices(
            skinning_data.vertices, dqs, skinning_data.weights
        )

        normals = None
        if skinning_data.normals:
            normals = [
                DualQuaternionSkinning.skin_normal(n, dqs, w)
                for n, w in zip(skinning_data.normals, skinning_data.weights)
            ]

        return vertices, normals
