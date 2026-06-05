"""Whitebox tests for skinning.py.

Tests LBS, DQS skinning implementations, dual quaternion operations,
and GPU data preparation.

Acceptance criteria:
- T-SKEL-1.4: DQS Implementation
  - DQ construction
  - Antipodality handling
  - Point/normal transformation
"""

import math
import pytest
from engine.core.math import Vec3, Vec4, Quat, Mat4, Transform
from engine.core.constants import MATH_EPSILON
from engine.animation.skeletal.skinning import (
    SkinningMethod, VertexWeight, SkinningData,
    DualQuaternion, LinearBlendSkinning, DualQuaternionSkinning,
    GPUSkinningData, prepare_gpu_skinning_data, skin_mesh,
    SkinningCache
)


# =============================================================================
# VertexWeight Tests
# =============================================================================

class TestVertexWeight:
    """Tests for VertexWeight dataclass."""

    def test_vertex_weight_default(self):
        """Test default vertex weight."""
        vw = VertexWeight()
        assert vw.bone_indices == (0, 0, 0, 0)
        assert vw.weights == (1.0, 0.0, 0.0, 0.0)

    def test_vertex_weight_custom(self):
        """Test custom vertex weight."""
        vw = VertexWeight(
            bone_indices=(0, 1, 2, 3),
            weights=(0.5, 0.3, 0.15, 0.05)
        )
        assert vw.bone_indices == (0, 1, 2, 3)
        assert abs(sum(vw.weights) - 1.0) < 1e-6

    def test_vertex_weight_pad_indices(self):
        """Test that bone indices are padded to 4."""
        vw = VertexWeight(bone_indices=(0, 1), weights=(0.5, 0.5, 0.0, 0.0))
        assert len(vw.bone_indices) == 4
        assert vw.bone_indices == (0, 1, 0, 0)

    def test_vertex_weight_pad_weights(self):
        """Test that weights are padded to 4."""
        vw = VertexWeight(bone_indices=(0, 0, 0, 0), weights=(1.0,))
        assert len(vw.weights) == 4
        assert vw.weights == (1.0, 0.0, 0.0, 0.0)

    def test_vertex_weight_normalize(self):
        """Test weight normalization."""
        vw = VertexWeight(
            bone_indices=(0, 1, 0, 0),
            weights=(2.0, 2.0, 0.0, 0.0)
        )
        normalized = vw.normalize()

        assert abs(normalized.weights[0] - 0.5) < 1e-6
        assert abs(normalized.weights[1] - 0.5) < 1e-6
        assert abs(sum(normalized.weights) - 1.0) < 1e-6

    def test_vertex_weight_normalize_zero(self):
        """Test normalizing zero weights."""
        vw = VertexWeight(
            bone_indices=(0, 1, 0, 0),
            weights=(0.0, 0.0, 0.0, 0.0)
        )
        normalized = vw.normalize()

        # Should fallback to single bone with weight 1
        assert normalized.weights[0] == 1.0

    def test_vertex_weight_is_normalized(self):
        """Test checking if weights are normalized."""
        normalized = VertexWeight(weights=(0.5, 0.3, 0.15, 0.05))
        unnormalized = VertexWeight(weights=(2.0, 2.0, 0.0, 0.0))

        assert normalized.is_normalized() is True
        assert unnormalized.is_normalized() is False

    def test_vertex_weight_influence_count(self):
        """Test counting non-zero influences."""
        vw1 = VertexWeight(weights=(1.0, 0.0, 0.0, 0.0))
        vw2 = VertexWeight(weights=(0.5, 0.3, 0.15, 0.05))
        vw3 = VertexWeight(weights=(0.7, 0.3, 0.0, 0.0))

        assert vw1.influence_count == 1
        assert vw2.influence_count == 4
        assert vw3.influence_count == 2

    def test_vertex_weight_serialization(self):
        """Test to_dict and from_dict."""
        original = VertexWeight(
            bone_indices=(0, 1, 2, 3),
            weights=(0.5, 0.3, 0.15, 0.05)
        )

        data = original.to_dict()
        restored = VertexWeight.from_dict(data)

        assert restored.bone_indices == original.bone_indices
        assert restored.weights == original.weights


# =============================================================================
# SkinningData Tests
# =============================================================================

class TestSkinningData:
    """Tests for SkinningData dataclass."""

    def test_skinning_data_creation(self):
        """Test creating skinning data."""
        data = SkinningData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0)],
            weights=[VertexWeight(), VertexWeight()],
            bind_pose_matrices=[Mat4.identity(), Mat4.identity()]
        )

        assert data.vertex_count == 2
        assert data.bone_count == 2

    def test_skinning_data_validate_success(self):
        """Test validation passes for valid data."""
        data = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight()],
            bind_pose_matrices=[Mat4.identity()]
        )

        errors = data.validate()
        assert len(errors) == 0

    def test_skinning_data_validate_vertex_weight_mismatch(self):
        """Test validation catches vertex/weight count mismatch."""
        data = SkinningData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0)],
            weights=[VertexWeight()],  # Only 1 weight for 2 vertices
            bind_pose_matrices=[Mat4.identity()]
        )

        errors = data.validate()
        assert any("!=" in e for e in errors)

    def test_skinning_data_validate_unnormalized_weights(self):
        """Test validation catches unnormalized weights."""
        data = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight(weights=(2.0, 2.0, 0.0, 0.0))],
            bind_pose_matrices=[Mat4.identity()]
        )

        errors = data.validate()
        assert any("not normalized" in e for e in errors)

    def test_skinning_data_validate_invalid_bone_index(self):
        """Test validation catches invalid bone indices."""
        data = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight(bone_indices=(99, 0, 0, 0))],
            bind_pose_matrices=[Mat4.identity()]
        )

        errors = data.validate()
        assert any("invalid bone index" in e for e in errors)


# =============================================================================
# DualQuaternion Tests - T-SKEL-1.4
# =============================================================================

class TestDualQuaternion:
    """Tests for T-SKEL-1.4: DQ construction and operations."""

    def test_dq_identity(self):
        """Test identity dual quaternion."""
        dq = DualQuaternion.identity()

        assert dq.real.w == 1.0
        assert dq.real.x == 0.0
        assert dq.real.y == 0.0
        assert dq.real.z == 0.0
        assert dq.dual.x == 0.0
        assert dq.dual.y == 0.0
        assert dq.dual.z == 0.0
        assert dq.dual.w == 0.0

    def test_dq_from_transform_translation_only(self):
        """Test DQ construction from translation."""
        translation = Vec3(1, 2, 3)
        rotation = Quat.identity()

        dq = DualQuaternion.from_transform(rotation, translation)

        # Verify translation extraction
        extracted = dq.to_translation()
        assert abs(extracted.x - 1.0) < 1e-6
        assert abs(extracted.y - 2.0) < 1e-6
        assert abs(extracted.z - 3.0) < 1e-6

    def test_dq_from_transform_rotation_only(self):
        """Test DQ construction from rotation."""
        translation = Vec3.zero()
        angle = math.pi / 2
        rotation = Quat(0, math.sin(angle/2), 0, math.cos(angle/2))  # 90 deg Y

        dq = DualQuaternion.from_transform(rotation, translation)

        # Verify rotation extraction
        extracted = dq.to_rotation()
        assert abs(extracted.y - rotation.y) < 1e-6
        assert abs(extracted.w - rotation.w) < 1e-6

    def test_dq_from_transform_combined(self):
        """Test DQ construction from rotation and translation."""
        translation = Vec3(5, 0, 0)
        angle = math.pi / 4  # 45 degrees
        rotation = Quat(0, 0, math.sin(angle/2), math.cos(angle/2))

        dq = DualQuaternion.from_transform(rotation, translation)

        # Verify both components
        ext_rot = dq.to_rotation()
        ext_trans = dq.to_translation()

        assert abs(ext_trans.x - 5.0) < 1e-6
        assert abs(ext_rot.w - rotation.w) < 1e-6

    def test_dq_from_matrix(self):
        """Test DQ construction from transformation matrix."""
        # Create a matrix with translation
        mat = Mat4.identity()
        mat.m[12] = 3.0  # x translation
        mat.m[13] = 4.0  # y translation
        mat.m[14] = 5.0  # z translation

        dq = DualQuaternion.from_matrix(mat)
        trans = dq.to_translation()

        assert abs(trans.x - 3.0) < 1e-5
        assert abs(trans.y - 4.0) < 1e-5
        assert abs(trans.z - 5.0) < 1e-5

    def test_dq_to_matrix_roundtrip(self):
        """Test DQ to matrix conversion preserves transform."""
        translation = Vec3(1, 2, 3)
        angle = math.pi / 6
        rotation = Quat(0, 0, math.sin(angle/2), math.cos(angle/2))

        dq = DualQuaternion.from_transform(rotation, translation)
        mat = dq.to_matrix()

        # Matrix translation should match
        assert abs(mat.m[12] - 1.0) < 1e-5
        assert abs(mat.m[13] - 2.0) < 1e-5
        assert abs(mat.m[14] - 3.0) < 1e-5

    def test_dq_normalized(self):
        """Test DQ normalization."""
        # Create non-normalized DQ
        dq = DualQuaternion(
            real=Quat(0.5, 0.5, 0.5, 0.5),  # Not unit
            dual=Quat(0.1, 0.1, 0.1, 0.1)
        )

        normalized = dq.normalized()

        length = math.sqrt(
            normalized.real.x**2 + normalized.real.y**2 +
            normalized.real.z**2 + normalized.real.w**2
        )
        assert abs(length - 1.0) < 1e-6

    def test_dq_addition(self):
        """Test DQ addition for blending."""
        dq1 = DualQuaternion(
            real=Quat(0.0, 0.0, 0.0, 1.0),
            dual=Quat(0.5, 0.0, 0.0, 0.0)
        )
        dq2 = DualQuaternion(
            real=Quat(0.0, 0.0, 0.0, 1.0),
            dual=Quat(0.5, 0.0, 0.0, 0.0)
        )

        result = dq1 + dq2

        assert abs(result.real.w - 2.0) < 1e-6
        assert abs(result.dual.x - 1.0) < 1e-6

    def test_dq_scalar_multiply(self):
        """Test DQ scalar multiplication for weighting."""
        dq = DualQuaternion(
            real=Quat(0.0, 0.0, 0.0, 1.0),
            dual=Quat(1.0, 0.0, 0.0, 0.0)
        )

        result = dq * 0.5

        assert abs(result.real.w - 0.5) < 1e-6
        assert abs(result.dual.x - 0.5) < 1e-6

    def test_dq_rmul(self):
        """Test right multiplication (scalar * dq)."""
        dq = DualQuaternion(
            real=Quat(0.0, 0.0, 0.0, 1.0),
            dual=Quat(1.0, 0.0, 0.0, 0.0)
        )

        result = 2.0 * dq

        assert abs(result.real.w - 2.0) < 1e-6

    def test_dq_transform_point(self):
        """Test T-SKEL-1.4: Point transformation."""
        # Translation only
        translation = Vec3(5, 0, 0)
        dq = DualQuaternion.from_transform(Quat.identity(), translation)

        point = Vec3(1, 2, 3)
        result = dq.transform_point(point)

        # Point should be translated
        assert abs(result.x - 6.0) < 1e-6
        assert abs(result.y - 2.0) < 1e-6
        assert abs(result.z - 3.0) < 1e-6

    def test_dq_transform_point_with_rotation(self):
        """Test point transformation with rotation."""
        # 90 degree rotation around Z axis
        angle = math.pi / 2
        rotation = Quat(0, 0, math.sin(angle/2), math.cos(angle/2))
        dq = DualQuaternion.from_transform(rotation, Vec3.zero())

        point = Vec3(1, 0, 0)
        result = dq.transform_point(point)

        # (1, 0, 0) rotated 90 deg around Z = (0, 1, 0)
        assert abs(result.x) < 1e-5
        assert abs(result.y - 1.0) < 1e-5
        assert abs(result.z) < 1e-5

    def test_dq_dot_product(self):
        """Test T-SKEL-1.4: DQ dot product for antipodality check."""
        dq1 = DualQuaternion(real=Quat(0, 0, 0, 1), dual=Quat(0, 0, 0, 0))
        dq2 = DualQuaternion(real=Quat(0, 0, 0, 1), dual=Quat(0, 0, 0, 0))
        dq3 = DualQuaternion(real=Quat(0, 0, 0, -1), dual=Quat(0, 0, 0, 0))

        # Same hemisphere
        assert dq1.dot(dq2) > 0

        # Opposite hemisphere (antipodal)
        assert dq1.dot(dq3) < 0


# =============================================================================
# Linear Blend Skinning Tests
# =============================================================================

class TestLinearBlendSkinning:
    """Tests for LBS implementation."""

    def test_lbs_compute_skinning_matrices(self):
        """Test computing skinning matrices."""
        world_transforms = [Mat4.identity()]
        bind_inverses = [Mat4.identity()]

        matrices = LinearBlendSkinning.compute_skinning_matrices(
            world_transforms, bind_inverses
        )

        assert len(matrices) == 1
        # Identity * identity = identity
        assert abs(matrices[0].m[0] - 1.0) < 1e-6

    def test_lbs_compute_skinning_matrices_mismatch(self):
        """Test skinning matrices with mismatched counts raises error."""
        world_transforms = [Mat4.identity(), Mat4.identity()]
        bind_inverses = [Mat4.identity()]

        with pytest.raises(ValueError, match="!="):
            LinearBlendSkinning.compute_skinning_matrices(
                world_transforms, bind_inverses
            )

    def test_lbs_skin_vertex_single_bone(self):
        """Test skinning vertex with single bone influence."""
        # Translation matrix
        mat = Mat4.identity()
        mat.m[12] = 5.0  # x translation

        weight = VertexWeight()  # Default: bone 0, weight 1.0
        vertex = Vec3(0, 0, 0)

        result = LinearBlendSkinning.skin_vertex(vertex, [mat], weight)

        assert abs(result.x - 5.0) < 1e-6

    def test_lbs_skin_vertex_multiple_bones(self):
        """Test skinning vertex with multiple bone influences."""
        mat0 = Mat4.identity()
        mat0.m[12] = 10.0  # x = 10

        mat1 = Mat4.identity()
        mat1.m[12] = 0.0  # x = 0

        weight = VertexWeight(
            bone_indices=(0, 1, 0, 0),
            weights=(0.5, 0.5, 0.0, 0.0)
        )
        vertex = Vec3(0, 0, 0)

        result = LinearBlendSkinning.skin_vertex(vertex, [mat0, mat1], weight)

        # 0.5 * 10 + 0.5 * 0 = 5
        assert abs(result.x - 5.0) < 1e-6

    def test_lbs_skin_vertices_batch(self):
        """Test batch skinning of vertices."""
        mat = Mat4.identity()
        mat.m[12] = 2.0

        vertices = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0)]
        weights = [VertexWeight() for _ in range(3)]

        results = LinearBlendSkinning.skin_vertices(vertices, [mat], weights)

        assert len(results) == 3
        assert abs(results[0].x - 2.0) < 1e-6
        assert abs(results[1].x - 3.0) < 1e-6
        assert abs(results[2].x - 4.0) < 1e-6

    def test_lbs_skin_normal(self):
        """Test normal skinning."""
        # 90 degree rotation around Z
        angle = math.pi / 2
        q = Quat(0, 0, math.sin(angle/2), math.cos(angle/2))
        mat = q.to_mat4()

        weight = VertexWeight()
        normal = Vec3(1, 0, 0)

        result = LinearBlendSkinning.skin_normal(normal, [mat], weight)

        # Normal (1, 0, 0) rotated 90 deg around Z = (0, 1, 0)
        assert abs(result.x) < 1e-5
        assert abs(result.y - 1.0) < 1e-5

    def test_lbs_skin_normals_batch(self):
        """Test batch normal skinning."""
        mat = Mat4.identity()  # No rotation

        normals = [Vec3(1, 0, 0), Vec3(0, 1, 0)]
        weights = [VertexWeight(), VertexWeight()]

        results = LinearBlendSkinning.skin_normals(normals, [mat], weights)

        assert len(results) == 2


# =============================================================================
# Dual Quaternion Skinning Tests - T-SKEL-1.4
# =============================================================================

class TestDualQuaternionSkinning:
    """Tests for T-SKEL-1.4: DQS implementation."""

    def test_dqs_matrix_to_dual_quaternion(self):
        """Test matrix to DQ conversion."""
        mat = Mat4.identity()
        mat.m[12] = 5.0

        dq = DualQuaternionSkinning.matrix_to_dual_quaternion(mat)

        trans = dq.to_translation()
        assert abs(trans.x - 5.0) < 1e-5

    def test_dqs_compute_skinning_dual_quaternions(self):
        """Test computing skinning DQs from matrices."""
        world = [Mat4.identity()]
        world[0].m[12] = 2.0

        bind_inv = [Mat4.identity()]

        dqs = DualQuaternionSkinning.compute_skinning_dual_quaternions(world, bind_inv)

        assert len(dqs) == 1
        trans = dqs[0].to_translation()
        assert abs(trans.x - 2.0) < 1e-5

    def test_dqs_handle_antipodality(self):
        """Test T-SKEL-1.4: Antipodality handling."""
        # Create two DQs in opposite hemispheres
        dq0 = DualQuaternion(real=Quat(0, 0, 0, 1), dual=Quat(0.5, 0, 0, 0))
        dq1 = DualQuaternion(real=Quat(0, 0, 0, -1), dual=Quat(-0.5, 0, 0, 0))

        weight = VertexWeight(
            bone_indices=(0, 1, 0, 0),
            weights=(0.5, 0.5, 0.0, 0.0)
        )

        adjusted = DualQuaternionSkinning._handle_antipodality(
            [dq0, dq1], weight, 0  # base_idx = 0
        )

        # After antipodality fix, both should be in same hemisphere
        assert adjusted[0].dot(adjusted[1]) >= 0

    def test_dqs_skin_vertex_single_bone(self):
        """Test T-SKEL-1.4: DQS point transformation single bone."""
        dq = DualQuaternion.from_transform(Quat.identity(), Vec3(3, 0, 0))
        weight = VertexWeight()
        vertex = Vec3(1, 0, 0)

        result = DualQuaternionSkinning.skin_vertex(vertex, [dq], weight)

        # 1 + 3 = 4
        assert abs(result.x - 4.0) < 1e-5

    def test_dqs_skin_vertex_multiple_bones(self):
        """Test DQS with multiple bone influences."""
        dq0 = DualQuaternion.from_transform(Quat.identity(), Vec3(10, 0, 0))
        dq1 = DualQuaternion.from_transform(Quat.identity(), Vec3(0, 0, 0))

        weight = VertexWeight(
            bone_indices=(0, 1, 0, 0),
            weights=(0.5, 0.5, 0.0, 0.0)
        )
        vertex = Vec3(0, 0, 0)

        result = DualQuaternionSkinning.skin_vertex(vertex, [dq0, dq1], weight)

        # Blended translation should be ~5
        assert abs(result.x - 5.0) < 1e-4

    def test_dqs_skin_vertices_batch(self):
        """Test batch DQS skinning."""
        dq = DualQuaternion.from_transform(Quat.identity(), Vec3(1, 0, 0))

        vertices = [Vec3(0, 0, 0), Vec3(1, 0, 0)]
        weights = [VertexWeight(), VertexWeight()]

        results = DualQuaternionSkinning.skin_vertices(vertices, [dq], weights)

        assert len(results) == 2
        assert abs(results[0].x - 1.0) < 1e-5
        assert abs(results[1].x - 2.0) < 1e-5

    def test_dqs_skin_normal(self):
        """Test T-SKEL-1.4: DQS normal transformation."""
        # 90 degree rotation around Z
        angle = math.pi / 2
        rotation = Quat(0, 0, math.sin(angle/2), math.cos(angle/2))
        dq = DualQuaternion.from_transform(rotation, Vec3.zero())

        weight = VertexWeight()
        normal = Vec3(1, 0, 0)

        result = DualQuaternionSkinning.skin_normal(normal, [dq], weight)

        # Normal should be rotated
        assert abs(result.x) < 1e-5
        assert abs(result.y - 1.0) < 1e-5


# =============================================================================
# GPU Skinning Data Tests
# =============================================================================

class TestGPUSkinningData:
    """Tests for GPU skinning data preparation."""

    def test_gpu_skinning_data_creation(self):
        """Test creating GPU skinning data."""
        data = GPUSkinningData(
            positions=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 2 vertices
            bone_indices=[0, 0, 0, 0, 0, 0, 0, 0],
            bone_weights=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        )

        assert data.vertex_count == 2

    def test_prepare_gpu_skinning_data_lbs(self):
        """Test preparing data for LBS GPU skinning."""
        skinning_data = SkinningData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0)],
            weights=[VertexWeight(), VertexWeight()],
            bind_pose_matrices=[Mat4.identity()]
        )

        world_transforms = [Mat4.identity()]

        gpu_data = prepare_gpu_skinning_data(
            skinning_data, world_transforms, SkinningMethod.LBS
        )

        assert len(gpu_data.positions) == 6  # 2 vertices * 3 components
        assert len(gpu_data.bone_indices) == 8  # 2 vertices * 4 indices
        assert len(gpu_data.bone_weights) == 8  # 2 vertices * 4 weights
        assert gpu_data.skinning_dual_quaternions is None  # LBS, no DQs

    def test_prepare_gpu_skinning_data_dqs(self):
        """Test preparing data for DQS GPU skinning."""
        skinning_data = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight()],
            bind_pose_matrices=[Mat4.identity()]
        )

        world_transforms = [Mat4.identity()]

        gpu_data = prepare_gpu_skinning_data(
            skinning_data, world_transforms, SkinningMethod.DQS
        )

        # DQS should have dual quaternion data
        assert gpu_data.skinning_dual_quaternions is not None
        assert len(gpu_data.skinning_dual_quaternions) == 8  # 1 bone * 8 components

    def test_prepare_gpu_skinning_data_with_normals(self):
        """Test GPU data includes normals when present."""
        skinning_data = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight()],
            bind_pose_matrices=[Mat4.identity()],
            normals=[Vec3(0, 1, 0)]
        )

        gpu_data = prepare_gpu_skinning_data(
            skinning_data, [Mat4.identity()], SkinningMethod.LBS
        )

        assert gpu_data.normals is not None
        assert len(gpu_data.normals) == 3

    def test_prepare_gpu_skinning_data_with_tangents(self):
        """Test GPU data includes tangents when present."""
        skinning_data = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight()],
            bind_pose_matrices=[Mat4.identity()],
            tangents=[Vec4(1, 0, 0, 1)]
        )

        gpu_data = prepare_gpu_skinning_data(
            skinning_data, [Mat4.identity()], SkinningMethod.LBS
        )

        assert gpu_data.tangents is not None
        assert len(gpu_data.tangents) == 4


# =============================================================================
# skin_mesh Function Tests
# =============================================================================

class TestSkinMesh:
    """Tests for skin_mesh function."""

    def test_skin_mesh_lbs(self):
        """Test skinning mesh with LBS."""
        skinning_data = SkinningData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0)],
            weights=[VertexWeight(), VertexWeight()],
            bind_pose_matrices=[Mat4.identity()]
        )

        # Translate by 5 in X
        world = [Mat4.identity()]
        world[0].m[12] = 5.0

        vertices, normals = skin_mesh(skinning_data, world, SkinningMethod.LBS)

        assert len(vertices) == 2
        assert abs(vertices[0].x - 5.0) < 1e-6
        assert abs(vertices[1].x - 6.0) < 1e-6
        assert normals is None  # No normals provided

    def test_skin_mesh_dqs(self):
        """Test skinning mesh with DQS."""
        skinning_data = SkinningData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0)],
            weights=[VertexWeight(), VertexWeight()],
            bind_pose_matrices=[Mat4.identity()]
        )

        world = [Mat4.identity()]
        world[0].m[12] = 3.0

        vertices, normals = skin_mesh(skinning_data, world, SkinningMethod.DQS)

        assert len(vertices) == 2
        assert abs(vertices[0].x - 3.0) < 1e-5
        assert abs(vertices[1].x - 4.0) < 1e-5

    def test_skin_mesh_with_normals(self):
        """Test skinning mesh with normals."""
        skinning_data = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight()],
            bind_pose_matrices=[Mat4.identity()],
            normals=[Vec3(1, 0, 0)]
        )

        world = [Mat4.identity()]

        vertices, normals = skin_mesh(skinning_data, world, SkinningMethod.LBS)

        assert normals is not None
        assert len(normals) == 1

    def test_skin_mesh_with_cache(self):
        """Test skinning with matrix cache."""
        skinning_data = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight()],
            bind_pose_matrices=[Mat4.identity()]
        )

        cache = SkinningCache()
        world = [Mat4.identity()]

        # First call - cache miss
        vertices1, _ = skin_mesh(skinning_data, world, SkinningMethod.LBS, cache)

        # Second call - should use cache
        vertices2, _ = skin_mesh(skinning_data, world, SkinningMethod.LBS, cache)

        assert abs(vertices1[0].x - vertices2[0].x) < 1e-6


# =============================================================================
# SkinningCache Tests
# =============================================================================

class TestSkinningCache:
    """Tests for SkinningCache class."""

    def test_skinning_cache_creation(self):
        """Test creating skinning cache."""
        cache = SkinningCache()
        assert cache._cached_matrices is None
        assert cache._cached_dqs is None

    def test_skinning_cache_get_matrices(self):
        """Test getting cached skinning matrices."""
        cache = SkinningCache()

        world = [Mat4.identity()]
        world[0].m[12] = 2.0
        bind_inv = [Mat4.identity()]

        matrices = cache.get_skinning_matrices(world, bind_inv)

        assert len(matrices) == 1
        assert abs(matrices[0].m[12] - 2.0) < 1e-6

    def test_skinning_cache_reuses_matrices(self):
        """Test cache reuses matrices when pose unchanged."""
        cache = SkinningCache()

        world = [Mat4.identity()]
        bind_inv = [Mat4.identity()]

        # First call
        matrices1 = cache.get_skinning_matrices(world, bind_inv)

        # Second call with same pose
        matrices2 = cache.get_skinning_matrices(world, bind_inv)

        # Should return cached result
        assert matrices1 is matrices2

    def test_skinning_cache_invalidates_on_change(self):
        """Test cache invalidates when pose changes."""
        cache = SkinningCache()

        world1 = [Mat4.identity()]
        bind_inv = [Mat4.identity()]

        matrices1 = cache.get_skinning_matrices(world1, bind_inv)

        # Change pose
        world2 = [Mat4.identity()]
        world2[0].m[12] = 5.0

        matrices2 = cache.get_skinning_matrices(world2, bind_inv)

        # Should recompute
        assert matrices1 is not matrices2
        assert abs(matrices2[0].m[12] - 5.0) < 1e-6

    def test_skinning_cache_get_dual_quaternions(self):
        """Test getting cached dual quaternions."""
        cache = SkinningCache()

        world = [Mat4.identity()]
        world[0].m[12] = 3.0
        bind_inv = [Mat4.identity()]

        dqs = cache.get_dual_quaternions(world, bind_inv)

        assert len(dqs) == 1
        trans = dqs[0].to_translation()
        assert abs(trans.x - 3.0) < 1e-5

    def test_skinning_cache_invalidate(self):
        """Test manual cache invalidation."""
        cache = SkinningCache()

        world = [Mat4.identity()]
        bind_inv = [Mat4.identity()]

        cache.get_skinning_matrices(world, bind_inv)
        cache.invalidate()

        assert cache._cached_matrices is None
        assert cache._cached_dqs is None
        assert cache._pose_hash is None
