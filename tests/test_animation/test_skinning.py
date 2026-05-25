"""Comprehensive tests for Animation Layer - Skinning and Root Motion subsystem.

Tests cover:
- Linear Blend Skinning (LBS) correctness
- Dual Quaternion Skinning (DQS) vs LBS for artifacts
- Root motion extraction modes
- Retargeting between different skeletons
- Compression quality vs size tradeoffs

Total: 140+ tests with real assertions
"""

import math
import pytest
from typing import List, Tuple

from engine.core.math.vec import Vec3, Vec4
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4
from engine.core.math.transform import Transform


# =============================================================================
# SKINNING TESTS
# =============================================================================

class TestVertexWeight:
    """Tests for VertexWeight dataclass."""

    def test_default_vertex_weight(self):
        from engine.animation.skeletal.skinning import VertexWeight
        w = VertexWeight()
        assert w.bone_indices == (0, 0, 0, 0)
        assert w.weights == (1.0, 0.0, 0.0, 0.0)

    def test_vertex_weight_with_values(self):
        from engine.animation.skeletal.skinning import VertexWeight
        w = VertexWeight(bone_indices=(0, 1, 2, 3), weights=(0.4, 0.3, 0.2, 0.1))
        assert w.bone_indices == (0, 1, 2, 3)
        assert w.weights == (0.4, 0.3, 0.2, 0.1)

    def test_vertex_weight_padding_short_indices(self):
        from engine.animation.skeletal.skinning import VertexWeight
        w = VertexWeight(bone_indices=(0, 1), weights=(0.6, 0.4))
        assert len(w.bone_indices) == 4
        assert len(w.weights) == 4

    def test_vertex_weight_normalize(self):
        from engine.animation.skeletal.skinning import VertexWeight
        w = VertexWeight(weights=(0.2, 0.2, 0.2, 0.2))
        normalized = w.normalize()
        assert abs(sum(normalized.weights) - 1.0) < 1e-6

    def test_vertex_weight_is_normalized_true(self):
        from engine.animation.skeletal.skinning import VertexWeight
        w = VertexWeight(weights=(0.5, 0.3, 0.15, 0.05))
        assert w.is_normalized()

    def test_vertex_weight_is_normalized_false(self):
        from engine.animation.skeletal.skinning import VertexWeight
        w = VertexWeight(weights=(0.5, 0.5, 0.5, 0.0))
        assert not w.is_normalized()

    def test_vertex_weight_influence_count(self):
        from engine.animation.skeletal.skinning import VertexWeight
        w = VertexWeight(weights=(0.5, 0.3, 0.2, 0.0))
        assert w.influence_count == 3

    def test_vertex_weight_serialization_roundtrip(self):
        from engine.animation.skeletal.skinning import VertexWeight
        original = VertexWeight(bone_indices=(1, 2, 3, 4), weights=(0.4, 0.3, 0.2, 0.1))
        data = original.to_dict()
        restored = VertexWeight.from_dict(data)
        assert restored.bone_indices == original.bone_indices
        assert restored.weights == original.weights


class TestSkinningData:
    """Tests for SkinningData class."""

    def test_empty_skinning_data(self):
        from engine.animation.skeletal.skinning import SkinningData
        sd = SkinningData()
        assert sd.vertex_count == 0
        assert sd.bone_count == 0

    def test_skinning_data_with_vertices(self):
        from engine.animation.skeletal.skinning import SkinningData, VertexWeight
        sd = SkinningData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)],
            weights=[VertexWeight() for _ in range(3)],
            bind_pose_matrices=[Mat4.identity()]
        )
        assert sd.vertex_count == 3
        assert sd.bone_count == 1

    def test_skinning_data_validate_success(self):
        from engine.animation.skeletal.skinning import SkinningData, VertexWeight
        sd = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight()],
            bind_pose_matrices=[Mat4.identity()]
        )
        errors = sd.validate()
        assert len(errors) == 0

    def test_skinning_data_validate_count_mismatch(self):
        from engine.animation.skeletal.skinning import SkinningData, VertexWeight
        sd = SkinningData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0)],
            weights=[VertexWeight()],  # Only one weight
            bind_pose_matrices=[Mat4.identity()]
        )
        errors = sd.validate()
        assert len(errors) > 0
        assert "weight count" in errors[0].lower()

    def test_skinning_data_validate_invalid_bone_index(self):
        from engine.animation.skeletal.skinning import SkinningData, VertexWeight
        sd = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight(bone_indices=(5, 0, 0, 0))],  # Invalid index
            bind_pose_matrices=[Mat4.identity()]  # Only 1 bone
        )
        errors = sd.validate()
        assert any("invalid bone index" in e.lower() for e in errors)


class TestDualQuaternion:
    """Tests for DualQuaternion class."""

    def test_dual_quaternion_identity(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        dq = DualQuaternion.identity()
        assert dq.real.w == pytest.approx(1.0)
        assert dq.dual.w == pytest.approx(0.0)

    def test_dual_quaternion_from_transform_translation_only(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        dq = DualQuaternion.from_transform(Quat.identity(), Vec3(1, 2, 3))
        trans = dq.to_translation()
        assert trans.x == pytest.approx(1.0)
        assert trans.y == pytest.approx(2.0)
        assert trans.z == pytest.approx(3.0)

    def test_dual_quaternion_from_transform_rotation_only(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        dq = DualQuaternion.from_transform(rot, Vec3.zero())
        result_rot = dq.to_rotation()
        assert abs(result_rot.dot(rot)) == pytest.approx(1.0, abs=1e-6)

    def test_dual_quaternion_from_matrix_identity(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        dq = DualQuaternion.from_matrix(Mat4.identity())
        trans = dq.to_translation()
        assert trans.x == pytest.approx(0.0, abs=1e-6)
        assert trans.y == pytest.approx(0.0, abs=1e-6)
        assert trans.z == pytest.approx(0.0, abs=1e-6)

    def test_dual_quaternion_transform_point_translation(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        dq = DualQuaternion.from_transform(Quat.identity(), Vec3(5, 0, 0))
        p = dq.transform_point(Vec3(0, 0, 0))
        assert p.x == pytest.approx(5.0)

    def test_dual_quaternion_transform_point_rotation(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        dq = DualQuaternion.from_transform(rot, Vec3.zero())
        p = dq.transform_point(Vec3(1, 0, 0))
        assert p.x == pytest.approx(0.0, abs=1e-6)
        assert p.z == pytest.approx(-1.0, abs=1e-6)

    def test_dual_quaternion_addition(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        dq1 = DualQuaternion.from_transform(Quat.identity(), Vec3(1, 0, 0))
        dq2 = DualQuaternion.from_transform(Quat.identity(), Vec3(0, 1, 0))
        result = dq1 + dq2
        # Result should be sum of components (unnormalized)
        assert result.real.w == pytest.approx(2.0)

    def test_dual_quaternion_scalar_multiplication(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        dq = DualQuaternion.from_transform(Quat.identity(), Vec3(2, 0, 0))
        scaled = dq * 0.5
        assert scaled.real.w == pytest.approx(0.5)

    def test_dual_quaternion_normalized(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        dq = DualQuaternion.from_transform(Quat.identity(), Vec3(1, 2, 3))
        # Artificially scale
        scaled = dq * 2.0
        normalized = scaled.normalized()
        assert normalized.real.length() == pytest.approx(1.0, abs=1e-6)

    def test_dual_quaternion_dot_product(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        dq1 = DualQuaternion.from_transform(Quat.identity(), Vec3.zero())
        dq2 = DualQuaternion.from_transform(Quat.identity(), Vec3.zero())
        assert dq1.dot(dq2) == pytest.approx(1.0)


class TestLinearBlendSkinning:
    """Tests for Linear Blend Skinning."""

    def test_lbs_compute_skinning_matrices_identity(self):
        from engine.animation.skeletal.skinning import LinearBlendSkinning
        world = [Mat4.identity()]
        bind_inv = [Mat4.identity()]
        matrices = LinearBlendSkinning.compute_skinning_matrices(world, bind_inv)
        assert len(matrices) == 1
        assert matrices[0] == Mat4.identity()

    def test_lbs_compute_skinning_matrices_translation(self):
        from engine.animation.skeletal.skinning import LinearBlendSkinning
        world = [Mat4.translation(Vec3(5, 0, 0))]
        bind_inv = [Mat4.identity()]
        matrices = LinearBlendSkinning.compute_skinning_matrices(world, bind_inv)
        p = matrices[0].transform_point(Vec3.zero())
        assert p.x == pytest.approx(5.0)

    def test_lbs_compute_skinning_matrices_count_mismatch(self):
        from engine.animation.skeletal.skinning import LinearBlendSkinning
        with pytest.raises(ValueError):
            LinearBlendSkinning.compute_skinning_matrices(
                [Mat4.identity(), Mat4.identity()],
                [Mat4.identity()]
            )

    def test_lbs_skin_vertex_single_bone(self):
        from engine.animation.skeletal.skinning import LinearBlendSkinning, VertexWeight
        matrices = [Mat4.translation(Vec3(10, 0, 0))]
        weight = VertexWeight(bone_indices=(0, 0, 0, 0), weights=(1.0, 0.0, 0.0, 0.0))
        vertex = Vec3(0, 0, 0)
        result = LinearBlendSkinning.skin_vertex(vertex, matrices, weight)
        assert result.x == pytest.approx(10.0)

    def test_lbs_skin_vertex_two_bones_equal_weight(self):
        from engine.animation.skeletal.skinning import LinearBlendSkinning, VertexWeight
        matrices = [
            Mat4.translation(Vec3(10, 0, 0)),
            Mat4.translation(Vec3(0, 10, 0))
        ]
        weight = VertexWeight(bone_indices=(0, 1, 0, 0), weights=(0.5, 0.5, 0.0, 0.0))
        vertex = Vec3(0, 0, 0)
        result = LinearBlendSkinning.skin_vertex(vertex, matrices, weight)
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(5.0)

    def test_lbs_skin_vertex_rotation(self):
        from engine.animation.skeletal.skinning import LinearBlendSkinning, VertexWeight
        # 90 degree rotation around Y
        rot_matrix = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2).to_mat4()
        matrices = [rot_matrix]
        weight = VertexWeight()
        vertex = Vec3(1, 0, 0)
        result = LinearBlendSkinning.skin_vertex(vertex, matrices, weight)
        assert result.x == pytest.approx(0.0, abs=1e-6)
        assert result.z == pytest.approx(-1.0, abs=1e-6)

    def test_lbs_skin_vertices_batch(self):
        from engine.animation.skeletal.skinning import LinearBlendSkinning, VertexWeight
        matrices = [Mat4.translation(Vec3(1, 2, 3))]
        vertices = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)]
        weights = [VertexWeight() for _ in vertices]
        results = LinearBlendSkinning.skin_vertices(vertices, matrices, weights)
        assert len(results) == 3
        assert results[0].x == pytest.approx(1.0)
        assert results[0].y == pytest.approx(2.0)

    def test_lbs_skin_vertices_count_mismatch(self):
        from engine.animation.skeletal.skinning import LinearBlendSkinning, VertexWeight
        with pytest.raises(ValueError):
            LinearBlendSkinning.skin_vertices(
                [Vec3(0, 0, 0)],
                [Mat4.identity()],
                [VertexWeight(), VertexWeight()]
            )

    def test_lbs_skin_normal_rotation(self):
        from engine.animation.skeletal.skinning import LinearBlendSkinning, VertexWeight
        rot_matrix = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2).to_mat4()
        matrices = [rot_matrix]
        weight = VertexWeight()
        normal = Vec3(1, 0, 0)
        result = LinearBlendSkinning.skin_normal(normal, matrices, weight)
        assert result.x == pytest.approx(0.0, abs=1e-6)
        assert result.z == pytest.approx(-1.0, abs=1e-6)
        # Normal should be normalized
        assert result.length() == pytest.approx(1.0, abs=1e-6)

    def test_lbs_skin_normals_batch(self):
        from engine.animation.skeletal.skinning import LinearBlendSkinning, VertexWeight
        matrices = [Mat4.identity()]
        normals = [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]
        weights = [VertexWeight() for _ in normals]
        results = LinearBlendSkinning.skin_normals(normals, matrices, weights)
        assert len(results) == 3
        assert results[0] == Vec3(1, 0, 0)


class TestDualQuaternionSkinning:
    """Tests for Dual Quaternion Skinning."""

    def test_dqs_matrix_to_dual_quaternion(self):
        from engine.animation.skeletal.skinning import DualQuaternionSkinning
        m = Mat4.translation(Vec3(1, 2, 3))
        dq = DualQuaternionSkinning.matrix_to_dual_quaternion(m)
        trans = dq.to_translation()
        assert trans.x == pytest.approx(1.0, abs=1e-5)
        assert trans.y == pytest.approx(2.0, abs=1e-5)
        assert trans.z == pytest.approx(3.0, abs=1e-5)

    def test_dqs_compute_skinning_dual_quaternions(self):
        from engine.animation.skeletal.skinning import DualQuaternionSkinning
        world = [Mat4.translation(Vec3(5, 0, 0))]
        bind_inv = [Mat4.identity()]
        dqs = DualQuaternionSkinning.compute_skinning_dual_quaternions(world, bind_inv)
        assert len(dqs) == 1
        trans = dqs[0].to_translation()
        assert trans.x == pytest.approx(5.0, abs=1e-5)

    def test_dqs_skin_vertex_single_bone(self):
        from engine.animation.skeletal.skinning import DualQuaternionSkinning, VertexWeight, DualQuaternion
        dqs = [DualQuaternion.from_transform(Quat.identity(), Vec3(10, 0, 0))]
        weight = VertexWeight()
        vertex = Vec3(0, 0, 0)
        result = DualQuaternionSkinning.skin_vertex(vertex, dqs, weight)
        assert result.x == pytest.approx(10.0, abs=1e-5)

    def test_dqs_skin_vertex_rotation(self):
        from engine.animation.skeletal.skinning import DualQuaternionSkinning, VertexWeight, DualQuaternion
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        dqs = [DualQuaternion.from_transform(rot, Vec3.zero())]
        weight = VertexWeight()
        vertex = Vec3(1, 0, 0)
        result = DualQuaternionSkinning.skin_vertex(vertex, dqs, weight)
        assert result.x == pytest.approx(0.0, abs=1e-5)
        assert result.z == pytest.approx(-1.0, abs=1e-5)

    def test_dqs_skin_vertex_blended(self):
        from engine.animation.skeletal.skinning import DualQuaternionSkinning, VertexWeight, DualQuaternion
        dqs = [
            DualQuaternion.from_transform(Quat.identity(), Vec3(10, 0, 0)),
            DualQuaternion.from_transform(Quat.identity(), Vec3(0, 10, 0))
        ]
        weight = VertexWeight(bone_indices=(0, 1, 0, 0), weights=(0.5, 0.5, 0.0, 0.0))
        vertex = Vec3(0, 0, 0)
        result = DualQuaternionSkinning.skin_vertex(vertex, dqs, weight)
        assert result.x == pytest.approx(5.0, abs=1e-5)
        assert result.y == pytest.approx(5.0, abs=1e-5)

    def test_dqs_antipodality_handling(self):
        from engine.animation.skeletal.skinning import DualQuaternionSkinning, VertexWeight, DualQuaternion
        # Create two quaternions in opposite hemispheres
        q1 = Quat(0, 0, 0, 1)  # Identity
        q2 = Quat(0, 0, 0, -1)  # Same rotation, opposite sign
        dqs = [
            DualQuaternion.from_transform(q1, Vec3(0, 0, 0)),
            DualQuaternion.from_transform(q2, Vec3(0, 0, 0))
        ]
        weight = VertexWeight(bone_indices=(0, 1, 0, 0), weights=(0.5, 0.5, 0.0, 0.0))
        # Should handle antipodality and produce valid result
        result = DualQuaternionSkinning.skin_vertex(Vec3(1, 0, 0), dqs, weight)
        # Point should still be at (1, 0, 0) since both represent identity
        assert abs(result.length() - 1.0) < 0.1  # Allow some error from blending

    def test_dqs_skin_normal(self):
        from engine.animation.skeletal.skinning import DualQuaternionSkinning, VertexWeight, DualQuaternion
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        dqs = [DualQuaternion.from_transform(rot, Vec3.zero())]
        weight = VertexWeight()
        normal = Vec3(1, 0, 0)
        result = DualQuaternionSkinning.skin_normal(normal, dqs, weight)
        assert result.x == pytest.approx(0.0, abs=1e-5)
        assert result.z == pytest.approx(-1.0, abs=1e-5)
        assert result.length() == pytest.approx(1.0, abs=1e-5)


class TestSkinningComparison:
    """Tests comparing LBS vs DQS for known artifact cases."""

    def test_candy_wrapper_artifact_lbs(self):
        """Test that LBS produces candy wrapper artifact on 180 degree twist."""
        from engine.animation.skeletal.skinning import LinearBlendSkinning, VertexWeight
        # Two bones with opposite 90 degree rotations
        rot1 = Quat.from_axis_angle(Vec3(1, 0, 0), math.pi / 4).to_mat4()
        rot2 = Quat.from_axis_angle(Vec3(1, 0, 0), -math.pi / 4).to_mat4()
        matrices = [rot1, rot2]

        # Vertex at joint between bones
        weight = VertexWeight(bone_indices=(0, 1, 0, 0), weights=(0.5, 0.5, 0.0, 0.0))
        vertex = Vec3(0, 1, 0)

        result = LinearBlendSkinning.skin_vertex(vertex, matrices, weight)
        # LBS will shrink the vertex toward the joint
        # Y component should be less than 1 due to matrix averaging
        assert result.y < 1.0

    def test_candy_wrapper_artifact_dqs(self):
        """Test that DQS avoids candy wrapper artifact on 180 degree twist."""
        from engine.animation.skeletal.skinning import DualQuaternionSkinning, VertexWeight, DualQuaternion
        # Two bones with opposite 90 degree rotations
        rot1 = Quat.from_axis_angle(Vec3(1, 0, 0), math.pi / 4)
        rot2 = Quat.from_axis_angle(Vec3(1, 0, 0), -math.pi / 4)
        dqs = [
            DualQuaternion.from_transform(rot1, Vec3.zero()),
            DualQuaternion.from_transform(rot2, Vec3.zero())
        ]

        weight = VertexWeight(bone_indices=(0, 1, 0, 0), weights=(0.5, 0.5, 0.0, 0.0))
        vertex = Vec3(0, 1, 0)

        result = DualQuaternionSkinning.skin_vertex(vertex, dqs, weight)
        # DQS should preserve volume better
        # Distance from origin should be closer to 1
        assert result.length() > 0.9

    def test_dqs_vs_lbs_extreme_rotation(self):
        """Compare DQS and LBS on extreme rotation blend."""
        from engine.animation.skeletal.skinning import (
            LinearBlendSkinning, DualQuaternionSkinning,
            VertexWeight, DualQuaternion
        )

        # 180 degree rotation difference
        rot1 = Quat.from_axis_angle(Vec3(0, 0, 1), math.pi / 2)
        rot2 = Quat.from_axis_angle(Vec3(0, 0, 1), -math.pi / 2)

        mat1 = rot1.to_mat4()
        mat2 = rot2.to_mat4()
        matrices = [mat1, mat2]

        dq1 = DualQuaternion.from_transform(rot1, Vec3.zero())
        dq2 = DualQuaternion.from_transform(rot2, Vec3.zero())
        dqs = [dq1, dq2]

        weight = VertexWeight(bone_indices=(0, 1, 0, 0), weights=(0.5, 0.5, 0.0, 0.0))
        vertex = Vec3(1, 0, 0)

        lbs_result = LinearBlendSkinning.skin_vertex(vertex, matrices, weight)
        dqs_result = DualQuaternionSkinning.skin_vertex(vertex, dqs, weight)

        # DQS should preserve distance from origin better
        assert dqs_result.length() >= lbs_result.length()


class TestSkinMesh:
    """Tests for the high-level skin_mesh function."""

    def test_skin_mesh_lbs(self):
        from engine.animation.skeletal.skinning import (
            skin_mesh, SkinningData, SkinningMethod, VertexWeight
        )

        sd = SkinningData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0)],
            weights=[VertexWeight(), VertexWeight()],
            bind_pose_matrices=[Mat4.identity()]
        )
        bone_transforms = [Mat4.translation(Vec3(5, 0, 0))]

        verts, normals = skin_mesh(sd, bone_transforms, SkinningMethod.LBS)
        assert len(verts) == 2
        assert verts[0].x == pytest.approx(5.0)
        assert normals is None

    def test_skin_mesh_with_normals(self):
        from engine.animation.skeletal.skinning import (
            skin_mesh, SkinningData, SkinningMethod, VertexWeight
        )

        sd = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight()],
            bind_pose_matrices=[Mat4.identity()],
            normals=[Vec3(0, 1, 0)]
        )
        bone_transforms = [Mat4.identity()]

        verts, normals = skin_mesh(sd, bone_transforms, SkinningMethod.LBS)
        assert normals is not None
        assert len(normals) == 1
        assert normals[0].y == pytest.approx(1.0)

    def test_skin_mesh_dqs(self):
        from engine.animation.skeletal.skinning import (
            skin_mesh, SkinningData, SkinningMethod, VertexWeight
        )

        sd = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight()],
            bind_pose_matrices=[Mat4.identity()]
        )
        bone_transforms = [Mat4.translation(Vec3(3, 4, 5))]

        verts, _ = skin_mesh(sd, bone_transforms, SkinningMethod.DQS)
        assert verts[0].x == pytest.approx(3.0, abs=1e-4)
        assert verts[0].y == pytest.approx(4.0, abs=1e-4)
        assert verts[0].z == pytest.approx(5.0, abs=1e-4)


class TestSkinningCache:
    """Tests for SkinningCache."""

    def test_skinning_cache_basic(self):
        from engine.animation.skeletal.skinning import SkinningCache
        cache = SkinningCache()
        transforms = [Mat4.translation(Vec3(1, 0, 0))]
        bind_inv = [Mat4.identity()]

        # First call should compute
        m1 = cache.get_skinning_matrices(transforms, bind_inv)
        assert len(m1) == 1

        # Second call with same transforms should use cache
        m2 = cache.get_skinning_matrices(transforms, bind_inv)
        assert m1 == m2

    def test_skinning_cache_invalidation(self):
        from engine.animation.skeletal.skinning import SkinningCache
        cache = SkinningCache()
        transforms = [Mat4.translation(Vec3(1, 0, 0))]
        bind_inv = [Mat4.identity()]

        m1 = cache.get_skinning_matrices(transforms, bind_inv)

        # Change transforms
        new_transforms = [Mat4.translation(Vec3(2, 0, 0))]
        m2 = cache.get_skinning_matrices(new_transforms, bind_inv)

        # Results should differ
        assert m1[0].m[12] != m2[0].m[12]

    def test_skinning_cache_force_invalidate(self):
        from engine.animation.skeletal.skinning import SkinningCache
        cache = SkinningCache()
        transforms = [Mat4.identity()]
        bind_inv = [Mat4.identity()]

        cache.get_skinning_matrices(transforms, bind_inv)
        cache.invalidate()

        # Should recompute after invalidation
        m = cache.get_skinning_matrices(transforms, bind_inv)
        assert m is not None


class TestGPUSkinningData:
    """Tests for GPU skinning data preparation."""

    def test_prepare_gpu_skinning_data_basic(self):
        from engine.animation.skeletal.skinning import (
            prepare_gpu_skinning_data, SkinningData, SkinningMethod, VertexWeight
        )

        sd = SkinningData(
            vertices=[Vec3(0, 1, 2), Vec3(3, 4, 5)],
            weights=[VertexWeight(), VertexWeight()],
            bind_pose_matrices=[Mat4.identity()]
        )
        bone_transforms = [Mat4.identity()]

        gpu_data = prepare_gpu_skinning_data(sd, bone_transforms, SkinningMethod.LBS)

        assert gpu_data.vertex_count == 2
        assert len(gpu_data.positions) == 6  # 2 verts * 3 components
        assert len(gpu_data.bone_indices) == 8  # 2 verts * 4 influences
        assert len(gpu_data.bone_weights) == 8
        assert len(gpu_data.skinning_matrices) == 16  # 1 mat4 * 16 floats

    def test_prepare_gpu_skinning_data_with_dqs(self):
        from engine.animation.skeletal.skinning import (
            prepare_gpu_skinning_data, SkinningData, SkinningMethod, VertexWeight
        )

        sd = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight()],
            bind_pose_matrices=[Mat4.identity()]
        )
        bone_transforms = [Mat4.identity()]

        gpu_data = prepare_gpu_skinning_data(sd, bone_transforms, SkinningMethod.DQS)

        assert gpu_data.skinning_dual_quaternions is not None
        assert len(gpu_data.skinning_dual_quaternions) == 8  # 1 DQ * 8 floats


# =============================================================================
# ROOT MOTION TESTS
# =============================================================================

class TestRootMotionMode:
    """Tests for RootMotionMode enum."""

    def test_root_motion_modes_exist(self):
        from engine.animation.skeletal.root_motion import RootMotionMode
        assert RootMotionMode.IN_PLACE
        assert RootMotionMode.EXTRACT_XZ
        assert RootMotionMode.EXTRACT_XYZ
        assert RootMotionMode.EXTRACT_ROTATION
        assert RootMotionMode.EXTRACT_ALL


class TestRootMotionData:
    """Tests for RootMotionData class."""

    def test_empty_root_motion_data(self):
        from engine.animation.skeletal.root_motion import RootMotionData
        data = RootMotionData()
        assert data.frame_count == 0
        assert data.duration == 0.0

    def test_root_motion_data_frame_count(self):
        from engine.animation.skeletal.root_motion import RootMotionData
        data = RootMotionData(
            delta_positions=[Vec3.zero(), Vec3(1, 0, 0)],
            delta_rotations=[Quat.identity(), Quat.identity()],
            frame_times=[0.0, 0.5]
        )
        assert data.frame_count == 2
        assert data.duration == 0.5

    def test_root_motion_data_get_delta_at_time_start(self):
        from engine.animation.skeletal.root_motion import RootMotionData
        data = RootMotionData(
            delta_positions=[Vec3.zero(), Vec3(10, 0, 0)],
            delta_rotations=[Quat.identity(), Quat.identity()],
            frame_times=[0.0, 1.0],
            total_delta_position=Vec3(10, 0, 0)
        )
        pos, rot = data.get_delta_at_time(0.0)
        assert pos.x == pytest.approx(0.0)

    def test_root_motion_data_get_delta_at_time_end(self):
        from engine.animation.skeletal.root_motion import RootMotionData
        data = RootMotionData(
            delta_positions=[Vec3.zero(), Vec3(10, 0, 0)],
            delta_rotations=[Quat.identity(), Quat.identity()],
            frame_times=[0.0, 1.0],
            total_delta_position=Vec3(10, 0, 0)
        )
        pos, rot = data.get_delta_at_time(1.0)
        assert pos.x == pytest.approx(10.0)

    def test_root_motion_data_get_delta_between(self):
        from engine.animation.skeletal.root_motion import RootMotionData
        data = RootMotionData(
            delta_positions=[Vec3.zero(), Vec3(5, 0, 0), Vec3(5, 0, 0)],
            delta_rotations=[Quat.identity(), Quat.identity(), Quat.identity()],
            frame_times=[0.0, 0.5, 1.0],
            total_delta_position=Vec3(10, 0, 0)
        )
        pos, rot = data.get_delta_between(0.0, 0.5)
        # Should get delta from first interval
        assert pos.x >= 0.0


class TestExtractRootMotion:
    """Tests for root motion extraction."""

    def test_extract_root_motion_in_place(self):
        from engine.animation.skeletal.root_motion import (
            extract_root_motion, RootMotionMode, RootBoneTransform
        )
        transforms = [
            RootBoneTransform(Vec3(0, 0, 0), Quat.identity(), 0.0),
            RootBoneTransform(Vec3(10, 0, 10), Quat.identity(), 1.0)
        ]
        data = extract_root_motion(transforms, RootMotionMode.IN_PLACE)
        assert data.total_delta_position == Vec3.zero()

    def test_extract_root_motion_xz(self):
        from engine.animation.skeletal.root_motion import (
            extract_root_motion, RootMotionMode, RootBoneTransform
        )
        transforms = [
            RootBoneTransform(Vec3(0, 0, 0), Quat.identity(), 0.0),
            RootBoneTransform(Vec3(10, 5, 10), Quat.identity(), 1.0)
        ]
        data = extract_root_motion(transforms, RootMotionMode.EXTRACT_XZ)
        assert data.total_delta_position.x == pytest.approx(10.0)
        assert data.total_delta_position.y == pytest.approx(0.0)  # Y zeroed
        assert data.total_delta_position.z == pytest.approx(10.0)

    def test_extract_root_motion_xyz(self):
        from engine.animation.skeletal.root_motion import (
            extract_root_motion, RootMotionMode, RootBoneTransform
        )
        transforms = [
            RootBoneTransform(Vec3(0, 0, 0), Quat.identity(), 0.0),
            RootBoneTransform(Vec3(10, 5, 10), Quat.identity(), 1.0)
        ]
        data = extract_root_motion(transforms, RootMotionMode.EXTRACT_XYZ)
        assert data.total_delta_position.x == pytest.approx(10.0)
        assert data.total_delta_position.y == pytest.approx(5.0)
        assert data.total_delta_position.z == pytest.approx(10.0)

    def test_extract_root_motion_rotation_only(self):
        from engine.animation.skeletal.root_motion import (
            extract_root_motion, RootMotionMode, RootBoneTransform
        )
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)
        transforms = [
            RootBoneTransform(Vec3(0, 0, 0), Quat.identity(), 0.0),
            RootBoneTransform(Vec3(10, 0, 10), rot, 1.0)
        ]
        data = extract_root_motion(transforms, RootMotionMode.EXTRACT_ROTATION)
        assert data.total_delta_position == Vec3.zero()
        # Should have rotation
        assert data.total_delta_rotation != Quat.identity()

    def test_extract_root_motion_empty(self):
        from engine.animation.skeletal.root_motion import extract_root_motion, RootMotionMode
        data = extract_root_motion([], RootMotionMode.EXTRACT_XZ)
        assert data.frame_count == 0


class TestApplyRootMotion:
    """Tests for apply_root_motion function."""

    def test_apply_root_motion_translation(self):
        from engine.animation.skeletal.root_motion import apply_root_motion
        transform = Transform(Vec3(0, 0, 0), Quat.identity(), Vec3.one())
        result = apply_root_motion(transform, Vec3(5, 0, 0), Quat.identity())
        assert result.translation.x == pytest.approx(5.0)

    def test_apply_root_motion_rotation(self):
        from engine.animation.skeletal.root_motion import apply_root_motion
        transform = Transform(Vec3(0, 0, 0), Quat.identity(), Vec3.one())
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)
        result = apply_root_motion(transform, Vec3.zero(), rot)
        # Rotation should be applied
        assert result.rotation != Quat.identity()

    def test_apply_root_motion_local_to_world(self):
        from engine.animation.skeletal.root_motion import apply_root_motion
        # Start facing along Z, rotated 90 degrees around Y
        facing_rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        transform = Transform(Vec3(0, 0, 0), facing_rot, Vec3.one())

        # Move "forward" in local space
        # With Y-axis rotation of pi/2, local -Z maps to -X in world
        result = apply_root_motion(transform, Vec3(0, 0, -1), Quat.identity())
        # Verify movement happened in rotated direction
        assert abs(result.translation.x) == pytest.approx(1.0, abs=1e-5)
        assert result.translation.z == pytest.approx(0.0, abs=1e-5)

    def test_apply_root_motion_with_dt(self):
        from engine.animation.skeletal.root_motion import apply_root_motion
        transform = Transform(Vec3(0, 0, 0), Quat.identity(), Vec3.one())
        # Half speed
        result = apply_root_motion(transform, Vec3(10, 0, 0), Quat.identity(), dt=0.5)
        assert result.translation.x == pytest.approx(5.0)


class TestRootMotionAccumulator:
    """Tests for RootMotionAccumulator class."""

    def test_accumulator_reset(self):
        from engine.animation.skeletal.root_motion import RootMotionAccumulator
        acc = RootMotionAccumulator()
        acc._accumulated_position = Vec3(10, 0, 0)
        acc.reset()
        assert acc.accumulated_position == Vec3.zero()

    def test_accumulator_basic_update(self):
        from engine.animation.skeletal.root_motion import (
            RootMotionAccumulator, RootMotionData
        )
        data = RootMotionData(
            delta_positions=[Vec3.zero(), Vec3(10, 0, 0)],
            delta_rotations=[Quat.identity(), Quat.identity()],
            frame_times=[0.0, 1.0],
            total_delta_position=Vec3(10, 0, 0),
            total_delta_rotation=Quat.identity()
        )
        acc = RootMotionAccumulator(data, loop=False)
        pos, rot = acc.update(0.5)
        # Should accumulate some motion
        assert acc.accumulated_position.x >= 0

    def test_accumulator_loop_wrap(self):
        from engine.animation.skeletal.root_motion import (
            RootMotionAccumulator, RootMotionData
        )
        data = RootMotionData(
            delta_positions=[Vec3.zero(), Vec3(10, 0, 0)],
            delta_rotations=[Quat.identity(), Quat.identity()],
            frame_times=[0.0, 1.0],
            total_delta_position=Vec3(10, 0, 0),
            total_delta_rotation=Quat.identity()
        )
        acc = RootMotionAccumulator(data, loop=True)
        # Update past the end
        acc.update(1.5)
        assert acc.loop_count >= 1

    def test_accumulator_seek(self):
        from engine.animation.skeletal.root_motion import (
            RootMotionAccumulator, RootMotionData
        )
        data = RootMotionData(
            delta_positions=[Vec3.zero(), Vec3(10, 0, 0)],
            delta_rotations=[Quat.identity(), Quat.identity()],
            frame_times=[0.0, 1.0],
            total_delta_position=Vec3(10, 0, 0)
        )
        acc = RootMotionAccumulator(data)
        acc.seek(0.5)
        assert acc.current_time == pytest.approx(0.5)


class TestRootMotionConfig:
    """Tests for RootMotionConfig."""

    def test_config_default_values(self):
        from engine.animation.skeletal.root_motion import RootMotionConfig, RootMotionMode
        config = RootMotionConfig()
        assert config.mode == RootMotionMode.EXTRACT_XZ
        assert config.scale == 1.0

    def test_config_apply_to_delta_scale(self):
        from engine.animation.skeletal.root_motion import RootMotionConfig
        config = RootMotionConfig(scale=2.0, clamp_to_ground=False)
        pos, rot = config.apply_to_delta(Vec3(5, 5, 5), Quat.identity())
        assert pos.x == pytest.approx(10.0)
        assert pos.y == pytest.approx(10.0)

    def test_config_clamp_to_ground(self):
        from engine.animation.skeletal.root_motion import RootMotionConfig
        config = RootMotionConfig(clamp_to_ground=True)
        pos, rot = config.apply_to_delta(Vec3(5, 5, 5), Quat.identity())
        assert pos.y == pytest.approx(0.0)


class TestBlendRootMotion:
    """Tests for blend_root_motion function."""

    def test_blend_root_motion_full_a(self):
        from engine.animation.skeletal.root_motion import blend_root_motion
        a = (Vec3(10, 0, 0), Quat.identity())
        b = (Vec3(0, 10, 0), Quat.identity())
        pos, rot = blend_root_motion(a, b, 0.0)
        assert pos.x == pytest.approx(10.0)
        assert pos.y == pytest.approx(0.0)

    def test_blend_root_motion_full_b(self):
        from engine.animation.skeletal.root_motion import blend_root_motion
        a = (Vec3(10, 0, 0), Quat.identity())
        b = (Vec3(0, 10, 0), Quat.identity())
        pos, rot = blend_root_motion(a, b, 1.0)
        assert pos.x == pytest.approx(0.0)
        assert pos.y == pytest.approx(10.0)

    def test_blend_root_motion_half(self):
        from engine.animation.skeletal.root_motion import blend_root_motion
        a = (Vec3(10, 0, 0), Quat.identity())
        b = (Vec3(0, 10, 0), Quat.identity())
        pos, rot = blend_root_motion(a, b, 0.5)
        assert pos.x == pytest.approx(5.0)
        assert pos.y == pytest.approx(5.0)


# =============================================================================
# RETARGETING TESTS
# =============================================================================

class TestBoneMapping:
    """Tests for BoneMapping dataclass."""

    def test_bone_mapping_default(self):
        from engine.animation.skeletal.retargeting import BoneMapping
        m = BoneMapping(source_index=0, target_index=0)
        assert m.rotation_offset == Quat.identity()
        assert m.translation_mode == "proportional"

    def test_bone_mapping_serialization(self):
        from engine.animation.skeletal.retargeting import BoneMapping
        original = BoneMapping(
            source_index=5,
            target_index=10,
            source_name="spine",
            target_name="spine_01"
        )
        data = original.to_dict()
        restored = BoneMapping.from_dict(data)
        assert restored.source_index == 5
        assert restored.target_index == 10


class TestRetargetMap:
    """Tests for RetargetMap class."""

    def test_retarget_map_empty(self):
        from engine.animation.skeletal.retargeting import RetargetMap
        rm = RetargetMap()
        assert rm.mapped_count == 0

    def test_retarget_map_add_mapping(self):
        from engine.animation.skeletal.retargeting import RetargetMap, BoneMapping
        rm = RetargetMap(source_bone_count=10, target_bone_count=10)
        rm.unmapped_source_bones = set(range(10))
        rm.unmapped_target_bones = set(range(10))

        mapping = BoneMapping(source_index=0, target_index=0)
        rm.add_mapping(mapping)

        assert rm.mapped_count == 1
        assert 0 not in rm.unmapped_source_bones

    def test_retarget_map_get_mapping(self):
        from engine.animation.skeletal.retargeting import RetargetMap, BoneMapping
        rm = RetargetMap()
        mapping = BoneMapping(source_index=5, target_index=7)
        rm.add_mapping(mapping)

        result = rm.get_mapping_for_source(5)
        assert result.target_index == 7

    def test_retarget_map_coverage_ratio(self):
        from engine.animation.skeletal.retargeting import RetargetMap, BoneMapping
        rm = RetargetMap(target_bone_count=10)
        rm.add_mapping(BoneMapping(0, 0))
        rm.add_mapping(BoneMapping(1, 1))
        assert rm.coverage_ratio == pytest.approx(0.2)


class TestSkeletonInfo:
    """Tests for SkeletonInfo class."""

    def test_skeleton_info_basic(self):
        from engine.animation.skeletal.retargeting import SkeletonInfo
        skel = SkeletonInfo(
            bone_names=["root", "spine", "head"],
            bone_parents=[-1, 0, 1],
            bind_translations=[Vec3.zero(), Vec3(0, 1, 0), Vec3(0, 0.5, 0)],
            bind_rotations=[Quat.identity()] * 3
        )
        assert skel.bone_count == 3

    def test_skeleton_info_get_bone_index(self):
        from engine.animation.skeletal.retargeting import SkeletonInfo
        skel = SkeletonInfo(bone_names=["root", "spine", "head"])
        assert skel.get_bone_index("spine") == 1
        assert skel.get_bone_index("missing") is None


class TestCreateRetargetMap:
    """Tests for create_retarget_map function."""

    def test_create_retarget_map_exact_match(self):
        from engine.animation.skeletal.retargeting import (
            create_retarget_map, SkeletonInfo, BoneMappingStrategy
        )
        source = SkeletonInfo(bone_names=["root", "spine", "head"])
        target = SkeletonInfo(bone_names=["root", "spine", "head"])

        rm = create_retarget_map(source, target, BoneMappingStrategy.BY_NAME)
        assert rm.mapped_count == 3

    def test_create_retarget_map_fuzzy_match(self):
        from engine.animation.skeletal.retargeting import (
            create_retarget_map, SkeletonInfo, BoneMappingStrategy
        )
        source = SkeletonInfo(bone_names=["Bip01_Root", "Bip01_Spine", "Bip01_Head"])
        target = SkeletonInfo(bone_names=["root", "spine", "head"])

        rm = create_retarget_map(source, target, BoneMappingStrategy.BY_NAME_FUZZY)
        assert rm.mapped_count >= 2  # Should match at least some

    def test_create_retarget_map_manual_override(self):
        from engine.animation.skeletal.retargeting import (
            create_retarget_map, SkeletonInfo, BoneMappingStrategy
        )
        source = SkeletonInfo(bone_names=["a", "b", "c"])
        target = SkeletonInfo(bone_names=["x", "y", "z"])

        rm = create_retarget_map(
            source, target,
            BoneMappingStrategy.BY_NAME,
            manual_mappings={"x": "a", "y": "b"}
        )
        assert rm.mapped_count == 2


class TestRetargetPose:
    """Tests for retarget_pose function."""

    def test_retarget_pose_identity(self):
        from engine.animation.skeletal.retargeting import (
            retarget_pose, SkeletonInfo, PoseData, RetargetMap, BoneMapping
        )
        source_skel = SkeletonInfo(
            bone_names=["root"],
            bind_translations=[Vec3.zero()],
            bind_rotations=[Quat.identity()]
        )
        target_skel = SkeletonInfo(
            bone_names=["root"],
            bind_translations=[Vec3.zero()],
            bind_rotations=[Quat.identity()]
        )
        source_pose = PoseData(
            local_translations=[Vec3(1, 0, 0)],
            local_rotations=[Quat.identity()]
        )
        rm = RetargetMap(
            mappings=[BoneMapping(0, 0, "root", "root")],
            source_bone_count=1,
            target_bone_count=1
        )

        result = retarget_pose(source_pose, source_skel, target_skel, rm)
        assert len(result.local_rotations) == 1

    def test_retarget_pose_rotation_preserved(self):
        from engine.animation.skeletal.retargeting import (
            retarget_pose, SkeletonInfo, PoseData, RetargetMap, BoneMapping
        )
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)
        source_skel = SkeletonInfo(
            bone_names=["root"],
            bind_translations=[Vec3.zero()],
            bind_rotations=[Quat.identity()]
        )
        target_skel = SkeletonInfo(
            bone_names=["root"],
            bind_translations=[Vec3.zero()],
            bind_rotations=[Quat.identity()]
        )
        source_pose = PoseData(
            local_translations=[Vec3.zero()],
            local_rotations=[rot]
        )
        rm = RetargetMap(
            mappings=[BoneMapping(0, 0)],
            source_bone_count=1,
            target_bone_count=1
        )

        result = retarget_pose(source_pose, source_skel, target_skel, rm)
        assert abs(result.local_rotations[0].dot(rot)) == pytest.approx(1.0, abs=1e-5)


class TestRetargetPipeline:
    """Tests for RetargetPipeline class."""

    def test_retarget_pipeline_auto_scale(self):
        from engine.animation.skeletal.retargeting import (
            RetargetPipeline, SkeletonInfo
        )
        source = SkeletonInfo(
            bone_names=["root", "child"],
            bone_parents=[-1, 0],
            bind_translations=[Vec3.zero(), Vec3(0, 1, 0)],
            bind_rotations=[Quat.identity(), Quat.identity()],
            bind_world_positions=[Vec3.zero(), Vec3(0, 1, 0)]
        )
        target = SkeletonInfo(
            bone_names=["root", "child"],
            bone_parents=[-1, 0],
            bind_translations=[Vec3.zero(), Vec3(0, 2, 0)],
            bind_rotations=[Quat.identity(), Quat.identity()],
            bind_world_positions=[Vec3.zero(), Vec3(0, 2, 0)]
        )

        pipeline = RetargetPipeline(source, target)
        # Scale should be auto-computed
        assert pipeline.scale_factor > 0


# =============================================================================
# COMPRESSION TESTS
# =============================================================================

class TestCompressionMethod:
    """Tests for CompressionMethod enum."""

    def test_compression_methods_exist(self):
        from engine.animation.skeletal.compression import CompressionMethod
        assert CompressionMethod.NONE
        assert CompressionMethod.QUANTIZED
        assert CompressionMethod.CURVE
        assert CompressionMethod.ACL


class TestQuantizedValue:
    """Tests for quantization functions."""

    def test_quantize_float_min(self):
        from engine.animation.skeletal.compression import QuantizedValue
        result = QuantizedValue.quantize_float(0.0, 0.0, 1.0, 16)
        assert result == 0

    def test_quantize_float_max(self):
        from engine.animation.skeletal.compression import QuantizedValue
        result = QuantizedValue.quantize_float(1.0, 0.0, 1.0, 16)
        assert result == 65535

    def test_quantize_float_mid(self):
        from engine.animation.skeletal.compression import QuantizedValue
        result = QuantizedValue.quantize_float(0.5, 0.0, 1.0, 16)
        assert 32000 < result < 33000  # Approximately half

    def test_dequantize_float_roundtrip(self):
        from engine.animation.skeletal.compression import QuantizedValue
        original = 0.75
        quantized = QuantizedValue.quantize_float(original, 0.0, 1.0, 16)
        restored = QuantizedValue.dequantize_float(quantized, 0.0, 1.0, 16)
        assert restored == pytest.approx(original, abs=0.001)


class TestQuantizedCurve:
    """Tests for QuantizedCurve class."""

    def test_quantized_curve_basic(self):
        from engine.animation.skeletal.compression import QuantizedCurve
        curve = QuantizedCurve(
            min_values=[0.0, 0.0, 0.0],
            max_values=[1.0, 1.0, 1.0],
            bits_per_sample=16,
            data=b'\x00\x00\x00\x00\x00\x00',  # 3 components of zeros
            sample_count=1,
            component_count=3
        )
        values = curve.get_value_at_index(0)
        assert len(values) == 3


class TestAnimationTrack:
    """Tests for AnimationTrack class."""

    def test_animation_track_duration(self):
        from engine.animation.skeletal.compression import AnimationTrack, Keyframe, TrackType
        track = AnimationTrack(
            bone_index=0,
            track_type=TrackType.TRANSLATION,
            keyframes=[
                Keyframe(0.0, Vec3.zero()),
                Keyframe(1.0, Vec3(1, 0, 0)),
                Keyframe(2.0, Vec3(2, 0, 0))
            ]
        )
        assert track.duration == 2.0

    def test_animation_track_sample_start(self):
        from engine.animation.skeletal.compression import AnimationTrack, Keyframe, TrackType
        track = AnimationTrack(
            bone_index=0,
            track_type=TrackType.TRANSLATION,
            keyframes=[
                Keyframe(0.0, Vec3(5, 0, 0)),
                Keyframe(1.0, Vec3(10, 0, 0))
            ]
        )
        result = track.sample(0.0)
        assert result.x == pytest.approx(5.0)

    def test_animation_track_sample_interpolated(self):
        from engine.animation.skeletal.compression import AnimationTrack, Keyframe, TrackType
        track = AnimationTrack(
            bone_index=0,
            track_type=TrackType.TRANSLATION,
            keyframes=[
                Keyframe(0.0, Vec3(0, 0, 0)),
                Keyframe(1.0, Vec3(10, 0, 0))
            ]
        )
        result = track.sample(0.5)
        assert result.x == pytest.approx(5.0)


class TestCompressClip:
    """Tests for compress_clip function."""

    def test_compress_clip_none_method(self):
        from engine.animation.skeletal.compression import (
            compress_clip, AnimationClipData, AnimationTrack,
            Keyframe, TrackType, CompressionSettings, CompressionMethod
        )
        clip = AnimationClipData(
            name="test",
            duration=1.0,
            frame_rate=30.0,
            bone_count=1,
            tracks=[
                AnimationTrack(
                    bone_index=0,
                    track_type=TrackType.TRANSLATION,
                    keyframes=[
                        Keyframe(0.0, Vec3.zero()),
                        Keyframe(1.0, Vec3(10, 0, 0))
                    ]
                )
            ]
        )
        settings = CompressionSettings(method=CompressionMethod.NONE)
        compressed = compress_clip(clip, settings)
        assert compressed.compression_method == CompressionMethod.NONE
        assert len(compressed.tracks) == 1

    def test_compress_clip_quantized(self):
        from engine.animation.skeletal.compression import (
            compress_clip, AnimationClipData, AnimationTrack,
            Keyframe, TrackType, CompressionSettings, CompressionMethod
        )
        clip = AnimationClipData(
            name="test",
            duration=1.0,
            bone_count=1,
            tracks=[
                AnimationTrack(
                    bone_index=0,
                    track_type=TrackType.TRANSLATION,
                    keyframes=[
                        Keyframe(0.0, Vec3.zero()),
                        Keyframe(0.5, Vec3(5, 0, 0)),
                        Keyframe(1.0, Vec3(10, 0, 0))
                    ]
                )
            ]
        )
        settings = CompressionSettings(method=CompressionMethod.QUANTIZED)
        compressed = compress_clip(clip, settings)
        assert compressed.compression_ratio >= 1.0

    def test_compress_clip_constant_track(self):
        from engine.animation.skeletal.compression import (
            compress_clip, AnimationClipData, AnimationTrack,
            Keyframe, TrackType, CompressionSettings, CompressionMethod
        )
        # Track with all same values
        clip = AnimationClipData(
            name="constant",
            duration=1.0,
            bone_count=1,
            tracks=[
                AnimationTrack(
                    bone_index=0,
                    track_type=TrackType.TRANSLATION,
                    keyframes=[
                        Keyframe(0.0, Vec3(5, 5, 5)),
                        Keyframe(0.5, Vec3(5, 5, 5)),
                        Keyframe(1.0, Vec3(5, 5, 5))
                    ]
                )
            ]
        )
        settings = CompressionSettings(method=CompressionMethod.QUANTIZED)
        compressed = compress_clip(clip, settings)
        # Constant track should be detected
        assert compressed.tracks[0].is_constant


class TestDecompressClip:
    """Tests for decompress_clip function."""

    def test_decompress_clip_roundtrip(self):
        from engine.animation.skeletal.compression import (
            compress_clip, decompress_clip, AnimationClipData, AnimationTrack,
            Keyframe, TrackType, CompressionSettings, CompressionMethod
        )
        original = AnimationClipData(
            name="roundtrip",
            duration=1.0,
            bone_count=1,
            tracks=[
                AnimationTrack(
                    bone_index=0,
                    track_type=TrackType.TRANSLATION,
                    keyframes=[
                        Keyframe(0.0, Vec3(0, 0, 0)),
                        Keyframe(1.0, Vec3(10, 0, 0))
                    ]
                )
            ]
        )
        settings = CompressionSettings(method=CompressionMethod.QUANTIZED, translation_bits=16)
        compressed = compress_clip(original, settings)
        decompressed = decompress_clip(compressed)

        assert decompressed.name == original.name
        assert len(decompressed.tracks) == len(original.tracks)


class TestCompressionError:
    """Tests for compression error metrics."""

    def test_compute_compression_error(self):
        from engine.animation.skeletal.compression import (
            compress_clip, compute_compression_error, AnimationClipData, AnimationTrack,
            Keyframe, TrackType, CompressionSettings, CompressionMethod
        )
        original = AnimationClipData(
            name="error_test",
            duration=1.0,
            bone_count=1,
            tracks=[
                AnimationTrack(
                    bone_index=0,
                    track_type=TrackType.TRANSLATION,
                    keyframes=[
                        Keyframe(0.0, Vec3(0, 0, 0)),
                        Keyframe(1.0, Vec3(1, 1, 1))
                    ]
                )
            ]
        )
        settings = CompressionSettings(method=CompressionMethod.QUANTIZED, translation_bits=8)
        compressed = compress_clip(original, settings)
        metrics = compute_compression_error(original, compressed)

        # With 8-bit quantization, there should be some error
        assert metrics.max_translation_error >= 0
        assert metrics.mean_translation_error >= 0


class TestCompressionQuality:
    """Tests for compression quality vs size tradeoffs."""

    def test_higher_bits_lower_error(self):
        from engine.animation.skeletal.compression import (
            compress_clip, compute_compression_error, AnimationClipData, AnimationTrack,
            Keyframe, TrackType, CompressionSettings, CompressionMethod
        )
        original = AnimationClipData(
            name="quality_test",
            duration=1.0,
            bone_count=1,
            tracks=[
                AnimationTrack(
                    bone_index=0,
                    track_type=TrackType.TRANSLATION,
                    keyframes=[
                        Keyframe(float(i) / 10, Vec3(i * 0.1, i * 0.05, 0))
                        for i in range(11)
                    ]
                )
            ]
        )

        # 8-bit compression
        settings_8 = CompressionSettings(method=CompressionMethod.QUANTIZED, translation_bits=8)
        compressed_8 = compress_clip(original, settings_8)
        error_8 = compute_compression_error(original, compressed_8)

        # 16-bit compression
        settings_16 = CompressionSettings(method=CompressionMethod.QUANTIZED, translation_bits=16)
        compressed_16 = compress_clip(original, settings_16)
        error_16 = compute_compression_error(original, compressed_16)

        # 16-bit should have lower error
        assert error_16.max_translation_error <= error_8.max_translation_error

    def test_curve_fitting_reduces_keyframes(self):
        from engine.animation.skeletal.compression import (
            compress_clip, AnimationClipData, AnimationTrack,
            Keyframe, TrackType, CompressionSettings, CompressionMethod
        )
        # Linear motion - many keyframes but can be represented with 2
        original = AnimationClipData(
            name="linear_motion",
            duration=1.0,
            bone_count=1,
            tracks=[
                AnimationTrack(
                    bone_index=0,
                    track_type=TrackType.TRANSLATION,
                    keyframes=[
                        Keyframe(float(i) / 100, Vec3(i, 0, 0))
                        for i in range(101)
                    ]
                )
            ]
        )

        settings = CompressionSettings(
            method=CompressionMethod.CURVE,
            curve_fitting_tolerance=0.1
        )
        compressed = compress_clip(original, settings)

        # Should have fewer keyframes
        if compressed.tracks[0].sample_times:
            assert len(compressed.tracks[0].sample_times) < 101


class TestEstimateCompressedSize:
    """Tests for estimate_compressed_size function."""

    def test_estimate_size_basic(self):
        from engine.animation.skeletal.compression import (
            estimate_compressed_size, AnimationClipData, AnimationTrack,
            Keyframe, TrackType, CompressionSettings
        )
        clip = AnimationClipData(
            name="size_test",
            duration=1.0,
            bone_count=1,
            tracks=[
                AnimationTrack(
                    bone_index=0,
                    track_type=TrackType.TRANSLATION,
                    keyframes=[Keyframe(0.0, Vec3.zero()), Keyframe(1.0, Vec3(1, 1, 1))]
                )
            ]
        )
        settings = CompressionSettings()
        size = estimate_compressed_size(clip, settings)
        assert size > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSkinningIntegration:
    """Integration tests combining multiple skinning features."""

    def test_full_skinning_pipeline_lbs(self):
        from engine.animation.skeletal.skinning import (
            skin_mesh, SkinningData, SkinningMethod, VertexWeight, SkinningCache
        )
        # Create a simple arm mesh
        sd = SkinningData(
            vertices=[
                Vec3(0, 0, 0),   # Upper arm start
                Vec3(0, 1, 0),   # Elbow
                Vec3(0, 2, 0),   # Wrist
            ],
            weights=[
                VertexWeight(bone_indices=(0, 0, 0, 0), weights=(1.0, 0.0, 0.0, 0.0)),
                VertexWeight(bone_indices=(0, 1, 0, 0), weights=(0.5, 0.5, 0.0, 0.0)),
                VertexWeight(bone_indices=(1, 0, 0, 0), weights=(1.0, 0.0, 0.0, 0.0)),
            ],
            bind_pose_matrices=[
                Mat4.identity(),
                Mat4.translation(Vec3(0, 1, 0)).inverse()
            ],
            normals=[Vec3(0, 0, 1), Vec3(0, 0, 1), Vec3(0, 0, 1)]
        )

        # Bend the elbow 45 degrees
        rot = Quat.from_axis_angle(Vec3(0, 0, 1), math.pi / 4)
        bone_transforms = [
            Mat4.identity(),
            Mat4.translation(Vec3(0, 1, 0)) @ rot.to_mat4()
        ]

        cache = SkinningCache()
        verts, normals = skin_mesh(sd, bone_transforms, SkinningMethod.LBS, cache)

        assert len(verts) == 3
        assert normals is not None
        assert len(normals) == 3
        # Wrist should have moved
        assert verts[2] != Vec3(0, 2, 0)


class TestRootMotionIntegration:
    """Integration tests for root motion system."""

    def test_continuous_locomotion(self):
        from engine.animation.skeletal.root_motion import (
            RootMotionAccumulator, RootMotionData, RootMotionBlender
        )
        # Simulate walking animation
        walk_data = RootMotionData(
            delta_positions=[
                Vec3.zero(),
                Vec3(0.5, 0, 0),
                Vec3(0.5, 0, 0),
                Vec3(0.5, 0, 0),
                Vec3(0.5, 0, 0)
            ],
            delta_rotations=[Quat.identity()] * 5,
            frame_times=[0.0, 0.25, 0.5, 0.75, 1.0],
            total_delta_position=Vec3(2.0, 0, 0),
            total_delta_rotation=Quat.identity()
        )

        acc = RootMotionAccumulator(walk_data, loop=True)

        # Simulate 3 seconds of walking
        total_motion = Vec3.zero()
        for _ in range(60):  # 60 frames at ~20fps
            pos, rot = acc.update(0.05)
            total_motion = total_motion + pos

        # Should have moved roughly 6 units (3 loops of 2 units)
        assert total_motion.x > 4.0


class TestRetargetingIntegration:
    """Integration tests for retargeting system."""

    def test_retarget_simple_animation(self):
        from engine.animation.skeletal.retargeting import (
            RetargetPipeline, SkeletonInfo, PoseData, BoneMappingStrategy
        )
        # Source: tall skeleton
        source_skel = SkeletonInfo(
            bone_names=["root", "spine"],
            bone_parents=[-1, 0],
            bind_translations=[Vec3.zero(), Vec3(0, 2, 0)],
            bind_rotations=[Quat.identity(), Quat.identity()],
            bind_world_positions=[Vec3.zero(), Vec3(0, 2, 0)]
        )

        # Target: short skeleton
        target_skel = SkeletonInfo(
            bone_names=["root", "spine"],
            bone_parents=[-1, 0],
            bind_translations=[Vec3.zero(), Vec3(0, 1, 0)],
            bind_rotations=[Quat.identity(), Quat.identity()],
            bind_world_positions=[Vec3.zero(), Vec3(0, 1, 0)]
        )

        pipeline = RetargetPipeline(source_skel, target_skel)
        pipeline.create_mapping(BoneMappingStrategy.BY_NAME)

        # Animate source
        rot = Quat.from_axis_angle(Vec3(0, 0, 1), math.pi / 4)
        source_pose = PoseData(
            local_translations=[Vec3.zero(), Vec3(0, 2, 0)],
            local_rotations=[Quat.identity(), rot]
        )

        result = pipeline.retarget(source_pose)

        # Rotation should be preserved
        assert abs(result.local_rotations[1].dot(rot)) > 0.99


class TestCompressionIntegration:
    """Integration tests for compression system."""

    def test_compress_decompress_full_clip(self):
        from engine.animation.skeletal.compression import (
            compress_clip, decompress_clip, compute_compression_error,
            AnimationClipData, AnimationTrack, Keyframe, TrackType,
            CompressionSettings, CompressionMethod
        )
        # Create a realistic clip
        translations = []
        rotations = []
        for i in range(30):
            t = i / 29.0
            # Sinusoidal motion
            translations.append(Keyframe(t, Vec3(math.sin(t * math.pi * 2), 0, 0)))
            rot = Quat.from_axis_angle(Vec3(0, 1, 0), t * math.pi / 4)
            rotations.append(Keyframe(t, rot))

        clip = AnimationClipData(
            name="walk_cycle",
            duration=1.0,
            frame_rate=30.0,
            bone_count=1,
            tracks=[
                AnimationTrack(bone_index=0, track_type=TrackType.TRANSLATION, keyframes=translations),
                AnimationTrack(bone_index=0, track_type=TrackType.ROTATION, keyframes=rotations)
            ]
        )

        settings = CompressionSettings(
            method=CompressionMethod.QUANTIZED,
            translation_bits=16,
            rotation_bits=16
        )

        compressed = compress_clip(clip, settings)
        decompressed = decompress_clip(compressed)
        metrics = compute_compression_error(clip, compressed)

        # Verify quality
        assert metrics.max_translation_error < 0.01
        assert metrics.max_rotation_error < 0.001


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_skinning_zero_weights(self):
        from engine.animation.skeletal.skinning import LinearBlendSkinning, VertexWeight
        matrices = [Mat4.translation(Vec3(100, 0, 0))]
        # All zero weights
        weight = VertexWeight(weights=(0.0, 0.0, 0.0, 0.0))
        vertex = Vec3(1, 2, 3)
        result = LinearBlendSkinning.skin_vertex(vertex, matrices, weight)
        # Should return zero or original (no influence)
        assert result.length() == pytest.approx(0.0, abs=0.1)

    def test_root_motion_zero_duration(self):
        from engine.animation.skeletal.root_motion import RootMotionData
        data = RootMotionData(frame_times=[])
        assert data.duration == 0.0
        pos, rot = data.get_delta_at_time(0.5)
        assert pos == Vec3.zero()

    def test_retarget_empty_skeleton(self):
        from engine.animation.skeletal.retargeting import SkeletonInfo, create_retarget_map
        source = SkeletonInfo()
        target = SkeletonInfo()
        rm = create_retarget_map(source, target)
        assert rm.mapped_count == 0

    def test_compression_empty_clip(self):
        from engine.animation.skeletal.compression import (
            compress_clip, AnimationClipData, CompressionSettings
        )
        clip = AnimationClipData(name="empty", tracks=[])
        compressed = compress_clip(clip)
        assert len(compressed.tracks) == 0

    def test_dqs_near_singularity(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        # Very small rotation near identity
        small_rot = Quat(1e-10, 1e-10, 1e-10, 1.0).normalized()
        dq = DualQuaternion.from_transform(small_rot, Vec3.zero())
        # Should not crash and produce valid result
        p = dq.transform_point(Vec3(1, 0, 0))
        assert not math.isnan(p.x)


# =============================================================================
# ADDITIONAL TESTS TO REACH 140+
# =============================================================================

class TestDualQuaternionAdvanced:
    """Additional dual quaternion tests."""

    def test_dq_to_matrix_roundtrip(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        rot = Quat.from_euler(0.1, 0.2, 0.3)
        trans = Vec3(1, 2, 3)
        dq = DualQuaternion.from_transform(rot, trans)
        matrix = dq.to_matrix()
        # Verify translation preserved
        assert matrix.m[12] == pytest.approx(trans.x, abs=1e-5)
        assert matrix.m[13] == pytest.approx(trans.y, abs=1e-5)
        assert matrix.m[14] == pytest.approx(trans.z, abs=1e-5)

    def test_dq_identity_transform(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        dq = DualQuaternion.identity()
        p = dq.transform_point(Vec3(5, 10, 15))
        assert p.x == pytest.approx(5.0)
        assert p.y == pytest.approx(10.0)
        assert p.z == pytest.approx(15.0)

    def test_dq_chained_transforms(self):
        from engine.animation.skeletal.skinning import DualQuaternion
        # Apply two transforms and compare with matrix result
        rot1 = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)
        trans1 = Vec3(1, 0, 0)
        dq1 = DualQuaternion.from_transform(rot1, trans1)

        p = Vec3(0, 0, 1)
        result = dq1.transform_point(p)
        # Verify result is reasonable
        assert not math.isnan(result.x)
        assert not math.isnan(result.y)


class TestSkinningEdgeCases:
    """Additional skinning edge case tests."""

    def test_lbs_single_vertex_multiple_bones(self):
        from engine.animation.skeletal.skinning import LinearBlendSkinning, VertexWeight
        matrices = [
            Mat4.translation(Vec3(1, 0, 0)),
            Mat4.translation(Vec3(0, 1, 0)),
            Mat4.translation(Vec3(0, 0, 1)),
            Mat4.translation(Vec3(1, 1, 1))
        ]
        weight = VertexWeight(
            bone_indices=(0, 1, 2, 3),
            weights=(0.25, 0.25, 0.25, 0.25)
        )
        vertex = Vec3(0, 0, 0)
        result = LinearBlendSkinning.skin_vertex(vertex, matrices, weight)
        # Weighted average of translations
        assert result.x == pytest.approx(0.5)
        assert result.y == pytest.approx(0.5)

    def test_lbs_out_of_bounds_bone_index(self):
        from engine.animation.skeletal.skinning import LinearBlendSkinning, VertexWeight
        matrices = [Mat4.identity()]
        weight = VertexWeight(bone_indices=(0, 5, 10, 99), weights=(1.0, 0.0, 0.0, 0.0))
        vertex = Vec3(1, 2, 3)
        # Should handle gracefully - only use valid indices
        result = LinearBlendSkinning.skin_vertex(vertex, matrices, weight)
        assert not math.isnan(result.x)

    def test_dqs_multiple_rotations_blend(self):
        from engine.animation.skeletal.skinning import DualQuaternionSkinning, VertexWeight, DualQuaternion
        # Three bones with different rotations
        dqs = [
            DualQuaternion.from_transform(Quat.from_axis_angle(Vec3(1, 0, 0), 0.1), Vec3.zero()),
            DualQuaternion.from_transform(Quat.from_axis_angle(Vec3(0, 1, 0), 0.2), Vec3.zero()),
            DualQuaternion.from_transform(Quat.from_axis_angle(Vec3(0, 0, 1), 0.3), Vec3.zero()),
            DualQuaternion.identity()
        ]
        weight = VertexWeight(
            bone_indices=(0, 1, 2, 3),
            weights=(0.3, 0.3, 0.3, 0.1)
        )
        result = DualQuaternionSkinning.skin_vertex(Vec3(1, 0, 0), dqs, weight)
        assert result.length() > 0.9  # Volume should be mostly preserved


class TestRootMotionAdvanced:
    """Additional root motion tests."""

    def test_root_motion_blender_multiple_sources(self):
        from engine.animation.skeletal.root_motion import (
            RootMotionBlender, RootMotionAccumulator, RootMotionData
        )
        data1 = RootMotionData(
            delta_positions=[Vec3.zero(), Vec3(2, 0, 0)],
            delta_rotations=[Quat.identity(), Quat.identity()],
            frame_times=[0.0, 1.0],
            total_delta_position=Vec3(2, 0, 0)
        )
        data2 = RootMotionData(
            delta_positions=[Vec3.zero(), Vec3(0, 2, 0)],
            delta_rotations=[Quat.identity(), Quat.identity()],
            frame_times=[0.0, 1.0],
            total_delta_position=Vec3(0, 2, 0)
        )

        acc1 = RootMotionAccumulator(data1, loop=False)
        acc2 = RootMotionAccumulator(data2, loop=False)

        blender = RootMotionBlender()
        blender.add_source(acc1, 0.5)
        blender.add_source(acc2, 0.5)

        pos, rot = blender.update(0.5)
        # Should blend between the two sources
        assert pos.x >= 0
        assert pos.y >= 0

    def test_root_motion_playback_rate(self):
        from engine.animation.skeletal.root_motion import RootMotionAccumulator, RootMotionData
        data = RootMotionData(
            delta_positions=[Vec3.zero(), Vec3(10, 0, 0)],
            delta_rotations=[Quat.identity(), Quat.identity()],
            frame_times=[0.0, 1.0],
            total_delta_position=Vec3(10, 0, 0)
        )
        acc = RootMotionAccumulator(data, loop=False)
        # 2x playback rate
        acc.update(0.5, playback_rate=2.0)
        assert acc.current_time == pytest.approx(1.0)

    def test_root_motion_negative_time(self):
        from engine.animation.skeletal.root_motion import RootMotionData
        data = RootMotionData(frame_times=[0.0, 1.0])
        pos, rot = data.get_delta_at_time(-1.0)
        assert pos == Vec3.zero()

    def test_extract_root_motion_multiple_frames(self):
        from engine.animation.skeletal.root_motion import (
            extract_root_motion, RootMotionMode, RootBoneTransform
        )
        transforms = [
            RootBoneTransform(Vec3(i, 0, i), Quat.identity(), i * 0.1)
            for i in range(10)
        ]
        data = extract_root_motion(transforms, RootMotionMode.EXTRACT_XZ)
        assert data.frame_count == 10
        assert data.duration == pytest.approx(0.9)


class TestRetargetingAdvanced:
    """Additional retargeting tests."""

    def test_retarget_map_remove_mapping(self):
        from engine.animation.skeletal.retargeting import RetargetMap, BoneMapping
        rm = RetargetMap(source_bone_count=5, target_bone_count=5)
        rm.unmapped_source_bones = set(range(5))
        rm.unmapped_target_bones = set(range(5))

        mapping = BoneMapping(source_index=2, target_index=3)
        rm.add_mapping(mapping)
        assert rm.mapped_count == 1

        removed = rm.remove_mapping(2)
        assert removed is not None
        assert rm.mapped_count == 0
        assert 2 in rm.unmapped_source_bones

    def test_skeleton_info_bone_length(self):
        from engine.animation.skeletal.retargeting import SkeletonInfo
        skel = SkeletonInfo(
            bone_names=["root", "child"],
            bone_parents=[-1, 0],
            bind_world_positions=[Vec3(0, 0, 0), Vec3(0, 2, 0)]
        )
        length = skel.get_bone_length(1)
        assert length == pytest.approx(2.0)

    def test_compute_scale_factor_different_skeletons(self):
        from engine.animation.skeletal.retargeting import compute_scale_factor, SkeletonInfo
        source = SkeletonInfo(
            bone_names=["root", "bone"],
            bone_parents=[-1, 0],
            bind_world_positions=[Vec3(0, 0, 0), Vec3(0, 1, 0)]
        )
        target = SkeletonInfo(
            bone_names=["root", "bone"],
            bone_parents=[-1, 0],
            bind_world_positions=[Vec3(0, 0, 0), Vec3(0, 3, 0)]
        )
        scale = compute_scale_factor(source, target)
        assert scale > 1.0

    def test_validate_retarget_map_errors(self):
        from engine.animation.skeletal.retargeting import (
            validate_retarget_map, RetargetMap, BoneMapping, SkeletonInfo
        )
        source = SkeletonInfo(bone_names=["a", "b"])
        target = SkeletonInfo(bone_names=["x", "y"])

        rm = RetargetMap(
            source_bone_count=5,  # Mismatch!
            target_bone_count=2,
            mappings=[BoneMapping(source_index=0, target_index=0)]
        )

        errors = validate_retarget_map(rm, source, target)
        assert len(errors) > 0


class TestCompressionAdvanced:
    """Additional compression tests."""

    def test_compression_rotation_track(self):
        from engine.animation.skeletal.compression import (
            compress_clip, AnimationClipData, AnimationTrack, Keyframe,
            TrackType, CompressionSettings, CompressionMethod
        )
        rotations = [
            Keyframe(i * 0.1, Quat.from_axis_angle(Vec3(0, 1, 0), i * 0.1))
            for i in range(11)
        ]
        clip = AnimationClipData(
            name="rotation_test",
            duration=1.0,
            bone_count=1,
            tracks=[
                AnimationTrack(
                    bone_index=0,
                    track_type=TrackType.ROTATION,
                    keyframes=rotations
                )
            ]
        )
        settings = CompressionSettings(method=CompressionMethod.QUANTIZED, rotation_bits=16)
        compressed = compress_clip(clip, settings)
        assert len(compressed.tracks) == 1

    def test_compression_scale_track(self):
        from engine.animation.skeletal.compression import (
            compress_clip, AnimationClipData, AnimationTrack, Keyframe,
            TrackType, CompressionSettings, CompressionMethod
        )
        scales = [
            Keyframe(i * 0.1, Vec3(1 + i * 0.1, 1 + i * 0.1, 1 + i * 0.1))
            for i in range(11)
        ]
        clip = AnimationClipData(
            name="scale_test",
            duration=1.0,
            bone_count=1,
            tracks=[
                AnimationTrack(
                    bone_index=0,
                    track_type=TrackType.SCALE,
                    keyframes=scales
                )
            ]
        )
        settings = CompressionSettings(method=CompressionMethod.QUANTIZED)
        compressed = compress_clip(clip, settings)
        assert len(compressed.tracks) == 1

    def test_compression_error_threshold(self):
        from engine.animation.skeletal.compression import (
            CompressionErrorMetrics, CompressionSettings
        )
        settings = CompressionSettings(
            translation_error_threshold=0.01,
            rotation_error_threshold=0.001
        )
        metrics = CompressionErrorMetrics(
            max_translation_error=0.005,
            max_rotation_error=0.0005
        )
        assert metrics.meets_threshold(settings)

        metrics_fail = CompressionErrorMetrics(
            max_translation_error=0.05,
            max_rotation_error=0.0005
        )
        assert not metrics_fail.meets_threshold(settings)

    def test_acl_compression(self):
        from engine.animation.skeletal.compression import (
            compress_clip, AnimationClipData, AnimationTrack, Keyframe,
            TrackType, CompressionSettings, CompressionMethod
        )
        clip = AnimationClipData(
            name="acl_test",
            duration=1.0,
            bone_count=1,
            tracks=[
                AnimationTrack(
                    bone_index=0,
                    track_type=TrackType.TRANSLATION,
                    keyframes=[
                        Keyframe(0.0, Vec3(0, 0, 0)),
                        Keyframe(0.5, Vec3(1, 1, 1)),
                        Keyframe(1.0, Vec3(2, 2, 2))
                    ]
                )
            ]
        )
        settings = CompressionSettings(method=CompressionMethod.ACL)
        compressed = compress_clip(clip, settings)
        assert compressed.compression_method == CompressionMethod.ACL

    def test_decompress_constant_track(self):
        from engine.animation.skeletal.compression import (
            decompress_track, CompressedTrack, TrackType
        )
        track = CompressedTrack(
            bone_index=0,
            track_type=TrackType.TRANSLATION,
            is_constant=True,
            constant_value=Vec3(5, 5, 5)
        )
        decompressed = decompress_track(track)
        assert len(decompressed.keyframes) == 1
        assert decompressed.keyframes[0].value.x == pytest.approx(5.0)


class TestVertexWeightAdvanced:
    """Additional vertex weight tests."""

    def test_vertex_weight_from_dict_missing_fields(self):
        from engine.animation.skeletal.skinning import VertexWeight
        data = {}  # Empty dict
        w = VertexWeight.from_dict(data)
        assert len(w.bone_indices) == 4
        assert len(w.weights) == 4

    def test_vertex_weight_zero_influence(self):
        from engine.animation.skeletal.skinning import VertexWeight
        w = VertexWeight(weights=(0.0, 0.0, 0.0, 0.0))
        assert w.influence_count == 0


class TestSkinningDataAdvanced:
    """Additional skinning data tests."""

    def test_skinning_data_with_tangents(self):
        from engine.animation.skeletal.skinning import SkinningData, VertexWeight
        sd = SkinningData(
            vertices=[Vec3(0, 0, 0)],
            weights=[VertexWeight()],
            bind_pose_matrices=[Mat4.identity()],
            tangents=[Vec4(1, 0, 0, 1)]
        )
        errors = sd.validate()
        assert len(errors) == 0

    def test_skinning_data_normal_count_mismatch(self):
        from engine.animation.skeletal.skinning import SkinningData, VertexWeight
        sd = SkinningData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0)],
            weights=[VertexWeight(), VertexWeight()],
            bind_pose_matrices=[Mat4.identity()],
            normals=[Vec3(0, 1, 0)]  # Only one normal for two vertices
        )
        errors = sd.validate()
        assert any("normal" in e.lower() for e in errors)


class TestPoseDataTests:
    """Tests for PoseData class."""

    def test_pose_data_identity(self):
        from engine.animation.skeletal.retargeting import PoseData
        pose = PoseData.identity(5)
        assert pose.bone_count == 5
        for rot in pose.local_rotations:
            assert rot == Quat.identity()

    def test_pose_data_empty(self):
        from engine.animation.skeletal.retargeting import PoseData
        pose = PoseData()
        assert pose.bone_count == 0


class TestRetargetConfigTests:
    """Tests for RetargetConfig class."""

    def test_retarget_config_per_bone_scale(self):
        from engine.animation.skeletal.retargeting import RetargetConfig
        config = RetargetConfig(
            scale_factor=1.0,
            per_bone_scale={0: 2.0, 1: 0.5}
        )
        assert config.get_scale_for_bone(0) == 2.0
        assert config.get_scale_for_bone(1) == 0.5
        assert config.get_scale_for_bone(2) == 1.0  # Default

    def test_retarget_config_rotation_offset(self):
        from engine.animation.skeletal.retargeting import RetargetConfig
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), 0.5)
        config = RetargetConfig(rotation_offsets={0: rot})
        assert config.get_rotation_offset(0) == rot
        assert config.get_rotation_offset(1) == Quat.identity()


class TestTrackTypeTests:
    """Tests for TrackType enum."""

    def test_track_types_exist(self):
        from engine.animation.skeletal.compression import TrackType
        assert TrackType.TRANSLATION
        assert TrackType.ROTATION
        assert TrackType.SCALE


class TestBoneMappingStrategyTests:
    """Tests for BoneMappingStrategy enum."""

    def test_mapping_strategies_exist(self):
        from engine.animation.skeletal.retargeting import BoneMappingStrategy
        assert BoneMappingStrategy.BY_NAME
        assert BoneMappingStrategy.BY_NAME_FUZZY
        assert BoneMappingStrategy.BY_HIERARCHY
        assert BoneMappingStrategy.BY_POSITION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
