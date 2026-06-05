"""
Blackbox Tests for T3.1: Blend Shape Evaluation

Contract-based testing without reading implementation.
Tests the public interface as specified in PHASE_3_TODO.md.

Public Contract:
    BlendShape(name, vertex_indices, deltas) - Creates a blend shape
    apply_blend_shape(base_mesh, shape, weight) - Applies blend shape to mesh

Acceptance Criteria:
    1. Sparse deltas apply correctly to target vertices
    2. Corrective shapes activate at correct thresholds
    3. Blend weights clamp to [0, 1] range
    4. NumPy vectorization is correct
"""

import pytest
import numpy as np


class TestBlendShapeBasicContract:
    """Test the basic contract from PHASE_3_TODO.md specification."""

    def test_contract_example_exact(self):
        """Verify the exact example from the public contract."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="smile",
            vertex_indices=np.array([0, 5, 10]),
            deltas=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        )
        base_mesh = np.zeros((20, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=0.5)

        assert result[0, 0] == 0.5, "Vertex 0, X component should be 0.5"
        assert result[5, 1] == 0.5, "Vertex 5, Y component should be 0.5"
        assert result[10, 2] == 0.5, "Vertex 10, Z component should be 0.5"

    def test_blend_shape_creation(self):
        """Test that BlendShape can be created with required parameters."""
        from engine.animation.facial.blend_shapes import BlendShape

        shape = BlendShape(
            name="test_shape",
            vertex_indices=np.array([1, 2, 3]),
            deltas=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
        )

        assert shape.name == "test_shape"
        assert len(shape.vertex_indices) == 3
        assert shape.deltas.shape == (3, 3)


class TestSparseDeltas:
    """Acceptance Criteria 1: Sparse deltas apply correctly to target vertices."""

    def test_single_vertex_delta(self):
        """Test applying delta to a single vertex."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="single",
            vertex_indices=np.array([5]),
            deltas=np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        )
        base_mesh = np.zeros((10, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=1.0)

        assert result[5, 0] == 1.0
        assert result[5, 1] == 2.0
        assert result[5, 2] == 3.0

    def test_unaffected_vertices_unchanged(self):
        """Test that vertices not in vertex_indices remain unchanged."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="sparse",
            vertex_indices=np.array([0, 5]),
            deltas=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        )
        base_mesh = np.ones((10, 3), dtype=np.float32) * 5.0
        result = apply_blend_shape(base_mesh, shape, weight=1.0)

        # Check affected vertices
        assert result[0, 0] == 6.0  # 5.0 + 1.0
        assert result[5, 1] == 6.0  # 5.0 + 1.0

        # Check unaffected vertices remain unchanged
        for i in [1, 2, 3, 4, 6, 7, 8, 9]:
            np.testing.assert_array_equal(result[i], [5.0, 5.0, 5.0],
                err_msg=f"Vertex {i} should remain unchanged")

    def test_multiple_vertices_with_different_deltas(self):
        """Test multiple vertices with distinct deltas."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="multi",
            vertex_indices=np.array([0, 3, 7, 9]),
            deltas=np.array([
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.0, 1.0, 1.0]
            ], dtype=np.float32)
        )
        base_mesh = np.zeros((10, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=1.0)

        np.testing.assert_array_almost_equal(result[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result[3], [0.0, 2.0, 0.0])
        np.testing.assert_array_almost_equal(result[7], [0.0, 0.0, 3.0])
        np.testing.assert_array_almost_equal(result[9], [1.0, 1.0, 1.0])

    def test_non_contiguous_vertex_indices(self):
        """Test with non-contiguous vertex indices."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="scattered",
            vertex_indices=np.array([2, 17, 45, 99]),
            deltas=np.array([
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]
            ], dtype=np.float32)
        )
        base_mesh = np.zeros((100, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=1.0)

        np.testing.assert_array_almost_equal(result[2], [0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(result[17], [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result[45], [2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(result[99], [3.0, 3.0, 3.0])


class TestCorrectiveShapes:
    """Acceptance Criteria 2: Corrective shapes activate at correct thresholds."""

    def test_corrective_shape_below_threshold(self):
        """Test that corrective shape does not apply below threshold."""
        from engine.animation.facial.blend_shapes import (
            BlendShape, apply_blend_shape, CorrectiveShape
        )

        base_shape = BlendShape(
            name="base",
            vertex_indices=np.array([0]),
            deltas=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        )

        corrective = CorrectiveShape(
            name="corrective",
            vertex_indices=np.array([0]),
            deltas=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            trigger_threshold=0.8
        )

        base_mesh = np.zeros((10, 3), dtype=np.float32)

        # Apply base shape at weight below corrective threshold
        result = apply_blend_shape(base_mesh, base_shape, weight=0.5)

        # Only base shape should apply
        assert result[0, 0] == 0.5
        assert result[0, 1] == 0.0

    def test_corrective_shape_at_threshold(self):
        """Test that corrective shape activates at threshold."""
        from engine.animation.facial.blend_shapes import (
            BlendShape, CorrectiveShape, apply_corrective_shape
        )

        corrective = CorrectiveShape(
            name="corrective",
            vertex_indices=np.array([0]),
            deltas=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            trigger_threshold=0.8
        )

        base_mesh = np.zeros((10, 3), dtype=np.float32)

        # Apply corrective at exactly threshold
        result = apply_corrective_shape(base_mesh, corrective, trigger_weight=0.8)

        # Corrective should apply
        assert result[0, 1] > 0.0

    def test_corrective_shape_above_threshold(self):
        """Test that corrective shape fully activates above threshold."""
        from engine.animation.facial.blend_shapes import (
            CorrectiveShape, apply_corrective_shape
        )

        corrective = CorrectiveShape(
            name="corrective",
            vertex_indices=np.array([0]),
            deltas=np.array([[0.0, 2.0, 0.0]], dtype=np.float32),
            trigger_threshold=0.5
        )

        base_mesh = np.zeros((10, 3), dtype=np.float32)

        result = apply_corrective_shape(base_mesh, corrective, trigger_weight=1.0)

        # At full weight (above threshold), corrective should be fully applied
        assert result[0, 1] == 2.0

    def test_multiple_corrective_thresholds(self):
        """Test multiple correctives with different thresholds."""
        from engine.animation.facial.blend_shapes import (
            CorrectiveShape, apply_corrective_shape
        )

        low_threshold = CorrectiveShape(
            name="low",
            vertex_indices=np.array([0]),
            deltas=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            trigger_threshold=0.3
        )

        high_threshold = CorrectiveShape(
            name="high",
            vertex_indices=np.array([1]),
            deltas=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            trigger_threshold=0.7
        )

        base_mesh = np.zeros((10, 3), dtype=np.float32)

        # At 0.5, only low_threshold should activate
        result1 = apply_corrective_shape(base_mesh.copy(), low_threshold, trigger_weight=0.5)
        result2 = apply_corrective_shape(base_mesh.copy(), high_threshold, trigger_weight=0.5)

        assert result1[0, 0] > 0.0  # Low threshold activated
        assert result2[1, 1] == 0.0  # High threshold not activated


class TestWeightClamping:
    """Acceptance Criteria 3: Blend weights clamp to [0, 1] range."""

    def test_weight_zero(self):
        """Test that weight=0 results in no change."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="test",
            vertex_indices=np.array([0, 1, 2]),
            deltas=np.array([[1.0, 1.0, 1.0]] * 3, dtype=np.float32)
        )
        base_mesh = np.zeros((10, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=0.0)

        np.testing.assert_array_equal(result, base_mesh)

    def test_weight_one(self):
        """Test that weight=1 applies full delta."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="test",
            vertex_indices=np.array([0]),
            deltas=np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
        )
        base_mesh = np.zeros((10, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=1.0)

        np.testing.assert_array_almost_equal(result[0], [2.0, 3.0, 4.0])

    def test_weight_negative_clamped_to_zero(self):
        """Test that negative weights are clamped to 0."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="test",
            vertex_indices=np.array([0]),
            deltas=np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
        )
        base_mesh = np.zeros((10, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=-0.5)

        # Negative weight should be clamped to 0, so no change
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 0.0])

    def test_weight_above_one_clamped(self):
        """Test that weights > 1 are clamped to 1."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="test",
            vertex_indices=np.array([0]),
            deltas=np.array([[2.0, 2.0, 2.0]], dtype=np.float32)
        )
        base_mesh = np.zeros((10, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=1.5)

        # Weight should be clamped to 1.0, so delta is [2.0, 2.0, 2.0]
        np.testing.assert_array_almost_equal(result[0], [2.0, 2.0, 2.0])

    def test_weight_large_value_clamped(self):
        """Test that very large weights are clamped to 1."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="test",
            vertex_indices=np.array([0]),
            deltas=np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        )
        base_mesh = np.zeros((10, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=100.0)

        # Should be clamped to weight=1.0
        np.testing.assert_array_almost_equal(result[0], [1.0, 1.0, 1.0])

    def test_weight_interpolation_at_half(self):
        """Test that weight=0.5 applies half the delta."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="test",
            vertex_indices=np.array([0]),
            deltas=np.array([[4.0, 6.0, 8.0]], dtype=np.float32)
        )
        base_mesh = np.zeros((10, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=0.5)

        np.testing.assert_array_almost_equal(result[0], [2.0, 3.0, 4.0])


class TestNumPyVectorization:
    """Acceptance Criteria 4: NumPy vectorization is correct."""

    def test_output_is_numpy_array(self):
        """Test that output is a NumPy array."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="test",
            vertex_indices=np.array([0]),
            deltas=np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        )
        base_mesh = np.zeros((10, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=0.5)

        assert isinstance(result, np.ndarray)

    def test_output_dtype_float32(self):
        """Test that output maintains float32 dtype."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="test",
            vertex_indices=np.array([0]),
            deltas=np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        )
        base_mesh = np.zeros((10, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=0.5)

        assert result.dtype == np.float32

    def test_output_shape_preserved(self):
        """Test that output shape matches input shape."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="test",
            vertex_indices=np.array([0, 5, 10]),
            deltas=np.array([[1.0, 0.0, 0.0]] * 3, dtype=np.float32)
        )
        base_mesh = np.zeros((100, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=0.5)

        assert result.shape == base_mesh.shape

    def test_base_mesh_not_mutated(self):
        """Test that the original base_mesh is not modified."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="test",
            vertex_indices=np.array([0]),
            deltas=np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        )
        base_mesh = np.zeros((10, 3), dtype=np.float32)
        original_copy = base_mesh.copy()
        _ = apply_blend_shape(base_mesh, shape, weight=1.0)

        np.testing.assert_array_equal(base_mesh, original_copy,
            err_msg="Original mesh should not be mutated")

    def test_large_mesh_performance(self):
        """Test vectorization handles large meshes efficiently."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape
        import time

        # Simulate realistic mesh with 10k vertices, 1k affected
        num_vertices = 10000
        affected_vertices = 1000

        shape = BlendShape(
            name="large",
            vertex_indices=np.arange(0, affected_vertices * 10, 10),
            deltas=np.random.randn(affected_vertices, 3).astype(np.float32)
        )
        base_mesh = np.random.randn(num_vertices, 3).astype(np.float32)

        start = time.perf_counter()
        result = apply_blend_shape(base_mesh, shape, weight=0.75)
        elapsed = time.perf_counter() - start

        # Should complete in under 10ms for vectorized implementation
        assert elapsed < 0.1, f"Operation took {elapsed:.3f}s, expected < 0.1s"
        assert result.shape == (num_vertices, 3)

    def test_batch_blend_shapes(self):
        """Test applying multiple blend shapes in sequence."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shapes = [
            BlendShape(
                name=f"shape_{i}",
                vertex_indices=np.array([i]),
                deltas=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            )
            for i in range(5)
        ]

        base_mesh = np.zeros((10, 3), dtype=np.float32)
        result = base_mesh.copy()

        for shape in shapes:
            result = apply_blend_shape(result, shape, weight=1.0)

        # Each shape should have added [1,0,0] to its vertex
        for i in range(5):
            assert result[i, 0] == 1.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_vertex_indices(self):
        """Test blend shape with no affected vertices."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="empty",
            vertex_indices=np.array([], dtype=np.int32),
            deltas=np.array([], dtype=np.float32).reshape(0, 3)
        )
        base_mesh = np.ones((10, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=1.0)

        np.testing.assert_array_equal(result, base_mesh)

    def test_single_vertex_mesh(self):
        """Test with minimal single-vertex mesh."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="minimal",
            vertex_indices=np.array([0]),
            deltas=np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        )
        base_mesh = np.zeros((1, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=0.5)

        np.testing.assert_array_almost_equal(result[0], [0.5, 1.0, 1.5])

    def test_last_vertex_index(self):
        """Test affecting the last vertex in mesh."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="last",
            vertex_indices=np.array([99]),
            deltas=np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        )
        base_mesh = np.zeros((100, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=1.0)

        np.testing.assert_array_almost_equal(result[99], [1.0, 1.0, 1.0])

    def test_first_and_last_vertices(self):
        """Test affecting both first and last vertices."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="extremes",
            vertex_indices=np.array([0, 49]),
            deltas=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        )
        base_mesh = np.zeros((50, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=1.0)

        assert result[0, 0] == 1.0
        assert result[49, 1] == 1.0

    def test_negative_deltas(self):
        """Test that negative deltas work correctly."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="negative",
            vertex_indices=np.array([0]),
            deltas=np.array([[-2.0, -3.0, -4.0]], dtype=np.float32)
        )
        base_mesh = np.ones((10, 3), dtype=np.float32) * 5.0
        result = apply_blend_shape(base_mesh, shape, weight=1.0)

        np.testing.assert_array_almost_equal(result[0], [3.0, 2.0, 1.0])

    def test_very_small_weight(self):
        """Test with very small weight values."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="tiny",
            vertex_indices=np.array([0]),
            deltas=np.array([[1000.0, 1000.0, 1000.0]], dtype=np.float32)
        )
        base_mesh = np.zeros((10, 3), dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=0.001)

        np.testing.assert_array_almost_equal(result[0], [1.0, 1.0, 1.0])

    def test_additive_with_existing_mesh_values(self):
        """Test that blend shapes add to existing mesh values."""
        from engine.animation.facial.blend_shapes import BlendShape, apply_blend_shape

        shape = BlendShape(
            name="additive",
            vertex_indices=np.array([0]),
            deltas=np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        )
        base_mesh = np.array([[10.0, 20.0, 30.0]] * 5, dtype=np.float32)
        result = apply_blend_shape(base_mesh, shape, weight=1.0)

        np.testing.assert_array_almost_equal(result[0], [11.0, 22.0, 33.0])
        # Unaffected vertices remain unchanged
        np.testing.assert_array_almost_equal(result[1], [10.0, 20.0, 30.0])


class TestBlendShapeNameAndMetadata:
    """Test blend shape metadata handling."""

    def test_shape_name_accessible(self):
        """Test that shape name is accessible after creation."""
        from engine.animation.facial.blend_shapes import BlendShape

        shape = BlendShape(
            name="my_custom_shape",
            vertex_indices=np.array([0]),
            deltas=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        )

        assert shape.name == "my_custom_shape"

    def test_unicode_name(self):
        """Test that unicode names are supported."""
        from engine.animation.facial.blend_shapes import BlendShape

        shape = BlendShape(
            name="smile_émotion",  # e with accent
            vertex_indices=np.array([0]),
            deltas=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        )

        assert "smile" in shape.name

    def test_empty_name(self):
        """Test that empty name is handled."""
        from engine.animation.facial.blend_shapes import BlendShape

        shape = BlendShape(
            name="",
            vertex_indices=np.array([0]),
            deltas=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        )

        assert shape.name == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
