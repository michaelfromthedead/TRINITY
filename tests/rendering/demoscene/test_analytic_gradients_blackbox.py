"""
TRINITY Analytic Gradients Blackbox Tests (T-DEMO-8.1)

Blackbox tests treating analytic gradient module as a black box,
testing external behavior and contracts without knowledge of internals.

Test Categories:
1. API contract tests
2. Return type validation
3. Input validation and error handling
4. Integration with SDF AST
5. Performance characteristics
"""

import math
import time
from typing import List

import pytest

from engine.rendering.demoscene.sdf_ast import Vec3
from engine.rendering.demoscene.analytic_gradients import (
    # Types
    GradientResult,
    CombinatorGradientResult,
    # Primitive functions
    gradient_sphere,
    gradient_box,
    gradient_torus,
    gradient_cylinder,
    gradient_cone,
    gradient_plane,
    gradient_capsule,
    gradient_ellipsoid,
    gradient_box_frame,
    gradient_rounded_box,
    gradient_octahedron,
    gradient_pyramid,
    # Combinator functions
    gradient_union,
    gradient_intersection,
    gradient_subtraction,
    gradient_smooth_union,
    gradient_smooth_intersection,
    gradient_smooth_subtraction,
    # Utilities
    validate_gradient,
    central_difference_gradient,
    normalize_gradient,
)


# =============================================================================
# Test 1-5: API Contract Tests
# =============================================================================


class TestAPIContracts:
    """Tests for API contracts and return types."""

    def test_gradient_sphere_returns_gradient_result(self):
        """gradient_sphere should return GradientResult."""
        result = gradient_sphere(Vec3(1.0, 0.0, 0.0), 1.0)
        assert isinstance(result, GradientResult)

    def test_gradient_result_has_required_fields(self):
        """GradientResult should have gradient, distance, and normal."""
        result = gradient_sphere(Vec3(1.0, 0.0, 0.0), 1.0)

        assert hasattr(result, 'gradient')
        assert hasattr(result, 'distance')
        assert hasattr(result, 'normal')

    def test_gradient_result_gradient_is_vec3(self):
        """Gradient should be Vec3."""
        result = gradient_sphere(Vec3(1.0, 0.0, 0.0), 1.0)
        assert isinstance(result.gradient, Vec3)

    def test_gradient_result_distance_is_float(self):
        """Distance should be float."""
        result = gradient_sphere(Vec3(1.0, 0.0, 0.0), 1.0)
        assert isinstance(result.distance, float)

    def test_gradient_result_normal_is_vec3(self):
        """Normal should be Vec3."""
        result = gradient_sphere(Vec3(1.0, 0.0, 0.0), 1.0)
        normal = result.normal
        assert isinstance(normal, Vec3)


class TestCombinatorContracts:
    """Tests for combinator API contracts."""

    def test_combinator_returns_combinator_result(self):
        """Combinator gradients should return CombinatorGradientResult."""
        grad_a = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)
        grad_b = GradientResult(Vec3(0.0, 1.0, 0.0), 0.5)

        result = gradient_union(grad_a, grad_b)
        assert isinstance(result, CombinatorGradientResult)

    def test_combinator_result_has_winner_id(self):
        """CombinatorGradientResult should have winner_id."""
        grad_a = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)
        grad_b = GradientResult(Vec3(0.0, 1.0, 0.0), 0.5)

        result = gradient_union(grad_a, grad_b)
        assert hasattr(result, 'winner_id')
        assert result.winner_id in [0, 1]

    def test_smooth_combinator_has_blend_weight(self):
        """Smooth combinators should have blend_weight."""
        grad_a = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)
        grad_b = GradientResult(Vec3(0.0, 1.0, 0.0), 0.5)

        result = gradient_smooth_union(grad_a, grad_b, k=0.5)
        assert hasattr(result, 'blend_weight')
        assert 0.0 <= result.blend_weight <= 1.0

    def test_hard_combinators_have_zero_blend_weight(self):
        """Hard combinators should have blend_weight = 0."""
        grad_a = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)
        grad_b = GradientResult(Vec3(0.0, 1.0, 0.0), 0.5)

        result = gradient_union(grad_a, grad_b)
        assert result.blend_weight == 0.0


# =============================================================================
# Test 6-10: Normal Normalization
# =============================================================================


class TestNormalNormalization:
    """Tests for normal vector normalization."""

    def test_normal_is_unit_length(self):
        """Normal should always be unit length."""
        result = gradient_sphere(Vec3(2.0, 3.0, 4.0), 1.0)
        normal = result.normal

        length = math.sqrt(normal.x**2 + normal.y**2 + normal.z**2)
        assert abs(length - 1.0) < 1e-6

    def test_normalize_gradient_returns_unit(self):
        """normalize_gradient should return unit vector."""
        vec = Vec3(3.0, 4.0, 0.0)
        normalized = normalize_gradient(vec)

        length = math.sqrt(normalized.x**2 + normalized.y**2 + normalized.z**2)
        assert abs(length - 1.0) < 1e-6

    def test_normalize_zero_vector(self):
        """Normalizing zero vector should return default."""
        vec = Vec3(0.0, 0.0, 0.0)
        normalized = normalize_gradient(vec)

        length = math.sqrt(normalized.x**2 + normalized.y**2 + normalized.z**2)
        assert abs(length - 1.0) < 1e-6

    def test_normal_direction_is_correct(self):
        """Normal should point in correct direction."""
        # On +x axis, normal should point +x
        result = gradient_sphere(Vec3(2.0, 0.0, 0.0), 1.0)
        assert result.normal.x > 0.9

        # On -y axis, normal should point -y
        result = gradient_sphere(Vec3(0.0, -2.0, 0.0), 1.0)
        assert result.normal.y < -0.9

    def test_combinator_normal_is_unit(self):
        """Combinator result normal should be unit length."""
        grad_a = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)
        grad_b = GradientResult(Vec3(0.0, 1.0, 0.0), 0.5)

        result = gradient_smooth_union(grad_a, grad_b, k=0.5)
        normal = result.normal

        length = math.sqrt(normal.x**2 + normal.y**2 + normal.z**2)
        assert abs(length - 1.0) < 1e-6


# =============================================================================
# Test 11-15: Distance Correctness
# =============================================================================


class TestDistanceCorrectness:
    """Tests for distance value correctness."""

    def test_sphere_distance_positive_outside(self):
        """Distance should be positive outside shape."""
        result = gradient_sphere(Vec3(2.0, 0.0, 0.0), 1.0)
        assert result.distance > 0

    def test_sphere_distance_negative_inside(self):
        """Distance should be negative inside shape."""
        result = gradient_sphere(Vec3(0.5, 0.0, 0.0), 1.0)
        assert result.distance < 0

    def test_sphere_distance_zero_on_surface(self):
        """Distance should be zero on surface."""
        result = gradient_sphere(Vec3(1.0, 0.0, 0.0), 1.0)
        assert abs(result.distance) < 1e-6

    def test_union_distance_is_minimum(self):
        """Union distance should be minimum of inputs."""
        grad_a = GradientResult(Vec3(1.0, 0.0, 0.0), 0.3)
        grad_b = GradientResult(Vec3(0.0, 1.0, 0.0), 0.7)

        result = gradient_union(grad_a, grad_b)
        assert abs(result.distance - 0.3) < 1e-10

    def test_intersection_distance_is_maximum(self):
        """Intersection distance should be maximum of inputs."""
        grad_a = GradientResult(Vec3(1.0, 0.0, 0.0), 0.3)
        grad_b = GradientResult(Vec3(0.0, 1.0, 0.0), 0.7)

        result = gradient_intersection(grad_a, grad_b)
        assert abs(result.distance - 0.7) < 1e-10


# =============================================================================
# Test 16-20: All Primitives Accessible
# =============================================================================


class TestAllPrimitivesAccessible:
    """Tests that all 12 primitives are accessible and work."""

    def test_gradient_sphere_accessible(self):
        """gradient_sphere should be callable."""
        result = gradient_sphere(Vec3(1.0, 0.0, 0.0), 1.0)
        assert result is not None

    def test_gradient_box_accessible(self):
        """gradient_box should be callable."""
        result = gradient_box(Vec3(1.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0))
        assert result is not None

    def test_gradient_torus_accessible(self):
        """gradient_torus should be callable."""
        result = gradient_torus(Vec3(1.0, 0.0, 0.0), 1.0, 0.25)
        assert result is not None

    def test_gradient_cylinder_accessible(self):
        """gradient_cylinder should be callable."""
        result = gradient_cylinder(Vec3(1.0, 0.0, 0.0), 0.5, 1.0)
        assert result is not None

    def test_gradient_cone_accessible(self):
        """gradient_cone should be callable."""
        result = gradient_cone(Vec3(1.0, 0.0, 0.0), math.pi/4, 1.0)
        assert result is not None

    def test_gradient_plane_accessible(self):
        """gradient_plane should be callable."""
        result = gradient_plane(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 0.0)
        assert result is not None

    def test_gradient_capsule_accessible(self):
        """gradient_capsule should be callable."""
        result = gradient_capsule(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, -0.5, 0.0),
            Vec3(0.0, 0.5, 0.0),
            0.25
        )
        assert result is not None

    def test_gradient_ellipsoid_accessible(self):
        """gradient_ellipsoid should be callable."""
        result = gradient_ellipsoid(Vec3(1.0, 0.0, 0.0), Vec3(1.0, 1.5, 1.0))
        assert result is not None

    def test_gradient_box_frame_accessible(self):
        """gradient_box_frame should be callable."""
        result = gradient_box_frame(Vec3(1.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), 0.1)
        assert result is not None

    def test_gradient_rounded_box_accessible(self):
        """gradient_rounded_box should be callable."""
        result = gradient_rounded_box(Vec3(1.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), 0.1)
        assert result is not None

    def test_gradient_octahedron_accessible(self):
        """gradient_octahedron should be callable."""
        result = gradient_octahedron(Vec3(1.0, 0.0, 0.0), 1.0)
        assert result is not None

    def test_gradient_pyramid_accessible(self):
        """gradient_pyramid should be callable."""
        result = gradient_pyramid(Vec3(1.0, 0.0, 0.0), 1.0)
        assert result is not None


# =============================================================================
# Test 21-25: All Combinators Accessible
# =============================================================================


class TestAllCombinatorsAccessible:
    """Tests that all combinator functions are accessible."""

    def setup_method(self):
        """Set up test fixtures."""
        self.grad_a = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)
        self.grad_b = GradientResult(Vec3(0.0, 1.0, 0.0), 0.5)

    def test_gradient_union_accessible(self):
        """gradient_union should be callable."""
        result = gradient_union(self.grad_a, self.grad_b)
        assert result is not None

    def test_gradient_intersection_accessible(self):
        """gradient_intersection should be callable."""
        result = gradient_intersection(self.grad_a, self.grad_b)
        assert result is not None

    def test_gradient_subtraction_accessible(self):
        """gradient_subtraction should be callable."""
        result = gradient_subtraction(self.grad_a, self.grad_b)
        assert result is not None

    def test_gradient_smooth_union_accessible(self):
        """gradient_smooth_union should be callable."""
        result = gradient_smooth_union(self.grad_a, self.grad_b, k=0.5)
        assert result is not None

    def test_gradient_smooth_intersection_accessible(self):
        """gradient_smooth_intersection should be callable."""
        result = gradient_smooth_intersection(self.grad_a, self.grad_b, k=0.5)
        assert result is not None

    def test_gradient_smooth_subtraction_accessible(self):
        """gradient_smooth_subtraction should be callable."""
        result = gradient_smooth_subtraction(self.grad_a, self.grad_b, k=0.5)
        assert result is not None


# =============================================================================
# Test 26-30: Validation Utilities
# =============================================================================


class TestValidationUtilities:
    """Tests for validation utility functions."""

    def test_central_difference_gradient_returns_vec3(self):
        """central_difference_gradient should return Vec3."""
        def sdf(p): return math.sqrt(p.x**2 + p.y**2 + p.z**2) - 1.0

        result = central_difference_gradient(sdf, Vec3(1.5, 0.0, 0.0))
        assert isinstance(result, Vec3)

    def test_validate_gradient_returns_tuple(self):
        """validate_gradient should return (bool, float)."""
        vec_a = Vec3(1.0, 0.0, 0.0)
        vec_b = Vec3(1.0, 0.0, 0.0)

        result = validate_gradient(vec_a, vec_b)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    def test_validate_gradient_passes_for_equal(self):
        """validate_gradient should pass for equal vectors."""
        vec_a = Vec3(1.0, 0.0, 0.0)
        vec_b = Vec3(1.0, 0.0, 0.0)

        is_valid, error = validate_gradient(vec_a, vec_b)
        assert is_valid
        assert error < 1e-6

    def test_validate_gradient_fails_for_different(self):
        """validate_gradient should fail for significantly different vectors."""
        vec_a = Vec3(1.0, 0.0, 0.0)
        vec_b = Vec3(0.0, 1.0, 0.0)

        is_valid, error = validate_gradient(vec_a, vec_b)
        assert not is_valid
        assert error > 0.5

    def test_validate_gradient_error_is_positive(self):
        """validate_gradient error should always be non-negative."""
        vec_a = Vec3(1.0, 2.0, 3.0)
        vec_b = Vec3(-1.0, -2.0, -3.0)

        is_valid, error = validate_gradient(vec_a, vec_b)
        assert error >= 0


# =============================================================================
# Test 31-35: Performance Characteristics
# =============================================================================


class TestPerformanceCharacteristics:
    """Tests for performance characteristics."""

    def test_primitive_gradient_is_fast(self):
        """Primitive gradient should complete quickly."""
        start = time.perf_counter()
        for _ in range(1000):
            gradient_sphere(Vec3(1.5, 0.5, 0.3), 1.0)
        elapsed = time.perf_counter() - start

        # Should complete 1000 calls in under 100ms
        assert elapsed < 0.1

    def test_combinator_gradient_is_fast(self):
        """Combinator gradient should complete quickly."""
        grad_a = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)
        grad_b = GradientResult(Vec3(0.0, 1.0, 0.0), 0.5)

        start = time.perf_counter()
        for _ in range(1000):
            gradient_union(grad_a, grad_b)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1

    def test_smooth_combinator_is_fast(self):
        """Smooth combinator should complete quickly."""
        grad_a = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)
        grad_b = GradientResult(Vec3(0.0, 1.0, 0.0), 0.5)

        start = time.perf_counter()
        for _ in range(1000):
            gradient_smooth_union(grad_a, grad_b, k=0.5)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1

    def test_gradient_result_creation_is_cheap(self):
        """GradientResult creation should be cheap."""
        start = time.perf_counter()
        for _ in range(10000):
            GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1

    def test_no_allocation_in_hot_path(self):
        """Gradient computation should minimize allocations."""
        # This is a best-effort test - we can't directly measure allocations
        # but we can check that repeated calls don't slow down
        times = []
        for _ in range(10):
            start = time.perf_counter()
            for _ in range(100):
                gradient_sphere(Vec3(1.5, 0.5, 0.3), 1.0)
            times.append(time.perf_counter() - start)

        # Times should be consistent (no memory pressure)
        avg = sum(times) / len(times)
        for t in times:
            assert t < avg * 2.0


# =============================================================================
# Test 36-40: Edge Case Handling
# =============================================================================


class TestEdgeCaseHandling:
    """Tests for edge case handling."""

    def test_very_small_radius(self):
        """Should handle very small radius without error."""
        result = gradient_sphere(Vec3(0.001, 0.0, 0.0), 0.0001)
        assert math.isfinite(result.distance)

    def test_very_large_radius(self):
        """Should handle very large radius without error."""
        result = gradient_sphere(Vec3(1000.0, 0.0, 0.0), 10000.0)
        assert math.isfinite(result.distance)

    def test_negative_coordinates(self):
        """Should handle negative coordinates."""
        result = gradient_sphere(Vec3(-5.0, -3.0, -2.0), 1.0)
        assert math.isfinite(result.distance)

    def test_zero_smoothness(self):
        """Should handle zero smoothness in smooth combinators."""
        grad_a = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)
        grad_b = GradientResult(Vec3(0.0, 1.0, 0.0), 0.5)

        # k=0 should behave like hard combinator
        result = gradient_smooth_union(grad_a, grad_b, k=0.0)
        assert math.isfinite(result.distance)

    def test_large_smoothness(self):
        """Should handle large smoothness values."""
        grad_a = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)
        grad_b = GradientResult(Vec3(0.0, 1.0, 0.0), 0.5)

        result = gradient_smooth_union(grad_a, grad_b, k=10.0)
        assert math.isfinite(result.distance)


# =============================================================================
# Test 41-45: GradientResult Immutability
# =============================================================================


class TestGradientResultImmutability:
    """Tests for GradientResult immutability."""

    def test_gradient_result_is_frozen(self):
        """GradientResult should be immutable."""
        result = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)

        with pytest.raises(AttributeError):
            result.distance = 1.0

    def test_combinator_result_is_frozen(self):
        """CombinatorGradientResult should be immutable."""
        result = CombinatorGradientResult(Vec3(1.0, 0.0, 0.0), 0.5, 0)

        with pytest.raises(AttributeError):
            result.winner_id = 1

    def test_vec3_in_result_is_original(self):
        """Vec3 in GradientResult should be the original Vec3."""
        vec = Vec3(1.0, 0.0, 0.0)
        result = GradientResult(vec, 0.5)

        # Same object (Vec3 is immutable)
        assert result.gradient.x == vec.x
        assert result.gradient.y == vec.y
        assert result.gradient.z == vec.z

    def test_result_can_be_hashed(self):
        """GradientResult should be hashable (frozen dataclass)."""
        result = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)
        # Should not raise
        hash(result)

    def test_result_can_be_in_set(self):
        """GradientResult should be usable in sets."""
        result1 = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)
        result2 = GradientResult(Vec3(1.0, 0.0, 0.0), 0.5)

        result_set = {result1, result2}
        assert len(result_set) == 1  # Should be equal and deduplicated
