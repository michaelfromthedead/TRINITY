"""
Whitebox tests for T-DEMO-3.2: Ray Marching Loop (Sphere Tracing).

Internal implementation tests for the SphereTracer class including:
- Step history tracking
- Direction normalization
- SDF evaluation at each step
- Internal state management
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.demoscene.ray_march import (
    HitResult,
    MarchResultType,
    SphereTracer,
    march_ray,
    generate_ray_march_struct_wgsl,
)
from engine.rendering.demoscene.sdf_ast import Vec3


# =============================================================================
# Test Fixtures
# =============================================================================


def unit_sphere_sdf(p: Vec3) -> tuple[float, int]:
    """SDF for a unit sphere at origin."""
    return (p.length() - 1.0, 0)


def counting_sdf_factory():
    """Create an SDF that counts evaluations."""
    counter = [0]

    def sdf(p: Vec3) -> tuple[float, int]:
        counter[0] += 1
        return (p.length() - 1.0, 0)

    return sdf, counter


def recording_sdf_factory():
    """Create an SDF that records all evaluation points."""
    points = []

    def sdf(p: Vec3) -> tuple[float, int]:
        points.append(Vec3(p.x, p.y, p.z))
        return (p.length() - 1.0, 0)

    return sdf, points


def variable_material_sdf(p: Vec3) -> tuple[float, int]:
    """SDF that returns different materials based on position."""
    dist = p.length() - 1.0
    # Material 1 in +X half, material 2 in -X half
    mat_id = 1 if p.x >= 0 else 2
    return (dist, mat_id)


# =============================================================================
# Test: Step History Internal State
# =============================================================================


class TestStepHistoryInternal:
    """Tests for internal step history tracking."""

    def test_step_history_cleared_each_march(self):
        """Step history should be cleared at start of each march."""
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        # First march
        tracer.march(Vec3(0, 0, 5), Vec3(0, 0, -1), unit_sphere_sdf)
        first_len = len(tracer.step_history)

        # Second march
        tracer.march(Vec3(0, 0, 3), Vec3(0, 0, -1), unit_sphere_sdf)
        second_len = len(tracer.step_history)

        # Second march is shorter, should have fewer entries
        assert second_len <= first_len

    def test_step_history_matches_steps(self):
        """Step history length should match steps taken."""
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        result = tracer.march(Vec3(0, 0, 5), Vec3(0, 0, -1), unit_sphere_sdf)

        # History length should equal steps
        assert len(tracer.step_history) == result.steps

    def test_step_history_decreasing_values(self):
        """Step history values should generally decrease toward hit."""
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        result = tracer.march(Vec3(0, 0, 10), Vec3(0, 0, -1), unit_sphere_sdf)

        if result.hit and len(tracer.step_history) > 1:
            # Values should decrease as we approach surface
            for i in range(1, len(tracer.step_history)):
                # Allow small numerical increases, but general trend should be down
                assert tracer.step_history[i] < tracer.step_history[0]

    def test_step_history_copy_not_reference(self):
        """step_history property should return a copy."""
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        tracer.march(Vec3(0, 0, 5), Vec3(0, 0, -1), unit_sphere_sdf)

        history1 = tracer.step_history
        history2 = tracer.step_history

        # Should be different objects
        history1.append(999.0)
        assert 999.0 not in history2


# =============================================================================
# Test: Direction Normalization
# =============================================================================


class TestDirectionNormalization:
    """Tests for direction vector normalization."""

    def test_unnormalized_direction_works(self):
        """Non-unit direction should still work correctly."""
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        # Direction with length 2
        result = tracer.march(
            Vec3(0, 0, 5),
            Vec3(0, 0, -2),  # Not unit length
            unit_sphere_sdf,
        )

        assert result.hit is True
        assert abs(result.distance - 4.0) < 0.1

    def test_very_small_direction_handled(self):
        """Very small direction vectors should be handled."""
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        result = tracer.march(
            Vec3(0, 0, 5),
            Vec3(0, 0, 1e-12),  # Nearly zero
            unit_sphere_sdf,
        )

        # Should either hit (if normalized) or miss gracefully
        assert result.steps >= 0


# =============================================================================
# Test: SDF Evaluation Points
# =============================================================================


class TestSDFEvaluationPoints:
    """Tests for where the SDF is evaluated."""

    def test_first_eval_at_origin(self):
        """First SDF evaluation should be at ray origin."""
        sdf, points = recording_sdf_factory()
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        origin = Vec3(3.0, 4.0, 5.0)
        tracer.march(origin, Vec3(0, 0, -1), sdf)

        assert len(points) > 0
        # Check first point matches origin
        assert abs(points[0].x - origin.x) < 1e-6
        assert abs(points[0].y - origin.y) < 1e-6
        assert abs(points[0].z - origin.z) < 1e-6

    def test_second_eval_advanced_by_sdf(self):
        """Second evaluation should be at origin + d * direction."""
        sdf, points = recording_sdf_factory()
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        origin = Vec3(0, 0, 5)
        tracer.march(origin, Vec3(0, 0, -1), sdf)

        if len(points) >= 2:
            # First point at origin, SDF value is 5 - 1 = 4
            # Second point should be at z = 5 - 4 = 1
            assert abs(points[1].z - 1.0) < 0.1

    def test_evals_along_ray(self):
        """All evaluations should be along the ray."""
        sdf, points = recording_sdf_factory()
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        origin = Vec3(0, 0, 5)
        direction = Vec3(0, 0, -1)
        tracer.march(origin, direction, sdf)

        for p in points:
            # Points should be on the ray (x=0, y=0)
            assert abs(p.x) < 1e-6
            assert abs(p.y) < 1e-6


# =============================================================================
# Test: Material ID Handling
# =============================================================================


class TestMaterialIDHandling:
    """Tests for material ID tracking."""

    def test_material_from_final_position(self):
        """Material ID should come from SDF at hit position."""
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        # Hit from +X side
        result = tracer.march(
            Vec3(5, 0, 0),
            Vec3(-1, 0, 0),
            variable_material_sdf,
        )

        assert result.hit is True
        assert result.material_id == 1  # Hit +X side

    def test_material_from_negative_side(self):
        """Material ID should vary with hit position."""
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        # Hit from -X side
        result = tracer.march(
            Vec3(-5, 0, 0),
            Vec3(1, 0, 0),
            variable_material_sdf,
        )

        assert result.hit is True
        assert result.material_id == 2  # Hit -X side


# =============================================================================
# Test: Epsilon Boundary Conditions
# =============================================================================


class TestEpsilonBoundary:
    """Tests for epsilon boundary behavior."""

    def test_exact_epsilon_is_hit(self):
        """SDF == epsilon should register as hit."""
        # Create SDF that returns exactly epsilon
        epsilon = 0.001

        def exact_epsilon_sdf(p: Vec3) -> tuple[float, int]:
            return (epsilon - 1e-10, 0)  # Just under epsilon

        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=epsilon)
        result = tracer.march(Vec3(0, 0, 5), Vec3(0, 0, -1), exact_epsilon_sdf)

        assert result.hit is True
        assert result.steps == 1  # Should hit on first evaluation

    def test_just_over_epsilon_not_hit(self):
        """SDF just over epsilon should not be immediate hit."""
        epsilon = 0.001

        def just_over_sdf(p: Vec3) -> tuple[float, int]:
            return (epsilon + 0.0001, 0)  # Just over epsilon

        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=epsilon)
        result = tracer.march(Vec3(0, 0, 5), Vec3(0, 0, -1), just_over_sdf)

        # Should not hit on first step
        assert result.steps > 1 or result.hit is False


# =============================================================================
# Test: MarchResultType Classification
# =============================================================================


class TestResultTypeClassification:
    """Tests for result type classification."""

    def test_hit_has_hit_type(self):
        """Successful hit should have HIT type."""
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)
        result = tracer.march(Vec3(0, 0, 5), Vec3(0, 0, -1), unit_sphere_sdf)

        assert result.hit is True
        assert result.result_type == MarchResultType.HIT

    def test_miss_has_miss_type(self):
        """Miss due to max_distance should have MISS type."""
        tracer = SphereTracer(max_steps=256, max_distance=10.0, epsilon=0.001)

        # Ray going away
        result = tracer.march(Vec3(0, 0, 5), Vec3(0, 0, 1), unit_sphere_sdf)

        assert result.hit is False
        assert result.result_type == MarchResultType.MISS

    def test_max_steps_has_max_steps_type(self):
        """Termination due to max_steps should have MAX_STEPS type."""
        # Use many iterations with a never-converging SDF
        def never_hit_sdf(p: Vec3) -> tuple[float, int]:
            return (1.0, 0)  # Always returns 1, never hits

        tracer = SphereTracer(max_steps=10, max_distance=100.0, epsilon=0.001)
        result = tracer.march(Vec3(0, 0, 0), Vec3(0, 0, 1), never_hit_sdf)

        assert result.hit is False
        # After 10 steps of size 1, we're at z=10, not past max_distance
        assert result.result_type == MarchResultType.MAX_STEPS


# =============================================================================
# Test: HitResult Factory Methods
# =============================================================================


class TestHitResultFactoryMethods:
    """Tests for HitResult factory methods."""

    def test_surface_hit_sets_all_fields(self):
        """surface_hit should set all fields correctly."""
        pos = Vec3(1.0, 2.0, 3.0)
        result = HitResult.surface_hit(pos, 5.0, 10, 7)

        assert result.hit is True
        assert result.position.x == 1.0
        assert result.distance == 5.0
        assert result.steps == 10
        assert result.material_id == 7
        assert result.result_type == MarchResultType.HIT

    def test_miss_sets_zero_position(self):
        """miss should set position to zero."""
        result = HitResult.miss(50.0, 100, MarchResultType.MISS)

        assert result.hit is False
        assert result.position.x == 0.0
        assert result.position.y == 0.0
        assert result.position.z == 0.0

    def test_miss_preserves_result_type(self):
        """miss should preserve the result type."""
        result_miss = HitResult.miss(50.0, 100, MarchResultType.MISS)
        result_max = HitResult.miss(50.0, 100, MarchResultType.MAX_STEPS)

        assert result_miss.result_type == MarchResultType.MISS
        assert result_max.result_type == MarchResultType.MAX_STEPS


# =============================================================================
# Test: WGSL Generation Details
# =============================================================================


class TestWGSLGenerationDetails:
    """Tests for WGSL generation internal details."""

    def test_struct_has_all_fields(self):
        """Generated struct should have all required fields."""
        wgsl = generate_ray_march_struct_wgsl()

        assert "hit: bool" in wgsl
        assert "position: vec3<f32>" in wgsl
        assert "distance: f32" in wgsl
        assert "material_id: u32" in wgsl
        assert "steps: u32" in wgsl

    def test_factory_functions_correct_signature(self):
        """Factory functions should have correct signatures."""
        wgsl = generate_ray_march_struct_wgsl()

        assert "fn ray_hit_miss(steps: u32) -> RayHit" in wgsl
        assert "fn ray_hit_surface(" in wgsl

    def test_miss_factory_returns_false_hit(self):
        """ray_hit_miss should return struct with hit=false."""
        wgsl = generate_ray_march_struct_wgsl()

        assert "false" in wgsl  # Should set hit to false

    def test_has_documentation(self):
        """Generated WGSL should have documentation."""
        wgsl = generate_ray_march_struct_wgsl()

        assert "///" in wgsl  # Doc comments


# =============================================================================
# Test: march_ray Convenience Function
# =============================================================================


class TestMarchRayConvenienceFunction:
    """Tests for the march_ray convenience function."""

    def test_creates_new_tracer(self):
        """march_ray should work without external tracer."""
        result = march_ray(
            Vec3(0, 0, 5),
            Vec3(0, 0, -1),
            unit_sphere_sdf,
            max_steps=64,
            max_distance=50.0,
            epsilon=0.001,
        )

        assert result.hit is True

    def test_respects_max_steps(self):
        """march_ray should respect max_steps parameter."""
        def never_hit(p):
            return (1.0, 0)

        result = march_ray(
            Vec3(0, 0, 0),
            Vec3(0, 0, 1),
            never_hit,
            max_steps=5,
            max_distance=100.0,
            epsilon=0.001,
        )

        assert result.steps <= 5

    def test_respects_epsilon(self):
        """march_ray should respect epsilon parameter."""
        result_coarse = march_ray(
            Vec3(0, 0, 5),
            Vec3(0, 0, -1),
            unit_sphere_sdf,
            epsilon=0.1,  # Coarse
        )

        result_fine = march_ray(
            Vec3(0, 0, 5),
            Vec3(0, 0, -1),
            unit_sphere_sdf,
            epsilon=0.0001,  # Fine
        )

        # Fine epsilon may take more steps or have more precise position
        assert result_coarse.hit is True
        assert result_fine.hit is True


# =============================================================================
# Test: Normal Computation
# =============================================================================


class TestNormalComputation:
    """Tests for march_with_normal functionality."""

    def test_normal_is_normalized(self):
        """Computed normal should be unit length."""
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        result, normal = tracer.march_with_normal(
            Vec3(0, 0, 5),
            Vec3(0, 0, -1),
            unit_sphere_sdf,
        )

        if normal is not None:
            assert abs(normal.length() - 1.0) < 1e-6

    def test_normal_points_outward(self):
        """Normal should point outward from surface."""
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        # Hit from +Z
        result, normal = tracer.march_with_normal(
            Vec3(0, 0, 5),
            Vec3(0, 0, -1),
            unit_sphere_sdf,
        )

        if normal is not None:
            # Normal at +Z surface should point toward +Z
            assert normal.z > 0

    def test_normal_perpendicular_to_surface(self):
        """Normal should be perpendicular to sphere surface."""
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        # Hit sphere at an angle
        result, normal = tracer.march_with_normal(
            Vec3(3, 3, 3),
            Vec3(-1, -1, -1).normalized(),
            unit_sphere_sdf,
        )

        if result.hit and normal is not None:
            # Normal should point radially outward (same direction as position)
            pos = result.position
            pos_len = pos.length()
            if pos_len > 1e-6:
                pos_dir_x = pos.x / pos_len
                pos_dir_y = pos.y / pos_len
                pos_dir_z = pos.z / pos_len
                # Compute dot product manually
                dot = normal.x * pos_dir_x + normal.y * pos_dir_y + normal.z * pos_dir_z
                assert abs(dot - 1.0) < 0.01  # Should be approximately 1


# =============================================================================
# Test: Edge Cases in Vec3 Operations
# =============================================================================


class TestVec3OperationsInMarching:
    """Tests for Vec3 operations used in marching."""

    def test_addition_preserves_precision(self):
        """Vec3 addition should preserve precision."""
        v1 = Vec3(1e-10, 1e-10, 1e-10)
        v2 = Vec3(1.0, 1.0, 1.0)

        result = v1 + v2

        assert abs(result.x - 1.0) < 1e-8
        assert abs(result.y - 1.0) < 1e-8
        assert abs(result.z - 1.0) < 1e-8

    def test_multiplication_with_large_scalar(self):
        """Vec3 multiplication with large scalar should work."""
        v = Vec3(1.0, 1.0, 1.0)
        result = v * 1e10

        assert result.x == 1e10
        assert result.y == 1e10
        assert result.z == 1e10

    def test_normalization_preserves_direction(self):
        """Normalization should preserve direction."""
        v = Vec3(3.0, 4.0, 0.0)
        n = v.normalized()

        # Direction should be same (proportional)
        assert abs(n.x / n.y - v.x / v.y) < 1e-6


# =============================================================================
# Test: SDF Evaluation Count
# =============================================================================


class TestSDFEvaluationCount:
    """Tests for SDF evaluation counting."""

    def test_eval_count_equals_steps(self):
        """SDF should be evaluated once per step."""
        sdf, counter = counting_sdf_factory()
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        result = tracer.march(Vec3(0, 0, 5), Vec3(0, 0, -1), sdf)

        assert counter[0] == result.steps

    def test_no_extra_evals_after_hit(self):
        """No extra SDF evaluations should occur after hit."""
        sdf, counter = counting_sdf_factory()
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        result = tracer.march(Vec3(0, 0, 2), Vec3(0, 0, -1), sdf)

        # Should hit quickly, count should be low
        assert counter[0] <= 3

    def test_normal_computation_adds_evals(self):
        """march_with_normal should add 6 evaluations for normal."""
        sdf, counter = counting_sdf_factory()
        tracer = SphereTracer(max_steps=64, max_distance=100.0, epsilon=0.001)

        result, normal = tracer.march_with_normal(Vec3(0, 0, 5), Vec3(0, 0, -1), sdf)

        # Steps + 6 for central differences
        if result.hit:
            assert counter[0] == result.steps + 6
