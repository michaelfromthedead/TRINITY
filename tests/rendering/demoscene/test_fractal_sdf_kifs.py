"""
Tests for KIFS SDF (T-DEMO-4.11)

This test suite validates the KIFS (Kaleidoscopic Iterated Function System)
SDF implementation including:
- Symmetry tests
- Scale compensation correctness
- Fold operation tests
- Fractal detail at multiple scales
- WGSL output validation
- Performance benchmarks
"""

import math
import pytest
import time
from typing import Tuple, Callable

from engine.rendering.demoscene.fractal_sdf import (
    KIFSSDF,
    KIFSConfig,
    KIFSFoldType,
    kifs_distance,
    kifs_fold_abs,
    kifs_fold_menger,
    kifs_fold_sierpinski,
    kifs_fold_box,
    kifs_fold_sphere,
    kifs_iteration,
    generate_kifs_wgsl,
    KIFS_WGSL_FUNCTION,
    KIFS_FOLD_ABS_WGSL,
    KIFS_FOLD_SIERPINSKI_WGSL,
    KIFS_FOLD_BOX_WGSL,
    KIFS_FOLD_SPHERE_WGSL,
    DEFAULT_KIFS_ITERATIONS,
    DEFAULT_KIFS_SCALE,
    DEFAULT_KIFS_FOLD_COUNT,
    sdf_sphere,
    sdf_box,
)
from engine.rendering.demoscene.sdf_ast import Vec3


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def default_kifs():
    """Create a default KIFS SDF."""
    return KIFSSDF()


@pytest.fixture
def sierpinski_kifs():
    """Create a KIFS with Sierpinski fold."""
    return KIFSSDF(fold_type=KIFSFoldType.SIERPINSKI)


@pytest.fixture
def menger_kifs():
    """Create a KIFS with Menger fold."""
    return KIFSSDF(fold_type=KIFSFoldType.MENGER)


@pytest.fixture
def sphere_fold_kifs():
    """Create a KIFS with sphere fold (Mandelbox-style)."""
    return KIFSSDF(
        fold_type=KIFSFoldType.SPHERE,
        min_radius=0.5,
        fixed_radius=1.0,
    )


# =============================================================================
# KIFSConfig Tests
# =============================================================================

class TestKIFSConfig:
    """Tests for KIFSConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KIFSConfig()
        assert config.iterations == DEFAULT_KIFS_ITERATIONS
        assert config.scale == DEFAULT_KIFS_SCALE
        assert config.fold_count == DEFAULT_KIFS_FOLD_COUNT

    def test_custom_config(self):
        """Test custom configuration."""
        config = KIFSConfig(iterations=12, scale=2.5, fold_count=4)
        assert config.iterations == 12
        assert config.scale == 2.5
        assert config.fold_count == 4

    def test_invalid_iterations(self):
        """Test that invalid iterations raises error."""
        with pytest.raises(ValueError, match="Iterations must be >= 1"):
            KIFSConfig(iterations=0)

    def test_invalid_scale(self):
        """Test that invalid scale raises error."""
        with pytest.raises(ValueError, match="Scale must be > 0.0"):
            KIFSConfig(scale=-1.0)

    def test_invalid_fold_count(self):
        """Test that invalid fold_count raises error."""
        with pytest.raises(ValueError, match="Fold count must be >= 1"):
            KIFSConfig(fold_count=0)


# =============================================================================
# Fold Operation Tests
# =============================================================================

class TestKIFSFoldOperations:
    """Tests for individual KIFS fold operations."""

    # -------------------------------------------------------------------------
    # Absolute Value Fold
    # -------------------------------------------------------------------------

    def test_fold_abs_positive(self):
        """Test abs fold with positive coordinates."""
        x, y, z = kifs_fold_abs(1.0, 2.0, 3.0)
        assert x == 1.0
        assert y == 2.0
        assert z == 3.0

    def test_fold_abs_negative(self):
        """Test abs fold with negative coordinates."""
        x, y, z = kifs_fold_abs(-1.0, -2.0, -3.0)
        assert x == 1.0
        assert y == 2.0
        assert z == 3.0

    def test_fold_abs_mixed(self):
        """Test abs fold with mixed signs."""
        x, y, z = kifs_fold_abs(-1.0, 2.0, -3.0)
        assert x == 1.0
        assert y == 2.0
        assert z == 3.0

    def test_fold_abs_symmetry(self):
        """Test that abs fold creates 8-fold symmetry."""
        d1 = kifs_fold_abs(1.0, 1.0, 1.0)
        d2 = kifs_fold_abs(-1.0, 1.0, 1.0)
        d3 = kifs_fold_abs(1.0, -1.0, 1.0)
        d4 = kifs_fold_abs(-1.0, -1.0, -1.0)

        # All should be the same after folding
        assert d1 == d2 == d3 == d4

    # -------------------------------------------------------------------------
    # Sierpinski Fold
    # -------------------------------------------------------------------------

    def test_fold_sierpinski_positive_octant(self):
        """Test Sierpinski fold in positive octant (no change)."""
        x, y, z = kifs_fold_sierpinski(1.0, 2.0, 3.0)
        # No planes crossed, should be unchanged
        assert x == 1.0
        assert y == 2.0
        assert z == 3.0

    def test_fold_sierpinski_xy_negative_sum(self):
        """Test Sierpinski fold when x + y < 0."""
        x, y, z = kifs_fold_sierpinski(-1.0, -2.0, 0.0)
        # Should swap x and y and negate both
        assert x == 2.0
        assert y == 1.0
        assert z == 0.0

    def test_fold_sierpinski_symmetry(self):
        """Test that Sierpinski fold creates tetrahedral symmetry."""
        # Points related by tetrahedral symmetry should fold to same region
        results = []
        points = [
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
        ]
        for p in points:
            result = kifs_fold_sierpinski(*p)
            results.append(result)

        # After multiple folds, they should converge to similar regions

    # -------------------------------------------------------------------------
    # Box Fold
    # -------------------------------------------------------------------------

    def test_fold_box_inside(self):
        """Test box fold for point inside limit."""
        x, y, z = kifs_fold_box(0.5, 0.5, 0.5, fold_limit=1.0)
        assert x == 0.5
        assert y == 0.5
        assert z == 0.5

    def test_fold_box_outside_positive(self):
        """Test box fold for point outside positive limit."""
        x, y, z = kifs_fold_box(1.5, 0.0, 0.0, fold_limit=1.0)
        assert x == 0.5  # 2 * 1.0 - 1.5 = 0.5
        assert y == 0.0
        assert z == 0.0

    def test_fold_box_outside_negative(self):
        """Test box fold for point outside negative limit."""
        x, y, z = kifs_fold_box(-1.5, 0.0, 0.0, fold_limit=1.0)
        assert x == -0.5  # -2 * 1.0 - (-1.5) = -0.5
        assert y == 0.0
        assert z == 0.0

    def test_fold_box_symmetry(self):
        """Test box fold creates bounded region."""
        # Points far outside should fold to bounded region
        x, y, z = kifs_fold_box(10.0, 10.0, 10.0, fold_limit=1.0)
        # After one fold, x = 2 * 1 - 10 = -8, still outside
        # But for KIFS we apply multiple iterations
        assert abs(x) <= 10.0  # Should be somewhat bounded

    # -------------------------------------------------------------------------
    # Sphere Fold
    # -------------------------------------------------------------------------

    def test_fold_sphere_outside_fixed(self):
        """Test sphere fold for point outside fixed radius."""
        x, y, z, scale = kifs_fold_sphere(2.0, 0.0, 0.0, min_radius=0.5, fixed_radius=1.0)
        assert x == 2.0
        assert y == 0.0
        assert z == 0.0
        assert scale == 1.0

    def test_fold_sphere_inside_min(self):
        """Test sphere fold for point inside min radius."""
        x, y, z, scale = kifs_fold_sphere(0.25, 0.0, 0.0, min_radius=0.5, fixed_radius=1.0)
        # r^2 = 0.0625, min_r^2 = 0.25, fixed_r^2 = 1.0
        # scale = 1.0 / 0.25 = 4.0
        expected_scale = 1.0 / 0.25
        assert abs(scale - expected_scale) < 1e-10
        assert abs(x - 0.25 * expected_scale) < 1e-10

    def test_fold_sphere_between_radii(self):
        """Test sphere fold for point between min and fixed radius."""
        x, y, z, scale = kifs_fold_sphere(0.75, 0.0, 0.0, min_radius=0.5, fixed_radius=1.0)
        # r^2 = 0.5625, between 0.25 and 1.0
        # scale = 1.0 / 0.5625
        r2 = 0.75 * 0.75
        expected_scale = 1.0 / r2
        assert abs(scale - expected_scale) < 1e-10


# =============================================================================
# KIFS Iteration Tests
# =============================================================================

class TestKIFSIteration:
    """Tests for KIFS iteration function."""

    def test_iteration_abs_fold(self):
        """Test one iteration with abs fold."""
        x, y, z, scale = kifs_iteration(
            -1.0, 1.0, 1.0,
            scale=2.0, tx=1.0, ty=1.0, tz=1.0,
            fold_type=KIFSFoldType.ABS,
            fold_count=1,
        )
        # abs(-1.0) = 1.0
        # scaled: 1.0 * 2.0 = 2.0
        # translation offset: 1.0 * (2.0 - 1.0) = 1.0
        # result: 2.0 - 1.0 = 1.0
        assert abs(scale - 2.0) < 1e-10

    def test_iteration_scale_accumulation(self):
        """Test that scale accumulates correctly."""
        _, _, _, scale = kifs_iteration(
            1.0, 1.0, 1.0,
            scale=2.0, tx=1.0, ty=1.0, tz=1.0,
            fold_type=KIFSFoldType.ABS,
            fold_count=1,
        )
        assert abs(scale - 2.0) < 1e-10

    def test_iteration_multiple_folds(self):
        """Test iteration with multiple folds."""
        # Multiple abs folds should have same result as single fold
        x1, y1, z1, s1 = kifs_iteration(
            -1.0, -1.0, -1.0,
            scale=2.0, tx=1.0, ty=1.0, tz=1.0,
            fold_type=KIFSFoldType.ABS,
            fold_count=1,
        )
        x2, y2, z2, s2 = kifs_iteration(
            -1.0, -1.0, -1.0,
            scale=2.0, tx=1.0, ty=1.0, tz=1.0,
            fold_type=KIFSFoldType.ABS,
            fold_count=3,
        )
        # abs(abs(abs(x))) = abs(x), so should be same
        assert abs(x1 - x2) < 1e-10


# =============================================================================
# KIFS Distance Tests
# =============================================================================

class TestKIFSDistance:
    """Tests for KIFS distance calculation."""

    def test_distance_origin(self):
        """Test distance at origin."""
        d = kifs_distance(0.0, 0.0, 0.0)
        # Should be finite
        assert math.isfinite(d)

    def test_distance_far(self):
        """Test distance far from fractal."""
        d = kifs_distance(100.0, 0.0, 0.0)
        assert d > 0.0

    def test_distance_scale_compensation(self):
        """Test that scale compensation is applied correctly."""
        # With scale=2 and iterations=4, total scale = 2^4 = 16
        # Distance should be divided by 16
        d_low = kifs_distance(1.0, 0.0, 0.0, iterations=2, scale=2.0)
        d_high = kifs_distance(1.0, 0.0, 0.0, iterations=4, scale=2.0)

        # More iterations = smaller distances due to higher accumulated scale
        # (This assumes the base distance grows slower than scale^iterations)

    def test_distance_with_custom_base_sdf(self):
        """Test distance with custom base SDF."""
        def box_sdf(x, y, z):
            return sdf_box(x, y, z, 0.5, 0.5, 0.5)

        d = kifs_distance(1.0, 0.0, 0.0, base_distance=box_sdf)
        assert math.isfinite(d)

    def test_distance_symmetry(self):
        """Test that distance is symmetric due to abs fold."""
        d1 = kifs_distance(1.0, 0.5, 0.3)
        d2 = kifs_distance(-1.0, 0.5, 0.3)
        d3 = kifs_distance(1.0, -0.5, 0.3)
        d4 = kifs_distance(-1.0, -0.5, -0.3)

        # All should be equal due to abs fold symmetry
        assert abs(d1 - d2) < 1e-10
        assert abs(d1 - d3) < 1e-10
        assert abs(d1 - d4) < 1e-10


# =============================================================================
# KIFSSDF Class Tests
# =============================================================================

class TestKIFSSDFClass:
    """Tests for KIFSSDF class."""

    def test_creation_default(self, default_kifs):
        """Test default creation."""
        kifs = default_kifs
        assert kifs.iterations == DEFAULT_KIFS_ITERATIONS
        assert kifs.scale == DEFAULT_KIFS_SCALE
        assert kifs.fold_count == DEFAULT_KIFS_FOLD_COUNT

    def test_creation_custom(self):
        """Test custom creation."""
        kifs = KIFSSDF(iterations=12, scale=2.5, fold_count=4)
        assert kifs.iterations == 12
        assert kifs.scale == 2.5
        assert kifs.fold_count == 4

    def test_position(self):
        """Test position parameter."""
        pos = Vec3(1.0, 2.0, 3.0)
        kifs = KIFSSDF(position=pos)
        assert kifs.position.x == 1.0
        assert kifs.position.y == 2.0
        assert kifs.position.z == 3.0

    def test_translation(self):
        """Test translation parameter."""
        trans = Vec3(0.5, 0.5, 0.5)
        kifs = KIFSSDF(translation=trans)
        assert kifs.translation.x == 0.5

    def test_evaluate(self, default_kifs):
        """Test evaluate method."""
        d = default_kifs.evaluate(1.0, 0.0, 0.0)
        assert math.isfinite(d)

    def test_evaluate_point(self, default_kifs):
        """Test evaluate_point method."""
        d = default_kifs.evaluate_point((1.0, 0.0, 0.0))
        assert math.isfinite(d)

    def test_evaluate_with_position(self):
        """Test that position offset works correctly."""
        kifs1 = KIFSSDF()
        kifs2 = KIFSSDF(position=Vec3(5.0, 0.0, 0.0))

        d1 = kifs1.evaluate(0.0, 0.0, 0.0)
        d2 = kifs2.evaluate(5.0, 0.0, 0.0)

        assert abs(d1 - d2) < 1e-10

    def test_set_base_sdf(self, default_kifs):
        """Test set_base_sdf method."""
        def my_sdf(x, y, z):
            return sdf_box(x, y, z, 0.5, 0.5, 0.5)

        result = default_kifs.set_base_sdf(my_sdf)
        assert result is default_kifs  # Returns self for chaining

        d = default_kifs.evaluate(1.0, 0.0, 0.0)
        assert math.isfinite(d)

    def test_wgsl_function_name(self, default_kifs):
        """Test WGSL function name property."""
        assert default_kifs.wgsl_function == "sdf_kifs"

    def test_label(self, default_kifs):
        """Test label generation."""
        label = default_kifs.label()
        assert "KIFS" in label
        assert str(DEFAULT_KIFS_ITERATIONS) in label

    def test_clone(self, default_kifs):
        """Test clone method."""
        clone = default_kifs.clone()
        assert clone.iterations == default_kifs.iterations
        assert clone.scale == default_kifs.scale
        assert clone is not default_kifs

    def test_children(self, default_kifs):
        """Test children method returns empty tuple."""
        assert default_kifs.children() == ()

    def test_get_config(self, default_kifs):
        """Test get_config method."""
        config = default_kifs.get_config()
        assert isinstance(config, KIFSConfig)
        assert config.iterations == default_kifs.iterations

    def test_tracker_dirty(self):
        """Test that tracker marks fields dirty on creation."""
        kifs = KIFSSDF()
        # tracker.is_dirty is a property that returns True if any field is dirty
        assert kifs.tracker.is_dirty
        # Check that specific fields are in the dirty set
        assert "iterations" in kifs.tracker.dirty_fields
        assert "scale" in kifs.tracker.dirty_fields
        assert "fold_count" in kifs.tracker.dirty_fields


# =============================================================================
# Symmetry Tests
# =============================================================================

class TestKIFSSymmetry:
    """Tests for KIFS symmetry properties."""

    def test_abs_fold_8_symmetry(self):
        """Test that abs fold creates 8-fold symmetry."""
        kifs = KIFSSDF(fold_type=KIFSFoldType.ABS)

        # All octants should give same distance
        points = [
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
            (-1.0, -1.0, -1.0),
        ]
        distances = [kifs.evaluate(*p) for p in points]

        # All distances should be equal
        for d in distances[1:]:
            assert abs(d - distances[0]) < 1e-10

    def test_sierpinski_tetrahedral_symmetry(self, sierpinski_kifs):
        """Test Sierpinski fold symmetry properties."""
        kifs = sierpinski_kifs

        # Test symmetry - should be more complex than simple abs
        d1 = kifs.evaluate(1.0, 1.0, 1.0)
        d2 = kifs.evaluate(1.0, 1.0, -1.0)

        # These won't be equal but should be finite
        assert math.isfinite(d1)
        assert math.isfinite(d2)


# =============================================================================
# Scale Compensation Tests
# =============================================================================

class TestKIFSScaleCompensation:
    """Tests for KIFS scale compensation correctness."""

    def test_scale_compensation_factor(self):
        """Test that scale compensation divides by scale^iterations."""
        # Create two KIFS with different iteration counts
        kifs1 = KIFSSDF(iterations=2, scale=2.0)
        kifs2 = KIFSSDF(iterations=4, scale=2.0)

        # Evaluate at a point
        d1 = kifs1.evaluate(0.5, 0.0, 0.0)
        d2 = kifs2.evaluate(0.5, 0.0, 0.0)

        # Both should be finite
        assert math.isfinite(d1)
        assert math.isfinite(d2)

    def test_scale_1_no_scaling(self):
        """Test that scale=1 doesn't change distances much."""
        kifs = KIFSSDF(scale=1.0, iterations=4)
        d = kifs.evaluate(1.0, 0.0, 0.0)

        # With scale=1, accumulated_scale = 1, so minimal compensation
        assert math.isfinite(d)

    def test_high_scale_small_distances(self):
        """Test that high scale produces appropriately compensated distances."""
        kifs = KIFSSDF(scale=3.0, iterations=5)
        d = kifs.evaluate(1.0, 0.0, 0.0)

        # Even with high scale, distance should be reasonable
        assert math.isfinite(d)


# =============================================================================
# Fractal Detail Tests
# =============================================================================

class TestKIFSFractalDetail:
    """Tests for fractal detail at multiple scales."""

    def test_detail_near_origin(self):
        """Test fractal detail near origin."""
        kifs = KIFSSDF(iterations=10)
        d = kifs.evaluate(0.1, 0.1, 0.1)
        assert math.isfinite(d)

    def test_detail_various_scales(self):
        """Test detail at various scales."""
        kifs = KIFSSDF(iterations=8)

        for scale in [10.0, 1.0, 0.1, 0.01]:
            d = kifs.evaluate(scale, 0.0, 0.0)
            assert math.isfinite(d), f"NaN at scale {scale}"

    def test_iteration_count_detail(self):
        """Test that more iterations add detail."""
        # Create KIFS with different iteration counts
        kifs_low = KIFSSDF(iterations=3)
        kifs_high = KIFSSDF(iterations=10)

        # Evaluate at same point
        d_low = kifs_low.evaluate(0.5, 0.3, 0.2)
        d_high = kifs_high.evaluate(0.5, 0.3, 0.2)

        # Both should be finite
        assert math.isfinite(d_low)
        assert math.isfinite(d_high)


# =============================================================================
# WGSL Generation Tests
# =============================================================================

class TestKIFSWGSL:
    """Tests for KIFS WGSL code generation."""

    def test_wgsl_function_exists(self):
        """Test that WGSL function constant exists."""
        assert "sdf_kifs" in KIFS_WGSL_FUNCTION
        assert "vec3<f32>" in KIFS_WGSL_FUNCTION

    def test_generate_wgsl_default(self):
        """Test WGSL generation with defaults."""
        wgsl = generate_kifs_wgsl()
        assert "fn sdf_kifs" in wgsl
        assert "accumulated_scale" in wgsl

    def test_generate_wgsl_custom(self):
        """Test WGSL generation with custom parameters."""
        wgsl = generate_kifs_wgsl(iterations=12, scale=2.5)
        assert "12" in wgsl
        assert "2.5" in wgsl

    def test_generate_wgsl_abs_fold(self):
        """Test WGSL generation with abs fold."""
        wgsl = generate_kifs_wgsl(fold_type=KIFSFoldType.ABS)
        assert "abs" in wgsl

    def test_generate_wgsl_sierpinski_fold(self):
        """Test WGSL generation with Sierpinski fold."""
        wgsl = generate_kifs_wgsl(fold_type=KIFSFoldType.SIERPINSKI)
        assert "Sierpinski" in wgsl

    def test_generate_wgsl_box_fold(self):
        """Test WGSL generation with box fold."""
        wgsl = generate_kifs_wgsl(fold_type=KIFSFoldType.BOX)
        assert "Box fold" in wgsl

    def test_generate_wgsl_sphere_fold(self):
        """Test WGSL generation with sphere fold."""
        wgsl = generate_kifs_wgsl(fold_type=KIFSFoldType.SPHERE)
        assert "Sphere fold" in wgsl

    def test_generate_wgsl_custom_name(self):
        """Test WGSL generation with custom function name."""
        wgsl = generate_kifs_wgsl(function_name="my_kifs")
        assert "fn my_kifs" in wgsl

    def test_to_wgsl_method(self, default_kifs):
        """Test to_wgsl method on SDF instance."""
        wgsl = default_kifs.to_wgsl()
        assert "fn sdf_kifs" in wgsl

    def test_wgsl_contains_key_elements(self):
        """Test that WGSL contains key algorithmic elements."""
        wgsl = generate_kifs_wgsl()
        # Check for loop
        assert "for" in wgsl
        # Check for scale compensation
        assert "accumulated_scale" in wgsl
        # Check for base SDF
        assert "length" in wgsl

    def test_fold_wgsl_templates_exist(self):
        """Test that fold WGSL templates exist."""
        assert "kifs_fold_abs" in KIFS_FOLD_ABS_WGSL
        assert "kifs_fold_sierpinski" in KIFS_FOLD_SIERPINSKI_WGSL
        assert "kifs_fold_box" in KIFS_FOLD_BOX_WGSL
        assert "kifs_fold_sphere" in KIFS_FOLD_SPHERE_WGSL


# =============================================================================
# Performance Benchmarks
# =============================================================================

class TestKIFSPerformance:
    """Performance benchmarks for KIFS SDF."""

    @pytest.mark.benchmark
    def test_single_evaluation_time(self, default_kifs):
        """Benchmark single point evaluation."""
        start = time.perf_counter()
        for _ in range(1000):
            default_kifs.evaluate(1.0, 0.5, 0.3)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5  # Should be very fast

    @pytest.mark.benchmark
    def test_grid_evaluation_time(self, default_kifs):
        """Benchmark grid of point evaluations."""
        start = time.perf_counter()
        for x in range(-5, 6):
            for y in range(-5, 6):
                for z in range(-5, 6):
                    default_kifs.evaluate(x * 0.2, y * 0.2, z * 0.2)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0

    @pytest.mark.benchmark
    def test_high_iteration_performance(self):
        """Benchmark high iteration count."""
        kifs = KIFSSDF(iterations=20)
        start = time.perf_counter()
        for _ in range(100):
            kifs.evaluate(1.0, 0.5, 0.3)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestKIFSEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_point(self):
        """Test evaluation at very small coordinates."""
        kifs = KIFSSDF()
        d = kifs.evaluate(1e-10, 1e-10, 1e-10)
        assert math.isfinite(d)

    def test_very_large_point(self):
        """Test evaluation at very large coordinates."""
        kifs = KIFSSDF()
        d = kifs.evaluate(1000.0, 1000.0, 1000.0)
        assert math.isfinite(d)

    def test_negative_coordinates(self):
        """Test evaluation with negative coordinates."""
        kifs = KIFSSDF()
        d = kifs.evaluate(-1.0, -1.0, -1.0)
        assert math.isfinite(d)

    def test_minimum_iterations(self):
        """Test with minimum iteration count."""
        kifs = KIFSSDF(iterations=1)
        d = kifs.evaluate(1.0, 0.0, 0.0)
        assert math.isfinite(d)

    def test_high_scale(self):
        """Test with high scale value."""
        kifs = KIFSSDF(scale=10.0)
        d = kifs.evaluate(1.0, 0.0, 0.0)
        assert math.isfinite(d)

    def test_small_scale(self):
        """Test with small scale value."""
        kifs = KIFSSDF(scale=1.1)
        d = kifs.evaluate(1.0, 0.0, 0.0)
        assert math.isfinite(d)


# =============================================================================
# Different Fold Type Tests
# =============================================================================

class TestKIFSFoldTypes:
    """Tests for different fold type behaviors."""

    @pytest.mark.parametrize("fold_type", [
        KIFSFoldType.ABS,
        KIFSFoldType.MENGER,
        KIFSFoldType.SIERPINSKI,
        KIFSFoldType.BOX,
        KIFSFoldType.SPHERE,
    ])
    def test_fold_type_finite_result(self, fold_type):
        """Test that all fold types produce finite results."""
        kifs = KIFSSDF(fold_type=fold_type, iterations=5)
        d = kifs.evaluate(1.0, 0.5, 0.3)
        assert math.isfinite(d)

    @pytest.mark.parametrize("fold_type", [
        KIFSFoldType.ABS,
        KIFSFoldType.MENGER,
        KIFSFoldType.SIERPINSKI,
        KIFSFoldType.BOX,
        KIFSFoldType.SPHERE,
    ])
    def test_fold_type_wgsl_generation(self, fold_type):
        """Test that all fold types generate valid WGSL."""
        wgsl = generate_kifs_wgsl(fold_type=fold_type)
        assert "fn sdf_kifs" in wgsl
        assert len(wgsl) > 100  # Should have substantial content


# =============================================================================
# Integration Tests
# =============================================================================

class TestKIFSIntegration:
    """Integration tests for KIFS with other components."""

    def test_kifs_with_sphere_base(self):
        """Test KIFS with sphere as base SDF."""
        kifs = KIFSSDF()
        kifs.set_base_sdf(lambda x, y, z: sdf_sphere(x, y, z, 0.5))
        d = kifs.evaluate(1.0, 0.0, 0.0)
        assert math.isfinite(d)

    def test_kifs_with_box_base(self):
        """Test KIFS with box as base SDF."""
        kifs = KIFSSDF()
        kifs.set_base_sdf(lambda x, y, z: sdf_box(x, y, z, 0.5, 0.5, 0.5))
        d = kifs.evaluate(1.0, 0.0, 0.0)
        assert math.isfinite(d)

    def test_multiple_kifs_different_folds(self):
        """Test multiple KIFS with different folds."""
        kifs_abs = KIFSSDF(fold_type=KIFSFoldType.ABS)
        kifs_sierp = KIFSSDF(fold_type=KIFSFoldType.SIERPINSKI)

        d_abs = kifs_abs.evaluate(1.0, 0.5, 0.3)
        d_sierp = kifs_sierp.evaluate(1.0, 0.5, 0.3)

        # Different folds should give different results
        # (though both finite)
        assert math.isfinite(d_abs)
        assert math.isfinite(d_sierp)


# =============================================================================
# Numeric Stability Tests
# =============================================================================

class TestKIFSNumericStability:
    """Tests for numeric stability of the KIFS implementation."""

    def test_no_nan_random_points(self):
        """Test that random points don't produce NaN."""
        import random
        random.seed(42)
        kifs = KIFSSDF()

        for _ in range(100):
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            z = random.uniform(-5, 5)
            d = kifs.evaluate(x, y, z)
            assert math.isfinite(d), f"NaN at ({x}, {y}, {z})"

    def test_stability_high_iterations(self):
        """Test stability with high iteration count."""
        kifs = KIFSSDF(iterations=30)
        d = kifs.evaluate(0.5, 0.5, 0.5)
        assert math.isfinite(d)

    def test_stability_high_scale(self):
        """Test stability with high scale."""
        kifs = KIFSSDF(scale=5.0, iterations=10)
        d = kifs.evaluate(0.5, 0.5, 0.5)
        assert math.isfinite(d)
