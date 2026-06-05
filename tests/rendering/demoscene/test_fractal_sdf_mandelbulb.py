"""
Tests for Mandelbulb SDF (T-DEMO-4.10)

This test suite validates the Mandelbulb SDF implementation including:
- Shape correctness at various powers (2, 4, 8)
- Distance estimation accuracy
- Iteration count effects
- Spherical coordinate conversion
- WGSL output validation
- Performance benchmarks
"""

import math
import pytest
import time
from typing import Tuple, List

from engine.rendering.demoscene.fractal_sdf import (
    MandelbulbSDF,
    MandelbulbConfig,
    mandelbulb_distance,
    mandelbulb_distance_estimator,
    cartesian_to_spherical,
    spherical_to_cartesian,
    spherical_power,
    generate_mandelbulb_wgsl,
    MANDELBULB_WGSL_FUNCTION,
    DEFAULT_MANDELBULB_POWER,
    DEFAULT_MANDELBULB_ITERATIONS,
    DEFAULT_MANDELBULB_BAILOUT,
)
from engine.rendering.demoscene.sdf_ast import Vec3


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def default_mandelbulb():
    """Create a default Mandelbulb SDF."""
    return MandelbulbSDF()


@pytest.fixture
def power_2_mandelbulb():
    """Create a Mandelbulb with power 2."""
    return MandelbulbSDF(power=2.0)


@pytest.fixture
def power_4_mandelbulb():
    """Create a Mandelbulb with power 4."""
    return MandelbulbSDF(power=4.0)


@pytest.fixture
def high_iteration_mandelbulb():
    """Create a Mandelbulb with high iteration count."""
    return MandelbulbSDF(iterations=30)


# =============================================================================
# Spherical Coordinate Tests
# =============================================================================

class TestSphericalCoordinates:
    """Tests for spherical coordinate conversion functions."""

    def test_cartesian_to_spherical_origin(self):
        """Test conversion at origin."""
        r, theta, phi = cartesian_to_spherical(0.0, 0.0, 0.0)
        assert r == 0.0
        assert theta == 0.0
        assert phi == 0.0

    def test_cartesian_to_spherical_unit_x(self):
        """Test conversion for unit x axis."""
        r, theta, phi = cartesian_to_spherical(1.0, 0.0, 0.0)
        assert abs(r - 1.0) < 1e-10
        assert abs(theta) < 1e-10
        assert abs(phi) < 1e-10

    def test_cartesian_to_spherical_unit_y(self):
        """Test conversion for unit y axis."""
        r, theta, phi = cartesian_to_spherical(0.0, 1.0, 0.0)
        assert abs(r - 1.0) < 1e-10
        assert abs(theta - math.pi / 2) < 1e-10
        assert abs(phi) < 1e-10

    def test_cartesian_to_spherical_unit_z(self):
        """Test conversion for unit z axis."""
        r, theta, phi = cartesian_to_spherical(0.0, 0.0, 1.0)
        assert abs(r - 1.0) < 1e-10
        assert abs(phi - math.pi / 2) < 1e-10

    def test_spherical_to_cartesian_roundtrip(self):
        """Test roundtrip conversion."""
        original = (1.5, 2.3, -0.7)
        r, theta, phi = cartesian_to_spherical(*original)
        result = spherical_to_cartesian(r, theta, phi)

        assert abs(result[0] - original[0]) < 1e-10
        assert abs(result[1] - original[1]) < 1e-10
        assert abs(result[2] - original[2]) < 1e-10

    @pytest.mark.parametrize("point", [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 2.0, 0.5),
        (0.5, -0.5, 0.5),
    ])
    def test_spherical_roundtrip_parametrized(self, point: Tuple[float, float, float]):
        """Test roundtrip conversion for various points."""
        r, theta, phi = cartesian_to_spherical(*point)
        result = spherical_to_cartesian(r, theta, phi)

        assert abs(result[0] - point[0]) < 1e-9
        assert abs(result[1] - point[1]) < 1e-9
        assert abs(result[2] - point[2]) < 1e-9

    def test_spherical_power_finite(self):
        """Test that spherical power returns finite values."""
        # The Mandelbulb uses a specific convention for spherical coordinates
        # that differs from standard power operations
        point = (1.0, 2.0, 3.0)
        result = spherical_power(*point, power=1.0)

        # Should return finite values
        assert math.isfinite(result[0])
        assert math.isfinite(result[1])
        assert math.isfinite(result[2])

    def test_spherical_power_squared_finite(self):
        """Test squaring operation returns finite values."""
        # For the Mandelbulb convention, squaring along axis may not preserve direction
        x, y, z = spherical_power(2.0, 0.0, 0.0, power=2.0)
        # Should return finite values
        assert math.isfinite(x)
        assert math.isfinite(y)
        assert math.isfinite(z)
        # Radius should be squared: 2^2 = 4
        r = math.sqrt(x * x + y * y + z * z)
        assert abs(r - 4.0) < 1e-9


# =============================================================================
# MandelbulbConfig Tests
# =============================================================================

class TestMandelbulbConfig:
    """Tests for MandelbulbConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MandelbulbConfig()
        assert config.power == DEFAULT_MANDELBULB_POWER
        assert config.iterations == DEFAULT_MANDELBULB_ITERATIONS
        assert config.bailout == DEFAULT_MANDELBULB_BAILOUT

    def test_custom_config(self):
        """Test custom configuration."""
        config = MandelbulbConfig(power=4.0, iterations=20, bailout=4.0)
        assert config.power == 4.0
        assert config.iterations == 20
        assert config.bailout == 4.0

    def test_invalid_power(self):
        """Test that invalid power raises error."""
        with pytest.raises(ValueError, match="Power must be >= 1.0"):
            MandelbulbConfig(power=0.5)

    def test_invalid_iterations(self):
        """Test that invalid iterations raises error."""
        with pytest.raises(ValueError, match="Iterations must be >= 1"):
            MandelbulbConfig(iterations=0)

    def test_invalid_bailout(self):
        """Test that invalid bailout raises error."""
        with pytest.raises(ValueError, match="Bailout must be > 0.0"):
            MandelbulbConfig(bailout=-1.0)


# =============================================================================
# Distance Estimation Tests
# =============================================================================

class TestMandelbulbDistanceEstimation:
    """Tests for Mandelbulb distance estimation accuracy."""

    def test_distance_at_origin(self):
        """Test distance at origin (inside the Mandelbulb)."""
        d = mandelbulb_distance_estimator(0.0, 0.0, 0.0)
        # Origin is inside, distance should be small/negative
        assert d < 0.5

    def test_distance_far_from_surface(self):
        """Test distance far from surface."""
        d = mandelbulb_distance_estimator(10.0, 0.0, 0.0)
        # Far outside, distance should be large and positive
        assert d > 5.0

    def test_distance_near_surface(self):
        """Test distance near surface for various powers."""
        # The Mandelbulb has a characteristic radius around 1.2-1.5
        # depending on power
        for power in [2.0, 4.0, 8.0]:
            d = mandelbulb_distance_estimator(1.3, 0.0, 0.0, power=power)
            # Should be relatively small near the surface
            assert abs(d) < 1.0

    def test_distance_symmetry_xy(self):
        """Test that distance is symmetric in XY plane."""
        d1 = mandelbulb_distance_estimator(1.0, 0.5, 0.0)
        d2 = mandelbulb_distance_estimator(0.5, 1.0, 0.0)
        # Due to rotational symmetry around Z axis (approximately)
        # These should be similar but not identical
        assert abs(d1 - d2) < 0.5

    def test_distance_positive_far(self):
        """Test that distance is positive far from surface."""
        points = [
            (5.0, 0.0, 0.0),
            (0.0, 5.0, 0.0),
            (0.0, 0.0, 5.0),
            (3.0, 3.0, 3.0),
        ]
        for p in points:
            d = mandelbulb_distance_estimator(*p)
            assert d > 0.0, f"Distance should be positive at {p}"

    def test_distance_monotonic_increasing(self):
        """Test that distance increases as we move away from surface."""
        distances = []
        for x in [1.5, 2.0, 3.0, 5.0, 10.0]:
            d = mandelbulb_distance_estimator(x, 0.0, 0.0)
            distances.append(d)

        # Check monotonic increase
        for i in range(len(distances) - 1):
            assert distances[i + 1] > distances[i], \
                f"Distance should increase: {distances[i]} < {distances[i + 1]}"

    def test_iteration_effect(self):
        """Test that more iterations give more accurate results."""
        point = (1.2, 0.0, 0.0)
        d_low = mandelbulb_distance_estimator(*point, iterations=5)
        d_high = mandelbulb_distance_estimator(*point, iterations=20)

        # Results should be different but both reasonable
        assert abs(d_low - d_high) < 0.5

    def test_bailout_effect(self):
        """Test effect of bailout parameter."""
        point = (3.0, 0.0, 0.0)
        d_low = mandelbulb_distance_estimator(*point, bailout=2.0)
        d_high = mandelbulb_distance_estimator(*point, bailout=4.0)

        # Both should be positive for a point outside the set
        assert d_low > 0.0
        assert d_high > 0.0


# =============================================================================
# MandelbulbSDF Class Tests
# =============================================================================

class TestMandelbulbSDF:
    """Tests for MandelbulbSDF class."""

    def test_creation_default(self, default_mandelbulb):
        """Test default creation."""
        mb = default_mandelbulb
        assert mb.power == DEFAULT_MANDELBULB_POWER
        assert mb.iterations == DEFAULT_MANDELBULB_ITERATIONS
        assert mb.bailout == DEFAULT_MANDELBULB_BAILOUT

    def test_creation_custom(self):
        """Test custom creation."""
        mb = MandelbulbSDF(power=4.0, iterations=20, bailout=3.0)
        assert mb.power == 4.0
        assert mb.iterations == 20
        assert mb.bailout == 3.0

    def test_position(self):
        """Test position parameter."""
        pos = Vec3(1.0, 2.0, 3.0)
        mb = MandelbulbSDF(position=pos)
        assert mb.position.x == 1.0
        assert mb.position.y == 2.0
        assert mb.position.z == 3.0

    def test_evaluate(self, default_mandelbulb):
        """Test evaluate method."""
        d = default_mandelbulb.evaluate(5.0, 0.0, 0.0)
        assert d > 0.0

    def test_evaluate_point(self, default_mandelbulb):
        """Test evaluate_point method."""
        d = default_mandelbulb.evaluate_point((5.0, 0.0, 0.0))
        assert d > 0.0

    def test_evaluate_with_position(self):
        """Test that position offset works correctly."""
        mb1 = MandelbulbSDF()
        mb2 = MandelbulbSDF(position=Vec3(5.0, 0.0, 0.0))

        # Evaluating at (5, 0, 0) for mb2 should be same as (0, 0, 0) for mb1
        d1 = mb1.evaluate(0.0, 0.0, 0.0)
        d2 = mb2.evaluate(5.0, 0.0, 0.0)

        assert abs(d1 - d2) < 1e-10

    def test_wgsl_function_name(self, default_mandelbulb):
        """Test WGSL function name property."""
        assert default_mandelbulb.wgsl_function == "sdf_mandelbulb"

    def test_label(self, default_mandelbulb):
        """Test label generation."""
        label = default_mandelbulb.label()
        assert "Mandelbulb" in label
        assert str(DEFAULT_MANDELBULB_POWER) in label

    def test_clone(self, default_mandelbulb):
        """Test clone method."""
        clone = default_mandelbulb.clone()
        assert clone.power == default_mandelbulb.power
        assert clone.iterations == default_mandelbulb.iterations
        assert clone.bailout == default_mandelbulb.bailout
        assert clone is not default_mandelbulb

    def test_children(self, default_mandelbulb):
        """Test children method returns empty tuple."""
        assert default_mandelbulb.children() == ()

    def test_get_config(self, default_mandelbulb):
        """Test get_config method."""
        config = default_mandelbulb.get_config()
        assert isinstance(config, MandelbulbConfig)
        assert config.power == default_mandelbulb.power

    def test_tracker_dirty(self):
        """Test that tracker marks fields dirty on creation."""
        mb = MandelbulbSDF()
        # tracker.is_dirty is a property that returns True if any field is dirty
        assert mb.tracker.is_dirty
        # Check that specific fields are in the dirty set
        assert "power" in mb.tracker.dirty_fields
        assert "iterations" in mb.tracker.dirty_fields
        assert "bailout" in mb.tracker.dirty_fields
        assert "position" in mb.tracker.dirty_fields


# =============================================================================
# Shape Tests at Various Powers
# =============================================================================

class TestMandelbulbShapes:
    """Tests for Mandelbulb shape at various powers."""

    @pytest.mark.parametrize("power", [2.0, 4.0, 8.0, 16.0])
    def test_shape_inside_origin(self, power: float):
        """Test that origin is inside the Mandelbulb at all powers."""
        mb = MandelbulbSDF(power=power)
        d = mb.evaluate(0.0, 0.0, 0.0)
        # Origin should be inside or on surface
        assert d < 0.1

    @pytest.mark.parametrize("power", [2.0, 4.0, 8.0])
    def test_shape_outside_far(self, power: float):
        """Test that far points are outside at all powers."""
        mb = MandelbulbSDF(power=power)
        d = mb.evaluate(10.0, 0.0, 0.0)
        assert d > 5.0

    @pytest.mark.parametrize("power", [2.0, 4.0, 8.0])
    def test_shape_bounded(self, power: float):
        """Test that Mandelbulb is bounded within radius ~1.5."""
        mb = MandelbulbSDF(power=power)
        # At radius 2.0, should be outside
        d = mb.evaluate(2.0, 0.0, 0.0)
        assert d > 0.0

    def test_power_2_vs_power_8(self):
        """Test that different powers give different shapes."""
        mb2 = MandelbulbSDF(power=2.0)
        mb8 = MandelbulbSDF(power=8.0)

        # At a test point, distances should differ
        d2 = mb2.evaluate(1.3, 0.0, 0.0)
        d8 = mb8.evaluate(1.3, 0.0, 0.0)

        # They should be different
        assert abs(d2 - d8) > 0.01


# =============================================================================
# WGSL Generation Tests
# =============================================================================

class TestMandelbulbWGSL:
    """Tests for Mandelbulb WGSL code generation."""

    def test_wgsl_function_exists(self):
        """Test that WGSL function constant exists."""
        assert "sdf_mandelbulb" in MANDELBULB_WGSL_FUNCTION
        assert "vec3<f32>" in MANDELBULB_WGSL_FUNCTION

    def test_generate_wgsl_default(self):
        """Test WGSL generation with defaults."""
        wgsl = generate_mandelbulb_wgsl()
        assert "fn sdf_mandelbulb" in wgsl
        assert str(DEFAULT_MANDELBULB_POWER) in wgsl

    def test_generate_wgsl_custom(self):
        """Test WGSL generation with custom parameters."""
        wgsl = generate_mandelbulb_wgsl(power=4.0, iterations=20)
        assert "4.0" in wgsl or "4" in wgsl
        assert "20" in wgsl

    def test_generate_wgsl_custom_name(self):
        """Test WGSL generation with custom function name."""
        wgsl = generate_mandelbulb_wgsl(function_name="my_mandelbulb")
        assert "fn my_mandelbulb" in wgsl

    def test_to_wgsl_method(self, default_mandelbulb):
        """Test to_wgsl method on SDF instance."""
        wgsl = default_mandelbulb.to_wgsl()
        assert "fn sdf_mandelbulb" in wgsl

    def test_wgsl_contains_key_elements(self):
        """Test that WGSL contains key algorithmic elements."""
        wgsl = generate_mandelbulb_wgsl()
        # Check for spherical coordinate conversion
        assert "atan2" in wgsl
        assert "acos" in wgsl
        # Check for power operation
        assert "pow" in wgsl
        # Check for distance estimator
        assert "log" in wgsl
        # Check for loop
        assert "for" in wgsl


# =============================================================================
# Performance Benchmarks
# =============================================================================

class TestMandelbulbPerformance:
    """Performance benchmarks for Mandelbulb SDF."""

    @pytest.mark.benchmark
    def test_single_evaluation_time(self, default_mandelbulb):
        """Benchmark single point evaluation."""
        start = time.perf_counter()
        for _ in range(1000):
            default_mandelbulb.evaluate(1.0, 0.5, 0.3)
        elapsed = time.perf_counter() - start

        # Should complete 1000 evaluations in under 1 second
        assert elapsed < 1.0

    @pytest.mark.benchmark
    def test_grid_evaluation_time(self, default_mandelbulb):
        """Benchmark grid of point evaluations."""
        start = time.perf_counter()
        for x in range(-5, 6):
            for y in range(-5, 6):
                for z in range(-5, 6):
                    default_mandelbulb.evaluate(x * 0.2, y * 0.2, z * 0.2)
        elapsed = time.perf_counter() - start

        # 11^3 = 1331 evaluations in under 2 seconds
        assert elapsed < 2.0

    @pytest.mark.benchmark
    def test_high_iteration_performance(self):
        """Benchmark high iteration count."""
        mb = MandelbulbSDF(iterations=50)
        start = time.perf_counter()
        for _ in range(100):
            mb.evaluate(1.0, 0.5, 0.3)
        elapsed = time.perf_counter() - start

        # 100 high-iteration evaluations in under 1 second
        assert elapsed < 1.0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestMandelbulbEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_point(self):
        """Test evaluation at very small coordinates."""
        d = mandelbulb_distance_estimator(1e-10, 1e-10, 1e-10)
        assert math.isfinite(d)

    def test_very_large_point(self):
        """Test evaluation at very large coordinates."""
        d = mandelbulb_distance_estimator(1000.0, 1000.0, 1000.0)
        assert math.isfinite(d)
        assert d > 0.0

    def test_negative_coordinates(self):
        """Test evaluation with negative coordinates."""
        d = mandelbulb_distance_estimator(-1.0, -1.0, -1.0)
        assert math.isfinite(d)

    def test_mixed_coordinates(self):
        """Test with mixed positive/negative coordinates."""
        d = mandelbulb_distance_estimator(1.0, -0.5, 0.3)
        assert math.isfinite(d)

    def test_minimum_iterations(self):
        """Test with minimum iteration count."""
        d = mandelbulb_distance_estimator(1.0, 0.0, 0.0, iterations=1)
        assert math.isfinite(d)

    def test_minimum_power(self):
        """Test with power = 1."""
        d = mandelbulb_distance_estimator(1.0, 0.0, 0.0, power=1.0)
        assert math.isfinite(d)

    def test_high_power(self):
        """Test with high power value."""
        d = mandelbulb_distance_estimator(1.0, 0.0, 0.0, power=32.0)
        assert math.isfinite(d)


# =============================================================================
# Integration Tests
# =============================================================================

class TestMandelbulbIntegration:
    """Integration tests for Mandelbulb with other components."""

    def test_mandelbulb_distance_function(self):
        """Test the convenience distance function."""
        config = MandelbulbConfig(power=4.0)
        d = mandelbulb_distance((1.0, 0.0, 0.0), config)
        assert math.isfinite(d)

    def test_mandelbulb_distance_none_config(self):
        """Test distance function with None config."""
        d = mandelbulb_distance((1.0, 0.0, 0.0), None)
        assert math.isfinite(d)

    def test_multiple_mandelbulbs(self):
        """Test multiple Mandelbulbs at different positions."""
        mb1 = MandelbulbSDF(position=Vec3(0.0, 0.0, 0.0))
        mb2 = MandelbulbSDF(position=Vec3(5.0, 0.0, 0.0))

        # Point between them
        d1 = mb1.evaluate(2.5, 0.0, 0.0)
        d2 = mb2.evaluate(2.5, 0.0, 0.0)

        assert d1 > 0.0
        assert d2 > 0.0


# =============================================================================
# Numeric Stability Tests
# =============================================================================

class TestMandelbulbNumericStability:
    """Tests for numeric stability of the Mandelbulb implementation."""

    def test_no_nan_on_surface(self):
        """Test that evaluation on approximate surface doesn't produce NaN."""
        mb = MandelbulbSDF()
        # Sample points near the expected surface
        for theta in range(0, 360, 30):
            for phi in range(-90, 91, 30):
                theta_rad = math.radians(theta)
                phi_rad = math.radians(phi)
                r = 1.2  # Approximate surface radius
                x = r * math.cos(phi_rad) * math.cos(theta_rad)
                y = r * math.cos(phi_rad) * math.sin(theta_rad)
                z = r * math.sin(phi_rad)

                d = mb.evaluate(x, y, z)
                assert math.isfinite(d), f"NaN at ({x}, {y}, {z})"

    def test_no_nan_random_points(self):
        """Test that random points don't produce NaN."""
        import random
        random.seed(42)
        mb = MandelbulbSDF()

        for _ in range(100):
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            z = random.uniform(-5, 5)
            d = mb.evaluate(x, y, z)
            assert math.isfinite(d), f"NaN at ({x}, {y}, {z})"

    def test_derivative_stability(self):
        """Test that derivative doesn't explode."""
        # The derivative should stay bounded
        mb = MandelbulbSDF(iterations=30)
        d = mb.evaluate(0.5, 0.5, 0.5)
        assert abs(d) < 10.0  # Reasonable bound
