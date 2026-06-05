"""
Tests for Depth of Field module (T-DEMO-3.12).

Tests cover:
  - DOF parameters and validation
  - Circle of confusion calculations
  - Lens disk sampling (uniform and stratified)
  - Ray jittering for thin lens model
  - WGSL code generation
  - Accumulation buffer

Requirements:
  - Objects at focal distance are sharp
  - Objects closer/further are blurred proportionally
  - Aperture controls blur intensity
  - 20+ tests verifying DOF behavior
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.demoscene.depth_of_field import (
    DOFParams,
    calculate_coc,
    calculate_coc_normalized,
    is_in_focus,
    sample_disk_uniform,
    sample_disk_stratified,
    sample_hexagon,
    DOFGenerator,
    AccumulationBuffer,
    generate_dof_wgsl,
    validate_dof_params,
)
from engine.rendering.demoscene.ray_generation import Ray, Vec3


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_dof_params() -> DOFParams:
    """Default DOF parameters."""
    return DOFParams(
        aperture=0.05,
        focal_distance=5.0,
        focus_range=2.0,
        samples_per_pixel=16,
    )


@pytest.fixture
def pinhole_params() -> DOFParams:
    """Pinhole camera (no DOF)."""
    return DOFParams(
        aperture=0.0,
        focal_distance=5.0,
        focus_range=2.0,
    )


@pytest.fixture
def primary_ray() -> Ray:
    """A primary ray looking down -Z axis."""
    return Ray(
        origin=Vec3(0.0, 0.0, 0.0),
        direction=Vec3(0.0, 0.0, -1.0),
    )


@pytest.fixture
def camera_vectors():
    """Camera right and up vectors."""
    return {
        'right': Vec3(1.0, 0.0, 0.0),
        'up': Vec3(0.0, 1.0, 0.0),
    }


# =============================================================================
# DOFParams Tests
# =============================================================================


class TestDOFParams:
    """Tests for DOFParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = DOFParams()
        assert params.aperture == 0.05
        assert params.focal_distance == 5.0
        assert params.focus_range == 2.0
        assert params.bokeh_shape == "circle"
        assert params.samples_per_pixel == 16

    def test_is_enabled_with_aperture(self, default_dof_params):
        """DOF is enabled when aperture > 0."""
        assert default_dof_params.is_enabled()

    def test_is_enabled_pinhole(self, pinhole_params):
        """DOF is disabled when aperture = 0."""
        assert not pinhole_params.is_enabled()

    def test_is_enabled_tiny_aperture(self):
        """Very small aperture should be considered disabled."""
        params = DOFParams(aperture=1e-10)
        assert not params.is_enabled()

    def test_custom_parameters(self):
        """Can create with custom parameters."""
        params = DOFParams(
            aperture=0.1,
            focal_distance=10.0,
            focus_range=4.0,
            bokeh_shape="hexagon",
            samples_per_pixel=32,
        )
        assert params.aperture == 0.1
        assert params.focal_distance == 10.0
        assert params.focus_range == 4.0
        assert params.bokeh_shape == "hexagon"
        assert params.samples_per_pixel == 32


# =============================================================================
# Circle of Confusion Tests
# =============================================================================


class TestCircleOfConfusion:
    """Tests for circle of confusion calculations."""

    def test_coc_at_focal_plane_is_zero(self):
        """COC at focal distance should be zero (sharp)."""
        coc = calculate_coc(
            hit_distance=5.0,
            focal_distance=5.0,
            aperture=0.05,
        )
        assert coc == pytest.approx(0.0, abs=1e-9)

    def test_coc_increases_with_distance_from_focal(self):
        """COC should increase as distance from focal plane increases."""
        focal = 5.0
        aperture = 0.05

        coc_near = calculate_coc(4.0, focal, aperture)
        coc_focal = calculate_coc(5.0, focal, aperture)
        coc_far = calculate_coc(6.0, focal, aperture)
        coc_very_far = calculate_coc(10.0, focal, aperture)

        assert coc_focal < coc_near
        assert coc_focal < coc_far
        assert coc_far < coc_very_far

    def test_coc_increases_with_aperture(self):
        """Larger aperture should produce larger COC."""
        hit = 10.0
        focal = 5.0

        coc_small = calculate_coc(hit, focal, aperture=0.01)
        coc_large = calculate_coc(hit, focal, aperture=0.1)

        assert coc_large > coc_small
        # Should be proportional
        assert coc_large / coc_small == pytest.approx(10.0, abs=0.1)

    def test_coc_zero_aperture_is_zero(self):
        """Zero aperture (pinhole) should have zero COC."""
        coc = calculate_coc(10.0, 5.0, aperture=0.0)
        assert coc == 0.0

    def test_coc_zero_distance_is_zero(self):
        """Zero hit distance should return zero COC."""
        coc = calculate_coc(0.0, 5.0, aperture=0.05)
        assert coc == 0.0

    def test_coc_normalized_in_range(self):
        """Normalized COC should be in [0, 1]."""
        for dist in [1.0, 5.0, 10.0, 50.0, 100.0]:
            coc = calculate_coc_normalized(dist, 5.0, 0.05, 2.0)
            assert 0.0 <= coc <= 1.0

    def test_coc_normalized_at_focal_is_zero(self):
        """Normalized COC at focal distance should be zero."""
        coc = calculate_coc_normalized(5.0, 5.0, 0.05, 2.0)
        assert coc == pytest.approx(0.0, abs=1e-9)


# =============================================================================
# Focus Range Tests
# =============================================================================


class TestFocusRange:
    """Tests for is_in_focus function."""

    def test_at_focal_distance_is_in_focus(self):
        """Objects at focal distance are in focus."""
        assert is_in_focus(5.0, focal_distance=5.0, focus_range=2.0)

    def test_within_focus_range_is_in_focus(self):
        """Objects within focus_range/2 of focal distance are in focus."""
        assert is_in_focus(4.5, focal_distance=5.0, focus_range=2.0)  # 0.5 < 1.0
        assert is_in_focus(5.5, focal_distance=5.0, focus_range=2.0)  # 0.5 < 1.0
        assert is_in_focus(4.0, focal_distance=5.0, focus_range=2.0)  # Exactly at edge

    def test_outside_focus_range_is_not_in_focus(self):
        """Objects outside focus_range/2 are out of focus."""
        assert not is_in_focus(3.5, focal_distance=5.0, focus_range=2.0)  # 1.5 > 1.0
        assert not is_in_focus(7.0, focal_distance=5.0, focus_range=2.0)  # 2.0 > 1.0

    def test_narrow_focus_range(self):
        """Narrow focus range should be more selective."""
        assert not is_in_focus(5.3, focal_distance=5.0, focus_range=0.5)


# =============================================================================
# Disk Sampling Tests
# =============================================================================


class TestDiskSampling:
    """Tests for lens disk sampling."""

    def test_uniform_center(self):
        """Sample at (0.5, 0.5) should give center."""
        x, y = sample_disk_uniform(0.5, 0.5)
        assert x == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(0.0, abs=1e-6)

    def test_uniform_samples_on_disk(self):
        """All uniform samples should be within unit disk."""
        for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
                x, y = sample_disk_uniform(u, v)
                radius = math.sqrt(x * x + y * y)
                assert radius <= 1.0 + 1e-6

    def test_stratified_samples_on_disk(self):
        """All stratified samples should be within unit disk."""
        for i in range(100):
            x, y = sample_disk_stratified(i, 100)
            radius = math.sqrt(x * x + y * y)
            assert radius <= 1.0 + 1e-6

    def test_stratified_center_sample(self):
        """First stratified sample should be near center."""
        x, y = sample_disk_stratified(0, 64)
        radius = math.sqrt(x * x + y * y)
        assert radius < 0.2  # Near center

    def test_stratified_distribution(self):
        """Stratified samples should cover disk uniformly."""
        xs, ys = [], []
        n_samples = 100
        for i in range(n_samples):
            x, y = sample_disk_stratified(i, n_samples, jitter=0.0)
            xs.append(x)
            ys.append(y)

        # Check reasonable spread
        assert min(xs) < -0.5
        assert max(xs) > 0.5
        assert min(ys) < -0.5
        assert max(ys) > 0.5

    def test_hexagon_samples_on_hexagon(self):
        """Hexagon samples should be within hexagonal shape."""
        for u in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for v in [0.1, 0.3, 0.5, 0.7, 0.9]:
                x, y = sample_hexagon(u, v)
                radius = math.sqrt(x * x + y * y)
                # Hexagon inscribed in unit circle
                assert radius <= 1.0 + 1e-6


# =============================================================================
# DOF Generator Tests
# =============================================================================


class TestDOFGenerator:
    """Tests for DOFGenerator class."""

    def test_pinhole_ray_unchanged(self, primary_ray, pinhole_params, camera_vectors):
        """Pinhole (aperture=0) should not modify ray."""
        dof = DOFGenerator()
        offset = (0.5, 0.5)
        jittered = dof.jitter_ray(
            primary_ray,
            pinhole_params,
            offset,
            camera_vectors['right'],
            camera_vectors['up'],
        )
        assert jittered.origin.x == pytest.approx(primary_ray.origin.x, abs=1e-9)
        assert jittered.origin.y == pytest.approx(primary_ray.origin.y, abs=1e-9)
        assert jittered.origin.z == pytest.approx(primary_ray.origin.z, abs=1e-9)
        assert jittered.direction.x == pytest.approx(primary_ray.direction.x, abs=1e-9)
        assert jittered.direction.y == pytest.approx(primary_ray.direction.y, abs=1e-9)
        assert jittered.direction.z == pytest.approx(primary_ray.direction.z, abs=1e-9)

    def test_center_sample_unchanged(self, primary_ray, default_dof_params, camera_vectors):
        """Center lens sample (0, 0) should not offset origin."""
        dof = DOFGenerator()
        jittered = dof.jitter_ray(
            primary_ray,
            default_dof_params,
            (0.0, 0.0),  # Center of lens
            camera_vectors['right'],
            camera_vectors['up'],
        )
        # Origin should be same as original
        assert jittered.origin.x == pytest.approx(0.0, abs=1e-9)
        assert jittered.origin.y == pytest.approx(0.0, abs=1e-9)
        assert jittered.origin.z == pytest.approx(0.0, abs=1e-9)

    def test_jitter_offsets_origin(self, primary_ray, default_dof_params, camera_vectors):
        """Non-center lens sample should offset ray origin."""
        dof = DOFGenerator()
        jittered = dof.jitter_ray(
            primary_ray,
            default_dof_params,
            (1.0, 0.0),  # Right edge of lens
            camera_vectors['right'],
            camera_vectors['up'],
        )
        # Origin should be offset in camera_right direction
        expected_offset = default_dof_params.aperture * 1.0
        assert jittered.origin.x == pytest.approx(expected_offset, abs=1e-9)
        assert jittered.origin.y == pytest.approx(0.0, abs=1e-9)
        assert jittered.origin.z == pytest.approx(0.0, abs=1e-9)

    def test_jittered_ray_converges_at_focal_plane(self, default_dof_params, camera_vectors):
        """All jittered rays should converge at the focal point."""
        dof = DOFGenerator()
        ray = Ray(
            origin=Vec3(0.0, 0.0, 0.0),
            direction=Vec3(0.0, 0.0, -1.0),
        )

        # Calculate expected focal point
        focal_point = ray.point_at(default_dof_params.focal_distance)

        # Test multiple sample offsets
        for offset in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.5, 0.5)]:
            jittered = dof.jitter_ray(
                ray,
                default_dof_params,
                offset,
                camera_vectors['right'],
                camera_vectors['up'],
            )

            # Find where jittered ray reaches focal distance
            # t such that jittered.origin + t * jittered.direction has z = focal_point.z
            t = (focal_point.z - jittered.origin.z) / jittered.direction.z
            hit_point = jittered.point_at(t)

            # All rays should converge at the same focal point
            assert hit_point.x == pytest.approx(focal_point.x, abs=1e-6)
            assert hit_point.y == pytest.approx(focal_point.y, abs=1e-6)

    def test_jitter_ray_simple(self, primary_ray, default_dof_params):
        """jitter_ray_simple should work without explicit camera vectors."""
        dof = DOFGenerator()
        jittered = dof.jitter_ray_simple(
            primary_ray,
            default_dof_params,
            (0.5, 0.5),
        )
        # Should return a valid ray
        assert jittered.direction.length() == pytest.approx(1.0, abs=1e-6)

    def test_get_sample_offset_center_first(self, default_dof_params):
        """First sample should be near center."""
        dof = DOFGenerator()
        x, y = dof.get_sample_offset(0, 16)
        radius = math.sqrt(x * x + y * y)
        assert radius < 0.3

    def test_get_sample_offset_coverage(self):
        """Sample offsets should cover the lens disk."""
        dof = DOFGenerator()
        xs, ys = [], []
        for i in range(64):
            x, y = dof.get_sample_offset(i, 64)
            xs.append(x)
            ys.append(y)

        # Should have good coverage
        assert min(xs) < -0.5
        assert max(xs) > 0.5
        assert min(ys) < -0.5
        assert max(ys) > 0.5

    def test_generate_samples_iterator(self, primary_ray, default_dof_params, camera_vectors):
        """generate_samples should yield correct number of rays."""
        dof = DOFGenerator()
        rays = list(dof.generate_samples(
            primary_ray,
            default_dof_params,
            camera_vectors['right'],
            camera_vectors['up'],
        ))
        assert len(rays) == default_dof_params.samples_per_pixel

    def test_generate_samples_pinhole_single_ray(self, primary_ray, pinhole_params, camera_vectors):
        """Pinhole should yield single ray."""
        dof = DOFGenerator()
        rays = list(dof.generate_samples(
            primary_ray,
            pinhole_params,
            camera_vectors['right'],
            camera_vectors['up'],
        ))
        assert len(rays) == 1


# =============================================================================
# Accumulation Buffer Tests
# =============================================================================


class TestAccumulationBuffer:
    """Tests for AccumulationBuffer class."""

    def test_initialization(self):
        """Buffer should initialize with zeros."""
        buf = AccumulationBuffer(width=8, height=8)
        assert buf.width == 8
        assert buf.height == 8
        assert buf.sample_count == 0

    def test_clear(self):
        """Clear should reset buffer and sample count."""
        buf = AccumulationBuffer(width=4, height=4)
        buf.accumulate(0, 0, Vec3(1.0, 0.0, 0.0))
        buf.increment_sample_count()
        buf.clear()
        assert buf.sample_count == 0
        assert buf.get_color(0, 0).x == 0.0

    def test_accumulate_and_get(self):
        """Accumulated colors should average correctly."""
        buf = AccumulationBuffer(width=4, height=4)

        # Add two samples
        buf.accumulate(1, 1, Vec3(1.0, 0.0, 0.0))
        buf.increment_sample_count()
        buf.accumulate(1, 1, Vec3(0.0, 1.0, 0.0))
        buf.increment_sample_count()

        result = buf.get_color(1, 1)
        assert result.x == pytest.approx(0.5, abs=1e-6)
        assert result.y == pytest.approx(0.5, abs=1e-6)
        assert result.z == pytest.approx(0.0, abs=1e-6)

    def test_out_of_bounds_ignored(self):
        """Out of bounds access should be safe."""
        buf = AccumulationBuffer(width=4, height=4)
        buf.accumulate(10, 10, Vec3(1.0, 0.0, 0.0))  # Should not crash
        result = buf.get_color(10, 10)
        assert result.x == 0.0


# =============================================================================
# WGSL Generation Tests
# =============================================================================


class TestWGSLGeneration:
    """Tests for WGSL code generation."""

    def test_generates_dof_params_struct(self):
        """Should include DOFParams struct."""
        wgsl = generate_dof_wgsl()
        assert "struct DOFParams" in wgsl

    def test_generates_sample_disk(self):
        """Should include sample_disk function."""
        wgsl = generate_dof_wgsl()
        assert "fn sample_disk" in wgsl

    def test_generates_jitter_ray(self):
        """Should include jitter_ray_dof function."""
        wgsl = generate_dof_wgsl()
        assert "fn jitter_ray_dof" in wgsl

    def test_generates_calculate_coc(self):
        """Should include calculate_coc function."""
        wgsl = generate_dof_wgsl()
        assert "fn calculate_coc" in wgsl

    def test_generates_is_in_focus(self):
        """Should include is_in_focus function."""
        wgsl = generate_dof_wgsl()
        assert "fn is_in_focus" in wgsl

    def test_generates_accumulation_helpers(self):
        """Should include accumulation helpers when requested."""
        wgsl = generate_dof_wgsl(include_accumulation=True)
        assert "fn accumulate_sample" in wgsl or "accumulation" in wgsl.lower()


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for validation helpers."""

    def test_valid_params_no_errors(self, default_dof_params):
        """Valid params should have no errors."""
        errors = validate_dof_params(default_dof_params)
        assert len(errors) == 0

    def test_negative_aperture_error(self):
        """Negative aperture should produce error."""
        params = DOFParams(aperture=-0.1)
        errors = validate_dof_params(params)
        assert len(errors) > 0
        assert "aperture" in errors[0].lower()

    def test_zero_focal_distance_error(self):
        """Zero focal distance should produce error."""
        params = DOFParams(focal_distance=0.0)
        errors = validate_dof_params(params)
        assert len(errors) > 0
        assert "focal" in errors[0].lower()

    def test_negative_focus_range_error(self):
        """Negative focus range should produce error."""
        params = DOFParams(focus_range=-1.0)
        errors = validate_dof_params(params)
        assert len(errors) > 0
        assert "focus" in errors[0].lower()

    def test_zero_samples_error(self):
        """Zero samples per pixel should produce error."""
        params = DOFParams(samples_per_pixel=0)
        errors = validate_dof_params(params)
        assert len(errors) > 0
        assert "samples" in errors[0].lower()

    def test_invalid_bokeh_shape_error(self):
        """Invalid bokeh shape should produce error."""
        params = DOFParams(bokeh_shape="triangle")
        errors = validate_dof_params(params)
        assert len(errors) > 0
        assert "bokeh" in errors[0].lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for DOF workflow."""

    def test_full_dof_workflow(self, camera_vectors):
        """Test complete DOF workflow from parameters to jittered rays."""
        # 1. Create DOF parameters
        params = DOFParams(
            aperture=0.05,
            focal_distance=5.0,
            focus_range=2.0,
            samples_per_pixel=8,
        )

        # 2. Create primary ray
        ray = Ray(
            origin=Vec3(0.0, 0.0, 0.0),
            direction=Vec3(0.0, 0.0, -1.0),
        )

        # 3. Generate jittered rays
        dof = DOFGenerator(seed=42)  # Deterministic
        rays = list(dof.generate_samples(
            ray,
            params,
            camera_vectors['right'],
            camera_vectors['up'],
        ))

        assert len(rays) == 8

        # 4. Verify all rays converge at focal plane
        focal_point = ray.point_at(params.focal_distance)
        for jittered in rays:
            # Find intersection with focal plane (z = -5)
            t = (focal_point.z - jittered.origin.z) / jittered.direction.z
            hit = jittered.point_at(t)
            assert hit.x == pytest.approx(focal_point.x, abs=1e-5)
            assert hit.y == pytest.approx(focal_point.y, abs=1e-5)

    def test_coc_consistent_with_focus_range(self):
        """Objects within focus range should have small COC."""
        params = DOFParams(
            aperture=0.05,
            focal_distance=5.0,
            focus_range=2.0,
        )

        # Object at focal distance
        coc_focal = calculate_coc(5.0, params.focal_distance, params.aperture)
        assert coc_focal == pytest.approx(0.0, abs=1e-9)

        # Object at edge of focus range
        coc_edge = calculate_coc(4.0, params.focal_distance, params.aperture)
        # Should be relatively small
        assert coc_edge < params.aperture * 0.3

        # Object far out of focus
        coc_far = calculate_coc(20.0, params.focal_distance, params.aperture)
        assert coc_far > coc_edge
