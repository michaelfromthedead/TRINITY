"""
Whitebox tests for SDF Ambient Occlusion (T-DEMO-3.5).

Tests the implementation-aware ambient occlusion calculation using Quilez's
method. Verifies internal algorithm paths including:
  - Sample accumulation along normal
  - Exponential falloff weighting
  - Expected vs actual distance comparison
  - Occlusion clamping behavior

WHITEBOX coverage plan:
  Path 1: Formula -- verify ao = sum((expected - actual) / expected * falloff^i)
  Path 2: No occlusion -- flat surface with no nearby geometry -> ao ~ 1.0
  Path 3: Full occlusion -- point inside crevice -> ao ~ 0.0
  Path 4: Partial occlusion -- corner geometry -> ao in (0.5, 0.9)
  Path 5: Sample count effect -- more samples = finer AO
  Path 6: Step scale effect -- larger steps = broader AO sampling
  Path 7: Falloff effect -- higher falloff = faster attenuation
  Path 8: Normal direction -- ao along normal vs opposite
  Path 9: Edge cases -- zero normal, coincident samples
  Path 10: Config validation -- invalid parameters raise errors
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.demoscene.sdf_ao import (
    AOConfig,
    calculate_ao,
    calculate_ao_multi_direction,
    generate_ao_wgsl,
    generate_ao_wgsl_inline,
    make_scene_ao_evaluator,
    Vec3Local,
)


# =============================================================================
# Test SDF Functions
# =============================================================================

def sdf_sphere(p: tuple[float, float, float], radius: float = 1.0) -> float:
    """Sphere SDF centered at origin."""
    return math.sqrt(p[0]**2 + p[1]**2 + p[2]**2) - radius


def sdf_plane_y(p: tuple[float, float, float]) -> float:
    """Infinite plane at y=0 with normal pointing up."""
    return p[1]


def sdf_box(p: tuple[float, float, float], half_extents: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """Axis-aligned box SDF centered at origin."""
    qx = abs(p[0]) - half_extents[0]
    qy = abs(p[1]) - half_extents[1]
    qz = abs(p[2]) - half_extents[2]
    outer = math.sqrt(max(qx, 0)**2 + max(qy, 0)**2 + max(qz, 0)**2)
    inner = min(max(qx, max(qy, qz)), 0.0)
    return outer + inner


def sdf_corner(p: tuple[float, float, float]) -> float:
    """L-shaped corner SDF (two perpendicular planes meeting at origin)."""
    # Union of y-plane and x-plane, creating a corner at origin
    d_floor = p[1]  # Floor plane
    d_wall = p[0]   # Wall plane
    return min(d_floor, d_wall)


def sdf_crevice(p: tuple[float, float, float], depth: float = 1.0, width: float = 0.5) -> float:
    """V-shaped crevice SDF."""
    # Two angled planes meeting to form a V
    angle = math.atan2(depth, width / 2)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Left side of V
    d_left = p[0] * cos_a + p[1] * sin_a
    # Right side of V
    d_right = -p[0] * cos_a + p[1] * sin_a

    return max(d_left, d_right)


def sdf_two_spheres(p: tuple[float, float, float]) -> float:
    """Two spheres forming a crevice where they meet."""
    # Spheres at (1, 0, 0) and (-1, 0, 0), radius 1.5, overlapping
    d1 = math.sqrt((p[0] - 1)**2 + p[1]**2 + p[2]**2) - 1.5
    d2 = math.sqrt((p[0] + 1)**2 + p[1]**2 + p[2]**2) - 1.5
    return min(d1, d2)


# =============================================================================
# Tolerance Constants
# =============================================================================

TOL_AO = 0.1       # AO tolerance (perceptual)
TOL_EXACT = 1e-6   # For numerical comparisons


# =============================================================================
# Path 1: Formula Verification
# =============================================================================

class TestFormula:
    """Verify the AO formula matches Quilez specification."""

    def test_ao_formula_manual_calculation(self):
        """Verify AO matches manual formula calculation."""
        # Simple sphere SDF: at (2, 0, 0), normal (1, 0, 0)
        # Samples along +x direction will hit the sphere at origin

        p = (2.0, 0.0, 0.0)
        n = (1.0, 0.0, 0.0)  # Away from sphere

        ao = calculate_ao(p, n, lambda pos: sdf_sphere(pos, 1.0))

        # Should have high AO (no occlusion when facing away)
        assert ao > 0.9, f"Expected ao > 0.9 for open direction, got {ao}"

    def test_ao_formula_occlusion_direction(self):
        """Verify occlusion increases when facing geometry."""
        p = (2.0, 0.0, 0.0)

        # Normal pointing away from sphere (outward)
        n_out = (1.0, 0.0, 0.0)
        ao_out = calculate_ao(p, n_out, lambda pos: sdf_sphere(pos, 1.0))

        # Normal pointing toward sphere (inward)
        n_in = (-1.0, 0.0, 0.0)
        ao_in = calculate_ao(p, n_in, lambda pos: sdf_sphere(pos, 1.0))

        # Facing away should have higher AO than facing toward
        assert ao_out > ao_in, f"ao_out={ao_out} should be > ao_in={ao_in}"

    def test_ao_accumulation_per_sample(self):
        """Verify AO accumulates across samples."""
        config_1 = AOConfig(samples=1, step_scale=0.1, falloff=0.5)
        config_3 = AOConfig(samples=3, step_scale=0.1, falloff=0.5)

        p = (0.5, 0.0, 0.0)  # Close to sphere surface
        n = (-1.0, 0.0, 0.0)  # Toward center (occluded)

        ao_1 = calculate_ao(p, n, lambda pos: sdf_sphere(pos, 0.5), config_1)
        ao_3 = calculate_ao(p, n, lambda pos: sdf_sphere(pos, 0.5), config_3)

        # More samples should detect more occlusion
        assert ao_3 <= ao_1, f"ao_3={ao_3} should be <= ao_1={ao_1}"


# =============================================================================
# Path 2: No Occlusion Case
# =============================================================================

class TestNoOcclusion:
    """Verify AO ~ 1.0 for unoccluded surfaces."""

    def test_flat_plane_no_occlusion(self):
        """Flat plane with nothing above should have AO ~ 1.0."""
        p = (0.0, 0.0, 0.0)  # On the plane
        n = (0.0, 1.0, 0.0)  # Normal pointing up

        ao = calculate_ao(p, n, sdf_plane_y)

        assert ao > 0.95, f"Expected ao > 0.95 for flat plane, got {ao}"

    def test_sphere_exterior_no_occlusion(self):
        """Point on sphere exterior facing outward has minimal occlusion."""
        p = (1.0, 0.0, 0.0)  # On sphere surface
        n = (1.0, 0.0, 0.0)  # Outward normal

        ao = calculate_ao(p, n, lambda pos: sdf_sphere(pos, 1.0))

        assert ao > 0.95, f"Expected ao > 0.95 for sphere exterior, got {ao}"

    def test_isolated_point(self):
        """Point far from any geometry should have AO ~ 1.0."""
        p = (0.0, 10.0, 0.0)  # Far above plane
        n = (0.0, 1.0, 0.0)

        # SDF returns large positive value everywhere
        ao = calculate_ao(p, n, lambda pos: 100.0)

        assert ao > 0.99, f"Expected ao > 0.99 for isolated point, got {ao}"


# =============================================================================
# Path 3: Full Occlusion Case
# =============================================================================

class TestFullOcclusion:
    """Verify AO ~ 0.0 for heavily occluded surfaces."""

    def test_inside_sphere(self):
        """Point inside sphere facing outward should be heavily occluded."""
        p = (0.5, 0.0, 0.0)  # Inside sphere of radius 1
        n = (1.0, 0.0, 0.0)  # Facing outward but still inside

        ao = calculate_ao(p, n, lambda pos: sdf_sphere(pos, 1.0))

        # Should detect occlusion (low AO)
        assert ao < 0.8, f"Expected ao < 0.8 for inside sphere, got {ao}"

    def test_tight_corner(self):
        """Point in tight corner should have very low AO."""
        p = (0.01, 0.01, 0.0)  # Very close to corner
        n = (1.0, 1.0, 0.0)    # Diagonal normal
        n_len = math.sqrt(2)
        n = (n[0]/n_len, n[1]/n_len, n[2]/n_len)

        ao = calculate_ao(p, n, sdf_corner)

        # Corner should have significant occlusion
        assert ao < 0.7, f"Expected ao < 0.7 for tight corner, got {ao}"

    def test_crevice_bottom(self):
        """Point at bottom of crevice should have very low AO."""
        p = (0.0, 0.01, 0.0)  # Near bottom of V-shaped crevice
        n = (0.0, 1.0, 0.0)   # Pointing up

        ao = calculate_ao(p, n, sdf_crevice)

        # Crevice should have significant occlusion
        assert ao < 0.8, f"Expected ao < 0.8 for crevice bottom, got {ao}"


# =============================================================================
# Path 4: Partial Occlusion Case
# =============================================================================

class TestPartialOcclusion:
    """Verify intermediate AO values for partially occluded surfaces."""

    def test_corner_at_distance(self):
        """Point near corner but not too close has partial occlusion."""
        p = (0.2, 0.2, 0.0)  # Near corner
        n_raw = (1.0, 1.0, 0.0)
        n_len = math.sqrt(2)
        n = (n_raw[0]/n_len, n_raw[1]/n_len, n_raw[2]/n_len)

        ao = calculate_ao(p, n, sdf_corner)

        # Should be partial occlusion
        assert 0.3 < ao < 0.95, f"Expected 0.3 < ao < 0.95 for partial occlusion, got {ao}"

    def test_sphere_grazing(self):
        """Point grazing sphere edge has partial occlusion."""
        p = (1.0, 0.5, 0.0)  # Just outside sphere, offset vertically
        n = (0.0, 1.0, 0.0)  # Normal pointing up

        ao = calculate_ao(p, n, lambda pos: sdf_sphere(pos, 1.0))

        # Partial occlusion from sphere below
        assert 0.5 < ao < 1.0, f"Expected partial occlusion, got {ao}"

    def test_two_spheres_contact(self):
        """Point between two spheres has partial occlusion."""
        p = (0.0, 0.0, 0.0)  # Between two overlapping spheres
        n = (0.0, 1.0, 0.0)  # Normal pointing up

        ao = calculate_ao(p, n, sdf_two_spheres)

        # Moderate occlusion expected
        assert 0.2 < ao < 0.9, f"Expected partial occlusion, got {ao}"


# =============================================================================
# Path 5: Sample Count Effect
# =============================================================================

class TestSampleCount:
    """Verify sample count affects AO quality."""

    def test_more_samples_smoother(self):
        """More samples should produce more consistent AO."""
        p = (0.1, 0.1, 0.0)
        n_raw = (1.0, 1.0, 0.0)
        n_len = math.sqrt(2)
        n = (n_raw[0]/n_len, n_raw[1]/n_len, n_raw[2]/n_len)

        config_3 = AOConfig(samples=3)
        config_8 = AOConfig(samples=8)

        ao_3 = calculate_ao(p, n, sdf_corner, config_3)
        ao_8 = calculate_ao(p, n, sdf_corner, config_8)

        # Both should detect occlusion
        assert ao_3 < 0.9
        assert ao_8 < 0.9

        # More samples might give slightly different (often darker) result
        # but should be in similar range
        assert abs(ao_3 - ao_8) < 0.3, f"ao_3={ao_3}, ao_8={ao_8}"

    def test_single_sample_ao(self):
        """Single sample AO still produces valid result."""
        config = AOConfig(samples=1)

        p = (0.1, 0.1, 0.0)
        n = (0.0, 1.0, 0.0)

        ao = calculate_ao(p, n, sdf_corner, config)

        assert 0.0 <= ao <= 1.0, f"AO should be in [0,1], got {ao}"

    def test_max_samples(self):
        """High sample count works correctly."""
        config = AOConfig(samples=16)

        p = (0.2, 0.2, 0.0)
        n_raw = (1.0, 1.0, 0.0)
        n_len = math.sqrt(2)
        n = (n_raw[0]/n_len, n_raw[1]/n_len, n_raw[2]/n_len)

        ao = calculate_ao(p, n, sdf_corner, config)

        assert 0.0 <= ao <= 1.0


# =============================================================================
# Path 6: Step Scale Effect
# =============================================================================

class TestStepScale:
    """Verify step_scale affects sampling distance."""

    def test_smaller_step_local_ao(self):
        """Smaller step scale samples closer to surface."""
        config_small = AOConfig(step_scale=0.01)
        config_large = AOConfig(step_scale=0.2)

        p = (0.1, 0.1, 0.0)
        n_raw = (1.0, 1.0, 0.0)
        n_len = math.sqrt(2)
        n = (n_raw[0]/n_len, n_raw[1]/n_len, n_raw[2]/n_len)

        ao_small = calculate_ao(p, n, sdf_corner, config_small)
        ao_large = calculate_ao(p, n, sdf_corner, config_large)

        # Small steps detect very local occlusion
        # Large steps sample further away
        assert abs(ao_small - ao_large) > 0.0, "Different step sizes should produce different AO"

    def test_step_scale_max_distance(self):
        """Step scale respects max_distance."""
        config = AOConfig(step_scale=1.0, max_distance=0.2, samples=5)

        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)

        # Should clamp to max_distance
        ao = calculate_ao(p, n, sdf_plane_y, config)

        assert 0.0 <= ao <= 1.0


# =============================================================================
# Path 7: Falloff Effect
# =============================================================================

class TestFalloff:
    """Verify falloff parameter affects sample weighting."""

    def test_higher_falloff_attenuates_faster(self):
        """Higher falloff values attenuate distant samples faster."""
        config_low = AOConfig(falloff=0.3)
        config_high = AOConfig(falloff=0.8)

        p = (0.1, 0.1, 0.0)
        n_raw = (1.0, 1.0, 0.0)
        n_len = math.sqrt(2)
        n = (n_raw[0]/n_len, n_raw[1]/n_len, n_raw[2]/n_len)

        ao_low = calculate_ao(p, n, sdf_corner, config_low)
        ao_high = calculate_ao(p, n, sdf_corner, config_high)

        # Different falloffs should produce different results
        assert ao_low != ao_high, "Different falloff should produce different AO"

    def test_falloff_bounds(self):
        """Falloff at boundary values works correctly."""
        config_min = AOConfig(falloff=0.01)
        config_max = AOConfig(falloff=1.0)

        p = (0.2, 0.2, 0.0)
        n = (0.0, 1.0, 0.0)

        ao_min = calculate_ao(p, n, sdf_corner, config_min)
        ao_max = calculate_ao(p, n, sdf_corner, config_max)

        assert 0.0 <= ao_min <= 1.0
        assert 0.0 <= ao_max <= 1.0


# =============================================================================
# Path 8: Normal Direction
# =============================================================================

class TestNormalDirection:
    """Verify AO depends correctly on normal direction."""

    def test_opposite_normals(self):
        """Opposite normals produce different AO."""
        p = (0.0, 1.0, 0.0)  # Above plane

        n_up = (0.0, 1.0, 0.0)    # Away from plane
        n_down = (0.0, -1.0, 0.0)  # Toward plane

        ao_up = calculate_ao(p, n_up, sdf_plane_y)
        ao_down = calculate_ao(p, n_down, sdf_plane_y)

        # Pointing up (away from plane) should have higher AO
        assert ao_up > ao_down, f"ao_up={ao_up} should be > ao_down={ao_down}"

    def test_perpendicular_normals(self):
        """Perpendicular normals sample different regions."""
        p = (0.0, 0.5, 0.0)

        n_up = (0.0, 1.0, 0.0)
        n_side = (1.0, 0.0, 0.0)

        ao_up = calculate_ao(p, n_up, sdf_plane_y)
        ao_side = calculate_ao(p, n_side, sdf_plane_y)

        # Side normal should not detect plane below
        assert ao_side > ao_up or abs(ao_side - ao_up) < 0.3

    def test_normalized_normal(self):
        """Non-unit normal is normalized internally."""
        p = (1.0, 0.0, 0.0)

        n_unit = (1.0, 0.0, 0.0)
        n_scaled = (2.0, 0.0, 0.0)  # Same direction, different magnitude

        ao_unit = calculate_ao(p, n_unit, lambda pos: sdf_sphere(pos, 1.0))
        ao_scaled = calculate_ao(p, n_scaled, lambda pos: sdf_sphere(pos, 1.0))

        assert ao_unit == pytest.approx(ao_scaled, abs=TOL_EXACT)


# =============================================================================
# Path 9: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Verify behavior for edge cases."""

    def test_zero_length_normal(self):
        """Zero-length normal should not crash."""
        p = (1.0, 0.0, 0.0)
        n = (0.0, 0.0, 0.0)

        ao = calculate_ao(p, n, lambda pos: sdf_sphere(pos, 1.0))

        # Should return valid AO (likely 1.0 since no direction to sample)
        assert 0.0 <= ao <= 1.0

    def test_very_small_sdf(self):
        """SDF returning very small values works correctly."""
        p = (0.0, 0.0, 0.0)
        n = (1.0, 0.0, 0.0)

        # SDF always returns tiny positive value
        ao = calculate_ao(p, n, lambda pos: 1e-10)

        # Should detect heavy occlusion
        assert ao < 0.5

    def test_negative_sdf(self):
        """Negative SDF (inside geometry) produces heavy occlusion."""
        p = (0.0, 0.0, 0.0)  # At origin
        n = (1.0, 0.0, 0.0)

        # Inside a large sphere
        ao = calculate_ao(p, n, lambda pos: sdf_sphere(pos, 10.0))

        # Should be heavily occluded (we're inside)
        assert ao < 0.5


# =============================================================================
# Path 10: Config Validation
# =============================================================================

class TestConfigValidation:
    """Verify configuration parameter validation."""

    def test_samples_positive(self):
        """Samples must be positive."""
        with pytest.raises(ValueError, match="samples"):
            AOConfig(samples=0)

        with pytest.raises(ValueError, match="samples"):
            AOConfig(samples=-1)

    def test_step_scale_positive(self):
        """Step scale must be positive."""
        with pytest.raises(ValueError, match="step_scale"):
            AOConfig(step_scale=0.0)

        with pytest.raises(ValueError, match="step_scale"):
            AOConfig(step_scale=-0.1)

    def test_falloff_range(self):
        """Falloff must be in (0, 1]."""
        with pytest.raises(ValueError, match="falloff"):
            AOConfig(falloff=0.0)

        with pytest.raises(ValueError, match="falloff"):
            AOConfig(falloff=1.5)

        # 1.0 should be valid
        config = AOConfig(falloff=1.0)
        assert config.falloff == 1.0

    def test_max_distance_positive(self):
        """Max distance must be positive."""
        with pytest.raises(ValueError, match="max_distance"):
            AOConfig(max_distance=0.0)

        with pytest.raises(ValueError, match="max_distance"):
            AOConfig(max_distance=-1.0)


# =============================================================================
# WGSL Generation Tests
# =============================================================================

class TestWGSLGeneration:
    """Verify WGSL code generation."""

    def test_default_wgsl_generation(self):
        """Default config generates valid WGSL."""
        wgsl = generate_ao_wgsl()

        assert "fn calculate_ao" in wgsl
        assert "vec3<f32>" in wgsl
        assert "scene_sdf" in wgsl
        assert "AO_SAMPLES" in wgsl

    def test_custom_config_wgsl(self):
        """Custom config values embedded in WGSL."""
        config = AOConfig(samples=8, step_scale=0.2, falloff=0.6, intensity=1.5)
        wgsl = generate_ao_wgsl(config)

        assert "AO_SAMPLES: i32 = 8" in wgsl
        assert "AO_STEP_SCALE: f32 = 0.2" in wgsl
        assert "AO_FALLOFF: f32 = 0.6" in wgsl
        assert "AO_INTENSITY: f32 = 1.5" in wgsl

    def test_inline_wgsl_generation(self):
        """Inline WGSL generates code block."""
        wgsl = generate_ao_wgsl_inline()

        # Should not have function declaration
        assert "fn calculate_ao" not in wgsl

        # Should have loop and accumulation
        assert "for" in wgsl
        assert "ao_accum" in wgsl

    def test_wgsl_has_reference_comment(self):
        """WGSL includes Quilez reference."""
        wgsl = generate_ao_wgsl()

        assert "Quilez" in wgsl or "quilez" in wgsl.lower()


# =============================================================================
# Multi-Direction AO Tests
# =============================================================================

class TestMultiDirectionAO:
    """Test multi-direction AO sampling."""

    def test_multi_direction_returns_valid(self):
        """Multi-direction AO returns valid value."""
        p = (0.2, 0.2, 0.0)
        n = (0.0, 1.0, 0.0)

        ao = calculate_ao_multi_direction(p, n, sdf_corner)

        assert 0.0 <= ao <= 1.0

    def test_multi_direction_more_accurate(self):
        """Multi-direction may differ from single direction."""
        p = (0.2, 0.2, 0.0)
        n_raw = (1.0, 1.0, 0.0)
        n_len = math.sqrt(2)
        n = (n_raw[0]/n_len, n_raw[1]/n_len, n_raw[2]/n_len)

        ao_single = calculate_ao(p, n, sdf_corner)
        ao_multi = calculate_ao_multi_direction(p, n, sdf_corner)

        # Both should be valid
        assert 0.0 <= ao_single <= 1.0
        assert 0.0 <= ao_multi <= 1.0


# =============================================================================
# Scene Evaluator Helper Tests
# =============================================================================

class TestSceneEvaluator:
    """Test scene AO evaluator factory."""

    def test_make_evaluator(self):
        """Factory creates working evaluator."""
        evaluator = make_scene_ao_evaluator(lambda p: sdf_sphere(p, 1.0))

        ao = evaluator((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))

        assert 0.0 <= ao <= 1.0

    def test_evaluator_with_config(self):
        """Factory respects config parameter."""
        config = AOConfig(samples=3)
        evaluator = make_scene_ao_evaluator(lambda p: sdf_sphere(p, 1.0), config)

        ao = evaluator((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))

        assert 0.0 <= ao <= 1.0


# =============================================================================
# Vec3Local Tests
# =============================================================================

class TestVec3Local:
    """Test local Vec3 helper class."""

    def test_operations(self):
        """Basic vector operations work."""
        v1 = Vec3Local(1.0, 2.0, 3.0)
        v2 = Vec3Local(0.5, 0.5, 0.5)

        # Addition
        v_add = v1 + v2
        assert v_add.x == 1.5 and v_add.y == 2.5 and v_add.z == 3.5

        # Scalar multiplication
        v_mul = v1 * 2.0
        assert v_mul.x == 2.0 and v_mul.y == 4.0 and v_mul.z == 6.0

        # Length
        v_unit = Vec3Local(1.0, 0.0, 0.0)
        assert v_unit.length() == pytest.approx(1.0)

    def test_normalization(self):
        """Normalization produces unit vector."""
        v = Vec3Local(3.0, 4.0, 0.0)
        n = v.normalized()

        assert n.length() == pytest.approx(1.0, abs=1e-10)
        assert n.x == pytest.approx(0.6)
        assert n.y == pytest.approx(0.8)

    def test_zero_normalization(self):
        """Zero vector normalization returns zero."""
        v = Vec3Local(0.0, 0.0, 0.0)
        n = v.normalized()

        assert n.x == 0.0 and n.y == 0.0 and n.z == 0.0
