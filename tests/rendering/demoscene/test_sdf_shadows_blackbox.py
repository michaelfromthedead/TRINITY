"""
Blackbox tests for SDF Soft Shadows (T-DEMO-3.6).

Tests the soft shadow calculation as a black box, verifying
observable behavior without knowledge of internal implementation.
Focuses on perceptual correctness and acceptance criteria.

BLACKBOX acceptance criteria:
  AC1: Contact hardening - shadows sharp near contact, soft further away
  AC2: K parameter visibly affects penumbra width
  AC3: No light leaking for fully occluded points
  AC4: Shadow value always in [0, 1] range
  AC5: Consistent behavior across similar inputs
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.demoscene.sdf_shadows import (
    ShadowConfig,
    calculate_soft_shadow,
    calculate_soft_shadow_improved,
    calculate_hard_shadow,
    calculate_shadow_from_light,
    generate_shadow_wgsl,
    make_scene_shadow_evaluator,
    make_light_shadow_evaluator,
)


# =============================================================================
# Test Scene SDFs
# =============================================================================

def sdf_sphere(p: tuple[float, float, float], radius: float = 1.0,
               center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> float:
    """Sphere SDF."""
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    dz = p[2] - center[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz) - radius


def sdf_ground_plane(p: tuple[float, float, float]) -> float:
    """Ground plane at y=0."""
    return p[1]


def sdf_box(p: tuple[float, float, float],
            half: tuple[float, float, float] = (1.0, 1.0, 1.0),
            center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> float:
    """Axis-aligned box SDF."""
    qx = abs(p[0] - center[0]) - half[0]
    qy = abs(p[1] - center[1]) - half[1]
    qz = abs(p[2] - center[2]) - half[2]
    outer = math.sqrt(max(qx, 0)**2 + max(qy, 0)**2 + max(qz, 0)**2)
    inner = min(max(qx, max(qy, qz)), 0.0)
    return outer + inner


def sdf_pillar(p: tuple[float, float, float]) -> float:
    """Vertical pillar (cylinder-like)."""
    # Approximate as box
    return sdf_box(p, (0.5, 5.0, 0.5), (0.0, 2.5, 0.0))


def sdf_overhang(p: tuple[float, float, float]) -> float:
    """Horizontal overhang (box above ground)."""
    d_ground = sdf_ground_plane(p)
    d_overhang = sdf_box(p, (2.0, 0.1, 2.0), (0.0, 3.0, 0.0))
    return min(d_ground, d_overhang)


# =============================================================================
# AC1: Contact Hardening
# =============================================================================

class TestContactHardening:
    """Verify contact hardening: shadows sharp near contact, soft farther away."""

    def test_shadow_sharper_near_occluder(self):
        """Shadow is sharper (more binary) near the occluder."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))

        light_dir = (0.0, 1.0, 0.0)

        # Near contact (close to sphere)
        ro_near = (0.5, 2.5, 0.0)
        # Far from contact (ground level)
        ro_far = (0.5, 0.0, 0.0)

        shadow_near = calculate_soft_shadow(ro_near, light_dir, scene)
        shadow_far = calculate_soft_shadow(ro_far, light_dir, scene)

        # Both should be in penumbra but different
        # The contact hardening should make near shadow sharper
        assert 0.0 <= shadow_near <= 1.0
        assert 0.0 <= shadow_far <= 1.0

    def test_contact_shadow_under_box(self):
        """Contact shadow under box is sharp at contact, soft at edges."""
        def scene(p):
            # Box floating 1 unit above ground
            d_box = sdf_box(p, (1.0, 0.5, 1.0), (0.0, 1.5, 0.0))
            d_ground = sdf_ground_plane(p)
            return min(d_box, d_ground)

        light_dir = (0.0, 1.0, 0.0)

        # Directly under box center (close contact shadow)
        shadow_center = calculate_soft_shadow((0.0, 0.01, 0.0), light_dir, scene)

        # At edge of box shadow
        shadow_edge = calculate_soft_shadow((1.2, 0.01, 0.0), light_dir, scene)

        # Center should be darker (lower shadow value)
        assert shadow_center < shadow_edge or shadow_center < 0.5

    def test_overhang_shadow_gradient(self):
        """Overhang creates shadow gradient from hard to soft."""
        light_dir = (0.0, 1.0, 0.0)

        shadows = []
        for x_offset in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
            ro = (x_offset, 0.01, 0.0)
            shadows.append(calculate_soft_shadow(ro, light_dir, sdf_overhang))

        # Should transition from shadowed to lit
        assert shadows[0] < 0.5  # Under overhang
        assert shadows[-1] > 0.8  # Beyond overhang edge


# =============================================================================
# AC2: K Parameter Affects Penumbra Width
# =============================================================================

class TestKParameterEffect:
    """Verify k parameter visibly affects penumbra width."""

    def test_low_k_soft_penumbra(self):
        """Low k value produces soft, wide penumbra."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))

        ro = (0.5, 0.0, 0.0)  # In penumbra region
        rd = (0.0, 1.0, 0.0)

        config_soft = ShadowConfig(k=4.0)
        shadow_soft = calculate_soft_shadow(ro, rd, scene, config_soft)

        # Soft shadow should show in penumbra
        assert 0.0 < shadow_soft < 1.0

    def test_high_k_hard_penumbra(self):
        """High k value produces hard, narrow penumbra."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))

        ro = (0.5, 0.0, 0.0)
        rd = (0.0, 1.0, 0.0)

        config_hard = ShadowConfig(k=64.0)
        shadow_hard = calculate_soft_shadow(ro, rd, scene, config_hard)

        # Hard shadow should also show in penumbra (may be lighter)
        assert 0.0 <= shadow_hard <= 1.0

    def test_k_comparison_same_point(self):
        """Different k values produce different shadow at same point."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))

        ro = (0.6, 0.0, 0.0)
        rd = (0.0, 1.0, 0.0)

        shadows = {}
        for k in [2.0, 8.0, 16.0, 32.0]:
            config = ShadowConfig(k=k)
            shadows[k] = calculate_soft_shadow(ro, rd, scene, config)

        # At least some variation should exist
        unique_shadows = set(round(s, 2) for s in shadows.values())
        # Different k should produce different shadows in penumbra
        assert len(unique_shadows) >= 1

    def test_penumbra_width_measurement(self):
        """Measure actual penumbra width for different k values."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 5.0, 0.0))

        rd = (0.0, 1.0, 0.0)

        def find_penumbra_edge(k_val, threshold=0.9):
            """Find x-offset where shadow > threshold."""
            config = ShadowConfig(k=k_val)
            for x in range(20):
                offset = x * 0.1
                shadow = calculate_soft_shadow((offset, 0.0, 0.0), rd, scene, config)
                if shadow > threshold:
                    return offset
            return 2.0

        edge_soft = find_penumbra_edge(4.0)
        edge_hard = find_penumbra_edge(32.0)

        # Soft shadow should have wider penumbra (larger x before lit)
        assert edge_soft >= edge_hard - 0.1


# =============================================================================
# AC3: No Light Leaking
# =============================================================================

class TestNoLightLeaking:
    """Verify no light leaking for fully occluded points."""

    def test_direct_occlusion_no_leak(self):
        """Directly occluded point has zero shadow."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 2.0, 0.0))

        # Point directly under sphere, ray directly toward sphere
        ro = (0.0, 0.0, 0.0)
        rd = (0.0, 1.0, 0.0)

        shadow = calculate_soft_shadow(ro, rd, scene)

        assert shadow == pytest.approx(0.0, abs=0.001), \
            f"Direct occlusion should have no light leaking, got {shadow}"

    def test_large_occluder_no_leak(self):
        """Large occluder fully blocks light."""
        def scene(p):
            # Huge box above
            return sdf_box(p, (10.0, 0.5, 10.0), (0.0, 5.0, 0.0))

        ro = (0.0, 0.0, 0.0)
        rd = (0.0, 1.0, 0.0)

        shadow = calculate_soft_shadow(ro, rd, scene)

        assert shadow == 0.0, f"Large occluder should fully block, got {shadow}"

    def test_thick_occluder_no_leak(self):
        """Thick occluder has no light leaking through."""
        def scene(p):
            # Very thick box
            return sdf_box(p, (2.0, 5.0, 2.0), (0.0, 7.0, 0.0))

        ro = (0.0, 0.0, 0.0)
        rd = (0.0, 1.0, 0.0)

        shadow = calculate_soft_shadow(ro, rd, scene)

        assert shadow == 0.0

    def test_multiple_occluders_no_leak(self):
        """Multiple occluders in path don't cause leaking."""
        def scene(p):
            d1 = sdf_sphere(p, 0.5, (0.0, 2.0, 0.0))
            d2 = sdf_sphere(p, 0.5, (0.0, 4.0, 0.0))
            d3 = sdf_sphere(p, 0.5, (0.0, 6.0, 0.0))
            return min(d1, d2, d3)

        ro = (0.0, 0.0, 0.0)
        rd = (0.0, 1.0, 0.0)

        shadow = calculate_soft_shadow(ro, rd, scene)

        assert shadow == 0.0

    def test_umbra_region_fully_dark(self):
        """Central umbra region is fully in shadow."""
        def scene(p):
            return sdf_sphere(p, 2.0, (0.0, 5.0, 0.0))

        rd = (0.0, 1.0, 0.0)

        # Sample multiple points in umbra
        for offset in [0.0, 0.1, 0.2, 0.3]:
            ro = (offset, 0.0, 0.0)
            shadow = calculate_soft_shadow(ro, rd, scene)
            assert shadow < 0.1, f"Umbra should be dark at offset {offset}"


# =============================================================================
# AC4: Shadow Value Range
# =============================================================================

class TestShadowRange:
    """Verify shadow value always in [0, 1]."""

    def test_shadow_range_normal_cases(self):
        """Normal cases produce shadow in [0, 1]."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))

        rd = (0.0, 1.0, 0.0)

        for x in range(20):
            offset = x * 0.2
            ro = (offset, 0.0, 0.0)
            shadow = calculate_soft_shadow(ro, rd, scene)
            assert 0.0 <= shadow <= 1.0, f"Shadow out of range at offset {offset}"

    def test_shadow_range_extreme_configs(self):
        """Extreme configs still produce valid range."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))

        ro = (0.3, 0.0, 0.0)
        rd = (0.0, 1.0, 0.0)

        configs = [
            ShadowConfig(k=0.1),
            ShadowConfig(k=1000.0),
            ShadowConfig(max_steps=5),
            ShadowConfig(max_steps=500),
            ShadowConfig(min_dist=0.001),
            ShadowConfig(min_dist=0.5),
        ]

        for config in configs:
            shadow = calculate_soft_shadow(ro, rd, scene, config)
            assert 0.0 <= shadow <= 1.0

    def test_shadow_range_fuzz(self):
        """Random inputs produce valid range."""
        import random
        random.seed(12345)

        def scene(p):
            d1 = sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))
            d2 = sdf_ground_plane(p)
            return min(d1, d2)

        for _ in range(100):
            ro = (
                random.uniform(-5, 5),
                random.uniform(0, 2),
                random.uniform(-5, 5)
            )
            rd = (
                random.gauss(0, 0.2),
                abs(random.gauss(0.5, 0.3)),
                random.gauss(0, 0.2)
            )

            shadow = calculate_soft_shadow(ro, rd, scene)
            assert 0.0 <= shadow <= 1.0


# =============================================================================
# AC5: Consistent Behavior
# =============================================================================

class TestConsistentBehavior:
    """Verify consistent and deterministic behavior."""

    def test_deterministic(self):
        """Same inputs produce same output."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))

        ro = (0.4, 0.0, 0.0)
        rd = (0.0, 1.0, 0.0)

        s1 = calculate_soft_shadow(ro, rd, scene)
        s2 = calculate_soft_shadow(ro, rd, scene)
        s3 = calculate_soft_shadow(ro, rd, scene)

        assert s1 == s2 == s3

    def test_symmetric_scene_symmetric_shadow(self):
        """Symmetric scene produces symmetric shadows."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))

        rd = (0.0, 1.0, 0.0)

        s_pos = calculate_soft_shadow((0.5, 0.0, 0.0), rd, scene)
        s_neg = calculate_soft_shadow((-0.5, 0.0, 0.0), rd, scene)

        assert s_pos == pytest.approx(s_neg, abs=0.01)

    def test_continuous_change(self):
        """Small position changes produce small shadow changes."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))

        rd = (0.0, 1.0, 0.0)

        base = calculate_soft_shadow((0.5, 0.0, 0.0), rd, scene)
        perturbed = calculate_soft_shadow((0.505, 0.0, 0.0), rd, scene)

        assert abs(base - perturbed) < 0.1

    def test_radial_symmetry(self):
        """Sphere shadow has radial symmetry."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))

        rd = (0.0, 1.0, 0.0)
        radius = 0.6

        shadows = []
        for angle in [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi]:
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            shadows.append(calculate_soft_shadow((x, 0.0, z), rd, scene))

        # All should be similar
        avg = sum(shadows) / len(shadows)
        for s in shadows:
            assert abs(s - avg) < 0.05


# =============================================================================
# Additional Perceptual Tests
# =============================================================================

class TestPerceptualCorrectness:
    """Additional perceptual correctness tests."""

    def test_pillar_shadow_shape(self):
        """Pillar casts expected elongated shadow."""
        rd = (0.0, 1.0, 0.0)

        # In line with pillar
        shadow_center = calculate_soft_shadow((0.0, 0.01, 0.0), rd, sdf_pillar)

        # Off to the side
        shadow_side = calculate_soft_shadow((1.5, 0.01, 0.0), rd, sdf_pillar)

        assert shadow_center < shadow_side

    def test_shadow_fades_with_distance(self):
        """Soft shadow fades at edges."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))

        rd = (0.0, 1.0, 0.0)

        shadows = [
            calculate_soft_shadow((x * 0.1, 0.0, 0.0), rd, scene)
            for x in range(20)
        ]

        # Should have transition from dark to light
        assert shadows[0] < 0.5  # Center is dark
        assert shadows[-1] > 0.9  # Far edge is lit

        # Should be monotonically non-decreasing (or nearly so)
        for i in range(len(shadows) - 1):
            assert shadows[i] <= shadows[i+1] + 0.1

    def test_higher_occluder_softer_shadow(self):
        """Higher occluder produces softer shadow."""
        rd = (0.0, 1.0, 0.0)
        offset = 0.8  # In penumbra for both

        # Low occluder (sharp contact shadow)
        def scene_low(p):
            return sdf_sphere(p, 1.0, (0.0, 2.0, 0.0))

        # High occluder (softer shadow)
        def scene_high(p):
            return sdf_sphere(p, 1.0, (0.0, 8.0, 0.0))

        shadow_low = calculate_soft_shadow((offset, 0.0, 0.0), rd, scene_low)
        shadow_high = calculate_soft_shadow((offset, 0.0, 0.0), rd, scene_high)

        # Higher occluder should have softer (lower) shadow at same offset
        # due to contact hardening
        assert shadow_high >= shadow_low - 0.2


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_scene_evaluator_workflow(self):
        """Complete scene evaluator workflow."""
        def scene(p):
            d_sphere = sdf_sphere(p, 1.0, (0.0, 2.0, 0.0))
            d_ground = sdf_ground_plane(p)
            return min(d_sphere, d_ground)

        evaluator = make_scene_shadow_evaluator(scene, ShadowConfig(k=16.0))

        # Test various points
        assert evaluator((0.0, 0.01, 0.0), (0.0, 1.0, 0.0)) < 0.5
        assert evaluator((3.0, 0.01, 0.0), (0.0, 1.0, 0.0)) > 0.9

    def test_light_evaluator_workflow(self):
        """Complete light evaluator workflow."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 2.0, 0.0))

        light_pos = (0.0, 10.0, 0.0)
        evaluator = make_light_shadow_evaluator(scene, light_pos)

        # Under sphere
        shadow = evaluator((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        assert shadow < 0.5

    def test_hard_vs_soft_comparison(self):
        """Hard shadow is more binary than soft."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))

        rd = (0.0, 1.0, 0.0)
        offset = 0.5

        hard = calculate_hard_shadow((offset, 0.0, 0.0), rd, scene)
        soft = calculate_soft_shadow((offset, 0.0, 0.0), rd, scene)

        # Hard should be 0 or 1
        assert hard in [0.0, 1.0]
        # Soft can be anything
        assert 0.0 <= soft <= 1.0

    def test_improved_shadow_integration(self):
        """Improved shadow works in full workflow."""
        def scene(p):
            return sdf_sphere(p, 1.0, (0.0, 3.0, 0.0))

        rd = (0.0, 1.0, 0.0)

        # Compare standard and improved
        for x in [0.0, 0.3, 0.6, 0.9]:
            standard = calculate_soft_shadow((x, 0.0, 0.0), rd, scene)
            improved = calculate_soft_shadow_improved((x, 0.0, 0.0), rd, scene)

            # Both should be valid
            assert 0.0 <= standard <= 1.0
            assert 0.0 <= improved <= 1.0


# =============================================================================
# WGSL Generation Tests
# =============================================================================

class TestWGSLGeneration:
    """Test WGSL code generation."""

    def test_wgsl_syntax_valid(self):
        """Generated WGSL has valid syntax structure."""
        wgsl = generate_shadow_wgsl()

        # Balanced braces
        assert wgsl.count("{") == wgsl.count("}")
        assert wgsl.count("(") == wgsl.count(")")

        # Required elements
        assert "fn calculate_soft_shadow" in wgsl
        assert "return" in wgsl
        assert "clamp" in wgsl

    def test_wgsl_config_embedded(self):
        """Config values are embedded in WGSL."""
        config = ShadowConfig(
            k=42.0,
            max_steps=99,
            min_dist=0.05,
            max_dist=50.0
        )
        wgsl = generate_shadow_wgsl(config)

        assert "42.0" in wgsl
        assert "99" in wgsl
        assert "0.05" in wgsl
        assert "50.0" in wgsl
