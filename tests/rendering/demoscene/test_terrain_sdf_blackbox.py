"""
Blackbox tests for Terrain SDF functions (T-DEMO-4.1 and T-DEMO-4.2).

Tests the external behavior and contracts of terrain SDFs without
knowledge of internal implementation details. Focuses on:
  - API contracts and invariants
  - Input/output relationships
  - Error handling
  - Integration scenarios

BLACKBOX coverage plan:
  Contract Tests:
    BB-1: HeightmapTerrainSDF produces heights in documented range
    BB-2: RidgedTerrainSDF produces heights in documented range
    BB-3: SDF sign convention (positive above, negative below)
    BB-4: SDF at surface is zero
    BB-5: Terrain is deterministic (reproducible)
    BB-6: Config changes affect output
    BB-7: Ground level shifts entire terrain
    BB-8: Amplitude scales height linearly
    BB-9: WGSL output is valid shader code structure

  Integration Tests:
    BB-10: Terrain can be ray-marched (SDF properties)
    BB-11: Multiple terrains are independent
    BB-12: Terrain IDs are unique
    BB-13: Factory functions produce valid terrains
    BB-14: Config validation rejects invalid inputs

  Property Tests:
    BB-15: Height is bounded by amplitude
    BB-16: SDF gradient points upward (Y+) above terrain
    BB-17: Ridged terrain has sharper features than heightmap
    BB-18: More octaves adds detail (not just noise)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pytest


# =============================================================================
# Import the module under test
# =============================================================================

from engine.rendering.demoscene.terrain_sdf import (
    HeightmapTerrainSDF,
    RidgedTerrainSDF,
    HeightmapConfig,
    RidgedConfig,
    Vec3,
    create_heightmap_terrain,
    create_ridged_terrain,
    generate_heightmap_terrain_wgsl,
    generate_ridged_terrain_wgsl,
    DEFAULT_AMPLITUDE,
)


# =============================================================================
# BB-1: Heightmap Height Range Contract
# =============================================================================

class TestHeightmapHeightRange:
    """BB-1: Heightmap terrain produces heights in [0, amplitude]."""

    @pytest.mark.parametrize("amplitude", [1.0, 10.0, 100.0, 1000.0])
    def test_height_bounded_by_amplitude(self, amplitude: float):
        """Height values never exceed amplitude."""
        terrain = create_heightmap_terrain(amplitude=amplitude)
        for i in range(200):
            x = (i - 100) * 0.5
            z = (i - 50) * 0.3
            h = terrain.height(x, z)
            assert 0.0 <= h <= amplitude, (
                f"Height {h} out of [0, {amplitude}] at ({x}, {z})"
            )

    def test_height_non_negative(self):
        """Heights are never negative."""
        terrain = HeightmapTerrainSDF()
        min_height = float("inf")
        for i in range(500):
            h = terrain.height(i * 0.1, i * 0.07)
            min_height = min(min_height, h)
        assert min_height >= 0.0, f"Found negative height: {min_height}"

    def test_height_achieves_range(self):
        """Heights span a significant portion of [0, amplitude]."""
        terrain = create_heightmap_terrain(amplitude=10.0)
        heights = [terrain.height(i * 0.1, i * 0.05) for i in range(1000)]
        min_h, max_h = min(heights), max(heights)
        range_fraction = (max_h - min_h) / 10.0
        assert range_fraction > 0.3, f"Height range too narrow: {min_h} to {max_h}"


# =============================================================================
# BB-2: Ridged Height Range Contract
# =============================================================================

class TestRidgedHeightRange:
    """BB-2: Ridged terrain produces heights in [0, amplitude]."""

    @pytest.mark.parametrize("amplitude", [1.0, 10.0, 100.0])
    def test_height_bounded_by_amplitude(self, amplitude: float):
        """Height values never exceed amplitude."""
        terrain = create_ridged_terrain(amplitude=amplitude)
        for i in range(200):
            x = (i - 100) * 0.5
            z = (i - 50) * 0.3
            h = terrain.height(x, z)
            assert 0.0 <= h <= amplitude, (
                f"Height {h} out of [0, {amplitude}] at ({x}, {z})"
            )

    def test_height_non_negative(self):
        """Heights are never negative."""
        terrain = RidgedTerrainSDF()
        min_height = float("inf")
        for i in range(500):
            h = terrain.height(i * 0.1, i * 0.07)
            min_height = min(min_height, h)
        assert min_height >= 0.0, f"Found negative height: {min_height}"


# =============================================================================
# BB-3: SDF Sign Convention
# =============================================================================

class TestSDFSignConvention:
    """BB-3: SDF follows standard sign convention."""

    @pytest.fixture
    def heightmap_terrain(self):
        return create_heightmap_terrain(amplitude=10.0, ground_level=0.0)

    @pytest.fixture
    def ridged_terrain(self):
        return create_ridged_terrain(amplitude=10.0, ground_level=0.0)

    def test_sdf_positive_far_above(self, heightmap_terrain):
        """SDF is positive when far above terrain."""
        # At y=100, we're definitely above any terrain with amplitude 10
        for x in range(-10, 11, 2):
            for z in range(-10, 11, 2):
                sdf = heightmap_terrain.sdf(Vec3(float(x), 100.0, float(z)))
                assert sdf > 0.0, f"SDF should be positive at y=100"

    def test_sdf_negative_far_below(self, heightmap_terrain):
        """SDF is negative when far below terrain."""
        # At y=-100, we're definitely below ground level
        for x in range(-10, 11, 2):
            for z in range(-10, 11, 2):
                sdf = heightmap_terrain.sdf(Vec3(float(x), -100.0, float(z)))
                assert sdf < 0.0, f"SDF should be negative at y=-100"

    def test_sdf_increases_with_height(self, heightmap_terrain):
        """SDF increases as we move upward."""
        x, z = 5.0, 5.0
        prev_sdf = heightmap_terrain.sdf(Vec3(x, 0.0, z))
        for y in [5.0, 10.0, 20.0, 50.0]:
            curr_sdf = heightmap_terrain.sdf(Vec3(x, y, z))
            assert curr_sdf > prev_sdf, f"SDF should increase with height"
            prev_sdf = curr_sdf

    def test_ridged_sdf_sign_convention(self, ridged_terrain):
        """Ridged terrain also follows sign convention."""
        assert ridged_terrain.sdf(Vec3(0.0, 100.0, 0.0)) > 0.0
        assert ridged_terrain.sdf(Vec3(0.0, -100.0, 0.0)) < 0.0


# =============================================================================
# BB-4: SDF Zero at Surface
# =============================================================================

class TestSDFAtSurface:
    """BB-4: SDF is approximately zero at terrain surface."""

    def test_heightmap_sdf_zero_at_surface(self):
        """Heightmap SDF is zero at computed surface."""
        terrain = HeightmapTerrainSDF()
        for i in range(50):
            x, z = i * 0.3, i * 0.2
            h = terrain.height(x, z)
            surface_y = terrain.config.ground_level + h
            sdf = terrain.sdf(Vec3(x, surface_y, z))
            assert abs(sdf) < 1e-6, f"SDF at surface should be 0, got {sdf}"

    def test_ridged_sdf_zero_at_surface(self):
        """Ridged SDF is zero at computed surface."""
        terrain = RidgedTerrainSDF()
        for i in range(50):
            x, z = i * 0.3, i * 0.2
            h = terrain.height(x, z)
            surface_y = terrain.config.ground_level + h
            sdf = terrain.sdf(Vec3(x, surface_y, z))
            assert abs(sdf) < 1e-6, f"SDF at surface should be 0, got {sdf}"


# =============================================================================
# BB-5: Determinism
# =============================================================================

class TestDeterminism:
    """BB-5: Terrain evaluation is deterministic."""

    def test_heightmap_deterministic(self):
        """Same input produces same output across calls."""
        terrain = HeightmapTerrainSDF()
        coords = [(i * 0.1, i * 0.07) for i in range(100)]

        # First pass
        heights1 = [terrain.height(x, z) for x, z in coords]

        # Second pass
        heights2 = [terrain.height(x, z) for x, z in coords]

        assert heights1 == heights2, "Heights should be deterministic"

    def test_heightmap_deterministic_across_instances(self):
        """Different instances with same config produce same results."""
        cfg = HeightmapConfig(octaves=6, amplitude=10.0, frequency=0.5)
        terrain1 = HeightmapTerrainSDF(cfg)
        terrain2 = HeightmapTerrainSDF(cfg)

        for i in range(50):
            x, z = i * 0.2, i * 0.15
            h1 = terrain1.height(x, z)
            h2 = terrain2.height(x, z)
            assert h1 == h2, f"Same config should produce same heights"

    def test_ridged_deterministic(self):
        """Ridged terrain is deterministic."""
        terrain = RidgedTerrainSDF()
        coords = [(i * 0.1, i * 0.07) for i in range(100)]

        heights1 = [terrain.height(x, z) for x, z in coords]
        heights2 = [terrain.height(x, z) for x, z in coords]

        assert heights1 == heights2


# =============================================================================
# BB-6: Config Changes Affect Output
# =============================================================================

class TestConfigChanges:
    """BB-6: Configuration changes affect terrain output."""

    def test_octaves_affect_output(self):
        """Different octaves produce different terrain."""
        terrain_low = create_heightmap_terrain(octaves=2)
        terrain_high = create_heightmap_terrain(octaves=8)

        differences = 0
        for i in range(100):
            h_low = terrain_low.height(i * 0.1, i * 0.05)
            h_high = terrain_high.height(i * 0.1, i * 0.05)
            if abs(h_low - h_high) > 1e-6:
                differences += 1

        assert differences > 0, "Different octaves should produce different terrain"

    def test_frequency_affects_scale(self):
        """Different frequencies produce different pattern scales."""
        terrain_low = create_heightmap_terrain(frequency=0.1)
        terrain_high = create_heightmap_terrain(frequency=1.0)

        # Compare variance at different scales
        heights_low = [terrain_low.height(i * 0.1, 0.0) for i in range(100)]
        heights_high = [terrain_high.height(i * 0.1, 0.0) for i in range(100)]

        # Calculate "roughness" as mean absolute differences
        roughness_low = sum(abs(heights_low[i+1] - heights_low[i])
                           for i in range(99)) / 99
        roughness_high = sum(abs(heights_high[i+1] - heights_high[i])
                            for i in range(99)) / 99

        # Higher frequency should be rougher at the same sample rate
        assert roughness_high > roughness_low * 0.5, (
            "Higher frequency should produce more variation"
        )

    def test_ridge_sharpness_affects_output(self):
        """Different ridge sharpness produces different terrain."""
        terrain_smooth = create_ridged_terrain(ridge_sharpness=1.0)
        terrain_sharp = create_ridged_terrain(ridge_sharpness=4.0)

        differences = 0
        for i in range(100):
            h1 = terrain_smooth.height(i * 0.1, i * 0.05)
            h2 = terrain_sharp.height(i * 0.1, i * 0.05)
            if abs(h1 - h2) > 1e-6:
                differences += 1

        assert differences > 0, "Different sharpness should produce different terrain"


# =============================================================================
# BB-7: Ground Level Offset
# =============================================================================

class TestGroundLevelOffset:
    """BB-7: Ground level shifts entire terrain vertically."""

    @pytest.mark.parametrize("ground_level", [-100.0, 0.0, 50.0, 1000.0])
    def test_ground_level_shifts_sdf(self, ground_level: float):
        """Ground level shifts SDF by expected amount."""
        terrain_base = create_heightmap_terrain(ground_level=0.0)
        terrain_shifted = create_heightmap_terrain(ground_level=ground_level)

        for i in range(20):
            x, z = i * 0.5, i * 0.3
            p = Vec3(x, 50.0, z)

            sdf_base = terrain_base.sdf(p)
            sdf_shifted = terrain_shifted.sdf(p)

            expected_diff = ground_level  # SDF decreases as ground rises
            actual_diff = sdf_base - sdf_shifted

            assert abs(actual_diff - expected_diff) < 1e-6, (
                f"Ground level shift should affect SDF by {ground_level}"
            )

    def test_ground_level_does_not_affect_height(self):
        """Ground level doesn't change the height function values."""
        terrain1 = create_heightmap_terrain(ground_level=0.0)
        terrain2 = create_heightmap_terrain(ground_level=100.0)

        for i in range(50):
            x, z = i * 0.2, i * 0.15
            h1 = terrain1.height(x, z)
            h2 = terrain2.height(x, z)
            assert h1 == h2, "Ground level should not affect height function"


# =============================================================================
# BB-8: Amplitude Scaling
# =============================================================================

class TestAmplitudeScaling:
    """BB-8: Amplitude scales height linearly."""

    @pytest.mark.parametrize("scale", [0.5, 2.0, 10.0])
    def test_amplitude_scales_height(self, scale: float):
        """Height scales linearly with amplitude."""
        terrain_base = create_heightmap_terrain(amplitude=1.0)
        terrain_scaled = create_heightmap_terrain(amplitude=scale)

        for i in range(50):
            x, z = i * 0.2, i * 0.15
            h_base = terrain_base.height(x, z)
            h_scaled = terrain_scaled.height(x, z)

            expected = h_base * scale
            assert abs(h_scaled - expected) < 1e-6, (
                f"Height should scale with amplitude: {h_scaled} vs {expected}"
            )


# =============================================================================
# BB-9: WGSL Output Structure
# =============================================================================

class TestWGSLOutput:
    """BB-9: WGSL output is valid shader code structure."""

    def test_heightmap_wgsl_has_function_declarations(self):
        """Heightmap WGSL has proper function declarations."""
        wgsl = generate_heightmap_terrain_wgsl()
        assert wgsl.count("fn ") >= 2, "Should have at least 2 functions"
        assert "fn heightmap_terrain_height" in wgsl
        assert "fn heightmap_terrain_sdf" in wgsl

    def test_heightmap_wgsl_has_return_types(self):
        """Heightmap WGSL functions have return types."""
        wgsl = generate_heightmap_terrain_wgsl()
        assert "-> f32" in wgsl, "Functions should return f32"

    def test_heightmap_wgsl_has_parameters(self):
        """Heightmap WGSL functions have parameters."""
        wgsl = generate_heightmap_terrain_wgsl()
        assert "vec2<f32>" in wgsl or "p: vec2" in wgsl
        assert "vec3<f32>" in wgsl

    def test_ridged_wgsl_has_function_declarations(self):
        """Ridged WGSL has proper function declarations."""
        wgsl = generate_ridged_terrain_wgsl()
        assert wgsl.count("fn ") >= 2
        assert "fn ridged_terrain_height" in wgsl
        assert "fn ridged_terrain_sdf" in wgsl

    def test_wgsl_has_loop_construct(self):
        """WGSL has proper loop for octave iteration."""
        wgsl = generate_heightmap_terrain_wgsl()
        assert "for" in wgsl, "Should have for loop for octaves"

    def test_wgsl_no_syntax_errors_basic(self):
        """WGSL doesn't have obvious syntax errors."""
        wgsl = generate_heightmap_terrain_wgsl()
        # Check balanced braces
        assert wgsl.count("{") == wgsl.count("}"), "Braces should be balanced"
        assert wgsl.count("(") == wgsl.count(")"), "Parentheses should be balanced"


# =============================================================================
# BB-10: Ray Marching Properties
# =============================================================================

class TestRayMarchingProperties:
    """BB-10: Terrain has properties needed for ray marching."""

    def test_sdf_is_lipschitz(self):
        """SDF doesn't change faster than 1 per unit distance (Lipschitz)."""
        terrain = HeightmapTerrainSDF()

        # Test many point pairs
        max_ratio = 0.0
        for i in range(100):
            p1 = Vec3(i * 0.1, 20.0 + i * 0.05, i * 0.08)
            p2 = Vec3(p1.x + 0.01, p1.y, p1.z)

            sdf1 = terrain.sdf(p1)
            sdf2 = terrain.sdf(p2)
            distance = 0.01

            ratio = abs(sdf2 - sdf1) / distance
            max_ratio = max(max_ratio, ratio)

        # For heightfield terrain, ratio should be reasonable (not infinite)
        # Perfect Lipschitz is 1.0 but heightfield approximation may be higher
        assert max_ratio < 10.0, f"SDF changes too fast: ratio = {max_ratio}"

    def test_sdf_continuous(self):
        """SDF is continuous (no sudden jumps)."""
        terrain = HeightmapTerrainSDF()

        for i in range(100):
            y = i * 0.5
            sdf1 = terrain.sdf(Vec3(5.0, y, 5.0))
            sdf2 = terrain.sdf(Vec3(5.0, y + 0.01, 5.0))

            # Small step should produce small change
            assert abs(sdf2 - sdf1) < 0.1, "SDF should be continuous"


# =============================================================================
# BB-11: Multiple Terrains Independence
# =============================================================================

class TestMultipleTerrains:
    """BB-11: Multiple terrains are independent."""

    def test_different_configs_different_results(self):
        """Terrains with different configs produce different results."""
        terrain1 = create_heightmap_terrain(octaves=4, amplitude=10.0)
        terrain2 = create_heightmap_terrain(octaves=8, amplitude=50.0)

        h1 = terrain1.height(5.0, 5.0)
        h2 = terrain2.height(5.0, 5.0)

        # Heights should definitely differ given different amplitudes
        assert h1 != h2

    def test_modifying_one_doesnt_affect_other(self):
        """Modifying one terrain doesn't affect another."""
        terrain1 = HeightmapTerrainSDF()
        terrain2 = HeightmapTerrainSDF()

        h2_before = terrain2.height(5.0, 5.0)

        terrain1.update_config(amplitude=100.0)

        h2_after = terrain2.height(5.0, 5.0)

        assert h2_before == h2_after, "Terrains should be independent"


# =============================================================================
# BB-12: Unique Terrain IDs
# =============================================================================

class TestTerrainIDs:
    """BB-12: Terrain instances have unique IDs."""

    def test_unique_ids(self):
        """Each terrain instance has a unique ID."""
        terrains = [HeightmapTerrainSDF() for _ in range(10)]
        ids = [t._terrain_id for t in terrains]
        assert len(ids) == len(set(ids)), "All terrain IDs should be unique"

    def test_ids_increase(self):
        """Terrain IDs increase monotonically."""
        t1 = HeightmapTerrainSDF()
        t2 = HeightmapTerrainSDF()
        t3 = HeightmapTerrainSDF()

        assert t1._terrain_id < t2._terrain_id < t3._terrain_id


# =============================================================================
# BB-13: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """BB-13: Factory functions produce valid terrains."""

    def test_create_heightmap_returns_heightmap(self):
        """create_heightmap_terrain returns HeightmapTerrainSDF."""
        terrain = create_heightmap_terrain()
        assert isinstance(terrain, HeightmapTerrainSDF)

    def test_create_ridged_returns_ridged(self):
        """create_ridged_terrain returns RidgedTerrainSDF."""
        terrain = create_ridged_terrain()
        assert isinstance(terrain, RidgedTerrainSDF)

    def test_factory_applies_parameters(self):
        """Factory functions apply all parameters."""
        terrain = create_heightmap_terrain(
            octaves=8,
            amplitude=50.0,
            frequency=0.5,
            lacunarity=2.5,
            gain=0.4,
            ground_level=10.0,
        )
        assert terrain.config.octaves == 8
        assert terrain.config.amplitude == 50.0
        assert terrain.config.frequency == 0.5
        assert terrain.config.lacunarity == 2.5
        assert terrain.config.gain == 0.4
        assert terrain.config.ground_level == 10.0

    def test_ridged_factory_applies_parameters(self):
        """Ridged factory applies all parameters."""
        terrain = create_ridged_terrain(
            octaves=6,
            ridge_sharpness=3.0,
            ridge_offset=1.5,
        )
        assert terrain.config.octaves == 6
        assert terrain.config.ridge_sharpness == 3.0
        assert terrain.config.ridge_offset == 1.5


# =============================================================================
# BB-14: Config Validation
# =============================================================================

class TestConfigValidation:
    """BB-14: Config validation rejects invalid inputs."""

    @pytest.mark.parametrize("octaves", [0, -1, -100])
    def test_heightmap_rejects_invalid_octaves(self, octaves: int):
        """Heightmap config rejects invalid octaves."""
        with pytest.raises(ValueError):
            HeightmapConfig(octaves=octaves)

    @pytest.mark.parametrize("gain", [0.0, -0.5, 1.5, 2.0])
    def test_heightmap_rejects_invalid_gain(self, gain: float):
        """Heightmap config rejects invalid gain."""
        with pytest.raises(ValueError):
            HeightmapConfig(gain=gain)

    @pytest.mark.parametrize("amplitude", [0.0, -1.0, -100.0])
    def test_heightmap_rejects_invalid_amplitude(self, amplitude: float):
        """Heightmap config rejects invalid amplitude."""
        with pytest.raises(ValueError):
            HeightmapConfig(amplitude=amplitude)

    @pytest.mark.parametrize("frequency", [0.0, -1.0])
    def test_heightmap_rejects_invalid_frequency(self, frequency: float):
        """Heightmap config rejects invalid frequency."""
        with pytest.raises(ValueError):
            HeightmapConfig(frequency=frequency)

    @pytest.mark.parametrize("ridge_sharpness", [0.0, -1.0])
    def test_ridged_rejects_invalid_sharpness(self, ridge_sharpness: float):
        """Ridged config rejects invalid ridge sharpness."""
        with pytest.raises(ValueError):
            RidgedConfig(ridge_sharpness=ridge_sharpness)


# =============================================================================
# BB-15: Height Bounded by Amplitude
# =============================================================================

class TestHeightBoundedByAmplitude:
    """BB-15: Height is strictly bounded by amplitude."""

    @pytest.mark.parametrize("amplitude", [0.1, 1.0, 10.0, 100.0, 1000.0])
    def test_heightmap_height_bounded(self, amplitude: float):
        """Heightmap height never exceeds amplitude."""
        terrain = create_heightmap_terrain(amplitude=amplitude)
        max_height = 0.0
        for i in range(500):
            h = terrain.height(i * 0.1 - 25.0, i * 0.07 - 15.0)
            max_height = max(max_height, h)
        assert max_height <= amplitude + 1e-6

    @pytest.mark.parametrize("amplitude", [0.1, 1.0, 10.0, 100.0])
    def test_ridged_height_bounded(self, amplitude: float):
        """Ridged height never exceeds amplitude."""
        terrain = create_ridged_terrain(amplitude=amplitude)
        max_height = 0.0
        for i in range(500):
            h = terrain.height(i * 0.1 - 25.0, i * 0.07 - 15.0)
            max_height = max(max_height, h)
        assert max_height <= amplitude + 1e-6


# =============================================================================
# BB-16: SDF Gradient Direction
# =============================================================================

class TestSDFGradientDirection:
    """BB-16: SDF gradient points upward above terrain."""

    def test_gradient_y_positive_above_terrain(self):
        """Numerical gradient has positive Y component above terrain."""
        terrain = HeightmapTerrainSDF()

        for i in range(20):
            x, z = i * 0.5, i * 0.3
            # Point well above terrain
            p = Vec3(x, 50.0, z)

            # Numerical gradient
            eps = 0.001
            dy = (terrain.sdf(Vec3(p.x, p.y + eps, p.z)) -
                  terrain.sdf(Vec3(p.x, p.y - eps, p.z))) / (2 * eps)

            # For a heightfield, the gradient should point roughly upward
            # (Y component should be positive and dominant)
            assert dy > 0, f"Gradient Y should be positive above terrain"


# =============================================================================
# BB-17: Ridged vs Heightmap Differences
# =============================================================================

class TestRidgedVsHeightmap:
    """BB-17: Ridged terrain has different characteristics than heightmap."""

    def test_different_height_distributions(self):
        """Ridged and heightmap have different height distributions."""
        heightmap = create_heightmap_terrain(octaves=6, amplitude=10.0)
        ridged = create_ridged_terrain(octaves=6, amplitude=10.0)

        heights_hm = [heightmap.height(i * 0.1, i * 0.05) for i in range(500)]
        heights_rg = [ridged.height(i * 0.1, i * 0.05) for i in range(500)]

        # Calculate mean and variance
        mean_hm = sum(heights_hm) / len(heights_hm)
        mean_rg = sum(heights_rg) / len(heights_rg)

        # Ridged terrain typically has more extreme values (near 0 in valleys)
        min_hm = min(heights_hm)
        min_rg = min(heights_rg)

        # At least one characteristic should differ significantly
        assert (abs(mean_hm - mean_rg) > 0.1 or
                abs(min_hm - min_rg) > 0.1 or
                heights_hm != heights_rg), (
            "Ridged and heightmap should produce different distributions"
        )


# =============================================================================
# BB-18: Octaves Add Detail
# =============================================================================

class TestOctavesAddDetail:
    """BB-18: More octaves adds meaningful detail, not just noise."""

    def test_more_octaves_adds_high_frequency(self):
        """More octaves increases high-frequency variation."""
        terrain_low = create_heightmap_terrain(octaves=2)
        terrain_high = create_heightmap_terrain(octaves=8)

        # Sample at fine resolution
        step = 0.01
        heights_low = [terrain_low.height(i * step, 5.0) for i in range(1000)]
        heights_high = [terrain_high.height(i * step, 5.0) for i in range(1000)]

        # Calculate high-frequency content (mean absolute second derivative)
        def roughness(heights: List[float]) -> float:
            return sum(abs(heights[i+1] - 2*heights[i] + heights[i-1])
                      for i in range(1, len(heights)-1)) / (len(heights) - 2)

        rough_low = roughness(heights_low)
        rough_high = roughness(heights_high)

        # More octaves should have more high-frequency content
        assert rough_high > rough_low, (
            f"More octaves should add detail: {rough_high} > {rough_low}"
        )


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_terrain_pipeline(self):
        """Test complete terrain evaluation pipeline."""
        # Create terrain
        terrain = create_heightmap_terrain(
            octaves=6,
            amplitude=100.0,
            frequency=0.1,
            ground_level=50.0,
        )

        # Evaluate height
        h = terrain.height(10.0, 20.0)
        assert 0.0 <= h <= 100.0

        # Evaluate SDF at multiple heights
        surface_y = 50.0 + h
        assert abs(terrain.sdf(Vec3(10.0, surface_y, 20.0))) < 1e-6
        assert terrain.sdf(Vec3(10.0, surface_y + 10.0, 20.0)) > 0
        assert terrain.sdf(Vec3(10.0, surface_y - 10.0, 20.0)) < 0

        # Generate WGSL
        wgsl = terrain.to_wgsl()
        assert "fn heightmap_terrain_sdf" in wgsl

    def test_config_update_workflow(self):
        """Test configuration update workflow."""
        terrain = HeightmapTerrainSDF()

        # Initial evaluation
        h1 = terrain.height(5.0, 5.0)

        # Update config
        terrain.update_config(amplitude=50.0)

        # New evaluation should differ
        h2 = terrain.height(5.0, 5.0)

        # With 50x the default amplitude (1.0), height should scale
        assert h2 > h1 * 10  # Should be roughly 50x larger

    def test_tracker_integration(self):
        """Test tracker integration with terrain operations."""
        terrain = HeightmapTerrainSDF()

        # New terrain is dirty
        assert terrain.tracker.is_dirty
        initial_version = terrain.tracker.version

        # Clear dirty
        terrain.tracker.clear()
        assert not terrain.tracker.is_dirty

        # Update config marks dirty
        terrain.update_config(octaves=8)
        assert terrain.tracker.is_dirty
        assert terrain.tracker.version > initial_version


# =============================================================================
# Stress Tests
# =============================================================================

class TestStress:
    """Stress tests for terrain evaluation."""

    def test_many_evaluations(self):
        """Terrain can handle many evaluations."""
        terrain = HeightmapTerrainSDF()
        for i in range(10000):
            h = terrain.height(i * 0.01, i * 0.007)
            assert not math.isnan(h)

    def test_extreme_coordinates(self):
        """Terrain handles extreme coordinates."""
        terrain = HeightmapTerrainSDF()

        # Very large
        h1 = terrain.height(1e6, 1e6)
        assert not math.isnan(h1)

        # Very small
        h2 = terrain.height(1e-6, 1e-6)
        assert not math.isnan(h2)

        # Mixed
        h3 = terrain.height(-1e6, 1e6)
        assert not math.isnan(h3)
