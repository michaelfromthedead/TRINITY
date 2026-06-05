"""
Blackbox tests for Procedural Palettes (T-DEMO-4.14, T-DEMO-4.15, T-DEMO-4.16).

Tests observable behavior without knowledge of internal implementation:

T-DEMO-4.14: TerrainPalette
  - Correct colors at known heights
  - Smooth transitions (no sudden jumps)
  - Full height range coverage

T-DEMO-4.15: ProceduralPattern
  - Correct pattern outputs for known inputs
  - Output always in [0, 1] range
  - Pattern visual characteristics

T-DEMO-4.16: PaletteLUT
  - Correct color lookup
  - Valid texture data output
  - Proper gradient generation

BLACKBOX coverage plan (40+ tests):
  Scenario 1-10:  TerrainPalette functional behavior
  Scenario 11-20: ProceduralPattern functional behavior
  Scenario 21-30: PaletteLUT functional behavior
  Scenario 31-40: Integration and edge cases
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.demoscene.procedural_palette import (
    # T-DEMO-4.14
    TerrainPalette,
    TerrainZone,
    DEFAULT_TERRAIN_ZONES,
    # T-DEMO-4.15
    ProceduralPattern,
    PatternType,
    # T-DEMO-4.16
    PaletteLUT,
    MaterialPaletteMap,
    # WGSL
    generate_palette_wgsl,
)


# =============================================================================
# Constants
# =============================================================================

TOL_COLOR = 0.05      # Color component tolerance for visual tests
TOL_PATTERN = 0.1     # Pattern value tolerance


# =============================================================================
# T-DEMO-4.14: TerrainPalette Blackbox Tests
# =============================================================================

class TestTerrainPaletteBlackbox:
    """Blackbox tests for TerrainPalette observable behavior."""

    def test_water_color_at_low_height(self) -> None:
        """Verify water-like color at very low heights."""
        palette = TerrainPalette()
        color = palette.sample(0.0)

        # Should be blueish (water)
        assert color[2] > color[0]  # Blue > Red
        assert color[2] > color[1]  # Blue > Green

    def test_snow_color_at_high_height(self) -> None:
        """Verify snow-like color at very high heights."""
        palette = TerrainPalette()
        color = palette.sample(1.0)

        # Should be white-ish (snow)
        assert color[0] > 0.9
        assert color[1] > 0.9
        assert color[2] > 0.9

    def test_vegetation_at_mid_height(self) -> None:
        """Verify greenish color at mid-range heights (vegetation zone)."""
        palette = TerrainPalette()
        color = palette.sample(0.4)

        # Should have green component (grass/vegetation)
        assert color[1] > 0.2

    def test_no_sudden_color_jumps(self) -> None:
        """Verify colors transition smoothly without sudden jumps."""
        palette = TerrainPalette()

        prev_color = palette.sample(0.0)
        for i in range(1, 101):
            height = i / 100.0
            color = palette.sample(height)

            # Each component should not jump more than 10% in 1% height change
            for j in range(3):
                diff = abs(color[j] - prev_color[j])
                assert diff < 0.15, f"Color jump at height {height}: {diff}"

            prev_color = color

    def test_height_range_coverage(self) -> None:
        """Verify palette covers full height range [0, 1]."""
        palette = TerrainPalette()

        # Should not raise for any valid height
        for i in range(11):
            height = i / 10.0
            color = palette.sample(height)
            assert len(color) == 3
            assert all(0.0 <= c <= 1.0 for c in color)

    def test_roughness_varies_with_height(self) -> None:
        """Verify roughness changes across height zones."""
        palette = TerrainPalette()

        roughness_values = []
        for i in range(11):
            height = i / 10.0
            _, roughness = palette.sample_with_roughness(height)
            roughness_values.append(roughness)

        # Roughness should vary (not all same value)
        assert max(roughness_values) - min(roughness_values) > 0.1

    def test_custom_terrain_zones(self) -> None:
        """Verify custom terrain zones work correctly."""
        zones = (
            TerrainZone(height=0.0, color=(1.0, 0.0, 0.0)),  # Red
            TerrainZone(height=0.5, color=(0.0, 1.0, 0.0)),  # Green
            TerrainZone(height=1.0, color=(0.0, 0.0, 1.0)),  # Blue
        )
        palette = TerrainPalette(zones=zones)

        # Check extreme heights
        red = palette.sample(0.0)
        assert red[0] > 0.9

        blue = palette.sample(1.0)
        assert blue[2] > 0.9

    def test_terrain_palette_is_deterministic(self) -> None:
        """Verify same input always produces same output."""
        palette = TerrainPalette()

        color1 = palette.sample(0.42)
        color2 = palette.sample(0.42)

        assert color1 == color2

    def test_out_of_range_heights_handled(self) -> None:
        """Verify out-of-range heights don't crash."""
        palette = TerrainPalette()

        # Should not raise
        _ = palette.sample(-100.0)
        _ = palette.sample(100.0)
        _ = palette.sample(float('inf'))

    def test_empty_zones_rejected(self) -> None:
        """Verify empty zone list is rejected."""
        with pytest.raises(ValueError):
            TerrainPalette(zones=())


# =============================================================================
# T-DEMO-4.15: ProceduralPattern Blackbox Tests
# =============================================================================

class TestProceduralPatternBlackbox:
    """Blackbox tests for ProceduralPattern observable behavior."""

    def test_stripes_alternating_values(self) -> None:
        """Verify stripes create alternating high/low values."""
        pattern = ProceduralPattern(PatternType.STRIPES, frequency=1.0)

        # Sample along X axis
        values = [pattern.evaluate((x * 0.1, 0.0, 0.0)) for x in range(10)]

        # Should have both high and low values
        assert max(values) > 0.7
        assert min(values) < 0.3

    def test_checkerboard_binary_values(self) -> None:
        """Verify checkerboard produces binary (0 or 1) values."""
        pattern = ProceduralPattern(PatternType.CHECKERBOARD, frequency=1.0)

        for x in range(-3, 4):
            for z in range(-3, 4):
                v = pattern.evaluate((x + 0.5, 0.0, z + 0.5))
                assert v == 0.0 or v == 1.0

    def test_checkerboard_alternates(self) -> None:
        """Verify adjacent checkerboard cells have different values."""
        pattern = ProceduralPattern(PatternType.CHECKERBOARD, frequency=1.0)

        v1 = pattern.evaluate((0.5, 0.0, 0.5))
        v2 = pattern.evaluate((1.5, 0.0, 0.5))
        v3 = pattern.evaluate((0.5, 0.0, 1.5))

        assert v1 != v2
        assert v1 != v3
        assert v2 == v3  # Diagonal cells same

    def test_wood_grain_ring_structure(self) -> None:
        """Verify wood grain creates ring-like structure."""
        pattern = ProceduralPattern(PatternType.WOOD_GRAIN, frequency=2.0, noise_seed=0)

        # Sample at increasing radii
        samples = [pattern.evaluate((r, 0.0, 0.0)) for r in [0.0, 0.5, 1.0, 1.5, 2.0]]

        # Should have variation (rings)
        assert max(samples) - min(samples) > 0.3

    def test_marble_veined_appearance(self) -> None:
        """Verify marble creates veined appearance."""
        pattern = ProceduralPattern(PatternType.MARBLE, frequency=1.0)

        # Sample along a line with smaller steps for smoother variation
        samples = [pattern.evaluate((x * 0.1, 0.0, 0.0)) for x in range(30)]

        # Should have continuous variation (veins) - most variations should be small
        variations = [abs(samples[i+1] - samples[i]) for i in range(len(samples)-1)]
        small_variations = sum(1 for v in variations if v < 0.5)
        assert small_variations > len(variations) * 0.8  # At least 80% are smooth

    def test_rust_patchy_appearance(self) -> None:
        """Verify rust creates patchy appearance."""
        pattern = ProceduralPattern(PatternType.RUST, frequency=1.0)

        # Sample grid
        values = []
        for x in range(5):
            for z in range(5):
                values.append(pattern.evaluate((x, 0.0, z)))

        # Should have variation (patches)
        assert max(values) - min(values) > 0.2

    def test_all_patterns_output_in_range(self) -> None:
        """Verify all pattern types output values in [0, 1]."""
        import random

        for pattern_type in PatternType:
            pattern = ProceduralPattern(pattern_type, frequency=1.0)

            for _ in range(50):
                p = (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5))
                v = pattern.evaluate(p)
                assert 0.0 <= v <= 1.0, f"{pattern_type.name} produced {v}"

    def test_pattern_frequency_affects_scale(self) -> None:
        """Verify frequency parameter affects pattern scale."""
        p_low = ProceduralPattern(PatternType.STRIPES, frequency=1.0)
        p_high = ProceduralPattern(PatternType.STRIPES, frequency=4.0)

        # High frequency should repeat more often in same range
        # Sample and count zero crossings
        low_values = [p_low.evaluate((x * 0.1, 0.0, 0.0)) for x in range(20)]
        high_values = [p_high.evaluate((x * 0.1, 0.0, 0.0)) for x in range(20)]

        def count_crossings(vals: list) -> int:
            return sum(1 for i in range(len(vals)-1)
                      if (vals[i] - 0.5) * (vals[i+1] - 0.5) < 0)

        low_crossings = count_crossings(low_values)
        high_crossings = count_crossings(high_values)

        assert high_crossings > low_crossings

    def test_pattern_color_evaluation(self) -> None:
        """Verify pattern can return colors when configured."""
        pattern = ProceduralPattern(
            PatternType.STRIPES,
            colors=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )

        color = pattern.evaluate_color((0.25, 0.0, 0.0))  # High value point

        # Should be near blue (high pattern value)
        assert color[2] > 0.5

    def test_pattern_without_colors_returns_grayscale(self) -> None:
        """Verify pattern without colors returns grayscale."""
        pattern = ProceduralPattern(PatternType.STRIPES)

        color = pattern.evaluate_color((0.0, 0.0, 0.0))

        # Should be grayscale (R=G=B)
        assert abs(color[0] - color[1]) < 0.01
        assert abs(color[1] - color[2]) < 0.01

    def test_radial_increases_with_distance(self) -> None:
        """Verify radial pattern increases with distance from origin."""
        pattern = ProceduralPattern(PatternType.RADIAL, frequency=0.5)

        v_origin = pattern.evaluate((0.0, 0.0, 0.0))
        v_near = pattern.evaluate((1.0, 0.0, 0.0))
        v_far = pattern.evaluate((2.0, 0.0, 0.0))

        assert v_origin < v_near < v_far

    def test_gradient_increases_along_axis(self) -> None:
        """Verify gradient patterns increase along their axis."""
        px = ProceduralPattern(PatternType.GRADIENT_X, frequency=0.5)

        v1 = px.evaluate((-1.0, 0.0, 0.0))
        v2 = px.evaluate((0.0, 0.0, 0.0))
        v3 = px.evaluate((1.0, 0.0, 0.0))

        assert v1 < v2 < v3

    def test_invalid_frequency_rejected(self) -> None:
        """Verify invalid frequency is rejected."""
        with pytest.raises(ValueError):
            ProceduralPattern(PatternType.STRIPES, frequency=0.0)

        with pytest.raises(ValueError):
            ProceduralPattern(PatternType.STRIPES, frequency=-1.0)


# =============================================================================
# T-DEMO-4.16: PaletteLUT Blackbox Tests
# =============================================================================

class TestPaletteLUTBlackbox:
    """Blackbox tests for PaletteLUT observable behavior."""

    def test_gradient_start_color(self) -> None:
        """Verify gradient LUT starts with correct color."""
        lut = PaletteLUT.from_gradient([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])

        color = lut.lookup(0.0)

        assert color[0] > 0.95  # Red
        assert color[2] < 0.05  # Not blue

    def test_gradient_end_color(self) -> None:
        """Verify gradient LUT ends with correct color."""
        lut = PaletteLUT.from_gradient([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])

        color = lut.lookup(1.0)

        assert color[0] < 0.05  # Not red
        assert color[2] > 0.95  # Blue

    def test_gradient_midpoint_color(self) -> None:
        """Verify gradient LUT interpolates at midpoint."""
        lut = PaletteLUT.from_gradient([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])

        color = lut.lookup(0.5)

        # Should be approximately gray
        assert 0.4 < color[0] < 0.6
        assert 0.4 < color[1] < 0.6
        assert 0.4 < color[2] < 0.6

    def test_bake_produces_valid_data(self) -> None:
        """Verify bake produces valid RGBA8 texture data."""
        lut = PaletteLUT.from_gradient([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])

        data = lut.bake()

        # Should be 1KB
        assert len(data) == 1024

        # Should be valid bytes (0-255)
        assert all(0 <= b <= 255 for b in data)

    def test_bake_first_entry_black(self) -> None:
        """Verify bake produces black at start of gradient."""
        lut = PaletteLUT.from_gradient([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])

        data = lut.bake()

        # First entry (RGBA)
        assert data[0] < 5    # R ~= 0
        assert data[1] < 5    # G ~= 0
        assert data[2] < 5    # B ~= 0
        assert data[3] > 250  # A ~= 1

    def test_bake_last_entry_white(self) -> None:
        """Verify bake produces white at end of gradient."""
        lut = PaletteLUT.from_gradient([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])

        data = lut.bake()

        # Last entry (index 255 * 4 bytes)
        offset = 255 * 4
        assert data[offset] > 250      # R ~= 1
        assert data[offset + 1] > 250  # G ~= 1
        assert data[offset + 2] > 250  # B ~= 1
        assert data[offset + 3] > 250  # A ~= 1

    def test_lookup_rgb_returns_three_components(self) -> None:
        """Verify lookup_rgb returns RGB without alpha."""
        lut = PaletteLUT.from_gradient([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])

        rgb = lut.lookup_rgb(0.5)

        assert len(rgb) == 3

    def test_bilinear_produces_smooth_lookup(self) -> None:
        """Verify bilinear filtering produces smooth color transitions."""
        lut = PaletteLUT.from_gradient([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], bilinear=True)

        # Sample at fractional indices
        v1 = lut.lookup(0.5)
        v2 = lut.lookup(0.51)

        # Should be very close
        assert abs(v1[0] - v2[0]) < 0.05

    def test_from_terrain_produces_valid_lut(self) -> None:
        """Verify from_terrain produces valid LUT."""
        terrain = TerrainPalette()
        lut = PaletteLUT.from_terrain(terrain)

        assert len(lut.entries) == 256

        # Should have terrain-like colors
        low = lut.lookup(0.0)
        high = lut.lookup(1.0)

        # Low should be bluish (water)
        assert low[2] > low[0]

        # High should be whitish (snow)
        assert high[0] > 0.8

    def test_set_entry_modifies_lookup(self) -> None:
        """Verify set_entry affects lookup."""
        lut = PaletteLUT.from_gradient([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])

        # Set middle entry to red
        lut.set_entry(128, (1.0, 0.0, 0.0, 1.0))

        color = lut.lookup(128 / 255.0)
        assert color[0] > 0.9  # Should be red now

    def test_multicolor_gradient(self) -> None:
        """Verify multi-color gradient interpolates correctly."""
        colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
        ]
        lut = PaletteLUT.from_gradient(colors)

        # Start should be red
        start = lut.lookup(0.0)
        assert start[0] > 0.9

        # Middle should be greenish
        middle = lut.lookup(0.5)
        assert middle[1] > 0.5

        # End should be blue
        end = lut.lookup(1.0)
        assert end[2] > 0.9

    def test_custom_alpha(self) -> None:
        """Verify custom alpha is applied."""
        lut = PaletteLUT.from_gradient([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], alpha=0.5)

        color = lut.lookup(0.5)
        assert abs(color[3] - 0.5) < 0.01


# =============================================================================
# MaterialPaletteMap Blackbox Tests
# =============================================================================

class TestMaterialPaletteMapBlackbox:
    """Blackbox tests for MaterialPaletteMap."""

    def test_different_materials_different_palettes(self) -> None:
        """Verify different materials use different palettes."""
        lut_red = PaletteLUT.from_gradient([(1.0, 0.0, 0.0), (1.0, 0.5, 0.5)])
        lut_blue = PaletteLUT.from_gradient([(0.0, 0.0, 1.0), (0.5, 0.5, 1.0)])

        mapper = MaterialPaletteMap()
        mapper.assign(0, lut_red)
        mapper.assign(1, lut_blue)

        color_0 = mapper.lookup(0, 0.5)
        color_1 = mapper.lookup(1, 0.5)

        # Material 0 should be reddish
        assert color_0[0] > color_0[2]

        # Material 1 should be bluish
        assert color_1[2] > color_1[0]

    def test_unassigned_material_uses_default(self) -> None:
        """Verify unassigned material uses default palette."""
        default_lut = PaletteLUT.from_gradient([(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)])
        mapper = MaterialPaletteMap(default_palette=default_lut)

        color = mapper.lookup(999, 0.5)

        # Should be gray (default)
        assert abs(color[0] - 0.5) < 0.1
        assert abs(color[1] - 0.5) < 0.1
        assert abs(color[2] - 0.5) < 0.1

    def test_unassigned_without_default_returns_gray(self) -> None:
        """Verify unassigned material without default returns gray."""
        mapper = MaterialPaletteMap()

        color = mapper.lookup(999, 0.5)

        assert color == (0.5, 0.5, 0.5, 1.0)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_terrain_to_lut_pipeline(self) -> None:
        """Verify terrain -> LUT pipeline produces correct results."""
        terrain = TerrainPalette()
        lut = PaletteLUT.from_terrain(terrain)

        # Lookup at various heights should match terrain sampling
        for i in range(10):
            height = i / 9.0
            terrain_color = terrain.sample(height)
            lut_color = lut.lookup_rgb(height)

            for j in range(3):
                assert abs(terrain_color[j] - lut_color[j]) < 0.02

    def test_pattern_to_lut_pipeline(self) -> None:
        """Verify pattern -> LUT pipeline works."""
        pattern = ProceduralPattern(
            PatternType.STRIPES,
            colors=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )

        # Create LUT from pattern sampled along X
        positions = [(i / 127.5 - 1.0, 0.0, 0.0) for i in range(256)]
        lut = PaletteLUT.from_pattern(pattern, sample_positions=positions)

        assert len(lut.entries) == 256

    def test_wgsl_generation_produces_valid_syntax(self) -> None:
        """Verify WGSL generation produces syntactically valid code."""
        terrain = TerrainPalette()
        pattern = ProceduralPattern(PatternType.MARBLE)
        lut = PaletteLUT.from_gradient([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])

        wgsl = generate_palette_wgsl(terrain=terrain, pattern=pattern, lut=lut)

        # Should contain function definitions
        assert "fn terrain_palette" in wgsl
        assert "fn procedural_pattern" in wgsl
        assert "fn palette_lookup" in wgsl

        # Should have proper WGSL syntax elements
        assert "vec3<f32>" in wgsl
        assert "return" in wgsl
        assert "}" in wgsl

    def test_material_palette_with_patterns(self) -> None:
        """Verify material palette map works with pattern-based LUTs."""
        # Create LUTs from patterns
        stripes = ProceduralPattern(PatternType.STRIPES, colors=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        checker = ProceduralPattern(PatternType.CHECKERBOARD, colors=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))

        pos = [(i / 127.5 - 1.0, 0.0, 0.0) for i in range(256)]
        lut_stripes = PaletteLUT.from_pattern(stripes, sample_positions=pos)
        lut_checker = PaletteLUT.from_pattern(checker, sample_positions=pos)

        mapper = MaterialPaletteMap()
        mapper.assign(0, lut_stripes)
        mapper.assign(1, lut_checker)

        # Should be able to lookup colors
        color_0 = mapper.lookup(0, 0.5)
        color_1 = mapper.lookup(1, 0.5)

        assert len(color_0) == 4
        assert len(color_1) == 4


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_single_color_gradient(self) -> None:
        """Verify single-color gradient fails gracefully."""
        with pytest.raises(ValueError):
            PaletteLUT.from_gradient([(1.0, 0.0, 0.0)])

    def test_many_color_gradient(self) -> None:
        """Verify many-color gradient works."""
        colors = [(i / 9.0, 0.0, 1.0 - i / 9.0) for i in range(10)]
        lut = PaletteLUT.from_gradient(colors)

        assert len(lut.entries) == 256

    def test_zero_frequency_pattern_rejected(self) -> None:
        """Verify zero frequency is rejected."""
        with pytest.raises(ValueError):
            ProceduralPattern(PatternType.STRIPES, frequency=0.0)

    def test_negative_blend_width_rejected(self) -> None:
        """Verify negative blend width is rejected."""
        zones = (TerrainZone(height=0.0, color=(0.0, 0.0, 0.0)),)
        with pytest.raises(ValueError):
            TerrainPalette(zones=zones, blend_width=-0.1)

    def test_excessive_blend_width_rejected(self) -> None:
        """Verify excessive blend width is rejected."""
        zones = (TerrainZone(height=0.0, color=(0.0, 0.0, 0.0)),)
        with pytest.raises(ValueError):
            TerrainPalette(zones=zones, blend_width=0.6)

    def test_pattern_with_extreme_positions(self) -> None:
        """Verify patterns handle extreme positions."""
        pattern = ProceduralPattern(PatternType.MARBLE)

        # Should not crash
        v1 = pattern.evaluate((1e6, 1e6, 1e6))
        v2 = pattern.evaluate((-1e6, -1e6, -1e6))

        assert 0.0 <= v1 <= 1.0
        assert 0.0 <= v2 <= 1.0

    def test_lut_with_transparent_alpha(self) -> None:
        """Verify LUT handles transparent alpha."""
        lut = PaletteLUT.from_gradient([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)], alpha=0.0)

        color = lut.lookup(0.5)
        assert color[3] == 0.0

    def test_terrain_with_coincident_heights(self) -> None:
        """Verify terrain handles zones at same height gracefully."""
        # Two zones at exactly the same height (after first at 0)
        # This is technically "sorted" but creates a degenerate case
        zones = (
            TerrainZone(height=0.0, color=(0.0, 0.0, 0.0)),
            TerrainZone(height=0.5, color=(0.5, 0.5, 0.5)),
            TerrainZone(height=0.5, color=(1.0, 1.0, 1.0)),  # Same height as previous
        )
        # This violates sorted requirement (not strictly increasing)
        with pytest.raises(ValueError):
            TerrainPalette(zones=zones)

    def test_default_terrain_produces_expected_colors(self) -> None:
        """Verify default terrain produces visually expected colors."""
        palette = TerrainPalette()

        # Water (low) should be dark and blue
        water = palette.sample(0.1)
        assert water[2] > 0.1  # Has blue

        # Sand should be brownish
        sand = palette.sample(0.3)
        assert sand[0] > sand[2]  # More red/yellow than blue

        # Grass should be green-dominant
        grass = palette.sample(0.4)
        assert grass[1] > 0.15  # Has green

        # Snow should be bright
        snow = palette.sample(0.95)
        assert sum(snow) > 2.5  # Bright overall
