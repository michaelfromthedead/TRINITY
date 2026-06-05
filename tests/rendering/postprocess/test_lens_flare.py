"""Tests for Lens Flare Post-Processing Effects.

Tests cover:
- Ghost UV mirroring across screen center
- Ghost position calculation with offset
- Chromatic shift directions for RGB channels
- Halo radial falloff calculation
- Streak direction and falloff
- Quality level configurations
- Budget estimates per quality level
- Settings validation and interpolation
- Factory function presets
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.postprocess.lens_flare import (
    GhostSettings,
    HaloSettings,
    LensFlareEffect,
    LensFlareQuality,
    LensFlareSettings,
    StreakSettings,
    create_cinematic_lens_flare,
    create_disabled_lens_flare,
    create_subtle_lens_flare,
)


class TestGhostUVMirroring:
    """Tests for ghost UV mirroring across screen center."""

    def test_mirror_uv_upper_right_to_lower_left(self) -> None:
        """UV in upper-right should mirror to lower-left."""
        effect = LensFlareEffect()
        result = effect.mirror_uv((0.8, 0.2))
        assert result[0] == pytest.approx(0.2)
        assert result[1] == pytest.approx(0.8)

    def test_mirror_uv_lower_left_to_upper_right(self) -> None:
        """UV in lower-left should mirror to upper-right."""
        effect = LensFlareEffect()
        result = effect.mirror_uv((0.2, 0.8))
        assert result[0] == pytest.approx(0.8)
        assert result[1] == pytest.approx(0.2)

    def test_mirror_uv_center_stays_center(self) -> None:
        """UV at center should stay at center."""
        effect = LensFlareEffect()
        result = effect.mirror_uv((0.5, 0.5))
        assert result == (0.5, 0.5)

    def test_mirror_uv_corner_to_opposite_corner(self) -> None:
        """Corner UV should mirror to opposite corner."""
        effect = LensFlareEffect()
        result = effect.mirror_uv((0.0, 0.0))
        assert result == (1.0, 1.0)

    def test_mirror_uv_is_involutory(self) -> None:
        """Mirroring twice should return original UV."""
        effect = LensFlareEffect()
        original = (0.3, 0.7)
        mirrored = effect.mirror_uv(original)
        back = effect.mirror_uv(mirrored)
        assert abs(back[0] - original[0]) < 1e-10
        assert abs(back[1] - original[1]) < 1e-10


class TestGhostPositionCalculation:
    """Tests for ghost UV position calculation."""

    def test_ghost_uv_with_zero_offset_at_center(self) -> None:
        """Ghost with zero offset should appear at screen center."""
        effect = LensFlareEffect()
        ghost = GhostSettings(offset=0.0)
        result = effect.calculate_ghost_uv((0.8, 0.2), ghost)
        assert abs(result[0] - 0.5) < 1e-10
        assert abs(result[1] - 0.5) < 1e-10

    def test_ghost_uv_with_full_offset_at_mirror(self) -> None:
        """Ghost with offset=1.0 should appear at mirrored position."""
        effect = LensFlareEffect()
        ghost = GhostSettings(offset=1.0)
        result = effect.calculate_ghost_uv((0.8, 0.2), ghost)
        assert abs(result[0] - 0.2) < 1e-10
        assert abs(result[1] - 0.8) < 1e-10

    def test_ghost_uv_with_half_offset_is_halfway(self) -> None:
        """Ghost with offset=0.5 should appear halfway between center and mirror."""
        effect = LensFlareEffect()
        ghost = GhostSettings(offset=0.5)
        # Source at (0.8, 0.2), mirrored is (0.2, 0.8)
        # Center is (0.5, 0.5)
        # Expected: center + (mirror - center) * 0.5
        # = (0.5, 0.5) + (0.2 - 0.5, 0.8 - 0.5) * 0.5
        # = (0.5 - 0.15, 0.5 + 0.15) = (0.35, 0.65)
        result = effect.calculate_ghost_uv((0.8, 0.2), ghost)
        assert abs(result[0] - 0.35) < 1e-10
        assert abs(result[1] - 0.65) < 1e-10

    def test_ghost_on_opposite_side_of_screen(self) -> None:
        """Ghost should always appear on opposite side of screen from source."""
        effect = LensFlareEffect()
        ghost = GhostSettings(offset=0.8)
        # Source in upper-right (x > 0.5, y < 0.5)
        result = effect.calculate_ghost_uv((0.9, 0.1), ghost)
        # Ghost should be in lower-left quadrant (x < 0.5, y > 0.5)
        assert result[0] < 0.5
        assert result[1] > 0.5


class TestChromaticShift:
    """Tests for chromatic aberration shift directions."""

    def test_chromatic_shift_red_outward(self) -> None:
        """Red channel should shift outward from center."""
        effect = LensFlareEffect()
        ghost = GhostSettings(chromatic_shift=0.1)
        ghost_uv = (0.7, 0.5)  # Right of center
        red_uv, green_uv, blue_uv = effect.calculate_ghost_chromatic_uv(ghost_uv, ghost)
        # Red shifts outward (further from center)
        assert red_uv[0] > ghost_uv[0]

    def test_chromatic_shift_blue_inward(self) -> None:
        """Blue channel should shift inward toward center."""
        effect = LensFlareEffect()
        ghost = GhostSettings(chromatic_shift=0.1)
        ghost_uv = (0.7, 0.5)  # Right of center
        red_uv, green_uv, blue_uv = effect.calculate_ghost_chromatic_uv(ghost_uv, ghost)
        # Blue shifts inward (closer to center)
        assert blue_uv[0] < ghost_uv[0]

    def test_chromatic_shift_green_unchanged(self) -> None:
        """Green channel should remain at original position."""
        effect = LensFlareEffect()
        ghost = GhostSettings(chromatic_shift=0.1)
        ghost_uv = (0.7, 0.5)
        red_uv, green_uv, blue_uv = effect.calculate_ghost_chromatic_uv(ghost_uv, ghost)
        assert green_uv == ghost_uv

    def test_chromatic_shift_zero_all_same(self) -> None:
        """Zero chromatic shift should keep all channels at same position."""
        effect = LensFlareEffect()
        ghost = GhostSettings(chromatic_shift=0.0)
        ghost_uv = (0.7, 0.5)
        red_uv, green_uv, blue_uv = effect.calculate_ghost_chromatic_uv(ghost_uv, ghost)
        assert red_uv == ghost_uv
        assert green_uv == ghost_uv
        assert blue_uv == ghost_uv

    def test_chromatic_shift_scales_with_dispersion(self) -> None:
        """Chromatic shift should scale with ghost_dispersion setting."""
        settings1 = LensFlareSettings(ghost_dispersion=0.5)
        settings2 = LensFlareSettings(ghost_dispersion=1.0)
        effect1 = LensFlareEffect(settings1)
        effect2 = LensFlareEffect(settings2)

        ghost = GhostSettings(chromatic_shift=0.1)
        ghost_uv = (0.7, 0.5)

        red1, _, _ = effect1.calculate_ghost_chromatic_uv(ghost_uv, ghost)
        red2, _, _ = effect2.calculate_ghost_chromatic_uv(ghost_uv, ghost)

        # Higher dispersion = larger shift
        shift1 = abs(red1[0] - ghost_uv[0])
        shift2 = abs(red2[0] - ghost_uv[0])
        assert shift2 > shift1


class TestHaloFalloff:
    """Tests for halo radial falloff calculation."""

    def test_halo_at_center_is_maximum(self) -> None:
        """Halo intensity at center should be maximum."""
        settings = LensFlareSettings(
            halo=HaloSettings(enabled=True, radius=0.3, intensity=1.0, falloff=1.0)
        )
        effect = LensFlareEffect(settings)
        intensity = effect.calculate_halo((0.5, 0.5))
        assert abs(intensity - 1.0) < 1e-10

    def test_halo_at_radius_is_zero(self) -> None:
        """Halo intensity at radius boundary should be zero."""
        settings = LensFlareSettings(
            halo=HaloSettings(enabled=True, radius=0.3, intensity=1.0, falloff=1.0)
        )
        effect = LensFlareEffect(settings)
        # At distance = radius from center
        uv = (0.5 + 0.3, 0.5)
        intensity = effect.calculate_halo(uv)
        assert abs(intensity) < 1e-10

    def test_halo_outside_radius_is_zero(self) -> None:
        """Halo intensity outside radius should be zero."""
        settings = LensFlareSettings(
            halo=HaloSettings(enabled=True, radius=0.3, intensity=1.0)
        )
        effect = LensFlareEffect(settings)
        # Far outside radius
        intensity = effect.calculate_halo((0.0, 0.0))
        assert intensity == 0.0

    def test_halo_falloff_affects_gradient(self) -> None:
        """Higher falloff exponent should create steeper gradient."""
        halo_soft = HaloSettings(enabled=True, radius=0.3, intensity=1.0, falloff=1.0)
        halo_hard = HaloSettings(enabled=True, radius=0.3, intensity=1.0, falloff=3.0)

        effect_soft = LensFlareEffect(LensFlareSettings(halo=halo_soft))
        effect_hard = LensFlareEffect(LensFlareSettings(halo=halo_hard))

        # Check intensity at halfway point
        uv = (0.5 + 0.15, 0.5)  # Halfway to edge
        soft_intensity = effect_soft.calculate_halo(uv)
        hard_intensity = effect_hard.calculate_halo(uv)

        # Both should be positive but hard falloff should be lower
        assert soft_intensity > 0
        assert hard_intensity > 0
        assert hard_intensity < soft_intensity

    def test_halo_disabled_returns_zero(self) -> None:
        """Disabled halo should always return zero."""
        settings = LensFlareSettings(halo=HaloSettings(enabled=False))
        effect = LensFlareEffect(settings)
        intensity = effect.calculate_halo((0.5, 0.5))
        assert intensity == 0.0


class TestStreakDirection:
    """Tests for anamorphic streak direction and falloff."""

    def test_horizontal_streak_direction(self) -> None:
        """Horizontal streak (direction=0) should extend along X axis."""
        settings = LensFlareSettings(
            streaks=StreakSettings(
                enabled=True, direction=0.0, length=0.5, spacing=0.2, falloff=1.0
            )
        )
        effect = LensFlareEffect(settings)
        source = (0.5, 0.5)

        # Point along X axis should have positive intensity
        intensity_x = effect.calculate_streak((0.6, 0.5), source)
        assert intensity_x > 0

        # Point along Y axis (outside spacing) should be zero
        intensity_y = effect.calculate_streak((0.5, 0.7), source)
        assert intensity_y == 0.0

    def test_vertical_streak_direction(self) -> None:
        """Vertical streak (direction=pi/2) should extend along Y axis."""
        settings = LensFlareSettings(
            streaks=StreakSettings(
                enabled=True,
                direction=math.pi / 2,
                length=0.5,
                spacing=0.2,
                falloff=1.0,
            )
        )
        effect = LensFlareEffect(settings)
        source = (0.5, 0.5)

        # Point along Y axis should have positive intensity
        intensity_y = effect.calculate_streak((0.5, 0.6), source)
        assert intensity_y > 0

        # Point along X axis (outside spacing) should be zero
        intensity_x = effect.calculate_streak((0.7, 0.5), source)
        assert intensity_x == 0.0

    def test_streak_falloff_with_distance(self) -> None:
        """Streak intensity should decrease with distance from source."""
        settings = LensFlareSettings(
            streaks=StreakSettings(
                enabled=True, direction=0.0, length=0.5, spacing=0.2, falloff=2.0
            )
        )
        effect = LensFlareEffect(settings)
        source = (0.5, 0.5)

        near = effect.calculate_streak((0.55, 0.5), source)
        far = effect.calculate_streak((0.7, 0.5), source)

        assert near > far

    def test_streak_beyond_length_is_zero(self) -> None:
        """Streak beyond configured length should be zero."""
        settings = LensFlareSettings(
            streaks=StreakSettings(enabled=True, direction=0.0, length=0.2, spacing=0.1)
        )
        effect = LensFlareEffect(settings)
        source = (0.5, 0.5)

        # Beyond length
        intensity = effect.calculate_streak((0.8, 0.5), source)
        assert intensity == 0.0

    def test_streak_disabled_returns_zero(self) -> None:
        """Disabled streaks should always return zero."""
        settings = LensFlareSettings(streaks=StreakSettings(enabled=False))
        effect = LensFlareEffect(settings)
        intensity = effect.calculate_streak((0.6, 0.5), (0.5, 0.5))
        assert intensity == 0.0


class TestQualityLevelConfigurations:
    """Tests for quality level ghost count and configurations."""

    def test_off_quality_has_no_ghosts(self) -> None:
        """OFF quality should have zero ghosts."""
        settings = LensFlareSettings(quality=LensFlareQuality.OFF)
        effect = LensFlareEffect(settings)
        assert len(effect.ghosts) == 0

    def test_low_quality_has_three_ghosts(self) -> None:
        """LOW quality should have 3 ghosts."""
        settings = LensFlareSettings(quality=LensFlareQuality.LOW)
        effect = LensFlareEffect(settings)
        assert len(effect.ghosts) == 3

    def test_medium_quality_has_six_ghosts(self) -> None:
        """MEDIUM quality should have 6 ghosts."""
        settings = LensFlareSettings(quality=LensFlareQuality.MEDIUM)
        effect = LensFlareEffect(settings)
        assert len(effect.ghosts) == 6

    def test_high_quality_has_eight_ghosts(self) -> None:
        """HIGH quality should have 8 ghosts."""
        settings = LensFlareSettings(quality=LensFlareQuality.HIGH)
        effect = LensFlareEffect(settings)
        assert len(effect.ghosts) == 8

    def test_ghosts_have_increasing_chromatic_shift(self) -> None:
        """Ghost chromatic shift should increase with offset (index)."""
        settings = LensFlareSettings(quality=LensFlareQuality.HIGH)
        effect = LensFlareEffect(settings)
        ghosts = effect.ghosts

        # Generally, later ghosts (further from source) have more shift
        # Check first and last
        assert ghosts[-1].chromatic_shift >= ghosts[0].chromatic_shift


class TestBudgetEstimates:
    """Tests for GPU time budget estimates."""

    def test_off_quality_budget_is_zero(self) -> None:
        """OFF quality should have zero budget."""
        settings = LensFlareSettings(quality=LensFlareQuality.OFF)
        effect = LensFlareEffect(settings)
        assert effect.get_budget_ms() == 0.0

    def test_low_quality_budget(self) -> None:
        """LOW quality budget should be ~0.03ms."""
        settings = LensFlareSettings(quality=LensFlareQuality.LOW)
        effect = LensFlareEffect(settings)
        assert effect.get_budget_ms() == pytest.approx(0.03)

    def test_medium_quality_budget(self) -> None:
        """MEDIUM quality budget should be ~0.05ms."""
        settings = LensFlareSettings(quality=LensFlareQuality.MEDIUM)
        effect = LensFlareEffect(settings)
        assert effect.get_budget_ms() == pytest.approx(0.05)

    def test_high_quality_budget(self) -> None:
        """HIGH quality budget should be ~0.08ms."""
        settings = LensFlareSettings(quality=LensFlareQuality.HIGH)
        effect = LensFlareEffect(settings)
        assert effect.get_budget_ms() == pytest.approx(0.08)

    def test_budget_increases_with_quality(self) -> None:
        """Budget should increase with quality level."""
        budgets = []
        for quality in [
            LensFlareQuality.OFF,
            LensFlareQuality.LOW,
            LensFlareQuality.MEDIUM,
            LensFlareQuality.HIGH,
        ]:
            settings = LensFlareSettings(quality=quality)
            effect = LensFlareEffect(settings)
            budgets.append(effect.get_budget_ms())

        for i in range(1, len(budgets)):
            assert budgets[i] >= budgets[i - 1]


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_threshold_must_be_positive(self) -> None:
        """Threshold must be in valid range."""
        with pytest.raises(ValueError, match="threshold"):
            LensFlareSettings(threshold=-1.0)

    def test_threshold_must_be_within_range(self) -> None:
        """Threshold above 10 should fail."""
        with pytest.raises(ValueError, match="threshold"):
            LensFlareSettings(threshold=15.0)

    def test_intensity_must_be_positive(self) -> None:
        """Intensity must be non-negative."""
        with pytest.raises(ValueError, match="intensity"):
            LensFlareSettings(intensity=-0.5)

    def test_ghost_count_must_be_within_range(self) -> None:
        """Ghost count must be 0-16."""
        with pytest.raises(ValueError, match="ghost_count"):
            LensFlareSettings(ghost_count=20)

    def test_ghost_dispersion_must_be_within_range(self) -> None:
        """Ghost dispersion must be 0-2."""
        with pytest.raises(ValueError, match="ghost_dispersion"):
            LensFlareSettings(ghost_dispersion=3.0)


class TestSettingsInterpolation:
    """Tests for settings lerp interpolation."""

    def test_lerp_intensity(self) -> None:
        """Intensity should interpolate linearly."""
        settings1 = LensFlareSettings(intensity=0.0)
        settings2 = LensFlareSettings(intensity=1.0)
        lerped = settings1.lerp(settings2, 0.5)
        assert lerped.intensity == pytest.approx(0.5)

    def test_lerp_threshold(self) -> None:
        """Threshold should interpolate linearly."""
        settings1 = LensFlareSettings(threshold=0.5)
        settings2 = LensFlareSettings(threshold=1.5)
        lerped = settings1.lerp(settings2, 0.5)
        assert lerped.threshold == pytest.approx(1.0)

    def test_lerp_quality_snaps_at_half(self) -> None:
        """Quality should snap to target at t >= 0.5."""
        settings1 = LensFlareSettings(quality=LensFlareQuality.LOW)
        settings2 = LensFlareSettings(quality=LensFlareQuality.HIGH)

        lerped_before = settings1.lerp(settings2, 0.4)
        assert lerped_before.quality == LensFlareQuality.LOW

        lerped_after = settings1.lerp(settings2, 0.6)
        assert lerped_after.quality == LensFlareQuality.HIGH


class TestFactoryFunctions:
    """Tests for factory function presets."""

    def test_cinematic_preset_is_high_quality(self) -> None:
        """Cinematic preset should use HIGH quality."""
        effect = create_cinematic_lens_flare()
        assert effect.settings.quality == LensFlareQuality.HIGH

    def test_cinematic_preset_has_streaks(self) -> None:
        """Cinematic preset should have streaks enabled."""
        effect = create_cinematic_lens_flare()
        assert effect.settings.streaks.enabled

    def test_subtle_preset_is_medium_quality(self) -> None:
        """Subtle preset should use MEDIUM quality."""
        effect = create_subtle_lens_flare()
        assert effect.settings.quality == LensFlareQuality.MEDIUM

    def test_subtle_preset_has_no_streaks(self) -> None:
        """Subtle preset should have streaks disabled."""
        effect = create_subtle_lens_flare()
        assert not effect.settings.streaks.enabled

    def test_disabled_preset_is_off(self) -> None:
        """Disabled preset should have OFF quality and disabled flag."""
        effect = create_disabled_lens_flare()
        assert effect.settings.quality == LensFlareQuality.OFF
        assert not effect.settings.enabled


class TestEffectInterface:
    """Tests for PostProcessEffect interface compliance."""

    def test_effect_name_is_lens_flare(self) -> None:
        """Effect name should be 'LensFlare'."""
        effect = LensFlareEffect()
        assert effect.name == "LensFlare"

    def test_effect_requires_bloom_buffer_input(self) -> None:
        """Effect should require bloom_buffer input."""
        effect = LensFlareEffect()
        inputs = effect.get_required_inputs()
        assert "bloom_buffer" in inputs

    def test_effect_outputs_color(self) -> None:
        """Effect should output color."""
        effect = LensFlareEffect()
        outputs = effect.get_outputs()
        assert "color" in outputs

    def test_effect_is_compute_effect(self) -> None:
        """Lens flare should be a compute effect."""
        effect = LensFlareEffect()
        assert effect.is_compute_effect()

    def test_effect_setup_stores_dimensions(self) -> None:
        """Setup should store width and height."""
        effect = LensFlareEffect()
        effect.setup(1920, 1080)
        assert effect._width == 1920
        assert effect._height == 1080

    def test_effect_cleanup_clears_buffer(self) -> None:
        """Cleanup should clear flare buffer."""
        effect = LensFlareEffect()
        effect._flare_buffer = [1, 2, 3]
        effect.cleanup()
        assert effect._flare_buffer is None
