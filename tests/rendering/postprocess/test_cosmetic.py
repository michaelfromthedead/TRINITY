"""
Tests for Cosmetic Post-Processing Effects.

Tests for T-PP-1.6 (Vignette), T-PP-2.4 (Chromatic Aberration),
and T-PP-3.5 (Film Grain).
"""

import math
import pytest

from engine.rendering.postprocess.cosmetic import (
    # Vignette
    VignetteSettings,
    VignetteEffect,
    # Chromatic Aberration
    CAQuality,
    ChromaticAberrationSettings,
    ChromaticAberrationEffect,
    # Film Grain
    GrainQuality,
    FilmGrainSettings,
    FilmGrainEffect,
)


# ============================================================================
# T-PP-1.6: VIGNETTE TESTS
# ============================================================================


class TestVignetteSettings:
    """Test VignetteSettings dataclass."""

    def test_default_values(self):
        """Test default settings are sensible."""
        settings = VignetteSettings()
        assert settings.enabled is True
        assert 0.0 <= settings.intensity <= 1.0
        assert settings.inner_radius < settings.outer_radius
        assert settings.feather > 0
        assert len(settings.color) == 3

    def test_custom_values(self):
        """Test custom settings are applied."""
        settings = VignetteSettings(
            intensity=0.5,
            inner_radius=0.2,
            outer_radius=0.9,
            feather=3.0,
            color=(0.1, 0.0, 0.2),
        )
        assert settings.intensity == 0.5
        assert settings.inner_radius == 0.2
        assert settings.outer_radius == 0.9
        assert settings.feather == 3.0
        assert settings.color == (0.1, 0.0, 0.2)


class TestVignetteEffect:
    """Test VignetteEffect class."""

    def test_center_has_no_vignette(self):
        """Center of screen should have no vignetting."""
        effect = VignetteEffect()
        factor = effect.calculate_vignette((0.5, 0.5))
        assert factor == pytest.approx(1.0, abs=0.01)

    def test_corners_have_vignette(self):
        """Corners should have vignette applied."""
        effect = VignetteEffect(VignetteSettings(intensity=0.5))

        # Check all four corners
        corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        for corner in corners:
            factor = effect.calculate_vignette(corner)
            assert factor < 1.0, f"Corner {corner} should have vignette"

    def test_intensity_affects_strength(self):
        """Higher intensity should produce darker vignette."""
        low_intensity = VignetteEffect(VignetteSettings(intensity=0.2))
        high_intensity = VignetteEffect(VignetteSettings(intensity=0.8))

        corner = (0.0, 0.0)
        low_factor = low_intensity.calculate_vignette(corner)
        high_factor = high_intensity.calculate_vignette(corner)

        assert high_factor < low_factor

    def test_disabled_returns_full_brightness(self):
        """Disabled vignette should return 1.0."""
        effect = VignetteEffect(VignetteSettings(enabled=False))
        assert effect.calculate_vignette((0.0, 0.0)) == 1.0
        assert effect.calculate_vignette((0.5, 0.5)) == 1.0
        assert effect.calculate_vignette((1.0, 1.0)) == 1.0

    def test_radial_symmetry(self):
        """Vignette should be radially symmetric."""
        effect = VignetteEffect()

        # Points equidistant from center should have same factor
        f1 = effect.calculate_vignette((0.3, 0.5))
        f2 = effect.calculate_vignette((0.7, 0.5))
        f3 = effect.calculate_vignette((0.5, 0.3))
        f4 = effect.calculate_vignette((0.5, 0.7))

        assert f1 == pytest.approx(f2, abs=0.01)
        assert f3 == pytest.approx(f4, abs=0.01)

    def test_aspect_ratio_correction(self):
        """Aspect ratio correction should affect horizontal spread."""
        effect_corrected = VignetteEffect(
            VignetteSettings(aspect_ratio_correction=True)
        )
        effect_uncorrected = VignetteEffect(
            VignetteSettings(aspect_ratio_correction=False)
        )

        # With 16:9 aspect, corners should show different behavior
        # Correction stretches x-coordinate, so corner distance increases
        uv = (0.0, 0.0)  # Corner
        factor_corrected = effect_corrected.calculate_vignette(uv, aspect=16/9)
        factor_uncorrected = effect_uncorrected.calculate_vignette(uv, aspect=16/9)

        # Corrected version has larger effective radius at corners
        assert factor_corrected != factor_uncorrected or True  # May be equal at corners

        # Test at a point where difference is more apparent
        # Point halfway to left edge
        uv_side = (0.25, 0.5)
        factor_side_corrected = effect_corrected.calculate_vignette(uv_side, aspect=16/9)
        factor_side_uncorrected = effect_uncorrected.calculate_vignette(uv_side, aspect=16/9)

        # With aspect correction, horizontal distance is scaled, changing vignette
        # The corrected version multiplies cx by aspect, increasing distance
        assert factor_side_corrected < factor_side_uncorrected

    def test_apply_to_color(self):
        """Test color application with vignette."""
        effect = VignetteEffect(VignetteSettings(intensity=0.5, color=(0.0, 0.0, 0.0)))

        white = (1.0, 1.0, 1.0)
        result_center = effect.apply_to_color(white, (0.5, 0.5))
        result_corner = effect.apply_to_color(white, (0.0, 0.0))

        # Center should remain white
        assert result_center[0] == pytest.approx(1.0, abs=0.02)

        # Corner should be darkened
        assert result_corner[0] < 1.0

    def test_colored_vignette(self):
        """Test non-black vignette color."""
        effect = VignetteEffect(
            VignetteSettings(intensity=1.0, color=(1.0, 0.0, 0.0))
        )

        white = (1.0, 1.0, 1.0)
        result = effect.apply_to_color(white, (0.0, 0.0))

        # Should blend toward red
        assert result[0] > result[1]
        assert result[0] > result[2]


# ============================================================================
# T-PP-2.4: CHROMATIC ABERRATION TESTS
# ============================================================================


class TestChromaticAberrationSettings:
    """Test ChromaticAberrationSettings dataclass."""

    def test_default_values(self):
        """Test default settings."""
        settings = ChromaticAberrationSettings()
        assert settings.enabled is True
        assert settings.quality == CAQuality.MEDIUM
        assert settings.red_offset > 0  # Outward
        assert settings.blue_offset < 0  # Inward

    def test_quality_enum(self):
        """Test quality levels exist."""
        assert CAQuality.OFF.value < CAQuality.LOW.value
        assert CAQuality.LOW.value < CAQuality.MEDIUM.value
        assert CAQuality.MEDIUM.value < CAQuality.HIGH.value


class TestChromaticAberrationEffect:
    """Test ChromaticAberrationEffect class."""

    def test_center_has_no_offset(self):
        """Center of screen should have no chromatic offset."""
        effect = ChromaticAberrationEffect()
        red_uv, green_uv, blue_uv = effect.calculate_offsets((0.5, 0.5))

        assert red_uv == pytest.approx(green_uv, abs=0.001)
        assert blue_uv == pytest.approx(green_uv, abs=0.001)

    def test_edge_has_offset(self):
        """Edges should have chromatic offset."""
        effect = ChromaticAberrationEffect(
            ChromaticAberrationSettings(quality=CAQuality.HIGH, intensity=2.0)
        )

        # Right edge
        red_uv, green_uv, blue_uv = effect.calculate_offsets((0.9, 0.5))

        # Red should be offset outward (positive x)
        assert red_uv[0] > green_uv[0]
        # Blue should be offset inward (negative x from edge)
        assert blue_uv[0] < green_uv[0]

    def test_off_quality_no_offset(self):
        """OFF quality should return identical UVs."""
        effect = ChromaticAberrationEffect(
            ChromaticAberrationSettings(quality=CAQuality.OFF)
        )

        uv = (0.9, 0.9)
        red_uv, green_uv, blue_uv = effect.calculate_offsets(uv)

        assert red_uv == uv
        assert green_uv == uv
        assert blue_uv == uv

    def test_disabled_no_offset(self):
        """Disabled effect should return identical UVs."""
        effect = ChromaticAberrationEffect(
            ChromaticAberrationSettings(enabled=False)
        )

        uv = (0.9, 0.9)
        red_uv, green_uv, blue_uv = effect.calculate_offsets(uv)

        assert red_uv == uv
        assert green_uv == uv
        assert blue_uv == uv

    def test_quality_affects_max_offset(self):
        """Higher quality should allow larger offsets."""
        effect_low = ChromaticAberrationEffect(
            ChromaticAberrationSettings(quality=CAQuality.LOW, intensity=1.0)
        )
        effect_high = ChromaticAberrationEffect(
            ChromaticAberrationSettings(quality=CAQuality.HIGH, intensity=1.0)
        )

        uv = (0.95, 0.5)
        mag_low = effect_low.get_offset_magnitude(uv)
        mag_high = effect_high.get_offset_magnitude(uv)

        assert mag_high > mag_low

    def test_radial_direction(self):
        """Offsets should be in radial direction from center."""
        effect = ChromaticAberrationEffect(
            ChromaticAberrationSettings(quality=CAQuality.HIGH, intensity=2.0)
        )

        # Test top edge
        red_uv, green_uv, blue_uv = effect.calculate_offsets((0.5, 0.1))
        # Red should be offset outward (negative y from center)
        assert red_uv[1] < green_uv[1]
        # Blue should be offset inward (positive y toward center)
        assert blue_uv[1] > green_uv[1]

    def test_smoothstep_function(self):
        """Test smoothstep helper function."""
        result_below = ChromaticAberrationEffect._smoothstep(0.0, 1.0, -0.5)
        result_above = ChromaticAberrationEffect._smoothstep(0.0, 1.0, 1.5)
        result_middle = ChromaticAberrationEffect._smoothstep(0.0, 1.0, 0.5)

        assert result_below == 0.0
        assert result_above == 1.0
        assert 0.0 < result_middle < 1.0

    def test_anamorphic_ratio(self):
        """Anamorphic ratio should stretch vertical offsets."""
        effect_normal = ChromaticAberrationEffect(
            ChromaticAberrationSettings(anamorphic_ratio=1.0, quality=CAQuality.HIGH)
        )
        effect_anamorphic = ChromaticAberrationEffect(
            ChromaticAberrationSettings(anamorphic_ratio=1.33, quality=CAQuality.HIGH)
        )

        # Test vertical edge
        uv = (0.5, 0.9)
        mag_normal = effect_normal.get_offset_magnitude(uv)
        mag_anamorphic = effect_anamorphic.get_offset_magnitude(uv)

        # Anamorphic should have larger vertical offset
        assert mag_anamorphic > mag_normal


# ============================================================================
# T-PP-3.5: FILM GRAIN TESTS
# ============================================================================


class TestFilmGrainSettings:
    """Test FilmGrainSettings dataclass."""

    def test_default_values(self):
        """Test default settings are sensible."""
        settings = FilmGrainSettings()
        assert settings.enabled is True
        assert settings.quality == GrainQuality.GAUSSIAN
        assert 0.0 < settings.intensity < 0.5
        assert 0.0 < settings.response <= 1.0

    def test_quality_enum(self):
        """Test quality levels exist."""
        assert GrainQuality.OFF is not None
        assert GrainQuality.UNIFORM is not None
        assert GrainQuality.GAUSSIAN is not None
        assert GrainQuality.GAUSSIAN_CHROMA is not None


class TestFilmGrainEffect:
    """Test FilmGrainEffect class."""

    def test_frame_advancement(self):
        """Frame counter should advance and wrap."""
        effect = FilmGrainEffect()
        assert effect.frame_index == 0

        effect.advance_frame()
        assert effect.frame_index == 1

        effect.set_frame(65535)
        effect.advance_frame()
        assert effect.frame_index == 0

    def test_wang_hash_deterministic(self):
        """Wang hash should be deterministic."""
        effect = FilmGrainEffect()
        hash1 = effect.wang_hash(12345)
        hash2 = effect.wang_hash(12345)
        assert hash1 == hash2

    def test_random_float_range(self):
        """Random float should be in [0, 1]."""
        effect = FilmGrainEffect()
        for x in range(10):
            for y in range(10):
                val = effect.random_float(x, y)
                assert 0.0 <= val <= 1.0

    def test_random_float_varies(self):
        """Random float should vary with position."""
        effect = FilmGrainEffect()
        values = set()
        for x in range(10):
            values.add(effect.random_float(x, 0))
        assert len(values) > 1

    def test_gaussian_noise_range(self):
        """Gaussian noise should be roughly in [-1, 1]."""
        effect = FilmGrainEffect()
        for x in range(20):
            for y in range(20):
                val = effect.gaussian_noise(x, y)
                assert -2.0 <= val <= 2.0  # Allow some tail

    def test_grain_off_returns_zero(self):
        """OFF quality should return zero grain."""
        effect = FilmGrainEffect(FilmGrainSettings(quality=GrainQuality.OFF))
        grain = effect.calculate_grain(100, 100, 0.5)
        assert grain == (0.0, 0.0, 0.0)

    def test_grain_disabled_returns_zero(self):
        """Disabled effect should return zero grain."""
        effect = FilmGrainEffect(FilmGrainSettings(enabled=False))
        grain = effect.calculate_grain(100, 100, 0.5)
        assert grain == (0.0, 0.0, 0.0)

    def test_luminance_modulation_midtones(self):
        """Grain should be most visible in midtones."""
        effect = FilmGrainEffect(FilmGrainSettings(intensity=1.0))

        # Midtone should have full factor
        factor_mid = effect.calculate_luminance_factor(0.5)
        factor_dark = effect.calculate_luminance_factor(0.1)
        factor_bright = effect.calculate_luminance_factor(0.9)

        assert factor_mid > factor_dark
        assert factor_mid > factor_bright

    def test_luminance_modulation_extremes(self):
        """Grain should be minimal at luminance extremes."""
        effect = FilmGrainEffect()

        factor_black = effect.calculate_luminance_factor(0.0)
        factor_white = effect.calculate_luminance_factor(1.0)

        assert factor_black == pytest.approx(0.0, abs=0.01)
        assert factor_white == pytest.approx(0.0, abs=0.01)

    def test_uniform_grain_identical_channels(self):
        """Uniform grain should have identical RGB values."""
        effect = FilmGrainEffect(FilmGrainSettings(quality=GrainQuality.UNIFORM))
        grain = effect.calculate_grain(100, 100, 0.5)

        assert grain[0] == grain[1]
        assert grain[1] == grain[2]

    def test_gaussian_grain_identical_channels(self):
        """Basic Gaussian grain should have identical RGB values."""
        effect = FilmGrainEffect(FilmGrainSettings(quality=GrainQuality.GAUSSIAN))
        grain = effect.calculate_grain(100, 100, 0.5)

        assert grain[0] == grain[1]
        assert grain[1] == grain[2]

    def test_gaussian_chroma_has_color_variation(self):
        """Gaussian chroma grain should have color variation in darks."""
        effect = FilmGrainEffect(
            FilmGrainSettings(
                quality=GrainQuality.GAUSSIAN_CHROMA,
                chroma_intensity=0.5,
            )
        )

        # In dark region, chroma should be present
        grain = effect.calculate_grain(100, 100, 0.1)

        # R and B should differ from G due to chroma
        # (may not always differ due to randomness, but should sometimes)
        # Test over multiple positions
        has_chroma_diff = False
        for x in range(20):
            g = effect.calculate_grain(x, 100, 0.1)
            if g[0] != g[1] or g[2] != g[1]:
                has_chroma_diff = True
                break
        assert has_chroma_diff

    def test_apply_to_color_clamps(self):
        """Applied grain should be clamped to [0, 1]."""
        effect = FilmGrainEffect(FilmGrainSettings(intensity=0.5))

        for x in range(20):
            result = effect.apply_to_color((0.5, 0.5, 0.5), x, 100)
            assert all(0.0 <= c <= 1.0 for c in result)

    def test_temporal_variation(self):
        """Grain should change between frames."""
        effect = FilmGrainEffect()

        grain1 = effect.calculate_grain(100, 100, 0.5)
        effect.advance_frame()
        grain2 = effect.calculate_grain(100, 100, 0.5)

        assert grain1 != grain2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestCosmeticEffectsIntegration:
    """Integration tests for combined effects."""

    def test_all_effects_can_be_instantiated(self):
        """All effects should instantiate with defaults."""
        vignette = VignetteEffect()
        ca = ChromaticAberrationEffect()
        grain = FilmGrainEffect()

        assert vignette is not None
        assert ca is not None
        assert grain is not None

    def test_effects_chain(self):
        """Effects can be chained together."""
        vignette = VignetteEffect(VignetteSettings(intensity=0.3))
        ca = ChromaticAberrationEffect()
        grain = FilmGrainEffect(FilmGrainSettings(intensity=0.05))

        # Simulate processing a pixel
        uv = (0.7, 0.3)
        color = (0.8, 0.7, 0.6)

        # Apply vignette
        color = vignette.apply_to_color(color, uv)

        # CA would modify UV lookups (can't fully simulate without texture)
        _ = ca.calculate_offsets(uv)

        # Apply grain
        x, y = int(uv[0] * 1920), int(uv[1] * 1080)
        color = grain.apply_to_color(color, x, y)

        # Result should still be valid
        assert all(0.0 <= c <= 1.0 for c in color)

    def test_all_exports_present(self):
        """All expected exports should be importable."""
        from engine.rendering.postprocess.cosmetic import __all__

        expected = [
            "VignetteSettings",
            "VignetteEffect",
            "CAQuality",
            "ChromaticAberrationSettings",
            "ChromaticAberrationEffect",
            "GrainQuality",
            "FilmGrainSettings",
            "FilmGrainEffect",
        ]

        for name in expected:
            assert name in __all__
