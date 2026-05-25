"""
Tests for Anti-Aliasing System

Tests FXAA, SMAA, and TAA with jitter, reprojection, and history.
"""

import math
import pytest

from engine.rendering.postprocess.antialiasing import (
    AAEffect,
    AAMethod,
    AASettings,
    FXAA,
    FXAAQuality,
    FXAASettings,
    JitterPattern,
    JitterSequence,
    SMAA,
    SMAAQuality,
    SMAASettings,
    TAA,
    TAASettings,
)


class TestJitterSequence:
    """Test JitterSequence for TAA sampling."""

    def test_halton_sequence_creation(self):
        """Test Halton sequence creation."""
        jitter = JitterSequence(JitterPattern.HALTON_16)

        assert jitter.sample_count == 16
        assert jitter.current_index == 0

    def test_halton_sample_range(self):
        """Test Halton samples are in valid range."""
        jitter = JitterSequence(JitterPattern.HALTON_16)

        for _ in range(16):
            x, y = jitter.next()
            assert -0.5 <= x <= 0.5
            assert -0.5 <= y <= 0.5

    def test_sequence_wraps(self):
        """Test sequence wraps after all samples."""
        jitter = JitterSequence(JitterPattern.HALTON_8)

        # Get all samples
        samples = [jitter.next() for _ in range(8)]

        # Next call should wrap to beginning
        assert jitter.current_index == 0
        next_sample = jitter.next()
        assert next_sample == samples[0]

    def test_sequence_reset(self):
        """Test sequence reset."""
        jitter = JitterSequence(JitterPattern.HALTON_16)

        # Advance a few samples
        for _ in range(5):
            jitter.next()

        assert jitter.current_index == 5

        jitter.reset()
        assert jitter.current_index == 0

    def test_uniform_pattern(self):
        """Test uniform 4-sample pattern."""
        jitter = JitterSequence(JitterPattern.UNIFORM_4)

        assert jitter.sample_count == 4

        samples = [jitter.next() for _ in range(4)]

        # Should have 4 unique samples
        assert len(set(samples)) == 4

    def test_rgss_pattern(self):
        """Test rotated grid pattern."""
        jitter = JitterSequence(JitterPattern.RGSS_4)

        assert jitter.sample_count == 4

    def test_projection_jitter(self):
        """Test getting jitter scaled for projection."""
        jitter = JitterSequence(JitterPattern.HALTON_16)

        jx, jy = jitter.get_projection_jitter(1920, 1080)

        # Should be scaled by resolution
        # Clip space is [-1, 1], so jitter should be small
        assert abs(jx) < 0.01
        assert abs(jy) < 0.01

    def test_different_resolutions(self):
        """Test jitter scales with resolution."""
        jitter = JitterSequence(JitterPattern.HALTON_16)

        jx_high, jy_high = jitter.get_projection_jitter(3840, 2160)

        jitter.reset()
        jx_low, jy_low = jitter.get_projection_jitter(1920, 1080)

        # Higher resolution should have smaller or equal clip-space jitter
        # (jitter offset divided by width/height gives smaller clip-space value)
        assert abs(jx_high) <= abs(jx_low) or abs(jx_high) < 0.01


class TestFXAASettings:
    """Test FXAASettings dataclass."""

    def test_default_settings(self):
        """Test default FXAA settings."""
        settings = FXAASettings()

        assert settings.quality == FXAAQuality.MEDIUM
        assert settings.edge_threshold == 0.166
        assert settings.subpixel_quality == 0.75

    def test_custom_settings(self):
        """Test custom FXAA settings."""
        settings = FXAASettings(
            quality=FXAAQuality.HIGH,
            edge_threshold=0.1,
        )

        assert settings.quality == FXAAQuality.HIGH
        assert settings.edge_threshold == 0.1


class TestSMAASettings:
    """Test SMAASettings dataclass."""

    def test_default_settings(self):
        """Test default SMAA settings."""
        settings = SMAASettings()

        assert settings.quality == SMAAQuality.HIGH
        assert settings.threshold == 0.1
        assert settings.max_search_steps == 16

    def test_custom_settings(self):
        """Test custom SMAA settings."""
        settings = SMAASettings(
            quality=SMAAQuality.ULTRA,
            max_search_steps=32,
        )

        assert settings.quality == SMAAQuality.ULTRA
        assert settings.max_search_steps == 32


class TestTAASettings:
    """Test TAASettings dataclass."""

    def test_default_settings(self):
        """Test default TAA settings."""
        settings = TAASettings()

        assert settings.jitter_pattern == JitterPattern.HALTON_16
        assert settings.history_weight == 0.9
        assert settings.color_box_clamping is True
        assert settings.variance_clipping is True
        assert settings.sharpen_enabled is True

    def test_custom_settings(self):
        """Test custom TAA settings."""
        settings = TAASettings(
            jitter_pattern=JitterPattern.HALTON_32,
            history_weight=0.95,
            sharpen_amount=0.5,
        )

        assert settings.jitter_pattern == JitterPattern.HALTON_32
        assert settings.history_weight == 0.95
        assert settings.sharpen_amount == 0.5


class TestAASettings:
    """Test AASettings dataclass."""

    def test_default_settings(self):
        """Test default AA settings."""
        settings = AASettings()

        assert settings.method == AAMethod.TAA
        assert settings.fxaa is not None
        assert settings.smaa is not None
        assert settings.taa is not None

    def test_custom_settings(self):
        """Test custom AA settings."""
        settings = AASettings(
            method=AAMethod.FXAA,
            fxaa=FXAASettings(quality=FXAAQuality.HIGH),
        )

        assert settings.method == AAMethod.FXAA
        assert settings.fxaa.quality == FXAAQuality.HIGH

    def test_settings_lerp(self):
        """Test settings interpolation."""
        settings1 = AASettings(
            taa=TAASettings(history_weight=0.8, sharpen_amount=0.1)
        )
        settings2 = AASettings(
            taa=TAASettings(history_weight=0.95, sharpen_amount=0.5)
        )

        lerped = settings1.lerp(settings2, 0.5)

        assert lerped.taa.history_weight == pytest.approx(0.875, 0.01)
        assert lerped.taa.sharpen_amount == pytest.approx(0.3, 0.01)


class TestFXAA:
    """Test FXAA processor."""

    def test_fxaa_creation(self):
        """Test FXAA creation."""
        fxaa = FXAA()
        assert fxaa is not None

    def test_fxaa_setup(self):
        """Test FXAA setup."""
        fxaa = FXAA()
        fxaa.setup(1920, 1080)
        # Should not raise


class TestSMAA:
    """Test SMAA processor."""

    def test_smaa_creation(self):
        """Test SMAA creation."""
        smaa = SMAA()
        assert smaa is not None

    def test_smaa_setup(self):
        """Test SMAA setup."""
        smaa = SMAA()
        smaa.setup(1920, 1080)
        # Should not raise


class TestTAA:
    """Test TAA processor."""

    def test_taa_creation(self):
        """Test TAA creation."""
        taa = TAA()
        assert taa is not None
        assert taa.history_valid is False

    def test_taa_setup(self):
        """Test TAA setup."""
        taa = TAA()
        taa.setup(1920, 1080, JitterPattern.HALTON_16)

        assert taa.jitter_sequence.sample_count == 16

    def test_taa_jitter_offset(self):
        """Test getting jitter offset."""
        taa = TAA()
        taa.setup(1920, 1080)

        x, y = taa.get_jitter_offset()

        assert -0.5 <= x <= 0.5
        assert -0.5 <= y <= 0.5

    def test_taa_jittered_projection(self):
        """Test applying jitter to projection matrix."""
        taa = TAA()
        taa.setup(1920, 1080)

        # Identity-like projection
        projection = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        jittered = taa.get_jittered_projection(projection)

        # Jitter should modify row 2
        assert jittered[2][0] != projection[2][0] or jittered[2][1] != projection[2][1]

    def test_taa_invalidate_history(self):
        """Test invalidating TAA history."""
        taa = TAA()
        taa.setup(1920, 1080)

        # Simulate having valid history
        taa._history_valid = True
        assert taa.history_valid is True

        taa.invalidate_history()
        assert taa.history_valid is False

    def test_taa_resolution_change_invalidates(self):
        """Test resolution change invalidates history."""
        taa = TAA()
        taa.setup(1920, 1080)
        taa._history_valid = True

        # Change resolution
        taa.setup(3840, 2160)

        assert taa.history_valid is False


class TestAAEffect:
    """Test AAEffect integration."""

    def test_effect_creation(self):
        """Test AA effect creation."""
        effect = AAEffect()

        assert effect.name == "AntiAliasing"
        assert effect.settings is not None

    def test_effect_with_custom_settings(self):
        """Test effect with custom settings."""
        settings = AASettings(method=AAMethod.SMAA)
        effect = AAEffect(settings)

        assert effect.settings.method == AAMethod.SMAA

    def test_effect_required_inputs_fxaa(self):
        """Test FXAA required inputs."""
        settings = AASettings(method=AAMethod.FXAA)
        effect = AAEffect(settings)

        inputs = effect.get_required_inputs()
        assert "color" in inputs
        # FXAA doesn't need depth or velocity
        assert "depth" not in inputs

    def test_effect_required_inputs_taa(self):
        """Test TAA required inputs."""
        settings = AASettings(method=AAMethod.TAA)
        effect = AAEffect(settings)

        inputs = effect.get_required_inputs()
        assert "color" in inputs
        assert "depth" in inputs
        assert "velocity" in inputs

    def test_effect_outputs(self):
        """Test effect outputs."""
        effect = AAEffect()
        outputs = effect.get_outputs()

        assert "color" in outputs

    def test_effect_jitter_offset_taa(self):
        """Test getting jitter offset when using TAA."""
        settings = AASettings(method=AAMethod.TAA)
        effect = AAEffect(settings)
        effect.setup(1920, 1080)

        x, y = effect.get_jitter_offset()

        assert -0.5 <= x <= 0.5
        assert -0.5 <= y <= 0.5

    def test_effect_jitter_offset_fxaa(self):
        """Test jitter is zero when not using TAA."""
        settings = AASettings(method=AAMethod.FXAA)
        effect = AAEffect(settings)

        x, y = effect.get_jitter_offset()

        assert x == 0.0
        assert y == 0.0

    def test_effect_jittered_projection_taa(self):
        """Test projection jitter with TAA."""
        settings = AASettings(method=AAMethod.TAA)
        effect = AAEffect(settings)
        effect.setup(1920, 1080)

        projection = [[1, 0, 0, 0]] * 4

        jittered = effect.get_jittered_projection(projection)

        # Should be modified
        assert jittered != projection

    def test_effect_jittered_projection_fxaa(self):
        """Test projection not jittered with FXAA."""
        settings = AASettings(method=AAMethod.FXAA)
        effect = AAEffect(settings)

        projection = [[1, 0, 0, 0]] * 4

        result = effect.get_jittered_projection(projection)

        # Should be unchanged
        assert result == projection

    def test_effect_invalidate_history(self):
        """Test invalidating history through effect."""
        effect = AAEffect()
        effect.setup(1920, 1080)

        effect._taa._history_valid = True
        effect.invalidate_history()

        assert effect._taa.history_valid is False

    def test_effect_cleanup(self):
        """Test effect cleanup."""
        effect = AAEffect()
        effect.setup(1920, 1080)
        effect.cleanup()
        # Should not raise


class TestAAMethod:
    """Test AAMethod enum."""

    def test_all_methods_exist(self):
        """Test all methods exist."""
        methods = [
            AAMethod.NONE,
            AAMethod.FXAA,
            AAMethod.SMAA,
            AAMethod.TAA,
        ]

        for method in methods:
            assert method is not None


class TestFXAAQuality:
    """Test FXAAQuality enum."""

    def test_all_qualities_exist(self):
        """Test all qualities exist."""
        qualities = [
            FXAAQuality.LOW,
            FXAAQuality.MEDIUM,
            FXAAQuality.HIGH,
            FXAAQuality.EXTREME,
        ]

        for quality in qualities:
            assert quality is not None


class TestSMAAQuality:
    """Test SMAAQuality enum."""

    def test_all_qualities_exist(self):
        """Test all qualities exist."""
        qualities = [
            SMAAQuality.LOW,
            SMAAQuality.MEDIUM,
            SMAAQuality.HIGH,
            SMAAQuality.ULTRA,
        ]

        for quality in qualities:
            assert quality is not None


class TestJitterPattern:
    """Test JitterPattern enum."""

    def test_all_patterns_exist(self):
        """Test all patterns exist."""
        patterns = [
            JitterPattern.HALTON_8,
            JitterPattern.HALTON_16,
            JitterPattern.HALTON_32,
            JitterPattern.UNIFORM_4,
            JitterPattern.RGSS_4,
        ]

        for pattern in patterns:
            assert pattern is not None

    def test_pattern_sample_counts(self):
        """Test each pattern has expected sample count."""
        expected_counts = {
            JitterPattern.HALTON_8: 8,
            JitterPattern.HALTON_16: 16,
            JitterPattern.HALTON_32: 32,
            JitterPattern.UNIFORM_4: 4,
            JitterPattern.RGSS_4: 4,
        }

        for pattern, count in expected_counts.items():
            jitter = JitterSequence(pattern)
            assert jitter.sample_count == count


class TestTAATemporalStability:
    """Test TAA temporal stability characteristics."""

    def test_jitter_sequence_low_discrepancy(self):
        """Verify Halton sequence has good coverage."""
        jitter = JitterSequence(JitterPattern.HALTON_16)

        samples = [jitter.next() for _ in range(16)]

        # Check that samples spread across the range
        x_coords = [s[0] for s in samples]
        y_coords = [s[1] for s in samples]

        # Should have variety in positions
        assert max(x_coords) > 0.3
        assert min(x_coords) < -0.3
        assert max(y_coords) > 0.3
        assert min(y_coords) < -0.3

        # No duplicate samples
        unique_samples = set(samples)
        assert len(unique_samples) == 16

    def test_history_weight_in_valid_range(self):
        """Verify history weight is constrained properly."""
        settings = TAASettings()

        # Default should be in valid range
        assert 0.8 <= settings.history_weight <= 0.98

    def test_taa_matrix_jitter_is_small(self):
        """Verify jitter applied to matrix is sub-pixel."""
        taa = TAA()
        taa.setup(1920, 1080)

        projection = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        jittered = taa.get_jittered_projection(projection)

        # Jitter should be small (sub-pixel in clip space)
        jitter_x = jittered[2][0]
        jitter_y = jittered[2][1]

        # Sub-pixel means less than 2/resolution
        assert abs(jitter_x) < 2.0 / 1920 * 2
        assert abs(jitter_y) < 2.0 / 1080 * 2


class TestAAConstants:
    """Test that AA uses centralized constants."""

    def test_fxaa_defaults_match_constants(self):
        """Verify FXAA defaults match constants module."""
        from engine.rendering.postprocess.constants import AA

        settings = FXAASettings()

        assert settings.edge_threshold == AA.FXAA_EDGE_THRESHOLD_DEFAULT
        assert settings.edge_threshold_min == AA.FXAA_EDGE_THRESHOLD_MIN_DEFAULT
        assert settings.subpixel_quality == AA.FXAA_SUBPIXEL_QUALITY_DEFAULT

    def test_taa_defaults_match_constants(self):
        """Verify TAA defaults match constants module."""
        from engine.rendering.postprocess.constants import AA

        settings = TAASettings()

        assert settings.history_weight == AA.TAA_HISTORY_WEIGHT_DEFAULT
        assert settings.sharpen_amount == AA.TAA_SHARPEN_AMOUNT_DEFAULT
        assert settings.velocity_rejection_threshold == AA.TAA_VELOCITY_THRESHOLD_DEFAULT
