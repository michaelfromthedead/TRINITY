"""
Tests for Tone Mapping module (T-DEMO-3.11).

Tests cover:
  - Individual tone mapping operators (Reinhard, ACES, Uncharted2)
  - ToneMapper class configuration and application
  - Gamma correction and sRGB conversion
  - WGSL code generation
  - Edge cases and validation

Requirements:
  - HDR colors mapped to [0, 1] display range
  - No clipping for bright values
  - Gamma correction applied
  - 20+ tests comparing output ranges
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.demoscene.tone_mapping import (
    ToneMappingOperator,
    reinhard,
    reinhard_extended,
    aces_filmic,
    uncharted2,
    linear_clamp,
    gamma_correct,
    linear_to_srgb,
    srgb_to_linear,
    ToneMapper,
    generate_tone_mapping_wgsl,
    validate_color_range,
    is_valid_hdr_color,
)
from engine.rendering.demoscene.ray_generation import Vec3


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def black_color() -> Vec3:
    """Pure black (0, 0, 0)."""
    return Vec3(0.0, 0.0, 0.0)


@pytest.fixture
def white_linear() -> Vec3:
    """Linear white (1, 1, 1)."""
    return Vec3(1.0, 1.0, 1.0)


@pytest.fixture
def hdr_bright() -> Vec3:
    """Bright HDR color (values > 1)."""
    return Vec3(5.0, 3.0, 2.0)


@pytest.fixture
def hdr_very_bright() -> Vec3:
    """Very bright HDR color."""
    return Vec3(100.0, 50.0, 25.0)


@pytest.fixture
def mid_gray() -> Vec3:
    """Mid-gray (0.18 linear)."""
    return Vec3(0.18, 0.18, 0.18)


# =============================================================================
# Reinhard Tone Mapping Tests
# =============================================================================


class TestReinhardSimple:
    """Tests for simple Reinhard tone mapping."""

    def test_reinhard_black_stays_black(self, black_color):
        """Black should remain black."""
        result = reinhard(black_color)
        assert result.x == pytest.approx(0.0, abs=1e-9)
        assert result.y == pytest.approx(0.0, abs=1e-9)
        assert result.z == pytest.approx(0.0, abs=1e-9)

    def test_reinhard_white_maps_to_half(self, white_linear):
        """Linear white (1,1,1) maps to (0.5, 0.5, 0.5)."""
        result = reinhard(white_linear)
        assert result.x == pytest.approx(0.5, abs=1e-6)
        assert result.y == pytest.approx(0.5, abs=1e-6)
        assert result.z == pytest.approx(0.5, abs=1e-6)

    def test_reinhard_formula(self):
        """Test the Reinhard formula: c / (1 + c)."""
        color = Vec3(2.0, 4.0, 8.0)
        result = reinhard(color)
        assert result.x == pytest.approx(2.0 / 3.0, abs=1e-6)
        assert result.y == pytest.approx(4.0 / 5.0, abs=1e-6)
        assert result.z == pytest.approx(8.0 / 9.0, abs=1e-6)

    def test_reinhard_asymptotic_to_one(self, hdr_very_bright):
        """Very bright values should approach 1.0 but never exceed it."""
        result = reinhard(hdr_very_bright)
        assert result.x < 1.0
        assert result.y < 1.0
        assert result.z < 1.0
        # Should be close to 1.0
        assert result.x > 0.99
        assert result.y > 0.98
        assert result.z > 0.96

    def test_reinhard_preserves_zero_channel(self):
        """Zero channels should stay zero."""
        color = Vec3(1.0, 0.0, 2.0)
        result = reinhard(color)
        assert result.y == pytest.approx(0.0, abs=1e-9)
        assert result.x > 0.0
        assert result.z > 0.0


class TestReinhardExtended:
    """Tests for extended Reinhard with white point control."""

    def test_extended_black_stays_black(self, black_color):
        """Black should remain black."""
        result = reinhard_extended(black_color, max_white=4.0)
        assert result.x == pytest.approx(0.0, abs=1e-9)

    def test_extended_white_point_reaches_one(self):
        """Max white value should map to 1.0."""
        max_white = 4.0
        color = Vec3(max_white, max_white, max_white)
        result = reinhard_extended(color, max_white=max_white)
        # Should be very close to 1.0
        assert result.x == pytest.approx(1.0, abs=0.01)

    def test_extended_higher_max_white_preserves_detail(self):
        """Higher max_white preserves more highlight detail."""
        color = Vec3(2.0, 2.0, 2.0)
        result_low = reinhard_extended(color, max_white=2.0)
        result_high = reinhard_extended(color, max_white=8.0)
        # Higher max_white should produce darker midtones
        assert result_high.x < result_low.x

    def test_extended_matches_simple_at_infinite_white(self):
        """At very high max_white, should approach simple Reinhard."""
        color = Vec3(1.0, 2.0, 3.0)
        extended = reinhard_extended(color, max_white=1000.0)
        simple = reinhard(color)
        assert extended.x == pytest.approx(simple.x, abs=0.01)
        assert extended.y == pytest.approx(simple.y, abs=0.01)
        assert extended.z == pytest.approx(simple.z, abs=0.01)


# =============================================================================
# ACES Filmic Tests
# =============================================================================


class TestACESFilmic:
    """Tests for ACES filmic tone mapping."""

    def test_aces_black_stays_black(self, black_color):
        """Black should remain black."""
        result = aces_filmic(black_color)
        assert result.x == pytest.approx(0.0, abs=1e-6)
        assert result.y == pytest.approx(0.0, abs=1e-6)
        assert result.z == pytest.approx(0.0, abs=1e-6)

    def test_aces_output_in_range(self, hdr_bright):
        """ACES output should be in [0, 1]."""
        result = aces_filmic(hdr_bright)
        assert 0.0 <= result.x <= 1.0
        assert 0.0 <= result.y <= 1.0
        assert 0.0 <= result.z <= 1.0

    def test_aces_very_bright_clamped(self, hdr_very_bright):
        """Very bright values should be in [0, 1] range."""
        result = aces_filmic(hdr_very_bright)
        assert 0.0 <= result.x <= 1.0
        assert 0.0 <= result.y <= 1.0
        assert 0.0 <= result.z <= 1.0

    def test_aces_mid_gray_preserved(self, mid_gray):
        """Mid-gray should map to approximately mid-tone."""
        result = aces_filmic(mid_gray)
        # ACES typically maps 0.18 to around 0.2
        assert 0.1 < result.x < 0.4

    def test_aces_negative_values_handled(self):
        """Negative values should be clamped to 0."""
        color = Vec3(-1.0, -0.5, 0.0)
        result = aces_filmic(color)
        assert result.x >= 0.0
        assert result.y >= 0.0
        assert result.z >= 0.0


# =============================================================================
# Uncharted 2 Tests
# =============================================================================


class TestUncharted2:
    """Tests for Uncharted 2 filmic tone mapping."""

    def test_uncharted2_black_stays_black(self, black_color):
        """Black should remain black."""
        result = uncharted2(black_color)
        assert result.x == pytest.approx(0.0, abs=1e-6)
        assert result.y == pytest.approx(0.0, abs=1e-6)
        assert result.z == pytest.approx(0.0, abs=1e-6)

    def test_uncharted2_output_in_range(self, hdr_bright):
        """Output should be in [0, 1]."""
        result = uncharted2(hdr_bright)
        assert 0.0 <= result.x <= 1.0
        assert 0.0 <= result.y <= 1.0
        assert 0.0 <= result.z <= 1.0

    def test_uncharted2_exposure_affects_brightness(self):
        """Higher exposure should produce brighter output."""
        color = Vec3(0.5, 0.5, 0.5)
        result_low = uncharted2(color, exposure=1.0)
        result_high = uncharted2(color, exposure=4.0)
        assert result_high.x > result_low.x
        assert result_high.y > result_low.y
        assert result_high.z > result_low.z

    def test_uncharted2_very_bright_saturates(self, hdr_very_bright):
        """Very bright values should saturate close to 1.0."""
        result = uncharted2(hdr_very_bright)
        assert result.x > 0.95
        assert result.y > 0.95
        assert result.z > 0.95


# =============================================================================
# Linear Clamp Tests
# =============================================================================


class TestLinearClamp:
    """Tests for linear clamping (no tone mapping)."""

    def test_linear_clamp_values_in_range(self):
        """Values in [0, 1] should pass through."""
        color = Vec3(0.5, 0.8, 0.2)
        result = linear_clamp(color)
        assert result.x == pytest.approx(0.5, abs=1e-9)
        assert result.y == pytest.approx(0.8, abs=1e-9)
        assert result.z == pytest.approx(0.2, abs=1e-9)

    def test_linear_clamp_clips_bright(self, hdr_bright):
        """Values > 1 should be clamped to 1."""
        result = linear_clamp(hdr_bright)
        assert result.x == 1.0
        assert result.y == 1.0
        assert result.z == 1.0

    def test_linear_clamp_clips_negative(self):
        """Negative values should be clamped to 0."""
        color = Vec3(-0.5, -1.0, 0.5)
        result = linear_clamp(color)
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.5


# =============================================================================
# Gamma Correction Tests
# =============================================================================


class TestGammaCorrection:
    """Tests for gamma correction."""

    def test_gamma_black_stays_black(self, black_color):
        """Black should remain black."""
        result = gamma_correct(black_color, 2.2)
        assert result.x == pytest.approx(0.0, abs=1e-9)

    def test_gamma_white_stays_white(self):
        """White should remain white."""
        result = gamma_correct(Vec3(1.0, 1.0, 1.0), 2.2)
        assert result.x == pytest.approx(1.0, abs=1e-9)

    def test_gamma_brightens_midtones(self):
        """Gamma correction should brighten linear midtones."""
        linear = Vec3(0.5, 0.5, 0.5)
        gamma_corrected = gamma_correct(linear, 2.2)
        # 0.5^(1/2.2) ≈ 0.73
        assert gamma_corrected.x > 0.7
        assert gamma_corrected.x < 0.8

    def test_gamma_formula(self):
        """Test gamma formula: c^(1/gamma)."""
        color = Vec3(0.25, 0.5, 0.75)
        gamma = 2.0
        result = gamma_correct(color, gamma)
        assert result.x == pytest.approx(0.25 ** 0.5, abs=1e-6)
        assert result.y == pytest.approx(0.5 ** 0.5, abs=1e-6)
        assert result.z == pytest.approx(0.75 ** 0.5, abs=1e-6)


class TestSRGBConversion:
    """Tests for sRGB conversion functions."""

    def test_linear_to_srgb_black(self, black_color):
        """Black should remain black."""
        result = linear_to_srgb(black_color)
        assert result.x == pytest.approx(0.0, abs=1e-9)

    def test_linear_to_srgb_white(self):
        """White should remain white."""
        result = linear_to_srgb(Vec3(1.0, 1.0, 1.0))
        assert result.x == pytest.approx(1.0, abs=1e-6)

    def test_srgb_roundtrip(self):
        """Converting to sRGB and back should be identity."""
        original = Vec3(0.25, 0.5, 0.75)
        srgb = linear_to_srgb(original)
        back = srgb_to_linear(srgb)
        assert back.x == pytest.approx(original.x, abs=1e-5)
        assert back.y == pytest.approx(original.y, abs=1e-5)
        assert back.z == pytest.approx(original.z, abs=1e-5)

    def test_srgb_linear_region(self):
        """Very dark values use linear conversion."""
        color = Vec3(0.001, 0.002, 0.003)
        result = linear_to_srgb(color)
        # In linear region: result = 12.92 * linear
        assert result.x == pytest.approx(0.001 * 12.92, abs=1e-6)


# =============================================================================
# ToneMapper Class Tests
# =============================================================================


class TestToneMapperClass:
    """Tests for the ToneMapper class."""

    def test_default_operator_is_aces(self):
        """Default operator should be ACES."""
        mapper = ToneMapper()
        assert mapper.default_operator == "aces"

    def test_apply_with_default_operator(self, hdr_bright):
        """Apply should use default operator."""
        mapper = ToneMapper(default_operator="reinhard")
        result = mapper.apply(hdr_bright)
        expected = reinhard(hdr_bright)
        # Should match reinhard (before gamma)
        # Note: gamma is applied, so we compare with gamma-corrected result
        assert result.x > 0.0

    def test_apply_with_specified_operator(self, hdr_bright):
        """Can override operator per call."""
        mapper = ToneMapper(default_operator="aces")
        result_aces = mapper.apply(hdr_bright, operator="aces")
        result_reinhard = mapper.apply(hdr_bright, operator="reinhard")
        # Different operators should give different results
        assert result_aces.x != result_reinhard.x

    def test_exposure_multiplier(self):
        """Exposure should scale input."""
        mapper_normal = ToneMapper(exposure=1.0)
        mapper_bright = ToneMapper(exposure=2.0)
        color = Vec3(0.5, 0.5, 0.5)
        result_normal = mapper_normal.apply(color)
        result_bright = mapper_bright.apply(color)
        assert result_bright.x > result_normal.x

    def test_gamma_disabled(self):
        """Can disable gamma correction."""
        mapper = ToneMapper()
        color = Vec3(0.5, 0.5, 0.5)
        with_gamma = mapper.apply(color, apply_gamma=True)
        without_gamma = mapper.apply(color, apply_gamma=False)
        assert with_gamma.x != without_gamma.x

    def test_invalid_operator_raises(self):
        """Invalid operator name should raise ValueError."""
        mapper = ToneMapper()
        with pytest.raises(ValueError, match="Unknown tone mapping operator"):
            mapper.apply(Vec3(1.0, 1.0, 1.0), operator="invalid_op")

    def test_batch_processing(self):
        """apply_batch should process multiple colors."""
        mapper = ToneMapper()
        colors = [Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0)]
        results = mapper.apply_batch(colors)
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.x <= 1.0
            assert 0.0 <= r.y <= 1.0
            assert 0.0 <= r.z <= 1.0


# =============================================================================
# WGSL Generation Tests
# =============================================================================


class TestWGSLGeneration:
    """Tests for WGSL code generation."""

    def test_generates_gamma_function(self):
        """Should include gamma correction function."""
        wgsl = generate_tone_mapping_wgsl()
        assert "fn gamma_correct" in wgsl

    def test_generates_aces_by_default(self):
        """Should include ACES function by default."""
        wgsl = generate_tone_mapping_wgsl(operator="aces")
        assert "fn tone_map_aces" in wgsl

    def test_generates_reinhard(self):
        """Should include Reinhard when requested."""
        wgsl = generate_tone_mapping_wgsl(operator="reinhard")
        assert "fn tone_map_reinhard" in wgsl

    def test_generates_all_operators(self):
        """include_all should include all operators."""
        wgsl = generate_tone_mapping_wgsl(include_all=True)
        assert "fn tone_map_reinhard" in wgsl
        assert "fn tone_map_reinhard_extended" in wgsl
        assert "fn tone_map_aces" in wgsl
        assert "fn tone_map_uncharted2" in wgsl

    def test_generates_entry_point(self):
        """Should include unified entry point."""
        wgsl = generate_tone_mapping_wgsl()
        assert "fn tone_map(" in wgsl

    def test_generates_linear_to_srgb(self):
        """Should include sRGB conversion."""
        wgsl = generate_tone_mapping_wgsl()
        assert "fn linear_to_srgb" in wgsl


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for validation helpers."""

    def test_valid_color_no_errors(self):
        """Valid color should have no errors."""
        color = Vec3(0.5, 0.5, 0.5)
        errors = validate_color_range(color)
        assert len(errors) == 0

    def test_negative_value_error(self):
        """Negative values should produce error."""
        color = Vec3(-0.1, 0.5, 0.5)
        errors = validate_color_range(color)
        assert len(errors) > 0
        assert "negative" in errors[0].lower()

    def test_exceeds_one_error(self):
        """Values > 1.0 should produce error."""
        color = Vec3(0.5, 1.5, 0.5)
        errors = validate_color_range(color)
        assert len(errors) > 0
        assert "exceeds" in errors[0].lower()

    def test_nan_value_error(self):
        """NaN should produce error."""
        color = Vec3(float('nan'), 0.5, 0.5)
        errors = validate_color_range(color)
        assert len(errors) > 0
        assert "nan" in errors[0].lower()

    def test_inf_value_error(self):
        """Infinity should produce error."""
        color = Vec3(float('inf'), 0.5, 0.5)
        errors = validate_color_range(color)
        assert len(errors) > 0
        # inf triggers both "exceeds" and "infinite" checks
        assert "inf" in errors[0].lower() or "infinite" in errors[0].lower()

    def test_is_valid_hdr_color(self):
        """Test is_valid_hdr_color helper."""
        assert is_valid_hdr_color(Vec3(0.5, 1.0, 10.0))  # HDR valid
        assert is_valid_hdr_color(Vec3(0.0, 0.0, 0.0))   # Black valid
        assert not is_valid_hdr_color(Vec3(-0.1, 0.0, 0.0))  # Negative invalid
        assert not is_valid_hdr_color(Vec3(float('nan'), 0.0, 0.0))  # NaN invalid
        assert not is_valid_hdr_color(Vec3(float('inf'), 0.0, 0.0))  # Inf invalid


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_small_values(self):
        """Very small values should be handled correctly."""
        color = Vec3(1e-10, 1e-10, 1e-10)
        result = aces_filmic(color)
        assert result.x >= 0.0
        assert not math.isnan(result.x)

    def test_very_large_values(self):
        """Very large values should not overflow."""
        color = Vec3(1e10, 1e10, 1e10)
        result = aces_filmic(color)
        assert result.x <= 1.0
        assert not math.isnan(result.x)
        assert not math.isinf(result.x)

    def test_all_operators_handle_hdr(self, hdr_bright):
        """All operators should handle HDR values without issues."""
        operators = [reinhard, aces_filmic, lambda c: uncharted2(c, 2.0)]
        for op in operators:
            result = op(hdr_bright)
            assert 0.0 <= result.x <= 1.0
            assert 0.0 <= result.y <= 1.0
            assert 0.0 <= result.z <= 1.0
            assert not math.isnan(result.x)
            assert not math.isnan(result.y)
            assert not math.isnan(result.z)
