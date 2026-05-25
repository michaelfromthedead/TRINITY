"""Tests for material system constants.

Verifies that constants have sensible values and are properly defined.
"""
import pytest

from engine.rendering.materials.constants import (
    # PBR ranges
    PBR_METALLIC_RANGE,
    PBR_ROUGHNESS_RANGE,
    PBR_NORMAL_SCALE_RANGE,
    PBR_AO_RANGE,
    PBR_BASE_COLOR_RANGE,
    PBR_EMISSIVE_MIN,
    # Clear coat
    CLEAR_COAT_INTENSITY_RANGE,
    CLEAR_COAT_ROUGHNESS_RANGE,
    CLEAR_COAT_IOR_RANGE,
    # Shader compilation
    SHADER_COMPILATION_TIMEOUT_SECONDS,
    HOT_RELOAD_POLL_INTERVAL_SECONDS,
    SHADER_HASH_LENGTH,
    # PSO cache
    PSO_CACHE_DEFAULT_MAX_SIZE,
    PSO_CACHE_MIN_SIZE,
    PSO_CACHE_MAX_SIZE,
    # Material graph
    MATERIAL_GRAPH_MAX_NODES,
    MATERIAL_GRAPH_MAX_DEPTH,
    # Safe division
    SAFE_DIVISION_EPSILON,
    # Color space
    SRGB_GAMMA,
    LUMINANCE_COEFFICIENTS_R,
    LUMINANCE_COEFFICIENTS_G,
    LUMINANCE_COEFFICIENTS_B,
)


class TestPBRRanges:
    """Test PBR parameter range constants."""

    def test_metallic_range(self):
        """Test metallic range is 0-1."""
        assert PBR_METALLIC_RANGE.min_value == 0.0
        assert PBR_METALLIC_RANGE.max_value == 1.0
        assert 0.0 <= PBR_METALLIC_RANGE.default_value <= 1.0

    def test_roughness_range(self):
        """Test roughness range is 0-1."""
        assert PBR_ROUGHNESS_RANGE.min_value == 0.0
        assert PBR_ROUGHNESS_RANGE.max_value == 1.0
        assert 0.0 <= PBR_ROUGHNESS_RANGE.default_value <= 1.0

    def test_normal_scale_range(self):
        """Test normal scale range allows values up to 2."""
        assert PBR_NORMAL_SCALE_RANGE.min_value == 0.0
        assert PBR_NORMAL_SCALE_RANGE.max_value == 2.0

    def test_ao_range(self):
        """Test ambient occlusion range is 0-1."""
        assert PBR_AO_RANGE.min_value == 0.0
        assert PBR_AO_RANGE.max_value == 1.0

    def test_emissive_min(self):
        """Test emissive minimum is non-negative."""
        assert PBR_EMISSIVE_MIN >= 0.0


class TestClearCoatRanges:
    """Test clear coat parameter range constants."""

    def test_intensity_range(self):
        """Test clear coat intensity is 0-1."""
        assert CLEAR_COAT_INTENSITY_RANGE.min_value == 0.0
        assert CLEAR_COAT_INTENSITY_RANGE.max_value == 1.0

    def test_ior_range(self):
        """Test clear coat IOR is realistic (1-3)."""
        assert CLEAR_COAT_IOR_RANGE.min_value >= 1.0
        assert CLEAR_COAT_IOR_RANGE.max_value <= 3.0
        # Default should be around typical clear coat IOR
        assert 1.4 <= CLEAR_COAT_IOR_RANGE.default_value <= 1.6


class TestShaderCompilationConstants:
    """Test shader compilation constants."""

    def test_timeout_is_positive(self):
        """Test compilation timeout is positive."""
        assert SHADER_COMPILATION_TIMEOUT_SECONDS > 0

    def test_poll_interval_is_positive(self):
        """Test hot-reload poll interval is positive."""
        assert HOT_RELOAD_POLL_INTERVAL_SECONDS > 0

    def test_hash_length_is_reasonable(self):
        """Test hash length produces unique-enough hashes."""
        # 16 hex chars = 64 bits = very low collision probability
        assert SHADER_HASH_LENGTH >= 8
        assert SHADER_HASH_LENGTH <= 64


class TestPSOCacheConstants:
    """Test PSO cache constants."""

    def test_cache_size_hierarchy(self):
        """Test cache size constants are properly ordered."""
        assert PSO_CACHE_MIN_SIZE < PSO_CACHE_DEFAULT_MAX_SIZE
        assert PSO_CACHE_DEFAULT_MAX_SIZE < PSO_CACHE_MAX_SIZE

    def test_default_size_is_reasonable(self):
        """Test default cache size is reasonable for typical use."""
        # Should be large enough for a complex scene
        assert PSO_CACHE_DEFAULT_MAX_SIZE >= 256
        # But not so large it wastes memory
        assert PSO_CACHE_DEFAULT_MAX_SIZE <= 4096


class TestMaterialGraphConstants:
    """Test material graph constants."""

    def test_max_nodes_is_reasonable(self):
        """Test max nodes allows complex graphs but prevents abuse."""
        assert MATERIAL_GRAPH_MAX_NODES >= 256
        assert MATERIAL_GRAPH_MAX_NODES <= 4096

    def test_max_depth_prevents_stack_overflow(self):
        """Test max depth is safe for recursion."""
        assert MATERIAL_GRAPH_MAX_DEPTH >= 32
        assert MATERIAL_GRAPH_MAX_DEPTH <= 256


class TestMathConstants:
    """Test mathematical constants."""

    def test_safe_division_epsilon(self):
        """Test safe division epsilon is small but non-zero."""
        assert SAFE_DIVISION_EPSILON > 0
        assert SAFE_DIVISION_EPSILON < 0.01

    def test_srgb_gamma(self):
        """Test sRGB gamma is the standard value."""
        assert abs(SRGB_GAMMA - 2.2) < 0.01

    def test_luminance_coefficients_sum_to_one(self):
        """Test Rec. 709 luminance coefficients sum to 1."""
        total = (
            LUMINANCE_COEFFICIENTS_R
            + LUMINANCE_COEFFICIENTS_G
            + LUMINANCE_COEFFICIENTS_B
        )
        assert abs(total - 1.0) < 0.001
