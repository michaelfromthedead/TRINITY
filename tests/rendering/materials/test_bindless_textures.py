"""Tests for Bindless Texture Arrays (T-MAT-5.7).

This module provides comprehensive whitebox and blackbox tests for the
bindless texture array system, including:

- TextureSlot metadata handling
- BindlessTextureArray slot management
- Free-list allocation and reuse
- WGSL code generation
- Capability detection and fallback
- Serialization for GPU upload

Test Categories:
    - Unit tests for TextureSlot
    - Unit tests for SamplerConfig
    - Unit tests for BindlessCapabilities
    - Integration tests for BindlessTextureArray
    - WGSL generation validation
    - Edge cases and error handling
"""

import pytest
import struct
import math
from typing import Dict, List, Optional

from trinity.materials.bindless import (
    # Constants
    MAX_BINDLESS_TEXTURES,
    DEFAULT_BINDLESS_CAPACITY,
    INVALID_TEXTURE_INDEX,
    MIN_TEXTURES_FOR_BINDLESS,
    # Enums
    TextureFormat,
    FilterMode,
    AddressMode,
    # Classes
    SamplerConfig,
    TextureSlot,
    BindlessCapabilities,
    BindlessArrayStats,
    BindlessTextureArray,
    # Factory functions
    create_bindless_array,
    create_default_slots,
)


# =============================================================================
# Constants Tests
# =============================================================================


class TestBindlessConstants:
    """Test bindless texture constants."""

    def test_max_bindless_textures(self) -> None:
        """MAX_BINDLESS_TEXTURES matches hardware convention (4096)."""
        assert MAX_BINDLESS_TEXTURES == 4096

    def test_default_capacity(self) -> None:
        """DEFAULT_BINDLESS_CAPACITY is reasonable subset of max."""
        assert DEFAULT_BINDLESS_CAPACITY == 1024
        assert DEFAULT_BINDLESS_CAPACITY <= MAX_BINDLESS_TEXTURES

    def test_invalid_texture_index(self) -> None:
        """INVALID_TEXTURE_INDEX is u32::MAX sentinel."""
        assert INVALID_TEXTURE_INDEX == 0xFFFFFFFF
        assert INVALID_TEXTURE_INDEX == 2**32 - 1

    def test_min_textures_for_bindless(self) -> None:
        """MIN_TEXTURES_FOR_BINDLESS is low enough for most hardware."""
        assert MIN_TEXTURES_FOR_BINDLESS == 16
        assert MIN_TEXTURES_FOR_BINDLESS > 0


# =============================================================================
# TextureFormat Tests
# =============================================================================


class TestTextureFormat:
    """Test TextureFormat enum functionality."""

    def test_all_formats_have_unique_values(self) -> None:
        """Each format has a unique integer value."""
        values = [fmt.value for fmt in TextureFormat]
        assert len(values) == len(set(values))

    def test_to_wgsl_common_formats(self) -> None:
        """to_wgsl() returns correct WGSL format strings."""
        assert TextureFormat.RGBA8_UNORM.to_wgsl() == "rgba8unorm"
        assert TextureFormat.RGBA8_UNORM_SRGB.to_wgsl() == "rgba8unorm-srgb"
        assert TextureFormat.RGBA16_FLOAT.to_wgsl() == "rgba16float"
        assert TextureFormat.RGBA32_FLOAT.to_wgsl() == "rgba32float"
        assert TextureFormat.R8_UNORM.to_wgsl() == "r8unorm"

    def test_is_srgb(self) -> None:
        """is_srgb() correctly identifies sRGB formats."""
        assert TextureFormat.RGBA8_UNORM_SRGB.is_srgb()
        assert TextureFormat.BC1_RGBA_UNORM_SRGB.is_srgb()
        assert TextureFormat.BC3_RGBA_UNORM_SRGB.is_srgb()
        assert TextureFormat.BC7_RGBA_UNORM_SRGB.is_srgb()

        assert not TextureFormat.RGBA8_UNORM.is_srgb()
        assert not TextureFormat.RGBA16_FLOAT.is_srgb()
        assert not TextureFormat.R8_UNORM.is_srgb()

    def test_is_compressed(self) -> None:
        """is_compressed() identifies block-compressed formats."""
        assert TextureFormat.BC1_RGBA_UNORM.is_compressed()
        assert TextureFormat.BC3_RGBA_UNORM.is_compressed()
        assert TextureFormat.BC5_RG_UNORM.is_compressed()
        assert TextureFormat.BC7_RGBA_UNORM.is_compressed()

        assert not TextureFormat.RGBA8_UNORM.is_compressed()
        assert not TextureFormat.RGBA16_FLOAT.is_compressed()

    def test_bytes_per_pixel(self) -> None:
        """bytes_per_pixel() returns correct values."""
        assert TextureFormat.RGBA8_UNORM.bytes_per_pixel() == 4.0
        assert TextureFormat.RGBA16_FLOAT.bytes_per_pixel() == 8.0
        assert TextureFormat.RGBA32_FLOAT.bytes_per_pixel() == 16.0
        assert TextureFormat.R8_UNORM.bytes_per_pixel() == 1.0
        assert TextureFormat.RG8_UNORM.bytes_per_pixel() == 2.0

        # BC formats: bits per block / pixels per block
        assert TextureFormat.BC1_RGBA_UNORM.bytes_per_pixel() == 0.5  # 8 bytes / 16 pixels
        assert TextureFormat.BC3_RGBA_UNORM.bytes_per_pixel() == 1.0  # 16 bytes / 16 pixels


# =============================================================================
# SamplerConfig Tests
# =============================================================================


class TestSamplerConfig:
    """Test SamplerConfig functionality."""

    def test_default_values(self) -> None:
        """Default config uses linear filtering and repeat wrapping."""
        config = SamplerConfig()
        assert config.mag_filter == FilterMode.LINEAR
        assert config.min_filter == FilterMode.LINEAR
        assert config.mipmap_filter == FilterMode.LINEAR
        assert config.address_u == AddressMode.REPEAT
        assert config.address_v == AddressMode.REPEAT
        assert config.address_w == AddressMode.REPEAT
        assert config.lod_min_clamp == 0.0
        assert config.lod_max_clamp == 32.0
        assert config.max_anisotropy == 1

    def test_content_hash_deterministic(self) -> None:
        """content_hash() produces consistent results."""
        config1 = SamplerConfig()
        config2 = SamplerConfig()
        assert config1.content_hash() == config2.content_hash()

    def test_content_hash_differs_for_different_configs(self) -> None:
        """Different configs produce different hashes."""
        config1 = SamplerConfig.linear_repeat()
        config2 = SamplerConfig.linear_clamp()
        config3 = SamplerConfig.nearest_repeat()

        hashes = {config1.content_hash(), config2.content_hash(), config3.content_hash()}
        assert len(hashes) == 3

    def test_linear_repeat_preset(self) -> None:
        """linear_repeat() returns standard linear filtering config."""
        config = SamplerConfig.linear_repeat()
        assert config.mag_filter == FilterMode.LINEAR
        assert config.address_u == AddressMode.REPEAT

    def test_linear_clamp_preset(self) -> None:
        """linear_clamp() returns edge-clamped config."""
        config = SamplerConfig.linear_clamp()
        assert config.mag_filter == FilterMode.LINEAR
        assert config.address_u == AddressMode.CLAMP_TO_EDGE
        assert config.address_v == AddressMode.CLAMP_TO_EDGE
        assert config.address_w == AddressMode.CLAMP_TO_EDGE

    def test_nearest_repeat_preset(self) -> None:
        """nearest_repeat() returns point filtering config."""
        config = SamplerConfig.nearest_repeat()
        assert config.mag_filter == FilterMode.NEAREST
        assert config.min_filter == FilterMode.NEAREST
        assert config.mipmap_filter == FilterMode.NEAREST

    def test_anisotropic_preset(self) -> None:
        """anisotropic() respects level bounds."""
        config_16 = SamplerConfig.anisotropic(16)
        assert config_16.max_anisotropy == 16

        config_4 = SamplerConfig.anisotropic(4)
        assert config_4.max_anisotropy == 4

        # Clamped to [1, 16]
        config_high = SamplerConfig.anisotropic(32)
        assert config_high.max_anisotropy == 16

        config_low = SamplerConfig.anisotropic(0)
        assert config_low.max_anisotropy == 1


# =============================================================================
# TextureSlot Tests
# =============================================================================


class TestTextureSlot:
    """Test TextureSlot metadata handling."""

    def test_default_values(self) -> None:
        """TextureSlot has sensible defaults."""
        slot = TextureSlot(name="test", width=256, height=256)
        assert slot.name == "test"
        assert slot.width == 256
        assert slot.height == 256
        assert slot.format == TextureFormat.RGBA8_UNORM
        assert slot.mip_levels == 1
        assert slot.layer_count == 1
        assert slot.index == INVALID_TEXTURE_INDEX
        assert not slot.valid
        assert slot.source_path is None
        assert slot.content_hash is None

    def test_byte_size_calculation(self) -> None:
        """byte_size() calculates correct memory usage."""
        slot = TextureSlot(
            name="test",
            width=1024,
            height=1024,
            format=TextureFormat.RGBA8_UNORM,  # 4 bytes per pixel
            mip_levels=1,
        )
        assert slot.byte_size(include_mipmaps=False) == 1024 * 1024 * 4

    def test_byte_size_with_mipmaps(self) -> None:
        """byte_size() includes mipmap chain when requested."""
        slot = TextureSlot(
            name="test",
            width=256,
            height=256,
            format=TextureFormat.RGBA8_UNORM,
            mip_levels=9,  # 256, 128, 64, 32, 16, 8, 4, 2, 1
        )
        base_size = 256 * 256 * 4
        size_with_mips = slot.byte_size(include_mipmaps=True)
        assert size_with_mips > base_size
        # Mipmap chain adds approximately 1/3 more
        assert size_with_mips < base_size * 1.5

    def test_byte_size_with_layers(self) -> None:
        """byte_size() accounts for array layers."""
        slot = TextureSlot(
            name="test",
            width=256,
            height=256,
            format=TextureFormat.RGBA8_UNORM,
            mip_levels=1,
            layer_count=6,  # Cubemap
        )
        single_layer = 256 * 256 * 4
        assert slot.byte_size() == single_layer * 6

    def test_max_mip_levels(self) -> None:
        """max_mip_levels() calculates correct chain length."""
        slot_256 = TextureSlot(name="test", width=256, height=256)
        assert slot_256.max_mip_levels() == 9  # 256 -> 1

        slot_1024 = TextureSlot(name="test", width=1024, height=1024)
        assert slot_1024.max_mip_levels() == 11  # 1024 -> 1

        slot_rect = TextureSlot(name="test", width=512, height=256)
        assert slot_rect.max_mip_levels() == 10  # max(512, 256) = 512 -> 1

    def test_is_power_of_two(self) -> None:
        """is_power_of_two() correctly identifies POT dimensions."""
        pot_slot = TextureSlot(name="test", width=256, height=512)
        assert pot_slot.is_power_of_two()

        npot_slot = TextureSlot(name="test", width=300, height=200)
        assert not npot_slot.is_power_of_two()

        mixed_slot = TextureSlot(name="test", width=256, height=300)
        assert not mixed_slot.is_power_of_two()

    def test_aspect_ratio(self) -> None:
        """aspect_ratio() calculates correct ratio."""
        square = TextureSlot(name="test", width=256, height=256)
        assert square.aspect_ratio() == 1.0

        wide = TextureSlot(name="test", width=1920, height=1080)
        assert abs(wide.aspect_ratio() - (1920 / 1080)) < 0.001

        tall = TextureSlot(name="test", width=100, height=200)
        assert tall.aspect_ratio() == 0.5

    def test_hash_and_equality(self) -> None:
        """TextureSlot is hashable and comparable."""
        slot1 = TextureSlot(name="test", width=256, height=256)
        slot1.index = 5
        slot2 = TextureSlot(name="test", width=256, height=256)
        slot2.index = 5
        slot3 = TextureSlot(name="other", width=256, height=256)
        slot3.index = 5

        assert slot1 == slot2
        assert slot1 != slot3
        assert hash(slot1) == hash(slot2)
        assert hash(slot1) != hash(slot3)


# =============================================================================
# BindlessCapabilities Tests
# =============================================================================


class TestBindlessCapabilities:
    """Test capability detection."""

    def test_default_is_fallback(self) -> None:
        """Default capabilities use fallback mode."""
        caps = BindlessCapabilities()
        assert caps.fallback_mode
        assert not caps.supports_binding_array
        assert caps.effective_max_bindless == 0

    def test_detect_without_limits(self) -> None:
        """detect() without limits returns fallback mode."""
        caps = BindlessCapabilities.detect(None)
        assert caps.fallback_mode
        assert caps.max_textures_per_shader_stage == 16

    def test_detect_with_low_limits(self) -> None:
        """detect() with low limits enables fallback mode."""
        limits = {"maxTexturesPerShaderStage": 8}
        caps = BindlessCapabilities.detect(limits)
        assert caps.fallback_mode
        assert not caps.supports_binding_array

    def test_detect_with_bindless_support(self) -> None:
        """detect() with high limits enables bindless mode."""
        limits = {
            "maxTexturesPerShaderStage": 1024,
            "maxSamplersPerShaderStage": 64,
        }
        caps = BindlessCapabilities.detect(limits)
        assert not caps.fallback_mode
        assert caps.supports_binding_array
        assert caps.effective_max_bindless == 1024

    def test_effective_max_capped(self) -> None:
        """effective_max_bindless is capped at MAX_BINDLESS_TEXTURES."""
        limits = {"maxTexturesPerShaderStage": 10000}
        caps = BindlessCapabilities.detect(limits)
        assert caps.effective_max_bindless == MAX_BINDLESS_TEXTURES

    def test_is_bindless_viable(self) -> None:
        """is_bindless_viable() checks requirements correctly."""
        caps_bindless = BindlessCapabilities.detect(
            {"maxTexturesPerShaderStage": 2048}
        )
        assert caps_bindless.is_bindless_viable(100)
        assert caps_bindless.is_bindless_viable(2048)
        assert not caps_bindless.is_bindless_viable(3000)

        caps_fallback = BindlessCapabilities()
        assert not caps_fallback.is_bindless_viable(1)


# =============================================================================
# BindlessTextureArray Tests - Basic Operations
# =============================================================================


class TestBindlessTextureArrayBasic:
    """Test basic BindlessTextureArray operations."""

    def test_empty_array(self) -> None:
        """Empty array has correct initial state."""
        array = BindlessTextureArray(max_textures=100)
        assert array.count == 0
        assert array.is_empty
        assert not array.is_full
        assert array.dirty_count == 0
        assert array.max_textures == 100

    def test_register_texture(self) -> None:
        """register() assigns index and marks dirty."""
        array = BindlessTextureArray()
        slot = TextureSlot(name="test", width=256, height=256)

        index = array.register(slot)

        assert index == 0
        assert array.count == 1
        assert not array.is_empty
        assert array.dirty_count == 1
        assert slot.index == 0
        assert slot.valid

    def test_register_multiple_textures(self) -> None:
        """Multiple registrations get sequential indices."""
        array = BindlessTextureArray()

        for i in range(5):
            slot = TextureSlot(name=f"tex_{i}", width=64, height=64)
            index = array.register(slot)
            assert index == i

        assert array.count == 5

    def test_register_same_name_replaces(self) -> None:
        """Registering same name updates existing slot."""
        array = BindlessTextureArray()
        slot1 = TextureSlot(name="test", width=128, height=128)
        slot2 = TextureSlot(name="test", width=256, height=256)

        idx1 = array.register(slot1)
        idx2 = array.register(slot2)

        assert idx1 == idx2
        assert array.count == 1

        retrieved = array.get(idx1)
        assert retrieved is not None
        assert retrieved.width == 256

    def test_register_full_array_raises(self) -> None:
        """Registering to full array raises RuntimeError."""
        array = BindlessTextureArray(max_textures=3)

        for i in range(3):
            array.register(TextureSlot(name=f"tex_{i}", width=64, height=64))

        assert array.is_full

        with pytest.raises(RuntimeError, match="full"):
            array.register(TextureSlot(name="overflow", width=64, height=64))


# =============================================================================
# BindlessTextureArray Tests - Removal and Free-list
# =============================================================================


class TestBindlessTextureArrayFreeList:
    """Test free-list allocation and removal."""

    def test_remove_by_index(self) -> None:
        """remove() returns slot to free-list."""
        array = BindlessTextureArray()
        idx = array.register(TextureSlot(name="test", width=64, height=64))

        assert array.remove(idx)
        assert array.count == 0
        assert not array.contains(idx)

    def test_remove_nonexistent_returns_false(self) -> None:
        """remove() returns False for unknown index."""
        array = BindlessTextureArray()
        assert not array.remove(999)

    def test_remove_by_name(self) -> None:
        """remove_by_name() finds and removes texture."""
        array = BindlessTextureArray()
        array.register(TextureSlot(name="test", width=64, height=64))

        assert array.remove_by_name("test")
        assert array.count == 0
        assert not array.contains_name("test")

    def test_remove_by_name_nonexistent_returns_false(self) -> None:
        """remove_by_name() returns False for unknown name."""
        array = BindlessTextureArray()
        assert not array.remove_by_name("nonexistent")

    def test_free_list_reuse(self) -> None:
        """Removed slots are reused for new registrations."""
        array = BindlessTextureArray()

        # Register 3 textures
        idx0 = array.register(TextureSlot(name="tex0", width=64, height=64))
        idx1 = array.register(TextureSlot(name="tex1", width=64, height=64))
        idx2 = array.register(TextureSlot(name="tex2", width=64, height=64))

        # Remove middle texture
        array.remove(idx1)
        assert array.count == 2

        # New texture should reuse idx1
        new_idx = array.register(TextureSlot(name="tex_new", width=128, height=128))
        assert new_idx == idx1

    def test_free_list_lifo_order(self) -> None:
        """Free-list uses LIFO ordering."""
        array = BindlessTextureArray()

        indices = [
            array.register(TextureSlot(name=f"tex{i}", width=64, height=64))
            for i in range(5)
        ]

        # Remove in order: 2, 0, 4
        array.remove(indices[2])
        array.remove(indices[0])
        array.remove(indices[4])

        # LIFO: 4, 0, 2
        reused1 = array.register(TextureSlot(name="new1", width=64, height=64))
        assert reused1 == indices[4]

        reused2 = array.register(TextureSlot(name="new2", width=64, height=64))
        assert reused2 == indices[0]

        reused3 = array.register(TextureSlot(name="new3", width=64, height=64))
        assert reused3 == indices[2]


# =============================================================================
# BindlessTextureArray Tests - Access Methods
# =============================================================================


class TestBindlessTextureArrayAccess:
    """Test texture access methods."""

    def test_get_by_index(self) -> None:
        """get() retrieves texture by index."""
        array = BindlessTextureArray()
        slot = TextureSlot(name="test", width=256, height=256)
        idx = array.register(slot)

        retrieved = array.get(idx)
        assert retrieved is not None
        assert retrieved.name == "test"
        assert retrieved.width == 256

    def test_get_nonexistent_returns_none(self) -> None:
        """get() returns None for unknown index."""
        array = BindlessTextureArray()
        assert array.get(999) is None

    def test_get_by_name(self) -> None:
        """get_by_name() retrieves texture by name."""
        array = BindlessTextureArray()
        array.register(TextureSlot(name="brick", width=512, height=512))

        slot = array.get_by_name("brick")
        assert slot is not None
        assert slot.width == 512

    def test_get_index_by_name(self) -> None:
        """get_index() returns index for name."""
        array = BindlessTextureArray()
        idx = array.register(TextureSlot(name="metal", width=128, height=128))

        assert array.get_index("metal") == idx
        assert array.get_index("unknown") == INVALID_TEXTURE_INDEX

    def test_contains(self) -> None:
        """contains() checks index existence."""
        array = BindlessTextureArray()
        idx = array.register(TextureSlot(name="test", width=64, height=64))

        assert array.contains(idx)
        assert not array.contains(999)

    def test_contains_name(self) -> None:
        """contains_name() checks name existence."""
        array = BindlessTextureArray()
        array.register(TextureSlot(name="brick", width=64, height=64))

        assert array.contains_name("brick")
        assert not array.contains_name("stone")

    def test_iter_slots(self) -> None:
        """iter_slots() iterates all registered slots."""
        array = BindlessTextureArray()
        names = {"a", "b", "c"}
        for name in names:
            array.register(TextureSlot(name=name, width=64, height=64))

        found_names = {slot.name for slot in array.iter_slots()}
        assert found_names == names

    def test_iter_indices(self) -> None:
        """iter_indices() iterates all registered indices."""
        array = BindlessTextureArray()
        expected_indices = set()
        for i in range(5):
            idx = array.register(TextureSlot(name=f"tex{i}", width=64, height=64))
            expected_indices.add(idx)

        found_indices = set(array.iter_indices())
        assert found_indices == expected_indices


# =============================================================================
# BindlessTextureArray Tests - Dirty Tracking
# =============================================================================


class TestBindlessTextureArrayDirty:
    """Test dirty tracking functionality."""

    def test_register_marks_dirty(self) -> None:
        """Registering a texture marks it dirty."""
        array = BindlessTextureArray()
        idx = array.register(TextureSlot(name="test", width=64, height=64))

        assert array.any_dirty()
        assert idx in array.get_dirty_indices()

    def test_remove_marks_dirty(self) -> None:
        """Removing a texture marks the slot dirty."""
        array = BindlessTextureArray()
        idx = array.register(TextureSlot(name="test", width=64, height=64))
        array.clear_dirty()

        array.remove(idx)
        assert array.any_dirty()
        assert idx in array.get_dirty_indices()

    def test_mark_dirty(self) -> None:
        """mark_dirty() adds index to dirty set."""
        array = BindlessTextureArray()
        idx = array.register(TextureSlot(name="test", width=64, height=64))
        array.clear_dirty()

        array.mark_dirty(idx)
        assert array.any_dirty()
        assert idx in array.get_dirty_indices()

    def test_mark_dirty_nonexistent_ignored(self) -> None:
        """mark_dirty() ignores unknown indices."""
        array = BindlessTextureArray()
        array.mark_dirty(999)
        assert not array.any_dirty()

    def test_clear_dirty(self) -> None:
        """clear_dirty() returns and clears dirty set."""
        array = BindlessTextureArray()
        idx = array.register(TextureSlot(name="test", width=64, height=64))

        dirty = array.clear_dirty()
        assert idx in dirty
        assert not array.any_dirty()
        assert len(array.get_dirty_indices()) == 0


# =============================================================================
# BindlessTextureArray Tests - Statistics
# =============================================================================


class TestBindlessTextureArrayStats:
    """Test statistics gathering."""

    def test_empty_stats(self) -> None:
        """Stats for empty array."""
        array = BindlessTextureArray(max_textures=100)
        stats = array.get_stats()

        assert stats.total_slots == 100
        assert stats.used_slots == 0
        assert stats.free_slots == 0
        assert stats.total_memory_bytes == 0
        assert stats.utilization() == 0.0

    def test_stats_with_textures(self) -> None:
        """Stats reflect registered textures."""
        array = BindlessTextureArray(max_textures=100)

        # Register textures with different formats
        array.register(TextureSlot(
            name="rgba",
            width=256,
            height=256,
            format=TextureFormat.RGBA8_UNORM,
        ))
        array.register(TextureSlot(
            name="hdr",
            width=128,
            height=128,
            format=TextureFormat.RGBA16_FLOAT,
        ))

        stats = array.get_stats()

        assert stats.used_slots == 2
        assert stats.total_memory_bytes > 0
        assert stats.utilization() == 2.0  # 2/100 * 100 = 2%
        assert TextureFormat.RGBA8_UNORM in stats.format_histogram
        assert TextureFormat.RGBA16_FLOAT in stats.format_histogram

    def test_stats_memory_calculation(self) -> None:
        """Stats correctly sum memory usage."""
        array = BindlessTextureArray()

        # 256x256 RGBA8 = 256KB
        array.register(TextureSlot(
            name="tex1",
            width=256,
            height=256,
            format=TextureFormat.RGBA8_UNORM,
            mip_levels=1,
        ))
        # 128x128 RGBA8 = 64KB
        array.register(TextureSlot(
            name="tex2",
            width=128,
            height=128,
            format=TextureFormat.RGBA8_UNORM,
            mip_levels=1,
        ))

        stats = array.get_stats()
        expected_memory = (256 * 256 * 4) + (128 * 128 * 4)
        assert stats.total_memory_bytes == expected_memory


# =============================================================================
# BindlessTextureArray Tests - WGSL Generation
# =============================================================================


class TestBindlessTextureArrayWGSL:
    """Test WGSL code generation."""

    def test_generate_declarations_bindless(self) -> None:
        """WGSL declarations for bindless mode."""
        caps = BindlessCapabilities.detect({"maxTexturesPerShaderStage": 1024})
        array = BindlessTextureArray(capabilities=caps)

        wgsl = array.generate_wgsl_declarations()

        assert "binding_array<texture_2d<f32>" in wgsl
        assert "binding_array<sampler" in wgsl
        assert "@group(1)" in wgsl
        assert "@binding(0)" in wgsl
        assert "@binding(1)" in wgsl

    def test_generate_declarations_custom_group(self) -> None:
        """WGSL declarations with custom binding group."""
        caps = BindlessCapabilities.detect({"maxTexturesPerShaderStage": 1024})
        array = BindlessTextureArray(capabilities=caps)

        wgsl = array.generate_wgsl_declarations(group=2, texture_binding=5)

        assert "@group(2)" in wgsl
        assert "@binding(5)" in wgsl

    def test_generate_declarations_fallback(self) -> None:
        """WGSL declarations for fallback mode."""
        array = BindlessTextureArray()  # Default fallback mode
        array.register(TextureSlot(name="brick", width=256, height=256))
        array.register(TextureSlot(name="stone", width=128, height=128))

        wgsl = array.generate_wgsl_declarations()

        assert "brick_tex" in wgsl
        assert "brick_sampler" in wgsl
        assert "stone_tex" in wgsl
        assert "Fallback" in wgsl

    def test_generate_sample_function_bindless(self) -> None:
        """WGSL sample function for bindless mode."""
        caps = BindlessCapabilities.detect({"maxTexturesPerShaderStage": 1024})
        array = BindlessTextureArray(capabilities=caps)

        wgsl = array.generate_wgsl_sample_function()

        assert "fn sample_bindless" in wgsl
        assert "texture_index" in wgsl
        assert "sampler_index" in wgsl
        assert "textureSample" in wgsl
        assert "0xFFFFFFFFu" in wgsl  # Invalid index check

    def test_generate_sample_function_fallback(self) -> None:
        """WGSL sample function placeholder for fallback mode."""
        array = BindlessTextureArray()

        wgsl = array.generate_wgsl_sample_function()

        assert "Fallback" in wgsl or "traditional" in wgsl.lower()


# =============================================================================
# BindlessTextureArray Tests - Serialization
# =============================================================================


class TestBindlessTextureArraySerialization:
    """Test GPU buffer serialization."""

    def test_to_bytes_empty(self) -> None:
        """Empty array produces empty bytes."""
        array = BindlessTextureArray()
        assert array.to_bytes() == b""

    def test_to_bytes_single_entry(self) -> None:
        """Single texture serializes to 24 bytes."""
        array = BindlessTextureArray()
        array.register(TextureSlot(
            name="test",
            width=1024,
            height=768,
            format=TextureFormat.RGBA8_UNORM,
            mip_levels=10,
            layer_count=1,
        ))

        data = array.to_bytes()
        assert len(data) == 24  # TextureTableEntry size

        # Unpack and verify
        width, height, mip_levels, fmt, layer_count, flags = struct.unpack(
            "<IIIIII", data
        )
        assert width == 1024
        assert height == 768
        assert mip_levels == 10
        assert fmt == int(TextureFormat.RGBA8_UNORM)
        assert layer_count == 1
        assert flags == 1  # valid

    def test_to_bytes_multiple_entries(self) -> None:
        """Multiple textures serialize sequentially."""
        array = BindlessTextureArray()

        for i in range(3):
            array.register(TextureSlot(
                name=f"tex{i}",
                width=100 + i,
                height=200 + i,
                format=TextureFormat.RGBA8_UNORM,
                mip_levels=1,
            ))

        data = array.to_bytes()
        assert len(data) == 24 * 3

        # Verify each entry
        for i in range(3):
            offset = i * 24
            width, height = struct.unpack_from("<II", data, offset)
            assert width == 100 + i
            assert height == 200 + i

    def test_to_bytes_with_holes(self) -> None:
        """Removed slots produce zero entries."""
        array = BindlessTextureArray()

        idx0 = array.register(TextureSlot(name="tex0", width=100, height=100))
        idx1 = array.register(TextureSlot(name="tex1", width=200, height=200))
        idx2 = array.register(TextureSlot(name="tex2", width=300, height=300))

        # Remove middle entry
        array.remove(idx1)

        data = array.to_bytes()
        assert len(data) == 24 * 3

        # Entry 0: valid
        w0, h0 = struct.unpack_from("<II", data, 0)
        assert w0 == 100

        # Entry 1: hole (zeros)
        w1, h1, mip1, fmt1, layer1, flags1 = struct.unpack_from("<IIIIII", data, 24)
        assert w1 == 0 and h1 == 0 and flags1 == 0

        # Entry 2: valid
        w2, h2 = struct.unpack_from("<II", data, 48)
        assert w2 == 300


# =============================================================================
# BindlessTextureArray Tests - Clear and Reserve
# =============================================================================


class TestBindlessTextureArrayClearReserve:
    """Test clear and reserve operations."""

    def test_clear(self) -> None:
        """clear() removes all textures."""
        array = BindlessTextureArray()
        for i in range(5):
            array.register(TextureSlot(name=f"tex{i}", width=64, height=64))

        array.clear()

        assert array.count == 0
        assert array.is_empty
        assert array.dirty_count == 0

    def test_register_after_clear(self) -> None:
        """New registrations work after clear."""
        array = BindlessTextureArray()
        array.register(TextureSlot(name="old", width=64, height=64))
        array.clear()

        idx = array.register(TextureSlot(name="new", width=128, height=128))
        assert idx == 0  # Starts from 0 again
        assert array.count == 1

    def test_reserve(self) -> None:
        """reserve() is a no-op hint (Python dicts auto-grow)."""
        array = BindlessTextureArray()
        # Should not raise
        array.reserve(1000)
        assert array.is_empty


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_bindless_array_default(self) -> None:
        """create_bindless_array() with defaults."""
        array = create_bindless_array()
        assert array.max_textures == DEFAULT_BINDLESS_CAPACITY
        assert array.fallback_mode  # No adapter limits provided

    def test_create_bindless_array_with_limits(self) -> None:
        """create_bindless_array() with adapter limits."""
        limits = {"maxTexturesPerShaderStage": 2048}
        array = create_bindless_array(max_textures=1024, adapter_limits=limits)

        assert array.max_textures == 1024
        assert not array.fallback_mode

    def test_create_default_slots(self) -> None:
        """create_default_slots() returns standard fallback textures."""
        slots = create_default_slots()

        assert "white" in slots
        assert "black" in slots
        assert "normal" in slots
        assert "gray" in slots

        for name, slot in slots.items():
            assert slot.width == 1
            assert slot.height == 1
            assert slot.mip_levels == 1


# =============================================================================
# Content Hash Deduplication Tests
# =============================================================================


class TestContentHashDeduplication:
    """Test texture deduplication via content hash."""

    def test_duplicate_hash_returns_existing_index(self) -> None:
        """Registering texture with same hash returns existing slot."""
        array = BindlessTextureArray()

        slot1 = TextureSlot(
            name="tex1",
            width=256,
            height=256,
            content_hash="abc123",
        )
        slot2 = TextureSlot(
            name="tex2",
            width=256,
            height=256,
            content_hash="abc123",  # Same hash
        )

        idx1 = array.register(slot1)
        idx2 = array.register(slot2)

        assert idx1 == idx2
        assert array.count == 1

    def test_different_hash_allocates_new_slot(self) -> None:
        """Different hashes allocate separate slots."""
        array = BindlessTextureArray()

        slot1 = TextureSlot(name="tex1", width=256, height=256, content_hash="hash1")
        slot2 = TextureSlot(name="tex2", width=256, height=256, content_hash="hash2")

        idx1 = array.register(slot1)
        idx2 = array.register(slot2)

        assert idx1 != idx2
        assert array.count == 2


# =============================================================================
# Sampler Registration Tests
# =============================================================================


class TestSamplerRegistration:
    """Test sampler deduplication."""

    def test_identical_samplers_deduplicated(self) -> None:
        """Identical sampler configs share the same index."""
        array = BindlessTextureArray()

        # Both use default sampler config
        slot1 = TextureSlot(name="tex1", width=64, height=64)
        slot2 = TextureSlot(name="tex2", width=64, height=64)

        array.register(slot1)
        array.register(slot2)

        # Internal sampler cache should have only one entry
        assert len(array._sampler_cache) == 1

    def test_different_samplers_separate_indices(self) -> None:
        """Different sampler configs get different indices."""
        array = BindlessTextureArray()

        slot1 = TextureSlot(
            name="tex1",
            width=64,
            height=64,
            sampler=SamplerConfig.linear_repeat(),
        )
        slot2 = TextureSlot(
            name="tex2",
            width=64,
            height=64,
            sampler=SamplerConfig.nearest_repeat(),
        )

        array.register(slot1)
        array.register(slot2)

        assert len(array._sampler_cache) == 2


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_max_textures_clamped_to_hardware_limit(self) -> None:
        """max_textures is clamped to MAX_BINDLESS_TEXTURES."""
        array = BindlessTextureArray(max_textures=10000)
        assert array.max_textures == MAX_BINDLESS_TEXTURES

    def test_fallback_callback_invoked(self) -> None:
        """on_fallback callback is called when fallback mode activated."""
        messages: List[str] = []

        def callback(msg: str) -> None:
            messages.append(msg)

        array = BindlessTextureArray(on_fallback=callback)

        assert array.fallback_mode
        assert len(messages) == 1
        assert "fallback" in messages[0].lower()

    def test_zero_dimension_texture(self) -> None:
        """Texture with zero dimensions still registers."""
        array = BindlessTextureArray()
        slot = TextureSlot(name="zero", width=0, height=0)

        # Should not raise
        idx = array.register(slot)
        assert idx == 0
        assert slot.byte_size() == 0
        assert slot.aspect_ratio() == 1.0  # Safe division

    def test_very_large_texture(self) -> None:
        """Large texture dimensions handled correctly."""
        array = BindlessTextureArray()
        slot = TextureSlot(
            name="huge",
            width=16384,
            height=16384,
            format=TextureFormat.RGBA32_FLOAT,
            mip_levels=15,
        )

        idx = array.register(slot)
        assert idx == 0

        # 16K x 16K x 16 bytes = 4 GB base (without mipmaps)
        base_size = 16384 * 16384 * 16
        assert slot.byte_size(include_mipmaps=False) == base_size

    def test_register_after_full_and_remove(self) -> None:
        """Can register after removing from full array."""
        array = BindlessTextureArray(max_textures=2)

        idx0 = array.register(TextureSlot(name="tex0", width=64, height=64))
        idx1 = array.register(TextureSlot(name="tex1", width=64, height=64))

        assert array.is_full

        array.remove(idx0)
        assert not array.is_full

        # Should reuse idx0
        new_idx = array.register(TextureSlot(name="tex_new", width=128, height=128))
        assert new_idx == idx0


# =============================================================================
# Integration Tests
# =============================================================================


class TestBindlessIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self) -> None:
        """Complete workflow: create, register, sample, remove."""
        # 1. Detect capabilities
        caps = BindlessCapabilities.detect({"maxTexturesPerShaderStage": 4096})
        assert caps.supports_binding_array

        # 2. Create array
        array = BindlessTextureArray(max_textures=1024, capabilities=caps)
        assert not array.fallback_mode

        # 3. Register textures
        albedo_idx = array.register(TextureSlot(
            name="brick_albedo",
            width=1024,
            height=1024,
            format=TextureFormat.RGBA8_UNORM_SRGB,
            mip_levels=10,
        ))
        normal_idx = array.register(TextureSlot(
            name="brick_normal",
            width=1024,
            height=1024,
            format=TextureFormat.RGBA8_UNORM,
            mip_levels=10,
        ))

        assert albedo_idx == 0
        assert normal_idx == 1

        # 4. Generate WGSL
        wgsl_decl = array.generate_wgsl_declarations()
        wgsl_func = array.generate_wgsl_sample_function()

        assert "binding_array" in wgsl_decl
        assert "sample_bindless" in wgsl_func

        # 5. Serialize for GPU
        data = array.to_bytes()
        assert len(data) == 24 * 2

        # 6. Get stats
        stats = array.get_stats()
        assert stats.used_slots == 2
        assert stats.total_memory_bytes > 0

        # 7. Remove texture
        array.remove(albedo_idx)
        assert array.count == 1

        # 8. Re-register (reuses slot)
        new_idx = array.register(TextureSlot(
            name="stone_albedo",
            width=512,
            height=512,
            format=TextureFormat.RGBA8_UNORM_SRGB,
            mip_levels=9,
        ))
        assert new_idx == albedo_idx

    def test_fallback_workflow(self) -> None:
        """Workflow in fallback mode (no bindless support)."""
        # No adapter limits = fallback mode
        array = BindlessTextureArray()
        assert array.fallback_mode

        # Register textures
        array.register(TextureSlot(name="diffuse", width=256, height=256))
        array.register(TextureSlot(name="specular", width=256, height=256))

        # WGSL uses traditional bindings
        wgsl = array.generate_wgsl_declarations()
        assert "diffuse_tex" in wgsl
        assert "specular_tex" in wgsl
        assert "binding_array" not in wgsl
