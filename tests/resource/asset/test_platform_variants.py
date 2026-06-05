"""Tests for platform-specific asset variants."""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from engine.resource.asset.content_hash import ContentHash
from engine.resource.asset.platform_variants import (
    DEFAULT_PLATFORM_FORMATS,
    PLATFORM_SUPPORTED_FORMATS,
    FormatConverter,
    GeneratedVariant,
    Platform,
    SourceAsset,
    TextureFormat,
    TextureFormatConverter,
    VariantCache,
    VariantGenerator,
    VariantKey,
    VariantRule,
    VariantRuleSet,
    VariantStorage,
    get_optimal_format,
    is_format_supported,
)


# ==============================================================================
# Platform Enum Tests
# ==============================================================================


class TestPlatform:
    """Tests for Platform enum."""

    def test_all_platforms_exist(self) -> None:
        platforms = list(Platform)
        assert len(platforms) == 6
        assert Platform.WINDOWS in platforms
        assert Platform.LINUX in platforms
        assert Platform.MACOS in platforms
        assert Platform.IOS in platforms
        assert Platform.ANDROID in platforms
        assert Platform.WEBGL in platforms

    def test_from_string_valid(self) -> None:
        assert Platform.from_string("windows") == Platform.WINDOWS
        assert Platform.from_string("WINDOWS") == Platform.WINDOWS
        assert Platform.from_string("Windows") == Platform.WINDOWS
        assert Platform.from_string("linux") == Platform.LINUX
        assert Platform.from_string("ios") == Platform.IOS

    def test_from_string_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown platform"):
            Platform.from_string("playstation")

    def test_is_desktop(self) -> None:
        assert Platform.WINDOWS.is_desktop is True
        assert Platform.LINUX.is_desktop is True
        assert Platform.MACOS.is_desktop is True
        assert Platform.IOS.is_desktop is False
        assert Platform.ANDROID.is_desktop is False
        assert Platform.WEBGL.is_desktop is False

    def test_is_mobile(self) -> None:
        assert Platform.WINDOWS.is_mobile is False
        assert Platform.IOS.is_mobile is True
        assert Platform.ANDROID.is_mobile is True

    def test_is_apple(self) -> None:
        assert Platform.MACOS.is_apple is True
        assert Platform.IOS.is_apple is True
        assert Platform.WINDOWS.is_apple is False
        assert Platform.ANDROID.is_apple is False

    def test_str(self) -> None:
        assert str(Platform.WINDOWS) == "windows"
        assert str(Platform.IOS) == "ios"


# ==============================================================================
# TextureFormat Enum Tests
# ==============================================================================


class TestTextureFormat:
    """Tests for TextureFormat enum."""

    def test_all_formats_exist(self) -> None:
        formats = list(TextureFormat)
        assert len(formats) >= 15

    def test_from_string_valid(self) -> None:
        assert TextureFormat.from_string("bc7") == TextureFormat.BC7
        assert TextureFormat.from_string("BC7") == TextureFormat.BC7
        assert TextureFormat.from_string("astc-4x4") == TextureFormat.ASTC_4x4
        assert TextureFormat.from_string("ASTC_4x4") == TextureFormat.ASTC_4x4
        assert TextureFormat.from_string("etc2_rgba") == TextureFormat.ETC2_RGBA

    def test_from_string_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown texture format"):
            TextureFormat.from_string("jpeg")

    def test_bits_per_pixel(self) -> None:
        assert TextureFormat.RGBA8.bits_per_pixel == 32.0
        assert TextureFormat.BC7.bits_per_pixel == 8.0
        assert TextureFormat.BC1.bits_per_pixel == 4.0
        assert TextureFormat.ASTC_4x4.bits_per_pixel == 8.0
        assert TextureFormat.ASTC_8x8.bits_per_pixel == 2.0
        assert TextureFormat.ETC2_RGBA.bits_per_pixel == 8.0

    def test_has_alpha(self) -> None:
        assert TextureFormat.RGBA8.has_alpha is True
        assert TextureFormat.BC7.has_alpha is True
        assert TextureFormat.BC3.has_alpha is True
        assert TextureFormat.BC1.has_alpha is False
        assert TextureFormat.BC4.has_alpha is False
        assert TextureFormat.ETC1.has_alpha is False
        assert TextureFormat.ETC2_RGB.has_alpha is False
        assert TextureFormat.ETC2_RGBA.has_alpha is True

    def test_is_compressed(self) -> None:
        assert TextureFormat.RGBA8.is_compressed is False
        assert TextureFormat.RGBA16F.is_compressed is False
        assert TextureFormat.BC7.is_compressed is True
        assert TextureFormat.ASTC_4x4.is_compressed is True
        assert TextureFormat.ETC2_RGBA.is_compressed is True

    def test_is_hdr(self) -> None:
        assert TextureFormat.RGBA8.is_hdr is False
        assert TextureFormat.RGBA16F.is_hdr is True
        assert TextureFormat.RGBA32F.is_hdr is True
        assert TextureFormat.BC6H.is_hdr is True
        assert TextureFormat.BC7.is_hdr is False

    def test_str(self) -> None:
        assert str(TextureFormat.BC7) == "bc7"
        assert str(TextureFormat.ASTC_4x4) == "astc_4x4"


# ==============================================================================
# Default Format Mappings Tests
# ==============================================================================


class TestDefaultFormatMappings:
    """Tests for default platform format mappings."""

    def test_all_platforms_have_defaults(self) -> None:
        for platform in Platform:
            assert platform in DEFAULT_PLATFORM_FORMATS

    def test_desktop_uses_bc7(self) -> None:
        assert DEFAULT_PLATFORM_FORMATS[Platform.WINDOWS] == TextureFormat.BC7
        assert DEFAULT_PLATFORM_FORMATS[Platform.LINUX] == TextureFormat.BC7

    def test_apple_uses_astc(self) -> None:
        assert DEFAULT_PLATFORM_FORMATS[Platform.MACOS] == TextureFormat.ASTC_4x4
        assert DEFAULT_PLATFORM_FORMATS[Platform.IOS] == TextureFormat.ASTC_4x4

    def test_android_uses_etc2(self) -> None:
        assert DEFAULT_PLATFORM_FORMATS[Platform.ANDROID] == TextureFormat.ETC2_RGBA

    def test_webgl_uses_etc2(self) -> None:
        assert DEFAULT_PLATFORM_FORMATS[Platform.WEBGL] == TextureFormat.ETC2_RGBA


class TestPlatformSupportedFormats:
    """Tests for platform format support matrix."""

    def test_all_platforms_have_support_matrix(self) -> None:
        for platform in Platform:
            assert platform in PLATFORM_SUPPORTED_FORMATS

    def test_all_platforms_support_rgba8(self) -> None:
        for platform in Platform:
            assert TextureFormat.RGBA8 in PLATFORM_SUPPORTED_FORMATS[platform]

    def test_desktop_supports_bc_formats(self) -> None:
        for platform in [Platform.WINDOWS, Platform.LINUX]:
            supported = PLATFORM_SUPPORTED_FORMATS[platform]
            assert TextureFormat.BC1 in supported
            assert TextureFormat.BC3 in supported
            assert TextureFormat.BC7 in supported

    def test_ios_supports_astc(self) -> None:
        supported = PLATFORM_SUPPORTED_FORMATS[Platform.IOS]
        assert TextureFormat.ASTC_4x4 in supported
        assert TextureFormat.ASTC_8x8 in supported

    def test_android_supports_etc2(self) -> None:
        supported = PLATFORM_SUPPORTED_FORMATS[Platform.ANDROID]
        assert TextureFormat.ETC2_RGB in supported
        assert TextureFormat.ETC2_RGBA in supported


# ==============================================================================
# Helper Function Tests
# ==============================================================================


class TestGetOptimalFormat:
    """Tests for get_optimal_format helper."""

    def test_windows_with_alpha(self) -> None:
        fmt = get_optimal_format(Platform.WINDOWS, has_alpha=True)
        assert fmt == TextureFormat.BC7

    def test_windows_without_alpha_high_quality(self) -> None:
        fmt = get_optimal_format(Platform.WINDOWS, has_alpha=False, quality_level=2)
        assert fmt == TextureFormat.BC7

    def test_windows_without_alpha_low_quality(self) -> None:
        fmt = get_optimal_format(Platform.WINDOWS, has_alpha=False, quality_level=1)
        assert fmt == TextureFormat.BC1

    def test_ios_quality_levels(self) -> None:
        assert get_optimal_format(Platform.IOS, quality_level=0) == TextureFormat.ASTC_8x8
        assert get_optimal_format(Platform.IOS, quality_level=1) == TextureFormat.ASTC_6x6
        assert get_optimal_format(Platform.IOS, quality_level=2) == TextureFormat.ASTC_5x5
        assert get_optimal_format(Platform.IOS, quality_level=3) == TextureFormat.ASTC_4x4

    def test_android_with_alpha(self) -> None:
        fmt = get_optimal_format(Platform.ANDROID, has_alpha=True)
        assert fmt == TextureFormat.ETC2_RGBA

    def test_android_without_alpha(self) -> None:
        fmt = get_optimal_format(Platform.ANDROID, has_alpha=False)
        assert fmt == TextureFormat.ETC2_RGB

    def test_hdr_on_desktop(self) -> None:
        fmt = get_optimal_format(Platform.WINDOWS, is_hdr=True)
        assert fmt == TextureFormat.BC6H

    def test_hdr_on_mobile(self) -> None:
        fmt = get_optimal_format(Platform.IOS, is_hdr=True)
        assert fmt == TextureFormat.RGBA16F


class TestIsFormatSupported:
    """Tests for is_format_supported helper."""

    def test_supported_format(self) -> None:
        assert is_format_supported(Platform.WINDOWS, TextureFormat.BC7) is True

    def test_unsupported_format(self) -> None:
        assert is_format_supported(Platform.IOS, TextureFormat.BC7) is False

    def test_universal_rgba8(self) -> None:
        for platform in Platform:
            assert is_format_supported(platform, TextureFormat.RGBA8) is True


# ==============================================================================
# VariantKey Tests
# ==============================================================================


class TestVariantKey:
    """Tests for VariantKey value object."""

    def test_create_valid_key(self) -> None:
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(
            source_hash=source_hash,
            platform=Platform.WINDOWS,
            format=TextureFormat.BC7,
        )
        assert key.source_hash == source_hash
        assert key.platform == Platform.WINDOWS
        assert key.format == TextureFormat.BC7

    def test_null_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be null"):
            VariantKey(
                source_hash=ContentHash.null(),
                platform=Platform.WINDOWS,
                format=TextureFormat.BC7,
            )

    def test_cache_key(self) -> None:
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(
            source_hash=source_hash,
            platform=Platform.WINDOWS,
            format=TextureFormat.BC7,
        )
        cache_key = key.cache_key
        assert source_hash.hex in cache_key
        assert "WINDOWS" in cache_key
        assert "BC7" in cache_key

    def test_compute_variant_hash(self) -> None:
        source_hash = ContentHash.from_content(b"test")
        key1 = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        key2 = VariantKey(source_hash, Platform.LINUX, TextureFormat.BC7)
        key3 = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC3)

        # Same key produces same hash
        assert key1.compute_variant_hash() == key1.compute_variant_hash()
        # Different platforms produce different hashes
        assert key1.compute_variant_hash() != key2.compute_variant_hash()
        # Different formats produce different hashes
        assert key1.compute_variant_hash() != key3.compute_variant_hash()

    def test_str(self) -> None:
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        s = str(key)
        assert source_hash.short_hex in s
        assert "WINDOWS" in s
        assert "BC7" in s

    def test_equality(self) -> None:
        source_hash = ContentHash.from_content(b"test")
        key1 = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        key2 = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        key3 = VariantKey(source_hash, Platform.LINUX, TextureFormat.BC7)
        assert key1 == key2
        assert key1 != key3


# ==============================================================================
# VariantRule Tests
# ==============================================================================


class TestVariantRule:
    """Tests for VariantRule matching."""

    def test_default_rule_matches_all(self) -> None:
        rule = VariantRule(target_format=TextureFormat.RGBA8)
        assert rule.matches(Platform.WINDOWS) is True
        assert rule.matches(Platform.IOS) is True
        assert rule.matches(Platform.ANDROID) is True

    def test_platform_specific_rule(self) -> None:
        rule = VariantRule(
            platform=Platform.WINDOWS,
            target_format=TextureFormat.BC7,
        )
        assert rule.matches(Platform.WINDOWS) is True
        assert rule.matches(Platform.LINUX) is False

    def test_alpha_requirement_true(self) -> None:
        rule = VariantRule(
            target_format=TextureFormat.BC3,
            requires_alpha=True,
        )
        assert rule.matches(Platform.WINDOWS, has_alpha=True) is True
        assert rule.matches(Platform.WINDOWS, has_alpha=False) is False

    def test_alpha_requirement_false(self) -> None:
        rule = VariantRule(
            target_format=TextureFormat.BC1,
            requires_alpha=False,
        )
        assert rule.matches(Platform.WINDOWS, has_alpha=False) is True
        assert rule.matches(Platform.WINDOWS, has_alpha=True) is False

    def test_hdr_requirement(self) -> None:
        rule = VariantRule(
            target_format=TextureFormat.BC6H,
            requires_hdr=True,
        )
        assert rule.matches(Platform.WINDOWS, is_hdr=True) is True
        assert rule.matches(Platform.WINDOWS, is_hdr=False) is False

    def test_custom_predicate(self) -> None:
        rule = VariantRule(
            target_format=TextureFormat.BC7,
            predicate=lambda ctx: ctx.get("size", 0) > 1024,
        )
        assert rule.matches(Platform.WINDOWS, context={"size": 2048}) is True
        assert rule.matches(Platform.WINDOWS, context={"size": 512}) is False
        assert rule.matches(Platform.WINDOWS, context={}) is False

    def test_combined_conditions(self) -> None:
        rule = VariantRule(
            platform=Platform.WINDOWS,
            target_format=TextureFormat.BC6H,
            requires_hdr=True,
        )
        assert rule.matches(Platform.WINDOWS, is_hdr=True) is True
        assert rule.matches(Platform.WINDOWS, is_hdr=False) is False
        assert rule.matches(Platform.LINUX, is_hdr=True) is False


# ==============================================================================
# VariantRuleSet Tests
# ==============================================================================


class TestVariantRuleSet:
    """Tests for VariantRuleSet collection."""

    def test_empty_ruleset(self) -> None:
        ruleset = VariantRuleSet()
        assert len(ruleset) == 0

    def test_add_rule(self) -> None:
        ruleset = VariantRuleSet()
        rule = VariantRule(platform=Platform.WINDOWS, target_format=TextureFormat.BC7)
        ruleset.add_rule(rule)
        assert len(ruleset) == 1

    def test_remove_rule(self) -> None:
        rule = VariantRule(platform=Platform.WINDOWS, target_format=TextureFormat.BC7)
        ruleset = VariantRuleSet([rule])
        assert ruleset.remove_rule(rule) is True
        assert len(ruleset) == 0

    def test_remove_nonexistent_rule(self) -> None:
        ruleset = VariantRuleSet()
        rule = VariantRule(platform=Platform.WINDOWS, target_format=TextureFormat.BC7)
        assert ruleset.remove_rule(rule) is False

    def test_clear(self) -> None:
        rules = [
            VariantRule(platform=Platform.WINDOWS, target_format=TextureFormat.BC7),
            VariantRule(platform=Platform.IOS, target_format=TextureFormat.ASTC_4x4),
        ]
        ruleset = VariantRuleSet(rules)
        ruleset.clear()
        assert len(ruleset) == 0

    def test_priority_ordering(self) -> None:
        low_priority = VariantRule(
            target_format=TextureFormat.RGBA8,
            priority=1,
        )
        high_priority = VariantRule(
            target_format=TextureFormat.BC7,
            priority=10,
        )
        ruleset = VariantRuleSet([low_priority, high_priority])

        # High priority should match first
        fmt = ruleset.match(Platform.WINDOWS)
        assert fmt == TextureFormat.BC7

    def test_match_falls_back_to_default(self) -> None:
        ruleset = VariantRuleSet()
        fmt = ruleset.match(Platform.WINDOWS)
        assert fmt == DEFAULT_PLATFORM_FORMATS[Platform.WINDOWS]

    def test_default_rules(self) -> None:
        ruleset = VariantRuleSet.default_rules()
        assert len(ruleset) > 0

        # Test platform-specific defaults
        assert ruleset.match(Platform.WINDOWS) == TextureFormat.BC7
        assert ruleset.match(Platform.IOS) == TextureFormat.ASTC_4x4
        assert ruleset.match(Platform.ANDROID, has_alpha=True) == TextureFormat.ETC2_RGBA
        assert ruleset.match(Platform.ANDROID, has_alpha=False) == TextureFormat.ETC2_RGB

    def test_iteration(self) -> None:
        rules = [
            VariantRule(platform=Platform.WINDOWS, target_format=TextureFormat.BC7),
            VariantRule(platform=Platform.IOS, target_format=TextureFormat.ASTC_4x4),
        ]
        ruleset = VariantRuleSet(rules)
        iterated = list(ruleset)
        assert len(iterated) == 2


# ==============================================================================
# SourceAsset Tests
# ==============================================================================


class TestSourceAsset:
    """Tests for SourceAsset dataclass."""

    def test_create_directly(self) -> None:
        source_hash = ContentHash.from_content(b"test")
        asset = SourceAsset(
            path=Path("/test/image.png"),
            content_hash=source_hash,
            has_alpha=True,
            is_hdr=False,
            width=256,
            height=256,
        )
        assert asset.path == Path("/test/image.png")
        assert asset.content_hash == source_hash

    def test_from_path(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake png data")
            path = Path(f.name)

        try:
            asset = SourceAsset.from_path(path)
            assert asset.path == path
            assert not asset.content_hash.is_null()
        finally:
            path.unlink()

    def test_from_path_hdr_detection(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as f:
            f.write(b"fake hdr data")
            path = Path(f.name)

        try:
            asset = SourceAsset.from_path(path)
            assert asset.is_hdr is True
        finally:
            path.unlink()

    def test_from_path_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            SourceAsset.from_path("/nonexistent/file.png")


# ==============================================================================
# GeneratedVariant Tests
# ==============================================================================


class TestGeneratedVariant:
    """Tests for GeneratedVariant dataclass."""

    def test_create_variant(self) -> None:
        source_hash = ContentHash.from_content(b"source")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        data = b"compressed data"
        variant_hash = ContentHash.from_content(data)

        variant = GeneratedVariant(
            key=key,
            data=data,
            variant_hash=variant_hash,
            generation_time_ms=10.5,
        )

        assert variant.key == key
        assert variant.data == data
        assert variant.compressed_size == len(data)
        assert variant.generation_time_ms == 10.5

    def test_auto_compute_compressed_size(self) -> None:
        source_hash = ContentHash.from_content(b"source")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        data = b"x" * 1000
        variant_hash = ContentHash.from_content(data)

        variant = GeneratedVariant(
            key=key,
            data=data,
            variant_hash=variant_hash,
        )

        assert variant.compressed_size == 1000


# ==============================================================================
# TextureFormatConverter Tests
# ==============================================================================


class TestTextureFormatConverter:
    """Tests for TextureFormatConverter simulation."""

    def test_can_convert_same_format(self) -> None:
        converter = TextureFormatConverter()
        assert converter.can_convert(TextureFormat.BC7, TextureFormat.BC7) is True

    def test_can_convert_different_formats(self) -> None:
        converter = TextureFormatConverter()
        assert converter.can_convert(TextureFormat.RGBA8, TextureFormat.BC7) is True
        assert converter.can_convert(TextureFormat.BC7, TextureFormat.ASTC_4x4) is True

    def test_convert_same_format_returns_unchanged(self) -> None:
        converter = TextureFormatConverter()
        data = b"original data"
        result = converter.convert(
            data,
            TextureFormat.RGBA8,
            TextureFormat.RGBA8,
            256,
            256,
        )
        assert result == data

    def test_convert_produces_smaller_output(self) -> None:
        converter = TextureFormatConverter()
        data = b"x" * 10000  # Simulated RGBA8 data
        result = converter.convert(
            data,
            TextureFormat.RGBA8,
            TextureFormat.BC7,
            256,
            256,
        )
        # BC7 is 8bpp vs RGBA8 at 32bpp = 1/4 size
        assert len(result) < len(data)

    def test_convert_includes_header(self) -> None:
        converter = TextureFormatConverter()
        data = b"x" * 1000
        result = converter.convert(
            data,
            TextureFormat.RGBA8,
            TextureFormat.BC7,
            256,
            256,
        )
        # Check for magic header
        assert result[:4] == b"TXFM"

    def test_convert_deterministic(self) -> None:
        converter = TextureFormatConverter()
        data = b"test texture data"
        result1 = converter.convert(data, TextureFormat.RGBA8, TextureFormat.BC7, 64, 64)
        result2 = converter.convert(data, TextureFormat.RGBA8, TextureFormat.BC7, 64, 64)
        assert result1 == result2


# ==============================================================================
# VariantCache Tests
# ==============================================================================


class TestVariantCache:
    """Tests for VariantCache LRU cache."""

    def test_create_cache(self) -> None:
        cache = VariantCache(max_size=10)
        assert len(cache) == 0
        assert cache.max_size == 10

    def test_invalid_max_size(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            VariantCache(max_size=0)

    def test_put_and_get(self) -> None:
        cache = VariantCache()
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        variant = GeneratedVariant(
            key=key,
            data=b"data",
            variant_hash=ContentHash.from_content(b"data"),
        )

        cache.put(variant)
        assert len(cache) == 1

        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved.data == b"data"

    def test_get_missing_returns_none(self) -> None:
        cache = VariantCache()
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        assert cache.get(key) is None

    def test_lru_eviction(self) -> None:
        cache = VariantCache(max_size=2)

        def make_variant(i: int) -> GeneratedVariant:
            source_hash = ContentHash.from_content(f"test{i}".encode())
            key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
            return GeneratedVariant(
                key=key,
                data=f"data{i}".encode(),
                variant_hash=ContentHash.from_content(f"data{i}".encode()),
            )

        v1 = make_variant(1)
        v2 = make_variant(2)
        v3 = make_variant(3)

        cache.put(v1)
        cache.put(v2)
        assert len(cache) == 2

        # Adding v3 should evict v1 (least recently used)
        cache.put(v3)
        assert len(cache) == 2
        assert cache.get(v1.key) is None
        assert cache.get(v2.key) is not None
        assert cache.get(v3.key) is not None

    def test_access_updates_lru_order(self) -> None:
        cache = VariantCache(max_size=2)

        def make_variant(i: int) -> GeneratedVariant:
            source_hash = ContentHash.from_content(f"test{i}".encode())
            key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
            return GeneratedVariant(
                key=key,
                data=f"data{i}".encode(),
                variant_hash=ContentHash.from_content(f"data{i}".encode()),
            )

        v1 = make_variant(1)
        v2 = make_variant(2)
        v3 = make_variant(3)

        cache.put(v1)
        cache.put(v2)

        # Access v1 to make it recently used
        cache.get(v1.key)

        # Adding v3 should evict v2 now (not v1)
        cache.put(v3)
        assert cache.get(v1.key) is not None
        assert cache.get(v2.key) is None
        assert cache.get(v3.key) is not None

    def test_invalidate(self) -> None:
        cache = VariantCache()
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        variant = GeneratedVariant(
            key=key,
            data=b"data",
            variant_hash=ContentHash.from_content(b"data"),
        )

        cache.put(variant)
        assert cache.invalidate(key) is True
        assert cache.get(key) is None
        assert cache.invalidate(key) is False

    def test_invalidate_source(self) -> None:
        cache = VariantCache()
        source_hash = ContentHash.from_content(b"test")

        # Add variants for different platforms
        for platform in [Platform.WINDOWS, Platform.IOS, Platform.ANDROID]:
            key = VariantKey(
                source_hash,
                platform,
                DEFAULT_PLATFORM_FORMATS[platform],
            )
            variant = GeneratedVariant(
                key=key,
                data=f"data_{platform}".encode(),
                variant_hash=ContentHash.from_content(f"data_{platform}".encode()),
            )
            cache.put(variant)

        assert len(cache) == 3
        removed = cache.invalidate_source(source_hash)
        assert removed == 3
        assert len(cache) == 0

    def test_clear(self) -> None:
        cache = VariantCache()
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        variant = GeneratedVariant(
            key=key,
            data=b"data",
            variant_hash=ContentHash.from_content(b"data"),
        )

        cache.put(variant)
        cleared = cache.clear()
        assert cleared == 1
        assert len(cache) == 0

    def test_contains(self) -> None:
        cache = VariantCache()
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        variant = GeneratedVariant(
            key=key,
            data=b"data",
            variant_hash=ContentHash.from_content(b"data"),
        )

        assert key not in cache
        cache.put(variant)
        assert key in cache


# ==============================================================================
# VariantStorage Tests
# ==============================================================================


class TestVariantStorage:
    """Tests for VariantStorage content-addressed storage."""

    def test_store_and_get(self) -> None:
        storage = VariantStorage()
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        variant = GeneratedVariant(
            key=key,
            data=b"compressed data",
            variant_hash=ContentHash.from_content(b"compressed data"),
        )

        content_hash = storage.store(variant)
        assert not content_hash.is_null()

        retrieved = storage.get(key)
        assert retrieved == b"compressed data"

    def test_get_missing_returns_none(self) -> None:
        storage = VariantStorage()
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        assert storage.get(key) is None

    def test_get_by_hash(self) -> None:
        storage = VariantStorage()
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        variant = GeneratedVariant(
            key=key,
            data=b"compressed data",
            variant_hash=ContentHash.from_content(b"compressed data"),
        )

        content_hash = storage.store(variant)
        retrieved = storage.get_by_hash(content_hash)
        assert retrieved == b"compressed data"

    def test_contains(self) -> None:
        storage = VariantStorage()
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        variant = GeneratedVariant(
            key=key,
            data=b"data",
            variant_hash=ContentHash.from_content(b"data"),
        )

        assert storage.contains(key) is False
        storage.store(variant)
        assert storage.contains(key) is True

    def test_remove(self) -> None:
        storage = VariantStorage()
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        variant = GeneratedVariant(
            key=key,
            data=b"data",
            variant_hash=ContentHash.from_content(b"data"),
        )

        storage.store(variant)
        assert storage.remove(key) is True
        assert storage.get(key) is None
        assert storage.remove(key) is False

    def test_remove_for_source(self) -> None:
        storage = VariantStorage()
        source_hash = ContentHash.from_content(b"test")

        # Store variants for multiple platforms
        for platform in [Platform.WINDOWS, Platform.IOS]:
            key = VariantKey(
                source_hash,
                platform,
                DEFAULT_PLATFORM_FORMATS[platform],
            )
            variant = GeneratedVariant(
                key=key,
                data=f"data_{platform}".encode(),
                variant_hash=ContentHash.from_content(f"data_{platform}".encode()),
            )
            storage.store(variant)

        assert len(storage) == 2
        removed = storage.remove_for_source(source_hash)
        assert removed == 2
        assert len(storage) == 0

    def test_get_stats(self) -> None:
        storage = VariantStorage()
        source_hash = ContentHash.from_content(b"test")
        key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
        variant = GeneratedVariant(
            key=key,
            data=b"data",
            variant_hash=ContentHash.from_content(b"data"),
        )

        storage.store(variant)
        stats = storage.get_stats()
        assert stats["variant_count"] == 1


# ==============================================================================
# VariantGenerator Tests
# ==============================================================================


class TestVariantGenerator:
    """Tests for VariantGenerator orchestration."""

    @pytest.fixture
    def temp_source_file(self) -> Path:
        """Create a temporary source file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Write enough data to simulate a texture
            f.write(b"fake png data " * 100)
            return Path(f.name)

    def test_create_generator(self) -> None:
        gen = VariantGenerator()
        assert gen.lazy_generation is True
        assert len(gen.rules) > 0

    def test_select_format(self, temp_source_file: Path) -> None:
        gen = VariantGenerator()
        source = SourceAsset.from_path(temp_source_file)

        fmt = gen.select_format(Platform.WINDOWS, source)
        assert fmt == TextureFormat.BC7

        fmt = gen.select_format(Platform.IOS, source)
        assert fmt == TextureFormat.ASTC_4x4

        temp_source_file.unlink()

    def test_generate_variant(self, temp_source_file: Path) -> None:
        gen = VariantGenerator()
        source = SourceAsset.from_path(temp_source_file)

        variant = gen.generate(source, Platform.WINDOWS)

        assert variant.key.platform == Platform.WINDOWS
        assert variant.key.format == TextureFormat.BC7
        assert len(variant.data) > 0
        assert not variant.variant_hash.is_null()

        temp_source_file.unlink()

    def test_generate_with_explicit_format(self, temp_source_file: Path) -> None:
        gen = VariantGenerator()
        source = SourceAsset.from_path(temp_source_file)

        variant = gen.generate(
            source,
            Platform.WINDOWS,
            target_format=TextureFormat.BC3,
        )

        assert variant.key.format == TextureFormat.BC3

        temp_source_file.unlink()

    def test_generate_caches_result(self, temp_source_file: Path) -> None:
        gen = VariantGenerator()
        source = SourceAsset.from_path(temp_source_file)

        variant1 = gen.generate(source, Platform.WINDOWS)
        variant2 = gen.generate(source, Platform.WINDOWS)

        # Should be cache hit
        assert variant1.data == variant2.data
        assert len(gen.cache) == 1

        temp_source_file.unlink()

    def test_generate_force_regenerate(self, temp_source_file: Path) -> None:
        gen = VariantGenerator()
        source = SourceAsset.from_path(temp_source_file)

        variant1 = gen.generate(source, Platform.WINDOWS)
        variant2 = gen.generate(source, Platform.WINDOWS, force_regenerate=True)

        # Both should exist and be equivalent
        assert variant1.data == variant2.data

        temp_source_file.unlink()

    def test_generate_all_platforms(self, temp_source_file: Path) -> None:
        gen = VariantGenerator()
        source = SourceAsset.from_path(temp_source_file)

        results = gen.generate_all_platforms(source)

        assert len(results) == len(Platform)
        for platform in Platform:
            assert platform in results

        temp_source_file.unlink()

    def test_get_or_generate_lazy(self, temp_source_file: Path) -> None:
        gen = VariantGenerator(lazy_generation=True)
        source = SourceAsset.from_path(temp_source_file)

        # Should generate on demand
        variant = gen.get_or_generate(source, Platform.IOS)
        assert variant.key.format == TextureFormat.ASTC_4x4

        temp_source_file.unlink()

    def test_get_or_generate_not_lazy_raises(self, temp_source_file: Path) -> None:
        gen = VariantGenerator(lazy_generation=False)
        source = SourceAsset.from_path(temp_source_file)

        with pytest.raises(KeyError, match="lazy generation disabled"):
            gen.get_or_generate(source, Platform.IOS)

        temp_source_file.unlink()

    def test_invalidate(self, temp_source_file: Path) -> None:
        gen = VariantGenerator()
        source = SourceAsset.from_path(temp_source_file)

        gen.generate(source, Platform.WINDOWS)
        gen.generate(source, Platform.IOS)

        removed = gen.invalidate(source.content_hash)
        assert removed >= 2

        temp_source_file.unlink()

    def test_get_stats(self) -> None:
        gen = VariantGenerator()
        stats = gen.get_stats()

        assert "cache_size" in stats
        assert "cache_max_size" in stats
        assert "rule_count" in stats
        assert "lazy_generation" in stats


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests for the variant system."""

    def test_same_source_correct_format_per_platform(self) -> None:
        """Same source produces correct format per platform."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"shared texture data " * 100)
            path = Path(f.name)

        try:
            gen = VariantGenerator()
            source = SourceAsset.from_path(path)

            # Generate for all platforms
            results = gen.generate_all_platforms(source)

            # Verify format selection per platform
            assert results[Platform.WINDOWS].key.format == TextureFormat.BC7
            assert results[Platform.LINUX].key.format == TextureFormat.BC7
            assert results[Platform.MACOS].key.format == TextureFormat.ASTC_4x4
            assert results[Platform.IOS].key.format == TextureFormat.ASTC_4x4
            assert results[Platform.ANDROID].key.format == TextureFormat.ETC2_RGBA
            assert results[Platform.WEBGL].key.format == TextureFormat.ETC2_RGBA
        finally:
            path.unlink()

    def test_content_addressed_storage_deduplication(self) -> None:
        """Variants stored with content addressing enable deduplication."""
        storage = VariantStorage()

        # Create two variants with identical content
        source1 = ContentHash.from_content(b"source1")
        source2 = ContentHash.from_content(b"source2")
        shared_data = b"identical compressed output"

        key1 = VariantKey(source1, Platform.WINDOWS, TextureFormat.BC7)
        key2 = VariantKey(source2, Platform.WINDOWS, TextureFormat.BC7)

        variant1 = GeneratedVariant(
            key=key1,
            data=shared_data,
            variant_hash=ContentHash.from_content(shared_data),
        )
        variant2 = GeneratedVariant(
            key=key2,
            data=shared_data,
            variant_hash=ContentHash.from_content(shared_data),
        )

        storage.store(variant1)
        storage.store(variant2)

        # Both keys should resolve to the same content
        assert storage.get(key1) == storage.get(key2)

    def test_on_demand_generation_with_caching(self) -> None:
        """On-demand generation with caching."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"texture data " * 50)
            path = Path(f.name)

        try:
            gen = VariantGenerator(lazy_generation=True)
            source = SourceAsset.from_path(path)

            # First request generates
            start = time.perf_counter()
            v1 = gen.get_or_generate(source, Platform.WINDOWS)
            first_time = time.perf_counter() - start

            # Second request hits cache
            start = time.perf_counter()
            v2 = gen.get_or_generate(source, Platform.WINDOWS)
            second_time = time.perf_counter() - start

            assert v1.data == v2.data
            # Cache hit should be faster (or at least not slower)
            assert second_time <= first_time * 2  # Allow some variance
        finally:
            path.unlink()


# ==============================================================================
# Thread Safety Tests
# ==============================================================================


class TestThreadSafety:
    """Tests for thread safety of variant components."""

    def test_variant_cache_concurrent_access(self) -> None:
        cache = VariantCache(max_size=100)
        errors: list[Exception] = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(50):
                    source_hash = ContentHash.from_content(f"test{worker_id}_{i}".encode())
                    key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
                    variant = GeneratedVariant(
                        key=key,
                        data=f"data{worker_id}_{i}".encode(),
                        variant_hash=ContentHash.from_content(f"data{worker_id}_{i}".encode()),
                    )
                    cache.put(variant)
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_variant_storage_concurrent_access(self) -> None:
        storage = VariantStorage()
        errors: list[Exception] = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(20):
                    source_hash = ContentHash.from_content(f"test{worker_id}_{i}".encode())
                    key = VariantKey(source_hash, Platform.WINDOWS, TextureFormat.BC7)
                    variant = GeneratedVariant(
                        key=key,
                        data=f"data{worker_id}_{i}".encode(),
                        variant_hash=ContentHash.from_content(f"data{worker_id}_{i}".encode()),
                    )
                    storage.store(variant)
                    storage.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
