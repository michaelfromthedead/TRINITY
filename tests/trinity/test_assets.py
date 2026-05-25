"""
Tests for Trinity Pattern - Tier 8: ASSETS Decorators
"""

import pytest

from trinity.decorators.assets import (
    VALID_COMPRESSION,
    VALID_RESIDENCY_PRIORITIES,
    AssetConfig,
    CookConfig,
    ImportSettingsConfig,
    ResidencyConfig,
    asset,
    cook,
    import_settings,
    preload,
    residency,
)
from trinity.decorators.ops import Op, decompose, expand


# ============================================================================
# Test @asset decorator
# ============================================================================


class TestAsset:
    """Tests for @asset decorator."""

    def test_asset_extensions_stored(self):
        """Extensions tuple is stored correctly."""

        @asset(extensions=(".png", ".jpg"))
        class Texture:
            pass

        assert Texture._asset is True
        assert Texture._asset_extensions == (".png", ".jpg")

    def test_asset_loader_callable_stored(self):
        """Loader callable is stored correctly."""

        def my_loader(path):
            return f"loaded:{path}"

        @asset(extensions=(".txt",), loader=my_loader)
        class TextFile:
            pass

        assert TextFile._asset_loader is my_loader

    def test_asset_no_loader_default(self):
        """Loader defaults to None if not provided."""

        @asset(extensions=(".obj",))
        class Mesh:
            pass

        assert Mesh._asset_loader is None

    def test_asset_applied_decorators(self):
        """_applied_decorators contains asset."""

        @asset(extensions=(".wav",))
        class Sound:
            pass

        assert "asset" in Sound._applied_decorators

    def test_asset_applied_steps(self):
        """_applied_steps contains TAG, REGISTER, DESCRIBE."""

        @asset(extensions=(".fbx",))
        class Model:
            pass

        steps = Model._applied_steps
        ops = [step.op for step in steps]

        assert Op.TAG in ops
        assert Op.REGISTER in ops
        assert Op.DESCRIBE in ops

    def test_asset_empty_extensions_raises(self):
        """Empty extensions raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):

            @asset(extensions=())
            class Bad:
                pass

    def test_asset_config_stored(self):
        """AssetConfig is stored correctly."""

        def loader(p):
            return p

        @asset(extensions=(".png",), loader=loader)
        class Tex:
            pass

        assert isinstance(Tex._asset_config, AssetConfig)
        assert Tex._asset_config.extensions == (".png",)
        assert Tex._asset_config.loader is loader

    def test_asset_non_tuple_list_extensions_raises(self):
        """Extensions must be tuple or list, not string."""
        with pytest.raises(TypeError, match="must be a tuple or list"):

            @asset(extensions=".png")
            class Bad:
                pass

    def test_asset_list_extensions_converted_to_tuple(self):
        """List extensions are stored as tuple."""

        @asset(extensions=[".jpg", ".png"])
        class Image:
            pass

        # Should be stored as tuple
        assert isinstance(Image._asset_extensions, tuple)
        assert Image._asset_extensions == (".jpg", ".png")


# ============================================================================
# Test @preload decorator
# ============================================================================


class TestPreload:
    """Tests for @preload decorator."""

    def test_preload_default_priority_is_zero(self):
        """Default priority is 0 and different from custom value."""

        @preload()
        class Asset:
            pass

        @preload(priority=5)
        class Custom:
            pass

        assert Asset._preload_priority == 0
        assert Custom._preload_priority == 5
        assert Asset._preload_priority != Custom._preload_priority

    def test_preload_custom_priority(self):
        """Custom priority is stored."""

        @preload(priority=10)
        class CriticalAsset:
            pass

        assert CriticalAsset._preload_priority == 10

    def test_preload_negative_priority(self):
        """Negative priority is allowed."""

        @preload(priority=-5)
        class LowPriority:
            pass

        assert LowPriority._preload_priority == -5

    def test_preload_applied_decorators(self):
        """_applied_decorators contains preload."""

        @preload()
        class P:
            pass

        assert "preload" in P._applied_decorators

    def test_preload_priority_can_be_float(self):
        """Priority can be float for fine-grained ordering."""

        @preload(priority=2.5)
        class Fractional:
            pass

        assert Fractional._preload_priority == 2.5


# ============================================================================
# Test @cook decorator
# ============================================================================


class TestCook:
    """Tests for @cook decorator."""

    def test_cook_compression_none(self):
        """Compression 'none' is valid."""

        @cook(compression="none")
        class Raw:
            pass

        assert Raw._cook_compression == "none"

    def test_cook_compression_lz4(self):
        """Compression 'lz4' is valid (default)."""

        @cook()
        class Fast:
            pass

        assert Fast._cook_compression == "lz4"

    def test_cook_compression_zstd(self):
        """Compression 'zstd' is valid."""

        @cook(compression="zstd")
        class Small:
            pass

        assert Small._cook_compression == "zstd"

    def test_cook_platform_none_vs_specific(self):
        """Platform None differs from platform-specific build."""

        @cook()
        class Universal:
            pass

        @cook(platform="linux")
        class Specific:
            pass

        assert Universal._cook_platform is None
        assert Specific._cook_platform == "linux"
        assert Universal._cook_platform != Specific._cook_platform

    def test_cook_platform_specific(self):
        """Platform can be specified."""

        @cook(platform="windows")
        class WinOnly:
            pass

        assert WinOnly._cook_platform == "windows"

    def test_cook_strip_debug_default_vs_explicit_false(self):
        """strip_debug defaults to True, differs from explicit False."""

        @cook()
        class Prod:
            pass

        @cook(strip_debug=False)
        class Dev:
            pass

        assert Prod._cook_strip_debug is True
        assert Dev._cook_strip_debug is False
        assert Prod._cook_strip_debug != Dev._cook_strip_debug


    def test_cook_invalid_compression_raises(self):
        """Invalid compression raises ValueError."""
        with pytest.raises(ValueError, match="Invalid compression"):

            @cook(compression="invalid")
            class Bad:
                pass

    def test_cook_config_stored(self):
        """CookConfig is stored correctly."""

        @cook(platform="linux", compression="zstd", strip_debug=False)
        class C:
            pass

        assert isinstance(C._cook_config, CookConfig)
        assert C._cook_config.platform == "linux"
        assert C._cook_config.compression == "zstd"
        assert C._cook_config.strip_debug is False

    def test_cook_case_sensitive_compression(self):
        """Compression validation is case-sensitive."""
        with pytest.raises(ValueError, match="Invalid compression"):

            @cook(compression="LZ4")
            class Bad:
                pass


# ============================================================================
# Test @residency decorator
# ============================================================================


class TestResidency:
    """Tests for @residency decorator."""

    @pytest.mark.parametrize(
        "priority", ["critical", "high", "normal", "low", "evictable"]
    )
    def test_residency_valid_priorities(self, priority):
        """All valid priorities work."""

        @residency(priority=priority)
        class Asset:
            pass

        assert Asset._residency_priority == priority

    def test_residency_invalid_priority_raises(self):
        """Invalid priority raises ValueError."""
        with pytest.raises(ValueError, match="Invalid priority"):

            @residency(priority="invalid")
            class Bad:
                pass

    def test_residency_min_mip_default(self):
        """min_mip defaults to 0."""

        @residency(priority="normal")
        class Tex:
            pass

        assert Tex._residency_min_mip == 0

    def test_residency_min_mip_custom(self):
        """min_mip can be customized."""

        @residency(priority="low", min_mip=3)
        class LowRes:
            pass

        assert LowRes._residency_min_mip == 3

    def test_residency_negative_min_mip_raises(self):
        """Negative min_mip raises ValueError."""
        with pytest.raises(ValueError, match="min_mip must be >= 0"):

            @residency(priority="normal", min_mip=-1)
            class Bad:
                pass

    def test_residency_config_stored(self):
        """ResidencyConfig is stored correctly."""

        @residency(priority="high", min_mip=2)
        class R:
            pass

        assert isinstance(R._residency_config, ResidencyConfig)
        assert R._residency_config.priority == "high"
        assert R._residency_config.min_mip == 2

    def test_residency_missing_priority_raises(self):
        """priority parameter is required (raises ValueError with None)."""
        with pytest.raises(ValueError, match="Invalid priority"):

            @residency()
            class Bad:
                pass

    def test_residency_case_sensitive_priority(self):
        """Priority validation is case-sensitive."""
        with pytest.raises(ValueError, match="Invalid priority"):

            @residency(priority="Normal")
            class Bad:
                pass


# ============================================================================
# Test @import_settings decorator
# ============================================================================


class TestImportSettings:
    """Tests for @import_settings decorator."""

    def test_import_settings_defaults_differ_from_custom(self):
        """Default values differ from custom values."""

        @import_settings()
        class DefaultModel:
            pass

        @import_settings(scale=2.0, axis_conversion=("Z", "Y", "X"), merge_meshes=True)
        class CustomModel:
            pass

        assert DefaultModel._import_scale == 1.0
        assert CustomModel._import_scale == 2.0
        assert DefaultModel._import_scale != CustomModel._import_scale

        assert DefaultModel._import_axis_conversion == ("X", "Y", "Z")
        assert CustomModel._import_axis_conversion == ("Z", "Y", "X")

        assert DefaultModel._import_merge_meshes is False
        assert CustomModel._import_merge_meshes is True

    def test_import_settings_custom_scale(self):
        """Custom scale is stored."""

        @import_settings(scale=0.01)
        class Small:
            pass

        assert Small._import_scale == 0.01

    def test_import_settings_custom_axis_conversion(self):
        """Custom axis conversion is stored."""

        @import_settings(axis_conversion=("Z", "X", "Y"))
        class Rotated:
            pass

        assert Rotated._import_axis_conversion == ("Z", "X", "Y")

    def test_import_settings_merge_meshes(self):
        """merge_meshes can be True."""

        @import_settings(merge_meshes=True)
        class Merged:
            pass

        assert Merged._import_merge_meshes is True

    def test_import_settings_config_stored(self):
        """ImportSettingsConfig is stored correctly."""

        @import_settings(scale=2.0, axis_conversion=("Y", "Z", "X"), merge_meshes=True)
        class I:
            pass

        assert isinstance(I._import_settings_config, ImportSettingsConfig)
        assert I._import_settings_config.scale == 2.0
        assert I._import_settings_config.axis_conversion == ("Y", "Z", "X")
        assert I._import_settings_config.merge_meshes is True

    def test_import_settings_negative_scale(self):
        """Negative scale is allowed (for mirroring)."""

        @import_settings(scale=-1.0)
        class Mirrored:
            pass

        assert Mirrored._import_scale == -1.0

    def test_import_settings_zero_scale(self):
        """Zero scale is allowed (edge case)."""

        @import_settings(scale=0.0)
        class Zero:
            pass

        assert Zero._import_scale == 0.0

    def test_import_settings_axis_list_converted_to_tuple(self):
        """List axis_conversion is converted to tuple."""

        @import_settings(axis_conversion=["Z", "X", "Y"])
        class Model:
            pass

        assert isinstance(Model._import_axis_conversion, tuple)
        assert Model._import_axis_conversion == ("Z", "X", "Y")


# ============================================================================
# Test introspection (decompose/expand)
# ============================================================================


class TestAssetsIntrospection:
    """Tests for introspection of ASSETS decorators."""

    @pytest.mark.parametrize(
        "dec",
        [asset, preload, cook, residency, import_settings],
    )
    def test_decompose_returns_steps(self, dec):
        """decompose returns steps list."""
        steps = decompose(dec)
        assert isinstance(steps, list)

    @pytest.mark.parametrize(
        "dec",
        [asset, preload, cook, residency, import_settings],
    )
    def test_expand_returns_string(self, dec):
        """expand returns string representation."""
        result = expand(dec)
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        "dec",
        [asset, preload, cook, residency, import_settings],
    )
    def test_all_have_register_assets(self, dec):
        """All ASSETS decorators register to 'assets' registry."""
        steps = decompose(dec)
        register_steps = [s for s in steps if s.op == Op.REGISTER]

        # Should have at least one REGISTER step with registry='assets'
        assert any(
            s.args.get("registry") == "assets" for s in register_steps
        ), f"{dec.__name__} missing REGISTER(assets)"


# ============================================================================
# Test stacking decorators
# ============================================================================


class TestStackingDecorators:
    """Tests for stacking multiple ASSETS decorators."""

    def test_asset_cook_residency_stacked(self):
        """Can stack @asset, @cook, @residency on same class."""

        @asset(extensions=(".dds",))
        @cook(compression="zstd")
        @residency(priority="high", min_mip=1)
        class Texture:
            pass

        assert Texture._asset is True
        assert Texture._cook is True
        assert Texture._residency is True
        assert len(Texture._applied_decorators) == 3

    def test_all_five_stacked(self):
        """Can stack all five ASSETS decorators."""

        @asset(extensions=(".gltf",))
        @preload(priority=10)
        @cook(platform="windows")
        @residency(priority="critical")
        @import_settings(scale=0.1)
        class Model:
            pass

        assert Model._asset is True
        assert Model._preload is True
        assert Model._cook is True
        assert Model._residency is True
        assert Model._import_settings is True
        assert len(Model._applied_decorators) == 5

    def test_stacked_configs_independent(self):
        """Stacked decorators have independent configs."""

        @asset(extensions=(".png",))
        @cook(compression="lz4")
        class Tex:
            pass

        assert Tex._asset_config.extensions == (".png",)
        assert Tex._cook_config.compression == "lz4"


# ============================================================================
# Test constants
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_valid_compression_contains_all(self):
        """VALID_COMPRESSION contains all expected values."""
        assert "none" in VALID_COMPRESSION
        assert "lz4" in VALID_COMPRESSION
        assert "zstd" in VALID_COMPRESSION
        assert len(VALID_COMPRESSION) == 3

    def test_valid_residency_priorities_contains_all(self):
        """VALID_RESIDENCY_PRIORITIES contains all expected values."""
        expected = {"critical", "high", "normal", "low", "evictable"}
        assert VALID_RESIDENCY_PRIORITIES == expected
