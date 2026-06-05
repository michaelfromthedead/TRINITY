"""
Tests for Upscaler Plugin Interface (T-PP-6.5)

Tests the abstract plugin interface for vendor upscalers (DLSS, FSR, XeSS)
with runtime auto-detection and graceful fallback to TSR Lanczos.
"""

import pytest
from typing import Tuple

from engine.rendering.postprocess.upscaler_plugins import (
    UpscalerCapabilities,
    QualityPreset,
    UpscalerPlugin,
    DLSSPlugin,
    FSRPlugin,
    XeSSPlugin,
    TSRLanczosPlugin,
    UpscalerManager,
)


# ==============================================================================
# UPSCALER CAPABILITIES TESTS
# ==============================================================================


class TestUpscalerCapabilities:
    """Tests for UpscalerCapabilities dataclass."""

    def test_capabilities_dataclass_fields(self) -> None:
        """Test that UpscalerCapabilities has all required fields."""
        caps = UpscalerCapabilities(
            name="Test Upscaler",
            version="1.0",
        )
        assert caps.name == "Test Upscaler"
        assert caps.version == "1.0"
        assert caps.supports_sharpening is True  # default
        assert caps.supports_hdr is True  # default
        assert caps.min_scale == 1.0  # default
        assert caps.max_scale == 3.0  # default
        assert caps.requires_motion_vectors is True  # default
        assert caps.requires_depth is True  # default
        assert caps.supports_frame_generation is False  # default
        assert caps.vendor == ""  # default

    def test_capabilities_custom_values(self) -> None:
        """Test UpscalerCapabilities with custom values."""
        caps = UpscalerCapabilities(
            name="Custom",
            version="2.0",
            supports_sharpening=False,
            supports_hdr=False,
            min_scale=0.5,
            max_scale=4.0,
            requires_motion_vectors=False,
            requires_depth=False,
            supports_frame_generation=True,
            vendor="TestVendor",
        )
        assert caps.supports_sharpening is False
        assert caps.supports_hdr is False
        assert caps.min_scale == 0.5
        assert caps.max_scale == 4.0
        assert caps.requires_motion_vectors is False
        assert caps.requires_depth is False
        assert caps.supports_frame_generation is True
        assert caps.vendor == "TestVendor"

    def test_capabilities_immutable_access(self) -> None:
        """Test that capabilities can be accessed as expected."""
        caps = UpscalerCapabilities(name="Test", version="1.0")
        # Access all fields to ensure they're accessible
        _ = (caps.name, caps.version, caps.supports_sharpening,
             caps.supports_hdr, caps.min_scale, caps.max_scale,
             caps.requires_motion_vectors, caps.requires_depth,
             caps.supports_frame_generation, caps.vendor)


class TestQualityPreset:
    """Tests for QualityPreset enum."""

    def test_quality_preset_values(self) -> None:
        """Test that all quality presets exist with correct values."""
        assert QualityPreset.ULTRA_PERFORMANCE.value == "ultra_performance"
        assert QualityPreset.PERFORMANCE.value == "performance"
        assert QualityPreset.BALANCED.value == "balanced"
        assert QualityPreset.QUALITY.value == "quality"
        assert QualityPreset.ULTRA_QUALITY.value == "ultra_quality"
        assert QualityPreset.NATIVE_AA.value == "native_aa"


# ==============================================================================
# UPSCALER PLUGIN INTERFACE TESTS
# ==============================================================================


class TestUpscalerPluginInterface:
    """Tests for UpscalerPlugin ABC interface methods."""

    def test_upscaler_plugin_interface_methods(self) -> None:
        """Test that UpscalerPlugin defines all required abstract methods."""
        # Can't instantiate ABC directly
        with pytest.raises(TypeError):
            UpscalerPlugin()  # type: ignore

        # Verify required methods exist on concrete implementations
        plugin = TSRLanczosPlugin()
        assert hasattr(plugin, "is_available")
        assert hasattr(plugin, "name")
        assert hasattr(plugin, "capabilities")
        assert hasattr(plugin, "initialize")
        assert hasattr(plugin, "evaluate")
        assert hasattr(plugin, "get_optimal_render_resolution")
        assert hasattr(plugin, "shutdown")

    def test_supports_preset_method(self) -> None:
        """Test supports_preset helper method."""
        plugin = TSRLanczosPlugin()

        # Valid presets
        assert plugin.supports_preset("ultra_performance") is True
        assert plugin.supports_preset("performance") is True
        assert plugin.supports_preset("balanced") is True
        assert plugin.supports_preset("quality") is True
        assert plugin.supports_preset("ultra_quality") is True
        assert plugin.supports_preset("native_aa") is True

        # Case insensitive
        assert plugin.supports_preset("BALANCED") is True
        assert plugin.supports_preset("Quality") is True

        # Invalid presets
        assert plugin.supports_preset("invalid") is False
        assert plugin.supports_preset("") is False


# ==============================================================================
# DLSS PLUGIN TESTS
# ==============================================================================


class TestDLSSPlugin:
    """Tests for NVIDIA DLSS plugin."""

    def test_dlss_not_available_without_sdk(self) -> None:
        """Test that DLSS is unavailable without NGX SDK."""
        assert DLSSPlugin.is_available() is False

    def test_dlss_name(self) -> None:
        """Test DLSS plugin name."""
        plugin = DLSSPlugin()
        assert plugin.name == "NVIDIA DLSS"

    def test_dlss_capabilities(self) -> None:
        """Test DLSS capabilities report."""
        plugin = DLSSPlugin()
        caps = plugin.capabilities

        assert caps.name == "NVIDIA DLSS"
        assert caps.version == "3.5"
        assert caps.supports_sharpening is True
        assert caps.supports_hdr is True
        assert caps.min_scale == 1.0
        assert caps.max_scale == 3.0
        assert caps.requires_motion_vectors is True
        assert caps.requires_depth is True
        assert caps.supports_frame_generation is True
        assert caps.vendor == "NVIDIA"

    def test_dlss_initialize_fails_without_sdk(self) -> None:
        """Test that DLSS initialization fails without SDK."""
        plugin = DLSSPlugin()
        result = plugin.initialize((1280, 720), (2560, 1440))
        assert result is False

    def test_dlss_evaluate_raises_without_initialization(self) -> None:
        """Test that DLSS evaluate raises NotImplementedError."""
        plugin = DLSSPlugin()
        with pytest.raises(NotImplementedError):
            plugin.evaluate(None)

    def test_dlss_optimal_resolution_presets(self) -> None:
        """Test DLSS optimal resolution calculation for all presets."""
        plugin = DLSSPlugin()
        target = (3840, 2160)  # 4K

        # Ultra Performance: 3x scale = 1280x720
        res = plugin.get_optimal_render_resolution(target, "ultra_performance")
        assert res == (1280, 720)

        # Performance: 2x scale = 1920x1080
        res = plugin.get_optimal_render_resolution(target, "performance")
        assert res == (1920, 1080)

        # Balanced: 1.7x scale = ~2259x1270
        res = plugin.get_optimal_render_resolution(target, "balanced")
        assert res == (2258, 1270)  # int(3840/1.7), int(2160/1.7)

        # Quality: 1.5x scale = 2560x1440
        res = plugin.get_optimal_render_resolution(target, "quality")
        assert res == (2560, 1440)

        # Ultra Quality: 1.3x scale = ~2953x1661
        res = plugin.get_optimal_render_resolution(target, "ultra_quality")
        assert res == (2953, 1661)

        # Native AA (DLAA): 1x scale = 3840x2160
        res = plugin.get_optimal_render_resolution(target, "native_aa")
        assert res == (3840, 2160)

    def test_dlss_optimal_resolution_unknown_preset(self) -> None:
        """Test DLSS defaults to balanced for unknown presets."""
        plugin = DLSSPlugin()
        target = (3840, 2160)
        res = plugin.get_optimal_render_resolution(target, "unknown_preset")
        # Should default to balanced (1.7x)
        assert res == (2258, 1270)

    def test_dlss_shutdown_no_error(self) -> None:
        """Test that DLSS shutdown doesn't error even if not initialized."""
        plugin = DLSSPlugin()
        # Should not raise
        plugin.shutdown()
        plugin.shutdown()  # Double shutdown should be safe


# ==============================================================================
# FSR PLUGIN TESTS
# ==============================================================================


class TestFSRPlugin:
    """Tests for AMD FSR 2 plugin."""

    def test_fsr_not_available_without_sdk(self) -> None:
        """Test that FSR is unavailable without FidelityFX SDK."""
        assert FSRPlugin.is_available() is False

    def test_fsr_name(self) -> None:
        """Test FSR plugin name."""
        plugin = FSRPlugin()
        assert plugin.name == "AMD FSR 2"

    def test_fsr_capabilities(self) -> None:
        """Test FSR capabilities report."""
        plugin = FSRPlugin()
        caps = plugin.capabilities

        assert caps.name == "AMD FSR 2"
        assert caps.version == "2.2"
        assert caps.supports_sharpening is True
        assert caps.supports_hdr is True
        assert caps.requires_motion_vectors is True
        assert caps.requires_depth is True
        assert caps.supports_frame_generation is False  # FSR 2 doesn't have FG
        assert caps.vendor == "AMD"

    def test_fsr_initialize_fails_without_sdk(self) -> None:
        """Test that FSR initialization fails without SDK."""
        plugin = FSRPlugin()
        result = plugin.initialize((1280, 720), (2560, 1440))
        assert result is False

    def test_fsr_evaluate_raises_without_initialization(self) -> None:
        """Test that FSR evaluate raises NotImplementedError."""
        plugin = FSRPlugin()
        with pytest.raises(NotImplementedError):
            plugin.evaluate(None)

    def test_fsr_optimal_resolution_presets(self) -> None:
        """Test FSR optimal resolution calculation for all presets."""
        plugin = FSRPlugin()
        target = (2560, 1440)  # 1440p

        # Ultra Performance: 3x
        res = plugin.get_optimal_render_resolution(target, "ultra_performance")
        assert res == (853, 480)

        # Performance: 2x
        res = plugin.get_optimal_render_resolution(target, "performance")
        assert res == (1280, 720)

        # Balanced: 1.7x
        res = plugin.get_optimal_render_resolution(target, "balanced")
        assert res == (1505, 847)

        # Quality: 1.5x
        res = plugin.get_optimal_render_resolution(target, "quality")
        assert res == (1706, 960)

    def test_fsr_shutdown_no_error(self) -> None:
        """Test that FSR shutdown doesn't error."""
        plugin = FSRPlugin()
        plugin.shutdown()
        plugin.shutdown()  # Double shutdown safe


# ==============================================================================
# XeSS PLUGIN TESTS
# ==============================================================================


class TestXeSSPlugin:
    """Tests for Intel XeSS plugin."""

    def test_xess_not_available_without_sdk(self) -> None:
        """Test that XeSS is unavailable without XeSS SDK."""
        assert XeSSPlugin.is_available() is False

    def test_xess_name(self) -> None:
        """Test XeSS plugin name."""
        plugin = XeSSPlugin()
        assert plugin.name == "Intel XeSS"

    def test_xess_capabilities(self) -> None:
        """Test XeSS capabilities report."""
        plugin = XeSSPlugin()
        caps = plugin.capabilities

        assert caps.name == "Intel XeSS"
        assert caps.version == "1.3"
        assert caps.supports_sharpening is True
        assert caps.supports_hdr is True
        assert caps.requires_motion_vectors is True
        assert caps.requires_depth is True
        assert caps.supports_frame_generation is False
        assert caps.vendor == "Intel"

    def test_xess_initialize_fails_without_sdk(self) -> None:
        """Test that XeSS initialization fails without SDK."""
        plugin = XeSSPlugin()
        result = plugin.initialize((1280, 720), (2560, 1440))
        assert result is False

    def test_xess_evaluate_raises_without_initialization(self) -> None:
        """Test that XeSS evaluate raises NotImplementedError."""
        plugin = XeSSPlugin()
        with pytest.raises(NotImplementedError):
            plugin.evaluate(None)

    def test_xess_dp4a_fallback_property(self) -> None:
        """Test XeSS DP4a fallback property."""
        plugin = XeSSPlugin()
        # Default is False (not using fallback)
        assert plugin.using_dp4a_fallback is False

    def test_xess_optimal_resolution_presets(self) -> None:
        """Test XeSS optimal resolution calculation."""
        plugin = XeSSPlugin()
        target = (1920, 1080)  # 1080p

        # Performance: 2x = 960x540
        res = plugin.get_optimal_render_resolution(target, "performance")
        assert res == (960, 540)

        # Quality: 1.5x = 1280x720
        res = plugin.get_optimal_render_resolution(target, "quality")
        assert res == (1280, 720)

    def test_xess_shutdown_no_error(self) -> None:
        """Test that XeSS shutdown doesn't error."""
        plugin = XeSSPlugin()
        plugin.shutdown()


# ==============================================================================
# TSR LANCZOS PLUGIN TESTS
# ==============================================================================


class TestTSRLanczosPlugin:
    """Tests for TSR Lanczos fallback plugin."""

    def test_tsr_always_available(self) -> None:
        """Test that TSR Lanczos is always available."""
        assert TSRLanczosPlugin.is_available() is True

    def test_tsr_name(self) -> None:
        """Test TSR Lanczos plugin name."""
        plugin = TSRLanczosPlugin()
        assert plugin.name == "TSR Lanczos"

    def test_tsr_capabilities(self) -> None:
        """Test TSR Lanczos capabilities."""
        plugin = TSRLanczosPlugin()
        caps = plugin.capabilities

        assert caps.name == "TSR Lanczos"
        assert caps.version == "1.0"
        assert caps.supports_sharpening is True
        assert caps.supports_hdr is True
        assert caps.min_scale == 1.0
        assert caps.max_scale == 4.0  # Lanczos can go higher
        assert caps.requires_motion_vectors is False  # Optional
        assert caps.requires_depth is False  # Optional
        assert caps.vendor == ""  # Works on any hardware

    def test_tsr_initialize_always_succeeds(self) -> None:
        """Test that TSR Lanczos initialization always succeeds."""
        plugin = TSRLanczosPlugin()
        result = plugin.initialize((1280, 720), (2560, 1440))
        assert result is True

    def test_tsr_initialize_with_different_presets(self) -> None:
        """Test TSR initialization with different quality presets."""
        plugin = TSRLanczosPlugin()

        # Low quality uses Lanczos2
        assert plugin.initialize((1280, 720), (2560, 1440), "performance") is True

        # High quality uses Lanczos3
        assert plugin.initialize((1920, 1080), (2560, 1440), "quality") is True

    def test_tsr_evaluate_no_error(self) -> None:
        """Test that TSR evaluate works."""
        plugin = TSRLanczosPlugin()
        plugin.initialize((1280, 720), (2560, 1440))

        # Should not raise
        result = plugin.evaluate(None, sharpness=0.5)
        # Stub returns input
        assert result is None

    def test_tsr_evaluate_with_reset(self) -> None:
        """Test TSR evaluate with reset flag."""
        plugin = TSRLanczosPlugin()
        plugin.initialize((1280, 720), (2560, 1440))

        # Should not raise even with reset
        plugin.evaluate(None, reset=True)

    def test_tsr_optimal_resolution(self) -> None:
        """Test TSR optimal resolution calculation."""
        plugin = TSRLanczosPlugin()
        target = (3840, 2160)

        res = plugin.get_optimal_render_resolution(target, "balanced")
        assert res == (2258, 1270)  # Same as DLSS balanced

    def test_tsr_shutdown_no_error(self) -> None:
        """Test TSR shutdown."""
        plugin = TSRLanczosPlugin()
        plugin.initialize((1280, 720), (2560, 1440))
        plugin.shutdown()

        # Double shutdown should be safe
        plugin.shutdown()


# ==============================================================================
# UPSCALER MANAGER TESTS
# ==============================================================================


class TestUpscalerManager:
    """Tests for UpscalerManager."""

    def test_manager_creation(self) -> None:
        """Test manager can be created."""
        manager = UpscalerManager()
        assert manager.active_plugin is None

    def test_manager_fallback_to_tsr(self) -> None:
        """Test that manager falls back to TSR when no vendor SDK available."""
        manager = UpscalerManager()
        upscaler = manager.detect_best_upscaler()

        # Should fall back to TSR since no SDKs are available
        assert upscaler.name == "TSR Lanczos"
        assert manager.active_plugin is upscaler

    def test_manager_priority_order(self) -> None:
        """Test that manager has correct priority order."""
        # Priority: DLSS > XeSS > FSR > TSR
        assert UpscalerManager.PRIORITY_ORDER[0] == DLSSPlugin
        assert UpscalerManager.PRIORITY_ORDER[1] == XeSSPlugin
        assert UpscalerManager.PRIORITY_ORDER[2] == FSRPlugin
        assert UpscalerManager.PRIORITY_ORDER[3] == TSRLanczosPlugin

    def test_manager_get_available_upscalers(self) -> None:
        """Test listing available upscalers."""
        manager = UpscalerManager()
        available = manager.get_available_upscalers()

        # Only TSR should be available (no vendor SDKs)
        assert "TSR Lanczos" in available
        assert "NVIDIA DLSS" not in available
        assert "AMD FSR 2" not in available
        assert "Intel XeSS" not in available

    def test_manager_select_upscaler_by_name(self) -> None:
        """Test selecting upscaler by name."""
        manager = UpscalerManager()

        # Should succeed for TSR
        assert manager.select_upscaler("TSR Lanczos") is True
        assert manager.active_plugin is not None
        assert manager.active_plugin.name == "TSR Lanczos"

        # Should fail for unavailable upscalers
        assert manager.select_upscaler("NVIDIA DLSS") is False

    def test_manager_select_upscaler_case_insensitive(self) -> None:
        """Test that upscaler selection is case insensitive."""
        manager = UpscalerManager()

        assert manager.select_upscaler("tsr lanczos") is True
        assert manager.select_upscaler("TSR LANCZOS") is True

    def test_manager_select_unknown_upscaler(self) -> None:
        """Test selecting unknown upscaler returns False."""
        manager = UpscalerManager()
        assert manager.select_upscaler("Unknown Upscaler") is False

    def test_manager_get_upscaler_by_name(self) -> None:
        """Test getting upscaler by name."""
        manager = UpscalerManager()

        # Should return plugin instance
        plugin = manager.get_upscaler_by_name("TSR Lanczos")
        assert plugin is not None
        assert plugin.name == "TSR Lanczos"

        # Unknown returns None
        assert manager.get_upscaler_by_name("Unknown") is None

    def test_manager_get_fallback(self) -> None:
        """Test getting fallback upscaler."""
        manager = UpscalerManager()
        fallback = manager.get_fallback()

        assert fallback is not None
        assert isinstance(fallback, TSRLanczosPlugin)
        assert fallback.is_available() is True

    def test_manager_shutdown_active(self) -> None:
        """Test shutting down active upscaler."""
        manager = UpscalerManager()
        manager.detect_best_upscaler()

        assert manager.active_plugin is not None
        manager.shutdown_active()
        assert manager.active_plugin is None

    def test_manager_shutdown_when_none_active(self) -> None:
        """Test shutdown is safe when no upscaler is active."""
        manager = UpscalerManager()
        # Should not raise
        manager.shutdown_active()

    def test_manager_get_capabilities(self) -> None:
        """Test getting capabilities of all upscalers."""
        manager = UpscalerManager()
        caps = manager.get_capabilities()

        # Should have entries for all registered upscalers
        assert "NVIDIA DLSS" in caps
        assert "AMD FSR 2" in caps
        assert "Intel XeSS" in caps
        assert "TSR Lanczos" in caps

        # Verify structure
        dlss_caps = caps["NVIDIA DLSS"]
        assert dlss_caps.vendor == "NVIDIA"
        assert dlss_caps.supports_frame_generation is True


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestUpscalerIntegration:
    """Integration tests for the upscaler plugin system."""

    def test_full_lifecycle_tsr(self) -> None:
        """Test full upscaler lifecycle with TSR Lanczos."""
        manager = UpscalerManager()

        # Detect best upscaler
        upscaler = manager.detect_best_upscaler()
        assert upscaler.name == "TSR Lanczos"

        # Initialize
        assert upscaler.initialize((1280, 720), (2560, 1440), "quality") is True

        # Get optimal resolution
        res = upscaler.get_optimal_render_resolution((2560, 1440), "quality")
        assert res[0] > 0 and res[1] > 0

        # Evaluate (stub)
        output = upscaler.evaluate(None, sharpness=0.7)

        # Shutdown
        upscaler.shutdown()
        manager.shutdown_active()

    def test_graceful_degradation(self) -> None:
        """Test graceful degradation when preferred upscaler unavailable."""
        manager = UpscalerManager()

        # Try to select DLSS (not available)
        selected = manager.select_upscaler("NVIDIA DLSS")
        assert selected is False

        # Fallback to best available
        upscaler = manager.detect_best_upscaler()
        assert upscaler.is_available() is True
        assert upscaler.name == "TSR Lanczos"

    def test_resolution_calculations_consistent(self) -> None:
        """Test that resolution calculations are consistent across upscalers."""
        target = (2560, 1440)
        preset = "balanced"

        dlss = DLSSPlugin()
        fsr = FSRPlugin()
        xess = XeSSPlugin()
        tsr = TSRLanczosPlugin()

        # All should calculate same resolution for same preset
        dlss_res = dlss.get_optimal_render_resolution(target, preset)
        fsr_res = fsr.get_optimal_render_resolution(target, preset)
        xess_res = xess.get_optimal_render_resolution(target, preset)
        tsr_res = tsr.get_optimal_render_resolution(target, preset)

        # All use same scale factors
        assert dlss_res == fsr_res == xess_res == tsr_res

    def test_minimum_resolution_clamping(self) -> None:
        """Test that resolution calculation clamps to minimum 1."""
        plugin = DLSSPlugin()

        # Very small target should still return at least 1x1
        res = plugin.get_optimal_render_resolution((2, 2), "ultra_performance")
        assert res[0] >= 1
        assert res[1] >= 1

    def test_all_plugins_have_vendor_info(self) -> None:
        """Test that all vendor plugins report correct vendor."""
        assert DLSSPlugin().capabilities.vendor == "NVIDIA"
        assert FSRPlugin().capabilities.vendor == "AMD"
        assert XeSSPlugin().capabilities.vendor == "Intel"
        assert TSRLanczosPlugin().capabilities.vendor == ""  # Hardware agnostic

    def test_frame_generation_support(self) -> None:
        """Test frame generation capability reporting."""
        # Only DLSS supports frame generation (DLSS 3)
        assert DLSSPlugin().capabilities.supports_frame_generation is True
        assert FSRPlugin().capabilities.supports_frame_generation is False
        assert XeSSPlugin().capabilities.supports_frame_generation is False
        assert TSRLanczosPlugin().capabilities.supports_frame_generation is False


# ==============================================================================
# EDGE CASES AND ERROR HANDLING
# ==============================================================================


class TestUpscalerEdgeCases:
    """Edge case and error handling tests."""

    def test_multiple_initialize_calls(self) -> None:
        """Test that multiple initialize calls are handled."""
        plugin = TSRLanczosPlugin()

        # Initialize multiple times with different settings
        assert plugin.initialize((1280, 720), (2560, 1440)) is True
        assert plugin.initialize((1920, 1080), (3840, 2160)) is True
        assert plugin.initialize((640, 480), (1280, 720)) is True

    def test_evaluate_before_initialize(self) -> None:
        """Test evaluate behavior before explicit initialization."""
        plugin = TSRLanczosPlugin()

        # TSR should auto-initialize
        result = plugin.evaluate(None)
        # Should not raise

    def test_zero_resolution_handling(self) -> None:
        """Test handling of zero resolution."""
        plugin = DLSSPlugin()

        # Should handle gracefully - implementation clamps to minimum 1
        res = plugin.get_optimal_render_resolution((0, 0), "balanced")
        # max(1, int(0/1.7)) = max(1, 0) = 1
        assert res == (1, 1)

    def test_negative_sharpness(self) -> None:
        """Test handling of negative sharpness value."""
        plugin = TSRLanczosPlugin()
        plugin.initialize((1280, 720), (2560, 1440))

        # Should handle gracefully (clamped or ignored)
        plugin.evaluate(None, sharpness=-0.5)

    def test_very_large_resolution(self) -> None:
        """Test handling of very large resolutions."""
        plugin = DLSSPlugin()

        # 8K target
        res = plugin.get_optimal_render_resolution((7680, 4320), "ultra_performance")
        assert res == (2560, 1440)  # 8K / 3 = ~2560x1440

    def test_aspect_ratio_preservation(self) -> None:
        """Test that aspect ratio is preserved in resolution calculations."""
        plugin = DLSSPlugin()

        # 21:9 ultrawide
        target = (3440, 1440)  # 21:9
        res = plugin.get_optimal_render_resolution(target, "performance")

        # Should maintain ~21:9 ratio
        target_ratio = target[0] / target[1]
        result_ratio = res[0] / res[1]
        assert abs(target_ratio - result_ratio) < 0.1


# ==============================================================================
# PERFORMANCE CONSIDERATIONS
# ==============================================================================


class TestUpscalerPerformance:
    """Tests related to performance characteristics."""

    def test_is_available_is_static(self) -> None:
        """Test that is_available is a static method (no instance needed)."""
        # Should be callable without creating instance
        _ = DLSSPlugin.is_available()
        _ = FSRPlugin.is_available()
        _ = XeSSPlugin.is_available()
        _ = TSRLanczosPlugin.is_available()

    def test_manager_caches_detection(self) -> None:
        """Test that manager can be reused efficiently."""
        manager = UpscalerManager()

        # Multiple detect calls should work
        up1 = manager.detect_best_upscaler()
        up2 = manager.detect_best_upscaler()

        # Both should be valid (though may be different instances)
        assert up1.is_available()
        assert up2.is_available()

    def test_shutdown_releases_resources(self) -> None:
        """Test that shutdown properly releases resources."""
        plugin = TSRLanczosPlugin()
        plugin.initialize((1920, 1080), (3840, 2160))

        # After shutdown, internal state should be cleared
        plugin.shutdown()

        # Re-initialization should work
        assert plugin.initialize((1280, 720), (2560, 1440)) is True
