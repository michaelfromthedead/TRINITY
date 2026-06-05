"""Tests for fallback selection logic (T-CC-0.13)."""
import pytest

from engine.rendering.quality.fallback_selector import (
    FallbackSelector,
    FallbackChainResult,
    StartupCapabilityCheck,
)
from engine.rendering.quality.capability_scorer import AdapterInfo, FeatureFlags, GPULimits, GPUBackend, GPUDeviceType
from engine.rendering.quality.gles_capabilities import GLESVersion
from engine.rendering.quality.quality_manager import QualityManagerConfig
from trinity.types import QualityTier


class TestFallbackChainResult:
    """Test FallbackChainResult dataclass."""

    def test_to_dict_basic(self):
        """Test basic serialization."""
        result = FallbackChainResult(
            selected_tier=QualityTier.HIGH,
            capability_score=0.75,
            gles_version=None,
            required_workarounds=[],
            available_features={"compute_shader", "storage_buffers"},
            unavailable_features={"ray_tracing"},
            warnings=[],
        )
        d = result.to_dict()
        assert d["selected_tier"] == "HIGH"
        assert d["capability_score"] == 0.75
        assert d["gles_version"] is None
        assert "compute_shader" in d["available_features"]

    def test_to_dict_with_gles_version(self):
        """Test serialization with GLES version."""
        result = FallbackChainResult(
            selected_tier=QualityTier.LOW,
            capability_score=0.3,
            gles_version=GLESVersion.GLES_30,
            required_workarounds=["compute_shader", "ssbo"],
            available_features=set(),
            unavailable_features={"compute_shader"},
            warnings=["GLES 3.0 detected"],
        )
        d = result.to_dict()
        assert d["selected_tier"] == "LOW"
        assert d["gles_version"] == "GLES_30"
        assert "compute_shader" in d["required_workarounds"]


class TestStartupCapabilityCheckCreation:
    """Test StartupCapabilityCheck creation."""

    def test_create_with_none(self):
        """Test creation with no adapter info."""
        check = StartupCapabilityCheck(None)
        assert check.adapter_info is not None
        assert check.gles_capabilities is None

    def test_create_with_adapter_info(self):
        """Test creation with AdapterInfo object."""
        info = AdapterInfo(
            name="Test GPU",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
        )
        check = StartupCapabilityCheck(info)
        assert check.adapter_info.name == "Test GPU"
        assert check.gles_capabilities is None

    def test_create_with_dict(self):
        """Test creation with dictionary."""
        info = {
            "name": "Test GPU",
            "backend": "vulkan",
            "device_type": "discrete",
        }
        check = StartupCapabilityCheck(info)
        assert check.adapter_info.name == "Test GPU"

    def test_create_gles_backend(self):
        """Test GLES capabilities are detected for OpenGL ES backend."""
        info = AdapterInfo(
            name="Mali GPU",
            backend=GPUBackend.OPENGLES,
            device_type=GPUDeviceType.INTEGRATED,
            features=FeatureFlags(compute_shader=False, texture_compression_etc2=True),
        )
        check = StartupCapabilityCheck(info)
        assert check.gles_capabilities is not None
        assert check.gles_capabilities.version == GLESVersion.GLES_30


class TestStartupCapabilityCheckScoring:
    """Test capability scoring."""

    def test_high_end_gpu_score(self):
        """Test high-end GPU gets high score."""
        info = AdapterInfo(
            name="RTX 4090",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
            features=FeatureFlags(
                compute_shader=True,
                storage_buffers=True,
                ray_tracing=True,
                ray_query=True,
                mesh_shader=True,
                bindless=True,
            ),
            limits=GPULimits(
                max_texture_dimension_2d=16384,
                max_storage_buffers_per_shader_stage=8,
            ),
        )
        check = StartupCapabilityCheck(info)
        score = check.capability_score
        assert score > 0.8

    def test_low_end_gpu_score(self):
        """Test low-end GPU gets low score."""
        info = AdapterInfo(
            name="Intel UHD 620",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.INTEGRATED,
            features=FeatureFlags(compute_shader=True),
            limits=GPULimits(max_texture_dimension_2d=4096),
        )
        check = StartupCapabilityCheck(info)
        score = check.capability_score
        assert score < 0.6


class TestStartupCapabilityCheckTierSelection:
    """Test tier selection logic."""

    def test_gles30_forced_low(self):
        """Test GLES 3.0 is forced to LOW tier."""
        info = AdapterInfo(
            name="Mali G52",
            backend=GPUBackend.OPENGLES,
            device_type=GPUDeviceType.INTEGRATED,
            features=FeatureFlags(compute_shader=False, texture_compression_etc2=True),
        )
        check = StartupCapabilityCheck(info)
        tier = check.select_tier()
        assert tier == QualityTier.LOW

    def test_gles31_not_forced_low(self):
        """Test GLES 3.1 is not forced to LOW."""
        info = AdapterInfo(
            name="Adreno 650",
            backend=GPUBackend.OPENGLES,
            device_type=GPUDeviceType.INTEGRATED,
            features=FeatureFlags(compute_shader=True, storage_buffers=True),
            limits=GPULimits(max_texture_dimension_2d=8192),
        )
        check = StartupCapabilityCheck(info)
        tier = check.select_tier()
        assert tier != QualityTier.LOW

    def test_discrete_gpu_high_tier(self):
        """Test discrete GPU with good features gets HIGH/ULTRA."""
        info = AdapterInfo(
            name="RTX 3070",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
            features=FeatureFlags(
                compute_shader=True,
                storage_buffers=True,
                ray_query=True,
            ),
            limits=GPULimits(max_texture_dimension_2d=16384),
        )
        check = StartupCapabilityCheck(info)
        tier = check.select_tier()
        assert tier in (QualityTier.HIGH, QualityTier.ULTRA)


class TestStartupCapabilityCheckWorkarounds:
    """Test workaround detection."""

    def test_gles30_workarounds(self):
        """Test GLES 3.0 requires compute workarounds."""
        info = AdapterInfo(
            name="Mali G52",
            backend=GPUBackend.OPENGLES,
            device_type=GPUDeviceType.INTEGRATED,
            features=FeatureFlags(compute_shader=False, texture_compression_etc2=True),
        )
        check = StartupCapabilityCheck(info)
        workarounds = check.get_required_workarounds()
        assert "compute_shader" in workarounds
        assert "gpu_culling" in workarounds

    def test_vulkan_no_workarounds(self):
        """Test Vulkan backend needs no GLES workarounds."""
        info = AdapterInfo(
            name="RTX 3070",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
        )
        check = StartupCapabilityCheck(info)
        workarounds = check.get_required_workarounds()
        assert len(workarounds) == 0


class TestStartupCapabilityCheckFeatures:
    """Test feature detection."""

    def test_available_features(self):
        """Test available features detection."""
        info = AdapterInfo(
            name="Test GPU",
            backend=GPUBackend.VULKAN,
            features=FeatureFlags(
                compute_shader=True,
                storage_buffers=True,
                ray_tracing=False,
                mesh_shader=True,
            ),
        )
        check = StartupCapabilityCheck(info)
        available = check.get_available_features()
        assert "compute_shader" in available
        assert "storage_buffers" in available
        assert "mesh_shader" in available
        assert "ray_tracing" not in available

    def test_unavailable_features(self):
        """Test unavailable features detection."""
        info = AdapterInfo(
            name="Test GPU",
            backend=GPUBackend.VULKAN,
            features=FeatureFlags(compute_shader=True),
        )
        check = StartupCapabilityCheck(info)
        unavailable = check.get_unavailable_features()
        assert "ray_tracing" in unavailable
        assert "mesh_shader" in unavailable
        assert "compute_shader" not in unavailable


class TestStartupCapabilityCheckWarnings:
    """Test warning generation."""

    def test_gles30_compute_warning(self):
        """Test GLES 3.0 generates compute warning."""
        info = AdapterInfo(
            name="Mali G52",
            backend=GPUBackend.OPENGLES,
            device_type=GPUDeviceType.INTEGRATED,
            features=FeatureFlags(compute_shader=False, texture_compression_etc2=True),
        )
        check = StartupCapabilityCheck(info)
        warnings = check.get_warnings()
        assert any("compute" in w.lower() for w in warnings)

    def test_integrated_gpu_warning(self):
        """Test integrated GPU generates warning."""
        info = AdapterInfo(
            name="Intel UHD",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.INTEGRATED,
        )
        check = StartupCapabilityCheck(info)
        warnings = check.get_warnings()
        assert any("integrated" in w.lower() for w in warnings)

    def test_no_ray_query_warning(self):
        """Test missing ray query generates warning."""
        info = AdapterInfo(
            name="GTX 1080",
            backend=GPUBackend.VULKAN,
            features=FeatureFlags(ray_query=False),
        )
        check = StartupCapabilityCheck(info)
        warnings = check.get_warnings()
        assert any("ray query" in w.lower() for w in warnings)


class TestStartupCapabilityCheckPerform:
    """Test full capability check."""

    def test_perform_returns_result(self):
        """Test perform returns complete result."""
        info = AdapterInfo(
            name="Test GPU",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
            features=FeatureFlags(compute_shader=True),
        )
        check = StartupCapabilityCheck(info)
        result = check.perform()
        assert isinstance(result, FallbackChainResult)
        assert result.selected_tier is not None
        assert 0.0 <= result.capability_score <= 1.0

    def test_perform_gles_result(self):
        """Test perform with GLES backend."""
        info = AdapterInfo(
            name="Mali",
            backend=GPUBackend.OPENGLES,
            features=FeatureFlags(compute_shader=False, texture_compression_etc2=True),
        )
        check = StartupCapabilityCheck(info)
        result = check.perform()
        assert result.gles_version is not None
        assert len(result.required_workarounds) > 0


class TestFallbackSelectorCreation:
    """Test FallbackSelector creation."""

    def test_create_default(self):
        """Test creation with defaults."""
        selector = FallbackSelector()
        assert selector.capability_check is not None
        assert selector.quality_manager is not None

    def test_create_with_adapter_info(self):
        """Test creation with adapter info."""
        info = AdapterInfo(name="Test GPU")
        selector = FallbackSelector(adapter_info=info)
        assert selector.capability_check.adapter_info.name == "Test GPU"

    def test_create_with_config(self):
        """Test creation with quality manager config."""
        config = QualityManagerConfig(
            default_tier=QualityTier.MEDIUM,
            auto_adjust=False,
        )
        selector = FallbackSelector(manager_config=config)
        assert not selector.quality_manager._config.auto_adjust


class TestFallbackSelectorInitialize:
    """Test FallbackSelector initialization."""

    def test_initialize_sets_tier(self):
        """Test initialize sets tier in quality manager."""
        info = AdapterInfo(
            name="High End GPU",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
            features=FeatureFlags(
                compute_shader=True,
                storage_buffers=True,
                ray_query=True,
            ),
            limits=GPULimits(max_texture_dimension_2d=16384),
        )
        selector = FallbackSelector(adapter_info=info)
        result = selector.initialize()

        manager_tier = selector.quality_manager.current_tier
        assert manager_tier == result.selected_tier

    def test_initialize_gles30_overrides(self):
        """Test GLES 3.0 applies compute-related overrides."""
        info = AdapterInfo(
            name="Mali",
            backend=GPUBackend.OPENGLES,
            features=FeatureFlags(compute_shader=False, texture_compression_etc2=True),
        )
        selector = FallbackSelector(adapter_info=info)
        selector.initialize()

        # Check compute-dependent subsystems are locked to LOW
        gpu_tier = selector.get_effective_tier("gpu_compute")
        particles_tier = selector.get_effective_tier("particles")
        assert gpu_tier == QualityTier.LOW
        assert particles_tier == QualityTier.LOW


class TestFallbackSelectorFallbacks:
    """Test fallback detection methods."""

    def test_get_fallback_for_compute(self):
        """Test getting fallback for compute feature on GLES 3.0."""
        info = AdapterInfo(
            name="Mali",
            backend=GPUBackend.OPENGLES,
            features=FeatureFlags(compute_shader=False, texture_compression_etc2=True),
        )
        selector = FallbackSelector(adapter_info=info)
        selector.initialize()

        fallback = selector.get_fallback_for_feature("compute_shader")
        assert fallback is not None
        assert "ping-pong" in fallback.lower()

    def test_no_fallback_for_vulkan(self):
        """Test no fallback needed for Vulkan."""
        info = AdapterInfo(
            name="RTX",
            backend=GPUBackend.VULKAN,
            features=FeatureFlags(compute_shader=True),
        )
        selector = FallbackSelector(adapter_info=info)
        selector.initialize()

        fallback = selector.get_fallback_for_feature("compute_shader")
        assert fallback is None

    def test_should_use_fallback_gles30(self):
        """Test should_use_fallback for GLES 3.0."""
        info = AdapterInfo(
            name="Mali",
            backend=GPUBackend.OPENGLES,
            features=FeatureFlags(compute_shader=False, texture_compression_etc2=True),
        )
        selector = FallbackSelector(adapter_info=info)
        selector.initialize()

        assert selector.should_use_fallback("compute_shader")
        assert selector.should_use_fallback("gpu_culling")

    def test_should_use_fallback_unavailable_feature(self):
        """Test should_use_fallback for unavailable features."""
        info = AdapterInfo(
            name="GTX 1080",
            backend=GPUBackend.VULKAN,
            features=FeatureFlags(ray_tracing=False),
        )
        selector = FallbackSelector(adapter_info=info)
        selector.initialize()

        assert selector.should_use_fallback("ray_tracing")


class TestFallbackSelectorLogging:
    """Test configuration logging."""

    def test_log_configuration_basic(self):
        """Test basic configuration log."""
        info = AdapterInfo(
            name="Test GPU",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
        )
        selector = FallbackSelector(adapter_info=info)
        selector.initialize()

        log = selector.log_configuration()
        assert "Test GPU" in log
        assert "VULKAN" in log
        assert "DISCRETE" in log
        assert "Capability Score" in log

    def test_log_configuration_gles(self):
        """Test GLES configuration log."""
        info = AdapterInfo(
            name="Mali",
            backend=GPUBackend.OPENGLES,
            features=FeatureFlags(compute_shader=False, texture_compression_etc2=True),
        )
        selector = FallbackSelector(adapter_info=info)
        selector.initialize()

        log = selector.log_configuration()
        assert "GLES Version" in log
        assert "Workarounds" in log
        assert "Warnings" in log


class TestFallbackSelectorIntegration:
    """Integration tests for full startup flow."""

    def test_full_startup_flow_high_end(self):
        """Test full startup flow for high-end GPU."""
        info = AdapterInfo(
            name="RTX 4090",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
            features=FeatureFlags(
                compute_shader=True,
                storage_buffers=True,
                ray_tracing=True,
                ray_query=True,
                mesh_shader=True,
                bindless=True,
                indirect_draw=True,
            ),
            limits=GPULimits(
                max_texture_dimension_2d=16384,
                max_storage_buffers_per_shader_stage=8,
            ),
        )
        selector = FallbackSelector(adapter_info=info)
        result = selector.initialize()

        assert result.selected_tier in (QualityTier.HIGH, QualityTier.ULTRA)
        assert result.capability_score > 0.8
        assert len(result.required_workarounds) == 0
        assert "ray_tracing" in result.available_features

    def test_full_startup_flow_gles30(self):
        """Test full startup flow for GLES 3.0 device."""
        info = AdapterInfo(
            name="Mali G52",
            backend=GPUBackend.OPENGLES,
            device_type=GPUDeviceType.INTEGRATED,
            features=FeatureFlags(
                compute_shader=False,
                texture_compression_etc2=True,
            ),
        )
        selector = FallbackSelector(adapter_info=info)
        result = selector.initialize()

        assert result.selected_tier == QualityTier.LOW
        assert result.gles_version == GLESVersion.GLES_30
        assert "compute_shader" in result.required_workarounds
        assert len(result.warnings) > 0

    def test_full_startup_flow_mid_range(self):
        """Test full startup flow for mid-range GPU."""
        info = AdapterInfo(
            name="GTX 1660",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
            features=FeatureFlags(
                compute_shader=True,
                storage_buffers=True,
                ray_tracing=False,
            ),
            limits=GPULimits(max_texture_dimension_2d=8192),
        )
        selector = FallbackSelector(adapter_info=info)
        result = selector.initialize()

        assert result.selected_tier in (QualityTier.MEDIUM, QualityTier.HIGH)
        assert "ray_tracing" in result.unavailable_features
