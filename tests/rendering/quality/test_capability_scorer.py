"""Tests for GPU capability scoring (T-CC-0.2)."""

import pytest

from engine.rendering.quality.capability_scorer import (
    AdapterInfo,
    CapabilityScorer,
    FeatureFlags,
    GPUBackend,
    GPUDeviceType,
    GPULimits,
)


class TestFeatureFlags:
    """Test FeatureFlags dataclass."""

    def test_default_values(self):
        """Test default feature values."""
        f = FeatureFlags()
        assert f.compute_shader is True
        assert f.storage_buffers is True
        assert f.ray_tracing is False
        assert f.bindless is False

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "ray_tracing": True,
            "bindless": True,
            "mesh_shader": True,
        }
        f = FeatureFlags.from_dict(data)
        assert f.ray_tracing is True
        assert f.bindless is True
        assert f.mesh_shader is True
        assert f.compute_shader is True  # default

    def test_from_dict_ignores_unknown(self):
        """Test that unknown keys are ignored."""
        data = {
            "ray_tracing": True,
            "unknown_feature": True,
        }
        f = FeatureFlags.from_dict(data)
        assert f.ray_tracing is True
        assert not hasattr(f, "unknown_feature")


class TestGPULimits:
    """Test GPULimits dataclass."""

    def test_default_values(self):
        """Test default limit values."""
        lim = GPULimits()
        assert lim.max_texture_dimension_2d == 8192
        assert lim.max_compute_invocations_per_workgroup == 256

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "max_texture_dimension_2d": 16384,
            "max_storage_buffer_binding_size": 2147483648,
        }
        lim = GPULimits.from_dict(data)
        assert lim.max_texture_dimension_2d == 16384
        assert lim.max_storage_buffer_binding_size == 2147483648

    def test_vram_estimate_8gb(self):
        """Test VRAM estimate for high-end GPU."""
        lim = GPULimits(max_storage_buffer_binding_size=2147483648)
        assert lim.vram_estimate_mb == 8192

    def test_vram_estimate_4gb(self):
        """Test VRAM estimate for mid-range GPU."""
        lim = GPULimits(max_storage_buffer_binding_size=1073741824)
        assert lim.vram_estimate_mb == 4096

    def test_vram_estimate_1gb(self):
        """Test VRAM estimate for low-end GPU."""
        lim = GPULimits(max_storage_buffer_binding_size=134217728)
        assert lim.vram_estimate_mb == 1024

    def test_vram_estimate_low(self):
        """Test VRAM estimate for very low-end GPU."""
        lim = GPULimits(max_storage_buffer_binding_size=67108864)
        assert lim.vram_estimate_mb == 512


class TestAdapterInfo:
    """Test AdapterInfo dataclass."""

    def test_default_values(self):
        """Test default adapter values."""
        info = AdapterInfo()
        assert info.name == "Unknown"
        assert info.backend == GPUBackend.UNKNOWN
        assert info.device_type == GPUDeviceType.UNKNOWN

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "name": "NVIDIA RTX 4090",
            "vendor": "NVIDIA",
            "backend": "vulkan",
            "device_type": "discrete",
            "features": {"ray_tracing": True, "bindless": True},
            "limits": {"max_texture_dimension_2d": 16384},
        }
        info = AdapterInfo.from_dict(data)
        assert info.name == "NVIDIA RTX 4090"
        assert info.vendor == "NVIDIA"
        assert info.backend == GPUBackend.VULKAN
        assert info.device_type == GPUDeviceType.DISCRETE
        assert info.features.ray_tracing is True
        assert info.limits.max_texture_dimension_2d == 16384

    def test_from_dict_case_insensitive_backend(self):
        """Test backend parsing is case-insensitive."""
        data = {"backend": "METAL"}
        info = AdapterInfo.from_dict(data)
        assert info.backend == GPUBackend.METAL


class TestCapabilityScorerBasic:
    """Test basic CapabilityScorer functionality."""

    def test_score_range(self):
        """Test score is in valid range."""
        scorer = CapabilityScorer()
        score = scorer.score()
        assert 0.0 <= score <= 1.0

    def test_default_adapter_low_score(self):
        """Test default adapter has low-ish score."""
        scorer = CapabilityScorer()
        score = scorer.score()
        assert score < 0.7  # Unknown device type, no RT, etc.

    def test_adapter_info_property(self):
        """Test adapter_info property."""
        info = AdapterInfo(name="Test GPU")
        scorer = CapabilityScorer(info)
        assert scorer.adapter_info.name == "Test GPU"


class TestCapabilityScorerHighEnd:
    """Test scoring for high-end GPUs."""

    def test_discrete_rtx_high_score(self):
        """Test discrete GPU with RT gets high score."""
        info = AdapterInfo(
            name="NVIDIA RTX 4090",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
            features=FeatureFlags(
                ray_tracing=True,
                bindless=True,
                mesh_shader=True,
                multi_draw_indirect=True,
            ),
            limits=GPULimits(
                max_texture_dimension_2d=16384,
                max_compute_invocations_per_workgroup=1024,
                max_storage_buffer_binding_size=2147483648,
                max_bindings_per_bind_group=1000,
                max_storage_buffers_per_shader_stage=8,
            ),
        )
        scorer = CapabilityScorer(info)
        score = scorer.score()
        assert score >= 0.85  # Should be ULTRA tier

    def test_discrete_no_rt_medium_high_score(self):
        """Test discrete GPU without RT gets medium-high score."""
        info = AdapterInfo(
            name="GTX 1080",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
            features=FeatureFlags(
                ray_tracing=False,
                bindless=False,
                multi_draw_indirect=True,
            ),
            limits=GPULimits(
                max_texture_dimension_2d=16384,
                max_storage_buffer_binding_size=1073741824,
            ),
        )
        scorer = CapabilityScorer(info)
        score = scorer.score()
        assert 0.5 <= score < 0.85


class TestCapabilityScorerLowEnd:
    """Test scoring for low-end GPUs."""

    def test_integrated_gpu_low_score(self):
        """Test integrated GPU gets lower score."""
        info = AdapterInfo(
            name="Intel UHD 630",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.INTEGRATED,
            features=FeatureFlags(
                ray_tracing=False,
                bindless=False,
            ),
            limits=GPULimits(
                max_texture_dimension_2d=4096,
                max_storage_buffer_binding_size=134217728,
            ),
        )
        scorer = CapabilityScorer(info)
        score = scorer.score()
        assert 0.25 <= score < 0.6

    def test_gles_mobile_very_low_score(self):
        """Test GLES mobile GPU gets low score."""
        info = AdapterInfo(
            name="Adreno 650",
            backend=GPUBackend.OPENGLES,
            device_type=GPUDeviceType.INTEGRATED,
            features=FeatureFlags(
                ray_tracing=False,
                compute_shader=True,
                texture_compression_etc2=True,
            ),
            limits=GPULimits(
                max_texture_dimension_2d=4096,
                max_storage_buffer_binding_size=67108864,
            ),
        )
        scorer = CapabilityScorer(info)
        score = scorer.score()
        assert score < 0.5  # LOW/MEDIUM tier boundary


class TestCapabilityScorerExplain:
    """Test score explanation functionality."""

    def test_explain_contains_components(self):
        """Test explain returns all score components."""
        scorer = CapabilityScorer()
        breakdown = scorer.explain()
        assert "device_type" in breakdown
        assert "features" in breakdown
        assert "limits" in breakdown
        assert "backend" in breakdown
        assert "total" in breakdown

    def test_explain_total_matches_score(self):
        """Test explain total matches computed score."""
        scorer = CapabilityScorer()
        breakdown = scorer.explain()
        assert abs(breakdown["total"] - scorer.score()) < 0.001

    def test_explain_components_positive(self):
        """Test all score components are non-negative."""
        info = AdapterInfo(
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
        )
        scorer = CapabilityScorer(info)
        breakdown = scorer.explain()
        for key, value in breakdown.items():
            assert value >= 0.0, f"{key} should be non-negative"
