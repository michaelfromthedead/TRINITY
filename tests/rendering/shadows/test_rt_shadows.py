"""
Tests for RT Shadow Dispatch System

Tests cover:
- RTShadowQuality enum values and behavior
- RTShadowParams validation and methods
- RTShadowDispatcher creation and device capability check
- Ray count per quality level calculations
- ShadowFallbackDispatcher CSM and contact shadow dispatch
- Technique selection logic
- Factory functions and cost estimation
- Error handling and edge cases
"""

from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, Mock, patch
from typing import Any

from engine.rendering.shadows.rt_shadows import (
    RTShadowQuality,
    RTShadowParams,
    RTShadowDispatcher,
    ShadowFallbackDispatcher,
    ShadowTechnique,
    create_shadow_dispatcher,
    estimate_shadow_cost,
    create_default_params,
    create_quality_params,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_texture():
    """Create a mock texture with valid state."""
    tex = MagicMock()
    tex.is_valid.return_value = True
    tex.desc.width = 1920
    tex.desc.height = 1080
    return tex


@pytest.fixture
def mock_texture_small():
    """Create a small mock texture."""
    tex = MagicMock()
    tex.is_valid.return_value = True
    tex.desc.width = 640
    tex.desc.height = 480
    return tex


@pytest.fixture
def mock_invalid_texture():
    """Create an invalid mock texture."""
    tex = MagicMock()
    tex.is_valid.return_value = False
    return tex


@pytest.fixture
def mock_buffer():
    """Create a mock buffer."""
    buf = MagicMock()
    buf.is_valid.return_value = True
    return buf


@pytest.fixture
def mock_tlas():
    """Create a mock TLAS handle."""
    tlas = MagicMock()
    tlas.handle_id = 42
    return tlas


@pytest.fixture
def mock_device_with_rt():
    """Create a mock device that supports RT."""
    device = MagicMock()
    device._adapter.query_features.return_value = MagicMock(ray_tracing=True)
    return device


@pytest.fixture
def mock_device_without_rt():
    """Create a mock device that does not support RT."""
    device = MagicMock()
    device._adapter.query_features.return_value = MagicMock(ray_tracing=False)
    return device


@pytest.fixture
def mock_device_no_adapter():
    """Create a mock device without adapter attribute."""
    device = MagicMock(spec=[])
    return device


# =============================================================================
# RTShadowQuality Tests
# =============================================================================


class TestRTShadowQuality:
    """Tests for RTShadowQuality enum."""

    def test_quality_low_value(self):
        """LOW quality should be 1 ray per pixel."""
        assert RTShadowQuality.LOW == 1
        assert int(RTShadowQuality.LOW) == 1

    def test_quality_medium_value(self):
        """MEDIUM quality should be 2 rays per pixel."""
        assert RTShadowQuality.MEDIUM == 2
        assert int(RTShadowQuality.MEDIUM) == 2

    def test_quality_high_value(self):
        """HIGH quality should be 4 rays per pixel."""
        assert RTShadowQuality.HIGH == 4
        assert int(RTShadowQuality.HIGH) == 4

    def test_quality_ultra_value(self):
        """ULTRA quality should be 8 rays per pixel."""
        assert RTShadowQuality.ULTRA == 8
        assert int(RTShadowQuality.ULTRA) == 8

    def test_quality_ordering(self):
        """Quality levels should be ordered LOW < MEDIUM < HIGH < ULTRA."""
        assert RTShadowQuality.LOW < RTShadowQuality.MEDIUM
        assert RTShadowQuality.MEDIUM < RTShadowQuality.HIGH
        assert RTShadowQuality.HIGH < RTShadowQuality.ULTRA

    def test_quality_is_intenum(self):
        """Quality should be usable as integer."""
        total = RTShadowQuality.LOW + RTShadowQuality.MEDIUM
        assert total == 3

    def test_quality_iteration(self):
        """Should be able to iterate over quality levels."""
        qualities = list(RTShadowQuality)
        assert len(qualities) == 4
        assert RTShadowQuality.LOW in qualities
        assert RTShadowQuality.ULTRA in qualities


# =============================================================================
# RTShadowParams Tests
# =============================================================================


class TestRTShadowParams:
    """Tests for RTShadowParams dataclass."""

    def test_default_params(self):
        """Default parameters should have sensible values."""
        params = RTShadowParams()
        assert params.quality == RTShadowQuality.MEDIUM
        assert params.max_distance == 1000.0
        assert params.bias == 0.001
        assert params.alpha_test_enabled is True
        assert params.alpha_cutoff == 0.5

    def test_custom_quality(self):
        """Should accept custom quality level."""
        params = RTShadowParams(quality=RTShadowQuality.ULTRA)
        assert params.quality == RTShadowQuality.ULTRA

    def test_custom_max_distance(self):
        """Should accept custom max distance."""
        params = RTShadowParams(max_distance=500.0)
        assert params.max_distance == 500.0

    def test_custom_bias(self):
        """Should accept custom bias value."""
        params = RTShadowParams(bias=0.005)
        assert params.bias == 0.005

    def test_alpha_test_disabled(self):
        """Should allow disabling alpha test."""
        params = RTShadowParams(alpha_test_enabled=False)
        assert params.alpha_test_enabled is False

    def test_custom_alpha_cutoff(self):
        """Should accept custom alpha cutoff."""
        params = RTShadowParams(alpha_cutoff=0.25)
        assert params.alpha_cutoff == 0.25

    def test_invalid_quality_type(self):
        """Should reject non-RTShadowQuality quality values."""
        with pytest.raises(TypeError, match="quality must be RTShadowQuality"):
            RTShadowParams(quality=2)  # type: ignore

    def test_invalid_max_distance_zero(self):
        """Should reject zero max_distance."""
        with pytest.raises(ValueError, match="max_distance must be positive"):
            RTShadowParams(max_distance=0.0)

    def test_invalid_max_distance_negative(self):
        """Should reject negative max_distance."""
        with pytest.raises(ValueError, match="max_distance must be positive"):
            RTShadowParams(max_distance=-100.0)

    def test_invalid_bias_negative(self):
        """Should reject negative bias."""
        with pytest.raises(ValueError, match="bias must be non-negative"):
            RTShadowParams(bias=-0.001)

    def test_bias_zero_valid(self):
        """Zero bias should be valid."""
        params = RTShadowParams(bias=0.0)
        assert params.bias == 0.0

    def test_invalid_alpha_cutoff_negative(self):
        """Should reject negative alpha cutoff."""
        with pytest.raises(ValueError, match="alpha_cutoff must be in"):
            RTShadowParams(alpha_cutoff=-0.1)

    def test_invalid_alpha_cutoff_over_one(self):
        """Should reject alpha cutoff > 1.0."""
        with pytest.raises(ValueError, match="alpha_cutoff must be in"):
            RTShadowParams(alpha_cutoff=1.5)

    def test_alpha_cutoff_boundary_zero(self):
        """Alpha cutoff of 0.0 should be valid."""
        params = RTShadowParams(alpha_cutoff=0.0)
        assert params.alpha_cutoff == 0.0

    def test_alpha_cutoff_boundary_one(self):
        """Alpha cutoff of 1.0 should be valid."""
        params = RTShadowParams(alpha_cutoff=1.0)
        assert params.alpha_cutoff == 1.0

    def test_get_shader_params(self):
        """get_shader_params should return tuple of shader values."""
        params = RTShadowParams(max_distance=500.0, bias=0.002, alpha_cutoff=0.75)
        shader_params = params.get_shader_params()
        assert shader_params == (500.0, 0.002, 0.75)
        assert len(shader_params) == 3

    def test_with_quality(self):
        """with_quality should create copy with new quality."""
        original = RTShadowParams(quality=RTShadowQuality.LOW, max_distance=800.0)
        updated = original.with_quality(RTShadowQuality.ULTRA)

        assert updated.quality == RTShadowQuality.ULTRA
        assert updated.max_distance == 800.0  # Preserved
        assert original.quality == RTShadowQuality.LOW  # Unchanged


# =============================================================================
# RTShadowDispatcher Tests
# =============================================================================


class TestRTShadowDispatcher:
    """Tests for RTShadowDispatcher class."""

    def test_create_dispatcher(self, mock_device_with_rt):
        """Should create dispatcher with device."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)
        assert dispatcher.device is mock_device_with_rt

    def test_is_initialized_false_initially(self, mock_device_with_rt):
        """Dispatcher should not be initialized before first dispatch."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)
        assert dispatcher.is_initialized is False

    def test_supports_rt_true(self, mock_device_with_rt):
        """supports_rt should return True when device has RT."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)
        assert dispatcher.supports_rt() is True

    def test_supports_rt_false(self, mock_device_without_rt):
        """supports_rt should return False when device lacks RT."""
        dispatcher = RTShadowDispatcher(mock_device_without_rt)
        assert dispatcher.supports_rt() is False

    def test_supports_rt_no_adapter(self, mock_device_no_adapter):
        """supports_rt should return False when adapter missing."""
        dispatcher = RTShadowDispatcher(mock_device_no_adapter)
        assert dispatcher.supports_rt() is False

    def test_supports_rt_cached(self, mock_device_with_rt):
        """supports_rt result should be cached."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)

        # First call
        result1 = dispatcher.supports_rt()
        # Change the mock
        mock_device_with_rt._adapter.query_features.return_value = MagicMock(ray_tracing=False)
        # Second call should return cached value
        result2 = dispatcher.supports_rt()

        assert result1 is True
        assert result2 is True  # Cached

    def test_get_ray_count_per_pixel(self, mock_device_with_rt):
        """get_ray_count_per_pixel should return quality value."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)

        assert dispatcher.get_ray_count_per_pixel(RTShadowQuality.LOW) == 1
        assert dispatcher.get_ray_count_per_pixel(RTShadowQuality.MEDIUM) == 2
        assert dispatcher.get_ray_count_per_pixel(RTShadowQuality.HIGH) == 4
        assert dispatcher.get_ray_count_per_pixel(RTShadowQuality.ULTRA) == 8

    def test_get_total_ray_count(self, mock_device_with_rt):
        """get_total_ray_count should compute width * height * rays."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)

        # 1920x1080 at MEDIUM (2 rays)
        total = dispatcher.get_total_ray_count(RTShadowQuality.MEDIUM, 1920, 1080)
        assert total == 1920 * 1080 * 2

    def test_dispatch_shadows_no_rt_support(
        self, mock_device_without_rt, mock_tlas, mock_texture, mock_buffer
    ):
        """dispatch_shadows should raise when RT not supported."""
        dispatcher = RTShadowDispatcher(mock_device_without_rt)

        with pytest.raises(RuntimeError, match="not supported"):
            dispatcher.dispatch_shadows(
                mock_tlas, mock_texture, mock_texture, mock_buffer, mock_texture
            )

    def test_dispatch_shadows_null_tlas(
        self, mock_device_with_rt, mock_texture, mock_buffer
    ):
        """dispatch_shadows should raise when tlas is None."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)

        with pytest.raises(ValueError, match="tlas cannot be None"):
            dispatcher.dispatch_shadows(
                None, mock_texture, mock_texture, mock_buffer, mock_texture
            )

    def test_dispatch_shadows_invalid_depth(
        self, mock_device_with_rt, mock_tlas, mock_invalid_texture, mock_texture, mock_buffer
    ):
        """dispatch_shadows should raise when depth buffer invalid."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)

        with pytest.raises(ValueError, match="depth_buffer is invalid"):
            dispatcher.dispatch_shadows(
                mock_tlas, mock_invalid_texture, mock_texture, mock_buffer, mock_texture
            )

    def test_dispatch_shadows_invalid_normal(
        self, mock_device_with_rt, mock_tlas, mock_texture, mock_invalid_texture, mock_buffer
    ):
        """dispatch_shadows should raise when normal buffer invalid."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)

        with pytest.raises(ValueError, match="normal_buffer is invalid"):
            dispatcher.dispatch_shadows(
                mock_tlas, mock_texture, mock_invalid_texture, mock_buffer, mock_texture
            )

    def test_dispatch_shadows_invalid_output(
        self, mock_device_with_rt, mock_tlas, mock_texture, mock_invalid_texture, mock_buffer
    ):
        """dispatch_shadows should raise when output texture invalid."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)

        with pytest.raises(ValueError, match="output texture is invalid"):
            dispatcher.dispatch_shadows(
                mock_tlas, mock_texture, mock_texture, mock_buffer, mock_invalid_texture
            )

    def test_dispatch_shadows_dimension_mismatch(
        self, mock_device_with_rt, mock_tlas, mock_texture, mock_texture_small, mock_buffer
    ):
        """dispatch_shadows should raise when dimensions mismatch."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)

        with pytest.raises(ValueError, match="do not match"):
            dispatcher.dispatch_shadows(
                mock_tlas, mock_texture, mock_texture, mock_buffer, mock_texture_small
            )

    def test_dispatch_shadows_success(
        self, mock_device_with_rt, mock_tlas, mock_texture, mock_buffer
    ):
        """dispatch_shadows should succeed with valid inputs."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)
        params = RTShadowParams(quality=RTShadowQuality.HIGH)

        # Should not raise
        dispatcher.dispatch_shadows(
            mock_tlas, mock_texture, mock_texture, mock_buffer, mock_texture, params
        )

        assert dispatcher.is_initialized is True

    def test_dispatch_shadows_default_params(
        self, mock_device_with_rt, mock_tlas, mock_texture, mock_buffer
    ):
        """dispatch_shadows should use default params when None."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)

        # Pass None for params
        dispatcher.dispatch_shadows(
            mock_tlas, mock_texture, mock_texture, mock_buffer, mock_texture, None
        )

        assert dispatcher.is_initialized is True

    def test_destroy(self, mock_device_with_rt, mock_tlas, mock_texture, mock_buffer):
        """destroy should reset initialized state."""
        dispatcher = RTShadowDispatcher(mock_device_with_rt)

        dispatcher.dispatch_shadows(
            mock_tlas, mock_texture, mock_texture, mock_buffer, mock_texture
        )
        assert dispatcher.is_initialized is True

        dispatcher.destroy()
        assert dispatcher.is_initialized is False


# =============================================================================
# ShadowFallbackDispatcher Tests
# =============================================================================


class TestShadowFallbackDispatcher:
    """Tests for ShadowFallbackDispatcher class."""

    def test_create_fallback_dispatcher(self, mock_device_without_rt):
        """Should create fallback dispatcher."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)
        assert dispatcher.device is mock_device_without_rt

    def test_default_cascade_count(self, mock_device_without_rt):
        """Default cascade count should be 4."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)
        assert dispatcher.cascade_count == 4

    def test_set_cascade_count_valid(self, mock_device_without_rt):
        """Should accept valid cascade count."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)
        dispatcher.cascade_count = 6
        assert dispatcher.cascade_count == 6

    def test_set_cascade_count_invalid_low(self, mock_device_without_rt):
        """Should reject cascade count < 1."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)
        with pytest.raises(ValueError, match="must be in"):
            dispatcher.cascade_count = 0

    def test_set_cascade_count_invalid_high(self, mock_device_without_rt):
        """Should reject cascade count > 8."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)
        with pytest.raises(ValueError, match="must be in"):
            dispatcher.cascade_count = 9

    def test_configure_csm(self, mock_device_without_rt):
        """configure_csm should update cascade settings."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)
        dispatcher.configure_csm(cascade_count=3, cascade_splits=[0.1, 0.4, 1.0])
        assert dispatcher.cascade_count == 3

    def test_configure_csm_splits_mismatch(self, mock_device_without_rt):
        """configure_csm should reject mismatched splits length."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)
        with pytest.raises(ValueError, match="must match cascade_count"):
            dispatcher.configure_csm(cascade_count=4, cascade_splits=[0.1, 0.5, 1.0])

    def test_configure_csm_splits_not_monotonic(self, mock_device_without_rt):
        """configure_csm should reject non-monotonic splits."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)
        with pytest.raises(ValueError, match="monotonically increasing"):
            dispatcher.configure_csm(cascade_count=4, cascade_splits=[0.5, 0.3, 0.7, 1.0])

    def test_dispatch_csm_success(self, mock_device_without_rt, mock_texture):
        """dispatch_csm should succeed with valid inputs."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)

        shadow_maps = [mock_texture for _ in range(4)]

        dispatcher.dispatch_csm(
            scene_bounds=((0, 0, 0), (100, 100, 100)),
            light_direction=(0.5, -0.7, 0.5),
            view_matrix=[1] * 16,
            projection_matrix=[1] * 16,
            shadow_maps=shadow_maps,
            output=mock_texture,
        )

    def test_dispatch_csm_wrong_shadow_map_count(
        self, mock_device_without_rt, mock_texture
    ):
        """dispatch_csm should reject wrong shadow map count."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)

        shadow_maps = [mock_texture for _ in range(3)]  # Wrong count

        with pytest.raises(ValueError, match="must match cascade_count"):
            dispatcher.dispatch_csm(
                scene_bounds=((0, 0, 0), (100, 100, 100)),
                light_direction=(0.5, -0.7, 0.5),
                view_matrix=[1] * 16,
                projection_matrix=[1] * 16,
                shadow_maps=shadow_maps,
                output=mock_texture,
            )

    def test_dispatch_csm_invalid_shadow_map(
        self, mock_device_without_rt, mock_texture, mock_invalid_texture
    ):
        """dispatch_csm should reject invalid shadow map."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)

        shadow_maps = [mock_texture, mock_invalid_texture, mock_texture, mock_texture]

        with pytest.raises(ValueError, match="shadow_maps"):
            dispatcher.dispatch_csm(
                scene_bounds=((0, 0, 0), (100, 100, 100)),
                light_direction=(0.5, -0.7, 0.5),
                view_matrix=[1] * 16,
                projection_matrix=[1] * 16,
                shadow_maps=shadow_maps,
                output=mock_texture,
            )

    def test_dispatch_csm_invalid_output(
        self, mock_device_without_rt, mock_texture, mock_invalid_texture
    ):
        """dispatch_csm should reject invalid output texture."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)

        shadow_maps = [mock_texture for _ in range(4)]

        with pytest.raises(ValueError, match="output texture is invalid"):
            dispatcher.dispatch_csm(
                scene_bounds=((0, 0, 0), (100, 100, 100)),
                light_direction=(0.5, -0.7, 0.5),
                view_matrix=[1] * 16,
                projection_matrix=[1] * 16,
                shadow_maps=shadow_maps,
                output=mock_invalid_texture,
            )

    def test_dispatch_contact_shadows_success(
        self, mock_device_without_rt, mock_texture
    ):
        """dispatch_contact_shadows should succeed with valid inputs."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)

        dispatcher.dispatch_contact_shadows(
            depth_buffer=mock_texture,
            normal_buffer=mock_texture,
            light_direction=(0.5, -0.7, 0.5),
            output=mock_texture,
        )

    def test_dispatch_contact_shadows_custom_params(
        self, mock_device_without_rt, mock_texture
    ):
        """dispatch_contact_shadows should accept custom parameters."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)

        dispatcher.dispatch_contact_shadows(
            depth_buffer=mock_texture,
            normal_buffer=mock_texture,
            light_direction=(0.5, -0.7, 0.5),
            output=mock_texture,
            ray_length=0.25,
            step_count=32,
        )

    def test_dispatch_contact_shadows_invalid_ray_length_zero(
        self, mock_device_without_rt, mock_texture
    ):
        """dispatch_contact_shadows should reject zero ray_length."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)

        with pytest.raises(ValueError, match="ray_length must be in"):
            dispatcher.dispatch_contact_shadows(
                depth_buffer=mock_texture,
                normal_buffer=mock_texture,
                light_direction=(0.5, -0.7, 0.5),
                output=mock_texture,
                ray_length=0.0,
            )

    def test_dispatch_contact_shadows_invalid_ray_length_over(
        self, mock_device_without_rt, mock_texture
    ):
        """dispatch_contact_shadows should reject ray_length > 1.0."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)

        with pytest.raises(ValueError, match="ray_length must be in"):
            dispatcher.dispatch_contact_shadows(
                depth_buffer=mock_texture,
                normal_buffer=mock_texture,
                light_direction=(0.5, -0.7, 0.5),
                output=mock_texture,
                ray_length=1.5,
            )

    def test_dispatch_contact_shadows_invalid_step_count_low(
        self, mock_device_without_rt, mock_texture
    ):
        """dispatch_contact_shadows should reject step_count < 1."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)

        with pytest.raises(ValueError, match="step_count must be in"):
            dispatcher.dispatch_contact_shadows(
                depth_buffer=mock_texture,
                normal_buffer=mock_texture,
                light_direction=(0.5, -0.7, 0.5),
                output=mock_texture,
                step_count=0,
            )

    def test_dispatch_contact_shadows_invalid_step_count_high(
        self, mock_device_without_rt, mock_texture
    ):
        """dispatch_contact_shadows should reject step_count > 64."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)

        with pytest.raises(ValueError, match="step_count must be in"):
            dispatcher.dispatch_contact_shadows(
                depth_buffer=mock_texture,
                normal_buffer=mock_texture,
                light_direction=(0.5, -0.7, 0.5),
                output=mock_texture,
                step_count=100,
            )

    def test_dispatch_contact_shadows_invalid_depth(
        self, mock_device_without_rt, mock_texture, mock_invalid_texture
    ):
        """dispatch_contact_shadows should reject invalid depth buffer."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)

        with pytest.raises(ValueError, match="depth_buffer is invalid"):
            dispatcher.dispatch_contact_shadows(
                depth_buffer=mock_invalid_texture,
                normal_buffer=mock_texture,
                light_direction=(0.5, -0.7, 0.5),
                output=mock_texture,
            )

    def test_dispatch_contact_shadows_invalid_output(
        self, mock_device_without_rt, mock_texture, mock_invalid_texture
    ):
        """dispatch_contact_shadows should reject invalid output."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)

        with pytest.raises(ValueError, match="output texture is invalid"):
            dispatcher.dispatch_contact_shadows(
                depth_buffer=mock_texture,
                normal_buffer=mock_texture,
                light_direction=(0.5, -0.7, 0.5),
                output=mock_invalid_texture,
            )

    def test_select_best_technique_rt_available(self, mock_device_without_rt):
        """select_best_technique should prefer RT when available."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)
        technique = dispatcher.select_best_technique(rt_available=True)
        assert technique == ShadowTechnique.RT_SHADOWS

    def test_select_best_technique_rt_unavailable(self, mock_device_without_rt):
        """select_best_technique should fallback to CSM when RT unavailable."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)
        technique = dispatcher.select_best_technique(rt_available=False)
        assert technique == ShadowTechnique.CSM

    def test_get_supported_techniques(self, mock_device_without_rt):
        """get_supported_techniques should return fallback techniques."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)
        techniques = dispatcher.get_supported_techniques()

        assert ShadowTechnique.CSM in techniques
        assert ShadowTechnique.CONTACT_SHADOWS in techniques
        assert ShadowTechnique.VSM in techniques
        assert ShadowTechnique.RT_SHADOWS not in techniques

    def test_destroy(self, mock_device_without_rt):
        """destroy should reset state."""
        dispatcher = ShadowFallbackDispatcher(mock_device_without_rt)
        dispatcher._csm_initialized = True
        dispatcher._contact_initialized = True

        dispatcher.destroy()

        assert dispatcher._csm_initialized is False
        assert dispatcher._contact_initialized is False


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateShadowDispatcher:
    """Tests for create_shadow_dispatcher factory function."""

    def test_returns_rt_dispatcher_when_supported(self, mock_device_with_rt):
        """Should return RTShadowDispatcher when RT supported."""
        dispatcher = create_shadow_dispatcher(mock_device_with_rt)
        assert isinstance(dispatcher, RTShadowDispatcher)

    def test_returns_fallback_dispatcher_when_not_supported(self, mock_device_without_rt):
        """Should return ShadowFallbackDispatcher when RT not supported."""
        dispatcher = create_shadow_dispatcher(mock_device_without_rt)
        assert isinstance(dispatcher, ShadowFallbackDispatcher)


# =============================================================================
# Cost Estimation Tests
# =============================================================================


class TestEstimateShadowCost:
    """Tests for estimate_shadow_cost function."""

    def test_rt_shadow_cost_basic(self):
        """estimate_shadow_cost should compute RT shadow metrics."""
        params = RTShadowParams(quality=RTShadowQuality.MEDIUM)
        resolution = (1920, 1080)

        cost = estimate_shadow_cost(params, resolution)

        assert cost["technique"] == ShadowTechnique.RT_SHADOWS
        assert cost["ray_count"] == 1920 * 1080 * 2
        assert cost["pixels"] == 1920 * 1080
        assert "memory_mb" in cost
        assert "relative_cost" in cost

    def test_rt_shadow_cost_quality_scales(self):
        """Higher quality should increase ray count."""
        resolution = (1920, 1080)

        low_cost = estimate_shadow_cost(
            RTShadowParams(quality=RTShadowQuality.LOW), resolution
        )
        ultra_cost = estimate_shadow_cost(
            RTShadowParams(quality=RTShadowQuality.ULTRA), resolution
        )

        assert ultra_cost["ray_count"] > low_cost["ray_count"]
        assert ultra_cost["relative_cost"] > low_cost["relative_cost"]

    def test_rt_shadow_cost_alpha_test_overhead(self):
        """Alpha testing should increase relative cost."""
        resolution = (1920, 1080)

        no_alpha = estimate_shadow_cost(
            RTShadowParams(alpha_test_enabled=False), resolution
        )
        with_alpha = estimate_shadow_cost(
            RTShadowParams(alpha_test_enabled=True), resolution
        )

        assert with_alpha["relative_cost"] > no_alpha["relative_cost"]

    def test_csm_cost(self):
        """CSM technique should have no ray count."""
        params = RTShadowParams()
        resolution = (1920, 1080)

        cost = estimate_shadow_cost(params, resolution, technique=ShadowTechnique.CSM)

        assert cost["technique"] == ShadowTechnique.CSM
        assert cost["ray_count"] == 0
        assert cost["relative_cost"] < 1.0  # Cheaper than RT

    def test_contact_shadow_cost(self):
        """Contact shadows should be cheapest."""
        params = RTShadowParams()
        resolution = (1920, 1080)

        cost = estimate_shadow_cost(
            params, resolution, technique=ShadowTechnique.CONTACT_SHADOWS
        )

        assert cost["ray_count"] == 0
        assert cost["relative_cost"] < 0.5


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_default_params(self):
        """create_default_params should return default RTShadowParams."""
        params = create_default_params()
        assert isinstance(params, RTShadowParams)
        assert params.quality == RTShadowQuality.MEDIUM

    def test_create_quality_params_low(self):
        """create_quality_params should configure for LOW quality."""
        params = create_quality_params(RTShadowQuality.LOW)
        assert params.quality == RTShadowQuality.LOW
        assert params.bias > 0.001  # Coarser bias for low quality

    def test_create_quality_params_ultra(self):
        """create_quality_params should configure for ULTRA quality."""
        params = create_quality_params(RTShadowQuality.ULTRA)
        assert params.quality == RTShadowQuality.ULTRA
        assert params.bias < 0.001  # Finer bias for high quality


# =============================================================================
# Integration-Style Tests
# =============================================================================


class TestIntegration:
    """Integration-style tests using mock TLAS."""

    def test_full_rt_shadow_workflow(
        self, mock_device_with_rt, mock_tlas, mock_texture, mock_buffer
    ):
        """Test complete RT shadow dispatch workflow."""
        # Create dispatcher via factory
        dispatcher = create_shadow_dispatcher(mock_device_with_rt)
        assert isinstance(dispatcher, RTShadowDispatcher)

        # Check RT support
        assert dispatcher.supports_rt()

        # Create params
        params = create_quality_params(RTShadowQuality.HIGH)
        assert params.quality == RTShadowQuality.HIGH

        # Estimate cost
        cost = estimate_shadow_cost(params, (1920, 1080))
        assert cost["ray_count"] == 1920 * 1080 * 4

        # Dispatch shadows
        dispatcher.dispatch_shadows(
            mock_tlas, mock_texture, mock_texture, mock_buffer, mock_texture, params
        )

        # Verify initialized
        assert dispatcher.is_initialized

    def test_fallback_workflow(self, mock_device_without_rt, mock_texture):
        """Test fallback shadow workflow."""
        # Create dispatcher via factory (should be fallback)
        dispatcher = create_shadow_dispatcher(mock_device_without_rt)
        assert isinstance(dispatcher, ShadowFallbackDispatcher)

        # Select technique
        technique = dispatcher.select_best_technique(rt_available=False)
        assert technique == ShadowTechnique.CSM

        # Configure CSM
        dispatcher.configure_csm(cascade_count=3, cascade_splits=[0.1, 0.3, 1.0])
        assert dispatcher.cascade_count == 3

        # Dispatch CSM
        shadow_maps = [mock_texture for _ in range(3)]
        dispatcher.dispatch_csm(
            scene_bounds=((0, 0, 0), (100, 100, 100)),
            light_direction=(0.5, -0.7, 0.5),
            view_matrix=[1] * 16,
            projection_matrix=[1] * 16,
            shadow_maps=shadow_maps,
            output=mock_texture,
        )

        # Dispatch contact shadows as enhancement
        dispatcher.dispatch_contact_shadows(
            depth_buffer=mock_texture,
            normal_buffer=mock_texture,
            light_direction=(0.5, -0.7, 0.5),
            output=mock_texture,
        )
