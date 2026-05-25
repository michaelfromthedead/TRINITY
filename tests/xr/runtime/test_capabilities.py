"""Tests for XR device capability detection and queries."""
from __future__ import annotations

import pytest
import sys

sys.path.insert(0, "/home/user/dev/AI_GAME_ENGINE")

from engine.xr.runtime.capabilities import (
    XRFeature,
    XRCapabilities,
    DisplaySpecs,
    TrackingCapabilities,
    RenderingCapabilities,
    InputCapabilities,
    detect_capabilities,
    create_fallback_capabilities,
)


class TestXRFeature:
    """Tests for XRFeature enumeration."""

    def test_feature_values_are_unique(self):
        """Verify all feature enum values are unique."""
        values = [f.value for f in XRFeature]
        assert len(values) == len(set(values))

    def test_common_features_exist(self):
        """Verify common XR features are defined."""
        assert XRFeature.HEAD_TRACKING
        assert XRFeature.HAND_TRACKING
        assert XRFeature.EYE_TRACKING
        assert XRFeature.FOVEATED_RENDERING
        assert XRFeature.PASSTHROUGH
        assert XRFeature.SPATIAL_ANCHORS


class TestDisplaySpecs:
    """Tests for DisplaySpecs dataclass."""

    def test_default_values(self):
        """Verify default display specs are reasonable."""
        specs = DisplaySpecs()
        assert specs.resolution_per_eye == (1920, 1920)
        assert specs.refresh_rate == 90.0
        assert specs.field_of_view_horizontal == 90.0
        assert specs.field_of_view_vertical == 90.0

    def test_custom_values(self):
        """Verify custom display specs are stored correctly."""
        specs = DisplaySpecs(
            resolution_per_eye=(2160, 2160),
            refresh_rate=120.0,
            supported_refresh_rates=(90.0, 120.0),
            field_of_view_horizontal=110.0,
            panel_type="OLED",
        )
        assert specs.resolution_per_eye == (2160, 2160)
        assert specs.refresh_rate == 120.0
        assert 120.0 in specs.supported_refresh_rates
        assert specs.panel_type == "OLED"

    def test_frozen_immutability(self):
        """Verify DisplaySpecs is immutable."""
        specs = DisplaySpecs()
        with pytest.raises(AttributeError):
            specs.refresh_rate = 120.0


class TestTrackingCapabilities:
    """Tests for TrackingCapabilities dataclass."""

    def test_default_values(self):
        """Verify default tracking capabilities."""
        caps = TrackingCapabilities()
        assert caps.tracking_frequency == 250.0
        assert caps.supports_guardian is True

    def test_custom_values(self):
        """Verify custom tracking capabilities."""
        caps = TrackingCapabilities(
            tracking_frequency=1000.0,
            position_accuracy=0.0005,
            supports_guardian=False,
        )
        assert caps.tracking_frequency == 1000.0
        assert caps.position_accuracy == 0.0005
        assert caps.supports_guardian is False


class TestRenderingCapabilities:
    """Tests for RenderingCapabilities dataclass."""

    def test_default_values(self):
        """Verify default rendering capabilities."""
        caps = RenderingCapabilities()
        assert caps.max_render_scale == 2.0
        assert caps.supports_multiview is True
        assert caps.supports_async_timewarp is True

    def test_foveation_levels(self):
        """Verify foveation level range."""
        caps = RenderingCapabilities(max_foveation_level=4)
        assert caps.max_foveation_level >= 0
        assert caps.max_foveation_level <= 4


class TestInputCapabilities:
    """Tests for InputCapabilities dataclass."""

    def test_default_values(self):
        """Verify default input capabilities."""
        caps = InputCapabilities()
        assert caps.controller_count == 2
        assert caps.hand_joint_count == 26

    def test_haptic_range(self):
        """Verify haptic frequency range."""
        caps = InputCapabilities()
        assert caps.haptic_frequency_range[0] < caps.haptic_frequency_range[1]


class TestXRCapabilities:
    """Tests for XRCapabilities aggregation class."""

    def test_default_values(self):
        """Verify default capabilities."""
        caps = XRCapabilities()
        assert caps.device_name == "Unknown XR Device"
        assert caps.vendor == "Unknown"
        assert len(caps.features) == 0

    def test_supports_single_feature(self):
        """Verify supports() works for single features."""
        caps = XRCapabilities(
            features=frozenset({XRFeature.HEAD_TRACKING, XRFeature.HAND_TRACKING})
        )
        assert caps.supports(XRFeature.HEAD_TRACKING) is True
        assert caps.supports(XRFeature.HAND_TRACKING) is True
        assert caps.supports(XRFeature.EYE_TRACKING) is False

    def test_supports_all(self):
        """Verify supports_all() checks all features."""
        caps = XRCapabilities(
            features=frozenset({
                XRFeature.HEAD_TRACKING,
                XRFeature.HAND_TRACKING,
                XRFeature.EYE_TRACKING,
            })
        )
        assert caps.supports_all(XRFeature.HEAD_TRACKING, XRFeature.HAND_TRACKING) is True
        assert caps.supports_all(XRFeature.HEAD_TRACKING, XRFeature.PASSTHROUGH) is False

    def test_supports_any(self):
        """Verify supports_any() checks any feature."""
        caps = XRCapabilities(
            features=frozenset({XRFeature.HEAD_TRACKING})
        )
        assert caps.supports_any(XRFeature.HEAD_TRACKING, XRFeature.HAND_TRACKING) is True
        assert caps.supports_any(XRFeature.EYE_TRACKING, XRFeature.PASSTHROUGH) is False

    def test_get_missing_features(self):
        """Verify get_missing_features() returns correct set."""
        caps = XRCapabilities(
            features=frozenset({XRFeature.HEAD_TRACKING})
        )
        missing = caps.get_missing_features(
            XRFeature.HEAD_TRACKING,
            XRFeature.HAND_TRACKING,
            XRFeature.EYE_TRACKING,
        )
        assert XRFeature.HEAD_TRACKING not in missing
        assert XRFeature.HAND_TRACKING in missing
        assert XRFeature.EYE_TRACKING in missing

    def test_convenience_properties(self):
        """Verify convenience property methods."""
        caps = XRCapabilities(
            features=frozenset({
                XRFeature.HAND_TRACKING,
                XRFeature.PASSTHROUGH,
            })
        )
        assert caps.has_hand_tracking is True
        assert caps.has_eye_tracking is False
        assert caps.has_passthrough is True
        assert caps.has_foveated_rendering is False


class TestDetectCapabilities:
    """Tests for detect_capabilities() function."""

    def test_openxr_capabilities(self):
        """Verify OpenXR capability detection."""
        caps = detect_capabilities("OpenXR")
        assert caps.device_name == "OpenXR Device"
        assert caps.supports(XRFeature.HEAD_TRACKING)
        assert caps.supports(XRFeature.HAND_TRACKING)

    def test_webxr_capabilities(self):
        """Verify WebXR capability detection."""
        caps = detect_capabilities("WebXR")
        assert caps.device_name == "WebXR Device"
        assert caps.supports(XRFeature.HEAD_TRACKING)
        # WebXR typically has more limited features
        assert caps.rendering.supports_multiview is False

    def test_steamvr_capabilities(self):
        """Verify SteamVR capability detection."""
        caps = detect_capabilities("SteamVR")
        assert caps.vendor == "Valve"
        assert caps.supports(XRFeature.HEAD_TRACKING)

    def test_oculus_capabilities(self):
        """Verify Oculus/Meta capability detection."""
        caps = detect_capabilities("Oculus")
        assert caps.vendor == "Meta"
        assert caps.supports(XRFeature.HAND_TRACKING)
        assert caps.supports(XRFeature.EYE_TRACKING)
        assert caps.supports(XRFeature.PASSTHROUGH)
        assert caps.supports(XRFeature.SCENE_UNDERSTANDING)

    def test_unknown_runtime_fallback(self):
        """Verify unknown runtime returns generic capabilities."""
        caps = detect_capabilities("UnknownRuntime")
        assert caps.device_name == "Generic XR Device"
        assert caps.supports(XRFeature.HEAD_TRACKING)

    def test_case_insensitive(self):
        """Verify runtime name matching is case-insensitive."""
        caps_lower = detect_capabilities("openxr")
        caps_upper = detect_capabilities("OPENXR")
        caps_mixed = detect_capabilities("OpenXR")

        assert caps_lower.device_name == caps_upper.device_name
        assert caps_lower.device_name == caps_mixed.device_name


class TestCreateFallbackCapabilities:
    """Tests for create_fallback_capabilities() function."""

    def test_fallback_is_minimal(self):
        """Verify fallback capabilities are minimal."""
        caps = create_fallback_capabilities()
        assert caps.device_name == "Fallback Device"
        assert len(caps.features) == 1
        assert XRFeature.HEAD_TRACKING in caps.features

    def test_fallback_has_no_advanced_features(self):
        """Verify fallback has no advanced features."""
        caps = create_fallback_capabilities()
        assert not caps.has_hand_tracking
        assert not caps.has_eye_tracking
        assert not caps.has_passthrough
        assert not caps.has_foveated_rendering

    def test_fallback_rendering_is_limited(self):
        """Verify fallback rendering is limited."""
        caps = create_fallback_capabilities()
        assert caps.rendering.max_render_scale == 1.0
        assert caps.rendering.supports_multiview is False
        assert caps.rendering.max_foveation_level == 0

    def test_fallback_input_is_minimal(self):
        """Verify fallback input is minimal."""
        caps = create_fallback_capabilities()
        assert caps.input.controller_count == 0
        assert caps.input.hand_joint_count == 0
