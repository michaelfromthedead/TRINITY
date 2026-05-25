"""Tests for XR platform integration.

Tests cover:
- XRDevice enumeration
- XRPlatform abstract class
- Platform implementations (OpenXR, SteamVR, Quest, Vision Pro, PSVR2)
- Device detection
- Platform capabilities
"""

import pytest
from unittest.mock import MagicMock, patch

from engine.xr.platform.platform_integration import (
    XRDevice,
    XRPlatformType,
    XRRuntime,
    XRDeviceCapabilities,
    XRPlatformInfo,
    XRPlatform,
    OpenXRPlatform,
    SteamVRPlatform,
    MetaQuestPlatform,
    AppleVisionProPlatform,
    PSVR2Platform,
    detect_xr_platform,
    get_device_capabilities,
)


class TestXRDeviceEnum:
    """Tests for XRDevice enumeration."""

    def test_pc_vr_devices_exist(self) -> None:
        """Test that PC VR devices are defined."""
        assert XRDevice.VALVE_INDEX
        assert XRDevice.HTC_VIVE
        assert XRDevice.HTC_VIVE_PRO
        assert XRDevice.OCULUS_RIFT
        assert XRDevice.HP_REVERB_G2
        assert XRDevice.WMR_GENERIC

    def test_standalone_devices_exist(self) -> None:
        """Test that standalone VR devices are defined."""
        assert XRDevice.META_QUEST_2
        assert XRDevice.META_QUEST_3
        assert XRDevice.META_QUEST_PRO
        assert XRDevice.PICO_4

    def test_ar_devices_exist(self) -> None:
        """Test that AR devices are defined."""
        assert XRDevice.ARCORE_PHONE
        assert XRDevice.ARKIT_PHONE
        assert XRDevice.HOLOLENS_2
        assert XRDevice.MAGIC_LEAP_2
        assert XRDevice.APPLE_VISION_PRO

    def test_console_devices_exist(self) -> None:
        """Test that console VR devices are defined."""
        assert XRDevice.PSVR2

    def test_unknown_device_exists(self) -> None:
        """Test that unknown device type exists."""
        assert XRDevice.UNKNOWN


class TestXRPlatformType:
    """Tests for XRPlatformType enumeration."""

    def test_platform_types_exist(self) -> None:
        """Test that all platform types are defined."""
        assert XRPlatformType.PC_VR
        assert XRPlatformType.STANDALONE_VR
        assert XRPlatformType.MOBILE_AR
        assert XRPlatformType.AR_HEADSET
        assert XRPlatformType.CONSOLE_VR
        assert XRPlatformType.UNKNOWN


class TestXRDeviceCapabilities:
    """Tests for XRDeviceCapabilities dataclass."""

    def test_default_capabilities(self) -> None:
        """Test default capability values."""
        caps = XRDeviceCapabilities()

        assert caps.supports_6dof_head is True
        assert caps.supports_6dof_controllers is True
        assert caps.supports_hand_tracking is False
        assert caps.supports_eye_tracking is False
        assert caps.supports_passthrough is False
        assert caps.is_tethered is True
        assert caps.performance_tier == "high"

    def test_custom_capabilities(self) -> None:
        """Test custom capability values."""
        caps = XRDeviceCapabilities(
            supports_hand_tracking=True,
            supports_eye_tracking=True,
            supports_passthrough=True,
            display_resolution=(2064, 2208),
            display_refresh_rates=[72.0, 90.0, 120.0],
            is_tethered=False,
            battery_capacity_mah=5060,
            performance_tier="medium",
        )

        assert caps.supports_hand_tracking is True
        assert caps.supports_eye_tracking is True
        assert caps.supports_passthrough is True
        assert caps.display_resolution == (2064, 2208)
        assert 120.0 in caps.display_refresh_rates
        assert caps.is_tethered is False
        assert caps.battery_capacity_mah == 5060
        assert caps.performance_tier == "medium"


class TestXRPlatformInfo:
    """Tests for XRPlatformInfo dataclass."""

    def test_default_platform_info(self) -> None:
        """Test default platform info values."""
        info = XRPlatformInfo()

        assert info.device == XRDevice.UNKNOWN
        assert info.platform_type == XRPlatformType.UNKNOWN
        assert info.runtime == XRRuntime.UNKNOWN
        assert info.device_name == ""
        assert isinstance(info.capabilities, XRDeviceCapabilities)

    def test_custom_platform_info(self) -> None:
        """Test custom platform info values."""
        caps = XRDeviceCapabilities(supports_hand_tracking=True)
        info = XRPlatformInfo(
            device=XRDevice.META_QUEST_3,
            platform_type=XRPlatformType.STANDALONE_VR,
            runtime=XRRuntime.OCULUS_MOBILE,
            device_name="Meta Quest 3",
            device_manufacturer="Meta",
            capabilities=caps,
        )

        assert info.device == XRDevice.META_QUEST_3
        assert info.platform_type == XRPlatformType.STANDALONE_VR
        assert info.runtime == XRRuntime.OCULUS_MOBILE
        assert info.device_name == "Meta Quest 3"
        assert info.device_manufacturer == "Meta"
        assert info.capabilities.supports_hand_tracking is True


class TestOpenXRPlatform:
    """Tests for OpenXR platform implementation."""

    def test_initialization(self) -> None:
        """Test OpenXR platform initialization."""
        platform = OpenXRPlatform()
        assert platform.initialized is False

        result = platform.initialize()
        assert result is True
        assert platform.initialized is True

    def test_shutdown(self) -> None:
        """Test OpenXR platform shutdown."""
        platform = OpenXRPlatform()
        platform.initialize()
        platform.shutdown()
        assert platform.initialized is False

    def test_detect_device(self) -> None:
        """Test device detection."""
        platform = OpenXRPlatform()
        info = platform.detect_device()

        assert isinstance(info, XRPlatformInfo)
        assert info.runtime == XRRuntime.OPENXR

    def test_is_available(self) -> None:
        """Test availability check."""
        platform = OpenXRPlatform()
        # Default returns True for development
        assert platform.is_available() is True

    def test_event_subscription(self) -> None:
        """Test event subscription and emission."""
        platform = OpenXRPlatform()
        received_events = []

        def on_event(*args, **kwargs):
            received_events.append((args, kwargs))

        platform.subscribe("test_event", on_event)
        platform._emit_event("test_event", "arg1", key="value")

        assert len(received_events) == 1
        assert received_events[0] == (("arg1",), {"key": "value"})

    def test_event_unsubscription(self) -> None:
        """Test event unsubscription."""
        platform = OpenXRPlatform()
        received_events = []

        def on_event(*args, **kwargs):
            received_events.append(args)

        platform.subscribe("test_event", on_event)
        platform.unsubscribe("test_event", on_event)
        platform._emit_event("test_event", "arg1")

        assert len(received_events) == 0


class TestMetaQuestPlatform:
    """Tests for Meta Quest platform implementation."""

    def test_initialization(self) -> None:
        """Test Quest platform initialization."""
        platform = MetaQuestPlatform()
        result = platform.initialize()
        assert result is True
        assert platform.initialized is True

    def test_detect_device_capabilities(self) -> None:
        """Test Quest device capabilities."""
        platform = MetaQuestPlatform()
        info = platform.detect_device()

        assert info.device == XRDevice.META_QUEST_3
        assert info.platform_type == XRPlatformType.STANDALONE_VR
        assert info.runtime == XRRuntime.OCULUS_MOBILE
        assert info.device_manufacturer == "Meta"

        caps = info.capabilities
        assert caps.supports_hand_tracking is True
        assert caps.supports_passthrough is True
        assert caps.supports_color_passthrough is True
        assert caps.supports_depth_sensing is True
        assert caps.supports_plane_detection is True
        assert caps.supports_spatial_anchors is True
        assert caps.is_tethered is False

    def test_request_passthrough(self) -> None:
        """Test passthrough request."""
        platform = MetaQuestPlatform()
        platform.initialize()

        result = platform.request_passthrough(True)
        assert result is True


class TestAppleVisionProPlatform:
    """Tests for Apple Vision Pro platform implementation."""

    def test_detect_device_capabilities(self) -> None:
        """Test Vision Pro device capabilities."""
        platform = AppleVisionProPlatform()
        info = platform.detect_device()

        assert info.device == XRDevice.APPLE_VISION_PRO
        assert info.platform_type == XRPlatformType.AR_HEADSET
        assert info.runtime == XRRuntime.VISIONOS
        assert info.device_manufacturer == "Apple"

        caps = info.capabilities
        assert caps.supports_hand_tracking is True
        assert caps.supports_eye_tracking is True
        assert caps.supports_6dof_controllers is False  # Uses hands
        assert caps.controller_type == "hands"
        assert caps.supports_hdr is True
        assert caps.performance_tier == "ultra"


class TestPSVR2Platform:
    """Tests for PSVR2 platform implementation."""

    def test_detect_device_capabilities(self) -> None:
        """Test PSVR2 device capabilities."""
        platform = PSVR2Platform()
        info = platform.detect_device()

        assert info.device == XRDevice.PSVR2
        assert info.platform_type == XRPlatformType.CONSOLE_VR
        assert info.runtime == XRRuntime.PSVR
        assert info.device_manufacturer == "Sony"

        caps = info.capabilities
        assert caps.supports_eye_tracking is True
        assert caps.supports_hand_tracking is False
        assert caps.supports_finger_tracking is True
        assert caps.supports_hdr is True
        assert caps.haptic_channels == 2
        assert caps.is_tethered is True


class TestGetDeviceCapabilities:
    """Tests for device capabilities lookup."""

    def test_valve_index_capabilities(self) -> None:
        """Test Valve Index capability lookup."""
        caps = get_device_capabilities(XRDevice.VALVE_INDEX)

        assert caps.supports_finger_tracking is True
        assert caps.display_resolution == (1440, 1600)
        assert 144.0 in caps.display_refresh_rates
        assert caps.field_of_view == (130.0, 115.0)

    def test_quest_3_capabilities(self) -> None:
        """Test Quest 3 capability lookup."""
        caps = get_device_capabilities(XRDevice.META_QUEST_3)

        assert caps.supports_hand_tracking is True
        assert caps.supports_passthrough is True
        assert caps.supports_color_passthrough is True
        assert caps.is_tethered is False

    def test_vision_pro_capabilities(self) -> None:
        """Test Vision Pro capability lookup."""
        caps = get_device_capabilities(XRDevice.APPLE_VISION_PRO)

        assert caps.supports_eye_tracking is True
        assert caps.supports_6dof_controllers is False
        assert caps.controller_type == "hands"
        assert caps.performance_tier == "ultra"

    def test_unknown_device_returns_defaults(self) -> None:
        """Test that unknown devices return default capabilities."""
        caps = get_device_capabilities(XRDevice.UNKNOWN)

        # Should return default capabilities
        assert isinstance(caps, XRDeviceCapabilities)
        assert caps.supports_6dof_head is True


class TestDetectXRPlatform:
    """Tests for platform detection."""

    def test_detect_returns_none_when_no_platform_available(self) -> None:
        """Test detection returns None when no XR platform is available."""
        with patch.object(OpenXRPlatform, 'is_available', return_value=False), \
             patch.object(MetaQuestPlatform, 'is_available', return_value=False), \
             patch.object(SteamVRPlatform, 'is_available', return_value=False), \
             patch.object(AppleVisionProPlatform, 'is_available', return_value=False), \
             patch.object(PSVR2Platform, 'is_available', return_value=False):
            result = detect_xr_platform()
            assert result is None, "Should return None when no platform is available"

    def test_detect_returns_platform_when_available(self) -> None:
        """Test detection returns a platform instance when available."""
        with patch.object(OpenXRPlatform, 'is_available', return_value=True), \
             patch.object(OpenXRPlatform, 'initialize', return_value=True):
            result = detect_xr_platform()
            assert result is not None, "Should return a platform when one is available"
            assert isinstance(result, XRPlatform), "Result should be an XRPlatform instance"
            assert hasattr(result, 'initialize'), "Platform should have initialize method"
            assert hasattr(result, 'shutdown'), "Platform should have shutdown method"

    @patch.object(MetaQuestPlatform, 'is_available', return_value=True)
    @patch.object(MetaQuestPlatform, 'initialize', return_value=True)
    def test_detect_prefers_quest(self, mock_init, mock_available) -> None:
        """Test that detection prefers Quest when available."""
        result = detect_xr_platform()
        assert isinstance(result, MetaQuestPlatform)
