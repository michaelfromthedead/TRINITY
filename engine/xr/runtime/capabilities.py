"""XR device capability detection and queries.

This module provides feature detection for XR devices, allowing the engine
to query available capabilities and gracefully degrade when features are
not supported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, FrozenSet


__all__ = [
    "XRFeature",
    "XRCapabilities",
    "DisplaySpecs",
    "TrackingCapabilities",
    "RenderingCapabilities",
    "InputCapabilities",
]


class XRFeature(Enum):
    """Enumeration of detectable XR features."""

    # Tracking features
    HEAD_TRACKING = auto()
    POSITIONAL_TRACKING = auto()
    CONTROLLER_TRACKING = auto()
    HAND_TRACKING = auto()
    EYE_TRACKING = auto()
    BODY_TRACKING = auto()
    FACE_TRACKING = auto()

    # Rendering features
    STEREO_RENDERING = auto()
    FOVEATED_RENDERING = auto()
    DYNAMIC_FOVEATION = auto()
    PASSTHROUGH = auto()
    DEPTH_SENSING = auto()
    MIXED_REALITY = auto()

    # Input features
    HAPTIC_FEEDBACK = auto()
    SPATIAL_ANCHORS = auto()
    PLANE_DETECTION = auto()
    MESH_DETECTION = auto()
    IMAGE_TRACKING = auto()
    OBJECT_TRACKING = auto()

    # Advanced features
    SCENE_UNDERSTANDING = auto()
    SPATIAL_AUDIO = auto()
    LIP_TRACKING = auto()
    KEYBOARD_TRACKING = auto()


@dataclass(slots=True, frozen=True)
class DisplaySpecs:
    """Display specifications for an XR device.

    Attributes:
        resolution_per_eye: Resolution (width, height) per eye in pixels.
        refresh_rate: Display refresh rate in Hz.
        supported_refresh_rates: All supported refresh rates.
        field_of_view_horizontal: Horizontal FOV in degrees.
        field_of_view_vertical: Vertical FOV in degrees.
        ipd_range: Supported IPD range (min, max) in meters.
        panel_type: Display panel type (LCD, OLED, etc.).
    """

    resolution_per_eye: tuple[int, int] = (1920, 1920)
    refresh_rate: float = 90.0
    supported_refresh_rates: tuple[float, ...] = (72.0, 90.0)
    field_of_view_horizontal: float = 90.0
    field_of_view_vertical: float = 90.0
    ipd_range: tuple[float, float] = (0.058, 0.072)
    panel_type: str = "LCD"


@dataclass(slots=True, frozen=True)
class TrackingCapabilities:
    """Tracking system capabilities.

    Attributes:
        tracking_frequency: Tracking update frequency in Hz.
        position_accuracy: Positional accuracy in meters.
        rotation_accuracy: Rotational accuracy in degrees.
        tracking_volume: Supported tracking volume (width, height, depth) in meters.
        supports_guardian: Whether guardian/boundary system is supported.
        guardian_data_available: Whether guardian boundary data can be queried.
    """

    tracking_frequency: float = 250.0
    position_accuracy: float = 0.001
    rotation_accuracy: float = 0.1
    tracking_volume: tuple[float, float, float] = (10.0, 3.0, 10.0)
    supports_guardian: bool = True
    guardian_data_available: bool = True


@dataclass(slots=True, frozen=True)
class RenderingCapabilities:
    """Rendering system capabilities.

    Attributes:
        max_render_scale: Maximum render resolution scale.
        supports_multiview: Whether multiview stereo rendering is supported.
        supports_instancing: Whether instanced stereo rendering is supported.
        max_foveation_level: Maximum foveation level (0-4).
        supports_application_spacewarp: Whether ASW/AST is supported.
        supports_async_timewarp: Whether ATW is supported.
        max_layer_count: Maximum compositor layers.
        supports_depth_composition: Whether depth-based composition is supported.
    """

    max_render_scale: float = 2.0
    supports_multiview: bool = True
    supports_instancing: bool = True
    max_foveation_level: int = 4
    supports_application_spacewarp: bool = False
    supports_async_timewarp: bool = True
    max_layer_count: int = 16
    supports_depth_composition: bool = True


@dataclass(slots=True, frozen=True)
class InputCapabilities:
    """Input system capabilities.

    Attributes:
        controller_count: Number of tracked controllers.
        haptic_channels: Number of haptic feedback channels per controller.
        haptic_frequency_range: Supported haptic frequency range (min, max) in Hz.
        hand_joint_count: Number of tracked hand joints (0 if not supported).
        eye_sample_rate: Eye tracking sample rate in Hz (0 if not supported).
        supports_skeletal_input: Whether skeletal hand input is supported.
    """

    controller_count: int = 2
    haptic_channels: int = 1
    haptic_frequency_range: tuple[float, float] = (100.0, 320.0)
    hand_joint_count: int = 26
    eye_sample_rate: float = 0.0
    supports_skeletal_input: bool = False


@dataclass(slots=True)
class XRCapabilities:
    """Complete XR device capabilities.

    This class aggregates all capability information for an XR device,
    providing a unified interface for feature queries.

    Attributes:
        device_name: Human-readable device name.
        vendor: Device vendor/manufacturer.
        features: Set of supported XR features.
        display: Display specifications.
        tracking: Tracking capabilities.
        rendering: Rendering capabilities.
        input: Input capabilities.
    """

    device_name: str = "Unknown XR Device"
    vendor: str = "Unknown"
    features: FrozenSet[XRFeature] = field(default_factory=frozenset)
    display: DisplaySpecs = field(default_factory=DisplaySpecs)
    tracking: TrackingCapabilities = field(default_factory=TrackingCapabilities)
    rendering: RenderingCapabilities = field(default_factory=RenderingCapabilities)
    input: InputCapabilities = field(default_factory=InputCapabilities)

    def supports(self, feature: XRFeature) -> bool:
        """Check if a specific feature is supported.

        Args:
            feature: The XRFeature to check.

        Returns:
            True if the feature is supported, False otherwise.
        """
        return feature in self.features

    def supports_all(self, *features: XRFeature) -> bool:
        """Check if all specified features are supported.

        Args:
            *features: Variable number of XRFeature values to check.

        Returns:
            True if all features are supported, False otherwise.
        """
        return all(f in self.features for f in features)

    def supports_any(self, *features: XRFeature) -> bool:
        """Check if any of the specified features are supported.

        Args:
            *features: Variable number of XRFeature values to check.

        Returns:
            True if at least one feature is supported, False otherwise.
        """
        return any(f in self.features for f in features)

    def get_missing_features(self, *required: XRFeature) -> FrozenSet[XRFeature]:
        """Get the set of required features that are not supported.

        Args:
            *required: Required features to check.

        Returns:
            Set of features that are required but not supported.
        """
        return frozenset(f for f in required if f not in self.features)

    @property
    def has_hand_tracking(self) -> bool:
        """Check if hand tracking is available."""
        return self.supports(XRFeature.HAND_TRACKING)

    @property
    def has_eye_tracking(self) -> bool:
        """Check if eye tracking is available."""
        return self.supports(XRFeature.EYE_TRACKING)

    @property
    def has_passthrough(self) -> bool:
        """Check if passthrough (mixed reality) is available."""
        return self.supports(XRFeature.PASSTHROUGH)

    @property
    def has_foveated_rendering(self) -> bool:
        """Check if foveated rendering is available."""
        return self.supports(XRFeature.FOVEATED_RENDERING)

    @property
    def has_spatial_anchors(self) -> bool:
        """Check if spatial anchors are available."""
        return self.supports(XRFeature.SPATIAL_ANCHORS)


def detect_capabilities(runtime_name: str) -> XRCapabilities:
    """Detect capabilities for a given XR runtime.

    This function queries the runtime to determine available features
    and device specifications.

    Args:
        runtime_name: Name of the XR runtime (e.g., "OpenXR", "WebXR").

    Returns:
        XRCapabilities instance with detected features.
    """
    # Base capabilities that most runtimes support
    base_features = frozenset({
        XRFeature.HEAD_TRACKING,
        XRFeature.POSITIONAL_TRACKING,
        XRFeature.CONTROLLER_TRACKING,
        XRFeature.STEREO_RENDERING,
        XRFeature.HAPTIC_FEEDBACK,
    })

    if runtime_name.lower() == "openxr":
        # OpenXR typically supports more advanced features
        features = base_features | frozenset({
            XRFeature.HAND_TRACKING,
            XRFeature.FOVEATED_RENDERING,
            XRFeature.SPATIAL_ANCHORS,
            XRFeature.PLANE_DETECTION,
        })
        return XRCapabilities(
            device_name="OpenXR Device",
            vendor="OpenXR Runtime",
            features=features,
            display=DisplaySpecs(
                resolution_per_eye=(1920, 1920),
                refresh_rate=90.0,
                supported_refresh_rates=(72.0, 80.0, 90.0, 120.0),
            ),
            input=InputCapabilities(
                hand_joint_count=26,
                supports_skeletal_input=True,
            ),
        )

    elif runtime_name.lower() == "webxr":
        # WebXR has more limited capabilities
        features = base_features | frozenset({
            XRFeature.SPATIAL_ANCHORS,
        })
        return XRCapabilities(
            device_name="WebXR Device",
            vendor="WebXR Runtime",
            features=features,
            rendering=RenderingCapabilities(
                supports_multiview=False,
                max_foveation_level=0,
            ),
        )

    elif runtime_name.lower() == "steamvr":
        features = base_features | frozenset({
            XRFeature.HAND_TRACKING,
            XRFeature.FOVEATED_RENDERING,
        })
        return XRCapabilities(
            device_name="SteamVR Device",
            vendor="Valve",
            features=features,
        )

    elif runtime_name.lower() == "oculus":
        features = base_features | frozenset({
            XRFeature.HAND_TRACKING,
            XRFeature.EYE_TRACKING,
            XRFeature.FOVEATED_RENDERING,
            XRFeature.DYNAMIC_FOVEATION,
            XRFeature.PASSTHROUGH,
            XRFeature.SPATIAL_ANCHORS,
            XRFeature.PLANE_DETECTION,
            XRFeature.SCENE_UNDERSTANDING,
        })
        return XRCapabilities(
            device_name="Meta Quest",
            vendor="Meta",
            features=features,
            display=DisplaySpecs(
                resolution_per_eye=(2064, 2208),
                refresh_rate=120.0,
                supported_refresh_rates=(72.0, 80.0, 90.0, 120.0),
            ),
            input=InputCapabilities(
                hand_joint_count=26,
                eye_sample_rate=90.0,
                supports_skeletal_input=True,
            ),
        )

    # Default/fallback capabilities
    return XRCapabilities(
        device_name="Generic XR Device",
        vendor="Unknown",
        features=base_features,
    )


def create_fallback_capabilities() -> XRCapabilities:
    """Create minimal fallback capabilities for graceful degradation.

    Returns:
        XRCapabilities with minimal feature set.
    """
    return XRCapabilities(
        device_name="Fallback Device",
        vendor="None",
        features=frozenset({XRFeature.HEAD_TRACKING}),
        display=DisplaySpecs(
            resolution_per_eye=(1280, 720),
            refresh_rate=60.0,
            supported_refresh_rates=(60.0,),
        ),
        tracking=TrackingCapabilities(
            tracking_frequency=60.0,
            supports_guardian=False,
        ),
        rendering=RenderingCapabilities(
            max_render_scale=1.0,
            supports_multiview=False,
            supports_instancing=False,
            max_foveation_level=0,
        ),
        input=InputCapabilities(
            controller_count=0,
            haptic_channels=0,
            hand_joint_count=0,
        ),
    )
