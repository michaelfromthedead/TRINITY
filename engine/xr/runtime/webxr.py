"""WebXR backend implementation.

This module provides the WebXR runtime implementation for browser-based
XR applications, supporting WebXR Device API.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable

from engine.xr.runtime.xr_runtime import (
    XRRuntime,
    XRRuntimeType,
    XRRuntimeError,
    Pose,
    ViewInfo,
)
from engine.xr.runtime.capabilities import (
    XRCapabilities,
    XRFeature,
    DisplaySpecs,
    TrackingCapabilities,
    RenderingCapabilities,
    InputCapabilities,
)
from engine.xr.runtime.session import (
    XRSessionState,
    XRSessionMode,
    XRReferenceSpace,
)


__all__ = [
    "WebXRRuntime",
    "WebXRError",
    "WebXRSessionMode",
    "WebXRInputSource",
]


logger = logging.getLogger(__name__)


class WebXRError(XRRuntimeError):
    """WebXR-specific error."""


class WebXRSessionMode(Enum):
    """WebXR session mode (mirrors XRSessionMode from WebXR API)."""

    INLINE = "inline"
    IMMERSIVE_VR = "immersive-vr"
    IMMERSIVE_AR = "immersive-ar"


class WebXRReferenceSpaceType(Enum):
    """WebXR reference space types."""

    VIEWER = "viewer"
    LOCAL = "local"
    LOCAL_FLOOR = "local-floor"
    BOUNDED_FLOOR = "bounded-floor"
    UNBOUNDED = "unbounded"


@dataclass(slots=True)
class WebXRInputSource:
    """Represents a WebXR input source (controller or hand).

    Attributes:
        handedness: "left", "right", or "none".
        target_ray_mode: "gaze", "tracked-pointer", or "screen".
        profiles: List of input profile strings.
        grip_pose: Grip space pose if available.
        target_ray_pose: Target ray space pose.
        gamepad_state: Gamepad button/axis state if available.
        hand_joint_poses: Hand tracking joint poses if available.
    """

    handedness: str = "none"
    target_ray_mode: str = "tracked-pointer"
    profiles: List[str] = field(default_factory=list)
    grip_pose: Optional[Pose] = None
    target_ray_pose: Optional[Pose] = None
    gamepad_state: Optional[Dict[str, Any]] = None
    hand_joint_poses: Optional[List[Pose]] = None


@dataclass(slots=True)
class WebXRFrameData:
    """Frame data from WebXR requestAnimationFrame callback.

    Attributes:
        time: DOMHighResTimeStamp.
        viewer_pose: Viewer (HMD) pose.
        views: List of view information for each eye.
        input_sources: Active input sources.
    """

    time: float = 0.0
    viewer_pose: Optional[Pose] = None
    views: List[ViewInfo] = field(default_factory=list)
    input_sources: List[WebXRInputSource] = field(default_factory=list)


class WebXRRuntime(XRRuntime):
    """WebXR runtime implementation.

    This class implements the XRRuntime interface for WebXR, enabling
    XR applications to run in web browsers.

    Note:
        This implementation simulates WebXR behavior for development.
        In production within a browser, it would use the WebXR Device API
        through JavaScript interop.
    """

    __slots__ = (
        "_session_mode",
        "_reference_space_type",
        "_frame_data",
        "_input_sources",
        "_is_inline",
        "_animation_frame_callback",
        "_last_frame_time",
        "_supported_features",
        "_granted_features",
        "_gl_context",
        "_xr_gl_layer",
    )

    # Feature strings (WebXR)
    FEATURE_LOCAL = "local"
    FEATURE_LOCAL_FLOOR = "local-floor"
    FEATURE_BOUNDED_FLOOR = "bounded-floor"
    FEATURE_UNBOUNDED = "unbounded"
    FEATURE_HAND_TRACKING = "hand-tracking"
    FEATURE_HIT_TEST = "hit-test"
    FEATURE_ANCHORS = "anchors"
    FEATURE_PLANE_DETECTION = "plane-detection"
    FEATURE_DEPTH_SENSING = "depth-sensing"

    def __init__(self) -> None:
        """Initialize the WebXR runtime."""
        super().__init__(XRRuntimeType.WEBXR)

        self._session_mode = WebXRSessionMode.IMMERSIVE_VR
        self._reference_space_type = WebXRReferenceSpaceType.LOCAL_FLOOR
        self._frame_data = WebXRFrameData()
        self._input_sources: List[WebXRInputSource] = []
        self._is_inline = False
        self._animation_frame_callback: Optional[Callable] = None
        self._last_frame_time = 0.0
        self._supported_features: List[str] = []
        self._granted_features: List[str] = []
        self._gl_context: Any = None
        self._xr_gl_layer: Any = None

    @property
    def session_mode(self) -> WebXRSessionMode:
        """Get the current WebXR session mode."""
        return self._session_mode

    @property
    def reference_space_type(self) -> WebXRReferenceSpaceType:
        """Get the current reference space type."""
        return self._reference_space_type

    @property
    def frame_data(self) -> WebXRFrameData:
        """Get the current frame data."""
        return self._frame_data

    @property
    def input_sources(self) -> List[WebXRInputSource]:
        """Get the list of active input sources."""
        return self._input_sources

    @property
    def is_inline_session(self) -> bool:
        """Check if running an inline (non-immersive) session."""
        return self._is_inline

    @property
    def granted_features(self) -> List[str]:
        """Get the list of granted WebXR features."""
        return list(self._granted_features)

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    def initialize(self) -> bool:
        """Initialize the WebXR runtime.

        In a browser environment, this would check navigator.xr availability.

        Returns:
            True if WebXR is available, False otherwise.
        """
        try:
            # Check if we're in a browser environment (simulated here)
            if not self._is_webxr_available():
                logger.warning("WebXR not available")
                return False

            # Query supported session modes
            supported_modes = self._get_supported_session_modes()
            if not supported_modes:
                logger.warning("No WebXR session modes supported")
                return False

            # Query supported features
            self._supported_features = self._get_supported_features()

            # Build capabilities
            self._capabilities = self._query_capabilities(supported_modes)

            # Update state
            self._state.runtime_name = "WebXR"
            self._state.runtime_version = "1.0"
            self._is_available = True

            logger.info(
                f"WebXR initialized. Supported modes: {[m.value for m in supported_modes]}"
            )
            return True

        except Exception as e:
            logger.exception(f"WebXR initialization failed: {e}")
            return False

    def shutdown(self) -> None:
        """Shut down the WebXR runtime."""
        try:
            if self._session is not None:
                self.destroy_session()

            self._is_available = False
            self._supported_features = []
            self._granted_features = []

            logger.info("WebXR shut down")

        except Exception as e:
            logger.exception(f"Error during WebXR shutdown: {e}")

    def poll_events(self) -> None:
        """Poll for WebXR events.

        In WebXR, events come through the XRSession event handlers.
        """
        # Process input source changes
        self._update_input_sources()

        # Check for session end
        if self._session is not None:
            if self._session.state == XRSessionState.STOPPING:
                self.emit("sessionend")

    # -------------------------------------------------------------------------
    # Frame Synchronization
    # -------------------------------------------------------------------------

    def wait_frame(self) -> bool:
        """Wait for the next frame.

        In WebXR, this is handled by requestAnimationFrame.

        Returns:
            True if frame is ready.
        """
        if not self.is_session_active:
            return False

        # Simulate frame timing (WebXR uses browser's rAF)
        current_time = time.perf_counter()
        target_frame_time = 1.0 / self._state.display_refresh_rate

        if self._last_frame_time > 0:
            elapsed = current_time - self._last_frame_time
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

        self._last_frame_time = time.perf_counter()
        self._frame_data.time = self._last_frame_time * 1000  # Convert to ms

        return True

    def begin_frame(self) -> bool:
        """Begin frame rendering.

        Returns:
            True if frame began successfully.
        """
        if not self.is_session_active:
            return False

        self._state.frame_number += 1

        # Get viewer pose and views for this frame
        self._frame_data.viewer_pose = self.get_head_pose()
        self._frame_data.views = [
            self.get_view_info(0),
            self.get_view_info(1),
        ]

        # Update input sources
        self._update_input_sources()

        # Update tracking
        self.update_tracking()

        return True

    def end_frame(self, views: Optional[List[ViewInfo]] = None) -> bool:
        """End frame and submit.

        Args:
            views: View information (unused in WebXR - views come from frame data).

        Returns:
            True if frame ended successfully.
        """
        if not self.is_session_active:
            return False

        # In WebXR, the frame is submitted automatically when
        # requestAnimationFrame callback returns

        if self._session is not None:
            self._session.update_frame_stats(frame_presented=True)

        return True

    # -------------------------------------------------------------------------
    # Tracking
    # -------------------------------------------------------------------------

    def get_head_pose(self, predicted_time: float = 0.0) -> Pose:
        """Get the viewer (HMD) pose.

        Args:
            predicted_time: Unused in WebXR (prediction handled by browser).

        Returns:
            The viewer pose.
        """
        # In WebXR, this comes from XRFrame.getViewerPose()
        # Simulated tracking data
        return Pose(
            position=(0.0, 1.6, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            is_valid=True,
        )

    def get_view_info(self, view_index: int, predicted_time: float = 0.0) -> ViewInfo:
        """Get view information for an eye.

        Args:
            view_index: 0 for left, 1 for right.
            predicted_time: Unused in WebXR.

        Returns:
            View information.
        """
        # In WebXR, this comes from XRView objects in XRViewerPose
        ipd = 0.063
        eye_offset = -ipd / 2 if view_index == 0 else ipd / 2

        return ViewInfo(
            pose=Pose(
                position=(eye_offset, 1.6, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            ),
            fov=(-0.785, 0.785, 0.785, -0.785),
            near_clip=0.1,
            far_clip=1000.0,
        )

    # -------------------------------------------------------------------------
    # WebXR-Specific Methods
    # -------------------------------------------------------------------------

    def request_session(
        self,
        mode: WebXRSessionMode,
        required_features: Optional[List[str]] = None,
        optional_features: Optional[List[str]] = None,
    ) -> bool:
        """Request a WebXR session.

        Args:
            mode: The session mode to request.
            required_features: Features that must be available.
            optional_features: Features to request if available.

        Returns:
            True if session request succeeded.
        """
        if required_features is None:
            required_features = []
        if optional_features is None:
            optional_features = []

        # Check if mode is supported
        supported_modes = self._get_supported_session_modes()
        if mode not in supported_modes:
            logger.error(f"Session mode {mode.value} not supported")
            return False

        # Check required features
        for feature in required_features:
            if feature not in self._supported_features:
                logger.error(f"Required feature {feature} not supported")
                return False

        # Determine granted features
        self._granted_features = list(required_features)
        for feature in optional_features:
            if feature in self._supported_features:
                self._granted_features.append(feature)

        self._session_mode = mode
        self._is_inline = mode == WebXRSessionMode.INLINE

        logger.info(
            f"WebXR session requested: {mode.value}, "
            f"features: {self._granted_features}"
        )

        return True

    def get_input_source(self, handedness: str) -> Optional[WebXRInputSource]:
        """Get an input source by handedness.

        Args:
            handedness: "left", "right", or "none".

        Returns:
            The input source, or None if not found.
        """
        for source in self._input_sources:
            if source.handedness == handedness:
                return source
        return None

    def get_controller_pose(self, hand: str, space: str = "grip") -> Optional[Pose]:
        """Get a controller pose.

        Args:
            hand: "left" or "right".
            space: "grip" or "target-ray".

        Returns:
            The controller pose, or None if not available.
        """
        source = self.get_input_source(hand)
        if source is None:
            return None

        if space == "grip":
            return source.grip_pose
        else:
            return source.target_ray_pose

    def get_gamepad_state(self, hand: str) -> Optional[Dict[str, Any]]:
        """Get gamepad state for a controller.

        Args:
            hand: "left" or "right".

        Returns:
            Gamepad state dict with 'buttons' and 'axes', or None.
        """
        source = self.get_input_source(hand)
        if source is None or source.gamepad_state is None:
            return None
        return source.gamepad_state

    def is_feature_granted(self, feature: str) -> bool:
        """Check if a feature was granted for this session.

        Args:
            feature: The feature string to check.

        Returns:
            True if feature was granted.
        """
        return feature in self._granted_features

    def set_reference_space(self, space_type: WebXRReferenceSpaceType) -> bool:
        """Set the reference space type.

        Args:
            space_type: The reference space to use.

        Returns:
            True if successful.
        """
        # Check if space type is available
        feature_map = {
            WebXRReferenceSpaceType.LOCAL: self.FEATURE_LOCAL,
            WebXRReferenceSpaceType.LOCAL_FLOOR: self.FEATURE_LOCAL_FLOOR,
            WebXRReferenceSpaceType.BOUNDED_FLOOR: self.FEATURE_BOUNDED_FLOOR,
            WebXRReferenceSpaceType.UNBOUNDED: self.FEATURE_UNBOUNDED,
        }

        required_feature = feature_map.get(space_type)
        if required_feature and not self.is_feature_granted(required_feature):
            logger.warning(f"Reference space {space_type.value} not available")
            return False

        self._reference_space_type = space_type
        logger.info(f"Reference space set to: {space_type.value}")
        return True

    def get_boundary_geometry(self) -> Optional[List[tuple[float, float, float]]]:
        """Get the boundary geometry for bounded-floor reference space.

        Returns:
            List of boundary points, or None if not available.
        """
        if self._reference_space_type != WebXRReferenceSpaceType.BOUNDED_FLOOR:
            return None

        if not self.is_feature_granted(self.FEATURE_BOUNDED_FLOOR):
            return None

        # In WebXR, this comes from XRBoundedReferenceSpace.boundsGeometry
        # Return simulated 2x2 meter bounds
        return [
            (-1.0, 0.0, -1.0),
            (1.0, 0.0, -1.0),
            (1.0, 0.0, 1.0),
            (-1.0, 0.0, 1.0),
        ]

    def perform_hit_test(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
    ) -> Optional[tuple[float, float, float]]:
        """Perform an AR hit test against real-world geometry.

        Args:
            origin: Ray origin point.
            direction: Ray direction vector.

        Returns:
            Hit point, or None if no hit.
        """
        if not self.is_feature_granted(self.FEATURE_HIT_TEST):
            return None

        # In real WebXR, this uses XRHitTestSource
        # Return simulated hit on floor plane
        if direction[1] < 0:  # Pointing downward
            t = -origin[1] / direction[1]
            if t > 0:
                return (
                    origin[0] + direction[0] * t,
                    0.0,
                    origin[2] + direction[2] * t,
                )
        return None

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _is_webxr_available(self) -> bool:
        """Check if WebXR is available."""
        # In browser, this checks navigator.xr existence
        # Simulate availability for development
        return True

    def _get_supported_session_modes(self) -> List[WebXRSessionMode]:
        """Get supported session modes."""
        # In browser, this calls navigator.xr.isSessionSupported()
        # Simulate VR support
        return [
            WebXRSessionMode.INLINE,
            WebXRSessionMode.IMMERSIVE_VR,
        ]

    def _get_supported_features(self) -> List[str]:
        """Get supported WebXR features."""
        return [
            self.FEATURE_LOCAL,
            self.FEATURE_LOCAL_FLOOR,
            # These depend on device capabilities
            # self.FEATURE_BOUNDED_FLOOR,
            # self.FEATURE_HAND_TRACKING,
            # self.FEATURE_HIT_TEST,
            # self.FEATURE_ANCHORS,
        ]

    def _query_capabilities(
        self,
        supported_modes: List[WebXRSessionMode],
    ) -> XRCapabilities:
        """Build capabilities based on WebXR support."""
        features = {
            XRFeature.HEAD_TRACKING,
            XRFeature.POSITIONAL_TRACKING,
            XRFeature.STEREO_RENDERING,
        }

        # Add features based on session mode support
        if WebXRSessionMode.IMMERSIVE_VR in supported_modes:
            features.add(XRFeature.CONTROLLER_TRACKING)
            features.add(XRFeature.HAPTIC_FEEDBACK)

        if WebXRSessionMode.IMMERSIVE_AR in supported_modes:
            features.add(XRFeature.PASSTHROUGH)
            features.add(XRFeature.MIXED_REALITY)

        # Add features based on granted features
        if self.FEATURE_HAND_TRACKING in self._supported_features:
            features.add(XRFeature.HAND_TRACKING)

        if self.FEATURE_HIT_TEST in self._supported_features:
            features.add(XRFeature.PLANE_DETECTION)

        if self.FEATURE_ANCHORS in self._supported_features:
            features.add(XRFeature.SPATIAL_ANCHORS)

        return XRCapabilities(
            device_name="WebXR Device",
            vendor="Browser",
            features=frozenset(features),
            display=DisplaySpecs(
                resolution_per_eye=(1440, 1600),
                refresh_rate=72.0,
                supported_refresh_rates=(72.0,),
            ),
            tracking=TrackingCapabilities(
                tracking_frequency=72.0,
                supports_guardian=self.FEATURE_BOUNDED_FLOOR in self._supported_features,
            ),
            rendering=RenderingCapabilities(
                supports_multiview=False,  # WebXR typically uses multi-pass
                max_foveation_level=0,
            ),
            input=InputCapabilities(
                controller_count=2,
                hand_joint_count=25 if self.FEATURE_HAND_TRACKING in self._supported_features else 0,
            ),
        )

    def _update_input_sources(self) -> None:
        """Update the list of active input sources."""
        # In WebXR, this comes from XRSession.inputSources
        # and inputsourceschange events

        # Simulate two controllers
        if not self._input_sources:
            self._input_sources = [
                WebXRInputSource(
                    handedness="left",
                    target_ray_mode="tracked-pointer",
                    profiles=["generic-trigger-squeeze-touchpad-thumbstick"],
                    grip_pose=Pose(position=(-0.3, 1.0, -0.3)),
                    target_ray_pose=Pose(position=(-0.3, 1.0, -0.3)),
                    gamepad_state={
                        "buttons": [
                            {"pressed": False, "touched": False, "value": 0.0},  # Trigger
                            {"pressed": False, "touched": False, "value": 0.0},  # Squeeze
                            {"pressed": False, "touched": False, "value": 0.0},  # Touchpad
                            {"pressed": False, "touched": False, "value": 0.0},  # Thumbstick
                        ],
                        "axes": [0.0, 0.0, 0.0, 0.0],  # Touchpad X/Y, Thumbstick X/Y
                    },
                ),
                WebXRInputSource(
                    handedness="right",
                    target_ray_mode="tracked-pointer",
                    profiles=["generic-trigger-squeeze-touchpad-thumbstick"],
                    grip_pose=Pose(position=(0.3, 1.0, -0.3)),
                    target_ray_pose=Pose(position=(0.3, 1.0, -0.3)),
                    gamepad_state={
                        "buttons": [
                            {"pressed": False, "touched": False, "value": 0.0},
                            {"pressed": False, "touched": False, "value": 0.0},
                            {"pressed": False, "touched": False, "value": 0.0},
                            {"pressed": False, "touched": False, "value": 0.0},
                        ],
                        "axes": [0.0, 0.0, 0.0, 0.0],
                    },
                ),
            ]

        self._frame_data.input_sources = self._input_sources
