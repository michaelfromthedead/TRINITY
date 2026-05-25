"""XR runtime abstraction layer with unified API.

This module provides the abstract base class for XR runtime implementations
and the XRRuntimeState resource for global XR state management.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Callable, Any, Dict, TypeVar, Generic

from engine.xr.runtime.capabilities import (
    XRCapabilities,
    XRFeature,
    detect_capabilities,
    create_fallback_capabilities,
)
from engine.xr.runtime.session import (
    XRSession,
    XRSessionConfig,
    XRSessionState,
    XRSessionMode,
    XRReferenceSpace,
)
from engine.xr.config import XR_CONFIG


__all__ = [
    "XRRuntimeType",
    "XRRuntimeState",
    "XRRuntime",
    "XRRuntimeError",
    "XRRuntimeNotAvailableError",
    "Pose",
    "ViewInfo",
]


logger = logging.getLogger(__name__)


class XRRuntimeType(Enum):
    """Supported XR runtime types."""

    OPENXR = auto()
    WEBXR = auto()
    STEAMVR = auto()
    OCULUS = auto()
    MOCK = auto()  # For testing


class XRRuntimeError(Exception):
    """Base exception for XR runtime errors."""


class XRRuntimeNotAvailableError(XRRuntimeError):
    """Raised when the requested XR runtime is not available."""


@dataclass(slots=True, frozen=True)
class Pose:
    """Represents a 6DOF pose in 3D space.

    Attributes:
        position: (x, y, z) position in meters.
        orientation: (x, y, z, w) quaternion rotation.
        linear_velocity: (x, y, z) linear velocity in m/s.
        angular_velocity: (x, y, z) angular velocity in rad/s.
        is_valid: Whether the pose data is valid.
    """

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_valid: bool = True

    @staticmethod
    def identity() -> "Pose":
        """Create an identity pose at origin."""
        return Pose()

    @staticmethod
    def invalid() -> "Pose":
        """Create an invalid pose marker."""
        return Pose(is_valid=False)


@dataclass(slots=True, frozen=True)
class ViewInfo:
    """Information about a single view (eye) for rendering.

    Attributes:
        pose: The view pose (eye position/orientation).
        fov: Field of view (left, right, up, down) in radians.
        near_clip: Near clipping plane distance.
        far_clip: Far clipping plane distance.
    """

    pose: Pose = field(default_factory=Pose.identity)
    fov: tuple[float, float, float, float] = (-0.785, 0.785, 0.785, -0.785)
    near_clip: float = 0.1
    far_clip: float = 1000.0


@dataclass(slots=True)
class XRRuntimeState:
    """Global XR runtime state resource.

    This is a singleton resource that tracks the current state of the
    XR runtime and session. Use `ResourceMeta` pattern for ECS integration.

    Attributes:
        runtime_name: Name of the active runtime.
        runtime_version: Version string of the runtime.
        session_state: Current session state name.
        display_refresh_rate: Current display refresh rate in Hz.
        display_resolution: Current display resolution per eye.
        field_of_view: Current FOV (horizontal, vertical) in degrees.
        render_scale: Current render resolution scale.
        foveated_level: Current foveation level (0-4).
        supports_hand_tracking: Whether hand tracking is available.
        supports_eye_tracking: Whether eye tracking is available.
        supports_passthrough: Whether passthrough is available.
        head_pose: Current HMD pose.
        frame_number: Current frame number.
        predicted_display_time: Predicted time when frame will be displayed.
    """

    runtime_name: str = ""
    runtime_version: str = "0.0.0"
    session_state: str = "idle"
    display_refresh_rate: float = XR_CONFIG.runtime.DEFAULT_REFRESH_RATE_HZ
    display_resolution: tuple[int, int] = (1920, 1080)
    field_of_view: tuple[float, float] = (90.0, 90.0)
    render_scale: float = 1.0
    foveated_level: int = 0
    supports_hand_tracking: bool = False
    supports_eye_tracking: bool = False
    supports_passthrough: bool = False
    head_pose: Pose = field(default_factory=Pose.identity)
    frame_number: int = 0
    predicted_display_time: float = 0.0

    def update_from_session(self, session: XRSession) -> None:
        """Update state from an XRSession instance.

        Args:
            session: The active XR session.
        """
        self.session_state = session.state.name.lower()
        if session.capabilities:
            caps = session.capabilities
            self.display_refresh_rate = caps.display.refresh_rate
            self.display_resolution = caps.display.resolution_per_eye
            self.field_of_view = (
                caps.display.field_of_view_horizontal,
                caps.display.field_of_view_vertical,
            )
            self.supports_hand_tracking = caps.has_hand_tracking
            self.supports_eye_tracking = caps.has_eye_tracking
            self.supports_passthrough = caps.has_passthrough

        self.render_scale = session.config.render_scale
        self.foveated_level = session.config.foveation_level


# Type for event callbacks
EventCallback = Callable[..., None]


class XRRuntime(ABC):
    """Abstract base class for XR runtime implementations.

    This class defines the unified API that all XR runtime backends must
    implement. It provides lifecycle management, device queries, and
    frame synchronization.

    Subclasses must implement all abstract methods to provide runtime-specific
    functionality.
    """

    __slots__ = (
        "_runtime_type",
        "_capabilities",
        "_session",
        "_state",
        "_is_available",
        "_event_handlers",
        "_head_pose",
        "_left_view",
        "_right_view",
    )

    def __init__(self, runtime_type: XRRuntimeType) -> None:
        """Initialize the runtime.

        Args:
            runtime_type: The type of runtime this instance represents.
        """
        self._runtime_type = runtime_type
        self._capabilities: Optional[XRCapabilities] = None
        self._session: Optional[XRSession] = None
        self._state = XRRuntimeState()
        self._is_available = False
        self._event_handlers: Dict[str, List[EventCallback]] = {}
        self._head_pose = Pose.identity()
        self._left_view = ViewInfo()
        self._right_view = ViewInfo()

    @property
    def runtime_type(self) -> XRRuntimeType:
        """Get the runtime type."""
        return self._runtime_type

    @property
    def capabilities(self) -> Optional[XRCapabilities]:
        """Get device capabilities (available after initialization)."""
        return self._capabilities

    @property
    def session(self) -> Optional[XRSession]:
        """Get the current XR session."""
        return self._session

    @property
    def state(self) -> XRRuntimeState:
        """Get the current runtime state resource."""
        return self._state

    @property
    def is_available(self) -> bool:
        """Check if the runtime is available on this system."""
        return self._is_available

    @property
    def is_session_active(self) -> bool:
        """Check if a session is currently active."""
        return self._session is not None and self._session.is_running

    @property
    def head_pose(self) -> Pose:
        """Get the current HMD pose."""
        return self._head_pose

    @property
    def left_view(self) -> ViewInfo:
        """Get the left eye view information."""
        return self._left_view

    @property
    def right_view(self) -> ViewInfo:
        """Get the right eye view information."""
        return self._right_view

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the XR runtime.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shut down the XR runtime and release resources."""
        ...

    @abstractmethod
    def poll_events(self) -> None:
        """Poll for XR system events.

        This should be called each frame to process runtime events
        like session state changes, input events, etc.
        """
        ...

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def create_session(self, config: Optional[XRSessionConfig] = None) -> XRSession:
        """Create a new XR session.

        Args:
            config: Session configuration. Uses defaults if not provided.

        Returns:
            The created XRSession instance.

        Raises:
            XRRuntimeError: If runtime is not initialized or session already exists.
        """
        if not self._is_available:
            raise XRRuntimeError("Runtime not initialized")

        if self._session is not None:
            raise XRRuntimeError("Session already exists")

        if config is None:
            config = XRSessionConfig()

        self._session = XRSession(config)

        if self._capabilities:
            self._session.initialize(self._capabilities)

        self._state.update_from_session(self._session)

        return self._session

    def destroy_session(self) -> None:
        """Destroy the current XR session."""
        if self._session is not None:
            if self._session.is_running or self._session.is_paused:
                self._session.stop()
            self._session = None
            self._state.session_state = "idle"

    def start_session(self) -> bool:
        """Start the current session (begin presenting).

        Returns:
            True if session started, False otherwise.
        """
        if self._session is None:
            logger.warning("No session to start")
            return False

        result = self._session.start()
        self._state.update_from_session(self._session)
        return result

    def stop_session(self) -> bool:
        """Stop the current session.

        Returns:
            True if session stopped, False otherwise.
        """
        if self._session is None:
            logger.warning("No session to stop")
            return False

        result = self._session.stop()
        self._state.update_from_session(self._session)
        return result

    # -------------------------------------------------------------------------
    # Frame Synchronization
    # -------------------------------------------------------------------------

    @abstractmethod
    def wait_frame(self) -> bool:
        """Wait for the next frame to be ready for rendering.

        This blocks until the compositor signals it's ready for the next frame.

        Returns:
            True if frame is ready, False if session ended.
        """
        ...

    @abstractmethod
    def begin_frame(self) -> bool:
        """Signal the start of frame rendering.

        Returns:
            True if frame began successfully, False otherwise.
        """
        ...

    @abstractmethod
    def end_frame(self, views: Optional[List[ViewInfo]] = None) -> bool:
        """Signal the end of frame rendering and submit to compositor.

        Args:
            views: View information for submitted frame.

        Returns:
            True if frame was submitted successfully, False otherwise.
        """
        ...

    # -------------------------------------------------------------------------
    # Tracking
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_head_pose(self, predicted_time: float = 0.0) -> Pose:
        """Get the current or predicted HMD pose.

        Args:
            predicted_time: Time offset for pose prediction in seconds.

        Returns:
            The HMD pose.
        """
        ...

    @abstractmethod
    def get_view_info(self, view_index: int, predicted_time: float = 0.0) -> ViewInfo:
        """Get view information for rendering.

        Args:
            view_index: 0 for left eye, 1 for right eye.
            predicted_time: Time offset for pose prediction.

        Returns:
            View information for the specified eye.
        """
        ...

    def update_tracking(self, predicted_time: float = 0.0) -> None:
        """Update all tracking data for the current frame.

        Args:
            predicted_time: Time offset for pose prediction.
        """
        self._head_pose = self.get_head_pose(predicted_time)
        self._left_view = self.get_view_info(0, predicted_time)
        self._right_view = self.get_view_info(1, predicted_time)
        self._state.head_pose = self._head_pose

    # -------------------------------------------------------------------------
    # Feature Queries
    # -------------------------------------------------------------------------

    def supports_feature(self, feature: XRFeature) -> bool:
        """Check if a feature is supported by this runtime.

        Args:
            feature: The feature to check.

        Returns:
            True if supported, False otherwise.
        """
        if self._capabilities is None:
            return False
        return self._capabilities.supports(feature)

    def is_feature_enabled(self, feature: XRFeature) -> bool:
        """Check if a feature is enabled in the current session.

        Args:
            feature: The feature to check.

        Returns:
            True if enabled, False otherwise.
        """
        if self._session is None:
            return False
        return self._session.is_feature_enabled(feature)

    # -------------------------------------------------------------------------
    # Event System
    # -------------------------------------------------------------------------

    def on(self, event_name: str, callback: EventCallback) -> None:
        """Register an event handler.

        Args:
            event_name: Name of the event to handle.
            callback: Callback function to invoke.
        """
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(callback)

    def off(self, event_name: str, callback: EventCallback) -> bool:
        """Unregister an event handler.

        Args:
            event_name: Name of the event.
            callback: Callback to remove.

        Returns:
            True if callback was found and removed.
        """
        if event_name not in self._event_handlers:
            return False
        try:
            self._event_handlers[event_name].remove(callback)
            return True
        except ValueError:
            return False

    def emit(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """Emit an event to all registered handlers.

        Args:
            event_name: Name of the event.
            *args: Positional arguments for handlers.
            **kwargs: Keyword arguments for handlers.
        """
        handlers = self._event_handlers.get(event_name, [])
        for handler in handlers:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Error in event handler for {event_name}: {e}")

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def recenter(self) -> bool:
        """Recenter the tracking origin to the current HMD position.

        Returns:
            True if recenter succeeded, False otherwise.
        """
        # Default implementation - subclasses may override
        logger.info("Recenter not implemented for this runtime")
        return False

    def set_render_scale(self, scale: float) -> None:
        """Set the render resolution scale.

        Args:
            scale: Scale factor (0.5 to 2.0).
        """
        scale = max(0.5, min(2.0, scale))
        if self._session is not None:
            self._session.config.render_scale = scale
            self._state.render_scale = scale

    def set_refresh_rate(self, rate: float) -> bool:
        """Set the display refresh rate.

        Args:
            rate: Target refresh rate in Hz.

        Returns:
            True if rate was set, False if not supported.
        """
        if self._capabilities is None:
            return False

        if rate not in self._capabilities.display.supported_refresh_rates:
            logger.warning(f"Refresh rate {rate}Hz not supported")
            return False

        if self._session is not None:
            self._session.config.target_refresh_rate = rate
            self._state.display_refresh_rate = rate
        return True

    def get_boundary_geometry(self) -> Optional[List[tuple[float, float, float]]]:
        """Get the guardian/playspace boundary geometry.

        Returns:
            List of boundary points, or None if not available.
        """
        # Default implementation - subclasses may override
        return None


def create_runtime(
    runtime_type: XRRuntimeType,
    fallback: bool = True,
) -> XRRuntime:
    """Factory function to create an XR runtime instance.

    Args:
        runtime_type: The type of runtime to create.
        fallback: Whether to fall back to mock runtime if unavailable.

    Returns:
        An XRRuntime instance.

    Raises:
        XRRuntimeNotAvailableError: If runtime is unavailable and fallback is False.
    """
    # Import here to avoid circular imports
    from engine.xr.runtime.openxr import OpenXRRuntime
    from engine.xr.runtime.webxr import WebXRRuntime

    runtime: XRRuntime

    if runtime_type == XRRuntimeType.OPENXR:
        runtime = OpenXRRuntime()
    elif runtime_type == XRRuntimeType.WEBXR:
        runtime = WebXRRuntime()
    elif runtime_type == XRRuntimeType.MOCK:
        runtime = _MockRuntime()
    else:
        # Try OpenXR as default
        runtime = OpenXRRuntime()

    if runtime.initialize():
        return runtime

    if fallback:
        logger.warning(
            f"Runtime {runtime_type.name} not available, falling back to mock"
        )
        mock = _MockRuntime()
        mock.initialize()
        return mock

    raise XRRuntimeNotAvailableError(
        f"Runtime {runtime_type.name} is not available on this system"
    )


def detect_available_runtimes() -> List[XRRuntimeType]:
    """Detect which XR runtimes are available on this system.

    Returns:
        List of available runtime types.
    """
    available = []

    # Check OpenXR
    try:
        from engine.xr.runtime.openxr import OpenXRRuntime
        runtime = OpenXRRuntime()
        if runtime.initialize():
            available.append(XRRuntimeType.OPENXR)
            runtime.shutdown()
    except Exception:
        pass

    # Check WebXR (browser environment)
    try:
        from engine.xr.runtime.webxr import WebXRRuntime
        runtime = WebXRRuntime()
        if runtime.initialize():
            available.append(XRRuntimeType.WEBXR)
            runtime.shutdown()
    except Exception:
        pass

    # Mock is always available
    available.append(XRRuntimeType.MOCK)

    return available


class _MockRuntime(XRRuntime):
    """Mock XR runtime for testing and fallback."""

    __slots__ = ("_frame_count", "_initialized")

    def __init__(self) -> None:
        super().__init__(XRRuntimeType.MOCK)
        self._frame_count = 0
        self._initialized = False

    def initialize(self) -> bool:
        if self._initialized:
            return True

        self._capabilities = create_fallback_capabilities()
        self._state.runtime_name = "Mock XR Runtime"
        self._state.runtime_version = "1.0.0"
        self._is_available = True
        self._initialized = True
        logger.info("Mock XR runtime initialized")
        return True

    def shutdown(self) -> None:
        if self._session is not None:
            self.destroy_session()
        self._event_handlers.clear()  # Clear all registered callbacks
        self._is_available = False
        self._initialized = False
        logger.info("Mock XR runtime shut down")

    def poll_events(self) -> None:
        # Mock runtime has no real events
        pass

    def wait_frame(self) -> bool:
        if not self.is_session_active:
            return False
        # Simulate frame timing
        import time
        time.sleep(1.0 / 90.0)  # Simulate 90Hz
        return True

    def begin_frame(self) -> bool:
        if not self.is_session_active:
            return False
        self._frame_count += 1
        self._state.frame_number = self._frame_count
        return True

    def end_frame(self, views: Optional[List[ViewInfo]] = None) -> bool:
        if not self.is_session_active:
            return False
        if self._session:
            self._session.update_frame_stats(frame_presented=True)
        return True

    def get_head_pose(self, predicted_time: float = 0.0) -> Pose:
        # Return a static pose for testing
        return Pose(
            position=(0.0, 1.6, 0.0),  # ~eye height
            orientation=(0.0, 0.0, 0.0, 1.0),
        )

    def get_view_info(self, view_index: int, predicted_time: float = 0.0) -> ViewInfo:
        # Simple stereo views
        ipd = XR_CONFIG.runtime.DEFAULT_IPD_MM / 1000.0  # Convert mm to meters
        offset = -ipd / 2 if view_index == 0 else ipd / 2
        return ViewInfo(
            pose=Pose(
                position=(offset, 1.6, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            ),
            fov=(-0.785, 0.785, 0.785, -0.785),
        )
