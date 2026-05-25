"""XR Runtime Foundation module.

This module provides the core XR runtime abstraction layer, supporting
multiple XR platforms including OpenXR, WebXR, SteamVR, and Oculus.

Key components:
- XRRuntime: Abstract base class for runtime implementations
- XRSession: Session state machine with lifecycle management
- XRCapabilities: Device feature detection and queries
- OpenXRRuntime: OpenXR backend implementation
- WebXRRuntime: WebXR backend implementation

Example usage:
    >>> from engine.xr.runtime import create_runtime, XRRuntimeType, XRSessionConfig
    >>>
    >>> # Create and initialize runtime
    >>> runtime = create_runtime(XRRuntimeType.OPENXR)
    >>>
    >>> # Create and start session
    >>> config = XRSessionConfig(enable_hand_tracking=True)
    >>> session = runtime.create_session(config)
    >>> runtime.start_session()
    >>>
    >>> # Main loop
    >>> while runtime.is_session_active:
    ...     runtime.poll_events()
    ...     if runtime.wait_frame():
    ...         runtime.begin_frame()
    ...         # Render frame using runtime.left_view and runtime.right_view
    ...         runtime.end_frame()
    >>>
    >>> # Cleanup
    >>> runtime.shutdown()
"""
from __future__ import annotations

# Core types and utilities
from engine.xr.runtime.xr_runtime import (
    XRRuntimeType,
    XRRuntimeState,
    XRRuntime,
    XRRuntimeError,
    XRRuntimeNotAvailableError,
    Pose,
    ViewInfo,
    create_runtime,
    detect_available_runtimes,
)

# Session management
from engine.xr.runtime.session import (
    XRSessionState,
    XRSessionMode,
    XRReferenceSpace,
    XRSessionConfig,
    XRSessionStats,
    XRSession,
    XRSessionError,
    InvalidStateTransitionError,
)

# Capability detection
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

# Backend implementations
from engine.xr.runtime.openxr import (
    OpenXRRuntime,
    OpenXRError,
    OpenXRSessionState,
)
from engine.xr.runtime.webxr import (
    WebXRRuntime,
    WebXRError,
    WebXRSessionMode,
    WebXRInputSource,
)


__all__ = [
    # Runtime types
    "XRRuntimeType",
    "XRRuntimeState",
    "XRRuntime",
    "XRRuntimeError",
    "XRRuntimeNotAvailableError",
    # Pose and view
    "Pose",
    "ViewInfo",
    # Factory functions
    "create_runtime",
    "detect_available_runtimes",
    # Session
    "XRSessionState",
    "XRSessionMode",
    "XRReferenceSpace",
    "XRSessionConfig",
    "XRSessionStats",
    "XRSession",
    "XRSessionError",
    "InvalidStateTransitionError",
    # Capabilities
    "XRFeature",
    "XRCapabilities",
    "DisplaySpecs",
    "TrackingCapabilities",
    "RenderingCapabilities",
    "InputCapabilities",
    "detect_capabilities",
    "create_fallback_capabilities",
    # OpenXR
    "OpenXRRuntime",
    "OpenXRError",
    "OpenXRSessionState",
    # WebXR
    "WebXRRuntime",
    "WebXRError",
    "WebXRSessionMode",
    "WebXRInputSource",
]
