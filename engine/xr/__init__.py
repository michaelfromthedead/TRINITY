"""XR Layer - Comprehensive VR/AR/MR Support.

The XR Layer provides complete support for virtual reality (VR), augmented reality (AR),
and mixed reality (MR) experiences. It abstracts hardware differences across platforms
while enabling immersive, comfortable, and interactive XR applications.

Example:
    >>> from engine.xr import (
    ...     XRRuntime, XRSession, XRSessionConfig,
    ...     create_runtime, XRRuntimeType
    ... )
    >>>
    >>> # Initialize XR
    >>> runtime = create_runtime(XRRuntimeType.OPENXR)
    >>> session = runtime.create_session(XRSessionConfig())
    >>> runtime.start_session()
"""

import logging

_logger = logging.getLogger(__name__)

# =============================================================================
# Configuration - Centralized XR constants (ALWAYS AVAILABLE)
# =============================================================================

from engine.xr.config import (
    XR_CONFIG,
    XRConfig,
    XRRuntimeConfig,
    XRRenderingConfig,
    XRInteractionConfig,
    XRSpatialConfig,
    XRAvatarConfig,
    XRLocomotionConfig,
    XRUIConfig,
    XRPlatformConfig,
)

# =============================================================================
# Runtime - XR runtime abstraction and session management (ALWAYS AVAILABLE)
# =============================================================================

from engine.xr.runtime.xr_runtime import (
    XRRuntime,
    XRRuntimeType,
    XRRuntimeState,
    XRRuntimeError,
    XRRuntimeNotAvailableError,
    Pose,
    ViewInfo,
    create_runtime,
    detect_available_runtimes,
)

from engine.xr.runtime.session import (
    XRSession,
    XRSessionConfig,
    XRSessionState,
    XRSessionMode,
    XRReferenceSpace,
    XRSessionStats,
    XRSessionError,
    InvalidStateTransitionError,
)

from engine.xr.runtime.capabilities import (
    XRCapabilities,
    XRFeature,
    DisplaySpecs,
    TrackingCapabilities,
    RenderingCapabilities,
    InputCapabilities,
    detect_capabilities,
    create_fallback_capabilities,
)

# Track which optional modules loaded successfully
_LOADED_MODULES = {"runtime": True}

# =============================================================================
# Optional modules - wrapped in try/except for graceful degradation
# =============================================================================

try:
    from engine.xr.input.hmd import *
    from engine.xr.input.controller import *
    from engine.xr.input.hand_tracking import *
    from engine.xr.input.eye_tracking import *
    from engine.xr.input.haptics import *
    from engine.xr.input.bindings import *
    _LOADED_MODULES["input"] = True
except Exception as e:
    _logger.debug(f"XR input module not fully available: {e}")
    _LOADED_MODULES["input"] = False

try:
    from engine.xr.rendering.stereo import *
    from engine.xr.rendering.foveated import *
    from engine.xr.rendering.reprojection import *
    from engine.xr.rendering.compositor import *
    _LOADED_MODULES["rendering"] = True
except Exception as e:
    _logger.debug(f"XR rendering module not fully available: {e}")
    _LOADED_MODULES["rendering"] = False

try:
    from engine.xr.interaction.interactable import *
    from engine.xr.interaction.grabbable import *
    from engine.xr.interaction.socket import *
    _LOADED_MODULES["interaction"] = True
except Exception as e:
    _logger.debug(f"XR interaction module not fully available: {e}")
    _LOADED_MODULES["interaction"] = False

try:
    from engine.xr.spatial.anchor import *
    from engine.xr.spatial.plane_detection import *
    from engine.xr.spatial.mesh_mapping import *
    _LOADED_MODULES["spatial"] = True
except Exception as e:
    _logger.debug(f"XR spatial module not fully available: {e}")
    _LOADED_MODULES["spatial"] = False

try:
    from engine.xr.avatars import *
    _LOADED_MODULES["avatars"] = True
except Exception as e:
    _logger.debug(f"XR avatars module not fully available: {e}")
    _LOADED_MODULES["avatars"] = False

try:
    from engine.xr.platform import *
    _LOADED_MODULES["platform"] = True
except Exception as e:
    _logger.debug(f"XR platform module not fully available: {e}")
    _LOADED_MODULES["platform"] = False

try:
    from engine.xr.locomotion import *
    _LOADED_MODULES["locomotion"] = True
except Exception as e:
    _logger.debug(f"XR locomotion module not fully available: {e}")
    _LOADED_MODULES["locomotion"] = False

try:
    from engine.xr.ui import *
    _LOADED_MODULES["ui"] = True
except Exception as e:
    _logger.debug(f"XR UI module not fully available: {e}")
    _LOADED_MODULES["ui"] = False


# =============================================================================
# Public API - Core runtime exports (always available)
# =============================================================================

__all__ = [
    # Configuration
    "XR_CONFIG",
    "XRConfig",
    "XRRuntimeConfig",
    "XRRenderingConfig",
    "XRInteractionConfig",
    "XRSpatialConfig",
    "XRAvatarConfig",
    "XRLocomotionConfig",
    "XRUIConfig",
    "XRPlatformConfig",
    # Runtime
    "XRRuntime",
    "XRRuntimeType",
    "XRRuntimeState",
    "XRRuntimeError",
    "XRRuntimeNotAvailableError",
    "Pose",
    "ViewInfo",
    "create_runtime",
    "detect_available_runtimes",
    # Session
    "XRSession",
    "XRSessionConfig",
    "XRSessionState",
    "XRSessionMode",
    "XRReferenceSpace",
    "XRSessionStats",
    "XRSessionError",
    "InvalidStateTransitionError",
    # Capabilities
    "XRCapabilities",
    "XRFeature",
    "DisplaySpecs",
    "TrackingCapabilities",
    "RenderingCapabilities",
    "InputCapabilities",
    "detect_capabilities",
    "create_fallback_capabilities",
    # Utility
    "get_loaded_modules",
]


def get_loaded_modules() -> dict:
    """Get the status of which XR submodules are loaded."""
    return dict(_LOADED_MODULES)


# Version info
__version__ = "1.0.0"
__author__ = "AI Game Engine Team"
