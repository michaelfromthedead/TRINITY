"""XR session state machine management.

This module provides the XRSession class which manages the lifecycle and
state transitions of an XR session, following a strict state machine pattern
with enter/exit hooks for clean resource management.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Any, Dict, List

from engine.xr.runtime.capabilities import XRCapabilities, XRFeature


__all__ = [
    "XRSessionState",
    "XRSessionMode",
    "XRReferenceSpace",
    "XRSessionConfig",
    "XRSession",
    "XRSessionError",
    "InvalidStateTransitionError",
]


logger = logging.getLogger(__name__)


class XRSessionError(Exception):
    """Base exception for XR session errors."""


class InvalidStateTransitionError(XRSessionError):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, from_state: "XRSessionState", to_state: "XRSessionState") -> None:
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Invalid transition from {from_state.name} to {to_state.name}"
        )


class XRSessionState(Enum):
    """XR session state enumeration.

    States follow a strict lifecycle:
        IDLE -> READY -> RUNNING -> STOPPING -> IDLE
        Any state can transition to STOPPING (for error/shutdown)
        PAUSED can be entered from RUNNING and return to RUNNING
    """

    IDLE = auto()       # Not initialized
    READY = auto()      # Initialized but not presenting
    RUNNING = auto()    # Actively presenting
    PAUSED = auto()     # Temporarily suspended (e.g., HMD removed)
    STOPPING = auto()   # Shutting down
    ERROR = auto()      # Error state


class XRSessionMode(Enum):
    """XR session presentation mode."""

    IMMERSIVE_VR = auto()   # Full VR immersion
    IMMERSIVE_AR = auto()   # AR with passthrough
    INLINE = auto()         # Non-immersive (preview)


class XRReferenceSpace(Enum):
    """XR reference space types."""

    LOCAL = auto()          # Device-relative (seated)
    LOCAL_FLOOR = auto()    # Device-relative with floor
    BOUNDED_FLOOR = auto()  # Room-scale with boundaries
    UNBOUNDED = auto()      # Unlimited (AR/MR)
    VIEWER = auto()         # Head-locked


# Valid state transitions
_VALID_TRANSITIONS: Dict[XRSessionState, frozenset[XRSessionState]] = {
    XRSessionState.IDLE: frozenset({XRSessionState.READY, XRSessionState.ERROR}),
    XRSessionState.READY: frozenset({
        XRSessionState.RUNNING,
        XRSessionState.STOPPING,
        XRSessionState.ERROR,
    }),
    XRSessionState.RUNNING: frozenset({
        XRSessionState.PAUSED,
        XRSessionState.STOPPING,
        XRSessionState.ERROR,
    }),
    XRSessionState.PAUSED: frozenset({
        XRSessionState.RUNNING,
        XRSessionState.STOPPING,
        XRSessionState.ERROR,
    }),
    XRSessionState.STOPPING: frozenset({XRSessionState.IDLE, XRSessionState.ERROR}),
    XRSessionState.ERROR: frozenset({XRSessionState.IDLE}),
}


# Type alias for state hooks
StateHook = Callable[["XRSession"], None]


@dataclass(slots=True)
class XRSessionConfig:
    """Configuration for XR session initialization.

    Attributes:
        mode: The XR session mode (VR, AR, inline).
        reference_space: The reference space type.
        enable_hand_tracking: Whether to enable hand tracking if available.
        enable_eye_tracking: Whether to enable eye tracking if available.
        enable_passthrough: Whether to enable passthrough if available.
        render_scale: Initial render resolution scale (0.5 to 2.0).
        target_refresh_rate: Target display refresh rate in Hz.
        enable_foveation: Whether to enable foveated rendering if available.
        foveation_level: Foveation level (0-4) if enabled.
    """

    mode: XRSessionMode = XRSessionMode.IMMERSIVE_VR
    reference_space: XRReferenceSpace = XRReferenceSpace.LOCAL_FLOOR
    enable_hand_tracking: bool = True
    enable_eye_tracking: bool = False
    enable_passthrough: bool = False
    render_scale: float = 1.0
    target_refresh_rate: float = 90.0
    enable_foveation: bool = True
    foveation_level: int = 2


@dataclass(slots=True)
class XRSessionStats:
    """Runtime statistics for an XR session.

    Attributes:
        frames_presented: Total frames presented.
        dropped_frames: Number of dropped frames.
        start_time: Session start timestamp.
        total_runtime: Total runtime in seconds.
        average_framerate: Average framerate during session.
        reprojection_count: Number of reprojected frames.
    """

    frames_presented: int = 0
    dropped_frames: int = 0
    start_time: float = 0.0
    total_runtime: float = 0.0
    average_framerate: float = 0.0
    reprojection_count: int = 0


@dataclass(slots=True)
class XRSession:
    """XR session state machine.

    Manages the lifecycle of an XR session with strict state transitions,
    enter/exit hooks, and resource management.

    Example:
        >>> config = XRSessionConfig(mode=XRSessionMode.IMMERSIVE_VR)
        >>> session = XRSession(config)
        >>> session.add_enter_hook(XRSessionState.RUNNING, on_session_start)
        >>> session.add_exit_hook(XRSessionState.RUNNING, on_session_end)
        >>> session.initialize(capabilities)
        >>> session.start()  # Begins presenting
        >>> session.stop()   # Graceful shutdown
    """

    config: XRSessionConfig
    _state: XRSessionState = field(default=XRSessionState.IDLE, init=False)
    _capabilities: Optional[XRCapabilities] = field(default=None, init=False)
    _stats: XRSessionStats = field(default_factory=XRSessionStats, init=False)
    _enter_hooks: Dict[XRSessionState, List[StateHook]] = field(
        default_factory=lambda: {s: [] for s in XRSessionState},
        init=False,
    )
    _exit_hooks: Dict[XRSessionState, List[StateHook]] = field(
        default_factory=lambda: {s: [] for s in XRSessionState},
        init=False,
    )
    _state_history: List[tuple[XRSessionState, float]] = field(
        default_factory=list,
        init=False,
    )
    _enabled_features: frozenset[XRFeature] = field(
        default_factory=frozenset,
        init=False,
    )
    _error_message: Optional[str] = field(default=None, init=False)

    @property
    def state(self) -> XRSessionState:
        """Current session state."""
        return self._state

    @property
    def capabilities(self) -> Optional[XRCapabilities]:
        """Device capabilities (available after initialization)."""
        return self._capabilities

    @property
    def stats(self) -> XRSessionStats:
        """Session runtime statistics."""
        return self._stats

    @property
    def enabled_features(self) -> frozenset[XRFeature]:
        """Features currently enabled for this session."""
        return self._enabled_features

    @property
    def is_running(self) -> bool:
        """Check if session is actively presenting."""
        return self._state == XRSessionState.RUNNING

    @property
    def is_ready(self) -> bool:
        """Check if session is initialized and ready to start."""
        return self._state in (XRSessionState.READY, XRSessionState.RUNNING)

    @property
    def is_paused(self) -> bool:
        """Check if session is paused."""
        return self._state == XRSessionState.PAUSED

    @property
    def error_message(self) -> Optional[str]:
        """Error message if session is in error state."""
        return self._error_message

    def can_transition_to(self, target: XRSessionState) -> bool:
        """Check if transition to target state is valid.

        Args:
            target: The target state to check.

        Returns:
            True if transition is valid, False otherwise.
        """
        return target in _VALID_TRANSITIONS.get(self._state, frozenset())

    def add_enter_hook(self, state: XRSessionState, hook: StateHook) -> None:
        """Register a hook to be called when entering a state.

        Args:
            state: The state to hook.
            hook: Callable to invoke when entering the state.
        """
        self._enter_hooks[state].append(hook)

    def add_exit_hook(self, state: XRSessionState, hook: StateHook) -> None:
        """Register a hook to be called when exiting a state.

        Args:
            state: The state to hook.
            hook: Callable to invoke when exiting the state.
        """
        self._exit_hooks[state].append(hook)

    def remove_enter_hook(self, state: XRSessionState, hook: StateHook) -> bool:
        """Remove an enter hook.

        Args:
            state: The state the hook was registered for.
            hook: The hook to remove.

        Returns:
            True if hook was found and removed, False otherwise.
        """
        try:
            self._enter_hooks[state].remove(hook)
            return True
        except ValueError:
            return False

    def remove_exit_hook(self, state: XRSessionState, hook: StateHook) -> bool:
        """Remove an exit hook.

        Args:
            state: The state the hook was registered for.
            hook: The hook to remove.

        Returns:
            True if hook was found and removed, False otherwise.
        """
        try:
            self._exit_hooks[state].remove(hook)
            return True
        except ValueError:
            return False

    def _transition_to(self, target: XRSessionState) -> None:
        """Execute a state transition with hooks.

        Args:
            target: The target state.

        Raises:
            InvalidStateTransitionError: If transition is not valid.
        """
        if not self.can_transition_to(target):
            raise InvalidStateTransitionError(self._state, target)

        old_state = self._state
        timestamp = time.monotonic()

        # Run exit hooks for current state
        for hook in self._exit_hooks[old_state]:
            try:
                hook(self)
            except Exception as e:
                logger.exception(f"Error in exit hook for {old_state.name}: {e}")

        # Update state
        self._state = target
        self._state_history.append((target, timestamp))

        logger.debug(f"XRSession: {old_state.name} -> {target.name}")

        # Run enter hooks for new state
        for hook in self._enter_hooks[target]:
            try:
                hook(self)
            except Exception as e:
                logger.exception(f"Error in enter hook for {target.name}: {e}")

    def initialize(self, capabilities: XRCapabilities) -> bool:
        """Initialize the session with device capabilities.

        Args:
            capabilities: The XR device capabilities.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        if self._state != XRSessionState.IDLE:
            logger.warning("Cannot initialize: session not in IDLE state")
            return False

        self._capabilities = capabilities

        # Determine which features to enable based on config and capabilities
        enabled = set()

        # Always enable basic tracking if available
        if capabilities.supports(XRFeature.HEAD_TRACKING):
            enabled.add(XRFeature.HEAD_TRACKING)
        if capabilities.supports(XRFeature.CONTROLLER_TRACKING):
            enabled.add(XRFeature.CONTROLLER_TRACKING)

        # Optional features based on config
        if self.config.enable_hand_tracking and capabilities.has_hand_tracking:
            enabled.add(XRFeature.HAND_TRACKING)

        if self.config.enable_eye_tracking and capabilities.has_eye_tracking:
            enabled.add(XRFeature.EYE_TRACKING)

        if self.config.enable_passthrough and capabilities.has_passthrough:
            enabled.add(XRFeature.PASSTHROUGH)

        if self.config.enable_foveation and capabilities.has_foveated_rendering:
            enabled.add(XRFeature.FOVEATED_RENDERING)
            if capabilities.supports(XRFeature.DYNAMIC_FOVEATION):
                enabled.add(XRFeature.DYNAMIC_FOVEATION)

        self._enabled_features = frozenset(enabled)

        try:
            self._transition_to(XRSessionState.READY)
            logger.info(
                f"XRSession initialized with features: "
                f"{[f.name for f in self._enabled_features]}"
            )
            return True
        except InvalidStateTransitionError as e:
            self._set_error(str(e))
            return False

    def start(self) -> bool:
        """Start the XR session (begin presenting).

        Returns:
            True if session started successfully, False otherwise.
        """
        if self._state != XRSessionState.READY:
            logger.warning(f"Cannot start: session in {self._state.name} state")
            return False

        try:
            self._transition_to(XRSessionState.RUNNING)
            self._stats.start_time = time.monotonic()
            logger.info("XRSession started")
            return True
        except InvalidStateTransitionError as e:
            self._set_error(str(e))
            return False

    def pause(self) -> bool:
        """Pause the XR session.

        Returns:
            True if session paused successfully, False otherwise.
        """
        if self._state != XRSessionState.RUNNING:
            logger.warning(f"Cannot pause: session in {self._state.name} state")
            return False

        try:
            self._transition_to(XRSessionState.PAUSED)
            logger.info("XRSession paused")
            return True
        except InvalidStateTransitionError as e:
            self._set_error(str(e))
            return False

    def resume(self) -> bool:
        """Resume a paused XR session.

        Returns:
            True if session resumed successfully, False otherwise.
        """
        if self._state != XRSessionState.PAUSED:
            logger.warning(f"Cannot resume: session in {self._state.name} state")
            return False

        try:
            self._transition_to(XRSessionState.RUNNING)
            logger.info("XRSession resumed")
            return True
        except InvalidStateTransitionError as e:
            self._set_error(str(e))
            return False

    def stop(self) -> bool:
        """Stop the XR session gracefully.

        Returns:
            True if session stopped successfully, False otherwise.
        """
        if self._state not in (
            XRSessionState.READY,
            XRSessionState.RUNNING,
            XRSessionState.PAUSED,
        ):
            logger.warning(f"Cannot stop: session in {self._state.name} state")
            return False

        try:
            # Update runtime stats before stopping
            if self._stats.start_time > 0:
                self._stats.total_runtime = time.monotonic() - self._stats.start_time
                if self._stats.total_runtime > 0 and self._stats.frames_presented > 0:
                    self._stats.average_framerate = (
                        self._stats.frames_presented / self._stats.total_runtime
                    )

            self._transition_to(XRSessionState.STOPPING)
            self._transition_to(XRSessionState.IDLE)
            logger.info("XRSession stopped")
            return True
        except InvalidStateTransitionError as e:
            self._set_error(str(e))
            return False

    def reset(self) -> bool:
        """Reset the session from error state.

        Returns:
            True if reset succeeded, False otherwise.
        """
        if self._state != XRSessionState.ERROR:
            logger.warning("Cannot reset: session not in ERROR state")
            return False

        try:
            self._transition_to(XRSessionState.IDLE)
            self._error_message = None
            self._stats = XRSessionStats()
            logger.info("XRSession reset")
            return True
        except InvalidStateTransitionError:
            return False

    def update_frame_stats(
        self,
        frame_presented: bool = True,
        was_reprojected: bool = False,
    ) -> None:
        """Update frame statistics.

        Args:
            frame_presented: Whether a frame was successfully presented.
            was_reprojected: Whether the frame was reprojected (ATW/ASW).
        """
        if frame_presented:
            self._stats.frames_presented += 1
        else:
            self._stats.dropped_frames += 1

        if was_reprojected:
            self._stats.reprojection_count += 1

    def get_state_history(self) -> List[tuple[XRSessionState, float]]:
        """Get the history of state transitions.

        Returns:
            List of (state, timestamp) tuples.
        """
        return list(self._state_history)

    def _set_error(self, message: str) -> None:
        """Transition to error state with message.

        Args:
            message: Error description.
        """
        logger.error(f"XRSession error: {message}")
        self._error_message = message
        try:
            self._transition_to(XRSessionState.ERROR)
        except InvalidStateTransitionError:
            # Force error state if normal transition fails
            self._state = XRSessionState.ERROR
            self._state_history.append((XRSessionState.ERROR, time.monotonic()))

    def is_feature_enabled(self, feature: XRFeature) -> bool:
        """Check if a feature is enabled for this session.

        Args:
            feature: The feature to check.

        Returns:
            True if the feature is enabled, False otherwise.
        """
        return feature in self._enabled_features
