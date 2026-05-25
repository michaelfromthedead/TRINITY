"""
Time Control - Pause, slow motion, fast forward, and frame stepping.

Provides a TimeController for manipulating game time during debugging.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class TimeState(Enum):
    """Current state of time control."""
    NORMAL = auto()
    PAUSED = auto()
    SLOW_MOTION = auto()
    FAST_FORWARD = auto()
    FRAME_STEP = auto()


@dataclass
class TimeControlConfig:
    """
    Configuration for time control.

    All time scale values and limits are defined here to avoid magic numbers.
    Override these values via config files or dependency injection.
    """
    # Time scale limits
    min_time_scale: float = 0.01  # Minimum time scale (1% speed)
    max_time_scale: float = 10.0  # Maximum time scale (10x speed)

    # Default presets
    default_slow_motion: float = 0.25  # 25% speed for slow-mo
    default_fast_forward: float = 2.0  # 2x speed for fast-forward
    frame_step_scale: float = 1.0  # Scale during frame stepping

    # Preset time scales (named constants for common values)
    preset_super_slow: float = 0.1   # 10% speed
    preset_slow: float = 0.25        # 25% speed
    preset_half: float = 0.5         # 50% speed
    preset_normal: float = 1.0       # 100% speed (real-time)
    preset_fast: float = 2.0         # 200% speed
    preset_super_fast: float = 4.0   # 400% speed

    # Build restrictions
    allow_in_shipping: bool = False  # Time control disabled in shipping builds


class TimeController:
    """
    Controls game time for debugging.

    SECURITY: This controller is automatically disabled in shipping builds
    to prevent time manipulation exploits.

    Supports:
    - Pause/resume
    - Time scale modification (slow motion, fast forward)
    - Frame stepping while paused
    - Time scale presets
    - Callbacks for state changes
    """

    def __init__(self, config: Optional[TimeControlConfig] = None) -> None:
        self._config = config or TimeControlConfig()

        # Initialize state
        self._time_scale: float = 1.0
        self._paused: bool = False
        self._state: TimeState = TimeState.NORMAL
        self._stored_scale: float = 1.0
        self._frame_step_pending: bool = False
        self._frame_step_count: int = 0
        self._pause_callbacks: List[Callable[[bool], None]] = []
        self._scale_callbacks: List[Callable[[float], None]] = []
        self._state_callbacks: List[Callable[[TimeState], None]] = []

        # Timing tracking
        self._pause_time: float = 0.0
        self._total_pause_duration: float = 0.0

        # Apply build restrictions
        self._build_allowed = self._check_build_allowed()

    def _check_build_allowed(self) -> bool:
        """Check if time control is allowed in this build."""
        import os

        # Check for shipping build
        if os.environ.get("GAME_BUILD_TYPE", "").upper() == "SHIPPING":
            if not self._config.allow_in_shipping:
                logger.info("TimeController disabled - shipping build")
                return False
        if os.environ.get("SHIPPING") == "1":
            if not self._config.allow_in_shipping:
                return False

        return True

    # Class-level presets that use config values
    @property
    def PRESET_SUPER_SLOW(self) -> float:
        """Get super slow preset value."""
        return self._config.preset_super_slow

    @property
    def PRESET_SLOW(self) -> float:
        """Get slow preset value."""
        return self._config.preset_slow

    @property
    def PRESET_HALF(self) -> float:
        """Get half-speed preset value."""
        return self._config.preset_half

    @property
    def PRESET_NORMAL(self) -> float:
        """Get normal speed preset value."""
        return self._config.preset_normal

    @property
    def PRESET_FAST(self) -> float:
        """Get fast preset value."""
        return self._config.preset_fast

    @property
    def PRESET_SUPER_FAST(self) -> float:
        """Get super fast preset value."""
        return self._config.preset_super_fast

    @property
    def time_scale(self) -> float:
        """Get the current time scale."""
        return self._time_scale

    @property
    def is_paused(self) -> bool:
        """Check if time is paused."""
        return self._paused

    @property
    def state(self) -> TimeState:
        """Get the current time state."""
        return self._state

    @property
    def frame_step_pending(self) -> bool:
        """Check if a frame step is pending."""
        return self._frame_step_pending

    def pause(self) -> None:
        """Pause the game time."""
        if not self._build_allowed:
            logger.warning("Time control not allowed in this build")
            return

        if self._paused:
            return

        self._paused = True
        self._stored_scale = self._time_scale
        self._time_scale = 0.0
        self._state = TimeState.PAUSED
        self._pause_time = time.time()

        logger.info("Time paused")
        self._notify_pause_callbacks(True)
        self._notify_state_callbacks(TimeState.PAUSED)

    def resume(self) -> None:
        """Resume the game time."""
        if not self._paused:
            return

        self._paused = False
        self._time_scale = self._stored_scale
        self._frame_step_pending = False
        self._update_state_from_scale()

        # Track pause duration
        pause_duration = time.time() - self._pause_time
        self._total_pause_duration += pause_duration

        logger.info("Time resumed (was paused for %.2fs)", pause_duration)
        self._notify_pause_callbacks(False)
        self._notify_state_callbacks(self._state)

    def toggle_pause(self) -> bool:
        """Toggle pause state. Returns new paused state."""
        if self._paused:
            self.resume()
        else:
            self.pause()
        return self._paused

    def set_time_scale(self, scale: float) -> float:
        """
        Set the time scale.

        Args:
            scale: Time scale (configurable, default 0.01 to 10.0)

        Returns:
            The clamped time scale that was set.
        """
        if not self._build_allowed:
            logger.warning("Time control not allowed in this build")
            return 1.0

        old_scale = self._time_scale

        # Clamp to valid range
        scale = max(self._config.min_time_scale, min(scale, self._config.max_time_scale))

        if self._paused:
            self._stored_scale = scale
        else:
            self._time_scale = scale
            self._update_state_from_scale()

        if scale != old_scale:
            logger.info("Time scale: %.2f -> %.2f", old_scale, scale)
            self._notify_scale_callbacks(scale)
            if not self._paused:
                self._notify_state_callbacks(self._state)

        return scale

    def get_time_scale(self) -> float:
        """Get the current time scale (or stored scale if paused)."""
        if self._paused:
            return self._stored_scale
        return self._time_scale

    def step_frame(self, count: int = 1) -> int:
        """
        Advance specified number of frames while paused.

        Args:
            count: Number of frames to advance

        Returns:
            Number of frame steps queued.
        """
        if not self._paused:
            logger.warning("Frame step ignored - not paused")
            return 0

        self._frame_step_pending = True
        self._frame_step_count = max(1, count)
        self._state = TimeState.FRAME_STEP

        logger.debug("Frame step: %d frames", self._frame_step_count)
        self._notify_state_callbacks(TimeState.FRAME_STEP)

        return self._frame_step_count

    def consume_frame_step(self) -> bool:
        """
        Consume a pending frame step.

        Called by the game loop to check if it should execute one frame.
        Returns True if a frame should be executed.
        """
        if not self._frame_step_pending:
            return False

        self._frame_step_count -= 1
        if self._frame_step_count <= 0:
            self._frame_step_pending = False
            self._frame_step_count = 0
            self._state = TimeState.PAUSED
            self._notify_state_callbacks(TimeState.PAUSED)

        return True

    def set_slow_motion(self, scale: Optional[float] = None) -> float:
        """
        Enter slow motion mode.

        Args:
            scale: Time scale (default from config)

        Returns:
            The time scale set.
        """
        scale = scale or self._config.default_slow_motion
        return self.set_time_scale(scale)

    def set_fast_forward(self, scale: Optional[float] = None) -> float:
        """
        Enter fast forward mode.

        Args:
            scale: Time scale (default from config)

        Returns:
            The time scale set.
        """
        scale = scale or self._config.default_fast_forward
        return self.set_time_scale(scale)

    def reset(self) -> None:
        """Reset to normal time."""
        self._paused = False
        self._time_scale = 1.0
        self._stored_scale = 1.0
        self._frame_step_pending = False
        self._frame_step_count = 0
        self._state = TimeState.NORMAL

        logger.info("Time reset to normal")
        self._notify_pause_callbacks(False)
        self._notify_scale_callbacks(1.0)
        self._notify_state_callbacks(TimeState.NORMAL)

    def apply_preset(self, preset: float) -> float:
        """Apply a time scale preset."""
        return self.set_time_scale(preset)

    def _update_state_from_scale(self) -> None:
        """Update the state based on current time scale."""
        if self._time_scale < 1.0:
            self._state = TimeState.SLOW_MOTION
        elif self._time_scale > 1.0:
            self._state = TimeState.FAST_FORWARD
        else:
            self._state = TimeState.NORMAL

    # =========================================================================
    # Callbacks
    # =========================================================================

    def add_pause_callback(self, callback: Callable[[bool], None]) -> None:
        """Add a callback for pause state changes."""
        self._pause_callbacks.append(callback)

    def remove_pause_callback(self, callback: Callable[[bool], None]) -> bool:
        """Remove a pause callback."""
        try:
            self._pause_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def add_scale_callback(self, callback: Callable[[float], None]) -> None:
        """Add a callback for time scale changes."""
        self._scale_callbacks.append(callback)

    def remove_scale_callback(self, callback: Callable[[float], None]) -> bool:
        """Remove a scale callback."""
        try:
            self._scale_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def add_state_callback(self, callback: Callable[[TimeState], None]) -> None:
        """Add a callback for state changes."""
        self._state_callbacks.append(callback)

    def remove_state_callback(self, callback: Callable[[TimeState], None]) -> bool:
        """Remove a state callback."""
        try:
            self._state_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def _notify_pause_callbacks(self, paused: bool) -> None:
        """Notify pause callbacks."""
        for callback in self._pause_callbacks:
            try:
                callback(paused)
            except Exception as e:
                logger.error("Pause callback error: %s", e)

    def _notify_scale_callbacks(self, scale: float) -> None:
        """Notify scale callbacks."""
        for callback in self._scale_callbacks:
            try:
                callback(scale)
            except Exception as e:
                logger.error("Scale callback error: %s", e)

    def _notify_state_callbacks(self, state: TimeState) -> None:
        """Notify state callbacks."""
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error("State callback error: %s", e)

    # =========================================================================
    # Console Commands
    # =========================================================================

    def cmd_pause(self) -> str:
        """Console command: pause"""
        self.pause()
        return "Game paused"

    def cmd_resume(self) -> str:
        """Console command: resume"""
        self.resume()
        return "Game resumed"

    def cmd_slomo(self, scale: float) -> str:
        """Console command: slomo <scale>"""
        actual = self.set_time_scale(scale)
        return f"Time scale set to {actual:.2f}"

    def cmd_step(self, count: int = 1) -> str:
        """Console command: step [count]"""
        stepped = self.step_frame(count)
        if stepped > 0:
            return f"Stepping {stepped} frame(s)"
        return "Cannot step - game is not paused"


# =============================================================================
# Console command registration helper
# =============================================================================

def register_time_commands(controller: TimeController, console: Any) -> None:
    """
    Register time control commands with a console.

    Args:
        controller: TimeController instance
        console: Console instance with register() method
    """
    try:
        console.register("pause", controller.cmd_pause, "Pause the game")
        console.register("resume", controller.cmd_resume, "Resume the game")
        console.register("slomo", controller.cmd_slomo, "Set time scale (slomo <scale>)")
        console.register("step", controller.cmd_step, "Step frame(s) while paused")
    except Exception as e:
        logger.warning("Failed to register time commands: %s", e)


# =============================================================================
# Singleton instance
# =============================================================================

_time_controller: Optional[TimeController] = None


def get_time_controller() -> TimeController:
    """Get the global time controller instance."""
    global _time_controller
    if _time_controller is None:
        _time_controller = TimeController()
    return _time_controller


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "TimeController",
    "TimeControlConfig",
    "TimeState",
    "get_time_controller",
    "register_time_commands",
]
