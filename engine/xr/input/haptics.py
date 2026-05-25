"""
XR Haptic feedback system.

Provides haptic effects for XR controllers including simple rumble,
patterned vibrations, and HD haptics support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.xr.input.controller import XRController, XRHand


class HapticType(Enum):
    """Types of haptic feedback effects."""
    RUMBLE = auto()           # Simple vibration
    PULSE = auto()            # Short pulse
    PATTERN = auto()          # Patterned vibration sequence
    HD_HAPTIC = auto()        # High-definition haptics
    ADAPTIVE = auto()         # Adaptive feedback (resistance, etc.)


class HapticWaveform(Enum):
    """Waveform shapes for haptic effects."""
    CONSTANT = auto()         # Constant amplitude
    SINE = auto()             # Sine wave
    SQUARE = auto()           # Square wave
    TRIANGLE = auto()         # Triangle wave
    SAWTOOTH = auto()         # Sawtooth wave
    CLICK = auto()            # Sharp click impulse
    BUZZ = auto()             # Buzzing sensation


@dataclass(slots=True)
class HapticEffect:
    """
    Describes a haptic feedback effect.

    Attributes:
        effect_type: Type of haptic effect
        amplitude: Vibration intensity (0.0 to 1.0)
        duration_ms: Effect duration in milliseconds
        frequency: Vibration frequency in Hz (for rumble)
        waveform: Waveform shape
        start_delay_ms: Delay before effect starts
        fade_in_ms: Fade-in duration
        fade_out_ms: Fade-out duration
    """
    effect_type: HapticType = HapticType.RUMBLE
    amplitude: float = 1.0
    duration_ms: float = 100.0
    frequency: float = 200.0
    waveform: HapticWaveform = HapticWaveform.CONSTANT
    start_delay_ms: float = 0.0
    fade_in_ms: float = 0.0
    fade_out_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate and clamp effect parameters."""
        self.amplitude = max(0.0, min(1.0, self.amplitude))
        self.duration_ms = max(0.0, self.duration_ms)
        self.frequency = max(0.0, min(500.0, self.frequency))
        self.start_delay_ms = max(0.0, self.start_delay_ms)
        self.fade_in_ms = max(0.0, self.fade_in_ms)
        self.fade_out_ms = max(0.0, self.fade_out_ms)

    @classmethod
    def click(cls, amplitude: float = 0.8) -> HapticEffect:
        """Create a short click effect."""
        return cls(
            effect_type=HapticType.PULSE,
            amplitude=amplitude,
            duration_ms=10.0,
            frequency=250.0,
            waveform=HapticWaveform.CLICK,
        )

    @classmethod
    def pulse(cls, amplitude: float = 0.6, duration_ms: float = 50.0) -> HapticEffect:
        """Create a pulse effect."""
        return cls(
            effect_type=HapticType.PULSE,
            amplitude=amplitude,
            duration_ms=duration_ms,
            frequency=200.0,
            waveform=HapticWaveform.SINE,
        )

    @classmethod
    def rumble(
        cls,
        amplitude: float = 0.5,
        duration_ms: float = 200.0,
        frequency: float = 160.0,
    ) -> HapticEffect:
        """Create a rumble effect."""
        return cls(
            effect_type=HapticType.RUMBLE,
            amplitude=amplitude,
            duration_ms=duration_ms,
            frequency=frequency,
            waveform=HapticWaveform.CONSTANT,
        )

    @classmethod
    def buzz(cls, amplitude: float = 0.4, duration_ms: float = 100.0) -> HapticEffect:
        """Create a buzzing effect."""
        return cls(
            effect_type=HapticType.RUMBLE,
            amplitude=amplitude,
            duration_ms=duration_ms,
            frequency=300.0,
            waveform=HapticWaveform.BUZZ,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize effect to dictionary."""
        return {
            "effect_type": self.effect_type.name,
            "amplitude": self.amplitude,
            "duration_ms": self.duration_ms,
            "frequency": self.frequency,
            "waveform": self.waveform.name,
            "start_delay_ms": self.start_delay_ms,
            "fade_in_ms": self.fade_in_ms,
            "fade_out_ms": self.fade_out_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HapticEffect:
        """Deserialize effect from dictionary."""
        return cls(
            effect_type=HapticType[data.get("effect_type", "RUMBLE")],
            amplitude=data.get("amplitude", 1.0),
            duration_ms=data.get("duration_ms", 100.0),
            frequency=data.get("frequency", 200.0),
            waveform=HapticWaveform[data.get("waveform", "CONSTANT")],
            start_delay_ms=data.get("start_delay_ms", 0.0),
            fade_in_ms=data.get("fade_in_ms", 0.0),
            fade_out_ms=data.get("fade_out_ms", 0.0),
        )


@dataclass
class HapticPattern:
    """
    A sequence of haptic effects forming a pattern.

    Attributes:
        name: Pattern name
        effects: List of (delay_ms, effect) tuples
        loop: Whether to loop the pattern
        loop_count: Number of loops (0 = infinite)
    """
    name: str = ""
    effects: List[Tuple[float, HapticEffect]] = field(default_factory=list)
    loop: bool = False
    loop_count: int = 0

    def add_effect(self, delay_ms: float, effect: HapticEffect) -> None:
        """Add an effect to the pattern."""
        self.effects.append((delay_ms, effect))

    @property
    def total_duration_ms(self) -> float:
        """Calculate total pattern duration."""
        if not self.effects:
            return 0.0

        max_end = 0.0
        for delay_ms, effect in self.effects:
            end = delay_ms + effect.start_delay_ms + effect.duration_ms
            max_end = max(max_end, end)

        return max_end

    @classmethod
    def heartbeat(cls) -> HapticPattern:
        """Create a heartbeat pattern."""
        pattern = cls(name="heartbeat")
        pattern.add_effect(0.0, HapticEffect.pulse(0.8, 60.0))
        pattern.add_effect(100.0, HapticEffect.pulse(0.5, 40.0))
        pattern.loop = True
        pattern.loop_count = 0  # Infinite
        return pattern

    @classmethod
    def success(cls) -> HapticPattern:
        """Create a success feedback pattern."""
        pattern = cls(name="success")
        pattern.add_effect(0.0, HapticEffect.pulse(0.6, 50.0))
        pattern.add_effect(80.0, HapticEffect.pulse(0.8, 80.0))
        return pattern

    @classmethod
    def error(cls) -> HapticPattern:
        """Create an error feedback pattern."""
        pattern = cls(name="error")
        pattern.add_effect(0.0, HapticEffect.buzz(0.9, 100.0))
        pattern.add_effect(150.0, HapticEffect.buzz(0.7, 80.0))
        pattern.add_effect(280.0, HapticEffect.buzz(0.5, 60.0))
        return pattern

    @classmethod
    def notification(cls) -> HapticPattern:
        """Create a notification pattern."""
        pattern = cls(name="notification")
        pattern.add_effect(0.0, HapticEffect.click(0.7))
        pattern.add_effect(50.0, HapticEffect.click(0.5))
        return pattern


@dataclass
class HapticCapabilities:
    """Haptic capabilities for a device."""
    supports_rumble: bool = True
    supports_hd_haptics: bool = False
    supports_adaptive: bool = False
    min_amplitude: float = 0.0
    max_amplitude: float = 1.0
    min_frequency: float = 0.0
    max_frequency: float = 500.0
    supported_waveforms: List[HapticWaveform] = field(
        default_factory=lambda: [HapticWaveform.CONSTANT]
    )


class HapticManager:
    """
    Manages haptic feedback for XR controllers.

    Handles effect queuing, playback, and device capability checking.
    """

    __slots__ = (
        "_device_capabilities",
        "_pending_effects",
        "_active_patterns",
        "_global_amplitude",
        "_enabled",
        "_on_effect_started",
        "_on_effect_completed",
    )

    def __init__(self) -> None:
        """Initialize the haptic manager."""
        # Device ID -> capabilities
        self._device_capabilities: Dict[str, HapticCapabilities] = {}
        # Device ID -> list of pending effects
        self._pending_effects: Dict[str, List[HapticEffect]] = {}
        # Device ID -> active pattern playback state
        self._active_patterns: Dict[str, Dict[str, Any]] = {}

        self._global_amplitude: float = 1.0
        self._enabled: bool = True

        # Callbacks
        self._on_effect_started: List[Callable[[str, HapticEffect], None]] = []
        self._on_effect_completed: List[Callable[[str, HapticEffect], None]] = []

    @property
    def enabled(self) -> bool:
        """Check if haptics are enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable haptics globally."""
        self._enabled = value
        if not value:
            self.stop_all()

    @property
    def global_amplitude(self) -> float:
        """Get global amplitude multiplier."""
        return self._global_amplitude

    @global_amplitude.setter
    def global_amplitude(self, value: float) -> None:
        """Set global amplitude multiplier (0-1)."""
        self._global_amplitude = max(0.0, min(1.0, value))

    # =========================================================================
    # DEVICE REGISTRATION
    # =========================================================================

    def register_device(
        self,
        device_id: str,
        capabilities: Optional[HapticCapabilities] = None,
    ) -> None:
        """
        Register a device with its haptic capabilities.

        Args:
            device_id: Unique device identifier
            capabilities: Device haptic capabilities
        """
        self._device_capabilities[device_id] = capabilities or HapticCapabilities()
        self._pending_effects[device_id] = []

    def unregister_device(self, device_id: str) -> None:
        """Unregister a device."""
        self._device_capabilities.pop(device_id, None)
        self._pending_effects.pop(device_id, None)
        self._active_patterns.pop(device_id, None)

    def get_capabilities(self, device_id: str) -> Optional[HapticCapabilities]:
        """Get capabilities for a device."""
        return self._device_capabilities.get(device_id)

    def supports_effect(self, device_id: str, effect: HapticEffect) -> bool:
        """Check if a device supports an effect."""
        caps = self._device_capabilities.get(device_id)
        if caps is None:
            return False

        if effect.effect_type == HapticType.RUMBLE and not caps.supports_rumble:
            return False
        if effect.effect_type == HapticType.HD_HAPTIC and not caps.supports_hd_haptics:
            return False
        if effect.effect_type == HapticType.ADAPTIVE and not caps.supports_adaptive:
            return False
        if effect.waveform not in caps.supported_waveforms:
            return False

        return True

    # =========================================================================
    # EFFECT PLAYBACK
    # =========================================================================

    def play(self, device_id: str, effect: HapticEffect) -> bool:
        """
        Queue a haptic effect for playback.

        Args:
            device_id: Target device ID
            effect: Effect to play

        Returns:
            True if effect was queued
        """
        if not self._enabled:
            return False

        if device_id not in self._pending_effects:
            return False

        if not self.supports_effect(device_id, effect):
            # Try to downgrade to simple rumble
            effect = HapticEffect(
                effect_type=HapticType.RUMBLE,
                amplitude=effect.amplitude,
                duration_ms=effect.duration_ms,
                frequency=effect.frequency,
            )

        # Apply global amplitude
        modified_effect = HapticEffect(
            effect_type=effect.effect_type,
            amplitude=effect.amplitude * self._global_amplitude,
            duration_ms=effect.duration_ms,
            frequency=effect.frequency,
            waveform=effect.waveform,
            start_delay_ms=effect.start_delay_ms,
            fade_in_ms=effect.fade_in_ms,
            fade_out_ms=effect.fade_out_ms,
        )

        self._pending_effects[device_id].append(modified_effect)

        for callback in self._on_effect_started:
            callback(device_id, modified_effect)

        return True

    def play_pattern(
        self,
        device_id: str,
        pattern: HapticPattern,
        pattern_id: Optional[str] = None,
    ) -> bool:
        """
        Start playing a haptic pattern.

        Args:
            device_id: Target device ID
            pattern: Pattern to play
            pattern_id: Optional ID for tracking (defaults to pattern name)

        Returns:
            True if pattern was started
        """
        if not self._enabled:
            return False

        if device_id not in self._pending_effects:
            return False

        pid = pattern_id or pattern.name or f"pattern_{id(pattern)}"

        self._active_patterns[device_id] = {
            "pattern": pattern,
            "pattern_id": pid,
            "current_index": 0,
            "loop_count": 0,
            "start_time": 0.0,
            "elapsed": 0.0,
        }

        return True

    def stop(self, device_id: str) -> None:
        """Stop all haptic effects on a device."""
        if device_id in self._pending_effects:
            self._pending_effects[device_id].clear()
        self._active_patterns.pop(device_id, None)

    def stop_pattern(self, device_id: str, pattern_id: Optional[str] = None) -> None:
        """Stop a specific pattern or all patterns on a device."""
        if device_id in self._active_patterns:
            if pattern_id is None:
                self._active_patterns.pop(device_id, None)
            elif self._active_patterns[device_id].get("pattern_id") == pattern_id:
                self._active_patterns.pop(device_id, None)

    def stop_all(self) -> None:
        """Stop all haptic effects on all devices."""
        for device_id in self._pending_effects:
            self._pending_effects[device_id].clear()
        self._active_patterns.clear()

    # =========================================================================
    # UPDATE AND RETRIEVAL
    # =========================================================================

    def update(self, delta_time_ms: float) -> None:
        """
        Update pattern playback state.

        Args:
            delta_time_ms: Time since last update in milliseconds
        """
        for device_id, state in list(self._active_patterns.items()):
            pattern: HapticPattern = state["pattern"]
            state["elapsed"] += delta_time_ms

            # Check for effects to trigger
            while state["current_index"] < len(pattern.effects):
                delay_ms, effect = pattern.effects[state["current_index"]]

                if state["elapsed"] >= delay_ms:
                    self.play(device_id, effect)
                    state["current_index"] += 1
                else:
                    break

            # Check for pattern completion
            if state["current_index"] >= len(pattern.effects):
                if pattern.loop:
                    if pattern.loop_count > 0 and state["loop_count"] >= pattern.loop_count:
                        # Done looping
                        self._active_patterns.pop(device_id, None)
                    else:
                        # Loop
                        state["current_index"] = 0
                        state["elapsed"] = 0.0
                        state["loop_count"] += 1
                else:
                    # Pattern complete
                    self._active_patterns.pop(device_id, None)

    def get_pending_effects(self, device_id: str) -> List[HapticEffect]:
        """
        Get and clear pending effects for a device.

        Args:
            device_id: Device to query

        Returns:
            List of pending effects
        """
        effects = self._pending_effects.get(device_id, [])
        self._pending_effects[device_id] = []
        return effects

    def has_pending_effects(self, device_id: str) -> bool:
        """Check if a device has pending effects."""
        return len(self._pending_effects.get(device_id, [])) > 0

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_effect_started(
        self,
        callback: Callable[[str, HapticEffect], None],
    ) -> None:
        """Register callback for when an effect starts."""
        self._on_effect_started.append(callback)

    def on_effect_completed(
        self,
        callback: Callable[[str, HapticEffect], None],
    ) -> None:
        """Register callback for when an effect completes."""
        self._on_effect_completed.append(callback)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================


_haptic_manager: Optional[HapticManager] = None


def get_haptic_manager() -> HapticManager:
    """Get the global haptic manager instance."""
    global _haptic_manager
    if _haptic_manager is None:
        _haptic_manager = HapticManager()
    return _haptic_manager


def reset_haptic_manager() -> None:
    """Reset the global haptic manager. Call during cleanup/testing."""
    global _haptic_manager
    if _haptic_manager is not None:
        _haptic_manager.stop_all()
    _haptic_manager = None


def play_haptic(
    device_id: str,
    amplitude: float = 1.0,
    duration_ms: float = 100.0,
    frequency: float = 200.0,
) -> bool:
    """
    Convenience function to play a simple haptic effect.

    Args:
        device_id: Target device ID
        amplitude: Vibration intensity (0-1)
        duration_ms: Duration in milliseconds
        frequency: Frequency in Hz

    Returns:
        True if effect was queued
    """
    return get_haptic_manager().play(
        device_id,
        HapticEffect.rumble(amplitude, duration_ms, frequency),
    )


def play_click(device_id: str, amplitude: float = 0.8) -> bool:
    """Play a click haptic effect."""
    return get_haptic_manager().play(device_id, HapticEffect.click(amplitude))


def play_pulse(
    device_id: str,
    amplitude: float = 0.6,
    duration_ms: float = 50.0,
) -> bool:
    """Play a pulse haptic effect."""
    return get_haptic_manager().play(device_id, HapticEffect.pulse(amplitude, duration_ms))


def stop_haptics(device_id: str) -> None:
    """Stop all haptic effects on a device."""
    get_haptic_manager().stop(device_id)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "HapticType",
    "HapticWaveform",
    # Data classes
    "HapticEffect",
    "HapticPattern",
    "HapticCapabilities",
    # Manager
    "HapticManager",
    "get_haptic_manager",
    "reset_haptic_manager",
    # Convenience functions
    "play_haptic",
    "play_click",
    "play_pulse",
    "stop_haptics",
]
