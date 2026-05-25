"""Haptic feedback system for input devices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .input_manager import InputDevice


class HapticType(Enum):
    """Types of haptic feedback effects."""
    RUMBLE = auto()  # Simple rumble/vibration
    ADAPTIVE_TRIGGER = auto()  # Adaptive trigger resistance (DualSense)
    HD_RUMBLE = auto()  # High-definition haptics (Nintendo Switch)


@dataclass(slots=True)
class HapticEffect:
    """Describes a haptic feedback effect."""
    type: HapticType
    intensity: float  # 0.0 to 1.0
    duration_ms: int
    frequency: Optional[float] = None  # For rumble effects (Hz)
    start_position: Optional[float] = None  # For adaptive triggers (0.0 to 1.0)
    end_position: Optional[float] = None  # For adaptive triggers (0.0 to 1.0)
    strength: Optional[float] = None  # For adaptive triggers (0.0 to 1.0)

    def __post_init__(self):
        """Validate effect parameters."""
        self.intensity = max(0.0, min(1.0, self.intensity))
        self.duration_ms = max(0, self.duration_ms)

        if self.frequency is not None:
            self.frequency = max(0.0, self.frequency)

        if self.start_position is not None:
            self.start_position = max(0.0, min(1.0, self.start_position))

        if self.end_position is not None:
            self.end_position = max(0.0, min(1.0, self.end_position))

        if self.strength is not None:
            self.strength = max(0.0, min(1.0, self.strength))


class Haptics:
    """Haptic feedback management system."""
    __slots__ = ('_pending_effects', '_device_capabilities')

    def __init__(self):
        """Initialize the haptics system."""
        self._pending_effects: dict[int, list[HapticEffect]] = {}
        self._device_capabilities: dict[int, set[HapticType]] = {}

    def play(self, device: InputDevice, effect: HapticEffect) -> bool:
        """Queue a haptic effect for a device.

        Args:
            device: The input device to play the effect on
            effect: The haptic effect to play

        Returns:
            True if effect was queued, False if device doesn't support it
        """
        device_id = device.id

        # Check if device supports this effect type
        capabilities = self._device_capabilities.get(device_id, set())
        if effect.type not in capabilities:
            return False

        # Add effect to queue
        if device_id not in self._pending_effects:
            self._pending_effects[device_id] = []

        self._pending_effects[device_id].append(effect)
        return True

    def stop(self, device: InputDevice) -> None:
        """Stop all haptic effects on a device.

        Args:
            device: The input device to stop effects on
        """
        device_id = device.id
        if device_id in self._pending_effects:
            self._pending_effects[device_id].clear()

    def supports_hd_rumble(self, device: InputDevice) -> bool:
        """Check if device supports HD rumble.

        Args:
            device: The input device to check

        Returns:
            True if device supports HD rumble
        """
        capabilities = self._device_capabilities.get(device.id, set())
        return HapticType.HD_RUMBLE in capabilities

    def supports_adaptive_triggers(self, device: InputDevice) -> bool:
        """Check if device supports adaptive triggers.

        Args:
            device: The input device to check

        Returns:
            True if device supports adaptive triggers
        """
        capabilities = self._device_capabilities.get(device.id, set())
        return HapticType.ADAPTIVE_TRIGGER in capabilities

    @property
    def pending_effects(self) -> list[tuple[int, HapticEffect]]:
        """Get all pending haptic effects for testing.

        Returns:
            List of (device_id, effect) tuples
        """
        effects = []
        for device_id, device_effects in self._pending_effects.items():
            for effect in device_effects:
                effects.append((device_id, effect))
        return effects

    def register_device_capabilities(
        self,
        device: InputDevice,
        capabilities: set[HapticType]
    ) -> None:
        """Register haptic capabilities for a device.

        Args:
            device: The input device
            capabilities: Set of supported haptic types
        """
        self._device_capabilities[device.id] = capabilities.copy()

    def clear_pending(self, device: InputDevice) -> list[HapticEffect]:
        """Clear and return pending effects for a device.

        Args:
            device: The input device

        Returns:
            List of pending effects that were cleared
        """
        device_id = device.id
        effects = self._pending_effects.get(device_id, [])
        self._pending_effects[device_id] = []
        return effects
