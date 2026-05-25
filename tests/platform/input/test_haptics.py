"""Tests for haptic feedback."""

import pytest

from engine.platform.input.haptics import HapticEffect, Haptics, HapticType
from engine.platform.input.gamepad import Gamepad


class TestHaptics:
    """Test suite for Haptics."""

    def test_initialization(self):
        """Test haptics system initialization."""
        haptics = Haptics()
        assert len(haptics.pending_effects) == 0

    def test_rumble_effect(self):
        """Test rumble haptic effect."""
        effect = HapticEffect(
            type=HapticType.RUMBLE,
            intensity=0.5,
            duration_ms=1000,
            frequency=200.0
        )

        assert effect.type == HapticType.RUMBLE
        assert effect.intensity == 0.5
        assert effect.duration_ms == 1000
        assert effect.frequency == 200.0

    def test_adaptive_trigger_effect(self):
        """Test adaptive trigger haptic effect."""
        effect = HapticEffect(
            type=HapticType.ADAPTIVE_TRIGGER,
            intensity=0.8,
            duration_ms=500,
            start_position=0.3,
            end_position=0.7,
            strength=0.6
        )

        assert effect.type == HapticType.ADAPTIVE_TRIGGER
        assert effect.start_position == 0.3
        assert effect.end_position == 0.7
        assert effect.strength == 0.6

    def test_hd_rumble_effect(self):
        """Test HD rumble haptic effect."""
        effect = HapticEffect(
            type=HapticType.HD_RUMBLE,
            intensity=0.7,
            duration_ms=200,
            frequency=320.0
        )

        assert effect.type == HapticType.HD_RUMBLE

    def test_intensity_clamping(self):
        """Test intensity is clamped to 0-1."""
        effect1 = HapticEffect(
            type=HapticType.RUMBLE,
            intensity=1.5,
            duration_ms=100
        )
        assert effect1.intensity == 1.0

        effect2 = HapticEffect(
            type=HapticType.RUMBLE,
            intensity=-0.5,
            duration_ms=100
        )
        assert effect2.intensity == 0.0

    def test_duration_clamping(self):
        """Test duration is clamped to non-negative."""
        effect = HapticEffect(
            type=HapticType.RUMBLE,
            intensity=0.5,
            duration_ms=-100
        )
        assert effect.duration_ms == 0

    def test_play_effect(self):
        """Test playing haptic effect on a device."""
        haptics = Haptics()
        gamepad = Gamepad(device_id=0)

        # Register device capabilities
        haptics.register_device_capabilities(
            gamepad,
            {HapticType.RUMBLE}
        )

        effect = HapticEffect(
            type=HapticType.RUMBLE,
            intensity=0.5,
            duration_ms=1000
        )

        success = haptics.play(gamepad, effect)
        assert success

        pending = haptics.pending_effects
        assert len(pending) == 1
        assert pending[0][0] == gamepad.id
        assert pending[0][1] == effect

    def test_play_unsupported_effect(self):
        """Test playing unsupported effect returns False."""
        haptics = Haptics()
        gamepad = Gamepad(device_id=0)

        # Register device with only rumble
        haptics.register_device_capabilities(
            gamepad,
            {HapticType.RUMBLE}
        )

        # Try to play adaptive trigger effect
        effect = HapticEffect(
            type=HapticType.ADAPTIVE_TRIGGER,
            intensity=0.5,
            duration_ms=100
        )

        success = haptics.play(gamepad, effect)
        assert not success
        assert len(haptics.pending_effects) == 0

    def test_stop_effects(self):
        """Test stopping all effects on a device."""
        haptics = Haptics()
        gamepad = Gamepad(device_id=0)

        haptics.register_device_capabilities(
            gamepad,
            {HapticType.RUMBLE}
        )

        # Play multiple effects
        for i in range(3):
            effect = HapticEffect(
                type=HapticType.RUMBLE,
                intensity=0.5,
                duration_ms=1000
            )
            haptics.play(gamepad, effect)

        assert len(haptics.pending_effects) == 3

        haptics.stop(gamepad)
        # Filter pending effects for this device
        device_effects = [e for e in haptics.pending_effects if e[0] == gamepad.id]
        assert len(device_effects) == 0

    def test_supports_hd_rumble(self):
        """Test HD rumble capability check."""
        haptics = Haptics()
        gamepad = Gamepad(device_id=0)

        haptics.register_device_capabilities(
            gamepad,
            {HapticType.HD_RUMBLE}
        )

        assert haptics.supports_hd_rumble(gamepad)

    def test_supports_adaptive_triggers(self):
        """Test adaptive trigger capability check."""
        haptics = Haptics()
        gamepad = Gamepad(device_id=0)

        haptics.register_device_capabilities(
            gamepad,
            {HapticType.ADAPTIVE_TRIGGER}
        )

        assert haptics.supports_adaptive_triggers(gamepad)

    def test_device_without_capabilities(self):
        """Test device without registered capabilities."""
        haptics = Haptics()
        gamepad = Gamepad(device_id=0)

        # Don't register capabilities
        assert not haptics.supports_hd_rumble(gamepad)
        assert not haptics.supports_adaptive_triggers(gamepad)

        effect = HapticEffect(
            type=HapticType.RUMBLE,
            intensity=0.5,
            duration_ms=100
        )

        success = haptics.play(gamepad, effect)
        assert not success

    def test_multiple_devices(self):
        """Test haptics with multiple devices."""
        haptics = Haptics()
        gamepad1 = Gamepad(device_id=0)
        gamepad2 = Gamepad(device_id=1)

        haptics.register_device_capabilities(
            gamepad1,
            {HapticType.RUMBLE}
        )
        haptics.register_device_capabilities(
            gamepad2,
            {HapticType.RUMBLE, HapticType.ADAPTIVE_TRIGGER}
        )

        effect = HapticEffect(
            type=HapticType.RUMBLE,
            intensity=0.5,
            duration_ms=100
        )

        haptics.play(gamepad1, effect)
        haptics.play(gamepad2, effect)

        pending = haptics.pending_effects
        assert len(pending) == 2

        device_ids = [e[0] for e in pending]
        assert 0 in device_ids
        assert 1 in device_ids

    def test_clear_pending(self):
        """Test clearing pending effects."""
        haptics = Haptics()
        gamepad = Gamepad(device_id=0)

        haptics.register_device_capabilities(
            gamepad,
            {HapticType.RUMBLE}
        )

        # Add effects
        for i in range(3):
            effect = HapticEffect(
                type=HapticType.RUMBLE,
                intensity=0.5,
                duration_ms=100
            )
            haptics.play(gamepad, effect)

        cleared = haptics.clear_pending(gamepad)
        assert len(cleared) == 3

        # Should be cleared now
        device_effects = [e for e in haptics.pending_effects if e[0] == gamepad.id]
        assert len(device_effects) == 0

    def test_parameter_validation(self):
        """Test haptic effect parameter validation."""
        # Test position clamping
        effect = HapticEffect(
            type=HapticType.ADAPTIVE_TRIGGER,
            intensity=0.5,
            duration_ms=100,
            start_position=1.5,
            end_position=-0.5,
            strength=2.0
        )

        assert effect.start_position == 1.0
        assert effect.end_position == 0.0
        assert effect.strength == 1.0

        # Test frequency clamping
        effect2 = HapticEffect(
            type=HapticType.RUMBLE,
            intensity=0.5,
            duration_ms=100,
            frequency=-100.0
        )

        assert effect2.frequency == 0.0

    def test_queuing_multiple_effects(self):
        """Test queuing multiple effects for same device."""
        haptics = Haptics()
        gamepad = Gamepad(device_id=0)

        haptics.register_device_capabilities(
            gamepad,
            {HapticType.RUMBLE, HapticType.HD_RUMBLE}
        )

        effect1 = HapticEffect(
            type=HapticType.RUMBLE,
            intensity=0.3,
            duration_ms=100
        )
        effect2 = HapticEffect(
            type=HapticType.HD_RUMBLE,
            intensity=0.7,
            duration_ms=200
        )

        haptics.play(gamepad, effect1)
        haptics.play(gamepad, effect2)

        pending = haptics.pending_effects
        assert len(pending) == 2
        assert pending[0][1].type == HapticType.RUMBLE
        assert pending[1][1].type == HapticType.HD_RUMBLE
