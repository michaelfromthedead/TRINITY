"""
Tests for XR haptic feedback system.

Tests haptic effects, patterns, manager, and convenience functions.
"""

import pytest

from engine.xr.input.haptics import (
    HapticType,
    HapticWaveform,
    HapticEffect,
    HapticPattern,
    HapticCapabilities,
    HapticManager,
    get_haptic_manager,
    play_haptic,
    play_click,
    play_pulse,
    stop_haptics,
)


# =============================================================================
# HAPTIC EFFECT TESTS
# =============================================================================


class TestHapticEffect:
    """Test HapticEffect dataclass."""

    def test_default_values(self):
        """Test default effect values."""
        effect = HapticEffect()

        assert effect.effect_type == HapticType.RUMBLE
        assert effect.amplitude == 1.0
        assert effect.duration_ms == 100.0
        assert effect.frequency == 200.0
        assert effect.waveform == HapticWaveform.CONSTANT

    def test_custom_values(self):
        """Test custom effect values."""
        effect = HapticEffect(
            effect_type=HapticType.PULSE,
            amplitude=0.7,
            duration_ms=50.0,
            frequency=300.0,
            waveform=HapticWaveform.SINE,
        )

        assert effect.effect_type == HapticType.PULSE
        assert effect.amplitude == 0.7
        assert effect.duration_ms == 50.0
        assert effect.frequency == 300.0
        assert effect.waveform == HapticWaveform.SINE

    def test_amplitude_clamping(self):
        """Test amplitude is clamped to 0-1."""
        effect1 = HapticEffect(amplitude=1.5)
        assert effect1.amplitude == 1.0

        effect2 = HapticEffect(amplitude=-0.5)
        assert effect2.amplitude == 0.0

    def test_duration_clamping(self):
        """Test duration is non-negative."""
        effect = HapticEffect(duration_ms=-100.0)
        assert effect.duration_ms == 0.0

    def test_frequency_clamping(self):
        """Test frequency is clamped."""
        effect1 = HapticEffect(frequency=-50.0)
        assert effect1.frequency == 0.0

        effect2 = HapticEffect(frequency=600.0)
        assert effect2.frequency == 500.0

    def test_fade_values(self):
        """Test fade in/out values."""
        effect = HapticEffect(
            fade_in_ms=20.0,
            fade_out_ms=30.0,
        )

        assert effect.fade_in_ms == 20.0
        assert effect.fade_out_ms == 30.0


# =============================================================================
# FACTORY METHOD TESTS
# =============================================================================


class TestHapticEffectFactories:
    """Test HapticEffect factory methods."""

    def test_click(self):
        """Test click effect factory."""
        effect = HapticEffect.click(amplitude=0.9)

        assert effect.effect_type == HapticType.PULSE
        assert effect.amplitude == 0.9
        assert effect.duration_ms == 10.0
        assert effect.waveform == HapticWaveform.CLICK

    def test_pulse(self):
        """Test pulse effect factory."""
        effect = HapticEffect.pulse(amplitude=0.7, duration_ms=80.0)

        assert effect.effect_type == HapticType.PULSE
        assert effect.amplitude == 0.7
        assert effect.duration_ms == 80.0
        assert effect.waveform == HapticWaveform.SINE

    def test_rumble(self):
        """Test rumble effect factory."""
        effect = HapticEffect.rumble(
            amplitude=0.5,
            duration_ms=150.0,
            frequency=180.0,
        )

        assert effect.effect_type == HapticType.RUMBLE
        assert effect.amplitude == 0.5
        assert effect.duration_ms == 150.0
        assert effect.frequency == 180.0

    def test_buzz(self):
        """Test buzz effect factory."""
        effect = HapticEffect.buzz(amplitude=0.6, duration_ms=120.0)

        assert effect.effect_type == HapticType.RUMBLE
        assert effect.waveform == HapticWaveform.BUZZ
        assert effect.frequency == 300.0


# =============================================================================
# HAPTIC EFFECT SERIALIZATION TESTS
# =============================================================================


class TestHapticEffectSerialization:
    """Test HapticEffect serialization."""

    def test_to_dict(self):
        """Test effect serializes to dictionary."""
        effect = HapticEffect(
            effect_type=HapticType.PULSE,
            amplitude=0.8,
            duration_ms=75.0,
            frequency=250.0,
            waveform=HapticWaveform.TRIANGLE,
        )

        data = effect.to_dict()

        assert data["effect_type"] == "PULSE"
        assert data["amplitude"] == 0.8
        assert data["duration_ms"] == 75.0
        assert data["frequency"] == 250.0
        assert data["waveform"] == "TRIANGLE"

    def test_from_dict(self):
        """Test effect deserializes from dictionary."""
        data = {
            "effect_type": "HD_HAPTIC",
            "amplitude": 0.65,
            "duration_ms": 200.0,
            "frequency": 150.0,
            "waveform": "SQUARE",
            "start_delay_ms": 10.0,
            "fade_in_ms": 5.0,
            "fade_out_ms": 10.0,
        }

        effect = HapticEffect.from_dict(data)

        assert effect.effect_type == HapticType.HD_HAPTIC
        assert effect.amplitude == 0.65
        assert effect.waveform == HapticWaveform.SQUARE
        assert effect.fade_in_ms == 5.0

    def test_round_trip(self):
        """Test serialization round-trip."""
        original = HapticEffect(
            effect_type=HapticType.PATTERN,
            amplitude=0.9,
            duration_ms=500.0,
            frequency=180.0,
            waveform=HapticWaveform.SAWTOOTH,
        )

        data = original.to_dict()
        restored = HapticEffect.from_dict(data)

        assert restored.effect_type == original.effect_type
        assert restored.amplitude == original.amplitude
        assert restored.duration_ms == original.duration_ms
        assert restored.waveform == original.waveform


# =============================================================================
# HAPTIC PATTERN TESTS
# =============================================================================


class TestHapticPattern:
    """Test HapticPattern."""

    def test_create_pattern(self):
        """Test creating a pattern."""
        pattern = HapticPattern(name="test_pattern")

        assert pattern.name == "test_pattern"
        assert len(pattern.effects) == 0
        assert pattern.loop is False

    def test_add_effect(self):
        """Test adding effects to pattern."""
        pattern = HapticPattern(name="multi_effect")
        pattern.add_effect(0.0, HapticEffect.click())
        pattern.add_effect(100.0, HapticEffect.pulse())

        assert len(pattern.effects) == 2
        assert pattern.effects[0][0] == 0.0
        assert pattern.effects[1][0] == 100.0

    def test_total_duration(self):
        """Test total duration calculation."""
        pattern = HapticPattern()
        pattern.add_effect(0.0, HapticEffect(duration_ms=50.0))
        pattern.add_effect(100.0, HapticEffect(duration_ms=100.0))

        # Duration = max(0 + 50, 100 + 100) = 200
        assert pattern.total_duration_ms == 200.0

    def test_loop_settings(self):
        """Test loop settings."""
        pattern = HapticPattern(loop=True, loop_count=3)

        assert pattern.loop is True
        assert pattern.loop_count == 3


# =============================================================================
# PRESET PATTERN TESTS
# =============================================================================


class TestPresetPatterns:
    """Test preset haptic patterns."""

    def test_heartbeat_pattern(self):
        """Test heartbeat pattern."""
        pattern = HapticPattern.heartbeat()

        assert pattern.name == "heartbeat"
        assert len(pattern.effects) == 2
        assert pattern.loop is True

    def test_success_pattern(self):
        """Test success pattern."""
        pattern = HapticPattern.success()

        assert pattern.name == "success"
        assert len(pattern.effects) == 2
        assert pattern.loop is False

    def test_error_pattern(self):
        """Test error pattern."""
        pattern = HapticPattern.error()

        assert pattern.name == "error"
        assert len(pattern.effects) == 3

    def test_notification_pattern(self):
        """Test notification pattern."""
        pattern = HapticPattern.notification()

        assert pattern.name == "notification"
        assert len(pattern.effects) == 2


# =============================================================================
# HAPTIC CAPABILITIES TESTS
# =============================================================================


class TestHapticCapabilities:
    """Test HapticCapabilities."""

    def test_default_capabilities(self):
        """Test default capabilities."""
        caps = HapticCapabilities()

        assert caps.supports_rumble is True
        assert caps.supports_hd_haptics is False
        assert caps.supports_adaptive is False
        assert caps.min_amplitude == 0.0
        assert caps.max_amplitude == 1.0

    def test_custom_capabilities(self):
        """Test custom capabilities."""
        caps = HapticCapabilities(
            supports_rumble=True,
            supports_hd_haptics=True,
            min_frequency=20.0,
            max_frequency=400.0,
            supported_waveforms=[
                HapticWaveform.CONSTANT,
                HapticWaveform.SINE,
                HapticWaveform.SQUARE,
            ],
        )

        assert caps.supports_hd_haptics is True
        assert caps.min_frequency == 20.0
        assert HapticWaveform.SINE in caps.supported_waveforms


# =============================================================================
# HAPTIC MANAGER TESTS
# =============================================================================


class TestHapticManager:
    """Test HapticManager."""

    def test_create_manager(self):
        """Test creating a haptic manager."""
        manager = HapticManager()

        assert manager.enabled is True
        assert manager.global_amplitude == 1.0

    def test_register_device(self):
        """Test registering a device."""
        manager = HapticManager()
        manager.register_device("device_1")

        caps = manager.get_capabilities("device_1")
        assert caps is not None

    def test_register_device_with_capabilities(self):
        """Test registering device with custom capabilities."""
        manager = HapticManager()
        caps = HapticCapabilities(supports_hd_haptics=True)
        manager.register_device("device_2", caps)

        retrieved = manager.get_capabilities("device_2")
        assert retrieved.supports_hd_haptics is True

    def test_unregister_device(self):
        """Test unregistering a device."""
        manager = HapticManager()
        manager.register_device("device_3")
        manager.unregister_device("device_3")

        caps = manager.get_capabilities("device_3")
        assert caps is None

    def test_play_effect(self):
        """Test playing an effect."""
        manager = HapticManager()
        manager.register_device("play_device")

        effect = HapticEffect.rumble()
        result = manager.play("play_device", effect)

        assert result is True
        assert manager.has_pending_effects("play_device")

    def test_play_disabled(self):
        """Test playing when disabled."""
        manager = HapticManager()
        manager.register_device("disabled_device")
        manager.enabled = False

        result = manager.play("disabled_device", HapticEffect.click())

        assert result is False

    def test_play_unknown_device(self):
        """Test playing to unknown device."""
        manager = HapticManager()

        result = manager.play("unknown", HapticEffect.click())

        assert result is False

    def test_get_pending_effects(self):
        """Test getting and clearing pending effects."""
        manager = HapticManager()
        manager.register_device("pending_device")

        manager.play("pending_device", HapticEffect.click())
        manager.play("pending_device", HapticEffect.pulse())

        effects = manager.get_pending_effects("pending_device")

        assert len(effects) == 2
        assert not manager.has_pending_effects("pending_device")

    def test_stop_device(self):
        """Test stopping effects on device."""
        manager = HapticManager()
        manager.register_device("stop_device")

        manager.play("stop_device", HapticEffect.rumble())
        manager.stop("stop_device")

        assert not manager.has_pending_effects("stop_device")

    def test_stop_all(self):
        """Test stopping all effects."""
        manager = HapticManager()
        manager.register_device("device_a")
        manager.register_device("device_b")

        manager.play("device_a", HapticEffect.click())
        manager.play("device_b", HapticEffect.pulse())
        manager.stop_all()

        assert not manager.has_pending_effects("device_a")
        assert not manager.has_pending_effects("device_b")

    def test_global_amplitude(self):
        """Test global amplitude multiplier."""
        manager = HapticManager()
        manager.register_device("amp_device")
        manager.global_amplitude = 0.5

        manager.play("amp_device", HapticEffect(amplitude=1.0))
        effects = manager.get_pending_effects("amp_device")

        assert effects[0].amplitude == 0.5

    def test_global_amplitude_clamping(self):
        """Test global amplitude is clamped."""
        manager = HapticManager()

        manager.global_amplitude = 1.5
        assert manager.global_amplitude == 1.0

        manager.global_amplitude = -0.5
        assert manager.global_amplitude == 0.0


# =============================================================================
# PATTERN PLAYBACK TESTS
# =============================================================================


class TestPatternPlayback:
    """Test pattern playback."""

    def test_play_pattern(self):
        """Test starting a pattern."""
        manager = HapticManager()
        manager.register_device("pattern_device")

        pattern = HapticPattern.success()
        result = manager.play_pattern("pattern_device", pattern)

        assert result is True

    def test_play_pattern_disabled(self):
        """Test pattern when disabled."""
        manager = HapticManager()
        manager.register_device("disabled_pattern")
        manager.enabled = False

        result = manager.play_pattern("disabled_pattern", HapticPattern.success())

        assert result is False

    def test_stop_pattern(self):
        """Test stopping a pattern."""
        manager = HapticManager()
        manager.register_device("stop_pattern_device")

        pattern = HapticPattern.heartbeat()
        manager.play_pattern("stop_pattern_device", pattern, "heartbeat")
        manager.stop_pattern("stop_pattern_device", "heartbeat")

        # Pattern should be stopped

    def test_pattern_update(self):
        """Test pattern update triggers effects."""
        manager = HapticManager()
        manager.register_device("update_device")

        pattern = HapticPattern()
        pattern.add_effect(0.0, HapticEffect.click())
        pattern.add_effect(50.0, HapticEffect.pulse())

        manager.play_pattern("update_device", pattern)

        # First update should trigger first effect
        manager.update(10.0)
        effects = manager.get_pending_effects("update_device")
        assert len(effects) >= 1


# =============================================================================
# CAPABILITY CHECK TESTS
# =============================================================================


class TestCapabilityChecks:
    """Test capability checking."""

    def test_supports_effect_rumble(self):
        """Test rumble support check."""
        manager = HapticManager()
        manager.register_device("rumble_device", HapticCapabilities(supports_rumble=True))

        effect = HapticEffect.rumble()
        assert manager.supports_effect("rumble_device", effect) is True

    def test_supports_effect_hd(self):
        """Test HD haptics support check."""
        manager = HapticManager()

        # Device without HD support
        manager.register_device("no_hd", HapticCapabilities(supports_hd_haptics=False))

        effect = HapticEffect(effect_type=HapticType.HD_HAPTIC)
        assert manager.supports_effect("no_hd", effect) is False

        # Device with HD support
        manager.register_device("has_hd", HapticCapabilities(supports_hd_haptics=True))
        assert manager.supports_effect("has_hd", effect) is True

    def test_supports_effect_waveform(self):
        """Test waveform support check."""
        manager = HapticManager()
        manager.register_device(
            "waveform_device",
            HapticCapabilities(
                supported_waveforms=[HapticWaveform.CONSTANT, HapticWaveform.SINE]
            ),
        )

        sine_effect = HapticEffect(waveform=HapticWaveform.SINE)
        assert manager.supports_effect("waveform_device", sine_effect) is True

        square_effect = HapticEffect(waveform=HapticWaveform.SQUARE)
        assert manager.supports_effect("waveform_device", square_effect) is False


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_haptic_manager(self):
        """Test getting global manager."""
        manager1 = get_haptic_manager()
        manager2 = get_haptic_manager()

        assert manager1 is manager2

    def test_play_haptic_function(self):
        """Test play_haptic convenience function."""
        manager = get_haptic_manager()
        manager.register_device("conv_device_1")

        result = play_haptic(
            "conv_device_1",
            amplitude=0.7,
            duration_ms=150.0,
            frequency=180.0,
        )

        assert result is True

    def test_play_click_function(self):
        """Test play_click convenience function."""
        manager = get_haptic_manager()
        manager.register_device("click_device")

        result = play_click("click_device", amplitude=0.9)

        assert result is True

    def test_play_pulse_function(self):
        """Test play_pulse convenience function."""
        manager = get_haptic_manager()
        manager.register_device("pulse_device")

        result = play_pulse("pulse_device", amplitude=0.6, duration_ms=80.0)

        assert result is True

    def test_stop_haptics_function(self):
        """Test stop_haptics convenience function."""
        manager = get_haptic_manager()
        manager.register_device("stop_conv_device")

        play_haptic("stop_conv_device")
        stop_haptics("stop_conv_device")

        assert not manager.has_pending_effects("stop_conv_device")


# =============================================================================
# CALLBACK TESTS
# =============================================================================


class TestHapticCallbacks:
    """Test haptic callbacks."""

    def test_effect_started_callback(self):
        """Test effect started callback."""
        manager = HapticManager()
        manager.register_device("callback_device")

        started_effects = []
        manager.on_effect_started(
            lambda device, effect: started_effects.append((device, effect))
        )

        effect = HapticEffect.click()
        manager.play("callback_device", effect)

        assert len(started_effects) == 1
        assert started_effects[0][0] == "callback_device"


# =============================================================================
# ENUM TESTS
# =============================================================================


class TestHapticEnums:
    """Test haptic enums."""

    def test_haptic_type_values(self):
        """Test HapticType enum values."""
        assert HapticType.RUMBLE.name == "RUMBLE"
        assert HapticType.PULSE.name == "PULSE"
        assert HapticType.PATTERN.name == "PATTERN"
        assert HapticType.HD_HAPTIC.name == "HD_HAPTIC"
        assert HapticType.ADAPTIVE.name == "ADAPTIVE"

    def test_haptic_waveform_values(self):
        """Test HapticWaveform enum values."""
        assert HapticWaveform.CONSTANT.name == "CONSTANT"
        assert HapticWaveform.SINE.name == "SINE"
        assert HapticWaveform.SQUARE.name == "SQUARE"
        assert HapticWaveform.TRIANGLE.name == "TRIANGLE"
        assert HapticWaveform.SAWTOOTH.name == "SAWTOOTH"
        assert HapticWaveform.CLICK.name == "CLICK"
        assert HapticWaveform.BUZZ.name == "BUZZ"
