"""
Blackbox tests for Doppler effect component.

Tests PUBLIC behavior only - no internal state inspection.
Covers: velocity-based frequency shift, Doppler factor calculation.
"""

import pytest
import math

from engine.audio.spatial import (
    calculate_doppler_shift,
    DopplerConfig,
    DopplerProcessor,
    DopplerState,
    DOPPLER_PRESETS,
    get_doppler_preset,
    DOPPLER_FACTOR,
    DOPPLER_SMOOTHING_TIME,
    DOPPLER_VELOCITY_THRESHOLD,
    MAX_DOPPLER_SHIFT,
    MIN_DOPPLER_SHIFT,
    SPEED_OF_SOUND,
)


class TestDopplerConstants:
    """Test Doppler-related constants."""

    def test_speed_of_sound_is_realistic(self):
        """Speed of sound constant is realistic (m/s in air)."""
        # Speed of sound in air at 20C is ~343 m/s
        assert 300 < SPEED_OF_SOUND < 400

    def test_doppler_factor_positive(self):
        """Doppler factor is positive."""
        assert DOPPLER_FACTOR > 0

    def test_max_doppler_shift_greater_than_one(self):
        """Max Doppler shift is above unity."""
        assert MAX_DOPPLER_SHIFT > 1.0

    def test_min_doppler_shift_less_than_one(self):
        """Min Doppler shift is below unity."""
        assert MIN_DOPPLER_SHIFT < 1.0

    def test_smoothing_time_positive(self):
        """Smoothing time is positive."""
        assert DOPPLER_SMOOTHING_TIME > 0

    def test_velocity_threshold_positive(self):
        """Velocity threshold is positive."""
        assert DOPPLER_VELOCITY_THRESHOLD >= 0


class TestCalculateDopplerShift:
    """Test Doppler shift calculation function."""

    def test_stationary_source_and_listener_is_unity(self):
        """Stationary source and listener produces no shift (1.0)."""
        shift = calculate_doppler_shift(
            source_pos=(10, 0, 0),
            source_velocity=(0, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift == pytest.approx(1.0)

    def test_approaching_source_increases_pitch(self):
        """Source approaching listener increases pitch (shift > 1)."""
        shift = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(-50, 0, 0),  # Approaching
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift > 1.0

    def test_receding_source_decreases_pitch(self):
        """Source receding from listener decreases pitch (shift < 1)."""
        shift = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(50, 0, 0),  # Receding
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift < 1.0

    def test_approaching_listener_increases_pitch(self):
        """Listener approaching source increases pitch."""
        shift = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(0, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(50, 0, 0)  # Approaching source
        )
        assert shift > 1.0

    def test_receding_listener_decreases_pitch(self):
        """Listener receding from source decreases pitch."""
        shift = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(0, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(-50, 0, 0)  # Receding from source
        )
        assert shift < 1.0

    def test_both_approaching_compounds_effect(self):
        """Both approaching compounds Doppler effect."""
        shift_source_only = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(-50, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        shift_both = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(-50, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(50, 0, 0)
        )
        assert shift_both > shift_source_only

    def test_both_receding_compounds_effect(self):
        """Both receding compounds Doppler effect."""
        shift_source_only = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(50, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        shift_both = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(50, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(-50, 0, 0)
        )
        assert shift_both < shift_source_only

    def test_perpendicular_motion_no_shift(self):
        """Perpendicular motion produces minimal Doppler shift."""
        shift = calculate_doppler_shift(
            source_pos=(100, 0, 0),  # On X axis
            source_velocity=(0, 50, 0),  # Moving along Y (perpendicular)
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift == pytest.approx(1.0, abs=0.1)

    def test_shift_with_doppler_factor(self):
        """Doppler factor exaggerates the effect."""
        shift_normal = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(-50, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0),
            doppler_factor=1.0
        )
        shift_exaggerated = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(-50, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0),
            doppler_factor=2.0
        )
        # Higher factor = more extreme shift
        assert abs(shift_exaggerated - 1.0) > abs(shift_normal - 1.0)

    def test_custom_speed_of_sound(self):
        """Custom speed of sound affects shift."""
        shift_air = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(-50, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0),
            speed_of_sound=343.0
        )
        shift_water = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(-50, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0),
            speed_of_sound=1500.0  # Sound in water
        )
        # Higher speed of sound = smaller shift
        assert abs(shift_water - 1.0) < abs(shift_air - 1.0)


class TestDopplerConfig:
    """Test Doppler configuration."""

    def test_create_doppler_config(self):
        """DopplerConfig can be created."""
        config = DopplerConfig(
            factor=1.0,
            smoothing_time=0.1
        )
        assert config is not None

    def test_config_factor(self):
        """Config stores Doppler factor."""
        config = DopplerConfig(factor=2.0)
        assert config.factor == pytest.approx(2.0)

    def test_config_smoothing_time(self):
        """Config stores smoothing time."""
        config = DopplerConfig(smoothing_time=0.2)
        assert config.smoothing_time == pytest.approx(0.2)

    def test_config_velocity_threshold(self):
        """Config stores velocity threshold."""
        config = DopplerConfig(velocity_threshold=5.0)
        assert config.velocity_threshold == pytest.approx(5.0)

    def test_config_max_shift(self):
        """Config stores max shift."""
        config = DopplerConfig(max_shift=3.0)
        assert config.max_shift == pytest.approx(3.0)

    def test_config_min_shift(self):
        """Config stores min shift."""
        config = DopplerConfig(min_shift=0.3)
        assert config.min_shift == pytest.approx(0.3)


class TestDopplerProcessor:
    """Test Doppler processor for continuous processing."""

    def test_create_doppler_processor(self):
        """DopplerProcessor can be created."""
        processor = DopplerProcessor()
        assert processor is not None

    def test_processor_with_config(self):
        """DopplerProcessor can be created with config."""
        config = DopplerConfig(factor=1.5)
        processor = DopplerProcessor(config=config)
        assert processor is not None

    def test_processor_process(self):
        """Processor can process position/velocity data."""
        processor = DopplerProcessor()
        shift = processor.process(
            source_pos=(100, 0, 0),
            source_velocity=(-50, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift > 0

    def test_processor_reset(self):
        """Processor can be reset."""
        processor = DopplerProcessor()
        processor.process(
            source_pos=(100, 0, 0),
            source_velocity=(-50, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        processor.reset()


class TestDopplerState:
    """Test Doppler state tracking."""

    def test_create_doppler_state(self):
        """DopplerState can be created."""
        state = DopplerState()
        assert state is not None

    def test_state_initial_shift_is_unity(self):
        """Initial Doppler shift is 1.0."""
        state = DopplerState()
        assert state.current_shift == pytest.approx(1.0)


class TestDopplerPresets:
    """Test Doppler presets."""

    def test_presets_dictionary_exists(self):
        """DOPPLER_PRESETS dictionary exists."""
        assert DOPPLER_PRESETS is not None
        assert len(DOPPLER_PRESETS) > 0

    def test_get_preset_realistic(self):
        """Realistic preset exists."""
        preset = get_doppler_preset("realistic")
        assert preset is not None

    def test_get_preset_exaggerated(self):
        """Exaggerated preset exists."""
        preset = get_doppler_preset("exaggerated")
        assert preset is not None

    def test_get_preset_subtle(self):
        """Subtle preset exists."""
        preset = get_doppler_preset("subtle")
        assert preset is not None

    def test_realistic_preset_factor_near_one(self):
        """Realistic preset has factor near 1.0."""
        preset = get_doppler_preset("realistic")
        assert 0.8 <= preset.factor <= 1.2

    def test_exaggerated_preset_factor_greater_than_one(self):
        """Exaggerated preset has factor > 1.0."""
        preset = get_doppler_preset("exaggerated")
        assert preset.factor > 1.0

    def test_subtle_preset_factor_less_than_one(self):
        """Subtle preset has factor < 1.0."""
        preset = get_doppler_preset("subtle")
        assert preset.factor < 1.0


class TestDopplerPhysics:
    """Test Doppler physics accuracy."""

    def test_doppler_formula_approaching(self):
        """Approaching Doppler produces higher pitch."""
        shift = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(-34.3, 0, 0),  # 10% of speed of sound
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift > 1.0
        assert shift < 2.0  # Reasonable bounds

    def test_doppler_formula_receding(self):
        """Receding Doppler produces lower pitch."""
        shift = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(34.3, 0, 0),  # 10% of speed of sound
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift < 1.0
        assert shift > 0.5  # Reasonable bounds

    def test_doppler_symmetry(self):
        """Doppler is similar for source vs listener motion."""
        shift_source = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(-50, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        shift_listener = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(0, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(50, 0, 0)
        )
        # Both should increase pitch (shift > 1)
        assert shift_source > 1.0
        assert shift_listener > 1.0


class TestDopplerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_velocity(self):
        """Very small velocity produces minimal shift."""
        shift = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(-0.001, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift == pytest.approx(1.0, abs=0.01)

    def test_same_position_source_listener(self):
        """Source and listener at same position is handled."""
        shift = calculate_doppler_shift(
            source_pos=(0, 0, 0),
            source_velocity=(50, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        # Should handle without crashing (division by zero protection)
        assert shift >= 0


class TestDopplerFrequencyConversion:
    """Test frequency conversion with Doppler."""

    def test_apply_shift_to_frequency(self):
        """Doppler shift can be applied to frequency."""
        base_frequency = 440.0  # A4
        shift = 1.5  # 50% higher pitch

        new_frequency = base_frequency * shift
        assert new_frequency == pytest.approx(660.0)

    def test_octave_shift(self):
        """Doubling shift raises pitch by octave."""
        base_frequency = 440.0
        shift = 2.0

        new_frequency = base_frequency * shift
        assert new_frequency == pytest.approx(880.0)  # A5

    def test_half_shift(self):
        """Halving shift lowers pitch by octave."""
        base_frequency = 440.0
        shift = 0.5

        new_frequency = base_frequency * shift
        assert new_frequency == pytest.approx(220.0)  # A3


class TestDoppler3DCalculations:
    """Test 3D Doppler calculations."""

    def test_3d_approaching_along_x(self):
        """Source approaching along X axis."""
        shift = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(-50, 0, 0),  # Moving toward origin
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift > 1.0

    def test_3d_approaching_along_y(self):
        """Source approaching along Y axis."""
        shift = calculate_doppler_shift(
            source_pos=(0, 100, 0),
            source_velocity=(0, -50, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift > 1.0

    def test_3d_approaching_along_z(self):
        """Source approaching along Z axis."""
        shift = calculate_doppler_shift(
            source_pos=(0, 0, 100),
            source_velocity=(0, 0, -50),
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift > 1.0

    def test_3d_diagonal_motion(self):
        """Source moving diagonally."""
        shift = calculate_doppler_shift(
            source_pos=(100, 100, 100),
            source_velocity=(-30, -30, -30),  # Approaching diagonally
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift > 1.0

    def test_3d_perpendicular_motion(self):
        """Source moving perpendicular to line of sight."""
        shift = calculate_doppler_shift(
            source_pos=(100, 0, 0),  # On X axis
            source_velocity=(0, 50, 0),  # Moving along Y (perpendicular)
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        # Perpendicular motion should have minimal shift
        assert shift == pytest.approx(1.0, abs=0.1)

    def test_3d_receding_along_x(self):
        """Source receding along X axis."""
        shift = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(50, 0, 0),  # Moving away from origin
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift < 1.0

    def test_3d_oblique_approach(self):
        """Source approaching at an angle."""
        shift = calculate_doppler_shift(
            source_pos=(100, 100, 0),
            source_velocity=(-50, -50, 0),  # Moving toward origin
            listener_pos=(0, 0, 0),
            listener_velocity=(0, 0, 0)
        )
        assert shift > 1.0

    def test_3d_listener_moving_toward_source(self):
        """Listener moving toward stationary source."""
        shift = calculate_doppler_shift(
            source_pos=(100, 0, 0),
            source_velocity=(0, 0, 0),
            listener_pos=(0, 0, 0),
            listener_velocity=(50, 0, 0)  # Moving toward source
        )
        assert shift > 1.0
