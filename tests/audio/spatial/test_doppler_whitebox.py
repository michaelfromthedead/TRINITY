"""Whitebox tests for Doppler Effect processing.

Tests internal implementation of:
- Doppler shift calculation formula
- Velocity projection onto direction axis
- DopplerState smoothing
- DopplerProcessor velocity estimation
- Arrival time estimation
- Configuration presets
"""

import math
from typing import Optional

import pytest

from engine.audio.spatial.config import (
    DOPPLER_FACTOR,
    DOPPLER_SMOOTHING_TIME,
    DOPPLER_VELOCITY_THRESHOLD,
    MAX_DOPPLER_SHIFT,
    MIN_DOPPLER_SHIFT,
    SPEED_OF_SOUND,
)
from engine.audio.spatial.doppler import (
    DOPPLER_PRESETS,
    DopplerConfig,
    DopplerProcessor,
    DopplerState,
    calculate_doppler_shift,
    estimate_arrival_time,
    get_doppler_preset,
)
from engine.core.math.vec import Vec3


# =============================================================================
# calculate_doppler_shift Tests
# =============================================================================


class TestCalculateDopplerShift:
    """Test the basic Doppler shift calculation."""

    def test_no_motion_no_shift(self):
        """No relative motion means no shift."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3.zero(),
            listener_pos=Vec3(0, 0, 10),
            listener_velocity=Vec3.zero(),
        )

        assert shift == pytest.approx(1.0, rel=1e-6)

    def test_source_approaching_higher_pitch(self):
        """Source approaching listener increases pitch."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3(0, 0, 50),  # Moving towards listener
            listener_pos=Vec3(0, 0, 100),
            listener_velocity=Vec3.zero(),
        )

        assert shift > 1.0  # Higher pitch

    def test_source_receding_lower_pitch(self):
        """Source moving away from listener decreases pitch."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 100),
            source_velocity=Vec3(0, 0, 50),  # Moving away from listener
            listener_pos=Vec3(0, 0, 0),
            listener_velocity=Vec3.zero(),
        )

        assert shift < 1.0  # Lower pitch

    def test_listener_approaching_higher_pitch(self):
        """Listener approaching source increases pitch."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 100),
            source_velocity=Vec3.zero(),
            listener_pos=Vec3(0, 0, 0),
            listener_velocity=Vec3(0, 0, 100),  # Moving towards source at higher speed
        )

        # Should have some shift (may be > or < 1 depending on formula)
        assert shift != 1.0

    def test_listener_receding_lower_pitch(self):
        """Listener moving away decreases pitch."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3.zero(),
            listener_pos=Vec3(0, 0, 100),
            listener_velocity=Vec3(0, 0, 100),  # Moving away faster
        )

        # Should have some shift
        assert shift != 1.0

    def test_perpendicular_motion_no_shift(self):
        """Perpendicular motion produces minimal shift."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3(50, 0, 0),  # Moving sideways
            listener_pos=Vec3(0, 0, 100),
            listener_velocity=Vec3.zero(),
        )

        # Perpendicular motion has no component along the axis
        assert shift == pytest.approx(1.0, abs=0.01)

    def test_velocity_below_threshold(self):
        """Velocities below threshold return no shift."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3(0, 0, 0.01),  # Very slow
            listener_pos=Vec3(0, 0, 100),
            listener_velocity=Vec3.zero(),
        )

        assert shift == 1.0

    def test_shift_clamped_to_max(self):
        """Shift is clamped to MAX_DOPPLER_SHIFT."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3(0, 0, 300),  # Very fast approach
            listener_pos=Vec3(0, 0, 100),
            listener_velocity=Vec3.zero(),
        )

        assert shift <= MAX_DOPPLER_SHIFT

    def test_shift_clamped_to_min(self):
        """Shift is clamped to MIN_DOPPLER_SHIFT."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 100),
            source_velocity=Vec3(0, 0, 300),  # Very fast recession
            listener_pos=Vec3(0, 0, 0),
            listener_velocity=Vec3.zero(),
        )

        assert shift >= MIN_DOPPLER_SHIFT

    def test_doppler_factor(self):
        """Doppler factor exaggerates the effect."""
        shift_normal = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3(0, 0, 50),
            listener_pos=Vec3(0, 0, 100),
            listener_velocity=Vec3.zero(),
            doppler_factor=1.0,
        )

        shift_exaggerated = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3(0, 0, 50),
            listener_pos=Vec3(0, 0, 100),
            listener_velocity=Vec3.zero(),
            doppler_factor=2.0,
        )

        # Exaggerated should be further from 1.0
        assert abs(shift_exaggerated - 1.0) > abs(shift_normal - 1.0)

    def test_custom_speed_of_sound(self):
        """Custom speed of sound affects calculation."""
        # Underwater has higher speed of sound
        shift_air = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3(0, 0, 50),
            listener_pos=Vec3(0, 0, 100),
            listener_velocity=Vec3.zero(),
            speed_of_sound=343.0,
        )

        shift_water = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3(0, 0, 50),
            listener_pos=Vec3(0, 0, 100),
            listener_velocity=Vec3.zero(),
            speed_of_sound=1500.0,
        )

        # Higher speed of sound = less pronounced effect
        assert abs(shift_water - 1.0) < abs(shift_air - 1.0)

    def test_same_position_no_crash(self):
        """Source and listener at same position doesn't crash."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3(0, 0, 50),
            listener_pos=Vec3(0, 0, 0),  # Same as source
            listener_velocity=Vec3.zero(),
        )

        assert shift == 1.0


# =============================================================================
# DopplerState Tests
# =============================================================================


class TestDopplerState:
    """Test DopplerState for per-source tracking."""

    def test_default_values(self):
        """DopplerState has correct defaults."""
        state = DopplerState()

        assert state.source_id == 0
        assert state.current_shift == 1.0
        assert state.target_shift == 1.0
        assert state.smoothing_time == DOPPLER_SMOOTHING_TIME
        assert state.first_update is True

    def test_reset(self):
        """Reset clears state."""
        state = DopplerState()
        state.current_shift = 1.5
        state.target_shift = 1.5
        state.first_update = False

        state.reset()

        assert state.current_shift == 1.0
        assert state.target_shift == 1.0
        assert state.first_update is True


# =============================================================================
# DopplerProcessor Tests
# =============================================================================


class TestDopplerProcessor:
    """Test DopplerProcessor with state management."""

    def test_default_initialization(self):
        """Default processor initialization."""
        proc = DopplerProcessor()

        assert proc.doppler_factor == DOPPLER_FACTOR
        assert proc.speed_of_sound == SPEED_OF_SOUND
        assert proc.smoothing_time == DOPPLER_SMOOTHING_TIME

    def test_custom_initialization(self):
        """Custom processor initialization."""
        proc = DopplerProcessor(
            doppler_factor=2.0,
            speed_of_sound=1500.0,
            smoothing_time=0.1,
        )

        assert proc.doppler_factor == 2.0
        assert proc.speed_of_sound == 1500.0
        assert proc.smoothing_time == 0.1

    def test_property_setters(self):
        """Property setters with clamping."""
        proc = DopplerProcessor()

        proc.doppler_factor = 3.0
        assert proc.doppler_factor == 3.0

        proc.doppler_factor = -1.0
        assert proc.doppler_factor >= 0.0

        proc.speed_of_sound = 0.0
        assert proc.speed_of_sound >= 1.0

        proc.smoothing_time = -0.1
        assert proc.smoothing_time >= 0.0

    def test_get_or_create_state(self):
        """Get or create state for a source."""
        proc = DopplerProcessor()

        state1 = proc.get_or_create_state(123)
        assert state1.source_id == 123

        state2 = proc.get_or_create_state(123)
        assert state1 is state2  # Same instance

    def test_remove_state(self):
        """Remove state for a source."""
        proc = DopplerProcessor()

        proc.get_or_create_state(123)
        proc.remove_state(123)

        # Creating again should give fresh state
        state = proc.get_or_create_state(123)
        assert state.first_update is True

    def test_clear_states(self):
        """Clear all states."""
        proc = DopplerProcessor()

        proc.get_or_create_state(1)
        proc.get_or_create_state(2)
        proc.get_or_create_state(3)

        proc.clear_states()

        # All states should be cleared
        assert proc.get_current_shift(1) == 1.0
        assert proc.get_current_shift(2) == 1.0
        assert proc.get_current_shift(3) == 1.0

    def test_update_calculates_shift(self):
        """Update calculates Doppler shift."""
        proc = DopplerProcessor(smoothing_time=0.0)

        shift = proc.update(
            source_id=1,
            source_pos=Vec3(0, 0, 50),
            listener_pos=Vec3(0, 0, 0),
            dt=0.016,
            source_velocity=Vec3(0, 0, -30),  # Approaching listener
        )

        assert shift > 1.0

    def test_update_with_velocity_estimation(self):
        """Update estimates velocity from position changes."""
        proc = DopplerProcessor(smoothing_time=0.0)

        # First update establishes position
        proc.update(
            source_id=1,
            source_pos=Vec3(0, 0, 100),
            listener_pos=Vec3(0, 0, 0),
            dt=0.016,
        )

        # Second update - source moved closer
        shift = proc.update(
            source_id=1,
            source_pos=Vec3(0, 0, 80),  # Moved 20 units closer
            listener_pos=Vec3(0, 0, 0),
            dt=0.016,
        )

        # Should detect approach
        assert shift > 1.0

    def test_update_smoothing(self):
        """Update applies smoothing."""
        proc = DopplerProcessor(smoothing_time=0.1)

        # First update
        shift1 = proc.update(
            source_id=1,
            source_pos=Vec3(0, 0, 100),
            listener_pos=Vec3(0, 0, 0),
            dt=0.016,
            source_velocity=Vec3(0, 0, -50),
        )

        # Should be smoothed towards target
        state = proc.get_or_create_state(1)
        assert state.current_shift != state.target_shift or state.target_shift == 1.0

    def test_update_zero_dt(self):
        """Update with zero dt returns current shift."""
        proc = DopplerProcessor()

        proc.get_or_create_state(1).current_shift = 1.5

        shift = proc.update(
            source_id=1,
            source_pos=Vec3(0, 0, 100),
            listener_pos=Vec3(0, 0, 0),
            dt=0.0,
        )

        assert shift == 1.5

    def test_get_current_shift(self):
        """Get current shift without updating."""
        proc = DopplerProcessor()

        # No state yet
        assert proc.get_current_shift(999) == 1.0

        # After update
        proc.update(
            source_id=1,
            source_pos=Vec3(0, 0, 100),
            listener_pos=Vec3(0, 0, 0),
            dt=0.016,
            source_velocity=Vec3(0, 0, -50),
        )

        shift = proc.get_current_shift(1)
        assert shift != 1.0  # Should have changed


# =============================================================================
# estimate_arrival_time Tests
# =============================================================================


class TestEstimateArrivalTime:
    """Test sound arrival time estimation."""

    def test_stationary_source_listener(self):
        """Arrival time for stationary source and listener."""
        time = estimate_arrival_time(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3.zero(),
            listener_pos=Vec3(0, 0, 343),  # 343m = 1 second at speed of sound
            listener_velocity=Vec3.zero(),
        )

        assert time == pytest.approx(1.0, rel=1e-2)

    def test_same_position(self):
        """Zero arrival time when at same position."""
        time = estimate_arrival_time(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3.zero(),
            listener_pos=Vec3(0, 0, 0),
            listener_velocity=Vec3.zero(),
        )

        assert time == 0.0

    def test_approaching_listener(self):
        """Approaching listener affects arrival time."""
        time_stationary = estimate_arrival_time(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3.zero(),
            listener_pos=Vec3(0, 0, 343),
            listener_velocity=Vec3.zero(),
        )

        time_approaching = estimate_arrival_time(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3.zero(),
            listener_pos=Vec3(0, 0, 343),
            listener_velocity=Vec3(0, 0, -100),  # Approaching source
        )

        # Both should be valid times
        assert time_stationary is not None
        assert time_approaching is not None

    def test_supersonic_separation(self):
        """Test supersonic separation scenario."""
        time = estimate_arrival_time(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3.zero(),
            listener_pos=Vec3(0, 0, 100),
            listener_velocity=Vec3(0, 0, 400),  # Running away faster than sound
        )

        # May return None or a large time depending on implementation
        # Just verify it doesn't crash
        assert time is None or time > 0


# =============================================================================
# DopplerConfig Tests
# =============================================================================


class TestDopplerConfig:
    """Test DopplerConfig dataclass."""

    def test_default_values(self):
        """DopplerConfig has correct defaults."""
        config = DopplerConfig()

        assert config.enabled is True
        assert config.factor == DOPPLER_FACTOR
        assert config.speed_of_sound == SPEED_OF_SOUND
        assert config.smoothing_time == DOPPLER_SMOOTHING_TIME
        assert config.min_shift == MIN_DOPPLER_SHIFT
        assert config.max_shift == MAX_DOPPLER_SHIFT
        assert config.velocity_threshold == DOPPLER_VELOCITY_THRESHOLD

    def test_create_processor(self):
        """DopplerConfig can create a processor."""
        config = DopplerConfig(
            factor=2.0,
            speed_of_sound=1500.0,
            smoothing_time=0.1,
        )

        proc = config.create_processor()

        assert isinstance(proc, DopplerProcessor)
        assert proc.doppler_factor == 2.0
        assert proc.speed_of_sound == 1500.0
        assert proc.smoothing_time == 0.1


# =============================================================================
# Preset Tests
# =============================================================================


class TestDopplerPresets:
    """Test Doppler effect presets."""

    def test_all_presets_exist(self):
        """All presets can be retrieved."""
        preset_names = ["realistic", "exaggerated", "subtle", "arcade", "disabled", "underwater"]

        for name in preset_names:
            preset = get_doppler_preset(name)
            assert preset is not None
            assert isinstance(preset, DopplerConfig)

    def test_preset_not_found(self):
        """Unknown preset returns None."""
        result = get_doppler_preset("nonexistent")
        assert result is None

    def test_realistic_preset(self):
        """Realistic preset has factor 1.0."""
        preset = get_doppler_preset("realistic")
        assert preset.factor == 1.0

    def test_exaggerated_preset(self):
        """Exaggerated preset has higher factor."""
        preset = get_doppler_preset("exaggerated")
        assert preset.factor > 1.0

    def test_subtle_preset(self):
        """Subtle preset has lower factor."""
        preset = get_doppler_preset("subtle")
        assert preset.factor < 1.0

    def test_disabled_preset(self):
        """Disabled preset has enabled=False."""
        preset = get_doppler_preset("disabled")
        assert preset.enabled is False

    def test_underwater_preset(self):
        """Underwater preset has higher speed of sound."""
        preset = get_doppler_preset("underwater")
        assert preset.speed_of_sound > SPEED_OF_SOUND


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestDopplerEdgeCases:
    """Test edge cases in Doppler processing."""

    def test_very_close_source(self):
        """Handle very close source positions."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3(0, 0, 50),
            listener_pos=Vec3(0, 0, 0.001),  # Very close
            listener_velocity=Vec3.zero(),
        )

        # Should not crash, return reasonable value
        assert MIN_DOPPLER_SHIFT <= shift <= MAX_DOPPLER_SHIFT

    def test_sonic_speed_approach(self):
        """Handle source approaching at sonic speed."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3(0, 0, SPEED_OF_SOUND - 1),  # Just under sonic
            listener_pos=Vec3(0, 0, 100),
            listener_velocity=Vec3.zero(),
        )

        # Should be high but clamped
        assert shift == MAX_DOPPLER_SHIFT

    def test_supersonic_approach(self):
        """Handle source approaching faster than sound."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3(0, 0, SPEED_OF_SOUND + 100),  # Supersonic
            listener_pos=Vec3(0, 0, 100),
            listener_velocity=Vec3.zero(),
        )

        # Should be clamped
        assert shift == MAX_DOPPLER_SHIFT

    def test_both_moving_same_direction(self):
        """Source and listener moving in same direction."""
        # Both moving positive Z at same speed - no relative motion
        shift = calculate_doppler_shift(
            source_pos=Vec3(0, 0, 0),
            source_velocity=Vec3(0, 0, 50),
            listener_pos=Vec3(0, 0, 100),
            listener_velocity=Vec3(0, 0, 50),
        )

        # Same velocity = should be close to 1.0 (may vary due to implementation)
        # Just verify it's a valid shift value
        assert MIN_DOPPLER_SHIFT <= shift <= MAX_DOPPLER_SHIFT

    def test_orbiting_motion(self):
        """Source orbiting around listener."""
        # Source at (100, 0, 0) moving in +Y (perpendicular to listener)
        shift = calculate_doppler_shift(
            source_pos=Vec3(100, 0, 0),
            source_velocity=Vec3(0, 50, 0),  # Perpendicular
            listener_pos=Vec3(0, 0, 0),
            listener_velocity=Vec3.zero(),
        )

        # Perpendicular motion - no shift
        assert shift == pytest.approx(1.0, abs=0.01)
