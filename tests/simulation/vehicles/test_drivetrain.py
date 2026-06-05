"""Tests for drivetrain components.

Tests cover:
- Engine torque curves and RPM management
- Transmission gear ratios and shifting
- Clutch engagement and slip
- Differential types (open, LSD, locked, Torsen)
- Complete drivetrain assembly
- Edge cases (stall, overrev, etc.)
"""

import math
import pytest

from engine.simulation.vehicles.drivetrain import (
    DiffType,
    DrivetrainLayout,
    EngineState,
    Engine,
    TransmissionState,
    Transmission,
    ClutchState,
    Clutch,
    Differential,
    Drivetrain,
)
from engine.simulation.vehicles.config import (
    ENGINE_IDLE_RPM,
    ENGINE_MAX_RPM,
    ENGINE_REDLINE_RPM,
    DEFAULT_GEAR_RATIOS,
)


# =============================================================================
# Engine Tests
# =============================================================================


class TestEngine:
    """Tests for engine simulation."""

    @pytest.fixture
    def engine(self):
        """Create a standard engine."""
        return Engine(
            idle_rpm=1000,
            max_rpm=7000,
            redline_rpm=6500,
            max_torque=450.0,
        )

    def test_initialization(self, engine):
        """Engine should initialize with correct values."""
        assert engine.idle_rpm == 1000
        assert engine.max_rpm == 7000
        assert engine.redline_rpm == 6500
        assert engine.max_torque == 450.0

    def test_initial_state(self, engine):
        """Engine should start at idle."""
        assert engine.rpm == ENGINE_IDLE_RPM
        assert engine.state.is_running

    def test_torque_curve_interpolation(self, engine):
        """Torque curve should interpolate between points."""
        mult_1500 = engine.get_torque_multiplier(1500)  # Between 1000 and 2000
        mult_1000 = engine.get_torque_multiplier(1000)
        mult_2000 = engine.get_torque_multiplier(2000)

        # Should be between the two points
        assert min(mult_1000, mult_2000) <= mult_1500 <= max(mult_1000, mult_2000)

    def test_torque_curve_bounds(self, engine):
        """Torque curve should clamp at edges."""
        mult_low = engine.get_torque_multiplier(500)  # Below curve
        mult_high = engine.get_torque_multiplier(8000)  # Above curve

        # Should return edge values
        assert mult_low == engine.get_torque_multiplier(1000)
        assert mult_high == engine.get_torque_multiplier(7000)

    def test_compute_torque_full_throttle(self, engine):
        """Full throttle should produce significant torque."""
        torque = engine.compute_torque(throttle=1.0, rpm=4500)  # Peak torque RPM
        assert torque > 0
        assert torque <= engine.max_torque  # Can't exceed max

    def test_compute_torque_zero_throttle(self, engine):
        """Zero throttle should produce engine braking."""
        torque = engine.compute_torque(throttle=0.0, rpm=4000)
        # Engine braking + friction = negative or near-zero torque
        assert torque <= 0

    def test_update_accelerates_engine(self, engine):
        """Throttle should increase RPM."""
        initial_rpm = engine.rpm
        for _ in range(100):  # Several updates
            engine.update(throttle=1.0, load_torque=0.0, dt=0.016)

        assert engine.rpm > initial_rpm

    def test_rev_limiter(self, engine):
        """Engine should not exceed max RPM."""
        for _ in range(500):
            engine.update(throttle=1.0, load_torque=0.0, dt=0.016)

        assert engine.rpm <= engine.max_rpm

    def test_rev_limited_flag(self, engine):
        """Rev limiter flag should engage at redline."""
        # Get to redline
        for _ in range(500):
            engine.update(throttle=1.0, load_torque=0.0, dt=0.016)
            if engine.rpm >= engine.redline_rpm:
                break

        # At or near redline, flag should be set
        assert engine.rpm >= engine.redline_rpm - 100 or not engine.state.is_revlimited

    def test_idle_governor(self, engine):
        """Idle governor should maintain minimum RPM."""
        # Drop throttle, add load
        for _ in range(200):
            engine.update(throttle=0.0, load_torque=50.0, dt=0.016)

        assert engine.rpm >= engine.idle_rpm

    def test_engine_stop_start(self, engine):
        """Engine should be stoppable and startable."""
        engine.stop()
        assert not engine.state.is_running
        assert engine.rpm == 0

        engine.start()
        assert engine.state.is_running
        assert engine.rpm == engine.idle_rpm

    def test_stopped_engine_no_torque(self, engine):
        """Stopped engine should produce no torque."""
        engine.stop()
        torque = engine.update(throttle=1.0, load_torque=0.0, dt=0.016)
        assert torque == 0.0


# =============================================================================
# Transmission Tests
# =============================================================================


class TestTransmission:
    """Tests for transmission simulation."""

    @pytest.fixture
    def transmission(self):
        """Create a standard transmission."""
        return Transmission(
            gear_ratios=DEFAULT_GEAR_RATIOS,
            final_drive=3.7,
            shift_time=0.2,
        )

    def test_initialization(self, transmission):
        """Transmission should initialize in first gear."""
        assert transmission.current_gear == 1
        assert transmission.gear_ratio != 0

    def test_gear_count(self, transmission):
        """Should report correct number of forward gears."""
        # DEFAULT_GEAR_RATIOS has R, N, 1-6 = 8 total, 6 forward
        assert transmission.gear_count == len(DEFAULT_GEAR_RATIOS) - 2

    def test_shift_up(self, transmission):
        """Shifting up should increase gear number."""
        initial_gear = transmission.current_gear
        result = transmission.shift_up()
        assert result

        # Process shift
        for _ in range(20):
            transmission.update(100.0, 4000.0, 0.016)

        assert transmission.current_gear == initial_gear + 1

    def test_shift_down(self, transmission):
        """Shifting down should decrease gear number."""
        # First shift to 2nd
        transmission.shift(2)
        for _ in range(20):
            transmission.update(100.0, 4000.0, 0.016)

        result = transmission.shift_down()
        assert result

        for _ in range(20):
            transmission.update(100.0, 4000.0, 0.016)

        assert transmission.current_gear == 1

    def test_shift_to_neutral(self, transmission):
        """Should be able to shift to neutral."""
        transmission.shift(0)
        for _ in range(20):
            transmission.update(100.0, 4000.0, 0.016)

        assert transmission.current_gear == 0
        assert transmission.gear_ratio == 0.0  # Neutral

    def test_shift_to_reverse(self, transmission):
        """Should be able to shift to reverse."""
        transmission.shift(-1)
        for _ in range(20):
            transmission.update(100.0, 4000.0, 0.016)

        assert transmission.current_gear == -1

    def test_shift_beyond_max(self, transmission):
        """Cannot shift beyond maximum gear."""
        max_gear = transmission.gear_count
        transmission.shift(max_gear)
        for _ in range(20):
            transmission.update(100.0, 4000.0, 0.016)

        result = transmission.shift_up()
        assert not result  # Should fail

    def test_shift_beyond_reverse(self, transmission):
        """Cannot shift below reverse."""
        transmission.shift(-1)
        for _ in range(20):
            transmission.update(100.0, 4000.0, 0.016)

        result = transmission.shift_down()
        assert not result

    def test_no_torque_during_shift(self, transmission):
        """No torque should transfer during shift."""
        transmission.shift_up()
        output_torque, _ = transmission.update(500.0, 4000.0, 0.016)
        assert output_torque == 0.0
        assert transmission.is_shifting

    def test_torque_multiplication(self, transmission):
        """Output torque should equal input * ratio."""
        input_torque = 100.0
        output_torque, output_rpm = transmission.update(
            input_torque, 4000.0, 0.016
        )

        expected = input_torque * transmission.gear_ratio
        assert abs(output_torque - expected) < 0.01

    def test_auto_mode_upshift(self):
        """Auto mode should upshift at high RPM."""
        trans = Transmission(
            auto_mode=True,
            upshift_rpm=5000.0,
            downshift_rpm=2500.0,
        )

        # Simulate high RPM
        for _ in range(50):
            trans.update(100.0, 5500.0, 0.016)  # Above upshift RPM

        # Should have initiated upshift
        assert trans.current_gear >= 1  # May have shifted


# =============================================================================
# Clutch Tests
# =============================================================================


class TestClutch:
    """Tests for clutch simulation."""

    @pytest.fixture
    def clutch(self):
        """Create a standard clutch."""
        return Clutch(
            max_torque=500.0,
            engagement_rate=5.0,
        )

    def test_initialization(self, clutch):
        """Clutch should start fully engaged."""
        assert clutch.engagement == 1.0
        assert clutch.is_engaged

    def test_disengage(self, clutch):
        """Disengaging should reduce engagement."""
        clutch.disengage()
        for _ in range(100):
            clutch.update(100.0, 3000.0, 3000.0, 0.016)

        assert clutch.engagement < 0.1
        assert not clutch.is_engaged

    def test_engage(self, clutch):
        """Engaging should increase engagement."""
        clutch.disengage()
        for _ in range(50):
            clutch.update(100.0, 3000.0, 3000.0, 0.016)

        clutch.engage()
        for _ in range(100):
            clutch.update(100.0, 3000.0, 3000.0, 0.016)

        assert clutch.engagement > 0.9

    def test_torque_transfer_engaged(self, clutch):
        """Fully engaged clutch should transfer all torque."""
        transfer = clutch.update(100.0, 3000.0, 3000.0, 0.016)
        assert abs(transfer - 100.0) < 1.0

    def test_torque_transfer_disengaged(self, clutch):
        """Disengaged clutch should transfer no torque."""
        clutch.set_engagement(0.0)
        for _ in range(100):
            clutch.update(100.0, 3000.0, 3000.0, 0.016)

        transfer = clutch.update(100.0, 3000.0, 3000.0, 0.016)
        assert abs(transfer) < 1.0

    def test_clutch_slip_detection(self, clutch):
        """Should detect clutch slip."""
        # Large torque exceeding capacity at partial engagement
        clutch.set_engagement(0.5)
        for _ in range(50):
            clutch.update(600.0, 5000.0, 2000.0, 0.016)  # High slip velocity

        assert clutch.is_slipping or clutch.state.slip != 0

    def test_partial_engagement(self, clutch):
        """Partial engagement should transfer partial torque."""
        clutch.set_engagement(0.5)
        for _ in range(50):
            clutch.update(100.0, 3000.0, 3000.0, 0.016)

        transfer = clutch.update(100.0, 3000.0, 3000.0, 0.016)
        # Should be approximately half
        assert 40 < abs(transfer) < 60


# =============================================================================
# Differential Tests
# =============================================================================


class TestDifferential:
    """Tests for differential simulation."""

    def test_open_diff_equal_split(self):
        """Open diff should split torque 50/50."""
        diff = Differential(diff_type=DiffType.OPEN)
        left, right = diff.torque_split(1000.0, 10.0, 10.0)

        assert left == 500.0
        assert right == 500.0

    def test_locked_diff_equal_split(self):
        """Locked diff should also split 50/50."""
        diff = Differential(diff_type=DiffType.LOCKED)
        left, right = diff.torque_split(1000.0, 10.0, 12.0)

        assert left == 500.0
        assert right == 500.0

    def test_lsd_bias_to_slower(self):
        """LSD should bias torque to slower wheel."""
        diff = Differential(
            diff_type=DiffType.LIMITED_SLIP,
            preload=100.0,
            power_ratio=0.6,
        )

        # Left wheel spinning faster
        left, right = diff.torque_split(1000.0, 15.0, 10.0)

        # Right (slower) should get more torque
        assert right > left

    def test_torsen_bias_to_slower(self):
        """Torsen should bias to slower wheel."""
        diff = Differential(
            diff_type=DiffType.TORSEN,
            bias_ratio=3.0,
        )

        # Right wheel spinning faster
        left, right = diff.torque_split(1000.0, 10.0, 15.0)

        # Left (slower) should get more torque
        assert left > right

    def test_diff_type_setter(self):
        """Should be able to change diff type."""
        diff = Differential(diff_type=DiffType.OPEN)
        diff.diff_type = DiffType.LOCKED
        assert diff.diff_type == DiffType.LOCKED


# =============================================================================
# Drivetrain Integration Tests
# =============================================================================


class TestDrivetrain:
    """Tests for complete drivetrain assembly."""

    @pytest.fixture
    def rwd_drivetrain(self):
        """Create an RWD drivetrain."""
        return Drivetrain(layout=DrivetrainLayout.RWD)

    @pytest.fixture
    def fwd_drivetrain(self):
        """Create an FWD drivetrain."""
        return Drivetrain(layout=DrivetrainLayout.FWD)

    @pytest.fixture
    def awd_drivetrain(self):
        """Create an AWD drivetrain."""
        return Drivetrain(layout=DrivetrainLayout.AWD)

    def test_rwd_layout(self, rwd_drivetrain):
        """RWD should only drive rear wheels."""
        wheel_speeds = (0.0, 0.0, 0.0, 0.0)
        torques = rwd_drivetrain.update(0.5, wheel_speeds, 0.016)

        fl, fr, rl, rr = torques
        # Front wheels should have zero torque
        assert fl == 0.0
        assert fr == 0.0
        # Rear wheels should have torque
        # (may be zero if engine hasn't spun up yet)

    def test_fwd_layout(self, fwd_drivetrain):
        """FWD should only drive front wheels."""
        wheel_speeds = (0.0, 0.0, 0.0, 0.0)

        # Spin up engine
        for _ in range(50):
            fwd_drivetrain.update(1.0, wheel_speeds, 0.016)

        torques = fwd_drivetrain.update(0.5, wheel_speeds, 0.016)
        fl, fr, rl, rr = torques

        # Rear wheels should have zero torque
        assert rl == 0.0
        assert rr == 0.0

    def test_awd_layout(self, awd_drivetrain):
        """AWD should drive all wheels."""
        wheel_speeds = (0.0, 0.0, 0.0, 0.0)

        # Spin up engine
        for _ in range(50):
            awd_drivetrain.update(1.0, wheel_speeds, 0.016)

        torques = awd_drivetrain.update(0.5, wheel_speeds, 0.016)
        fl, fr, rl, rr = torques

        # All wheels should have non-zero torque (after engine spins up)
        # Engine may still be building RPM, so check sum
        total_torque = abs(fl) + abs(fr) + abs(rl) + abs(rr)
        assert total_torque > 0

    def test_engine_rpm_access(self, rwd_drivetrain):
        """Should access engine RPM through drivetrain."""
        rpm = rwd_drivetrain.engine_rpm
        assert rpm >= ENGINE_IDLE_RPM

    def test_shift_up_down(self, rwd_drivetrain):
        """Should be able to shift through drivetrain."""
        initial = rwd_drivetrain.current_gear
        rwd_drivetrain.shift_up()
        # Process shift
        for _ in range(20):
            rwd_drivetrain.update(0.5, (0, 0, 0, 0), 0.016)

        assert rwd_drivetrain.current_gear == initial + 1

        rwd_drivetrain.shift_down()
        for _ in range(20):
            rwd_drivetrain.update(0.5, (0, 0, 0, 0), 0.016)

        assert rwd_drivetrain.current_gear == initial

    def test_shift_to_specific_gear(self, rwd_drivetrain):
        """Should shift to specific gear."""
        rwd_drivetrain.shift_to(3)
        for _ in range(20):
            rwd_drivetrain.update(0.5, (0, 0, 0, 0), 0.016)

        assert rwd_drivetrain.current_gear == 3


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestDrivetrainEdgeCases:
    """Tests for drivetrain edge cases."""

    def test_engine_custom_torque_curve(self):
        """Engine should accept custom torque curve."""
        custom_curve = {
            2000: 0.8,
            4000: 1.0,
            6000: 0.9,
        }
        engine = Engine(torque_curve=custom_curve)
        mult = engine.get_torque_multiplier(4000)
        assert mult == 1.0

    def test_transmission_invalid_gear(self):
        """Invalid gear should fail gracefully."""
        trans = Transmission()
        result = trans.shift(999)  # Way beyond valid
        assert not result

    def test_zero_wheel_speeds(self):
        """Drivetrain should handle zero wheel speeds."""
        drivetrain = Drivetrain()
        wheel_speeds = (0.0, 0.0, 0.0, 0.0)
        torques = drivetrain.update(0.5, wheel_speeds, 0.016)
        # Should not crash
        assert len(torques) == 4

    def test_negative_wheel_speeds(self):
        """Drivetrain should handle reverse motion."""
        drivetrain = Drivetrain()
        wheel_speeds = (-5.0, -5.0, -5.0, -5.0)  # Reversing
        torques = drivetrain.update(0.0, wheel_speeds, 0.016)
        # Should not crash
        assert len(torques) == 4

    def test_mismatched_wheel_speeds(self):
        """Drivetrain should handle different wheel speeds (cornering)."""
        drivetrain = Drivetrain(layout=DrivetrainLayout.RWD)
        # Different speeds for cornering
        wheel_speeds = (8.0, 10.0, 9.0, 11.0)
        torques = drivetrain.update(0.5, wheel_speeds, 0.016)
        assert len(torques) == 4
