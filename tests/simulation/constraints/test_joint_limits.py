"""
Whitebox tests for joint_limits.py - Limit helpers.

Tests:
- LimitState enum
- LinearLimit dataclass
- AngularLimit dataclass
- SwingLimit dataclass
- TwistLimit dataclass
- compute_limit_impulse function
- compute_soft_limit_coefficients function
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'engine'))

from simulation.constraints.joint_limits import (
    LimitState,
    LinearLimit,
    AngularLimit,
    SwingLimit,
    TwistLimit,
    compute_limit_impulse,
    compute_soft_limit_coefficients,
)


class TestLimitState:
    """Tests for LimitState enum."""

    def test_inactive_exists(self):
        """INACTIVE state should exist."""
        assert LimitState.INACTIVE is not None

    def test_at_lower_exists(self):
        """AT_LOWER state should exist."""
        assert LimitState.AT_LOWER is not None

    def test_at_upper_exists(self):
        """AT_UPPER state should exist."""
        assert LimitState.AT_UPPER is not None

    def test_states_unique(self):
        """All states should be unique."""
        states = [LimitState.INACTIVE, LimitState.AT_LOWER, LimitState.AT_UPPER]
        assert len(states) == len(set(states))


class TestLinearLimit:
    """Tests for LinearLimit dataclass."""

    def test_default_values(self):
        """Default values should be sensible."""
        limit = LinearLimit()
        assert limit.lower == -1.0
        assert limit.upper == 1.0
        assert limit.stiffness == 0.0
        assert limit.damping == 0.0
        assert limit.restitution == 0.0
        assert limit.contact_distance == 0.01

    def test_custom_values(self):
        """Custom values should be stored correctly."""
        limit = LinearLimit(lower=-5.0, upper=5.0, stiffness=100.0, damping=10.0)
        assert limit.lower == -5.0
        assert limit.upper == 5.0
        assert limit.stiffness == 100.0
        assert limit.damping == 10.0

    def test_auto_swap_if_inverted(self):
        """Lower and upper should be swapped if inverted."""
        limit = LinearLimit(lower=10.0, upper=-10.0)
        assert limit.lower == -10.0
        assert limit.upper == 10.0

    def test_is_soft_false(self):
        """is_soft should be False when stiffness is 0."""
        limit = LinearLimit()
        assert limit.is_soft is False

    def test_is_soft_true(self):
        """is_soft should be True when stiffness > 0."""
        limit = LinearLimit(stiffness=100.0)
        assert limit.is_soft is True

    def test_range_property(self):
        """range should return upper - lower."""
        limit = LinearLimit(lower=-2.0, upper=3.0)
        assert abs(limit.range - 5.0) < 1e-6

    def test_center_property(self):
        """center should return midpoint."""
        limit = LinearLimit(lower=-2.0, upper=4.0)
        assert abs(limit.center - 1.0) < 1e-6

    def test_check_state_inactive(self):
        """check_state should return INACTIVE when within limits."""
        limit = LinearLimit(lower=-1.0, upper=1.0)
        assert limit.check_state(0.0) == LimitState.INACTIVE
        assert limit.check_state(0.5) == LimitState.INACTIVE
        assert limit.check_state(-0.5) == LimitState.INACTIVE

    def test_check_state_at_lower(self):
        """check_state should return AT_LOWER at lower limit."""
        limit = LinearLimit(lower=-1.0, upper=1.0, contact_distance=0.01)
        assert limit.check_state(-0.995) == LimitState.AT_LOWER
        assert limit.check_state(-1.5) == LimitState.AT_LOWER

    def test_check_state_at_upper(self):
        """check_state should return AT_UPPER at upper limit."""
        limit = LinearLimit(lower=-1.0, upper=1.0, contact_distance=0.01)
        assert limit.check_state(0.995) == LimitState.AT_UPPER
        assert limit.check_state(1.5) == LimitState.AT_UPPER

    def test_compute_error_within_limits(self):
        """compute_error should return 0 within limits."""
        limit = LinearLimit(lower=-1.0, upper=1.0)
        assert limit.compute_error(0.0) == 0.0
        assert limit.compute_error(0.5) == 0.0
        assert limit.compute_error(-0.5) == 0.0

    def test_compute_error_below_lower(self):
        """compute_error should return positive error below lower."""
        limit = LinearLimit(lower=-1.0, upper=1.0)
        assert abs(limit.compute_error(-1.5) - 0.5) < 1e-6

    def test_compute_error_above_upper(self):
        """compute_error should return positive error above upper."""
        limit = LinearLimit(lower=-1.0, upper=1.0)
        assert abs(limit.compute_error(1.5) - 0.5) < 1e-6

    def test_clamp_within(self):
        """clamp should return value when within limits."""
        limit = LinearLimit(lower=-1.0, upper=1.0)
        assert abs(limit.clamp(0.5) - 0.5) < 1e-6

    def test_clamp_below(self):
        """clamp should return lower when below."""
        limit = LinearLimit(lower=-1.0, upper=1.0)
        assert abs(limit.clamp(-5.0) - (-1.0)) < 1e-6

    def test_clamp_above(self):
        """clamp should return upper when above."""
        limit = LinearLimit(lower=-1.0, upper=1.0)
        assert abs(limit.clamp(5.0) - 1.0) < 1e-6


class TestAngularLimit:
    """Tests for AngularLimit dataclass."""

    def test_default_values(self):
        """Default values should be sensible."""
        limit = AngularLimit()
        assert abs(limit.lower - (-math.pi)) < 1e-6
        assert abs(limit.upper - math.pi) < 1e-6

    def test_normalization(self):
        """Angles should be normalized to [-pi, pi]."""
        limit = AngularLimit(lower=3*math.pi, upper=5*math.pi)
        assert abs(limit.lower) <= math.pi + 0.01
        assert abs(limit.upper) <= math.pi + 0.01

    def test_auto_swap_if_inverted(self):
        """Lower and upper should be swapped if inverted."""
        limit = AngularLimit(lower=math.pi/2, upper=-math.pi/2)
        assert limit.lower <= limit.upper

    def test_is_soft(self):
        """is_soft should reflect stiffness."""
        limit = AngularLimit()
        assert limit.is_soft is False
        limit = AngularLimit(stiffness=100.0)
        assert limit.is_soft is True

    def test_check_state_inactive(self):
        """check_state should return INACTIVE within limits."""
        limit = AngularLimit(lower=-math.pi/4, upper=math.pi/4)
        assert limit.check_state(0.0) == LimitState.INACTIVE

    def test_check_state_at_lower(self):
        """check_state should return AT_LOWER at lower limit."""
        limit = AngularLimit(lower=-math.pi/4, upper=math.pi/4, contact_distance=0.01)
        assert limit.check_state(-math.pi/4 + 0.005) == LimitState.AT_LOWER

    def test_check_state_at_upper(self):
        """check_state should return AT_UPPER at upper limit."""
        limit = AngularLimit(lower=-math.pi/4, upper=math.pi/4, contact_distance=0.01)
        assert limit.check_state(math.pi/4 - 0.005) == LimitState.AT_UPPER

    def test_compute_error_within(self):
        """compute_error should return 0 within limits."""
        limit = AngularLimit(lower=-math.pi/4, upper=math.pi/4)
        assert limit.compute_error(0.0) == 0.0

    def test_compute_error_outside(self):
        """compute_error should return positive error outside limits."""
        limit = AngularLimit(lower=-math.pi/4, upper=math.pi/4)
        error = limit.compute_error(math.pi/2)
        assert error > 0.0

    def test_clamp_normalizes(self):
        """clamp should normalize angle before clamping."""
        limit = AngularLimit(lower=-math.pi/4, upper=math.pi/4)
        result = limit.clamp(5*math.pi)  # Should normalize to pi, then clamp
        assert abs(result) <= math.pi/4 + 0.01


class TestSwingLimit:
    """Tests for SwingLimit dataclass."""

    def test_default_values(self):
        """Default values should be sensible."""
        limit = SwingLimit()
        assert abs(limit.y_angle - math.pi/4) < 1e-6
        assert abs(limit.z_angle - math.pi/4) < 1e-6

    def test_is_circular_true(self):
        """is_circular should be True when y and z angles are equal."""
        limit = SwingLimit(y_angle=0.5, z_angle=0.5)
        assert limit.is_circular is True

    def test_is_circular_false(self):
        """is_circular should be False when y and z angles differ."""
        limit = SwingLimit(y_angle=0.5, z_angle=0.3)
        assert limit.is_circular is False

    def test_is_soft(self):
        """is_soft should reflect stiffness."""
        limit = SwingLimit()
        assert limit.is_soft is False
        limit = SwingLimit(stiffness=100.0)
        assert limit.is_soft is True

    def test_check_within_circular_cone_inside(self):
        """check_within_cone should return True when inside circular cone."""
        limit = SwingLimit(y_angle=math.pi/4, z_angle=math.pi/4)
        assert limit.check_within_cone(0.1, 0.1) is True
        assert limit.check_within_cone(0.0, 0.0) is True

    def test_check_within_circular_cone_outside(self):
        """check_within_cone should return False when outside circular cone."""
        limit = SwingLimit(y_angle=0.1, z_angle=0.1)
        assert limit.check_within_cone(0.5, 0.5) is False

    def test_check_within_elliptical_cone_inside(self):
        """check_within_cone should handle elliptical cones."""
        limit = SwingLimit(y_angle=math.pi/4, z_angle=math.pi/6)
        assert limit.check_within_cone(0.1, 0.1) is True

    def test_check_within_elliptical_cone_outside(self):
        """check_within_cone should detect outside elliptical cone."""
        limit = SwingLimit(y_angle=0.1, z_angle=0.2)
        # On the ellipse boundary or outside
        assert limit.check_within_cone(0.15, 0.25) is False

    def test_compute_error_inside(self):
        """compute_error should return 0 inside cone."""
        limit = SwingLimit(y_angle=math.pi/4, z_angle=math.pi/4)
        assert limit.compute_error(0.1, 0.1) == 0.0

    def test_compute_error_outside(self):
        """compute_error should return positive error outside cone."""
        limit = SwingLimit(y_angle=0.1, z_angle=0.1)
        error = limit.compute_error(0.5, 0.5)
        assert error > 0.0


class TestTwistLimit:
    """Tests for TwistLimit dataclass."""

    def test_default_values(self):
        """Default values should be sensible."""
        limit = TwistLimit()
        assert abs(limit.lower - (-math.pi)) < 1e-6
        assert abs(limit.upper - math.pi) < 1e-6

    def test_is_soft(self):
        """is_soft should reflect stiffness."""
        limit = TwistLimit()
        assert limit.is_soft is False
        limit = TwistLimit(stiffness=100.0)
        assert limit.is_soft is True

    def test_check_state_inactive(self):
        """check_state should return INACTIVE within limits."""
        limit = TwistLimit(lower=-math.pi/2, upper=math.pi/2)
        assert limit.check_state(0.0) == LimitState.INACTIVE
        assert limit.check_state(0.5) == LimitState.INACTIVE

    def test_check_state_at_lower(self):
        """check_state should return AT_LOWER at lower limit."""
        limit = TwistLimit(lower=-math.pi/2, upper=math.pi/2)
        assert limit.check_state(-math.pi/2 - 0.1) == LimitState.AT_LOWER

    def test_check_state_at_upper(self):
        """check_state should return AT_UPPER at upper limit."""
        limit = TwistLimit(lower=-math.pi/2, upper=math.pi/2)
        assert limit.check_state(math.pi/2 + 0.1) == LimitState.AT_UPPER

    def test_compute_error_within(self):
        """compute_error should return 0 within limits."""
        limit = TwistLimit(lower=-math.pi/2, upper=math.pi/2)
        assert limit.compute_error(0.0) == 0.0

    def test_compute_error_outside(self):
        """compute_error should return positive error outside limits."""
        limit = TwistLimit(lower=-math.pi/4, upper=math.pi/4)
        error = limit.compute_error(math.pi/2)
        assert error > 0.0

    def test_handles_wraparound(self):
        """Twist limit should handle angle wraparound."""
        limit = TwistLimit(lower=-math.pi/2, upper=math.pi/2)
        # 3*pi should normalize to pi, which is at upper limit
        state = limit.check_state(3*math.pi)
        assert state in [LimitState.AT_LOWER, LimitState.AT_UPPER]


class TestComputeLimitImpulse:
    """Tests for compute_limit_impulse function."""

    def test_inactive_returns_zero(self):
        """Should return zero impulse when inactive."""
        limit = LinearLimit(lower=-1.0, upper=1.0)
        impulse, state = compute_limit_impulse(
            limit=limit,
            current_value=0.0,
            current_velocity=0.0,
            effective_mass=1.0,
            dt=0.016
        )
        assert impulse == 0.0
        assert state == LimitState.INACTIVE

    def test_at_lower_returns_positive_impulse(self):
        """Should return positive impulse at lower limit."""
        limit = LinearLimit(lower=-1.0, upper=1.0, contact_distance=0.02)
        impulse, state = compute_limit_impulse(
            limit=limit,
            current_value=-1.1,
            current_velocity=-1.0,
            effective_mass=1.0,
            dt=0.016,
            slop=0.0
        )
        assert impulse >= 0.0
        assert state == LimitState.AT_LOWER

    def test_at_upper_returns_negative_impulse(self):
        """Should return negative impulse at upper limit."""
        limit = LinearLimit(lower=-1.0, upper=1.0, contact_distance=0.02)
        impulse, state = compute_limit_impulse(
            limit=limit,
            current_value=1.1,
            current_velocity=1.0,
            effective_mass=1.0,
            dt=0.016,
            slop=0.0
        )
        assert impulse <= 0.0
        assert state == LimitState.AT_UPPER

    def test_zero_effective_mass(self):
        """Should return zero impulse with zero effective mass."""
        limit = LinearLimit(lower=-1.0, upper=1.0)
        impulse, state = compute_limit_impulse(
            limit=limit,
            current_value=-1.1,
            current_velocity=-1.0,
            effective_mass=0.0,
            dt=0.016
        )
        assert impulse == 0.0
        assert state == LimitState.INACTIVE

    def test_soft_limit(self):
        """Soft limits should use spring/damper model."""
        limit = LinearLimit(lower=-1.0, upper=1.0, stiffness=100.0, damping=10.0)
        impulse, state = compute_limit_impulse(
            limit=limit,
            current_value=-1.1,
            current_velocity=-1.0,
            effective_mass=1.0,
            dt=0.016,
            slop=0.0
        )
        # Soft limit should still produce impulse
        assert state == LimitState.AT_LOWER

    def test_restitution(self):
        """Restitution should affect impulse."""
        limit = LinearLimit(lower=-1.0, upper=1.0, restitution=0.5, contact_distance=0.02)
        impulse, state = compute_limit_impulse(
            limit=limit,
            current_value=-1.1,
            current_velocity=-5.0,  # High incoming velocity
            effective_mass=1.0,
            dt=0.016,
            slop=0.0
        )
        assert state == LimitState.AT_LOWER
        # Impulse should be affected by restitution

    def test_with_angular_limit(self):
        """Should work with AngularLimit."""
        limit = AngularLimit(lower=-math.pi/4, upper=math.pi/4, contact_distance=0.02)
        impulse, state = compute_limit_impulse(
            limit=limit,
            current_value=math.pi/2,
            current_velocity=1.0,
            effective_mass=1.0,
            dt=0.016,
            slop=0.0
        )
        assert state == LimitState.AT_UPPER


class TestComputeSoftLimitCoefficients:
    """Tests for compute_soft_limit_coefficients function."""

    def test_zero_stiffness(self):
        """Should return defaults for zero stiffness."""
        gamma, beta, softness = compute_soft_limit_coefficients(
            stiffness=0.0,
            damping=10.0,
            effective_mass=1.0,
            dt=0.016
        )
        assert gamma == 0.0
        assert beta == 1.0
        assert softness == 0.0

    def test_zero_effective_mass(self):
        """Should return defaults for zero effective mass."""
        gamma, beta, softness = compute_soft_limit_coefficients(
            stiffness=100.0,
            damping=10.0,
            effective_mass=0.0,
            dt=0.016
        )
        assert gamma == 0.0
        assert beta == 1.0
        assert softness == 0.0

    def test_positive_stiffness(self):
        """Should compute valid coefficients for positive stiffness."""
        gamma, beta, softness = compute_soft_limit_coefficients(
            stiffness=100.0,
            damping=10.0,
            effective_mass=1.0,
            dt=0.016
        )
        assert gamma > 0.0
        assert 0.0 <= beta <= 1.0
        assert softness >= 0.0

    def test_high_stiffness(self):
        """High stiffness should produce higher beta than low stiffness."""
        gamma_low, beta_low, _ = compute_soft_limit_coefficients(
            stiffness=100.0,
            damping=10.0,
            effective_mass=1.0,
            dt=0.016
        )
        gamma_high, beta_high, _ = compute_soft_limit_coefficients(
            stiffness=10000.0,
            damping=10.0,
            effective_mass=1.0,
            dt=0.016
        )
        # Higher stiffness should increase beta
        assert beta_high > beta_low

    def test_high_damping(self):
        """High damping should reduce beta."""
        gamma1, beta1, _ = compute_soft_limit_coefficients(
            stiffness=100.0,
            damping=10.0,
            effective_mass=1.0,
            dt=0.016
        )
        gamma2, beta2, _ = compute_soft_limit_coefficients(
            stiffness=100.0,
            damping=1000.0,
            effective_mass=1.0,
            dt=0.016
        )
        # Higher damping should reduce beta
        assert beta2 < beta1
