"""
Limit Helpers for Joint Constraints.

Provides limit functionality for joints that support constrained
movement (angle limits, position limits).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum, auto
import math

from ..solver.config import BAUMGARTE_FACTOR, SLOP


class LimitState(Enum):
    """Current state of a limit."""
    INACTIVE = auto()   # Within limits
    AT_LOWER = auto()   # At lower limit
    AT_UPPER = auto()   # At upper limit


@dataclass
class LinearLimit:
    """
    Linear (distance) limit configuration.

    Constrains movement along an axis between lower and upper bounds.

    Attributes:
        lower: Lower limit (minimum position).
        upper: Upper limit (maximum position).
        stiffness: Soft limit spring stiffness (0 = hard limit).
        damping: Soft limit damping coefficient.
        restitution: Bounce coefficient when hitting limit.
        contact_distance: Distance at which limit begins to apply.
    """
    lower: float = -1.0
    upper: float = 1.0
    stiffness: float = 0.0
    damping: float = 0.0
    restitution: float = 0.0
    contact_distance: float = 0.01

    def __post_init__(self):
        if self.lower > self.upper:
            self.lower, self.upper = self.upper, self.lower

    @property
    def is_soft(self) -> bool:
        """Check if this is a soft limit."""
        return self.stiffness > 0

    @property
    def range(self) -> float:
        """Get limit range."""
        return self.upper - self.lower

    @property
    def center(self) -> float:
        """Get center of limit range."""
        return (self.lower + self.upper) / 2

    def check_state(self, value: float) -> LimitState:
        """
        Check current limit state.

        Args:
            value: Current position value.

        Returns:
            LimitState indicating if at/beyond limits.
        """
        # Use contact_distance for early activation to prevent tunneling
        if value <= self.lower + self.contact_distance:
            return LimitState.AT_LOWER
        elif value >= self.upper - self.contact_distance:
            return LimitState.AT_UPPER
        else:
            return LimitState.INACTIVE

    def compute_error(self, value: float) -> float:
        """
        Compute error from limit.

        Args:
            value: Current position value.

        Returns:
            Error (positive = beyond limit, 0 = within limits).
        """
        if value < self.lower:
            return self.lower - value
        elif value > self.upper:
            return value - self.upper
        else:
            return 0.0

    def clamp(self, value: float) -> float:
        """Clamp value to within limits."""
        return max(self.lower, min(self.upper, value))


@dataclass
class AngularLimit:
    """
    Angular (rotation) limit configuration.

    Constrains rotation between lower and upper bounds.
    Handles wraparound for angles.

    Attributes:
        lower: Lower limit (minimum angle in radians).
        upper: Upper limit (maximum angle in radians).
        stiffness: Soft limit spring stiffness (0 = hard limit).
        damping: Soft limit damping coefficient.
        restitution: Bounce coefficient when hitting limit.
        contact_distance: Angular distance at which limit begins to apply.
    """
    lower: float = -math.pi
    upper: float = math.pi
    stiffness: float = 0.0
    damping: float = 0.0
    restitution: float = 0.0
    contact_distance: float = 0.01

    def __post_init__(self):
        # Normalize to [-pi, pi]
        self.lower = self._normalize_angle(self.lower)
        self.upper = self._normalize_angle(self.upper)

        # Ensure lower <= upper (swap if needed)
        if self.lower > self.upper:
            self.lower, self.upper = self.upper, self.lower

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    @property
    def is_soft(self) -> bool:
        """Check if this is a soft limit."""
        return self.stiffness > 0

    @property
    def range(self) -> float:
        """Get limit range."""
        return self.upper - self.lower

    @property
    def center(self) -> float:
        """Get center of limit range."""
        return (self.lower + self.upper) / 2

    def check_state(self, angle: float) -> LimitState:
        """
        Check current limit state.

        Args:
            angle: Current angle in radians.

        Returns:
            LimitState indicating if at/beyond limits.
        """
        angle = self._normalize_angle(angle)

        # Use contact_distance for early activation to prevent tunneling
        if angle <= self.lower + self.contact_distance:
            return LimitState.AT_LOWER
        elif angle >= self.upper - self.contact_distance:
            return LimitState.AT_UPPER
        else:
            return LimitState.INACTIVE

    def compute_error(self, angle: float) -> float:
        """
        Compute error from limit.

        Args:
            angle: Current angle in radians.

        Returns:
            Error (positive = beyond limit, 0 = within limits).
        """
        angle = self._normalize_angle(angle)

        if angle < self.lower:
            return self.lower - angle
        elif angle > self.upper:
            return angle - self.upper
        else:
            return 0.0

    def clamp(self, angle: float) -> float:
        """Clamp angle to within limits."""
        angle = self._normalize_angle(angle)
        return max(self.lower, min(self.upper, angle))


@dataclass
class SwingLimit:
    """
    Swing limit configuration (cone limit).

    Constrains the angle between two axes using a cone.
    Used for ball joint swing limits.

    Attributes:
        y_angle: Half-angle of cone in Y direction (radians).
        z_angle: Half-angle of cone in Z direction (radians).
        stiffness: Soft limit spring stiffness.
        damping: Soft limit damping coefficient.
        restitution: Bounce coefficient.
    """
    y_angle: float = math.pi / 4
    z_angle: float = math.pi / 4
    stiffness: float = 0.0
    damping: float = 0.0
    restitution: float = 0.0

    @property
    def is_circular(self) -> bool:
        """Check if cone is circular (y_angle == z_angle)."""
        return abs(self.y_angle - self.z_angle) < 1e-6

    @property
    def is_soft(self) -> bool:
        """Check if this is a soft limit."""
        return self.stiffness > 0

    def check_within_cone(self, swing_y: float, swing_z: float) -> bool:
        """
        Check if swing angles are within cone.

        Args:
            swing_y: Swing angle in Y direction.
            swing_z: Swing angle in Z direction.

        Returns:
            True if within cone.
        """
        if self.is_circular:
            # Circular cone: check angle magnitude
            angle = math.sqrt(swing_y * swing_y + swing_z * swing_z)
            return angle <= self.y_angle
        else:
            # Elliptical cone: check ellipse equation
            # (y/y_max)^2 + (z/z_max)^2 <= 1
            y_norm = swing_y / self.y_angle if self.y_angle > 0 else 0
            z_norm = swing_z / self.z_angle if self.z_angle > 0 else 0
            return y_norm * y_norm + z_norm * z_norm <= 1.0

    def compute_error(self, swing_y: float, swing_z: float) -> float:
        """
        Compute error from cone surface.

        Args:
            swing_y: Swing angle in Y direction.
            swing_z: Swing angle in Z direction.

        Returns:
            Error distance from cone surface (0 if inside).
        """
        if self.is_circular:
            angle = math.sqrt(swing_y * swing_y + swing_z * swing_z)
            return max(0.0, angle - self.y_angle)
        else:
            y_norm = swing_y / self.y_angle if self.y_angle > 0 else 0
            z_norm = swing_z / self.z_angle if self.z_angle > 0 else 0
            ellipse_dist = y_norm * y_norm + z_norm * z_norm
            if ellipse_dist <= 1.0:
                return 0.0
            # Approximate error (not exact for ellipse)
            return (math.sqrt(ellipse_dist) - 1.0) * min(self.y_angle, self.z_angle)


@dataclass
class TwistLimit:
    """
    Twist limit configuration.

    Constrains rotation around the twist (X) axis.

    Attributes:
        lower: Lower twist angle (radians).
        upper: Upper twist angle (radians).
        stiffness: Soft limit spring stiffness.
        damping: Soft limit damping coefficient.
    """
    lower: float = -math.pi
    upper: float = math.pi
    stiffness: float = 0.0
    damping: float = 0.0

    @property
    def is_soft(self) -> bool:
        """Check if this is a soft limit."""
        return self.stiffness > 0

    def check_state(self, twist: float) -> LimitState:
        """Check current twist limit state."""
        # Normalize twist
        while twist > math.pi:
            twist -= 2 * math.pi
        while twist < -math.pi:
            twist += 2 * math.pi

        if twist <= self.lower:
            return LimitState.AT_LOWER
        elif twist >= self.upper:
            return LimitState.AT_UPPER
        else:
            return LimitState.INACTIVE

    def compute_error(self, twist: float) -> float:
        """Compute twist error from limit."""
        # Normalize
        while twist > math.pi:
            twist -= 2 * math.pi
        while twist < -math.pi:
            twist += 2 * math.pi

        if twist < self.lower:
            return self.lower - twist
        elif twist > self.upper:
            return twist - self.upper
        else:
            return 0.0


def compute_limit_impulse(
    limit: LinearLimit | AngularLimit,
    current_value: float,
    current_velocity: float,
    effective_mass: float,
    dt: float,
    baumgarte_factor: float = BAUMGARTE_FACTOR,
    slop: float = SLOP
) -> Tuple[float, LimitState]:
    """
    Compute impulse for a limit constraint.

    Args:
        limit: Limit configuration.
        current_value: Current position/angle.
        current_velocity: Current velocity.
        effective_mass: Effective mass of constraint.
        dt: Time step.
        baumgarte_factor: Baumgarte stabilization factor.
        slop: Allowed penetration before correction.

    Returns:
        Tuple of (impulse, limit_state).
    """
    if effective_mass == 0:
        return 0.0, LimitState.INACTIVE

    state = limit.check_state(current_value)

    if state == LimitState.INACTIVE:
        return 0.0, state

    # Compute error
    error = limit.compute_error(current_value)
    error = max(0.0, error - slop)

    if error <= 0:
        return 0.0, LimitState.INACTIVE

    # Compute bias velocity
    bias = baumgarte_factor * error / dt

    # Add restitution for velocity
    if hasattr(limit, 'restitution') and limit.restitution > 0:
        if state == LimitState.AT_LOWER and current_velocity < 0:
            bias -= limit.restitution * current_velocity
        elif state == LimitState.AT_UPPER and current_velocity > 0:
            bias -= limit.restitution * current_velocity

    # Soft limit: add spring/damper forces
    if limit.is_soft:
        # Spring force
        spring_impulse = limit.stiffness * error * dt * effective_mass
        # Damper force
        damper_impulse = limit.damping * current_velocity * dt * effective_mass

        impulse = spring_impulse - damper_impulse
    else:
        # Hard limit: compute impulse from velocity error
        # Target: bring velocity to zero at limit, plus correction
        if state == LimitState.AT_LOWER:
            velocity_error = -current_velocity + bias
        else:  # AT_UPPER
            velocity_error = -current_velocity - bias

        impulse = effective_mass * velocity_error

    # Clamp to non-negative for inequality constraint
    # Lower limit: impulse pushes positive
    # Upper limit: impulse pushes negative
    if state == LimitState.AT_LOWER:
        impulse = max(0.0, impulse)
    else:
        impulse = min(0.0, impulse)

    return impulse, state


def compute_soft_limit_coefficients(
    stiffness: float,
    damping: float,
    effective_mass: float,
    dt: float
) -> Tuple[float, float, float]:
    """
    Compute soft constraint coefficients.

    For soft limits using spring-damper model.

    Args:
        stiffness: Spring stiffness.
        damping: Damping coefficient.
        effective_mass: Effective mass.
        dt: Time step.

    Returns:
        Tuple of (gamma, beta, softness).
    """
    if stiffness <= 0 or effective_mass <= 0:
        return 0.0, 1.0, 0.0

    omega = math.sqrt(stiffness / effective_mass)
    d = 2.0 * damping * omega

    c_plus_dtk = d + dt * stiffness
    if c_plus_dtk <= 0:
        return 0.0, 1.0, 0.0

    gamma = 1.0 / c_plus_dtk
    beta = dt * stiffness * gamma
    softness = gamma / dt

    return gamma, beta, softness
