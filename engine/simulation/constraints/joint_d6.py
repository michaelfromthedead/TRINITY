"""
D6 Joint (Configurable Joint) Implementation.

A D6 joint is a highly configurable joint that allows independent
configuration of each of the 6 degrees of freedom. Each axis can be
locked, limited, or free.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum, auto
import math

from .joint_base import Joint, JointState
from .joint_motors import Motor, MotorMode
from .joint_limits import LinearLimit, AngularLimit, LimitState
from ..solver.jacobian import Vec3, Mat3, Quaternion, Jacobian
from ..solver.constraint_solver import RigidBody, ConstraintType
from ..solver.config import SolverConfig


class D6MotionType(Enum):
    """Motion type for each axis."""
    LOCKED = auto()    # Axis is completely locked
    LIMITED = auto()   # Axis has limits
    FREE = auto()      # Axis is unconstrained


class D6Axis(Enum):
    """Axes for D6 joint configuration."""
    LINEAR_X = 0
    LINEAR_Y = 1
    LINEAR_Z = 2
    ANGULAR_X = 3  # Twist
    ANGULAR_Y = 4  # Swing1
    ANGULAR_Z = 5  # Swing2


@dataclass
class D6AxisConfig:
    """Configuration for a single axis."""
    motion: D6MotionType = D6MotionType.LOCKED
    lower_limit: float = 0.0
    upper_limit: float = 0.0
    stiffness: float = 0.0  # For soft limits
    damping: float = 0.0
    motor_enabled: bool = False
    motor: Optional[Motor] = None


class D6Joint(Joint):
    """
    D6 Joint (6 DOF configurable joint).

    Each of the 6 degrees of freedom can be independently configured:
    - LINEAR_X, LINEAR_Y, LINEAR_Z: Translation along each axis
    - ANGULAR_X (twist), ANGULAR_Y (swing1), ANGULAR_Z (swing2): Rotation

    Each axis can be:
    - LOCKED: No motion allowed
    - LIMITED: Motion within specified limits
    - FREE: Unrestricted motion

    Supports per-axis motors and soft limits.

    Attributes:
        axis_config: Configuration for each axis.
        use_swing_cone: Use cone limit for combined swing instead of per-axis.
    """

    def __init__(
        self,
        body_a: RigidBody,
        body_b: Optional[RigidBody] = None,
        local_anchor_a: Vec3 = None,
        local_anchor_b: Vec3 = None,
        break_force: float = 0.0,
        break_torque: float = 0.0
    ):
        """
        Initialize D6 joint.

        Args:
            body_a: First rigid body.
            body_b: Second rigid body.
            local_anchor_a: Anchor in body A's local space.
            local_anchor_b: Anchor in body B's local space.
            break_force: Force threshold for breaking.
            break_torque: Torque threshold for breaking.
        """
        super().__init__(
            body_a, body_b,
            local_anchor_a, local_anchor_b,
            break_force, break_torque
        )

        # Store reference frame
        if body_b is not None:
            self._reference_orientation = (
                body_b.orientation.conjugate() * body_a.orientation
            )
        else:
            self._reference_orientation = body_a.orientation

        # Axis configuration
        self._axis_config: Dict[D6Axis, D6AxisConfig] = {
            axis: D6AxisConfig() for axis in D6Axis
        }

        # Use swing cone instead of per-axis swing limits
        self._use_swing_cone = False
        self._swing_cone_angle = math.pi / 4

        # Local reference axes
        self._local_x_axis_a = Vec3.unit_x()
        self._local_y_axis_a = Vec3.unit_y()
        self._local_z_axis_a = Vec3.unit_z()

        # Initialize constraint storage (max 6 + limits + motors)
        max_rows = 12
        self._jacobians = [Jacobian() for _ in range(max_rows)]
        self._effective_masses = [0.0] * max_rows
        self._biases = [0.0] * max_rows
        self._accumulated_impulse = [0.0] * max_rows
        self._warm_start_impulse = [0.0] * max_rows
        self._lower_limits = [float("-inf")] * max_rows
        self._upper_limits = [float("inf")] * max_rows

        self._active_constraint_count = 0
        self._axis_to_row: Dict[D6Axis, int] = {}

    def set_motion(self, axis: D6Axis, motion: D6MotionType) -> None:
        """
        Set motion type for an axis.

        Args:
            axis: Which axis to configure.
            motion: Motion type (LOCKED, LIMITED, FREE).
        """
        self._axis_config[axis].motion = motion

    def get_motion(self, axis: D6Axis) -> D6MotionType:
        """Get motion type for an axis."""
        return self._axis_config[axis].motion

    def set_linear_limit(
        self,
        axis: D6Axis,
        lower: float,
        upper: float,
        stiffness: float = 0.0,
        damping: float = 0.0
    ) -> None:
        """
        Set linear limit for an axis.

        Args:
            axis: LINEAR_X, LINEAR_Y, or LINEAR_Z.
            lower: Lower limit.
            upper: Upper limit.
            stiffness: Soft limit stiffness (0 = hard limit).
            damping: Soft limit damping.
        """
        if axis not in [D6Axis.LINEAR_X, D6Axis.LINEAR_Y, D6Axis.LINEAR_Z]:
            raise ValueError("Linear limits only apply to linear axes")

        config = self._axis_config[axis]
        config.motion = D6MotionType.LIMITED
        config.lower_limit = lower
        config.upper_limit = upper
        config.stiffness = stiffness
        config.damping = damping

    def set_angular_limit(
        self,
        axis: D6Axis,
        lower: float,
        upper: float,
        stiffness: float = 0.0,
        damping: float = 0.0
    ) -> None:
        """
        Set angular limit for an axis.

        Args:
            axis: ANGULAR_X (twist), ANGULAR_Y, or ANGULAR_Z.
            lower: Lower limit in radians.
            upper: Upper limit in radians.
            stiffness: Soft limit stiffness.
            damping: Soft limit damping.
        """
        if axis not in [D6Axis.ANGULAR_X, D6Axis.ANGULAR_Y, D6Axis.ANGULAR_Z]:
            raise ValueError("Angular limits only apply to angular axes")

        config = self._axis_config[axis]
        config.motion = D6MotionType.LIMITED
        config.lower_limit = lower
        config.upper_limit = upper
        config.stiffness = stiffness
        config.damping = damping

    def set_swing_cone_limit(self, angle: float) -> None:
        """
        Set swing cone limit (combined swing1 and swing2).

        Args:
            angle: Cone half-angle in radians.
        """
        self._use_swing_cone = True
        self._swing_cone_angle = angle
        self._axis_config[D6Axis.ANGULAR_Y].motion = D6MotionType.LIMITED
        self._axis_config[D6Axis.ANGULAR_Z].motion = D6MotionType.LIMITED

    def set_twist_limit(self, lower: float, upper: float) -> None:
        """
        Set twist limit (rotation around X axis).

        Args:
            lower: Lower twist limit in radians.
            upper: Upper twist limit in radians.
        """
        self.set_angular_limit(D6Axis.ANGULAR_X, lower, upper)

    def set_motor(
        self,
        axis: D6Axis,
        mode: MotorMode,
        target: float,
        max_force: float
    ) -> None:
        """
        Configure motor for an axis.

        Args:
            axis: Which axis to motorize.
            mode: VELOCITY or POSITION mode.
            target: Target velocity or position.
            max_force: Maximum motor force/torque.
        """
        config = self._axis_config[axis]
        config.motor_enabled = True
        config.motor = Motor(mode, target, max_force)

    def disable_motor(self, axis: D6Axis) -> None:
        """Disable motor for an axis."""
        self._axis_config[axis].motor_enabled = False

    def lock_all(self) -> None:
        """Lock all 6 DOF."""
        for axis in D6Axis:
            self._axis_config[axis].motion = D6MotionType.LOCKED

    def free_all(self) -> None:
        """Free all 6 DOF."""
        for axis in D6Axis:
            self._axis_config[axis].motion = D6MotionType.FREE

    def get_constraint_count(self) -> int:
        """Get number of active constraint rows."""
        return self._active_constraint_count

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """Prepare D6 joint for solving."""
        if self._state != JointState.ACTIVE:
            return

        self._body_a.update_world_inertia()
        if self._body_b is not None:
            self._body_b.update_world_inertia()

        # Get anchor positions and r vectors
        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()
        r_a, r_b = self._get_r_vectors()

        # Transform reference axes to world space
        x_axis = self._body_a.local_to_world_direction(self._local_x_axis_a)
        y_axis = self._body_a.local_to_world_direction(self._local_y_axis_a)
        z_axis = self._body_a.local_to_world_direction(self._local_z_axis_a)

        # Position error
        position_error = anchor_b - anchor_a

        # Angular error
        if self._body_b is not None:
            q_current = self._body_b.orientation.conjugate() * self._body_a.orientation
        else:
            q_current = self._body_a.orientation

        q_error = q_current * self._reference_orientation.conjugate()
        if q_error.w < 0:
            q_error = Quaternion(-q_error.x, -q_error.y, -q_error.z, -q_error.w)

        angular_error = Vec3(q_error.x, q_error.y, q_error.z) * 2.0

        self._active_constraint_count = 0
        self._axis_to_row.clear()

        # ============ LINEAR CONSTRAINTS ============
        linear_axes = [
            (D6Axis.LINEAR_X, x_axis, position_error.dot(x_axis)),
            (D6Axis.LINEAR_Y, y_axis, position_error.dot(y_axis)),
            (D6Axis.LINEAR_Z, z_axis, position_error.dot(z_axis)),
        ]

        for axis_enum, world_axis, error in linear_axes:
            axis_config = self._axis_config[axis_enum]

            if axis_config.motion == D6MotionType.LOCKED:
                self._setup_linear_constraint(
                    world_axis, r_a, r_b, error, dt, config
                )
                self._axis_to_row[axis_enum] = self._active_constraint_count - 1

            elif axis_config.motion == D6MotionType.LIMITED:
                # Check if at limits
                if error < axis_config.lower_limit:
                    limit_error = axis_config.lower_limit - error
                    self._setup_linear_limit_constraint(
                        -world_axis, r_a, r_b, limit_error, dt, config, lower=True
                    )
                    self._axis_to_row[axis_enum] = self._active_constraint_count - 1

                elif error > axis_config.upper_limit:
                    limit_error = error - axis_config.upper_limit
                    self._setup_linear_limit_constraint(
                        world_axis, r_a, r_b, limit_error, dt, config, lower=False
                    )
                    self._axis_to_row[axis_enum] = self._active_constraint_count - 1

            # Motor
            if axis_config.motor_enabled and axis_config.motor:
                self._setup_linear_motor(
                    axis_enum, world_axis, r_a, r_b, error, dt
                )

        # ============ ANGULAR CONSTRAINTS ============
        angular_axes = [
            (D6Axis.ANGULAR_X, x_axis, angular_error.dot(x_axis)),
            (D6Axis.ANGULAR_Y, y_axis, angular_error.dot(y_axis)),
            (D6Axis.ANGULAR_Z, z_axis, angular_error.dot(z_axis)),
        ]

        for axis_enum, world_axis, error in angular_axes:
            axis_config = self._axis_config[axis_enum]

            if axis_config.motion == D6MotionType.LOCKED:
                self._setup_angular_constraint(
                    world_axis, error, dt, config
                )
                self._axis_to_row[axis_enum] = self._active_constraint_count - 1

            elif axis_config.motion == D6MotionType.LIMITED:
                # Check if at limits
                if error < axis_config.lower_limit:
                    limit_error = axis_config.lower_limit - error
                    self._setup_angular_limit_constraint(
                        -world_axis, limit_error, dt, config, lower=True
                    )
                    self._axis_to_row[axis_enum] = self._active_constraint_count - 1

                elif error > axis_config.upper_limit:
                    limit_error = error - axis_config.upper_limit
                    self._setup_angular_limit_constraint(
                        world_axis, limit_error, dt, config, lower=False
                    )
                    self._axis_to_row[axis_enum] = self._active_constraint_count - 1

            # Motor
            if axis_config.motor_enabled and axis_config.motor:
                self._setup_angular_motor(
                    axis_enum, world_axis, error, dt
                )

    def _setup_linear_constraint(
        self,
        axis: Vec3,
        r_a: Vec3,
        r_b: Vec3,
        error: float,
        dt: float,
        config: SolverConfig
    ) -> None:
        """Set up a locked linear constraint."""
        idx = self._active_constraint_count

        self._jacobians[idx] = Jacobian(
            linear_a=-axis,
            angular_a=-r_a.cross(axis),
            linear_b=axis,
            angular_b=r_b.cross(axis)
        )
        self._effective_masses[idx] = self._compute_effective_mass(self._jacobians[idx])
        self._biases[idx] = config.baumgarte_factor * error / dt
        self._lower_limits[idx] = float("-inf")
        self._upper_limits[idx] = float("inf")

        self._active_constraint_count += 1

    def _setup_linear_limit_constraint(
        self,
        axis: Vec3,
        error: float,
        dt: float,
        config: SolverConfig,
        lower: bool
    ) -> None:
        """Set up a linear limit constraint."""
        r_a, r_b = self._get_r_vectors()
        idx = self._active_constraint_count

        self._jacobians[idx] = Jacobian(
            linear_a=-axis,
            angular_a=-r_a.cross(axis),
            linear_b=axis,
            angular_b=r_b.cross(axis)
        )
        self._effective_masses[idx] = self._compute_effective_mass(self._jacobians[idx])
        self._biases[idx] = config.baumgarte_factor * error / dt
        self._lower_limits[idx] = 0.0
        self._upper_limits[idx] = float("inf")

        self._active_constraint_count += 1

    def _setup_angular_constraint(
        self,
        axis: Vec3,
        error: float,
        dt: float,
        config: SolverConfig
    ) -> None:
        """Set up a locked angular constraint."""
        idx = self._active_constraint_count

        self._jacobians[idx] = Jacobian(
            linear_a=Vec3.zero(),
            angular_a=-axis,
            linear_b=Vec3.zero(),
            angular_b=axis
        )
        self._effective_masses[idx] = self._compute_effective_mass(self._jacobians[idx])
        self._biases[idx] = config.baumgarte_factor * error / dt
        self._lower_limits[idx] = float("-inf")
        self._upper_limits[idx] = float("inf")

        self._active_constraint_count += 1

    def _setup_angular_limit_constraint(
        self,
        axis: Vec3,
        error: float,
        dt: float,
        config: SolverConfig,
        lower: bool
    ) -> None:
        """Set up an angular limit constraint."""
        idx = self._active_constraint_count

        self._jacobians[idx] = Jacobian(
            linear_a=Vec3.zero(),
            angular_a=-axis,
            linear_b=Vec3.zero(),
            angular_b=axis
        )
        self._effective_masses[idx] = self._compute_effective_mass(self._jacobians[idx])
        self._biases[idx] = config.baumgarte_factor * error / dt
        self._lower_limits[idx] = 0.0
        self._upper_limits[idx] = float("inf")

        self._active_constraint_count += 1

    def _setup_linear_motor(
        self,
        axis_enum: D6Axis,
        world_axis: Vec3,
        r_a: Vec3,
        r_b: Vec3,
        current_pos: float,
        dt: float
    ) -> None:
        """Set up linear motor constraint."""
        motor = self._axis_config[axis_enum].motor
        if not motor:
            return

        idx = self._active_constraint_count

        self._jacobians[idx] = Jacobian(
            linear_a=-world_axis,
            angular_a=-r_a.cross(world_axis),
            linear_b=world_axis,
            angular_b=r_b.cross(world_axis)
        )
        self._effective_masses[idx] = self._compute_effective_mass(self._jacobians[idx])

        if motor.mode == MotorMode.VELOCITY:
            # Get current velocity along axis
            rel_vel = self._body_a.velocity
            if self._body_b:
                rel_vel = rel_vel - self._body_b.velocity
            current_vel = rel_vel.dot(world_axis)
            self._biases[idx] = motor.target - current_vel
        else:
            self._biases[idx] = (motor.target - current_pos) / dt

        max_impulse = motor.max_force * dt
        self._lower_limits[idx] = -max_impulse
        self._upper_limits[idx] = max_impulse

        self._active_constraint_count += 1

    def _setup_angular_motor(
        self,
        axis_enum: D6Axis,
        world_axis: Vec3,
        current_angle: float,
        dt: float
    ) -> None:
        """Set up angular motor constraint."""
        motor = self._axis_config[axis_enum].motor
        if not motor:
            return

        idx = self._active_constraint_count

        self._jacobians[idx] = Jacobian(
            linear_a=Vec3.zero(),
            angular_a=-world_axis,
            linear_b=Vec3.zero(),
            angular_b=world_axis
        )
        self._effective_masses[idx] = self._compute_effective_mass(self._jacobians[idx])

        if motor.mode == MotorMode.VELOCITY:
            rel_ang_vel = self._body_a.angular_velocity
            if self._body_b:
                rel_ang_vel = rel_ang_vel - self._body_b.angular_velocity
            current_vel = rel_ang_vel.dot(world_axis)
            self._biases[idx] = motor.target - current_vel
        else:
            self._biases[idx] = (motor.target - current_angle) / dt

        max_impulse = motor.max_force * dt
        self._lower_limits[idx] = -max_impulse
        self._upper_limits[idx] = max_impulse

        self._active_constraint_count += 1

    def _solve_position_internal(self, max_correction: float) -> float:
        """Solve position constraints for D6 joint."""
        max_error = 0.0

        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()
        position_error = anchor_b - anchor_a
        max_error = max(max_error, position_error.length())

        # For simplicity, apply corrections for locked linear axes
        r_a, r_b = self._get_r_vectors()
        x_axis = self._body_a.local_to_world_direction(self._local_x_axis_a)
        y_axis = self._body_a.local_to_world_direction(self._local_y_axis_a)
        z_axis = self._body_a.local_to_world_direction(self._local_z_axis_a)

        for axis_enum, world_axis in [
            (D6Axis.LINEAR_X, x_axis),
            (D6Axis.LINEAR_Y, y_axis),
            (D6Axis.LINEAR_Z, z_axis),
        ]:
            if self._axis_config[axis_enum].motion == D6MotionType.LOCKED:
                error = position_error.dot(world_axis)
                if abs(error) > 1e-6:
                    self._apply_position_correction(
                        world_axis, r_a, r_b, error, max_correction
                    )

        return max_error

    def _apply_position_correction(
        self,
        axis: Vec3,
        r_a: Vec3,
        r_b: Vec3,
        error: float,
        max_correction: float
    ) -> None:
        """Apply position correction along an axis."""
        total_inv_mass = 0.0
        if not self._body_a.is_static:
            total_inv_mass += self._body_a.inv_mass
        if self._body_b is not None and not self._body_b.is_static:
            total_inv_mass += self._body_b.inv_mass

        if total_inv_mass < 1e-10:
            return

        correction = max(-max_correction, min(max_correction, error)) * 0.2

        if not self._body_a.is_static:
            self._body_a.position = self._body_a.position - axis * (
                correction * self._body_a.inv_mass / total_inv_mass
            )

        if self._body_b is not None and not self._body_b.is_static:
            self._body_b.position = self._body_b.position + axis * (
                correction * self._body_b.inv_mass / total_inv_mass
            )

    @classmethod
    def create_fixed(cls, body_a: RigidBody, body_b: RigidBody) -> "D6Joint":
        """Create D6 joint with all axes locked (like fixed joint)."""
        joint = cls(body_a, body_b)
        joint.lock_all()
        return joint

    @classmethod
    def create_hinge(
        cls,
        body_a: RigidBody,
        body_b: RigidBody,
        world_anchor: Vec3,
        world_axis: Vec3
    ) -> "D6Joint":
        """Create D6 joint configured as hinge."""
        local_anchor_a = body_a.world_to_local(world_anchor)
        local_anchor_b = body_b.world_to_local(world_anchor)

        joint = cls(body_a, body_b, local_anchor_a, local_anchor_b)

        # Lock all except rotation around specified axis
        joint.lock_all()

        # Free the hinge axis (assuming X is twist/hinge)
        joint.set_motion(D6Axis.ANGULAR_X, D6MotionType.FREE)

        return joint

    @classmethod
    def create_slider(
        cls,
        body_a: RigidBody,
        body_b: RigidBody,
        world_anchor: Vec3,
        world_axis: Vec3
    ) -> "D6Joint":
        """Create D6 joint configured as slider."""
        local_anchor_a = body_a.world_to_local(world_anchor)
        local_anchor_b = body_b.world_to_local(world_anchor)

        joint = cls(body_a, body_b, local_anchor_a, local_anchor_b)

        # Lock all except translation along specified axis
        joint.lock_all()

        # Free the slider axis (assuming X)
        joint.set_motion(D6Axis.LINEAR_X, D6MotionType.FREE)

        return joint
