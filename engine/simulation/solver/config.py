"""
Solver Configuration for Constraint Solver Module.

Contains default parameters and configuration dataclass for tuning
physics constraint solving behavior.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto


# ============================================================================
# Default Solver Constants
# ============================================================================

# Number of iterations for velocity constraint solving
# Higher values = more accurate but slower
DEFAULT_VELOCITY_ITERATIONS: int = 8

# Number of iterations for position constraint solving
# Typically fewer than velocity iterations as position correction is less critical
DEFAULT_POSITION_ITERATIONS: int = 3

# Baumgarte stabilization factor
# Controls how aggressively position errors are corrected
# Range: 0.0 to 1.0, typically 0.1 to 0.3
BAUMGARTE_FACTOR: float = 0.2

# Penetration slop (allowance)
# Small penetration depth that is ignored to prevent jitter
# Measured in world units (meters)
SLOP: float = 0.01

# Warm starting factor
# How much of previous frame's impulse to apply at start
# Range: 0.0 (disabled) to 1.0 (full warm starting)
WARM_START_FACTOR: float = 0.8

# Maximum velocity for position correction
# Prevents explosive corrections from large penetrations
# Measured in world units per second
MAX_CORRECTION_VELOCITY: float = 10.0

# Successive Over-Relaxation (SOR) factor
# 1.0 = standard Gauss-Seidel
# > 1.0 = over-relaxation (faster convergence, may be unstable)
# < 1.0 = under-relaxation (slower but more stable)
RELAXATION_FACTOR: float = 1.0

# Maximum linear velocity (clamp)
MAX_LINEAR_VELOCITY: float = 200.0

# Maximum angular velocity (clamp)
MAX_ANGULAR_VELOCITY: float = 50.0

# Minimum mass for a body to be considered dynamic
MIN_DYNAMIC_MASS: float = 0.001

# Default friction coefficient
DEFAULT_FRICTION: float = 0.4

# Default restitution (bounciness)
DEFAULT_RESTITUTION: float = 0.0

# Position correction factor for joints
# Controls how much of the position error is corrected per iteration
# Range: 0.0 to 1.0, typically 0.1 to 0.3
POSITION_CORRECTION_FACTOR: float = 0.2

# Contact constraint bias velocity
CONTACT_BIAS_VELOCITY: float = 0.001

# Minimum normal impulse for contact
MIN_CONTACT_IMPULSE: float = 1e-8

# ============================================================================
# Motor Controller Constants
# ============================================================================

# Default motor PID gains
MOTOR_DEFAULT_KP: float = 100.0  # Position gain (proportional)
MOTOR_DEFAULT_KD: float = 10.0   # Velocity gain (derivative/damping)
MOTOR_DEFAULT_KI: float = 0.0    # Integral gain

# Default motor limits
MOTOR_DEFAULT_MAX_FORCE: float = 100.0      # Maximum motor force/torque
MOTOR_DEFAULT_MAX_VELOCITY: float = 10.0     # Maximum motor velocity
MOTOR_DEFAULT_MAX_ACCELERATION: float = 100.0  # Maximum motor acceleration

# ============================================================================
# XPBD Solver Constants
# ============================================================================

# Default velocity damping factor for XPBD solver
# Range: 0.0 to 1.0, where 1.0 = no damping
XPBD_VELOCITY_DAMPING: float = 0.99


class SolverType(Enum):
    """Enumeration of available solver types."""
    SEQUENTIAL_IMPULSE = auto()
    TEMPORAL_GAUSS_SEIDEL = auto()
    XPBD = auto()


class WarmStartMode(Enum):
    """Warm starting modes for constraint solver."""
    DISABLED = auto()
    NORMAL = auto()
    AGGRESSIVE = auto()


@dataclass
class SolverConfig:
    """
    Configuration for constraint solvers.

    Attributes:
        velocity_iterations: Number of velocity constraint iterations per step.
        position_iterations: Number of position constraint iterations per step.
        baumgarte_factor: Baumgarte stabilization coefficient.
        slop: Allowed penetration depth before correction.
        warm_start_factor: How much previous impulses influence current frame.
        max_correction_velocity: Maximum velocity for position corrections.
        relaxation_factor: SOR relaxation factor.
        solver_type: Type of solver to use.
        warm_start_mode: Warm starting strategy.
        enable_sleeping: Whether to allow bodies to sleep.
        sleep_velocity_threshold: Linear velocity below which bodies may sleep.
        sleep_angular_threshold: Angular velocity below which bodies may sleep.
        sleep_time_threshold: Time body must be slow before sleeping.
        substeps: Number of substeps per physics step.
        max_linear_velocity: Maximum linear velocity clamp.
        max_angular_velocity: Maximum angular velocity clamp.
        contact_bias: Velocity bias for contact constraints.
        use_block_solver: Whether to use block solving for contacts.
        friction_iterations: Additional iterations for friction solving.
        enable_gyroscopic_torque: Whether to apply gyroscopic forces.
    """

    velocity_iterations: int = DEFAULT_VELOCITY_ITERATIONS
    position_iterations: int = DEFAULT_POSITION_ITERATIONS
    baumgarte_factor: float = BAUMGARTE_FACTOR
    slop: float = SLOP
    warm_start_factor: float = WARM_START_FACTOR
    max_correction_velocity: float = MAX_CORRECTION_VELOCITY
    relaxation_factor: float = RELAXATION_FACTOR
    solver_type: SolverType = SolverType.SEQUENTIAL_IMPULSE
    warm_start_mode: WarmStartMode = WarmStartMode.NORMAL
    enable_sleeping: bool = True
    sleep_velocity_threshold: float = 0.05
    sleep_angular_threshold: float = 0.05
    sleep_time_threshold: float = 0.5
    substeps: int = 1
    max_linear_velocity: float = MAX_LINEAR_VELOCITY
    max_angular_velocity: float = MAX_ANGULAR_VELOCITY
    contact_bias: float = CONTACT_BIAS_VELOCITY
    use_block_solver: bool = True
    friction_iterations: int = 2
    enable_gyroscopic_torque: bool = True

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.velocity_iterations < 1:
            raise ValueError("velocity_iterations must be at least 1")
        if self.position_iterations < 0:
            raise ValueError("position_iterations cannot be negative")
        if not 0.0 <= self.baumgarte_factor <= 1.0:
            raise ValueError("baumgarte_factor must be between 0 and 1")
        if self.slop < 0:
            raise ValueError("slop cannot be negative")
        if not 0.0 <= self.warm_start_factor <= 1.0:
            raise ValueError("warm_start_factor must be between 0 and 1")
        if self.max_correction_velocity <= 0:
            raise ValueError("max_correction_velocity must be positive")
        if self.relaxation_factor <= 0 or self.relaxation_factor > 2:
            raise ValueError("relaxation_factor should be in (0, 2]")
        if self.substeps < 1:
            raise ValueError("substeps must be at least 1")

    @classmethod
    def default(cls) -> "SolverConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def high_quality(cls) -> "SolverConfig":
        """Create high quality configuration for accurate simulations."""
        return cls(
            velocity_iterations=16,
            position_iterations=6,
            substeps=2,
            relaxation_factor=1.0,
            warm_start_factor=0.9,
            use_block_solver=True,
            friction_iterations=4,
        )

    @classmethod
    def performance(cls) -> "SolverConfig":
        """Create performance-focused configuration."""
        return cls(
            velocity_iterations=4,
            position_iterations=2,
            substeps=1,
            relaxation_factor=1.2,
            warm_start_factor=0.7,
            use_block_solver=False,
            friction_iterations=1,
        )

    @classmethod
    def xpbd_default(cls) -> "SolverConfig":
        """Create default XPBD solver configuration."""
        return cls(
            solver_type=SolverType.XPBD,
            velocity_iterations=1,
            position_iterations=10,
            substeps=4,
            baumgarte_factor=0.0,  # XPBD doesn't use Baumgarte
            warm_start_factor=0.0,  # XPBD typically doesn't warm start
        )

    @classmethod
    def tgs_default(cls) -> "SolverConfig":
        """Create default TGS solver configuration."""
        return cls(
            solver_type=SolverType.TEMPORAL_GAUSS_SEIDEL,
            velocity_iterations=8,
            position_iterations=4,
            substeps=2,
            relaxation_factor=1.0,
            warm_start_factor=0.85,
        )

    def with_iterations(
        self,
        velocity: Optional[int] = None,
        position: Optional[int] = None
    ) -> "SolverConfig":
        """Return a copy with modified iteration counts."""
        return SolverConfig(
            velocity_iterations=velocity if velocity is not None else self.velocity_iterations,
            position_iterations=position if position is not None else self.position_iterations,
            baumgarte_factor=self.baumgarte_factor,
            slop=self.slop,
            warm_start_factor=self.warm_start_factor,
            max_correction_velocity=self.max_correction_velocity,
            relaxation_factor=self.relaxation_factor,
            solver_type=self.solver_type,
            warm_start_mode=self.warm_start_mode,
            enable_sleeping=self.enable_sleeping,
            sleep_velocity_threshold=self.sleep_velocity_threshold,
            sleep_angular_threshold=self.sleep_angular_threshold,
            sleep_time_threshold=self.sleep_time_threshold,
            substeps=self.substeps,
            max_linear_velocity=self.max_linear_velocity,
            max_angular_velocity=self.max_angular_velocity,
            contact_bias=self.contact_bias,
            use_block_solver=self.use_block_solver,
            friction_iterations=self.friction_iterations,
            enable_gyroscopic_torque=self.enable_gyroscopic_torque,
        )

    def with_substeps(self, substeps: int) -> "SolverConfig":
        """Return a copy with modified substep count."""
        config = SolverConfig(
            velocity_iterations=self.velocity_iterations,
            position_iterations=self.position_iterations,
            baumgarte_factor=self.baumgarte_factor,
            slop=self.slop,
            warm_start_factor=self.warm_start_factor,
            max_correction_velocity=self.max_correction_velocity,
            relaxation_factor=self.relaxation_factor,
            solver_type=self.solver_type,
            warm_start_mode=self.warm_start_mode,
            enable_sleeping=self.enable_sleeping,
            sleep_velocity_threshold=self.sleep_velocity_threshold,
            sleep_angular_threshold=self.sleep_angular_threshold,
            sleep_time_threshold=self.sleep_time_threshold,
            substeps=substeps,
            max_linear_velocity=self.max_linear_velocity,
            max_angular_velocity=self.max_angular_velocity,
            contact_bias=self.contact_bias,
            use_block_solver=self.use_block_solver,
            friction_iterations=self.friction_iterations,
            enable_gyroscopic_torque=self.enable_gyroscopic_torque,
        )
        return config
