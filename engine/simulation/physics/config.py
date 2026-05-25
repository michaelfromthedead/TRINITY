"""
Physics Configuration Module

Defines all physics engine constants and configuration parameters.
These values control simulation accuracy, performance, and behavior.
"""

from dataclasses import dataclass, field
from typing import Tuple
from enum import Enum, auto


class PhysicsBackend(Enum):
    """Available physics computation backends."""
    PYTHON = auto()
    NUMPY = auto()
    CYTHON = auto()
    NATIVE = auto()


class BroadphaseType(Enum):
    """Broadphase collision detection algorithms."""
    BRUTE_FORCE = auto()
    SPATIAL_HASH = auto()
    BVH = auto()
    OCTREE = auto()
    SWEEP_AND_PRUNE = auto()


class NarrowphaseType(Enum):
    """Narrowphase collision detection algorithms."""
    GJK_EPA = auto()
    SAT = auto()
    MPR = auto()


class SolverType(Enum):
    """Constraint solver algorithms."""
    SEQUENTIAL_IMPULSE = auto()
    PROJECTED_GAUSS_SEIDEL = auto()
    JACOBI = auto()


# =============================================================================
# Default Physics Constants
# =============================================================================

# Gravity vector in world space (m/s^2)
DEFAULT_GRAVITY: Tuple[float, float, float] = (0.0, -9.81, 0.0)

# Default simulation timestep (seconds) - 60 Hz
DEFAULT_TIMESTEP: float = 1.0 / 60.0

# Minimum allowed timestep (seconds) - prevents numerical instability
MIN_TIMESTEP: float = 1e-6

# Maximum number of physics substeps per frame
MAX_SUBSTEPS: int = 8

# Sleep thresholds for bodies
SLEEP_THRESHOLD_LINEAR: float = 0.01  # m/s
SLEEP_THRESHOLD_ANGULAR: float = 0.01  # rad/s
SLEEP_TIME_THRESHOLD: float = 0.5  # seconds

# Maximum number of physics bodies in simulation
MAX_BODIES: int = 65536

# Constraint solver iterations
SOLVER_ITERATIONS: int = 10
POSITION_ITERATIONS: int = 4

# Velocity solver iterations for stability
VELOCITY_ITERATIONS: int = 8

# Contact constraints
MAX_CONTACTS_PER_PAIR: int = 4
CONTACT_CACHE_SIZE: int = 32768
CONTACT_PENETRATION_TOLERANCE: float = 0.005  # 5mm
CONTACT_BAUMGARTE_FACTOR: float = 0.2
CONTACT_SLOP: float = 0.001  # 1mm penetration allowed

# Continuous Collision Detection (CCD)
CCD_MOTION_THRESHOLD: float = 0.1  # m
CCD_SWEPT_SPHERE_RADIUS: float = 0.01  # m
CCD_MAX_ITERATIONS: int = 32

# Joint constraints
MAX_JOINTS: int = 16384
JOINT_ERROR_TOLERANCE: float = 0.001
JOINT_WARMSTARTING_FACTOR: float = 0.85

# Island management
MIN_ISLAND_SIZE: int = 1
MAX_ISLAND_SIZE: int = 512
ISLAND_MERGE_THRESHOLD: int = 64

# Collision detection
COLLISION_EPSILON: float = 1e-6
AABB_EXTENSION: float = 0.1  # m - adds margin to AABBs
GJK_MAX_ITERATIONS: int = 64
EPA_MAX_ITERATIONS: int = 64
EPA_TOLERANCE: float = 1e-4

# Mass properties
MIN_MASS: float = 1e-6  # kg
MAX_MASS: float = 1e12  # kg
DEFAULT_DENSITY: float = 1000.0  # kg/m^3 (water density)
MIN_INERTIA: float = 1e-6

# Damping
DEFAULT_LINEAR_DAMPING: float = 0.01
DEFAULT_ANGULAR_DAMPING: float = 0.05
MAX_LINEAR_VELOCITY: float = 500.0  # m/s
MAX_ANGULAR_VELOCITY: float = 100.0  # rad/s

# Layer/group filtering
MAX_COLLISION_LAYERS: int = 32
DEFAULT_COLLISION_MASK: int = 0xFFFFFFFF

# Shape margins and thresholds
DEFAULT_SHAPE_MARGIN: float = 0.04  # Default collision margin in meters
MIN_SHAPE_RADIUS: float = 0.001  # Minimum radius for shapes in meters
MIN_SHAPE_DIMENSION: float = 0.001  # Minimum dimension for boxes/cylinders

# Convex hull settings
MIN_CONVEX_HULL_POINTS: int = 4  # Minimum points for convex hull
CONVEX_HULL_FILL_RATIO: float = 0.6  # Approximate fill ratio for mass estimation

# Box overlap test expansion factor
BOX_OVERLAP_EXPANSION: float = 1.8  # Conservative AABB expansion for OBB tests

# Default query distance
DEFAULT_RAYCAST_DISTANCE: float = 1000.0  # Default max raycast distance in meters

# Numerical tolerances for comparisons
FLOAT_COMPARISON_EPSILON: float = 1e-10  # For near-zero comparisons


@dataclass
class PhysicsConfig:
    """
    Complete physics simulation configuration.

    Attributes:
        gravity: World gravity vector (x, y, z) in m/s^2
        timestep: Fixed simulation timestep in seconds
        max_substeps: Maximum physics steps per frame
        solver_iterations: Velocity solver iterations
        position_iterations: Position correction iterations
        velocity_iterations: Velocity correction iterations
        max_bodies: Maximum simultaneous bodies
        max_joints: Maximum simultaneous joints
        sleep_threshold_linear: Linear velocity for sleep (m/s)
        sleep_threshold_angular: Angular velocity for sleep (rad/s)
        sleep_time_threshold: Time below threshold before sleep (s)
        enable_sleeping: Whether bodies can sleep
        enable_ccd: Global CCD enable
        broadphase_type: Broadphase algorithm
        narrowphase_type: Narrowphase algorithm
        solver_type: Constraint solver algorithm
        contact_baumgarte: Baumgarte stabilization factor
        contact_slop: Allowed penetration depth
        warmstarting: Enable constraint warmstarting
        warmstarting_factor: Warmstarting blend factor
        linear_damping: Default linear damping
        angular_damping: Default angular damping
        max_linear_velocity: Velocity clamp
        max_angular_velocity: Angular velocity clamp
        collision_epsilon: Numerical tolerance
        aabb_extension: AABB margin
        backend: Physics computation backend
    """

    # Core simulation
    gravity: Tuple[float, float, float] = DEFAULT_GRAVITY
    timestep: float = DEFAULT_TIMESTEP
    max_substeps: int = MAX_SUBSTEPS

    # Solver
    solver_iterations: int = SOLVER_ITERATIONS
    position_iterations: int = POSITION_ITERATIONS
    velocity_iterations: int = VELOCITY_ITERATIONS
    solver_type: SolverType = SolverType.SEQUENTIAL_IMPULSE

    # Capacity
    max_bodies: int = MAX_BODIES
    max_joints: int = MAX_JOINTS

    # Sleeping
    sleep_threshold_linear: float = SLEEP_THRESHOLD_LINEAR
    sleep_threshold_angular: float = SLEEP_THRESHOLD_ANGULAR
    sleep_time_threshold: float = SLEEP_TIME_THRESHOLD
    enable_sleeping: bool = True

    # CCD
    enable_ccd: bool = True
    ccd_motion_threshold: float = CCD_MOTION_THRESHOLD

    # Broadphase/Narrowphase
    broadphase_type: BroadphaseType = BroadphaseType.BVH
    narrowphase_type: NarrowphaseType = NarrowphaseType.GJK_EPA

    # Contact solver
    contact_baumgarte: float = CONTACT_BAUMGARTE_FACTOR
    contact_slop: float = CONTACT_SLOP
    max_contacts_per_pair: int = MAX_CONTACTS_PER_PAIR

    # Warmstarting
    warmstarting: bool = True
    warmstarting_factor: float = JOINT_WARMSTARTING_FACTOR

    # Damping
    linear_damping: float = DEFAULT_LINEAR_DAMPING
    angular_damping: float = DEFAULT_ANGULAR_DAMPING
    max_linear_velocity: float = MAX_LINEAR_VELOCITY
    max_angular_velocity: float = MAX_ANGULAR_VELOCITY

    # Numerical
    collision_epsilon: float = COLLISION_EPSILON
    aabb_extension: float = AABB_EXTENSION

    # Backend
    backend: PhysicsBackend = PhysicsBackend.NUMPY

    # Island management
    enable_islands: bool = True
    min_island_size: int = MIN_ISLAND_SIZE
    max_island_size: int = MAX_ISLAND_SIZE

    def validate(self) -> bool:
        """
        Validate configuration values are within acceptable ranges.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If any configuration value is invalid
        """
        if self.timestep <= 0:
            raise ValueError(f"Timestep must be positive, got {self.timestep}")

        if self.max_substeps < 1:
            raise ValueError(f"Max substeps must be >= 1, got {self.max_substeps}")

        if self.solver_iterations < 1:
            raise ValueError(f"Solver iterations must be >= 1, got {self.solver_iterations}")

        if self.position_iterations < 1:
            raise ValueError(f"Position iterations must be >= 1, got {self.position_iterations}")

        if self.max_bodies < 1:
            raise ValueError(f"Max bodies must be >= 1, got {self.max_bodies}")

        if self.sleep_threshold_linear < 0:
            raise ValueError(f"Sleep threshold linear must be >= 0, got {self.sleep_threshold_linear}")

        if self.sleep_threshold_angular < 0:
            raise ValueError(f"Sleep threshold angular must be >= 0, got {self.sleep_threshold_angular}")

        if self.sleep_time_threshold < 0:
            raise ValueError(f"Sleep time threshold must be >= 0, got {self.sleep_time_threshold}")

        if not (0 <= self.contact_baumgarte <= 1):
            raise ValueError(f"Contact baumgarte must be in [0, 1], got {self.contact_baumgarte}")

        if not (0 <= self.warmstarting_factor <= 1):
            raise ValueError(f"Warmstarting factor must be in [0, 1], got {self.warmstarting_factor}")

        if self.linear_damping < 0:
            raise ValueError(f"Linear damping must be >= 0, got {self.linear_damping}")

        if self.angular_damping < 0:
            raise ValueError(f"Angular damping must be >= 0, got {self.angular_damping}")

        if self.max_linear_velocity <= 0:
            raise ValueError(f"Max linear velocity must be > 0, got {self.max_linear_velocity}")

        if self.max_angular_velocity <= 0:
            raise ValueError(f"Max angular velocity must be > 0, got {self.max_angular_velocity}")

        return True

    def copy(self) -> 'PhysicsConfig':
        """Create a deep copy of this configuration."""
        return PhysicsConfig(
            gravity=self.gravity,
            timestep=self.timestep,
            max_substeps=self.max_substeps,
            solver_iterations=self.solver_iterations,
            position_iterations=self.position_iterations,
            velocity_iterations=self.velocity_iterations,
            solver_type=self.solver_type,
            max_bodies=self.max_bodies,
            max_joints=self.max_joints,
            sleep_threshold_linear=self.sleep_threshold_linear,
            sleep_threshold_angular=self.sleep_threshold_angular,
            sleep_time_threshold=self.sleep_time_threshold,
            enable_sleeping=self.enable_sleeping,
            enable_ccd=self.enable_ccd,
            ccd_motion_threshold=self.ccd_motion_threshold,
            broadphase_type=self.broadphase_type,
            narrowphase_type=self.narrowphase_type,
            contact_baumgarte=self.contact_baumgarte,
            contact_slop=self.contact_slop,
            max_contacts_per_pair=self.max_contacts_per_pair,
            warmstarting=self.warmstarting,
            warmstarting_factor=self.warmstarting_factor,
            linear_damping=self.linear_damping,
            angular_damping=self.angular_damping,
            max_linear_velocity=self.max_linear_velocity,
            max_angular_velocity=self.max_angular_velocity,
            collision_epsilon=self.collision_epsilon,
            aabb_extension=self.aabb_extension,
            backend=self.backend,
            enable_islands=self.enable_islands,
            min_island_size=self.min_island_size,
            max_island_size=self.max_island_size,
        )


# Preset configurations for common use cases
PRESET_HIGH_QUALITY = PhysicsConfig(
    solver_iterations=20,
    position_iterations=8,
    velocity_iterations=16,
    enable_ccd=True,
    max_substeps=16,
)

PRESET_PERFORMANCE = PhysicsConfig(
    solver_iterations=4,
    position_iterations=2,
    velocity_iterations=4,
    enable_ccd=False,
    max_substeps=4,
    enable_sleeping=True,
)

PRESET_MOBILE = PhysicsConfig(
    solver_iterations=2,
    position_iterations=1,
    velocity_iterations=2,
    enable_ccd=False,
    max_substeps=2,
    enable_sleeping=True,
    max_bodies=1024,
)

PRESET_DETERMINISTIC = PhysicsConfig(
    solver_iterations=10,
    position_iterations=4,
    velocity_iterations=8,
    enable_ccd=True,
    max_substeps=1,
    timestep=1.0 / 120.0,  # Fixed 120 Hz
    enable_sleeping=False,  # Sleeping can introduce non-determinism
)
