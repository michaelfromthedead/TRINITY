"""IK system configuration constants.

All magic numbers for the IK subsystem are centralized here for
easy tuning and maintenance. These values have been carefully chosen
based on common animation requirements and numerical stability needs.
"""

from __future__ import annotations

# ============================================================================
# Solver Tolerances
# ============================================================================

# Default convergence tolerance for position-based solvers
IK_DEFAULT_TOLERANCE: float = 0.001

# Tight tolerance for high-precision requirements
IK_TOLERANCE_TIGHT: float = 0.0001

# Loose tolerance for performance-critical scenarios
IK_TOLERANCE_LOOSE: float = 0.01

# Angular tolerance for rotation goals (radians, ~0.57 degrees)
IK_ROTATION_TOLERANCE: float = 0.01

# ============================================================================
# Iteration Limits
# ============================================================================

# Two-bone IK doesn't iterate (analytical solution)
# These are for iterative solvers

# FABRIK default iterations - good balance of quality and performance
FABRIK_DEFAULT_MAX_ITERATIONS: int = 10

# CCD default iterations - usually converges quickly
CCD_DEFAULT_MAX_ITERATIONS: int = 10

# Jacobian default iterations - may need more for complex poses
JACOBIAN_DEFAULT_MAX_ITERATIONS: int = 50

# Full body IK iterations
FULLBODY_DEFAULT_MAX_ITERATIONS: int = 10

# Multi-chain FABRIK outer loop iterations
FABRIK_MULTI_CHAIN_MAX_ITERATIONS: int = 20

# ============================================================================
# Damping Factors
# ============================================================================

# Damped Least Squares (DLS) damping factor
# Higher = more stable but slower convergence
JACOBIAN_DLS_DAMPING: float = 0.5

# CCD damping factor for smooth motion
CCD_DEFAULT_DAMPING: float = 1.0

# ============================================================================
# Soft IK Parameters
# ============================================================================

# Soft IK ratio - when to start softening (fraction of max reach)
SOFT_IK_DEFAULT_RATIO: float = 0.9

# Soft IK blend - how much soft IK affects final position
SOFT_IK_DEFAULT_BLEND: float = 0.5

# Soft IK falloff rate constant
SOFT_IK_FALLOFF_RATE: float = 2.0

# ============================================================================
# Foot Placement Parameters
# ============================================================================

# Maximum raycast distance for ground detection
FOOT_PLACEMENT_RAY_LENGTH: float = 2.0

# Default foot height above ground
FOOT_PLACEMENT_FOOT_HEIGHT: float = 0.1

# Blend speed for smooth foot transitions (units per second)
FOOT_PLACEMENT_BLEND_SPEED: float = 10.0

# Maximum pelvis can drop to reach feet
FOOT_PLACEMENT_MAX_PELVIS_DROP: float = 0.5

# Maximum pelvis can raise for elevated feet
FOOT_PLACEMENT_MAX_PELVIS_RAISE: float = 0.2

# Weight for toe alignment to terrain
FOOT_PLACEMENT_TOE_ALIGN_WEIGHT: float = 0.5

# Leg reach safety margin (fraction of max reach)
FOOT_PLACEMENT_REACH_SAFETY_MARGIN: float = 0.95

# Multi-leg pelvis drop cap
MULTI_LEG_MAX_PELVIS_DROP: float = 0.5

# ============================================================================
# Joint Limits (radians)
# ============================================================================

# Default minimum bend angle for two-bone (prevents hyperextension)
JOINT_MIN_BEND_ANGLE: float = 0.1

# Default maximum bend angle for two-bone
JOINT_MAX_BEND_ANGLE: float = 3.04159  # pi - 0.1

# Default cone angle for ball-socket joints (90 degrees)
JOINT_DEFAULT_CONE_ANGLE: float = 1.5708  # pi/2

# Maximum look-at angle (90 degrees)
LOOK_AT_MAX_ANGLE: float = 1.5708  # pi/2

# ============================================================================
# Look-At Solver Weights
# ============================================================================

# Default weight distribution for look-at
LOOK_AT_HEAD_WEIGHT: float = 0.6
LOOK_AT_NECK_WEIGHT: float = 0.3
LOOK_AT_SPINE_WEIGHT: float = 0.1

# ============================================================================
# Full Body IK Parameters
# ============================================================================

# Spine stiffness (0 = flexible, 1 = rigid)
FULLBODY_SPINE_STIFFNESS: float = 0.5

# ============================================================================
# Goal Blending
# ============================================================================

# Default blend speed for goal transitions
GOAL_BLENDER_DEFAULT_SPEED: float = 10.0

# ============================================================================
# Step Sizes
# ============================================================================

# Jacobian step size multiplier
JACOBIAN_DEFAULT_STEP_SIZE: float = 1.0

# ============================================================================
# Numerical Safety
# ============================================================================

# Minimum bone length to consider valid (prevents division by zero)
MIN_BONE_LENGTH: float = 1e-6

# Small epsilon for polygon edge length checks
POLYGON_EDGE_MIN_LENGTH: float = 1e-6
