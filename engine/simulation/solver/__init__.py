"""
Constraint Solver Module for AI Game Engine.

This module provides various constraint solvers for physics simulation:
- Sequential Impulse Solver (standard approach)
- Temporal Gauss-Seidel Solver (improved convergence)
- Extended Position Based Dynamics (XPBD) Solver (soft constraints)

Also includes:
- Jacobian computation utilities
- Island management for optimization
"""

from .config import (
    DEFAULT_VELOCITY_ITERATIONS,
    DEFAULT_POSITION_ITERATIONS,
    BAUMGARTE_FACTOR,
    SLOP,
    WARM_START_FACTOR,
    MAX_CORRECTION_VELOCITY,
    RELAXATION_FACTOR,
    SolverConfig,
)
from .constraint_solver import ConstraintSolver, Constraint
from .tgs_solver import TGSSolver
from .xpbd_solver import XPBDSolver, XPBDConstraint
from .jacobian import (
    Jacobian,
    compute_jacobian,
    compute_effective_mass,
    apply_impulse,
    compute_relative_velocity,
)
from .island_manager import Island, IslandManager

__all__ = [
    # Config
    "DEFAULT_VELOCITY_ITERATIONS",
    "DEFAULT_POSITION_ITERATIONS",
    "BAUMGARTE_FACTOR",
    "SLOP",
    "WARM_START_FACTOR",
    "MAX_CORRECTION_VELOCITY",
    "RELAXATION_FACTOR",
    "SolverConfig",
    # Solvers
    "ConstraintSolver",
    "Constraint",
    "TGSSolver",
    "XPBDSolver",
    "XPBDConstraint",
    # Jacobian
    "Jacobian",
    "compute_jacobian",
    "compute_effective_mass",
    "apply_impulse",
    "compute_relative_velocity",
    # Islands
    "Island",
    "IslandManager",
]
