"""Position Based Fluids (PBF) solver.

This module implements position-based fluids including:
- Position prediction
- Density constraint solving
- Lambda (Lagrange multiplier) computation
- Position correction
- Vorticity confinement
- XSPH viscosity
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import (
    PARTICLE_RADIUS,
    SMOOTHING_LENGTH,
    REST_DENSITY,
    MAX_PARTICLES,
    MAX_NEIGHBORS,
    GRID_CELL_SIZE,
    FLUID_SUBSTEPS,
    PBF_ITERATIONS,
    PBF_LAMBDA_EPSILON,
    PBF_TENSILE_K,
    PBF_TENSILE_N,
    PBF_DELTA_Q_RATIO,
    FluidMaterial,
    PBFConfig,
)
from .sph import SPHKernels, SpatialHashGrid


# =============================================================================
# Type Aliases
# =============================================================================

Vector3 = NDArray[np.float64]


# =============================================================================
# PBF Particle
# =============================================================================

@dataclass
class PBFParticle:
    """PBF fluid particle.

    Attributes:
        position: Current position
        predicted: Predicted position (after external forces)
        velocity: Velocity vector
        mass: Particle mass
        lambda_: Lagrange multiplier
        delta_p: Position correction
        neighbors: Indices of neighboring particles
        omega: Vorticity vector (for confinement)
    """
    position: Vector3 = field(default_factory=lambda: np.zeros(3))
    predicted: Vector3 = field(default_factory=lambda: np.zeros(3))
    velocity: Vector3 = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    lambda_: float = 0.0
    delta_p: Vector3 = field(default_factory=lambda: np.zeros(3))
    neighbors: List[int] = field(default_factory=list)
    omega: Vector3 = field(default_factory=lambda: np.zeros(3))


# =============================================================================
# PBF Solver
# =============================================================================

class PBFSolver:
    """Position Based Fluids solver.

    Implements the PBF algorithm from Macklin & Muller (2013):
    1. Predict positions with external forces
    2. Find neighbors
    3. Iteratively solve density constraints
    4. Update velocities from position changes
    5. Apply vorticity confinement
    6. Apply XSPH viscosity

    Attributes:
        particles: List of fluid particles
        material: Fluid material properties
        config: PBF configuration
        smoothing_length: Kernel smoothing length (h)
        grid: Spatial hash grid for neighbor search
        gravity: Gravity acceleration
        bounds_min/max: Domain boundaries
    """

    def __init__(
        self,
        material: Optional[FluidMaterial] = None,
        config: Optional[PBFConfig] = None,
        smoothing_length: float = SMOOTHING_LENGTH,
        gravity: Optional[Vector3] = None,
        bounds_min: Optional[Vector3] = None,
        bounds_max: Optional[Vector3] = None
    ):
        """Initialize PBF solver.

        Args:
            material: Fluid material properties
            config: PBF-specific configuration
            smoothing_length: Kernel smoothing length
            gravity: Gravity acceleration vector
            bounds_min/max: Domain boundaries
        """
        self.material = material or FluidMaterial.water()
        self.config = config or PBFConfig()
        self.smoothing_length = smoothing_length

        self.particles: List[PBFParticle] = []
        self.grid = SpatialHashGrid(smoothing_length)

        self.gravity = gravity if gravity is not None else np.array([0.0, -9.81, 0.0])
        self.bounds_min = bounds_min if bounds_min is not None else np.array([-5.0, -5.0, -5.0])
        self.bounds_max = bounds_max if bounds_max is not None else np.array([5.0, 5.0, 5.0])

        # Precompute mass and constants
        self._compute_constants()

    def _compute_constants(self) -> None:
        """Compute solver constants."""
        h = self.smoothing_length
        spacing = h * 0.5
        self.particle_mass = self.material.rest_density * spacing ** 3

        # Relaxation parameter for constraint solving
        # Using config value instead of hardcoded magic number
        self.epsilon = PBF_LAMBDA_EPSILON

        # Tensile instability correction constants from config
        self.k_tensile = PBF_TENSILE_K
        self.n_tensile = PBF_TENSILE_N
        delta_q = PBF_DELTA_Q_RATIO * h
        self.w_delta_q = SPHKernels.poly6(delta_q * delta_q, h)

    def add_particle(
        self,
        position: Vector3,
        velocity: Optional[Vector3] = None
    ) -> int:
        """Add a particle to the simulation.

        Args:
            position: Initial position
            velocity: Initial velocity

        Returns:
            Index of new particle
        """
        particle = PBFParticle(
            position=position.copy(),
            predicted=position.copy(),
            velocity=velocity.copy() if velocity is not None else np.zeros(3),
            mass=self.particle_mass
        )
        self.particles.append(particle)
        return len(self.particles) - 1

    def add_block(
        self,
        min_corner: Vector3,
        max_corner: Vector3,
        spacing: Optional[float] = None
    ) -> List[int]:
        """Add a block of particles."""
        spacing = spacing or self.smoothing_length * 0.5
        indices = []

        x = min_corner[0]
        while x <= max_corner[0]:
            y = min_corner[1]
            while y <= max_corner[1]:
                z = min_corner[2]
                while z <= max_corner[2]:
                    if len(self.particles) < MAX_PARTICLES:
                        idx = self.add_particle(np.array([x, y, z]))
                        indices.append(idx)
                    z += spacing
                y += spacing
            x += spacing

        return indices

    def predict_positions(self, dt: float) -> None:
        """Predict positions using external forces.

        x* = x + v*dt + a*dt^2

        Args:
            dt: Timestep
        """
        for p in self.particles:
            # Apply gravity
            p.velocity += self.gravity * dt
            p.predicted = p.position + p.velocity * dt

    def _build_grid(self) -> None:
        """Build spatial hash grid from predicted positions."""
        self.grid.clear()
        for i, p in enumerate(self.particles):
            self.grid.insert(i, p.predicted)

    def _find_neighbors(self) -> None:
        """Find neighbors for each particle."""
        h = self.smoothing_length

        for i, p in enumerate(self.particles):
            candidates = self.grid.get_neighbors(p.predicted, h)
            p.neighbors = []

            for j in candidates:
                if i == j:
                    continue

                dist_sq = np.sum((p.predicted - self.particles[j].predicted) ** 2)
                if dist_sq < h * h:
                    p.neighbors.append(j)

                if len(p.neighbors) >= MAX_NEIGHBORS:
                    break

    def compute_density_constraint(self, particle_index: int) -> float:
        """Compute density constraint for a particle.

        C_i = rho_i / rho_0 - 1

        Args:
            particle_index: Particle index

        Returns:
            Constraint value (0 when satisfied)
        """
        p = self.particles[particle_index]
        h = self.smoothing_length

        # Compute density
        density = p.mass * SPHKernels.poly6(0.0, h)  # Self contribution

        for j in p.neighbors:
            pj = self.particles[j]
            r_sq = np.sum((p.predicted - pj.predicted) ** 2)
            density += pj.mass * SPHKernels.poly6(r_sq, h)

        # Constraint
        return density / self.material.rest_density - 1.0

    def compute_lambda(self, particle_index: int) -> float:
        """Compute Lagrange multiplier lambda for a particle.

        lambda = -C_i / (sum(|grad_pk C_i|^2) + epsilon)

        Handles edge cases:
        - Division by zero when grad_sum is very small
        - Large constraint values that could cause instability

        Args:
            particle_index: Particle index

        Returns:
            Lambda value (clamped to prevent instability)
        """
        p = self.particles[particle_index]
        h = self.smoothing_length
        rho_0 = self.material.rest_density

        # Compute constraint
        C = self.compute_density_constraint(particle_index)

        # If constraint is essentially satisfied, skip computation
        if abs(C) < 1e-10:
            return 0.0

        # Compute gradient sum
        grad_sum = 0.0
        grad_i = np.zeros(3)  # Gradient w.r.t. this particle

        for j in p.neighbors:
            pj = self.particles[j]
            r = p.predicted - pj.predicted
            dist = np.linalg.norm(r)

            if dist < 1e-10:
                continue

            # Gradient w.r.t. neighbor
            grad_j = SPHKernels.spiky_gradient(r, dist, h) / rho_0
            grad_sum += np.dot(grad_j, grad_j)

            # Accumulate for self gradient
            grad_i -= grad_j

        # Self gradient contribution
        grad_sum += np.dot(grad_i, grad_i)

        # Prevent division by zero - use scaled epsilon based on constraint magnitude
        # This prevents instability when both C and grad_sum are very small
        denominator = grad_sum + self.epsilon

        # Additional safeguard: if gradient sum is too small relative to constraint,
        # limit the correction to prevent explosion
        if denominator < self.epsilon * 10:
            # Degenerate case - very few neighbors or nearly uniform distribution
            # Return a conservative lambda to avoid instability
            return -np.sign(C) * min(abs(C), 0.1)

        lambda_val = -C / denominator

        # Clamp lambda to prevent excessive corrections
        max_lambda = 1000.0  # Maximum correction factor
        return np.clip(lambda_val, -max_lambda, max_lambda)

    def compute_position_correction(self, particle_index: int) -> Vector3:
        """Compute position correction for a particle.

        delta_p_i = (1/rho_0) * sum_j (lambda_i + lambda_j + s_corr) * grad W

        Args:
            particle_index: Particle index

        Returns:
            Position correction vector
        """
        p = self.particles[particle_index]
        h = self.smoothing_length
        rho_0 = self.material.rest_density

        delta_p = np.zeros(3)

        for j in p.neighbors:
            pj = self.particles[j]
            r = p.predicted - pj.predicted
            dist = np.linalg.norm(r)

            if dist < 1e-10:
                continue

            # Tensile instability correction
            r_sq = dist * dist
            w = SPHKernels.poly6(r_sq, h)
            s_corr = 0.0
            if self.w_delta_q > 1e-10:
                s_corr = -self.k_tensile * (w / self.w_delta_q) ** self.n_tensile

            # Position correction
            grad = SPHKernels.spiky_gradient(r, dist, h)
            delta_p += (p.lambda_ + pj.lambda_ + s_corr) * grad

        return delta_p / rho_0

    def solve_constraints(self, iterations: int) -> None:
        """Iteratively solve density constraints.

        Each iteration:
        1. Rebuild neighbor structure (positions changed)
        2. Compute lambda for each particle
        3. Compute and apply position corrections

        Args:
            iterations: Number of solver iterations
        """
        for _ in range(iterations):
            # Rebuild grid and neighbors since positions changed
            self._build_grid()
            self._find_neighbors()

            # Compute all lambdas
            for i, p in enumerate(self.particles):
                p.lambda_ = self.compute_lambda(i)

            # Compute all position corrections
            for i, p in enumerate(self.particles):
                p.delta_p = self.compute_position_correction(i)

            # Apply corrections
            for p in self.particles:
                p.predicted += p.delta_p

            # Enforce boundaries during solving
            self._apply_boundaries_predicted()

    def _apply_boundaries_predicted(self) -> None:
        """Apply boundary conditions to predicted positions."""
        margin = PARTICLE_RADIUS

        for p in self.particles:
            for dim in range(3):
                if p.predicted[dim] < self.bounds_min[dim] + margin:
                    p.predicted[dim] = self.bounds_min[dim] + margin
                elif p.predicted[dim] > self.bounds_max[dim] - margin:
                    p.predicted[dim] = self.bounds_max[dim] - margin

    def update_velocities(self, dt: float) -> None:
        """Update velocities from position changes.

        v = (x* - x) / dt

        Args:
            dt: Timestep
        """
        inv_dt = 1.0 / dt if dt > 1e-10 else 0.0

        for p in self.particles:
            p.velocity = (p.predicted - p.position) * inv_dt

    def apply_vorticity_confinement(self, dt: float) -> None:
        """Apply vorticity confinement force.

        Restores energy lost to numerical dissipation.

        Args:
            dt: Timestep
        """
        strength = self.config.vorticity_strength
        if strength < 1e-10:
            return

        h = self.smoothing_length

        # Compute vorticity omega = curl(v)
        for i, p in enumerate(self.particles):
            omega = np.zeros(3)

            for j in p.neighbors:
                pj = self.particles[j]
                r = p.predicted - pj.predicted
                dist = np.linalg.norm(r)

                if dist < 1e-10:
                    continue

                v_diff = pj.velocity - p.velocity
                grad = SPHKernels.spiky_gradient(r, dist, h)

                omega += np.cross(v_diff, grad)

            p.omega = omega

        # Apply vorticity force
        for i, p in enumerate(self.particles):
            # Compute gradient of |omega|
            eta = np.zeros(3)

            for j in p.neighbors:
                pj = self.particles[j]
                r = p.predicted - pj.predicted
                dist = np.linalg.norm(r)

                if dist < 1e-10:
                    continue

                omega_diff = np.linalg.norm(pj.omega) - np.linalg.norm(p.omega)
                grad = SPHKernels.spiky_gradient(r, dist, h)

                eta += omega_diff * grad / (dist + 1e-10)

            eta_norm = np.linalg.norm(eta)
            if eta_norm > 1e-10:
                N = eta / eta_norm
                f_vorticity = strength * np.cross(N, p.omega)
                p.velocity += f_vorticity * dt

    def apply_xsph_viscosity(self) -> None:
        """Apply XSPH viscosity for coherent motion.

        v_new = v + c * sum_j (v_j - v_i) * W
        """
        c = self.config.xsph_viscosity
        if c < 1e-10:
            return

        h = self.smoothing_length

        # Store corrections
        velocity_corrections = []

        for i, p in enumerate(self.particles):
            correction = np.zeros(3)

            for j in p.neighbors:
                pj = self.particles[j]
                r_sq = np.sum((p.predicted - pj.predicted) ** 2)
                w = SPHKernels.poly6(r_sq, h)

                # Average density for weighting
                avg_density = 0.5 * (p.mass / self.material.rest_density +
                                     pj.mass / self.material.rest_density)

                correction += (pj.velocity - p.velocity) * w / avg_density

            velocity_corrections.append(c * correction)

        # Apply corrections
        for p, corr in zip(self.particles, velocity_corrections):
            p.velocity += corr

    def finalize_positions(self) -> None:
        """Update positions from predicted values."""
        for p in self.particles:
            p.position = p.predicted.copy()

    def step(self, dt: float, substeps: int = FLUID_SUBSTEPS) -> None:
        """Advance simulation by one timestep.

        Args:
            dt: Total timestep
            substeps: Number of substeps
        """
        sub_dt = dt / substeps

        for _ in range(substeps):
            self._substep(sub_dt)

    def _substep(self, dt: float) -> None:
        """Perform a single substep."""
        # 1. Predict positions
        self.predict_positions(dt)

        # 2. Build neighbor structure
        self._build_grid()
        self._find_neighbors()

        # 3. Solve constraints
        self.solve_constraints(self.config.iterations)

        # 4. Update velocities
        self.update_velocities(dt)

        # 5. Apply vorticity confinement
        self.apply_vorticity_confinement(dt)

        # 6. Apply XSPH viscosity
        self.apply_xsph_viscosity()

        # 7. Update positions
        self.finalize_positions()

    def get_positions(self) -> NDArray[np.float64]:
        """Get all particle positions."""
        return np.array([p.position for p in self.particles])

    def get_velocities(self) -> NDArray[np.float64]:
        """Get all particle velocities."""
        return np.array([p.velocity for p in self.particles])

    @property
    def num_particles(self) -> int:
        """Number of particles in simulation."""
        return len(self.particles)

    def compute_average_constraint_error(self) -> float:
        """Compute average density constraint error."""
        if not self.particles:
            return 0.0

        total_error = 0.0
        for i in range(len(self.particles)):
            C = self.compute_density_constraint(i)
            total_error += abs(C)

        return total_error / len(self.particles)
