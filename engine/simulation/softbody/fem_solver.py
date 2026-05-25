"""Finite Element Method solver for soft body simulation.

This module implements FEM-based deformation simulation including:
- Tetrahedral mesh representation
- Deformation gradient computation
- Strain energy computation
- Corotational FEM for large deformations
- Neo-Hookean material model
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import (
    DEFAULT_YOUNG_MODULUS,
    DEFAULT_POISSON_RATIO,
    FEM_TOLERANCE,
    FEM_MAX_ITERATIONS,
    MIN_TET_VOLUME,
    FEM_MIN_JACOBIAN,
    FEM_INVERSION_HANDLING,
    SoftBodyMaterial,
)


# =============================================================================
# Type Aliases
# =============================================================================

Vector3 = NDArray[np.float64]  # Shape: (3,)
Matrix3x3 = NDArray[np.float64]  # Shape: (3, 3)
Matrix4x4 = NDArray[np.float64]  # Shape: (4, 4)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TetrahedralMesh:
    """Tetrahedral mesh representation for FEM simulation.

    Attributes:
        vertices: Vertex positions, shape (N, 3)
        tetrahedra: Tetrahedron indices, shape (M, 4)
        rest_vertices: Rest pose vertex positions
        velocities: Vertex velocities, shape (N, 3)
        masses: Per-vertex masses, shape (N,)
        fixed: Boolean array marking fixed vertices
    """
    vertices: NDArray[np.float64]  # (N, 3)
    tetrahedra: NDArray[np.int32]  # (M, 4)
    rest_vertices: Optional[NDArray[np.float64]] = None
    velocities: Optional[NDArray[np.float64]] = None
    masses: Optional[NDArray[np.float64]] = None
    fixed: Optional[NDArray[np.bool_]] = None

    def __post_init__(self):
        """Initialize derived data."""
        n_vertices = len(self.vertices)

        if self.rest_vertices is None:
            self.rest_vertices = self.vertices.copy()

        if self.velocities is None:
            self.velocities = np.zeros((n_vertices, 3), dtype=np.float64)

        if self.masses is None:
            self.masses = np.ones(n_vertices, dtype=np.float64)

        if self.fixed is None:
            self.fixed = np.zeros(n_vertices, dtype=np.bool_)

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return len(self.vertices)

    @property
    def num_tetrahedra(self) -> int:
        """Number of tetrahedra in the mesh."""
        return len(self.tetrahedra)

    def get_tetrahedron_vertices(self, tet_index: int) -> NDArray[np.float64]:
        """Get the 4 vertex positions of a tetrahedron.

        Args:
            tet_index: Index of the tetrahedron

        Returns:
            Array of shape (4, 3) with vertex positions
        """
        indices = self.tetrahedra[tet_index]
        return self.vertices[indices]

    def get_rest_tetrahedron_vertices(self, tet_index: int) -> NDArray[np.float64]:
        """Get the 4 rest pose vertex positions of a tetrahedron."""
        indices = self.tetrahedra[tet_index]
        return self.rest_vertices[indices]

    def compute_volume(self, tet_index: int) -> float:
        """Compute signed volume of a tetrahedron.

        Args:
            tet_index: Index of the tetrahedron

        Returns:
            Signed volume (positive if correctly oriented)
        """
        verts = self.get_tetrahedron_vertices(tet_index)
        return compute_tetrahedron_volume(verts[0], verts[1], verts[2], verts[3])

    def compute_rest_volume(self, tet_index: int) -> float:
        """Compute rest pose volume of a tetrahedron."""
        verts = self.get_rest_tetrahedron_vertices(tet_index)
        return compute_tetrahedron_volume(verts[0], verts[1], verts[2], verts[3])

    def compute_total_volume(self) -> float:
        """Compute total volume of the mesh."""
        return sum(self.compute_volume(i) for i in range(self.num_tetrahedra))

    def compute_center_of_mass(self) -> Vector3:
        """Compute center of mass of the mesh."""
        total_mass = np.sum(self.masses)
        if total_mass < 1e-10:
            return np.mean(self.vertices, axis=0)
        return np.sum(self.vertices * self.masses[:, np.newaxis], axis=0) / total_mass

    def compute_mass_from_density(self, density: float) -> None:
        """Compute vertex masses from density and tetrahedron volumes.

        Args:
            density: Mass per unit volume
        """
        self.masses = np.zeros(self.num_vertices, dtype=np.float64)

        for i in range(self.num_tetrahedra):
            vol = abs(self.compute_rest_volume(i))
            mass_per_vertex = density * vol / 4.0
            for vi in self.tetrahedra[i]:
                self.masses[vi] += mass_per_vertex


@dataclass
class FEMElement:
    """Precomputed data for a single FEM element (tetrahedron).

    Attributes:
        tet_index: Index in the mesh
        rest_volume: Volume in rest pose
        inv_Dm: Inverse of the reference shape matrix
        B: Strain-displacement matrix
    """
    tet_index: int
    rest_volume: float
    inv_Dm: Matrix3x3
    B: Optional[NDArray[np.float64]] = None


# =============================================================================
# Material Models
# =============================================================================

class MaterialModel(ABC):
    """Abstract base class for hyperelastic material models."""

    @abstractmethod
    def compute_stress(
        self,
        F: Matrix3x3,
        lam: float,
        mu: float
    ) -> Matrix3x3:
        """Compute first Piola-Kirchhoff stress tensor.

        Args:
            F: Deformation gradient
            lam: First Lame parameter
            mu: Second Lame parameter (shear modulus)

        Returns:
            First Piola-Kirchhoff stress tensor P
        """
        pass

    @abstractmethod
    def compute_energy(
        self,
        F: Matrix3x3,
        lam: float,
        mu: float
    ) -> float:
        """Compute strain energy density.

        Args:
            F: Deformation gradient
            lam: First Lame parameter
            mu: Second Lame parameter

        Returns:
            Strain energy density (per unit volume)
        """
        pass


class NeoHookeanMaterial(MaterialModel):
    """Neo-Hookean hyperelastic material model.

    Good for rubber-like materials with large deformations.
    Energy: W = (mu/2)(I1 - 3) - mu*ln(J) + (lam/2)(ln(J))^2
    where I1 = tr(F^T F), J = det(F)
    """

    def compute_stress(
        self,
        F: Matrix3x3,
        lam: float,
        mu: float
    ) -> Matrix3x3:
        """Compute Neo-Hookean first Piola-Kirchhoff stress.

        Handles inverted elements using configurable strategy to prevent
        deformation gradient singularities.
        """
        J = np.linalg.det(F)

        # Handle inverted/degenerate elements robustly
        if J <= FEM_MIN_JACOBIAN:
            if FEM_INVERSION_HANDLING == "reflect":
                # Reflect the deformation to positive Jacobian
                U, sigma, Vt = np.linalg.svd(F)
                # Clamp minimum singular value
                sigma = np.maximum(sigma, FEM_MIN_JACOBIAN ** (1/3))
                # Ensure positive Jacobian
                if np.prod(sigma) < 0:
                    sigma[np.argmin(sigma)] *= -1
                F = U @ np.diag(sigma) @ Vt
                J = np.prod(sigma)
            else:  # "clamp" or "penalty"
                # Clamp J to minimum value
                J = FEM_MIN_JACOBIAN

        # Compute F inverse transpose safely
        try:
            F_inv_T = np.linalg.inv(F).T
        except np.linalg.LinAlgError:
            # Fallback: use pseudoinverse for degenerate F
            F_inv_T = np.linalg.pinv(F).T

        # P = mu * F - mu * F^{-T} + lam * ln(J) * F^{-T}
        P = mu * F - mu * F_inv_T + lam * math.log(J) * F_inv_T

        return P

    def compute_energy(
        self,
        F: Matrix3x3,
        lam: float,
        mu: float
    ) -> float:
        """Compute Neo-Hookean strain energy density.

        Handles inverted elements to prevent negative Jacobian singularity.
        """
        J = np.linalg.det(F)

        # Handle inverted/degenerate elements
        if J <= FEM_MIN_JACOBIAN:
            J = FEM_MIN_JACOBIAN

        # I1 = tr(F^T F) = ||F||^2
        I1 = np.sum(F * F)
        ln_J = math.log(J)

        # W = (mu/2)(I1 - 3) - mu*ln(J) + (lam/2)(ln(J))^2
        return 0.5 * mu * (I1 - 3.0) - mu * ln_J + 0.5 * lam * ln_J * ln_J


class CorotationalMaterial(MaterialModel):
    """Corotational linear elastic material model.

    Handles large rotations while using linear elasticity.
    Extracts rotation from F, then applies linear stress to strain.
    """

    def __init__(self, use_svd: bool = True):
        """Initialize corotational material.

        Args:
            use_svd: Use SVD for polar decomposition (more stable)
        """
        self.use_svd = use_svd

    def _polar_decomposition(self, F: Matrix3x3) -> Tuple[Matrix3x3, Matrix3x3]:
        """Compute polar decomposition F = R * S.

        Args:
            F: Deformation gradient

        Returns:
            Tuple of (R, S) where R is rotation, S is symmetric stretch
        """
        if self.use_svd:
            U, sigma, Vt = np.linalg.svd(F)
            R = U @ Vt
            # Ensure proper rotation (det = 1)
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            S = Vt.T @ np.diag(sigma) @ Vt
        else:
            # Iterative polar decomposition
            R = F.copy()
            for _ in range(10):
                R_inv_T = np.linalg.inv(R).T
                R_new = 0.5 * (R + R_inv_T)
                if np.linalg.norm(R_new - R) < 1e-6:
                    break
                R = R_new
            S = R.T @ F

        return R, S

    def compute_stress(
        self,
        F: Matrix3x3,
        lam: float,
        mu: float
    ) -> Matrix3x3:
        """Compute corotational first Piola-Kirchhoff stress."""
        R, S = self._polar_decomposition(F)

        # Green strain in rotated frame: E = (S - I)
        I = np.eye(3)
        strain = S - I

        # Linear stress-strain relationship (2nd Piola-Kirchhoff in rotated frame)
        trace_strain = np.trace(strain)
        stress_rotated = 2.0 * mu * strain + lam * trace_strain * I

        # Transform to 1st Piola-Kirchhoff: P = R * stress_rotated
        P = R @ stress_rotated

        return P

    def compute_energy(
        self,
        F: Matrix3x3,
        lam: float,
        mu: float
    ) -> float:
        """Compute corotational strain energy density."""
        R, S = self._polar_decomposition(F)

        # Strain: E = S - I
        I = np.eye(3)
        strain = S - I

        # Linear elastic energy: W = mu * ||E||^2 + (lam/2) * tr(E)^2
        trace_strain = np.trace(strain)
        W = mu * np.sum(strain * strain) + 0.5 * lam * trace_strain * trace_strain

        return W


class StVenantKirchhoffMaterial(MaterialModel):
    """St. Venant-Kirchhoff material model.

    Simple extension of linear elasticity to large strains.
    Note: Can become unstable under compression.
    """

    def compute_stress(
        self,
        F: Matrix3x3,
        lam: float,
        mu: float
    ) -> Matrix3x3:
        """Compute St. Venant-Kirchhoff first Piola-Kirchhoff stress."""
        # Green strain: E = 0.5 * (F^T F - I)
        I = np.eye(3)
        E = 0.5 * (F.T @ F - I)

        # 2nd Piola-Kirchhoff: S = 2*mu*E + lam*tr(E)*I
        trace_E = np.trace(E)
        S = 2.0 * mu * E + lam * trace_E * I

        # 1st Piola-Kirchhoff: P = F * S
        P = F @ S

        return P

    def compute_energy(
        self,
        F: Matrix3x3,
        lam: float,
        mu: float
    ) -> float:
        """Compute St. Venant-Kirchhoff strain energy density."""
        # Green strain: E = 0.5 * (F^T F - I)
        I = np.eye(3)
        E = 0.5 * (F.T @ F - I)

        trace_E = np.trace(E)
        W = mu * np.sum(E * E) + 0.5 * lam * trace_E * trace_E

        return W


# =============================================================================
# Utility Functions
# =============================================================================

def compute_tetrahedron_volume(
    v0: Vector3,
    v1: Vector3,
    v2: Vector3,
    v3: Vector3
) -> float:
    """Compute signed volume of tetrahedron.

    Args:
        v0, v1, v2, v3: Four vertex positions

    Returns:
        Signed volume (1/6 * det([v1-v0, v2-v0, v3-v0]))
    """
    d1 = v1 - v0
    d2 = v2 - v0
    d3 = v3 - v0
    return np.dot(d1, np.cross(d2, d3)) / 6.0


def compute_deformation_gradient(
    current: NDArray[np.float64],  # (4, 3)
    inv_Dm: Matrix3x3
) -> Matrix3x3:
    """Compute deformation gradient for a tetrahedron.

    F = Ds * inv(Dm) where:
    - Ds = [x1-x0, x2-x0, x3-x0] (deformed shape matrix)
    - Dm = [X1-X0, X2-X0, X3-X0] (reference shape matrix)

    Args:
        current: Current vertex positions, shape (4, 3)
        inv_Dm: Inverse of reference shape matrix

    Returns:
        Deformation gradient F, shape (3, 3)
    """
    # Deformed shape matrix
    Ds = np.column_stack([
        current[1] - current[0],
        current[2] - current[0],
        current[3] - current[0]
    ])

    return Ds @ inv_Dm


def compute_strain_energy(
    mesh: TetrahedralMesh,
    elements: List[FEMElement],
    material: MaterialModel,
    lam: float,
    mu: float
) -> float:
    """Compute total strain energy of the mesh.

    Args:
        mesh: Tetrahedral mesh
        elements: Precomputed FEM elements
        material: Material model
        lam: First Lame parameter
        mu: Second Lame parameter

    Returns:
        Total strain energy
    """
    total_energy = 0.0

    for elem in elements:
        current = mesh.get_tetrahedron_vertices(elem.tet_index)
        F = compute_deformation_gradient(current, elem.inv_Dm)
        energy_density = material.compute_energy(F, lam, mu)
        total_energy += energy_density * elem.rest_volume

    return total_energy


# =============================================================================
# FEM Solver
# =============================================================================

class FEMSolver:
    """Finite Element Method solver for soft body deformation.

    Supports multiple material models:
    - Neo-Hookean (rubber-like)
    - Corotational (large rotations)
    - St. Venant-Kirchhoff (simple but can be unstable)

    Attributes:
        mesh: Tetrahedral mesh
        material: Material model
        elements: Precomputed per-element data
        young_modulus: Material stiffness
        poisson_ratio: Material compressibility
        gravity: Gravity acceleration vector
        damping: Velocity damping factor
    """

    def __init__(
        self,
        mesh: TetrahedralMesh,
        material: Optional[MaterialModel] = None,
        young_modulus: float = DEFAULT_YOUNG_MODULUS,
        poisson_ratio: float = DEFAULT_POISSON_RATIO,
        gravity: Optional[Vector3] = None,
        damping: float = 0.99
    ):
        """Initialize FEM solver.

        Args:
            mesh: Tetrahedral mesh to simulate
            material: Material model (default: CorotationalMaterial)
            young_modulus: Young's modulus (stiffness)
            poisson_ratio: Poisson's ratio (0-0.5)
            gravity: Gravity vector (default: [0, -9.81, 0])
            damping: Velocity damping per step
        """
        self.mesh = mesh
        self.material = material or CorotationalMaterial()
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.gravity = gravity if gravity is not None else np.array([0.0, -9.81, 0.0])
        self.damping = damping

        # Compute Lame parameters
        self.lam, self.mu = self._compute_lame_parameters()

        # Precompute element data
        self.elements: List[FEMElement] = []
        self._precompute_elements()

        # Force accumulator
        self.forces = np.zeros_like(mesh.vertices)

    def _compute_lame_parameters(self) -> Tuple[float, float]:
        """Compute Lame parameters from Young's modulus and Poisson's ratio."""
        E = self.young_modulus
        nu = min(self.poisson_ratio, 0.4999)  # Clamp to avoid singularity

        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        return lam, mu

    def _precompute_elements(self) -> None:
        """Precompute per-element reference data."""
        self.elements.clear()

        for i in range(self.mesh.num_tetrahedra):
            rest = self.mesh.get_rest_tetrahedron_vertices(i)

            # Reference shape matrix Dm = [X1-X0, X2-X0, X3-X0]
            Dm = np.column_stack([
                rest[1] - rest[0],
                rest[2] - rest[0],
                rest[3] - rest[0]
            ])

            # Inverse of reference shape matrix
            try:
                inv_Dm = np.linalg.inv(Dm)
            except np.linalg.LinAlgError:
                # Degenerate tetrahedron, use pseudoinverse
                inv_Dm = np.linalg.pinv(Dm)

            # Rest volume
            rest_volume = abs(compute_tetrahedron_volume(
                rest[0], rest[1], rest[2], rest[3]
            ))

            self.elements.append(FEMElement(
                tet_index=i,
                rest_volume=rest_volume,
                inv_Dm=inv_Dm
            ))

    def compute_elastic_forces(self) -> NDArray[np.float64]:
        """Compute elastic forces on all vertices.

        Returns:
            Force array, shape (N, 3)
        """
        forces = np.zeros_like(self.mesh.vertices)

        for elem in self.elements:
            if elem.rest_volume < MIN_TET_VOLUME:
                continue

            current = self.mesh.get_tetrahedron_vertices(elem.tet_index)

            # Deformation gradient
            F = compute_deformation_gradient(current, elem.inv_Dm)

            # First Piola-Kirchhoff stress
            P = self.material.compute_stress(F, self.lam, self.mu)

            # Force on each vertex: f_i = -V0 * P * grad_N_i
            # For linear tetrahedra, grad_N = inv_Dm^T * [basis gradients]
            H = -elem.rest_volume * P @ elem.inv_Dm.T

            indices = self.mesh.tetrahedra[elem.tet_index]

            # Forces on vertices 1, 2, 3 are columns of H
            forces[indices[1]] += H[:, 0]
            forces[indices[2]] += H[:, 1]
            forces[indices[3]] += H[:, 2]

            # Force on vertex 0 balances the others
            forces[indices[0]] -= H[:, 0] + H[:, 1] + H[:, 2]

        return forces

    def compute_gravity_forces(self) -> NDArray[np.float64]:
        """Compute gravity forces on all vertices."""
        return self.mesh.masses[:, np.newaxis] * self.gravity

    def step(self, dt: float) -> None:
        """Advance simulation by one timestep.

        Args:
            dt: Timestep duration
        """
        # Accumulate forces
        self.forces.fill(0)
        self.forces += self.compute_elastic_forces()
        self.forces += self.compute_gravity_forces()

        # Semi-implicit Euler integration
        inv_masses = np.where(
            self.mesh.masses > 1e-10,
            1.0 / self.mesh.masses,
            0.0
        )

        # Update velocities
        acceleration = self.forces * inv_masses[:, np.newaxis]
        self.mesh.velocities += acceleration * dt

        # Apply damping
        self.mesh.velocities *= self.damping

        # Update positions
        self.mesh.vertices += self.mesh.velocities * dt

        # Apply fixed vertex constraints
        self.mesh.vertices[self.mesh.fixed] = self.mesh.rest_vertices[self.mesh.fixed]
        self.mesh.velocities[self.mesh.fixed] = 0.0

    def compute_total_energy(self) -> float:
        """Compute total strain energy of the mesh."""
        return compute_strain_energy(
            self.mesh, self.elements, self.material, self.lam, self.mu
        )

    def set_material_properties(
        self,
        young_modulus: Optional[float] = None,
        poisson_ratio: Optional[float] = None
    ) -> None:
        """Update material properties.

        Args:
            young_modulus: New Young's modulus
            poisson_ratio: New Poisson's ratio
        """
        if young_modulus is not None:
            self.young_modulus = young_modulus
        if poisson_ratio is not None:
            self.poisson_ratio = poisson_ratio

        self.lam, self.mu = self._compute_lame_parameters()

    def reset_to_rest_pose(self) -> None:
        """Reset mesh to rest configuration."""
        self.mesh.vertices = self.mesh.rest_vertices.copy()
        self.mesh.velocities.fill(0)
        self.forces.fill(0)

    def apply_external_force(
        self,
        vertex_index: int,
        force: Vector3
    ) -> None:
        """Apply external force to a vertex.

        Args:
            vertex_index: Index of vertex to apply force to
            force: Force vector
        """
        if 0 <= vertex_index < self.mesh.num_vertices:
            self.forces[vertex_index] += force

    def set_fixed_vertices(self, indices: Sequence[int]) -> None:
        """Set vertices as fixed (immovable).

        Args:
            indices: Indices of vertices to fix
        """
        self.mesh.fixed.fill(False)
        for i in indices:
            if 0 <= i < self.mesh.num_vertices:
                self.mesh.fixed[i] = True
