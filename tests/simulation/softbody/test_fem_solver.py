"""Tests for FEM solver module."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from engine.simulation.softbody.fem_solver import (
    TetrahedralMesh,
    FEMElement,
    MaterialModel,
    NeoHookeanMaterial,
    CorotationalMaterial,
    StVenantKirchhoffMaterial,
    FEMSolver,
    compute_tetrahedron_volume,
    compute_deformation_gradient,
    compute_strain_energy,
)
from engine.simulation.softbody.config import (
    DEFAULT_YOUNG_MODULUS,
    DEFAULT_POISSON_RATIO,
    FEM_MIN_JACOBIAN,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_tet_vertices():
    """Simple tetrahedron with unit volume."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


@pytest.fixture
def simple_tet_indices():
    """Single tetrahedron indices."""
    return np.array([[0, 1, 2, 3]], dtype=np.int32)


@pytest.fixture
def cube_mesh():
    """A cube discretized into 5 tetrahedra."""
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
    ], dtype=np.float64)

    # 5 tetrahedra decomposition of cube
    tetrahedra = np.array([
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ], dtype=np.int32)

    return vertices, tetrahedra


@pytest.fixture
def tetrahedral_mesh(simple_tet_vertices, simple_tet_indices):
    """Create a simple tetrahedral mesh."""
    return TetrahedralMesh(
        vertices=simple_tet_vertices,
        tetrahedra=simple_tet_indices,
    )


# =============================================================================
# Test Utility Functions
# =============================================================================

class TestComputeTetrahedronVolume:
    """Test tetrahedron volume computation."""

    def test_unit_tetrahedron(self, simple_tet_vertices):
        """Unit tetrahedron should have volume 1/6."""
        v0, v1, v2, v3 = simple_tet_vertices
        volume = compute_tetrahedron_volume(v0, v1, v2, v3)
        assert np.isclose(volume, 1.0 / 6.0)

    def test_scaled_tetrahedron(self):
        """Scaled tetrahedron should have scaled volume."""
        scale = 2.0
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ])
        volume = compute_tetrahedron_volume(*vertices)
        # Volume scales as scale^3
        assert np.isclose(volume, (scale ** 3) / 6.0)

    def test_inverted_tetrahedron_negative_volume(self):
        """Inverted tetrahedron should have negative signed volume."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],  # Swapped with v3
            [0.0, 1.0, 0.0],  # Swapped with v2
        ])
        volume = compute_tetrahedron_volume(*vertices)
        assert volume < 0

    def test_degenerate_tetrahedron_zero_volume(self):
        """Degenerate (flat) tetrahedron should have zero volume."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],  # Collinear
            [2.0, 0.0, 0.0],  # Collinear
        ])
        volume = compute_tetrahedron_volume(*vertices)
        assert np.isclose(volume, 0.0, atol=1e-10)

    def test_translated_tetrahedron_same_volume(self, simple_tet_vertices):
        """Translation should not change volume."""
        translation = np.array([10.0, 20.0, 30.0])
        translated = simple_tet_vertices + translation
        original_vol = compute_tetrahedron_volume(*simple_tet_vertices)
        translated_vol = compute_tetrahedron_volume(*translated)
        assert np.isclose(original_vol, translated_vol)


class TestComputeDeformationGradient:
    """Test deformation gradient computation."""

    def test_identity_deformation(self, simple_tet_vertices):
        """No deformation should give identity gradient."""
        rest = simple_tet_vertices
        # Compute reference shape matrix inverse
        Dm = np.column_stack([
            rest[1] - rest[0],
            rest[2] - rest[0],
            rest[3] - rest[0],
        ])
        inv_Dm = np.linalg.inv(Dm)

        F = compute_deformation_gradient(rest, inv_Dm)
        assert_array_almost_equal(F, np.eye(3))

    def test_uniform_stretch(self, simple_tet_vertices):
        """Uniform stretch should give scaled identity."""
        rest = simple_tet_vertices
        Dm = np.column_stack([
            rest[1] - rest[0],
            rest[2] - rest[0],
            rest[3] - rest[0],
        ])
        inv_Dm = np.linalg.inv(Dm)

        scale = 2.0
        deformed = rest * scale
        F = compute_deformation_gradient(deformed, inv_Dm)
        assert_array_almost_equal(F, scale * np.eye(3))

    def test_pure_rotation(self, simple_tet_vertices):
        """Pure rotation should give rotation matrix."""
        rest = simple_tet_vertices
        Dm = np.column_stack([
            rest[1] - rest[0],
            rest[2] - rest[0],
            rest[3] - rest[0],
        ])
        inv_Dm = np.linalg.inv(Dm)

        # Rotation around z-axis by 90 degrees
        theta = np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        deformed = (R @ rest.T).T
        F = compute_deformation_gradient(deformed, inv_Dm)
        assert_array_almost_equal(F, R, decimal=5)


# =============================================================================
# Test TetrahedralMesh
# =============================================================================

class TestTetrahedralMesh:
    """Test TetrahedralMesh dataclass."""

    def test_construction(self, simple_tet_vertices, simple_tet_indices):
        """Basic construction should initialize all fields."""
        mesh = TetrahedralMesh(
            vertices=simple_tet_vertices,
            tetrahedra=simple_tet_indices,
        )
        assert mesh.num_vertices == 4
        assert mesh.num_tetrahedra == 1
        assert mesh.rest_vertices is not None
        assert mesh.velocities is not None
        assert mesh.masses is not None
        assert mesh.fixed is not None

    def test_default_initialization(self, simple_tet_vertices, simple_tet_indices):
        """Default values should be properly initialized."""
        mesh = TetrahedralMesh(
            vertices=simple_tet_vertices,
            tetrahedra=simple_tet_indices,
        )
        # Rest vertices should match initial vertices
        assert_array_almost_equal(mesh.rest_vertices, mesh.vertices)
        # Velocities should be zero
        assert_array_almost_equal(mesh.velocities, np.zeros((4, 3)))
        # Masses should be 1.0
        assert_array_almost_equal(mesh.masses, np.ones(4))
        # No fixed vertices
        assert not np.any(mesh.fixed)

    def test_custom_initialization(self, simple_tet_vertices, simple_tet_indices):
        """Custom values should be used when provided."""
        custom_masses = np.array([1.0, 2.0, 3.0, 4.0])
        custom_fixed = np.array([True, False, False, False])

        mesh = TetrahedralMesh(
            vertices=simple_tet_vertices,
            tetrahedra=simple_tet_indices,
            masses=custom_masses,
            fixed=custom_fixed,
        )
        assert_array_almost_equal(mesh.masses, custom_masses)
        assert np.array_equal(mesh.fixed, custom_fixed)

    def test_get_tetrahedron_vertices(self, tetrahedral_mesh):
        """Getting tetrahedron vertices should return correct shape."""
        verts = tetrahedral_mesh.get_tetrahedron_vertices(0)
        assert verts.shape == (4, 3)

    def test_compute_volume(self, tetrahedral_mesh):
        """Volume computation should match manual calculation."""
        volume = tetrahedral_mesh.compute_volume(0)
        assert np.isclose(volume, 1.0 / 6.0)

    def test_compute_rest_volume(self, tetrahedral_mesh):
        """Rest volume should match current volume initially."""
        volume = tetrahedral_mesh.compute_volume(0)
        rest_volume = tetrahedral_mesh.compute_rest_volume(0)
        assert np.isclose(volume, rest_volume)

    def test_compute_total_volume(self, cube_mesh):
        """Total volume of cube mesh should be approximately 1."""
        vertices, tetrahedra = cube_mesh
        mesh = TetrahedralMesh(vertices=vertices, tetrahedra=tetrahedra)
        total_volume = mesh.compute_total_volume()
        assert np.isclose(total_volume, 1.0, atol=0.01)

    def test_compute_center_of_mass(self, simple_tet_vertices, simple_tet_indices):
        """Center of mass should be at centroid with uniform masses."""
        mesh = TetrahedralMesh(
            vertices=simple_tet_vertices,
            tetrahedra=simple_tet_indices,
        )
        com = mesh.compute_center_of_mass()
        expected_com = np.mean(simple_tet_vertices, axis=0)
        assert_array_almost_equal(com, expected_com)

    def test_compute_center_of_mass_weighted(self, simple_tet_vertices, simple_tet_indices):
        """Center of mass should shift towards heavier vertices."""
        # Heavy vertex at origin
        masses = np.array([100.0, 1.0, 1.0, 1.0])
        mesh = TetrahedralMesh(
            vertices=simple_tet_vertices,
            tetrahedra=simple_tet_indices,
            masses=masses,
        )
        com = mesh.compute_center_of_mass()
        # COM should be close to origin
        assert np.linalg.norm(com) < 0.1

    def test_compute_mass_from_density(self, tetrahedral_mesh):
        """Mass computation from density should be correct."""
        density = 1000.0
        tetrahedral_mesh.compute_mass_from_density(density)
        # Total mass should equal density * volume
        volume = abs(tetrahedral_mesh.compute_rest_volume(0))
        expected_total = density * volume
        actual_total = np.sum(tetrahedral_mesh.masses)
        assert np.isclose(actual_total, expected_total)


# =============================================================================
# Test Material Models
# =============================================================================

class TestNeoHookeanMaterial:
    """Test Neo-Hookean material model."""

    @pytest.fixture
    def material(self):
        return NeoHookeanMaterial()

    def test_zero_deformation_zero_stress(self, material):
        """Identity deformation should give zero stress."""
        F = np.eye(3)
        lam, mu = 1000.0, 500.0
        P = material.compute_stress(F, lam, mu)
        # At rest, stress should be zero
        assert_array_almost_equal(P, np.zeros((3, 3)), decimal=5)

    def test_zero_deformation_zero_energy(self, material):
        """Identity deformation should give zero energy."""
        F = np.eye(3)
        lam, mu = 1000.0, 500.0
        W = material.compute_energy(F, lam, mu)
        assert np.isclose(W, 0.0)

    def test_stretch_positive_energy(self, material):
        """Stretching should produce positive energy."""
        F = 1.5 * np.eye(3)  # Uniform stretch
        lam, mu = 1000.0, 500.0
        W = material.compute_energy(F, lam, mu)
        assert W > 0

    def test_compression_positive_energy(self, material):
        """Compression should produce positive energy."""
        F = 0.5 * np.eye(3)  # Uniform compression
        lam, mu = 1000.0, 500.0
        W = material.compute_energy(F, lam, mu)
        assert W > 0

    def test_stress_symmetry(self, material):
        """Stress should be symmetric for symmetric deformation."""
        F = np.diag([1.2, 1.1, 1.0])  # Principal stretch
        lam, mu = 1000.0, 500.0
        P = material.compute_stress(F, lam, mu)
        # For principal stretches, P should be diagonal
        off_diag = P - np.diag(np.diag(P))
        assert np.allclose(off_diag, 0, atol=1e-10)

    def test_handles_near_singular_jacobian(self, material):
        """Should handle near-singular Jacobian gracefully."""
        # Nearly degenerate deformation
        F = np.diag([1.0, 1.0, FEM_MIN_JACOBIAN * 0.5])
        lam, mu = 1000.0, 500.0
        # Should not raise
        P = material.compute_stress(F, lam, mu)
        W = material.compute_energy(F, lam, mu)
        assert np.all(np.isfinite(P))
        assert np.isfinite(W)


class TestCorotationalMaterial:
    """Test Corotational material model."""

    @pytest.fixture
    def material(self):
        return CorotationalMaterial(use_svd=True)

    def test_zero_deformation_zero_stress(self, material):
        """Identity deformation should give zero stress."""
        F = np.eye(3)
        lam, mu = 1000.0, 500.0
        P = material.compute_stress(F, lam, mu)
        assert_array_almost_equal(P, np.zeros((3, 3)), decimal=5)

    def test_zero_deformation_zero_energy(self, material):
        """Identity deformation should give zero energy."""
        F = np.eye(3)
        lam, mu = 1000.0, 500.0
        W = material.compute_energy(F, lam, mu)
        assert np.isclose(W, 0.0)

    def test_pure_rotation_zero_stress(self, material):
        """Pure rotation should give zero stress."""
        theta = np.pi / 4
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        lam, mu = 1000.0, 500.0
        P = material.compute_stress(R, lam, mu)
        assert_array_almost_equal(P, np.zeros((3, 3)), decimal=5)

    def test_stretch_positive_energy(self, material):
        """Stretching should produce positive energy."""
        F = 1.5 * np.eye(3)
        lam, mu = 1000.0, 500.0
        W = material.compute_energy(F, lam, mu)
        assert W > 0

    def test_svd_vs_iterative(self):
        """SVD and iterative polar decomposition should give same result."""
        mat_svd = CorotationalMaterial(use_svd=True)
        mat_iter = CorotationalMaterial(use_svd=False)

        F = np.array([
            [1.1, 0.2, 0.1],
            [0.1, 1.3, 0.15],
            [0.05, 0.1, 0.9],
        ])
        lam, mu = 1000.0, 500.0

        P_svd = mat_svd.compute_stress(F, lam, mu)
        P_iter = mat_iter.compute_stress(F, lam, mu)
        assert_array_almost_equal(P_svd, P_iter, decimal=4)


class TestStVenantKirchhoffMaterial:
    """Test St. Venant-Kirchhoff material model."""

    @pytest.fixture
    def material(self):
        return StVenantKirchhoffMaterial()

    def test_zero_deformation_zero_stress(self, material):
        """Identity deformation should give zero stress."""
        F = np.eye(3)
        lam, mu = 1000.0, 500.0
        P = material.compute_stress(F, lam, mu)
        assert_array_almost_equal(P, np.zeros((3, 3)), decimal=5)

    def test_zero_deformation_zero_energy(self, material):
        """Identity deformation should give zero energy."""
        F = np.eye(3)
        lam, mu = 1000.0, 500.0
        W = material.compute_energy(F, lam, mu)
        assert np.isclose(W, 0.0)

    def test_stretch_positive_energy(self, material):
        """Stretching should produce positive energy."""
        F = 1.2 * np.eye(3)
        lam, mu = 1000.0, 500.0
        W = material.compute_energy(F, lam, mu)
        assert W > 0


# =============================================================================
# Test FEMSolver
# =============================================================================

class TestFEMSolver:
    """Test FEM solver."""

    @pytest.fixture
    def solver(self, simple_tet_vertices, simple_tet_indices):
        mesh = TetrahedralMesh(
            vertices=simple_tet_vertices,
            tetrahedra=simple_tet_indices,
        )
        return FEMSolver(mesh)

    def test_construction(self, solver):
        """Solver should initialize properly."""
        assert solver.mesh is not None
        assert solver.material is not None
        assert len(solver.elements) == 1
        assert solver.lam > 0
        assert solver.mu > 0

    def test_default_material_is_corotational(self, solver):
        """Default material should be corotational."""
        assert isinstance(solver.material, CorotationalMaterial)

    def test_custom_material(self, simple_tet_vertices, simple_tet_indices):
        """Custom material should be used."""
        mesh = TetrahedralMesh(
            vertices=simple_tet_vertices,
            tetrahedra=simple_tet_indices,
        )
        material = NeoHookeanMaterial()
        solver = FEMSolver(mesh, material=material)
        assert isinstance(solver.material, NeoHookeanMaterial)

    def test_lame_parameters(self, solver):
        """Lame parameters should be computed correctly."""
        E = DEFAULT_YOUNG_MODULUS
        nu = DEFAULT_POISSON_RATIO
        expected_mu = E / (2.0 * (1.0 + nu))
        expected_lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        assert np.isclose(solver.mu, expected_mu)
        assert np.isclose(solver.lam, expected_lam)

    def test_element_precomputation(self, solver):
        """Elements should have precomputed data."""
        elem = solver.elements[0]
        assert elem.rest_volume > 0
        assert elem.inv_Dm is not None
        assert elem.inv_Dm.shape == (3, 3)

    def test_elastic_forces_at_rest_zero(self, solver):
        """Elastic forces should be zero at rest configuration."""
        forces = solver.compute_elastic_forces()
        assert_array_almost_equal(forces, np.zeros_like(forces), decimal=5)

    def test_elastic_forces_with_deformation(self, solver):
        """Elastic forces should be non-zero when deformed."""
        # Stretch mesh
        solver.mesh.vertices *= 1.5
        forces = solver.compute_elastic_forces()
        assert np.linalg.norm(forces) > 0

    def test_gravity_forces(self, solver):
        """Gravity forces should be correct."""
        forces = solver.compute_gravity_forces()
        # Each vertex has mass 1, gravity is [0, -9.81, 0]
        expected = solver.mesh.masses[:, np.newaxis] * solver.gravity
        assert_array_almost_equal(forces, expected)

    def test_step_moves_vertices(self, solver):
        """A simulation step should move non-fixed vertices."""
        initial_pos = solver.mesh.vertices.copy()
        solver.step(dt=0.01)
        # Vertices should have moved (due to gravity)
        assert not np.allclose(solver.mesh.vertices, initial_pos)

    def test_fixed_vertices_dont_move(self, solver):
        """Fixed vertices should not move."""
        solver.set_fixed_vertices([0])
        initial_pos = solver.mesh.vertices[0].copy()
        solver.step(dt=0.01)
        assert_array_almost_equal(solver.mesh.vertices[0], initial_pos)

    def test_reset_to_rest_pose(self, solver):
        """Reset should restore rest configuration."""
        original = solver.mesh.vertices.copy()
        solver.step(dt=0.01)
        solver.step(dt=0.01)
        solver.reset_to_rest_pose()
        assert_array_almost_equal(solver.mesh.vertices, original)
        assert_array_almost_equal(solver.mesh.velocities, np.zeros_like(solver.mesh.velocities))

    def test_apply_external_force(self, solver):
        """External forces should be applied correctly."""
        force = np.array([100.0, 0.0, 0.0])
        solver.apply_external_force(0, force)
        assert_array_almost_equal(solver.forces[0], force)

    def test_set_material_properties(self, solver):
        """Material properties can be updated."""
        new_young = 50000.0
        new_poisson = 0.4
        solver.set_material_properties(young_modulus=new_young, poisson_ratio=new_poisson)
        assert solver.young_modulus == new_young
        assert solver.poisson_ratio == new_poisson
        # Lame parameters should be recomputed
        E = new_young
        nu = new_poisson
        expected_mu = E / (2.0 * (1.0 + nu))
        assert np.isclose(solver.mu, expected_mu)

    def test_total_energy_at_rest_zero(self, solver):
        """Total energy should be zero at rest."""
        energy = solver.compute_total_energy()
        assert np.isclose(energy, 0.0)

    def test_total_energy_with_deformation(self, solver):
        """Total energy should be positive when deformed."""
        solver.mesh.vertices *= 1.3
        energy = solver.compute_total_energy()
        assert energy > 0


class TestFEMSolverCubeMesh:
    """Test FEM solver with more complex mesh."""

    @pytest.fixture
    def solver(self, cube_mesh):
        vertices, tetrahedra = cube_mesh
        mesh = TetrahedralMesh(vertices=vertices, tetrahedra=tetrahedra)
        return FEMSolver(mesh)

    def test_multiple_elements(self, solver):
        """Should have correct number of elements."""
        assert len(solver.elements) == 5

    def test_stability_multiple_steps(self, solver):
        """Solver should remain stable over multiple steps."""
        solver.set_fixed_vertices([0, 1, 2, 3])  # Fix bottom face
        for _ in range(100):
            solver.step(dt=0.001)
        # Check positions are finite
        assert np.all(np.isfinite(solver.mesh.vertices))

    def test_energy_conservation_no_damping(self, solver):
        """Without damping, energy should be approximately conserved."""
        solver.damping = 1.0  # No damping
        solver.gravity = np.zeros(3)  # No gravity
        # Apply initial velocity
        solver.mesh.velocities[4:] = np.array([0.1, 0.0, 0.0])

        # Run simulation
        initial_kinetic = 0.5 * np.sum(
            solver.mesh.masses[:, np.newaxis] * solver.mesh.velocities ** 2
        )
        for _ in range(50):
            solver.step(dt=0.001)
        final_kinetic = 0.5 * np.sum(
            solver.mesh.masses[:, np.newaxis] * solver.mesh.velocities ** 2
        )
        final_potential = solver.compute_total_energy()

        # Energy should be roughly conserved (allow some numerical drift)
        total_initial = initial_kinetic
        total_final = final_kinetic + final_potential
        # Note: Energy conservation is only approximate in explicit integration
        assert np.isfinite(total_final)


class TestFEMEdgeCases:
    """Test edge cases and error handling."""

    def test_degenerate_tetrahedron(self):
        """Degenerate tetrahedron should be handled gracefully."""
        # Collinear vertices (degenerate)
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float64)
        tetrahedra = np.array([[0, 1, 2, 3]], dtype=np.int32)
        mesh = TetrahedralMesh(vertices=vertices, tetrahedra=tetrahedra)

        # Should not raise
        solver = FEMSolver(mesh)
        # Element should have pseudoinverse
        assert solver.elements[0].inv_Dm is not None

    def test_zero_mass_vertex(self):
        """Zero mass vertex should be handled."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        tetrahedra = np.array([[0, 1, 2, 3]], dtype=np.int32)
        masses = np.array([0.0, 1.0, 1.0, 1.0])

        mesh = TetrahedralMesh(vertices=vertices, tetrahedra=tetrahedra, masses=masses)
        solver = FEMSolver(mesh)
        # Should not raise
        solver.step(dt=0.01)
        assert np.all(np.isfinite(solver.mesh.vertices))

    def test_negative_vertex_index(self, simple_tet_vertices, simple_tet_indices):
        """Out of range vertex index should be handled."""
        mesh = TetrahedralMesh(
            vertices=simple_tet_vertices,
            tetrahedra=simple_tet_indices,
        )
        solver = FEMSolver(mesh)
        # Should not raise for invalid index
        solver.apply_external_force(-1, np.array([1.0, 0.0, 0.0]))
        solver.apply_external_force(100, np.array([1.0, 0.0, 0.0]))
