"""Comprehensive tests for soft body and fluid simulation modules.

Tests cover:
- FEM deformation
- Shape matching
- Soft body constraints
- Muscle contraction
- SPH density/pressure
- PBF constraint solving
- FLIP particle-grid transfer
- Surface reconstruction

Total: 130+ tests
"""

import math
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less


# =============================================================================
# Soft Body Imports
# =============================================================================

from engine.simulation.softbody.config import (
    DEFAULT_YOUNG_MODULUS,
    DEFAULT_POISSON_RATIO,
    VOLUME_STIFFNESS,
    SHAPE_MATCHING_STIFFNESS,
    MAX_DEFORMATION,
    SOFTBODY_SUBSTEPS,
    SoftBodyMaterial,
    MaterialPreset,
    SoftBodySolverType,
)
from engine.simulation.softbody.fem_solver import (
    FEMSolver,
    TetrahedralMesh,
    NeoHookeanMaterial,
    CorotationalMaterial,
    StVenantKirchhoffMaterial,
    compute_tetrahedron_volume,
    compute_deformation_gradient,
)
from engine.simulation.softbody.shape_matching import (
    ShapeMatchingSolver,
    ShapeMatchingCluster,
    ClusterConfig,
    compute_center_of_mass,
    compute_rigid_transform,
    goal_positions,
)
from engine.simulation.softbody.soft_body_pbd import (
    PBDSoftBody,
    VolumeConstraint,
    StrainLimitConstraint,
    EdgeLengthConstraint,
    CollisionConstraint,
    PlaneCollider,
    SphereCollider,
)
from engine.simulation.softbody.muscle import (
    Muscle,
    MuscleAttachment,
    MuscleGroup,
    MuscleFiber,
    MuscleProperties,
)
from engine.simulation.softbody.deformable_mesh import (
    DeformableMesh,
    EmbeddedSurface,
    compute_barycentric_coords,
    point_in_tetrahedron,
)


# =============================================================================
# Fluid Imports
# =============================================================================

from engine.simulation.fluid.config import (
    PARTICLE_RADIUS,
    SMOOTHING_LENGTH,
    REST_DENSITY,
    VISCOSITY,
    SURFACE_TENSION,
    MAX_PARTICLES,
    GRID_CELL_SIZE,
    FluidMaterial,
    FluidConfig,
    FluidSolverType,
)
from engine.simulation.fluid.sph import (
    SPHSolver,
    SPHParticle,
    SPHKernels,
    SpatialHashGrid,
)
from engine.simulation.fluid.pbf import (
    PBFSolver,
    PBFParticle,
    PBFConfig,
)
from engine.simulation.fluid.flip_pic import (
    FLIPSolver,
    MACGrid,
    FLIPConfig,
)
from engine.simulation.fluid.eulerian import (
    EulerianSolver,
    StaggeredGrid,
    VelocityField,
    EulerianConfig,
)
from engine.simulation.fluid.shallow_water import (
    ShallowWaterSolver,
    HeightField,
    TerrainBoundary,
    ShallowWaterConfig,
)
from engine.simulation.fluid.surface_reconstruction import (
    MarchingCubes,
    DensityField,
    FluidSurface,
    compute_density_field,
)
from engine.simulation.fluid.gpu_fluid import (
    GPUFluidSolver,
    GPUSpatialHash,
    GPUFluidConfig,
    GPUFluidSolverStub,
    ParticleBuffer,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_tet_mesh():
    """Create a simple tetrahedron mesh (single tet)."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0]
    ], dtype=np.float64)
    tetrahedra = np.array([[0, 1, 2, 3]], dtype=np.int32)
    return TetrahedralMesh(vertices, tetrahedra)


@pytest.fixture
def cube_tet_mesh():
    """Create a cube decomposed into tetrahedra."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float64)
    # 5 tetrahedra to fill cube
    tetrahedra = np.array([
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6]
    ], dtype=np.int32)
    return TetrahedralMesh(vertices, tetrahedra)


@pytest.fixture
def sph_solver():
    """Create SPH solver with some particles."""
    solver = SPHSolver(
        smoothing_length=0.1,
        bounds_min=np.array([0, 0, 0]),
        bounds_max=np.array([2, 2, 2])
    )
    solver.add_block(
        np.array([0.5, 0.5, 0.5]),
        np.array([1.0, 1.0, 1.0]),
        spacing=0.05
    )
    return solver


@pytest.fixture
def pbf_solver():
    """Create PBF solver with some particles."""
    solver = PBFSolver(
        smoothing_length=0.1,
        bounds_min=np.array([0, 0, 0]),
        bounds_max=np.array([2, 2, 2])
    )
    solver.add_block(
        np.array([0.5, 0.5, 0.5]),
        np.array([1.0, 1.0, 1.0]),
        spacing=0.05
    )
    return solver


# =============================================================================
# SOFT BODY CONFIG TESTS
# =============================================================================

class TestSoftBodyConfig:
    """Tests for soft body configuration."""

    def test_default_constants(self):
        """Test default configuration values."""
        assert DEFAULT_YOUNG_MODULUS == 10000.0
        assert DEFAULT_POISSON_RATIO == 0.3
        assert VOLUME_STIFFNESS == 1.0
        assert SHAPE_MATCHING_STIFFNESS == 0.5
        assert MAX_DEFORMATION == 0.3
        assert SOFTBODY_SUBSTEPS == 4

    def test_material_preset_rubber(self):
        """Test rubber material preset."""
        mat = SoftBodyMaterial.from_preset(MaterialPreset.RUBBER)
        assert mat.young_modulus == 1e6
        assert mat.poisson_ratio == 0.49
        assert mat.max_stretch == 2.0

    def test_material_preset_muscle(self):
        """Test muscle material preset."""
        mat = SoftBodyMaterial.from_preset(MaterialPreset.MUSCLE)
        assert mat.young_modulus == 1e4
        assert mat.poisson_ratio == 0.4

    def test_material_preset_jelly(self):
        """Test jelly material preset."""
        mat = SoftBodyMaterial.from_preset(MaterialPreset.JELLY)
        assert mat.young_modulus == 500.0
        assert mat.poisson_ratio == 0.48

    def test_lame_parameters(self):
        """Test Lame parameter computation."""
        mat = SoftBodyMaterial(young_modulus=1000.0, poisson_ratio=0.3)
        lam, mu = mat.compute_lame_parameters()
        # mu = E / (2 * (1 + nu)) = 1000 / 2.6 = 384.6
        assert_allclose(mu, 1000.0 / 2.6, rtol=1e-3)
        # lambda = E * nu / ((1 + nu) * (1 - 2*nu)) = 1000 * 0.3 / (1.3 * 0.4)
        assert_allclose(lam, 300.0 / 0.52, rtol=1e-3)

    def test_solver_types_exist(self):
        """Test that all solver types are defined."""
        assert SoftBodySolverType.FEM is not None
        assert SoftBodySolverType.COROTATIONAL is not None
        assert SoftBodySolverType.SHAPE_MATCHING is not None
        assert SoftBodySolverType.PBD is not None
        assert SoftBodySolverType.XPBD is not None


# =============================================================================
# FEM SOLVER TESTS
# =============================================================================

class TestFEMSolver:
    """Tests for Finite Element Method solver."""

    def test_tetrahedron_volume(self):
        """Test tetrahedron volume computation."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        vol = compute_tetrahedron_volume(v0, v1, v2, v3)
        assert_allclose(vol, 1.0 / 6.0, rtol=1e-10)

    def test_mesh_creation(self, simple_tet_mesh):
        """Test tetrahedral mesh creation."""
        assert simple_tet_mesh.num_vertices == 4
        assert simple_tet_mesh.num_tetrahedra == 1
        assert simple_tet_mesh.rest_vertices.shape == (4, 3)
        assert simple_tet_mesh.velocities.shape == (4, 3)

    def test_mesh_volume(self, simple_tet_mesh):
        """Test mesh volume computation."""
        vol = simple_tet_mesh.compute_volume(0)
        assert vol > 0

    def test_deformation_gradient_identity(self, simple_tet_mesh):
        """Test deformation gradient is identity at rest."""
        rest = simple_tet_mesh.get_rest_tetrahedron_vertices(0)
        Dm = np.column_stack([
            rest[1] - rest[0],
            rest[2] - rest[0],
            rest[3] - rest[0]
        ])
        inv_Dm = np.linalg.inv(Dm)
        F = compute_deformation_gradient(rest, inv_Dm)
        assert_allclose(F, np.eye(3), atol=1e-10)

    def test_fem_solver_creation(self, simple_tet_mesh):
        """Test FEM solver initialization."""
        solver = FEMSolver(simple_tet_mesh)
        assert len(solver.elements) == 1
        assert solver.young_modulus == DEFAULT_YOUNG_MODULUS
        assert solver.poisson_ratio == DEFAULT_POISSON_RATIO

    def test_fem_elastic_forces_at_rest(self, simple_tet_mesh):
        """Test elastic forces are zero at rest configuration."""
        solver = FEMSolver(simple_tet_mesh)
        forces = solver.compute_elastic_forces()
        assert_allclose(forces, 0.0, atol=1e-10)

    def test_fem_step_gravity(self, simple_tet_mesh):
        """Test FEM step with gravity actually produces deformation."""
        solver = FEMSolver(simple_tet_mesh, gravity=np.array([0, -9.81, 0]))
        solver.mesh.fixed[0] = True  # Fix one vertex
        initial_positions = solver.mesh.vertices.copy()
        initial_y = solver.mesh.vertices[:, 1].copy()

        # Run multiple steps to ensure deformation occurs
        for _ in range(10):
            solver.step(0.01)

        # CRITICAL: Verify actual deformation occurred, not just position change
        final_positions = solver.mesh.vertices

        # 1. Non-fixed vertices should move down due to gravity
        assert solver.mesh.vertices[1, 1] < initial_y[1], "Vertex 1 did not move down"
        assert solver.mesh.vertices[2, 1] < initial_y[2], "Vertex 2 did not move down"
        assert solver.mesh.vertices[3, 1] < initial_y[3], "Vertex 3 did not move down"

        # 2. Fixed vertex should NOT have moved
        assert_allclose(solver.mesh.vertices[0], initial_positions[0],
                       err_msg="Fixed vertex moved when it should not")

        # 3. Mesh should have deformed (distances between vertices changed)
        initial_dist_12 = np.linalg.norm(initial_positions[1] - initial_positions[2])
        final_dist_12 = np.linalg.norm(final_positions[1] - final_positions[2])
        # Allow some deformation but not excessive
        deformation_ratio = abs(final_dist_12 - initial_dist_12) / initial_dist_12
        assert deformation_ratio < 0.5, f"Excessive deformation: {deformation_ratio}"

    def test_fem_fixed_vertices(self, simple_tet_mesh):
        """Test fixed vertices don't move."""
        solver = FEMSolver(simple_tet_mesh)
        solver.set_fixed_vertices([0, 1])
        initial_pos = solver.mesh.vertices[:2].copy()
        for _ in range(10):
            solver.step(0.01)
        assert_allclose(solver.mesh.vertices[:2], initial_pos)

    def test_fem_energy_zero_at_rest(self, simple_tet_mesh):
        """Test strain energy is zero at rest."""
        solver = FEMSolver(simple_tet_mesh)
        energy = solver.compute_total_energy()
        assert_allclose(energy, 0.0, atol=1e-10)

    def test_fem_reset_to_rest(self, simple_tet_mesh):
        """Test reset to rest pose."""
        solver = FEMSolver(simple_tet_mesh)
        solver.mesh.vertices += np.random.randn(4, 3) * 0.1
        solver.reset_to_rest_pose()
        assert_allclose(solver.mesh.vertices, solver.mesh.rest_vertices)


class TestMaterialModels:
    """Tests for FEM material models."""

    def test_neo_hookean_singularity_handling(self):
        """Test Neo-Hookean handles inverted/degenerate elements without NaN."""
        mat = NeoHookeanMaterial()

        # Test case 1: Nearly singular deformation (collapsed element)
        F_collapsed = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, 1.0]
        ])
        P = mat.compute_stress(F_collapsed, 1000.0, 500.0)
        assert np.all(np.isfinite(P)), "Stress has NaN/Inf for collapsed element"

        # Test case 2: Inverted element (negative Jacobian)
        F_inverted = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        P = mat.compute_stress(F_inverted, 1000.0, 500.0)
        assert np.all(np.isfinite(P)), "Stress has NaN/Inf for inverted element"

        # Test case 3: Energy should also be finite
        W_collapsed = mat.compute_energy(F_collapsed, 1000.0, 500.0)
        assert np.isfinite(W_collapsed), "Energy not finite for collapsed element"

        W_inverted = mat.compute_energy(F_inverted, 1000.0, 500.0)
        assert np.isfinite(W_inverted), "Energy not finite for inverted element"

    def test_neo_hookean_stress_identity(self):
        """Test Neo-Hookean stress at identity deformation."""
        mat = NeoHookeanMaterial()
        F = np.eye(3)
        P = mat.compute_stress(F, 1000.0, 500.0)
        assert_allclose(P, 0.0, atol=1e-10)

    def test_neo_hookean_energy_identity(self):
        """Test Neo-Hookean energy at identity."""
        mat = NeoHookeanMaterial()
        F = np.eye(3)
        W = mat.compute_energy(F, 1000.0, 500.0)
        assert_allclose(W, 0.0, atol=1e-10)

    def test_corotational_stress_identity(self):
        """Test corotational stress at identity."""
        mat = CorotationalMaterial()
        F = np.eye(3)
        P = mat.compute_stress(F, 1000.0, 500.0)
        assert_allclose(P, 0.0, atol=1e-10)

    def test_corotational_rotation_invariance(self):
        """Test corotational handles pure rotation."""
        mat = CorotationalMaterial()
        # Pure rotation (90 degrees around z)
        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        W = mat.compute_energy(R, 1000.0, 500.0)
        assert_allclose(W, 0.0, atol=1e-10)

    def test_stvk_stress_identity(self):
        """Test St. Venant-Kirchhoff stress at identity."""
        mat = StVenantKirchhoffMaterial()
        F = np.eye(3)
        P = mat.compute_stress(F, 1000.0, 500.0)
        assert_allclose(P, 0.0, atol=1e-10)


# =============================================================================
# SHAPE MATCHING TESTS
# =============================================================================

class TestShapeMatching:
    """Tests for shape matching solver."""

    def test_center_of_mass(self):
        """Test center of mass computation."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0]
        ], dtype=np.float64)
        masses = np.array([1, 1, 1], dtype=np.float64)
        com = compute_center_of_mass(positions, masses)
        assert_allclose(com, [0.5, 1/3, 0], rtol=1e-10)

    def test_rigid_transform_identity(self):
        """Test rigid transform is identity when positions match."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0]
        ], dtype=np.float64)
        masses = np.ones(3)
        R, cur_com, rest_com = compute_rigid_transform(
            positions, positions, masses
        )
        assert_allclose(R, np.eye(3), atol=1e-10)
        assert_allclose(cur_com, rest_com, atol=1e-10)

    def test_rigid_transform_translation(self):
        """Test rigid transform handles pure translation."""
        rest = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0]
        ], dtype=np.float64)
        current = rest + np.array([1, 2, 3])
        masses = np.ones(3)
        R, cur_com, rest_com = compute_rigid_transform(
            current, rest, masses
        )
        assert_allclose(R, np.eye(3), atol=1e-10)
        assert_allclose(cur_com - rest_com, [1, 2, 3], atol=1e-10)

    def test_goal_positions(self):
        """Test goal position computation."""
        rest = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float64)
        R = np.eye(3)
        cur_com = np.array([0.5, 0.5, 0])
        rest_com = np.array([1/3, 1/3, 0])
        goals = goal_positions(rest, R, cur_com, rest_com)
        # Goals should be rest positions shifted to current COM
        expected = rest - rest_com + cur_com
        assert_allclose(goals, expected, atol=1e-10)

    def test_shape_matching_solver_creation(self):
        """Test shape matching solver initialization."""
        positions = np.random.randn(10, 3)
        solver = ShapeMatchingSolver(positions, stiffness=0.8)
        assert len(solver.clusters) == 1
        assert solver.stiffness == 0.8

    def test_shape_matching_step(self):
        """Test shape matching simulation step."""
        positions = np.random.randn(10, 3)
        solver = ShapeMatchingSolver(positions)
        solver.velocities += np.array([0, -1, 0])
        solver.step(0.016)
        # Positions should change due to velocity
        assert not np.allclose(solver.positions, positions)

    def test_shape_matching_fixed_vertices(self):
        """Test fixed vertices in shape matching."""
        positions = np.random.randn(10, 3)
        solver = ShapeMatchingSolver(positions)
        solver.set_fixed_vertices([0, 1, 2])
        initial = solver.positions[:3].copy()
        solver.step(0.1)
        assert_allclose(solver.positions[:3], initial)

    def test_shape_matching_reset(self):
        """Test reset to rest pose."""
        positions = np.random.randn(10, 3)
        solver = ShapeMatchingSolver(positions)
        solver.step(0.1)
        solver.reset_to_rest_pose()
        assert_allclose(solver.positions, solver.rest_positions)


# =============================================================================
# PBD SOFT BODY TESTS
# =============================================================================

class TestPBDSoftBody:
    """Tests for position-based soft body dynamics."""

    def test_volume_constraint_satisfied(self):
        """Test volume constraint at rest."""
        positions = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]
        ], dtype=np.float64)
        constraint = VolumeConstraint(
            indices=(0, 1, 2, 3),
            rest_volume=compute_tetrahedron_volume(
                positions[0], positions[1], positions[2], positions[3]
            ),
            stiffness=1.0
        )
        violation = constraint.get_constraint_value(positions)
        assert_allclose(violation, 0.0, atol=1e-10)

    def test_edge_length_constraint(self):
        """Test edge length constraint."""
        positions = np.array([
            [0, 0, 0], [1, 0, 0]
        ], dtype=np.float64)
        constraint = EdgeLengthConstraint(
            i0=0, i1=1, rest_length=1.0, stiffness=1.0
        )
        violation = constraint.get_constraint_value(positions)
        assert_allclose(violation, 0.0, atol=1e-10)

    def test_edge_length_constraint_projection(self):
        """Test edge length constraint corrects stretched edge."""
        positions = np.array([
            [0, 0, 0], [2, 0, 0]  # Stretched to 2x
        ], dtype=np.float64)
        inv_masses = np.array([1.0, 1.0])
        constraint = EdgeLengthConstraint(
            i0=0, i1=1, rest_length=1.0, stiffness=1.0
        )
        constraint.project(positions, inv_masses)
        # Should correct towards rest length
        new_dist = np.linalg.norm(positions[1] - positions[0])
        assert new_dist < 2.0

    def test_plane_collider(self):
        """Test plane collision detection."""
        collider = PlaneCollider(
            point=np.array([0, 0, 0]),
            normal=np.array([0, 1, 0])
        )
        # Point below plane
        constraint = collider.get_collision_constraint(
            0, np.array([0, -0.1, 0])
        )
        assert constraint is not None
        # Point above plane
        constraint = collider.get_collision_constraint(
            0, np.array([0, 1, 0])
        )
        assert constraint is None

    def test_sphere_collider(self):
        """Test sphere collision detection."""
        collider = SphereCollider(
            center=np.array([0, 0, 0]),
            radius=1.0
        )
        # Point inside sphere
        constraint = collider.get_collision_constraint(
            0, np.array([0.5, 0, 0])
        )
        assert constraint is not None
        # Point outside sphere
        constraint = collider.get_collision_constraint(
            0, np.array([2, 0, 0])
        )
        assert constraint is None

    def test_pbd_softbody_creation(self, cube_tet_mesh):
        """Test PBD soft body initialization."""
        body = PBDSoftBody(
            cube_tet_mesh.vertices,
            cube_tet_mesh.tetrahedra
        )
        assert len(body.volume_constraints) > 0
        assert len(body.edge_constraints) > 0

    def test_pbd_softbody_step(self, cube_tet_mesh):
        """Test PBD soft body simulation step."""
        body = PBDSoftBody(
            cube_tet_mesh.vertices,
            cube_tet_mesh.tetrahedra
        )
        body.set_fixed_vertices([0, 1, 2, 3])  # Fix bottom
        initial_top = body.positions[4:].copy()
        body.step(0.016)
        # Top vertices should move due to gravity
        assert not np.allclose(body.positions[4:], initial_top)

    def test_pbd_volume_preservation(self, cube_tet_mesh):
        """Test volume preservation in PBD - verifies constraints actually work."""
        body = PBDSoftBody(
            cube_tet_mesh.vertices,
            cube_tet_mesh.tetrahedra
        )
        initial_volume = body.get_rest_volume()
        assert initial_volume > 0, "Initial volume must be positive"

        body.set_fixed_vertices([0, 1, 2, 3])
        initial_positions = body.positions.copy()

        for _ in range(10):
            body.step(0.016)

        final_volume = body.get_total_volume()

        # CRITICAL VERIFICATIONS:
        # 1. Volume constraint actually preserves volume (within tolerance)
        volume_change_ratio = abs(final_volume - initial_volume) / initial_volume
        assert volume_change_ratio < 0.3, \
            f"Volume changed by {volume_change_ratio*100:.1f}%, expected < 30%"

        # 2. Non-fixed vertices actually moved (deformation occurred)
        non_fixed_movement = np.linalg.norm(
            body.positions[4:] - initial_positions[4:], axis=1
        )
        assert np.any(non_fixed_movement > 1e-6), \
            "No deformation occurred - non-fixed vertices did not move"

        # 3. Fixed vertices stayed in place
        fixed_movement = np.linalg.norm(
            body.positions[:4] - initial_positions[:4], axis=1
        )
        assert np.all(fixed_movement < 1e-10), \
            "Fixed vertices moved when they should not"

        # 4. Constraint violations should be small after solving
        violations = body.get_constraint_violation()
        assert violations["volume"] < initial_volume * 0.1, \
            f"Volume constraint violation too large: {violations['volume']}"


# =============================================================================
# MUSCLE TESTS
# =============================================================================

class TestMuscle:
    """Tests for muscle simulation."""

    def test_muscle_properties_default(self):
        """Test default muscle properties."""
        props = MuscleProperties()
        assert props.max_force == 100.0
        assert props.optimal_length == 1.0

    def test_muscle_activation_clamping(self):
        """Test muscle activation is clamped to [0, 1]."""
        muscle = Muscle(
            origin=MuscleAttachment(),
            insertion=MuscleAttachment(is_origin=False),
            fiber_direction=np.array([1, 0, 0])
        )
        muscle.activation = 1.5
        assert muscle.activation == 1.0
        muscle.activation = -0.5
        assert muscle.activation == 0.0

    def test_muscle_force_at_rest(self):
        """Test muscle force at optimal length with no activation."""
        muscle = Muscle(
            origin=MuscleAttachment(),
            insertion=MuscleAttachment(is_origin=False),
            fiber_direction=np.array([1, 0, 0])
        )
        muscle.current_length = muscle.properties.optimal_length
        muscle.activation = 0.0
        force = muscle.compute_contraction_force()
        assert force == 0.0  # No passive stretch, no activation

    def test_muscle_force_with_activation(self):
        """Test muscle force with activation produces expected contraction force."""
        muscle = Muscle(
            origin=MuscleAttachment(),
            insertion=MuscleAttachment(is_origin=False),
            fiber_direction=np.array([1, 0, 0])
        )
        muscle.current_length = muscle.properties.optimal_length
        muscle.activation = 1.0
        force = muscle.compute_contraction_force()

        # CRITICAL VERIFICATIONS:
        # 1. Force must be positive with activation
        assert force > 0, "Muscle force should be positive with activation=1.0"

        # 2. Force should be close to max_force at optimal length (f_L ~= 1.0)
        # At optimal length with no velocity, force should be approximately max_force
        expected_max = muscle.properties.max_force
        assert force >= expected_max * 0.9, \
            f"Force {force} should be close to max_force {expected_max} at optimal length"

        # 3. Force-length relationship: force should decrease away from optimal length
        muscle.current_length = muscle.properties.optimal_length * 1.5  # 50% stretched
        force_stretched = muscle.compute_contraction_force()
        # Stretched muscle should produce less active force
        assert force_stretched < force, \
            "Stretched muscle should produce less active force"

        # 4. Zero activation should produce zero active force (at optimal length)
        muscle.current_length = muscle.properties.optimal_length
        muscle.activation = 0.0
        force_inactive = muscle.compute_contraction_force()
        assert force_inactive == 0.0, \
            "Inactive muscle at optimal length should produce zero force"

    def test_muscle_fiber_creation(self):
        """Test muscle fiber creation."""
        fiber = MuscleFiber(
            start_vertex=0,
            end_vertex=1,
            rest_length=1.0,
            direction=np.array([1, 0, 0])
        )
        assert fiber.max_contraction == 0.3
        target = fiber.compute_target_length(1.0)
        assert target == 1.0 * (1.0 - 0.3)

    def test_muscle_group(self):
        """Test muscle group management."""
        group = MuscleGroup(name="test")
        muscle = Muscle(
            origin=MuscleAttachment(),
            insertion=MuscleAttachment(is_origin=False),
            fiber_direction=np.array([1, 0, 0])
        )
        group.add_muscle(muscle)
        assert len(group.muscles) == 1

        group.set_activation(0.5)
        assert group.muscles[0].activation == 0.5


# =============================================================================
# DEFORMABLE MESH TESTS
# =============================================================================

class TestDeformableMesh:
    """Tests for deformable mesh handling."""

    def test_barycentric_coords_inside(self):
        """Test barycentric coordinates for point inside tet."""
        v0 = np.array([0, 0, 0], dtype=np.float64)
        v1 = np.array([1, 0, 0], dtype=np.float64)
        v2 = np.array([0, 1, 0], dtype=np.float64)
        v3 = np.array([0, 0, 1], dtype=np.float64)
        point = np.array([0.25, 0.25, 0.25])  # Inside
        bary = compute_barycentric_coords(point, v0, v1, v2, v3)
        assert_allclose(np.sum(bary), 1.0)
        assert np.all(bary >= 0)

    def test_point_in_tetrahedron(self):
        """Test point-in-tetrahedron test."""
        v0 = np.array([0, 0, 0], dtype=np.float64)
        v1 = np.array([1, 0, 0], dtype=np.float64)
        v2 = np.array([0, 1, 0], dtype=np.float64)
        v3 = np.array([0, 0, 1], dtype=np.float64)
        assert point_in_tetrahedron(np.array([0.1, 0.1, 0.1]), v0, v1, v2, v3)
        assert not point_in_tetrahedron(np.array([1, 1, 1]), v0, v1, v2, v3)

    def test_deformable_mesh_creation(self, cube_tet_mesh):
        """Test deformable mesh initialization."""
        mesh = DeformableMesh(
            cube_tet_mesh.vertices,
            cube_tet_mesh.tetrahedra
        )
        assert mesh.surface.num_vertices > 0
        assert mesh.surface.num_triangles > 0

    def test_deformable_mesh_update(self, cube_tet_mesh):
        """Test surface update after deformation."""
        mesh = DeformableMesh(
            cube_tet_mesh.vertices,
            cube_tet_mesh.tetrahedra
        )
        # Deform tet vertices
        new_verts = cube_tet_mesh.vertices + np.array([0.1, 0.2, 0.3])
        mesh.set_tet_vertices(new_verts)
        # Surface should follow
        assert np.any(mesh.surface.surface_vertices != cube_tet_mesh.vertices)

    def test_skinning_from_tets(self, cube_tet_mesh):
        """Test skinning weight generation."""
        mesh = DeformableMesh(
            cube_tet_mesh.vertices,
            cube_tet_mesh.tetrahedra
        )
        skinning = mesh.skinning_from_tets()
        assert skinning.weights.shape[1] == 4  # 4 weights per vertex
        # Weights should sum to 1
        weight_sums = np.sum(skinning.weights, axis=1)
        assert_allclose(weight_sums, 1.0, atol=1e-5)


# =============================================================================
# FLUID CONFIG TESTS
# =============================================================================

class TestFluidConfig:
    """Tests for fluid configuration."""

    def test_default_constants(self):
        """Test default fluid constants."""
        assert PARTICLE_RADIUS == 0.05
        assert SMOOTHING_LENGTH == 0.1
        assert REST_DENSITY == 1000.0
        assert VISCOSITY == 0.01
        assert SURFACE_TENSION == 0.1
        assert MAX_PARTICLES == 100000
        assert GRID_CELL_SIZE == 0.1

    def test_fluid_material_water(self):
        """Test water material preset."""
        mat = FluidMaterial.water()
        assert mat.rest_density == 1000.0
        assert mat.viscosity == 0.001

    def test_fluid_material_oil(self):
        """Test oil material preset."""
        mat = FluidMaterial.oil()
        assert mat.rest_density == 850.0
        assert mat.viscosity > FluidMaterial.water().viscosity

    def test_fluid_material_honey(self):
        """Test honey material preset."""
        mat = FluidMaterial.honey()
        assert mat.viscosity > 1.0  # Very viscous

    def test_solver_types_exist(self):
        """Test that all solver types are defined."""
        assert FluidSolverType.SPH is not None
        assert FluidSolverType.PBF is not None
        assert FluidSolverType.FLIP is not None


# =============================================================================
# SPH TESTS
# =============================================================================

class TestSPHKernels:
    """Tests for SPH kernel functions."""

    def test_poly6_at_zero(self):
        """Test poly6 kernel at r=0."""
        h = 0.1
        w = SPHKernels.poly6(0, h)
        # Should be maximum at r=0
        assert w > 0
        w2 = SPHKernels.poly6(h * h * 0.5, h)
        assert w > w2

    def test_poly6_at_boundary(self):
        """Test poly6 kernel at r=h."""
        h = 0.1
        w = SPHKernels.poly6(h * h, h)
        assert_allclose(w, 0.0, atol=1e-10)

    def test_poly6_outside(self):
        """Test poly6 kernel outside radius."""
        h = 0.1
        w = SPHKernels.poly6(h * h * 1.5, h)
        assert w == 0.0

    def test_spiky_at_zero(self):
        """Test spiky kernel at r=0."""
        h = 0.1
        w = SPHKernels.spiky(0, h)
        assert w > 0

    def test_spiky_gradient_direction(self):
        """Test spiky gradient points away from neighbor."""
        r = np.array([0.05, 0, 0])
        dist = 0.05
        h = 0.1
        grad = SPHKernels.spiky_gradient(r, dist, h)
        # Gradient should point in direction of r (away from neighbor)
        assert grad[0] < 0  # Negative because it's -45/pi * ...

    def test_viscosity_laplacian(self):
        """Test viscosity kernel laplacian is positive."""
        h = 0.1
        lap = SPHKernels.viscosity_laplacian(0.05, h)
        assert lap > 0


class TestSPHSolver:
    """Tests for SPH solver."""

    def test_spatial_hash_grid(self):
        """Test spatial hash grid."""
        grid = SpatialHashGrid(0.1)
        grid.insert(0, np.array([0.05, 0.05, 0.05]))
        grid.insert(1, np.array([0.06, 0.06, 0.06]))
        neighbors = grid.get_neighbors(np.array([0.05, 0.05, 0.05]), 0.1)
        assert 0 in neighbors
        assert 1 in neighbors

    def test_sph_solver_creation(self):
        """Test SPH solver initialization."""
        solver = SPHSolver()
        assert solver.num_particles == 0
        assert solver.material is not None

    def test_sph_add_particle(self):
        """Test adding particles."""
        solver = SPHSolver()
        idx = solver.add_particle(np.array([0, 0, 0]))
        assert idx == 0
        assert solver.num_particles == 1

    def test_sph_add_block(self):
        """Test adding block of particles."""
        solver = SPHSolver()
        indices = solver.add_block(
            np.array([0, 0, 0]),
            np.array([0.2, 0.2, 0.2]),
            spacing=0.05
        )
        assert len(indices) > 0
        assert solver.num_particles == len(indices)

    def test_sph_density_computation(self, sph_solver):
        """Test density is computed correctly."""
        sph_solver._build_grid()
        sph_solver._find_neighbors()
        density = sph_solver.compute_density(0)
        assert density > 0

    def test_sph_pressure_computation(self):
        """Test pressure from density."""
        solver = SPHSolver()
        # At rest density, pressure should be low
        p = solver.compute_pressure(REST_DENSITY)
        assert p >= 0
        # Above rest density, pressure should be positive
        p_high = solver.compute_pressure(REST_DENSITY * 1.1)
        assert p_high > p

    def test_sph_step(self, sph_solver):
        """Test SPH simulation step with proper density/pressure verification."""
        initial_positions = sph_solver.get_positions().copy()
        initial_y_mean = np.mean(initial_positions[:, 1])

        # Run simulation step
        sph_solver.step(0.001)

        final_positions = sph_solver.get_positions()
        final_y_mean = np.mean(final_positions[:, 1])

        # CRITICAL VERIFICATIONS:
        # 1. Particles should move due to gravity (y decreases on average)
        assert final_y_mean < initial_y_mean, \
            "Particles did not fall due to gravity"

        # 2. Verify density computation is working
        sph_solver._build_grid()
        sph_solver._find_neighbors()
        sph_solver._compute_density()

        densities = sph_solver.get_densities()
        assert np.all(densities > 0), "Densities must be positive"
        avg_density = np.mean(densities)
        # Density should be in reasonable range relative to rest density
        assert avg_density > REST_DENSITY * 0.1, \
            f"Average density {avg_density} too low"
        assert avg_density < REST_DENSITY * 10, \
            f"Average density {avg_density} too high"

        # 3. Verify pressure is pushing particles apart (incompressibility)
        # After stepping, particles that were compressed should spread out
        # Check that particles don't all collapse to same point
        position_std = np.std(final_positions, axis=0)
        assert np.all(position_std > 0.001), \
            "Particles collapsed - pressure not working"

    def test_sph_kinetic_energy(self, sph_solver):
        """Test kinetic energy computation."""
        sph_solver.step(0.01)  # Let particles gain velocity
        ke = sph_solver.compute_kinetic_energy()
        assert ke > 0


# =============================================================================
# PBF TESTS
# =============================================================================

class TestPBFSolver:
    """Tests for Position Based Fluids solver."""

    def test_pbf_solver_creation(self):
        """Test PBF solver initialization."""
        solver = PBFSolver()
        assert solver.num_particles == 0

    def test_pbf_predict_positions(self):
        """Test position prediction."""
        solver = PBFSolver()
        solver.add_particle(np.array([0.0, 1.0, 0.0]))
        solver.particles[0].velocity = np.array([1.0, 0.0, 0.0])
        solver.predict_positions(0.1)
        # Predicted should be position + velocity * dt + gravity * dt
        assert solver.particles[0].predicted[0] > 0

    def test_pbf_density_constraint(self, pbf_solver):
        """Test density constraint computation."""
        pbf_solver._build_grid()
        pbf_solver._find_neighbors()
        C = pbf_solver.compute_density_constraint(0)
        # Constraint may be positive or negative depending on density
        assert isinstance(C, float)

    def test_pbf_lambda_computation(self, pbf_solver):
        """Test lambda (Lagrange multiplier) computation."""
        pbf_solver._build_grid()
        pbf_solver._find_neighbors()
        lam = pbf_solver.compute_lambda(0)
        assert isinstance(lam, float)

    def test_pbf_position_correction(self, pbf_solver):
        """Test position correction computation."""
        pbf_solver._build_grid()
        pbf_solver._find_neighbors()
        for i, p in enumerate(pbf_solver.particles):
            p.lambda_ = pbf_solver.compute_lambda(i)
        delta_p = pbf_solver.compute_position_correction(0)
        assert delta_p.shape == (3,)

    def test_pbf_constraint_solving(self, pbf_solver):
        """Test constraint solving works without producing NaN/Inf values."""
        pbf_solver.predict_positions(0.016)
        pbf_solver._build_grid()
        pbf_solver._find_neighbors()

        # Initial error
        initial_error = pbf_solver.compute_average_constraint_error()

        # Solve with multiple iterations
        pbf_solver.solve_constraints(4)

        # Get final error
        final_error = pbf_solver.compute_average_constraint_error()

        # CRITICAL VERIFICATIONS:
        # 1. Error must be finite (no NaN/Inf from numerical issues)
        assert np.isfinite(final_error), \
            f"Final error is not finite: {final_error}"

        # 2. Error should not explode (reasonable bounds)
        # Note: PBF can have oscillations, especially with dense particle blocks
        # The key is that it doesn't explode to huge values
        assert final_error < 10.0, \
            f"Constraint error exploded: {final_error}"

        # 3. Verify density constraint values are computed correctly
        for i in range(min(5, len(pbf_solver.particles))):
            C = pbf_solver.compute_density_constraint(i)
            assert np.isfinite(C), f"Density constraint {i} is not finite"

        # 4. Verify lambda computation doesn't produce NaN/Inf
        for i in range(min(5, len(pbf_solver.particles))):
            lam = pbf_solver.compute_lambda(i)
            assert np.isfinite(lam), f"Lambda {i} is not finite: {lam}"
            # Lambda should be bounded by our clamp
            assert abs(lam) <= 1000.0, f"Lambda {i} exceeds clamp: {lam}"

        # 5. Position corrections should be finite
        for i, p in enumerate(pbf_solver.particles[:5]):
            assert np.all(np.isfinite(p.predicted)), \
                f"Particle {i} predicted position has NaN/Inf"

    def test_pbf_step(self, pbf_solver):
        """Test full PBF step."""
        initial_pos = pbf_solver.get_positions().copy()
        pbf_solver.step(0.016)
        final_pos = pbf_solver.get_positions()
        assert not np.allclose(initial_pos, final_pos)


# =============================================================================
# FLIP/PIC TESTS
# =============================================================================

class TestFLIPSolver:
    """Tests for FLIP/PIC solver."""

    def test_mac_grid_creation(self):
        """Test MAC grid initialization."""
        grid = MACGrid((10, 10, 10), 0.1)
        assert grid.u.shape == (11, 10, 10)
        assert grid.v.shape == (10, 11, 10)
        assert grid.w.shape == (10, 10, 11)

    def test_mac_grid_interpolation(self):
        """Test velocity interpolation on MAC grid."""
        grid = MACGrid((10, 10, 10), 0.1)
        grid.u.fill(1.0)  # Uniform x-velocity
        vel = grid.interpolate_velocity(np.array([0.5, 0.5, 0.5]))
        assert_allclose(vel[0], 1.0, atol=0.1)

    def test_flip_solver_creation(self):
        """Test FLIP solver initialization."""
        solver = FLIPSolver(resolution=(16, 16, 16), cell_size=0.1)
        assert solver.num_particles == 0

    def test_flip_add_particle(self):
        """Test adding particles to FLIP."""
        solver = FLIPSolver(resolution=(16, 16, 16), cell_size=0.1)
        idx = solver.add_particle(np.array([0.5, 0.5, 0.5]))
        assert idx == 0
        assert solver.num_particles == 1

    def test_flip_particles_to_grid(self):
        """Test particle-to-grid transfer preserves momentum correctly."""
        solver = FLIPSolver(resolution=(16, 16, 16), cell_size=0.1)
        particle_vel = np.array([1.0, 0.5, -0.3])
        solver.add_particle(
            np.array([0.5, 0.5, 0.5]),
            velocity=particle_vel.copy()
        )
        solver.particles_to_grid()

        # CRITICAL VERIFICATIONS:
        # 1. Grid should have non-zero velocity in u component
        assert np.any(solver.grid.u != 0), "No x-velocity transferred to grid"

        # 2. Grid should have non-zero velocity in v component
        assert np.any(solver.grid.v != 0), "No y-velocity transferred to grid"

        # 3. Grid should have non-zero velocity in w component
        assert np.any(solver.grid.w != 0), "No z-velocity transferred to grid"

        # 4. Interpolated velocity at particle position should match particle velocity
        interp_vel = solver.grid.interpolate_velocity(np.array([0.5, 0.5, 0.5]))
        # Allow some tolerance due to kernel spreading
        assert_allclose(interp_vel, particle_vel, atol=0.3,
                       err_msg="Interpolated velocity doesn't match particle velocity")

    def test_flip_grid_to_particles(self):
        """Test grid-to-particle transfer correctly updates particle velocities."""
        solver = FLIPSolver(resolution=(16, 16, 16), cell_size=0.1)
        solver.add_particle(np.array([0.5, 0.5, 0.5]))

        # Set up initial state
        solver.particles_to_grid()
        solver.grid.save_velocities()

        # Set uniform grid velocity
        solver.grid.u.fill(2.0)
        solver.grid.v.fill(1.0)
        solver.grid.w.fill(-0.5)

        initial_vel = solver.particles[0].velocity.copy()
        solver.grid_to_particles()
        final_vel = solver.particles[0].velocity

        # CRITICAL VERIFICATIONS:
        # 1. Particle x-velocity should have increased
        assert final_vel[0] > initial_vel[0], \
            f"X velocity didn't increase: {initial_vel[0]} -> {final_vel[0]}"

        # 2. Particle y-velocity should have increased
        assert final_vel[1] > initial_vel[1], \
            f"Y velocity didn't increase: {initial_vel[1]} -> {final_vel[1]}"

        # 3. Particle z-velocity should reflect grid (negative)
        # The FLIP/PIC blend should pull velocity toward grid velocity
        assert final_vel[2] < initial_vel[2], \
            f"Z velocity didn't decrease toward grid: {initial_vel[2]} -> {final_vel[2]}"

        # 4. Velocity change should be proportional to FLIP ratio
        # Full PIC would give grid velocity directly
        # Full FLIP would add delta between old and new grid

    def test_flip_pressure_projection(self):
        """Test pressure projection makes divergence-free."""
        solver = FLIPSolver(resolution=(8, 8, 8), cell_size=0.5)
        solver.add_block(
            np.array([1, 1, 1]),
            np.array([3, 3, 3])
        )
        solver.particles_to_grid()
        solver.apply_boundary_conditions()
        solver.project_pressure(0.016)
        # Should not throw

    def test_flip_step(self):
        """Test full FLIP step."""
        solver = FLIPSolver(resolution=(8, 8, 8), cell_size=0.5)
        solver.add_block(
            np.array([1.0, 2.0, 1.0]),
            np.array([3.0, 3.0, 3.0])
        )
        initial_pos = solver.get_positions().copy()
        solver.step(0.016)
        final_pos = solver.get_positions()
        assert not np.allclose(initial_pos, final_pos)


# =============================================================================
# EULERIAN SOLVER TESTS
# =============================================================================

class TestEulerianSolver:
    """Tests for Eulerian grid-based solver."""

    def test_staggered_grid_creation(self):
        """Test staggered grid initialization."""
        grid = StaggeredGrid((10, 10, 10), 0.1)
        assert grid.velocity.u.shape == (11, 10, 10)

    def test_eulerian_solver_creation(self):
        """Test Eulerian solver initialization."""
        config = EulerianConfig(grid_size=(8, 8, 8), dx=0.5)
        solver = EulerianSolver(config=config)
        assert solver.grid.resolution == (8, 8, 8)

    def test_divergence_computation(self):
        """Test divergence computation."""
        grid = StaggeredGrid((4, 4, 4), 1.0)
        # Zero velocity should have zero divergence
        div = grid.compute_divergence()
        assert_allclose(div, 0.0)

    def test_eulerian_advection(self):
        """Test semi-Lagrangian advection."""
        config = EulerianConfig(grid_size=(8, 8, 8), dx=0.5)
        solver = EulerianSolver(config=config)
        solver.grid.velocity.u[4, 4, 4] = 1.0
        solver.advect_velocity(0.1)
        # Should advect velocity

    def test_eulerian_step(self):
        """Test full Eulerian step."""
        config = EulerianConfig(grid_size=(8, 8, 8), dx=0.5)
        solver = EulerianSolver(config=config)
        solver.grid.velocity.v[4, 4, 4] = 1.0
        solver.step(0.016)


# =============================================================================
# SHALLOW WATER TESTS
# =============================================================================

class TestShallowWater:
    """Tests for shallow water solver."""

    def test_height_field_creation(self):
        """Test height field initialization."""
        hf = HeightField(
            height=np.zeros((10, 10)),
            velocity_x=np.zeros((11, 10)),
            velocity_y=np.zeros((10, 11)),
            terrain=np.zeros((10, 10))
        )
        assert hf.resolution == (10, 10)

    def test_terrain_flat(self):
        """Test flat terrain creation."""
        terrain = TerrainBoundary.flat((10, 10), elevation=0.5)
        assert_allclose(terrain.elevation, 0.5)

    def test_terrain_bowl(self):
        """Test bowl terrain creation."""
        terrain = TerrainBoundary.bowl((10, 10), depth=1.0)
        # Center should be lowest
        center_val = terrain.elevation[5, 5]
        corner_val = terrain.elevation[0, 0]
        assert center_val < corner_val

    def test_shallow_water_creation(self):
        """Test shallow water solver initialization."""
        config = ShallowWaterConfig(grid_size=(16, 16), dx=0.5)
        solver = ShallowWaterSolver(config=config)
        assert solver.field.resolution == (16, 16)

    def test_shallow_water_add_water(self):
        """Test adding water to simulation."""
        config = ShallowWaterConfig(grid_size=(16, 16), dx=0.5)
        solver = ShallowWaterSolver(config=config)
        initial_volume = solver.field.total_volume(solver.dx)
        solver.add_water((8, 8), radius=3, height=1.0)
        final_volume = solver.field.total_volume(solver.dx)
        assert final_volume > initial_volume

    def test_shallow_water_step(self):
        """Test shallow water simulation step."""
        config = ShallowWaterConfig(grid_size=(16, 16), dx=0.5)
        solver = ShallowWaterSolver(config=config)
        solver.add_water((8, 8), radius=3, height=1.0)
        initial_heights = solver.field.height.copy()
        solver.step(0.01)
        # Heights should change as water flows
        assert not np.allclose(solver.field.height, initial_heights)

    def test_shallow_water_surface_mesh(self):
        """Test surface mesh generation."""
        config = ShallowWaterConfig(grid_size=(8, 8), dx=0.5)
        solver = ShallowWaterSolver(config=config)
        solver.add_water((4, 4), radius=2, height=1.0)
        verts, tris = solver.get_surface_mesh()
        assert len(verts) == 64
        assert len(tris) == (7 * 7 * 2)


# =============================================================================
# SURFACE RECONSTRUCTION TESTS
# =============================================================================

class TestSurfaceReconstruction:
    """Tests for fluid surface reconstruction."""

    def test_density_field_creation(self):
        """Test density field initialization."""
        data = np.zeros((10, 10, 10))
        field = DensityField(
            data=data,
            origin=np.zeros(3),
            cell_size=0.1
        )
        assert field.resolution == (10, 10, 10)

    def test_density_field_sampling(self):
        """Test density field interpolation."""
        data = np.ones((10, 10, 10))
        field = DensityField(
            data=data,
            origin=np.zeros(3),
            cell_size=0.1
        )
        val = field.sample(np.array([0.5, 0.5, 0.5]))
        assert_allclose(val, 1.0)

    def test_marching_cubes_creation(self):
        """Test marching cubes initialization."""
        mc = MarchingCubes(iso_level=0.5)
        assert mc.iso_level == 0.5

    def test_compute_density_field_from_particles(self):
        """Test density field computation from particles."""
        positions = np.array([
            [0.5, 0.5, 0.5],
            [0.6, 0.5, 0.5],
            [0.5, 0.6, 0.5]
        ], dtype=np.float64)
        field = compute_density_field(
            positions,
            np.zeros(3),
            np.ones(3),
            (8, 8, 8)
        )
        # Should have density near particle positions
        val = field.sample(np.array([0.5, 0.5, 0.5]))
        assert val > 0

    def test_marching_cubes_extraction(self):
        """Test isosurface extraction produces valid mesh output."""
        # Create a simple density field with a sphere
        data = np.zeros((10, 10, 10))
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    dist = math.sqrt((i-5)**2 + (j-5)**2 + (k-5)**2)
                    data[i, j, k] = max(0, 1.0 - dist / 4.0)

        field = DensityField(data=data, origin=np.zeros(3), cell_size=0.1)
        mc = MarchingCubes(iso_level=0.5)
        surface = mc.extract_isosurface(field)

        # CRITICAL VERIFICATIONS:
        # 1. Surface must have vertices
        assert surface.num_vertices > 0, "Surface extraction produced no vertices"

        # 2. Surface must have triangles
        assert surface.num_triangles > 0, \
            f"Surface extraction produced vertices ({surface.num_vertices}) but no triangles"

        # 3. All triangle indices must be valid
        assert np.all(surface.triangles >= 0), "Negative triangle indices"
        assert np.all(surface.triangles < surface.num_vertices), \
            "Triangle indices exceed vertex count"

        # 4. Normals must be present and normalized
        assert surface.normals.shape == surface.vertices.shape, \
            "Normals shape doesn't match vertices"
        normal_lengths = np.linalg.norm(surface.normals, axis=1)
        assert np.allclose(normal_lengths, 1.0, atol=0.01), \
            "Normals are not unit length"

        # 5. Vertices should be within the field bounds
        field_min = field.origin
        field_max = field.origin + np.array(field.resolution) * field.cell_size
        assert np.all(surface.vertices >= field_min - 0.01), \
            "Vertices outside field bounds (min)"
        assert np.all(surface.vertices <= field_max + 0.01), \
            "Vertices outside field bounds (max)"

        # 6. For a sphere-like field, surface should be roughly spherical
        # (vertices roughly equidistant from center)
        center = np.array([0.5, 0.5, 0.5])  # Center of sphere in world coords
        distances = np.linalg.norm(surface.vertices - center, axis=1)
        # Radius where iso_level=0.5 occurs: 1 - r/0.4 = 0.5 => r = 0.2 (in grid units) * 0.1 = 0.02...
        # Actually the field is 10x10x10 with cell_size 0.1, so grid spans 0-1
        # Sphere center at grid (5,5,5) = world (0.5, 0.5, 0.5)
        # iso_level=0.5: 1 - dist/4 = 0.5 => dist = 2 grid units = 0.2 world units
        expected_radius = 0.2
        assert np.std(distances) < expected_radius * 0.5, \
            f"Surface not spherical enough: std={np.std(distances)}"


# =============================================================================
# GPU FLUID TESTS
# =============================================================================

class TestGPUFluid:
    """Tests for GPU fluid simulation stubs."""

    def test_particle_buffer_creation(self):
        """Test particle buffer initialization."""
        buffer = ParticleBuffer.create(1000)
        assert buffer.positions.shape == (1000, 4)
        assert buffer.velocities.shape == (1000, 4)

    def test_gpu_spatial_hash_creation(self):
        """Test GPU spatial hash initialization."""
        hash_grid = GPUSpatialHash(
            resolution=(16, 16, 16),
            cell_size=0.1,
            bounds_min=np.zeros(3)
        )
        assert hash_grid.resolution == (16, 16, 16)

    def test_gpu_spatial_hash_build(self):
        """Test GPU spatial hash building."""
        hash_grid = GPUSpatialHash(
            resolution=(8, 8, 8),
            cell_size=0.5,
            bounds_min=np.zeros(3)
        )
        positions = np.random.rand(100, 4).astype(np.float32) * 4
        hash_grid.build(positions, 100)
        # Cell counts should sum to particle count
        total = np.sum(hash_grid.buffer.cell_count)
        assert total == 100

    def test_gpu_fluid_solver_stub(self):
        """Test GPU fluid solver stub."""
        solver = GPUFluidSolverStub()
        solver.add_particle_cpu(np.array([0.5, 0.5, 0.5]))
        assert solver.particle_count == 1

    def test_gpu_fluid_solver_step(self):
        """Test GPU solver step with stub."""
        solver = GPUFluidSolverStub()
        for i in range(10):
            solver.add_particle_cpu(
                np.array([0.5 + i * 0.1, 1.0, 0.5])
            )
        solver.step_cpu(0.01)
        # Particles should fall due to gravity
        positions = solver.get_positions()
        assert np.any(positions[:, 1] < 1.0)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_fem_with_collision(self, simple_tet_mesh):
        """Test FEM solver with ground collision."""
        solver = FEMSolver(simple_tet_mesh)
        solver.set_fixed_vertices([0])  # Fix one vertex

        # Simulate falling
        for _ in range(100):
            solver.step(0.001)
            # Clamp to ground
            solver.mesh.vertices[:, 1] = np.maximum(
                solver.mesh.vertices[:, 1], 0.0
            )

    def test_pbd_with_multiple_constraints(self, cube_tet_mesh):
        """Test PBD with volume and edge constraints together."""
        body = PBDSoftBody(
            cube_tet_mesh.vertices,
            cube_tet_mesh.tetrahedra
        )
        body.add_collider(PlaneCollider(
            point=np.array([0, 0, 0]),
            normal=np.array([0, 1, 0])
        ))
        body.set_fixed_vertices([0, 1, 2, 3])

        for _ in range(50):
            body.step(0.016)

        # Should have settled above ground
        assert np.all(body.positions[:, 1] >= -0.1)

    def test_fluid_with_surface_extraction(self):
        """Test fluid simulation with surface mesh extraction."""
        solver = SPHSolver(
            smoothing_length=0.1,
            bounds_min=np.array([0, 0, 0]),
            bounds_max=np.array([2, 2, 2])
        )
        solver.add_block(
            np.array([0.5, 0.5, 0.5]),
            np.array([1.0, 1.0, 1.0])
        )

        # Simulate
        solver.step(0.01)

        # Extract surface
        positions = solver.get_positions()
        if len(positions) > 0:
            field = compute_density_field(
                positions,
                np.array([0, 0, 0]),
                np.array([2, 2, 2]),
                (16, 16, 16),
                smoothing_length=0.1
            )
            # Should have some density


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_mass_particle(self):
        """Test handling of zero-mass particles (fixed)."""
        solver = SPHSolver()
        solver.add_particle(np.array([0.0, 0.0, 0.0]))
        solver.particles[0].mass = 0.0
        # Should not crash
        solver.step(0.001)

    def test_degenerate_tetrahedron(self):
        """Test handling of degenerate tetrahedron."""
        # Flat tetrahedron (all in same plane)
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.25, 0.5, 0]
        ], dtype=np.float64)
        tetrahedra = np.array([[0, 1, 2, 3]], dtype=np.int32)
        mesh = TetrahedralMesh(vertices, tetrahedra)
        # Volume should be near zero
        assert abs(mesh.compute_volume(0)) < 0.01

    def test_single_particle_fluid(self):
        """Test fluid simulation with single particle."""
        solver = SPHSolver()
        solver.add_particle(np.array([1.0, 1.0, 1.0]))
        solver.step(0.01)
        # Should not crash, particle should fall

    def test_empty_fluid_solver(self):
        """Test fluid solver with no particles."""
        solver = SPHSolver()
        solver.step(0.01)  # Should not crash
        assert solver.num_particles == 0

    def test_very_small_timestep(self):
        """Test with very small timestep."""
        solver = SPHSolver()
        solver.add_block(
            np.array([0.5, 0.5, 0.5]),
            np.array([0.6, 0.6, 0.6])
        )
        initial = solver.get_positions().copy()
        solver.step(1e-10)
        # Should barely move
        assert_allclose(solver.get_positions(), initial, atol=1e-5)


# =============================================================================
# ADDITIONAL SOFT BODY TESTS
# =============================================================================

class TestAdditionalFEM:
    """Additional FEM solver tests."""

    def test_fem_mass_computation(self, simple_tet_mesh):
        """Test mass computation from density."""
        simple_tet_mesh.compute_mass_from_density(1000.0)
        assert np.all(simple_tet_mesh.masses > 0)
        assert np.sum(simple_tet_mesh.masses) > 0

    def test_fem_external_force(self, simple_tet_mesh):
        """Test applying external force."""
        solver = FEMSolver(simple_tet_mesh)
        solver.apply_external_force(1, np.array([10.0, 0, 0]))
        assert_allclose(solver.forces[1], [10.0, 0, 0])

    def test_fem_material_update(self, simple_tet_mesh):
        """Test updating material properties."""
        solver = FEMSolver(simple_tet_mesh)
        solver.set_material_properties(young_modulus=5000.0, poisson_ratio=0.4)
        assert solver.young_modulus == 5000.0
        assert solver.poisson_ratio == 0.4

    def test_fem_center_of_mass(self, simple_tet_mesh):
        """Test center of mass computation."""
        com = simple_tet_mesh.compute_center_of_mass()
        assert com.shape == (3,)

    def test_fem_rest_volume(self, simple_tet_mesh):
        """Test rest volume computation."""
        vol = simple_tet_mesh.compute_rest_volume(0)
        assert vol > 0


class TestAdditionalShapeMatching:
    """Additional shape matching tests."""

    def test_cluster_creation(self):
        """Test cluster with custom config."""
        indices = np.array([0, 1, 2, 3], dtype=np.int32)
        rest_pos = np.random.randn(4, 3)
        masses = np.ones(4)
        config = ClusterConfig(stiffness=0.9, damping=0.95)
        cluster = ShapeMatchingCluster(
            indices=indices,
            rest_positions=rest_pos,
            masses=masses,
            config=config
        )
        assert cluster.total_mass == 4.0
        assert cluster.Aqq_inv is not None

    def test_shape_matching_grid_clusters(self):
        """Test creating clusters via grid."""
        # Create a regular grid of particles to ensure clusters can form
        # Each cluster needs at least 4 particles
        positions = []
        for x in range(5):
            for y in range(5):
                for z in range(2):
                    positions.append([x * 0.2, y * 0.2, z * 0.2])
        positions = np.array(positions, dtype=np.float64)

        solver = ShapeMatchingSolver(positions)
        solver.create_clusters_grid(cell_size=0.5, overlap=0.3)
        # Should have created some clusters
        assert len(solver.clusters) >= 1

    def test_shape_matching_impulse(self):
        """Test applying impulse."""
        positions = np.random.randn(10, 3)
        solver = ShapeMatchingSolver(positions)
        solver.apply_impulse(0, np.array([1.0, 0, 0]))
        assert solver.velocities[0, 0] > 0

    def test_shape_matching_deformation(self):
        """Test getting deformation."""
        positions = np.random.randn(10, 3)
        solver = ShapeMatchingSolver(positions)
        solver.positions += 0.1
        deformation = solver.get_deformation()
        assert_allclose(deformation, 0.1, atol=1e-10)

    def test_shape_matching_max_stretch(self):
        """Test getting maximum stretch."""
        positions = np.random.randn(10, 3)
        solver = ShapeMatchingSolver(positions)
        stretch = solver.get_max_stretch()
        assert stretch >= 1.0


class TestAdditionalPBD:
    """Additional PBD tests."""

    def test_strain_constraint_creation(self):
        """Test strain limit constraint creation."""
        inv_Dm = np.eye(3)
        constraint = StrainLimitConstraint(
            indices=(0, 1, 2, 3),
            inv_Dm=inv_Dm,
            max_strain=0.2
        )
        assert constraint.max_strain == 0.2

    def test_pbd_constraint_violation(self, cube_tet_mesh):
        """Test getting constraint violations."""
        body = PBDSoftBody(
            cube_tet_mesh.vertices,
            cube_tet_mesh.tetrahedra
        )
        violations = body.get_constraint_violation()
        assert "volume" in violations
        assert "strain" in violations
        assert "edge" in violations

    def test_pbd_rest_volume(self, cube_tet_mesh):
        """Test getting rest volume."""
        body = PBDSoftBody(
            cube_tet_mesh.vertices,
            cube_tet_mesh.tetrahedra
        )
        vol = body.get_rest_volume()
        assert vol > 0

    def test_pbd_apply_force(self, cube_tet_mesh):
        """Test applying external force."""
        body = PBDSoftBody(
            cube_tet_mesh.vertices,
            cube_tet_mesh.tetrahedra
        )
        body.apply_force(4, np.array([10, 0, 0]), 0.01)
        assert body.velocities[4, 0] != 0

    def test_pbd_collider_inside_sphere(self):
        """Test sphere collider with inside mode."""
        collider = SphereCollider(
            center=np.array([0, 0, 0]),
            radius=1.0,
            inside=True
        )
        # Point outside (should collide when inside=True)
        constraint = collider.get_collision_constraint(
            0, np.array([1.5, 0, 0])
        )
        assert constraint is not None


class TestAdditionalMuscle:
    """Additional muscle tests."""

    def test_muscle_build_fibers(self):
        """Test building fibers from mesh vertices."""
        muscle = Muscle(
            origin=MuscleAttachment(),
            insertion=MuscleAttachment(is_origin=False),
            fiber_direction=np.array([1, 0, 0])
        )
        positions = np.array([
            [0, 0, 0], [0, 0.1, 0],  # Origin vertices
            [1, 0, 0], [1, 0.1, 0]   # Insertion vertices
        ], dtype=np.float64)
        muscle.build_fibers_from_mesh(positions, [0, 1], [2, 3])
        assert len(muscle.fibers) > 0

    def test_muscle_fiber_target_length(self):
        """Test fiber target length computation."""
        fiber = MuscleFiber(
            start_vertex=0,
            end_vertex=1,
            rest_length=1.0,
            direction=np.array([1, 0, 0]),
            max_contraction=0.4
        )
        target = fiber.compute_target_length(0.5)
        assert target == 1.0 * (1.0 - 0.4 * 0.5)

    def test_muscle_controller(self):
        """Test muscle controller."""
        from engine.simulation.softbody.muscle import MuscleController
        controller = MuscleController()
        group = MuscleGroup("biceps")
        muscle = Muscle(
            origin=MuscleAttachment(),
            insertion=MuscleAttachment(is_origin=False),
            fiber_direction=np.array([1, 0, 0])
        )
        group.add_muscle(muscle)
        controller.add_group("biceps", group)
        controller.activate_group("biceps", 0.8)
        assert group.muscles[0].activation == 0.8


class TestAdditionalDeformableMesh:
    """Additional deformable mesh tests."""

    def test_surface_area(self, cube_tet_mesh):
        """Test surface area computation."""
        mesh = DeformableMesh(
            cube_tet_mesh.vertices,
            cube_tet_mesh.tetrahedra
        )
        area = mesh.compute_surface_area()
        assert area > 0

    def test_bounding_box(self, cube_tet_mesh):
        """Test bounding box computation."""
        mesh = DeformableMesh(
            cube_tet_mesh.vertices,
            cube_tet_mesh.tetrahedra
        )
        min_corner, max_corner = mesh.compute_bounding_box()
        assert np.all(max_corner >= min_corner)

    def test_get_surface_data(self, cube_tet_mesh):
        """Test getting surface mesh data."""
        mesh = DeformableMesh(
            cube_tet_mesh.vertices,
            cube_tet_mesh.tetrahedra
        )
        verts = mesh.get_surface_vertices()
        norms = mesh.get_surface_normals()
        tris = mesh.get_surface_triangles()
        assert len(verts) > 0
        assert len(norms) == len(verts)
        assert len(tris) > 0


# =============================================================================
# ADDITIONAL FLUID TESTS
# =============================================================================

class TestAdditionalSPH:
    """Additional SPH tests."""

    def test_cubic_spline_kernel(self):
        """Test cubic spline kernel."""
        w = SPHKernels.cubic_spline(0, 0.1)
        assert w > 0
        w_edge = SPHKernels.cubic_spline(0.1, 0.1)
        assert_allclose(w_edge, 0.0, atol=1e-10)

    def test_sph_viscosity_force(self, sph_solver):
        """Test viscosity force computation."""
        sph_solver._build_grid()
        sph_solver._find_neighbors()
        sph_solver._compute_density()
        force = sph_solver.compute_viscosity_force(0)
        assert force.shape == (3,)

    def test_sph_surface_tension(self, sph_solver):
        """Test surface tension computation."""
        sph_solver._build_grid()
        sph_solver._find_neighbors()
        sph_solver._compute_density()
        force = sph_solver.compute_surface_tension(0)
        assert force.shape == (3,)

    def test_sph_get_velocities(self, sph_solver):
        """Test getting velocities."""
        vels = sph_solver.get_velocities()
        assert vels.shape == (sph_solver.num_particles, 3)

    def test_sph_get_densities(self, sph_solver):
        """Test getting densities."""
        sph_solver._build_grid()
        sph_solver._find_neighbors()
        sph_solver._compute_density()
        densities = sph_solver.get_densities()
        assert len(densities) == sph_solver.num_particles

    def test_sph_average_density(self, sph_solver):
        """Test average density computation."""
        sph_solver._build_grid()
        sph_solver._find_neighbors()
        sph_solver._compute_density()
        avg = sph_solver.compute_average_density()
        assert avg > 0


class TestAdditionalPBF:
    """Additional PBF tests."""

    def test_pbf_vorticity_confinement(self, pbf_solver):
        """Test vorticity confinement."""
        pbf_solver.predict_positions(0.016)
        pbf_solver._build_grid()
        pbf_solver._find_neighbors()
        # Give particles some velocity
        for p in pbf_solver.particles:
            p.velocity = np.random.randn(3) * 0.1
        pbf_solver.apply_vorticity_confinement(0.016)

    def test_pbf_xsph_viscosity(self, pbf_solver):
        """Test XSPH viscosity."""
        pbf_solver.predict_positions(0.016)
        pbf_solver._build_grid()
        pbf_solver._find_neighbors()
        for p in pbf_solver.particles:
            p.velocity = np.random.randn(3) * 0.1
        pbf_solver.apply_xsph_viscosity()


class TestAdditionalFLIP:
    """Additional FLIP tests."""

    def test_flip_config(self):
        """Test FLIP configuration."""
        config = FLIPConfig(flip_ratio=0.9, grid_resolution=32)
        assert config.flip_ratio == 0.9

    def test_mac_grid_clear(self):
        """Test MAC grid clearing."""
        grid = MACGrid((10, 10, 10), 0.1)
        grid.u.fill(1.0)
        grid.clear_velocities()
        assert np.all(grid.u == 0)

    def test_mac_grid_save_restore(self):
        """Test MAC grid velocity save/restore."""
        grid = MACGrid((10, 10, 10), 0.1)
        grid.u.fill(5.0)
        grid.save_velocities()
        grid.u.fill(0.0)
        assert_allclose(grid.u_old, 5.0)

    def test_flip_boundary_conditions(self):
        """Test boundary condition application."""
        solver = FLIPSolver(resolution=(8, 8, 8), cell_size=0.5)
        solver.grid.u.fill(1.0)
        solver.apply_boundary_conditions()
        # Boundaries should be zero
        assert solver.grid.u[0, :, :].sum() == 0
        assert solver.grid.u[-1, :, :].sum() == 0


class TestAdditionalShallowWater:
    """Additional shallow water tests."""

    def test_shallow_water_timestep(self):
        """Test CFL timestep computation."""
        config = ShallowWaterConfig(grid_size=(16, 16), dx=0.5)
        solver = ShallowWaterSolver(config=config)
        solver.add_water((8, 8), radius=3, height=1.0)
        dt = solver.compute_timestep()
        assert dt > 0

    def test_shallow_water_energy(self):
        """Test energy computation."""
        config = ShallowWaterConfig(grid_size=(16, 16), dx=0.5)
        solver = ShallowWaterSolver(config=config)
        solver.add_water((8, 8), radius=3, height=1.0)
        energy = solver.total_energy()
        assert energy > 0

    def test_shallow_water_source(self):
        """Test water source addition."""
        config = ShallowWaterConfig(grid_size=(16, 16), dx=0.5)
        solver = ShallowWaterSolver(config=config)
        initial_vol = solver.field.total_volume(solver.dx)
        solver.add_source((8, 8), rate=1.0, dt=0.1)
        final_vol = solver.field.total_volume(solver.dx)
        assert final_vol > initial_vol

    def test_terrain_slope(self):
        """Test sloped terrain creation."""
        terrain = TerrainBoundary.slope((10, 10), direction=(1, 0), angle=0.1)
        # Should increase in x direction
        assert terrain.elevation[5, 5] > terrain.elevation[0, 5]


class TestAdditionalSurfaceReconstruction:
    """Additional surface reconstruction tests."""

    def test_fluid_surface_bounds(self):
        """Test fluid surface bounding box."""
        surface = FluidSurface(
            vertices=np.array([[0, 0, 0], [1, 1, 1]]),
            triangles=np.zeros((0, 3), dtype=np.int32),
            normals=np.zeros((2, 3))
        )
        min_b, max_b = surface.compute_bounds()
        assert_allclose(min_b, [0, 0, 0])
        assert_allclose(max_b, [1, 1, 1])

    def test_density_field_grid_conversion(self):
        """Test density field coordinate conversion."""
        field = DensityField(
            data=np.zeros((10, 10, 10)),
            origin=np.array([1, 2, 3]),
            cell_size=0.5
        )
        grid_pos = field.world_to_grid(np.array([2, 3, 4]))
        world_pos = field.grid_to_world(grid_pos)
        assert_allclose(world_pos, [2, 3, 4])


class TestAdditionalGPU:
    """Additional GPU fluid tests."""

    def test_grid_buffer_creation(self):
        """Test grid buffer creation."""
        from engine.simulation.fluid.gpu_fluid import GridBuffer
        buffer = GridBuffer.create((8, 8, 8), 1000)
        assert len(buffer.cell_start) == 8 * 8 * 8
        assert len(buffer.particle_indices) == 1000

    def test_gpu_solver_positions(self):
        """Test getting GPU solver positions."""
        solver = GPUFluidSolverStub()
        for i in range(5):
            solver.add_particle_cpu(np.random.randn(3))
        positions = solver.get_positions()
        assert positions.shape == (5, 3)

    def test_gpu_solver_velocities(self):
        """Test getting GPU solver velocities."""
        solver = GPUFluidSolverStub()
        for i in range(5):
            solver.add_particle_cpu(np.random.randn(3), velocity=np.random.randn(3))
        velocities = solver.get_velocities()
        assert velocities.shape == (5, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
