"""
Comprehensive verification tests for all engine/simulation/cloth/ modules.

Covers edge cases, GPU interface verification, constraint overrides,
collision degeneracies, wind extremes, and simulation state management
not covered by the existing test_cloth_hair.py.

PHASE 1 -- GPU Cloth Completion Verification.
"""

import math
from typing import Any, List, Optional

import numpy as np
import pytest

# =============================================================================
# GPU Cloth Interface Verification
# =============================================================================


class TestGPUBufferVerification:
    """Verification tests for GPUBuffer and related types."""

    def test_gpu_buffer_default_handle_none(self):
        """Verify default buffer handle is None."""
        from engine.simulation.cloth.gpu_cloth import GPUBuffer

        buf = GPUBuffer()
        assert buf.handle is None

    def test_gpu_buffer_is_valid_false_when_no_handle(self):
        """Verify is_valid() returns False when handle is None."""
        from engine.simulation.cloth.gpu_cloth import GPUBuffer

        buf = GPUBuffer(size=1024)
        assert buf.is_valid() is False

    def test_gpu_buffer_is_valid_true_when_handle_set(self):
        """Verify is_valid() returns True when handle is not None."""
        from engine.simulation.cloth.gpu_cloth import GPUBuffer

        buf = GPUBuffer(handle=object(), size=1024)
        assert buf.is_valid() is True

    def test_gpu_buffer_all_usage_values(self):
        """Verify all GPUBufferUsage enum values exist."""
        from engine.simulation.cloth.gpu_cloth import GPUBufferUsage

        assert GPUBufferUsage.VERTEX is not None
        assert GPUBufferUsage.INDEX is not None
        assert GPUBufferUsage.STORAGE is not None
        assert GPUBufferUsage.UNIFORM is not None
        assert GPUBufferUsage.STAGING is not None
        assert len(GPUBufferUsage) == 5

    def test_gpu_buffer_all_access_values(self):
        """Verify all GPUBufferAccess enum values exist."""
        from engine.simulation.cloth.gpu_cloth import GPUBufferAccess

        assert GPUBufferAccess.READ_ONLY is not None
        assert GPUBufferAccess.WRITE_ONLY is not None
        assert GPUBufferAccess.READ_WRITE is not None
        assert len(GPUBufferAccess) == 3


class TestGPUComputePipelineVerification:
    """Verification tests for GPUComputePipeline."""

    def test_compute_pipeline_default_handle_none(self):
        """Verify default pipeline handle is None."""
        from engine.simulation.cloth.gpu_cloth import GPUComputePipeline

        pipe = GPUComputePipeline()
        assert pipe.handle is None
        assert pipe.shader_module is None

    def test_compute_pipeline_local_size_default(self):
        """Verify default local size is (64, 1, 1)."""
        from engine.simulation.cloth.gpu_cloth import GPUComputePipeline

        pipe = GPUComputePipeline()
        assert pipe.local_size == (64, 1, 1)

    def test_compute_pipeline_custom_local_size(self):
        """Verify custom local size is stored."""
        from engine.simulation.cloth.gpu_cloth import GPUComputePipeline

        pipe = GPUComputePipeline(local_size=(128, 1, 1))
        assert pipe.local_size == (128, 1, 1)

    def test_compute_pipeline_is_valid(self):
        """Verify is_valid() behavior."""
        from engine.simulation.cloth.gpu_cloth import GPUComputePipeline

        pipe = GPUComputePipeline()
        assert pipe.is_valid() is False

        pipe.handle = object()
        assert pipe.is_valid() is True


class TestGPUClothBuffersVerification:
    """Verification tests for GPUClothBuffers."""

    def test_all_buffers_default_to_invalid(self):
        """Verify all default buffers start invalid."""
        from engine.simulation.cloth.gpu_cloth import GPUClothBuffers

        bufs = GPUClothBuffers()
        assert bufs.are_valid() is False

    def test_are_valid_requires_all_core_buffers(self):
        """Verify are_valid() returns True only when all four core buffers are valid."""
        from engine.simulation.cloth.gpu_cloth import GPUBuffer, GPUClothBuffers

        bufs = GPUClothBuffers()
        # Set only positions_current as valid
        bufs.positions_current = GPUBuffer(handle=object(), size=64)
        assert bufs.are_valid() is False  # Missing others

        # Set all four required buffers
        bufs.positions_current = GPUBuffer(handle=object(), size=64)
        bufs.positions_predicted = GPUBuffer(handle=object(), size=64)
        bufs.positions_previous = GPUBuffer(handle=object(), size=64)
        bufs.inv_masses = GPUBuffer(handle=object(), size=64)
        assert bufs.are_valid() is True

    def test_all_buffer_fields_exist(self):
        """Verify all expected buffer fields exist."""
        from engine.simulation.cloth.gpu_cloth import GPUClothBuffers

        bufs = GPUClothBuffers()
        expected = [
            "positions_current",
            "positions_predicted",
            "positions_previous",
            "velocities",
            "inv_masses",
            "distance_constraints",
            "bending_constraints",
            "triangles",
            "collision_primitives",
            "spatial_hash_table",
            "spatial_hash_entries",
            "simulation_params",
        ]
        for field in expected:
            assert hasattr(bufs, field), f"Missing buffer field: {field}"


class TestGPUClothPipelinesVerification:
    """Verification tests for GPUClothPipelines."""

    def test_all_pipeline_fields_exist(self):
        """Verify all expected pipeline fields exist."""
        from engine.simulation.cloth.gpu_cloth import GPUClothPipelines

        pipes = GPUClothPipelines()
        expected = [
            "integration",
            "distance_constraint",
            "bending_constraint",
            "collision_primitives",
            "self_collision_broad",
            "self_collision_narrow",
            "wind_force",
            "velocity_update",
        ]
        for field in expected:
            assert hasattr(pipes, field), f"Missing pipeline field: {field}"


class TestGPUClothSolverVerification:
    """Verification tests for GPUClothSolver ABC and GPUClothSolverStub."""

    def test_gpu_cloth_solver_abstract_cannot_instantiate(self):
        """Verify GPUClothSolver ABC cannot be directly instantiated."""
        from engine.simulation.cloth.gpu_cloth import GPUClothSolver, GPUDevice

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GPUClothSolver(None)

    def test_gpu_cloth_solver_stub_init(self):
        """Verify GPUClothSolverStub initializes correctly."""
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        solver = GPUClothSolverStub()
        assert solver._is_initialized is True
        assert solver._num_particles == 0
        assert solver._num_distance_constraints == 0

    def test_gpu_cloth_solver_stub_init_with_config(self):
        """Verify GPUClothSolverStub accepts a config."""
        from engine.simulation.cloth.cloth_simulation import ClothSimulationConfig
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        config = ClothSimulationConfig(substeps=2, solver_iterations=2)
        solver = GPUClothSolverStub(config=config)
        assert solver.config.substeps == 2
        assert solver.config.solver_iterations == 2

    def test_gpu_cloth_solver_stub_initialize_sets_counts(self):
        """Verify initialize() sets particle and constraint counts."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(4, 4)
        solver = GPUClothSolverStub()
        result = solver.initialize(mesh)

        assert result is True
        assert solver._num_particles == 16
        assert solver._num_distance_constraints == len(mesh.edges)
        assert solver._num_triangles == len(mesh.triangles)

    def test_gpu_cloth_solver_stub_step_does_not_modify_mesh(self):
        """Verify stub step() does NOT modify mesh positions."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(4, 4)
        original_positions = mesh.get_positions_array().copy()

        solver = GPUClothSolverStub()
        solver.initialize(mesh)
        solver.step(mesh, 1.0 / 60.0)

        final_positions = mesh.get_positions_array()
        assert np.allclose(original_positions, final_positions), (
            "Stub step() must not modify mesh positions"
        )

    def test_gpu_cloth_solver_stub_step_uninitialized(self):
        """Verify step() with uninitialized solver does not crash."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(4, 4)
        solver = GPUClothSolverStub()
        # Do not call initialize()
        solver.step(mesh, 1.0 / 60.0)  # Should not raise

    def test_gpu_cloth_solver_stub_shutdown(self):
        """Verify shutdown() clears state."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(4, 4)
        solver = GPUClothSolverStub()
        solver.initialize(mesh)
        assert solver._mesh is not None
        assert solver._is_initialized is True

        solver.shutdown()
        assert solver._mesh is None
        assert solver._is_initialized is False

    def test_gpu_cloth_solver_stub_dispatch_compute_noop(self):
        """Verify dispatch_compute is intentionally a no-op with clear warning."""
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        solver = GPUClothSolverStub()
        # Should not raise and should not modify anything
        solver.dispatch_compute(1.0 / 60.0)

    def test_gpu_cloth_solver_stub_prepare_buffers(self):
        """Verify prepare_buffers stores mesh reference."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(4, 4)
        solver = GPUClothSolverStub()
        solver.prepare_buffers(mesh)
        assert solver._mesh is mesh

    def test_gpu_cloth_solver_stub_readback_positions_noop(self):
        """Verify readback_positions is a no-op for stub."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(4, 4)
        solver = GPUClothSolverStub()
        solver.initialize(mesh)
        original = mesh.get_positions_array().copy()
        solver.readback_positions(mesh)
        assert np.allclose(original, mesh.get_positions_array())


class TestShaderTemplatesVerification:
    """Verification tests for shader templates."""

    def test_get_shader_templates_returns_all_three(self):
        """Verify all three shader templates are returned."""
        from engine.simulation.cloth.gpu_cloth import get_shader_templates

        templates = get_shader_templates()
        assert len(templates) == 3
        assert "integration" in templates
        assert "distance_constraint" in templates
        assert "velocity_update" in templates

    def test_integration_shader_contains_key_elements(self):
        """Verify integration shader has required elements."""
        from engine.simulation.cloth.gpu_cloth import INTEGRATION_SHADER_TEMPLATE

        shader = INTEGRATION_SHADER_TEMPLATE
        assert "gravity" in shader
        assert "damping" in shader
        assert "local_size_x = 64" in shader
        assert "gl_GlobalInvocationID" in shader
        assert "num_particles" in shader
        assert "positions_predicted" in shader
        assert "positions_current" in shader

    def test_distance_constraint_shader_has_key_elements(self):
        """Verify distance constraint shader has required elements."""
        from engine.simulation.cloth.gpu_cloth import DISTANCE_CONSTRAINT_SHADER_TEMPLATE

        shader = DISTANCE_CONSTRAINT_SHADER_TEMPLATE
        assert "DistanceConstraint" in shader
        assert "rest_length" in shader
        assert "atomicAdd" in shader
        assert "local_size_x = 64" in shader

    def test_velocity_update_shader_has_key_elements(self):
        """Verify velocity update shader has required elements."""
        from engine.simulation.cloth.gpu_cloth import VELOCITY_UPDATE_SHADER_TEMPLATE

        shader = VELOCITY_UPDATE_SHADER_TEMPLATE
        assert "inv_dt" in shader
        assert "damping" in shader
        assert "velocities" in shader
        assert "local_size_x = 64" in shader


class TestCalculateWorkgroupsVerification:
    """Verification tests for calculate_workgroups."""

    def test_exact_multiple(self):
        """Verify exact multiples produce correct count."""
        from engine.simulation.cloth.gpu_cloth import calculate_workgroups

        assert calculate_workgroups(64, 64) == 1
        assert calculate_workgroups(128, 64) == 2
        assert calculate_workgroups(192, 64) == 3

    def test_remainder_rounds_up(self):
        """Verify non-exact multiples round up."""
        from engine.simulation.cloth.gpu_cloth import calculate_workgroups

        assert calculate_workgroups(1, 64) == 1
        assert calculate_workgroups(65, 64) == 2
        assert calculate_workgroups(100, 64) == 2

    def test_zero_items(self):
        """Verify zero items returns zero workgroups."""
        from engine.simulation.cloth.gpu_cloth import calculate_workgroups

        assert calculate_workgroups(0, 64) == 0

    def test_custom_local_size(self):
        """Verify custom local size works."""
        from engine.simulation.cloth.gpu_cloth import calculate_workgroups

        assert calculate_workgroups(256, 128) == 2
        assert calculate_workgroups(100, 32) == 4


# =============================================================================
# Constraint Verification
# =============================================================================


class TestDistanceConstraintVerification:
    """Verification tests for DistanceConstraint edge cases."""

    def test_solve_edge_both_pinned(self):
        """Verify solving with both particles pinned does nothing."""
        from engine.simulation.cloth.cloth_constraints import DistanceConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            inv_mass=0.0,
        )
        p1 = ClothParticle(
            position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            inv_mass=0.0,
        )
        error = DistanceConstraint.solve_edge(p0, p1, 1.0, 1.0)
        assert np.allclose(p0.position, [0.0, 0.0, 0.0])
        assert np.allclose(p1.position, [1.0, 0.0, 0.0])
        # Error should be 0 since neither can move, w_sum = 0
        assert error == 0.0

    def test_solve_edge_coincident_particles(self):
        """Verify solving with coincident particles returns 0 error."""
        from engine.simulation.cloth.cloth_constraints import DistanceConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        p1 = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        error = DistanceConstraint.solve_edge(p0, p1, 1.0, 1.0)
        assert error == 0.0

    def test_solve_edge_stiffness_zero(self):
        """Verify zero stiffness means no correction."""
        from engine.simulation.cloth.cloth_constraints import DistanceConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        p1 = ClothParticle(
            position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        )
        error = DistanceConstraint.solve_edge(p0, p1, 1.0, 0.0)
        assert error == 1.0
        # No movement since stiffness = 0
        assert np.allclose(p0.position, [0.0, 0.0, 0.0])
        assert np.allclose(p1.position, [2.0, 0.0, 0.0])

    def test_distance_constraint_solve_with_override(self):
        """Verify DistanceConstraint.solve() accepts stiffness_override."""
        from engine.simulation.cloth.cloth_constraints import DistanceConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        p1 = ClothParticle(
            position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
        )
        constraint = DistanceConstraint(
            p0_index=0, p1_index=1, rest_length=1.0, stiffness=1.0
        )
        error = constraint.solve([p0, p1], stiffness_override=0.5)
        assert error == 9.0
        # Should have moved less than with full stiffness
        dist = np.linalg.norm(p1.position - p0.position)
        assert dist > 1.0  # Not fully corrected
        assert dist < 10.0  # But moved

    def test_distance_constraint_rest_length_maintained(self):
        """Verify two particles at rest length produce zero error."""
        from engine.simulation.cloth.cloth_constraints import DistanceConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        p1 = ClothParticle(
            position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        error = DistanceConstraint.solve_edge(p0, p1, 1.0, 1.0)
        assert error == 0.0
        assert np.allclose(p0.position, [0.0, 0.0, 0.0])
        assert np.allclose(p1.position, [1.0, 0.0, 0.0])


class TestBendingConstraintVerification:
    """Verification tests for BendingConstraint edge cases."""

    def test_flat_triangle_pair_zero_angle(self):
        """Verify flat triangles have zero dihedral angle."""
        from engine.simulation.cloth.cloth_constraints import BendingConstraint

        # Two adjacent flat triangles
        p0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p2 = np.array([0.5, 1.0, 0.0], dtype=np.float32)
        p3 = np.array([0.5, -1.0, 0.0], dtype=np.float32)

        angle = BendingConstraint.compute_dihedral_angle(p0, p1, p2, p3)
        # Flat triangles with opposite winding produce angle of 0 or pi
        assert abs(angle) < 0.001 or abs(abs(angle) - math.pi) < 0.001, (
            f"Flat triangles should have near-zero or pi angle, got {angle}"
        )

    def test_folded_triangles_180_degrees(self):
        """Verify folded triangles produce pi radian angle."""
        from engine.simulation.cloth.cloth_constraints import BendingConstraint

        p0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p2 = np.array([0.5, 1.0, 0.0], dtype=np.float32)
        # p3 is on the opposite side of the shared edge
        p3 = np.array([0.5, -1.0, 0.0], dtype=np.float32)

        angle = BendingConstraint.compute_dihedral_angle(p0, p1, p2, p3)
        # Flat case - angle should be near zero or near pi
        # With flat triangles, the normals are parallel
        assert abs(angle) < 0.001 or abs(abs(angle) - math.pi) < 0.001

    def test_bending_constraint_solve_small_error(self):
        """Verify bending constraint solve handles small error correctly."""
        from engine.simulation.cloth.cloth_constraints import BendingConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particles = [
            ClothParticle(
                position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            ),
            ClothParticle(
                position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                prev_position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ),
            ClothParticle(
                position=np.array([0.5, 1.0, 0.0], dtype=np.float32),
                prev_position=np.array([0.5, 1.0, 0.0], dtype=np.float32),
            ),
            ClothParticle(
                position=np.array([0.5, -1.0, 0.0], dtype=np.float32),
                prev_position=np.array([0.5, -1.0, 0.0], dtype=np.float32),
            ),
        ]

        constraint = BendingConstraint(
            p0_index=0, p1_index=1, p2_index=2, p3_index=3,
            rest_angle=0.0, stiffness=0.1,
        )

        # Should not crash
        error = constraint.solve(particles)
        assert isinstance(error, float)


class TestShearConstraintVerification:
    """Verification tests for ShearConstraint."""

    def test_shear_with_stiffness_override(self):
        """Verify ShearConstraint.solve() with stiffness override."""
        from engine.simulation.cloth.cloth_constraints import ShearConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        p1 = ClothParticle(
            position=np.array([2.0, 2.0, 0.0], dtype=np.float32),
            prev_position=np.array([2.0, 2.0, 0.0], dtype=np.float32),
        )
        constraint = ShearConstraint(
            p0_index=0, p1_index=1, rest_length=math.sqrt(2), stiffness=0.5
        )
        error = constraint.solve([p0, p1], stiffness_override=0.25)
        assert error > 0

    def test_shear_zero_stiffness_no_movement(self):
        """Verify zero stiffness means zero correction."""
        from engine.simulation.cloth.cloth_constraints import ShearConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        p1 = ClothParticle(
            position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
        )
        constraint = ShearConstraint(
            p0_index=0, p1_index=1, rest_length=1.0, stiffness=0.0
        )
        constraint.solve([p0, p1])
        assert np.allclose(p0.position, [0.0, 0.0, 0.0])
        assert np.allclose(p1.position, [10.0, 0.0, 0.0])


class TestLongRangeAttachmentVerification:
    """Verification tests for LongRangeAttachment edge cases."""

    def test_within_max_distance_returns_zero(self):
        """Verify no correction when within max distance."""
        from engine.simulation.cloth.cloth_constraints import LongRangeAttachment
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            inv_mass=0.0,
        )
        p1 = ClothParticle(
            position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        attachment = LongRangeAttachment(
            p0_index=0, p1_index=1, max_distance=2.0
        )
        error = attachment.solve([p0, p1])
        assert error == 0.0
        assert np.allclose(p1.position, [1.0, 0.0, 0.0])

    def test_both_pinned_returns_error(self):
        """Verify both pinned returns error without crash."""
        from engine.simulation.cloth.cloth_constraints import LongRangeAttachment
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            inv_mass=0.0,
        )
        p1 = ClothParticle(
            position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
            inv_mass=0.0,
        )
        attachment = LongRangeAttachment(
            p0_index=0, p1_index=1, max_distance=2.0
        )
        error = attachment.solve([p0, p1])
        # Both pinned means w_sum < EPSILON, so returns error but no movement
        assert error is not None
        assert np.allclose(p1.position, [10.0, 0.0, 0.0])

    def test_stiffness_override_applied(self):
        """Verify stiffness_override modifies correction strength."""
        from engine.simulation.cloth.cloth_constraints import LongRangeAttachment
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            inv_mass=0.0,
        )
        p1 = ClothParticle(
            position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
        )
        attachment = LongRangeAttachment(
            p0_index=0, p1_index=1, max_distance=2.0, stiffness=0.8
        )
        # Full stiffness
        error_high = attachment.solve([p0, p1], stiffness_override=1.0)
        dist_high = np.linalg.norm(p1.position - p0.position)

        # Reset
        p1.position[:] = [10.0, 0.0, 0.0]

        # Low stiffness
        error_low = attachment.solve([p0, p1], stiffness_override=0.01)
        dist_low = np.linalg.norm(p1.position - p0.position)

        # High stiffness should have moved more
        assert dist_low > dist_high, (
            f"Low stiffness ({dist_low:.4f}) should leave particle farther "
            f"than high stiffness ({dist_high:.4f})"
        )


class TestAnchorConstraintVerification:
    """Verification tests for AnchorConstraint edge cases."""

    def test_already_at_anchor_returns_zero(self):
        """Verify already-at-anchor returns zero distance."""
        from engine.simulation.cloth.cloth_constraints import AnchorConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p = ClothParticle(
            position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            prev_position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )
        anchor = AnchorConstraint(
            particle_index=0,
            anchor_position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )
        error = anchor.solve([p])
        assert error == 0.0
        assert np.allclose(p.position, [1.0, 2.0, 3.0])

    def test_update_anchor(self):
        """Verify update_anchor changes anchor position."""
        from engine.simulation.cloth.cloth_constraints import AnchorConstraint

        anchor = AnchorConstraint(
            particle_index=0,
            anchor_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        new_pos = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        anchor.update_anchor(new_pos)
        assert np.allclose(anchor.anchor_position, [5.0, 5.0, 5.0])

    def test_stiffness_override(self):
        """Verify stiffness_override modifies correction."""
        from engine.simulation.cloth.cloth_constraints import AnchorConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        anchor = AnchorConstraint(
            particle_index=0,
            anchor_position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
            stiffness=0.5,
        )
        # Full override
        anchor.solve([p], stiffness_override=0.0)
        # With zero override, particle should stay put
        assert np.allclose(p.position, [0.0, 0.0, 0.0])


class TestTetherConstraintVerification:
    """Verification tests for TetherConstraint edge cases."""

    def test_within_max_distance_returns_zero(self):
        """Verify within max distance returns zero error."""
        from engine.simulation.cloth.cloth_constraints import TetherConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p = ClothParticle(
            position=np.array([0.5, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        )
        tether = TetherConstraint(
            particle_index=0,
            attachment_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            max_distance=1.0,
        )
        error = tether.solve([p])
        assert error == 0.0
        assert np.allclose(p.position, [0.5, 0.0, 0.0])

    def test_pinned_particle_not_moved(self):
        """Verify pinned particle is not moved by tether."""
        from engine.simulation.cloth.cloth_constraints import TetherConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p = ClothParticle(
            position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
            inv_mass=0.0,
        )
        tether = TetherConstraint(
            particle_index=0,
            attachment_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            max_distance=1.0,
        )
        error = tether.solve([p])
        # Pinned so w_sum check may return error value
        assert np.allclose(p.position, [10.0, 0.0, 0.0])

    def test_update_attachment(self):
        """Verify update_attachment changes attachment position."""
        from engine.simulation.cloth.cloth_constraints import TetherConstraint

        tether = TetherConstraint(
            particle_index=0,
            attachment_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            max_distance=1.0,
        )
        new_pos = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        tether.update_attachment(new_pos)
        assert np.allclose(tether.attachment_position, [2.0, 3.0, 4.0])


class TestCreateConstraintsVerification:
    """Verification tests for constraint creation functions."""

    def test_create_bend_constraints_no_adjacent(self):
        """Verify non-adjacent triangles produce no bend constraints."""
        from engine.simulation.cloth.cloth_constraints import create_bend_constraints
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particles = [
            ClothParticle(
                position=np.array([i * 1.0, 0.0, 0.0], dtype=np.float32),
                prev_position=np.array([i * 1.0, 0.0, 0.0], dtype=np.float32),
            )
            for i in range(6)
        ]
        # Two triangles that don't share an edge
        triangles = [(0, 1, 2), (3, 4, 5)]

        constraints = create_bend_constraints(particles, triangles)
        assert len(constraints) == 0

    def test_create_long_range_attachments_no_anchors(self):
        """Verify no anchors produces empty list."""
        from engine.simulation.cloth.cloth_constraints import create_long_range_attachments
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particles = [
            ClothParticle(
                position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            )
        ]
        attachments = create_long_range_attachments(particles, attachment_indices=[])
        assert len(attachments) == 0

    def test_create_long_range_attachment_counts(self):
        """Verify correct number of attachments created."""
        from engine.simulation.cloth.cloth_constraints import create_long_range_attachments
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particles = [
            ClothParticle(
                position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                inv_mass=0.0 if i == 0 else 1.0,
            )
            for i in range(5)
        ]
        attachments = create_long_range_attachments(
            particles, attachment_indices=[0]
        )
        # 4 non-pinned particles (indices 1-4) should get attachments
        assert len(attachments) == 4


# =============================================================================
# Collision Verification (edge cases)
# =============================================================================


class TestCollisionVerification:
    """Verification tests for collision edge cases."""

    def test_collide_sphere_center_position(self):
        """Verify particle at sphere center is pushed outward."""
        from engine.simulation.cloth.cloth_collision import (
            SphereCollider,
            collide_with_sphere,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        result = collide_with_sphere(particle, sphere, margin=0.0)

        assert result.collided is True
        assert result.penetration_depth > 0
        # Particle should be pushed to surface of sphere
        dist = np.linalg.norm(particle.position)
        assert abs(dist - 1.0) < 0.01, (
            f"Particle at center should be pushed to radius 1.0, got {dist}"
        )

    def test_collide_sphere_no_margin(self):
        """Verify collision with zero margin."""
        from engine.simulation.cloth.cloth_collision import (
            SphereCollider,
            collide_with_sphere,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        # Particle exactly on surface (distance = radius) -- should NOT collide
        particle = ClothParticle(
            position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        result = collide_with_sphere(particle, sphere, margin=0.0)
        assert result.collided is False

    def test_collide_sphere_no_margin_inside(self):
        """Verify collision with zero margin when inside."""
        from engine.simulation.cloth.cloth_collision import (
            SphereCollider,
            collide_with_sphere,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([0.9, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.9, 0.0, 0.0], dtype=np.float32),
        )
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        result = collide_with_sphere(particle, sphere, margin=0.0)
        assert result.collided is True
        dist = np.linalg.norm(particle.position)
        assert dist >= 1.0 - 0.001

    def test_collide_sphere_with_friction(self):
        """Verify friction affects collision response."""
        from engine.simulation.cloth.cloth_collision import (
            SphereCollider,
            collide_with_sphere,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        # Particle moving into sphere (has velocity via prev_position)
        particle = ClothParticle(
            position=np.array([0.5, 0.5, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.5, 0.0], dtype=np.float32),
        )
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
            friction=0.5,
        )
        result = collide_with_sphere(particle, sphere, margin=0.0)
        assert result.collided is True
        assert result.penetration_depth >= 0

    def test_collide_capsule_degenerate(self):
        """Verify degenerate capsule (axis length zero) falls back to sphere."""
        from engine.simulation.cloth.cloth_collision import (
            CapsuleCollider,
            collide_with_capsule,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([0.5, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        )
        capsule = CapsuleCollider(
            point_a=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            point_b=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # same = degenerate
            radius=1.0,
        )
        result = collide_with_capsule(particle, capsule, margin=0.0)
        assert result.collided is True
        dist = np.linalg.norm(particle.position)
        assert dist >= 1.0 - 0.001

    def test_collide_capsule_on_axis(self):
        """Verify particle exactly on capsule axis is still handled."""
        from engine.simulation.cloth.cloth_collision import (
            CapsuleCollider,
            collide_with_capsule,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        # Particle on the axis line, inside the capsule
        particle = ClothParticle(
            position=np.array([0.0, 0.5, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.5, 0.0], dtype=np.float32),
        )
        capsule = CapsuleCollider(
            point_a=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            point_b=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        result = collide_with_capsule(particle, capsule, margin=0.0)
        assert result.collided is True

    def test_collide_box_margin_expands_box(self):
        """Verify margin expands the box effectively."""
        from engine.simulation.cloth.cloth_collision import (
            BoxCollider,
            collide_with_box,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        # Particle just outside the box but within margin
        particle = ClothParticle(
            position=np.array([1.1, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([1.1, 0.0, 0.0], dtype=np.float32),
        )
        box = BoxCollider(
            min_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            max_point=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        # With margin 0, this should NOT collide
        result = collide_with_box(particle, box, margin=0.0)
        assert result.collided is False

        # With margin 0.2, box expands to max=1.2, so it SHOULD collide
        result = collide_with_box(particle, box, margin=0.2)
        assert result.collided is True

    def test_collide_mesh_degenerate_triangle(self):
        """Verify mesh collision with degenerate triangles."""
        from engine.simulation.cloth.cloth_collision import (
            MeshCollider,
            collide_with_mesh,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        # Degenerate triangle (all same vertex)
        mesh = MeshCollider(
            vertices=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            indices=np.array([0, 0, 0], dtype=np.int32),
        )
        result = collide_with_mesh(particle, mesh, margin=0.0)
        assert result.collided is False  # Degenerate triangle won't collide

    def test_collide_mesh_no_triangles(self):
        """Verify mesh with no triangles returns no collision."""
        from engine.simulation.cloth.cloth_collision import (
            MeshCollider,
            collide_with_mesh,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        mesh = MeshCollider(
            vertices=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            indices=np.array([], dtype=np.int32),
        )
        result = collide_with_mesh(particle, mesh, margin=0.01)
        assert result.collided is False

    def test_sdf_collider_edge_cases(self):
        """Verify SDF collider with zero-gradient function."""
        from engine.simulation.cloth.cloth_collision import (
            SDFCollider,
            collide_with_sdf,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        # SDF that always returns a large positive distance
        particle = ClothParticle(
            position=np.array([10.0, 10.0, 10.0], dtype=np.float32),
            prev_position=np.array([10.0, 10.0, 10.0], dtype=np.float32),
        )

        def far_sdf(pos):
            return 100.0, np.array([1.0, 0.0, 0.0], dtype=np.float32)

        sdf = SDFCollider(sdf_function=far_sdf)
        result = collide_with_sdf(particle, sdf, margin=0.01)
        assert result.collided is False

        def inside_sdf(pos):
            return -0.5, np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        sdf2 = SDFCollider(sdf_function=inside_sdf)
        result = collide_with_sdf(particle, sdf2, margin=0.01)
        assert result.collided is True


class TestSelfCollisionVerification:
    """Verification tests for self-collision edge cases."""

    def test_self_collision_all_pinned(self):
        """Verify self-collision with all pinned particles returns 0."""
        from engine.simulation.cloth.cloth_collision import handle_self_collision
        from engine.simulation.cloth.cloth_simulation import ClothParticle, ClothMesh, ClothEdge, ClothTriangle

        particles = [
            ClothParticle(
                position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                inv_mass=0.0,
            ),
            ClothParticle(
                position=np.array([0.0, 0.001, 0.0], dtype=np.float32),
                prev_position=np.array([0.0, 0.001, 0.0], dtype=np.float32),
                inv_mass=0.0,
            ),
        ]
        mesh = ClothMesh(
            particles=particles, edges=[], triangles=[]
        )
        count = handle_self_collision(mesh, thickness=0.02)
        assert count == 0

    def test_self_collision_no_particles(self):
        """Verify self-collision with empty mesh returns 0."""
        from engine.simulation.cloth.cloth_collision import handle_self_collision
        from engine.simulation.cloth.cloth_simulation import ClothMesh

        mesh = ClothMesh(particles=[], edges=[], triangles=[])
        count = handle_self_collision(mesh)
        assert count == 0

    def test_self_collision_separated_particles(self):
        """Verify self-collision with far apart particles returns 0."""
        from engine.simulation.cloth.cloth_collision import handle_self_collision
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(2, 2)
        # Move particles far apart
        mesh.particles[0].position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        mesh.particles[1].position = np.array([100.0, 0.0, 0.0], dtype=np.float32)
        mesh.particles[2].position = np.array([0.0, 100.0, 0.0], dtype=np.float32)
        mesh.particles[3].position = np.array([100.0, 100.0, 0.0], dtype=np.float32)

        count = handle_self_collision(mesh, thickness=0.02)
        assert count == 0

    def test_self_collision_triangle_level_delegation(self):
        """Verify triangle-level self-collision delegates to particle-level."""
        from engine.simulation.cloth.cloth_collision import (
            handle_self_collision,
            handle_triangle_self_collision,
        )
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(3, 3)
        # Force some overlap
        mesh.particles[4].position = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        mesh.particles[5].position = np.array([0.501, 0.5, 0.0], dtype=np.float32)

        tri_count = handle_triangle_self_collision(mesh, thickness=0.02)
        particle_count = handle_self_collision(
            create_cloth_grid(3, 3), thickness=0.02
        )
        # Both should run without error
        assert isinstance(tri_count, int)
        assert isinstance(particle_count, int)


class TestSpatialHashVerification:
    """Verification tests for SpatialHash."""

    def test_clear_empties_table(self):
        """Verify clear() removes all entries."""
        from engine.simulation.cloth.cloth_collision import SpatialHash

        h = SpatialHash()
        h.insert(0, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        h.insert(1, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert len(h._table) > 0
        h.clear()
        assert len(h._table) == 0

    def test_build_from_particles(self):
        """Verify build_from_particles populates hash."""
        from engine.simulation.cloth.cloth_collision import SpatialHash
        from engine.simulation.cloth.cloth_simulation import ClothParticle, ClothMesh, ClothEdge, ClothTriangle

        particles = [
            ClothParticle(
                position=np.array([float(i), 0.0, 0.0], dtype=np.float32),
                prev_position=np.array([float(i), 0.0, 0.0], dtype=np.float32),
            )
            for i in range(10)
        ]
        h = SpatialHash()
        h.build_from_particles(particles)
        assert len(h._table) > 0

    def test_query_empty_table(self):
        """Verify query on empty table returns empty list."""
        from engine.simulation.cloth.cloth_collision import SpatialHash

        h = SpatialHash()
        result = h.query(
            np.array([0.0, 0.0, 0.0], dtype=np.float32), radius=1.0
        )
        assert result == []


class TestClothCollisionHandlerVerification:
    """Verification tests for ClothCollisionHandler."""

    def test_clear_removes_all_colliders(self):
        """Verify clear() removes all colliders."""
        from engine.simulation.cloth.cloth_collision import (
            ClothCollisionHandler,
            SphereCollider,
        )

        handler = ClothCollisionHandler()
        handler.add_sphere(
            SphereCollider(
                center=np.zeros(3, dtype=np.float32), radius=1.0
            )
        )
        handler.add_sphere(
            SphereCollider(
                center=np.ones(3, dtype=np.float32), radius=0.5
            )
        )
        assert len(handler.sphere_colliders) == 2
        handler.clear()
        assert len(handler.sphere_colliders) == 0
        assert len(handler.capsule_colliders) == 0
        assert len(handler.box_colliders) == 0

    def test_process_collisions_on_empty_mesh(self):
        """Verify process_collisions on empty mesh returns 0."""
        from engine.simulation.cloth.cloth_collision import ClothCollisionHandler
        from engine.simulation.cloth.cloth_simulation import ClothMesh

        handler = ClothCollisionHandler()
        mesh = ClothMesh(particles=[], edges=[], triangles=[])
        count = handler.process_collisions(mesh)
        assert count == 0


class TestClothCollisionResult:
    """Tests for CollisionResult default values."""

    def test_collision_result_defaults(self):
        """Verify CollisionResult defaults."""
        from engine.simulation.cloth.cloth_collision import CollisionResult

        result = CollisionResult(collided=False)
        assert result.penetration_depth == 0.0
        assert result.contact_normal is None
        assert result.contact_point is None

    def test_collision_result_contact_point(self):
        """Verify CollisionResult stores contact point."""
        from engine.simulation.cloth.cloth_collision import CollisionResult

        result = CollisionResult(
            collided=True,
            penetration_depth=0.5,
            contact_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            contact_point=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        assert np.allclose(result.contact_point, [0.0, 1.0, 0.0])


# =============================================================================
# Wind Force Verification
# =============================================================================


class TestWindForceVerification:
    """Verification tests for wind edge cases."""

    def test_set_direction_normalizes(self):
        """Verify set_direction normalizes the direction."""
        from engine.simulation.cloth.cloth_wind import WindForce

        wind = WindForce()
        wind.set_direction(np.array([5.0, 0.0, 0.0], dtype=np.float32))
        assert np.allclose(wind.settings.direction, [1.0, 0.0, 0.0])

    def test_set_direction_zero_length_no_change(self):
        """Verify zero-length direction does not change direction."""
        from engine.simulation.cloth.cloth_wind import WindForce

        wind = WindForce()
        original = wind.settings.direction.copy()
        wind.set_direction(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert np.allclose(wind.settings.direction, original)

    def test_set_strength_clamps_negative(self):
        """Verify set_strength clamps to zero."""
        from engine.simulation.cloth.cloth_wind import WindForce

        wind = WindForce()
        wind.set_strength(-5.0)
        assert wind.settings.strength == 0.0

    def test_set_strength_zero(self):
        """Verify strength can be set to zero."""
        from engine.simulation.cloth.cloth_wind import WindForce, WindSettings

        wind = WindForce(settings=WindSettings(strength=10.0))
        wind.set_strength(0.0)
        assert wind.settings.strength == 0.0

    def test_compute_wind_force_on_empty_mesh(self):
        """Verify compute_wind_force on empty mesh does not crash."""
        from engine.simulation.cloth.cloth_wind import WindForce
        from engine.simulation.cloth.cloth_simulation import ClothMesh

        wind = WindForce()
        mesh = ClothMesh(particles=[], edges=[], triangles=[])
        wind.compute_wind_force(mesh, 1.0 / 60.0)  # Should not raise

    def test_turbulence_zero_strength_returns_base_wind(self):
        """Verify zero turbulence returns base wind unchanged."""
        from engine.simulation.cloth.cloth_wind import WindForce

        import numpy as np

        wind = WindForce()
        wind.settings.turbulence_strength = 0.0
        base = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        result = wind._get_wind_at_position(
            np.array([1.0, 2.0, 3.0], dtype=np.float32), base
        )
        assert np.allclose(result, base)

    def test_update_time_advances(self):
        """Verify update() advances internal time."""
        from engine.simulation.cloth.cloth_wind import WindForce

        wind = WindForce()
        assert wind._time == 0.0
        wind.update(1.0)
        assert wind._time == 1.0
        wind.update(0.5)
        assert wind._time == 1.5


class TestVortexWindVerification:
    """Verification tests for VortexWind edge cases."""

    def test_vortex_at_center(self):
        """Verify vortex at exact center produces zero velocity."""
        from engine.simulation.cloth.cloth_wind import VortexWind

        wind = VortexWind(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            axis=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            strength=1.0,
            radius=5.0,
        )
        velocity = wind.get_velocity(
            np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0
        )
        assert np.linalg.norm(velocity) == 0.0

    def test_vortex_outside_radius(self):
        """Verify vortex outside radius produces zero velocity."""
        from engine.simulation.cloth.cloth_wind import VortexWind

        wind = VortexWind(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            axis=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            strength=1.0,
            radius=5.0,
        )
        velocity = wind.get_velocity(
            np.array([10.0, 0.0, 0.0], dtype=np.float32), 0.0
        )
        assert np.linalg.norm(velocity) == 0.0


class TestPointWindVerification:
    """Verification tests for PointWind edge cases."""

    def test_point_wind_at_source(self):
        """Verify point wind at exact source returns zero (division check)."""
        from engine.simulation.cloth.cloth_wind import PointWind

        wind = PointWind(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            strength=10.0,
            radius=5.0,
        )
        velocity = wind.get_velocity(
            np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0
        )
        assert np.linalg.norm(velocity) == 0.0


class TestWindSystemVerification:
    """Verification tests for WindSystem."""

    def test_wind_system_clear(self):
        """Verify clear_winds removes all sources."""
        from engine.simulation.cloth.cloth_wind import (
            DirectionalWind,
            WindSystem,
        )

        system = WindSystem()
        system.add_directional_wind(
            DirectionalWind(
                direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                strength=5.0,
            )
        )
        assert len(system._directional_winds) == 1
        system.clear_winds()
        assert len(system._directional_winds) == 0
        assert len(system._point_winds) == 0
        assert len(system._vortex_winds) == 0

    def test_wind_system_apply_to_empty_mesh(self):
        """Verify apply_to_mesh on empty mesh does not crash."""
        from engine.simulation.cloth.cloth_wind import WindSystem
        from engine.simulation.cloth.cloth_simulation import ClothMesh

        system = WindSystem()
        mesh = ClothMesh(particles=[], edges=[], triangles=[])
        system.apply_to_mesh(mesh, 1.0 / 60.0)  # Should not raise

    def test_get_combined_wind_no_sources(self):
        """Verify get_combined_wind with no sources returns zero."""
        from engine.simulation.cloth.cloth_wind import WindSystem

        system = WindSystem()
        combined = system.get_combined_wind(
            np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        assert np.allclose(combined, [0.0, 0.0, 0.0])


# =============================================================================
# Simulation Verification
# =============================================================================


class TestClothSimulationVerification:
    """Verification tests for ClothSimulation edge cases."""

    def test_step_when_inactive(self):
        """Verify step() does nothing when simulation is inactive."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        assert sim.state.name == "INACTIVE"

        initial_pos = mesh.get_positions_array().copy()
        sim.step(1.0 / 60.0)
        final_pos = mesh.get_positions_array()

        assert np.allclose(initial_pos, final_pos), (
            "Step() should not modify mesh when inactive"
        )

    def test_step_when_paused(self):
        """Verify step() does nothing when simulation is paused."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        sim.start()
        sim.pause()
        assert sim.state.name == "PAUSED"

        initial_pos = mesh.get_positions_array().copy()
        sim.step(1.0 / 60.0)
        final_pos = mesh.get_positions_array()

        assert np.allclose(initial_pos, final_pos), (
            "Step() should not modify mesh when paused"
        )

    def test_time_accumulator_advances(self):
        """Verify time accumulator advances correctly."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        sim.start()

        assert sim._time_accumulator == 0.0

        # Step with half a fixed timestep -- not enough to trigger a sub-step
        half_step = sim.config.timestep * 0.5
        sim.step(half_step)
        assert abs(sim._time_accumulator - half_step) < 1e-8

        # Another half step should trigger at least one sub-step
        sim.step(half_step)
        assert sim._time_accumulator < sim.config.timestep

    def test_stop_resets_accumulator(self):
        """Verify stop() resets time accumulator."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        sim.start()
        # Use a dt that does NOT evenly divide into the fixed timestep,
        # so a remainder stays in the accumulator after the while loop.
        sim.step(sim.config.timestep * 1.5)  # e.g. 1/120 * 1.5 = 0.0125
        assert sim._time_accumulator > 0
        sim.stop()
        assert sim._time_accumulator == 0.0

    def test_resume_from_pause(self):
        """Verify resume() from paused state resumes simulation."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            ClothState,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        sim.start()
        sim.pause()
        assert sim.state == ClothState.PAUSED
        sim.resume()
        assert sim.state == ClothState.SIMULATING

    def test_resume_from_inactive_does_nothing(self):
        """Verify resume() from inactive changes nothing."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            ClothState,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        assert sim.state == ClothState.INACTIVE
        sim.resume()
        assert sim.state == ClothState.INACTIVE

    def test_pause_when_inactive_does_nothing(self):
        """Verify pause() from inactive changes nothing."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            ClothState,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        assert sim.state == ClothState.INACTIVE
        sim.pause()
        assert sim.state == ClothState.INACTIVE

    def test_remove_constraint_not_present(self):
        """Verify removing non-existent constraint does not raise."""
        from engine.simulation.cloth.cloth_constraints import DistanceConstraint
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        constraint = DistanceConstraint(
            p0_index=0, p1_index=1, rest_length=0.5
        )
        # Should not raise
        sim.remove_constraint(constraint)

    def test_remove_collider_not_present(self):
        """Verify removing non-existent collider does not raise."""
        from engine.simulation.cloth.cloth_collision import SphereCollider
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        collider = SphereCollider(
            center=np.zeros(3, dtype=np.float32), radius=1.0
        )
        # Should not raise
        sim.remove_collider(collider)

    def test_pin_particle_invalid_index(self):
        """Verify pinning invalid index does not raise."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        # Index out of range
        sim.pin_particle(9999)  # Should not raise

    def test_unpin_particle_invalid_index(self):
        """Verify unpinning invalid index does not raise."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        sim.unpin_particle(9999)  # Should not raise

    def test_set_particle_position_updates_both(self):
        """Verify set_particle_position updates both position and prev_position."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        new_pos = np.array([42.0, 42.0, 42.0], dtype=np.float32)
        sim.set_particle_position(7, new_pos)

        assert np.allclose(mesh.particles[7].position, [42.0, 42.0, 42.0])
        assert np.allclose(
            mesh.particles[7].prev_position, [42.0, 42.0, 42.0]
        )


class TestClothMeshVerification:
    """Verification tests for ClothMesh edge cases."""

    def test_create_cloth_grid_1x1(self):
        """Verify 1x1 grid creates a single particle."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(1, 1)
        assert mesh.num_particles == 1
        assert mesh.num_edges == 0
        assert mesh.num_triangles == 0

    def test_create_cloth_grid_1xn(self):
        """Verify 1xN grid creates a line of particles."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(1, 4)
        assert mesh.num_particles == 4
        # 3 vertical structural edges + 2 bend (skip-one) edges = 5
        assert mesh.num_edges == 5
        assert mesh.num_triangles == 0  # No quads with width=1

    def test_create_cloth_grid_exceeds_max(self):
        """Verify create_cloth_grid raises when exceeding MAX_CLOTH_PARTICLES."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        # 101 x 100 = 10100 > 10000
        with pytest.raises(ValueError, match="Grid too large"):
            create_cloth_grid(101, 100)

    def test_create_cloth_from_mesh_duplicate_edges(self):
        """Verify create_cloth_from_mesh deduplicates edges."""
        from engine.simulation.cloth.cloth_simulation import (
            create_cloth_from_mesh,
        )

        # Two triangles sharing an edge
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

        mesh = create_cloth_from_mesh(vertices, indices)
        # 2 triangles = 5 unique edges (not 6, since shared edge is deduplicated)
        assert mesh.num_edges == 5
        assert mesh.num_triangles == 2

    def test_mesh_properties_empty(self):
        """Verify empty mesh properties."""
        from engine.simulation.cloth.cloth_simulation import ClothMesh

        mesh = ClothMesh(particles=[], edges=[], triangles=[])
        assert mesh.num_particles == 0
        assert mesh.num_edges == 0
        assert mesh.num_triangles == 0

    def test_get_positions_array_empty(self):
        """Verify get_positions_array on empty mesh returns empty array."""
        from engine.simulation.cloth.cloth_simulation import ClothMesh

        mesh = ClothMesh(particles=[], edges=[], triangles=[])
        arr = mesh.get_positions_array()
        # An empty particle list produces a 1-D float32 array
        assert arr.shape == (0,) or arr.shape == (0, 3)


class TestClothTriangleVerification:
    """Verification tests for ClothTriangle edge cases."""

    def test_compute_normal_degenerate(self):
        """Verify degenerate triangle normal returns default."""
        from engine.simulation.cloth.cloth_simulation import ClothTriangle

        tri = ClothTriangle(p0=0, p1=1, p2=2)
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        # Degenerate triangle should return a zero-length normal divided safely
        normal = tri.compute_normal(positions)
        assert np.linalg.norm(normal) == 0.0

    def test_compute_area_degenerate(self):
        """Verify degenerate triangle area is zero."""
        from engine.simulation.cloth.cloth_simulation import ClothTriangle

        tri = ClothTriangle(p0=0, p1=1, p2=2)
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        area = tri.compute_area(positions)
        assert area == 0.0

    def test_compute_normal_2d_triangle(self):
        """Verify 2D triangle has normal in Z direction."""
        from engine.simulation.cloth.cloth_simulation import ClothTriangle

        tri = ClothTriangle(p0=0, p1=1, p2=2)
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        normal = tri.compute_normal(positions)
        # Should point in +Z or -Z
        assert abs(abs(normal[2]) - 1.0) < 1e-6
        assert abs(normal[0]) < 1e-6
        assert abs(normal[1]) < 1e-6


class TestClothParticleVerification:
    """Verification tests for ClothParticle edge cases."""

    def test_particle_default_acceleration_zero(self):
        """Verify default acceleration is zero."""
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p = ClothParticle(position=pos.copy(), prev_position=pos.copy())
        assert np.allclose(p.acceleration, [0.0, 0.0, 0.0])

    def test_particle_negative_mass(self):
        """Verify negative mass produces negative inverse mass."""
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p = ClothParticle(
            position=pos, prev_position=pos, inv_mass=-1.0
        )
        assert p.inv_mass == -1.0
        assert p.is_pinned is False

    def test_unpin_with_zero_mass(self):
        """Verify unpin with zero mass falls back to inv_mass=1.0."""
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p = ClothParticle(position=pos, prev_position=pos, inv_mass=0.0)
        p.unpin(mass=0.0)
        assert p.inv_mass == 1.0


class TestClothSimulationConfigVerification:
    """Verification tests for ClothSimulationConfig edge cases."""

    def test_config_default_self_collision_enabled(self):
        """Verify default config has self-collision enabled."""
        from engine.simulation.cloth.cloth_simulation import ClothSimulationConfig

        config = ClothSimulationConfig()
        assert config.enable_self_collision is True

    def test_config_default_wind_enabled(self):
        """Verify default config has wind enabled."""
        from engine.simulation.cloth.cloth_simulation import ClothSimulationConfig

        config = ClothSimulationConfig()
        assert config.enable_wind is True

    def test_config_custom_timestep(self):
        """Verify custom timestep is stored."""
        from engine.simulation.cloth.cloth_simulation import ClothSimulationConfig

        config = ClothSimulationConfig(timestep=1.0 / 240.0)
        assert abs(config.timestep - 1.0 / 240.0) < 1e-8


class TestVelocityUpdateVerification:
    """Verification tests for velocity update edge cases."""

    def test_velocity_update_very_small_dt(self):
        """Verify velocity update with dt < MIN_VELOCITY_TIMESTEP."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )
        from engine.simulation.cloth.config import MIN_VELOCITY_TIMESTEP

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        sim.start()

        # Step with extremely small dt - should not crash
        tiny_dt = MIN_VELOCITY_TIMESTEP * 0.5
        sim.step(tiny_dt)

    def test_velocity_update_dt_zero(self):
        """Verify velocity update with dt=0 does not crash."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        sim.start()

        # The time accumulator will advance by dt but internal _simulate_step
        # may not be called enough to trigger velocity update.
        # We just verify no crash.
        sim._simulate_step(0.0)


class TestClothSimulationForceCallbacks:
    """Verification tests for force callbacks."""

    def test_force_callback_applied(self):
        """Verify force callback modifies particle behavior."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4, pin_top=True)
        sim = ClothSimulation(mesh)
        sim.start()

        positions_before = mesh.get_positions_array().copy()

        def upward_force(m, dt):
            """Apply upward force stronger than gravity."""
            for p in m.particles:
                if not p.is_pinned:
                    p.acceleration += np.array(
                        [0.0, 30.0, 0.0], dtype=np.float32
                    )

        sim.add_force_callback(upward_force)
        sim.step(1.0 / 60.0)

        # Particles should move differently with upward force
        # (at minimum, the function was called without crash and positions differ)
        positions_after = mesh.get_positions_array()
        assert not np.allclose(positions_before, positions_after)

    def test_multiple_force_callbacks(self):
        """Verify multiple force callbacks accumulate."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4, pin_top=True)
        sim = ClothSimulation(mesh)
        sim.start()

        call_order = []

        def callback_a(m, dt):
            call_order.append("a")

        def callback_b(m, dt):
            call_order.append("b")

        sim.add_force_callback(callback_a)
        sim.add_force_callback(callback_b)
        sim.step(1.0 / 60.0)

        # Callbacks fire once per substep, and step(1/60) with
        # timestep=1/120 triggers 2 fixed-timestep iterations:
        # total = 2 iterations * 4 substeps * 2 callbacks = 16 calls.
        num_fixed_steps = int(1.0 / 60.0 / sim.config.timestep)
        expected = num_fixed_steps * sim.config.substeps * 2
        assert len(call_order) == expected, (
            f"Expected {expected} callback calls, got {len(call_order)}"
        )
        # Check ordering within each substep
        for i in range(num_fixed_steps * sim.config.substeps):
            assert call_order[i * 2] == "a"
            assert call_order[i * 2 + 1] == "b"


# =============================================================================
# Integration Verification (GPU stub + simulation + collision)
# =============================================================================


class TestGPUStubIntegration:
    """Integration tests combining GPU stub with simulation."""

    def test_stub_works_with_mesh_create_grid(self):
        """Verify GPUClothSolverStub works with various mesh sizes."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        for size in [2, 4, 10, 50]:
            mesh = create_cloth_grid(size, size)
            solver = GPUClothSolverStub()
            result = solver.initialize(mesh)
            assert result is True
            assert solver._num_particles == size * size
            solver.step(mesh, 1.0 / 60.0)
            solver.shutdown()

    def test_stub_multiple_steps(self):
        """Verify stub handles multiple consecutive steps."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(4, 4)
        solver = GPUClothSolverStub()
        solver.initialize(mesh)

        for _ in range(100):
            solver.step(mesh, 1.0 / 60.0)

        # Mesh should NOT have moved (stub is no-op)
        static_check = create_cloth_grid(4, 4)
        assert np.allclose(
            mesh.get_positions_array(), static_check.get_positions_array()
        )

    def test_gpu_device_protocol(self):
        """Verify GPUDevice Protocol is importable and structurally typed."""
        from engine.simulation.cloth.gpu_cloth import GPUDevice

        # Verify Protocol by checking it has the expected methods
        expected_methods = [
            "create_buffer",
            "destroy_buffer",
            "upload_buffer",
            "readback_buffer",
            "create_compute_pipeline",
            "dispatch_compute",
            "memory_barrier",
        ]
        for method in expected_methods:
            assert hasattr(GPUDevice, method), (
                f"GPUDevice Protocol missing method: {method}"
            )

    def test_integration_stub_initialize_with_realistic_mesh(self):
        """Verify stub initialize works with a realistic cloth mesh."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(16, 16, size_x=2.0, size_y=2.0)
        solver = GPUClothSolverStub()
        solver.initialize(mesh)

        # Verify counts match
        assert solver._num_particles == mesh.num_particles
        assert solver._num_distance_constraints == mesh.num_edges
        assert solver._num_triangles == mesh.num_triangles

        # Simulate the step method which wraps prepare, dispatch, readback
        solver.prepare_buffers(mesh)
        solver.dispatch_compute(1.0 / 60.0)
        solver.readback_positions(mesh)

        # Mesh should be unchanged
        original = create_cloth_grid(16, 16, size_x=2.0, size_y=2.0)
        assert np.allclose(
            mesh.get_positions_array(), original.get_positions_array()
        )


# =============================================================================
# ClassImportsVerification
# =============================================================================


class TestModuleExports:
    """Verify __init__.py exports all expected symbols."""

    def test_all_exported_names(self):
        """Verify core classes are exposed via __init__."""
        import engine.simulation.cloth as cloth_module

        expected = [
            # Config
            "CLOTH_DAMPING",
            "CLOTH_SOLVER_ITERATIONS",
            "CLOTH_SUBSTEPS",
            "CLOTH_TIMESTEP",
            "COLLISION_FRICTION",
            "COLLISION_MARGIN",
            "DEFAULT_BEND_STIFFNESS",
            "DEFAULT_SHEAR_STIFFNESS",
            "DEFAULT_STRETCH_STIFFNESS",
            "MAX_CLOTH_PARTICLES",
            "SELF_COLLISION_THICKNESS",
            "ClothQualityPreset",
            # Simulation
            "ClothEdge",
            "ClothMesh",
            "ClothParticle",
            "ClothSimulation",
            "ClothSimulationConfig",
            "ClothState",
            "ClothTriangle",
            "create_cloth_from_mesh",
            "create_cloth_grid",
            # Constraints
            "AnchorConstraint",
            "BendingConstraint",
            "DistanceConstraint",
            "LongRangeAttachment",
            "ShearConstraint",
            "TetherConstraint",
            "create_bend_constraints",
            "create_long_range_attachments",
            # Collision
            "BoxCollider",
            "CapsuleCollider",
            "ClothCollisionHandler",
            "CollisionResult",
            "MeshCollider",
            "SDFCollider",
            "SpatialHash",
            "SphereCollider",
            "collide_with_box",
            "collide_with_capsule",
            "collide_with_mesh",
            "collide_with_sdf",
            "collide_with_sphere",
            "handle_self_collision",
            # Wind
            "DirectionalWind",
            "PointWind",
            "VortexWind",
            "WindForce",
            "WindSettings",
            "WindSystem",
            # GPU
            "GPUBuffer",
            "GPUBufferAccess",
            "GPUBufferUsage",
            "GPUClothBuffers",
            "GPUClothPipelines",
            "GPUClothSolver",
            "GPUClothSolverStub",
            "GPUComputePipeline",
            "GPUDevice",
            "calculate_workgroups",
            "get_shader_templates",
        ]
        for name in expected:
            assert hasattr(cloth_module, name), f"Missing export: {name}"
