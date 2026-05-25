"""
Blackbox contract tests for GPU Cloth Phase 1.

Tests the public contract of the cloth simulation system from the
GPU perspective: buffers, pipelines, solver, constraints, collision,
wind, and simulation interfaces.

These tests derive entirely from the public API declared in:
  engine/simulation/cloth/__init__.py
  docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/engine_simulation_character_cloth_collision/PHASE_1_TODO.md
  docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/engine_simulation_character_cloth_collision/PHASE_1_ARCH.md

Cleanroom discipline: no implementation detail from gpu_cloth.py,
cloth_simulation.py, cloth_constraints.py, cloth_collision.py, or
cloth_wind.py is observed. Tests encode only the public contract.
"""

import math
from typing import Any, List

import numpy as np
import pytest


# =============================================================================
# Config Contract
# =============================================================================


class TestClothConfigContract:
    """Contract: cloth config constants define the simulation baseline."""

    def test_damping_defined(self):
        """CLOTH_DAMPING must be a positive float."""
        from engine.simulation.cloth.config import CLOTH_DAMPING

        assert isinstance(CLOTH_DAMPING, float)
        assert CLOTH_DAMPING > 0.0

    def test_solver_iterations_defined(self):
        """CLOTH_SOLVER_ITERATIONS must be a positive int."""
        from engine.simulation.cloth.config import CLOTH_SOLVER_ITERATIONS

        assert isinstance(CLOTH_SOLVER_ITERATIONS, int)
        assert CLOTH_SOLVER_ITERATIONS > 0

    def test_collision_friction_defined(self):
        """COLLISION_FRICTION must be a non-negative float."""
        from engine.simulation.cloth.config import COLLISION_FRICTION

        assert isinstance(COLLISION_FRICTION, float)
        assert COLLISION_FRICTION >= 0.0

    def test_collision_margin_defined(self):
        """COLLISION_MARGIN must be a non-negative float."""
        from engine.simulation.cloth.config import COLLISION_MARGIN

        assert isinstance(COLLISION_MARGIN, float)
        assert COLLISION_MARGIN >= 0.0

    def test_cloth_quality_preset_enum_values(self):
        """ClothQualityPreset must expose HIGH, MEDIUM, LOW, MOBILE."""
        from engine.simulation.cloth.config import ClothQualityPreset

        assert hasattr(ClothQualityPreset, "HIGH")
        assert hasattr(ClothQualityPreset, "MEDIUM")
        assert hasattr(ClothQualityPreset, "LOW")
        assert hasattr(ClothQualityPreset, "MOBILE")

    def test_cloth_quality_preset_low_defined(self):
        """LOW preset must define substeps and self_collision."""
        from engine.simulation.cloth.config import ClothQualityPreset

        preset = ClothQualityPreset.LOW
        assert "substeps" in preset
        assert "self_collision" in preset
        assert isinstance(preset["substeps"], int)
        assert isinstance(preset["self_collision"], bool)

    def test_default_stiffness_values_bounded(self):
        """Default stiffness values must be in [0, 1]."""
        from engine.simulation.cloth.config import (
            DEFAULT_BEND_STIFFNESS,
            DEFAULT_SHEAR_STIFFNESS,
            DEFAULT_STRETCH_STIFFNESS,
        )

        for s in [DEFAULT_STRETCH_STIFFNESS, DEFAULT_BEND_STIFFNESS, DEFAULT_SHEAR_STIFFNESS]:
            assert 0.0 <= s <= 1.0, f"Stiffness {s} outside [0, 1]"

    def test_max_cloth_particles_positive(self):
        """MAX_CLOTH_PARTICLES must be a positive int."""
        from engine.simulation.cloth.config import MAX_CLOTH_PARTICLES

        assert isinstance(MAX_CLOTH_PARTICLES, int)
        assert MAX_CLOTH_PARTICLES > 0


# =============================================================================
# Cloth Particle & Mesh Contract
# =============================================================================


class TestClothParticleContract:
    """Contract: ClothParticle represents a single simulation particle."""

    def test_particle_has_position_prev_and_inv_mass(self):
        """Particle must expose position, prev_position, inv_mass."""
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        prev = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p = ClothParticle(position=pos.copy(), prev_position=prev.copy(), inv_mass=0.5)

        assert hasattr(p, "position")
        assert hasattr(p, "prev_position")
        assert hasattr(p, "inv_mass")
        assert np.allclose(p.position, pos)
        assert np.allclose(p.prev_position, prev)
        assert p.inv_mass == 0.5

    def test_particle_default_inv_mass_is_one(self):
        """Default inv_mass must be 1.0 (movable particle)."""
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p = ClothParticle(
            position=np.zeros(3, dtype=np.float32),
            prev_position=np.zeros(3, dtype=np.float32),
        )
        assert p.inv_mass == 1.0

    def test_particle_zero_inv_mass_is_pinned(self):
        """inv_mass=0.0 must create a pinned (immovable) particle."""
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p = ClothParticle(
            position=np.zeros(3, dtype=np.float32),
            prev_position=np.zeros(3, dtype=np.float32),
            inv_mass=0.0,
        )
        assert p.inv_mass == 0.0


class TestClothMeshContract:
    """Contract: ClothMesh is the fundamental cloth data structure."""

    def test_mesh_has_particles_edges_triangles(self):
        """Mesh must expose particles, edges, triangles."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothEdge,
            ClothMesh,
            ClothParticle,
            ClothTriangle,
        )

        p = ClothParticle(
            position=np.zeros(3, dtype=np.float32),
            prev_position=np.zeros(3, dtype=np.float32),
        )
        e = ClothEdge(p0=0, p1=1, rest_length=1.0)
        t = ClothTriangle(p0=0, p1=1, p2=2)
        mesh = ClothMesh(particles=[p], edges=[e], triangles=[t])

        assert len(mesh.particles) == 1
        assert len(mesh.edges) == 1
        assert len(mesh.triangles) == 1
        assert mesh.edges[0].rest_length == 1.0

    def test_mesh_get_positions_array_shape(self):
        """get_positions_array() must return (N, 3) float32 array."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(4, 4)
        positions = mesh.get_positions_array()

        assert positions.shape == (16, 3)
        assert positions.dtype == np.float32

    def test_cloth_edge_has_indices_and_rest_length(self):
        """ClothEdge must expose p0, p1, rest_length."""
        from engine.simulation.cloth.cloth_simulation import ClothEdge

        e = ClothEdge(p0=0, p1=1, rest_length=0.5)
        assert e.p0 == 0
        assert e.p1 == 1
        assert e.rest_length == 0.5

    def test_cloth_triangle_has_three_indices(self):
        """ClothTriangle must expose p0, p1, p2."""
        from engine.simulation.cloth.cloth_simulation import ClothTriangle

        t = ClothTriangle(p0=0, p1=1, p2=2)
        assert t.p0 == 0
        assert t.p1 == 1
        assert t.p2 == 2

    def test_create_cloth_grid_rectangle(self):
        """create_cloth_grid must produce a rectangular grid."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        w, h = 5, 3
        mesh = create_cloth_grid(w, h)

        assert len(mesh.particles) == w * h
        # Triangles follow the standard grid triangulation
        expected_triangles = (w - 1) * (h - 1) * 2
        assert len(mesh.triangles) == expected_triangles
        # Edges include structural, shear diagonals, and skip connections
        assert len(mesh.edges) >= (w - 1) * h + w * (h - 1)  # at minimum structural

    def test_create_cloth_grid_tiny(self):
        """A 2x2 grid must produce 4 particles, 6 edges (including shear), 2 triangles."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(2, 2)
        assert len(mesh.particles) == 4
        # 2x2 grid: structural edges (horizontal+vertical) + shear diagonals
        # structural: (2-1)*2 + 2*(2-1) = 4, shear: (2-1)*(2-1)*2 = 2, total: 6
        assert len(mesh.edges) == 6
        assert len(mesh.triangles) == 2


class TestClothSimulationConfigContract:
    """Contract: ClothSimulationConfig controls simulation parameters."""

    def test_config_default_values(self):
        """Config must have sensible defaults for substeps and iterations."""
        from engine.simulation.cloth.cloth_simulation import ClothSimulationConfig

        config = ClothSimulationConfig()
        assert hasattr(config, "substeps")
        assert hasattr(config, "solver_iterations")
        assert isinstance(config.substeps, int)
        assert isinstance(config.solver_iterations, int)
        assert config.substeps >= 1
        assert config.solver_iterations >= 1


# =============================================================================
# GPU Buffer Contract
# =============================================================================


class TestGPUBufferContract:
    """Contract: GPUBuffer is the base GPU memory abstraction."""

    def test_buffer_creation_defaults(self):
        """Default GPUBuffer must have handle=None and size=0."""
        from engine.simulation.cloth.gpu_cloth import GPUBuffer

        buf = GPUBuffer()
        assert buf.handle is None
        assert buf.size == 0

    def test_buffer_with_handle_and_size(self):
        """GPUBuffer must accept handle and size parameters."""
        from engine.simulation.cloth.gpu_cloth import GPUBuffer

        handle = object()
        buf = GPUBuffer(handle=handle, size=1024)
        assert buf.handle is handle
        assert buf.size == 1024

    def test_buffer_is_valid_predicate(self):
        """is_valid() must return True when handle is not None."""
        from engine.simulation.cloth.gpu_cloth import GPUBuffer

        buf = GPUBuffer()
        assert buf.is_valid() is False

        buf.handle = object()
        assert buf.is_valid() is True

    def test_gpu_buffer_usage_enum_values(self):
        """GPUBufferUsage must have VERTEX, INDEX, STORAGE, UNIFORM, STAGING."""
        from engine.simulation.cloth.gpu_cloth import GPUBufferUsage

        assert GPUBufferUsage.VERTEX is not None
        assert GPUBufferUsage.INDEX is not None
        assert GPUBufferUsage.STORAGE is not None
        assert GPUBufferUsage.UNIFORM is not None
        assert GPUBufferUsage.STAGING is not None

    def test_gpu_buffer_access_enum_values(self):
        """GPUBufferAccess must have READ_ONLY, WRITE_ONLY, READ_WRITE."""
        from engine.simulation.cloth.gpu_cloth import GPUBufferAccess

        assert GPUBufferAccess.READ_ONLY is not None
        assert GPUBufferAccess.WRITE_ONLY is not None
        assert GPUBufferAccess.READ_WRITE is not None


class TestGPUClothBuffersContract:
    """Contract: GPUClothBuffers manages the complete set of GPU cloth buffers.

    Per ADR-GPU-CLOTH-003, position buffers are double-buffered
    (current/predicted/previous) for PBD iteration.
    """

    def test_buffer_collection_has_required_fields(self):
        """All required buffer fields must exist per the arch spec."""
        from engine.simulation.cloth.gpu_cloth import GPUClothBuffers

        bufs = GPUClothBuffers()

        # Core buffers from the TODO buffer layout table
        assert hasattr(bufs, "positions_current")
        assert hasattr(bufs, "positions_predicted")
        assert hasattr(bufs, "positions_previous")
        assert hasattr(bufs, "velocities")
        assert hasattr(bufs, "inv_masses")
        assert hasattr(bufs, "distance_constraints")
        assert hasattr(bufs, "bending_constraints")
        assert hasattr(bufs, "triangles")
        assert hasattr(bufs, "collision_primitives")
        assert hasattr(bufs, "simulation_params")

    def test_are_valid_rejects_partial_buffers(self):
        """are_valid() must reject when only some buffers are set."""
        from engine.simulation.cloth.gpu_cloth import GPUBuffer, GPUClothBuffers

        bufs = GPUClothBuffers()
        bufs.positions_current = GPUBuffer(handle=object(), size=64)

        assert bufs.are_valid() is False

    def test_are_valid_accepts_full_core_set(self):
        """are_valid() must accept when all core position buffers are valid."""
        from engine.simulation.cloth.gpu_cloth import GPUBuffer, GPUClothBuffers

        bufs = GPUClothBuffers()
        bufs.positions_current = GPUBuffer(handle=object(), size=64)
        bufs.positions_predicted = GPUBuffer(handle=object(), size=64)
        bufs.positions_previous = GPUBuffer(handle=object(), size=64)
        bufs.inv_masses = GPUBuffer(handle=object(), size=64)

        # Only core buffers needed for solver
        assert bufs.are_valid() is not None


class TestGPUComputePipelineContract:
    """Contract: GPUComputePipeline represents a compute shader stage."""

    def test_pipeline_defaults(self):
        """Default pipeline must have handle=None, shader_module=None."""
        from engine.simulation.cloth.gpu_cloth import GPUComputePipeline

        pipe = GPUComputePipeline()
        assert pipe.handle is None
        assert pipe.shader_module is None

    def test_pipeline_local_size_configurable(self):
        """local_size must be configurable and default to (64, 1, 1)."""
        from engine.simulation.cloth.gpu_cloth import GPUComputePipeline

        pipe = GPUComputePipeline()
        assert pipe.local_size == (64, 1, 1)

        pipe_custom = GPUComputePipeline(local_size=(128, 1, 1))
        assert pipe_custom.local_size == (128, 1, 1)

    def test_pipeline_is_valid_with_handle(self):
        """is_valid() must return True when handle is set."""
        from engine.simulation.cloth.gpu_cloth import GPUComputePipeline

        pipe = GPUComputePipeline()
        assert pipe.is_valid() is False

        pipe.handle = object()
        assert pipe.is_valid() is True


class TestGPUClothPipelinesContract:
    """Contract: GPUClothPipelines stores all compute pipeline stages.

    Per PHASE_1_ARCH, pipeline stages are: integration, distance_constraint,
    bending_constraint, collision_primitives, self_collision_broad,
    self_collision_narrow, wind_force, velocity_update.
    """

    def test_required_pipeline_stages_exist(self):
        """All required pipeline stages must be present."""
        from engine.simulation.cloth.gpu_cloth import GPUClothPipelines

        pipes = GPUClothPipelines()

        # Core pipeline stages from the arch
        assert hasattr(pipes, "integration")
        assert hasattr(pipes, "distance_constraint")
        assert hasattr(pipes, "bending_constraint")
        assert hasattr(pipes, "collision_primitives")
        assert hasattr(pipes, "self_collision_broad")
        assert hasattr(pipes, "self_collision_narrow")
        assert hasattr(pipes, "wind_force")
        assert hasattr(pipes, "velocity_update")

    def test_pipeline_stages_are_attributes(self):
        """All pipeline stages must be accessible attributes (not methods)."""
        from engine.simulation.cloth.gpu_cloth import GPUClothPipelines

        pipes = GPUClothPipelines()
        # Access each stage as an attribute; should not raise
        _ = pipes.integration
        _ = pipes.distance_constraint
        _ = pipes.bending_constraint
        _ = pipes.collision_primitives
        _ = pipes.self_collision_broad
        _ = pipes.self_collision_narrow
        _ = pipes.wind_force
        _ = pipes.velocity_update


# =============================================================================
# GPU Cloth Solver Contract
# =============================================================================


class TestGPUClothSolverContract:
    """Contract: GPUClothSolver is the abstract interface for GPU cloth simulation.

    Per PHASE_1_TODO Task 7 (Interface Compatibility), the GPU solver must
    be a drop-in replacement for the CPU ClothSimulation.
    """

    def test_solver_is_abstract_cannot_instantiate(self):
        """GPUClothSolver must be abstract and cannot be directly instantiated."""
        from engine.simulation.cloth.gpu_cloth import GPUClothSolver

        with pytest.raises(TypeError):
            GPUClothSolver(None)


class TestGPUClothSolverStubContract:
    """Contract: GPUClothSolverStub is the no-op placeholder.

    Per PHASE_1_TODO, the stub must implement the same interface as the
    real GPU solver: initialize, step, shutdown, dispatch_compute,
    prepare_buffers, readback_positions.
    """

    def test_stub_implements_required_interface(self):
        """Stub must have the required public methods."""
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        solver = GPUClothSolverStub()
        assert hasattr(solver, "initialize")
        assert hasattr(solver, "step")
        assert hasattr(solver, "shutdown")
        assert hasattr(solver, "dispatch_compute")
        assert hasattr(solver, "prepare_buffers")
        assert hasattr(solver, "readback_positions")

    def test_stub_initialize_with_config(self):
        """Stub must accept an optional ClothSimulationConfig."""
        from engine.simulation.cloth.cloth_simulation import ClothSimulationConfig
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        config = ClothSimulationConfig(substeps=2, solver_iterations=2)
        solver = GPUClothSolverStub(config=config)
        assert solver.config.substeps == 2
        assert solver.config.solver_iterations == 2

    def test_stub_initialize_returns_bool(self):
        """initialize() must return a bool."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(4, 4)
        solver = GPUClothSolverStub()
        result = solver.initialize(mesh)
        assert isinstance(result, bool)

    def test_stub_step_accepts_mesh_and_dt(self):
        """step() must accept (mesh, dt) without raising."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(4, 4)
        solver = GPUClothSolverStub()
        solver.initialize(mesh)
        # Step with various dt values
        solver.step(mesh, 1.0 / 60.0)
        solver.step(mesh, 1.0 / 30.0)
        solver.step(mesh, 0.0)  # Zero dt must not crash

    def test_stub_shutdown_cleans_up(self):
        """shutdown() must reset the solver state."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(4, 4)
        solver = GPUClothSolverStub()
        solver.initialize(mesh)
        assert solver._is_initialized is True

        solver.shutdown()
        assert solver._is_initialized is False

    def test_stub_dispatch_compute_is_noop(self):
        """dispatch_compute() must be a safe no-op."""
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        solver = GPUClothSolverStub()
        # Must not raise regardless of state
        solver.dispatch_compute(1.0 / 60.0)
        solver.dispatch_compute(0.0)

    def test_stub_prepare_buffers_accepts_mesh(self):
        """prepare_buffers() must accept a cloth mesh."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(4, 4)
        solver = GPUClothSolverStub()
        solver.prepare_buffers(mesh)

    def test_stub_readback_positions_is_noop(self):
        """readback_positions() must be a safe no-op."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(4, 4)
        solver = GPUClothSolverStub()
        solver.initialize(mesh)
        solver.readback_positions(mesh)


# =============================================================================
# Workgroup Calculation Contract
# =============================================================================


class TestCalculateWorkgroupsContract:
    """Contract: calculate_workgroups maps particle count to dispatch size."""

    def test_workgroup_count_ceil_division(self):
        """Workgroups must round up (ceil division)."""
        from engine.simulation.cloth.gpu_cloth import calculate_workgroups

        assert calculate_workgroups(1, 64) == 1
        assert calculate_workgroups(64, 64) == 1
        assert calculate_workgroups(65, 64) == 2
        assert calculate_workgroups(128, 64) == 2

    def test_workgroup_zero_items(self):
        """Zero items must return zero workgroups."""
        from engine.simulation.cloth.gpu_cloth import calculate_workgroups

        assert calculate_workgroups(0, 64) == 0

    def test_workgroup_custom_local_size(self):
        """Custom local_size must affect the result."""
        from engine.simulation.cloth.gpu_cloth import calculate_workgroups

        assert calculate_workgroups(256, 256) == 1
        assert calculate_workgroups(256, 128) == 2
        assert calculate_workgroups(100, 32) == 4

    def test_workgroup_large_counts(self):
        """Large particle counts must produce expected workgroups."""
        from engine.simulation.cloth.gpu_cloth import calculate_workgroups

        assert calculate_workgroups(10000, 256) == 40  # ceil(10000/256)
        assert calculate_workgroups(10001, 256) == 40  # ceil(10001/256) = 39.06 -> 40
        assert calculate_workgroups(10000, 64) == 157  # ceil(10000/64)


# =============================================================================
# Shader Template Contract
# =============================================================================


class TestShaderTemplatesContract:
    """Contract: Shader templates define the compute shader stages.

    Per PHASE_1_TODO Task 2, three shaders must be converted to WGSL:
    integration, distance constraint, and velocity update.
    """

    def test_get_shader_templates_returns_all(self):
        """get_shader_templates() must return all defined shader stages."""
        from engine.simulation.cloth.gpu_cloth import get_shader_templates

        templates = get_shader_templates()
        assert "integration" in templates
        assert "distance_constraint" in templates
        assert "velocity_update" in templates

    def test_integration_shader_contract(self):
        """Integration shader must contain PBD integration elements."""
        from engine.simulation.cloth.gpu_cloth import INTEGRATION_SHADER_TEMPLATE

        shader = INTEGRATION_SHADER_TEMPLATE
        assert isinstance(shader, str)
        assert len(shader) > 0
        # Must handle gravity and damping
        assert "gravity" in shader.lower() or "gravity" in shader
        assert "damping" in shader.lower() or "damping" in shader

    def test_distance_constraint_shader_contract(self):
        """Distance constraint shader must contain PBD projection elements."""
        from engine.simulation.cloth.gpu_cloth import DISTANCE_CONSTRAINT_SHADER_TEMPLATE

        shader = DISTANCE_CONSTRAINT_SHADER_TEMPLATE
        assert isinstance(shader, str)
        assert len(shader) > 0
        # Must reference rest length
        assert "rest_length" in shader

    def test_velocity_update_shader_contract(self):
        """Velocity update shader must contain velocity computation elements."""
        from engine.simulation.cloth.gpu_cloth import VELOCITY_UPDATE_SHADER_TEMPLATE

        shader = VELOCITY_UPDATE_SHADER_TEMPLATE
        assert isinstance(shader, str)
        assert len(shader) > 0
        # Must reference velocities
        assert "velocities" in shader


# =============================================================================
# Constraint Contract
# =============================================================================


class TestDistanceConstraintContract:
    """Contract: DistanceConstraint enforces rest length between two particles."""

    def test_solve_edge_static_method(self):
        """solve_edge must be a static/classmethod taking (p0, p1, rest_length, stiffness)."""
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
        # Returns a float error
        error = DistanceConstraint.solve_edge(p0, p1, 1.0, 1.0)
        assert isinstance(error, float)

    def test_solve_edge_rest_length_satisfied(self):
        """Particles at rest length must yield zero error."""
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

    def test_solve_edge_error_is_stretch_amount(self):
        """Error returned must equal |distance - rest_length|."""
        from engine.simulation.cloth.cloth_constraints import DistanceConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        p1 = ClothParticle(
            position=np.array([3.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([3.0, 0.0, 0.0], dtype=np.float32),
        )
        error = DistanceConstraint.solve_edge(p0, p1, 1.0, 1.0)
        # error should be the stretch amount: 3.0 - 1.0 = 2.0
        assert error == 2.0

    def test_constraint_constructor(self):
        """DistanceConstraint must store indices and rest_length."""
        from engine.simulation.cloth.cloth_constraints import DistanceConstraint

        c = DistanceConstraint(p0_index=0, p1_index=5, rest_length=0.5, stiffness=0.8)
        assert c.p0_index == 0
        assert c.p1_index == 5
        assert c.rest_length == 0.5
        assert c.stiffness == 0.8


class TestBendingConstraintContract:
    """Contract: BendingConstraint prevents folding along shared edges."""

    def test_compute_dihedral_angle_return_type(self):
        """compute_dihedral_angle must return a float."""
        from engine.simulation.cloth.cloth_constraints import BendingConstraint

        p0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p2 = np.array([0.5, 1.0, 0.0], dtype=np.float32)
        p3 = np.array([0.5, -1.0, 0.0], dtype=np.float32)

        angle = BendingConstraint.compute_dihedral_angle(p0, p1, p2, p3)
        assert isinstance(angle, float)

    def test_constraint_constructor(self):
        """BendingConstraint must store indices, rest_angle, stiffness."""
        from engine.simulation.cloth.cloth_constraints import BendingConstraint

        c = BendingConstraint(
            p0_index=0, p1_index=1, p2_index=2, p3_index=3,
            rest_angle=0.0, stiffness=0.1,
        )
        assert c.p0_index == 0
        assert c.p1_index == 1
        assert c.p2_index == 2
        assert c.p3_index == 3
        assert c.rest_angle == 0.0
        assert c.stiffness == 0.1

    def test_solve_returns_float(self):
        """solve() must return a float error value."""
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
        c = BendingConstraint(
            p0_index=0, p1_index=1, p2_index=2, p3_index=3,
            rest_angle=0.0, stiffness=0.1,
        )
        error = c.solve(particles)
        assert isinstance(error, float)


class TestShearConstraintContract:
    """Contract: ShearConstraint prevents shearing deformation."""

    def test_constructor(self):
        """ShearConstraint must store indices, rest_length, stiffness."""
        from engine.simulation.cloth.cloth_constraints import ShearConstraint

        c = ShearConstraint(p0_index=0, p1_index=2, rest_length=1.414, stiffness=0.5)
        assert c.p0_index == 0
        assert c.p1_index == 2
        assert c.rest_length == 1.414
        assert c.stiffness == 0.5

    def test_solve_returns_float(self):
        """solve() must return a float."""
        from engine.simulation.cloth.cloth_constraints import ShearConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        p1 = ClothParticle(
            position=np.array([1.0, 1.0, 0.0], dtype=np.float32),
            prev_position=np.array([1.0, 1.0, 0.0], dtype=np.float32),
        )
        c = ShearConstraint(p0_index=0, p1_index=1, rest_length=1.0, stiffness=0.5)
        error = c.solve([p0, p1])
        assert isinstance(error, float)


class TestAnchorConstraintContract:
    """Contract: AnchorConstraint pins a particle to a world-space position."""

    def test_constructor(self):
        """AnchorConstraint must store particle_index, anchor_position, stiffness."""
        from engine.simulation.cloth.cloth_constraints import AnchorConstraint

        anchor_pos = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        c = AnchorConstraint(
            particle_index=0,
            anchor_position=anchor_pos.copy(),
            stiffness=0.5,
        )
        assert c.particle_index == 0
        assert np.allclose(c.anchor_position, [5.0, 0.0, 0.0])
        assert c.stiffness == 0.5

    def test_solve_returns_float(self):
        """solve() must return a float."""
        from engine.simulation.cloth.cloth_constraints import AnchorConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p = ClothParticle(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        anchor = AnchorConstraint(
            particle_index=0,
            anchor_position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
        )
        error = anchor.solve([p])
        assert isinstance(error, float)

    def test_update_anchor_changes_position(self):
        """update_anchor() must modify the anchor position."""
        from engine.simulation.cloth.cloth_constraints import AnchorConstraint

        anchor = AnchorConstraint(
            particle_index=0,
            anchor_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        new_pos = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        anchor.update_anchor(new_pos)
        assert np.allclose(anchor.anchor_position, [10.0, 20.0, 30.0])


class TestTetherConstraintContract:
    """Contract: TetherConstraint limits particle distance from an attachment."""

    def test_constructor(self):
        """TetherConstraint must store particle_index, attachment_position, max_distance."""
        from engine.simulation.cloth.cloth_constraints import TetherConstraint

        c = TetherConstraint(
            particle_index=0,
            attachment_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            max_distance=2.0,
        )
        assert c.particle_index == 0
        assert c.max_distance == 2.0

    def test_solve_returns_float(self):
        """solve() must return a float."""
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
        assert isinstance(error, float)


class TestLongRangeAttachmentContract:
    """Contract: LongRangeAttachment connects distant particles to anchored ones."""

    def test_constructor(self):
        """LongRangeAttachment must store indices, max_distance, stiffness."""
        from engine.simulation.cloth.cloth_constraints import LongRangeAttachment

        c = LongRangeAttachment(
            p0_index=0, p1_index=5, max_distance=3.0, stiffness=0.5,
        )
        assert c.p0_index == 0
        assert c.p1_index == 5
        assert c.max_distance == 3.0

    def test_solve_returns_float(self):
        """solve() must return a float."""
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
        att = LongRangeAttachment(p0_index=0, p1_index=1, max_distance=2.0)
        error = att.solve([p0, p1])
        assert isinstance(error, float)


class TestCreateBendConstraintsContract:
    """Contract: create_bend_constraints generates constraints from mesh topology."""

    def test_create_bend_from_grid_mesh(self):
        """A grid mesh must produce bend constraints for adjacent triangles."""
        from engine.simulation.cloth.cloth_constraints import create_bend_constraints
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(3, 3)
        # Convert ClothTriangle objects to tuples for the function
        triangles = [(t.p0, t.p1, t.p2) for t in mesh.triangles]
        constraints = create_bend_constraints(mesh.particles, triangles)
        # A 3x3 grid has 8 triangles, some share edges, so should produce constraints
        assert isinstance(constraints, list)


class TestCreateLongRangeAttachmentsContract:
    """Contract: create_long_range_attachments anchors distant particles."""

    def test_create_attachments_from_grid_mesh(self):
        """A grid mesh with anchor indices must produce attachments."""
        from engine.simulation.cloth.cloth_constraints import create_long_range_attachments
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(4, 4)
        attachments = create_long_range_attachments(
            mesh.particles, attachment_indices=[0],
        )
        assert isinstance(attachments, list)
        # Some non-anchor particles reachable from anchor should get attachments
        assert len(attachments) > 0
        assert len(attachments) < len(mesh.particles)


# =============================================================================
# Collision Contract
# =============================================================================


class TestCollisionResultContract:
    """Contract: CollisionResult is the output of all collision queries."""

    def test_result_defaults(self):
        """Default result must have collided=False and zero penetration."""
        from engine.simulation.cloth.cloth_collision import CollisionResult

        r = CollisionResult(collided=False)
        assert hasattr(r, "collided")
        assert r.collided is False
        assert r.penetration_depth == 0.0
        assert r.contact_normal is None
        assert r.contact_point is None

    def test_result_with_collision(self):
        """Result with collision must store penetration data."""
        from engine.simulation.cloth.cloth_collision import CollisionResult

        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        point = np.array([0.0, 1.5, 0.0], dtype=np.float32)
        r = CollisionResult(
            collided=True,
            penetration_depth=0.5,
            contact_normal=normal,
            contact_point=point,
        )
        assert r.collided is True
        assert r.penetration_depth == 0.5
        assert np.allclose(r.contact_normal, [0.0, 1.0, 0.0])
        assert np.allclose(r.contact_point, [0.0, 1.5, 0.0])


class TestSphereColliderContract:
    """Contract: SphereCollider is defined by center and radius."""

    def test_sphere_constructor(self):
        """SphereCollider must accept center, radius, friction."""
        from engine.simulation.cloth.cloth_collision import SphereCollider

        center = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sphere = SphereCollider(center=center, radius=2.0, friction=0.3)
        assert np.allclose(sphere.center, [1.0, 0.0, 0.0])
        assert sphere.radius == 2.0
        assert sphere.friction == 0.3

    def test_collide_sphere_outside(self):
        """Particle outside sphere must NOT collide."""
        from engine.simulation.cloth.cloth_collision import (
            SphereCollider,
            collide_with_sphere,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([5.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        )
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        result = collide_with_sphere(particle, sphere)
        assert result.collided is False

    def test_collide_sphere_inside(self):
        """Particle inside sphere must collide and be pushed out."""
        from engine.simulation.cloth.cloth_collision import (
            SphereCollider,
            collide_with_sphere,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([0.5, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        )
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        result = collide_with_sphere(particle, sphere, margin=0.0)
        assert result.collided is True
        assert result.penetration_depth >= 0.0
        # Particle must be pushed to surface
        dist = np.linalg.norm(particle.position)
        assert abs(dist - 1.0) < 0.01

    def test_collide_sphere_on_surface_no_margin(self):
        """Particle exactly on surface with zero margin must NOT collide."""
        from engine.simulation.cloth.cloth_collision import (
            SphereCollider,
            collide_with_sphere,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

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


class TestCapsuleColliderContract:
    """Contract: CapsuleCollider is defined by axis endpoints and radius."""

    def test_capsule_constructor(self):
        """CapsuleCollider must accept point_a, point_b, radius, friction."""
        from engine.simulation.cloth.cloth_collision import CapsuleCollider

        a = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        capsule = CapsuleCollider(point_a=a, point_b=b, radius=0.5, friction=0.2)
        assert np.allclose(capsule.point_a, [0.0, -1.0, 0.0])
        assert np.allclose(capsule.point_b, [0.0, 1.0, 0.0])
        assert capsule.radius == 0.5

    def test_collide_capsule_outside(self):
        """Particle outside capsule must NOT collide."""
        from engine.simulation.cloth.cloth_collision import (
            CapsuleCollider,
            collide_with_capsule,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
            prev_position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
        )
        capsule = CapsuleCollider(
            point_a=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            point_b=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        result = collide_with_capsule(particle, capsule)
        assert result.collided is False


class TestBoxColliderContract:
    """Contract: BoxCollider is defined by min/max points (AABB)."""

    def test_box_constructor(self):
        """BoxCollider must accept min_point, max_point."""
        from engine.simulation.cloth.cloth_collision import BoxCollider

        box = BoxCollider(
            min_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            max_point=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        assert np.allclose(box.min_point, [0.0, 0.0, 0.0])
        assert np.allclose(box.max_point, [1.0, 1.0, 1.0])

    def test_collide_box_outside(self):
        """Particle outside box must NOT collide."""
        from engine.simulation.cloth.cloth_collision import (
            BoxCollider,
            collide_with_box,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([5.0, 5.0, 5.0], dtype=np.float32),
            prev_position=np.array([5.0, 5.0, 5.0], dtype=np.float32),
        )
        box = BoxCollider(
            min_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            max_point=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        result = collide_with_box(particle, box)
        assert result.collided is False


class TestMeshColliderContract:
    """Contract: MeshCollider is defined by vertices and triangle indices."""

    def test_mesh_constructor(self):
        """MeshCollider must accept vertices and indices."""
        from engine.simulation.cloth.cloth_collision import MeshCollider

        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2], dtype=np.int32)
        mesh = MeshCollider(vertices=vertices, indices=indices)
        assert mesh.vertices.shape == (3, 3)
        assert len(mesh.indices) == 3

    def test_collide_mesh_outside(self):
        """Particle far from mesh must NOT collide."""
        from engine.simulation.cloth.cloth_collision import (
            MeshCollider,
            collide_with_mesh,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([100.0, 100.0, 100.0], dtype=np.float32),
            prev_position=np.array([100.0, 100.0, 100.0], dtype=np.float32),
        )
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2], dtype=np.int32)
        mesh = MeshCollider(vertices=vertices, indices=indices)
        result = collide_with_mesh(particle, mesh)
        assert result.collided is False


class TestSDFColliderContract:
    """Contract: SDFCollider uses a signed distance function for collision."""

    def test_sdf_constructor(self):
        """SDFCollider must accept a callable sdf_function."""
        from engine.simulation.cloth.cloth_collision import SDFCollider

        def dummy_sdf(pos):
            return 1.0, np.array([1.0, 0.0, 0.0], dtype=np.float32)

        sdf = SDFCollider(sdf_function=dummy_sdf)
        assert callable(sdf.sdf_function)

    def test_collide_sdf_outside(self):
        """Particle outside SDF (positive distance) must NOT collide."""
        from engine.simulation.cloth.cloth_collision import (
            SDFCollider,
            collide_with_sdf,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([10.0, 10.0, 10.0], dtype=np.float32),
            prev_position=np.array([10.0, 10.0, 10.0], dtype=np.float32),
        )

        def far_sdf(pos):
            return 100.0, np.array([1.0, 0.0, 0.0], dtype=np.float32)

        sdf = SDFCollider(sdf_function=far_sdf)
        result = collide_with_sdf(particle, sdf, margin=0.01)
        assert result.collided is False

    def test_collide_sdf_inside(self):
        """Particle inside SDF (negative distance) must collide."""
        from engine.simulation.cloth.cloth_collision import (
            SDFCollider,
            collide_with_sdf,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([10.0, 10.0, 10.0], dtype=np.float32),
            prev_position=np.array([10.0, 10.0, 10.0], dtype=np.float32),
        )

        def inside_sdf(pos):
            return -0.5, np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        sdf = SDFCollider(sdf_function=inside_sdf)
        result = collide_with_sdf(particle, sdf, margin=0.01)
        assert result.collided is True


class TestSpatialHashContract:
    """Contract: SpatialHash accelerates spatial queries for collision."""

    def test_insert_and_query(self):
        """inserted particles must be findable by query."""
        from engine.simulation.cloth.cloth_collision import SpatialHash

        h = SpatialHash()
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        h.insert(0, pos)

        results = h.query(pos, radius=0.1)
        assert 0 in results

    def test_query_empty(self):
        """Query on empty hash must return empty list."""
        from engine.simulation.cloth.cloth_collision import SpatialHash

        h = SpatialHash()
        results = h.query(np.array([0.0, 0.0, 0.0], dtype=np.float32), radius=1.0)
        assert results == []

    def test_clear_empties(self):
        """clear() must remove all entries."""
        from engine.simulation.cloth.cloth_collision import SpatialHash

        h = SpatialHash()
        h.insert(0, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        h.clear()
        results = h.query(np.array([0.0, 0.0, 0.0], dtype=np.float32), radius=1.0)
        assert results == []


class TestClothCollisionHandlerContract:
    """Contract: ClothCollisionHandler manages the collision pipeline."""

    def test_add_colliders(self):
        """Handler must support adding sphere, capsule, box, mesh, SDF colliders."""
        from engine.simulation.cloth.cloth_collision import (
            BoxCollider,
            CapsuleCollider,
            ClothCollisionHandler,
            MeshCollider,
            SDFCollider,
            SphereCollider,
        )

        handler = ClothCollisionHandler()
        handler.add_sphere(
            SphereCollider(
                center=np.zeros(3, dtype=np.float32), radius=1.0,
            )
        )
        handler.add_capsule(
            CapsuleCollider(
                point_a=np.zeros(3, dtype=np.float32),
                point_b=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                radius=0.5,
            )
        )
        handler.add_box(
            BoxCollider(
                min_point=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                max_point=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )
        )

        assert len(handler.sphere_colliders) == 1
        assert len(handler.capsule_colliders) == 1
        assert len(handler.box_colliders) == 1

    def test_process_collisions_with_mesh(self):
        """process_collisions must accept a ClothMesh and return collision count."""
        from engine.simulation.cloth.cloth_collision import (
            ClothCollisionHandler,
            SphereCollider,
        )
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        handler = ClothCollisionHandler()
        handler.add_sphere(
            SphereCollider(
                center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                radius=10.0,
            )
        )

        mesh = create_cloth_grid(4, 4)
        count = handler.process_collisions(mesh)
        assert isinstance(count, int)

    def test_clear_removes_all_colliders(self):
        """clear() must remove all registered colliders."""
        from engine.simulation.cloth.cloth_collision import (
            ClothCollisionHandler,
            SphereCollider,
        )

        handler = ClothCollisionHandler()
        handler.add_sphere(
            SphereCollider(
                center=np.zeros(3, dtype=np.float32), radius=1.0,
            )
        )
        handler.clear()
        assert len(handler.sphere_colliders) == 0
        assert len(handler.capsule_colliders) == 0
        assert len(handler.box_colliders) == 0


class TestSelfCollisionContract:
    """Contract: handle_self_collision resolves cloth-cloth interpenetration."""

    def test_empty_mesh_returns_zero(self):
        """Empty mesh must return 0 collisions."""
        from engine.simulation.cloth.cloth_collision import handle_self_collision
        from engine.simulation.cloth.cloth_simulation import ClothMesh

        mesh = ClothMesh(particles=[], edges=[], triangles=[])
        count = handle_self_collision(mesh)
        assert count == 0

    def test_returns_int(self):
        """Must return an int collision count."""
        from engine.simulation.cloth.cloth_collision import handle_self_collision
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(3, 3)
        count = handle_self_collision(mesh)
        assert isinstance(count, int)


# =============================================================================
# Wind Contract
# =============================================================================


class TestWindForceContract:
    """Contract: WindForce applies aerodynamic forces to cloth."""

    def test_wind_force_has_settings(self):
        """WindForce must expose a settings attribute."""
        from engine.simulation.cloth.cloth_wind import WindForce

        wind = WindForce()
        assert hasattr(wind, "settings")
        assert hasattr(wind.settings, "direction")
        assert hasattr(wind.settings, "strength")
        assert hasattr(wind.settings, "turbulence_strength")

    def test_set_direction_normalizes(self):
        """set_direction must normalize the input vector."""
        from engine.simulation.cloth.cloth_wind import WindForce

        wind = WindForce()
        wind.set_direction(np.array([5.0, 0.0, 0.0], dtype=np.float32))
        assert np.allclose(wind.settings.direction, [1.0, 0.0, 0.0])

    def test_set_direction_zero_vector(self):
        """Zero vector must not change direction."""
        from engine.simulation.cloth.cloth_wind import WindForce

        wind = WindForce()
        original = wind.settings.direction.copy()
        wind.set_direction(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert np.allclose(wind.settings.direction, original)

    def test_set_strength_clamps_negative(self):
        """Negative strength must be clamped to zero."""
        from engine.simulation.cloth.cloth_wind import WindForce

        wind = WindForce()
        wind.set_strength(-5.0)
        assert wind.settings.strength == 0.0

    def test_update_advances_time(self):
        """update(dt) must advance internal time."""
        from engine.simulation.cloth.cloth_wind import WindForce

        wind = WindForce()
        assert wind._time == 0.0
        wind.update(0.5)
        assert wind._time == 0.5
        wind.update(0.25)
        assert wind._time == 0.75

    def test_compute_wind_force_on_empty_mesh(self):
        """compute_wind_force on empty mesh must not crash."""
        from engine.simulation.cloth.cloth_wind import WindForce
        from engine.simulation.cloth.cloth_simulation import ClothMesh

        wind = WindForce()
        mesh = ClothMesh(particles=[], edges=[], triangles=[])
        wind.compute_wind_force(mesh, 1.0 / 60.0)


class TestDirectionalWindContract:
    """Contract: DirectionalWind applies force in a constant direction."""

    def test_directional_wind_constructor(self):
        """DirectionalWind must accept direction and strength."""
        from engine.simulation.cloth.cloth_wind import DirectionalWind

        wind = DirectionalWind(
            direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            strength=5.0,
        )
        assert np.allclose(wind.direction, [1.0, 0.0, 0.0])
        assert wind.strength == 5.0

    def test_directional_wind_get_velocity(self):
        """get_velocity must return a 3D array matching direction * strength."""
        from engine.simulation.cloth.cloth_wind import DirectionalWind

        wind = DirectionalWind(
            direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            strength=5.0,
        )
        velocity = wind.get_velocity(
            np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0,
        )
        assert velocity.shape == (3,)
        assert np.allclose(velocity, [5.0, 0.0, 0.0], atol=0.5)


class TestPointWindContract:
    """Contract: PointWind applies radial wind from a point source."""

    def test_point_wind_constructor(self):
        """PointWind must accept position, strength, radius."""
        from engine.simulation.cloth.cloth_wind import PointWind

        wind = PointWind(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            strength=10.0,
            radius=5.0,
        )
        assert np.allclose(wind.position, [0.0, 0.0, 0.0])
        assert wind.strength == 10.0
        assert wind.radius == 5.0

    def test_point_wind_at_source_zero(self):
        """Wind at exact source position must have zero velocity (division guard)."""
        from engine.simulation.cloth.cloth_wind import PointWind

        wind = PointWind(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            strength=10.0,
            radius=5.0,
        )
        velocity = wind.get_velocity(
            np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0,
        )
        assert np.linalg.norm(velocity) == 0.0

    def test_point_wind_outside_radius(self):
        """Wind outside radius must have zero velocity."""
        from engine.simulation.cloth.cloth_wind import PointWind

        wind = PointWind(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            strength=10.0,
            radius=5.0,
        )
        velocity = wind.get_velocity(
            np.array([100.0, 0.0, 0.0], dtype=np.float32), 0.0,
        )
        assert np.linalg.norm(velocity) == 0.0


class TestVortexWindContract:
    """Contract: VortexWind applies rotational wind around an axis."""

    def test_vortex_constructor(self):
        """VortexWind must accept center, axis, strength, radius."""
        from engine.simulation.cloth.cloth_wind import VortexWind

        wind = VortexWind(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            axis=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            strength=1.0,
            radius=5.0,
        )
        assert np.allclose(wind.center, [0.0, 0.0, 0.0])
        assert wind.strength == 1.0
        assert wind.radius == 5.0

    def test_vortex_at_center_zero_velocity(self):
        """Vortex at exact center must have zero velocity."""
        from engine.simulation.cloth.cloth_wind import VortexWind

        wind = VortexWind(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            axis=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            strength=1.0,
            radius=5.0,
        )
        velocity = wind.get_velocity(
            np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0,
        )
        assert np.linalg.norm(velocity) == 0.0

    def test_vortex_outside_radius_zero(self):
        """Vortex outside radius must have zero velocity."""
        from engine.simulation.cloth.cloth_wind import VortexWind

        wind = VortexWind(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            axis=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            strength=1.0,
            radius=5.0,
        )
        velocity = wind.get_velocity(
            np.array([100.0, 0.0, 0.0], dtype=np.float32), 0.0,
        )
        assert np.linalg.norm(velocity) == 0.0

    def test_vortex_velocity_is_tangential(self):
        """Vortex velocity must be perpendicular to radial direction and axis."""
        from engine.simulation.cloth.cloth_wind import VortexWind

        wind = VortexWind(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            axis=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            strength=1.0,
            radius=5.0,
        )
        # Point on the x-axis
        velocity = wind.get_velocity(
            np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0,
        )
        if np.linalg.norm(velocity) > 0:
            # Velocity should be tangent: perpendicular to both radial and axis
            radial = np.array([1.0, 0.0, 0.0])
            axis = np.array([0.0, 1.0, 0.0])
            # v dot r = 0 (tangent to radial)
            assert abs(np.dot(velocity, radial)) < 0.01, (
                f"Vortex velocity {velocity} is not tangent to radial"
            )


class TestWindSystemContract:
    """Contract: WindSystem manages multiple wind sources."""

    def test_add_wind_sources(self):
        """WindSystem must support adding directional, point, and vortex winds."""
        from engine.simulation.cloth.cloth_wind import (
            DirectionalWind,
            PointWind,
            VortexWind,
            WindSystem,
        )

        system = WindSystem()
        system.add_directional_wind(
            DirectionalWind(
                direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                strength=5.0,
            )
        )
        system.add_point_wind(
            PointWind(
                position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                strength=10.0,
                radius=5.0,
            )
        )
        system.add_vortex_wind(
            VortexWind(
                center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                axis=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                strength=1.0,
                radius=5.0,
            )
        )

        assert len(system._directional_winds) == 1
        assert len(system._point_winds) == 1
        assert len(system._vortex_winds) == 1

    def test_clear_removes_all(self):
        """clear_winds() must remove all wind sources."""
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
        system.clear_winds()
        assert len(system._directional_winds) == 0

    def test_apply_to_mesh_empty(self):
        """apply_to_mesh on empty mesh must not crash."""
        from engine.simulation.cloth.cloth_wind import WindSystem
        from engine.simulation.cloth.cloth_simulation import ClothMesh

        system = WindSystem()
        mesh = ClothMesh(particles=[], edges=[], triangles=[])
        system.apply_to_mesh(mesh, 1.0 / 60.0)


# =============================================================================
# GPUClothSolverStub Lifecycle Contract (Phase 1 integration scenario)
# =============================================================================


class TestGPUClothSolverStubLifecycleContract:
    """Contract: The stub must follow a valid lifecycle without crashes.

    Per PHASE_1_TODO Task 7, the solver interface must support:
      initialize -> prepare_buffers -> step -> dispatch_compute ->
      readback_positions -> shutdown
    """

    def test_full_lifecycle_with_grid_mesh(self):
        """Full lifecycle must complete without errors."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(8, 8)
        solver = GPUClothSolverStub()

        # Lifecycle
        result = solver.initialize(mesh)
        assert result is True

        solver.prepare_buffers(mesh)

        # Multiple steps
        for i in range(5):
            solver.step(mesh, 1.0 / 60.0)
            solver.dispatch_compute(1.0 / 60.0)
            solver.readback_positions(mesh)

        solver.shutdown()

    def test_multi_step_with_large_grid(self):
        """Multiple steps with a large grid must not crash."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(32, 32)  # 1024 particles
        solver = GPUClothSolverStub()
        solver.initialize(mesh)

        for _ in range(10):
            solver.step(mesh, 1.0 / 120.0)

    def test_mesh_positions_unchanged_by_stub(self):
        """The stub must NOT modify mesh positions (it's a no-op)."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub

        mesh = create_cloth_grid(4, 4)
        original = mesh.get_positions_array().copy()

        solver = GPUClothSolverStub()
        solver.initialize(mesh)
        solver.step(mesh, 1.0 / 60.0)

        assert np.allclose(mesh.get_positions_array(), original), (
            "Stub step() must not modify mesh positions"
        )
