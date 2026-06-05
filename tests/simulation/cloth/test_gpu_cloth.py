"""
Whitebox tests for GPU cloth solver stubs and utilities.

Tests:
- GPUBufferUsage: buffer usage enum
- GPUBufferAccess: buffer access enum
- GPUBuffer: buffer data class
- GPUClothBuffers: cloth simulation buffers
- GPUComputePipeline: compute pipeline data class
- GPUClothPipelines: cloth solver pipelines
- GPUClothSolverStub: stub solver implementation
- Utility functions: calculate_workgroups, get_shader_templates
"""

import numpy as np
import pytest

from engine.simulation.cloth.cloth_simulation import ClothMesh, ClothParticle, create_cloth_grid
from engine.simulation.cloth.gpu_cloth import (
    GPUBuffer,
    GPUBufferAccess,
    GPUBufferUsage,
    GPUClothBuffers,
    GPUClothPipelines,
    GPUClothSolverStub,
    GPUComputePipeline,
    calculate_workgroups,
    get_shader_templates,
)


class TestGPUBufferUsage:
    """Test GPUBufferUsage enum."""

    def test_buffer_usage_values(self):
        """Test that all expected usage values exist."""
        assert GPUBufferUsage.VERTEX is not None
        assert GPUBufferUsage.INDEX is not None
        assert GPUBufferUsage.STORAGE is not None
        assert GPUBufferUsage.UNIFORM is not None
        assert GPUBufferUsage.STAGING is not None

    def test_buffer_usage_distinct(self):
        """Test that usage values are distinct."""
        usages = [
            GPUBufferUsage.VERTEX,
            GPUBufferUsage.INDEX,
            GPUBufferUsage.STORAGE,
            GPUBufferUsage.UNIFORM,
            GPUBufferUsage.STAGING,
        ]
        assert len(set(usages)) == len(usages)


class TestGPUBufferAccess:
    """Test GPUBufferAccess enum."""

    def test_buffer_access_values(self):
        """Test that all expected access values exist."""
        assert GPUBufferAccess.READ_ONLY is not None
        assert GPUBufferAccess.WRITE_ONLY is not None
        assert GPUBufferAccess.READ_WRITE is not None

    def test_buffer_access_distinct(self):
        """Test that access values are distinct."""
        accesses = [
            GPUBufferAccess.READ_ONLY,
            GPUBufferAccess.WRITE_ONLY,
            GPUBufferAccess.READ_WRITE,
        ]
        assert len(set(accesses)) == len(accesses)


class TestGPUBuffer:
    """Test GPUBuffer data class."""

    def test_buffer_default_values(self):
        """Test buffer default values."""
        buffer = GPUBuffer()

        assert buffer.handle is None
        assert buffer.size == 0
        assert buffer.usage == GPUBufferUsage.STORAGE
        assert buffer.access == GPUBufferAccess.READ_WRITE
        assert buffer.staging_buffer is None

    def test_buffer_custom_values(self):
        """Test buffer with custom values."""
        buffer = GPUBuffer(
            handle="mock_handle",
            size=1024,
            usage=GPUBufferUsage.VERTEX,
            access=GPUBufferAccess.READ_ONLY,
        )

        assert buffer.handle == "mock_handle"
        assert buffer.size == 1024
        assert buffer.usage == GPUBufferUsage.VERTEX
        assert buffer.access == GPUBufferAccess.READ_ONLY

    def test_buffer_is_valid_with_handle(self):
        """Buffer with handle should be valid."""
        buffer = GPUBuffer(handle="mock_handle")

        assert buffer.is_valid()

    def test_buffer_is_invalid_without_handle(self):
        """Buffer without handle should be invalid."""
        buffer = GPUBuffer()

        assert not buffer.is_valid()


class TestGPUClothBuffers:
    """Test GPUClothBuffers data class."""

    def test_cloth_buffers_default_values(self):
        """Test cloth buffers default values."""
        buffers = GPUClothBuffers()

        assert not buffers.positions_current.is_valid()
        assert not buffers.positions_predicted.is_valid()
        assert not buffers.velocities.is_valid()

    def test_cloth_buffers_are_valid_all_required(self):
        """Test are_valid requires all essential buffers."""
        buffers = GPUClothBuffers()

        # Initially invalid
        assert not buffers.are_valid()

        # Set only some required buffers
        buffers.positions_current = GPUBuffer(handle="h1", size=100)
        assert not buffers.are_valid()

        # Set all required
        buffers.positions_predicted = GPUBuffer(handle="h2", size=100)
        buffers.positions_previous = GPUBuffer(handle="h3", size=100)
        buffers.inv_masses = GPUBuffer(handle="h4", size=100)

        assert buffers.are_valid()


class TestGPUComputePipeline:
    """Test GPUComputePipeline data class."""

    def test_pipeline_default_values(self):
        """Test pipeline default values."""
        pipeline = GPUComputePipeline()

        assert pipeline.handle is None
        assert pipeline.shader_module is None
        assert pipeline.local_size == (64, 1, 1)

    def test_pipeline_custom_values(self):
        """Test pipeline with custom values."""
        pipeline = GPUComputePipeline(
            handle="pipeline_handle",
            shader_module="shader",
            local_size=(128, 2, 1),
        )

        assert pipeline.handle == "pipeline_handle"
        assert pipeline.shader_module == "shader"
        assert pipeline.local_size == (128, 2, 1)

    def test_pipeline_is_valid_with_handle(self):
        """Pipeline with handle should be valid."""
        pipeline = GPUComputePipeline(handle="handle")

        assert pipeline.is_valid()

    def test_pipeline_is_invalid_without_handle(self):
        """Pipeline without handle should be invalid."""
        pipeline = GPUComputePipeline()

        assert not pipeline.is_valid()


class TestGPUClothPipelines:
    """Test GPUClothPipelines data class."""

    def test_pipelines_default_values(self):
        """Test pipelines default values."""
        pipelines = GPUClothPipelines()

        assert not pipelines.integration.is_valid()
        assert not pipelines.distance_constraint.is_valid()
        assert not pipelines.bending_constraint.is_valid()
        assert not pipelines.collision_primitives.is_valid()
        assert not pipelines.self_collision_broad.is_valid()
        assert not pipelines.self_collision_narrow.is_valid()
        assert not pipelines.wind_force.is_valid()
        assert not pipelines.velocity_update.is_valid()


class TestGPUClothSolverStub:
    """Test GPUClothSolverStub class."""

    @pytest.fixture
    def simple_mesh(self):
        """Create a simple cloth mesh."""
        return create_cloth_grid(width=4, height=4, pin_top=True)

    def test_stub_creation_no_device(self):
        """Test stub creation without device."""
        solver = GPUClothSolverStub()

        assert solver._is_initialized

    def test_stub_initialize(self, simple_mesh):
        """Test stub initialization with mesh."""
        solver = GPUClothSolverStub()

        result = solver.initialize(simple_mesh)

        assert result is True
        assert solver._num_particles == len(simple_mesh.particles)
        assert solver._num_distance_constraints == len(simple_mesh.edges)
        assert solver._num_triangles == len(simple_mesh.triangles)

    def test_stub_shutdown(self, simple_mesh):
        """Test stub shutdown."""
        solver = GPUClothSolverStub()
        solver.initialize(simple_mesh)

        solver.shutdown()

        assert solver._mesh is None
        assert not solver._is_initialized

    def test_stub_prepare_buffers(self, simple_mesh):
        """Test stub prepare_buffers stores mesh reference."""
        solver = GPUClothSolverStub()

        solver.prepare_buffers(simple_mesh)

        assert solver._mesh is simple_mesh

    def test_stub_dispatch_compute_no_op(self, simple_mesh):
        """Test stub dispatch_compute does nothing."""
        solver = GPUClothSolverStub()
        solver.initialize(simple_mesh)

        # Get initial positions
        initial_positions = simple_mesh.get_positions_array().copy()

        solver.dispatch_compute(dt=0.016)

        # Positions should not change (stub does nothing)
        final_positions = simple_mesh.get_positions_array()
        assert np.allclose(initial_positions, final_positions)

    def test_stub_readback_positions_no_op(self, simple_mesh):
        """Test stub readback_positions does nothing."""
        solver = GPUClothSolverStub()
        solver.initialize(simple_mesh)

        # Should not crash
        solver.readback_positions(simple_mesh)

    def test_stub_step_does_nothing(self, simple_mesh):
        """Test stub step does not modify mesh."""
        solver = GPUClothSolverStub()
        solver.initialize(simple_mesh)

        initial_positions = simple_mesh.get_positions_array().copy()

        solver.step(simple_mesh, dt=0.016)

        # Stub does not simulate - positions unchanged
        final_positions = simple_mesh.get_positions_array()
        assert np.allclose(initial_positions, final_positions)

    def test_stub_step_when_not_initialized(self, simple_mesh):
        """Test step does nothing when not initialized."""
        solver = GPUClothSolverStub()
        solver._is_initialized = False

        # Should not crash
        solver.step(simple_mesh, dt=0.016)


class TestCalculateWorkgroups:
    """Test calculate_workgroups utility function."""

    def test_exact_multiple(self):
        """Test when items is exact multiple of local size."""
        groups = calculate_workgroups(num_items=128, local_size=64)

        assert groups == 2

    def test_not_exact_multiple(self):
        """Test when items is not exact multiple of local size."""
        groups = calculate_workgroups(num_items=100, local_size=64)

        assert groups == 2  # ceil(100/64) = 2

    def test_less_than_local_size(self):
        """Test when items is less than local size."""
        groups = calculate_workgroups(num_items=30, local_size=64)

        assert groups == 1

    def test_zero_items(self):
        """Test with zero items."""
        groups = calculate_workgroups(num_items=0, local_size=64)

        assert groups == 0

    def test_default_local_size(self):
        """Test default local size is 64."""
        groups = calculate_workgroups(num_items=128)

        assert groups == 2

    def test_large_item_count(self):
        """Test with large item count."""
        groups = calculate_workgroups(num_items=10000, local_size=64)

        assert groups == 157  # ceil(10000/64) = 157


class TestGetShaderTemplates:
    """Test get_shader_templates utility function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        templates = get_shader_templates()

        assert isinstance(templates, dict)

    def test_contains_integration_shader(self):
        """Test that integration shader template exists."""
        templates = get_shader_templates()

        assert "integration" in templates
        assert len(templates["integration"]) > 0

    def test_contains_distance_constraint_shader(self):
        """Test that distance constraint shader template exists."""
        templates = get_shader_templates()

        assert "distance_constraint" in templates
        assert len(templates["distance_constraint"]) > 0

    def test_contains_velocity_update_shader(self):
        """Test that velocity update shader template exists."""
        templates = get_shader_templates()

        assert "velocity_update" in templates
        assert len(templates["velocity_update"]) > 0

    def test_shader_templates_are_strings(self):
        """Test that all templates are strings."""
        templates = get_shader_templates()

        for name, template in templates.items():
            assert isinstance(template, str), f"Template {name} should be string"

    def test_integration_shader_has_gravity(self):
        """Test that integration shader mentions gravity."""
        templates = get_shader_templates()

        assert "gravity" in templates["integration"].lower()

    def test_distance_constraint_shader_has_rest_length(self):
        """Test that distance constraint shader mentions rest_length."""
        templates = get_shader_templates()

        assert "rest_length" in templates["distance_constraint"]

    def test_velocity_update_shader_has_damping(self):
        """Test that velocity update shader mentions damping."""
        templates = get_shader_templates()

        assert "damping" in templates["velocity_update"]
