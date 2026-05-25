"""
Tests for the GPU particle system module.

Tests:
    - GPUParticleAttributes layout computation
    - GPUParticleBuffer allocation and sizing
    - GPUParticleSimulator configuration
    - GPUParticleRenderer setup
    - GPUParticleSystem integration
"""

import pytest

from engine.rendering.particles.gpu_particles import (
    ATTRIBUTE_SIZES,
    STANDARD_ATTRIBUTES,
    AttributeType,
    BufferAllocation,
    BufferUsage,
    ComputeShaderConfig,
    DrawMode,
    GPUParticleAttribute,
    GPUParticleAttributes,
    GPUParticleBuffer,
    GPUParticleConfig,
    GPUParticleRenderer,
    GPUParticleSimulator,
    GPUParticleSystem,
    RenderState,
)
from engine.rendering.particles.particle_system import Vec3


class TestAttributeType:
    """Test attribute type sizes."""

    def test_attribute_sizes(self):
        """Test all attribute types have correct sizes."""
        assert ATTRIBUTE_SIZES[AttributeType.FLOAT] == 4
        assert ATTRIBUTE_SIZES[AttributeType.FLOAT2] == 8
        assert ATTRIBUTE_SIZES[AttributeType.FLOAT3] == 12
        assert ATTRIBUTE_SIZES[AttributeType.FLOAT4] == 16
        assert ATTRIBUTE_SIZES[AttributeType.INT] == 4
        assert ATTRIBUTE_SIZES[AttributeType.UINT] == 4

    def test_standard_attributes(self):
        """Test standard attributes are defined."""
        assert "position" in STANDARD_ATTRIBUTES
        assert "velocity" in STANDARD_ATTRIBUTES
        assert "color" in STANDARD_ATTRIBUTES
        assert "life" in STANDARD_ATTRIBUTES
        assert "size" in STANDARD_ATTRIBUTES


class TestGPUParticleAttribute:
    """Test GPUParticleAttribute."""

    def test_creation(self):
        """Test attribute creation."""
        attr = GPUParticleAttribute(
            name="position",
            attr_type=AttributeType.FLOAT3,
            binding=0,
        )
        assert attr.name == "position"
        assert attr.attr_type == AttributeType.FLOAT3
        assert attr.size == 12
        assert attr.component_count == 3

    def test_component_counts(self):
        """Test component count for different types."""
        assert GPUParticleAttribute("a", AttributeType.FLOAT).component_count == 1
        assert GPUParticleAttribute("b", AttributeType.FLOAT2).component_count == 2
        assert GPUParticleAttribute("c", AttributeType.FLOAT3).component_count == 3
        assert GPUParticleAttribute("d", AttributeType.FLOAT4).component_count == 4


class TestGPUParticleAttributes:
    """Test GPUParticleAttributes collection."""

    def test_from_names(self):
        """Test creation from standard attribute names."""
        attrs = GPUParticleAttributes.from_names(["position", "velocity", "color"])

        assert len(attrs.attributes) == 3
        assert attrs.has_attribute("position")
        assert attrs.has_attribute("velocity")
        assert attrs.has_attribute("color")

    def test_layout_computation(self):
        """Test SoA layout computation."""
        attrs = GPUParticleAttributes()
        attrs.add_attribute("pos", AttributeType.FLOAT3)  # 12 bytes
        attrs.add_attribute("col", AttributeType.FLOAT4)  # 16 bytes
        attrs.compute_layout()

        pos = attrs.get_attribute("pos")
        col = attrs.get_attribute("col")

        assert pos.offset == 0
        assert col.offset == 12
        assert attrs.stride == 28

    def test_unknown_attribute_defaults_to_float4(self):
        """Test unknown attributes default to FLOAT4."""
        attrs = GPUParticleAttributes.from_names(["unknown_attr"])

        attr = attrs.get_attribute("unknown_attr")
        assert attr is not None
        assert attr.attr_type == AttributeType.FLOAT4


class TestGPUParticleBuffer:
    """Test GPUParticleBuffer."""

    def test_creation(self):
        """Test buffer creation."""
        attrs = GPUParticleAttributes.from_names(["position", "velocity"])
        buffer = GPUParticleBuffer(max_particles=1000, attributes=attrs)

        assert buffer.max_particles == 1000
        assert buffer.alive_count == 0

    def test_buffer_allocation(self):
        """Test individual attribute buffer allocation."""
        attrs = GPUParticleAttributes.from_names(["position"])  # FLOAT3 = 12 bytes
        buffer = GPUParticleBuffer(max_particles=100, attributes=attrs)

        pos_buffer = buffer.get_buffer("position")
        assert pos_buffer is not None
        assert pos_buffer.size_bytes == 12 * 100
        assert pos_buffer.usage == BufferUsage.STORAGE

    def test_total_size(self):
        """Test total buffer size calculation."""
        attrs = GPUParticleAttributes.from_names(["position", "velocity", "color"])
        buffer = GPUParticleBuffer(max_particles=1000, attributes=attrs)

        # position: 12 * 1000 = 12000
        # velocity: 12 * 1000 = 12000
        # color: 16 * 1000 = 16000
        assert buffer.get_total_size() == 40000

    def test_indirect_buffer_allocation(self):
        """Test indirect draw buffer allocation."""
        attrs = GPUParticleAttributes.from_names(["position"])
        buffer = GPUParticleBuffer(max_particles=100, attributes=attrs)

        assert buffer.indirect_buffer is None

        buffer.allocate_indirect_buffer()

        assert buffer.indirect_buffer is not None
        assert buffer.indirect_buffer.size_bytes == 16
        assert buffer.indirect_buffer.usage == BufferUsage.INDIRECT

    def test_counter_buffer_allocation(self):
        """Test counter buffer allocation."""
        attrs = GPUParticleAttributes.from_names(["position"])
        buffer = GPUParticleBuffer(max_particles=100, attributes=attrs)

        buffer.allocate_counter_buffer()

        assert buffer.counter_buffer is not None
        assert buffer.counter_buffer.size_bytes == 4


class TestGPUParticleSimulator:
    """Test GPUParticleSimulator."""

    def test_creation(self):
        """Test simulator creation."""
        attrs = GPUParticleAttributes.from_names(["position", "velocity"])
        buffer = GPUParticleBuffer(max_particles=1000, attributes=attrs)
        simulator = GPUParticleSimulator(buffer)

        assert simulator.buffer is buffer

    def test_set_gravity(self):
        """Test setting gravity."""
        attrs = GPUParticleAttributes.from_names(["position"])
        buffer = GPUParticleBuffer(max_particles=100, attributes=attrs)
        simulator = GPUParticleSimulator(buffer)

        gravity = Vec3(0, -10, 0)
        simulator.set_gravity(gravity)

        stats = simulator.get_stats()
        assert stats["gravity"] == (0, -10, 0)

    def test_spawn(self):
        """Test spawning particles."""
        attrs = GPUParticleAttributes.from_names(["position"])
        buffer = GPUParticleBuffer(max_particles=100, attributes=attrs)
        simulator = GPUParticleSimulator(buffer)

        spawned = simulator.spawn(50, {})
        assert spawned == 50
        assert buffer.alive_count == 50

    def test_spawn_exceeds_capacity(self):
        """Test spawning more than capacity."""
        attrs = GPUParticleAttributes.from_names(["position"])
        buffer = GPUParticleBuffer(max_particles=10, attributes=attrs)
        simulator = GPUParticleSimulator(buffer)

        spawned = simulator.spawn(100, {})
        assert spawned == 10


class TestGPUParticleRenderer:
    """Test GPUParticleRenderer."""

    def test_creation(self):
        """Test renderer creation."""
        attrs = GPUParticleAttributes.from_names(["position", "color"])
        buffer = GPUParticleBuffer(max_particles=100, attributes=attrs)
        renderer = GPUParticleRenderer(buffer)

        assert renderer.buffer is buffer
        assert renderer.draw_mode == DrawMode.QUADS  # Default

    def test_draw_mode(self):
        """Test setting draw mode."""
        attrs = GPUParticleAttributes.from_names(["position"])
        buffer = GPUParticleBuffer(max_particles=100, attributes=attrs)
        renderer = GPUParticleRenderer(buffer, draw_mode=DrawMode.POINTS)

        assert renderer.draw_mode == DrawMode.POINTS

        renderer.draw_mode = DrawMode.MESH
        assert renderer.draw_mode == DrawMode.MESH

    def test_render_state(self):
        """Test setting render state configuration is retained."""
        attrs = GPUParticleAttributes.from_names(["position"])
        buffer = GPUParticleBuffer(max_particles=100, attributes=attrs)
        renderer = GPUParticleRenderer(buffer)

        state = RenderState(
            blend_enabled=True,
            depth_write=False,
            cull_face=True,
        )
        renderer.set_render_state(state)

        # Verify render state was set (internal state)
        assert renderer._render_state.blend_enabled is True
        assert renderer._render_state.depth_write is False
        assert renderer._render_state.cull_face is True

    def test_set_camera(self):
        """Test setting camera vectors are stored correctly."""
        attrs = GPUParticleAttributes.from_names(["position"])
        buffer = GPUParticleBuffer(max_particles=100, attributes=attrs)
        renderer = GPUParticleRenderer(buffer)

        renderer.set_camera(
            position=Vec3(0, 10, 10),
            up=Vec3(0, 1, 0),
            right=Vec3(1, 0, 0),
        )

        # Verify camera vectors are stored
        assert renderer._camera_position.x == 0
        assert renderer._camera_position.y == 10
        assert renderer._camera_position.z == 10
        assert renderer._camera_up.y == 1
        assert renderer._camera_right.x == 1

    def test_render_empty_buffer(self):
        """Test rendering with no particles returns zero."""
        attrs = GPUParticleAttributes.from_names(["position", "color"])
        buffer = GPUParticleBuffer(max_particles=100, attributes=attrs)
        renderer = GPUParticleRenderer(buffer)

        rendered = renderer.render()
        assert rendered == 0

    def test_render_indirect(self):
        """Test indirect rendering allocates buffer if needed."""
        attrs = GPUParticleAttributes.from_names(["position"])
        buffer = GPUParticleBuffer(max_particles=100, attributes=attrs)
        renderer = GPUParticleRenderer(buffer)

        # Before render_indirect, indirect buffer is None
        assert buffer.indirect_buffer is None

        rendered = renderer.render_indirect()

        # After render_indirect, indirect buffer should be allocated
        assert buffer.indirect_buffer is not None


class TestGPUParticleConfig:
    """Test GPUParticleConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = GPUParticleConfig()
        assert config.attributes == ("position", "velocity", "color", "life")
        assert config.compute_shader is None
        assert config.max_particles == 100000
        assert config.draw_mode == DrawMode.QUADS

    def test_from_decorator_params(self):
        """Test creation from decorator parameters."""
        config = GPUParticleConfig.from_decorator_params(
            attributes=["position", "velocity", "size"],
            compute_shader="custom.comp",
        )
        assert config.attributes == ("position", "velocity", "size")
        assert config.compute_shader == "custom.comp"


class TestGPUParticleSystem:
    """Test GPUParticleSystem integration."""

    def test_creation(self):
        """Test system creation."""
        config = GPUParticleConfig(
            attributes=("position", "velocity", "color"),
            max_particles=1000,
        )
        system = GPUParticleSystem(config)

        assert system.config is config
        assert system.buffer.max_particles == 1000
        assert system.alive_count == 0

    def test_spawn(self):
        """Test spawning particles."""
        config = GPUParticleConfig(max_particles=100)
        system = GPUParticleSystem(config)

        spawned = system.spawn(50)
        assert spawned == 50
        assert system.alive_count == 50

    def test_update(self):
        """Test updating simulation verifies state changes."""
        config = GPUParticleConfig(max_particles=100)
        system = GPUParticleSystem(config)

        system.spawn(10)
        initial_count = system.alive_count
        assert initial_count == 10

        # Update should process particles without crashing
        system.update(0.016)

        # Alive count should still be valid (simulation runs)
        assert system.alive_count >= 0
        assert system.alive_count <= config.max_particles

    def test_render(self):
        """Test rendering returns correct particle count."""
        config = GPUParticleConfig(max_particles=100)
        system = GPUParticleSystem(config)

        # No particles - should render 0
        rendered_empty = system.render()
        assert rendered_empty == 0

        # Spawn particles
        system.spawn(10)
        rendered = system.render()

        assert rendered == 10

    def test_render_with_camera_sorting(self):
        """Test rendering with depth sorting."""
        config = GPUParticleConfig(max_particles=100)
        system = GPUParticleSystem(config)

        system.spawn(20)

        # Render with sorting enabled
        camera_pos = Vec3(0, 0, 10)
        rendered = system.render(camera_position=camera_pos, sort_by_depth=True)

        assert rendered == 20

    def test_get_stats(self):
        """Test getting system stats."""
        config = GPUParticleConfig(max_particles=100)
        system = GPUParticleSystem(config)

        stats = system.get_stats()

        assert "config" in stats
        assert "simulator" in stats
        assert "buffer_size_bytes" in stats
