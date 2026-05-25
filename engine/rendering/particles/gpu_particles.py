"""
GPU Compute Particles Implementation.

Provides GPU-accelerated particle simulation using compute shaders.
Uses Structure-of-Arrays (SoA) layout for cache-efficient GPU access.

Architecture:
    GPUParticleBuffer - SoA layout buffer for particle attributes
    GPUParticleAttributes - Particle attribute definitions
    GPUParticleSimulator - Compute shader simulation dispatcher
    GPUParticleRenderer - GPU buffer to draw call renderer

Supports @gpu_particle decorator configuration for:
    - Custom attribute lists
    - Custom compute shaders
    - Flexible attribute types
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from engine.rendering.particles.particle_system import (
    Particle,
    ParticleState,
    SimulationMode,
    Vec3,
    Vec4,
)
from engine.rendering.particles.constants import PARTICLE_CONSTANTS


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class AttributeType(Enum):
    """GPU particle attribute data types."""

    FLOAT = auto()  # 4 bytes
    FLOAT2 = auto()  # 8 bytes (vec2)
    FLOAT3 = auto()  # 12 bytes (vec3)
    FLOAT4 = auto()  # 16 bytes (vec4)
    INT = auto()  # 4 bytes
    INT2 = auto()  # 8 bytes (ivec2)
    INT3 = auto()  # 12 bytes (ivec3)
    INT4 = auto()  # 16 bytes (ivec4)
    UINT = auto()  # 4 bytes
    UINT2 = auto()  # 8 bytes (uvec2)
    UINT3 = auto()  # 12 bytes (uvec3)
    UINT4 = auto()  # 16 bytes (uvec4)


class BufferUsage(Enum):
    """GPU buffer usage flags."""

    STORAGE = auto()  # Storage buffer (read/write in compute)
    VERTEX = auto()  # Vertex buffer (read in vertex shader)
    INDIRECT = auto()  # Indirect draw arguments
    UNIFORM = auto()  # Uniform buffer


class DrawMode(Enum):
    """Particle rendering draw mode."""

    POINTS = auto()  # GL_POINTS / point primitives
    QUADS = auto()  # Camera-facing quads (2 triangles per particle)
    MESH = auto()  # Instanced mesh rendering


# Attribute type sizes in bytes
ATTRIBUTE_SIZES: dict[AttributeType, int] = {
    AttributeType.FLOAT: 4,
    AttributeType.FLOAT2: 8,
    AttributeType.FLOAT3: 12,
    AttributeType.FLOAT4: 16,
    AttributeType.INT: 4,
    AttributeType.INT2: 8,
    AttributeType.INT3: 12,
    AttributeType.INT4: 16,
    AttributeType.UINT: 4,
    AttributeType.UINT2: 8,
    AttributeType.UINT3: 12,
    AttributeType.UINT4: 16,
}

# Standard particle attributes
STANDARD_ATTRIBUTES: dict[str, AttributeType] = {
    "position": AttributeType.FLOAT3,
    "velocity": AttributeType.FLOAT3,
    "acceleration": AttributeType.FLOAT3,
    "color": AttributeType.FLOAT4,
    "life": AttributeType.FLOAT2,  # (current_age, max_lifetime)
    "size": AttributeType.FLOAT2,  # (width, height)
    "rotation": AttributeType.FLOAT2,  # (angle, angular_velocity)
    "uv": AttributeType.FLOAT4,  # (u_min, v_min, u_max, v_max)
    "random": AttributeType.FLOAT4,  # Random seeds per particle
    "flags": AttributeType.UINT,  # Bitfield flags
}


# =============================================================================
# GPU PARTICLE ATTRIBUTES
# =============================================================================


@dataclass
class GPUParticleAttribute:
    """Definition of a single GPU particle attribute."""

    name: str
    attr_type: AttributeType
    offset: int = 0  # Byte offset in buffer (set during layout)
    binding: int = 0  # Shader binding point

    @property
    def size(self) -> int:
        """Size in bytes."""
        return ATTRIBUTE_SIZES[self.attr_type]

    @property
    def component_count(self) -> int:
        """Number of components (1-4)."""
        type_name = self.attr_type.name
        if type_name.endswith("4"):
            return 4
        elif type_name.endswith("3"):
            return 3
        elif type_name.endswith("2"):
            return 2
        return 1


@dataclass
class GPUParticleAttributes:
    """
    Collection of particle attributes for GPU buffer layout.

    Manages the SoA (Structure of Arrays) layout where each attribute
    has its own contiguous buffer region for optimal GPU access.
    """

    attributes: list[GPUParticleAttribute] = field(default_factory=list)
    _layout_computed: bool = False
    _total_stride: int = 0

    @classmethod
    def from_names(cls, attribute_names: list[str]) -> "GPUParticleAttributes":
        """
        Create attributes from a list of standard attribute names.

        Args:
            attribute_names: List of attribute names (e.g., ["position", "velocity", "color"])

        Returns:
            GPUParticleAttributes with standard types for each name
        """
        attrs = cls()
        for name in attribute_names:
            if name in STANDARD_ATTRIBUTES:
                attrs.add_attribute(name, STANDARD_ATTRIBUTES[name])
            else:
                # Default unknown attributes to FLOAT4
                attrs.add_attribute(name, AttributeType.FLOAT4)
        attrs.compute_layout()
        return attrs

    def add_attribute(
        self,
        name: str,
        attr_type: AttributeType,
        binding: int = -1,
    ) -> None:
        """Add an attribute to the layout."""
        if binding < 0:
            binding = len(self.attributes)
        self.attributes.append(
            GPUParticleAttribute(name=name, attr_type=attr_type, binding=binding)
        )
        self._layout_computed = False

    def compute_layout(self) -> None:
        """Compute byte offsets for SoA layout."""
        offset = 0
        for attr in self.attributes:
            # Align to 4-byte boundary
            if offset % 4 != 0:
                offset += 4 - (offset % 4)
            attr.offset = offset
            offset += attr.size
        self._total_stride = offset
        self._layout_computed = True

    @property
    def stride(self) -> int:
        """Total stride per particle in bytes."""
        if not self._layout_computed:
            self.compute_layout()
        return self._total_stride

    def get_attribute(self, name: str) -> Optional[GPUParticleAttribute]:
        """Get attribute by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr
        return None

    def has_attribute(self, name: str) -> bool:
        """Check if attribute exists."""
        return self.get_attribute(name) is not None


# =============================================================================
# GPU PARTICLE BUFFER
# =============================================================================


@dataclass
class BufferAllocation:
    """Represents a GPU buffer allocation."""

    size_bytes: int
    usage: BufferUsage
    data: Optional[bytes] = None
    handle: Optional[Any] = None  # GPU-specific handle


class GPUParticleBuffer:
    """
    Structure-of-Arrays (SoA) GPU buffer for particle attributes.

    Each attribute has its own contiguous memory region:
    [position0, position1, ..., positionN][velocity0, velocity1, ..., velocityN]...

    This layout is optimal for GPU compute shaders that process
    one attribute at a time across all particles.

    Attributes:
        max_particles: Maximum number of particles
        attributes: Attribute layout definition
    """

    def __init__(
        self,
        max_particles: int,
        attributes: GPUParticleAttributes,
    ) -> None:
        self._max_particles = max_particles
        self._attributes = attributes
        self._alive_count = 0

        # Ensure layout is computed
        if not attributes._layout_computed:
            attributes.compute_layout()

        # Calculate buffer sizes
        self._attribute_buffers: dict[str, BufferAllocation] = {}
        self._allocate_buffers()

        # Indirect draw buffer for GPU-driven rendering
        self._indirect_buffer: Optional[BufferAllocation] = None
        self._counter_buffer: Optional[BufferAllocation] = None

    def _allocate_buffers(self) -> None:
        """Allocate GPU buffers for each attribute."""
        for attr in self._attributes.attributes:
            buffer_size = attr.size * self._max_particles
            self._attribute_buffers[attr.name] = BufferAllocation(
                size_bytes=buffer_size,
                usage=BufferUsage.STORAGE,
            )

    @property
    def max_particles(self) -> int:
        return self._max_particles

    @property
    def alive_count(self) -> int:
        return self._alive_count

    @property
    def attributes(self) -> GPUParticleAttributes:
        return self._attributes

    def get_buffer(self, attribute_name: str) -> Optional[BufferAllocation]:
        """Get buffer allocation for an attribute."""
        return self._attribute_buffers.get(attribute_name)

    def get_buffer_size(self, attribute_name: str) -> int:
        """Get buffer size in bytes for an attribute."""
        buf = self._attribute_buffers.get(attribute_name)
        return buf.size_bytes if buf else 0

    def get_total_size(self) -> int:
        """Get total GPU memory usage in bytes."""
        return sum(buf.size_bytes for buf in self._attribute_buffers.values())

    def set_alive_count(self, count: int) -> None:
        """Set the number of alive particles (from GPU counter)."""
        self._alive_count = min(count, self._max_particles)

    def allocate_indirect_buffer(self) -> None:
        """Allocate indirect draw arguments buffer."""
        # IndirectDrawArgs: vertex_count(4), instance_count(4), first_vertex(4), first_instance(4)
        self._indirect_buffer = BufferAllocation(
            size_bytes=PARTICLE_CONSTANTS.INDIRECT_BUFFER_SIZE,
            usage=BufferUsage.INDIRECT,
        )

    def allocate_counter_buffer(self) -> None:
        """Allocate atomic counter buffer for alive particles."""
        self._counter_buffer = BufferAllocation(
            size_bytes=PARTICLE_CONSTANTS.COUNTER_BUFFER_SIZE,
            usage=BufferUsage.STORAGE,
        )

    @property
    def indirect_buffer(self) -> Optional[BufferAllocation]:
        return self._indirect_buffer

    @property
    def counter_buffer(self) -> Optional[BufferAllocation]:
        return self._counter_buffer


# =============================================================================
# GPU PARTICLE SIMULATOR
# =============================================================================


@dataclass
class ComputeShaderConfig:
    """Configuration for a compute shader."""

    shader_path: Optional[str] = None
    shader_source: Optional[str] = None
    workgroup_size: Tuple[int, int, int] = (64, 1, 1)
    entry_point: str = "main"


class GPUParticleSimulator:
    """
    GPU compute shader particle simulation.

    Runs simulation passes on the GPU:
    1. Spawn pass: Initialize new particles
    2. Update pass: Apply forces, age particles
    3. Compact pass: Remove dead particles, update counter
    4. Sort pass (optional): Sort for rendering order

    The simulator uses a ping-pong buffer scheme for double buffering.
    """

    def __init__(
        self,
        buffer: GPUParticleBuffer,
        compute_shader: Optional[str] = None,
    ) -> None:
        self._buffer = buffer
        self._compute_shader = compute_shader

        # Compute configurations using centralized workgroup sizes
        workgroup = (
            PARTICLE_CONSTANTS.WORKGROUP_SIZE_X,
            PARTICLE_CONSTANTS.WORKGROUP_SIZE_Y,
            PARTICLE_CONSTANTS.WORKGROUP_SIZE_Z,
        )
        self._spawn_config = ComputeShaderConfig(
            shader_path=compute_shader,
            workgroup_size=workgroup,
        )
        self._update_config = ComputeShaderConfig(
            shader_path=compute_shader,
            workgroup_size=workgroup,
        )
        self._compact_config = ComputeShaderConfig(
            workgroup_size=workgroup,
        )

        # Simulation parameters using centralized constants
        self._gravity = Vec3(0, PARTICLE_CONSTANTS.DEFAULT_GRAVITY_Y, 0)
        self._wind = Vec3(0, 0, 0)
        self._time_scale = 1.0
        bounds_val = PARTICLE_CONSTANTS.DEFAULT_BOUNDS_MIN
        self._bounds_min = Vec3(bounds_val, bounds_val, bounds_val)
        bounds_val = PARTICLE_CONSTANTS.DEFAULT_BOUNDS_MAX
        self._bounds_max = Vec3(bounds_val, bounds_val, bounds_val)

        # Per-frame uniforms
        self._uniforms: dict[str, Any] = {}

    @property
    def buffer(self) -> GPUParticleBuffer:
        return self._buffer

    def set_gravity(self, gravity: Vec3) -> None:
        """Set gravity vector for simulation."""
        self._gravity = gravity

    def set_wind(self, wind: Vec3) -> None:
        """Set global wind vector."""
        self._wind = wind

    def set_bounds(self, min_bounds: Vec3, max_bounds: Vec3) -> None:
        """Set simulation bounds for particle death/wrapping."""
        self._bounds_min = min_bounds
        self._bounds_max = max_bounds

    def set_uniform(self, name: str, value: Any) -> None:
        """Set a custom uniform for the compute shader."""
        self._uniforms[name] = value

    def _calculate_dispatch_size(self, particle_count: int) -> Tuple[int, int, int]:
        """Calculate compute dispatch dimensions."""
        workgroup_x = self._update_config.workgroup_size[0]
        dispatch_x = (particle_count + workgroup_x - 1) // workgroup_x
        return (dispatch_x, 1, 1)

    def spawn(self, count: int, spawn_params: dict[str, Any]) -> int:
        """
        Spawn new particles via compute shader.

        Args:
            count: Number of particles to spawn
            spawn_params: Parameters for spawn shader (position, velocity, etc.)

        Returns:
            Number of particles actually spawned
        """
        # Calculate available slots
        available = self._buffer.max_particles - self._buffer.alive_count
        actual_spawn = min(count, available)

        if actual_spawn <= 0:
            return 0

        # In actual implementation, this would:
        # 1. Bind spawn compute shader
        # 2. Set uniforms (spawn_params, count, etc.)
        # 3. Dispatch compute shader
        # 4. Update alive count

        # Placeholder: just update the count
        self._buffer.set_alive_count(self._buffer.alive_count + actual_spawn)
        return actual_spawn

    def update(self, dt: float) -> None:
        """
        Run simulation update pass.

        Args:
            dt: Delta time in seconds
        """
        if self._buffer.alive_count == 0:
            return

        dispatch_size = self._calculate_dispatch_size(self._buffer.alive_count)

        # In actual implementation, this would:
        # 1. Bind update compute shader
        # 2. Set uniforms (dt, gravity, wind, etc.)
        # 3. Bind attribute buffers
        # 4. Dispatch compute shader
        # 5. Memory barrier

        # Placeholder for simulation logic
        pass

    def compact(self) -> None:
        """
        Run compaction pass to remove dead particles.

        Uses stream compaction to gather alive particles contiguously.
        """
        if self._buffer.alive_count == 0:
            return

        # In actual implementation, this would:
        # 1. Run prefix sum on alive flags
        # 2. Scatter alive particles to new positions
        # 3. Read back or use atomic counter for new alive count

        # Placeholder
        pass

    def sort(self, camera_position: Vec3) -> None:
        """
        Sort particles by distance to camera (for correct alpha blending).

        Uses GPU radix sort or bitonic sort.
        """
        if self._buffer.alive_count == 0:
            return

        # In actual implementation:
        # 1. Calculate sort keys (distance to camera)
        # 2. Run GPU sort algorithm
        # 3. Reorder particle indices

        pass

    def get_stats(self) -> dict[str, Any]:
        """Get simulator statistics."""
        return {
            "max_particles": self._buffer.max_particles,
            "alive_particles": self._buffer.alive_count,
            "buffer_size_bytes": self._buffer.get_total_size(),
            "gravity": (self._gravity.x, self._gravity.y, self._gravity.z),
            "wind": (self._wind.x, self._wind.y, self._wind.z),
        }


# =============================================================================
# GPU PARTICLE RENDERER
# =============================================================================


@dataclass
class RenderState:
    """Rendering state configuration."""

    blend_enabled: bool = True
    blend_src: str = "SRC_ALPHA"
    blend_dst: str = "ONE_MINUS_SRC_ALPHA"
    depth_write: bool = False
    depth_test: bool = True
    cull_face: bool = False


class GPUParticleRenderer:
    """
    Renderer for GPU particle buffers.

    Draws particles directly from GPU buffers without CPU readback.
    Supports multiple draw modes:
    - Points: Simple point sprites
    - Quads: Camera-facing billboard quads
    - Mesh: Instanced mesh particles
    """

    def __init__(
        self,
        buffer: GPUParticleBuffer,
        draw_mode: DrawMode = DrawMode.QUADS,
    ) -> None:
        self._buffer = buffer
        self._draw_mode = draw_mode
        self._render_state = RenderState()

        # Shader paths
        self._vertex_shader: Optional[str] = None
        self._fragment_shader: Optional[str] = None
        self._geometry_shader: Optional[str] = None  # For point-to-quad expansion

        # Texture atlas for particle sprites
        self._texture_atlas: Optional[Any] = None
        self._atlas_rows: int = 1
        self._atlas_cols: int = 1

        # Camera-facing parameters
        self._camera_position = Vec3()
        self._camera_up = Vec3(0, 1, 0)
        self._camera_right = Vec3(1, 0, 0)

    @property
    def buffer(self) -> GPUParticleBuffer:
        return self._buffer

    @property
    def draw_mode(self) -> DrawMode:
        return self._draw_mode

    @draw_mode.setter
    def draw_mode(self, mode: DrawMode) -> None:
        self._draw_mode = mode

    def set_render_state(self, state: RenderState) -> None:
        """Set rendering state configuration."""
        self._render_state = state

    def set_texture_atlas(
        self,
        texture: Any,
        rows: int = 1,
        cols: int = 1,
    ) -> None:
        """Set texture atlas for particle sprites."""
        self._texture_atlas = texture
        self._atlas_rows = rows
        self._atlas_cols = cols

    def set_camera(
        self,
        position: Vec3,
        up: Vec3,
        right: Vec3,
    ) -> None:
        """Set camera vectors for billboard orientation."""
        self._camera_position = position
        self._camera_up = up
        self._camera_right = right

    def set_shaders(
        self,
        vertex: Optional[str] = None,
        fragment: Optional[str] = None,
        geometry: Optional[str] = None,
    ) -> None:
        """Set custom shaders for rendering."""
        self._vertex_shader = vertex
        self._fragment_shader = fragment
        self._geometry_shader = geometry

    def render(self) -> int:
        """
        Render particles from GPU buffer.

        Returns:
            Number of particles rendered
        """
        if self._buffer.alive_count == 0:
            return 0

        # In actual implementation, this would:
        # 1. Bind shaders
        # 2. Set render state (blending, depth, etc.)
        # 3. Bind attribute buffers as vertex inputs
        # 4. Set uniforms (camera matrices, atlas info, etc.)
        # 5. Issue draw call

        if self._draw_mode == DrawMode.POINTS:
            return self._render_points()
        elif self._draw_mode == DrawMode.QUADS:
            return self._render_quads()
        elif self._draw_mode == DrawMode.MESH:
            return self._render_mesh()

        return 0

    def _render_points(self) -> int:
        """Render as point sprites."""
        # glDrawArrays(GL_POINTS, 0, alive_count)
        return self._buffer.alive_count

    def _render_quads(self) -> int:
        """Render as camera-facing quads."""
        # If using geometry shader: glDrawArrays(GL_POINTS, 0, alive_count)
        # If using instancing: glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, alive_count)
        return self._buffer.alive_count

    def _render_mesh(self) -> int:
        """Render as instanced meshes."""
        # glDrawElementsInstanced(mesh_indices, alive_count)
        return self._buffer.alive_count

    def render_indirect(self) -> int:
        """
        Render using indirect draw buffer.

        The GPU writes draw arguments directly, avoiding CPU readback.
        """
        if self._buffer.indirect_buffer is None:
            self._buffer.allocate_indirect_buffer()

        # In actual implementation:
        # glDrawArraysIndirect(GL_POINTS, indirect_buffer)

        return self._buffer.alive_count


# =============================================================================
# GPU PARTICLE CONFIG
# =============================================================================


@dataclass(frozen=True)
class GPUParticleConfig:
    """
    Configuration for GPU particle system from @gpu_particle decorator.

    Attributes:
        attributes: List of attribute names
        compute_shader: Path to custom compute shader
        max_particles: Maximum particle count
        draw_mode: How to render particles
    """

    attributes: tuple[str, ...] = ("position", "velocity", "color", "life")
    compute_shader: Optional[str] = None
    max_particles: int = PARTICLE_CONSTANTS.GPU_MAX_PARTICLES
    draw_mode: DrawMode = DrawMode.QUADS

    @classmethod
    def from_decorator_params(
        cls,
        attributes: list[str],
        compute_shader: Optional[str] = None,
        **kwargs: Any,
    ) -> "GPUParticleConfig":
        """Create config from @gpu_particle decorator parameters."""
        return cls(
            attributes=tuple(attributes),
            compute_shader=compute_shader,
            **kwargs,
        )


# =============================================================================
# GPU PARTICLE SYSTEM (Combined)
# =============================================================================


class GPUParticleSystem:
    """
    Complete GPU particle system combining buffer, simulator, and renderer.

    Provides a high-level interface for GPU-accelerated particle effects.
    """

    def __init__(
        self,
        config: GPUParticleConfig,
    ) -> None:
        self._config = config

        # Create attribute layout
        self._attributes = GPUParticleAttributes.from_names(list(config.attributes))

        # Create buffer
        self._buffer = GPUParticleBuffer(
            max_particles=config.max_particles,
            attributes=self._attributes,
        )

        # Allocate indirect buffers
        self._buffer.allocate_indirect_buffer()
        self._buffer.allocate_counter_buffer()

        # Create simulator and renderer
        self._simulator = GPUParticleSimulator(
            buffer=self._buffer,
            compute_shader=config.compute_shader,
        )
        self._renderer = GPUParticleRenderer(
            buffer=self._buffer,
            draw_mode=config.draw_mode,
        )

    @property
    def config(self) -> GPUParticleConfig:
        return self._config

    @property
    def buffer(self) -> GPUParticleBuffer:
        return self._buffer

    @property
    def simulator(self) -> GPUParticleSimulator:
        return self._simulator

    @property
    def renderer(self) -> GPUParticleRenderer:
        return self._renderer

    @property
    def alive_count(self) -> int:
        return self._buffer.alive_count

    def spawn(self, count: int, **spawn_params: Any) -> int:
        """Spawn new particles."""
        return self._simulator.spawn(count, spawn_params)

    def update(self, dt: float) -> None:
        """Update simulation."""
        self._simulator.update(dt)
        self._simulator.compact()

    def render(
        self,
        camera_position: Optional[Vec3] = None,
        sort_by_depth: bool = True,
    ) -> int:
        """
        Render particles.

        Args:
            camera_position: Camera position for billboarding and sorting
            sort_by_depth: Whether to sort particles by depth

        Returns:
            Number of particles rendered
        """
        if camera_position and sort_by_depth:
            self._simulator.sort(camera_position)

        return self._renderer.render()

    def get_stats(self) -> dict[str, Any]:
        """Get combined statistics."""
        return {
            "config": {
                "attributes": self._config.attributes,
                "max_particles": self._config.max_particles,
                "draw_mode": self._config.draw_mode.name,
            },
            "simulator": self._simulator.get_stats(),
            "buffer_size_bytes": self._buffer.get_total_size(),
        }


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Enums
    "AttributeType",
    "BufferUsage",
    "DrawMode",
    # Constants
    "ATTRIBUTE_SIZES",
    "STANDARD_ATTRIBUTES",
    # Attributes
    "GPUParticleAttribute",
    "GPUParticleAttributes",
    # Buffer
    "BufferAllocation",
    "GPUParticleBuffer",
    # Simulator
    "ComputeShaderConfig",
    "GPUParticleSimulator",
    # Renderer
    "RenderState",
    "GPUParticleRenderer",
    # Config
    "GPUParticleConfig",
    # Combined System
    "GPUParticleSystem",
]
