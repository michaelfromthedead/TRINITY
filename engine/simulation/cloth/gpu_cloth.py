"""
GPU acceleration stubs for cloth simulation.

Provides interfaces for GPU-accelerated cloth solving using compute shaders.
This module defines the API - actual GPU implementation depends on the
rendering backend (Vulkan, DirectX, Metal, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

from .cloth_simulation import ClothMesh, ClothSimulationConfig


class GPUBufferUsage(Enum):
    """Buffer usage flags for GPU resources."""

    VERTEX = auto()
    INDEX = auto()
    STORAGE = auto()
    UNIFORM = auto()
    STAGING = auto()


class GPUBufferAccess(Enum):
    """Buffer access patterns."""

    READ_ONLY = auto()
    WRITE_ONLY = auto()
    READ_WRITE = auto()


@dataclass
class GPUBuffer:
    """
    Represents a GPU buffer resource.

    This is a placeholder - actual implementation depends on graphics API.
    """

    handle: Any = None  # Native handle (VkBuffer, ID3D12Resource, etc.)
    size: int = 0
    usage: GPUBufferUsage = GPUBufferUsage.STORAGE
    access: GPUBufferAccess = GPUBufferAccess.READ_WRITE

    # Staging buffer for CPU readback
    staging_buffer: Optional[Any] = None

    def is_valid(self) -> bool:
        """Check if buffer is valid."""
        return self.handle is not None


@dataclass
class GPUClothBuffers:
    """Collection of GPU buffers for cloth simulation."""

    # Position buffers (double buffered for ping-pong)
    positions_current: GPUBuffer = field(default_factory=GPUBuffer)
    positions_predicted: GPUBuffer = field(default_factory=GPUBuffer)

    # Previous positions for velocity calculation
    positions_previous: GPUBuffer = field(default_factory=GPUBuffer)

    # Velocity buffer
    velocities: GPUBuffer = field(default_factory=GPUBuffer)

    # Inverse mass per particle
    inv_masses: GPUBuffer = field(default_factory=GPUBuffer)

    # Constraint data
    distance_constraints: GPUBuffer = field(default_factory=GPUBuffer)
    bending_constraints: GPUBuffer = field(default_factory=GPUBuffer)

    # Triangle data for wind
    triangles: GPUBuffer = field(default_factory=GPUBuffer)

    # Collision data
    collision_primitives: GPUBuffer = field(default_factory=GPUBuffer)

    # Spatial hash for self-collision
    spatial_hash_table: GPUBuffer = field(default_factory=GPUBuffer)
    spatial_hash_entries: GPUBuffer = field(default_factory=GPUBuffer)

    # Uniform buffer for simulation parameters
    simulation_params: GPUBuffer = field(default_factory=GPUBuffer)

    def are_valid(self) -> bool:
        """Check if all required buffers are valid."""
        return (
            self.positions_current.is_valid()
            and self.positions_predicted.is_valid()
            and self.positions_previous.is_valid()
            and self.inv_masses.is_valid()
        )


@dataclass
class GPUComputePipeline:
    """Represents a GPU compute pipeline."""

    handle: Any = None  # Native handle
    shader_module: Any = None
    local_size: Tuple[int, int, int] = (64, 1, 1)

    def is_valid(self) -> bool:
        """Check if pipeline is valid."""
        return self.handle is not None


@dataclass
class GPUClothPipelines:
    """Collection of compute pipelines for cloth simulation."""

    # Integration pipeline (apply forces, predict positions)
    integration: GPUComputePipeline = field(default_factory=GPUComputePipeline)

    # Constraint solving pipelines
    distance_constraint: GPUComputePipeline = field(default_factory=GPUComputePipeline)
    bending_constraint: GPUComputePipeline = field(default_factory=GPUComputePipeline)

    # Collision pipelines
    collision_primitives: GPUComputePipeline = field(default_factory=GPUComputePipeline)
    self_collision_broad: GPUComputePipeline = field(default_factory=GPUComputePipeline)
    self_collision_narrow: GPUComputePipeline = field(default_factory=GPUComputePipeline)

    # Wind force pipeline
    wind_force: GPUComputePipeline = field(default_factory=GPUComputePipeline)

    # Velocity update pipeline
    velocity_update: GPUComputePipeline = field(default_factory=GPUComputePipeline)


class GPUDevice(Protocol):
    """Protocol for GPU device interface."""

    def create_buffer(
        self,
        size: int,
        usage: GPUBufferUsage,
        data: Optional[bytes] = None,
    ) -> GPUBuffer:
        """Create a GPU buffer."""
        ...

    def destroy_buffer(self, buffer: GPUBuffer) -> None:
        """Destroy a GPU buffer."""
        ...

    def upload_buffer(
        self,
        buffer: GPUBuffer,
        data: bytes,
        offset: int = 0,
    ) -> None:
        """Upload data to a GPU buffer."""
        ...

    def readback_buffer(
        self,
        buffer: GPUBuffer,
        size: int,
        offset: int = 0,
    ) -> bytes:
        """Read data back from a GPU buffer."""
        ...

    def create_compute_pipeline(
        self,
        shader_code: bytes,
        entry_point: str,
    ) -> GPUComputePipeline:
        """Create a compute pipeline."""
        ...

    def dispatch_compute(
        self,
        pipeline: GPUComputePipeline,
        groups_x: int,
        groups_y: int = 1,
        groups_z: int = 1,
    ) -> None:
        """Dispatch a compute shader."""
        ...

    def memory_barrier(self) -> None:
        """Insert a memory barrier."""
        ...


class GPUClothSolver(ABC):
    """
    Abstract base class for GPU-accelerated cloth simulation.

    Provides the interface for GPU cloth solving. Concrete implementations
    should be provided for specific graphics APIs.
    """

    def __init__(
        self,
        device: GPUDevice,
        config: Optional[ClothSimulationConfig] = None,
    ) -> None:
        """
        Initialize the GPU cloth solver.

        Args:
            device: GPU device interface
            config: Simulation configuration
        """
        self.device = device
        self.config = config or ClothSimulationConfig()

        self._buffers: Optional[GPUClothBuffers] = None
        self._pipelines: Optional[GPUClothPipelines] = None

        self._num_particles: int = 0
        self._num_distance_constraints: int = 0
        self._num_bending_constraints: int = 0
        self._num_triangles: int = 0

        self._is_initialized: bool = False

    @abstractmethod
    def initialize(self, mesh: ClothMesh) -> bool:
        """
        Initialize GPU resources for the cloth mesh.

        Args:
            mesh: The cloth mesh to simulate

        Returns:
            True if initialization succeeded
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Release all GPU resources."""
        pass

    @abstractmethod
    def prepare_buffers(self, mesh: ClothMesh) -> None:
        """
        Upload mesh data to GPU buffers.

        Args:
            mesh: The cloth mesh
        """
        pass

    @abstractmethod
    def dispatch_compute(self, dt: float) -> None:
        """
        Dispatch compute shaders for one simulation step.

        Args:
            dt: Timestep duration
        """
        pass

    @abstractmethod
    def readback_positions(self, mesh: ClothMesh) -> None:
        """
        Read simulated positions back to CPU.

        Args:
            mesh: The cloth mesh to update
        """
        pass

    def step(self, mesh: ClothMesh, dt: float) -> None:
        """
        Perform one complete simulation step on GPU.

        Args:
            mesh: The cloth mesh
            dt: Timestep duration
        """
        if not self._is_initialized:
            return

        self.prepare_buffers(mesh)
        self.dispatch_compute(dt)
        self.readback_positions(mesh)


class GPUClothSolverStub(GPUClothSolver):
    """
    Stub implementation of GPU cloth solver.

    This is a placeholder that does NOT perform simulation.
    When GPU is unavailable, use ClothSimulation (CPU) directly instead.

    NOTE: This stub exists only to satisfy the GPU solver interface for testing.
    It does NOT fall back to CPU - if you need simulation, use ClothSimulation.
    """

    def __init__(
        self,
        device: Optional[GPUDevice] = None,
        config: Optional[ClothSimulationConfig] = None,
    ) -> None:
        """Initialize stub solver."""
        # Use a dummy device if none provided
        super().__init__(device, config)  # type: ignore
        self._is_initialized = True
        self._mesh: Optional[ClothMesh] = None

    def initialize(self, mesh: ClothMesh) -> bool:
        """Initialize with mesh reference for potential CPU fallback."""
        self._num_particles = len(mesh.particles)
        self._num_distance_constraints = len(mesh.edges)
        self._num_triangles = len(mesh.triangles)
        self._mesh = mesh
        return True

    def shutdown(self) -> None:
        """Shutdown and release mesh reference."""
        self._mesh = None
        self._is_initialized = False

    def prepare_buffers(self, mesh: ClothMesh) -> None:
        """Prepare buffers - stores mesh reference for step()."""
        self._mesh = mesh

    def dispatch_compute(self, dt: float) -> None:
        """
        Dispatch compute - STUB: Does nothing.

        WARNING: This stub does not simulate. Use ClothSimulation for CPU simulation.
        This method intentionally does nothing to make it clear that GPU simulation
        requires a real implementation (Vulkan, DirectX, Metal, etc.).
        """
        # Intentional no-op - GPU stubs do not simulate
        # Use ClothSimulation for CPU-based simulation
        pass

    def readback_positions(self, mesh: ClothMesh) -> None:
        """Readback positions - no-op since dispatch_compute doesn't modify anything."""
        pass

    def step(self, mesh: ClothMesh, dt: float) -> None:
        """
        Perform one complete simulation step.

        WARNING: This stub does NOT simulate. The mesh will not be modified.
        For actual simulation, use ClothSimulation instead.

        Args:
            mesh: The cloth mesh (will NOT be modified)
            dt: Timestep duration (ignored)
        """
        if not self._is_initialized:
            return
        # Intentionally does nothing - this is a stub
        # Document clearly that no simulation occurs


# =============================================================================
# Compute Shader Source Templates (GLSL/SPIR-V style pseudocode)
# =============================================================================

INTEGRATION_SHADER_TEMPLATE = """
// Integration compute shader
// Workgroup size: 64 particles per invocation

layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer PositionsCurrent {
    vec3 positions_current[];
};

layout(std430, binding = 1) buffer PositionsPrevious {
    vec3 positions_previous[];
};

layout(std430, binding = 2) buffer PositionsPredicted {
    vec3 positions_predicted[];
};

layout(std430, binding = 3) buffer Velocities {
    vec3 velocities[];
};

layout(std430, binding = 4) buffer InvMasses {
    float inv_masses[];
};

layout(std140, binding = 5) uniform SimParams {
    vec3 gravity;
    float dt;
    float damping;
    int num_particles;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_particles) return;

    float inv_mass = inv_masses[idx];
    if (inv_mass == 0.0) {
        // Pinned particle - don't move
        positions_predicted[idx] = positions_current[idx];
        return;
    }

    vec3 pos = positions_current[idx];
    vec3 vel = velocities[idx];

    // Apply gravity
    vel += gravity * dt;

    // Predict position
    vec3 predicted = pos + vel * dt;

    // Apply damping
    vel *= damping;

    // Store results
    positions_previous[idx] = pos;
    positions_predicted[idx] = predicted;
    velocities[idx] = vel;
}
"""

DISTANCE_CONSTRAINT_SHADER_TEMPLATE = """
// Distance constraint compute shader
// Solves one distance constraint per invocation

layout(local_size_x = 64) in;

struct DistanceConstraint {
    uint p0;
    uint p1;
    float rest_length;
    float stiffness;
};

layout(std430, binding = 0) buffer Positions {
    vec3 positions[];
};

layout(std430, binding = 1) buffer InvMasses {
    float inv_masses[];
};

layout(std430, binding = 2) buffer Constraints {
    DistanceConstraint constraints[];
};

layout(std140, binding = 3) uniform Params {
    int num_constraints;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_constraints) return;

    DistanceConstraint c = constraints[idx];

    vec3 p0 = positions[c.p0];
    vec3 p1 = positions[c.p1];
    float w0 = inv_masses[c.p0];
    float w1 = inv_masses[c.p1];

    vec3 delta = p1 - p0;
    float current_length = length(delta);

    if (current_length < 1e-8) return;

    float error = current_length - c.rest_length;
    float w_sum = w0 + w1;

    if (w_sum < 1e-8) return;

    float correction = error / current_length * c.stiffness;
    vec3 dir = delta / current_length;

    // Apply corrections (using atomics for thread safety)
    if (w0 > 0.0) {
        atomicAdd(positions[c.p0].x, dir.x * correction * w0 / w_sum);
        atomicAdd(positions[c.p0].y, dir.y * correction * w0 / w_sum);
        atomicAdd(positions[c.p0].z, dir.z * correction * w0 / w_sum);
    }

    if (w1 > 0.0) {
        atomicAdd(positions[c.p1].x, -dir.x * correction * w1 / w_sum);
        atomicAdd(positions[c.p1].y, -dir.y * correction * w1 / w_sum);
        atomicAdd(positions[c.p1].z, -dir.z * correction * w1 / w_sum);
    }
}
"""

VELOCITY_UPDATE_SHADER_TEMPLATE = """
// Velocity update compute shader
// Updates velocities from position changes

layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer PositionsCurrent {
    vec3 positions_current[];
};

layout(std430, binding = 1) buffer PositionsPrevious {
    vec3 positions_previous[];
};

layout(std430, binding = 2) buffer Velocities {
    vec3 velocities[];
};

layout(std140, binding = 3) uniform Params {
    float inv_dt;
    float damping;
    int num_particles;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_particles) return;

    vec3 pos_curr = positions_current[idx];
    vec3 pos_prev = positions_previous[idx];

    // Velocity from position delta
    vec3 vel = (pos_curr - pos_prev) * inv_dt;

    // Apply damping
    vel *= damping;

    velocities[idx] = vel;
}
"""


def get_shader_templates() -> Dict[str, str]:
    """
    Get compute shader templates.

    Returns:
        Dictionary of shader name to source code
    """
    return {
        "integration": INTEGRATION_SHADER_TEMPLATE,
        "distance_constraint": DISTANCE_CONSTRAINT_SHADER_TEMPLATE,
        "velocity_update": VELOCITY_UPDATE_SHADER_TEMPLATE,
    }


def calculate_workgroups(
    num_items: int,
    local_size: int = 64,
) -> int:
    """
    Calculate number of workgroups needed.

    Args:
        num_items: Total items to process
        local_size: Items per workgroup

    Returns:
        Number of workgroups
    """
    return (num_items + local_size - 1) // local_size
