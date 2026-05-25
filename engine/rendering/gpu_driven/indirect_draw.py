"""
Indirect Draw Generation for GPU-driven rendering.

Implements GPU-side generation of indirect draw commands from culled instances.
Enables rendering thousands of objects with minimal CPU overhead.

References:
- RENDERING_CONTEXT.md Section 6.2 GPU-Driven Rendering Pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Sequence


# =============================================================================
# INDIRECT DRAW ARGUMENT STRUCTURES
# =============================================================================


@dataclass(slots=True)
class DrawIndexedIndirectArgs:
    """
    Arguments for an indexed indirect draw call.

    Matches the GPU layout for vkCmdDrawIndexedIndirect / DrawIndexedInstanced.
    """
    index_count: int = 0  # Number of indices to draw
    instance_count: int = 0  # Number of instances
    first_index: int = 0  # Start index in index buffer
    vertex_offset: int = 0  # Offset added to vertex indices
    first_instance: int = 0  # Start instance ID

    def to_bytes(self) -> bytes:
        """Pack to GPU buffer format (5 x uint32)."""
        import struct
        return struct.pack(
            "<5I",
            self.index_count,
            self.instance_count,
            self.first_index,
            self.vertex_offset,
            self.first_instance,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "DrawIndexedIndirectArgs":
        """Unpack from GPU buffer format."""
        import struct
        values = struct.unpack("<5I", data[:20])
        return cls(
            index_count=values[0],
            instance_count=values[1],
            first_index=values[2],
            vertex_offset=values[3],
            first_instance=values[4],
        )

    @classmethod
    def byte_size(cls) -> int:
        """Size in bytes of the struct."""
        return 20  # 5 * sizeof(uint32)


@dataclass(slots=True)
class DrawIndirectArgs:
    """
    Arguments for a non-indexed indirect draw call.

    Matches the GPU layout for vkCmdDrawIndirect / DrawInstanced.
    """
    vertex_count: int = 0  # Number of vertices to draw
    instance_count: int = 0  # Number of instances
    first_vertex: int = 0  # Start vertex index
    first_instance: int = 0  # Start instance ID

    def to_bytes(self) -> bytes:
        """Pack to GPU buffer format (4 x uint32)."""
        import struct
        return struct.pack(
            "<4I",
            self.vertex_count,
            self.instance_count,
            self.first_vertex,
            self.first_instance,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "DrawIndirectArgs":
        """Unpack from GPU buffer format."""
        import struct
        values = struct.unpack("<4I", data[:16])
        return cls(
            vertex_count=values[0],
            instance_count=values[1],
            first_vertex=values[2],
            first_instance=values[3],
        )

    @classmethod
    def byte_size(cls) -> int:
        """Size in bytes of the struct."""
        return 16  # 4 * sizeof(uint32)


@dataclass(slots=True)
class DispatchIndirectArgs:
    """
    Arguments for an indirect compute dispatch.

    Matches the GPU layout for vkCmdDispatchIndirect / DispatchIndirect.
    """
    group_count_x: int = 1
    group_count_y: int = 1
    group_count_z: int = 1

    def to_bytes(self) -> bytes:
        """Pack to GPU buffer format (3 x uint32)."""
        import struct
        return struct.pack(
            "<3I",
            self.group_count_x,
            self.group_count_y,
            self.group_count_z,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "DispatchIndirectArgs":
        """Unpack from GPU buffer format."""
        import struct
        values = struct.unpack("<3I", data[:12])
        return cls(
            group_count_x=values[0],
            group_count_y=values[1],
            group_count_z=values[2],
        )

    @classmethod
    def byte_size(cls) -> int:
        """Size in bytes of the struct."""
        return 12  # 3 * sizeof(uint32)


# =============================================================================
# DRAW COMMAND STRUCTURES
# =============================================================================


class DrawCommandType(IntEnum):
    """Type of draw command."""
    INDEXED = auto()  # DrawIndexedIndirect
    NON_INDEXED = auto()  # DrawIndirect
    MESH_SHADER = auto()  # DispatchMeshIndirect (for meshlet rendering)


@dataclass
class DrawCommand:
    """
    A draw command representing a single renderable batch.

    Contains all information needed to issue a draw call.
    """
    # Draw type
    command_type: DrawCommandType = DrawCommandType.INDEXED

    # Draw arguments
    args: DrawIndexedIndirectArgs | DrawIndirectArgs = field(
        default_factory=DrawIndexedIndirectArgs
    )

    # Resource bindings
    mesh_id: int = 0  # Index into mesh buffer
    material_id: int = 0  # Index into material buffer
    instance_buffer_offset: int = 0  # Offset into instance data buffer

    # LOD information
    lod_level: int = 0

    # Sorting key for state-based sorting
    sort_key: int = 0

    # Debug info
    debug_name: str = ""


# =============================================================================
# INDIRECT DRAW BUFFER
# =============================================================================


@dataclass
class IndirectDrawBufferConfig:
    """Configuration for indirect draw buffer."""
    max_draw_commands: int = 65536  # Maximum number of draw commands
    instance_data_size: int = 64  # Bytes per instance
    max_instances: int = 1_000_000  # Maximum total instances


class IndirectDrawBuffer:
    """
    GPU buffer for storing indirect draw commands.

    This buffer is written to by GPU culling compute shaders
    and read by the graphics pipeline for indirect drawing.

    Memory layout:
    - Draw count (uint32): Number of valid draw commands
    - Draw arguments array: Contiguous array of draw argument structs
    - Instance data array: Per-instance transform and custom data
    """

    def __init__(self, config: Optional[IndirectDrawBufferConfig] = None) -> None:
        self._config = config or IndirectDrawBufferConfig()
        self._draw_commands: list[DrawCommand] = []
        self._instance_data: list[bytes] = []
        self._draw_count: int = 0

        # Pre-allocate buffers
        self._draw_args_buffer: bytearray = bytearray(
            self._config.max_draw_commands * DrawIndexedIndirectArgs.byte_size()
        )
        self._count_buffer: bytearray = bytearray(4)  # Single uint32

    @property
    def config(self) -> IndirectDrawBufferConfig:
        return self._config

    @property
    def draw_count(self) -> int:
        return self._draw_count

    @property
    def draw_commands(self) -> list[DrawCommand]:
        return self._draw_commands

    def clear(self) -> None:
        """Clear all draw commands for a new frame."""
        self._draw_commands.clear()
        self._instance_data.clear()
        self._draw_count = 0

    def add_draw_command(self, command: DrawCommand) -> bool:
        """
        Add a draw command to the buffer.

        Returns:
            True if command was added, False if buffer is full
        """
        if self._draw_count >= self._config.max_draw_commands:
            return False

        self._draw_commands.append(command)
        self._draw_count += 1
        return True

    def add_instance_data(self, data: bytes) -> int:
        """
        Add instance data and return its offset.

        Args:
            data: Raw instance data bytes

        Returns:
            Offset into instance buffer
        """
        offset = len(self._instance_data) * self._config.instance_data_size
        self._instance_data.append(data)
        return offset

    def build_gpu_buffer(self) -> tuple[bytes, bytes]:
        """
        Build GPU-ready buffers.

        Returns:
            Tuple of (draw_args_buffer, count_buffer)
        """
        import struct

        # Build count buffer
        count_bytes = struct.pack("<I", self._draw_count)

        # Build draw args buffer
        draw_args = bytearray()
        for cmd in self._draw_commands:
            if isinstance(cmd.args, DrawIndexedIndirectArgs):
                draw_args.extend(cmd.args.to_bytes())
            elif isinstance(cmd.args, DrawIndirectArgs):
                draw_args.extend(cmd.args.to_bytes())

        return bytes(draw_args), count_bytes

    def get_instance_buffer(self) -> bytes:
        """Get the instance data buffer."""
        return b"".join(self._instance_data)


# =============================================================================
# MESH/MATERIAL BATCH INFO
# =============================================================================


@dataclass(slots=True)
class MeshBatchInfo:
    """Information about a mesh for batching."""
    mesh_id: int
    index_count: int
    first_index: int
    vertex_offset: int
    material_id: int
    lod_level: int = 0


@dataclass(slots=True)
class InstanceInfo:
    """Information about a single instance."""
    instance_id: int
    batch_key: int  # Combined mesh_id + material_id for grouping
    transform_data: bytes  # Raw transform matrix data
    custom_data: bytes = b""  # Additional per-instance data


# =============================================================================
# DRAW COMMAND GENERATOR
# =============================================================================


class DrawCommandGenerator:
    """
    Generates indirect draw commands from culled instances.

    Workflow:
    1. Receive list of visible instance indices from culling
    2. Group instances by mesh + material (batch key)
    3. Generate one draw command per unique batch
    4. Set up instance data offsets
    """

    def __init__(self, config: Optional[IndirectDrawBufferConfig] = None) -> None:
        self._config = config or IndirectDrawBufferConfig()
        self._mesh_batches: dict[int, MeshBatchInfo] = {}  # mesh_id -> batch info
        self._instances: list[InstanceInfo] = []

    @property
    def config(self) -> IndirectDrawBufferConfig:
        return self._config

    def register_mesh_batch(
        self,
        mesh_id: int,
        index_count: int,
        first_index: int,
        vertex_offset: int,
        material_id: int,
        lod_level: int = 0,
    ) -> None:
        """Register a mesh batch for draw command generation."""
        self._mesh_batches[mesh_id] = MeshBatchInfo(
            mesh_id=mesh_id,
            index_count=index_count,
            first_index=first_index,
            vertex_offset=vertex_offset,
            material_id=material_id,
            lod_level=lod_level,
        )

    def add_instance(
        self,
        instance_id: int,
        mesh_id: int,
        material_id: int,
        transform_data: bytes,
        custom_data: bytes = b"",
    ) -> None:
        """Add an instance to be drawn."""
        # Create batch key: combine mesh_id and material_id
        batch_key = (mesh_id << 16) | (material_id & 0xFFFF)

        self._instances.append(InstanceInfo(
            instance_id=instance_id,
            batch_key=batch_key,
            transform_data=transform_data,
            custom_data=custom_data,
        ))

    def clear(self) -> None:
        """Clear all instances for a new frame."""
        self._instances.clear()

    def generate_commands(
        self,
        visible_indices: Sequence[int],
        draw_buffer: IndirectDrawBuffer,
    ) -> None:
        """
        Generate draw commands for visible instances.

        Args:
            visible_indices: Indices of visible instances (from culling)
            draw_buffer: Buffer to write draw commands to
        """
        draw_buffer.clear()

        # Group visible instances by batch key
        batches: dict[int, list[int]] = {}  # batch_key -> list of instance indices

        for vis_idx in visible_indices:
            if vis_idx >= len(self._instances):
                continue

            instance = self._instances[vis_idx]
            if instance.batch_key not in batches:
                batches[instance.batch_key] = []
            batches[instance.batch_key].append(vis_idx)

        # Generate draw command for each batch
        for batch_key, instance_indices in batches.items():
            mesh_id = batch_key >> 16
            material_id = batch_key & 0xFFFF

            if mesh_id not in self._mesh_batches:
                continue

            mesh_batch = self._mesh_batches[mesh_id]

            # Create draw args
            args = DrawIndexedIndirectArgs(
                index_count=mesh_batch.index_count,
                instance_count=len(instance_indices),
                first_index=mesh_batch.first_index,
                vertex_offset=mesh_batch.vertex_offset,
                first_instance=0,  # Set below
            )

            # Add instance data and get offset
            first_instance_offset = 0
            for i, inst_idx in enumerate(instance_indices):
                instance = self._instances[inst_idx]
                instance_data = instance.transform_data + instance.custom_data

                if i == 0:
                    first_instance_offset = draw_buffer.add_instance_data(instance_data)
                else:
                    draw_buffer.add_instance_data(instance_data)

            args.first_instance = first_instance_offset // self._config.instance_data_size

            # Create draw command
            command = DrawCommand(
                command_type=DrawCommandType.INDEXED,
                args=args,
                mesh_id=mesh_id,
                material_id=material_id,
                instance_buffer_offset=first_instance_offset,
                lod_level=mesh_batch.lod_level,
                sort_key=self._compute_sort_key(material_id, mesh_id),
            )

            draw_buffer.add_draw_command(command)

    def _compute_sort_key(self, material_id: int, mesh_id: int) -> int:
        """
        Compute sort key for draw command ordering.

        Sorting order:
        1. Material (minimize state changes)
        2. Mesh (GPU vertex cache efficiency)
        """
        return (material_id << 16) | (mesh_id & 0xFFFF)

    def has_mesh_batch(self, mesh_id: int) -> bool:
        """Check if a mesh batch is registered."""
        return mesh_id in self._mesh_batches

    @property
    def instance_count(self) -> int:
        """Get the number of registered instances."""
        return len(self._instances)


# =============================================================================
# MULTI-DRAW INDIRECT COMMAND BUFFER
# =============================================================================


class MultiDrawIndirectBuffer:
    """
    Buffer for multi-draw indirect rendering.

    Supports rendering multiple meshes with different arguments
    in a single draw call using multi-draw indirect.
    """

    def __init__(self, max_draws: int = 65536) -> None:
        self._max_draws = max_draws
        self._draws: list[DrawIndexedIndirectArgs] = []
        self._draw_count: int = 0

    @property
    def max_draws(self) -> int:
        return self._max_draws

    @property
    def draw_count(self) -> int:
        return self._draw_count

    def clear(self) -> None:
        """Clear all draws."""
        self._draws.clear()
        self._draw_count = 0

    def add_draw(self, args: DrawIndexedIndirectArgs) -> bool:
        """Add a draw to the multi-draw buffer."""
        if self._draw_count >= self._max_draws:
            return False

        self._draws.append(args)
        self._draw_count += 1
        return True

    def build_buffer(self) -> bytes:
        """Build GPU-ready buffer with all draw arguments."""
        data = bytearray()
        for draw in self._draws:
            data.extend(draw.to_bytes())
        return bytes(data)

    def get_stride(self) -> int:
        """Get stride between draw arguments."""
        return DrawIndexedIndirectArgs.byte_size()


# =============================================================================
# DRAW COMMAND COMPACTION
# =============================================================================


class DrawCommandCompactor:
    """
    Compacts draw commands by merging compatible batches.

    Merges draw commands that:
    - Use the same mesh
    - Use the same material
    - Have contiguous instance ranges
    """

    def compact(
        self,
        commands: Sequence[DrawCommand],
    ) -> list[DrawCommand]:
        """
        Compact a list of draw commands by merging compatible batches.

        Args:
            commands: List of draw commands to compact

        Returns:
            Compacted list of draw commands
        """
        if not commands:
            return []

        # Sort by sort key for optimal merging
        sorted_commands = sorted(commands, key=lambda c: c.sort_key)

        compacted: list[DrawCommand] = []
        current = sorted_commands[0]

        for i in range(1, len(sorted_commands)):
            next_cmd = sorted_commands[i]

            # Check if commands can be merged
            if self._can_merge(current, next_cmd):
                current = self._merge(current, next_cmd)
            else:
                compacted.append(current)
                current = next_cmd

        compacted.append(current)
        return compacted

    def _can_merge(self, a: DrawCommand, b: DrawCommand) -> bool:
        """Check if two draw commands can be merged."""
        # Must be same type, mesh, material, and LOD
        if a.command_type != b.command_type:
            return False
        if a.mesh_id != b.mesh_id:
            return False
        if a.material_id != b.material_id:
            return False
        if a.lod_level != b.lod_level:
            return False

        # Check if instance ranges are contiguous
        if isinstance(a.args, DrawIndexedIndirectArgs) and isinstance(
            b.args, DrawIndexedIndirectArgs
        ):
            a_end = a.args.first_instance + a.args.instance_count
            return b.args.first_instance == a_end

        return False

    def _merge(self, a: DrawCommand, b: DrawCommand) -> DrawCommand:
        """Merge two compatible draw commands."""
        merged = DrawCommand(
            command_type=a.command_type,
            mesh_id=a.mesh_id,
            material_id=a.material_id,
            lod_level=a.lod_level,
            sort_key=a.sort_key,
            instance_buffer_offset=a.instance_buffer_offset,
        )

        if isinstance(a.args, DrawIndexedIndirectArgs) and isinstance(
            b.args, DrawIndexedIndirectArgs
        ):
            merged.args = DrawIndexedIndirectArgs(
                index_count=a.args.index_count,
                instance_count=a.args.instance_count + b.args.instance_count,
                first_index=a.args.first_index,
                vertex_offset=a.args.vertex_offset,
                first_instance=a.args.first_instance,
            )
        elif isinstance(a.args, DrawIndirectArgs) and isinstance(
            b.args, DrawIndirectArgs
        ):
            merged.args = DrawIndirectArgs(
                vertex_count=a.args.vertex_count,
                instance_count=a.args.instance_count + b.args.instance_count,
                first_vertex=a.args.first_vertex,
                first_instance=a.args.first_instance,
            )

        return merged


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Argument structures
    "DrawIndexedIndirectArgs",
    "DrawIndirectArgs",
    "DispatchIndirectArgs",
    # Draw command
    "DrawCommandType",
    "DrawCommand",
    # Buffers
    "IndirectDrawBufferConfig",
    "IndirectDrawBuffer",
    "MultiDrawIndirectBuffer",
    # Batch info
    "MeshBatchInfo",
    "InstanceInfo",
    # Generation
    "DrawCommandGenerator",
    "DrawCommandCompactor",
]
