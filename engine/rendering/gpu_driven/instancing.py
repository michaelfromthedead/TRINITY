"""
Instance Batching for GPU-driven rendering.

Implements efficient batching of instances with the same mesh and material
for multi-draw indirect rendering.

References:
- RENDERING_CONTEXT.md Section 6.2 GPU-Driven Rendering Pipeline
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Optional, Sequence

from engine.rendering.gpu_driven.culling import Vec3, Vec4
from engine.rendering.gpu_driven.indirect_draw import (
    DrawCommand,
    DrawCommandType,
    DrawIndexedIndirectArgs,
    IndirectDrawBuffer,
    MultiDrawIndirectBuffer,
)


# =============================================================================
# INSTANCE DATA STRUCTURES
# =============================================================================


@dataclass(slots=True)
class Mat4x4:
    """4x4 Matrix for transforms (row-major storage)."""
    m: list[float] = field(default_factory=lambda: [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])

    @classmethod
    def identity(cls) -> "Mat4x4":
        """Create identity matrix."""
        return cls()

    @classmethod
    def translation(cls, x: float, y: float, z: float) -> "Mat4x4":
        """Create translation matrix."""
        mat = cls()
        mat.m[3] = x
        mat.m[7] = y
        mat.m[11] = z
        return mat

    @classmethod
    def scale(cls, sx: float, sy: float, sz: float) -> "Mat4x4":
        """Create scale matrix."""
        mat = cls()
        mat.m[0] = sx
        mat.m[5] = sy
        mat.m[10] = sz
        return mat

    @classmethod
    def from_translation_rotation_scale(
        cls,
        translation: Vec3,
        rotation_quat: tuple[float, float, float, float],
        scale: Vec3,
    ) -> "Mat4x4":
        """
        Create transform matrix from TRS.

        Args:
            translation: Translation vector
            rotation_quat: Quaternion (x, y, z, w)
            scale: Scale vector
        """
        qx, qy, qz, qw = rotation_quat

        # Compute rotation matrix from quaternion
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz

        mat = cls()
        mat.m[0] = (1.0 - 2.0 * (yy + zz)) * scale.x
        mat.m[1] = 2.0 * (xy + wz) * scale.x
        mat.m[2] = 2.0 * (xz - wy) * scale.x
        mat.m[3] = translation.x

        mat.m[4] = 2.0 * (xy - wz) * scale.y
        mat.m[5] = (1.0 - 2.0 * (xx + zz)) * scale.y
        mat.m[6] = 2.0 * (yz + wx) * scale.y
        mat.m[7] = translation.y

        mat.m[8] = 2.0 * (xz + wy) * scale.z
        mat.m[9] = 2.0 * (yz - wx) * scale.z
        mat.m[10] = (1.0 - 2.0 * (xx + yy)) * scale.z
        mat.m[11] = translation.z

        mat.m[12] = 0.0
        mat.m[13] = 0.0
        mat.m[14] = 0.0
        mat.m[15] = 1.0

        return mat

    def to_bytes(self) -> bytes:
        """Pack to GPU buffer format (16 x float32)."""
        return struct.pack("<16f", *self.m)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Mat4x4":
        """Unpack from GPU buffer format."""
        values = struct.unpack("<16f", data[:64])
        return cls(m=list(values))

    @classmethod
    def byte_size(cls) -> int:
        return 64  # 16 * sizeof(float)


@dataclass
class InstanceData:
    """
    Per-instance data for GPU rendering.

    Standard layout for instance buffer:
    - Transform matrix (4x4 float, 64 bytes)
    - Custom data (configurable)
    """
    # Unique instance identifier
    instance_id: int = 0

    # World transform matrix
    transform: Mat4x4 = field(default_factory=Mat4x4.identity)

    # LOD level for this instance
    lod_index: int = 0

    # Material override index (-1 = use default)
    material_override: int = -1

    # User-defined custom data
    custom_data: bytes = b""

    # Flags
    visible: bool = True
    cast_shadow: bool = True
    receive_shadow: bool = True

    def to_gpu_format(self, custom_data_size: int = 0) -> bytes:
        """
        Pack instance data to GPU buffer format.

        Args:
            custom_data_size: Size of custom data padding (0 = no padding)

        Returns:
            Packed bytes ready for GPU upload
        """
        data = bytearray()

        # Transform matrix (64 bytes)
        data.extend(self.transform.to_bytes())

        # Instance metadata (16 bytes)
        # uint32 instance_id
        # uint32 lod_index
        # int32 material_override
        # uint32 flags
        flags = (
            (1 if self.visible else 0)
            | (2 if self.cast_shadow else 0)
            | (4 if self.receive_shadow else 0)
        )
        data.extend(struct.pack(
            "<IIiI",
            self.instance_id,
            self.lod_index,
            self.material_override,
            flags,
        ))

        # Custom data (padded to specified size)
        if custom_data_size > 0:
            custom = self.custom_data[:custom_data_size]
            data.extend(custom)
            # Pad to specified size
            if len(custom) < custom_data_size:
                data.extend(b"\x00" * (custom_data_size - len(custom)))

        return bytes(data)

    @classmethod
    def base_size(cls) -> int:
        """Size of base instance data (transform + metadata)."""
        return 64 + 16  # 80 bytes


# =============================================================================
# BATCH KEY
# =============================================================================


@dataclass(frozen=True, slots=True)
class BatchKey:
    """
    Unique key identifying a renderable batch.

    Instances with the same batch key can be drawn together.
    """
    mesh_id: int
    material_id: int
    lod_level: int = 0
    render_layer: int = 0

    def __hash__(self) -> int:
        return hash((self.mesh_id, self.material_id, self.lod_level, self.render_layer))

    def to_sort_key(self) -> int:
        """Convert to integer sort key."""
        # Sort order: layer > material > mesh > lod
        return (
            (self.render_layer << 48)
            | (self.material_id << 32)
            | (self.mesh_id << 16)
            | self.lod_level
        )


# =============================================================================
# INSTANCE BATCH
# =============================================================================


@dataclass
class InstanceBatch:
    """
    A batch of instances sharing the same mesh and material.

    All instances in a batch can be drawn with a single instanced draw call.
    """
    # Batch identifier
    key: BatchKey

    # Mesh information
    index_count: int = 0
    first_index: int = 0
    vertex_offset: int = 0

    # Instances in this batch
    instances: list[InstanceData] = field(default_factory=list)

    # Buffer offsets (set during buffer building)
    instance_buffer_offset: int = 0

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    @property
    def is_empty(self) -> bool:
        return len(self.instances) == 0

    def add_instance(self, instance: InstanceData) -> None:
        """Add an instance to the batch."""
        self.instances.append(instance)

    def clear_instances(self) -> None:
        """Clear all instances (keep mesh/material info)."""
        self.instances.clear()

    def to_draw_args(self, first_instance: int = 0) -> DrawIndexedIndirectArgs:
        """Create draw arguments for this batch."""
        return DrawIndexedIndirectArgs(
            index_count=self.index_count,
            instance_count=self.instance_count,
            first_index=self.first_index,
            vertex_offset=self.vertex_offset,
            first_instance=first_instance,
        )


# =============================================================================
# INSTANCE BATCHER
# =============================================================================


class InstanceBatcher:
    """
    Batches instances by mesh and material for efficient rendering.

    Workflow:
    1. Register mesh/material combinations
    2. Add instances for the frame
    3. Build batches for rendering
    4. Generate draw commands
    """

    def __init__(self) -> None:
        # Registered batch templates (keyed by mesh_id + material_id)
        self._batch_templates: dict[BatchKey, InstanceBatch] = {}

        # Current frame batches
        self._active_batches: dict[BatchKey, InstanceBatch] = {}

        # Instance data buffer
        self._instance_buffer: bytearray = bytearray()
        self._custom_data_size: int = 0

    @property
    def batch_count(self) -> int:
        return len(self._active_batches)

    @property
    def instance_count(self) -> int:
        return sum(b.instance_count for b in self._active_batches.values())

    def set_custom_data_size(self, size: int) -> None:
        """Set the size of per-instance custom data."""
        self._custom_data_size = size

    def register_batch(
        self,
        mesh_id: int,
        material_id: int,
        index_count: int,
        first_index: int = 0,
        vertex_offset: int = 0,
        lod_level: int = 0,
        render_layer: int = 0,
    ) -> BatchKey:
        """
        Register a mesh/material combination for batching.

        Args:
            mesh_id: Mesh resource ID
            material_id: Material resource ID
            index_count: Number of indices in mesh
            first_index: Start index in index buffer
            vertex_offset: Vertex offset in vertex buffer
            lod_level: LOD level for this batch
            render_layer: Render layer for sorting

        Returns:
            BatchKey identifying this batch type
        """
        key = BatchKey(
            mesh_id=mesh_id,
            material_id=material_id,
            lod_level=lod_level,
            render_layer=render_layer,
        )

        batch = InstanceBatch(
            key=key,
            index_count=index_count,
            first_index=first_index,
            vertex_offset=vertex_offset,
        )

        self._batch_templates[key] = batch
        return key

    def add_instance(
        self,
        batch_key: BatchKey,
        instance: InstanceData,
    ) -> bool:
        """
        Add an instance to be rendered.

        Args:
            batch_key: Key identifying the batch type
            instance: Instance data

        Returns:
            True if instance was added, False if batch not found
        """
        if batch_key not in self._batch_templates:
            return False

        # Get or create active batch for this key
        if batch_key not in self._active_batches:
            template = self._batch_templates[batch_key]
            self._active_batches[batch_key] = InstanceBatch(
                key=batch_key,
                index_count=template.index_count,
                first_index=template.first_index,
                vertex_offset=template.vertex_offset,
            )

        self._active_batches[batch_key].add_instance(instance)
        return True

    def add_instance_quick(
        self,
        mesh_id: int,
        material_id: int,
        transform: Mat4x4,
        instance_id: int = 0,
        lod_level: int = 0,
    ) -> bool:
        """
        Quick method to add an instance with minimal parameters.

        Looks up batch by mesh_id and material_id.
        """
        key = BatchKey(mesh_id=mesh_id, material_id=material_id, lod_level=lod_level)

        if key not in self._batch_templates:
            return False

        instance = InstanceData(
            instance_id=instance_id,
            transform=transform,
            lod_index=lod_level,
        )

        return self.add_instance(key, instance)

    def clear(self) -> None:
        """Clear all active batches for a new frame."""
        self._active_batches.clear()
        self._instance_buffer.clear()

    def build_instance_buffer(self) -> bytes:
        """
        Build the instance data buffer for GPU upload.

        Returns:
            Packed instance buffer bytes
        """
        self._instance_buffer.clear()
        instance_size = InstanceData.base_size() + self._custom_data_size

        for batch in self._active_batches.values():
            batch.instance_buffer_offset = len(self._instance_buffer)

            for instance in batch.instances:
                data = instance.to_gpu_format(self._custom_data_size)
                self._instance_buffer.extend(data)

        return bytes(self._instance_buffer)

    def generate_draw_commands(
        self,
        draw_buffer: IndirectDrawBuffer,
        sort_batches: bool = True,
    ) -> int:
        """
        Generate draw commands for all batches.

        Args:
            draw_buffer: Buffer to write commands to
            sort_batches: Whether to sort batches for optimal rendering

        Returns:
            Number of draw commands generated
        """
        batches = list(self._active_batches.values())

        # Filter empty batches
        batches = [b for b in batches if not b.is_empty]

        # Sort batches by sort key
        if sort_batches:
            batches.sort(key=lambda b: b.key.to_sort_key())

        # Build instance buffer and get offsets
        self.build_instance_buffer()

        count = 0
        instance_size = InstanceData.base_size() + self._custom_data_size

        for batch in batches:
            first_instance = batch.instance_buffer_offset // instance_size

            command = DrawCommand(
                command_type=DrawCommandType.INDEXED,
                args=batch.to_draw_args(first_instance),
                mesh_id=batch.key.mesh_id,
                material_id=batch.key.material_id,
                lod_level=batch.key.lod_level,
                instance_buffer_offset=batch.instance_buffer_offset,
                sort_key=batch.key.to_sort_key(),
            )

            if draw_buffer.add_draw_command(command):
                count += 1

        return count

    def get_batches(self) -> list[InstanceBatch]:
        """Get all active batches."""
        return list(self._active_batches.values())


# =============================================================================
# MULTI-DRAW INDIRECT MANAGER
# =============================================================================


class MultiDrawIndirectManager:
    """
    Manages multi-draw indirect rendering for maximum efficiency.

    Combines multiple draw calls into a single GPU dispatch using
    multi-draw indirect, minimizing CPU-GPU synchronization.
    """

    def __init__(self, max_draws: int = 65536) -> None:
        self._multi_draw_buffer = MultiDrawIndirectBuffer(max_draws)
        self._batcher = InstanceBatcher()

        # Draw ranges for each material (for binding optimization)
        self._material_draw_ranges: dict[int, tuple[int, int]] = {}

    @property
    def batcher(self) -> InstanceBatcher:
        return self._batcher

    @property
    def draw_count(self) -> int:
        return self._multi_draw_buffer.draw_count

    def register_mesh(
        self,
        mesh_id: int,
        material_id: int,
        index_count: int,
        first_index: int = 0,
        vertex_offset: int = 0,
    ) -> BatchKey:
        """Register a mesh for multi-draw indirect."""
        return self._batcher.register_batch(
            mesh_id=mesh_id,
            material_id=material_id,
            index_count=index_count,
            first_index=first_index,
            vertex_offset=vertex_offset,
        )

    def add_instance(
        self,
        batch_key: BatchKey,
        transform: Mat4x4,
        instance_id: int = 0,
    ) -> bool:
        """Add an instance to be drawn."""
        instance = InstanceData(
            instance_id=instance_id,
            transform=transform,
        )
        return self._batcher.add_instance(batch_key, instance)

    def begin_frame(self) -> None:
        """Begin a new frame, clearing previous data."""
        self._multi_draw_buffer.clear()
        self._batcher.clear()
        self._material_draw_ranges.clear()

    def build(self) -> tuple[bytes, bytes]:
        """
        Build GPU buffers for multi-draw indirect.

        Returns:
            Tuple of (draw_args_buffer, instance_buffer)
        """
        # Build instance buffer
        instance_buffer = self._batcher.build_instance_buffer()

        # Get sorted batches
        batches = self._batcher.get_batches()
        batches = [b for b in batches if not b.is_empty]
        batches.sort(key=lambda b: b.key.to_sort_key())

        # Build multi-draw buffer and track material ranges
        instance_size = InstanceData.base_size()
        current_material = -1
        material_start = 0

        for i, batch in enumerate(batches):
            first_instance = batch.instance_buffer_offset // instance_size
            args = batch.to_draw_args(first_instance)

            self._multi_draw_buffer.add_draw(args)

            # Track material draw ranges
            if batch.key.material_id != current_material:
                if current_material >= 0:
                    self._material_draw_ranges[current_material] = (
                        material_start,
                        i - material_start,
                    )
                current_material = batch.key.material_id
                material_start = i

        # Close last material range
        if current_material >= 0:
            self._material_draw_ranges[current_material] = (
                material_start,
                len(batches) - material_start,
            )

        draw_buffer = self._multi_draw_buffer.build_buffer()
        return draw_buffer, instance_buffer

    def get_material_draw_range(self, material_id: int) -> tuple[int, int]:
        """
        Get the draw range for a specific material.

        Returns:
            (start_draw_index, draw_count) for the material
        """
        return self._material_draw_ranges.get(material_id, (0, 0))

    def get_draw_stride(self) -> int:
        """Get the stride between draw arguments."""
        return self._multi_draw_buffer.get_stride()


# =============================================================================
# INSTANCE CULLING INTEGRATION
# =============================================================================


class CulledInstanceBatcher:
    """
    Instance batcher integrated with GPU culling.

    Maintains instance bounds for culling and only batches visible instances.
    """

    def __init__(self) -> None:
        self._batcher = InstanceBatcher()
        self._instance_bounds: list[tuple[BatchKey, InstanceData]] = []

    @property
    def batcher(self) -> InstanceBatcher:
        return self._batcher

    def register_batch(
        self,
        mesh_id: int,
        material_id: int,
        index_count: int,
        first_index: int = 0,
        vertex_offset: int = 0,
    ) -> BatchKey:
        """Register a mesh/material batch."""
        return self._batcher.register_batch(
            mesh_id=mesh_id,
            material_id=material_id,
            index_count=index_count,
            first_index=first_index,
            vertex_offset=vertex_offset,
        )

    def add_instance(
        self,
        batch_key: BatchKey,
        instance: InstanceData,
    ) -> int:
        """
        Add an instance and return its index for culling reference.

        Returns:
            Index of the instance in the culling list
        """
        idx = len(self._instance_bounds)
        self._instance_bounds.append((batch_key, instance))
        return idx

    def clear(self) -> None:
        """Clear all instances for a new frame."""
        self._batcher.clear()
        self._instance_bounds.clear()

    def apply_visibility(self, visible_indices: Sequence[int]) -> None:
        """
        Apply visibility results from culling.

        Only visible instances will be batched for rendering.

        Args:
            visible_indices: Indices of visible instances from culling
        """
        for idx in visible_indices:
            if idx < len(self._instance_bounds):
                batch_key, instance = self._instance_bounds[idx]
                self._batcher.add_instance(batch_key, instance)

    def build_and_generate(
        self,
        draw_buffer: IndirectDrawBuffer,
    ) -> tuple[bytes, int]:
        """
        Build instance buffer and generate draw commands.

        Returns:
            Tuple of (instance_buffer_bytes, draw_command_count)
        """
        instance_buffer = self._batcher.build_instance_buffer()
        draw_count = self._batcher.generate_draw_commands(draw_buffer)
        return instance_buffer, draw_count


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Transform
    "Mat4x4",
    # Instance data
    "InstanceData",
    # Batching
    "BatchKey",
    "InstanceBatch",
    "InstanceBatcher",
    # Multi-draw
    "MultiDrawIndirectManager",
    # Culling integration
    "CulledInstanceBatcher",
]
