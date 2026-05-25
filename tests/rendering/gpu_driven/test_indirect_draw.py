"""
Tests for Indirect Draw Generation.

Tests indirect draw argument structures, command generation, and multi-draw buffers.
"""

import struct
import pytest

from engine.rendering.gpu_driven.indirect_draw import (
    DrawIndexedIndirectArgs,
    DrawIndirectArgs,
    DispatchIndirectArgs,
    DrawCommandType,
    DrawCommand,
    IndirectDrawBufferConfig,
    IndirectDrawBuffer,
    MultiDrawIndirectBuffer,
    MeshBatchInfo,
    InstanceInfo,
    DrawCommandGenerator,
    DrawCommandCompactor,
)


# =============================================================================
# INDIRECT DRAW ARGUMENT TESTS
# =============================================================================


class TestDrawIndexedIndirectArgs:
    """Tests for DrawIndexedIndirectArgs."""

    def test_creation(self) -> None:
        """Test DrawIndexedIndirectArgs creation."""
        args = DrawIndexedIndirectArgs(
            index_count=36,
            instance_count=100,
            first_index=0,
            vertex_offset=0,
            first_instance=0,
        )

        assert args.index_count == 36
        assert args.instance_count == 100

    def test_to_bytes(self) -> None:
        """Test packing to bytes."""
        args = DrawIndexedIndirectArgs(
            index_count=36,
            instance_count=100,
            first_index=10,
            vertex_offset=5,
            first_instance=50,
        )

        data = args.to_bytes()
        assert len(data) == 20  # 5 * 4 bytes

        # Verify packed values
        values = struct.unpack("<5I", data)
        assert values == (36, 100, 10, 5, 50)

    def test_from_bytes(self) -> None:
        """Test unpacking from bytes."""
        original = DrawIndexedIndirectArgs(
            index_count=36,
            instance_count=100,
            first_index=10,
            vertex_offset=5,
            first_instance=50,
        )

        data = original.to_bytes()
        restored = DrawIndexedIndirectArgs.from_bytes(data)

        assert restored.index_count == original.index_count
        assert restored.instance_count == original.instance_count
        assert restored.first_index == original.first_index
        assert restored.vertex_offset == original.vertex_offset
        assert restored.first_instance == original.first_instance

    def test_byte_size(self) -> None:
        """Test byte size constant."""
        assert DrawIndexedIndirectArgs.byte_size() == 20


class TestDrawIndirectArgs:
    """Tests for DrawIndirectArgs."""

    def test_creation(self) -> None:
        """Test DrawIndirectArgs creation."""
        args = DrawIndirectArgs(
            vertex_count=24,
            instance_count=50,
            first_vertex=0,
            first_instance=0,
        )

        assert args.vertex_count == 24
        assert args.instance_count == 50

    def test_to_bytes(self) -> None:
        """Test packing to bytes."""
        args = DrawIndirectArgs(
            vertex_count=24,
            instance_count=50,
            first_vertex=10,
            first_instance=25,
        )

        data = args.to_bytes()
        assert len(data) == 16

        values = struct.unpack("<4I", data)
        assert values == (24, 50, 10, 25)

    def test_byte_size(self) -> None:
        """Test byte size constant."""
        assert DrawIndirectArgs.byte_size() == 16


class TestDispatchIndirectArgs:
    """Tests for DispatchIndirectArgs."""

    def test_creation(self) -> None:
        """Test DispatchIndirectArgs creation."""
        args = DispatchIndirectArgs(
            group_count_x=64,
            group_count_y=32,
            group_count_z=1,
        )

        assert args.group_count_x == 64
        assert args.group_count_y == 32
        assert args.group_count_z == 1

    def test_to_bytes(self) -> None:
        """Test packing to bytes."""
        args = DispatchIndirectArgs(
            group_count_x=64,
            group_count_y=32,
            group_count_z=16,
        )

        data = args.to_bytes()
        assert len(data) == 12

        values = struct.unpack("<3I", data)
        assert values == (64, 32, 16)


# =============================================================================
# DRAW COMMAND TESTS
# =============================================================================


class TestDrawCommand:
    """Tests for DrawCommand."""

    def test_creation(self) -> None:
        """Test DrawCommand creation."""
        args = DrawIndexedIndirectArgs(index_count=36, instance_count=10)
        command = DrawCommand(
            command_type=DrawCommandType.INDEXED,
            args=args,
            mesh_id=1,
            material_id=2,
            lod_level=0,
        )

        assert command.command_type == DrawCommandType.INDEXED
        assert command.mesh_id == 1
        assert command.material_id == 2

    def test_sort_key(self) -> None:
        """Test sort key computation."""
        cmd1 = DrawCommand(mesh_id=0, material_id=0, sort_key=0)
        cmd2 = DrawCommand(mesh_id=0, material_id=1, sort_key=1)

        assert cmd1.sort_key < cmd2.sort_key


# =============================================================================
# INDIRECT DRAW BUFFER TESTS
# =============================================================================


class TestIndirectDrawBuffer:
    """Tests for IndirectDrawBuffer."""

    def test_creation(self) -> None:
        """Test IndirectDrawBuffer creation."""
        config = IndirectDrawBufferConfig(max_draw_commands=1000)
        buffer = IndirectDrawBuffer(config)

        assert buffer.config.max_draw_commands == 1000
        assert buffer.draw_count == 0

    def test_add_draw_command(self) -> None:
        """Test adding draw commands."""
        buffer = IndirectDrawBuffer()

        args = DrawIndexedIndirectArgs(index_count=36, instance_count=10)
        command = DrawCommand(args=args, mesh_id=1, material_id=1)

        result = buffer.add_draw_command(command)
        assert result
        assert buffer.draw_count == 1

    def test_buffer_full(self) -> None:
        """Test buffer full condition."""
        config = IndirectDrawBufferConfig(max_draw_commands=2)
        buffer = IndirectDrawBuffer(config)

        args = DrawIndexedIndirectArgs(index_count=36, instance_count=10)

        # Add up to limit
        buffer.add_draw_command(DrawCommand(args=args))
        buffer.add_draw_command(DrawCommand(args=args))

        # Should fail
        result = buffer.add_draw_command(DrawCommand(args=args))
        assert not result
        assert buffer.draw_count == 2

    def test_clear(self) -> None:
        """Test clearing buffer."""
        buffer = IndirectDrawBuffer()

        args = DrawIndexedIndirectArgs(index_count=36, instance_count=10)
        buffer.add_draw_command(DrawCommand(args=args))

        buffer.clear()
        assert buffer.draw_count == 0
        assert len(buffer.draw_commands) == 0

    def test_add_instance_data(self) -> None:
        """Test adding instance data."""
        buffer = IndirectDrawBuffer(IndirectDrawBufferConfig(instance_data_size=64))

        data = b"\x00" * 64
        offset = buffer.add_instance_data(data)

        assert offset == 0

        offset2 = buffer.add_instance_data(data)
        assert offset2 == 64

    def test_build_gpu_buffer(self) -> None:
        """Test building GPU buffer."""
        buffer = IndirectDrawBuffer()

        args = DrawIndexedIndirectArgs(index_count=36, instance_count=10)
        buffer.add_draw_command(DrawCommand(args=args))

        draw_args, count = buffer.build_gpu_buffer()

        # Verify count
        count_value = struct.unpack("<I", count)[0]
        assert count_value == 1

        # Verify draw args
        assert len(draw_args) == DrawIndexedIndirectArgs.byte_size()


# =============================================================================
# MULTI-DRAW INDIRECT BUFFER TESTS
# =============================================================================


class TestMultiDrawIndirectBuffer:
    """Tests for MultiDrawIndirectBuffer."""

    def test_creation(self) -> None:
        """Test MultiDrawIndirectBuffer creation."""
        buffer = MultiDrawIndirectBuffer(max_draws=1000)

        assert buffer.max_draws == 1000
        assert buffer.draw_count == 0

    def test_add_draws(self) -> None:
        """Test adding draws."""
        buffer = MultiDrawIndirectBuffer()

        args = DrawIndexedIndirectArgs(index_count=36, instance_count=10)
        result = buffer.add_draw(args)

        assert result
        assert buffer.draw_count == 1

    def test_build_buffer(self) -> None:
        """Test building buffer."""
        buffer = MultiDrawIndirectBuffer()

        args1 = DrawIndexedIndirectArgs(index_count=36, instance_count=10)
        args2 = DrawIndexedIndirectArgs(index_count=24, instance_count=5)

        buffer.add_draw(args1)
        buffer.add_draw(args2)

        data = buffer.build_buffer()
        expected_size = DrawIndexedIndirectArgs.byte_size() * 2
        assert len(data) == expected_size

    def test_stride(self) -> None:
        """Test stride calculation."""
        buffer = MultiDrawIndirectBuffer()
        assert buffer.get_stride() == DrawIndexedIndirectArgs.byte_size()


# =============================================================================
# DRAW COMMAND GENERATOR TESTS
# =============================================================================


class TestDrawCommandGenerator:
    """Tests for DrawCommandGenerator."""

    def test_register_mesh_batch(self) -> None:
        """Test registering mesh batches."""
        generator = DrawCommandGenerator()

        generator.register_mesh_batch(
            mesh_id=1,
            index_count=36,
            first_index=0,
            vertex_offset=0,
            material_id=1,
        )

        # Batch should be registered
        assert generator.has_mesh_batch(1)

    def test_add_instance(self) -> None:
        """Test adding instances."""
        generator = DrawCommandGenerator()

        transform_data = b"\x00" * 64
        generator.add_instance(
            instance_id=0,
            mesh_id=1,
            material_id=1,
            transform_data=transform_data,
        )

        assert generator.instance_count == 1

    def test_generate_commands(self) -> None:
        """Test generating draw commands."""
        generator = DrawCommandGenerator()

        # Register mesh
        generator.register_mesh_batch(
            mesh_id=1,
            index_count=36,
            first_index=0,
            vertex_offset=0,
            material_id=1,
        )

        # Add instances
        transform_data = b"\x00" * 64
        generator.add_instance(0, 1, 1, transform_data)
        generator.add_instance(1, 1, 1, transform_data)

        # Generate commands
        draw_buffer = IndirectDrawBuffer()
        generator.generate_commands([0, 1], draw_buffer)

        assert draw_buffer.draw_count == 1  # Single batch
        assert draw_buffer.draw_commands[0].args.instance_count == 2

    def test_multiple_batches(self) -> None:
        """Test generating commands for multiple batches."""
        generator = DrawCommandGenerator()

        # Register two meshes
        generator.register_mesh_batch(
            mesh_id=1,
            index_count=36,
            first_index=0,
            vertex_offset=0,
            material_id=1,
        )
        generator.register_mesh_batch(
            mesh_id=2,
            index_count=24,
            first_index=0,
            vertex_offset=0,
            material_id=2,
        )

        # Add instances for each
        transform_data = b"\x00" * 64
        generator.add_instance(0, 1, 1, transform_data)
        generator.add_instance(1, 2, 2, transform_data)

        # Generate commands
        draw_buffer = IndirectDrawBuffer()
        generator.generate_commands([0, 1], draw_buffer)

        assert draw_buffer.draw_count == 2

    def test_clear(self) -> None:
        """Test clearing instances."""
        generator = DrawCommandGenerator()

        transform_data = b"\x00" * 64
        generator.add_instance(0, 1, 1, transform_data)

        generator.clear()
        assert generator.instance_count == 0


# =============================================================================
# DRAW COMMAND COMPACTOR TESTS
# =============================================================================


class TestDrawCommandCompactor:
    """Tests for DrawCommandCompactor."""

    def test_compact_mergeable(self) -> None:
        """Test compacting mergeable commands."""
        compactor = DrawCommandCompactor()

        # Two commands that can be merged
        cmd1 = DrawCommand(
            command_type=DrawCommandType.INDEXED,
            args=DrawIndexedIndirectArgs(
                index_count=36,
                instance_count=10,
                first_instance=0,
            ),
            mesh_id=1,
            material_id=1,
            lod_level=0,
            sort_key=0,
        )
        cmd2 = DrawCommand(
            command_type=DrawCommandType.INDEXED,
            args=DrawIndexedIndirectArgs(
                index_count=36,
                instance_count=5,
                first_instance=10,  # Contiguous with cmd1
            ),
            mesh_id=1,
            material_id=1,
            lod_level=0,
            sort_key=0,
        )

        compacted = compactor.compact([cmd1, cmd2])

        assert len(compacted) == 1
        assert compacted[0].args.instance_count == 15

    def test_compact_not_mergeable(self) -> None:
        """Test commands that cannot be merged."""
        compactor = DrawCommandCompactor()

        # Different materials
        cmd1 = DrawCommand(
            command_type=DrawCommandType.INDEXED,
            args=DrawIndexedIndirectArgs(index_count=36, instance_count=10),
            mesh_id=1,
            material_id=1,
            sort_key=1,
        )
        cmd2 = DrawCommand(
            command_type=DrawCommandType.INDEXED,
            args=DrawIndexedIndirectArgs(index_count=36, instance_count=5),
            mesh_id=1,
            material_id=2,  # Different material
            sort_key=2,
        )

        compacted = compactor.compact([cmd1, cmd2])

        assert len(compacted) == 2

    def test_compact_empty(self) -> None:
        """Test compacting empty list."""
        compactor = DrawCommandCompactor()
        compacted = compactor.compact([])
        assert len(compacted) == 0
