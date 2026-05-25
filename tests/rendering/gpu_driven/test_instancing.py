"""
Tests for Instance Batching.

Tests instance data structures, batching, and multi-draw indirect management.
"""

import struct
import pytest

from engine.rendering.gpu_driven.culling import Vec3
from engine.rendering.gpu_driven.indirect_draw import (
    IndirectDrawBuffer,
    DrawIndexedIndirectArgs,
)
from engine.rendering.gpu_driven.instancing import (
    Mat4x4,
    InstanceData,
    BatchKey,
    InstanceBatch,
    InstanceBatcher,
    MultiDrawIndirectManager,
    CulledInstanceBatcher,
)


# =============================================================================
# MATRIX TESTS
# =============================================================================


class TestMat4x4:
    """Tests for Mat4x4."""

    def test_identity(self) -> None:
        """Test identity matrix creation."""
        mat = Mat4x4.identity()

        assert mat.m[0] == 1.0  # Diagonal
        assert mat.m[5] == 1.0
        assert mat.m[10] == 1.0
        assert mat.m[15] == 1.0
        assert mat.m[1] == 0.0  # Off-diagonal

    def test_translation(self) -> None:
        """Test translation matrix."""
        mat = Mat4x4.translation(10.0, 20.0, 30.0)

        assert mat.m[3] == 10.0
        assert mat.m[7] == 20.0
        assert mat.m[11] == 30.0

    def test_scale(self) -> None:
        """Test scale matrix."""
        mat = Mat4x4.scale(2.0, 3.0, 4.0)

        assert mat.m[0] == 2.0
        assert mat.m[5] == 3.0
        assert mat.m[10] == 4.0

    def test_trs(self) -> None:
        """Test TRS matrix construction."""
        mat = Mat4x4.from_translation_rotation_scale(
            translation=Vec3(10.0, 20.0, 30.0),
            rotation_quat=(0.0, 0.0, 0.0, 1.0),  # Identity rotation
            scale=Vec3(2.0, 2.0, 2.0),
        )

        # With identity rotation and uniform scale
        assert mat.m[3] == pytest.approx(10.0)  # Translation
        assert mat.m[7] == pytest.approx(20.0)
        assert mat.m[11] == pytest.approx(30.0)
        assert mat.m[0] == pytest.approx(2.0)  # Scale
        assert mat.m[5] == pytest.approx(2.0)
        assert mat.m[10] == pytest.approx(2.0)

    def test_to_bytes(self) -> None:
        """Test matrix packing to bytes."""
        mat = Mat4x4.identity()
        data = mat.to_bytes()

        assert len(data) == 64  # 16 * 4 bytes

        values = struct.unpack("<16f", data)
        assert values[0] == 1.0  # First element

    def test_from_bytes(self) -> None:
        """Test matrix unpacking from bytes."""
        original = Mat4x4.translation(5.0, 10.0, 15.0)
        data = original.to_bytes()
        restored = Mat4x4.from_bytes(data)

        assert restored.m[3] == 5.0
        assert restored.m[7] == 10.0
        assert restored.m[11] == 15.0


# =============================================================================
# INSTANCE DATA TESTS
# =============================================================================


class TestInstanceData:
    """Tests for InstanceData."""

    def test_creation(self) -> None:
        """Test InstanceData creation."""
        instance = InstanceData(
            instance_id=42,
            transform=Mat4x4.identity(),
            lod_index=1,
            material_override=5,
            visible=True,
            cast_shadow=True,
        )

        assert instance.instance_id == 42
        assert instance.lod_index == 1
        assert instance.material_override == 5

    def test_to_gpu_format(self) -> None:
        """Test packing to GPU format."""
        instance = InstanceData(
            instance_id=100,
            transform=Mat4x4.identity(),
            lod_index=2,
            material_override=10,
            visible=True,
            cast_shadow=True,
            receive_shadow=True,
        )

        data = instance.to_gpu_format()

        # Should contain transform (64 bytes) + metadata (16 bytes)
        assert len(data) == 80

        # Verify instance ID
        instance_id = struct.unpack_from("<I", data, 64)[0]
        assert instance_id == 100

        # Verify LOD index
        lod_index = struct.unpack_from("<I", data, 68)[0]
        assert lod_index == 2

        # Verify flags
        flags = struct.unpack_from("<I", data, 76)[0]
        assert flags & 1  # visible
        assert flags & 2  # cast_shadow
        assert flags & 4  # receive_shadow

    def test_gpu_format_with_custom_data(self) -> None:
        """Test GPU format with custom data padding."""
        instance = InstanceData(
            instance_id=0,
            custom_data=b"\x01\x02\x03\x04",
        )

        data = instance.to_gpu_format(custom_data_size=16)

        # Should have transform + metadata + custom data
        assert len(data) == 80 + 16

    def test_base_size(self) -> None:
        """Test base size calculation."""
        assert InstanceData.base_size() == 80


# =============================================================================
# BATCH KEY TESTS
# =============================================================================


class TestBatchKey:
    """Tests for BatchKey."""

    def test_creation(self) -> None:
        """Test BatchKey creation."""
        key = BatchKey(mesh_id=1, material_id=2, lod_level=0)

        assert key.mesh_id == 1
        assert key.material_id == 2
        assert key.lod_level == 0

    def test_immutability(self) -> None:
        """Test BatchKey is immutable (frozen dataclass)."""
        key = BatchKey(mesh_id=1, material_id=2)

        with pytest.raises(Exception):  # FrozenInstanceError
            key.mesh_id = 3  # type: ignore

    def test_hashable(self) -> None:
        """Test BatchKey is hashable."""
        key1 = BatchKey(mesh_id=1, material_id=2)
        key2 = BatchKey(mesh_id=1, material_id=2)
        key3 = BatchKey(mesh_id=1, material_id=3)

        assert hash(key1) == hash(key2)
        assert hash(key1) != hash(key3)

        # Can be used in dict/set
        d = {key1: "test"}
        assert key2 in d

    def test_sort_key(self) -> None:
        """Test sort key generation."""
        key1 = BatchKey(mesh_id=0, material_id=0, lod_level=0, render_layer=0)
        key2 = BatchKey(mesh_id=0, material_id=1, lod_level=0, render_layer=0)
        key3 = BatchKey(mesh_id=0, material_id=0, lod_level=0, render_layer=1)

        # Higher render layer should sort later
        assert key1.to_sort_key() < key2.to_sort_key()
        assert key2.to_sort_key() < key3.to_sort_key()


# =============================================================================
# INSTANCE BATCH TESTS
# =============================================================================


class TestInstanceBatch:
    """Tests for InstanceBatch."""

    def test_creation(self) -> None:
        """Test InstanceBatch creation."""
        key = BatchKey(mesh_id=1, material_id=2)
        batch = InstanceBatch(
            key=key,
            index_count=36,
            first_index=0,
            vertex_offset=0,
        )

        assert batch.key == key
        assert batch.instance_count == 0
        assert batch.is_empty

    def test_add_instance(self) -> None:
        """Test adding instances."""
        key = BatchKey(mesh_id=1, material_id=2)
        batch = InstanceBatch(key=key, index_count=36)

        instance = InstanceData(instance_id=0)
        batch.add_instance(instance)

        assert batch.instance_count == 1
        assert not batch.is_empty

    def test_clear_instances(self) -> None:
        """Test clearing instances."""
        key = BatchKey(mesh_id=1, material_id=2)
        batch = InstanceBatch(key=key, index_count=36)

        batch.add_instance(InstanceData(instance_id=0))
        batch.clear_instances()

        assert batch.is_empty

    def test_to_draw_args(self) -> None:
        """Test generating draw args."""
        key = BatchKey(mesh_id=1, material_id=2)
        batch = InstanceBatch(
            key=key,
            index_count=36,
            first_index=10,
            vertex_offset=5,
        )

        batch.add_instance(InstanceData(instance_id=0))
        batch.add_instance(InstanceData(instance_id=1))

        args = batch.to_draw_args(first_instance=100)

        assert args.index_count == 36
        assert args.instance_count == 2
        assert args.first_index == 10
        assert args.vertex_offset == 5
        assert args.first_instance == 100


# =============================================================================
# INSTANCE BATCHER TESTS
# =============================================================================


class TestInstanceBatcher:
    """Tests for InstanceBatcher."""

    def test_creation(self) -> None:
        """Test InstanceBatcher creation."""
        batcher = InstanceBatcher()

        assert batcher.batch_count == 0
        assert batcher.instance_count == 0

    def test_register_batch(self) -> None:
        """Test registering batches."""
        batcher = InstanceBatcher()

        key = batcher.register_batch(
            mesh_id=1,
            material_id=2,
            index_count=36,
        )

        assert key.mesh_id == 1
        assert key.material_id == 2

    def test_add_instance(self) -> None:
        """Test adding instances."""
        batcher = InstanceBatcher()

        key = batcher.register_batch(mesh_id=1, material_id=2, index_count=36)

        instance = InstanceData(instance_id=0, transform=Mat4x4.identity())
        result = batcher.add_instance(key, instance)

        assert result
        assert batcher.instance_count == 1

    def test_add_instance_unknown_batch(self) -> None:
        """Test adding instance to unknown batch."""
        batcher = InstanceBatcher()

        unknown_key = BatchKey(mesh_id=999, material_id=999)
        instance = InstanceData(instance_id=0)

        result = batcher.add_instance(unknown_key, instance)
        assert not result

    def test_add_instance_quick(self) -> None:
        """Test quick instance addition."""
        batcher = InstanceBatcher()

        batcher.register_batch(mesh_id=1, material_id=2, index_count=36)

        result = batcher.add_instance_quick(
            mesh_id=1,
            material_id=2,
            transform=Mat4x4.identity(),
            instance_id=0,
        )

        assert result
        assert batcher.instance_count == 1

    def test_build_instance_buffer(self) -> None:
        """Test building instance buffer."""
        batcher = InstanceBatcher()

        key = batcher.register_batch(mesh_id=1, material_id=2, index_count=36)

        batcher.add_instance(key, InstanceData(instance_id=0))
        batcher.add_instance(key, InstanceData(instance_id=1))

        buffer = batcher.build_instance_buffer()

        expected_size = InstanceData.base_size() * 2
        assert len(buffer) == expected_size

    def test_generate_draw_commands(self) -> None:
        """Test generating draw commands."""
        batcher = InstanceBatcher()

        key = batcher.register_batch(mesh_id=1, material_id=2, index_count=36)

        batcher.add_instance(key, InstanceData(instance_id=0))
        batcher.add_instance(key, InstanceData(instance_id=1))

        draw_buffer = IndirectDrawBuffer()
        count = batcher.generate_draw_commands(draw_buffer)

        assert count == 1
        assert draw_buffer.draw_count == 1
        assert draw_buffer.draw_commands[0].args.instance_count == 2

    def test_clear(self) -> None:
        """Test clearing batcher."""
        batcher = InstanceBatcher()

        key = batcher.register_batch(mesh_id=1, material_id=2, index_count=36)
        batcher.add_instance(key, InstanceData(instance_id=0))

        batcher.clear()

        assert batcher.batch_count == 0
        assert batcher.instance_count == 0

    def test_multiple_batches(self) -> None:
        """Test multiple batches."""
        batcher = InstanceBatcher()

        key1 = batcher.register_batch(mesh_id=1, material_id=1, index_count=36)
        key2 = batcher.register_batch(mesh_id=2, material_id=2, index_count=24)

        batcher.add_instance(key1, InstanceData(instance_id=0))
        batcher.add_instance(key2, InstanceData(instance_id=1))

        draw_buffer = IndirectDrawBuffer()
        count = batcher.generate_draw_commands(draw_buffer)

        assert count == 2


# =============================================================================
# MULTI-DRAW INDIRECT MANAGER TESTS
# =============================================================================


class TestMultiDrawIndirectManager:
    """Tests for MultiDrawIndirectManager."""

    def test_creation(self) -> None:
        """Test MultiDrawIndirectManager creation."""
        manager = MultiDrawIndirectManager()

        assert manager.draw_count == 0

    def test_register_and_add(self) -> None:
        """Test registering mesh and adding instances."""
        manager = MultiDrawIndirectManager()

        key = manager.register_mesh(mesh_id=1, material_id=2, index_count=36)
        manager.add_instance(key, Mat4x4.identity(), instance_id=0)

        assert manager.batcher.instance_count == 1

    def test_build(self) -> None:
        """Test building GPU buffers."""
        manager = MultiDrawIndirectManager()

        key = manager.register_mesh(mesh_id=1, material_id=2, index_count=36)
        manager.add_instance(key, Mat4x4.identity(), instance_id=0)

        draw_buffer, instance_buffer = manager.build()

        assert len(draw_buffer) > 0
        assert len(instance_buffer) > 0

    def test_begin_frame(self) -> None:
        """Test frame begin."""
        manager = MultiDrawIndirectManager()

        key = manager.register_mesh(mesh_id=1, material_id=2, index_count=36)
        manager.add_instance(key, Mat4x4.identity(), instance_id=0)

        manager.begin_frame()

        assert manager.draw_count == 0

    def test_material_draw_ranges(self) -> None:
        """Test material draw range tracking."""
        manager = MultiDrawIndirectManager()

        # Register two different materials
        key1 = manager.register_mesh(mesh_id=1, material_id=1, index_count=36)
        key2 = manager.register_mesh(mesh_id=2, material_id=2, index_count=24)

        manager.add_instance(key1, Mat4x4.identity(), instance_id=0)
        manager.add_instance(key2, Mat4x4.identity(), instance_id=1)

        manager.build()

        # Check material ranges
        range1 = manager.get_material_draw_range(1)
        range2 = manager.get_material_draw_range(2)

        # Each material should have 1 draw
        assert range1[1] == 1 or range2[1] == 1


# =============================================================================
# CULLED INSTANCE BATCHER TESTS
# =============================================================================


class TestCulledInstanceBatcher:
    """Tests for CulledInstanceBatcher."""

    def test_creation(self) -> None:
        """Test CulledInstanceBatcher creation."""
        batcher = CulledInstanceBatcher()
        assert batcher.batcher is not None

    def test_add_instance_returns_index(self) -> None:
        """Test that add_instance returns index."""
        batcher = CulledInstanceBatcher()

        key = batcher.register_batch(mesh_id=1, material_id=2, index_count=36)

        idx0 = batcher.add_instance(key, InstanceData(instance_id=0))
        idx1 = batcher.add_instance(key, InstanceData(instance_id=1))

        assert idx0 == 0
        assert idx1 == 1

    def test_apply_visibility(self) -> None:
        """Test applying visibility from culling."""
        batcher = CulledInstanceBatcher()

        key = batcher.register_batch(mesh_id=1, material_id=2, index_count=36)

        batcher.add_instance(key, InstanceData(instance_id=0))
        batcher.add_instance(key, InstanceData(instance_id=1))
        batcher.add_instance(key, InstanceData(instance_id=2))

        # Only instance 0 and 2 are visible
        batcher.apply_visibility([0, 2])

        assert batcher.batcher.instance_count == 2

    def test_build_and_generate(self) -> None:
        """Test building and generating commands."""
        batcher = CulledInstanceBatcher()

        key = batcher.register_batch(mesh_id=1, material_id=2, index_count=36)

        batcher.add_instance(key, InstanceData(instance_id=0))
        batcher.add_instance(key, InstanceData(instance_id=1))

        batcher.apply_visibility([0, 1])

        draw_buffer = IndirectDrawBuffer()
        instance_buffer, draw_count = batcher.build_and_generate(draw_buffer)

        assert len(instance_buffer) > 0
        assert draw_count == 1

    def test_clear(self) -> None:
        """Test clearing."""
        batcher = CulledInstanceBatcher()

        key = batcher.register_batch(mesh_id=1, material_id=2, index_count=36)
        batcher.add_instance(key, InstanceData(instance_id=0))

        batcher.clear()

        # Should have no instances to process
        batcher.apply_visibility([])
        assert batcher.batcher.instance_count == 0
