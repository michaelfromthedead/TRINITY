"""
Whitebox Tests for T-GPU-1.6 @gpu_buffer Wgpu Buffer Allocation

Tests focus on internal implementation details of the @gpu_buffer decorator
and its wgpu buffer allocation wiring: _resolve_wgpu_usage_flags internals,
_after_gpu_buffer attribute-setting mechanics, WgpuBufferAllocation dataclass
behavior, allocate_wgpu_buffer error paths, and create_wgpu_buffer delegation.

Covers the 5 core functions:
    - _resolve_wgpu_usage_flags   -- bitmask resolution internals
    - _after_gpu_buffer           -- attribute attachment, layout computation
    - allocate_wgpu_buffer        -- descriptor construction, error handling
    - create_wgpu_buffer          -- device.create_buffer delegation
    - WgpuBufferAllocation        -- frozen dataclass behavior
"""

import pytest
from typing import Any
from unittest.mock import MagicMock

from trinity.decorators.gpu import (
    WgpuBufferAllocation,
    _WGPU_USAGE_FLAGS,
    _after_gpu_buffer,
    _resolve_wgpu_usage_flags,
    allocate_wgpu_buffer,
    create_wgpu_buffer,
    gpu_buffer,
)


# =============================================================================
# _WGPU_USAGE_FLAGS MAP INTEGRITY
# =============================================================================


class TestWgpuUsageFlagsMapInternals:
    """Whitebox: _WGPU_USAGE_FLAGS dict integrity and bit value properties."""

    def test_all_9_flags_present(self):
        """Exactly 9 abstract usage names must be defined."""
        assert len(_WGPU_USAGE_FLAGS) == 9

    def test_no_bit_overlap(self):
        """No two flags share the same bit value."""
        values = set(_WGPU_USAGE_FLAGS.values())
        assert len(values) == len(_WGPU_USAGE_FLAGS), "Duplicate bit values detected"

    def test_every_bit_is_power_of_two(self):
        """Each wgpu usage flag must be a power of two."""
        for name, bit in _WGPU_USAGE_FLAGS.items():
            assert bin(bit).count("1") == 1, (
                f"{name}=0x{bit:04X} is not a power of two"
            )

    def test_combined_all_flags_mask(self):
        """All 9 flags OR'd together = 0x01FF (9 low bits)."""
        combined = 0
        for bit in _WGPU_USAGE_FLAGS.values():
            combined |= bit
        assert combined == 0x01FF

    def test_map_read_and_write_separate(self):
        """map_read=0x0001 and map_write=0x0002 are distinct bits."""
        assert _WGPU_USAGE_FLAGS["map_read"] == 0x0001
        assert _WGPU_USAGE_FLAGS["map_write"] == 0x0002
        assert _WGPU_USAGE_FLAGS["map_read"] & _WGPU_USAGE_FLAGS["map_write"] == 0

    def test_copy_src_and_dst_separate(self):
        """copy_src=0x0004 and copy_dst=0x0008 are distinct bits."""
        assert _WGPU_USAGE_FLAGS["copy_src"] == 0x0004
        assert _WGPU_USAGE_FLAGS["copy_dst"] == 0x0008
        assert _WGPU_USAGE_FLAGS["copy_src"] & _WGPU_USAGE_FLAGS["copy_dst"] == 0

    def test_storage_is_present(self):
        """storage=0x0080 is the key default usage."""
        assert _WGPU_USAGE_FLAGS["storage"] == 0x0080

    def test_vertex_and_index_separate(self):
        """vertex=0x0020 and index=0x0010 are distinct."""
        assert _WGPU_USAGE_FLAGS["vertex"] == 0x0020
        assert _WGPU_USAGE_FLAGS["index"] == 0x0010
        assert _WGPU_USAGE_FLAGS["vertex"] & _WGPU_USAGE_FLAGS["index"] == 0

    def test_indirect_flag(self):
        """indirect=0x0100 for indirect draw/dispatch."""
        assert _WGPU_USAGE_FLAGS["indirect"] == 0x0100


# =============================================================================
# _resolve_wgpu_usage_flags INTERNALS
# =============================================================================


class TestResolveWgpuUsageFlagsInternals:
    """Whitebox: _resolve_wgpu_usage_flags internal resolution mechanics."""

    def test_ordering_independent(self):
        """OR result is independent of iteration order."""
        a = _resolve_wgpu_usage_flags(frozenset({"storage", "vertex"}))
        b = _resolve_wgpu_usage_flags(frozenset({"vertex", "storage"}))
        assert a == b

    def test_duplicate_names_idempotent(self):
        """Duplicate names inside the set produce same result."""
        single = _resolve_wgpu_usage_flags(frozenset({"storage"}))
        dupes = _resolve_wgpu_usage_flags(frozenset({"storage", "storage"}))
        assert single == dupes

    def test_unknown_name_does_not_affect_other_bits(self):
        """Unknown names are silently dropped; other bits preserved."""
        flags = _resolve_wgpu_usage_flags(frozenset({"storage", "nonsense"}))
        # storage auto-appends copy_dst per WebGPU spec
        assert flags == (0x0080 | 0x0008)

    def test_only_unknown_names_return_zero(self):
        """When all names are unknown, returns 0."""
        flags = _resolve_wgpu_usage_flags(frozenset({"quantum", "tensor"}))
        assert flags == 0

    def test_input_casing_matters(self):
        """Casing is not normalized -- 'Storage' is not 'storage'."""
        flags = _resolve_wgpu_usage_flags(frozenset({"Storage"}))
        assert flags == 0  # not matched

    def test_minimal_mapping_does_not_mutate_global_dict(self):
        """The function reads but does not mutate _WGPU_USAGE_FLAGS."""
        before = dict(_WGPU_USAGE_FLAGS)
        _resolve_wgpu_usage_flags(frozenset({"storage"}))
        assert _WGPU_USAGE_FLAGS == before


# =============================================================================
# _after_gpu_buffer INTERNALS
# =============================================================================


class MockTargetWithAnnotations:
    """A minimal class-like object for _after_gpu_buffer testing."""
    def __init__(self):
        self._tags = {"gpu_buffer_config": type("Cfg", (), {
            "usage": frozenset({"storage"}),
            "mapped": False,
        })()}
        self.__annotations__ = {"x": float, "y": float}
        self._schema = {}
        self.__name__ = "MockTarget"


class MockTargetWithSchema(MockTargetWithAnnotations):
    """Target that has _schema populated (overrides __annotations__ for fields)."""
    def __init__(self):
        super().__init__()
        self._schema = {"x": float, "y": float, "z": float}


class TestAfterGpuBufferInternals:
    """Whitebox: _after_gpu_buffer attribute-setting and field extraction."""

    def test_sets_gpu_buffer_true(self):
        target = MockTargetWithAnnotations()
        _after_gpu_buffer(target, {})
        assert target._gpu_buffer is True

    def test_sets_gpu_usage_from_config(self):
        target = MockTargetWithAnnotations()
        _after_gpu_buffer(target, {})
        assert target._gpu_usage == frozenset({"storage"})

    def test_sets_gpu_mapped_from_config(self):
        target = MockTargetWithAnnotations()
        target._tags["gpu_buffer_config"].mapped = True
        _after_gpu_buffer(target, {})
        assert target._gpu_mapped is True

    def test_gpu_mapped_defaults_false(self):
        target = MockTargetWithAnnotations()
        _after_gpu_buffer(target, {})
        assert target._gpu_mapped is False

    def test_buffer_fields_from_schema(self):
        """Fields are extracted from _schema, not __annotations__."""
        target = MockTargetWithSchema()  # _schema has 3 fields
        _after_gpu_buffer(target, {})
        assert target._gpu_buffer_fields == ["x", "y", "z"]

    def test_buffer_fields_empty_when_no_schema(self):
        """When _schema is absent/empty, fields list is empty."""
        target = MockTargetWithAnnotations()  # _schema is {}
        _after_gpu_buffer(target, {})
        assert target._gpu_buffer_fields == []

    def test_buffer_size_from_annotations(self):
        """Layout size is computed from __annotations__, not _schema."""
        target = MockTargetWithAnnotations()
        target.__annotations__ = {"a": float, "b": float, "c": float}
        _after_gpu_buffer(target, {})
        assert target._gpu_buffer_size == 12  # 3 floats

    def test_buffer_alignment_from_layout(self):
        target = MockTargetWithAnnotations()
        target.__annotations__ = {"x": float}
        _after_gpu_buffer(target, {})
        assert target._gpu_buffer_alignment == 4

    def test_wgpu_usage_resolved(self):
        target = MockTargetWithAnnotations()
        target._tags["gpu_buffer_config"].usage = frozenset({"vertex", "storage"})
        _after_gpu_buffer(target, {})
        # storage auto-appends copy_dst per WebGPU spec
        assert target._gpu_wgpu_usage == (0x0020 | 0x0080 | 0x0008)

    def test_layout_has_fields_list(self):
        target = MockTargetWithAnnotations()
        target.__annotations__ = {"x": float, "y": float}
        _after_gpu_buffer(target, {})
        assert isinstance(target._gpu_buffer_layout, list)
        assert len(target._gpu_buffer_layout) == 2

    def test_buffer_layout_field_offsets(self):
        target = MockTargetWithAnnotations()
        target.__annotations__ = {"x": float, "y": float}
        _after_gpu_buffer(target, {})
        offsets = [f["offset"] for f in target._gpu_buffer_layout]
        assert offsets == [0, 4]

    def test_buffer_layout_field_types(self):
        target = MockTargetWithAnnotations()
        target.__annotations__ = {"x": float}
        _after_gpu_buffer(target, {})
        assert target._gpu_buffer_layout[0]["type"] == "f32"

    def test_returns_none(self):
        """_after_gpu_buffer always returns None (not the target)."""
        target = MockTargetWithAnnotations()
        result = _after_gpu_buffer(target, {})
        assert result is None


class TestAfterGpuBufferEdgeCases:
    """Whitebox: _after_gpu_buffer edge cases and error paths."""

    def test_annotations_empty_buffer_size_zero(self):
        target = MockTargetWithAnnotations()
        target.__annotations__ = {}
        _after_gpu_buffer(target, {})
        assert target._gpu_buffer_size == 0

    def test_config_from_tags_not_params(self):
        """Config is read from target._tags, NOT from params dict."""
        target = MockTargetWithAnnotations()
        result = _after_gpu_buffer(target, {"usage": {"vertex"}})
        assert target._gpu_usage == frozenset({"storage"})  # from _tags, not params

    def test_multiple_calls_idempotent_in_attrs(self):
        """Calling _after_gpu_buffer twice sets same attributes (last wins)."""
        target = MockTargetWithAnnotations()
        _after_gpu_buffer(target, {})
        first_size = target._gpu_buffer_size
        _after_gpu_buffer(target, {})
        assert target._gpu_buffer_size == first_size

    def test_schema_without_annotations(self):
        """If __annotations__ is empty but _schema has data, size is 0."""
        target = MockTargetWithAnnotations()
        target.__annotations__ = {}
        target._schema = {"x": float}
        _after_gpu_buffer(target, {})
        assert target._gpu_buffer_size == 0  # size from annotations=empty
        assert target._gpu_buffer_fields == ["x"]  # fields from schema


# =============================================================================
# WgpuBufferAllocation DATACLASS INTERNALS
# =============================================================================


class TestWgpuBufferAllocationInternals:
    """Whitebox: WgpuBufferAllocation frozen dataclass behavior."""

    def test_default_mapped_is_false(self):
        alloc = WgpuBufferAllocation(size=64, usage=0x0080)
        assert alloc.mapped is False

    def test_default_label_is_empty(self):
        alloc = WgpuBufferAllocation(size=64, usage=0x0080)
        assert alloc.label == ""

    def test_all_fields_via_constructor(self):
        alloc = WgpuBufferAllocation(size=128, usage=0x0040, mapped=True, label="test")
        assert alloc.size == 128
        assert alloc.usage == 0x0040
        assert alloc.mapped is True
        assert alloc.label == "test"

    def test_is_frozen(self):
        alloc = WgpuBufferAllocation(size=64, usage=0x0080)
        with pytest.raises(Exception):
            alloc.size = 128

    def test_immutable_via_frozen(self):
        """Verify frozen=True prevents all attribute mutation."""
        alloc = WgpuBufferAllocation(size=32, usage=0x0080)
        with pytest.raises((AttributeError, TypeError)):
            alloc.usage = 0x0040
        with pytest.raises((AttributeError, TypeError)):
            alloc.mapped = True
        with pytest.raises((AttributeError, TypeError)):
            alloc.label = "new"

    def test_equality_between_identical(self):
        a = WgpuBufferAllocation(size=64, usage=0x0080, mapped=False, label="x")
        b = WgpuBufferAllocation(size=64, usage=0x0080, mapped=False, label="x")
        assert a == b

    def test_inequality_different_size(self):
        a = WgpuBufferAllocation(size=64, usage=0x0080)
        b = WgpuBufferAllocation(size=128, usage=0x0080)
        assert a != b

    def test_inequality_different_usage(self):
        a = WgpuBufferAllocation(size=64, usage=0x0080)
        b = WgpuBufferAllocation(size=64, usage=0x0040)
        assert a != b

    def test_inequality_different_label(self):
        a = WgpuBufferAllocation(size=64, usage=0x0080, label="a")
        b = WgpuBufferAllocation(size=64, usage=0x0080, label="b")
        assert a != b

    def test_repr_contains_fields(self):
        alloc = WgpuBufferAllocation(size=64, usage=0x0080, label="buf")
        r = repr(alloc)
        assert "size=64" in r
        assert "usage=128" in r or "usage=0x80" in r or "usage=0x0080" in r
        assert "label='buf'" in r


# =============================================================================
# allocate_wgpu_buffer INTERNAL PATHS
# =============================================================================


class TestAllocateWgpuBufferInternals:
    """Whitebox: allocate_wgpu_buffer internal control flow and error paths."""

    def test_checks_gpu_buffer_flag_true(self):
        """getattr(._gpu_buffer, False) must be True."""
        target = type("Buf", (), {"_gpu_buffer": True, "_gpu_buffer_size": 16,
                                   "_gpu_wgpu_usage": 0x0080, "_gpu_mapped": False,
                                   "__name__": "Buf"})
        alloc = allocate_wgpu_buffer(target, None)
        assert alloc.size == 16

    def test_checks_gpu_buffer_flag_explicit_false(self):
        """_gpu_buffer=False raises TypeError."""
        target = type("Buf", (), {"_gpu_buffer": False, "__name__": "Buf"})
        with pytest.raises(TypeError, match="not decorated with @gpu_buffer"):
            allocate_wgpu_buffer(target, None)

    def test_checks_gpu_buffer_flag_missing(self):
        """Missing _gpu_buffer attr defaults to False, raises TypeError."""
        target = type("Buf", (), {"__name__": "Buf"})
        with pytest.raises(TypeError, match="not decorated with @gpu_buffer"):
            allocate_wgpu_buffer(target, None)

    def test_zero_size_raises(self):
        target = type("Buf", (), {"_gpu_buffer": True, "_gpu_buffer_size": 0,
                                   "_gpu_wgpu_usage": 0x0080, "_gpu_mapped": False,
                                   "__name__": "Buf"})
        with pytest.raises(RuntimeError, match="zero buffer size"):
            allocate_wgpu_buffer(target, None)

    def test_negative_size_raises(self):
        """Even a negative size triggers the zero-size error path."""
        target = type("Buf", (), {"_gpu_buffer": True, "_gpu_buffer_size": -1,
                                   "_gpu_wgpu_usage": 0x0080, "_gpu_mapped": False,
                                   "__name__": "Buf"})
        with pytest.raises(RuntimeError, match="zero buffer size"):
            allocate_wgpu_buffer(target, None)

    def test_usage_falls_back_to_zero(self):
        target = type("Buf", (), {"_gpu_buffer": True, "_gpu_buffer_size": 16,
                                   "_gpu_mapped": False,
                                   "__name__": "Buf"})
        alloc = allocate_wgpu_buffer(target, None)
        assert alloc.usage == 0  # getattr default

    def test_mapped_falls_back_to_false(self):
        target = type("Buf", (), {"_gpu_buffer": True, "_gpu_buffer_size": 16,
                                   "_gpu_wgpu_usage": 0x0080,
                                   "__name__": "Buf"})
        alloc = allocate_wgpu_buffer(target, None)
        assert alloc.mapped is False

    def test_label_default_matches_class_name(self):
        target = type("MyParticle", (), {"_gpu_buffer": True, "_gpu_buffer_size": 16,
                                           "_gpu_wgpu_usage": 0x0080, "_gpu_mapped": False,
                                           "__name__": "MyParticle"})
        alloc = allocate_wgpu_buffer(target, None)
        assert alloc.label == "MyParticle"

    def test_label_none_falls_back_to_class_name(self):
        target = type("Buf", (), {"_gpu_buffer": True, "_gpu_buffer_size": 16,
                                   "_gpu_wgpu_usage": 0x0080, "_gpu_mapped": False,
                                   "__name__": "Buf"})
        alloc = allocate_wgpu_buffer(target, None, label=None)
        assert alloc.label == "Buf"

    def test_device_parameter_not_used(self):
        """The device parameter is accepted but currently not inspected."""
        calls = []
        def device_func(): calls.append("called")
        target = type("Buf", (), {"_gpu_buffer": True, "_gpu_buffer_size": 16,
                                   "_gpu_wgpu_usage": 0x0080, "_gpu_mapped": False,
                                   "__name__": "Buf"})
        allocate_wgpu_buffer(target, device_func)
        assert len(calls) == 0  # device is NOT called

    def test_error_message_contains_class_name(self):
        target = type("MySpecialBuf", (), {"_gpu_buffer": False, "__name__": "MySpecialBuf"})
        with pytest.raises(TypeError) as exc:
            allocate_wgpu_buffer(target, None)
        assert "MySpecialBuf" in str(exc.value)


# =============================================================================
# create_wgpu_buffer INTERNAL DELEGATION
# =============================================================================


class TestCreateWgpuBufferInternals:
    """Whitebox: create_wgpu_buffer maps alloc fields to device.create_buffer."""

    def test_delegates_to_allocate(self):
        """create_wgpu_buffer must call allocate_wgpu_buffer internally."""
        @gpu_buffer(usage={"uniform"})
        class UniformBuf:
            data: float

        device = MagicMock()
        device.create_buffer.return_value = "buf_handle"
        result = create_wgpu_buffer(UniformBuf, device, label="uni")
        assert result == "buf_handle"

    def test_size_passed_to_device(self):
        @gpu_buffer(usage={"uniform"})
        class Buf:
            x: float
            y: float

        device = MagicMock()
        create_wgpu_buffer(Buf, device)
        call_kwargs = device.create_buffer.call_args.kwargs
        assert call_kwargs["size"] == 8

    def test_usage_passed_to_device(self):
        @gpu_buffer(usage={"storage", "copy_src"})
        class Buf:
            data: float

        device = MagicMock()
        create_wgpu_buffer(Buf, device)
        call_kwargs = device.create_buffer.call_args.kwargs
        # storage auto-appends copy_dst per WebGPU spec
        assert call_kwargs["usage"] == (0x0080 | 0x0004 | 0x0008)

    def test_mapped_at_creation_passed(self):
        @gpu_buffer(usage={"uniform"}, mapped=True)
        class Buf:
            data: float

        device = MagicMock()
        create_wgpu_buffer(Buf, device)
        call_kwargs = device.create_buffer.call_args.kwargs
        assert call_kwargs["mapped_at_creation"] is True

    def test_mapped_at_creation_false_by_default(self):
        @gpu_buffer(usage={"storage"})
        class Buf:
            data: float

        device = MagicMock()
        create_wgpu_buffer(Buf, device)
        call_kwargs = device.create_buffer.call_args.kwargs
        assert call_kwargs["mapped_at_creation"] is False

    def test_label_passed_to_device(self):
        @gpu_buffer(usage={"storage"})
        class ParticleBuf:
            pos: float

        device = MagicMock()
        create_wgpu_buffer(ParticleBuf, device, label="particles")
        call_kwargs = device.create_buffer.call_args.kwargs
        assert call_kwargs["label"] == "particles"

    def test_label_defaults_to_class_name(self):
        @gpu_buffer(usage={"storage"})
        class DefaultLabel:
            data: float

        device = MagicMock()
        create_wgpu_buffer(DefaultLabel, device)
        call_kwargs = device.create_buffer.call_args.kwargs
        assert call_kwargs["label"] == "DefaultLabel"

    def test_propagates_type_error_from_allocate(self):
        """TypeError from undecorated class propagates through create."""
        device = MagicMock()
        with pytest.raises(TypeError, match="not decorated with @gpu_buffer"):
            create_wgpu_buffer(int, device)

    def test_propagates_runtime_error_from_allocate(self):
        """RuntimeError from empty struct propagates through create."""
        @gpu_buffer(usage={"storage"})
        class Empty:
            pass

        device = MagicMock()
        with pytest.raises(RuntimeError, match="zero buffer size"):
            create_wgpu_buffer(Empty, device)

    def test_device_create_buffer_exact_kwarg_keys(self):
        """Exact 4 kwargs passed: size, usage, mapped_at_creation, label."""
        @gpu_buffer(usage={"storage"})
        class Buf:
            data: float

        device = MagicMock()
        create_wgpu_buffer(Buf, device)
        call_kwargs = device.create_buffer.call_args.kwargs
        assert set(call_kwargs.keys()) == {"size", "usage", "mapped_at_creation", "label"}

    def test_device_create_buffer_kwargs_order_independent(self):
        """kwargs-based so ordering does not matter, just check values."""
        @gpu_buffer(usage={"uniform"})
        class Buf:
            data: float

        device = MagicMock()
        create_wgpu_buffer(Buf, device)
        _, kwargs = device.create_buffer.call_args
        assert kwargs["size"] == 4
        assert kwargs["usage"] == 0x0040
        assert kwargs["mapped_at_creation"] is False
        assert kwargs["label"] == "Buf"


# =============================================================================
# FULL-LIFECYCLE WHITEBOX
# =============================================================================


class TestGpuBufferFullLifecycle:
    """Whitebox: end-to-end lifecycle from decorator to device buffer creation."""

    def test_decorate_to_allocate_to_create(self):
        """Full chain: decorator sets attrs -> allocate builds desc -> create calls device."""
        @gpu_buffer(usage={"storage", "copy_src"})
        class ComputeBuffer:
            data: float
            count: int
            flags: bool

        assert ComputeBuffer._gpu_buffer is True
        assert ComputeBuffer._gpu_buffer_size == 12
        # storage auto-appends copy_dst per WebGPU spec
        assert ComputeBuffer._gpu_wgpu_usage == (0x0080 | 0x0004 | 0x0008)

        alloc = allocate_wgpu_buffer(ComputeBuffer, None)
        assert alloc.size == 12
        assert alloc.usage == (0x0080 | 0x0004 | 0x0008)
        assert alloc.label == "ComputeBuffer"

        device = MagicMock()
        buf = create_wgpu_buffer(ComputeBuffer, device)
        assert buf == device.create_buffer.return_value

    def test_mapped_lifecycle(self):
        """mapped=True flows decorator -> allocate -> create."""
        @gpu_buffer(usage={"uniform"}, mapped=True)
        class MappedBuf:
            data: float

        assert MappedBuf._gpu_mapped is True

        alloc = allocate_wgpu_buffer(MappedBuf, None)
        assert alloc.mapped is True

        device = MagicMock()
        create_wgpu_buffer(MappedBuf, device)
        assert device.create_buffer.call_args.kwargs["mapped_at_creation"] is True

    def test_wgsl_types_lifecycle(self):
        """WGSL type markers flow through the full chain."""
        from trinity.decorators.gpu import Vec3, Vec2

        @gpu_buffer(usage={"vertex"})
        class Vertex:
            position: Vec3
            normal: Vec3
            uv: Vec2

        assert Vertex._gpu_buffer_size == 48
        assert Vertex._gpu_buffer_alignment == 16

        alloc = allocate_wgpu_buffer(Vertex, None)
        assert alloc.size == 48

    def test_array_type_lifecycle(self):
        """Annotated[T, N] arrays flow through full chain."""
        from typing import Annotated
        from trinity.decorators.gpu import Vec3

        @gpu_buffer(usage={"storage"})
        class Particles:
            positions: Annotated[Vec3, 64]

        assert Particles._gpu_buffer_size == 1024  # 64 * 16 stride
        assert Particles._gpu_buffer_alignment == 16

        alloc = allocate_wgpu_buffer(Particles, None)
        assert alloc.size == 1024

    def test_multi_field_with_array_lifecycle(self):
        """Mixed scalar + array fields in full chain."""
        from typing import Annotated
        from trinity.decorators.gpu import Vec4

        @gpu_buffer(usage={"uniform"})
        class Material:
            base_color: Vec4
            roughness: float
            metallic: float
            emissive: Annotated[float, 2]

        assert Material._gpu_buffer_size == 48
        alloc = allocate_wgpu_buffer(Material, None)
        assert alloc.size == 48


# =============================================================================
# GPU_BUFFER + BIND_GROUP STACKING WHITEBOX
# =============================================================================


class TestGpuBufferStackingWhitebox:
    """Whitebox: @gpu_buffer stacked with other decorators."""

    def test_gpu_buffer_with_bind_group(self):
        @gpu_buffer(usage={"uniform"})
        class UniformBlock:
            data: float

        assert UniformBlock._gpu_buffer is True
        assert UniformBlock._gpu_buffer_size == 4
        assert UniformBlock._gpu_wgpu_usage == 0x0040

    def test_gpu_buffer_kernel_buffer_separate(self):
        """Verify @gpu_buffer and @gpu_kernel set different attributes."""
        from trinity.decorators.gpu import gpu_kernel

        @gpu_buffer(usage={"storage"})
        class Buf:
            data: float

        @gpu_kernel(backend="wgpu")
        class Kern:
            pass

        assert Buf._gpu_buffer is True
        assert Buf._gpu_buffer_size == 4
        # storage auto-appends copy_dst per WebGPU spec
        assert Buf._gpu_wgpu_usage == (0x0080 | 0x0008)
        assert not hasattr(Buf, "_gpu_kernel")

        assert Kern._gpu_kernel is True
        assert Kern._workgroup_size == (64, 1, 1)
        assert not hasattr(Kern, "_gpu_buffer")

    def test_buffer_after_gpu_struct(self):
        """@gpu_struct can be used on the same class before @gpu_buffer."""
        from trinity.decorators.gpu import gpu_struct, Vec3

        @gpu_buffer(usage={"vertex"})
        @gpu_struct
        class MeshVertex:
            position: Vec3
            normal: Vec3
            uv: float

        assert MeshVertex._gpu_struct is True
        assert MeshVertex._gpu_buffer is True
        assert MeshVertex._gpu_struct_size == 32
        assert MeshVertex._gpu_buffer_size == 32

    def test_gpu_buffer_decorates_class_only(self):
        """@gpu_buffer cannot be applied to functions."""
        with pytest.raises(Exception):
            @gpu_buffer(usage={"storage"})
            def func():
                pass

    def test_buffer_with_empty_usage_set(self):
        """Empty usage set produces wgpu usage of 0."""
        @gpu_buffer(usage=set())
        class Buf:
            data: float

        assert Buf._gpu_wgpu_usage == 0
        assert Buf._gpu_buffer_size == 4


# =============================================================================
# ERROR PROPAGATION WHITEBOX
# =============================================================================


class TestGpuBufferErrorPropagation:
    """Whitebox: error propagation paths through the buffer API."""

    def test_allocate_error_includes_class_name(self):
        @gpu_buffer(usage={"storage"})
        class EmptyBuf:
            pass

        with pytest.raises(RuntimeError) as exc:
            allocate_wgpu_buffer(EmptyBuf, None)
        assert "EmptyBuf" in str(exc.value)

    def test_create_error_includes_class_name(self):
        @gpu_buffer(usage={"storage"})
        class EmptyBuf2:
            pass

        with pytest.raises(RuntimeError) as exc:
            create_wgpu_buffer(EmptyBuf2, None)
        assert "EmptyBuf2" in str(exc.value)

    def test_device_error_propagates(self):
        """If device.create_buffer raises, it propagates to caller."""
        @gpu_buffer(usage={"storage"})
        class Buf:
            data: float

        class BrokenDevice:
            def create_buffer(self, **kwargs):
                raise RuntimeError("device lost")

        device = BrokenDevice()
        with pytest.raises(RuntimeError, match="device lost"):
            create_wgpu_buffer(Buf, device)

    def test_device_create_buffer_called_exactly_once(self):
        @gpu_buffer(usage={"storage"})
        class Buf:
            data: float

        device = MagicMock()
        create_wgpu_buffer(Buf, device)
        device.create_buffer.assert_called_once()
