"""
Black-box (cleanroom) tests for @gpu_buffer.
"""

from __future__ import annotations

import pytest
from trinity.decorators import (
    VALID_BUFFER_USAGE, WgpuBufferAllocation,
    allocate_wgpu_buffer, create_wgpu_buffer,
    gpu_buffer, gpu_struct, Vec2, Vec3, Vec4, Mat4,
)
from trinity.decorators.gpu import f32

_WGPU_VERTEX = 0x0020
_WGPU_INDEX = 0x0010
_WGPU_UNIFORM = 0x0040
_WGPU_STORAGE = 0x0080
_WGPU_INDIRECT = 0x0100
_WGPU_MAP_READ = 0x0001
_WGPU_MAP_WRITE = 0x0002
_WGPU_COPY_SRC = 0x0004
_WGPU_COPY_DST = 0x0008

@gpu_struct
class _Pos2: x: f32; y: f32
@gpu_struct
class _Vertex: position: Vec3; normal: Vec3; uv: Vec2
@gpu_struct
class _Pos3: x: f32; y: f32; z: f32
@gpu_struct
class _Color4: r: f32; g: f32; b: f32; a: f32

class TestInvalidUsage:
    def test_unknown_usage_flag_raises(self):
        with pytest.raises(ValueError):
            @gpu_buffer(usage={"vertex", "typo_flag"})
            class BadBuffer: data: f32
    def test_empty_usage_set_ok(self):
        @gpu_buffer(usage=frozenset())
        class EmptyBuffer: data: f32
        assert allocate_wgpu_buffer(EmptyBuffer).usage == 0
    def test_invalid_usage_single_string_raises(self):
        with pytest.raises(ValueError):
            @gpu_buffer(usage="not_a_flag")
            class BadFlag: data: f32
    def test_all_valid_flags_accepted(self):
        for flag in VALID_BUFFER_USAGE:
            @gpu_buffer(usage={flag})
            class T: data: f32
            assert allocate_wgpu_buffer(T).size > 0
    def test_mixed_valid_and_invalid_raises(self):
        with pytest.raises(ValueError):
            @gpu_buffer(usage={"vertex", "bogus"})
            class MixedBad: data: f32

class TestBufferSize:
    def test_single_float_field(self):
        @gpu_buffer(usage={"uniform"})
        class T: value: f32
        assert allocate_wgpu_buffer(T).size == 4
    def test_two_float_fields(self):
        @gpu_buffer(usage={"storage"})
        class T: x: f32; y: f32
        assert allocate_wgpu_buffer(T).size == 8
    def test_vec3_field(self):
        @gpu_buffer(usage={"storage"})
        class T: pos: Vec3
        assert allocate_wgpu_buffer(T).size == 12
    def test_vec4_field(self):
        @gpu_buffer(usage={"storage"})
        class T: color: Vec4
        assert allocate_wgpu_buffer(T).size == 16
    def test_mat4_field(self):
        @gpu_buffer(usage={"uniform"})
        class T: matrix: Mat4
        assert allocate_wgpu_buffer(T).size == 64
    def test_mixed_fields_vertex(self):
        @gpu_buffer(usage={"vertex"})
        class T: position: Vec3; normal: Vec3
        assert allocate_wgpu_buffer(T).size == 32
    def test_struct_with_gpu_struct_field(self):
        @gpu_buffer(usage={"storage"})
        class T: pos: _Pos2; color: _Color4
        assert allocate_wgpu_buffer(T).size >= 24
    def test_arrays_in_struct_fields(self):
        @gpu_buffer(usage={"storage"})
        class T: data: f32[3]; extra: Vec3
        assert allocate_wgpu_buffer(T).size == 32
    def test_empty_class_zero_size(self):
        @gpu_buffer(usage={"storage"})
        class T: pass
        assert allocate_wgpu_buffer(T).size == 0

class TestUsageBitmask:
    def test_vertex(self):
        @gpu_buffer(usage={"vertex"})
        class T: pos: Vec3
        assert allocate_wgpu_buffer(T).usage & _WGPU_VERTEX
    def test_index(self):
        @gpu_buffer(usage={"index"})
        class T: data: f32
        assert allocate_wgpu_buffer(T).usage & _WGPU_INDEX
    def test_uniform(self):
        @gpu_buffer(usage={"uniform"})
        class T: value: f32
        assert allocate_wgpu_buffer(T).usage & _WGPU_UNIFORM
    def test_storage(self):
        @gpu_buffer(usage={"storage"})
        class T: data: f32
        assert allocate_wgpu_buffer(T).usage & _WGPU_STORAGE
    def test_indirect(self):
        @gpu_buffer(usage={"indirect"})
        class T: data: f32
        assert allocate_wgpu_buffer(T).usage & _WGPU_INDIRECT
    def test_map_read(self):
        @gpu_buffer(usage={"map_read"})
        class T: data: f32
        assert allocate_wgpu_buffer(T).usage & _WGPU_MAP_READ
    def test_map_write(self):
        @gpu_buffer(usage={"map_write"})
        class T: data: f32
        assert allocate_wgpu_buffer(T).usage & _WGPU_MAP_WRITE
    def test_copy_src(self):
        @gpu_buffer(usage={"copy_src"})
        class T: data: f32
        assert allocate_wgpu_buffer(T).usage & _WGPU_COPY_SRC
    def test_copy_dst(self):
        @gpu_buffer(usage={"copy_dst"})
        class T: data: f32
        assert allocate_wgpu_buffer(T).usage & _WGPU_COPY_DST
    def test_multiple_bits_orred(self):
        @gpu_buffer(usage={"vertex", "storage", "copy_dst"})
        class T: data: f32
        e = _WGPU_VERTEX | _WGPU_STORAGE | _WGPU_COPY_DST
        assert allocate_wgpu_buffer(T).usage == e

class TestMappedFlag:
    def test_default_false(self):
        @gpu_buffer(usage={"storage"})
        class T: data: f32
        assert not allocate_wgpu_buffer(T).mapped
    def test_explicit_true(self):
        @gpu_buffer(usage={"storage"}, mapped=True)
        class T: data: f32
        assert allocate_wgpu_buffer(T).mapped
    def test_explicit_false(self):
        @gpu_buffer(usage={"storage"}, mapped=False)
        class T: data: f32
        assert not allocate_wgpu_buffer(T).mapped

class TestLabel:
    def test_default_is_class_name(self):
        @gpu_buffer(usage={"storage"})
        class MyData: value: f32
        assert allocate_wgpu_buffer(MyData).label == "MyData"
    def test_explicit_override(self):
        @gpu_buffer(usage={"storage"})
        class T: value: f32
        assert allocate_wgpu_buffer(T, label="custom").label == "custom"
    def test_empty_string(self):
        @gpu_buffer(usage={"storage"})
        class T: value: f32
        assert allocate_wgpu_buffer(T, label="").label == ""

class TestCreateWgpuBuffer:
    def test_with_mock_device(self):
        @gpu_buffer(usage={"storage"})
        class T: value: f32
        calls = []
        class D:
            def create_buffer(self, **desc):
                calls.append(desc); return "ok"
        assert create_wgpu_buffer(T, device=D()) == "ok"
        assert calls[0]["size"] == 4 and calls[0]["usage"] & _WGPU_STORAGE
    def test_none_device_returns_allocation(self):
        @gpu_buffer(usage={"vertex"})
        class T: pos: Vec3
        assert isinstance(create_wgpu_buffer(T, device=None), WgpuBufferAllocation)
    def test_no_create_buffer_returns_allocation(self):
        @gpu_buffer(usage={"uniform"})
        class T: value: f32
        assert isinstance(create_wgpu_buffer(T, device=object()), WgpuBufferAllocation)
    def test_passes_correct_descriptor(self):
        @gpu_buffer(usage={"storage", "copy_src"}, mapped=True)
        class T: x: f32; y: f32; z: f32
        captured = {}
        class D:
            def create_buffer(self, **desc):
                captured.update(desc); return desc
        create_wgpu_buffer(T, device=D(), label="test_buf")
        assert captured["size"] == 12 and captured["usage"] & _WGPU_STORAGE
        assert captured["mapped_at_creation"] and captured["label"] == "test_buf"

class TestErrorContracts:
    def test_allocate_on_undecorated_class_raises(self):
        class T: data: f32
        with pytest.raises((TypeError, AttributeError)):
            allocate_wgpu_buffer(T)
    def test_create_on_undecorated_class_raises(self):
        class T: data: f32
        with pytest.raises((TypeError, AttributeError)):
            create_wgpu_buffer(T)
    def test_allocate_on_non_class_raises(self):
        with pytest.raises((TypeError, AttributeError)):
            allocate_wgpu_buffer(42)
    def test_create_on_non_class_raises(self):
        with pytest.raises((TypeError, AttributeError)):
            create_wgpu_buffer("not_a_class")
    def test_allocate_on_gpu_struct_raises(self):
        with pytest.raises((TypeError, AttributeError)):
            allocate_wgpu_buffer(_Pos2)

class TestAllValidFlags:
    def test_all_nine_flags_combined(self):
        @gpu_buffer(usage=set(VALID_BUFFER_USAGE))
        class T: data: f32
        e = _WGPU_VERTEX|_WGPU_INDEX|_WGPU_UNIFORM|_WGPU_STORAGE|_WGPU_INDIRECT|_WGPU_MAP_READ|_WGPU_MAP_WRITE|_WGPU_COPY_SRC|_WGPU_COPY_DST
        assert allocate_wgpu_buffer(T).usage == e

class TestWgslTypes:
    def test_f32(self):
        @gpu_buffer(usage={"storage"})
        class T: value: f32
        assert allocate_wgpu_buffer(T).size == 4
    def test_vec2(self):
        @gpu_buffer(usage={"storage"})
        class T: pos: Vec2
        assert allocate_wgpu_buffer(T).size == 8
    def test_vec3(self):
        @gpu_buffer(usage={"storage"})
        class T: pos: Vec3
        assert allocate_wgpu_buffer(T).size == 12
    def test_vec4(self):
        @gpu_buffer(usage={"storage"})
        class T: color: Vec4
        assert allocate_wgpu_buffer(T).size == 16
    def test_mat4(self):
        @gpu_buffer(usage={"uniform"})
        class T: transform: Mat4
        assert allocate_wgpu_buffer(T).size == 64

class TestDecoratorStacking:
    def test_gpu_struct_and_gpu_buffer(self):
        @gpu_struct
        @gpu_buffer(usage={"storage"})
        class T:
            x: f32
            y: f32
        assert allocate_wgpu_buffer(T).size >= 8
    def test_stacked_with_mapped(self):
        @gpu_struct
        @gpu_buffer(usage={"storage"}, mapped=True)
        class T:
            data: f32
        assert allocate_wgpu_buffer(T).mapped

class TestEdgeCases:
    def test_float_4_bytes(self):
        @gpu_buffer(usage={"storage"})
        class T: value: float
        assert allocate_wgpu_buffer(T).size == 4
    def test_int_4_bytes(self):
        @gpu_buffer(usage={"storage"})
        class T: count: int
        assert allocate_wgpu_buffer(T).size == 4
    def test_bool_4_bytes(self):
        @gpu_buffer(usage={"storage"})
        class T: flag: bool
        assert allocate_wgpu_buffer(T).size == 4
    def test_mixed_primitives(self):
        @gpu_buffer(usage={"storage"})
        class T: a: float; b: int; c: bool
        assert allocate_wgpu_buffer(T).size == 12
    def test_zero_size_buffer(self):
        @gpu_buffer(usage={"storage"})
        class T: pass
        a = allocate_wgpu_buffer(T)
        assert a.size == 0 and a.usage & _WGPU_STORAGE
    def test_usage_as_list(self):
        @gpu_buffer(usage=["vertex", "index"])
        class T: pos: Vec3
        a = allocate_wgpu_buffer(T)
        assert a.usage & _WGPU_VERTEX and a.usage & _WGPU_INDEX

class TestAllocationContract:
    def test_frozen(self):
        a = WgpuBufferAllocation(size=64, usage=_WGPU_STORAGE)
        import dataclasses
        # Check if the dataclass is frozen by verifying it has __dataclass_fields__
        # and trying to mutate would raise FrozenInstanceError
        assert hasattr(a, "__dataclass_fields__")
        # Verify it's actually frozen by checking the decorator metadata
        assert dataclasses.fields(a.__class__)[0].name == "size"
        try:
            a.size = 128
            assert False, "Expected FrozenInstanceError"
        except dataclasses.FrozenInstanceError:
            pass  # Expected behavior
    def test_equality(self):
        a = WgpuBufferAllocation(size=4, usage=_WGPU_STORAGE)
        b = WgpuBufferAllocation(size=4, usage=_WGPU_STORAGE)
        assert a == b
    def test_mapped_default(self):
        assert not WgpuBufferAllocation(size=4, usage=_WGPU_UNIFORM).mapped

class TestPublicApiSurface:
    def test_frozenset(self):
        assert isinstance(VALID_BUFFER_USAGE, frozenset)
    def test_nine_flags(self):
        assert len(VALID_BUFFER_USAGE) == 9
    def test_callable(self):
        assert callable(gpu_buffer) and callable(create_wgpu_buffer)
