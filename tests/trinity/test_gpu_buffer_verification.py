"""
T-GPU-1.6: WHITEBOX+BLACKBOX verification tests for @gpu_buffer fixes.

WHITEBOX:
  1. Each of the 6 corrected flag bits matches the WebGPU/wgpu-py spec.
  2. _resolve_wgpu_usage_flags produces correct combined bitmasks.
  3. allocate_wgpu_buffer resolves correct usage from @gpu_buffer classes.
  4. VALID_BUFFER_USAGE covers all mapped flag names.

BLACKBOX:
  1. All 4 re-exports (VALID_BUFFER_USAGE, WgpuBufferAllocation,
     allocate_wgpu_buffer, create_wgpu_buffer) are importable from
     trinity.decorators.
  2. All 4 re-exports appear in trinity.decorators.__all__.
"""

import pytest

_WGPU_SPEC_FLAGS: dict[str, int] = {
    "vertex":   0x0020,
    "index":    0x0010,
    "uniform":  0x0040,
    "storage":  0x0080,
    "indirect": 0x0100,
    "map_read": 0x0001,
    "map_write": 0x0002,
    "copy_src": 0x0004,
    "copy_dst": 0x0008,
}

class TestWgpuUsageFlagBits:
    def test_internal_flags_match_spec(self):
        from trinity.decorators.gpu import _WGPU_USAGE_FLAGS
        assert _WGPU_USAGE_FLAGS == _WGPU_SPEC_FLAGS

    def test_index_flag_correct(self):
        from trinity.decorators.gpu import _WGPU_USAGE_FLAGS
        assert _WGPU_USAGE_FLAGS["index"] == 0x0010

    def test_uniform_flag_correct(self):
        from trinity.decorators.gpu import _WGPU_USAGE_FLAGS
        assert _WGPU_USAGE_FLAGS["uniform"] == 0x0040

    def test_map_read_flag_correct(self):
        from trinity.decorators.gpu import _WGPU_USAGE_FLAGS
        assert _WGPU_USAGE_FLAGS["map_read"] == 0x0001

    def test_map_write_flag_correct(self):
        from trinity.decorators.gpu import _WGPU_USAGE_FLAGS
        assert _WGPU_USAGE_FLAGS["map_write"] == 0x0002

    def test_copy_src_flag_correct(self):
        from trinity.decorators.gpu import _WGPU_USAGE_FLAGS
        assert _WGPU_USAGE_FLAGS["copy_src"] == 0x0004

    def test_copy_dst_flag_correct(self):
        from trinity.decorators.gpu import _WGPU_USAGE_FLAGS
        assert _WGPU_USAGE_FLAGS["copy_dst"] == 0x0008

    def test_vertex_flag_unchanged(self):
        from trinity.decorators.gpu import _WGPU_USAGE_FLAGS
        assert _WGPU_USAGE_FLAGS["vertex"] == 0x0020

    def test_storage_flag_unchanged(self):
        from trinity.decorators.gpu import _WGPU_USAGE_FLAGS
        assert _WGPU_USAGE_FLAGS["storage"] == 0x0080

    def test_indirect_flag_unchanged(self):
        from trinity.decorators.gpu import _WGPU_USAGE_FLAGS
        assert _WGPU_USAGE_FLAGS["indirect"] == 0x0100

    def test_no_duplicate_bit_values(self):
        from trinity.decorators.gpu import _WGPU_USAGE_FLAGS
        bits = list(_WGPU_USAGE_FLAGS.values())
        assert len(bits) == len(set(bits))

    def test_all_flags_are_powers_of_two(self):
        from trinity.decorators.gpu import _WGPU_USAGE_FLAGS
        for name, value in _WGPU_USAGE_FLAGS.items():
            assert value & (value - 1) == 0 and value != 0

class TestResolveWgpuUsageFlags:
    def test_single_flag(self):
        from trinity.decorators.gpu import _resolve_wgpu_usage_flags
        assert _resolve_wgpu_usage_flags(frozenset({"vertex"})) == 0x0020

    def test_multi_flag(self):
        from trinity.decorators.gpu import _resolve_wgpu_usage_flags
        assert _resolve_wgpu_usage_flags(frozenset({"vertex", "index"})) == (0x0020 | 0x0010)

    def test_storage_copy_src(self):
        from trinity.decorators.gpu import _resolve_wgpu_usage_flags
        assert _resolve_wgpu_usage_flags(frozenset({"storage", "copy_src"})) == (0x0080 | 0x0004)

    def test_uniform_copy_dst(self):
        from trinity.decorators.gpu import _resolve_wgpu_usage_flags
        assert _resolve_wgpu_usage_flags(frozenset({"uniform", "copy_dst"})) == (0x0040 | 0x0008)

    def test_empty_set(self):
        from trinity.decorators.gpu import _resolve_wgpu_usage_flags
        assert _resolve_wgpu_usage_flags(frozenset()) == 0

    def test_map_read_write(self):
        from trinity.decorators.gpu import _resolve_wgpu_usage_flags
        assert _resolve_wgpu_usage_flags(frozenset({"map_read", "map_write"})) == (0x0001 | 0x0002)

    def test_all_flags(self):
        from trinity.decorators.gpu import _resolve_wgpu_usage_flags
        result = _resolve_wgpu_usage_flags(frozenset(_WGPU_SPEC_FLAGS.keys()))
        assert result == sum(_WGPU_SPEC_FLAGS.values())

class TestAllocateWgpuBufferFlags:
    def _alloc(self, cls):
        from trinity.decorators.gpu import allocate_wgpu_buffer
        return allocate_wgpu_buffer(cls, device=None)

    def test_single_vertex_usage(self):
        from trinity.decorators.gpu import gpu_buffer
        @gpu_buffer(usage={"vertex"})
        class VertexBuf: pass
        assert self._alloc(VertexBuf).usage == 0x0020

    def test_storage_copy_src_usage(self):
        from trinity.decorators.gpu import gpu_buffer
        @gpu_buffer(usage={"storage", "copy_src"})
        class StorageBuf: pass
        assert self._alloc(StorageBuf).usage == (0x0080 | 0x0004)

    def test_index_usage(self):
        from trinity.decorators.gpu import gpu_buffer
        @gpu_buffer(usage={"index"})
        class IndexBuf: pass
        assert self._alloc(IndexBuf).usage == 0x0010

    def test_uniform_usage(self):
        from trinity.decorators.gpu import gpu_buffer
        @gpu_buffer(usage={"uniform"})
        class UniformBuf: pass
        assert self._alloc(UniformBuf).usage == 0x0040

    def test_default_storage_usage(self):
        from trinity.decorators.gpu import gpu_buffer
        @gpu_buffer()
        class DefaultBuf: pass
        assert self._alloc(DefaultBuf).usage == 0x0080

    def test_multiple_usage_combines(self):
        from trinity.decorators.gpu import gpu_buffer
        @gpu_buffer(usage={"vertex", "index", "storage", "copy_src", "copy_dst"})
        class MultiBuf: pass
        expected = 0x0020 | 0x0010 | 0x0080 | 0x0004 | 0x0008
        assert self._alloc(MultiBuf).usage == expected

    def test_map_read_write_usage(self):
        from trinity.decorators.gpu import gpu_buffer
        @gpu_buffer(usage={"map_read", "map_write"})
        class MapBuf: pass
        assert self._alloc(MapBuf).usage == (0x0001 | 0x0002)

class TestValidBufferUsage:
    def test_contains_all_flag_names(self):
        from trinity.decorators.gpu import VALID_BUFFER_USAGE, _WGPU_USAGE_FLAGS
        for name in _WGPU_USAGE_FLAGS:
            assert name in VALID_BUFFER_USAGE

    def test_no_extra_names(self):
        from trinity.decorators.gpu import VALID_BUFFER_USAGE, _WGPU_USAGE_FLAGS
        for name in VALID_BUFFER_USAGE:
            assert name in _WGPU_USAGE_FLAGS

    def test_is_frozenset(self):
        from trinity.decorators.gpu import VALID_BUFFER_USAGE
        assert isinstance(VALID_BUFFER_USAGE, frozenset)

class TestReExportsFromInit:
    def test_import_valid_buffer_usage(self):
        from trinity.decorators import VALID_BUFFER_USAGE
        assert isinstance(VALID_BUFFER_USAGE, frozenset)
        assert "vertex" in VALID_BUFFER_USAGE

    def test_import_wgpu_buffer_allocation(self):
        from trinity.decorators import WgpuBufferAllocation
        alloc = WgpuBufferAllocation(size=256, usage=0x0080)
        assert alloc.size == 256
        assert alloc.usage == 0x0080
        assert alloc.mapped is False
        assert alloc.label == ""

    def test_import_allocate_wgpu_buffer(self):
        from trinity.decorators import allocate_wgpu_buffer
        assert callable(allocate_wgpu_buffer)

    def test_import_create_wgpu_buffer(self):
        from trinity.decorators import create_wgpu_buffer
        assert callable(create_wgpu_buffer)

    def test_allocate_wgpu_buffer_rejects_non_gpu_buffer(self):
        from trinity.decorators import allocate_wgpu_buffer
        class NotAGpuBuffer: pass
        with pytest.raises(TypeError, match="not decorated with @gpu_buffer"):
            allocate_wgpu_buffer(NotAGpuBuffer, None)

    def test_create_wgpu_buffer_rejects_non_gpu_buffer(self):
        from trinity.decorators import create_wgpu_buffer
        class NotAGpuBuffer: pass
        with pytest.raises(TypeError, match="not decorated with @gpu_buffer"):
            create_wgpu_buffer(NotAGpuBuffer, None)

    def test_wgpu_buffer_allocation_with_label(self):
        from trinity.decorators import WgpuBufferAllocation
        alloc = WgpuBufferAllocation(size=64, usage=0x0040, mapped=True, label="UBO")
        assert alloc.size == 64
        assert alloc.usage == 0x0040
        assert alloc.mapped is True
        assert alloc.label == "UBO"

    def test_import_all_re_exports_from_all(self):
        from trinity.decorators import __all__
        expected = {
            "VALID_BUFFER_USAGE",
            "WgpuBufferAllocation",
            "allocate_wgpu_buffer",
            "create_wgpu_buffer",
        }
        for name in expected:
            assert name in __all__, f"{name} missing from trinity.decorators.__all__"
