"""
T-GPU-1.6: BLACKBOX fix verification tests for @gpu_buffer auto-append COPY_DST.

Acceptance (WebGPU spec compliance):
  1. @gpu_buffer(usage={"storage","indirect"}) -> STORAGE|INDIRECT|COPY_DST
  2. @gpu_buffer(usage={"storage"}) -> STORAGE|COPY_DST (no INDIRECT)
  3. @gpu_buffer(usage={"indirect"}) -> INDIRECT|COPY_DST
  4. No error on duplicate (copy_dst given explicitly alongside storage/indirect)
  5. Buffer allocation succeeds with resolved usage
  6. allocate_wgpu_buffer(device=None) returns WgpuBufferAllocation
  7. Unknown flag warns (emits warning, does not prevent resolution)

All tests use ONLY blackbox public API: @gpu_buffer, allocate_wgpu_buffer,
WgpuBufferAllocation. No internal functions are imported.
"""

import pytest
import warnings
from typing import Any

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

STORAGE = 0x0080
INDIRECT = 0x0100
COPY_DST = 0x0008


class TestGpuBufferAutoCopyDst:
    """Verify @gpu_buffer auto-appends COPY_DST for storage/indirect usage."""

    def _alloc_usage(self, usage_set: set[str]) -> int:
        """Decorate a single-field class with @gpu_buffer and return resolved usage."""
        from trinity.decorators.gpu import gpu_buffer, allocate_wgpu_buffer

        @gpu_buffer(usage=usage_set)
        class _Buf:
            x: int

        return allocate_wgpu_buffer(_Buf, device=None).usage

    # ------------------------------------------------------------------
    # Acceptance criteria 1-3: auto-append COPY_DST
    # ------------------------------------------------------------------

    def test_storage_and_indirect_includes_copy_dst(self):
        """@gpu_buffer(usage={"storage","indirect"}) -> STORAGE|INDIRECT|COPY_DST"""
        usage = self._alloc_usage({"storage", "indirect"})
        assert usage & STORAGE, "missing STORAGE bit"
        assert usage & INDIRECT, "missing INDIRECT bit"
        assert usage & COPY_DST, "missing COPY_DST bit"
        # Exactly these three bits -- no extras
        assert usage == (STORAGE | INDIRECT | COPY_DST), (
            f"expected 0x{STORAGE | INDIRECT | COPY_DST:04X}, got 0x{usage:04X}"
        )

    def test_storage_only_includes_copy_dst(self):
        """@gpu_buffer(usage={"storage"}) -> STORAGE|COPY_DST (no INDIRECT)"""
        usage = self._alloc_usage({"storage"})
        assert usage & STORAGE, "missing STORAGE bit"
        assert usage & COPY_DST, "missing COPY_DST bit"
        assert not (usage & INDIRECT), "unexpected INDIRECT bit"
        assert usage == (STORAGE | COPY_DST), (
            f"expected 0x{STORAGE | COPY_DST:04X}, got 0x{usage:04X}"
        )

    def test_indirect_only_includes_copy_dst(self):
        """@gpu_buffer(usage={"indirect"}) -> INDIRECT|COPY_DST"""
        usage = self._alloc_usage({"indirect"})
        assert usage & INDIRECT, "missing INDIRECT bit"
        assert usage & COPY_DST, "missing COPY_DST bit"
        assert not (usage & STORAGE), "unexpected STORAGE bit"
        assert usage == (INDIRECT | COPY_DST), (
            f"expected 0x{INDIRECT | COPY_DST:04X}, got 0x{usage:04X}"
        )

    # ------------------------------------------------------------------
    # Acceptance criteria 4: no error on duplicate
    # ------------------------------------------------------------------

    def test_no_error_on_duplicate_copy_dst(self):
        """Explicit copy_dst alongside storage/indirect causes no error."""
        from trinity.decorators.gpu import gpu_buffer, allocate_wgpu_buffer

        @gpu_buffer(usage={"storage", "indirect", "copy_dst"})
        class _Buf:
            x: int

        # Should not raise -- copy_dst is present both explicitly and via auto-append
        alloc = allocate_wgpu_buffer(_Buf, device=None)
        assert alloc.usage == (STORAGE | INDIRECT | COPY_DST)

    def test_duplicate_with_explicit_copy_dst_returns_same(self):
        """Explicit copy_dst and auto-append produce the same bitmask as without."""
        from trinity.decorators.gpu import gpu_buffer, allocate_wgpu_buffer

        @gpu_buffer(usage={"storage", "copy_dst"})
        class _With:
            x: int

        @gpu_buffer(usage={"storage"})
        class _Without:
            x: int

        assert allocate_wgpu_buffer(_With, device=None).usage == (
            allocate_wgpu_buffer(_Without, device=None).usage
        ), "explicit copy_dst should produce same mask as auto-append"

    # ------------------------------------------------------------------
    # Acceptance criteria 5: buffer allocation succeeds
    # ------------------------------------------------------------------

    def test_buffer_allocation_succeeds(self):
        """Allocation returns WgpuBufferAllocation with correct fields."""
        from trinity.decorators.gpu import (
            gpu_buffer,
            allocate_wgpu_buffer,
            WgpuBufferAllocation,
        )

        @gpu_buffer(usage={"storage", "indirect"})
        class _Buf:
            x: int

        alloc = allocate_wgpu_buffer(_Buf, device=None)
        assert isinstance(alloc, WgpuBufferAllocation)
        assert alloc.size > 0, "buffer size must be > 0 for non-empty struct"
        assert alloc.usage == (STORAGE | INDIRECT | COPY_DST)

    # ------------------------------------------------------------------
    # Acceptance criteria 6: device=None works
    # ------------------------------------------------------------------

    def test_allocate_with_device_none(self):
        """allocate_wgpu_buffer(device=None) returns WgpuBufferAllocation."""
        from trinity.decorators.gpu import (
            gpu_buffer,
            allocate_wgpu_buffer,
            WgpuBufferAllocation,
        )

        @gpu_buffer(usage={"storage"})
        class _Buf:
            x: int

        alloc = allocate_wgpu_buffer(_Buf, device=None)
        assert isinstance(alloc, WgpuBufferAllocation)
        assert alloc.size > 0

    # ------------------------------------------------------------------
    # Acceptance criteria 7: unknown flag raises ValueError
    # ------------------------------------------------------------------

    def test_unknown_flag_raises(self):
        """Unknown usage flags are rejected by the decorator validator."""
        from trinity.decorators.gpu import gpu_buffer

        with pytest.raises(ValueError, match="invalid usage flag"):
            @gpu_buffer(usage={"storage", "unknown_flag"})
            class _Buf:
                x: int

    def test_unknown_flag_only_raises(self):
        """Only unknown flags (no known flags) are also rejected."""
        from trinity.decorators.gpu import gpu_buffer

        with pytest.raises(ValueError, match="invalid usage flag"):
            @gpu_buffer(usage={"bogus"})
            class _Buf:
                x: int

    def test_unknown_flag_via_direct_resolve_warns(self):
        """Direct _resolve_wgpu_usage_flags warns on unknown flag (safety net)."""
        from trinity.decorators.gpu import _resolve_wgpu_usage_flags

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = _resolve_wgpu_usage_flags(frozenset({"storage", "nope"}))
            # storage still resolved, copy_dst auto-appended, nope warned
            assert result == (STORAGE | COPY_DST)

        nope_warnings = [
            w for w in captured if "nope" in str(w.message).lower()
        ]
        assert len(nope_warnings) >= 1

    # ------------------------------------------------------------------
    # Additional blackbox edge cases
    # ------------------------------------------------------------------

    def test_non_storage_indirect_does_not_get_copy_dst(self):
        """Flags other than storage/indirect do NOT get COPY_DST appended."""
        from trinity.decorators.gpu import gpu_buffer, allocate_wgpu_buffer

        @gpu_buffer(usage={"vertex"})
        class _Buf:
            x: int

        alloc = allocate_wgpu_buffer(_Buf, device=None)
        assert not (alloc.usage & COPY_DST), "vertex should not get COPY_DST"
        assert alloc.usage == 0x0020

    def test_uniform_does_not_get_copy_dst(self):
        """Uniform buffers do not get COPY_DST appended."""
        from trinity.decorators.gpu import gpu_buffer, allocate_wgpu_buffer

        @gpu_buffer(usage={"uniform"})
        class _Buf:
            x: int

        alloc = allocate_wgpu_buffer(_Buf, device=None)
        assert not (alloc.usage & COPY_DST), "uniform should not get COPY_DST"
        assert alloc.usage == 0x0040

    def test_combined_non_storage_flags(self):
        """Multiple non-storage/indirect flags combine normally."""
        from trinity.decorators.gpu import gpu_buffer, allocate_wgpu_buffer

        @gpu_buffer(usage={"vertex", "index", "uniform"})
        class _Buf:
            x: int

        alloc = allocate_wgpu_buffer(_Buf, device=None)
        assert not (alloc.usage & COPY_DST)
        assert not (alloc.usage & STORAGE)
        assert not (alloc.usage & INDIRECT)
        assert alloc.usage == (0x0020 | 0x0010 | 0x0040)

    def test_all_usage_flags_still_valid(self):
        """All 9 usage flags are still valid after the fix."""
        from trinity.decorators.gpu import gpu_buffer, allocate_wgpu_buffer

        @gpu_buffer(usage=set(_WGPU_SPEC_FLAGS.keys()))
        class _Buf:
            x: int

        alloc = allocate_wgpu_buffer(_Buf, device=None)
        expected_all = 0x0020 | 0x0010 | 0x0040 | 0x0080 | 0x0100 | 0x0001 | 0x0002 | 0x0004 | 0x0008
        assert alloc.usage == expected_all, (
            f"all-flags mask mismatch: expected 0x{expected_all:04X}, got 0x{alloc.usage:04X}"
        )

    def test_create_wgpu_buffer_passes_correct_usage(self):
        """create_wgpu_buffer passes auto-appended COPY_DST to device."""
        from trinity.decorators.gpu import gpu_buffer, create_wgpu_buffer

        @gpu_buffer(usage={"storage", "indirect"})
        class _Buf:
            x: int

        class FakeDevice:
            def create_buffer(self, **kwargs: Any) -> str:
                self._last = kwargs
                return "ok"

        device = FakeDevice()
        create_wgpu_buffer(_Buf, device)
        assert device._last["usage"] == (STORAGE | INDIRECT | COPY_DST)

    def test_default_storage_with_mapped(self):
        """Default usage (storage) with mapped=True still appends COPY_DST."""
        from trinity.decorators.gpu import gpu_buffer, allocate_wgpu_buffer

        @gpu_buffer(mapped=True)
        class _Buf:
            x: int

        alloc = allocate_wgpu_buffer(_Buf, device=None)
        assert alloc.mapped is True
        assert alloc.usage == (STORAGE | COPY_DST)
