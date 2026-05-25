"""
T-GPU-1.6 FIX#2 — Cleanroom (blackbox) tests for @gpu_buffer.

Acceptance criteria:
  1. @gpu_buffer(usage={"storage","indirect"}) -> resolved flags include COPY_DST
  2. @gpu_buffer(usage={"storage"}) -> resolved flags do NOT include COPY_DST
  3. @gpu_buffer(usage={"indirect"}) -> resolved flags include COPY_DST
  4. allocate_wgpu_buffer works and returns a WgpuBufferAllocation
  5. create_wgpu_buffer works (returns allocation when device is None/mock)
  6. Unknown flag warns (ValueError for direct, warning for string)
"""

from __future__ import annotations

import pytest
from trinity.decorators import (
    VALID_BUFFER_USAGE,
    WgpuBufferAllocation,
    allocate_wgpu_buffer,
    create_wgpu_buffer,
    gpu_buffer,
)
from trinity.decorators.gpu import _resolve_wgpu_usage_flags

COPY_DST = 0x0008
STORAGE  = 0x0080
INDIRECT = 0x0100
VERTEX   = 0x0020


# ── helpers ──────────────────────────────────────────────────────────────────

def _resolved_usage(*names: str) -> int:
    """Decorate an anonymous class and return the resolved usage bitmask."""
    @gpu_buffer(usage=set(names))
    class _Buf:
        data: float
    return allocate_wgpu_buffer(_Buf, device=None).usage


# =============================================================================
# Criterion 1: storage+indirect includes COPY_DST
# =============================================================================

class TestStorageIndirectIncludesCopyDst:
    def test_storage_indirect_has_copy_dst(self):
        usage = _resolved_usage("storage", "indirect")
        assert usage & COPY_DST, "storage+indirect should include COPY_DST"

    def test_storage_indirect_has_storage(self):
        usage = _resolved_usage("storage", "indirect")
        assert usage & STORAGE

    def test_storage_indirect_has_indirect(self):
        usage = _resolved_usage("storage", "indirect")
        assert usage & INDIRECT

    def test_storage_indirect_bitmask_exact(self):
        """Verify the exact combined bitmask."""
        usage = _resolved_usage("storage", "indirect")
        # indirect auto-appends COPY_DST
        assert usage == (STORAGE | INDIRECT | COPY_DST)


# =============================================================================
# Criterion 2: storage alone has NO COPY_DST
# =============================================================================

class TestStorageOnlyNoCopyDst:
    def test_storage_only_no_copy_dst(self):
        usage = _resolved_usage("storage")
        assert not (usage & COPY_DST), "storage alone should NOT include COPY_DST"

    def test_storage_only_has_storage(self):
        usage = _resolved_usage("storage")
        assert usage & STORAGE

    def test_storage_only_exact(self):
        usage = _resolved_usage("storage")
        assert usage == STORAGE

    def test_storage_and_other_no_indirect(self):
        """storage + vertex should still NOT have COPY_DST."""
        usage = _resolved_usage("storage", "vertex")
        assert not (usage & COPY_DST)
        assert usage & STORAGE
        assert usage & VERTEX


# =============================================================================
# Criterion 3: indirect alone has COPY_DST
# =============================================================================

class TestIndirectOnlyIncludesCopyDst:
    def test_indirect_only_has_copy_dst(self):
        usage = _resolved_usage("indirect")
        assert usage & COPY_DST, "indirect alone should include COPY_DST"

    def test_indirect_only_has_indirect(self):
        usage = _resolved_usage("indirect")
        assert usage & INDIRECT

    def test_indirect_only_exact(self):
        usage = _resolved_usage("indirect")
        assert usage == (INDIRECT | COPY_DST)

    def test_indirect_with_other_flags(self):
        """indirect + vertex should include COPY_DST."""
        usage = _resolved_usage("indirect", "vertex")
        assert usage & COPY_DST
        assert usage & INDIRECT
        assert usage & VERTEX


# =============================================================================
# Criterion 4: allocate_wgpu_buffer works
# =============================================================================

class TestAllocateWgpuBufferWorks:
    def test_returns_wgpu_buffer_allocation(self):
        @gpu_buffer(usage={"storage"})
        class T:
            data: float
        result = allocate_wgpu_buffer(T, device=None)
        assert isinstance(result, WgpuBufferAllocation)

    def test_correct_size(self):
        @gpu_buffer(usage={"storage"})
        class T:
            x: float
            y: float
        result = allocate_wgpu_buffer(T, device=None)
        assert result.size == 8

    def test_correct_label_default(self):
        @gpu_buffer(usage={"storage"})
        class MyBuf:
            data: float
        result = allocate_wgpu_buffer(MyBuf, device=None)
        assert result.label == "MyBuf"

    def test_custom_label(self):
        @gpu_buffer(usage={"storage"})
        class T:
            data: float
        result = allocate_wgpu_buffer(T, device=None, label="custom_label")
        assert result.label == "custom_label"

    def test_mapped_default_false(self):
        @gpu_buffer(usage={"storage"})
        class T:
            data: float
        result = allocate_wgpu_buffer(T, device=None)
        assert not result.mapped

    def test_undecorated_class_raises(self):
        class T:
            data: float
        with pytest.raises((TypeError, AttributeError)):
            allocate_wgpu_buffer(T, device=None)


# =============================================================================
# Criterion 5: create_wgpu_buffer works (handles device=None gracefully)
# =============================================================================

class TestCreateWgpuBufferWorks:
    def test_none_device_returns_allocation(self):
        @gpu_buffer(usage={"storage"})
        class T:
            data: float
        result = create_wgpu_buffer(T, device=None)
        assert isinstance(result, WgpuBufferAllocation)

    def test_object_no_create_buffer_returns_allocation(self):
        @gpu_buffer(usage={"storage"})
        class T:
            data: float
        result = create_wgpu_buffer(T, device=object())
        assert isinstance(result, WgpuBufferAllocation)

    def test_mock_device_calls_create_buffer(self):
        @gpu_buffer(usage={"storage"})
        class T:
            data: float
        calls = []
        class MockDevice:
            def create_buffer(self, **kw):
                calls.append(kw)
                return "ok"
        result = create_wgpu_buffer(T, device=MockDevice())
        assert result == "ok"
        assert len(calls) == 1
        assert calls[0]["size"] == 4

    def test_undecorated_class_raises(self):
        class T:
            data: float
        with pytest.raises((TypeError, AttributeError)):
            create_wgpu_buffer(T, device=None)


# =============================================================================
# Criterion 6: unknown flag warns via ValueError for @gpu_buffer
# =============================================================================

class TestUnknownFlagWarns:
    def test_unknown_flag_in_decorator_raises_value_error(self):
        """@gpu_buffer(usage={"unknown"}) should raise ValueError."""
        with pytest.raises(ValueError):
            @gpu_buffer(usage={"unknown"})
            class T:
                data: float

    def test_mixed_valid_and_unknown_raises(self):
        with pytest.raises(ValueError):
            @gpu_buffer(usage={"storage", "bogus"})
            class T:
                data: float
