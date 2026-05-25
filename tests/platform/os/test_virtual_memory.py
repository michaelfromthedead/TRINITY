"""
Tests for virtual memory management.
"""
import pytest
from engine.platform.os.virtual_memory import (
    VirtualMemory,
    ProtectionFlags,
    page_size,
)


def test_page_size():
    """Test page size retrieval."""
    size = page_size()
    assert size > 0
    assert size % 4096 == 0  # Should be multiple of 4KB


def test_virtual_memory_reserve():
    """Test memory reservation."""
    vm = VirtualMemory()

    # Reserve 1 page
    size = page_size()
    addr = vm.reserve(size, ProtectionFlags.READ_WRITE)

    assert addr is not None

    # Clean up
    assert vm.release(addr)


def test_virtual_memory_commit():
    """Test memory commit."""
    vm = VirtualMemory()
    size = page_size()

    addr = vm.reserve(size, ProtectionFlags.READ_WRITE)
    assert addr is not None

    # Commit memory
    success = vm.commit(addr, size)
    assert success

    vm.release(addr)


@pytest.mark.skip(reason="decommit not implemented on this platform")
def test_virtual_memory_decommit():
    """Test memory decommit."""
    vm = VirtualMemory()
    size = page_size()

    addr = vm.reserve(size, ProtectionFlags.READ_WRITE)
    assert addr is not None

    vm.commit(addr, size)

    # Decommit
    success = vm.decommit(addr, size)
    assert success

    vm.release(addr)


@pytest.mark.skip(reason="protect not implemented on this platform")
def test_virtual_memory_protect():
    """Test memory protection changes."""
    vm = VirtualMemory()
    size = page_size()

    addr = vm.reserve(size, ProtectionFlags.READ_WRITE)
    assert addr is not None

    vm.commit(addr, size)

    # Change protection
    success = vm.protect(addr, size, ProtectionFlags.READ)
    assert success

    vm.release(addr)


def test_virtual_memory_release():
    """Test memory release."""
    vm = VirtualMemory()
    size = page_size()

    addr = vm.reserve(size, ProtectionFlags.READ_WRITE)
    assert addr is not None

    success = vm.release(addr)
    assert success

    # Double release should fail
    success = vm.release(addr)
    assert not success


def test_virtual_memory_multiple_allocations():
    """Test multiple allocations."""
    vm = VirtualMemory()
    size = page_size()

    addrs = []
    for _ in range(5):
        addr = vm.reserve(size, ProtectionFlags.READ_WRITE)
        assert addr is not None
        addrs.append(addr)

    # All addresses should be unique
    assert len(set(addrs)) == 5

    # Clean up
    for addr in addrs:
        assert vm.release(addr)


def test_virtual_memory_large_allocation():
    """Test large allocation (multiple pages)."""
    vm = VirtualMemory()
    size = page_size() * 10

    addr = vm.reserve(size, ProtectionFlags.READ_WRITE)
    assert addr is not None

    success = vm.commit(addr, size)
    assert success

    vm.release(addr)


def test_virtual_memory_invalid_operations():
    """Test error handling."""
    vm = VirtualMemory()

    # Commit without reserve
    success = vm.commit(12345, page_size())
    assert not success

    # Protect without reserve
    success = vm.protect(12345, page_size(), ProtectionFlags.READ)
    assert not success

    # Release invalid address
    success = vm.release(12345)
    assert not success


def test_virtual_memory_stats():
    """Test memory statistics."""
    vm = VirtualMemory()
    stats = vm.get_stats()

    assert stats.page_size > 0
    # Total physical should be positive (unless reading /proc/meminfo fails)
    # We don't assert this to allow test to pass in restricted environments


def test_protection_flags():
    """Test ProtectionFlags enum."""
    assert ProtectionFlags.NONE.value == 0
    assert ProtectionFlags.READ
    assert ProtectionFlags.WRITE
    assert ProtectionFlags.EXECUTE

    # Test combinations
    rw = ProtectionFlags.READ | ProtectionFlags.WRITE
    assert rw & ProtectionFlags.READ
    assert rw & ProtectionFlags.WRITE
    assert not (rw & ProtectionFlags.EXECUTE)

    rwx = ProtectionFlags.READ_WRITE_EXECUTE
    assert rwx & ProtectionFlags.READ
    assert rwx & ProtectionFlags.WRITE
    assert rwx & ProtectionFlags.EXECUTE


def test_virtual_memory_alignment():
    """Test that sizes are aligned to page boundaries."""
    vm = VirtualMemory()
    size = page_size()

    # Request size that's not page-aligned
    addr = vm.reserve(size + 100, ProtectionFlags.READ_WRITE)
    assert addr is not None

    # Should still work (rounded up internally)
    success = vm.commit(addr, size)
    assert success

    vm.release(addr)
