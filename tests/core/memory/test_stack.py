"""Tests for StackAllocator."""

import pytest

from engine.core.memory.stack import StackAllocator


class TestStackAllocator:
    def test_allocate_sequential(self):
        a = StackAllocator(256)
        assert a.allocate(10) == 0
        assert a.allocate(20) == 10

    def test_get_marker(self):
        a = StackAllocator(256)
        assert a.get_marker() == 0
        a.allocate(50)
        assert a.get_marker() == 50

    def test_free_to_marker(self):
        a = StackAllocator(256)
        a.allocate(20)
        marker = a.get_marker()
        a.allocate(30)
        a.allocate(40)
        assert a.used_bytes == 90
        a.free_to_marker(marker)
        assert a.used_bytes == 20

    def test_free_lifo(self):
        a = StackAllocator(128)
        a.allocate(32)
        off2 = a.allocate(32)
        a.free(off2)
        assert a.used_bytes == 32

    def test_overflow_raises(self):
        a = StackAllocator(32)
        a.allocate(30)
        with pytest.raises(MemoryError):
            a.allocate(10)

    def test_reset(self):
        a = StackAllocator(128)
        a.allocate(100)
        a.reset()
        assert a.used_bytes == 0

    def test_invalid_marker_raises(self):
        a = StackAllocator(64)
        a.allocate(32)
        with pytest.raises(ValueError):
            a.free_to_marker(64)
        with pytest.raises(ValueError):
            a.free_to_marker(-1)

    def test_invalid_free_raises(self):
        a = StackAllocator(64)
        a.allocate(10)
        with pytest.raises(ValueError):
            a.free(20)  # beyond current offset
