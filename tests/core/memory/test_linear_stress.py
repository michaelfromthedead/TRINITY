"""Stress tests for LinearAllocator."""

import pytest

from engine.core.memory.linear import LinearAllocator


class TestLinearAllocatorStress:
    def test_stress_large_alloc_overflow(self) -> None:
        """Allocate near capacity and verify overflow detection."""
        a = LinearAllocator(1048576)  # 1 MiB
        a.allocate(1024000)
        # 1,024,000 + 1 = 1,024,001 < 1,048,576, so allocate(1) would NOT overflow.
        # Use 148,576 so sum 1,172,576 > 1,048,576 triggers MemoryError.
        with pytest.raises(MemoryError):
            a.allocate(148576)
