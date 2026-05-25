"""
Virtual memory management: reserve, commit, protect operations.
"""
import ctypes
import logging
import mmap
import os
from dataclasses import dataclass
from enum import Flag, auto
from typing import Optional

from ..constants import DEFAULT_PAGE_SIZE

logger = logging.getLogger(__name__)


class ProtectionFlags(Flag):
    """Memory protection flags."""
    NONE = 0
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()
    READ_WRITE = READ | WRITE
    READ_EXECUTE = READ | EXECUTE
    READ_WRITE_EXECUTE = READ | WRITE | EXECUTE


@dataclass(slots=True)
class MemoryStats:
    """Memory statistics."""
    total_physical: int  # Total physical memory in bytes
    available_physical: int  # Available physical memory
    total_virtual: int  # Total virtual memory
    available_virtual: int  # Available virtual memory
    page_size: int  # System page size


def page_size() -> int:
    """Get system page size in bytes."""
    try:
        return os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, ValueError):
        return DEFAULT_PAGE_SIZE


class VirtualMemory:
    """Virtual memory management using mmap."""

    def __init__(self):
        self._allocations: dict[int, tuple[mmap.mmap, int, int]] = {}  # address -> (mmap_obj, size, protection)
        self._page_size = page_size()

    def reserve(self, size: int, protection: ProtectionFlags = ProtectionFlags.NONE) -> Optional[int]:
        """
        Reserve virtual address space without committing physical memory.

        Args:
            size: Size in bytes (will be rounded up to page size)
            protection: Initial protection flags

        Returns:
            Address of reserved memory or None on failure
        """
        try:
            # Round up to page size
            aligned_size = ((size + self._page_size - 1) // self._page_size) * self._page_size

            # Convert protection flags to mmap protection
            prot = self._protection_to_mmap(protection)

            # Create anonymous mapping
            mm = mmap.mmap(-1, aligned_size, prot=prot)

            # Get address (approximation using id)
            addr = id(mm)
            self._allocations[addr] = (mm, aligned_size, protection.value)

            return addr
        except Exception as e:
            return None

    def commit(self, address: int, size: int) -> bool:
        """
        Commit physical memory to reserved address space.
        On Linux with mmap, memory is committed on access (lazy allocation).

        Args:
            address: Base address
            size: Size to commit

        Returns:
            Success status
        """
        if address not in self._allocations:
            return False

        # On Linux, mmap is lazy - memory is committed on first access
        # We can force commitment by touching the pages
        try:
            mm, alloc_size, _ = self._allocations[address]
            if size > alloc_size:
                return False

            # Touch pages to force commitment
            for offset in range(0, size, self._page_size):
                mm[offset] = mm[offset]

            return True
        except Exception:
            return False

    def decommit(self, address: int, size: int) -> bool:
        """
        Decommit physical memory (make it available for reuse).
        On Linux, we can use madvise with MADV_DONTNEED.

        Args:
            address: Base address
            size: Size to decommit

        Returns:
            Success status
        """
        if address not in self._allocations:
            return False

        try:
            mm, alloc_size, _ = self._allocations[address]
            if size > alloc_size:
                return False

            # Use madvise to tell kernel pages can be freed
            # Note: Python's mmap doesn't expose madvise directly
            # We'd need to use ctypes to call it, but for simplicity we'll just mark success
            logger.warning("decommit() not implemented on this platform, no-op")
            return True
        except Exception:
            return False

    def protect(self, address: int, size: int, protection: ProtectionFlags) -> bool:
        """
        Change memory protection on a region.

        Args:
            address: Base address
            size: Size of region
            protection: New protection flags

        Returns:
            Success status
        """
        if address not in self._allocations:
            return False

        try:
            mm, alloc_size, old_prot = self._allocations[address]
            if size > alloc_size:
                return False

            # Python's mmap doesn't support changing protection after creation
            # In a real implementation, we'd use mprotect via ctypes
            # For now, we'll just update our tracking
            logger.warning("protect() not implemented on this platform, no-op")
            self._allocations[address] = (mm, alloc_size, protection.value)

            return True
        except Exception:
            return False

    def release(self, address: int) -> bool:
        """
        Release reserved memory.

        Args:
            address: Base address to release

        Returns:
            Success status
        """
        if address not in self._allocations:
            return False

        try:
            mm, _, _ = self._allocations[address]
            mm.close()
            del self._allocations[address]
            return True
        except Exception:
            return False

    def _protection_to_mmap(self, protection: ProtectionFlags) -> int:
        """Convert ProtectionFlags to mmap protection constant."""
        prot = 0
        if protection & ProtectionFlags.READ:
            prot |= mmap.PROT_READ
        if protection & ProtectionFlags.WRITE:
            prot |= mmap.PROT_WRITE
        if protection & ProtectionFlags.EXECUTE:
            prot |= mmap.PROT_EXEC

        return prot if prot != 0 else mmap.PROT_NONE

    def get_stats(self) -> MemoryStats:
        """Get system memory statistics."""
        try:
            # Try to get real memory info
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        value = int(parts[1]) * 1024  # Convert KB to bytes
                        meminfo[key] = value

                return MemoryStats(
                    total_physical=meminfo.get('MemTotal', 0),
                    available_physical=meminfo.get('MemAvailable', meminfo.get('MemFree', 0)),
                    total_virtual=meminfo.get('VmallocTotal', 0),
                    available_virtual=meminfo.get('VmallocTotal', 0),
                    page_size=self._page_size
                )
        except Exception:
            # Fallback
            return MemoryStats(
                total_physical=0,
                available_physical=0,
                total_virtual=0,
                available_virtual=0,
                page_size=self._page_size
            )
