"""
System information: CPU, memory, environment variables.
"""
import os
import platform
from dataclasses import dataclass
from typing import Optional

from ..constants import DEFAULT_CACHE_LINE_SIZE, HYPERTHREADING_RATIO, BYTES_PER_KB


@dataclass(slots=True)
class CPUInfo:
    """CPU information."""
    logical_count: int
    physical_count: int
    architecture: str
    processor: str


@dataclass(slots=True)
class MemoryInfo:
    """Memory information in bytes."""
    total: int
    available: int
    used: int
    percent: float


class SystemInfo:
    """System information provider."""

    @staticmethod
    def cpu_count() -> int:
        """Get number of logical CPU cores."""
        count = os.cpu_count()
        return count if count is not None else 1

    @staticmethod
    def cpu_count_physical() -> int:
        """
        Get number of physical CPU cores.
        Falls back to logical count if not determinable.
        """
        try:
            # Try to read from /sys on Linux
            with open('/sys/devices/system/cpu/present', 'r') as f:
                content = f.read().strip()
                # Format is like "0-7" for 8 cores
                if '-' in content:
                    start, end = content.split('-')
                    return int(end) - int(start) + 1
        except Exception:
            pass

        # Fallback: assume physical = logical / HYPERTHREADING_RATIO (for hyperthreading)
        logical = SystemInfo.cpu_count()
        return max(1, logical // HYPERTHREADING_RATIO)

    @staticmethod
    def get_cpu_info() -> CPUInfo:
        """Get detailed CPU information."""
        return CPUInfo(
            logical_count=SystemInfo.cpu_count(),
            physical_count=SystemInfo.cpu_count_physical(),
            architecture=platform.machine(),
            processor=platform.processor()
        )

    @staticmethod
    def total_memory() -> int:
        """Get total physical memory in bytes."""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        # Value is in kB
                        kb = int(line.split()[1])
                        return kb * BYTES_PER_KB
        except Exception:
            pass

        return 0

    @staticmethod
    def available_memory() -> int:
        """Get available physical memory in bytes."""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        kb = int(line.split()[1])
                        return kb * BYTES_PER_KB
                    elif line.startswith('MemFree:'):
                        # Fallback to MemFree if MemAvailable not present
                        kb = int(line.split()[1])
                        return kb * BYTES_PER_KB
        except Exception:
            pass

        return 0

    @staticmethod
    def get_memory_info() -> MemoryInfo:
        """Get detailed memory information."""
        total = SystemInfo.total_memory()
        available = SystemInfo.available_memory()
        used = total - available
        percent = (used / total * 100) if total > 0 else 0.0

        return MemoryInfo(
            total=total,
            available=available,
            used=used,
            percent=percent
        )

    @staticmethod
    def cache_line_size() -> int:
        """
        Get CPU cache line size in bytes.
        Returns typical value if not determinable.
        """
        try:
            # Try to read from sysfs (Linux)
            with open('/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size', 'r') as f:
                return int(f.read().strip())
        except Exception:
            pass

        # Typical cache line size for modern CPUs
        return DEFAULT_CACHE_LINE_SIZE

    @staticmethod
    def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable."""
        return os.environ.get(key, default)

    @staticmethod
    def set_env(key: str, value: str):
        """Set environment variable."""
        os.environ[key] = value

    @staticmethod
    def unset_env(key: str):
        """Unset environment variable."""
        if key in os.environ:
            del os.environ[key]

    @staticmethod
    def get_all_env() -> dict[str, str]:
        """Get all environment variables."""
        return dict(os.environ)

    @staticmethod
    def platform_name() -> str:
        """Get platform name (Linux, Windows, Darwin, etc.)."""
        return platform.system()

    @staticmethod
    def platform_version() -> str:
        """Get platform version."""
        return platform.release()

    @staticmethod
    def hostname() -> str:
        """Get system hostname."""
        return platform.node()
