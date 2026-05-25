"""RHI Ray tracing support (stub implementation)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import List, Optional
import threading

from ..constants import GPU_ADDRESS_START, ACCELERATION_STRUCTURE_ALIGNMENT


class BuildFlags(Flag):
    """Acceleration structure build flags."""
    PREFER_FAST_TRACE = auto()
    PREFER_FAST_BUILD = auto()
    ALLOW_UPDATE = auto()


@dataclass
class BLASDesc:
    """Bottom-level acceleration structure descriptor."""
    vertex_buffer: 'Buffer'
    vertex_count: int
    vertex_stride: int
    index_buffer: Optional['Buffer'] = None
    index_count: int = 0
    build_flags: BuildFlags = BuildFlags.PREFER_FAST_TRACE


@dataclass
class TLASDesc:
    """Top-level acceleration structure descriptor."""
    instance_count: int
    instance_buffer: 'Buffer'
    build_flags: BuildFlags = BuildFlags.PREFER_FAST_TRACE


class AccelerationStructure(ABC):
    """Abstract acceleration structure."""

    @classmethod
    @abstractmethod
    def create_blas(cls, device: 'Device', desc: BLASDesc) -> AccelerationStructure:
        """Create bottom-level acceleration structure."""
        pass

    @classmethod
    @abstractmethod
    def create_tlas(cls, device: 'Device', desc: TLASDesc) -> AccelerationStructure:
        """Create top-level acceleration structure."""
        pass

    @property
    @abstractmethod
    def gpu_address(self) -> int:
        """Get GPU address."""
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if acceleration structure is valid."""
        pass


class NullAccelerationStructure(AccelerationStructure):
    """Null implementation of AccelerationStructure."""

    _next_address = GPU_ADDRESS_START
    _lock = threading.Lock()

    def __init__(self, device: 'Device', desc):
        self._device = device
        self._desc = desc
        self._valid = True

        with NullAccelerationStructure._lock:
            self._gpu_address = NullAccelerationStructure._next_address
            # Allocate 64KB per structure (arbitrary for null impl)
            NullAccelerationStructure._next_address += ACCELERATION_STRUCTURE_ALIGNMENT

    @classmethod
    def create_blas(cls, device: 'Device', desc: BLASDesc) -> AccelerationStructure:
        """Create bottom-level acceleration structure."""
        return cls(device, desc)

    @classmethod
    def create_tlas(cls, device: 'Device', desc: TLASDesc) -> AccelerationStructure:
        """Create top-level acceleration structure."""
        return cls(device, desc)

    @property
    def gpu_address(self) -> int:
        """Get GPU address."""
        return self._gpu_address

    def is_valid(self) -> bool:
        """Check if acceleration structure is valid."""
        return self._valid


# Forward declarations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .device import Device
    from .resources import Buffer
