"""RHI Ray tracing support (stub implementation)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import List, Optional, Dict
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


@dataclass(frozen=True)
class BLASHandle:
    """Handle to a bottom-level acceleration structure."""
    handle_id: int


@dataclass(frozen=True)
class TLASHandle:
    """Handle to a top-level acceleration structure."""
    handle_id: int


@dataclass
class TLASInstance:
    """Instance in a top-level acceleration structure."""
    blas_handle: BLASHandle
    transform: List[float] = field(default_factory=lambda: [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0
    ])  # 3x4 row-major transform matrix
    instance_id: int = 0
    mask: int = 0xFF
    sbt_offset: int = 0
    flags: int = 0

    def __post_init__(self):
        """Validate transform after initialization."""
        if len(self.transform) != 12:
            raise ValueError(
                f"Transform must have exactly 12 floats (3x4 matrix), got {len(self.transform)}"
            )


class BLASManager:
    """Manages bottom-level acceleration structures."""

    def __init__(self, device: 'Device'):
        self._device = device
        self._structures: Dict[int, AccelerationStructure] = {}
        self._descs: Dict[int, BLASDesc] = {}
        self._next_id = 1  # Start at 1 so handle_id > 0
        self._lock = threading.Lock()

    def build_static(self, desc: BLASDesc) -> BLASHandle:
        """Build a static BLAS (no updates allowed)."""
        with self._lock:
            handle_id = self._next_id
            self._next_id += 1
            # For static, don't set ALLOW_UPDATE
            build_desc = BLASDesc(
                vertex_buffer=desc.vertex_buffer,
                vertex_count=desc.vertex_count,
                vertex_stride=desc.vertex_stride,
                index_buffer=desc.index_buffer,
                index_count=desc.index_count,
                build_flags=desc.build_flags & ~BuildFlags.ALLOW_UPDATE,
            )
            structure = NullAccelerationStructure.create_blas(self._device, build_desc)
            self._structures[handle_id] = structure
            self._descs[handle_id] = build_desc
            return BLASHandle(handle_id=handle_id)

    def build_dynamic(self, desc: BLASDesc) -> BLASHandle:
        """Build a dynamic BLAS (supports updates)."""
        with self._lock:
            handle_id = self._next_id
            self._next_id += 1
            # For dynamic, set ALLOW_UPDATE
            build_desc = BLASDesc(
                vertex_buffer=desc.vertex_buffer,
                vertex_count=desc.vertex_count,
                vertex_stride=desc.vertex_stride,
                index_buffer=desc.index_buffer,
                index_count=desc.index_count,
                build_flags=desc.build_flags | BuildFlags.ALLOW_UPDATE,
            )
            structure = NullAccelerationStructure.create_blas(self._device, build_desc)
            self._structures[handle_id] = structure
            self._descs[handle_id] = build_desc
            return BLASHandle(handle_id=handle_id)

    def refit(self, handle: BLASHandle, vertex_buffer: 'Buffer') -> None:
        """Refit a dynamic BLAS with updated vertex data."""
        with self._lock:
            if handle.handle_id not in self._structures:
                raise KeyError(f"Invalid BLAS handle: {handle.handle_id}")
            # In a real impl, would refit using new vertex buffer
            # For null impl, just verify it exists

    def compact(self, handle: BLASHandle) -> BLASHandle:
        """Compact a BLAS to reduce memory."""
        with self._lock:
            if handle.handle_id not in self._structures:
                raise KeyError(f"Invalid BLAS handle: {handle.handle_id}")
            # Create compacted copy
            new_handle_id = self._next_id
            self._next_id += 1
            old_desc = self._descs[handle.handle_id]
            structure = NullAccelerationStructure.create_blas(self._device, old_desc)
            self._structures[new_handle_id] = structure
            self._descs[new_handle_id] = old_desc
            return BLASHandle(handle_id=new_handle_id)

    def release(self, handle: BLASHandle) -> None:
        """Release a BLAS handle."""
        with self._lock:
            if handle.handle_id in self._structures:
                del self._structures[handle.handle_id]
            if handle.handle_id in self._descs:
                del self._descs[handle.handle_id]

    def get_structure(self, handle: BLASHandle) -> Optional[AccelerationStructure]:
        """Get acceleration structure by handle."""
        with self._lock:
            return self._structures.get(handle.handle_id)


class TLASManager:
    """Manages top-level acceleration structures."""

    def __init__(self, device: 'Device'):
        self._device = device
        self._structures: Dict[int, AccelerationStructure] = {}
        self._next_id = 1  # Start at 1 so handle_id > 0
        self._lock = threading.Lock()

    def build_frame(self, instances: List[TLASInstance]) -> TLASHandle:
        """Build a TLAS from instances for the current frame."""
        with self._lock:
            handle_id = self._next_id
            self._next_id += 1
            # Create a mock desc for the null implementation
            # In a real impl, would build from instances
            structure = NullAccelerationStructure(self._device, instances)
            self._structures[handle_id] = structure
            return TLASHandle(handle_id=handle_id)

    def release(self, handle: TLASHandle) -> None:
        """Release a TLAS handle."""
        with self._lock:
            if handle.handle_id in self._structures:
                del self._structures[handle.handle_id]

    def get_structure(self, handle: TLASHandle) -> Optional[AccelerationStructure]:
        """Get acceleration structure by handle."""
        with self._lock:
            return self._structures.get(handle.handle_id)


class BLASPool:
    """Pool for BLAS structures with reference counting by mesh name."""

    def __init__(self, manager: BLASManager):
        self._manager = manager
        self._entries: Dict[str, BLASHandle] = {}
        self._ref_counts: Dict[str, int] = {}
        self._handle_to_name: Dict[int, str] = {}
        self._lock = threading.Lock()

    def register(self, mesh_name: str, handle: BLASHandle) -> None:
        """Register a BLAS handle with a mesh name."""
        with self._lock:
            if mesh_name in self._entries:
                raise ValueError(f"Mesh '{mesh_name}' already registered")
            self._entries[mesh_name] = handle
            self._ref_counts[mesh_name] = 1
            self._handle_to_name[handle.handle_id] = mesh_name

    def acquire(self, mesh_name: str) -> Optional[BLASHandle]:
        """Acquire a BLAS handle by mesh name, incrementing reference count."""
        with self._lock:
            if mesh_name not in self._entries:
                return None
            self._ref_counts[mesh_name] += 1
            return self._entries[mesh_name]

    def release(self, handle: BLASHandle) -> bool:
        """
        Release a BLAS handle, decrementing reference count.

        Returns True if the entry was removed (ref count hit 0).
        """
        with self._lock:
            mesh_name = self._handle_to_name.get(handle.handle_id)
            if mesh_name is None:
                return False
            self._ref_counts[mesh_name] -= 1
            if self._ref_counts[mesh_name] <= 0:
                del self._entries[mesh_name]
                del self._ref_counts[mesh_name]
                del self._handle_to_name[handle.handle_id]
                return True
            return False

    def contains(self, mesh_name: str) -> bool:
        """Check if mesh is registered in pool."""
        with self._lock:
            return mesh_name in self._entries

    def size(self) -> int:
        """Get number of registered meshes."""
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        """Clear all entries from pool."""
        with self._lock:
            self._entries.clear()
            self._ref_counts.clear()
            self._handle_to_name.clear()

    def get_ref_count(self, mesh_name: str) -> int:
        """Get reference count for a mesh."""
        with self._lock:
            return self._ref_counts.get(mesh_name, 0)


# Forward declarations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .device import Device
    from .resources import Buffer
