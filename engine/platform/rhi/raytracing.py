"""RHI Ray tracing support (stub implementation)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Dict, List, Optional
import threading

from ..constants import GPU_ADDRESS_START, ACCELERATION_STRUCTURE_ALIGNMENT


# =============================================================================
# Handle Types
# =============================================================================

@dataclass(frozen=True)
class BLASHandle:
    """Handle to a bottom-level acceleration structure."""
    handle_id: int

    def __hash__(self) -> int:
        return hash(self.handle_id)


@dataclass(frozen=True)
class TLASHandle:
    """Handle to a top-level acceleration structure."""
    handle_id: int

    def __hash__(self) -> int:
        return hash(self.handle_id)


# =============================================================================
# Instance Descriptor
# =============================================================================

@dataclass
class TLASInstance:
    """Instance descriptor for TLAS building.

    Describes a single instance referencing a BLAS with its transform.
    """
    blas_handle: BLASHandle
    transform: List[float]  # 4x3 row-major (12 floats)
    instance_id: int = 0
    mask: int = 0xFF
    flags: int = 0

    def __post_init__(self) -> None:
        if len(self.transform) != 12:
            raise ValueError(
                f"Transform must have 12 floats (4x3 row-major), got {len(self.transform)}"
            )


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


# =============================================================================
# Manager Classes
# =============================================================================

class BLASManager:
    """Manager for bottom-level acceleration structures.

    Provides methods to build, refit, and compact BLAS instances.
    Thread-safe handle allocation.
    """

    _next_handle_id: int = 1
    _lock: threading.Lock = threading.Lock()

    def __init__(self, device: 'Device') -> None:
        """Initialize BLAS manager.

        Args:
            device: The RHI device to use for BLAS creation.
        """
        self._device = device
        self._structures: Dict[int, AccelerationStructure] = {}

    def _allocate_handle(self) -> BLASHandle:
        """Allocate a unique handle ID (thread-safe)."""
        with BLASManager._lock:
            handle_id = BLASManager._next_handle_id
            BLASManager._next_handle_id += 1
        return BLASHandle(handle_id=handle_id)

    def build_static(self, desc: BLASDesc) -> BLASHandle:
        """Build a static BLAS optimized for fast tracing.

        Static BLAS cannot be updated after creation. Use for geometry
        that never changes (terrain, buildings, etc.).

        Args:
            desc: BLAS descriptor with geometry data.

        Returns:
            Handle to the created BLAS.
        """
        handle = self._allocate_handle()
        # Stub: In real implementation, would call into Rust backend
        structure = NullAccelerationStructure.create_blas(self._device, desc)
        self._structures[handle.handle_id] = structure
        return handle

    def build_dynamic(self, desc: BLASDesc) -> BLASHandle:
        """Build a dynamic BLAS that can be updated.

        Dynamic BLAS has ALLOW_UPDATE flag set, allowing refit operations.
        Use for deformable geometry (skinned meshes, cloth, etc.).

        Args:
            desc: BLAS descriptor with geometry data.

        Returns:
            Handle to the created BLAS.
        """
        # Ensure ALLOW_UPDATE flag is set
        dynamic_desc = BLASDesc(
            vertex_buffer=desc.vertex_buffer,
            vertex_count=desc.vertex_count,
            vertex_stride=desc.vertex_stride,
            index_buffer=desc.index_buffer,
            index_count=desc.index_count,
            build_flags=desc.build_flags | BuildFlags.ALLOW_UPDATE
        )
        handle = self._allocate_handle()
        structure = NullAccelerationStructure.create_blas(self._device, dynamic_desc)
        self._structures[handle.handle_id] = structure
        return handle

    def refit(self, handle: BLASHandle, vertex_buffer: 'Buffer') -> None:
        """Refit an existing BLAS with updated vertex data.

        Only valid for BLAS built with ALLOW_UPDATE flag.
        Faster than full rebuild but may reduce trace performance.

        Args:
            handle: Handle to the BLAS to refit.
            vertex_buffer: New vertex buffer with updated positions.

        Raises:
            KeyError: If handle is not valid.
        """
        if handle.handle_id not in self._structures:
            raise KeyError(f"Invalid BLAS handle: {handle.handle_id}")
        # Stub: In real implementation, would update the structure
        # The vertex buffer reference is stored for the refit operation
        _ = vertex_buffer

    def compact(self, handle: BLASHandle) -> BLASHandle:
        """Compact a BLAS to reduce memory usage.

        Compaction creates a new, smaller BLAS. The original handle
        remains valid until explicitly released.

        Args:
            handle: Handle to the BLAS to compact.

        Returns:
            Handle to the compacted BLAS (may be same as input).

        Raises:
            KeyError: If handle is not valid.
        """
        if handle.handle_id not in self._structures:
            raise KeyError(f"Invalid BLAS handle: {handle.handle_id}")
        # Stub: In real implementation, would perform compaction
        # For null impl, just return a new handle pointing to same structure
        new_handle = self._allocate_handle()
        self._structures[new_handle.handle_id] = self._structures[handle.handle_id]
        return new_handle

    def get_structure(self, handle: BLASHandle) -> Optional[AccelerationStructure]:
        """Get the acceleration structure for a handle.

        Args:
            handle: Handle to look up.

        Returns:
            The acceleration structure, or None if handle invalid.
        """
        return self._structures.get(handle.handle_id)

    def release(self, handle: BLASHandle) -> None:
        """Release a BLAS handle.

        Args:
            handle: Handle to release.
        """
        self._structures.pop(handle.handle_id, None)


class TLASManager:
    """Manager for top-level acceleration structures.

    Handles per-frame TLAS building from BLAS instances.
    """

    _next_handle_id: int = 1
    _lock: threading.Lock = threading.Lock()

    def __init__(self, device: 'Device') -> None:
        """Initialize TLAS manager.

        Args:
            device: The RHI device to use for TLAS creation.
        """
        self._device = device
        self._structures: Dict[int, AccelerationStructure] = {}

    def _allocate_handle(self) -> TLASHandle:
        """Allocate a unique handle ID (thread-safe)."""
        with TLASManager._lock:
            handle_id = TLASManager._next_handle_id
            TLASManager._next_handle_id += 1
        return TLASHandle(handle_id=handle_id)

    def build_frame(self, instances: List[TLASInstance]) -> TLASHandle:
        """Build a TLAS for the current frame.

        Creates a top-level acceleration structure from a list of
        BLAS instances with their transforms.

        Args:
            instances: List of TLAS instances referencing BLAS handles.

        Returns:
            Handle to the created TLAS.
        """
        handle = self._allocate_handle()
        # Stub: In real implementation, would:
        # 1. Create instance buffer from instances list
        # 2. Build TLAS referencing the BLAS handles
        desc = TLASDesc(
            instance_count=len(instances),
            instance_buffer=None,  # Would be real buffer
            build_flags=BuildFlags.PREFER_FAST_BUILD
        )
        structure = NullAccelerationStructure.create_tlas(self._device, desc)
        self._structures[handle.handle_id] = structure
        return handle

    def get_structure(self, handle: TLASHandle) -> Optional[AccelerationStructure]:
        """Get the acceleration structure for a handle.

        Args:
            handle: Handle to look up.

        Returns:
            The acceleration structure, or None if handle invalid.
        """
        return self._structures.get(handle.handle_id)

    def release(self, handle: TLASHandle) -> None:
        """Release a TLAS handle.

        Args:
            handle: Handle to release.
        """
        self._structures.pop(handle.handle_id, None)


class BLASPool:
    """Reference-counted pool for shared BLAS instances.

    Manages BLAS sharing across multiple users of the same mesh asset.
    Uses reference counting to track when BLAS can be safely released.
    """

    def __init__(self, blas_manager: BLASManager) -> None:
        """Initialize the BLAS pool.

        Args:
            blas_manager: The BLAS manager to use for building structures.
        """
        self._blas_manager = blas_manager
        self._pool: Dict[str, BLASHandle] = {}
        self._ref_counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def acquire(self, mesh_asset_id: str) -> Optional[BLASHandle]:
        """Acquire a BLAS handle for a mesh asset.

        If the mesh asset already has a BLAS, increments reference count.
        If not, returns None (caller should build and register).

        Args:
            mesh_asset_id: Unique identifier for the mesh asset.

        Returns:
            Handle to the BLAS if exists, None otherwise.
        """
        with self._lock:
            if mesh_asset_id in self._pool:
                self._ref_counts[mesh_asset_id] += 1
                return self._pool[mesh_asset_id]
            return None

    def register(self, mesh_asset_id: str, handle: BLASHandle) -> None:
        """Register a newly built BLAS with the pool.

        Args:
            mesh_asset_id: Unique identifier for the mesh asset.
            handle: Handle to the BLAS to register.

        Raises:
            ValueError: If mesh_asset_id already registered.
        """
        with self._lock:
            if mesh_asset_id in self._pool:
                raise ValueError(f"Mesh asset already registered: {mesh_asset_id}")
            self._pool[mesh_asset_id] = handle
            self._ref_counts[mesh_asset_id] = 1

    def release(self, handle: BLASHandle) -> bool:
        """Release a reference to a BLAS handle.

        Decrements reference count. When count reaches zero, the BLAS
        is removed from the pool and should be destroyed.

        Args:
            handle: Handle to release.

        Returns:
            True if BLAS was removed from pool (ref count hit zero),
            False if still referenced.
        """
        with self._lock:
            # Find the mesh_asset_id for this handle
            mesh_asset_id = None
            for asset_id, pooled_handle in self._pool.items():
                if pooled_handle.handle_id == handle.handle_id:
                    mesh_asset_id = asset_id
                    break

            if mesh_asset_id is None:
                return False

            self._ref_counts[mesh_asset_id] -= 1
            if self._ref_counts[mesh_asset_id] <= 0:
                del self._pool[mesh_asset_id]
                del self._ref_counts[mesh_asset_id]
                self._blas_manager.release(handle)
                return True
            return False

    def get_ref_count(self, mesh_asset_id: str) -> int:
        """Get the reference count for a mesh asset.

        Args:
            mesh_asset_id: Unique identifier for the mesh asset.

        Returns:
            Reference count, or 0 if not in pool.
        """
        with self._lock:
            return self._ref_counts.get(mesh_asset_id, 0)

    def contains(self, mesh_asset_id: str) -> bool:
        """Check if a mesh asset is in the pool.

        Args:
            mesh_asset_id: Unique identifier for the mesh asset.

        Returns:
            True if mesh asset is pooled.
        """
        with self._lock:
            return mesh_asset_id in self._pool

    def clear(self) -> None:
        """Clear all entries from the pool.

        Releases all BLAS handles. Use with caution.
        """
        with self._lock:
            for handle in self._pool.values():
                self._blas_manager.release(handle)
            self._pool.clear()
            self._ref_counts.clear()

    def size(self) -> int:
        """Get number of entries in the pool.

        Returns:
            Number of pooled BLAS handles.
        """
        with self._lock:
            return len(self._pool)


# Forward declarations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .device import Device
    from .resources import Buffer
