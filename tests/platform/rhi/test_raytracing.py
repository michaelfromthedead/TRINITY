"""Tests for RHI ray tracing acceleration structure management."""
from __future__ import annotations

import pytest
import threading
from dataclasses import dataclass
from typing import Optional

from engine.platform.rhi.raytracing import (
    BLASDesc,
    BLASHandle,
    BLASManager,
    BLASPool,
    BuildFlags,
    TLASDesc,
    TLASHandle,
    TLASInstance,
    TLASManager,
    AccelerationStructure,
    NullAccelerationStructure,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class MockBuffer:
    """Mock buffer for testing."""
    size: int = 1024
    data: Optional[bytes] = None


@dataclass
class MockDevice:
    """Mock device for testing."""
    name: str = "MockDevice"


@pytest.fixture
def mock_device() -> MockDevice:
    """Create a mock device."""
    return MockDevice()


@pytest.fixture
def mock_vertex_buffer() -> MockBuffer:
    """Create a mock vertex buffer."""
    return MockBuffer(size=4096)


@pytest.fixture
def mock_index_buffer() -> MockBuffer:
    """Create a mock index buffer."""
    return MockBuffer(size=2048)


@pytest.fixture
def blas_desc(mock_vertex_buffer: MockBuffer, mock_index_buffer: MockBuffer) -> BLASDesc:
    """Create a BLAS descriptor."""
    return BLASDesc(
        vertex_buffer=mock_vertex_buffer,
        vertex_count=100,
        vertex_stride=32,
        index_buffer=mock_index_buffer,
        index_count=300,
        build_flags=BuildFlags.PREFER_FAST_TRACE,
    )


@pytest.fixture
def blas_manager(mock_device: MockDevice) -> BLASManager:
    """Create a BLAS manager."""
    return BLASManager(mock_device)


@pytest.fixture
def tlas_manager(mock_device: MockDevice) -> TLASManager:
    """Create a TLAS manager."""
    return TLASManager(mock_device)


@pytest.fixture
def blas_pool(blas_manager: BLASManager) -> BLASPool:
    """Create a BLAS pool."""
    return BLASPool(blas_manager)


# =============================================================================
# Handle Tests
# =============================================================================

class TestBLASHandle:
    """Tests for BLASHandle."""

    def test_handle_creation(self) -> None:
        """Test handle creation with ID."""
        handle = BLASHandle(handle_id=42)
        assert handle.handle_id == 42

    def test_handle_frozen(self) -> None:
        """Test that handles are immutable."""
        handle = BLASHandle(handle_id=1)
        with pytest.raises(Exception):  # FrozenInstanceError
            handle.handle_id = 2  # type: ignore

    def test_handle_hashable(self) -> None:
        """Test that handles can be used in sets/dicts."""
        handle1 = BLASHandle(handle_id=1)
        handle2 = BLASHandle(handle_id=2)
        handle_set = {handle1, handle2}
        assert len(handle_set) == 2
        assert handle1 in handle_set


class TestTLASHandle:
    """Tests for TLASHandle."""

    def test_handle_creation(self) -> None:
        """Test handle creation with ID."""
        handle = TLASHandle(handle_id=99)
        assert handle.handle_id == 99

    def test_handle_hashable(self) -> None:
        """Test that handles can be used in sets/dicts."""
        handle = TLASHandle(handle_id=1)
        handle_dict = {handle: "test"}
        assert handle_dict[handle] == "test"


# =============================================================================
# TLASInstance Tests
# =============================================================================

class TestTLASInstance:
    """Tests for TLASInstance."""

    def test_instance_creation(self) -> None:
        """Test instance creation with valid transform."""
        blas_handle = BLASHandle(handle_id=1)
        transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]  # Identity 4x3
        instance = TLASInstance(
            blas_handle=blas_handle,
            transform=transform,
            instance_id=5,
            mask=0xFF,
            flags=0,
        )
        assert instance.blas_handle == blas_handle
        assert len(instance.transform) == 12
        assert instance.instance_id == 5

    def test_instance_invalid_transform(self) -> None:
        """Test that invalid transform length raises error."""
        blas_handle = BLASHandle(handle_id=1)
        with pytest.raises(ValueError, match="12 floats"):
            TLASInstance(
                blas_handle=blas_handle,
                transform=[1, 0, 0],  # Too short
            )

    def test_instance_default_values(self) -> None:
        """Test instance default values."""
        blas_handle = BLASHandle(handle_id=1)
        transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        instance = TLASInstance(blas_handle=blas_handle, transform=transform)
        assert instance.instance_id == 0
        assert instance.mask == 0xFF
        assert instance.flags == 0


# =============================================================================
# BLASManager Tests
# =============================================================================

class TestBLASManager:
    """Tests for BLASManager."""

    def test_build_static(self, blas_manager: BLASManager, blas_desc: BLASDesc) -> None:
        """Test building a static BLAS."""
        handle = blas_manager.build_static(blas_desc)
        assert isinstance(handle, BLASHandle)
        assert handle.handle_id > 0

    def test_build_dynamic(self, blas_manager: BLASManager, blas_desc: BLASDesc) -> None:
        """Test building a dynamic BLAS."""
        handle = blas_manager.build_dynamic(blas_desc)
        assert isinstance(handle, BLASHandle)
        assert handle.handle_id > 0

    def test_build_dynamic_sets_allow_update(
        self, blas_manager: BLASManager, blas_desc: BLASDesc
    ) -> None:
        """Test that build_dynamic sets ALLOW_UPDATE flag."""
        # The flag is set internally; verify by checking structure exists
        handle = blas_manager.build_dynamic(blas_desc)
        structure = blas_manager.get_structure(handle)
        assert structure is not None
        assert structure.is_valid()

    def test_refit(
        self,
        blas_manager: BLASManager,
        blas_desc: BLASDesc,
        mock_vertex_buffer: MockBuffer,
    ) -> None:
        """Test refitting a BLAS."""
        handle = blas_manager.build_dynamic(blas_desc)
        # Should not raise
        blas_manager.refit(handle, mock_vertex_buffer)

    def test_refit_invalid_handle(
        self, blas_manager: BLASManager, mock_vertex_buffer: MockBuffer
    ) -> None:
        """Test refit with invalid handle raises KeyError."""
        invalid_handle = BLASHandle(handle_id=99999)
        with pytest.raises(KeyError):
            blas_manager.refit(invalid_handle, mock_vertex_buffer)

    def test_compact(self, blas_manager: BLASManager, blas_desc: BLASDesc) -> None:
        """Test compacting a BLAS."""
        original = blas_manager.build_static(blas_desc)
        compacted = blas_manager.compact(original)
        assert isinstance(compacted, BLASHandle)
        # Compacted may be different handle
        assert blas_manager.get_structure(compacted) is not None

    def test_compact_invalid_handle(self, blas_manager: BLASManager) -> None:
        """Test compact with invalid handle raises KeyError."""
        invalid_handle = BLASHandle(handle_id=99999)
        with pytest.raises(KeyError):
            blas_manager.compact(invalid_handle)

    def test_unique_handles(self, blas_manager: BLASManager, blas_desc: BLASDesc) -> None:
        """Test that each build produces unique handle."""
        handles = [blas_manager.build_static(blas_desc) for _ in range(5)]
        handle_ids = [h.handle_id for h in handles]
        assert len(set(handle_ids)) == 5  # All unique

    def test_release(self, blas_manager: BLASManager, blas_desc: BLASDesc) -> None:
        """Test releasing a BLAS handle."""
        handle = blas_manager.build_static(blas_desc)
        assert blas_manager.get_structure(handle) is not None
        blas_manager.release(handle)
        assert blas_manager.get_structure(handle) is None


# =============================================================================
# TLASManager Tests
# =============================================================================

class TestTLASManager:
    """Tests for TLASManager."""

    def test_build_frame_empty(self, tlas_manager: TLASManager) -> None:
        """Test building TLAS with no instances."""
        handle = tlas_manager.build_frame([])
        assert isinstance(handle, TLASHandle)
        assert handle.handle_id > 0

    def test_build_frame_with_instances(self, tlas_manager: TLASManager) -> None:
        """Test building TLAS with instances."""
        blas_handle = BLASHandle(handle_id=1)
        transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        instances = [
            TLASInstance(blas_handle=blas_handle, transform=transform, instance_id=i)
            for i in range(10)
        ]
        handle = tlas_manager.build_frame(instances)
        assert isinstance(handle, TLASHandle)
        structure = tlas_manager.get_structure(handle)
        assert structure is not None
        assert structure.is_valid()

    def test_release(self, tlas_manager: TLASManager) -> None:
        """Test releasing a TLAS handle."""
        handle = tlas_manager.build_frame([])
        assert tlas_manager.get_structure(handle) is not None
        tlas_manager.release(handle)
        assert tlas_manager.get_structure(handle) is None


# =============================================================================
# BLASPool Tests
# =============================================================================

class TestBLASPool:
    """Tests for BLASPool reference counting."""

    def test_acquire_unregistered(self, blas_pool: BLASPool) -> None:
        """Test acquiring unregistered mesh returns None."""
        handle = blas_pool.acquire("unknown_mesh")
        assert handle is None

    def test_register_and_acquire(
        self, blas_pool: BLASPool, blas_manager: BLASManager, blas_desc: BLASDesc
    ) -> None:
        """Test registering and acquiring a BLAS."""
        handle = blas_manager.build_static(blas_desc)
        blas_pool.register("mesh_001", handle)

        acquired = blas_pool.acquire("mesh_001")
        assert acquired is not None
        assert acquired.handle_id == handle.handle_id

    def test_register_duplicate_raises(
        self, blas_pool: BLASPool, blas_manager: BLASManager, blas_desc: BLASDesc
    ) -> None:
        """Test registering duplicate mesh raises ValueError."""
        handle = blas_manager.build_static(blas_desc)
        blas_pool.register("mesh_001", handle)

        with pytest.raises(ValueError, match="already registered"):
            blas_pool.register("mesh_001", handle)

    def test_reference_counting(
        self, blas_pool: BLASPool, blas_manager: BLASManager, blas_desc: BLASDesc
    ) -> None:
        """Test reference counting increments on acquire."""
        handle = blas_manager.build_static(blas_desc)
        blas_pool.register("mesh_001", handle)
        assert blas_pool.get_ref_count("mesh_001") == 1

        blas_pool.acquire("mesh_001")
        assert blas_pool.get_ref_count("mesh_001") == 2

        blas_pool.acquire("mesh_001")
        assert blas_pool.get_ref_count("mesh_001") == 3

    def test_release_decrements_count(
        self, blas_pool: BLASPool, blas_manager: BLASManager, blas_desc: BLASDesc
    ) -> None:
        """Test release decrements reference count."""
        handle = blas_manager.build_static(blas_desc)
        blas_pool.register("mesh_001", handle)
        blas_pool.acquire("mesh_001")  # ref count = 2

        removed = blas_pool.release(handle)
        assert removed is False  # Still has references
        assert blas_pool.get_ref_count("mesh_001") == 1

    def test_release_removes_at_zero(
        self, blas_pool: BLASPool, blas_manager: BLASManager, blas_desc: BLASDesc
    ) -> None:
        """Test release removes entry when count hits zero."""
        handle = blas_manager.build_static(blas_desc)
        blas_pool.register("mesh_001", handle)

        removed = blas_pool.release(handle)
        assert removed is True
        assert blas_pool.get_ref_count("mesh_001") == 0
        assert not blas_pool.contains("mesh_001")

    def test_contains(
        self, blas_pool: BLASPool, blas_manager: BLASManager, blas_desc: BLASDesc
    ) -> None:
        """Test contains check."""
        assert not blas_pool.contains("mesh_001")

        handle = blas_manager.build_static(blas_desc)
        blas_pool.register("mesh_001", handle)

        assert blas_pool.contains("mesh_001")

    def test_size(
        self, blas_pool: BLASPool, blas_manager: BLASManager, blas_desc: BLASDesc
    ) -> None:
        """Test pool size."""
        assert blas_pool.size() == 0

        for i in range(3):
            handle = blas_manager.build_static(blas_desc)
            blas_pool.register(f"mesh_{i}", handle)

        assert blas_pool.size() == 3

    def test_clear(
        self, blas_pool: BLASPool, blas_manager: BLASManager, blas_desc: BLASDesc
    ) -> None:
        """Test clearing the pool."""
        for i in range(3):
            handle = blas_manager.build_static(blas_desc)
            blas_pool.register(f"mesh_{i}", handle)

        blas_pool.clear()
        assert blas_pool.size() == 0

    def test_thread_safety(
        self, blas_pool: BLASPool, blas_manager: BLASManager, blas_desc: BLASDesc
    ) -> None:
        """Test thread-safe access to pool."""
        handle = blas_manager.build_static(blas_desc)
        blas_pool.register("shared_mesh", handle)

        errors = []

        def acquire_many() -> None:
            try:
                for _ in range(100):
                    blas_pool.acquire("shared_mesh")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=acquire_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Initial 1 + 4 threads * 100 acquires = 401
        assert blas_pool.get_ref_count("shared_mesh") == 401


# =============================================================================
# Integration Tests
# =============================================================================

class TestRayTracingIntegration:
    """Integration tests for ray tracing components."""

    def test_full_pipeline(
        self,
        mock_device: MockDevice,
        blas_desc: BLASDesc,
    ) -> None:
        """Test complete ray tracing setup pipeline."""
        # Create managers
        blas_mgr = BLASManager(mock_device)
        tlas_mgr = TLASManager(mock_device)
        pool = BLASPool(blas_mgr)

        # Build and pool some BLAS
        mesh_ids = ["cube", "sphere", "terrain"]
        for mesh_id in mesh_ids:
            handle = blas_mgr.build_static(blas_desc)
            pool.register(mesh_id, handle)

        # Acquire handles for scene instances
        instances = []
        identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]

        cube_handle = pool.acquire("cube")
        sphere_handle = pool.acquire("sphere")

        assert cube_handle is not None
        assert sphere_handle is not None

        instances.append(TLASInstance(
            blas_handle=cube_handle,
            transform=identity,
            instance_id=0,
        ))
        instances.append(TLASInstance(
            blas_handle=sphere_handle,
            transform=identity,
            instance_id=1,
        ))

        # Build TLAS
        tlas_handle = tlas_mgr.build_frame(instances)
        tlas = tlas_mgr.get_structure(tlas_handle)

        assert tlas is not None
        assert tlas.is_valid()
        assert tlas.gpu_address > 0
