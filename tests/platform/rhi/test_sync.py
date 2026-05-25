"""Tests for RHI synchronization."""
import pytest
import threading
import time
from engine.platform.rhi import (
    Fence, NullFence, ResourceState, BarrierType, BarrierDesc,
    NullAdapter, NullDevice, DeviceConfig
)


@pytest.fixture
def device():
    """Create test device."""
    adapter = NullAdapter()
    return NullDevice.create(adapter, DeviceConfig(adapter=adapter))


def test_fence_creation(device):
    """Test fence creation."""
    fence = NullFence.create(device, initial=0)

    assert fence is not None
    assert isinstance(fence, Fence)
    assert fence.value == 0


def test_fence_initial_value(device):
    """Test fence initial value."""
    fence = NullFence.create(device, initial=42)
    assert fence.value == 42


def test_fence_signal(device):
    """Test fence signal."""
    fence = NullFence.create(device, initial=0)

    fence.signal(10)
    assert fence.value == 10

    fence.signal(20)
    assert fence.value == 20


def test_fence_is_complete(device):
    """Test fence completion check."""
    fence = NullFence.create(device, initial=0)

    assert fence.is_complete(0)
    assert not fence.is_complete(1)

    fence.signal(5)

    assert fence.is_complete(0)
    assert fence.is_complete(5)
    assert not fence.is_complete(6)


def test_fence_wait_immediate(device):
    """Test fence wait when already signaled."""
    fence = NullFence.create(device, initial=10)

    # Should return immediately since already at value
    result = fence.wait(5, timeout_ms=1000)
    assert result is True


def test_fence_wait_timeout(device):
    """Test fence wait with timeout."""
    fence = NullFence.create(device, initial=0)

    # Should timeout since fence won't be signaled
    start = time.time()
    result = fence.wait(100, timeout_ms=100)
    elapsed = (time.time() - start) * 1000

    assert result is False
    assert elapsed >= 90  # Should wait at least ~100ms


def test_fence_wait_signal_threaded(device):
    """Test fence wait/signal across threads with one thread waiting, another signaling."""
    fence = NullFence.create(device, initial=0)

    wait_result = [None]

    def wait_thread():
        # Wait for fence to reach value 5
        wait_result[0] = fence.wait(5, timeout_ms=1000)

    def signal_thread():
        time.sleep(0.1)  # Wait a bit before signaling
        fence.signal(5)

    # Start both threads
    t_wait = threading.Thread(target=wait_thread)
    t_signal = threading.Thread(target=signal_thread)

    t_wait.start()
    t_signal.start()

    # Join both
    t_wait.join()
    t_signal.join()

    # Verify the wait succeeded after signal
    assert wait_result[0] is True
    assert fence.value == 5


def test_fence_wait_infinite(device):
    """Test fence wait with infinite timeout."""
    fence = NullFence.create(device, initial=0)

    def signal_after_delay():
        time.sleep(0.05)
        fence.signal(1)

    t = threading.Thread(target=signal_after_delay)
    t.start()

    # Wait with infinite timeout (-1)
    result = fence.wait(1, timeout_ms=-1)

    t.join()
    assert result is True


def test_resource_state_transition_barrier():
    """Test creating barrier descriptors with transition states."""
    # Create a transition barrier from COPY_DST to SHADER_RESOURCE
    barrier = BarrierDesc(
        type=BarrierType.TRANSITION,
        resource=None,
        state_before=ResourceState.COPY_DST,
        state_after=ResourceState.SHADER_RESOURCE
    )

    # Verify all fields are stored correctly
    assert barrier.type == BarrierType.TRANSITION
    assert barrier.state_before == ResourceState.COPY_DST
    assert barrier.state_after == ResourceState.SHADER_RESOURCE

    # Create another transition
    barrier2 = BarrierDesc(
        type=BarrierType.TRANSITION,
        resource=None,
        state_before=ResourceState.RENDER_TARGET,
        state_after=ResourceState.PRESENT
    )

    assert barrier2.state_before == ResourceState.RENDER_TARGET
    assert barrier2.state_after == ResourceState.PRESENT


def test_barrier_type_uav_vs_transition():
    """Test creating different barrier types."""
    # UAV barrier
    uav_barrier = BarrierDesc(type=BarrierType.UAV, resource=None)
    assert uav_barrier.type == BarrierType.UAV

    # Transition barrier
    transition_barrier = BarrierDesc(
        type=BarrierType.TRANSITION,
        resource=None,
        state_before=ResourceState.COMMON,
        state_after=ResourceState.UNORDERED_ACCESS
    )
    assert transition_barrier.type == BarrierType.TRANSITION

    # Verify they're different
    assert uav_barrier.type != transition_barrier.type


def test_barrier_desc_creation():
    """Test barrier descriptor creation."""
    barrier = BarrierDesc(
        type=BarrierType.TRANSITION,
        resource=None,
        state_before=ResourceState.COPY_DST,
        state_after=ResourceState.SHADER_RESOURCE
    )

    assert barrier.type == BarrierType.TRANSITION
    assert barrier.state_before == ResourceState.COPY_DST
    assert barrier.state_after == ResourceState.SHADER_RESOURCE


def test_uav_barrier_desc():
    """Test UAV barrier descriptor."""
    barrier = BarrierDesc(
        type=BarrierType.UAV,
        resource=None
    )

    assert barrier.type == BarrierType.UAV


def test_fence_multiple_threads(device):
    """Test fence with multiple waiting threads."""
    fence = NullFence.create(device, initial=0)
    results = [False, False, False]

    def wait_thread(index):
        result = fence.wait(10, timeout_ms=1000)
        results[index] = result

    # Start multiple waiting threads
    threads = [threading.Thread(target=wait_thread, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()

    # Signal after a bit
    time.sleep(0.1)
    fence.signal(10)

    # Wait for all threads
    for t in threads:
        t.join()

    # All should have succeeded
    assert all(results)


def test_fence_incremental_signal(device):
    """Test fence with incremental signaling."""
    fence = NullFence.create(device, initial=0)

    fence.signal(1)
    assert fence.is_complete(1)
    assert not fence.is_complete(2)

    fence.signal(2)
    assert fence.is_complete(2)
    assert not fence.is_complete(3)

    fence.signal(3)
    assert fence.is_complete(3)
