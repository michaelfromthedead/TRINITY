"""
Tests for threading primitives.
"""
import time
import pytest
from engine.platform.os.threading import (
    Thread,
    ThreadConfig,
    ThreadPriority,
    Mutex,
    RWLock,
    Semaphore,
    CondVar,
    Barrier,
    ThreadLocalStorage,
)


def test_thread_basic():
    """Test basic thread creation and execution."""
    result = []

    def worker():
        result.append(1)

    config = ThreadConfig(name="test-thread")
    thread = Thread(target=worker, config=config)

    thread.start()
    thread.join()

    assert result == [1]
    assert not thread.is_alive()


def test_thread_with_args():
    """Test thread with arguments."""
    result = []

    def worker(a, b):
        result.append(a + b)

    thread = Thread(target=worker, args=(5, 3))
    thread.start()
    thread.join()

    assert result == [8]


def test_thread_affinity():
    """Test thread with CPU affinity (may not have effect without privileges)."""
    result = []

    def worker():
        result.append(1)

    config = ThreadConfig(affinity=[0], priority=ThreadPriority.HIGH)
    thread = Thread(target=worker, config=config)

    thread.start()
    thread.join()

    assert result == [1]


def test_mutex_basic():
    """Test mutex lock/unlock."""
    mutex = Mutex()

    mutex.lock()
    mutex.unlock()


def test_mutex_context_manager():
    """Test mutex as context manager."""
    mutex = Mutex()
    counter = [0]

    def worker():
        for _ in range(100):
            with mutex:
                counter[0] += 1

    threads = [Thread(target=worker) for _ in range(10)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    assert counter[0] == 1000


def test_mutex_try_lock():
    """Test try_lock."""
    mutex = Mutex()

    assert mutex.try_lock()
    assert not mutex.try_lock()
    mutex.unlock()


def test_mutex_try_lock_for():
    """Test try_lock_for with timeout."""
    mutex = Mutex()

    assert mutex.try_lock_for(0.1)
    assert not mutex.try_lock_for(0.1)
    mutex.unlock()


def test_rwlock_read():
    """Test RWLock read operations."""
    rwlock = RWLock()
    data = [0]

    def reader():
        with rwlock.read():
            time.sleep(0.01)
            _ = data[0]

    threads = [Thread(target=reader) for _ in range(5)]

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - start

    # Multiple readers should run concurrently
    assert elapsed < 0.1


def test_rwlock_write():
    """Test RWLock write operations."""
    rwlock = RWLock()
    data = [0]

    def writer():
        with rwlock.write():
            data[0] += 1
            time.sleep(0.01)

    threads = [Thread(target=writer) for _ in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert data[0] == 5


def test_rwlock_mixed():
    """Test RWLock with mixed read/write."""
    rwlock = RWLock()
    data = [0]
    reads = []

    def reader():
        with rwlock.read():
            reads.append(data[0])

    def writer():
        with rwlock.write():
            data[0] += 1

    # Start some readers
    reader_threads = [Thread(target=reader) for _ in range(3)]
    for t in reader_threads:
        t.start()

    # Write
    writer_thread = Thread(target=writer)
    writer_thread.start()
    writer_thread.join()

    # More readers
    for t in reader_threads:
        t.join()

    assert data[0] == 1


def test_semaphore():
    """Test semaphore."""
    sem = Semaphore(2)
    counter = [0]

    def worker():
        with sem:
            counter[0] += 1
            time.sleep(0.05)
            counter[0] -= 1

    threads = [Thread(target=worker) for _ in range(4)]

    for t in threads:
        t.start()

    time.sleep(0.02)
    # At most 2 should be running
    assert counter[0] <= 2

    for t in threads:
        t.join()


def test_condvar():
    """Test condition variable."""
    mutex = Mutex()
    condvar = CondVar(mutex)
    ready = [False]

    def waiter():
        with condvar:
            while not ready[0]:
                condvar.wait()

    def notifier():
        time.sleep(0.05)
        with condvar:
            ready[0] = True
            condvar.notify()

    waiter_thread = Thread(target=waiter)
    notifier_thread = Thread(target=notifier)

    waiter_thread.start()
    notifier_thread.start()

    waiter_thread.join()
    notifier_thread.join()

    assert ready[0]


def test_barrier():
    """Test barrier."""
    barrier = Barrier(3)
    results = []

    def worker(worker_id):
        results.append(f"start_{worker_id}")
        barrier.wait()
        results.append(f"end_{worker_id}")

    threads = [Thread(target=worker, args=(i,)) for i in range(3)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # All starts should come before all ends
    starts = [r for r in results if r.startswith("start")]
    ends = [r for r in results if r.startswith("end")]

    assert len(starts) == 3
    assert len(ends) == 3

    # Find the index of the last start and first end
    last_start_idx = max(results.index(s) for s in starts)
    first_end_idx = min(results.index(e) for e in ends)

    # All starts must complete before any ends
    assert last_start_idx < first_end_idx


def test_barrier_properties():
    """Test barrier properties."""
    barrier = Barrier(5)

    assert barrier.parties == 5
    assert barrier.n_waiting == 0
    assert not barrier.broken


def test_thread_local_storage():
    """Test thread-local storage."""
    tls = ThreadLocalStorage()
    results = {}

    def worker(worker_id):
        tls.set("id", worker_id)
        time.sleep(0.01)
        results[worker_id] = tls.get("id")

    threads = [Thread(target=worker, args=(i,)) for i in range(5)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Each thread should have its own value
    for i in range(5):
        assert results[i] == i


def test_thread_local_storage_operations():
    """Test TLS operations."""
    tls = ThreadLocalStorage()

    assert not tls.has("key")
    tls.set("key", "value")
    assert tls.has("key")
    assert tls.get("key") == "value"

    tls.delete("key")
    assert not tls.has("key")
    assert tls.get("key") is None
    assert tls.get("key", "default") == "default"
