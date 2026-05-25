"""
Tests for atomic operations.
"""
import time
import pytest
from engine.platform.os.threading import Thread
from engine.platform.os.atomics import (
    AtomicInt,
    AtomicFloat,
    AtomicBool,
    AtomicRef,
)


def test_atomic_int_basic():
    """Test basic AtomicInt operations."""
    atomic = AtomicInt(10)

    assert atomic.load() == 10

    atomic.store(20)
    assert atomic.load() == 20

    old = atomic.exchange(30)
    assert old == 20
    assert atomic.load() == 30


def test_atomic_int_compare_exchange():
    """Test compare-exchange operation."""
    atomic = AtomicInt(10)

    # Successful exchange
    success, value = atomic.compare_exchange(10, 20)
    assert success
    assert value == 20
    assert atomic.load() == 20

    # Failed exchange
    success, value = atomic.compare_exchange(10, 30)
    assert not success
    assert value == 20
    assert atomic.load() == 20


def test_atomic_int_fetch_add():
    """Test fetch-add operation."""
    atomic = AtomicInt(10)

    old = atomic.fetch_add(5)
    assert old == 10
    assert atomic.load() == 15

    new = atomic.add_fetch(5)
    assert new == 20
    assert atomic.load() == 20


def test_atomic_int_fetch_sub():
    """Test fetch-sub operation."""
    atomic = AtomicInt(10)

    old = atomic.fetch_sub(3)
    assert old == 10
    assert atomic.load() == 7

    new = atomic.sub_fetch(2)
    assert new == 5
    assert atomic.load() == 5


def test_atomic_int_increment_decrement():
    """Test increment and decrement."""
    atomic = AtomicInt(10)

    value = atomic.increment()
    assert value == 11

    value = atomic.decrement()
    assert value == 10


def test_atomic_int_concurrent():
    """Test AtomicInt under concurrent access."""
    atomic = AtomicInt(0)
    iterations = 1000

    def worker():
        for _ in range(iterations):
            atomic.increment()

    threads = [Thread(target=worker) for _ in range(10)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    assert atomic.load() == 10 * iterations


def test_atomic_float_basic():
    """Test basic AtomicFloat operations."""
    atomic = AtomicFloat(1.5)

    assert atomic.load() == 1.5

    atomic.store(2.5)
    assert atomic.load() == 2.5

    old = atomic.exchange(3.5)
    assert old == 2.5
    assert atomic.load() == 3.5


def test_atomic_float_compare_exchange():
    """Test float compare-exchange."""
    atomic = AtomicFloat(1.5)

    success, value = atomic.compare_exchange(1.5, 2.5)
    assert success
    assert value == 2.5

    success, value = atomic.compare_exchange(1.5, 3.5)
    assert not success
    assert value == 2.5


def test_atomic_float_arithmetic():
    """Test atomic float arithmetic."""
    atomic = AtomicFloat(10.0)

    old = atomic.fetch_add(5.5)
    assert old == 10.0
    assert atomic.load() == 15.5

    old = atomic.fetch_sub(3.5)
    assert old == 15.5
    assert atomic.load() == 12.0


def test_atomic_bool_basic():
    """Test basic AtomicBool operations."""
    atomic = AtomicBool(False)

    assert not atomic.load()

    atomic.store(True)
    assert atomic.load()

    old = atomic.exchange(False)
    assert old
    assert not atomic.load()


def test_atomic_bool_test_and_set():
    """Test test-and-set operation."""
    atomic = AtomicBool(False)

    old = atomic.test_and_set()
    assert not old
    assert atomic.load()

    old = atomic.test_and_set()
    assert old


def test_atomic_bool_clear():
    """Test clear operation."""
    atomic = AtomicBool(True)

    atomic.clear()
    assert not atomic.load()


def test_atomic_bool_compare_exchange():
    """Test bool compare-exchange."""
    atomic = AtomicBool(False)

    success, value = atomic.compare_exchange(False, True)
    assert success
    assert value

    success, value = atomic.compare_exchange(False, False)
    assert not success
    assert value


def test_atomic_bool_concurrent():
    """Test AtomicBool for synchronization."""
    atomic = AtomicBool(False)
    counter = [0]

    def worker():
        # Spin until we can acquire
        while atomic.test_and_set():
            pass  # Busy wait, testing atomicity without sleep

        # Critical section
        counter[0] += 1
        # Increase iterations to stress-test atomicity
        for _ in range(1000):
            temp = counter[0]
            counter[0] = temp

        # Release
        atomic.clear()

    threads = [Thread(target=worker) for _ in range(5)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    assert counter[0] == 5


def test_atomic_ref_basic():
    """Test basic AtomicRef operations."""
    obj1 = {"value": 1}
    obj2 = {"value": 2}

    atomic = AtomicRef(obj1)

    assert atomic.load() is obj1

    atomic.store(obj2)
    assert atomic.load() is obj2

    old = atomic.exchange(obj1)
    assert old is obj2
    assert atomic.load() is obj1


def test_atomic_ref_compare_exchange():
    """Test ref compare-exchange."""
    obj1 = {"value": 1}
    obj2 = {"value": 2}
    obj3 = {"value": 3}

    atomic = AtomicRef(obj1)

    # Successful exchange
    success, value = atomic.compare_exchange(obj1, obj2)
    assert success
    assert value is obj2

    # Failed exchange (wrong expected)
    success, value = atomic.compare_exchange(obj1, obj3)
    assert not success
    assert value is obj2


def test_atomic_ref_none():
    """Test AtomicRef with None."""
    atomic = AtomicRef(None)

    assert atomic.load() is None

    obj = {"value": 1}
    atomic.store(obj)
    assert atomic.load() is obj

    old = atomic.exchange(None)
    assert old is obj
    assert atomic.load() is None


def test_atomic_ref_concurrent():
    """Test AtomicRef under concurrent updates."""
    atomic = AtomicRef([])
    objects = [{"id": i} for i in range(10)]

    def worker(obj):
        atomic.store(obj)
        time.sleep(0.001)

    threads = [Thread(target=worker, args=(obj,)) for obj in objects]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Should have one of the objects
    result = atomic.load()
    assert result in objects
