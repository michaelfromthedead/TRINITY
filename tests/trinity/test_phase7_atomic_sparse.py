"""Tests for AtomicDescriptor and SparseDescriptor."""

import threading

import pytest

from trinity.descriptors.atomic import AtomicDescriptor, compare_and_swap
from trinity.descriptors.sparse import SparseDescriptor, sparse_count


# =========================================================================
# Test fixtures
# =========================================================================


class AtomicEntity:
    health = AtomicDescriptor(field_type=int)
    name = AtomicDescriptor(field_type=str)


class SparseEntity:
    flags = SparseDescriptor(default=0, field_type=int)
    tag = SparseDescriptor(default="none", field_type=str)


# =========================================================================
# AtomicDescriptor tests
# =========================================================================


class TestAtomicDescriptor:
    def test_basic_get_set(self):
        e = AtomicEntity()
        e.health = 100
        assert e.health == 100

    def test_thread_safe_get_set(self):
        e = AtomicEntity()
        e.health = 0
        errors = []

        def increment():
            for _ in range(1000):
                current = e.health
                # Use CAS for true atomic increment
                while not compare_and_swap(e, "health", current, current + 1):
                    current = e.health

        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert e.health == 4000

    def test_compare_and_swap_success(self):
        e = AtomicEntity()
        e.health = 50
        result = compare_and_swap(e, "health", 50, 75)
        assert result is True
        assert e.health == 75

    def test_compare_and_swap_failure(self):
        e = AtomicEntity()
        e.health = 50
        result = compare_and_swap(e, "health", 99, 75)
        assert result is False
        assert e.health == 50

    def test_compare_and_swap_not_atomic_descriptor(self):
        class Plain:
            x = 10

        with pytest.raises(TypeError, match="not an AtomicDescriptor"):
            compare_and_swap(Plain(), "x", 10, 20)

    def test_get_metadata(self):
        desc = AtomicEntity.__dict__["health"]
        meta = desc.get_metadata()
        assert meta["descriptor_id"] == "atomic"
        assert "lock_key" in meta

    def test_descriptor_steps(self):
        desc = AtomicEntity.__dict__["health"]
        steps = desc.descriptor_steps
        assert len(steps) == 1
        assert steps[0].args["get"] == "atomic_get"


# =========================================================================
# SparseDescriptor tests
# =========================================================================


class TestSparseDescriptor:
    def test_default_value_returned(self):
        e = SparseEntity()
        assert e.flags == 0
        assert e.tag == "none"

    def test_non_default_stored(self):
        e = SparseEntity()
        e.flags = 42
        assert e.flags == 42

    def test_setting_default_removes_entry(self):
        e = SparseEntity()
        e.flags = 42
        assert e.flags == 42
        e.flags = 0  # default
        assert e.flags == 0
        # Verify actually removed from store
        store = getattr(SparseEntity, "_sparse_flags")
        assert (id(e), "flags") not in store

    def test_delete(self):
        e = SparseEntity()
        e.flags = 99
        del e.flags
        assert e.flags == 0  # returns default after delete

    def test_sparse_count(self):
        # Clear store first
        store = getattr(SparseEntity, "_sparse_flags")
        store.clear()

        a = SparseEntity()
        b = SparseEntity()
        c = SparseEntity()

        a.flags = 1
        b.flags = 2
        # c stays default

        assert sparse_count(SparseEntity, "flags") == 2

    def test_multiple_instances_independent(self):
        a = SparseEntity()
        b = SparseEntity()
        a.flags = 10
        b.flags = 20
        assert a.flags == 10
        assert b.flags == 20

    def test_get_metadata(self):
        desc = SparseEntity.__dict__["flags"]
        meta = desc.get_metadata()
        assert meta["descriptor_id"] == "sparse"
        assert meta["default"] == 0

    def test_descriptor_steps(self):
        desc = SparseEntity.__dict__["flags"]
        steps = desc.descriptor_steps
        assert len(steps) == 1
        assert steps[0].args["get"] == "sparse_get"
