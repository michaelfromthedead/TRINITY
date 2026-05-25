"""Tests for entity creation, generation, recycling, null entity, validity."""
import pytest
from engine.core.ecs.entity import Entity, EntityAllocator


class TestEntity:
    def test_pack_unpack(self):
        e = Entity(42, 7)
        assert e.index == 42
        assert e.generation == 7

    def test_null(self):
        n = Entity.null()
        assert not n.is_valid()
        assert repr(n) == "Entity(null)"

    def test_valid(self):
        e = Entity(0, 0)
        assert e.is_valid()

    def test_equality_and_hash(self):
        a = Entity(1, 2)
        b = Entity(1, 2)
        c = Entity(1, 3)
        assert a == b
        assert a != c
        assert hash(a) == hash(b)
        assert {a, b} == {a}

    def test_repr(self):
        e = Entity(5, 3)
        assert "index=5" in repr(e)
        assert "gen=3" in repr(e)


class TestEntityAllocator:
    def test_allocate_sequential(self):
        alloc = EntityAllocator()
        e0 = alloc.allocate()
        e1 = alloc.allocate()
        assert e0.index == 0
        assert e1.index == 1
        assert e0.generation == 0

    def test_deallocate_and_recycle(self):
        alloc = EntityAllocator()
        e0 = alloc.allocate()
        assert alloc.is_alive(e0)
        alloc.deallocate(e0)
        assert not alloc.is_alive(e0)

        # Recycled entity gets same index, bumped generation
        e1 = alloc.allocate()
        assert e1.index == 0
        assert e1.generation == 1

    def test_is_alive_null(self):
        alloc = EntityAllocator()
        assert not alloc.is_alive(Entity.null())

    def test_generation_wraps(self):
        alloc = EntityAllocator()
        e = alloc.allocate()
        for _ in range(255):
            alloc.deallocate(e)
            e = alloc.allocate()
        assert e.generation == 255
        alloc.deallocate(e)
        e = alloc.allocate()
        assert e.generation == 0  # wrapped
