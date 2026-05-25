"""Tests for query matching, With/Without filters, iteration."""
import pytest
from engine.core.ecs.archetype import ArchetypeGraph
from engine.core.ecs.component import component_id
from engine.core.ecs.entity import Entity
from engine.core.ecs.query import Query, QueryDescriptor, With, Without


class A:
    __slots__ = ("v",)
    def __init__(self, v=0): self.v = v

class B:
    __slots__ = ("v",)
    def __init__(self, v=0): self.v = v

class C:
    __slots__ = ("v",)
    def __init__(self, v=0): self.v = v


A_ID = component_id(A)
B_ID = component_id(B)
C_ID = component_id(C)


def _populate(graph: ArchetypeGraph):
    """Add some entities to archetypes for testing."""
    # Archetype {A, B}
    mask_ab = frozenset({A_ID, B_ID})
    arch_ab = graph.get_or_create(mask_ab)
    e0 = Entity(0, 0)
    arch_ab.add_entity(e0, {A_ID: A(1), B_ID: B(2)})

    # Archetype {A}
    mask_a = frozenset({A_ID})
    arch_a = graph.get_or_create(mask_a)
    e1 = Entity(1, 0)
    arch_a.add_entity(e1, {A_ID: A(3)})

    # Archetype {A, C}
    mask_ac = frozenset({A_ID, C_ID})
    arch_ac = graph.get_or_create(mask_ac)
    e2 = Entity(2, 0)
    arch_ac.add_entity(e2, {A_ID: A(5), C_ID: C(6)})


class TestQuery:
    def test_match_required(self):
        g = ArchetypeGraph()
        _populate(g)
        desc = QueryDescriptor(required=(A_ID,))
        q = Query(desc, g)
        results = list(q.iter())
        assert len(results) == 3  # all have A

    def test_match_multiple_required(self):
        g = ArchetypeGraph()
        _populate(g)
        desc = QueryDescriptor(required=(A_ID, B_ID))
        q = Query(desc, g)
        results = list(q.iter())
        assert len(results) == 1

    def test_with_filter(self):
        g = ArchetypeGraph()
        _populate(g)
        desc = QueryDescriptor(required=(A_ID,), with_=frozenset({C_ID}))
        q = Query(desc, g)
        results = list(q.iter())
        assert len(results) == 1
        assert results[0][1].v == 5

    def test_without_filter(self):
        g = ArchetypeGraph()
        _populate(g)
        desc = QueryDescriptor(required=(A_ID,), without=frozenset({B_ID}))
        q = Query(desc, g)
        results = list(q.iter())
        assert len(results) == 2  # {A} and {A, C}

    def test_result_tuple_shape(self):
        g = ArchetypeGraph()
        _populate(g)
        desc = QueryDescriptor(required=(A_ID, B_ID))
        q = Query(desc, g)
        results = list(q.iter())
        entity, a_comp, b_comp = results[0]
        assert isinstance(entity, Entity)
        assert isinstance(a_comp, A)
        assert isinstance(b_comp, B)

    def test_empty_query(self):
        g = ArchetypeGraph()
        desc = QueryDescriptor(required=(A_ID,))
        q = Query(desc, g)
        assert list(q.iter()) == []
