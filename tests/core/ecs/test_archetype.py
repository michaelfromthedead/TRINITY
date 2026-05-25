"""Tests for archetype storage, SoA layout, add/remove entity, swap-remove."""
import pytest
from engine.core.ecs.archetype import Archetype, ArchetypeGraph
from engine.core.ecs.component import component_id
from engine.core.ecs.entity import Entity


class Pos:
    __slots__ = ("x", "y")
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Vel:
    __slots__ = ("vx", "vy")
    def __init__(self, vx=0, vy=0):
        self.vx = vx
        self.vy = vy


POS_ID = component_id(Pos)
VEL_ID = component_id(Vel)


class TestArchetype:
    def test_add_and_get(self):
        mask = frozenset({POS_ID})
        arch = Archetype(mask)
        e = Entity(0, 0)
        p = Pos(1, 2)
        arch.add_entity(e, {POS_ID: p})
        assert len(arch) == 1
        assert arch.get_component(e, POS_ID) is p

    def test_set_component(self):
        mask = frozenset({POS_ID})
        arch = Archetype(mask)
        e = Entity(0, 0)
        arch.add_entity(e, {POS_ID: Pos(0, 0)})
        new_p = Pos(5, 5)
        arch.set_component(e, POS_ID, new_p)
        assert arch.get_component(e, POS_ID) is new_p

    def test_swap_remove(self):
        mask = frozenset({POS_ID})
        arch = Archetype(mask)
        e0 = Entity(0, 0)
        e1 = Entity(1, 0)
        e2 = Entity(2, 0)
        arch.add_entity(e0, {POS_ID: Pos(0, 0)})
        arch.add_entity(e1, {POS_ID: Pos(1, 1)})
        arch.add_entity(e2, {POS_ID: Pos(2, 2)})

        # Remove middle element
        removed = arch.remove_entity(e0)
        assert removed is not None
        assert len(arch) == 2
        # e2 should have been swapped into row 0
        assert arch.get_component(e2, POS_ID).x == 2
        assert arch.get_component(e1, POS_ID).x == 1

    def test_remove_last(self):
        mask = frozenset({POS_ID})
        arch = Archetype(mask)
        e = Entity(0, 0)
        arch.add_entity(e, {POS_ID: Pos(9, 9)})
        removed = arch.remove_entity(e)
        assert removed is not None
        assert len(arch) == 0

    def test_soa_layout(self):
        mask = frozenset({POS_ID, VEL_ID})
        arch = Archetype(mask)
        e0 = Entity(0, 0)
        e1 = Entity(1, 0)
        arch.add_entity(e0, {POS_ID: Pos(1, 2), VEL_ID: Vel(3, 4)})
        arch.add_entity(e1, {POS_ID: Pos(5, 6), VEL_ID: Vel(7, 8)})
        # Columns are separate lists
        assert len(arch.columns[POS_ID]) == 2
        assert len(arch.columns[VEL_ID]) == 2


class TestArchetypeGraph:
    def test_get_or_create(self):
        g = ArchetypeGraph()
        mask = frozenset({POS_ID})
        a1 = g.get_or_create(mask)
        a2 = g.get_or_create(mask)
        assert a1 is a2

    def test_add_edge(self):
        g = ArchetypeGraph()
        src = frozenset({POS_ID})
        target = g.get_add_target(src, VEL_ID)
        assert target == frozenset({POS_ID, VEL_ID})

    def test_remove_edge(self):
        g = ArchetypeGraph()
        src = frozenset({POS_ID, VEL_ID})
        target = g.get_remove_target(src, VEL_ID)
        assert target == frozenset({POS_ID})
