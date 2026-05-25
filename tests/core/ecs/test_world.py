"""Tests for World: spawn, destroy, components, queries, is_alive."""
import pytest
from engine.core.ecs.world import World


class Pos:
    __slots__ = ("x", "y")
    def __init__(self, x=0, y=0):
        self.x = x; self.y = y


class Vel:
    __slots__ = ("vx", "vy")
    def __init__(self, vx=0, vy=0):
        self.vx = vx; self.vy = vy


class Tag:
    __slots__ = ()


class TestWorld:
    def test_spawn_and_alive(self):
        w = World()
        e = w.spawn(Pos(1, 2))
        assert w.is_alive(e)

    def test_destroy(self):
        w = World()
        e = w.spawn(Pos())
        w.destroy(e)
        assert not w.is_alive(e)

    def test_get_component(self):
        w = World()
        p = Pos(3, 4)
        e = w.spawn(p)
        assert w.get_component(e, Pos) is p

    def test_has_component(self):
        w = World()
        e = w.spawn(Pos())
        assert w.has_component(e, Pos)
        assert not w.has_component(e, Vel)

    def test_add_component(self):
        w = World()
        e = w.spawn(Pos())
        v = Vel(1, 1)
        w.add_component(e, v)
        assert w.has_component(e, Vel)
        assert w.get_component(e, Vel) is v

    def test_remove_component(self):
        w = World()
        e = w.spawn(Pos(), Vel())
        w.remove_component(e, Vel)
        assert not w.has_component(e, Vel)
        assert w.has_component(e, Pos)

    def test_overwrite_component(self):
        w = World()
        e = w.spawn(Pos(0, 0))
        new_p = Pos(9, 9)
        w.add_component(e, new_p)
        assert w.get_component(e, Pos) is new_p

    def test_query_basic(self):
        w = World()
        w.spawn(Pos(1, 1), Vel(2, 2))
        w.spawn(Pos(3, 3))
        results = list(w.query(Pos, Vel))
        assert len(results) == 1
        entity, pos, vel = results[0]
        assert pos.x == 1
        assert vel.vx == 2

    def test_query_with_filter(self):
        w = World()
        w.spawn(Pos(1, 1), Tag())
        w.spawn(Pos(2, 2))
        results = list(w.query(Pos, with_=(Tag,)))
        assert len(results) == 1
        assert results[0][1].x == 1

    def test_query_without_filter(self):
        w = World()
        w.spawn(Pos(1, 1), Tag())
        w.spawn(Pos(2, 2))
        results = list(w.query(Pos, without=(Tag,)))
        assert len(results) == 1
        assert results[0][1].x == 2

    def test_for_each(self):
        w = World()
        w.spawn(Pos(1, 0))
        w.spawn(Pos(2, 0))
        xs = []
        w.for_each(Pos, callback=lambda p: xs.append(p.x))
        assert sorted(xs) == [1, 2]

    def test_destroy_removes_from_query(self):
        w = World()
        e = w.spawn(Pos(1, 1))
        w.destroy(e)
        results = list(w.query(Pos))
        assert len(results) == 0

    def test_get_component_dead_entity(self):
        w = World()
        e = w.spawn(Pos())
        w.destroy(e)
        assert w.get_component(e, Pos) is None
