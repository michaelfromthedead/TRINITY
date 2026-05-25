"""Tests for parent/child relationships and cascade delete."""
import pytest
from engine.core.ecs.world import World
from engine.core.ecs.hierarchy import (
    set_parent, remove_parent, get_parent, get_children, destroy_hierarchy,
)


class Pos:
    __slots__ = ("x",)
    def __init__(self, x=0): self.x = x


class TestHierarchy:
    def test_set_and_get_parent(self):
        w = World()
        parent = w.spawn(Pos(0))
        child = w.spawn(Pos(1))
        set_parent(w, child, parent)
        assert get_parent(w, child) == parent

    def test_get_children(self):
        w = World()
        parent = w.spawn(Pos())
        c1 = w.spawn(Pos())
        c2 = w.spawn(Pos())
        set_parent(w, c1, parent)
        set_parent(w, c2, parent)
        children = get_children(w, parent)
        assert set(children) == {c1, c2}

    def test_remove_parent(self):
        w = World()
        parent = w.spawn(Pos())
        child = w.spawn(Pos())
        set_parent(w, child, parent)
        remove_parent(w, child)
        assert get_parent(w, child) is None
        assert get_children(w, parent) == []

    def test_reparent(self):
        w = World()
        p1 = w.spawn(Pos())
        p2 = w.spawn(Pos())
        child = w.spawn(Pos())
        set_parent(w, child, p1)
        set_parent(w, child, p2)
        assert get_parent(w, child) == p2
        assert get_children(w, p1) == []
        assert get_children(w, p2) == [child]

    def test_cascade_delete(self):
        w = World()
        root = w.spawn(Pos())
        c1 = w.spawn(Pos())
        c2 = w.spawn(Pos())
        gc = w.spawn(Pos())  # grandchild
        set_parent(w, c1, root)
        set_parent(w, c2, root)
        set_parent(w, gc, c1)
        destroy_hierarchy(w, root)
        assert not w.is_alive(root)
        assert not w.is_alive(c1)
        assert not w.is_alive(c2)
        assert not w.is_alive(gc)
