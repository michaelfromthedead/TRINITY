"""Tests for CommandBuffer: deferred spawn, despawn, insert, remove, flush."""
import pytest
from engine.core.ecs.command_buffer import CommandBuffer
from engine.core.ecs.world import World


class Pos:
    __slots__ = ("x",)
    def __init__(self, x=0): self.x = x


class Vel:
    __slots__ = ("v",)
    def __init__(self, v=0): self.v = v


class TestCommandBuffer:
    def test_spawn(self):
        w = World()
        cb = CommandBuffer()
        cmd = cb.spawn(Pos(5))
        assert len(cb) == 1
        cb.flush(w)
        assert cmd.entity is not None
        assert w.is_alive(cmd.entity)
        assert w.get_component(cmd.entity, Pos).x == 5

    def test_despawn(self):
        w = World()
        e = w.spawn(Pos())
        cb = CommandBuffer()
        cb.despawn(e)
        cb.flush(w)
        assert not w.is_alive(e)

    def test_insert(self):
        w = World()
        e = w.spawn(Pos())
        cb = CommandBuffer()
        cb.insert(e, Vel(10))
        cb.flush(w)
        assert w.get_component(e, Vel).v == 10

    def test_remove(self):
        w = World()
        e = w.spawn(Pos(), Vel())
        cb = CommandBuffer()
        cb.remove(e, Vel)
        cb.flush(w)
        assert not w.has_component(e, Vel)

    def test_flush_clears(self):
        w = World()
        cb = CommandBuffer()
        cb.spawn(Pos())
        cb.flush(w)
        assert len(cb) == 0
        cb.flush(w)  # no-op
