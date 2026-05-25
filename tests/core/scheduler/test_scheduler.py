"""Tests for SystemScheduler."""

import pytest

from engine.core.scheduler import Phase, SystemScheduler, SystemAccess


class TestRegisterAndRun:
    def test_register_callable_and_run(self):
        scheduler = SystemScheduler(parallel_dispatch=False)
        calls = []

        def my_system(world, dt):
            calls.append(("my_system", world, dt))

        scheduler.register_system(my_system)
        scheduler.run("world", 0.016)
        assert len(calls) == 1
        assert calls[0] == ("my_system", "world", 0.016)

    def test_register_object_with_update(self):
        scheduler = SystemScheduler(parallel_dispatch=False)
        calls = []

        class MySystem:
            def update(self, world, dt):
                calls.append(("obj", world, dt))

        scheduler.register_system(MySystem())
        scheduler.run("w", 0.01)
        assert calls == [("obj", "w", 0.01)]

    def test_systems_receive_world_and_delta(self):
        scheduler = SystemScheduler(parallel_dispatch=False)
        received = {}

        def sys(world, dt):
            received["world"] = world
            received["dt"] = dt

        scheduler.register_system(sys)
        scheduler.run({"entities": []}, 1.0 / 60)
        assert received["world"] == {"entities": []}
        assert abs(received["dt"] - 1.0 / 60) < 1e-9


class TestPhaseOrdering:
    def test_pre_update_before_update_before_post_update(self):
        scheduler = SystemScheduler(parallel_dispatch=False)
        order = []

        def pre(w, dt):
            order.append("pre")

        def upd(w, dt):
            order.append("upd")

        def post(w, dt):
            order.append("post")

        scheduler.register_system(pre, phase=Phase.PRE_UPDATE)
        scheduler.register_system(upd, phase=Phase.UPDATE)
        scheduler.register_system(post, phase=Phase.POST_UPDATE)
        scheduler.run(None, 0.016)
        assert order == ["pre", "upd", "post"]

    def test_render_phases_after_update_phases(self):
        scheduler = SystemScheduler(parallel_dispatch=False)
        order = []

        scheduler.register_system(lambda w, dt: order.append("update"), phase=Phase.UPDATE)
        scheduler.register_system(lambda w, dt: order.append("render"), phase=Phase.RENDER)
        scheduler.run(None, 0.016)
        assert order == ["update", "render"]


class TestDependencyOrdering:
    def test_a_before_b(self):
        scheduler = SystemScheduler(parallel_dispatch=False)
        order = []

        b_id = scheduler.register_system(lambda w, dt: order.append("B"))
        a_id = scheduler.register_system(lambda w, dt: order.append("A"))
        scheduler.add_dependency(a_id, b_id)  # A before B
        scheduler.run(None, 0.016)
        assert order.index("A") < order.index("B")


class TestRunIf:
    def test_conditional_skip(self):
        scheduler = SystemScheduler(parallel_dispatch=False)
        calls = []
        enabled = False

        scheduler.register_system(
            lambda w, dt: calls.append("skipped"),
            run_if=lambda: enabled,
        )
        scheduler.run(None, 0.016)
        assert calls == []

    def test_conditional_run(self):
        scheduler = SystemScheduler(parallel_dispatch=False)
        calls = []

        scheduler.register_system(
            lambda w, dt: calls.append("ran"),
            run_if=lambda: True,
        )
        scheduler.run(None, 0.016)
        assert calls == ["ran"]
