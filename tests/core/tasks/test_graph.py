"""Tests for engine.core.tasks.graph — TaskGraph and TaskGraphBuilder."""

import threading
import time

import pytest

from engine.core.tasks.graph import TaskGraph, TaskGraphBuilder, TaskState
from engine.core.tasks.scheduler import TaskScheduler


@pytest.fixture
def executor():
    s = TaskScheduler(worker_count=2)
    yield s
    s.shutdown()


class TestTaskGraph:
    def test_add_tasks(self):
        g = TaskGraph()
        a = g.add_task("a", lambda: 1)
        b = g.add_task("b", lambda: 2)
        assert a != b
        assert len(g.nodes) == 2

    def test_add_dependency(self):
        g = TaskGraph()
        a = g.add_task("a", lambda: 1)
        b = g.add_task("b", lambda: 2)
        g.add_dependency(b, a)  # b depends on a
        assert a in g.nodes[b].dependencies

    def test_compile_topological_order(self):
        g = TaskGraph()
        a = g.add_task("a", lambda: None)
        b = g.add_task("b", lambda: None)
        c = g.add_task("c", lambda: None)
        g.add_dependency(b, a)  # b after a
        g.add_dependency(c, b)  # c after b
        order = g.compile()
        assert order.index(a) < order.index(b) < order.index(c)

    def test_compile_detects_cycle(self):
        g = TaskGraph()
        a = g.add_task("a", lambda: None)
        b = g.add_task("b", lambda: None)
        g.add_dependency(a, b)
        g.add_dependency(b, a)
        with pytest.raises(ValueError, match="cycle"):
            g.compile()

    def test_empty_graph(self):
        g = TaskGraph()
        order = g.compile()
        assert order == []

    def test_fence_node(self):
        g = TaskGraph()
        a = g.add_task("a", lambda: None)
        fence = g.add_fence("sync")
        b = g.add_task("b", lambda: None)
        g.add_dependency(fence, a)
        g.add_dependency(b, fence)
        order = g.compile()
        assert order.index(a) < order.index(fence) < order.index(b)


class TestTaskGraphExecution:
    def test_execute_runs_all(self, executor):
        results = []
        lock = threading.Lock()

        def append(val):
            with lock:
                results.append(val)

        g = TaskGraph()
        g.add_task("a", append, "a")
        g.add_task("b", append, "b")
        g.compile()
        g.execute(executor)
        assert g.is_complete()
        assert sorted(results) == ["a", "b"]

    def test_execute_respects_dependencies(self, executor):
        order = []
        lock = threading.Lock()

        def task(name):
            time.sleep(0.02)
            with lock:
                order.append(name)

        g = TaskGraph()
        a = g.add_task("a", task, "a")
        b = g.add_task("b", task, "b")
        g.add_dependency(b, a)
        g.compile()
        g.execute(executor)
        assert order.index("a") < order.index("b")

    def test_execute_fence(self, executor):
        order = []
        lock = threading.Lock()

        def task(name):
            with lock:
                order.append(name)

        g = TaskGraph()
        a = g.add_task("a", task, "a")
        fence = g.add_fence("barrier")
        b = g.add_task("b", task, "b")
        g.add_dependency(fence, a)
        g.add_dependency(b, fence)
        g.compile()
        g.execute(executor)
        assert order.index("a") < order.index("b")

    def test_node_states_after_execution(self, executor):
        g = TaskGraph()
        g.add_task("ok", lambda: 42)
        g.compile()
        g.execute(executor)
        node = list(g.nodes.values())[0]
        assert node.state == TaskState.COMPLETE
        assert node.result == 42


class TestTaskGraphBuilder:
    def test_fluent_api(self, executor):
        order = []
        lock = threading.Lock()

        def t(name):
            time.sleep(0.01)
            with lock:
                order.append(name)

        b = TaskGraphBuilder()
        t1 = b.task("load_mesh", t, "mesh")
        t2 = b.task("load_tex", t, "tex")
        t3 = b.task("create_mat", t, "mat")
        t3.depends_on(t1, t2)
        t4 = b.task("finalize", t, "fin")
        t4.depends_on(t3)
        graph = b.build()

        graph.execute(executor)
        assert order.index("mat") > order.index("mesh")
        assert order.index("mat") > order.index("tex")
        assert order.index("fin") > order.index("mat")

    def test_depends_on_by_name(self, executor):
        b = TaskGraphBuilder()
        b.task("first", lambda: None)
        ref = b.task("second", lambda: None)
        ref.depends_on("first")
        graph = b.build()
        order = graph._sorted
        name_order = [graph.nodes[nid].name for nid in order]
        assert name_order.index("first") < name_order.index("second")

    def test_builder_fence(self, executor):
        b = TaskGraphBuilder()
        b.task("a", lambda: None)
        f = b.fence("sync")
        f.depends_on("a")
        t = b.task("b", lambda: None)
        t.depends_on(f)
        graph = b.build()
        graph.execute(executor)
        assert graph.is_complete()
