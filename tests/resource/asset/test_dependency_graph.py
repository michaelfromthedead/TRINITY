"""Tests for DependencyGraph."""
import pytest

from engine.resource.asset.dependency_graph import DependencyGraph


class TestDependencyGraph:
    def test_add_and_get_dependents(self) -> None:
        g = DependencyGraph()
        g.add_dependency(1, 2)
        assert g.get_dependents(2) == {1}

    def test_load_order_simple(self) -> None:
        g = DependencyGraph()
        g.add_dependency(2, 1)  # 2 depends on 1
        order = g.get_load_order([2])
        assert order.index(1) < order.index(2)

    def test_load_order_chain(self) -> None:
        g = DependencyGraph()
        g.add_dependency(3, 2)
        g.add_dependency(2, 1)
        order = g.get_load_order([3])
        assert order == [1, 2, 3]

    def test_load_order_diamond(self) -> None:
        g = DependencyGraph()
        g.add_dependency(4, 2)
        g.add_dependency(4, 3)
        g.add_dependency(2, 1)
        g.add_dependency(3, 1)
        order = g.get_load_order([4])
        assert order.index(1) < order.index(2)
        assert order.index(1) < order.index(3)
        assert order.index(2) < order.index(4)
        assert order.index(3) < order.index(4)

    def test_cycle_detection_self(self) -> None:
        g = DependencyGraph()
        with pytest.raises(ValueError, match="Self-dependency"):
            g.add_dependency(1, 1)

    def test_cycle_detection_indirect(self) -> None:
        g = DependencyGraph()
        g.add_dependency(1, 2)
        g.add_dependency(2, 3)
        with pytest.raises(ValueError, match="[Cc]ycle"):
            g.add_dependency(3, 1)

    def test_remove_clears_edges(self) -> None:
        g = DependencyGraph()
        g.add_dependency(2, 1)
        g.remove(2)
        assert g.get_dependents(1) == set()

    def test_get_dependents_empty(self) -> None:
        g = DependencyGraph()
        assert g.get_dependents(999) == set()

    def test_load_order_independent_nodes(self) -> None:
        g = DependencyGraph()
        # No dependencies between them
        g.add_dependency(1, 10)
        g.add_dependency(2, 20)
        order = g.get_load_order([1, 2])
        assert set(order) == {1, 2, 10, 20}
        assert order.index(10) < order.index(1)
        assert order.index(20) < order.index(2)
