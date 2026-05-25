"""Tests for SystemGraph."""

import pytest

from engine.core.scheduler.graph import CycleDetectedError, SystemGraph


class TestTopologicalSort:
    def test_valid_ordering(self):
        g = SystemGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_node(2)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        result = g.topological_sort()
        assert result.index(0) < result.index(1) < result.index(2)

    def test_diamond(self):
        g = SystemGraph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        result = g.topological_sort()
        assert result.index(0) < result.index(1)
        assert result.index(0) < result.index(2)
        assert result.index(1) < result.index(3)
        assert result.index(2) < result.index(3)

    def test_empty_graph(self):
        g = SystemGraph()
        assert g.topological_sort() == []

    def test_single_node(self):
        g = SystemGraph()
        g.add_node(42)
        assert g.topological_sort() == [42]


class TestCycleDetection:
    def test_simple_cycle(self):
        g = SystemGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        with pytest.raises(CycleDetectedError):
            g.detect_cycles()

    def test_three_node_cycle(self):
        g = SystemGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        with pytest.raises(CycleDetectedError):
            g.topological_sort()


class TestParallelGroups:
    def test_independent_nodes_single_group(self):
        g = SystemGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_node(2)
        groups = g.get_parallel_groups()
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]

    def test_chain_separate_groups(self):
        g = SystemGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        groups = g.get_parallel_groups()
        assert groups == [[0], [1], [2]]

    def test_diamond_groups(self):
        g = SystemGraph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        groups = g.get_parallel_groups()
        assert groups[0] == [0]
        assert sorted(groups[1]) == [1, 2]
        assert groups[2] == [3]

    def test_empty_graph(self):
        g = SystemGraph()
        assert g.get_parallel_groups() == []
