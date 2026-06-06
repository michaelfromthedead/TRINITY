"""Tests for asset dependency graph with rebuild cascade.

Comprehensive test suite covering:
- DependencyGraph (legacy API backward compatibility)
- DependencyEdge with types (import, reference, embed)
- AssetDependencyGraph (full-featured graph)
- RebuildPlanner (incremental rebuilds)
- Cycle detection and reporting
- Topological sort for rebuild order
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import MagicMock, call, patch

import pytest

from engine.resource.asset.content_hash import ContentHash
from engine.resource.asset.dependency_graph import (
    AssetDependencyGraph,
    AssetNode,
    CycleError,
    DependencyEdge,
    DependencyGraph,
    DependencyType,
    RebuildPlan,
    RebuildPlanner,
    RebuildResult,
    RebuildStats,
)


# =============================================================================
# Legacy DependencyGraph Tests (Backward Compatibility)
# =============================================================================


class TestDependencyGraph:
    """Tests for the legacy DependencyGraph API."""

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


# =============================================================================
# DependencyEdge Tests
# =============================================================================


class TestDependencyEdge:
    """Tests for DependencyEdge with typed relationships."""

    def test_create_import_edge(self) -> None:
        edge = DependencyEdge.import_edge(1, 2)
        assert edge.source == 1
        assert edge.target == 2
        assert edge.dep_type == DependencyType.IMPORT

    def test_create_reference_edge(self) -> None:
        edge = DependencyEdge.reference_edge(1, 2)
        assert edge.dep_type == DependencyType.REFERENCE

    def test_create_embed_edge(self) -> None:
        edge = DependencyEdge.embed_edge(1, 2)
        assert edge.dep_type == DependencyType.EMBED

    def test_edge_with_metadata(self) -> None:
        edge = DependencyEdge.import_edge(1, 2, version="1.0", optional="false")
        assert edge.get_metadata("version") == "1.0"
        assert edge.get_metadata("optional") == "false"
        assert edge.get_metadata("missing") is None

    def test_edge_with_additional_metadata(self) -> None:
        edge = DependencyEdge.import_edge(1, 2, version="1.0")
        edge2 = edge.with_metadata(author="test")
        assert edge2.get_metadata("version") == "1.0"
        assert edge2.get_metadata("author") == "test"
        # Original unchanged
        assert edge.get_metadata("author") is None

    def test_edge_self_dependency_raises(self) -> None:
        with pytest.raises(ValueError, match="Self-dependency"):
            DependencyEdge(1, 1, DependencyType.IMPORT)

    def test_edge_is_frozen(self) -> None:
        edge = DependencyEdge.import_edge(1, 2)
        with pytest.raises(AttributeError):
            edge.source = 3  # type: ignore

    def test_edge_repr(self) -> None:
        edge = DependencyEdge.import_edge(1, 2)
        assert "1" in repr(edge)
        assert "2" in repr(edge)
        assert "import" in repr(edge)

    def test_dependency_type_str(self) -> None:
        assert str(DependencyType.IMPORT) == "import"
        assert str(DependencyType.REFERENCE) == "reference"
        assert str(DependencyType.EMBED) == "embed"


# =============================================================================
# AssetNode Tests
# =============================================================================


class TestAssetNode:
    """Tests for AssetNode content tracking."""

    def test_create_node(self) -> None:
        node = AssetNode(asset_id=1)
        assert node.asset_id == 1
        assert node.content_hash is None
        assert node.rebuild_count == 0

    def test_node_with_hash(self) -> None:
        h = ContentHash.from_content(b"test")
        node = AssetNode(asset_id=1, content_hash=h)
        assert node.content_hash == h

    def test_needs_rebuild_no_hash(self) -> None:
        node = AssetNode(asset_id=1)
        assert node.needs_rebuild(None) is True

    def test_needs_rebuild_same_hash(self) -> None:
        h = ContentHash.from_content(b"test")
        node = AssetNode(asset_id=1, content_hash=h)
        assert node.needs_rebuild(h) is False

    def test_needs_rebuild_different_hash(self) -> None:
        h1 = ContentHash.from_content(b"test1")
        h2 = ContentHash.from_content(b"test2")
        node = AssetNode(asset_id=1, content_hash=h1)
        assert node.needs_rebuild(h2) is True

    def test_mark_rebuilt(self) -> None:
        h = ContentHash.from_content(b"new")
        node = AssetNode(asset_id=1)
        node.mark_rebuilt(h)
        assert node.content_hash == h
        assert node.rebuild_count == 1
        assert node.last_rebuild_time > 0

    def test_node_metadata(self) -> None:
        node = AssetNode(asset_id=1, path="/test/path.txt", metadata={"key": "value"})
        assert node.path == "/test/path.txt"
        assert node.metadata["key"] == "value"


# =============================================================================
# AssetDependencyGraph Tests
# =============================================================================


class TestAssetDependencyGraph:
    """Tests for the full-featured AssetDependencyGraph."""

    def test_add_node(self) -> None:
        graph = AssetDependencyGraph()
        node = graph.add_node(1, path="/test.txt")
        assert node.asset_id == 1
        assert node.path == "/test.txt"
        assert graph.has_node(1)

    def test_add_node_with_hash(self) -> None:
        graph = AssetDependencyGraph()
        h = ContentHash.from_content(b"data")
        node = graph.add_node(1, content_hash=h)
        assert node.content_hash == h

    def test_update_existing_node(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_node(1, path="/old.txt")
        graph.add_node(1, path="/new.txt")
        node = graph.get_node(1)
        assert node is not None
        assert node.path == "/new.txt"

    def test_get_nonexistent_node(self) -> None:
        graph = AssetDependencyGraph()
        assert graph.get_node(999) is None

    def test_remove_node(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_node(1)
        assert graph.remove_node(1) is True
        assert graph.has_node(1) is False
        assert graph.remove_node(1) is False  # Already removed

    def test_remove_node_clears_edges(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2)
        graph.add_dependency(3, 1)
        graph.remove_node(1)
        assert not graph.has_edge(1, 2)
        assert not graph.has_edge(3, 1)

    def test_node_count(self) -> None:
        graph = AssetDependencyGraph()
        assert graph.node_count() == 0
        graph.add_node(1)
        graph.add_node(2)
        assert graph.node_count() == 2

    def test_add_edge(self) -> None:
        graph = AssetDependencyGraph()
        edge = DependencyEdge.import_edge(1, 2)
        graph.add_edge(edge)
        assert graph.has_edge(1, 2)
        assert graph.has_node(1)
        assert graph.has_node(2)

    def test_add_dependency_convenience(self) -> None:
        graph = AssetDependencyGraph()
        edge = graph.add_dependency(1, 2, DependencyType.EMBED)
        assert edge.dep_type == DependencyType.EMBED
        assert graph.has_edge(1, 2)

    def test_get_edge(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2, DependencyType.IMPORT, file="test.py")
        edge = graph.get_edge(1, 2)
        assert edge is not None
        assert edge.get_metadata("file") == "test.py"

    def test_remove_edge(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2)
        assert graph.remove_edge(1, 2) is True
        assert not graph.has_edge(1, 2)
        assert graph.remove_edge(1, 2) is False

    def test_edge_count(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2)
        graph.add_dependency(2, 3)
        assert graph.edge_count() == 2

    def test_get_dependencies(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2)
        graph.add_dependency(1, 3)
        deps = graph.get_dependencies(1)
        assert deps == {2, 3}

    def test_get_dependents(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 3)
        graph.add_dependency(2, 3)
        dependents = graph.get_dependents(3)
        assert dependents == {1, 2}

    def test_get_transitive_dependencies(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2)
        graph.add_dependency(2, 3)
        graph.add_dependency(3, 4)
        trans = graph.get_transitive_dependencies(1)
        assert trans == {2, 3, 4}

    def test_get_transitive_dependents(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 4)
        graph.add_dependency(2, 1)
        graph.add_dependency(3, 2)
        trans = graph.get_transitive_dependents(4)
        assert trans == {1, 2, 3}

    def test_edges_from(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2, DependencyType.IMPORT)
        graph.add_dependency(1, 3, DependencyType.REFERENCE)
        edges = graph.get_edges_from(1)
        assert len(edges) == 2
        targets = {e.target for e in edges}
        assert targets == {2, 3}

    def test_edges_to(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 3)
        graph.add_dependency(2, 3)
        edges = graph.get_edges_to(3)
        assert len(edges) == 2

    def test_edges_of_type(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2, DependencyType.IMPORT)
        graph.add_dependency(3, 4, DependencyType.EMBED)
        graph.add_dependency(5, 6, DependencyType.IMPORT)
        imports = list(graph.edges_of_type(DependencyType.IMPORT))
        assert len(imports) == 2

    def test_clear(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2)
        graph.add_dependency(3, 4)
        graph.clear()
        assert graph.node_count() == 0
        assert graph.edge_count() == 0


# =============================================================================
# Cycle Detection Tests
# =============================================================================


class TestCycleDetection:
    """Tests for cycle detection and error reporting."""

    def test_self_dependency_raises_value_error(self) -> None:
        # DependencyEdge validates self-dependency before reaching graph
        with pytest.raises(ValueError, match="Self-dependency"):
            DependencyEdge(1, 1, DependencyType.IMPORT)

    def test_self_dependency_via_add_raises_cycle_error(self) -> None:
        # Graph catches self-dependency at add_dependency level
        graph = AssetDependencyGraph()
        with pytest.raises(CycleError) as exc_info:
            graph.add_dependency(1, 1)
        assert 1 in exc_info.value.cycle_path

    def test_direct_cycle_raises(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2)
        with pytest.raises(CycleError):
            graph.add_dependency(2, 1)

    def test_indirect_cycle_raises(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2)
        graph.add_dependency(2, 3)
        graph.add_dependency(3, 4)
        with pytest.raises(CycleError) as exc_info:
            graph.add_dependency(4, 1)
        assert len(exc_info.value.cycle_path) > 0

    def test_cycle_error_path(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2)
        graph.add_dependency(2, 3)
        try:
            graph.add_dependency(3, 1)
            pytest.fail("Expected CycleError")
        except CycleError as e:
            # Cycle path should contain the nodes involved
            assert 1 in e.cycle_path or 2 in e.cycle_path or 3 in e.cycle_path

    def test_detect_cycles_empty_graph(self) -> None:
        graph = AssetDependencyGraph()
        assert graph.detect_cycles() == []

    def test_detect_cycles_acyclic(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2)
        graph.add_dependency(2, 3)
        assert graph.detect_cycles() == []

    def test_is_acyclic(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2)
        graph.add_dependency(2, 3)
        assert graph.is_acyclic() is True

    def test_cycle_error_str(self) -> None:
        error = CycleError("Test cycle", cycle_path=[1, 2, 3, 1])
        s = str(error)
        assert "1" in s
        assert "2" in s
        assert "3" in s


# =============================================================================
# Topological Sort Tests
# =============================================================================


class TestTopologicalSort:
    """Tests for topological sorting and rebuild order."""

    def test_topological_sort_simple(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(2, 1)  # 2 depends on 1
        order = graph.topological_sort()
        assert order.index(1) < order.index(2)

    def test_topological_sort_chain(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(3, 2)
        graph.add_dependency(2, 1)
        order = graph.topological_sort()
        assert order == [1, 2, 3]

    def test_topological_sort_diamond(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(4, 2)
        graph.add_dependency(4, 3)
        graph.add_dependency(2, 1)
        graph.add_dependency(3, 1)
        order = graph.topological_sort()
        assert order.index(1) < order.index(2)
        assert order.index(1) < order.index(3)
        assert order.index(2) < order.index(4)
        assert order.index(3) < order.index(4)

    def test_topological_sort_subset(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 10)
        graph.add_dependency(2, 20)
        graph.add_dependency(3, 1)
        order = graph.topological_sort_subset([3])
        assert 10 in order
        assert 1 in order
        assert 3 in order
        assert 2 not in order
        assert 20 not in order

    def test_topological_sort_empty(self) -> None:
        graph = AssetDependencyGraph()
        assert graph.topological_sort() == []

    def test_topological_sort_single_node(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_node(1)
        assert graph.topological_sort() == [1]


# =============================================================================
# Rebuild Order Tests
# =============================================================================


class TestRebuildOrder:
    """Tests for rebuild cascade ordering."""

    def test_rebuild_order_single_change(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(2, 1)  # 2 depends on 1
        graph.add_dependency(3, 2)  # 3 depends on 2
        order = graph.get_rebuild_order([1])
        # When 1 changes, 2 and 3 need rebuild
        assert 1 in order
        assert 2 in order
        assert 3 in order
        assert order.index(1) < order.index(2)
        assert order.index(2) < order.index(3)

    def test_rebuild_order_multiple_changes(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(3, 1)
        graph.add_dependency(4, 2)
        graph.add_dependency(5, 3)
        graph.add_dependency(5, 4)
        order = graph.get_rebuild_order([1, 2])
        assert set(order) == {1, 2, 3, 4, 5}

    def test_rebuild_order_isolated_node(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_node(1)
        graph.add_dependency(3, 2)
        order = graph.get_rebuild_order([1])
        assert order == [1]

    def test_rebuild_order_respects_dependencies(self) -> None:
        graph = AssetDependencyGraph()
        # A depends on B and C, B and C depend on D
        graph.add_dependency(1, 2)
        graph.add_dependency(1, 3)
        graph.add_dependency(2, 4)
        graph.add_dependency(3, 4)
        order = graph.get_rebuild_order([4])
        # D must come before B and C, which must come before A
        assert order.index(4) < order.index(2)
        assert order.index(4) < order.index(3)
        assert order.index(2) < order.index(1)
        assert order.index(3) < order.index(1)


# =============================================================================
# Content Hash / Incremental Rebuild Tests
# =============================================================================


class TestIncrementalRebuild:
    """Tests for incremental rebuild with content hashing."""

    def test_update_content_hash(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_node(1)
        h = ContentHash.from_content(b"test")
        assert graph.update_content_hash(1, h) is True
        assert graph.get_content_hash(1) == h

    def test_update_content_hash_unchanged(self) -> None:
        graph = AssetDependencyGraph()
        h = ContentHash.from_content(b"test")
        graph.add_node(1, content_hash=h)
        assert graph.update_content_hash(1, h) is False

    def test_update_content_hash_nonexistent(self) -> None:
        graph = AssetDependencyGraph()
        h = ContentHash.from_content(b"test")
        assert graph.update_content_hash(999, h) is False

    def test_needs_rebuild_no_existing_hash(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_node(1)
        h = ContentHash.from_content(b"test")
        assert graph.needs_rebuild(1, h) is True

    def test_needs_rebuild_hash_changed(self) -> None:
        graph = AssetDependencyGraph()
        h1 = ContentHash.from_content(b"old")
        h2 = ContentHash.from_content(b"new")
        graph.add_node(1, content_hash=h1)
        assert graph.needs_rebuild(1, h2) is True

    def test_needs_rebuild_hash_unchanged(self) -> None:
        graph = AssetDependencyGraph()
        h = ContentHash.from_content(b"same")
        graph.add_node(1, content_hash=h)
        assert graph.needs_rebuild(1, h) is False

    def test_mark_rebuilt(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_node(1)
        h = ContentHash.from_content(b"test")
        graph.mark_rebuilt(1, h)
        node = graph.get_node(1)
        assert node is not None
        assert node.content_hash == h
        assert node.rebuild_count == 1


# =============================================================================
# RebuildPlanner Tests
# =============================================================================


class TestRebuildPlanner:
    """Tests for RebuildPlanner planning and execution."""

    def test_plan_rebuild_simple(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(2, 1)
        graph.add_dependency(3, 2)
        planner = RebuildPlanner(graph)
        plan = planner.plan_rebuild([1])
        assert set(plan.assets_to_rebuild) == {1, 2, 3}
        assert plan.changed_sources == {1}
        assert plan.affected_dependents == {2, 3}

    def test_plan_rebuild_with_skipping(self) -> None:
        # Skipping applies to source assets (in changed_assets) that have
        # unchanged content hash. Dependents always rebuild if their
        # dependencies changed.
        graph = AssetDependencyGraph()
        h_unchanged = ContentHash.from_content(b"unchanged")
        graph.add_node(1)  # Will change
        graph.add_node(2)  # Depends on 1, must rebuild
        graph.add_node(3, content_hash=h_unchanged)  # Same hash, skip
        graph.add_dependency(2, 1)

        def hash_provider(aid: int) -> ContentHash | None:
            if aid == 3:
                return h_unchanged  # Same hash = source unchanged
            return None  # Changed or unknown

        planner = RebuildPlanner(graph, hash_provider=hash_provider)
        # Request rebuild for 1 and 3
        plan = planner.plan_rebuild([1, 3], check_unchanged=True)
        # Asset 1 changed (no hash), 2 depends on 1 so must rebuild
        assert 1 in plan.assets_to_rebuild
        assert 2 in plan.assets_to_rebuild
        # Asset 3 has unchanged source hash, so skip
        assert 3 not in plan.assets_to_rebuild
        assert 3 in plan.skipped_unchanged

    def test_plan_full_rebuild(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(2, 1)
        graph.add_dependency(3, 1)
        planner = RebuildPlanner(graph)
        plan = planner.plan_full_rebuild()
        assert set(plan.assets_to_rebuild) == {1, 2, 3}

    def test_execute_success(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(2, 1)
        rebuilt: List[int] = []

        def rebuild_callback(aid: int, node: AssetNode | None) -> bool:
            rebuilt.append(aid)
            return True

        planner = RebuildPlanner(graph, rebuild_callback=rebuild_callback)
        plan = planner.plan_rebuild([1])
        result = planner.execute(plan)

        assert result.success
        assert set(result.rebuilt_assets) == {1, 2}
        assert result.stats.rebuilt == 2
        assert result.stats.failed == 0

    def test_execute_with_failure(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(2, 1)

        def rebuild_callback(aid: int, node: AssetNode | None) -> bool:
            if aid == 2:
                raise RuntimeError("Build failed")
            return True

        planner = RebuildPlanner(graph, rebuild_callback=rebuild_callback)
        plan = planner.plan_rebuild([1])
        result = planner.execute(plan)

        assert not result.success
        assert 1 in result.rebuilt_assets
        assert len(result.failed_assets) == 1
        assert result.failed_assets[0][0] == 2

    def test_execute_stop_on_failure(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(2, 1)
        graph.add_dependency(3, 2)
        graph.add_dependency(4, 3)

        def rebuild_callback(aid: int, node: AssetNode | None) -> bool:
            if aid == 2:
                raise RuntimeError("Build failed")
            return True

        planner = RebuildPlanner(
            graph, rebuild_callback=rebuild_callback, stop_on_failure=True
        )
        plan = planner.plan_rebuild([1])
        result = planner.execute(plan)

        # Should stop after failure on 2
        assert 1 in result.rebuilt_assets
        assert 3 not in result.rebuilt_assets
        assert 4 not in result.rebuilt_assets

    def test_execute_incremental(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(2, 1)
        rebuilt: List[int] = []

        def rebuild_callback(aid: int, node: AssetNode | None) -> bool:
            rebuilt.append(aid)
            return True

        planner = RebuildPlanner(graph, rebuild_callback=rebuild_callback)
        result = planner.execute_incremental([1])

        assert result.success
        assert set(rebuilt) == {1, 2}

    def test_execute_no_callback_raises(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_node(1)
        planner = RebuildPlanner(graph)
        plan = planner.plan_rebuild([1])
        with pytest.raises(RuntimeError, match="No rebuild callback"):
            planner.execute(plan)

    def test_set_rebuild_callback(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_node(1)
        planner = RebuildPlanner(graph)
        planner.set_rebuild_callback(lambda aid, node: True)
        plan = planner.plan_rebuild([1])
        result = planner.execute(plan)
        assert result.success

    def test_rebuild_updates_content_hash(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_node(1)
        h = ContentHash.from_content(b"new_content")

        def hash_provider(aid: int) -> ContentHash:
            return h

        planner = RebuildPlanner(
            graph,
            rebuild_callback=lambda aid, node: True,
            hash_provider=hash_provider,
        )
        plan = planner.plan_rebuild([1])
        planner.execute(plan)

        assert graph.get_content_hash(1) == h


# =============================================================================
# Parallel Rebuild Tests
# =============================================================================


class TestParallelRebuild:
    """Tests for parallel rebuild execution."""

    def test_get_parallel_groups_simple(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(2, 1)
        graph.add_dependency(3, 1)  # 2 and 3 both depend on 1
        planner = RebuildPlanner(graph, rebuild_callback=lambda a, n: True)
        plan = planner.plan_rebuild([1])
        groups = planner.get_parallel_groups(plan)

        assert len(groups) == 2
        assert groups[0] == [1]  # 1 first (no dependencies)
        assert set(groups[1]) == {2, 3}  # 2 and 3 can be parallel

    def test_get_parallel_groups_chain(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(2, 1)
        graph.add_dependency(3, 2)
        graph.add_dependency(4, 3)
        planner = RebuildPlanner(graph, rebuild_callback=lambda a, n: True)
        plan = planner.plan_rebuild([1])
        groups = planner.get_parallel_groups(plan)

        # Chain = no parallelism
        assert len(groups) == 4
        for i, group in enumerate(groups):
            assert len(group) == 1

    def test_get_parallel_groups_diamond(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(2, 1)
        graph.add_dependency(3, 1)
        graph.add_dependency(4, 2)
        graph.add_dependency(4, 3)
        planner = RebuildPlanner(graph, rebuild_callback=lambda a, n: True)
        plan = planner.plan_rebuild([1])
        groups = planner.get_parallel_groups(plan)

        assert len(groups) == 3
        assert groups[0] == [1]
        assert set(groups[1]) == {2, 3}  # Parallel
        assert groups[2] == [4]

    def test_execute_parallel(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(2, 1)
        graph.add_dependency(3, 1)
        graph.add_dependency(4, 1)
        rebuild_order: List[int] = []
        lock = threading.Lock()

        def rebuild_callback(aid: int, node: AssetNode | None) -> bool:
            with lock:
                rebuild_order.append(aid)
            return True

        planner = RebuildPlanner(graph, rebuild_callback=rebuild_callback)
        plan = planner.plan_rebuild([1])
        result = planner.execute_parallel(plan, max_workers=4)

        assert result.success
        assert set(result.rebuilt_assets) == {1, 2, 3, 4}
        # 1 must be rebuilt before 2, 3, 4
        assert rebuild_order.index(1) < min(
            rebuild_order.index(2), rebuild_order.index(3), rebuild_order.index(4)
        )


# =============================================================================
# RebuildStats Tests
# =============================================================================


class TestRebuildStats:
    """Tests for rebuild statistics."""

    def test_duration_ms(self) -> None:
        stats = RebuildStats(start_time=1.0, end_time=1.5)
        assert stats.duration_ms == 500.0

    def test_success_rate_all_rebuilt(self) -> None:
        stats = RebuildStats(total_assets=10, rebuilt=10, skipped=0, failed=0)
        assert stats.success_rate == 100.0

    def test_success_rate_with_skips(self) -> None:
        stats = RebuildStats(total_assets=10, rebuilt=6, skipped=4, failed=0)
        assert stats.success_rate == 100.0

    def test_success_rate_with_failures(self) -> None:
        stats = RebuildStats(total_assets=10, rebuilt=7, skipped=1, failed=2)
        assert stats.success_rate == 80.0

    def test_success_rate_empty(self) -> None:
        stats = RebuildStats(total_assets=0)
        assert stats.success_rate == 100.0


# =============================================================================
# RebuildPlan Tests
# =============================================================================


class TestRebuildPlan:
    """Tests for RebuildPlan data structure."""

    def test_total_affected(self) -> None:
        plan = RebuildPlan(
            assets_to_rebuild=[1, 2, 3],
            changed_sources={1},
            affected_dependents={2, 3},
            skipped_unchanged=set(),
        )
        assert plan.total_affected == 3

    def test_repr(self) -> None:
        plan = RebuildPlan(
            assets_to_rebuild=[1, 2],
            changed_sources={1},
            affected_dependents={2},
            skipped_unchanged={3},
        )
        s = repr(plan)
        assert "rebuild=2" in s
        assert "sources=1" in s
        assert "skipped=1" in s


# =============================================================================
# RebuildResult Tests
# =============================================================================


class TestRebuildResult:
    """Tests for RebuildResult data structure."""

    def test_success_true(self) -> None:
        result = RebuildResult(
            stats=RebuildStats(),
            rebuilt_assets=[1, 2],
            skipped_assets=[3],
            failed_assets=[],
        )
        assert result.success is True

    def test_success_false(self) -> None:
        result = RebuildResult(
            stats=RebuildStats(),
            rebuilt_assets=[1],
            skipped_assets=[],
            failed_assets=[(2, "error")],
        )
        assert result.success is False


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_add_nodes(self) -> None:
        graph = AssetDependencyGraph()
        threads: List[threading.Thread] = []

        def add_nodes(start: int) -> None:
            for i in range(start, start + 100):
                graph.add_node(i)

        for i in range(5):
            t = threading.Thread(target=add_nodes, args=(i * 100,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert graph.node_count() == 500

    def test_concurrent_add_edges(self) -> None:
        graph = AssetDependencyGraph()
        # Pre-add nodes
        for i in range(1, 101):
            graph.add_node(i)

        threads: List[threading.Thread] = []

        def add_edges(thread_id: int) -> None:
            for i in range(1, 20):
                try:
                    # Each thread adds edges to its own "target" to avoid cycles
                    graph.add_dependency(i * 5 + thread_id, 100)
                except (CycleError, ValueError):
                    pass  # May conflict

        for i in range(5):
            t = threading.Thread(target=add_edges, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have some edges
        assert graph.edge_count() > 0


# =============================================================================
# Subgraph Tests
# =============================================================================


class TestSubgraph:
    """Tests for subgraph extraction."""

    def test_subgraph_basic(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2)
        graph.add_dependency(3, 4)
        sub = graph.subgraph([1, 2])
        assert sub.node_count() == 2
        assert sub.has_edge(1, 2)
        assert not sub.has_node(3)
        assert not sub.has_node(4)

    def test_subgraph_preserves_node_data(self) -> None:
        graph = AssetDependencyGraph()
        h = ContentHash.from_content(b"test")
        graph.add_node(1, content_hash=h, path="/test.txt")
        sub = graph.subgraph([1])
        node = sub.get_node(1)
        assert node is not None
        assert node.content_hash == h
        assert node.path == "/test.txt"


# =============================================================================
# Graph Statistics Tests
# =============================================================================


class TestGraphStats:
    """Tests for graph statistics."""

    def test_get_stats(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2, DependencyType.IMPORT)
        graph.add_dependency(3, 4, DependencyType.EMBED)
        graph.add_dependency(5, 6, DependencyType.IMPORT)

        stats = graph.get_stats()
        assert stats["node_count"] == 6
        assert stats["edge_count"] == 3
        assert stats["edge_types"]["import"] == 2
        assert stats["edge_types"]["embed"] == 1
        assert stats["is_acyclic"] is True

    def test_len_and_contains(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_node(1)
        graph.add_node(2)
        assert len(graph) == 2
        assert 1 in graph
        assert 3 not in graph

    def test_repr(self) -> None:
        graph = AssetDependencyGraph()
        graph.add_dependency(1, 2)
        s = repr(graph)
        assert "nodes=2" in s
        assert "edges=1" in s
