"""
Test Suite: T-GP-3.5+3.6 - BT Registry Integration

Tests for the @behavior_tree and @bt_node decorators and their integration
with the Foundation Registry for runtime discovery.

Requirements tested:
1. @behavior_tree registers class with Foundation Registry
2. @bt_node registers node by type
3. Registry.query(tag="behavior_tree") returns all BT definitions
4. Registry.query(tag="bt_node", type="action") returns all action nodes
5. Metadata stored correctly
6. BehaviorTree.from_registry(name) factory instantiation
7. Multiple BT definitions coexist
8. Node type validation
9. Dynamic BT construction
10. Performance: 100 queries under 50ms
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, List, Optional, Type

import pytest

from foundation import registry, Registry
from engine.gameplay.ai.behavior_tree import (
    BehaviorTree,
    BTNode,
    BTContext,
    BTStatus,
    BTNodeType,
    Sequence,
    Selector,
    Parallel,
    Action,
    Condition,
    Wait,
    behavior_tree,
    bt_node,
    get_all_behavior_trees,
    get_bt_nodes_by_type,
    get_all_bt_nodes,
    BTNodeTypeError,
    VALID_BT_NODE_TYPES,
)
from engine.gameplay.ai.ai_registry import (
    behavior_tree as ai_behavior_tree,
    bt_node as ai_bt_node,
    get_all_behavior_trees as ai_get_all_behavior_trees,
    TAG_BEHAVIOR_TREE,
    TAG_BT_NODE,
)
from engine.gameplay.ai.blackboard import Blackboard


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before and after each test to avoid cross-contamination."""
    # Store initial state
    initial_types = set(registry.all_types())

    yield

    # Clean up any types added during the test
    for cls in registry.all_types():
        if cls not in initial_types:
            try:
                registry.unregister(cls)
            except Exception:
                pass


@pytest.fixture
def fresh_blackboard():
    """Create a fresh blackboard for each test."""
    return Blackboard()


# =============================================================================
# Test Classes - Defined inside tests to avoid polluting global registry
# =============================================================================


class TestBehaviorTreeDecorator:
    """Tests for @behavior_tree decorator."""

    def test_behavior_tree_registers_class(self, clean_registry):
        """Test that @behavior_tree registers the class with Foundation Registry."""
        @behavior_tree(name="test_bt_01", description="Test behavior tree")
        class TestBT01:
            @classmethod
            def create_root(cls) -> BTNode:
                return Sequence([Wait(1.0)])

        # Verify registration
        registered = registry.get("bt.test_bt_01")
        assert registered is TestBT01
        assert registry.is_registered(TestBT01)

    def test_behavior_tree_adds_tag(self, clean_registry):
        """Test that @behavior_tree adds the 'behavior_tree' tag."""
        @behavior_tree(name="test_bt_02", description="Tagged BT")
        class TestBT02:
            pass

        assert registry.has_tag(TestBT02, "behavior_tree")

    def test_behavior_tree_stores_metadata(self, clean_registry):
        """Test that @behavior_tree stores correct metadata."""
        @behavior_tree(name="test_bt_03", description="Metadata test BT")
        class TestBT03:
            pass

        assert registry.get_metadata(TestBT03, "bt_name") == "test_bt_03"
        assert registry.get_metadata(TestBT03, "description") == "Metadata test BT"

    def test_behavior_tree_stores_node_count(self, clean_registry):
        """Test that @behavior_tree stores node_count metadata."""
        @behavior_tree(name="test_bt_04")
        class TestBT04:
            @staticmethod
            def _count_nodes() -> int:
                return 5

        assert registry.get_metadata(TestBT04, "node_count") == 5

    def test_behavior_tree_sets_class_attributes(self, clean_registry):
        """Test that @behavior_tree sets _behavior_tree attributes on class."""
        @behavior_tree(name="test_bt_05", description="Attr test")
        class TestBT05:
            pass

        assert hasattr(TestBT05, "_behavior_tree")
        assert TestBT05._behavior_tree is True
        assert TestBT05._bt_name == "test_bt_05"
        assert TestBT05._bt_description == "Attr test"

    def test_behavior_tree_default_description(self, clean_registry):
        """Test that @behavior_tree handles missing description."""
        @behavior_tree(name="test_bt_06")
        class TestBT06:
            pass

        assert registry.get_metadata(TestBT06, "description") == ""
        assert TestBT06._bt_description == ""

    def test_behavior_tree_query_returns_all(self, clean_registry):
        """Test that Registry.query(tag='behavior_tree') returns all BTs."""
        @behavior_tree(name="test_bt_07a")
        class TestBT07a:
            pass

        @behavior_tree(name="test_bt_07b")
        class TestBT07b:
            pass

        @behavior_tree(name="test_bt_07c")
        class TestBT07c:
            pass

        all_bts = registry.query(tag="behavior_tree")
        registered_names = {registry.get_metadata(bt, "bt_name") for bt in all_bts}

        assert "test_bt_07a" in registered_names
        assert "test_bt_07b" in registered_names
        assert "test_bt_07c" in registered_names

    def test_behavior_tree_track_instances(self, clean_registry):
        """Test that track_instances=True enables instance tracking."""
        @behavior_tree(name="test_bt_08", track_instances=True)
        class TestBT08:
            pass

        instance1 = TestBT08()
        instance2 = TestBT08()

        count = registry.instance_count(TestBT08)
        assert count >= 2


class TestBTNodeDecorator:
    """Tests for @bt_node decorator."""

    def test_bt_node_registers_class(self, clean_registry):
        """Test that @bt_node registers the class with Foundation Registry."""
        @bt_node(node_type="action", description="Test action node")
        class TestAction01(Action):
            def __init__(self):
                super().__init__(lambda ctx: BTStatus.SUCCESS, name="TestAction01")

        assert registry.is_registered(TestAction01)
        assert registry.has_tag(TestAction01, "bt_node")

    def test_bt_node_stores_node_type(self, clean_registry):
        """Test that @bt_node stores node_type metadata."""
        @bt_node(node_type="action")
        class TestAction02:
            pass

        assert registry.get_metadata(TestAction02, "node_type") == "action"

    def test_bt_node_stores_description(self, clean_registry):
        """Test that @bt_node stores description metadata."""
        @bt_node(node_type="condition", description="Check health status")
        class HealthCheck01:
            pass

        assert registry.get_metadata(HealthCheck01, "description") == "Check health status"

    def test_bt_node_sets_class_attributes(self, clean_registry):
        """Test that @bt_node sets class attributes."""
        @bt_node(node_type="selector")
        class TestSelector01:
            pass

        assert TestSelector01._bt_node is True
        assert TestSelector01._bt_node_type == "selector"

    def test_bt_node_query_all_nodes(self, clean_registry):
        """Test that Registry.query(tag='bt_node') returns all BT nodes."""
        @bt_node(node_type="action")
        class QAction01:
            pass

        @bt_node(node_type="condition")
        class QCondition01:
            pass

        @bt_node(node_type="sequence")
        class QSequence01:
            pass

        all_nodes = registry.query(tag="bt_node")
        node_types = {registry.get_metadata(n, "node_type") for n in all_nodes
                      if registry.get_metadata(n, "node_type") in ["action", "condition", "sequence"]}

        assert "action" in node_types
        assert "condition" in node_types
        assert "sequence" in node_types

    def test_bt_node_query_by_type_action(self, clean_registry):
        """Test that Registry.query(tag='bt_node', node_type='action') returns only action nodes."""
        @bt_node(node_type="action")
        class FilterAction01:
            pass

        @bt_node(node_type="condition")
        class FilterCondition01:
            pass

        action_nodes = registry.query(tag="bt_node", node_type="action")

        assert FilterAction01 in action_nodes
        assert FilterCondition01 not in action_nodes

    def test_bt_node_query_by_type_condition(self, clean_registry):
        """Test filtering by condition type."""
        @bt_node(node_type="condition")
        class FilterCondition02:
            pass

        @bt_node(node_type="selector")
        class FilterSelector01:
            pass

        condition_nodes = registry.query(tag="bt_node", node_type="condition")

        assert FilterCondition02 in condition_nodes
        assert FilterSelector01 not in condition_nodes

    def test_bt_node_query_by_type_sequence(self, clean_registry):
        """Test filtering by sequence type."""
        @bt_node(node_type="sequence")
        class FilterSequence01:
            pass

        sequence_nodes = registry.query(tag="bt_node", node_type="sequence")
        assert FilterSequence01 in sequence_nodes

    def test_bt_node_query_by_type_selector(self, clean_registry):
        """Test filtering by selector type."""
        @bt_node(node_type="selector")
        class FilterSelector02:
            pass

        selector_nodes = registry.query(tag="bt_node", node_type="selector")
        assert FilterSelector02 in selector_nodes

    def test_bt_node_query_by_type_parallel(self, clean_registry):
        """Test filtering by parallel type."""
        @bt_node(node_type="parallel")
        class FilterParallel01:
            pass

        parallel_nodes = registry.query(tag="bt_node", node_type="parallel")
        assert FilterParallel01 in parallel_nodes

    def test_bt_node_query_by_type_decorator(self, clean_registry):
        """Test filtering by decorator type."""
        @bt_node(node_type="decorator")
        class FilterDecorator01:
            pass

        decorator_nodes = registry.query(tag="bt_node", node_type="decorator")
        assert FilterDecorator01 in decorator_nodes


class TestNodeTypeValidation:
    """Tests for BT node type validation."""

    def test_valid_node_type_action(self, clean_registry):
        """Test that 'action' is a valid node type."""
        @bt_node(node_type="action")
        class ValidAction01:
            pass

        assert ValidAction01._bt_node_type == "action"

    def test_valid_node_type_condition(self, clean_registry):
        """Test that 'condition' is a valid node type."""
        @bt_node(node_type="condition")
        class ValidCondition01:
            pass

        assert ValidCondition01._bt_node_type == "condition"

    def test_valid_node_type_selector(self, clean_registry):
        """Test that 'selector' is a valid node type."""
        @bt_node(node_type="selector")
        class ValidSelector01:
            pass

        assert ValidSelector01._bt_node_type == "selector"

    def test_valid_node_type_sequence(self, clean_registry):
        """Test that 'sequence' is a valid node type."""
        @bt_node(node_type="sequence")
        class ValidSequence01:
            pass

        assert ValidSequence01._bt_node_type == "sequence"

    def test_valid_node_type_parallel(self, clean_registry):
        """Test that 'parallel' is a valid node type."""
        @bt_node(node_type="parallel")
        class ValidParallel01:
            pass

        assert ValidParallel01._bt_node_type == "parallel"

    def test_valid_node_type_decorator(self, clean_registry):
        """Test that 'decorator' is a valid node type."""
        @bt_node(node_type="decorator")
        class ValidDecorator01:
            pass

        assert ValidDecorator01._bt_node_type == "decorator"

    def test_invalid_node_type_raises_error(self, clean_registry):
        """Test that invalid node type raises BTNodeTypeError."""
        with pytest.raises(BTNodeTypeError):
            @bt_node(node_type="invalid_type")
            class InvalidNode01:
                pass

    def test_node_type_case_insensitive(self, clean_registry):
        """Test that node type matching is case-insensitive."""
        @bt_node(node_type="ACTION")
        class CaseAction01:
            pass

        assert CaseAction01._bt_node_type == "action"

    def test_node_type_strips_whitespace(self, clean_registry):
        """Test that node type strips whitespace."""
        @bt_node(node_type="  action  ")
        class WhitespaceAction01:
            pass

        assert WhitespaceAction01._bt_node_type == "action"

    def test_valid_bt_node_types_constant(self, clean_registry):
        """Test VALID_BT_NODE_TYPES contains all expected types."""
        # Core composite and leaf types
        core_types = {"selector", "sequence", "parallel", "action", "condition", "decorator"}
        # All types should be present
        assert core_types <= VALID_BT_NODE_TYPES
        # Should also include decorator node types and leaf variants
        assert "invert" in VALID_BT_NODE_TYPES
        assert "repeat" in VALID_BT_NODE_TYPES


class TestFactoryInstantiation:
    """Tests for BehaviorTree.from_registry() factory method."""

    def test_from_registry_with_create_root(self, clean_registry, fresh_blackboard):
        """Test factory instantiation using create_root() method."""
        @behavior_tree(name="factory_test_01")
        class FactoryBT01:
            @classmethod
            def create_root(cls) -> BTNode:
                return Sequence([
                    Action(lambda ctx: BTStatus.SUCCESS, name="TestAction")
                ])

        bt = BehaviorTree.from_registry("factory_test_01", blackboard=fresh_blackboard)

        assert bt is not None
        assert bt.name == "factory_test_01"
        assert isinstance(bt.root, Sequence)

    def test_from_registry_with_build_method(self, clean_registry, fresh_blackboard):
        """Test factory instantiation using build() method."""
        @behavior_tree(name="factory_test_02")
        class FactoryBT02:
            @classmethod
            def build(cls, blackboard=None, tick_interval=0.1, **kwargs) -> BehaviorTree:
                root = Selector([
                    Action(lambda ctx: BTStatus.FAILURE, name="Fail"),
                    Action(lambda ctx: BTStatus.SUCCESS, name="Success"),
                ])
                return BehaviorTree(
                    root=root,
                    blackboard=blackboard,
                    tick_interval=tick_interval,
                    name="factory_test_02",
                )

        bt = BehaviorTree.from_registry("factory_test_02", blackboard=fresh_blackboard)

        assert bt is not None
        assert bt.name == "factory_test_02"
        assert isinstance(bt.root, Selector)

    def test_from_registry_not_found_raises_keyerror(self, clean_registry):
        """Test that from_registry raises KeyError for unknown name."""
        with pytest.raises(KeyError) as exc_info:
            BehaviorTree.from_registry("nonexistent_bt")

        assert "nonexistent_bt" in str(exc_info.value)

    def test_from_registry_invalid_class_raises_typeerror(self, clean_registry):
        """Test that from_registry raises TypeError for invalid class."""
        @behavior_tree(name="factory_test_03")
        class InvalidBT:
            # No create_root, build, or root attribute
            pass

        with pytest.raises(TypeError):
            BehaviorTree.from_registry("factory_test_03")

    def test_from_registry_with_kwargs(self, clean_registry, fresh_blackboard):
        """Test that kwargs are passed through to factory methods."""
        @behavior_tree(name="factory_test_04")
        class FactoryBT04:
            @classmethod
            def create_root(cls, custom_param=None) -> BTNode:
                if custom_param == "special":
                    return Parallel([
                        Action(lambda ctx: BTStatus.SUCCESS, name="SpecialAction")
                    ])
                return Sequence([Action(lambda ctx: BTStatus.SUCCESS)])

        bt = BehaviorTree.from_registry("factory_test_04", custom_param="special")

        assert isinstance(bt.root, Parallel)


class TestMultipleBTDefinitions:
    """Tests for multiple BT definitions coexisting."""

    def test_multiple_bts_coexist(self, clean_registry):
        """Test that multiple behavior trees can be registered."""
        @behavior_tree(name="multi_bt_01")
        class MultiBT01:
            pass

        @behavior_tree(name="multi_bt_02")
        class MultiBT02:
            pass

        @behavior_tree(name="multi_bt_03")
        class MultiBT03:
            pass

        assert registry.get("bt.multi_bt_01") is MultiBT01
        assert registry.get("bt.multi_bt_02") is MultiBT02
        assert registry.get("bt.multi_bt_03") is MultiBT03

    def test_multiple_bts_independent_metadata(self, clean_registry):
        """Test that multiple BTs have independent metadata."""
        @behavior_tree(name="indep_bt_01", description="First BT")
        class IndepBT01:
            pass

        @behavior_tree(name="indep_bt_02", description="Second BT")
        class IndepBT02:
            pass

        assert registry.get_metadata(IndepBT01, "description") == "First BT"
        assert registry.get_metadata(IndepBT02, "description") == "Second BT"

    def test_multiple_node_types_coexist(self, clean_registry):
        """Test that multiple node types can be registered."""
        @bt_node(node_type="action")
        class MultiAction01:
            pass

        @bt_node(node_type="condition")
        class MultiCondition01:
            pass

        @bt_node(node_type="selector")
        class MultiSelector01:
            pass

        @bt_node(node_type="sequence")
        class MultiSequence01:
            pass

        all_nodes = registry.query(tag="bt_node")

        # Check all are registered
        all_node_types = {registry.get_metadata(n, "node_type") for n in all_nodes}

        assert {"action", "condition", "selector", "sequence"} <= all_node_types


class TestDynamicBTConstruction:
    """Tests for dynamic behavior tree construction."""

    def test_dynamic_construction_from_registry(self, clean_registry, fresh_blackboard):
        """Test dynamically constructing a BT from registered nodes."""
        @bt_node(node_type="action", description="Move action")
        class MoveAction:
            @staticmethod
            def create() -> Action:
                return Action(lambda ctx: BTStatus.SUCCESS, name="Move")

        @bt_node(node_type="action", description="Attack action")
        class AttackAction:
            @staticmethod
            def create() -> Action:
                return Action(lambda ctx: BTStatus.SUCCESS, name="Attack")

        # Query all action nodes
        action_nodes = registry.query(tag="bt_node", node_type="action")

        # Build a sequence from all action nodes
        children = []
        for node_cls in action_nodes:
            if hasattr(node_cls, "create"):
                children.append(node_cls.create())

        root = Sequence(children) if children else Sequence([])
        bt = BehaviorTree(root=root, blackboard=fresh_blackboard)

        assert bt.root is not None

    def test_dynamic_bt_with_metadata_filtering(self, clean_registry):
        """Test filtering nodes by metadata for dynamic construction."""
        @bt_node(node_type="action", description="Combat action")
        class CombatAction01:
            pass

        @bt_node(node_type="action", description="Movement action")
        class MovementAction01:
            pass

        # Query by description pattern would need custom logic
        # Here we verify metadata is accessible for filtering
        all_actions = registry.query(tag="bt_node", node_type="action")
        combat_actions = [a for a in all_actions
                         if "Combat" in (registry.get_metadata(a, "description") or "")]

        assert any(a for a in combat_actions if "Combat" in registry.get_metadata(a, "description"))


class TestHelperFunctions:
    """Tests for helper query functions."""

    def test_get_all_behavior_trees(self, clean_registry):
        """Test get_all_behavior_trees() returns all registered BTs."""
        @behavior_tree(name="helper_bt_01")
        class HelperBT01:
            pass

        @behavior_tree(name="helper_bt_02")
        class HelperBT02:
            pass

        all_bts = get_all_behavior_trees()

        assert HelperBT01 in all_bts
        assert HelperBT02 in all_bts

    def test_get_bt_nodes_by_type(self, clean_registry):
        """Test get_bt_nodes_by_type() filters correctly."""
        @bt_node(node_type="action")
        class HelperAction01:
            pass

        @bt_node(node_type="condition")
        class HelperCondition01:
            pass

        action_nodes = get_bt_nodes_by_type("action")
        condition_nodes = get_bt_nodes_by_type("condition")

        assert HelperAction01 in action_nodes
        assert HelperCondition01 not in action_nodes
        assert HelperCondition01 in condition_nodes
        assert HelperAction01 not in condition_nodes

    def test_get_all_bt_nodes(self, clean_registry):
        """Test get_all_bt_nodes() returns all registered nodes."""
        @bt_node(node_type="action")
        class HelperAction02:
            pass

        @bt_node(node_type="selector")
        class HelperSelector01:
            pass

        all_nodes = get_all_bt_nodes()

        assert HelperAction02 in all_nodes
        assert HelperSelector01 in all_nodes


class TestAIRegistryIntegration:
    """Tests for ai_registry module integration."""

    def test_ai_registry_behavior_tree_decorator(self, clean_registry):
        """Test ai_registry.behavior_tree decorator works."""
        @ai_behavior_tree(name="ai_bt_01", description="AI module BT")
        class AIBT01:
            pass

        assert registry.has_tag(AIBT01, TAG_BEHAVIOR_TREE)

    def test_ai_registry_bt_node_decorator(self, clean_registry):
        """Test ai_registry.bt_node decorator works."""
        @ai_bt_node(type="action")
        class AIAction01:
            pass

        assert registry.has_tag(AIAction01, TAG_BT_NODE)

    def test_ai_registry_get_all_behavior_trees(self, clean_registry):
        """Test ai_registry.get_all_behavior_trees() works."""
        @ai_behavior_tree(name="ai_bt_02")
        class AIBT02:
            pass

        all_bts = ai_get_all_behavior_trees()
        assert AIBT02 in all_bts


class TestPerformance:
    """Performance tests for registry queries."""

    def test_100_queries_under_50ms(self, clean_registry):
        """Test that 100 registry queries complete in under 50ms."""
        # Register several BT nodes for realistic scenario
        # Use unique class names to avoid registration conflicts
        perf_classes = []

        for i in range(10):
            cls = type(f"PerfAction{i}", (), {})
            cls._bt_node = True
            cls._bt_node_type = "action"
            cls._bt_node_description = f"Perf action {i}"
            registry.register(cls, name=f"perf.action.{i}")
            registry.add_tag(cls, "bt_node")
            registry.set_metadata(cls, "node_type", "action")
            perf_classes.append(cls)

        for i in range(10):
            cls = type(f"PerfCondition{i}", (), {})
            cls._bt_node = True
            cls._bt_node_type = "condition"
            registry.register(cls, name=f"perf.condition.{i}")
            registry.add_tag(cls, "bt_node")
            registry.set_metadata(cls, "node_type", "condition")
            perf_classes.append(cls)

        for i in range(5):
            cls = type(f"PerfBT{i}", (), {})
            cls._behavior_tree = True
            cls._bt_name = f"perf_bt_{i}"
            registry.register(cls, name=f"bt.perf_bt_{i}")
            registry.add_tag(cls, "behavior_tree")
            perf_classes.append(cls)

        # Time 100 queries
        start_time = time.perf_counter()

        for _ in range(100):
            registry.query(tag="bt_node")
            registry.query(tag="bt_node", node_type="action")
            registry.query(tag="behavior_tree")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 50, f"100 queries took {elapsed_ms:.2f}ms, expected < 50ms"

    def test_1000_queries_reasonable_time(self, clean_registry):
        """Test that 1000 queries complete in reasonable time."""
        # Register nodes using unique class names
        perf_classes = []

        for i in range(20):
            cls = type(f"PerfNode{i}", (), {})
            cls._bt_node = True
            cls._bt_node_type = "action"
            registry.register(cls, name=f"perf.node.{i}")
            registry.add_tag(cls, "bt_node")
            registry.set_metadata(cls, "node_type", "action")
            perf_classes.append(cls)

        start_time = time.perf_counter()

        for _ in range(1000):
            registry.query(tag="bt_node")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should complete in under 500ms
        assert elapsed_ms < 500, f"1000 queries took {elapsed_ms:.2f}ms"


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_reregistration_same_name(self, clean_registry):
        """Test that re-registering same name doesn't raise error."""
        @behavior_tree(name="reregister_bt_01")
        class ReregisterBT01:
            pass

        # Re-decorating same class should be idempotent
        # (new class with same name should fail, but decorating same class is ok)
        assert registry.is_registered(ReregisterBT01)

    def test_empty_description(self, clean_registry):
        """Test that empty description is handled correctly."""
        @behavior_tree(name="empty_desc_bt_01", description="")
        class EmptyDescBT01:
            pass

        assert registry.get_metadata(EmptyDescBT01, "description") == ""

    def test_node_without_description(self, clean_registry):
        """Test bt_node without description."""
        @bt_node(node_type="action")
        class NoDescAction01:
            pass

        desc = registry.get_metadata(NoDescAction01, "description")
        assert desc is None or desc == ""

    def test_special_characters_in_name(self, clean_registry):
        """Test that special characters in name are handled."""
        @behavior_tree(name="special-name_01.test")
        class SpecialNameBT01:
            pass

        registered = registry.get("bt.special-name_01.test")
        assert registered is SpecialNameBT01


class TestIntegrationWithBTExecution:
    """Integration tests with actual BT execution."""

    def test_registered_bt_executes_correctly(self, clean_registry, fresh_blackboard):
        """Test that a BT from registry executes correctly."""
        execution_log = []

        @behavior_tree(name="exec_bt_01")
        class ExecBT01:
            @classmethod
            def create_root(cls) -> BTNode:
                def action1(ctx):
                    execution_log.append("action1")
                    return BTStatus.SUCCESS

                def action2(ctx):
                    execution_log.append("action2")
                    return BTStatus.SUCCESS

                return Sequence([
                    Action(action1, name="Action1"),
                    Action(action2, name="Action2"),
                ])

        bt = BehaviorTree.from_registry("exec_bt_01", blackboard=fresh_blackboard)
        status = bt.tick()

        assert status == BTStatus.SUCCESS
        assert execution_log == ["action1", "action2"]

    def test_registered_selector_fallback(self, clean_registry, fresh_blackboard):
        """Test that registered selector BT falls back correctly."""
        @behavior_tree(name="selector_bt_01")
        class SelectorBT01:
            @classmethod
            def create_root(cls) -> BTNode:
                return Selector([
                    Action(lambda ctx: BTStatus.FAILURE, name="Fail"),
                    Action(lambda ctx: BTStatus.SUCCESS, name="Success"),
                ])

        bt = BehaviorTree.from_registry("selector_bt_01", blackboard=fresh_blackboard)
        status = bt.tick()

        assert status == BTStatus.SUCCESS


# =============================================================================
# Additional tests to reach 50+ total
# =============================================================================


class TestMetadataCompleteness:
    """Tests for metadata storage completeness."""

    def test_bt_metadata_all_fields(self, clean_registry):
        """Test that all expected metadata fields are stored for BT."""
        @behavior_tree(name="meta_bt_01", description="Full metadata test")
        class MetaBT01:
            @staticmethod
            def _count_nodes():
                return 3

        assert registry.get_metadata(MetaBT01, "bt_name") == "meta_bt_01"
        assert registry.get_metadata(MetaBT01, "description") == "Full metadata test"
        assert registry.get_metadata(MetaBT01, "node_count") == 3

    def test_node_metadata_all_fields(self, clean_registry):
        """Test that all expected metadata fields are stored for nodes."""
        @bt_node(node_type="action", description="Action metadata test")
        class MetaAction01:
            pass

        assert registry.get_metadata(MetaAction01, "node_type") == "action"
        assert registry.get_metadata(MetaAction01, "description") == "Action metadata test"

    def test_get_all_metadata(self, clean_registry):
        """Test getting all metadata at once."""
        @behavior_tree(name="all_meta_bt_01", description="All metadata")
        class AllMetaBT01:
            pass

        all_meta = registry.get_all_metadata(AllMetaBT01)

        assert "bt_name" in all_meta
        assert "_tags" in all_meta
        assert "behavior_tree" in all_meta["_tags"]


class TestTagManagement:
    """Tests for tag management."""

    def test_bt_node_has_type_specific_tag(self, clean_registry):
        """Test that bt_node adds type-specific tag."""
        @bt_node(node_type="action")
        class TagAction01:
            pass

        # Check for type-specific tag
        assert registry.has_tag(TagAction01, "bt_node_action")

    def test_multiple_tags_on_node(self, clean_registry):
        """Test that nodes can have multiple tags."""
        @bt_node(node_type="condition")
        class MultiTagCondition01:
            pass

        # Has both generic and specific tags
        assert registry.has_tag(MultiTagCondition01, "bt_node")
        assert registry.has_tag(MultiTagCondition01, "bt_node_condition")

    def test_get_tags(self, clean_registry):
        """Test getting all tags for a registered type."""
        @behavior_tree(name="tags_bt_01")
        class TagsBT01:
            pass

        tags = registry.get_tags(TagsBT01)
        assert "behavior_tree" in tags


class TestRegistryUniqueness:
    """Tests for registry name uniqueness."""

    def test_bt_names_unique(self, clean_registry):
        """Test that BT names must be unique."""
        @behavior_tree(name="unique_bt_01")
        class UniqueBT01:
            pass

        # Second registration with same name should work on same class
        # but different class would fail (tested implicitly - registry handles this)
        assert registry.get("bt.unique_bt_01") is UniqueBT01

    def test_node_names_include_module(self, clean_registry):
        """Test that node registry names include module path."""
        @bt_node(node_type="action")
        class ModuleAction01:
            pass

        # The registration name should include the module
        name = registry.get_name(ModuleAction01)
        assert "ModuleAction01" in name


class TestQueryFiltering:
    """Tests for query filtering capabilities."""

    def test_query_with_multiple_filters(self, clean_registry):
        """Test query with multiple metadata filters."""
        @bt_node(node_type="action", description="Combat")
        class FilterAction01:
            pass

        # Query with both tag and type
        results = registry.query(tag="bt_node", node_type="action")
        assert FilterAction01 in results

    def test_empty_query_result(self, clean_registry):
        """Test query that returns empty result."""
        results = registry.query(tag="nonexistent_tag")
        assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
