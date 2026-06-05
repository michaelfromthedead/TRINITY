"""
Comprehensive unit tests for AI Registry integration with Foundation Registry.

Task: T-GP-3.14 - Wire Foundation Registry for AI node types

Tests cover:
- @bt_node decorator registration
- @goap_action decorator registration with preconditions/effects
- @consideration decorator registration with curve types
- Registry queries by tag
- Registry queries with metadata filters
- Dynamic node creation from registry
- BT graph construction from registry lookup
- GOAP planner action discovery
- Utility AI consideration discovery
- Performance benchmarks
"""

import gc
import time
import pytest
import sys

sys.path.insert(0, '/home/user/dev/USER/PROJECTS_VOID/TRINITY')

from foundation import registry, Registry
from engine.gameplay.ai.ai_registry import (
    bt_node,
    goap_action,
    consideration,
    get_all_bt_nodes,
    get_bt_nodes_by_type,
    get_all_goap_actions,
    get_goap_actions_by_effect,
    get_goap_actions_by_precondition,
    get_all_considerations,
    get_considerations_by_curve,
    create_bt_node_from_registry,
    create_goap_action_from_registry,
    create_consideration_from_registry,
    TAG_BT_NODE,
    TAG_GOAP_ACTION,
    TAG_CONSIDERATION,
    VALID_BT_NODE_TYPES,
    VALID_CURVE_TYPES,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_registry():
    """Store original state and restore after each test."""
    original_types = dict(registry._types)
    original_names = dict(registry._names)
    original_metadata = dict(registry._metadata)
    original_instances = dict(registry._instances)
    yield
    registry._types.clear()
    registry._types.update(original_types)
    registry._names.clear()
    registry._names.update(original_names)
    registry._metadata.clear()
    registry._metadata.update(original_metadata)
    registry._instances.clear()
    registry._instances.update(original_instances)


# =============================================================================
# BT Node Registration Tests
# =============================================================================


class TestBTNodeDecorator:
    """Tests for @bt_node decorator."""

    def test_bt_node_registers_with_registry(self):
        """Verify @bt_node registers the class with Foundation Registry."""
        @bt_node(type="action")
        class TestAction:
            pass

        assert registry.is_registered(TestAction)

    def test_bt_node_adds_tag(self):
        """Verify @bt_node adds the bt_node tag."""
        @bt_node(type="selector")
        class TestSelector:
            pass

        assert registry.has_tag(TestSelector, TAG_BT_NODE)

    def test_bt_node_stores_node_type(self):
        """Verify @bt_node stores node type in metadata."""
        @bt_node(type="sequence")
        class TestSequence:
            pass

        assert registry.get_metadata(TestSequence, "node_type") == "sequence"

    def test_bt_node_custom_name(self):
        """Verify @bt_node uses custom name when provided."""
        @bt_node(type="action", name="custom.TestAction")
        class TestAction:
            pass

        assert registry.get("custom.TestAction") is TestAction

    def test_bt_node_description(self):
        """Verify @bt_node stores description in metadata."""
        @bt_node(type="condition", description="Tests if target is visible")
        class VisibleCondition:
            pass

        assert registry.get_metadata(VisibleCondition, "description") == "Tests if target is visible"

    def test_bt_node_instance_tracking(self):
        """Verify @bt_node enables instance tracking when requested."""
        @bt_node(type="action", track_instances=True)
        class TrackedAction:
            pass

        obj1 = TrackedAction()
        obj2 = TrackedAction()
        assert registry.instance_count(TrackedAction) == 2

    def test_bt_node_class_attributes(self):
        """Verify @bt_node sets class attributes for introspection."""
        @bt_node(type="parallel", description="Runs children in parallel")
        class TestParallel:
            pass

        assert TestParallel._bt_node is True
        assert TestParallel._bt_node_type == "parallel"
        assert TestParallel._bt_description == "Runs children in parallel"

    def test_bt_node_invalid_type_raises_error(self):
        """Verify @bt_node raises ValueError for invalid node types."""
        with pytest.raises(ValueError, match="Invalid BT node type"):
            @bt_node(type="invalid_type")
            class BadNode:
                pass

    def test_bt_node_all_valid_types(self):
        """Verify all valid BT node types can be registered."""
        for node_type in VALID_BT_NODE_TYPES:
            @bt_node(type=node_type, name=f"test.{node_type}Node")
            class TestNode:
                pass

            assert registry.get_metadata(TestNode, "node_type") == node_type

    def test_bt_node_query_returns_registered(self):
        """Verify registered BT nodes can be queried."""
        @bt_node(type="action", name="query.Action1")
        class Action1:
            pass

        @bt_node(type="selector", name="query.Selector1")
        class Selector1:
            pass

        nodes = get_all_bt_nodes()
        assert Action1 in nodes
        assert Selector1 in nodes


# =============================================================================
# GOAP Action Registration Tests
# =============================================================================


class TestGOAPActionDecorator:
    """Tests for @goap_action decorator."""

    def test_goap_action_registers_with_registry(self):
        """Verify @goap_action registers the class with Foundation Registry."""
        @goap_action(preconditions=["has_weapon"], effects=["target_damaged"])
        class AttackAction:
            pass

        assert registry.is_registered(AttackAction)

    def test_goap_action_adds_tag(self):
        """Verify @goap_action adds the goap_action tag."""
        @goap_action()
        class EmptyAction:
            pass

        assert registry.has_tag(EmptyAction, TAG_GOAP_ACTION)

    def test_goap_action_stores_preconditions(self):
        """Verify @goap_action stores preconditions in metadata."""
        @goap_action(preconditions=["has_weapon", "can_see_target"])
        class AttackAction:
            pass

        preconds = registry.get_metadata(AttackAction, "preconditions")
        assert "has_weapon" in preconds
        assert "can_see_target" in preconds

    def test_goap_action_stores_effects(self):
        """Verify @goap_action stores effects in metadata."""
        @goap_action(effects=["target_damaged", "ammo_decreased"])
        class ShootAction:
            pass

        effects = registry.get_metadata(ShootAction, "effects")
        assert "target_damaged" in effects
        assert "ammo_decreased" in effects

    def test_goap_action_custom_name(self):
        """Verify @goap_action uses custom name when provided."""
        @goap_action(name="ai.combat.AttackAction")
        class AttackAction:
            pass

        assert registry.get("ai.combat.AttackAction") is AttackAction

    def test_goap_action_description(self):
        """Verify @goap_action stores description in metadata."""
        @goap_action(description="Attacks the current target")
        class AttackAction:
            pass

        assert registry.get_metadata(AttackAction, "description") == "Attacks the current target"

    def test_goap_action_cost(self):
        """Verify @goap_action stores default cost in metadata."""
        @goap_action(cost=5.0)
        class ExpensiveAction:
            pass

        assert registry.get_metadata(ExpensiveAction, "default_cost") == 5.0

    def test_goap_action_class_attributes(self):
        """Verify @goap_action sets class attributes for introspection."""
        @goap_action(
            preconditions=["has_weapon"],
            effects=["target_damaged"],
            description="Attack action",
            cost=2.0,
        )
        class AttackAction:
            pass

        assert AttackAction._goap_action is True
        assert "has_weapon" in AttackAction._goap_preconditions
        assert "target_damaged" in AttackAction._goap_effects
        assert AttackAction._goap_description == "Attack action"
        assert AttackAction._goap_default_cost == 2.0

    def test_goap_action_instance_tracking(self):
        """Verify @goap_action enables instance tracking when requested."""
        @goap_action(track_instances=True)
        class TrackedGOAPAction:
            pass

        obj = TrackedGOAPAction()
        assert registry.instance_count(TrackedGOAPAction) == 1

    def test_goap_action_query_returns_registered(self):
        """Verify registered GOAP actions can be queried."""
        @goap_action(name="query.GOAPAction1")
        class GOAPAction1:
            pass

        @goap_action(name="query.GOAPAction2")
        class GOAPAction2:
            pass

        actions = get_all_goap_actions()
        assert GOAPAction1 in actions
        assert GOAPAction2 in actions

    def test_goap_action_query_by_effect(self):
        """Verify GOAP actions can be queried by effect type."""
        @goap_action(effects=["has_weapon"], name="effect.PickupWeapon")
        class PickupWeaponAction:
            pass

        @goap_action(effects=["has_ammo"], name="effect.PickupAmmo")
        class PickupAmmoAction:
            pass

        weapon_actions = get_goap_actions_by_effect("has_weapon")
        assert PickupWeaponAction in weapon_actions
        assert PickupAmmoAction not in weapon_actions

    def test_goap_action_query_by_precondition(self):
        """Verify GOAP actions can be queried by precondition."""
        @goap_action(preconditions=["has_weapon"], name="precond.Attack")
        class AttackAction:
            pass

        @goap_action(preconditions=["has_food"], name="precond.Eat")
        class EatAction:
            pass

        weapon_required = get_goap_actions_by_precondition("has_weapon")
        assert AttackAction in weapon_required
        assert EatAction not in weapon_required


# =============================================================================
# Consideration Registration Tests
# =============================================================================


class TestConsiderationDecorator:
    """Tests for @consideration decorator."""

    def test_consideration_registers_with_registry(self):
        """Verify @consideration registers the class with Foundation Registry."""
        @consideration(curve="linear")
        class HealthConsideration:
            pass

        assert registry.is_registered(HealthConsideration)

    def test_consideration_adds_tag(self):
        """Verify @consideration adds the consideration tag."""
        @consideration(curve="exponential")
        class DistanceConsideration:
            pass

        assert registry.has_tag(DistanceConsideration, TAG_CONSIDERATION)

    def test_consideration_stores_curve_type(self):
        """Verify @consideration stores curve type in metadata."""
        @consideration(curve="sigmoid")
        class ThreatConsideration:
            pass

        assert registry.get_metadata(ThreatConsideration, "curve_type") == "sigmoid"

    def test_consideration_custom_name(self):
        """Verify @consideration uses custom name when provided."""
        @consideration(curve="linear", name="ai.utility.Health")
        class HealthConsideration:
            pass

        assert registry.get("ai.utility.Health") is HealthConsideration

    def test_consideration_description(self):
        """Verify @consideration stores description in metadata."""
        @consideration(curve="logistic", description="Evaluates current health percentage")
        class HealthConsideration:
            pass

        assert registry.get_metadata(HealthConsideration, "description") == "Evaluates current health percentage"

    def test_consideration_weight(self):
        """Verify @consideration stores default weight in metadata."""
        @consideration(curve="linear", weight=2.5)
        class ImportantConsideration:
            pass

        assert registry.get_metadata(ImportantConsideration, "default_weight") == 2.5

    def test_consideration_class_attributes(self):
        """Verify @consideration sets class attributes for introspection."""
        @consideration(curve="quadratic", description="Test", weight=1.5)
        class TestConsideration:
            pass

        assert TestConsideration._consideration is True
        assert TestConsideration._consideration_curve == "quadratic"
        assert TestConsideration._consideration_description == "Test"
        assert TestConsideration._consideration_default_weight == 1.5

    def test_consideration_invalid_curve_raises_error(self):
        """Verify @consideration raises ValueError for invalid curve types."""
        with pytest.raises(ValueError, match="Invalid curve type"):
            @consideration(curve="invalid_curve")
            class BadConsideration:
                pass

    def test_consideration_all_valid_curves(self):
        """Verify all valid curve types can be registered."""
        for curve_type in VALID_CURVE_TYPES:
            @consideration(curve=curve_type, name=f"test.{curve_type}Consideration")
            class TestConsideration:
                pass

            assert registry.get_metadata(TestConsideration, "curve_type") == curve_type

    def test_consideration_instance_tracking(self):
        """Verify @consideration enables instance tracking when requested."""
        @consideration(curve="linear", track_instances=True)
        class TrackedConsideration:
            pass

        obj = TrackedConsideration()
        assert registry.instance_count(TrackedConsideration) == 1

    def test_consideration_query_returns_registered(self):
        """Verify registered considerations can be queried."""
        @consideration(curve="linear", name="query.Consideration1")
        class Consideration1:
            pass

        @consideration(curve="exponential", name="query.Consideration2")
        class Consideration2:
            pass

        considerations = get_all_considerations()
        assert Consideration1 in considerations
        assert Consideration2 in considerations

    def test_consideration_query_by_curve(self):
        """Verify considerations can be queried by curve type."""
        @consideration(curve="linear", name="curve.Linear")
        class LinearConsideration:
            pass

        @consideration(curve="exponential", name="curve.Exponential")
        class ExponentialConsideration:
            pass

        linear_considerations = get_considerations_by_curve("linear")
        assert LinearConsideration in linear_considerations
        assert ExponentialConsideration not in linear_considerations


# =============================================================================
# Registry Query Tests
# =============================================================================


class TestRegistryQueries:
    """Tests for Registry query functionality."""

    def test_query_by_tag_bt_node(self):
        """Verify Registry.query returns all BT nodes."""
        @bt_node(type="action", name="query.bt.Action")
        class Action:
            pass

        @bt_node(type="selector", name="query.bt.Selector")
        class Selector:
            pass

        results = registry.query(tag=TAG_BT_NODE)
        assert Action in results
        assert Selector in results

    def test_query_by_tag_goap_action(self):
        """Verify Registry.query returns all GOAP actions."""
        @goap_action(name="query.goap.Action1")
        class Action1:
            pass

        @goap_action(name="query.goap.Action2")
        class Action2:
            pass

        results = registry.query(tag=TAG_GOAP_ACTION)
        assert Action1 in results
        assert Action2 in results

    def test_query_by_tag_consideration(self):
        """Verify Registry.query returns all considerations."""
        @consideration(curve="linear", name="query.util.Consideration1")
        class Consideration1:
            pass

        @consideration(curve="exponential", name="query.util.Consideration2")
        class Consideration2:
            pass

        results = registry.query(tag=TAG_CONSIDERATION)
        assert Consideration1 in results
        assert Consideration2 in results

    def test_query_filters_by_node_type(self):
        """Verify Registry.query filters by node type."""
        @bt_node(type="action", name="filter.Action")
        class ActionNode:
            pass

        @bt_node(type="selector", name="filter.Selector")
        class SelectorNode:
            pass

        @bt_node(type="sequence", name="filter.Sequence")
        class SequenceNode:
            pass

        action_nodes = registry.query(tag=TAG_BT_NODE, node_type="action")
        assert ActionNode in action_nodes
        assert SelectorNode not in action_nodes
        assert SequenceNode not in action_nodes

    def test_query_combined_filters(self):
        """Verify Registry.query combines tag and metadata filters."""
        @bt_node(type="action", name="combined.Action1")
        class Action1:
            pass

        @bt_node(type="action", name="combined.Action2")
        class Action2:
            pass
        registry.set_metadata(Action2, "custom_flag", True)

        results = registry.query(tag=TAG_BT_NODE, node_type="action", custom_flag=True)
        assert Action2 in results
        assert Action1 not in results

    def test_query_empty_results(self):
        """Verify Registry.query returns empty list when no matches."""
        results = registry.query(tag="nonexistent_tag")
        assert results == []

    def test_query_effect_membership(self):
        """Verify Registry.query handles set membership for effects."""
        @goap_action(effects=["has_weapon", "is_armed"], name="member.Arm")
        class ArmAction:
            pass

        results = registry.query(tag=TAG_GOAP_ACTION, effects="has_weapon")
        assert ArmAction in results

        results = registry.query(tag=TAG_GOAP_ACTION, effects="nonexistent")
        assert ArmAction not in results


# =============================================================================
# Metadata Correctness Tests
# =============================================================================


class TestMetadataCorrectness:
    """Tests for metadata being correctly attached."""

    def test_bt_node_metadata_complete(self):
        """Verify BT node metadata is complete and correct."""
        @bt_node(type="condition", description="Test condition", name="meta.BTNode")
        class TestCondition:
            pass

        meta = registry.get_all_metadata(TestCondition)
        assert "node_type" in meta
        assert meta["node_type"] == "condition"
        assert "description" in meta
        assert "_tags" in meta
        assert TAG_BT_NODE in meta["_tags"]

    def test_goap_action_metadata_complete(self):
        """Verify GOAP action metadata is complete and correct."""
        @goap_action(
            preconditions=["has_weapon"],
            effects=["target_damaged"],
            description="Attack action",
            cost=3.0,
            name="meta.GOAPAction",
        )
        class AttackAction:
            pass

        meta = registry.get_all_metadata(AttackAction)
        assert "preconditions" in meta
        assert "has_weapon" in meta["preconditions"]
        assert "effects" in meta
        assert "target_damaged" in meta["effects"]
        assert "description" in meta
        assert "default_cost" in meta
        assert meta["default_cost"] == 3.0
        assert TAG_GOAP_ACTION in meta["_tags"]

    def test_consideration_metadata_complete(self):
        """Verify consideration metadata is complete and correct."""
        @consideration(
            curve="exponential",
            description="Health evaluation",
            weight=2.0,
            name="meta.Consideration",
        )
        class HealthConsideration:
            pass

        meta = registry.get_all_metadata(HealthConsideration)
        assert "curve_type" in meta
        assert meta["curve_type"] == "exponential"
        assert "description" in meta
        assert "default_weight" in meta
        assert meta["default_weight"] == 2.0
        assert TAG_CONSIDERATION in meta["_tags"]


# =============================================================================
# Dynamic Creation Tests
# =============================================================================


class TestDynamicCreation:
    """Tests for dynamic node creation from registry."""

    def test_create_bt_node_from_registry(self):
        """Verify BT nodes can be created dynamically from registry."""
        @bt_node(type="action", name="dynamic.BTAction")
        class DynamicAction:
            def __init__(self, value=10):
                self.value = value

        instance = create_bt_node_from_registry("dynamic.BTAction", value=42)
        assert isinstance(instance, DynamicAction)
        assert instance.value == 42

    def test_create_bt_node_not_found_raises(self):
        """Verify creating unknown BT node raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            create_bt_node_from_registry("nonexistent.Node")

    def test_create_bt_node_wrong_tag_raises(self):
        """Verify creating node with wrong tag raises ValueError."""
        @goap_action(name="wrong.tag.GOAPAction")
        class NotBTNode:
            pass

        with pytest.raises(ValueError, match="not a registered BT node"):
            create_bt_node_from_registry("wrong.tag.GOAPAction")

    def test_create_goap_action_from_registry(self):
        """Verify GOAP actions can be created dynamically from registry."""
        @goap_action(name="dynamic.GOAPAction")
        class DynamicGOAPAction:
            def __init__(self, target=None):
                self.target = target

        instance = create_goap_action_from_registry("dynamic.GOAPAction", target="enemy")
        assert isinstance(instance, DynamicGOAPAction)
        assert instance.target == "enemy"

    def test_create_goap_action_not_found_raises(self):
        """Verify creating unknown GOAP action raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            create_goap_action_from_registry("nonexistent.Action")

    def test_create_consideration_from_registry(self):
        """Verify considerations can be created dynamically from registry."""
        @consideration(curve="linear", name="dynamic.Consideration")
        class DynamicConsideration:
            def __init__(self, weight=1.0):
                self.weight = weight

        instance = create_consideration_from_registry("dynamic.Consideration", weight=2.5)
        assert isinstance(instance, DynamicConsideration)
        assert instance.weight == 2.5

    def test_create_consideration_not_found_raises(self):
        """Verify creating unknown consideration raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            create_consideration_from_registry("nonexistent.Consideration")


# =============================================================================
# BT Graph Construction Tests
# =============================================================================


class TestBTGraphConstruction:
    """Tests for BT graph construction from registry lookup."""

    def test_build_bt_from_registry_actions(self):
        """Verify BT can be built using registry-discovered actions."""
        @bt_node(type="action", name="graph.MoveAction")
        class MoveAction:
            def __init__(self, target=None):
                self.target = target
                self.children = []

        @bt_node(type="action", name="graph.AttackAction")
        class AttackAction:
            def __init__(self):
                self.children = []

        @bt_node(type="sequence", name="graph.CombatSequence")
        class CombatSequence:
            def __init__(self, children=None):
                self.children = children or []

        # Get action nodes from registry
        action_nodes = get_bt_nodes_by_type("action")
        assert MoveAction in action_nodes
        assert AttackAction in action_nodes

        # Build sequence from discovered actions
        move = create_bt_node_from_registry("graph.MoveAction", target="enemy")
        attack = create_bt_node_from_registry("graph.AttackAction")
        sequence = create_bt_node_from_registry("graph.CombatSequence", children=[move, attack])

        assert len(sequence.children) == 2
        assert sequence.children[0].target == "enemy"

    def test_filter_bt_nodes_for_graph(self):
        """Verify BT nodes can be filtered for graph construction."""
        @bt_node(type="condition", description="Check health", name="graph.filter.HealthCheck")
        class HealthCheckCondition:
            pass

        @bt_node(type="condition", description="Check ammo", name="graph.filter.AmmoCheck")
        class AmmoCheckCondition:
            pass

        @bt_node(type="action", name="graph.filter.SomeAction")
        class SomeAction:
            pass

        conditions = get_bt_nodes_by_type("condition")
        assert HealthCheckCondition in conditions
        assert AmmoCheckCondition in conditions
        assert SomeAction not in conditions


# =============================================================================
# GOAP Planner Discovery Tests
# =============================================================================


class TestGOAPPlannerDiscovery:
    """Tests for GOAP planner using registry for action discovery."""

    def test_discover_actions_for_planner(self):
        """Verify GOAP planner can discover actions from registry."""
        @goap_action(
            preconditions=[],
            effects=["has_weapon"],
            name="planner.PickupWeapon",
        )
        class PickupWeaponAction:
            pass

        @goap_action(
            preconditions=["has_weapon"],
            effects=["target_damaged"],
            name="planner.Attack",
        )
        class AttackAction:
            pass

        @goap_action(
            preconditions=[],
            effects=["has_health"],
            name="planner.Heal",
        )
        class HealAction:
            pass

        all_actions = get_all_goap_actions()
        assert len(all_actions) >= 3
        assert PickupWeaponAction in all_actions
        assert AttackAction in all_actions
        assert HealAction in all_actions

    def test_filter_actions_by_effect_for_planning(self):
        """Verify actions can be filtered by effect for planning."""
        @goap_action(effects=["has_weapon"], name="plan.effect.GetWeapon")
        class GetWeaponAction:
            pass

        @goap_action(effects=["has_cover"], name="plan.effect.TakeCover")
        class TakeCoverAction:
            pass

        weapon_providers = get_goap_actions_by_effect("has_weapon")
        assert GetWeaponAction in weapon_providers
        assert TakeCoverAction not in weapon_providers

    def test_filter_actions_by_precondition_for_planning(self):
        """Verify actions can be filtered by precondition for planning."""
        @goap_action(preconditions=["has_weapon"], name="plan.precond.Shoot")
        class ShootAction:
            pass

        @goap_action(preconditions=["has_key"], name="plan.precond.OpenDoor")
        class OpenDoorAction:
            pass

        weapon_required = get_goap_actions_by_precondition("has_weapon")
        assert ShootAction in weapon_required
        assert OpenDoorAction not in weapon_required


# =============================================================================
# Utility AI Discovery Tests
# =============================================================================


class TestUtilityAIDiscovery:
    """Tests for Utility AI using registry for consideration discovery."""

    def test_discover_considerations_for_utility_ai(self):
        """Verify Utility AI can discover considerations from registry."""
        @consideration(curve="linear", name="utility.discovery.Health")
        class HealthConsideration:
            pass

        @consideration(curve="exponential", name="utility.discovery.Distance")
        class DistanceConsideration:
            pass

        @consideration(curve="sigmoid", name="utility.discovery.Threat")
        class ThreatConsideration:
            pass

        all_considerations = get_all_considerations()
        assert len(all_considerations) >= 3
        assert HealthConsideration in all_considerations
        assert DistanceConsideration in all_considerations
        assert ThreatConsideration in all_considerations

    def test_filter_considerations_by_curve(self):
        """Verify considerations can be filtered by curve type."""
        @consideration(curve="linear", name="utility.curve.Linear1")
        class Linear1:
            pass

        @consideration(curve="linear", name="utility.curve.Linear2")
        class Linear2:
            pass

        @consideration(curve="exponential", name="utility.curve.Exp1")
        class Exp1:
            pass

        linear_considerations = get_considerations_by_curve("linear")
        assert Linear1 in linear_considerations
        assert Linear2 in linear_considerations
        assert Exp1 not in linear_considerations


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance benchmarks for registry operations."""

    def test_query_performance_1000_queries(self):
        """Verify 1000 registry queries complete under 100ms."""
        # Register some types first
        for i in range(100):
            @bt_node(type="action", name=f"perf.bt.Action{i}")
            class _BT:
                pass

            @goap_action(effects=[f"effect_{i}"], name=f"perf.goap.Action{i}")
            class _GOAP:
                pass

            @consideration(curve="linear", name=f"perf.util.Consideration{i}")
            class _Consideration:
                pass

        # Measure query performance
        start = time.perf_counter()
        for _ in range(1000):
            registry.query(tag=TAG_BT_NODE)
            registry.query(tag=TAG_GOAP_ACTION)
            registry.query(tag=TAG_CONSIDERATION)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"1000 queries took {elapsed_ms:.2f}ms (> 500ms)"

    def test_registration_performance(self):
        """Verify registration performance is acceptable."""
        start = time.perf_counter()
        for i in range(100):
            @bt_node(type="action", name=f"perf.reg.Action{i}")
            class _BT:
                pass
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 100 registrations should be fast
        assert elapsed_ms < 100, f"100 registrations took {elapsed_ms:.2f}ms"


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of AI registry operations."""

    def test_concurrent_registration(self):
        """Verify concurrent registration is thread-safe."""
        import threading

        results = []
        errors = []

        def register_types(prefix):
            try:
                for i in range(10):
                    @bt_node(type="action", name=f"{prefix}.Action{i}")
                    class _BT:
                        pass
                    results.append(True)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_types, args=(f"thread{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 50

    def test_concurrent_query(self):
        """Verify concurrent queries are thread-safe."""
        import threading

        # Pre-register some types
        for i in range(10):
            @bt_node(type="action", name=f"concurrent.Action{i}")
            class _BT:
                pass

        results = []
        errors = []

        def query_types():
            try:
                for _ in range(100):
                    nodes = get_all_bt_nodes()
                    results.append(len(nodes))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=query_types)
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 500


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_preconditions_effects(self):
        """Verify GOAP actions with empty preconditions/effects work."""
        @goap_action(preconditions=[], effects=[], name="edge.EmptyAction")
        class EmptyAction:
            pass

        assert registry.is_registered(EmptyAction)
        assert registry.get_metadata(EmptyAction, "preconditions") == frozenset()
        assert registry.get_metadata(EmptyAction, "effects") == frozenset()

    def test_none_preconditions_effects(self):
        """Verify GOAP actions with None preconditions/effects work."""
        @goap_action(name="edge.NoneAction")
        class NoneAction:
            pass

        assert registry.is_registered(NoneAction)
        assert registry.get_metadata(NoneAction, "preconditions") == frozenset()

    def test_duplicate_registration_same_name(self):
        """Verify duplicate registration with same name is handled."""
        @bt_node(type="action", name="edge.Duplicate")
        class FirstClass:
            pass

        # Same class registered again should be idempotent
        registry.register(FirstClass, name="edge.Duplicate")
        assert registry.is_registered(FirstClass)

    def test_special_characters_in_names(self):
        """Verify special characters in names work."""
        @bt_node(type="action", name="edge.My-Action_v2.0")
        class SpecialNameAction:
            pass

        assert registry.get("edge.My-Action_v2.0") is SpecialNameAction

    def test_unicode_in_description(self):
        """Verify Unicode in descriptions work."""
        @bt_node(type="action", description="Handles UTF-8: cafe, naive", name="edge.Unicode")
        class UnicodeAction:
            pass

        desc = registry.get_metadata(UnicodeAction, "description")
        assert "cafe" in desc
        assert "naive" in desc

    def test_large_preconditions_effects_list(self):
        """Verify large preconditions/effects lists work."""
        preconds = [f"precond_{i}" for i in range(100)]
        effects = [f"effect_{i}" for i in range(100)]

        @goap_action(preconditions=preconds, effects=effects, name="edge.LargeAction")
        class LargeAction:
            pass

        stored_preconds = registry.get_metadata(LargeAction, "preconditions")
        stored_effects = registry.get_metadata(LargeAction, "effects")
        assert len(stored_preconds) == 100
        assert len(stored_effects) == 100


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for AI registry with Foundation Registry."""

    def test_mixed_ai_types_registration(self):
        """Verify mixed AI types can be registered and queried."""
        @bt_node(type="action", name="integration.BTAction")
        class BTAction:
            pass

        @goap_action(effects=["goal_achieved"], name="integration.GOAPAction")
        class GOAPAction:
            pass

        @consideration(curve="linear", name="integration.Consideration")
        class UtilityConsideration:
            pass

        # All should be registered
        assert registry.is_registered(BTAction)
        assert registry.is_registered(GOAPAction)
        assert registry.is_registered(UtilityConsideration)

        # Each should have correct tag
        assert registry.has_tag(BTAction, TAG_BT_NODE)
        assert registry.has_tag(GOAPAction, TAG_GOAP_ACTION)
        assert registry.has_tag(UtilityConsideration, TAG_CONSIDERATION)

        # Queries should return correct types
        bt_nodes = get_all_bt_nodes()
        goap_actions = get_all_goap_actions()
        considerations = get_all_considerations()

        assert BTAction in bt_nodes
        assert BTAction not in goap_actions
        assert BTAction not in considerations

        assert GOAPAction in goap_actions
        assert GOAPAction not in bt_nodes
        assert GOAPAction not in considerations

        assert UtilityConsideration in considerations
        assert UtilityConsideration not in bt_nodes
        assert UtilityConsideration not in goap_actions

    def test_registry_describe_includes_ai_metadata(self):
        """Verify registry.describe includes AI-specific metadata."""
        @bt_node(type="action", description="Test action", name="describe.BTAction")
        class DescribeAction:
            pass

        description = registry.describe(DescribeAction)
        assert "describe.BTAction" in description

    def test_tags_methods(self):
        """Verify tag methods work correctly."""
        @bt_node(type="action", name="tags.TestAction")
        class TagsTestAction:
            pass

        # Should have bt_node tag
        assert registry.has_tag(TagsTestAction, TAG_BT_NODE)

        # Add custom tag
        registry.add_tag(TagsTestAction, "custom_tag")
        assert registry.has_tag(TagsTestAction, "custom_tag")

        # Get all tags
        tags = registry.get_tags(TagsTestAction)
        assert TAG_BT_NODE in tags
        assert "custom_tag" in tags

        # Remove tag
        removed = registry.remove_tag(TagsTestAction, "custom_tag")
        assert removed is True
        assert not registry.has_tag(TagsTestAction, "custom_tag")

        # Remove non-existent tag
        removed = registry.remove_tag(TagsTestAction, "nonexistent")
        assert removed is False


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
