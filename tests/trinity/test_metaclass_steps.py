"""Tests for _metaclass_steps across all Phase 3 metaclasses."""
import warnings

import pytest

from trinity.decorators.ops import Op, Step
from trinity.metaclasses import (
    AssetMeta,
    ComponentMeta,
    EngineMeta,
    EventMeta,
    ProtocolMeta,
    ResourceMeta,
    StateMeta,
    SystemMeta,
)


# ---------------------------------------------------------------------------
# Fixtures: clear all registries before/after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_all_registries():
    """Clear all metaclass registries to avoid test pollution."""
    for meta in (EngineMeta, ComponentMeta, SystemMeta, ResourceMeta,
                 EventMeta, StateMeta, AssetMeta, ProtocolMeta):
        meta.clear_registry()
    yield
    for meta in (EngineMeta, ComponentMeta, SystemMeta, ResourceMeta,
                 EventMeta, StateMeta, AssetMeta, ProtocolMeta):
        meta.clear_registry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_step(steps, op, **kwargs):
    """Check if any step matches op and optional arg constraints."""
    for s in steps:
        if s.op == op:
            if not kwargs:
                return True
            if all(s.args.get(k) == v for k, v in kwargs.items()):
                return True
    return False


def _count_steps(steps, op):
    return sum(1 for s in steps if s.op == op)


# ===========================================================================
# EngineMeta
# ===========================================================================

class TestEngineMetaSteps:
    def test_has_metaclass_steps(self):
        cls = EngineMeta("MyEngine", (), {})
        assert hasattr(cls, "_metaclass_steps")
        assert isinstance(cls._metaclass_steps, list)

    def test_register_step(self):
        cls = EngineMeta("EngSteps1", (), {})
        steps = cls._metaclass_steps
        assert _has_step(steps, Op.REGISTER, registry="engine_types")

    def test_base_class_skipped(self):
        """Base class names get no REGISTER(engine_types) step."""
        cls = EngineMeta("EngineBase", (), {})
        assert cls._metaclass_steps == []

    def test_steps_are_step_objects(self):
        cls = EngineMeta("EngSteps2", (), {})
        for s in cls._metaclass_steps:
            assert isinstance(s, Step)


# ===========================================================================
# ComponentMeta
# ===========================================================================

class TestComponentMetaSteps:
    def test_has_metaclass_steps(self):
        cls = ComponentMeta("Comp1", (), {})
        assert isinstance(cls._metaclass_steps, list)
        assert len(cls._metaclass_steps) > 0

    def test_tag_component_id(self):
        cls = ComponentMeta("Comp2", (), {})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="component_id")

    def test_tag_component_name(self):
        cls = ComponentMeta("Comp3", (), {})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="component_name")

    def test_validate_step(self):
        cls = ComponentMeta("Comp4", (), {})
        assert _has_step(cls._metaclass_steps, Op.VALIDATE, constraint="component_rules")

    def test_register_component_registry(self):
        cls = ComponentMeta("Comp5", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="component_registry")

    def test_register_foundation(self):
        cls = ComponentMeta("Comp6", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="foundation")

    def test_parent_register_step_preserved(self):
        cls = ComponentMeta("Comp7", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_field_describe_steps(self):
        cls = ComponentMeta("Comp8", (), {"__annotations__": {"x": int, "y": float}})
        assert _has_step(cls._metaclass_steps, Op.DESCRIBE, field="x")
        assert _has_step(cls._metaclass_steps, Op.DESCRIBE, field="y")

    def test_intercept_steps_for_fields(self):
        cls = ComponentMeta("Comp9", (), {"__annotations__": {"hp": int}})
        assert _has_step(cls._metaclass_steps, Op.INTERCEPT, field="hp")

    def test_pooled_extra_steps(self):
        ns = {"_pooled_config": {"max_size": 64}}
        cls = ComponentMeta("Comp10", (), ns)
        assert _has_step(cls._metaclass_steps, Op.TAG, key="pooled", value=True)
        assert _has_step(cls._metaclass_steps, Op.HOOK, event="on_create", callback="pool_allocate")

    def test_budgeted_extra_steps(self):
        ns = {"_budget_config": {"max_instances": 100}}
        cls = ComponentMeta("Comp11", (), ns)
        assert _has_step(cls._metaclass_steps, Op.TAG, key="budgeted", value=True)
        assert _has_step(cls._metaclass_steps, Op.VALIDATE, constraint="budget_limit")

    def test_base_component_skipped(self):
        cls = ComponentMeta("Component", (), {})
        # Base "Component" returns early; only EngineMeta base-class skip applies
        # so it should have an empty list (base name is in _BASE_CLASS_NAMES)
        assert cls._metaclass_steps == []

    def test_all_steps_are_step_objects(self):
        cls = ComponentMeta("Comp12", (), {"__annotations__": {"a": int}})
        for s in cls._metaclass_steps:
            assert isinstance(s, Step)
            assert isinstance(s.op, Op)


# ===========================================================================
# SystemMeta
# ===========================================================================

class TestSystemMetaSteps:
    def _make_system(self, name="Sys1", ns=None):
        if ns is None:
            ns = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SystemMeta(name, (), ns)

    def test_has_metaclass_steps(self):
        cls = self._make_system("Sys2")
        assert isinstance(cls._metaclass_steps, list)
        assert len(cls._metaclass_steps) > 0

    def test_tag_system_id(self):
        cls = self._make_system("Sys3")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="system_id")

    def test_tag_system_name(self):
        cls = self._make_system("Sys4")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="system_name")

    def test_tag_defaults(self):
        cls = self._make_system("Sys5")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="system_phase")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="reads")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="writes")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="exclusive")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="priority")

    def test_validate_step(self):
        cls = self._make_system("Sys6")
        assert _has_step(cls._metaclass_steps, Op.VALIDATE, constraint="system_declarations")

    def test_describe_dependencies(self):
        cls = self._make_system("Sys7")
        assert _has_step(cls._metaclass_steps, Op.DESCRIBE)

    def test_register_system_registry(self):
        cls = self._make_system("Sys8")
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="system_registry")

    def test_parent_register_preserved(self):
        cls = self._make_system("Sys9")
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_base_system_skipped(self):
        cls = SystemMeta("System", (), {})
        assert cls._metaclass_steps == []


# ===========================================================================
# ResourceMeta
# ===========================================================================

class TestResourceMetaSteps:
    def test_has_metaclass_steps(self):
        cls = ResourceMeta("Res1", (), {})
        assert isinstance(cls._metaclass_steps, list)
        assert len(cls._metaclass_steps) > 0

    def test_tag_resource_id(self):
        cls = ResourceMeta("Res2", (), {})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="resource_id")

    def test_tag_resource_name(self):
        cls = ResourceMeta("Res3", (), {})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="resource_name")

    def test_tag_priority_and_lazy(self):
        cls = ResourceMeta("Res4", (), {})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="resource_priority")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="resource_lazy")

    def test_register_resource_registry(self):
        cls = ResourceMeta("Res5", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="resource_registry")

    def test_hook_singleton_enforce(self):
        cls = ResourceMeta("Res6", (), {})
        assert _has_step(cls._metaclass_steps, Op.HOOK, event="on_create", callback="singleton_enforce")

    def test_parent_register_preserved(self):
        cls = ResourceMeta("Res7", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_base_resource_skipped(self):
        cls = ResourceMeta("Resource", (), {})
        assert cls._metaclass_steps == []


# ===========================================================================
# EventMeta
# ===========================================================================

class TestEventMetaSteps:
    def test_has_metaclass_steps(self):
        cls = EventMeta("Evt1", (), {})
        assert isinstance(cls._metaclass_steps, list)
        assert len(cls._metaclass_steps) > 0

    def test_tag_event_id_and_name(self):
        cls = EventMeta("Evt2", (), {})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="event_id")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="event_name")

    def test_describe_fields(self):
        cls = EventMeta("Evt3", (), {"__annotations__": {"msg": str}})
        assert _has_step(cls._metaclass_steps, Op.DESCRIBE, field="msg")

    def test_tag_parents(self):
        cls = EventMeta("Evt4", (), {})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="event_parents")

    def test_tag_defaults(self):
        cls = EventMeta("Evt5", (), {})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="event_priority")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="event_channels")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="event_pooled")

    def test_validate_step(self):
        cls = EventMeta("Evt6", (), {})
        assert _has_step(cls._metaclass_steps, Op.VALIDATE, constraint="event_data_only")

    def test_register_event_registry(self):
        cls = EventMeta("Evt7", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="event_registry")

    def test_parent_register_preserved(self):
        cls = EventMeta("Evt8", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_pooled_event_hook_steps(self):
        cls = EventMeta("Evt9", (), {"_event_pooled": True})
        assert _has_step(cls._metaclass_steps, Op.HOOK, event="pool_acquire")
        assert _has_step(cls._metaclass_steps, Op.HOOK, event="pool_release")

    def test_non_pooled_no_hook(self):
        cls = EventMeta("Evt10", (), {})
        assert not _has_step(cls._metaclass_steps, Op.HOOK, event="pool_acquire")

    def test_base_event_skipped(self):
        cls = EventMeta("Event", (), {})
        assert cls._metaclass_steps == []


# ===========================================================================
# StateMeta
# ===========================================================================

class TestStateMetaSteps:
    def test_has_metaclass_steps(self):
        cls = StateMeta("St1", (), {})
        assert isinstance(cls._metaclass_steps, list)
        assert len(cls._metaclass_steps) > 0

    def test_tag_state_id_and_name(self):
        cls = StateMeta("St2", (), {})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="state_id")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="state_name")

    def test_tag_transitions(self):
        cls = StateMeta("St3", (), {})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="state_transitions")

    def test_register_global(self):
        cls = StateMeta("St4", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="state_global")

    def test_parent_register_preserved(self):
        cls = StateMeta("St5", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_on_enter_hook(self):
        def my_enter():
            pass
        cls = StateMeta("St6", (), {"_state_on_enter": my_enter})
        assert _has_step(cls._metaclass_steps, Op.HOOK, event="on_enter")

    def test_on_exit_hook(self):
        def my_exit():
            pass
        cls = StateMeta("St7", (), {"_state_on_exit": my_exit})
        assert _has_step(cls._metaclass_steps, Op.HOOK, event="on_exit")

    def test_no_hooks_by_default(self):
        cls = StateMeta("St8", (), {})
        assert not _has_step(cls._metaclass_steps, Op.HOOK, event="on_enter")
        assert not _has_step(cls._metaclass_steps, Op.HOOK, event="on_exit")

    def test_machine_register(self):
        class FakeMachine:
            pass
        cls = StateMeta("St9", (), {"_state_machine_cls": FakeMachine})
        assert _has_step(cls._metaclass_steps, Op.REGISTER,
                         registry=f"state_machine:{FakeMachine.__name__}")

    def test_base_state_skipped(self):
        cls = StateMeta("State", (), {})
        assert cls._metaclass_steps == []


# ===========================================================================
# AssetMeta
# ===========================================================================

class TestAssetMetaSteps:
    def _make_asset(self, name, ns=None):
        base_ns = {"_asset_extensions": (".test",)}
        if ns:
            base_ns.update(ns)
        return AssetMeta(name, (), base_ns)

    def test_has_metaclass_steps(self):
        cls = self._make_asset("Ast1")
        assert isinstance(cls._metaclass_steps, list)
        assert len(cls._metaclass_steps) > 0

    def test_tag_asset_id(self):
        cls = self._make_asset("Ast2", {"_asset_extensions": (".ast2",)})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="asset_id")

    def test_tag_asset_type_code(self):
        cls = self._make_asset("Ast3", {"_asset_extensions": (".ast3",)})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="asset_type_code")

    def test_validate_extensions(self):
        cls = self._make_asset("Ast4", {"_asset_extensions": (".ast4",)})
        assert _has_step(cls._metaclass_steps, Op.VALIDATE, constraint="asset_extensions_required")
        assert _has_step(cls._metaclass_steps, Op.VALIDATE, constraint="extension_uniqueness")

    def test_tag_extensions(self):
        cls = self._make_asset("Ast5", {"_asset_extensions": (".ast5",)})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="extensions")

    def test_tag_defaults(self):
        cls = self._make_asset("Ast6", {"_asset_extensions": (".ast6",)})
        assert _has_step(cls._metaclass_steps, Op.TAG, key="cache_policy")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="hot_reload")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="asset_priority")

    def test_register_extension_map(self):
        cls = self._make_asset("Ast7", {"_asset_extensions": (".ast7",)})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="asset_extension_map")

    def test_register_asset_registry(self):
        cls = self._make_asset("Ast8", {"_asset_extensions": (".ast8",)})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="asset_registry")

    def test_parent_register_preserved(self):
        cls = self._make_asset("Ast9", {"_asset_extensions": (".ast9",)})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_base_asset_skipped(self):
        cls = AssetMeta("Asset", (), {})
        assert cls._metaclass_steps == []


# ===========================================================================
# ProtocolMeta
# ===========================================================================

class TestProtocolMetaSteps:
    def _make_proto(self, name, ns=None):
        base_ns = {"_protocol_version": 1}
        if ns:
            base_ns.update(ns)
        return ProtocolMeta(name, (), base_ns)

    def test_has_metaclass_steps(self):
        cls = self._make_proto("Proto1")
        assert isinstance(cls._metaclass_steps, list)
        assert len(cls._metaclass_steps) > 0

    def test_tag_protocol_id_and_name(self):
        cls = self._make_proto("Proto2")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="protocol_id")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="protocol_name")

    def test_validate_version(self):
        cls = self._make_proto("Proto3")
        assert _has_step(cls._metaclass_steps, Op.VALIDATE, constraint="protocol_version_valid")

    def test_tag_version(self):
        cls = self._make_proto("Proto4")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="protocol_version", value=1)

    def test_tag_min_version(self):
        cls = self._make_proto("Proto5")
        assert _has_step(cls._metaclass_steps, Op.TAG, key="protocol_min_version")

    def test_validate_min_lte_version(self):
        cls = self._make_proto("Proto6")
        assert _has_step(cls._metaclass_steps, Op.VALIDATE, constraint="min_version_lte_version")

    def test_register_protocol_registry(self):
        cls = self._make_proto("Proto7")
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="protocol_registry")

    def test_parent_register_preserved(self):
        cls = self._make_proto("Proto8")
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_base_protocol_skipped(self):
        cls = ProtocolMeta("Protocol", (), {})
        assert cls._metaclass_steps == []


# ===========================================================================
# Cross-cutting: parent REGISTER(engine_types) step preserved
# ===========================================================================

class TestParentStepsPreserved:
    """Test that child metaclasses preserve EngineMeta's REGISTER step."""

    def test_component_preserves_engine_register(self):
        cls = ComponentMeta("CrossComp", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_system_preserves_engine_register(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls = SystemMeta("CrossSys", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_resource_preserves_engine_register(self):
        cls = ResourceMeta("CrossRes", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_event_preserves_engine_register(self):
        cls = EventMeta("CrossEvt", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_state_preserves_engine_register(self):
        cls = StateMeta("CrossSt", (), {})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_asset_preserves_engine_register(self):
        cls = AssetMeta("CrossAst", (), {"_asset_extensions": (".cross",)})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_protocol_preserves_engine_register(self):
        cls = ProtocolMeta("CrossPr", (), {"_protocol_version": 1})
        assert _has_step(cls._metaclass_steps, Op.REGISTER, registry="engine_types")

    def test_engine_register_is_first_step(self):
        """The REGISTER(engine_types) step from EngineMeta should be first."""
        cls = ComponentMeta("CrossComp2", (), {})
        first = cls._metaclass_steps[0]
        assert first.op == Op.REGISTER
        assert first.args.get("registry") == "engine_types"


# ===========================================================================
# Step count sanity checks
# ===========================================================================

class TestStepCounts:
    def test_engine_meta_one_step(self):
        """EngineMeta adds exactly 1 step (REGISTER) for non-base classes."""
        cls = EngineMeta("CountEng", (), {})
        assert len(cls._metaclass_steps) == 1

    def test_resource_meta_step_count(self):
        """ResourceMeta: 1(parent REGISTER) + 4 TAGs + 1 REGISTER + 1 HOOK = 7."""
        cls = ResourceMeta("CountRes", (), {})
        assert len(cls._metaclass_steps) == 7

    def test_event_basic_step_count(self):
        """EventMeta with no fields, not pooled: parent + 2 TAG id/name + 1 TAG parents +
        3 TAG defaults + 1 VALIDATE + 1 REGISTER = 9."""
        cls = EventMeta("CountEvt", (), {})
        assert len(cls._metaclass_steps) == 9

    def test_event_pooled_adds_two_hooks(self):
        cls_plain = EventMeta("CountEvtP1", (), {})
        cls_pooled = EventMeta("CountEvtP2", (), {"_event_pooled": True})
        assert len(cls_pooled._metaclass_steps) == len(cls_plain._metaclass_steps) + 2
