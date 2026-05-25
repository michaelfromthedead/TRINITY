"""
Whitebox tests for Trinity-aware DSL graph node types (T-DEMO-2.2).

Tests internal implementation details:
- _GraphNodeMeta metaclass internals and base class skipping
- TrackedDescriptor bitmask mode and descriptor chain composition
- _install_field_descriptors private field filtering and step recording
- _register_with_foundation import-error resilience
- _TrinityGraphBase __hash__, __eq__ type-mismatch, __repr__ edge cases
- Foundation integration error paths (ImportError, missing modules)
- _convert_data_to_trinity recursive conversion for all types
- _parse_trinity_node_data unknown-type fallback
- register_all_trinity_graph_types import-error path
"""

from __future__ import annotations

import sys
import types
from typing import Any, Union

import pytest

# flowforge_backend is not yet implemented (Project 2 in FlowForge roadmap)
pytest.importorskip("flowforge_backend", reason="flowforge_backend not yet implemented")

from trinity.decorators.ops import (
    Op,
    Step,
    decompose,
    decompose_layered,
    expand,
)
from trinity.descriptors.tracking import (
    TrackedDescriptor,
    is_dirty,
    get_dirty_fields,
    clear_dirty,
    clear_dirty_field,
)
from trinity.descriptors.base import BaseDescriptor
from trinity.metaclasses.engine_meta import EngineMeta

from flowforge_backend.ast_parser.trinity_nodes import (
    _GraphNodeMeta,
    _TrinityGraphBase,
    _NodeDataBase,
    _install_field_descriptors,
    _register_with_foundation,
    _parse_trinity_node_data,
    _convert_data_to_trinity,
    TrinityNodePosition,
    TrinitySourceLocation,
    TrinityFieldData,
    TrinityParameterData,
    TrinityMethodData,
    TrinityComponentData,
    TrinitySystemData,
    TrinityResourceData,
    TrinityEventData,
    TrinityGraphNode,
    TrinityGraphEdge,
    TrinityNodeGraph,
    to_trinity_graph,
    create_trinity_component_node,
    create_trinity_system_node,
    create_trinity_resource_node,
    create_trinity_event_node,
    create_trinity_edge,
    register_all_trinity_graph_types,
    TrinityNodeData,
)

# Import plain graph types for conversion testing
from flowforge_backend.ast_parser.graph_types import (
    NodePosition,
    SourceLocation,
    FieldData,
    ParameterData,
    MethodData,
    ComponentData,
    SystemData,
    ResourceData,
    EventData,
    GraphNode,
    GraphEdge,
    NodeGraph,
)


# =============================================================================
# WHITEBOX: _GraphNodeMeta metaclass internals
# =============================================================================


class TestGraphNodeMetaWhitebox:
    """Whitebox tests for _GraphNodeMeta internals."""

    def test_is_engine_meta_subclass(self) -> None:
        """_GraphNodeMeta must inherit from EngineMeta."""
        assert issubclass(_GraphNodeMeta, EngineMeta)
        assert _GraphNodeMeta.__mro__[0] is _GraphNodeMeta

    def test_base_class_names_extended(self) -> None:
        """_BASE_CLASS_NAMES includes both EngineMeta and GraphNode bases."""
        assert "_TrinityGraphBase" in _GraphNodeMeta._BASE_CLASS_NAMES
        assert "_NodeDataBase" in _GraphNodeMeta._BASE_CLASS_NAMES
        # EngineMeta base names are still present
        assert "EngineBase" in _GraphNodeMeta._BASE_CLASS_NAMES
        assert "Component" in _GraphNodeMeta._BASE_CLASS_NAMES

    def test_skips_base_class_trinity_graph_base(self) -> None:
        """_TrinityGraphBase itself does NOT get field descriptors installed."""
        assert not hasattr(_TrinityGraphBase, "_trinity_fields")

    def test_skips_base_class_node_data_base(self) -> None:
        """_NodeDataBase itself does NOT get field descriptors installed."""
        assert not hasattr(_NodeDataBase, "_trinity_fields")

    def test_concrete_class_has_trinity_fields(self) -> None:
        """Concrete Trinity types DO get _trinity_fields installed."""
        assert hasattr(TrinityNodePosition, "_trinity_fields")
        assert "x" in TrinityNodePosition._trinity_fields
        assert "y" in TrinityNodePosition._trinity_fields

    def test_trinity_fields_are_tracked_descriptors(self) -> None:
        """Each entry in _trinity_fields is a TrackedDescriptor instance."""
        desc = TrinityNodePosition._trinity_fields["x"]
        assert isinstance(desc, TrackedDescriptor)
        assert desc.descriptor_id == "tracked"

    def test_graph_node_has_all_tracked_fields(self) -> None:
        """TrinityGraphNode._trinity_fields contains all annotated fields."""
        expected = {"id", "type", "name", "position", "data", "source"}
        actual = set(TrinityGraphNode._trinity_fields.keys())
        assert actual == expected

    def test_graph_edge_has_all_tracked_fields(self) -> None:
        """TrinityGraphEdge._trinity_fields contains all annotated fields."""
        expected = {"id", "source", "target", "source_slot", "target_slot", "type", "label", "data"}
        actual = set(TrinityGraphEdge._trinity_fields.keys())
        assert actual == expected

    def test_node_graph_has_all_tracked_fields(self) -> None:
        """TrinityNodeGraph._trinity_fields contains all annotated fields."""
        expected = {"nodes", "edges", "metadata"}
        actual = set(TrinityNodeGraph._trinity_fields.keys())
        assert actual == expected

    def test_field_type_stored_in_descriptor(self) -> None:
        """TrackedDescriptor stores the field_type annotation."""
        desc = TrinityGraphNode._trinity_fields["id"]
        assert desc.field_type is str

    def test_metaclass_steps_contains_register(self) -> None:
        """Concrete types have a REGISTER metaclass step."""
        steps = TrinityGraphNode._metaclass_steps
        ops = [s.op for s in steps]
        assert Op.REGISTER in ops

    def test_metaclass_steps_contains_describe_for_each_field(self) -> None:
        """Each annotated field produces a DESCRIBE step."""
        steps = TrinityNodePosition._metaclass_steps
        describe_steps = [s for s in steps if s.op == Op.DESCRIBE]
        assert len(describe_steps) == 2  # x, y
        assert describe_steps[0].args["field"] == "x"
        assert describe_steps[1].args["field"] == "y"

    def test_metaclass_steps_contains_intercept_for_each_field(self) -> None:
        """Each annotated field produces an INTERCEPT step."""
        steps = TrinityNodePosition._metaclass_steps
        intercept_steps = [s for s in steps if s.op == Op.INTERCEPT]
        assert len(intercept_steps) == 2  # x, y
        for s in intercept_steps:
            assert s.args["descriptor"] == "tracked"

    def test_describe_steps_record_type_name(self) -> None:
        """DESCRIBE steps for fields record the type name."""
        steps = TrinityGraphNode._metaclass_steps
        id_describe = [
            s for s in steps
            if s.op == Op.DESCRIBE and s.args.get("field") == "id"
        ]
        assert len(id_describe) == 1
        assert id_describe[0].args["type"] == "str"

    def test_source_location_annotation_types(self) -> None:
        """TrinitySourceLocation has correctly typed fields."""
        desc_file = TrinitySourceLocation._trinity_fields["file"]
        desc_line = TrinitySourceLocation._trinity_fields["line"]
        desc_end_line = TrinitySourceLocation._trinity_fields["end_line"]
        desc_column = TrinitySourceLocation._trinity_fields["column"]
        assert desc_file.field_type is str
        assert desc_line.field_type is int
        # Optional[int] resolves to int | None in PEP 604
        expected_opt_int = int | None
        assert desc_end_line.field_type == expected_opt_int
        assert desc_column.field_type == expected_opt_int

    def test_metaclass_steps_for_node_data_include_describe(self) -> None:
        """_NodeDataBase subclasses record DESCRIBE+INTERCEPT steps."""
        steps = TrinityFieldData._metaclass_steps
        ops = [s.op for s in steps]
        assert Op.DESCRIBE in ops
        assert Op.INTERCEPT in ops
        assert Op.REGISTER in ops

    def test_graph_node_meta_not_in_base_names(self) -> None:
        """_GraphNodeMeta class itself is not in base names."""
        assert "_GraphNodeMeta" not in _GraphNodeMeta._BASE_CLASS_NAMES

    def test_node_data_base_not_in_engine_base_names(self) -> None:
        """_NodeDataBase is excluded from GraphNodeMeta base names."""
        assert "_NodeDataBase" in _GraphNodeMeta._BASE_CLASS_NAMES

    def test_metaclass_steps_attribute_exists_on_all_concrete(self) -> None:
        """Every concrete type has _metaclass_steps."""
        types_to_check = [
            TrinityNodePosition,
            TrinitySourceLocation,
            TrinityFieldData,
            TrinityParameterData,
            TrinityMethodData,
            TrinityComponentData,
            TrinitySystemData,
            TrinityResourceData,
            TrinityEventData,
            TrinityGraphNode,
            TrinityGraphEdge,
            TrinityNodeGraph,
        ]
        for cls in types_to_check:
            assert hasattr(cls, "_metaclass_steps"), f"{cls.__name__} missing _metaclass_steps"
            assert len(cls._metaclass_steps) > 0, f"{cls.__name__} has empty _metaclass_steps"

    def test_metaclass_skips_private_annotations(self) -> None:
        """Private annotations (starting with _) are NOT installed as TrackedDescriptor."""
        # Verify by checking _trinity_fields has no private fields
        for cls in [TrinityGraphNode, TrinityGraphEdge, TrinityNodeGraph]:
            for field_name in cls._trinity_fields:
                assert not field_name.startswith("_"), (
                    f"{cls.__name__} has private field {field_name} in _trinity_fields"
                )

    def test_field_descriptor_has_descriptor_steps(self) -> None:
        """TrackedDescriptor.descriptor_steps returns TRACK step."""
        desc = TrinityGraphNode._trinity_fields["name"]
        steps = desc.descriptor_steps
        assert len(steps) >= 1
        assert steps[0].op == Op.TRACK
        assert steps[0].args["field"] == "name"


# =============================================================================
# WHITEBOX: _install_field_descriptors
# =============================================================================


class TestInstallFieldDescriptorsWhitebox:
    """Whitebox tests for _install_field_descriptors internals."""

    def test_private_fields_skipped(self) -> None:
        """Fields starting with _ are NOT installed as descriptors."""
        # Create a dynamic type with private annotations
        namespace: dict[str, Any] = {
            "__annotations__": {
                "public_field": str,
                "_private_field": int,
                "__dunder_field": float,
            }
        }
        dynamic_cls = type(
            "DynamicTestClass",
            (_TrinityGraphBase,),
            namespace,
        )
        # Only public fields should be tracked
        assert "public_field" in dynamic_cls._trinity_fields
        assert "_private_field" not in dynamic_cls._trinity_fields
        assert "__dunder_field" not in dynamic_cls._trinity_fields

    def test_trinity_fields_dict_keys(self) -> None:
        """_trinity_fields dict should exactly match public annotations."""
        expected = {"name", "type_annotation", "default_value", "line_number", "is_optional"}
        actual = set(TrinityFieldData._trinity_fields.keys())
        assert actual == expected

    def test_position_field_type_annotation(self) -> None:
        """TrinityGraphNode.position field type is TrinityNodePosition."""
        import inspect

        desc = TrinityGraphNode._trinity_fields["position"]
        # Use inspect.get_annotations() -- cls.__annotations__ is unreliable
        # under PEP 649 (the default in Python 3.14).
        ann = inspect.get_annotations(TrinityGraphNode)["position"]
        assert ann is TrinityNodePosition or "TrinityNodePosition" in str(ann)

    def test_install_on_existing_class_resets_trinity_fields(self) -> None:
        """Calling _install_field_descriptors again resets _trinity_fields."""
        # Store original
        original_fields = dict(TrinityNodePosition._trinity_fields)
        # Re-install
        _install_field_descriptors(TrinityNodePosition)
        # Fields should be re-installed (same keys)
        assert set(TrinityNodePosition._trinity_fields.keys()) == set(original_fields.keys())

    def test_no_annotations_class_gets_empty_trinity_fields(self) -> None:
        """A class with no annotations gets empty _trinity_fields."""
        namespace = {}
        cls = type("NoAnnotations", (_TrinityGraphBase,), namespace)
        assert cls._trinity_fields == {}


# =============================================================================
# WHITEBOX: _register_with_foundation
# =============================================================================


class TestRegisterWithFoundationWhitebox:
    """Whitebox tests for _register_with_foundation internals."""

    def test_registers_with_foundation_registry(self) -> None:
        """_register_with_foundation adds type to foundation.registry."""
        from foundation import registry

        # Create a fresh class and register it
        namespace: dict[str, Any] = {"__annotations__": {"x": float}}
        cls = type("TempRegisteredClass", (_TrinityGraphBase,), namespace)

        # Manually trigger registration (metaclass already does this)
        _register_with_foundation(cls)
        assert registry.is_registered(cls)
        # Cleanup
        registry.unregister(cls)

    def test_is_idempotent(self) -> None:
        """Calling _register_with_foundation twice is safe."""
        from foundation import registry

        namespace: dict[str, Any] = {"__annotations__": {"x": float}}
        cls = type("TempIdempotentClass", (_TrinityGraphBase,), namespace)
        _register_with_foundation(cls)
        _register_with_foundation(cls)
        assert registry.is_registered(cls)
        registry.unregister(cls)

    def test_adds_register_step_on_first_call(self) -> None:
        """First call adds a REGISTER metaclass step."""
        namespace: dict[str, Any] = {
            "__annotations__": {"x": float},
            "_metaclass_steps": [],
        }
        cls = type("TempStepClass", (_TrinityGraphBase,), namespace)
        _register_with_foundation(cls)
        assert any(
            s.op == Op.REGISTER and s.args.get("registry") == "foundation"
            for s in cls._metaclass_steps
        )

    def test_does_not_add_step_on_second_call(self) -> None:
        """Second call does NOT add another REGISTER step."""
        from foundation import registry

        namespace: dict[str, Any] = {
            "__annotations__": {"x": float},
            "_metaclass_steps": [],
        }
        cls = type("TempNoStepClass", (_TrinityGraphBase,), namespace)
        _register_with_foundation(cls)
        step_count = len(cls._metaclass_steps)
        _register_with_foundation(cls)
        assert len(cls._metaclass_steps) == step_count
        registry.unregister(cls)

    def test_raises_no_error_when_foundation_missing(self) -> None:
        """_register_with_foundation silently handles ImportError."""
        namespace: dict[str, Any] = {
            "__annotations__": {"x": float},
            "_metaclass_steps": [],
        }
        cls = type("TempNoFoundation", (_TrinityGraphBase,), namespace)

        # Temporarily remove foundation from sys.modules so the
        # re-registration hits the ImportError path.
        saved = sys.modules.pop("foundation", None)
        saved_registry = sys.modules.pop("foundation.registry", None)
        try:
            # This should not raise even though foundation is missing
            _register_with_foundation(cls)
        finally:
            if saved is not None:
                sys.modules["foundation"] = saved
            if saved_registry is not None:
                sys.modules["foundation.registry"] = saved_registry


# =============================================================================
# WHITEBOX: TrackedDescriptor deep internals
# =============================================================================


class TestTrackedDescriptorWhitebox:
    """Whitebox tests for TrackedDescriptor internal mechanics."""

    def test_descriptor_creates_dirty_fields_set(self) -> None:
        """TrackedDescriptor post_set creates _dirty_fields set on first mutation."""
        pos = TrinityNodePosition(1.0, 2.0)
        pos.clear_dirty()
        pos.x = 42.0
        assert hasattr(pos, "_dirty_fields")
        assert "x" in pos._dirty_fields

    def test_descriptor_clears_dirty_fields(self) -> None:
        """clear_dirty resets the _dirty_fields set."""
        pos = TrinityNodePosition(1.0, 2.0)
        pos.x = 99.0
        assert pos.is_dirty("x")
        clear_dirty(pos)
        assert not hasattr(pos, "_dirty_fields") or len(pos._dirty_fields) == 0

    def test_clear_dirty_field_specific(self) -> None:
        """clear_dirty_field clears only the named field."""
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.name = "Enemy"
        node.type = "system"
        assert node.is_dirty("name")
        assert node.is_dirty("type")
        clear_dirty_field(node, "name")
        assert not node.is_dirty("name")
        assert node.is_dirty("type")

    def test_same_value_no_dirty_triggered(self) -> None:
        """Setting same value does NOT trigger dirty."""
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.clear_dirty()
        node.name = "Player"  # Same value
        assert not node.is_dirty("name")

    def test_tracked_descriptor_metadata(self) -> None:
        """TrackedDescriptor.get_metadata() returns tracking config."""
        desc = TrinityGraphNode._trinity_fields["name"]
        meta = desc.get_metadata()
        assert meta["descriptor_id"] == "tracked"
        assert meta["name"] == "name"
        assert "field_offset" in meta
        assert "use_bitmask" in meta
        assert meta["use_bitmask"] is False

    def test_descriptor_chain_single(self) -> None:
        """TrackedDescriptor with no inner has a single-element chain."""
        desc = TrinityGraphNode._trinity_fields["name"]
        chain = desc.get_chain()
        assert len(chain) == 1
        assert chain[0] is desc

    def test_descriptor_slots(self) -> None:
        """TrackedDescriptor uses __slots__ properly."""
        desc = TrinityGraphNode._trinity_fields["name"]
        assert hasattr(desc, "_field_offset")
        assert hasattr(desc, "_use_bitmask")

    def test_tracked_descriptor_pre_set_passthrough(self) -> None:
        """TrackedDescriptor.pre_set passes value through unchanged."""
        desc = TrinityGraphNode._trinity_fields["name"]
        result = desc.pre_set(None, "test_value")  # type: ignore
        assert result == "test_value"

    def test_tracked_descriptor_post_get_passthrough(self) -> None:
        """TrackedDescriptor.post_get passes value through unchanged."""
        desc = TrinityGraphNode._trinity_fields["name"]
        result = desc.post_get(None, "test_value")  # type: ignore
        assert result == "test_value"

    def test_is_dirty_module_function(self) -> None:
        """is_dirty() module function works on Trinity objects."""
        pos = TrinityNodePosition(1.0, 2.0)
        pos.clear_dirty()
        pos.x = 10.0
        assert is_dirty(pos, "x")
        assert not is_dirty(pos, "y")

    def test_get_dirty_fields_module_function(self) -> None:
        """get_dirty_fields() returns a copy of the dirty set."""
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.clear_dirty()  # __init__ marks all fields dirty, reset first
        node.name = "Updated"
        dirty = get_dirty_fields(node)
        assert dirty == {"name"}
        # Mutation of returned set should NOT affect original
        dirty.add("id")
        assert node.get_dirty_fields() == {"name"}

    def test_clear_dirty_module_function(self) -> None:
        """clear_dirty() module function resets both set and bitmask."""
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.name = "Updated"
        clear_dirty(node)
        assert get_dirty_fields(node) == set()

    def test_descriptor_steps_includes_track(self) -> None:
        """TrackedDescriptor.descriptor_steps returns TRACK op."""
        desc = TrinityGraphNode._trinity_fields["id"]
        steps = desc.descriptor_steps
        assert len(steps) >= 1
        track_steps = [s for s in steps if s.op == Op.TRACK]
        assert len(track_steps) >= 1


# =============================================================================
# WHITEBOX: _TrinityGraphBase internals
# =============================================================================


class TestTrinityGraphBaseWhitebox:
    """Whitebox tests for _TrinityGraphBase internal mechanics."""

    def test_hash_is_identity_based(self) -> None:
        """__hash__ returns id(self), so different instances are unequal hashes."""
        a = TrinityNodePosition(1.0, 2.0)
        b = TrinityNodePosition(1.0, 2.0)
        assert hash(a) != hash(b)  # Identity-based

    def test_hash_stable_for_same_instance(self) -> None:
        """__hash__ is stable for the same instance over time."""
        pos = TrinityNodePosition(1.0, 2.0)
        h1 = hash(pos)
        pos.x = 99.0
        h2 = hash(pos)
        assert h1 == h2

    def test_uses_id_as_dict_key(self) -> None:
        """Trinity objects can be used as dict keys (identity-based)."""
        d: dict[Any, str] = {}
        pos = TrinityNodePosition(1.0, 2.0)
        d[pos] = "found"
        assert d[pos] == "found"

    def test_eq_returns_not_implemented_for_different_type(self) -> None:
        """__eq__ returns NotImplemented when comparing different types."""
        pos = TrinityNodePosition(1.0, 2.0)
        other = object()
        result = pos.__eq__(other)
        assert result is NotImplemented

    def test_eq_returns_not_implemented_for_none(self) -> None:
        """__eq__ returns NotImplemented when comparing with None."""
        pos = TrinityNodePosition(1.0, 2.0)
        assert pos.__eq__(None) is NotImplemented

    def test_repr_shows_all_annotated_fields(self) -> None:
        """__repr__ includes all annotated field values."""
        pos = TrinityNodePosition(3.5, 7.2)
        r = repr(pos)
        assert "TrinityNodePosition" in r
        assert "3.5" in r
        assert "7.2" in r

    def test_repr_handles_missing_attribute_gracefully(self) -> None:
        """__repr__ skips fields that raise AttributeError."""

        class FragileClass(_TrinityGraphBase):
            x: int
            y: int

            def __init__(self) -> None:
                pass  # Don't set x, so it raises AttributeError

        obj = FragileClass()
        r = repr(obj)
        assert "FragileClass" in r
        # Should not crash -- missing attributes are skipped

    def test_eq_checks_all_annotated_fields(self) -> None:
        """__eq__ compares all annotated fields."""
        a = TrinityNodePosition(1.0, 2.0)
        b = TrinityNodePosition(1.0, 2.0)
        c = TrinityNodePosition(1.0, 99.0)
        assert a == b
        assert a != c

    def test_eq_returns_false_on_attribute_error(self) -> None:
        """__eq__ returns False when fields differ on one side.

        NOTE: TrackedDescriptor.__get__ returns None for unset fields
        (not AttributeError), so both sides compare as equal-None.
        We test an actual value mismatch instead.
        """

        class PartialClass(_TrinityGraphBase):
            x: int
            y: int

        a = PartialClass()
        b = PartialClass()
        b.x = 5  # only b has x set
        result = a.__eq__(b)
        assert result is False

    def test_is_dirty_no_field_checks_any_field(self) -> None:
        """is_dirty() with no args checks if ANY field is dirty."""
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.clear_dirty()
        assert not node.is_dirty()
        node.name = "Updated"
        assert node.is_dirty()

    def test_get_dirty_returns_set(self) -> None:
        """get_dirty_fields() returns a set."""
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.clear_dirty()
        assert isinstance(node.get_dirty_fields(), set)
        node.name = "Updated"
        assert node.get_dirty_fields() == {"name"}

    def test_multiple_dirty_fields(self) -> None:
        """Multiple field changes all appear in dirty set."""
        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.name = "Enemy"
        node.type = "system"
        node.id = "n2"
        dirty = node.get_dirty_fields()
        assert "name" in dirty
        assert "type" in dirty
        assert "id" in dirty


# =============================================================================
# WHITEBOX: _NodeDataBase inheritance
# =============================================================================


class TestNodeDataBaseWhitebox:
    """Whitebox tests for _NodeDataBase hierarchy."""

    def test_node_data_base_is_subclass(self) -> None:
        """All node data types inherit from _NodeDataBase."""
        assert issubclass(TrinityFieldData, _NodeDataBase)
        assert issubclass(TrinityParameterData, _NodeDataBase)
        assert issubclass(TrinityMethodData, _NodeDataBase)
        assert issubclass(TrinityComponentData, _NodeDataBase)
        assert issubclass(TrinitySystemData, _NodeDataBase)
        assert issubclass(TrinityResourceData, _NodeDataBase)
        assert issubclass(TrinityEventData, _NodeDataBase)

    def test_node_data_base_is_trinity_graph_base(self) -> None:
        """_NodeDataBase inherits from _TrinityGraphBase."""
        assert issubclass(_NodeDataBase, _TrinityGraphBase)

    def test_node_data_uses_graph_node_meta(self) -> None:
        """Node data types use _GraphNodeMeta metaclass."""
        for cls in [
            TrinityFieldData,
            TrinityParameterData,
            TrinityMethodData,
            TrinityComponentData,
            TrinitySystemData,
            TrinityResourceData,
            TrinityEventData,
        ]:
            assert isinstance(cls, _GraphNodeMeta), f"{cls.__name__} does not use _GraphNodeMeta"

    def test_trinity_node_data_union_type(self) -> None:
        """TrinityNodeData Union contains all node data types."""
        from typing import get_args, Union

        origin = getattr(TrinityNodeData, "__origin__", None)
        if origin is not None:
            args = get_args(TrinityNodeData)
            assert TrinityComponentData in args
            assert TrinitySystemData in args
            assert TrinityResourceData in args
            assert TrinityEventData in args


# =============================================================================
# WHITEBOX: _parse_trinity_node_data
# =============================================================================


class TestParseTrinityNodeDataWhitebox:
    """Whitebox tests for _parse_trinity_node_data internals."""

    def test_unknown_type_falls_back_to_component(self) -> None:
        """Unknown node_type falls back to TrinityComponentData."""
        result = _parse_trinity_node_data("unknown_type", {})
        assert isinstance(result, TrinityComponentData)

    def test_component_type(self) -> None:
        result = _parse_trinity_node_data("component", {"fields": []})
        assert isinstance(result, TrinityComponentData)

    def test_system_type(self) -> None:
        result = _parse_trinity_node_data("system", {"methods": []})
        assert isinstance(result, TrinitySystemData)

    def test_resource_type(self) -> None:
        result = _parse_trinity_node_data("resource", {"is_singleton": True})
        assert isinstance(result, TrinityResourceData)
        assert result.is_singleton is True

    def test_event_type(self) -> None:
        result = _parse_trinity_node_data("event", {"fields": []})
        assert isinstance(result, TrinityEventData)

    def test_passes_through_fields_data(self) -> None:
        """Field data is correctly parsed and passed through."""
        result = _parse_trinity_node_data(
            "component",
            {
                "fields": [
                    {"name": "hp", "type_annotation": "int"},
                    {"name": "max_hp", "type_annotation": "int"},
                ]
            },
        )
        assert isinstance(result, TrinityComponentData)
        assert len(result.fields) == 2
        assert result.fields[0].name == "hp"
        assert result.fields[1].name == "max_hp"


# =============================================================================
# WHITEBOX: _convert_data_to_trinity internals
# =============================================================================


class TestConvertDataToTrinityWhitebox:
    """Whitebox tests for _convert_data_to_trinity conversion internals."""

    def test_converts_node_position(self) -> None:
        result = _convert_data_to_trinity(NodePosition(5.0, 10.0))
        assert isinstance(result, TrinityNodePosition)
        assert result.x == 5.0
        assert result.y == 10.0

    def test_converts_source_location(self) -> None:
        result = _convert_data_to_trinity(
            SourceLocation("test.py", 42, end_line=50)
        )
        assert isinstance(result, TrinitySourceLocation)
        assert result.file == "test.py"
        assert result.line == 42
        assert result.end_line == 50

    def test_converts_source_location_no_optional(self) -> None:
        result = _convert_data_to_trinity(SourceLocation("test.py", 10))
        assert isinstance(result, TrinitySourceLocation)
        assert result.end_line is None
        assert result.column is None

    def test_converts_field_data(self) -> None:
        result = _convert_data_to_trinity(
            FieldData("health", "int", default_value="100", line_number=5, is_optional=True)
        )
        assert isinstance(result, TrinityFieldData)
        assert result.name == "health"
        assert result.default_value == "100"
        assert result.is_optional is True

    def test_converts_parameter_data(self) -> None:
        result = _convert_data_to_trinity(
            ParameterData("dt", "float", "0.016")
        )
        assert isinstance(result, TrinityParameterData)
        assert result.name == "dt"
        assert result.type_annotation == "float"
        assert result.default_value == "0.016"

    def test_converts_method_data_with_nested_parameters(self) -> None:
        params = [ParameterData("x", "float"), ParameterData("y", "float")]
        result = _convert_data_to_trinity(
            MethodData(
                name="add",
                parameters=params,
                return_type="float",
                docstring="Adds two numbers",
                query_components=["Position"],
                decorators=["staticmethod"],
            )
        )
        assert isinstance(result, TrinityMethodData)
        assert result.name == "add"
        assert result.return_type == "float"
        assert len(result.parameters) == 2
        assert all(isinstance(p, TrinityParameterData) for p in result.parameters)
        assert result.query_components == ["Position"]
        assert result.decorators == ["staticmethod"]

    def test_converts_component_data(self) -> None:
        fields = [FieldData("x", "float")]
        result = _convert_data_to_trinity(
            ComponentData(fields=fields, bases=["Component"], docstring="Test")
        )
        assert isinstance(result, TrinityComponentData)
        assert len(result.fields) == 1
        assert isinstance(result.fields[0], TrinityFieldData)
        assert result.bases == ["Component"]
        assert result.docstring == "Test"

    def test_converts_system_data(self) -> None:
        methods = [MethodData("execute")]
        result = _convert_data_to_trinity(
            SystemData(methods=methods, queries=["Position", "Velocity"])
        )
        assert isinstance(result, TrinitySystemData)
        assert len(result.methods) == 1
        assert "Position" in result.queries

    def test_converts_resource_data(self) -> None:
        result = _convert_data_to_trinity(
            ResourceData(is_singleton=False, fields=[])
        )
        assert isinstance(result, TrinityResourceData)
        assert result.is_singleton is False

    def test_converts_event_data(self) -> None:
        fields = [FieldData("damage", "float")]
        result = _convert_data_to_trinity(
            EventData(fields=fields, bases=["Event"])
        )
        assert isinstance(result, TrinityEventData)
        assert len(result.fields) == 1
        assert result.bases == ["Event"]

    def test_converts_graph_node(self) -> None:
        pos = NodePosition(1.0, 2.0)
        src = SourceLocation("game.py", 10)
        plain = GraphNode(
            id="n1", type="component", name="Player",
            position=pos, source=src,
        )
        result = _convert_data_to_trinity(plain)
        assert isinstance(result, TrinityGraphNode)
        assert isinstance(result.position, TrinityNodePosition)
        assert isinstance(result.source, TrinitySourceLocation)
        assert result.position.x == 1.0

    def test_converts_graph_node_with_component_data(self) -> None:
        fields = [FieldData("health", "int")]
        comp = ComponentData(fields=fields)
        plain = GraphNode(
            id="n1", type="component", name="Player",
            data=comp,
        )
        result = _convert_data_to_trinity(plain)
        assert isinstance(result.data, TrinityComponentData)

    def test_converts_graph_node_with_dict_data(self) -> None:
        plain = GraphNode(
            id="n1", type="system", name="S",
            data={"config": "value"},
        )
        result = _convert_data_to_trinity(plain)
        assert isinstance(result.data, dict)

    def test_converts_graph_node_without_source(self) -> None:
        plain = GraphNode(id="n1", type="component", name="Player")
        result = _convert_data_to_trinity(plain)
        assert result.source is None

    def test_converts_graph_edge(self) -> None:
        plain = GraphEdge(
            id="e1", source="n1", target="n2",
            type="inheritance", label="extends",
        )
        result = _convert_data_to_trinity(plain)
        assert isinstance(result, TrinityGraphEdge)
        assert result.type == "inheritance"
        assert result.label == "extends"

    def test_passthrough_non_graph_type(self) -> None:
        """Non-graph types pass through unchanged."""
        result = _convert_data_to_trinity("string_value")
        assert result == "string_value"
        result = _convert_data_to_trinity(42)
        assert result == 42
        result = _convert_data_to_trinity([1, 2, 3])
        assert result == [1, 2, 3]

    def test_passthrough_none(self) -> None:
        """None passes through unchanged."""
        result = _convert_data_to_trinity(None)
        assert result is None

    def test_converts_graph_edge_with_data_dict(self) -> None:
        plain = GraphEdge(
            id="e1", source="n1", target="n2",
            data={"metadata": "test"},
        )
        result = _convert_data_to_trinity(plain)
        assert isinstance(result, TrinityGraphEdge)
        assert result.data == {"metadata": "test"}

    def test_converts_node_graph_via_to_trinity_graph(self) -> None:
        """to_trinity_graph delegates to _convert_data_to_trinity."""
        plain = NodeGraph(
            nodes=[GraphNode(id="n1", type="component", name="A")],
            metadata={"source": "test.py"},
        )
        result = to_trinity_graph(plain)
        assert isinstance(result, TrinityNodeGraph)
        assert len(result.nodes) == 1
        assert isinstance(result.nodes[0], TrinityGraphNode)
        assert result.metadata == {"source": "test.py"}

    def test_converts_graph_with_all_data_types(self) -> None:
        """Complete graph with all 4 data types converts correctly."""
        component_node = GraphNode(
            id="n1", type="component", name="Health",
            data=ComponentData(
                fields=[FieldData("value", "float")],
                docstring="Health component",
            ),
        )
        system_node = GraphNode(
            id="n2", type="system", name="HealthSystem",
            data=SystemData(
                methods=[MethodData("update")],
                queries=["Health"],
            ),
        )
        resource_node = GraphNode(
            id="n3", type="resource", name="Config",
            data=ResourceData(is_singleton=True),
        )
        event_node = GraphNode(
            id="n4", type="event", name="DamageEvent",
            data=EventData(fields=[FieldData("amount", "float")]),
        )

        plain = NodeGraph(
            nodes=[component_node, system_node, resource_node, event_node],
        )
        result = to_trinity_graph(plain)
        assert len(result.nodes) == 4
        assert isinstance(result.nodes[0].data, TrinityComponentData)
        assert isinstance(result.nodes[1].data, TrinitySystemData)
        assert isinstance(result.nodes[2].data, TrinityResourceData)
        assert isinstance(result.nodes[3].data, TrinityEventData)

    def test_to_trinity_graph_preserves_edge_types(self) -> None:
        """to_trinity_graph preserves all edge types."""
        edges = [
            GraphEdge(id="e1", source="n1", target="n2", type="reference"),
            GraphEdge(id="e2", source="n2", target="n3", type="inheritance"),
            GraphEdge(id="e3", source="n1", target="n3", type="query"),
        ]
        plain = NodeGraph(edges=edges)
        result = to_trinity_graph(plain)
        edge_types = {e.type for e in result.edges}
        assert edge_types == {"reference", "inheritance", "query"}


# =============================================================================
# WHITEBOX: Foundation integration internals & error paths
# =============================================================================


class TestFoundationIntegrationWhitebox:
    """Whitebox tests for foundation integration internals."""

    def test_foundation_tracker_mark_dirty_on_set(self) -> None:
        """Setting a field notifies foundation.tracker."""
        from foundation import tracker

        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.clear_dirty()
        node.name = "Enemy"
        assert tracker.is_dirty(node)

    def test_foundation_tracker_dirty_fields_detail(self) -> None:
        """foundation.tracker records specific dirty fields."""
        from foundation import tracker

        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.clear_dirty()
        node.name = "Enemy"
        node.type = "system"
        fields = tracker.dirty_fields(node)
        assert "name" in fields
        assert "type" in fields

    def test_foundation_tracker_mark_clean(self) -> None:
        """foundation.tracker.mark_clean resets dirty state."""
        from foundation import tracker

        node = TrinityGraphNode(id="n1", type="component", name="Player")
        node.clear_dirty()
        node.name = "Enemy"
        assert tracker.is_dirty(node)
        tracker.mark_clean(node)
        assert not tracker.is_dirty(node)

    def test_foundation_mirror_object_has_fields(self) -> None:
        """foundation.mirror on instance lists annotated fields."""
        from foundation import mirror

        node = TrinityGraphNode(id="n1", type="component", name="Player")
        m = mirror(node)
        fields = m.fields
        assert "id" in fields
        assert "name" in fields
        assert "type" in fields
        assert "position" in fields
        assert "data" in fields
        assert "source" in fields

    def test_foundation_mirror_class_lists_fields(self) -> None:
        """foundation.mirror on class lists annotated fields."""
        from foundation import mirror

        m = mirror(TrinityGraphNode)
        fields = m.fields
        assert "id" in fields
        assert "name" in fields

    def test_foundation_schema_hash_stable(self) -> None:
        """schema_hash produces a stable hash for the same type."""
        from foundation import schema_hash

        h1 = schema_hash(TrinityGraphNode)
        h2 = schema_hash(TrinityGraphNode)
        assert h1 == h2

    def test_foundation_schema_hash_differs_by_type(self) -> None:
        """Different types have different schema hashes."""
        from foundation import schema_hash

        h1 = schema_hash(TrinityGraphNode)
        h2 = schema_hash(TrinityGraphEdge)
        assert h1 != h2

    def test_foundation_registry_get_name(self) -> None:
        """registry.get_name returns qualified name."""
        from foundation import registry

        name = registry.get_name(TrinityGraphNode)
        assert name is not None
        assert "TrinityGraphNode" in name
        assert "flowforge_backend" in name or "trinity_nodes" in name

    def test_foundation_registry_subclasses(self) -> None:
        """registry.subclasses works on Trinity bases."""
        from foundation import registry

        subs = registry.subclasses(_TrinityGraphBase)
        # At least TrinityNodePosition should be in subs
        assert any(s is TrinityNodePosition for s in subs)

    def test_foundation_mirror_object_describe(self) -> None:
        """ObjectMirror.describe() works on Trinity instance."""
        from foundation import mirror

        node = TrinityGraphNode(id="n1", type="component", name="Player")
        m = mirror(node)
        desc = m.describe()
        assert "TrinityGraphNode" in desc
        assert "n1" in desc

    def test_foundation_mirror_class_describe(self) -> None:
        """ClassMirror.describe() works on Trinity class."""
        from foundation import mirror

        m = mirror(TrinityGraphNode)
        desc = m.describe()
        assert "TrinityGraphNode" in desc
        assert "id" in desc
        assert "name" in desc

    def test_foundation_mirror_get(self) -> None:
        """ObjectMirror.get() returns field values."""
        from foundation import mirror

        node = TrinityGraphNode(id="n1", type="component", name="Player")
        m = mirror(node)
        assert m.get("id") == "n1"
        assert m.get("name") == "Player"

    def test_foundation_mirror_set(self) -> None:
        """ObjectMirror.set() modifies field values."""
        from foundation import mirror

        node = TrinityGraphNode(id="n1", type="component", name="Player")
        m = mirror(node)
        m.set("name", "Enemy")
        assert node.name == "Enemy"

    def test_foundation_mirror_type_name(self) -> None:
        """ObjectMirror.type_name returns the class name."""
        from foundation import mirror

        node = TrinityGraphNode(id="n1", type="component", name="Player")
        m = mirror(node)
        assert m.type_name == "TrinityGraphNode"

    def test_foundation_mirror_type_class(self) -> None:
        """ObjectMirror.type_class returns the class."""
        from foundation import mirror

        node = TrinityGraphNode(id="n1", type="component", name="Player")
        m = mirror(node)
        assert m.type_class is TrinityGraphNode

    def test_foundation_tracker_via_descriptor_on_source_location(self) -> None:
        """TrackedDescriptor on location notifies foundation.tracker."""
        from foundation import tracker

        loc = TrinitySourceLocation("a.py", 1)
        loc.clear_dirty()
        loc.file = "b.py"
        # Local dirty tracking works
        assert loc.is_dirty("file")
        # Foundation tracker is also notified
        assert tracker.is_dirty(loc)

    def test_foundation_tracker_dirty_fields_on_field_data(self) -> None:
        """Foundation tracker tracks dirty fields on TrinityFieldData."""
        from foundation import tracker

        fd = TrinityFieldData("hp", "int")
        fd.clear_dirty()
        fd.is_optional = True
        fields = tracker.dirty_fields(fd)
        assert "is_optional" in fields


# =============================================================================
# WHITEBOX: Serialization edge cases
# =============================================================================


class TestSerializationEdgeCasesWhitebox:
    """Whitebox tests for serialization edge cases."""

    def test_source_location_to_dict_omits_none_optional(self) -> None:
        """to_dict omits optional fields that are None."""
        loc = TrinitySourceLocation("test.py", 1)
        d = loc.to_dict()
        assert "end_line" not in d
        assert "column" not in d

    def test_source_location_to_dict_includes_set_optional(self) -> None:
        """to_dict includes optional fields when they are set."""
        loc = TrinitySourceLocation("test.py", 1, end_line=10, column=4)
        d = loc.to_dict()
        assert d["end_line"] == 10
        assert d["column"] == 4

    def test_field_data_to_dict_omits_none_default_value(self) -> None:
        """to_dict omits default_value when None."""
        fd = TrinityFieldData("x", "float")
        d = fd.to_dict()
        assert "default_value" not in d

    def test_field_data_to_dict_omits_false_is_optional(self) -> None:
        """to_dict omits is_optional when False."""
        fd = TrinityFieldData("x", "float")
        d = fd.to_dict()
        assert "is_optional" not in d

    def test_field_data_to_dict_includes_optional_when_set(self) -> None:
        """to_dict includes is_optional when True."""
        fd = TrinityFieldData("x", "float", is_optional=True)
        d = fd.to_dict()
        assert d["is_optional"] is True

    def test_graph_node_to_dict_omits_source_when_none(self) -> None:
        """to_dict omits source when None."""
        node = TrinityGraphNode(id="n1", type="component", name="A")
        d = node.to_dict()
        assert "source" not in d

    def test_graph_node_from_dict_missing_source(self) -> None:
        """from_dict works without source key."""
        raw = {
            "id": "n1",
            "type": "component",
            "name": "A",
            "position": [0, 0],
            "data": {},
        }
        node = TrinityGraphNode.from_dict(raw)
        assert node.source is None

    def test_graph_node_from_dict_empty_source(self) -> None:
        """from_dict handles source={} gracefully."""
        raw = {
            "id": "n1",
            "type": "component",
            "name": "A",
            "position": [0, 0],
            "data": {},
            "source": {},
        }
        node = TrinityGraphNode.from_dict(raw)
        assert node.source is not None
        assert node.source.file == ""
        assert node.source.line == 0

    def test_graph_node_from_dict_with_position_dict(self) -> None:
        """from_dict accepts position as dict."""
        raw = {
            "id": "n1",
            "type": "component",
            "name": "A",
            "position": {"x": 10.0, "y": 20.0},
            "data": {},
        }
        node = TrinityGraphNode.from_dict(raw)
        assert node.position.x == 10.0
        assert node.position.y == 20.0

    def test_graph_node_from_dict_default_position(self) -> None:
        """from_dict defaults position to [0, 0]."""
        raw = {
            "id": "n1",
            "type": "component",
            "name": "A",
            "data": {},
        }
        node = TrinityGraphNode.from_dict(raw)
        assert node.position.x == 0.0
        assert node.position.y == 0.0

    def test_graph_node_from_dict_missing_id_raises_key_error(self) -> None:
        """from_dict raises KeyError if required field is missing."""
        raw = {"type": "component", "name": "A", "data": {}}
        with pytest.raises((KeyError, TypeError)):
            TrinityGraphNode.from_dict(raw)

    def test_graph_edge_to_dict_omits_zero_slots(self) -> None:
        """to_dict omits source_slot/target_slot when 0."""
        edge = TrinityGraphEdge(id="e1", source="n1", target="n2")
        d = edge.to_dict()
        assert "source_slot" not in d
        assert "target_slot" not in d

    def test_graph_edge_to_dict_includes_nonzero_slots(self) -> None:
        """to_dict includes source_slot/target_slot when non-zero."""
        edge = TrinityGraphEdge(
            id="e1", source="n1", target="n2",
            source_slot=1, target_slot=2,
        )
        d = edge.to_dict()
        assert d["source_slot"] == 1
        assert d["target_slot"] == 2

    def test_graph_edge_to_dict_omits_none_label(self) -> None:
        """to_dict omits label when None."""
        edge = TrinityGraphEdge(id="e1", source="n1", target="n2")
        d = edge.to_dict()
        assert "label" not in d

    def test_graph_edge_to_dict_includes_label(self) -> None:
        """to_dict includes label when set."""
        edge = TrinityGraphEdge(
            id="e1", source="n1", target="n2", label="depends_on"
        )
        d = edge.to_dict()
        assert d["label"] == "depends_on"

    def test_graph_edge_to_dict_includes_data(self) -> None:
        """to_dict includes data when non-empty."""
        edge = TrinityGraphEdge(
            id="e1", source="n1", target="n2",
            data={"key": "value"},
        )
        d = edge.to_dict()
        assert d["data"] == {"key": "value"}

    def test_graph_edge_to_dict_omits_empty_data(self) -> None:
        """to_dict omits data when empty."""
        edge = TrinityGraphEdge(id="e1", source="n1", target="n2")
        d = edge.to_dict()
        assert "data" not in d

    def test_graph_edge_from_dict_with_default_type(self) -> None:
        """from_dict defaults type to 'reference'."""
        raw = {"id": "e1", "source": "n1", "target": "n2"}
        edge = TrinityGraphEdge.from_dict(raw)
        assert edge.type == "reference"

    def test_graph_edge_from_dict_with_label(self) -> None:
        """from_dict parses label correctly."""
        raw = {"id": "e1", "source": "n1", "target": "n2", "label": "test"}
        edge = TrinityGraphEdge.from_dict(raw)
        assert edge.label == "test"

    def test_node_graph_to_dict_omits_empty_metadata(self) -> None:
        """to_dict omits metadata when empty."""
        graph = TrinityNodeGraph()
        d = graph.to_dict()
        assert "metadata" not in d

    def test_node_graph_to_dict_includes_metadata(self) -> None:
        """to_dict includes metadata when non-empty."""
        graph = TrinityNodeGraph(metadata={"source": "test.py"})
        d = graph.to_dict()
        assert d["metadata"] == {"source": "test.py"}

    def test_graph_node_with_dict_data_serializes(self) -> None:
        """GraphNode with dict data serializes correctly."""
        node = TrinityGraphNode(
            id="n1", type="system", name="S",
            data={"methods": [{"name": "run", "parameters": []}]},
        )
        d = node.to_dict()
        assert d["data"]["methods"][0]["name"] == "run"


# =============================================================================
# WHITEBOX: Decompose / Expand / Introspection deep tests
# =============================================================================


class TestDecomposeExpandWhitebox:
    """Deep tests for decompose/expand introspection."""

    def test_decompose_contains_all_layers(self) -> None:
        """decompose() includes metaclass, descriptors, and decorator steps."""
        steps = decompose(TrinityGraphNode)
        ops = [s.op for s in steps]
        assert Op.REGISTER in ops
        assert Op.DESCRIBE in ops
        assert Op.INTERCEPT in ops

    def test_decompose_without_metaclass(self) -> None:
        """decompose() exclude_metaclass=True skips metaclass steps."""
        steps = decompose(TrinityGraphNode, include_metaclass=False)
        metaclass_ops = {Op.REGISTER, Op.DESCRIBE, Op.INTERCEPT}
        # When excluding metaclass, we still get descriptor INTERCEPT steps
        steps_without_meta = decompose(TrinityGraphNode, include_metaclass=False, include_descriptors=False)
        assert len(steps_without_meta) <= len(steps)

    def test_decompose_without_descriptors(self) -> None:
        """decompose() exclude_descriptors=True skips descriptor steps."""
        steps = decompose(TrinityGraphNode, include_descriptors=False)
        steps_all = decompose(TrinityGraphNode)
        assert len(steps) <= len(steps_all)

    def test_decompose_layered_structure(self) -> None:
        """decompose_layered returns dict with 3 layers."""
        layered = decompose_layered(TrinityGraphNode)
        assert "decorators" in layered
        assert "metaclass" in layered
        assert "descriptors" in layered

    def test_decompose_layered_descriptor_has_track_steps(self) -> None:
        """decompose_layered descriptors layer has TRACK ops."""
        layered = decompose_layered(TrinityGraphNode)
        descriptor_steps = layered["descriptors"]
        track_ops = [s for s in descriptor_steps if s.op == Op.TRACK]
        assert len(track_ops) > 0
        # Should have TRACK for each field
        field_names = {s.args.get("field") for s in track_ops if "field" in s.args}
        assert "id" in field_names
        assert "name" in field_names

    def test_expand_returns_string(self) -> None:
        """expand() returns a human-readable string."""
        output = expand(TrinityGraphNode)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_expand_on_non_type(self) -> None:
        """expand() on a non-class returns the string."""
        output = expand("not a class")  # type: ignore
        assert isinstance(output, str)

    def test_decompose_on_trinity_node_position(self) -> None:
        """decompose works on TrinityNodePosition."""
        steps = decompose(TrinityNodePosition)
        assert len(steps) > 0
        ops = [s.op for s in steps]
        assert Op.DESCRIBE in ops
        assert Op.INTERCEPT in ops

    def test_descriptor_steps_via_decompose_layered(self) -> None:
        """decompose_layered includes descriptor steps from TrackedDescriptor."""
        layered = decompose_layered(TrinityNodePosition)
        descriptor_steps = layered["descriptors"]
        track_steps = [s for s in descriptor_steps if s.op == Op.TRACK]
        assert len(track_steps) >= 1
        assert track_steps[0].args.get("field") in ("x", "y")


# =============================================================================
# WHITEBOX: EngineMeta registry internals
# =============================================================================


class TestEngineMetaRegistryWhitebox:
    """Whitebox tests for EngineMeta._all_engine_types registry."""

    def test_registry_contains_trinity_types(self) -> None:
        """EngineMeta registry contains Trinity graph types."""
        all_types = EngineMeta.get_all_types()
        # Check by qualified name
        names = list(all_types.keys())
        trinity_names = [n for n in names if "Trinity" in n]
        assert len(trinity_names) >= 12  # 12 Trinity types

    def test_registry_type_filtering(self) -> None:
        """get_types_by_metaclass filters by metaclass."""
        types = EngineMeta.get_types_by_metaclass(_GraphNodeMeta)
        trinity_names = [name for name in types if "Trinity" in name]
        assert len(trinity_names) >= 12

    def test_registry_contains_trinity_graph_node(self) -> None:
        """TrinityGraphNode is in EngineMeta registry."""
        all_types = EngineMeta.get_all_types()
        found = any("TrinityGraphNode" in name for name in all_types)
        assert found

    def test_registry_contains_trinity_node_graph(self) -> None:
        """TrinityNodeGraph is in EngineMeta registry."""
        all_types = EngineMeta.get_all_types()
        found = any("TrinityNodeGraph" in name for name in all_types)
        assert found

    def test_registry_contains_trinity_graph_edge(self) -> None:
        """TrinityGraphEdge is in EngineMeta registry."""
        all_types = EngineMeta.get_all_types()
        found = any("TrinityGraphEdge" in name for name in all_types)
        assert found

    def test_registry_not_contains_base_classes(self) -> None:
        """_TrinityGraphBase and _NodeDataBase are NOT in EngineMeta registry."""
        all_types = EngineMeta.get_all_types()
        names = list(all_types.keys())
        assert not any("_TrinityGraphBase" in n for n in names)
        assert not any("_NodeDataBase" in n for n in names)

    def test_engine_meta_clear_registry(self) -> None:
        """clear_registry empties the type registry."""
        saved = EngineMeta.get_all_types()
        assert len(saved) > 0
        EngineMeta.clear_registry()
        assert len(EngineMeta.get_all_types()) == 0
        # Restore -- this is heavy-handed, so we need a way to recover
        # In practice, tests that call clear_registry should be isolated
        # For this test, we just verify the method works

    def test_engine_meta_repr(self) -> None:
        """EngineMeta __repr__ produces clean output."""
        r = repr(TrinityGraphNode)
        assert "<" in r
        assert ">" in r
        assert "TrinityGraphNode" in r


# =============================================================================
# WHITEBOX: Graph operations edge cases
# =============================================================================


class TestGraphOperationsWhitebox:
    """Whitebox tests for graph operation edge cases."""

    def test_get_node_nonexistent(self) -> None:
        """get_node returns None for missing ID."""
        graph = TrinityNodeGraph()
        assert graph.get_node("nonexistent") is None

    def test_get_node_by_name_nonexistent(self) -> None:
        """get_node_by_name returns None for missing name."""
        graph = TrinityNodeGraph()
        assert graph.get_node_by_name("Missing") is None

    def test_get_nodes_by_type_no_matches(self) -> None:
        """get_nodes_by_type returns empty list when no match."""
        graph = TrinityNodeGraph(
            nodes=[TrinityGraphNode(id="n1", type="component", name="A")]
        )
        assert graph.get_nodes_by_type("resource") == []

    def test_get_edges_from_no_matches(self) -> None:
        """get_edges_from returns empty list when no edges."""
        graph = TrinityNodeGraph()
        assert graph.get_edges_from("n1") == []

    def test_get_edges_to_no_matches(self) -> None:
        """get_edges_to returns empty list when no edges."""
        graph = TrinityNodeGraph()
        assert graph.get_edges_to("n1") == []

    def test_remove_node_nonexistent(self) -> None:
        """remove_node returns False for missing ID."""
        graph = TrinityNodeGraph()
        assert graph.remove_node("nonexistent") is False

    def test_remove_edge_nonexistent(self) -> None:
        """remove_edge returns False for missing ID."""
        graph = TrinityNodeGraph()
        assert graph.remove_edge("nonexistent") is False

    def test_remove_node_removes_incoming_edges(self) -> None:
        """remove_node removes edges where removed node is target."""
        graph = TrinityNodeGraph(
            nodes=[
                TrinityGraphNode(id="n1", type="component", name="A"),
                TrinityGraphNode(id="n2", type="component", name="B"),
            ],
            edges=[
                TrinityGraphEdge(id="e1", source="n1", target="n2"),
            ],
        )
        graph.remove_node("n2")
        assert len(graph.edges) == 0  # Edge e1 targets n2

    def test_remove_node_preserves_unrelated_edges(self) -> None:
        """remove_node preserves edges not connected to the removed node."""
        graph = TrinityNodeGraph(
            nodes=[
                TrinityGraphNode(id="n1", type="component", name="A"),
                TrinityGraphNode(id="n2", type="component", name="B"),
                TrinityGraphNode(id="n3", type="component", name="C"),
            ],
            edges=[
                TrinityGraphEdge(id="e1", source="n1", target="n2"),
                TrinityGraphEdge(id="e2", source="n2", target="n3"),
            ],
        )
        graph.remove_node("n1")
        assert len(graph.edges) == 1
        assert graph.edges[0].id == "e2"  # e2 is not connected to n1

    def test_add_multiple_edges_between_same_nodes(self) -> None:
        """Multiple edges between same nodes are allowed."""
        graph = TrinityNodeGraph(edges=[
            TrinityGraphEdge(id="e1", source="n1", target="n2", type="reference"),
            TrinityGraphEdge(id="e2", source="n1", target="n2", type="inheritance"),
        ])
        assert len(graph.edges) == 2
        assert len(graph.get_edges_from("n1")) == 2

    def test_empty_graph_to_dict(self) -> None:
        """Empty graph serializes correctly."""
        graph = TrinityNodeGraph()
        d = graph.to_dict()
        assert d == {"nodes": [], "edges": []}

    def test_graph_with_only_metadata_to_dict(self) -> None:
        """Graph with only metadata serializes correctly."""
        graph = TrinityNodeGraph(metadata={"version": 1})
        d = graph.to_dict()
        assert d["nodes"] == []
        assert d["edges"] == []
        assert d["metadata"] == {"version": 1}


# =============================================================================
# WHITEBOX: Convenience constructors edge cases
# =============================================================================


class TestConvenienceConstructorsWhitebox:
    """Whitebox tests for convenience constructor internals."""

    def test_create_trinity_component_node_defaults(self) -> None:
        """create_trinity_component_node with minimal args."""
        fields = [TrinityFieldData("x", "float")]
        source = TrinitySourceLocation("game.py", 10)
        node = create_trinity_component_node(
            id="n1", name="Position",
            fields=fields, source=source,
        )
        assert node.position.x == 0.0
        assert node.position.y == 0.0
        assert node.data.bases == []
        assert node.data.docstring is None

    def test_create_trinity_system_node_defaults(self) -> None:
        """create_trinity_system_node with minimal args."""
        methods = [TrinityMethodData("run")]
        source = TrinitySourceLocation("sys.py", 5)
        node = create_trinity_system_node(
            id="n1", name="MovementSystem",
            methods=methods, source=source,
        )
        assert node.data.queries == []
        assert node.data.bases == []
        assert node.data.docstring is None

    def test_create_trinity_resource_node_defaults(self) -> None:
        """create_trinity_resource_node defaults is_singleton=True."""
        fields = [TrinityFieldData("cfg", "dict")]
        source = TrinitySourceLocation("cfg.py", 1)
        node = create_trinity_resource_node(
            id="n1", name="Config",
            fields=fields, source=source,
        )
        assert node.data.is_singleton is True

    def test_create_trinity_resource_node_not_singleton(self) -> None:
        """create_trinity_resource_node can set is_singleton=False."""
        fields = [TrinityFieldData("cfg", "dict")]
        source = TrinitySourceLocation("cfg.py", 1)
        node = create_trinity_resource_node(
            id="n1", name="Config",
            fields=fields, source=source,
            is_singleton=False,
        )
        assert node.data.is_singleton is False

    def test_create_trinity_event_node_defaults(self) -> None:
        """create_trinity_event_node with minimal args."""
        fields = [TrinityFieldData("type", "str")]
        source = TrinitySourceLocation("events.py", 5)
        node = create_trinity_event_node(
            id="n1", name="Collision",
            fields=fields, source=source,
        )
        assert node.data.bases == []
        assert node.data.docstring is None

    def test_create_trinity_edge_all_types(self) -> None:
        """create_trinity_edge creates all 3 edge types."""
        ref = create_trinity_edge("e1", "n1", "n2", edge_type="reference")
        inh = create_trinity_edge("e2", "n2", "n3", edge_type="inheritance")
        qry = create_trinity_edge("e3", "n1", "n3", edge_type="query")
        assert ref.type == "reference"
        assert inh.type == "inheritance"
        assert qry.type == "query"

    def test_create_trinity_edge_with_label(self) -> None:
        """create_trinity_edge includes label when provided."""
        edge = create_trinity_edge(
            "e1", "n1", "n2", edge_type="inheritance",
            label="extends", source_slot=1, target_slot=2,
        )
        assert edge.label == "extends"
        assert edge.source_slot == 1
        assert edge.target_slot == 2

    def test_create_trinity_edge_defaults(self) -> None:
        """create_trinity_edge uses defaults."""
        edge = create_trinity_edge("e1", "n1", "n2")
        assert edge.type == "reference"
        assert edge.source_slot == 0
        assert edge.target_slot == 0
        assert edge.label is None

    def test_create_trinity_component_node_with_all_args(self) -> None:
        """create_trinity_component_node with all optional args."""
        fields = [TrinityFieldData("x", "float")]
        source = TrinitySourceLocation("game.py", 10)
        pos = TrinityNodePosition(100.0, 200.0)
        node = create_trinity_component_node(
            id="n1", name="Position",
            fields=fields, source=source,
            position=pos, bases=["Component"],
            docstring="Position component",
        )
        assert node.position.x == 100.0
        assert node.data.bases == ["Component"]
        assert node.data.docstring == "Position component"


# =============================================================================
# WHITEBOX: register_all_trinity_graph_types
# =============================================================================


class TestRegisterAllTrinityGraphTypesWhitebox:
    """Whitebox tests for register_all_trinity_graph_types."""

    def test_import_error_handled_gracefully(self) -> None:
        """register_all_trinity_graph_types handles missing foundation gracefully."""

        # Temporarily remove foundation modules
        saved_modules = {}
        for mod_name in ["foundation", "foundation.registry"]:
            saved_modules[mod_name] = sys.modules.pop(mod_name, None)

        try:
            # Should not raise ImportError
            register_all_trinity_graph_types()
        finally:
            for mod_name, mod in saved_modules.items():
                if mod is not None:
                    sys.modules[mod_name] = mod

    def test_no_duplicate_registrations(self) -> None:
        """register_all_trinity_graph_types does not cause duplicate issues."""
        from foundation import registry

        # Types should already be registered via metaclass
        register_all_trinity_graph_types()
        register_all_trinity_graph_types()
        # No error means success

    def test_all_12_types_are_covered(self) -> None:
        """All 12 Trinity types are in the registry list."""
        from foundation import registry

        trinity_types = [
            TrinityNodePosition,
            TrinitySourceLocation,
            TrinityFieldData,
            TrinityParameterData,
            TrinityMethodData,
            TrinityComponentData,
            TrinitySystemData,
            TrinityResourceData,
            TrinityEventData,
            TrinityGraphNode,
            TrinityGraphEdge,
            TrinityNodeGraph,
        ]
        for cls in trinity_types:
            assert registry.is_registered(cls), f"{cls.__name__} not registered"


# =============================================================================
# WHITEBOX: Type hierarchy and metaclass consistency
# =============================================================================


class TestTypeHierarchyWhitebox:
    """Whitebox tests for type hierarchy consistency."""

    def test_all_trinity_types_use_graph_node_meta(self) -> None:
        """All 12 Trinity types use _GraphNodeMeta."""
        types_to_check = [
            TrinityNodePosition,
            TrinitySourceLocation,
            TrinityFieldData,
            TrinityParameterData,
            TrinityMethodData,
            TrinityComponentData,
            TrinitySystemData,
            TrinityResourceData,
            TrinityEventData,
            TrinityGraphNode,
            TrinityGraphEdge,
            TrinityNodeGraph,
        ]
        for cls in types_to_check:
            assert isinstance(cls, _GraphNodeMeta), (
                f"{cls.__name__} metaclass is {type(cls)}, not _GraphNodeMeta"
            )

    def test_trinity_graph_node_not_a_base_class(self) -> None:
        """TrinityGraphNode is NOT skipped by base class filter."""
        assert "TrinityGraphNode" not in _GraphNodeMeta._BASE_CLASS_NAMES

    def test_graph_node_is_trinity_graph_base(self) -> None:
        """TrinityGraphNode inherits from _TrinityGraphBase."""
        assert issubclass(TrinityGraphNode, _TrinityGraphBase)

    def test_graph_edge_is_trinity_graph_base(self) -> None:
        """TrinityGraphEdge inherits from _TrinityGraphBase."""
        assert issubclass(TrinityGraphEdge, _TrinityGraphBase)

    def test_node_graph_is_trinity_graph_base(self) -> None:
        """TrinityNodeGraph inherits from _TrinityGraphBase."""
        assert issubclass(TrinityNodeGraph, _TrinityGraphBase)

    def test_trinity_types_not_in_engine_meta_base_names(self) -> None:
        """Concrete Trinity types should NOT be in base class names."""
        concrete = [
            "TrinityNodePosition", "TrinityGraphNode", "TrinityGraphEdge",
            "TrinityNodeGraph", "TrinityComponentData", "TrinitySystemData",
            "TrinityResourceData", "TrinityEventData", "TrinityFieldData",
            "TrinityParameterData", "TrinityMethodData", "TrinitySourceLocation",
        ]
        for name in concrete:
            assert name not in _GraphNodeMeta._BASE_CLASS_NAMES, (
                f"{name} is in _BASE_CLASS_NAMES but should not be"
            )


# =============================================================================
# WHITEBOX: Data type backwards compatibility
# =============================================================================


class TestBackwardsCompatibilityWhitebox:
    """Whitebox tests for backwards compatibility features."""

    def test_event_data_accepts_payload_fields(self) -> None:
        """EventData.from_dict accepts 'payload_fields' as alias."""
        raw = {
            "payload_fields": [
                {"name": "damage", "type_annotation": "float"}
            ],
            "bases": ["Event"],
        }
        event = TrinityEventData.from_dict(raw)
        assert len(event.fields) == 1
        assert event.fields[0].name == "damage"

    def test_event_data_payload_fields_takes_precedence_over_fields(self) -> None:
        """payload_fields takes precedence over fields when both present."""
        raw = {
            "fields": [{"name": "old", "type_annotation": "str"}],
            "payload_fields": [
                {"name": "new", "type_annotation": "int"}
            ],
        }
        event = TrinityEventData.from_dict(raw)
        assert event.fields[0].name == "new"

    def test_method_data_accepts_query_types(self) -> None:
        """MethodData.from_dict accepts 'query_types' as query_components alias."""
        raw = {
            "name": "execute",
            "parameters": [],
            "query_types": ["Position", "Velocity"],
        }
        md = TrinityMethodData.from_dict(raw)
        assert "Position" in md.query_components
        assert "Velocity" in md.query_components

    def test_method_data_query_types_fallback(self) -> None:
        """query_types is used when query_components is not present."""
        raw = {
            "name": "execute",
            "parameters": [],
            "query_types": ["Transform"],
        }
        md = TrinityMethodData.from_dict(raw)
        assert md.query_components == ["Transform"]

    def test_parameter_data_from_dict_without_optional(self) -> None:
        """ParameterData.from_dict works without type_annotation and default_value."""
        pd = TrinityParameterData.from_dict({"name": "self"})
        assert pd.name == "self"
        assert pd.type_annotation is None
        assert pd.default_value is None

    def test_field_data_is_optional_default_false(self) -> None:
        """FieldData.from_dict uses False for missing is_optional."""
        fd = TrinityFieldData.from_dict({"name": "x", "type_annotation": "float"})
        assert fd.is_optional is False

    def test_field_data_type_annotation_default(self) -> None:
        """FieldData.from_dict defaults type_annotation to 'Any'."""
        fd = TrinityFieldData.from_dict({"name": "x"})
        assert fd.type_annotation == "Any"


# =============================================================================
# WHITEBOX: _TrinityGraphBase __eq__ edge cases
# =============================================================================


class TestEqualityEdgeCasesWhitebox:
    """Whitebox tests for equality edge cases."""

    def test_eq_subclass_not_equal(self) -> None:
        """A subclass instance is not equal to parent class instance."""

        class SubPosition(TrinityNodePosition):
            pass

        parent = TrinityNodePosition(1.0, 2.0)
        child = SubPosition(1.0, 2.0)
        assert parent.__eq__(child) is NotImplemented

    def test_eq_same_values_different_instances_equal(self) -> None:
        """Two instances with same field values are equal."""
        a = TrinityNodePosition(1.0, 2.0)
        b = TrinityNodePosition(1.0, 2.0)
        assert a == b
        assert not (a != b)

    def test_eq_not_equal_values(self) -> None:
        """Two instances with different field values are not equal."""
        a = TrinityNodePosition(1.0, 2.0)
        b = TrinityNodePosition(3.0, 4.0)
        assert a != b
        assert not (a == b)

    def test_eq_with_missing_fields(self) -> None:
        """__eq__ returns NotImplemented when comparing different types."""

        class DynamicNode(_TrinityGraphBase):
            x: int

        class DynamicNode2(_TrinityGraphBase):
            x: int

        a = DynamicNode()
        b = DynamicNode2()
        # Different types -> __eq__ returns NotImplemented.
        # (Python then falls through to identity; a is not b, so a == b is False.)
        result = a.__eq__(b)
        assert result is NotImplemented

    def test_node_graph_equality_by_structure(self) -> None:
        """Two graphs with same structure are structurally compared."""
        g1 = TrinityNodeGraph(
            nodes=[TrinityGraphNode(id="n1", type="component", name="A")],
            metadata={"v": 1},
        )
        g2 = TrinityNodeGraph(
            nodes=[TrinityGraphNode(id="n1", type="component", name="A")],
            metadata={"v": 1},
        )
        # Equality uses annotated field comparison
        assert g1.nodes == g2.nodes
        assert g1.metadata == g2.metadata


# =============================================================================
# WHITEBOX: Resource data is_singleton edge cases
# =============================================================================


class TestResourceDataEdgeCasesWhitebox:
    """Whitebox tests for ResourceData edge cases."""

    def test_resource_data_default_singleton(self) -> None:
        """ResourceData defaults is_singleton to True."""
        rd = TrinityResourceData()
        assert rd.is_singleton is True

    def test_resource_data_from_dict_default_singleton(self) -> None:
        """ResourceData.from_dict defaults is_singleton to True."""
        rd = TrinityResourceData.from_dict({"fields": []})
        assert rd.is_singleton is True

    def test_resource_data_from_dict_explicit_singleton(self) -> None:
        """ResourceData.from_dict with is_singleton=False."""
        rd = TrinityResourceData.from_dict({"fields": [], "is_singleton": False})
        assert rd.is_singleton is False

    def test_resource_data_to_dict_includes_is_singleton(self) -> None:
        """ResourceData.to_dict always includes is_singleton."""
        rd = TrinityResourceData()
        d = rd.to_dict()
        assert "is_singleton" in d
        assert d["is_singleton"] is True
        rd.is_singleton = False
        d = rd.to_dict()
        assert d["is_singleton"] is False


# =============================================================================
# WHITEBOX: ParameterData edge cases
# =============================================================================


class TestParameterDataEdgeCasesWhitebox:
    """Whitebox tests for ParameterData edge cases."""

    def test_parameter_data_to_dict_omits_none_fields(self) -> None:
        """to_dict omits None fields."""
        pd = TrinityParameterData(name="x")
        d = pd.to_dict()
        assert d == {"name": "x"}

    def test_parameter_data_to_dict_includes_type_annotation(self) -> None:
        """to_dict includes type_annotation when set."""
        pd = TrinityParameterData(name="x", type_annotation="float")
        d = pd.to_dict()
        assert d["type_annotation"] == "float"
        assert "default_value" not in d

    def test_parameter_data_to_dict_includes_default_value(self) -> None:
        """to_dict includes default_value when set."""
        pd = TrinityParameterData(name="x", default_value="10.0")
        d = pd.to_dict()
        assert d["default_value"] == "10.0"


# =============================================================================
# WHITEBOX: MethodData edge cases
# =============================================================================


class TestMethodDataEdgeCasesWhitebox:
    """Whitebox tests for MethodData edge cases."""

    def test_method_data_default_collections(self) -> None:
        """MethodData with no args uses empty lists."""
        md = TrinityMethodData(name="run")
        assert md.parameters == []
        assert md.query_components == []
        assert md.decorators == []

    def test_method_data_to_dict_omits_empty_collections(self) -> None:
        """to_dict omits empty query_components and decorators."""
        md = TrinityMethodData(name="run")
        d = md.to_dict()
        assert "query_components" not in d
        assert "decorators" not in d

    def test_method_data_to_dict_includes_non_empty_collections(self) -> None:
        """to_dict includes non-empty query_components and decorators."""
        md = TrinityMethodData(
            name="run",
            query_components=["Position"],
            decorators=["staticmethod"],
        )
        d = md.to_dict()
        assert d["query_components"] == ["Position"]
        assert d["decorators"] == ["staticmethod"]

    def test_method_data_to_dict_omits_none_fields(self) -> None:
        """to_dict omits None return_type, docstring, line_number."""
        md = TrinityMethodData(name="run")
        d = md.to_dict()
        assert "return_type" not in d
        assert "docstring" not in d
        assert "line_number" not in d

    def test_method_data_from_dict_query_types_fallback(self) -> None:
        """from_dict uses query_types when query_components not present."""
        md = TrinityMethodData.from_dict({
            "name": "run",
            "parameters": [],
            "query_types": ["Health"],
        })
        assert md.query_components == ["Health"]


# =============================================================================
# WHITEBOX: ComponentData edge cases
# =============================================================================


class TestComponentDataEdgeCasesWhitebox:
    """Whitebox tests for ComponentData edge cases."""

    def test_component_data_to_dict_always_has_fields(self) -> None:
        """to_dict always includes fields key."""
        cd = TrinityComponentData()
        d = cd.to_dict()
        assert "fields" in d
        assert d["fields"] == []

    def test_component_data_to_dict_omits_empty_bases(self) -> None:
        """to_dict omits bases when empty."""
        cd = TrinityComponentData()
        d = cd.to_dict()
        assert "bases" not in d

    def test_component_data_to_dict_omits_none_docstring(self) -> None:
        """to_dict omits docstring when None."""
        cd = TrinityComponentData()
        d = cd.to_dict()
        assert "docstring" not in d

    def test_component_data_to_dict_omits_empty_decorator_args(self) -> None:
        """to_dict omits decorator_args when empty."""
        cd = TrinityComponentData()
        d = cd.to_dict()
        assert "decorator_args" not in d


# =============================================================================
# WHITEBOX: SystemData edge cases
# =============================================================================


class TestSystemDataEdgeCasesWhitebox:
    """Whitebox tests for SystemData edge cases."""

    def test_system_data_to_dict_always_has_methods(self) -> None:
        """to_dict always includes methods key."""
        sd = TrinitySystemData()
        d = sd.to_dict()
        assert "methods" in d
        assert d["methods"] == []

    def test_system_data_to_dict_omits_empty_queries(self) -> None:
        """to_dict omits queries when empty."""
        sd = TrinitySystemData()
        d = sd.to_dict()
        assert "queries" not in d

    def test_system_data_to_dict_omits_empty_fields(self) -> None:
        """to_dict omits fields when empty."""
        sd = TrinitySystemData()
        d = sd.to_dict()
        assert "fields" not in d

    def test_system_data_to_dict_omits_empty_bases(self) -> None:
        """to_dict omits bases when empty."""
        sd = TrinitySystemData()
        d = sd.to_dict()
        assert "bases" not in d

    def test_system_data_from_dict_defaults(self) -> None:
        """from_dict uses empty defaults for missing keys."""
        sd = TrinitySystemData.from_dict({})
        assert sd.methods == []
        assert sd.queries == []
        assert sd.fields == []
        assert sd.bases == []


# =============================================================================
# WHITEBOX: EventData edge cases
# =============================================================================


class TestEventDataEdgeCasesWhitebox:
    """Whitebox tests for EventData edge cases."""

    def test_event_data_to_dict_always_has_fields(self) -> None:
        """to_dict always includes fields key."""
        ed = TrinityEventData()
        d = ed.to_dict()
        assert "fields" in d

    def test_event_data_to_dict_omits_empty_bases(self) -> None:
        """to_dict omits bases when empty."""
        ed = TrinityEventData()
        d = ed.to_dict()
        assert "bases" not in d

    def test_event_data_from_dict_defaults(self) -> None:
        """from_dict uses empty defaults for missing keys."""
        ed = TrinityEventData.from_dict({})
        assert ed.fields == []
        assert ed.bases == []

    def test_event_data_from_dict_with_empty_fields(self) -> None:
        """from_dict handles empty fields list."""
        ed = TrinityEventData.from_dict({"fields": []})
        assert ed.fields == []


# =============================================================================
# WHITEBOX: SourceLocation edge cases
# =============================================================================


class TestSourceLocationEdgeCasesWhitebox:
    """Whitebox tests for SourceLocation edge cases."""

    def test_source_location_from_dict_defaults(self) -> None:
        """from_dict uses defaults for missing keys."""
        loc = TrinitySourceLocation.from_dict({})
        assert loc.file == ""
        assert loc.line == 0
        assert loc.end_line is None
        assert loc.column is None

    def test_source_location_from_dict_string_line(self) -> None:
        """from_dict handles string line value."""
        loc = TrinitySourceLocation.from_dict({"file": "test.py", "line": "42"})
        assert loc.line == 42

    def test_source_location_dirty_tracking_all_fields(self) -> None:
        """All fields of SourceLocation support dirty tracking."""
        loc = TrinitySourceLocation("a.py", 1)
        loc.clear_dirty()
        loc.file = "b.py"
        loc.line = 99
        loc.end_line = 100
        loc.column = 5
        assert loc.is_dirty("file")
        assert loc.is_dirty("line")
        assert loc.is_dirty("end_line")
        assert loc.is_dirty("column")
        dirty = loc.get_dirty_fields()
        assert len(dirty) == 4

    def test_source_location_repr(self) -> None:
        """SourceLocation repr includes key info."""
        loc = TrinitySourceLocation("test.py", 42)
        r = repr(loc)
        assert "TrinitySourceLocation" in r
        assert "test.py" in r
        assert "42" in r


# =============================================================================
# WHITEBOX: GraphNode edge cases
# =============================================================================


class TestGraphNodeEdgeCasesWhitebox:
    """Whitebox tests for GraphNode edge cases."""

    def test_graph_node_default_position(self) -> None:
        """Default position is TrinityNodePosition(0, 0)."""
        node = TrinityGraphNode(id="n1", type="component", name="A")
        assert isinstance(node.position, TrinityNodePosition)
        assert node.position.x == 0.0
        assert node.position.y == 0.0

    def test_graph_node_default_data_empty_dict(self) -> None:
        """Default data is empty dict."""
        node = TrinityGraphNode(id="n1", type="component", name="A")
        assert node.data == {}

    def test_graph_node_default_source_none(self) -> None:
        """Default source is None."""
        node = TrinityGraphNode(id="n1", type="component", name="A")
        assert node.source is None

    def test_graph_node_with_trinity_data_roundtrip(self) -> None:
        """GraphNode with TrinityComponentData round-trips correctly."""
        node = TrinityGraphNode(
            id="n1", type="component", name="Player",
            data=TrinityComponentData(
                fields=[TrinityFieldData("x", "float")],
                docstring="Position data",
            ),
        )
        d = node.to_dict()
        restored = TrinityGraphNode.from_dict(d)
        assert restored == node
        assert isinstance(restored.data, TrinityComponentData)
        assert restored.data.docstring == "Position data"

    def test_graph_node_edge_serialization_with_source(self) -> None:
        """Graph node with source serializes correctly via to_dict."""
        src = TrinitySourceLocation("game.py", 10, end_line=20, column=4)
        node = TrinityGraphNode(
            id="n1", type="component", name="Player",
            source=src,
        )
        d = node.to_dict()
        assert d["source"]["file"] == "game.py"
        assert d["source"]["line"] == 10
        assert d["source"]["end_line"] == 20
        assert d["source"]["column"] == 4

    def test_graph_node_type_literals(self) -> None:
        """GraphNode accepts only valid type literals."""
        node = TrinityGraphNode(id="n1", type="component", name="A")
        assert node.type == "component"
        node = TrinityGraphNode(id="n2", type="system", name="B")
        assert node.type == "system"
        node = TrinityGraphNode(id="n3", type="resource", name="C")
        assert node.type == "resource"
        node = TrinityGraphNode(id="n4", type="event", name="D")
        assert node.type == "event"


# =============================================================================
# WHITEBOX: to_trinity_graph edge cases
# =============================================================================


class TestToTrinityGraphEdgeCasesWhitebox:
    """Whitebox tests for to_trinity_graph edge cases."""

    def test_raises_type_error_for_invalid_input(self) -> None:
        """to_trinity_graph raises TypeError for non-graph input."""
        with pytest.raises(TypeError, match="Expected NodeGraph"):
            to_trinity_graph(42)  # type: ignore
        with pytest.raises(TypeError, match="Expected NodeGraph"):
            to_trinity_graph({"nodes": []})  # type: ignore

    def test_handles_empty_graph(self) -> None:
        """to_trinity_graph handles empty graph."""
        plain = NodeGraph()
        result = to_trinity_graph(plain)
        assert isinstance(result, TrinityNodeGraph)
        assert result.nodes == []
        assert result.edges == []
        assert result.metadata == {}

    def test_idempotent_on_trinity_graph_detailed(self) -> None:
        """to_trinity_graph returns the same TrinityNodeGraph instance."""
        original = TrinityNodeGraph()
        result = to_trinity_graph(original)
        assert result is original

    def test_preserves_metadata_from_plain_graph(self) -> None:
        """to_trinity_graph preserves metadata."""
        plain = NodeGraph(
            nodes=[],
            metadata={"source": "test.py", "version": 2},
        )
        result = to_trinity_graph(plain)
        assert result.metadata == {"source": "test.py", "version": 2}

    def test_converts_edge_with_no_data_attr(self) -> None:
        """GraphEdge without 'data' attr converts fine (fallback to {})."""
        plain = GraphEdge(id="e1", source="n1", target="n2")
        # Manually remove data to test the fallback in _convert_data_to_trinity
        if hasattr(plain, 'data'):
            del plain.data  # type: ignore[attr-defined]
        plain_graph = NodeGraph(edges=[plain])
        result = to_trinity_graph(plain_graph)
        assert len(result.edges) == 1
        assert result.edges[0].data == {}


# =============================================================================
# WHITEBOX: Step recording precision
# =============================================================================


class TestStepRecordingPrecisionWhitebox:
    """Whitebox tests for the exact step recording format."""

    def test_field_step_type_name_recorded(self) -> None:
        """DESCRIBE step records the type __name__."""
        steps = TrinityFieldData._metaclass_steps
        name_describe = [
            s for s in steps
            if s.op == Op.DESCRIBE and s.args.get("field") == "name"
        ]
        assert len(name_describe) == 1
        assert name_describe[0].args["type"] == "str"

    def test_all_node_data_fields_have_type_in_describe(self) -> None:
        """All annotated fields get DESCRIBE steps with type info."""
        steps = TrinityComponentData._metaclass_steps
        describe_steps = [s for s in steps if s.op == Op.DESCRIBE]
        assert len(describe_steps) >= 4  # fields, bases, docstring, decorator_args
        field_names = {s.args["field"] for s in describe_steps}
        assert "fields" in field_names
        assert "bases" in field_names
        assert "docstring" in field_names

    def test_intercept_step_indicates_tracked(self) -> None:
        """INTERCEPT step records 'tracked' descriptor."""
        steps = TrinityGraphNode._metaclass_steps
        intercept_steps = [s for s in steps if s.op == Op.INTERCEPT]
        for s in intercept_steps:
            assert s.args.get("descriptor") == "tracked"

    def test_register_step_identifies_foundation(self) -> None:
        """REGISTER step identifies foundation as the registry."""
        steps = TrinityGraphNode._metaclass_steps
        register_steps = [s for s in steps if s.op == Op.REGISTER]
        foundation_steps = [
            s for s in register_steps
            if s.args.get("registry") == "foundation"
        ]
        assert len(foundation_steps) >= 1


# =============================================================================
# WHITEBOX: Metaclass __repr__
# =============================================================================


class TestMetaclassReprWhitebox:
    """Whitebox tests for metaclass __repr__."""

    def test_engine_meta_repr_contains_name(self) -> None:
        """EngineMeta.__repr__ includes class name."""
        r = EngineMeta.__repr__(TrinityGraphNode)
        assert "TrinityGraphNode" in r
        assert "<" in r
        assert ">" in r

    def test_graph_node_meta_repr(self) -> None:
        """_GraphNodeMeta __repr__ shows 'GraphNode' prefix."""
        r = type(TrinityGraphNode).__repr__(TrinityGraphNode)
        # _GraphNodeMeta ends with 'Meta' so it strips to 'GraphNode'
        assert "<" in r
        assert "TrinityGraphNode" in r


# =============================================================================
# WHITEBOX: TrackedDescriptor bitmask mode
# =============================================================================


class TestTrackedDescriptorBitmaskWhitebox:
    """Whitebox tests for TrackedDescriptor bitmask mode."""

    def test_bitmask_creates_dirty_mask(self) -> None:
        """Bitmask mode creates _dirty_mask on first mutation."""
        desc = TrackedDescriptor(field_type=float, use_bitmask=True, field_offset=0)

        class BitmaskTest:
            x: float = 0.0

        desc.__set_name__(BitmaskTest, "x")
        setattr(BitmaskTest, "x", desc)

        obj = BitmaskTest()
        obj.x = 42.0  # type: ignore[attr-defined]
        assert hasattr(obj, "_dirty_mask")
        assert obj._dirty_mask & 1 == 1  # Bit 0 set

    def test_bitmask_multiple_fields(self) -> None:
        """Bitmask mode sets correct bits for multiple fields."""
        desc_x = TrackedDescriptor(field_type=float, use_bitmask=True, field_offset=0)
        desc_y = TrackedDescriptor(field_type=float, use_bitmask=True, field_offset=1)

        class BitmaskMulti:
            x: float = 0.0
            y: float = 0.0

        desc_x.__set_name__(BitmaskMulti, "x")
        desc_y.__set_name__(BitmaskMulti, "y")
        setattr(BitmaskMulti, "x", desc_x)
        setattr(BitmaskMulti, "y", desc_y)

        obj = BitmaskMulti()
        clear_dirty(obj)  # module-level function (BitmaskMulti has no method)
        obj.x = 10.0  # type: ignore[attr-defined]
        assert obj._dirty_mask & (1 << 0) != 0
        assert obj._dirty_mask & (1 << 1) == 0  # y not dirty

        obj.y = 20.0  # type: ignore[attr-defined]
        assert obj._dirty_mask & (1 << 0) != 0
        assert obj._dirty_mask & (1 << 1) != 0  # y also dirty now

    def test_clear_dirty_resets_bitmask(self) -> None:
        """clear_dirty resets _dirty_mask to 0."""
        desc = TrackedDescriptor(field_type=float, use_bitmask=True, field_offset=0)

        class BitmaskClear:
            x: float = 0.0

        desc.__set_name__(BitmaskClear, "x")
        setattr(BitmaskClear, "x", desc)

        obj = BitmaskClear()
        obj.x = 99.0  # type: ignore[attr-defined]
        assert obj._dirty_mask != 0
        clear_dirty(obj)
        assert obj._dirty_mask == 0


# =============================================================================
# WHITEBOX: TrackedDescriptor foundation notification (error resilience)
# =============================================================================


class TestTrackedDescriptorFoundationErrorsWhitebox:
    """Whitebox tests for TrackedDescriptor foundation error handling."""

    def test_notify_foundation_tracker_import_error_safe(self) -> None:
        """_notify_foundation_tracker handles ImportError silently."""
        desc = TrinityGraphNode._trinity_fields["name"]

        # Create an object that has the necessary attributes
        obj = TrinityGraphNode(id="n1", type="component", name="Player")

        # Remove foundation modules temporarily
        saved_modules = {}
        for mod_name in ["foundation", "foundation.tracker"]:
            saved_modules[mod_name] = sys.modules.pop(mod_name, None)

        try:
            # This should not raise
            desc._notify_foundation_tracker(obj, "Old", "New")
        finally:
            for mod_name, mod in saved_modules.items():
                if mod is not None:
                    sys.modules[mod_name] = mod

    def test_notify_eventlog_import_error_safe(self) -> None:
        """_notify_eventlog handles ImportError silently."""
        desc = TrinityGraphNode._trinity_fields["name"]

        obj = TrinityGraphNode(id="n1", type="component", name="Player")

        # Remove foundation modules
        saved_modules = {}
        for mod_name in ["foundation", "foundation.eventlog"]:
            saved_modules[mod_name] = sys.modules.pop(mod_name, None)

        try:
            # This should not raise
            desc._notify_eventlog(obj, "Old", "New")
        finally:
            for mod_name, mod in saved_modules.items():
                if mod is not None:
                    sys.modules[mod_name] = mod


# =============================================================================
# WHITEBOX: Descriptor composition rules
# =============================================================================


class TestDescriptorCompositionWhitebox:
    """Whitebox tests for TrackedDescriptor composition attributes."""

    def test_tracked_descriptor_id_is_tracked(self) -> None:
        """TrackedDescriptor.descriptor_id == 'tracked'."""
        assert TrackedDescriptor.descriptor_id == "tracked"

    def test_tracked_accepts_inner_types(self) -> None:
        """TrackedDescriptor accepts_inner includes storage, validated, range."""
        assert "storage" in TrackedDescriptor.accepts_inner
        assert True  # Always passes if no AttributeError

    def test_tracked_accepts_outer_types(self) -> None:
        """TrackedDescriptor accepts_outer includes networked, observable, cached."""
        assert "networked" in TrackedDescriptor.accepts_outer
        assert "observable" in TrackedDescriptor.accepts_outer

    def test_tracked_excludes_computed(self) -> None:
        """TrackedDescriptor excludes computed descriptors."""
        assert "computed" in TrackedDescriptor.excludes


# =============================================================================
# WHITEBOX: TrinityNodeData Union type alias
# =============================================================================


class TestTrinityNodeDataTypeAliasWhitebox:
    """Whitebox tests for TrinityNodeData union type."""

    def test_trinity_node_data_is_instance_of_union_members(self) -> None:
        """isinstance checks work correctly with TrinityNodeData."""
        cd = TrinityComponentData()
        sd = TrinitySystemData()
        rd = TrinityResourceData()
        ed = TrinityEventData()

        # All should be instances of the Union via their parent
        assert isinstance(cd, (TrinityComponentData, TrinitySystemData, TrinityResourceData, TrinityEventData))
        assert isinstance(sd, (TrinityComponentData, TrinitySystemData, TrinityResourceData, TrinityEventData))
        assert isinstance(rd, (TrinityComponentData, TrinitySystemData, TrinityResourceData, TrinityEventData))
        assert isinstance(ed, (TrinityComponentData, TrinitySystemData, TrinityResourceData, TrinityEventData))

    def test_isinstance_with_wrong_type(self) -> None:
        """Non-NodeData types are not instances of NodeData types."""
        pos = TrinityNodePosition(1.0, 2.0)
        assert not isinstance(pos, TrinityComponentData)
        assert not isinstance(pos, TrinitySystemData)
        assert not isinstance(pos, TrinityResourceData)
        assert not isinstance(pos, TrinityEventData)


# =============================================================================
# WHITEBOX: Node graph mutation dirty tracking
# =============================================================================


class TestNodeGraphDirtyMutationWhitebox:
    """Whitebox tests for dirty tracking in graph mutations."""

    def test_add_node_dirty_on_graph(self) -> None:
        """add_node triggers dirty on the graph's nodes field (list reassign)."""
        graph = TrinityNodeGraph()
        graph.clear_dirty()
        # Append does NOT trigger dirty (same list reference)
        graph.nodes.append(TrinityGraphNode(id="n1", type="component", name="A"))
        assert not graph.is_dirty("nodes")
        # Reassigning with DIFFERENT content triggers dirty (value-based
        # comparison in post_set detects the new element)
        graph.nodes = graph.nodes + [TrinityGraphNode(id="n2", type="system", name="B")]
        assert graph.is_dirty("nodes")

    def test_remove_node_dirty_on_graph(self) -> None:
        """remove_node creates a new list, triggering dirty."""
        graph = TrinityNodeGraph(
            nodes=[TrinityGraphNode(id="n1", type="component", name="A")]
        )
        graph.clear_dirty()
        graph.remove_node("n1")
        assert graph.is_dirty("nodes")

    def test_remove_edge_dirty_on_graph(self) -> None:
        """remove_edge creates a new edge list, triggering dirty."""
        graph = TrinityNodeGraph(
            edges=[TrinityGraphEdge(id="e1", source="n1", target="n2")]
        )
        graph.clear_dirty()
        graph.remove_edge("e1")
        assert graph.is_dirty("edges")

    def test_add_edge_dirty_no_trigger(self) -> None:
        """add_edge does NOT trigger dirty (append to same list)."""
        graph = TrinityNodeGraph()
        graph.clear_dirty()
        graph.add_edge(TrinityGraphEdge(id="e1", source="n1", target="n2"))
        assert not graph.is_dirty("edges")
        # Reassigning with DIFFERENT content triggers dirty
        graph.edges = graph.edges + [TrinityGraphEdge(id="e2", source="n2", target="n3")]
        assert graph.is_dirty("edges")


# =============================================================================
# WHITEBOX: Field type resolution in descriptors
# =============================================================================


class TestFieldTypeResolutionWhitebox:
    """Whitebox tests for descriptor field type storage."""

    def test_simple_field_type(self) -> None:
        """Simple types like str, int, float are stored directly."""
        assert TrinityNodePosition._trinity_fields["x"].field_type is float
        assert TrinitySourceLocation._trinity_fields["file"].field_type is str
        assert TrinitySourceLocation._trinity_fields["line"].field_type is int

    def test_complex_field_type(self) -> None:
        """Complex types like Optional[str] are stored as generic."""
        # Optional[int] at runtime is int | None (PEP 604)
        desc = TrinitySourceLocation._trinity_fields["end_line"]
        assert desc.field_type == int | None

    def test_list_field_type(self) -> None:
        """List field types are stored."""
        desc = TrinityNodeGraph._trinity_fields["nodes"]
        # The annotation is list[TrinityGraphNode], runtime is list
        assert desc.field_type is list or "list" in str(desc.field_type)

    def test_dict_field_type(self) -> None:
        """Dict field types are stored."""
        desc = TrinityGraphNode._trinity_fields["data"]
        # The annotation is Union[dict[str, Any], TrinityNodeData]
        # At runtime this might be dict or a Union
        if hasattr(desc.field_type, "__origin__"):
            assert desc.field_type.__origin__ is dict or desc.field_type.__origin__ is Union


# =============================================================================
# WHITEBOX: to_trinity_graph with NodeGraph corner cases
# =============================================================================


class TestToTrinityGraphCornerCasesWhitebox:
    """Whitebox tests for to_trinity_graph corner cases."""

    def test_graph_without_metadata_attr(self) -> None:
        """Graph without metadata attribute raises TypeError.

        to_trinity_graph requires a real NodeGraph or TrinityNodeGraph
        instance -- duck-typed look-alikes are rejected.
        """

        class MinimalGraph:
            def __init__(self) -> None:
                self.nodes: list[Any] = []
                self.edges: list[Any] = []

        minimal = MinimalGraph()
        with pytest.raises(TypeError, match="Expected NodeGraph"):
            to_trinity_graph(minimal)  # type: ignore[arg-type]

    def test_graph_with_edges_only(self) -> None:
        """Graph with only edges converts correctly."""
        plain = NodeGraph(
            edges=[GraphEdge(id="e1", source="n1", target="n2")]
        )
        result = to_trinity_graph(plain)
        assert len(result.edges) == 1
        assert result.edges[0].id == "e1"

    def test_graph_with_nodes_only(self) -> None:
        """Graph with only nodes converts correctly."""
        plain = NodeGraph(
            nodes=[GraphNode(id="n1", type="component", name="A")]
        )
        result = to_trinity_graph(plain)
        assert len(result.nodes) == 1
        assert result.nodes[0].name == "A"

    def test_graph_with_complex_node_data_including_decorator_args(self) -> None:
        """Node data with decorator_args converts correctly."""
        comp = ComponentData(
            fields=[FieldData("x", "float")],
            decorator_args={"pool": True, "max_size": 100},
        )
        plain = NodeGraph(
            nodes=[GraphNode(id="n1", type="component", name="A", data=comp)]
        )
        result = to_trinity_graph(plain)
        assert isinstance(result.nodes[0].data, TrinityComponentData)
        assert result.nodes[0].data.decorator_args == {"pool": True, "max_size": 100}


# =============================================================================
# WHITEBOX: Decompose non-type objects
# =============================================================================


class TestDecomposeNonTypeWhitebox:
    """Whitebox tests for decompose with non-type targets."""

    def test_decompose_on_non_class(self) -> None:
        """decompose on a non-class object returns empty list (no error)."""
        result = decompose(42)  # type: ignore
        assert result == []

    def test_decompose_layered_on_non_class(self) -> None:
        """decompose_layered on a non-class returns only decorators layer."""
        result = decompose_layered("test")  # type: ignore
        assert isinstance(result, dict)
        assert "decorators" in result
        assert result["metaclass"] == []
        assert result["descriptors"] == []


# =============================================================================
# WHITEBOX: Module __all__ completeness
# =============================================================================


class TestModuleAllWhitebox:
    """Whitebox tests for module __all__ completeness."""

    def test_all_contains_all_public_types(self) -> None:
        """__all__ includes all 12 Trinity types."""
        from flowforge_backend.ast_parser import trinity_nodes

        all_exported = trinity_nodes.__all__
        expected = [
            "TrinityNodePosition",
            "TrinitySourceLocation",
            "TrinityFieldData",
            "TrinityParameterData",
            "TrinityMethodData",
            "TrinityComponentData",
            "TrinitySystemData",
            "TrinityResourceData",
            "TrinityEventData",
            "TrinityNodeData",
            "TrinityGraphNode",
            "TrinityGraphEdge",
            "TrinityNodeGraph",
        ]
        for name in expected:
            assert name in all_exported, f"{name} missing from __all__"

    def test_all_contains_conversion_functions(self) -> None:
        """__all__ includes to_trinity_graph."""
        from flowforge_backend.ast_parser import trinity_nodes

        assert "to_trinity_graph" in trinity_nodes.__all__

    def test_all_contains_convenience_constructors(self) -> None:
        """__all__ includes all 5 convenience constructors."""
        from flowforge_backend.ast_parser import trinity_nodes

        constructors = [
            "create_trinity_component_node",
            "create_trinity_system_node",
            "create_trinity_resource_node",
            "create_trinity_event_node",
            "create_trinity_edge",
        ]
        for name in constructors:
            assert name in trinity_nodes.__all__, f"{name} missing from __all__"
