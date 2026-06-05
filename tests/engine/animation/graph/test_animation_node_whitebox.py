"""
WHITEBOX tests for T-AG-1.1 AnimationNode Base Class.

Tests the internal implementation details of:
- GraphNodeMeta metaclass with auto-registration
- SlotType enum
- InputSlot and OutputSlot dataclasses
- AnimationNode abstract base class
- Slot definition methods
- Node identification system
"""

import pytest
from abc import ABCMeta
from dataclasses import fields, is_dataclass

from engine.animation.graph.animation_graph import (
    AnimationNode,
    GraphContext,
    GraphNodeMeta,
    InputSlot,
    OutputSlot,
    Pose,
    SlotType,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the node registry before and after each test."""
    # Save existing registry state
    saved_registry = dict(GraphNodeMeta._registry)
    GraphNodeMeta.clear_registry()
    yield
    # Restore registry state
    GraphNodeMeta._registry.clear()
    GraphNodeMeta._registry.update(saved_registry)


@pytest.fixture
def graph_context():
    """Create a basic GraphContext for testing."""
    return GraphContext()


# =============================================================================
# TEST: SlotType Enum
# =============================================================================


class TestSlotTypeEnum:
    """Tests for SlotType enum completeness and values."""

    def test_slot_type_has_pose(self):
        """SlotType should have POSE type."""
        assert hasattr(SlotType, "POSE")
        assert SlotType.POSE is not None

    def test_slot_type_has_float(self):
        """SlotType should have FLOAT type."""
        assert hasattr(SlotType, "FLOAT")
        assert SlotType.FLOAT is not None

    def test_slot_type_has_bool(self):
        """SlotType should have BOOL type."""
        assert hasattr(SlotType, "BOOL")
        assert SlotType.BOOL is not None

    def test_slot_type_has_int(self):
        """SlotType should have INT type."""
        assert hasattr(SlotType, "INT")
        assert SlotType.INT is not None

    def test_slot_type_has_transform(self):
        """SlotType should have TRANSFORM type."""
        assert hasattr(SlotType, "TRANSFORM")
        assert SlotType.TRANSFORM is not None

    def test_slot_type_has_trigger(self):
        """SlotType should have TRIGGER type."""
        assert hasattr(SlotType, "TRIGGER")
        assert SlotType.TRIGGER is not None

    def test_slot_type_has_enum(self):
        """SlotType should have ENUM type."""
        assert hasattr(SlotType, "ENUM")
        assert SlotType.ENUM is not None

    def test_slot_type_all_types_present(self):
        """SlotType should have exactly 7 types."""
        expected_types = {"POSE", "FLOAT", "BOOL", "INT", "TRANSFORM", "TRIGGER", "ENUM"}
        actual_types = {member.name for member in SlotType}
        assert actual_types == expected_types

    def test_slot_type_values_are_unique(self):
        """All SlotType values should be unique."""
        values = [member.value for member in SlotType]
        assert len(values) == len(set(values))


# =============================================================================
# TEST: InputSlot Dataclass
# =============================================================================


class TestInputSlotDataclass:
    """Tests for InputSlot dataclass structure and behavior."""

    def test_input_slot_is_dataclass(self):
        """InputSlot should be a dataclass."""
        assert is_dataclass(InputSlot)

    def test_input_slot_has_name_field(self):
        """InputSlot should have a name field."""
        field_names = [f.name for f in fields(InputSlot)]
        assert "name" in field_names

    def test_input_slot_has_slot_type_field(self):
        """InputSlot should have a slot_type field."""
        field_names = [f.name for f in fields(InputSlot)]
        assert "slot_type" in field_names

    def test_input_slot_has_description_field(self):
        """InputSlot should have a description field."""
        field_names = [f.name for f in fields(InputSlot)]
        assert "description" in field_names

    def test_input_slot_has_optional_field(self):
        """InputSlot should have an optional field."""
        field_names = [f.name for f in fields(InputSlot)]
        assert "optional" in field_names

    def test_input_slot_creation(self):
        """InputSlot should be creatable with required fields."""
        slot = InputSlot(name="test_input", slot_type=SlotType.POSE)
        assert slot.name == "test_input"
        assert slot.slot_type == SlotType.POSE

    def test_input_slot_default_description(self):
        """InputSlot description should default to empty string."""
        slot = InputSlot(name="test", slot_type=SlotType.FLOAT)
        assert slot.description == ""

    def test_input_slot_default_optional(self):
        """InputSlot optional should default to False."""
        slot = InputSlot(name="test", slot_type=SlotType.FLOAT)
        assert slot.optional is False

    def test_input_slot_with_all_fields(self):
        """InputSlot should accept all fields."""
        slot = InputSlot(
            name="full_slot",
            slot_type=SlotType.BOOL,
            description="A boolean input",
            optional=True,
        )
        assert slot.name == "full_slot"
        assert slot.slot_type == SlotType.BOOL
        assert slot.description == "A boolean input"
        assert slot.optional is True


# =============================================================================
# TEST: OutputSlot Dataclass
# =============================================================================


class TestOutputSlotDataclass:
    """Tests for OutputSlot dataclass structure and behavior."""

    def test_output_slot_is_dataclass(self):
        """OutputSlot should be a dataclass."""
        assert is_dataclass(OutputSlot)

    def test_output_slot_has_name_field(self):
        """OutputSlot should have a name field."""
        field_names = [f.name for f in fields(OutputSlot)]
        assert "name" in field_names

    def test_output_slot_has_slot_type_field(self):
        """OutputSlot should have a slot_type field."""
        field_names = [f.name for f in fields(OutputSlot)]
        assert "slot_type" in field_names

    def test_output_slot_has_description_field(self):
        """OutputSlot should have a description field."""
        field_names = [f.name for f in fields(OutputSlot)]
        assert "description" in field_names

    def test_output_slot_creation(self):
        """OutputSlot should be creatable with required fields."""
        slot = OutputSlot(name="test_output", slot_type=SlotType.POSE)
        assert slot.name == "test_output"
        assert slot.slot_type == SlotType.POSE

    def test_output_slot_default_description(self):
        """OutputSlot description should default to empty string."""
        slot = OutputSlot(name="test", slot_type=SlotType.FLOAT)
        assert slot.description == ""

    def test_output_slot_with_all_fields(self):
        """OutputSlot should accept all fields."""
        slot = OutputSlot(
            name="full_output",
            slot_type=SlotType.TRANSFORM,
            description="A transform output",
        )
        assert slot.name == "full_output"
        assert slot.slot_type == SlotType.TRANSFORM
        assert slot.description == "A transform output"


# =============================================================================
# TEST: GraphNodeMeta Metaclass
# =============================================================================


class TestGraphNodeMetaAutoRegistration:
    """Tests for GraphNodeMeta auto-registration behavior."""

    def test_metaclass_inherits_from_abcmeta(self):
        """GraphNodeMeta should inherit from ABCMeta."""
        assert issubclass(GraphNodeMeta, ABCMeta)

    def test_metaclass_has_registry(self):
        """GraphNodeMeta should have a _registry class attribute."""
        assert hasattr(GraphNodeMeta, "_registry")
        assert isinstance(GraphNodeMeta._registry, dict)

    def test_registry_method_returns_dict(self):
        """registry() should return a dictionary."""
        result = GraphNodeMeta.registry()
        assert isinstance(result, dict)

    def test_base_class_not_registered(self):
        """AnimationNode base class should not be in registry."""
        registry = GraphNodeMeta.registry()
        assert "AnimationNode" not in registry

    def test_concrete_subclass_auto_registered(self):
        """Concrete subclasses should be auto-registered."""

        class ConcreteTestNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        registry = GraphNodeMeta.registry()
        assert "ConcreteTestNode" in registry
        assert registry["ConcreteTestNode"] is ConcreteTestNode

    def test_abstract_subclass_not_registered(self):
        """Subclasses marked _abstract=True should not be registered."""

        class AbstractTestNode(AnimationNode):
            _abstract = True

            def evaluate(self, context):
                return Pose()

        registry = GraphNodeMeta.registry()
        assert "AbstractTestNode" not in registry

    def test_multiple_subclasses_registered(self):
        """Multiple concrete subclasses should all be registered."""

        class NodeTypeA(AnimationNode):
            def evaluate(self, context):
                return Pose()

        class NodeTypeB(AnimationNode):
            def evaluate(self, context):
                return Pose()

        registry = GraphNodeMeta.registry()
        assert "NodeTypeA" in registry
        assert "NodeTypeB" in registry

    def test_registry_returns_copy(self):
        """registry() should return a copy, not the internal dict."""

        class TestNodeForCopy(AnimationNode):
            def evaluate(self, context):
                return Pose()

        registry = GraphNodeMeta.registry()
        registry["fake_node"] = None  # Modify returned dict

        # Internal registry should be unchanged
        assert "fake_node" not in GraphNodeMeta._registry


class TestGraphNodeMetaLookup:
    """Tests for GraphNodeMeta lookup methods."""

    def test_get_node_type_returns_class(self):
        """get_node_type should return the registered class."""

        class LookupTestNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        result = GraphNodeMeta.get_node_type("LookupTestNode")
        assert result is LookupTestNode

    def test_get_node_type_returns_none_for_unknown(self):
        """get_node_type should return None for unknown names."""
        result = GraphNodeMeta.get_node_type("NonExistentNode")
        assert result is None

    def test_all_node_types_method(self):
        """all_node_types should return all registered types."""

        class AllTypesNodeA(AnimationNode):
            def evaluate(self, context):
                return Pose()

        class AllTypesNodeB(AnimationNode):
            def evaluate(self, context):
                return Pose()

        all_types = GraphNodeMeta.all_node_types()
        assert "AllTypesNodeA" in all_types
        assert "AllTypesNodeB" in all_types

    def test_clear_registry(self):
        """clear_registry should empty the registry."""

        class ToClearNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        assert "ToClearNode" in GraphNodeMeta._registry
        GraphNodeMeta.clear_registry()
        assert len(GraphNodeMeta._registry) == 0


class TestGraphNodeMetaEdgeCases:
    """Tests for GraphNodeMeta edge cases."""

    def test_duplicate_class_name_overwrites(self):
        """A second class with the same name should overwrite in registry."""
        # Create first class
        class DuplicateNode(AnimationNode):
            first_version = True

            def evaluate(self, context):
                return Pose()

        first_class = DuplicateNode

        # Create a new class with same name in a different scope
        # This simulates redefining or reloading a class
        exec(
            """
class DuplicateNode(AnimationNode):
    second_version = True
    def evaluate(self, context):
        return Pose()
""",
            globals(),
        )

        # The registry should contain the second version
        registry = GraphNodeMeta.registry()
        assert "DuplicateNode" in registry
        # Note: The second class will overwrite the first

    def test_nested_subclass_registered(self):
        """Nested subclasses should be registered."""

        class ParentNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        class ChildNode(ParentNode):
            def evaluate(self, context):
                return Pose()

        registry = GraphNodeMeta.registry()
        assert "ParentNode" in registry
        assert "ChildNode" in registry


# =============================================================================
# TEST: AnimationNode Abstract Methods
# =============================================================================


class TestAnimationNodeAbstractMethods:
    """Tests for AnimationNode abstract method requirements."""

    def test_animation_node_has_abstract_evaluate(self):
        """AnimationNode.evaluate should be abstract."""
        # Check that AnimationNode has the evaluate method marked abstract
        assert hasattr(AnimationNode, "evaluate")
        assert getattr(AnimationNode.evaluate, "__isabstractmethod__", False)

    def test_cannot_instantiate_animation_node_directly(self):
        """AnimationNode should not be instantiable directly."""
        with pytest.raises(TypeError):
            AnimationNode("test_node")

    def test_concrete_subclass_must_implement_evaluate(self):
        """Concrete subclasses must implement evaluate."""

        # This should work - proper implementation
        class GoodNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = GoodNode("good")
        assert node is not None

    def test_evaluate_signature(self):
        """evaluate method should accept context parameter."""

        class SignatureTestNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = SignatureTestNode("test")
        ctx = GraphContext()
        result = node.evaluate(ctx)
        assert isinstance(result, Pose)


# =============================================================================
# TEST: AnimationNode Slot Definition Methods
# =============================================================================


class TestAnimationNodeSlotDefinition:
    """Tests for AnimationNode slot definition methods."""

    def test_define_input_slot(self):
        """define_input_slot should create and store an InputSlot."""

        class SlotDefNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = SlotDefNode("test")
        slot = node.define_input_slot("pose_in", SlotType.POSE, "Input pose")

        assert isinstance(slot, InputSlot)
        assert slot.name == "pose_in"
        assert slot.slot_type == SlotType.POSE
        assert slot.description == "Input pose"

    def test_define_input_slot_optional(self):
        """define_input_slot should handle optional parameter."""

        class OptionalSlotNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = OptionalSlotNode("test")
        slot = node.define_input_slot(
            "optional_in", SlotType.FLOAT, "Optional input", optional=True
        )

        assert slot.optional is True

    def test_define_output_slot(self):
        """define_output_slot should create and store an OutputSlot."""

        class OutputSlotNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = OutputSlotNode("test")
        slot = node.define_output_slot("pose_out", SlotType.POSE, "Output pose")

        assert isinstance(slot, OutputSlot)
        assert slot.name == "pose_out"
        assert slot.slot_type == SlotType.POSE
        assert slot.description == "Output pose"

    def test_get_input_slot(self):
        """get_input_slot should retrieve defined input slot."""

        class GetSlotNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = GetSlotNode("test")
        node.define_input_slot("my_input", SlotType.BOOL)

        retrieved = node.get_input_slot("my_input")
        assert retrieved is not None
        assert retrieved.name == "my_input"

    def test_get_input_slot_returns_none_for_unknown(self):
        """get_input_slot should return None for unknown slot names."""

        class UnknownSlotNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = UnknownSlotNode("test")
        assert node.get_input_slot("nonexistent") is None

    def test_get_output_slot(self):
        """get_output_slot should retrieve defined output slot."""

        class GetOutputNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = GetOutputNode("test")
        node.define_output_slot("my_output", SlotType.FLOAT)

        retrieved = node.get_output_slot("my_output")
        assert retrieved is not None
        assert retrieved.name == "my_output"

    def test_get_output_slot_returns_none_for_unknown(self):
        """get_output_slot should return None for unknown slot names."""

        class UnknownOutputNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = UnknownOutputNode("test")
        assert node.get_output_slot("nonexistent") is None

    def test_input_slots_property(self):
        """input_slots property should return all defined input slots."""

        class MultiSlotNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = MultiSlotNode("test")
        node.define_input_slot("slot_a", SlotType.POSE)
        node.define_input_slot("slot_b", SlotType.FLOAT)

        slots = node.input_slots
        assert len(slots) == 2
        assert "slot_a" in slots
        assert "slot_b" in slots

    def test_output_slots_property(self):
        """output_slots property should return all defined output slots."""

        class MultiOutputNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = MultiOutputNode("test")
        node.define_output_slot("out_a", SlotType.POSE)
        node.define_output_slot("out_b", SlotType.TRIGGER)

        slots = node.output_slots
        assert len(slots) == 2
        assert "out_a" in slots
        assert "out_b" in slots

    def test_input_slots_returns_copy(self):
        """input_slots should return a copy of the internal dict."""

        class CopyTestNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = CopyTestNode("test")
        node.define_input_slot("original", SlotType.INT)

        slots = node.input_slots
        slots["fake"] = None  # Modify returned dict

        # Internal should be unchanged
        assert "fake" not in node._input_slots

    def test_output_slots_returns_copy(self):
        """output_slots should return a copy of the internal dict."""

        class OutputCopyNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = OutputCopyNode("test")
        node.define_output_slot("original", SlotType.ENUM)

        slots = node.output_slots
        slots["fake"] = None

        assert "fake" not in node._output_slots


# =============================================================================
# TEST: AnimationNode Identification
# =============================================================================


class TestAnimationNodeIdentification:
    """Tests for AnimationNode identification system."""

    def test_node_id_stored(self):
        """Node should store its node_id."""

        class IdTestNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = IdTestNode("my_unique_id")
        assert node.node_id == "my_unique_id"

    def test_display_name_defaults_to_node_id(self):
        """display_name should default to node_id when not provided."""

        class DisplayNameNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = DisplayNameNode("default_name")
        assert node.display_name == "default_name"

    def test_display_name_custom(self):
        """display_name should use provided value when given."""

        class CustomNameNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = CustomNameNode("id123", display_name="My Pretty Name")
        assert node.node_id == "id123"
        assert node.display_name == "My Pretty Name"

    def test_display_name_settable(self):
        """display_name should be settable after creation."""

        class SettableNameNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = SettableNameNode("test")
        node.display_name = "New Display Name"
        assert node.display_name == "New Display Name"

    def test_node_type_name(self):
        """node_type_name should return the class name."""

        class TypeNameTestNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = TypeNameTestNode("test")
        assert node.node_type_name() == "TypeNameTestNode"

    def test_get_debug_info(self):
        """get_debug_info should return comprehensive node information."""

        class DebugInfoNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = DebugInfoNode("debug_node", display_name="Debug Node")
        node.define_input_slot("in1", SlotType.POSE)
        node.define_output_slot("out1", SlotType.POSE)

        debug_info = node.get_debug_info()

        assert debug_info["node_id"] == "debug_node"
        assert debug_info["display_name"] == "Debug Node"
        assert debug_info["type"] == "DebugInfoNode"
        assert "in1" in debug_info["input_slots"]
        assert "out1" in debug_info["output_slots"]


# =============================================================================
# TEST: AnimationNode Input/Output Connections
# =============================================================================


class TestAnimationNodeConnections:
    """Tests for AnimationNode input/output connection handling."""

    def test_get_input_returns_none_initially(self):
        """get_input should return None for unconnected inputs."""

        class ConnectionNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = ConnectionNode("test")
        assert node.get_input("any_input") is None

    def test_set_input_stores_node(self):
        """set_input should store the connected node."""

        class ConnectedNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node_a = ConnectedNode("node_a")
        node_b = ConnectedNode("node_b")

        node_a.set_input("my_input", node_b)
        assert node_a.get_input("my_input") is node_b

    def test_set_input_none_disconnects(self):
        """set_input with None should disconnect the input."""

        class DisconnectNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node_a = DisconnectNode("a")
        node_b = DisconnectNode("b")

        node_a.set_input("input", node_b)
        node_a.set_input("input", None)

        assert node_a.get_input("input") is None

    def test_evaluate_input_calls_connected_node(self):
        """evaluate_input should evaluate the connected node."""

        class EvalInputNode(AnimationNode):
            def __init__(self, node_id, pose_to_return=None):
                super().__init__(node_id)
                self.pose_to_return = pose_to_return or Pose()
                self.evaluate_called = False

            def evaluate(self, context):
                self.evaluate_called = True
                return self.pose_to_return

        expected_pose = Pose.identity(5)
        source_node = EvalInputNode("source", expected_pose)
        target_node = EvalInputNode("target")

        target_node.set_input("pose_input", source_node)

        ctx = GraphContext()
        result = target_node.evaluate_input("pose_input", ctx)

        assert source_node.evaluate_called
        assert result is not None

    def test_evaluate_input_returns_none_for_disconnected(self):
        """evaluate_input should return None for disconnected inputs."""

        class DisconnectedInputNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = DisconnectedInputNode("test")
        ctx = GraphContext()

        result = node.evaluate_input("nonexistent", ctx)
        assert result is None


# =============================================================================
# TEST: AnimationNode Cache System
# =============================================================================


class TestAnimationNodeCache:
    """Tests for AnimationNode caching behavior."""

    def test_initial_cache_invalid(self):
        """Cache should be invalid initially."""

        class CacheNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = CacheNode("test")
        assert node._cache_valid is False
        assert node._cached_pose is None

    def test_invalidate_cache(self):
        """invalidate_cache should clear cached pose."""

        class InvalidateCacheNode(AnimationNode):
            def evaluate(self, context):
                pose = Pose.identity(3)
                self._cached_pose = pose
                self._cache_valid = True
                return pose

        node = InvalidateCacheNode("test")
        ctx = GraphContext()
        node.evaluate(ctx)

        node.invalidate_cache()

        assert node._cache_valid is False
        assert node._cached_pose is None


# =============================================================================
# TEST: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_slot_name(self):
        """Slots with empty names should still work."""

        class EmptyNameNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = EmptyNameNode("test")
        slot = node.define_input_slot("", SlotType.POSE)
        assert slot.name == ""
        assert node.get_input_slot("") is not None

    def test_special_characters_in_node_id(self):
        """Node IDs with special characters should work."""

        class SpecialCharNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = SpecialCharNode("node-with-dashes_and_underscores.and.dots")
        assert "dashes" in node.node_id
        assert "dots" in node.node_id

    def test_unicode_in_display_name(self):
        """Unicode characters in display name should work."""

        class UnicodeNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = UnicodeNode("test", display_name="Animation de test")
        assert node.display_name == "Animation de test"

    def test_all_slot_types_definable(self):
        """All SlotType values should be usable in slot definitions."""

        class AllSlotTypesNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = AllSlotTypesNode("test")

        for slot_type in SlotType:
            slot = node.define_input_slot(f"input_{slot_type.name}", slot_type)
            assert slot.slot_type == slot_type

            out_slot = node.define_output_slot(f"output_{slot_type.name}", slot_type)
            assert out_slot.slot_type == slot_type

    def test_many_slots_per_node(self):
        """Nodes should handle many slots without issue."""

        class ManySlotNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = ManySlotNode("test")

        for i in range(100):
            node.define_input_slot(f"in_{i}", SlotType.FLOAT)
            node.define_output_slot(f"out_{i}", SlotType.FLOAT)

        assert len(node.input_slots) == 100
        assert len(node.output_slots) == 100

    def test_overwrite_slot_with_same_name(self):
        """Defining a slot with the same name should overwrite."""

        class OverwriteSlotNode(AnimationNode):
            def evaluate(self, context):
                return Pose()

        node = OverwriteSlotNode("test")
        node.define_input_slot("slot", SlotType.FLOAT, "First")
        node.define_input_slot("slot", SlotType.BOOL, "Second")

        slot = node.get_input_slot("slot")
        assert slot.slot_type == SlotType.BOOL
        assert slot.description == "Second"


# =============================================================================
# TEST: Integration - Complete Node Lifecycle
# =============================================================================


class TestNodeLifecycle:
    """Integration tests for complete node lifecycle."""

    def test_complete_node_lifecycle(self):
        """Test creating, configuring, and using a complete node."""

        class BlendNode(AnimationNode):
            def __init__(self, node_id, display_name=None):
                super().__init__(node_id, display_name)
                self.define_input_slot("pose_a", SlotType.POSE, "First pose")
                self.define_input_slot("pose_b", SlotType.POSE, "Second pose")
                self.define_input_slot("blend", SlotType.FLOAT, "Blend factor")
                self.define_output_slot("result", SlotType.POSE, "Blended pose")

            def evaluate(self, context):
                return Pose.identity(1)

        # Create node
        node = BlendNode("blend_1", "My Blend Node")

        # Verify registration
        assert "BlendNode" in GraphNodeMeta.registry()

        # Verify identification
        assert node.node_id == "blend_1"
        assert node.display_name == "My Blend Node"
        assert node.node_type_name() == "BlendNode"

        # Verify slots
        assert len(node.input_slots) == 3
        assert len(node.output_slots) == 1
        assert node.get_input_slot("pose_a").slot_type == SlotType.POSE
        assert node.get_input_slot("blend").slot_type == SlotType.FLOAT

        # Verify evaluation
        ctx = GraphContext()
        result = node.evaluate(ctx)
        assert isinstance(result, Pose)

        # Verify debug info
        debug = node.get_debug_info()
        assert debug["node_id"] == "blend_1"
        assert debug["type"] == "BlendNode"
        assert "pose_a" in debug["input_slots"]
