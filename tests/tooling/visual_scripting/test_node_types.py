"""
Tests for FlowForge node types.

Tests all core node types including events, flow control, functions, and variables.
"""

import pytest

from engine.tooling.visual_scripting.node_types import (
    Pin,
    PinDirection,
    PinKind,
    Node,
    NodeCategory,
    NodeMetadata,
    # Event nodes
    EventNode,
    BeginPlayNode,
    TickNode,
    InputActionNode,
    CustomEventNode,
    # Flow control
    FlowControlNode,
    BranchNode,
    SequenceNode,
    ForLoopNode,
    WhileLoopNode,
    ForEachLoopNode,
    SwitchNode,
    GateNode,
    DoOnceNode,
    FlipFlopNode,
    # Functions
    FunctionNode,
    PureFunctionNode,
    CallFunctionNode,
    # Variables
    VariableNode,
    GetVariableNode,
    SetVariableNode,
    # Macros
    MacroNode,
    MacroInputNode,
    MacroOutputNode,
    # Literals
    LiteralNode,
    BoolLiteralNode,
    IntLiteralNode,
    FloatLiteralNode,
    StringLiteralNode,
    VectorLiteralNode,
    # Utilities
    PrintStringNode,
    DelayNode,
    # Registry
    NODE_REGISTRY,
    register_node,
    get_node_class,
    get_nodes_by_category,
    search_nodes,
)
from engine.tooling.visual_scripting.data_types import (
    BoolType,
    IntType,
    FloatType,
    StringType,
    ExecutionType,
    WildcardType,
)


class TestPin:
    """Tests for Pin class."""

    def test_create_data_pin(self):
        pin = Pin(
            id="test_pin",
            name="Value",
            direction=PinDirection.INPUT,
            kind=PinKind.DATA,
            data_type=IntType
        )
        assert pin.name == "Value"
        assert pin.direction == PinDirection.INPUT
        assert pin.kind == PinKind.DATA
        assert pin.data_type == IntType

    def test_create_execution_pin(self):
        pin = Pin(
            id="exec_pin",
            name="In",
            direction=PinDirection.INPUT,
            kind=PinKind.EXECUTION
        )
        assert pin.kind == PinKind.EXECUTION
        assert pin.data_type == ExecutionType

    def test_auto_generate_id(self):
        pin = Pin(
            id=None,
            name="Test",
            direction=PinDirection.OUTPUT,
            kind=PinKind.DATA
        )
        assert pin.id is not None
        assert len(pin.id) > 0

    def test_default_value(self):
        pin = Pin(
            id="pin",
            name="Value",
            direction=PinDirection.INPUT,
            kind=PinKind.DATA,
            data_type=IntType,
            default_value=42
        )
        assert pin.get_value() == 42

    def test_set_value(self):
        pin = Pin(
            id="pin",
            name="Value",
            direction=PinDirection.INPUT,
            kind=PinKind.DATA,
            data_type=IntType,
            default_value=0
        )
        pin.set_value(100)
        assert pin.get_value() == 100

    def test_set_value_coerces_type(self):
        pin = Pin(
            id="pin",
            name="Value",
            direction=PinDirection.INPUT,
            kind=PinKind.DATA,
            data_type=IntType,
            default_value=0
        )
        pin.set_value(3.7)  # Should coerce to int
        assert pin.get_value() == 3

    def test_can_connect_same_kind(self):
        output_pin = Pin(
            id="out",
            name="Out",
            direction=PinDirection.OUTPUT,
            kind=PinKind.DATA,
            data_type=IntType
        )
        input_pin = Pin(
            id="in",
            name="In",
            direction=PinDirection.INPUT,
            kind=PinKind.DATA,
            data_type=IntType
        )
        assert output_pin.can_connect_to(input_pin)

    def test_cannot_connect_same_direction(self):
        pin1 = Pin(
            id="out1",
            name="Out1",
            direction=PinDirection.OUTPUT,
            kind=PinKind.DATA,
            data_type=IntType
        )
        pin2 = Pin(
            id="out2",
            name="Out2",
            direction=PinDirection.OUTPUT,
            kind=PinKind.DATA,
            data_type=IntType
        )
        assert not pin1.can_connect_to(pin2)

    def test_cannot_connect_different_kinds(self):
        exec_pin = Pin(
            id="exec",
            name="Exec",
            direction=PinDirection.OUTPUT,
            kind=PinKind.EXECUTION
        )
        data_pin = Pin(
            id="data",
            name="Data",
            direction=PinDirection.INPUT,
            kind=PinKind.DATA,
            data_type=IntType
        )
        assert not exec_pin.can_connect_to(data_pin)

    def test_execution_pins_connect(self):
        out_exec = Pin(
            id="out",
            name="Out",
            direction=PinDirection.OUTPUT,
            kind=PinKind.EXECUTION
        )
        in_exec = Pin(
            id="in",
            name="In",
            direction=PinDirection.INPUT,
            kind=PinKind.EXECUTION
        )
        assert out_exec.can_connect_to(in_exec)


class TestNodeMetadata:
    """Tests for NodeMetadata."""

    def test_create_metadata(self):
        meta = NodeMetadata(
            display_name="Test Node",
            category=NodeCategory.UTILITY,
            description="A test node",
            keywords=["test", "example"]
        )
        assert meta.display_name == "Test Node"
        assert meta.category == NodeCategory.UTILITY
        assert "test" in meta.keywords

    def test_default_values(self):
        meta = NodeMetadata(
            display_name="Node",
            category=NodeCategory.CUSTOM
        )
        assert meta.is_latent is False
        assert meta.is_pure is False
        assert meta.is_deprecated is False


class TestBeginPlayNode:
    """Tests for BeginPlayNode."""

    def test_create(self):
        node = BeginPlayNode()
        assert node.id is not None

    def test_has_output_pin(self):
        node = BeginPlayNode()
        assert "Out" in node.output_pins

    def test_no_input_pins(self):
        node = BeginPlayNode()
        assert len(node.input_pins) == 0

    def test_metadata_category(self):
        meta = BeginPlayNode.get_metadata()
        assert meta.category == NodeCategory.EVENT

    def test_execute_returns_output(self):
        node = BeginPlayNode()
        result = node.execute(None)
        assert result == node.output_pins["Out"].id


class TestTickNode:
    """Tests for TickNode."""

    def test_has_delta_time_output(self):
        node = TickNode()
        assert "DeltaTime" in node.output_pins

    def test_delta_time_type(self):
        node = TickNode()
        delta_pin = node.output_pins["DeltaTime"]
        assert delta_pin.data_type == FloatType

    def test_execute_with_context(self):
        node = TickNode()

        class MockContext:
            delta_time = 0.016

        result = node.execute(MockContext())
        assert result is not None
        assert node.output_pins["DeltaTime"].get_value() == 0.016


class TestInputActionNode:
    """Tests for InputActionNode."""

    def test_create_with_action_name(self):
        node = InputActionNode(action_name="Jump")
        assert node.action_name == "Jump"

    def test_has_pressed_and_released(self):
        node = InputActionNode()
        assert "Pressed" in node.output_pins
        assert "Released" in node.output_pins

    def test_has_key_output(self):
        node = InputActionNode()
        assert "Key" in node.output_pins


class TestCustomEventNode:
    """Tests for CustomEventNode."""

    def test_create_with_event_name(self):
        node = CustomEventNode(event_name="OnDamage")
        assert node.event_name == "OnDamage"

    def test_add_parameter(self):
        node = CustomEventNode(event_name="OnDamage")
        pin = node.add_parameter("Amount", FloatType)
        assert "Amount" in node.output_pins
        assert pin.data_type == FloatType


class TestBranchNode:
    """Tests for BranchNode."""

    def test_has_condition_input(self):
        node = BranchNode()
        assert "Condition" in node.input_pins
        assert node.input_pins["Condition"].data_type == BoolType

    def test_has_true_false_outputs(self):
        node = BranchNode()
        assert "True" in node.output_pins
        assert "False" in node.output_pins

    def test_execute_true_branch(self):
        node = BranchNode()
        node.input_pins["Condition"].set_value(True)
        result = node.execute(None)
        assert result == node.output_pins["True"].id

    def test_execute_false_branch(self):
        node = BranchNode()
        node.input_pins["Condition"].set_value(False)
        result = node.execute(None)
        assert result == node.output_pins["False"].id

    def test_metadata_is_flow_control(self):
        meta = BranchNode.get_metadata()
        assert meta.category == NodeCategory.FLOW_CONTROL


class TestSequenceNode:
    """Tests for SequenceNode."""

    def test_default_outputs(self):
        node = SequenceNode()
        assert "Then 0" in node.output_pins
        assert "Then 1" in node.output_pins

    def test_custom_output_count(self):
        node = SequenceNode(num_outputs=4)
        assert len([p for p in node.output_pins if "Then" in p]) == 4

    def test_add_output(self):
        node = SequenceNode(num_outputs=2)
        node.add_output()
        assert node.num_outputs == 3
        assert "Then 2" in node.output_pins

    def test_execute_sequence(self):
        node = SequenceNode(num_outputs=3)
        result1 = node.execute(None)
        result2 = node.execute(None)
        result3 = node.execute(None)

        assert result1 == node.output_pins["Then 0"].id
        assert result2 == node.output_pins["Then 1"].id
        assert result3 == node.output_pins["Then 2"].id


class TestForLoopNode:
    """Tests for ForLoopNode."""

    def test_has_index_pins(self):
        node = ForLoopNode()
        assert "FirstIndex" in node.input_pins
        assert "LastIndex" in node.input_pins
        assert "Index" in node.output_pins

    def test_has_loop_body_and_completed(self):
        node = ForLoopNode()
        assert "LoopBody" in node.output_pins
        assert "Completed" in node.output_pins

    def test_is_latent(self):
        meta = ForLoopNode.get_metadata()
        assert meta.is_latent is True

    def test_execute_loop(self):
        node = ForLoopNode()
        node.input_pins["FirstIndex"].set_value(0)
        node.input_pins["LastIndex"].set_value(2)

        # First iteration
        result = node.execute(None)
        assert result == node.output_pins["LoopBody"].id
        assert node.output_pins["Index"].get_value() == 0

    def test_reset(self):
        node = ForLoopNode()
        node._current_index = 5
        node.reset()
        assert node._current_index == 0


class TestWhileLoopNode:
    """Tests for WhileLoopNode."""

    def test_has_condition_input(self):
        node = WhileLoopNode()
        assert "Condition" in node.input_pins

    def test_execute_while_true(self):
        node = WhileLoopNode()
        node.input_pins["Condition"].set_value(True)
        result = node.execute(None)
        assert result == node.output_pins["LoopBody"].id

    def test_execute_while_false(self):
        node = WhileLoopNode()
        node.input_pins["Condition"].set_value(False)
        result = node.execute(None)
        assert result == node.output_pins["Completed"].id


class TestForEachLoopNode:
    """Tests for ForEachLoopNode."""

    def test_has_array_input(self):
        node = ForEachLoopNode()
        assert "Array" in node.input_pins

    def test_has_element_and_index_outputs(self):
        node = ForEachLoopNode()
        assert "Element" in node.output_pins
        assert "Index" in node.output_pins

    def test_execute_foreach(self):
        node = ForEachLoopNode()
        node.input_pins["Array"].set_value([10, 20, 30])

        result = node.execute(None)
        assert result == node.output_pins["LoopBody"].id
        assert node.output_pins["Index"].get_value() == 0
        assert node.output_pins["Element"].get_value() == 10


class TestSwitchNode:
    """Tests for SwitchNode."""

    def test_default_cases(self):
        node = SwitchNode()
        assert "Case 0" in node.output_pins
        assert "Case 1" in node.output_pins
        assert "Default" in node.output_pins

    def test_add_case(self):
        node = SwitchNode(cases=[0, 1])
        node.add_case(5)
        assert "Case 5" in node.output_pins

    def test_execute_matching_case(self):
        node = SwitchNode(cases=[0, 1, 2])
        node.input_pins["Selection"].set_value(1)
        result = node.execute(None)
        assert result == node.output_pins["Case 1"].id

    def test_execute_default(self):
        node = SwitchNode(cases=[0, 1, 2])
        node.input_pins["Selection"].set_value(99)
        result = node.execute(None)
        assert result == node.output_pins["Default"].id


class TestGateNode:
    """Tests for GateNode."""

    def test_has_control_inputs(self):
        node = GateNode()
        assert "Enter" in node.input_pins
        assert "Open" in node.input_pins
        assert "Close" in node.input_pins
        assert "Toggle" in node.input_pins

    def test_open_close_toggle(self):
        node = GateNode()
        assert node._is_open is True

        node.close()
        assert node._is_open is False

        node.open()
        assert node._is_open is True

        node.toggle()
        assert node._is_open is False

    def test_execute_when_open(self):
        node = GateNode()
        node.open()
        result = node.execute(None)
        assert result == node.output_pins["Exit"].id

    def test_execute_when_closed(self):
        node = GateNode()
        node.close()
        result = node.execute(None)
        assert result is None


class TestDoOnceNode:
    """Tests for DoOnceNode."""

    def test_execute_once(self):
        node = DoOnceNode()
        result1 = node.execute(None)
        result2 = node.execute(None)

        assert result1 == node.output_pins["Completed"].id
        assert result2 is None

    def test_reset_allows_reexecute(self):
        node = DoOnceNode()
        node.execute(None)
        node.reset()
        result = node.execute(None)
        assert result == node.output_pins["Completed"].id


class TestFlipFlopNode:
    """Tests for FlipFlopNode."""

    def test_alternates_outputs(self):
        node = FlipFlopNode()

        result1 = node.execute(None)
        result2 = node.execute(None)
        result3 = node.execute(None)

        assert result1 == node.output_pins["A"].id
        assert result2 == node.output_pins["B"].id
        assert result3 == node.output_pins["A"].id

    def test_is_a_output(self):
        node = FlipFlopNode()
        node.execute(None)
        assert node.output_pins["IsA"].get_value() is True

        node.execute(None)
        assert node.output_pins["IsA"].get_value() is False


class TestCallFunctionNode:
    """Tests for CallFunctionNode."""

    def test_create_with_function_name(self):
        node = CallFunctionNode(function_name="MyFunc", target_class="MyClass")
        assert node.function_name == "MyFunc"
        assert node.target_class == "MyClass"

    def test_has_target_input(self):
        node = CallFunctionNode()
        assert "Target" in node.input_pins


class TestGetVariableNode:
    """Tests for GetVariableNode."""

    def test_create_with_variable(self):
        node = GetVariableNode(variable_name="Health", variable_type=FloatType)
        assert node.variable_name == "Health"
        assert node.variable_type == FloatType

    def test_has_value_output(self):
        node = GetVariableNode(variable_name="Health", variable_type=FloatType)
        assert "Value" in node.output_pins
        assert node.output_pins["Value"].data_type == FloatType

    def test_is_pure(self):
        meta = GetVariableNode.get_metadata()
        assert meta.is_pure is True


class TestSetVariableNode:
    """Tests for SetVariableNode."""

    def test_has_value_input(self):
        node = SetVariableNode(variable_name="Health", variable_type=FloatType)
        assert "Value" in node.input_pins

    def test_has_execution_pins(self):
        node = SetVariableNode(variable_name="Health")
        assert "In" in node.input_pins
        assert "Out" in node.output_pins

    def test_has_new_value_output(self):
        node = SetVariableNode(variable_name="Health")
        assert "NewValue" in node.output_pins


class TestMacroNode:
    """Tests for MacroNode."""

    def test_create_with_macro_name(self):
        node = MacroNode(macro_name="MyMacro", macro_id="macro_123")
        assert node.macro_name == "MyMacro"
        assert node.macro_id == "macro_123"

    def test_set_interface(self):
        node = MacroNode(macro_name="TestMacro")
        node.set_interface(
            inputs=[("In", ExecutionType, True), ("Value", IntType, False)],
            outputs=[("Out", ExecutionType, True), ("Result", IntType, False)]
        )

        assert "In" in node.input_pins
        assert "Value" in node.input_pins
        assert "Out" in node.output_pins
        assert "Result" in node.output_pins


class TestLiteralNodes:
    """Tests for literal/constant nodes."""

    def test_bool_literal(self):
        node = BoolLiteralNode(value=True)
        assert node.value is True
        assert node.output_pins["Value"].get_value() is True

    def test_int_literal(self):
        node = IntLiteralNode(value=42)
        assert node.value == 42
        assert node.output_pins["Value"].get_value() == 42

    def test_float_literal(self):
        node = FloatLiteralNode(value=3.14)
        assert abs(node.value - 3.14) < 0.0001

    def test_string_literal(self):
        node = StringLiteralNode(value="Hello")
        assert node.value == "Hello"

    def test_set_value_updates_pin(self):
        node = IntLiteralNode(value=0)
        node.value = 100
        assert node.output_pins["Value"].get_value() == 100


class TestPrintStringNode:
    """Tests for PrintStringNode."""

    def test_has_string_input(self):
        node = PrintStringNode()
        assert "String" in node.input_pins
        assert node.input_pins["String"].data_type == StringType

    def test_has_options(self):
        node = PrintStringNode()
        assert "PrintToScreen" in node.input_pins
        assert "PrintToLog" in node.input_pins
        assert "Duration" in node.input_pins


class TestDelayNode:
    """Tests for DelayNode."""

    def test_has_duration_input(self):
        node = DelayNode()
        assert "Duration" in node.input_pins
        assert node.input_pins["Duration"].data_type == FloatType

    def test_is_latent(self):
        meta = DelayNode.get_metadata()
        assert meta.is_latent is True


class TestNodeRegistry:
    """Tests for node registry functions."""

    def test_registry_has_nodes(self):
        assert len(NODE_REGISTRY) > 0

    def test_get_node_class(self):
        node_class = get_node_class("Branch")
        assert node_class == BranchNode

    def test_get_nodes_by_category(self):
        events = get_nodes_by_category(NodeCategory.EVENT)
        assert BeginPlayNode in events
        assert TickNode in events

    def test_search_nodes_by_name(self):
        results = search_nodes("branch")
        assert BranchNode in results

    def test_search_nodes_by_keyword(self):
        results = search_nodes("if")
        assert BranchNode in results

    def test_register_node(self):
        class TestNode(Node):
            def _setup_pins(self):
                pass

            @classmethod
            def get_metadata(cls):
                return NodeMetadata(
                    display_name="TestNode",
                    category=NodeCategory.CUSTOM
                )

            def execute(self, context):
                return None

        register_node(TestNode)
        assert "TestNode" in NODE_REGISTRY


class TestNodeValidation:
    """Tests for node validation."""

    def test_validate_returns_list(self):
        node = BranchNode()
        errors = node.validate()
        assert isinstance(errors, list)

    def test_deprecated_node_warning(self):
        class DeprecatedNode(Node):
            def _setup_pins(self):
                pass

            @classmethod
            def get_metadata(cls):
                return NodeMetadata(
                    display_name="OldNode",
                    category=NodeCategory.CUSTOM,
                    is_deprecated=True,
                    deprecated_message="Use NewNode instead"
                )

            def execute(self, context):
                return None

        node = DeprecatedNode()
        errors = node.validate()
        assert any("deprecated" in e.lower() for e in errors)


class TestNodePinManagement:
    """Tests for node pin management methods."""

    def test_get_input_pin(self):
        node = BranchNode()
        pin = node.get_input_pin("Condition")
        assert pin is not None
        assert pin.name == "Condition"

    def test_get_output_pin(self):
        node = BranchNode()
        pin = node.get_output_pin("True")
        assert pin is not None
        assert pin.name == "True"

    def test_get_all_pins(self):
        node = BranchNode()
        all_pins = node.get_all_pins()
        assert len(all_pins) > 0
        assert all(isinstance(p, Pin) for p in all_pins)

    def test_create_custom_pins(self):
        node = BranchNode()
        new_input = node.create_data_input("Custom", IntType, 10)
        new_output = node.create_data_output("Result", FloatType)

        assert "Custom" in node.input_pins
        assert "Result" in node.output_pins
        assert new_input.default_value == 10
