"""
FlowForge Node Types - Core node types for visual scripting.

Provides the fundamental node types:
- EventNode: BeginPlay, Tick, InputAction, Custom Events
- FunctionNode: Pure functions, function calls
- FlowControlNode: Branch, Sequence, ForLoop, WhileLoop, Switch, Gate
- VariableNode: Get, Set, local/member variables
- MacroNode: Reusable subgraph references
- Literal nodes for constants
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .data_types import (
    BlueprintType,
    BoolType,
    ExecutionType,
    FloatType,
    IntType,
    ObjectType,
    StringType,
    Vector3Type,
    WildcardType,
)


class PinDirection(Enum):
    """Direction of a node pin."""
    INPUT = auto()
    OUTPUT = auto()


class PinKind(Enum):
    """Kind of pin (execution or data)."""
    EXECUTION = auto()
    DATA = auto()


@dataclass
class Pin:
    """A connection point on a node."""
    id: str
    name: str
    direction: PinDirection
    kind: PinKind
    data_type: Type[BlueprintType] = WildcardType
    default_value: Any = None
    is_connected: bool = False
    is_hidden: bool = False
    is_array: bool = False
    description: str = ""
    connected_to: List[str] = field(default_factory=list)  # List of pin IDs

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.kind == PinKind.EXECUTION:
            self.data_type = ExecutionType

    def can_connect_to(self, other: Pin) -> bool:
        """Check if this pin can connect to another pin."""
        # Cannot connect to same direction
        if self.direction == other.direction:
            return False

        # Both must be same kind
        if self.kind != other.kind:
            return False

        # For execution pins, just check direction
        if self.kind == PinKind.EXECUTION:
            return True

        # For data pins, check type compatibility
        from .data_types import can_connect_types

        if self.direction == PinDirection.OUTPUT:
            return can_connect_types(self.data_type, other.data_type)
        else:
            return can_connect_types(other.data_type, self.data_type)

    def get_value(self) -> Any:
        """Get the current value of this pin."""
        return self.default_value

    def set_value(self, value: Any) -> None:
        """Set the value of this pin."""
        if self.data_type:
            self.default_value = self.data_type.coerce(value)
        else:
            self.default_value = value


class NodeCategory(Enum):
    """Categories for organizing nodes."""
    EVENT = "Events"
    FLOW_CONTROL = "Flow Control"
    FUNCTION = "Functions"
    VARIABLE = "Variables"
    MATH = "Math"
    STRING = "String"
    VECTOR = "Vector"
    TRANSFORM = "Transform"
    OBJECT = "Object"
    ARRAY = "Array"
    UTILITY = "Utilities"
    MACRO = "Macros"
    CUSTOM = "Custom"
    DEBUG = "Debug"
    AI = "AI"
    PHYSICS = "Physics"
    AUDIO = "Audio"
    UI = "UI"


@dataclass
class NodeMetadata:
    """Metadata about a node type."""
    display_name: str
    category: NodeCategory
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    icon: str = ""
    color: Tuple[int, int, int] = (64, 64, 64)
    is_latent: bool = False  # Node that takes time (async)
    is_pure: bool = False  # No side effects, no execution pins
    is_const: bool = False  # Doesn't modify any state
    is_deprecated: bool = False
    deprecated_message: str = ""
    compact_node_title: str = ""  # Short title for compact view


class Node(ABC):
    """Base class for all blueprint nodes."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0),
        comment: str = ""
    ):
        self.id = node_id or str(uuid.uuid4())
        self.position = position
        self.comment = comment
        self.is_disabled = False
        self.input_pins: Dict[str, Pin] = {}
        self.output_pins: Dict[str, Pin] = {}
        self._setup_pins()

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> NodeMetadata:
        """Return metadata about this node type."""
        pass

    @abstractmethod
    def _setup_pins(self) -> None:
        """Initialize input and output pins."""
        pass

    @abstractmethod
    def execute(self, context: Any) -> Optional[str]:
        """Execute this node. Returns the ID of the next execution pin to follow."""
        pass

    def get_input_pin(self, name: str) -> Optional[Pin]:
        """Get an input pin by name."""
        return self.input_pins.get(name)

    def get_output_pin(self, name: str) -> Optional[Pin]:
        """Get an output pin by name."""
        return self.output_pins.get(name)

    def add_input_pin(self, pin: Pin) -> Pin:
        """Add an input pin."""
        self.input_pins[pin.name] = pin
        return pin

    def add_output_pin(self, pin: Pin) -> Pin:
        """Add an output pin."""
        self.output_pins[pin.name] = pin
        return pin

    def create_execution_input(self, name: str = "In") -> Pin:
        """Create a standard execution input pin."""
        pin = Pin(
            id=f"{self.id}_{name}_exec_in",
            name=name,
            direction=PinDirection.INPUT,
            kind=PinKind.EXECUTION
        )
        return self.add_input_pin(pin)

    def create_execution_output(self, name: str = "Out") -> Pin:
        """Create a standard execution output pin."""
        pin = Pin(
            id=f"{self.id}_{name}_exec_out",
            name=name,
            direction=PinDirection.OUTPUT,
            kind=PinKind.EXECUTION
        )
        return self.add_output_pin(pin)

    def create_data_input(
        self,
        name: str,
        data_type: Type[BlueprintType],
        default_value: Any = None
    ) -> Pin:
        """Create a data input pin."""
        pin = Pin(
            id=f"{self.id}_{name}_data_in",
            name=name,
            direction=PinDirection.INPUT,
            kind=PinKind.DATA,
            data_type=data_type,
            default_value=default_value if default_value is not None else data_type.default_value
        )
        return self.add_input_pin(pin)

    def create_data_output(
        self,
        name: str,
        data_type: Type[BlueprintType],
        default_value: Any = None
    ) -> Pin:
        """Create a data output pin."""
        pin = Pin(
            id=f"{self.id}_{name}_data_out",
            name=name,
            direction=PinDirection.OUTPUT,
            kind=PinKind.DATA,
            data_type=data_type,
            default_value=default_value if default_value is not None else data_type.default_value
        )
        return self.add_output_pin(pin)

    def get_all_pins(self) -> List[Pin]:
        """Get all pins on this node."""
        return list(self.input_pins.values()) + list(self.output_pins.values())

    def validate(self) -> List[str]:
        """Validate this node. Returns list of error messages."""
        errors = []
        metadata = self.get_metadata()
        if metadata.is_deprecated:
            errors.append(f"Node '{metadata.display_name}' is deprecated: {metadata.deprecated_message}")
        return errors


# =============================================================================
# EVENT NODES
# =============================================================================


class EventNode(Node):
    """Base class for event nodes (entry points)."""

    def _setup_pins(self) -> None:
        self.create_execution_output("Out")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Event",
            category=NodeCategory.EVENT,
            description="Base event node",
            color=(180, 0, 0)
        )


class BeginPlayNode(EventNode):
    """Event that fires when the game/level starts."""

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Event BeginPlay",
            category=NodeCategory.EVENT,
            description="Called when the game starts or actor is spawned",
            keywords=["start", "spawn", "begin", "init"],
            color=(180, 0, 0)
        )

    def execute(self, context: Any) -> Optional[str]:
        out_pin = self.get_output_pin("Out")
        return out_pin.id if out_pin else None


class TickNode(EventNode):
    """Event that fires every frame."""

    def _setup_pins(self) -> None:
        super()._setup_pins()
        self.create_data_output("DeltaTime", FloatType, 0.016)

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Event Tick",
            category=NodeCategory.EVENT,
            description="Called every frame",
            keywords=["update", "frame", "loop"],
            color=(180, 0, 0)
        )

    def execute(self, context: Any) -> Optional[str]:
        # Set delta time from context
        delta_pin = self.get_output_pin("DeltaTime")
        if delta_pin and hasattr(context, "delta_time"):
            delta_pin.set_value(context.delta_time)
        out_pin = self.get_output_pin("Out")
        return out_pin.id if out_pin else None


class InputActionNode(EventNode):
    """Event that fires on input action."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0),
        action_name: str = "Jump"
    ):
        self.action_name = action_name
        super().__init__(node_id, position)

    def _setup_pins(self) -> None:
        self.create_execution_output("Pressed")
        self.create_execution_output("Released")
        self.create_data_output("Key", StringType, "")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Input Action",
            category=NodeCategory.EVENT,
            description="Called when an input action is triggered",
            keywords=["input", "key", "button", "action"],
            color=(180, 0, 0)
        )

    def execute(self, context: Any) -> Optional[str]:
        # Determine which output to use based on input state
        if hasattr(context, "input_pressed") and context.input_pressed:
            return self.get_output_pin("Pressed").id
        return self.get_output_pin("Released").id


class CustomEventNode(EventNode):
    """User-defined custom event."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0),
        event_name: str = "CustomEvent"
    ):
        self.event_name = event_name
        self.parameter_pins: List[Tuple[str, Type[BlueprintType]]] = []
        super().__init__(node_id, position)

    def _setup_pins(self) -> None:
        self.create_execution_output("Out")

    def add_parameter(self, name: str, data_type: Type[BlueprintType]) -> Pin:
        """Add a parameter output pin."""
        self.parameter_pins.append((name, data_type))
        return self.create_data_output(name, data_type)

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Custom Event",
            category=NodeCategory.EVENT,
            description="User-defined event that can be called from other blueprints",
            keywords=["custom", "event", "function", "callback"],
            color=(180, 0, 0)
        )

    def execute(self, context: Any) -> Optional[str]:
        return self.get_output_pin("Out").id


# =============================================================================
# FLOW CONTROL NODES
# =============================================================================


class FlowControlNode(Node):
    """Base class for flow control nodes."""

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Flow Control",
            category=NodeCategory.FLOW_CONTROL,
            description="Base flow control node",
            color=(255, 255, 255)
        )


class BranchNode(FlowControlNode):
    """If/else branching node."""

    def _setup_pins(self) -> None:
        self.create_execution_input("In")
        self.create_data_input("Condition", BoolType, False)
        self.create_execution_output("True")
        self.create_execution_output("False")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Branch",
            category=NodeCategory.FLOW_CONTROL,
            description="If/else conditional branching",
            keywords=["if", "else", "condition", "branch"],
            compact_node_title="Branch",
            color=(255, 255, 255)
        )

    def execute(self, context: Any) -> Optional[str]:
        condition = self.get_input_pin("Condition").get_value()
        if condition:
            return self.get_output_pin("True").id
        return self.get_output_pin("False").id


class SequenceNode(FlowControlNode):
    """Execute multiple outputs in sequence."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0),
        num_outputs: int = 2
    ):
        self.num_outputs = max(2, num_outputs)
        self._current_index = 0
        super().__init__(node_id, position)

    def _setup_pins(self) -> None:
        self.create_execution_input("In")
        for i in range(self.num_outputs):
            self.create_execution_output(f"Then {i}")

    def add_output(self) -> Pin:
        """Add another sequence output."""
        new_index = self.num_outputs
        self.num_outputs += 1
        return self.create_execution_output(f"Then {new_index}")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Sequence",
            category=NodeCategory.FLOW_CONTROL,
            description="Execute multiple outputs in order",
            keywords=["sequence", "order", "multiple"],
            color=(255, 255, 255)
        )

    def execute(self, context: Any) -> Optional[str]:
        if self._current_index < self.num_outputs:
            pin = self.get_output_pin(f"Then {self._current_index}")
            self._current_index += 1
            return pin.id if pin else None
        self._current_index = 0
        return None

    def get_next_output(self) -> Optional[str]:
        """Get the next output in sequence."""
        return self.execute(None)


class ForLoopNode(FlowControlNode):
    """For loop with index."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0)
    ):
        self._current_index = 0
        super().__init__(node_id, position)

    def _setup_pins(self) -> None:
        self.create_execution_input("In")
        self.create_data_input("FirstIndex", IntType, 0)
        self.create_data_input("LastIndex", IntType, 10)
        self.create_execution_output("LoopBody")
        self.create_data_output("Index", IntType, 0)
        self.create_execution_output("Completed")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="For Loop",
            category=NodeCategory.FLOW_CONTROL,
            description="Loop from first to last index",
            keywords=["for", "loop", "iterate", "index"],
            is_latent=True,
            color=(255, 255, 255)
        )

    def execute(self, context: Any) -> Optional[str]:
        first = self.get_input_pin("FirstIndex").get_value()
        last = self.get_input_pin("LastIndex").get_value()

        if self._current_index <= last:
            self.get_output_pin("Index").set_value(self._current_index)
            self._current_index += 1
            return self.get_output_pin("LoopBody").id

        self._current_index = first
        return self.get_output_pin("Completed").id

    def reset(self) -> None:
        """Reset loop to initial state."""
        self._current_index = self.get_input_pin("FirstIndex").get_value()


class WhileLoopNode(FlowControlNode):
    """While loop with condition."""

    def _setup_pins(self) -> None:
        self.create_execution_input("In")
        self.create_data_input("Condition", BoolType, True)
        self.create_execution_output("LoopBody")
        self.create_execution_output("Completed")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="While Loop",
            category=NodeCategory.FLOW_CONTROL,
            description="Loop while condition is true",
            keywords=["while", "loop", "condition"],
            is_latent=True,
            color=(255, 255, 255)
        )

    def execute(self, context: Any) -> Optional[str]:
        condition = self.get_input_pin("Condition").get_value()
        if condition:
            return self.get_output_pin("LoopBody").id
        return self.get_output_pin("Completed").id


class ForEachLoopNode(FlowControlNode):
    """For each loop over array."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0)
    ):
        self._current_index = 0
        super().__init__(node_id, position)

    def _setup_pins(self) -> None:
        self.create_execution_input("In")
        self.create_data_input("Array", WildcardType, [])
        self.create_execution_output("LoopBody")
        self.create_data_output("Element", WildcardType)
        self.create_data_output("Index", IntType, 0)
        self.create_execution_output("Completed")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="For Each Loop",
            category=NodeCategory.FLOW_CONTROL,
            description="Loop over each element in an array",
            keywords=["foreach", "for", "each", "array", "loop"],
            is_latent=True,
            color=(255, 255, 255)
        )

    def execute(self, context: Any) -> Optional[str]:
        array = self.get_input_pin("Array").get_value()
        if hasattr(array, "items"):
            array = array.items

        if self._current_index < len(array):
            self.get_output_pin("Index").set_value(self._current_index)
            self.get_output_pin("Element").set_value(array[self._current_index])
            self._current_index += 1
            return self.get_output_pin("LoopBody").id

        self._current_index = 0
        return self.get_output_pin("Completed").id


class SwitchNode(FlowControlNode):
    """Switch/case on integer or enum."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0),
        cases: Optional[List[int]] = None
    ):
        self.cases = cases or [0, 1, 2]
        super().__init__(node_id, position)

    def _setup_pins(self) -> None:
        self.create_execution_input("In")
        self.create_data_input("Selection", IntType, 0)
        for case_val in self.cases:
            self.create_execution_output(f"Case {case_val}")
        self.create_execution_output("Default")

    def add_case(self, value: int) -> Pin:
        """Add a case output."""
        if value not in self.cases:
            self.cases.append(value)
            return self.create_execution_output(f"Case {value}")
        return self.get_output_pin(f"Case {value}")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Switch on Int",
            category=NodeCategory.FLOW_CONTROL,
            description="Branch based on integer value",
            keywords=["switch", "case", "select"],
            color=(255, 255, 255)
        )

    def execute(self, context: Any) -> Optional[str]:
        selection = self.get_input_pin("Selection").get_value()
        case_pin = self.get_output_pin(f"Case {selection}")
        if case_pin:
            return case_pin.id
        return self.get_output_pin("Default").id


class GateNode(FlowControlNode):
    """Gate that can be opened/closed."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0)
    ):
        self._is_open = True
        super().__init__(node_id, position)

    def _setup_pins(self) -> None:
        self.create_execution_input("Enter")
        self.create_execution_input("Open")
        self.create_execution_input("Close")
        self.create_execution_input("Toggle")
        self.create_data_input("StartClosed", BoolType, False)
        self.create_execution_output("Exit")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Gate",
            category=NodeCategory.FLOW_CONTROL,
            description="Gate that can be opened or closed",
            keywords=["gate", "open", "close", "toggle"],
            color=(255, 255, 255)
        )

    def execute(self, context: Any) -> Optional[str]:
        # This is simplified - actual execution depends on which input was triggered
        if self._is_open:
            return self.get_output_pin("Exit").id
        return None

    def open(self) -> None:
        """Open the gate."""
        self._is_open = True

    def close(self) -> None:
        """Close the gate."""
        self._is_open = False

    def toggle(self) -> None:
        """Toggle the gate state."""
        self._is_open = not self._is_open


class DoOnceNode(FlowControlNode):
    """Execute only once until reset."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0)
    ):
        self._has_executed = False
        super().__init__(node_id, position)

    def _setup_pins(self) -> None:
        self.create_execution_input("In")
        self.create_execution_input("Reset")
        self.create_data_input("StartClosed", BoolType, False)
        self.create_execution_output("Completed")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Do Once",
            category=NodeCategory.FLOW_CONTROL,
            description="Execute only once until reset",
            keywords=["once", "single", "reset"],
            color=(255, 255, 255)
        )

    def execute(self, context: Any) -> Optional[str]:
        if not self._has_executed:
            self._has_executed = True
            return self.get_output_pin("Completed").id
        return None

    def reset(self) -> None:
        """Reset to allow execution again."""
        self._has_executed = False


class FlipFlopNode(FlowControlNode):
    """Alternates between two outputs."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0)
    ):
        self._is_a = True
        super().__init__(node_id, position)

    def _setup_pins(self) -> None:
        self.create_execution_input("In")
        self.create_execution_output("A")
        self.create_execution_output("B")
        self.create_data_output("IsA", BoolType, True)

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Flip Flop",
            category=NodeCategory.FLOW_CONTROL,
            description="Alternates between A and B outputs",
            keywords=["flip", "flop", "toggle", "alternate"],
            color=(255, 255, 255)
        )

    def execute(self, context: Any) -> Optional[str]:
        self.get_output_pin("IsA").set_value(self._is_a)
        if self._is_a:
            self._is_a = False
            return self.get_output_pin("A").id
        else:
            self._is_a = True
            return self.get_output_pin("B").id


# =============================================================================
# FUNCTION NODES
# =============================================================================


class FunctionNode(Node):
    """Base class for function call nodes."""

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Function",
            category=NodeCategory.FUNCTION,
            description="Function call node",
            color=(0, 100, 200)
        )


class PureFunctionNode(FunctionNode):
    """Pure function node (no execution pins, no side effects)."""

    def _setup_pins(self) -> None:
        # Pure functions have no execution pins
        pass

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Pure Function",
            category=NodeCategory.FUNCTION,
            description="Pure function with no side effects",
            is_pure=True,
            is_const=True,
            color=(80, 200, 120)
        )

    def execute(self, context: Any) -> Optional[str]:
        # Pure functions don't continue execution flow
        return None


class CallFunctionNode(FunctionNode):
    """Call a blueprint function."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0),
        function_name: str = "",
        target_class: str = ""
    ):
        self.function_name = function_name
        self.target_class = target_class
        super().__init__(node_id, position)

    def _setup_pins(self) -> None:
        self.create_execution_input("In")
        self.create_data_input("Target", ObjectType)
        self.create_execution_output("Out")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Call Function",
            category=NodeCategory.FUNCTION,
            description="Call a function on an object",
            keywords=["call", "function", "method"],
            color=(0, 100, 200)
        )

    def execute(self, context: Any) -> Optional[str]:
        # Function execution would be handled by the runtime
        return self.get_output_pin("Out").id


# =============================================================================
# VARIABLE NODES
# =============================================================================


class VariableNode(Node):
    """Base class for variable access nodes."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0),
        variable_name: str = "",
        variable_type: Type[BlueprintType] = WildcardType
    ):
        self.variable_name = variable_name
        self.variable_type = variable_type
        super().__init__(node_id, position)

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Variable",
            category=NodeCategory.VARIABLE,
            description="Variable access node",
            color=(0, 160, 0)
        )


class GetVariableNode(VariableNode):
    """Get the value of a variable."""

    def _setup_pins(self) -> None:
        self.create_data_output("Value", self.variable_type)

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Get Variable",
            category=NodeCategory.VARIABLE,
            description="Get the value of a variable",
            keywords=["get", "variable", "read"],
            is_pure=True,
            color=(0, 160, 0)
        )

    def execute(self, context: Any) -> Optional[str]:
        # Pure node, no execution flow
        if hasattr(context, "get_variable"):
            value = context.get_variable(self.variable_name)
            self.get_output_pin("Value").set_value(value)
        return None


class SetVariableNode(VariableNode):
    """Set the value of a variable."""

    def _setup_pins(self) -> None:
        self.create_execution_input("In")
        self.create_data_input("Value", self.variable_type)
        self.create_execution_output("Out")
        self.create_data_output("NewValue", self.variable_type)

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Set Variable",
            category=NodeCategory.VARIABLE,
            description="Set the value of a variable",
            keywords=["set", "variable", "write", "assign"],
            color=(0, 160, 0)
        )

    def execute(self, context: Any) -> Optional[str]:
        value = self.get_input_pin("Value").get_value()
        if hasattr(context, "set_variable"):
            context.set_variable(self.variable_name, value)
        self.get_output_pin("NewValue").set_value(value)
        return self.get_output_pin("Out").id


# =============================================================================
# MACRO NODES
# =============================================================================


class MacroNode(Node):
    """Reference to a reusable subgraph (macro)."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0),
        macro_name: str = "",
        macro_id: str = ""
    ):
        self.macro_name = macro_name
        self.macro_id = macro_id
        self.input_definitions: List[Tuple[str, Type[BlueprintType], bool]] = []
        self.output_definitions: List[Tuple[str, Type[BlueprintType], bool]] = []
        super().__init__(node_id, position)

    def _setup_pins(self) -> None:
        # Pins are set up based on macro definition
        pass

    def set_interface(
        self,
        inputs: List[Tuple[str, Type[BlueprintType], bool]],
        outputs: List[Tuple[str, Type[BlueprintType], bool]]
    ) -> None:
        """Set the macro interface (inputs and outputs).

        Each tuple is (name, type, is_execution).
        """
        self.input_definitions = inputs
        self.output_definitions = outputs

        for name, dtype, is_exec in inputs:
            if is_exec:
                self.create_execution_input(name)
            else:
                self.create_data_input(name, dtype)

        for name, dtype, is_exec in outputs:
            if is_exec:
                self.create_execution_output(name)
            else:
                self.create_data_output(name, dtype)

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Macro",
            category=NodeCategory.MACRO,
            description="Reusable subgraph",
            keywords=["macro", "subgraph", "reuse"],
            color=(100, 100, 200)
        )

    def execute(self, context: Any) -> Optional[str]:
        # Macro execution is handled by runtime expanding the subgraph
        # Return first execution output for now
        for pin in self.output_pins.values():
            if pin.kind == PinKind.EXECUTION:
                return pin.id
        return None


class MacroInputNode(Node):
    """Macro input tunnel node."""

    def _setup_pins(self) -> None:
        # Configured when macro is defined
        pass

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Macro Inputs",
            category=NodeCategory.MACRO,
            description="Input tunnel for macro",
            color=(100, 100, 200)
        )

    def execute(self, context: Any) -> Optional[str]:
        return None


class MacroOutputNode(Node):
    """Macro output tunnel node."""

    def _setup_pins(self) -> None:
        # Configured when macro is defined
        pass

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Macro Outputs",
            category=NodeCategory.MACRO,
            description="Output tunnel for macro",
            color=(100, 100, 200)
        )

    def execute(self, context: Any) -> Optional[str]:
        return None


# =============================================================================
# LITERAL/CONSTANT NODES
# =============================================================================


class LiteralNode(PureFunctionNode):
    """Base class for literal/constant value nodes."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        position: Tuple[float, float] = (0, 0),
        value: Any = None
    ):
        self._value = value
        super().__init__(node_id, position)

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, v: Any) -> None:
        self._value = v
        out_pin = self.get_output_pin("Value")
        if out_pin:
            out_pin.set_value(v)


class BoolLiteralNode(LiteralNode):
    """Boolean constant node."""

    def _setup_pins(self) -> None:
        self.create_data_output("Value", BoolType, self._value or False)

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Bool Literal",
            category=NodeCategory.UTILITY,
            description="Boolean constant value",
            is_pure=True,
            color=(139, 0, 0)
        )


class IntLiteralNode(LiteralNode):
    """Integer constant node."""

    def _setup_pins(self) -> None:
        self.create_data_output("Value", IntType, self._value or 0)

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Int Literal",
            category=NodeCategory.UTILITY,
            description="Integer constant value",
            is_pure=True,
            color=(0, 230, 180)
        )


class FloatLiteralNode(LiteralNode):
    """Float constant node."""

    def _setup_pins(self) -> None:
        self.create_data_output("Value", FloatType, self._value or 0.0)

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Float Literal",
            category=NodeCategory.UTILITY,
            description="Float constant value",
            is_pure=True,
            color=(0, 200, 80)
        )


class StringLiteralNode(LiteralNode):
    """String constant node."""

    def _setup_pins(self) -> None:
        self.create_data_output("Value", StringType, self._value or "")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="String Literal",
            category=NodeCategory.UTILITY,
            description="String constant value",
            is_pure=True,
            color=(255, 0, 200)
        )


class VectorLiteralNode(LiteralNode):
    """Vector constant node."""

    def _setup_pins(self) -> None:
        from .data_types import Vector3
        self.create_data_output("Value", Vector3Type, self._value or Vector3())

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Vector Literal",
            category=NodeCategory.UTILITY,
            description="Vector constant value",
            is_pure=True,
            color=(255, 200, 0)
        )


# =============================================================================
# UTILITY NODES
# =============================================================================


class PrintStringNode(Node):
    """Debug print node."""

    def _setup_pins(self) -> None:
        self.create_execution_input("In")
        self.create_data_input("String", StringType, "Hello")
        self.create_data_input("PrintToScreen", BoolType, True)
        self.create_data_input("PrintToLog", BoolType, True)
        self.create_data_input("Duration", FloatType, 2.0)
        self.create_execution_output("Out")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Print String",
            category=NodeCategory.DEBUG,
            description="Print a string for debugging",
            keywords=["print", "debug", "log", "display"],
            color=(255, 200, 0)
        )

    def execute(self, context: Any) -> Optional[str]:
        text = self.get_input_pin("String").get_value()
        to_screen = self.get_input_pin("PrintToScreen").get_value()
        to_log = self.get_input_pin("PrintToLog").get_value()

        if hasattr(context, "print_string"):
            context.print_string(text, to_screen, to_log)

        return self.get_output_pin("Out").id


class DelayNode(Node):
    """Delay execution for a duration."""

    def _setup_pins(self) -> None:
        self.create_execution_input("In")
        self.create_data_input("Duration", FloatType, 1.0)
        self.create_execution_output("Completed")

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Delay",
            category=NodeCategory.FLOW_CONTROL,
            description="Delay execution for specified duration",
            keywords=["delay", "wait", "timer", "sleep"],
            is_latent=True,
            color=(255, 128, 0)
        )

    def execute(self, context: Any) -> Optional[str]:
        # Delay is handled by runtime
        return self.get_output_pin("Completed").id


# =============================================================================
# NODE REGISTRY
# =============================================================================


NODE_REGISTRY: Dict[str, Type[Node]] = {}


def register_node(node_class: Type[Node]) -> Type[Node]:
    """Register a node type."""
    metadata = node_class.get_metadata()
    NODE_REGISTRY[metadata.display_name] = node_class
    return node_class


def get_node_class(name: str) -> Optional[Type[Node]]:
    """Get a node class by its display name."""
    return NODE_REGISTRY.get(name)


def get_nodes_by_category(category: NodeCategory) -> List[Type[Node]]:
    """Get all registered nodes in a category."""
    return [
        cls for cls in NODE_REGISTRY.values()
        if cls.get_metadata().category == category
    ]


def search_nodes(query: str) -> List[Type[Node]]:
    """Search nodes by name or keywords."""
    query = query.lower()
    results = []
    for cls in NODE_REGISTRY.values():
        meta = cls.get_metadata()
        if query in meta.display_name.lower():
            results.append(cls)
        elif any(query in kw.lower() for kw in meta.keywords):
            results.append(cls)
    return results


# Register built-in nodes
for node_cls in [
    BeginPlayNode, TickNode, InputActionNode, CustomEventNode,
    BranchNode, SequenceNode, ForLoopNode, WhileLoopNode, ForEachLoopNode,
    SwitchNode, GateNode, DoOnceNode, FlipFlopNode,
    CallFunctionNode, GetVariableNode, SetVariableNode,
    MacroNode, MacroInputNode, MacroOutputNode,
    BoolLiteralNode, IntLiteralNode, FloatLiteralNode, StringLiteralNode, VectorLiteralNode,
    PrintStringNode, DelayNode
]:
    register_node(node_cls)
