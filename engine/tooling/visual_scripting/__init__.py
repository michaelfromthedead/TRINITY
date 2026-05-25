"""
FlowForge Visual Scripting System

A node-based visual scripting system for the AI Game Engine.

This module provides:
- Graph-based blueprint editing
- Node library with search and categories
- Blueprint execution runtime
- Debugging with breakpoints and stepping
- Bytecode compilation
- Serialization with versioning

Usage:
    from engine.tooling.visual_scripting import (
        BlueprintGraph,
        GraphEditor,
        BlueprintRuntime,
        BlueprintDebugger,
        compile_blueprint,
        save_blueprint,
        load_blueprint,
    )

    # Create a new blueprint
    graph = BlueprintGraph(name="MyBlueprint")

    # Add nodes
    from engine.tooling.visual_scripting.node_types import BeginPlayNode, PrintStringNode
    begin = BeginPlayNode(position=(0, 0))
    graph.add_node(begin)

    # Execute
    runtime = get_runtime()
    vm = runtime.register_blueprint(graph)
    runtime.begin_play(graph.id)
"""

# Data types
from .data_types import (
    BlueprintType,
    DataTypeCategory,
    TypeColor,
    WireColors,
    BoolType,
    IntType,
    FloatType,
    StringType,
    Vector2,
    Vector3,
    Vector2Type,
    Vector3Type,
    Rotator,
    RotatorType,
    Transform,
    TransformType,
    ObjectRef,
    ObjectType,
    ArrayValue,
    ArrayType,
    MapValue,
    MapType,
    SetValue,
    SetType,
    ExecutionType,
    WildcardType,
    TYPE_REGISTRY,
    get_type_by_name,
    register_type,
    can_connect_types,
    convert_value,
)

# Node types
from .node_types import (
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

# Execution context
from .execution_context import (
    VariableScope,
    Variable,
    StackFrame,
    ExecutionState,
    ExecutionError,
    ExecutionContext,
    ExecutionContextPool,
    get_context_pool,
    acquire_context,
    release_context,
)

# Graph editor
from .graph_editor import (
    SelectionMode,
    Connection,
    ViewState,
    EditorAction,
    UndoStack,
    ClipboardData,
    BlueprintGraph,
    GraphEditor,
)

# Node library
from .node_library import (
    CategoryInfo,
    CATEGORY_INFO,
    NodeEntry,
    SearchResult,
    NodeLibrary,
    get_node_library,
    register_custom_node,
    search_library,
)

# Runtime
from .blueprint_runtime import (
    VMInstruction,
    VMOp,
    LatentAction,
    ExecutionStats,
    EventDispatcher,
    BlueprintVM,
    BlueprintRuntime,
    get_runtime,
    execute_blueprint,
)

# Debugging
from .blueprint_debug import (
    BreakpointType,
    StepMode,
    Breakpoint,
    WatchExpression,
    ExecutionHistoryEntry,
    NodeProfile,
    DebugState,
    BlueprintDebugger,
)

# Compiler
from .blueprint_compiler import (
    OpCode,
    BytecodeInstruction,
    ConstantPool,
    CompiledFunction,
    CompiledBlueprint,
    OptimizationLevel,
    CompilerError,
    CompilationResult,
    BlueprintCompiler,
    BytecodeSerializer,
    compile_blueprint,
)

# Serializer
from .blueprint_serializer import (
    SerializationFormat,
    SerializationOptions,
    BlueprintHeader,
    BlueprintSerializer,
    IncrementalSerializer,
    save_blueprint,
    load_blueprint,
    export_blueprint_json,
    import_blueprint_json,
)


__all__ = [
    # Data types
    "BlueprintType",
    "DataTypeCategory",
    "TypeColor",
    "WireColors",
    "BoolType",
    "IntType",
    "FloatType",
    "StringType",
    "Vector2",
    "Vector3",
    "Vector2Type",
    "Vector3Type",
    "Rotator",
    "RotatorType",
    "Transform",
    "TransformType",
    "ObjectRef",
    "ObjectType",
    "ArrayValue",
    "ArrayType",
    "MapValue",
    "MapType",
    "SetValue",
    "SetType",
    "ExecutionType",
    "WildcardType",
    "TYPE_REGISTRY",
    "get_type_by_name",
    "register_type",
    "can_connect_types",
    "convert_value",
    # Node types
    "Pin",
    "PinDirection",
    "PinKind",
    "Node",
    "NodeCategory",
    "NodeMetadata",
    "EventNode",
    "BeginPlayNode",
    "TickNode",
    "InputActionNode",
    "CustomEventNode",
    "FlowControlNode",
    "BranchNode",
    "SequenceNode",
    "ForLoopNode",
    "WhileLoopNode",
    "ForEachLoopNode",
    "SwitchNode",
    "GateNode",
    "DoOnceNode",
    "FlipFlopNode",
    "FunctionNode",
    "PureFunctionNode",
    "CallFunctionNode",
    "VariableNode",
    "GetVariableNode",
    "SetVariableNode",
    "MacroNode",
    "MacroInputNode",
    "MacroOutputNode",
    "LiteralNode",
    "BoolLiteralNode",
    "IntLiteralNode",
    "FloatLiteralNode",
    "StringLiteralNode",
    "VectorLiteralNode",
    "PrintStringNode",
    "DelayNode",
    "NODE_REGISTRY",
    "register_node",
    "get_node_class",
    "get_nodes_by_category",
    "search_nodes",
    # Execution context
    "VariableScope",
    "Variable",
    "StackFrame",
    "ExecutionState",
    "ExecutionError",
    "ExecutionContext",
    "ExecutionContextPool",
    "get_context_pool",
    "acquire_context",
    "release_context",
    # Graph editor
    "SelectionMode",
    "Connection",
    "ViewState",
    "EditorAction",
    "UndoStack",
    "ClipboardData",
    "BlueprintGraph",
    "GraphEditor",
    # Node library
    "CategoryInfo",
    "CATEGORY_INFO",
    "NodeEntry",
    "SearchResult",
    "NodeLibrary",
    "get_node_library",
    "register_custom_node",
    "search_library",
    # Runtime
    "VMInstruction",
    "VMOp",
    "LatentAction",
    "ExecutionStats",
    "EventDispatcher",
    "BlueprintVM",
    "BlueprintRuntime",
    "get_runtime",
    "execute_blueprint",
    # Debugging
    "BreakpointType",
    "StepMode",
    "Breakpoint",
    "WatchExpression",
    "ExecutionHistoryEntry",
    "NodeProfile",
    "DebugState",
    "BlueprintDebugger",
    # Compiler
    "OpCode",
    "BytecodeInstruction",
    "ConstantPool",
    "CompiledFunction",
    "CompiledBlueprint",
    "OptimizationLevel",
    "CompilerError",
    "CompilationResult",
    "BlueprintCompiler",
    "BytecodeSerializer",
    "compile_blueprint",
    # Serializer
    "SerializationFormat",
    "SerializationOptions",
    "BlueprintHeader",
    "BlueprintSerializer",
    "IncrementalSerializer",
    "save_blueprint",
    "load_blueprint",
    "export_blueprint_json",
    "import_blueprint_json",
]
