"""AST Parser module for FlowForge Backend.

This module provides Python AST parsing capabilities for Trinity ECS code:
- Parsing Python code into AST representation
- Extracting classes, methods, and fields
- Special handling for Trinity ECS patterns (System, Query)
- Building node graphs with edges representing relationships

Example:
    from flowforge_backend.ast_parser import TrinityASTVisitor

    visitor = TrinityASTVisitor()
    module_def = visitor.parse('''
    from trinity.ecs import System

    class MovementSystem(System):
        speed: float = 1.0

        def update(self, dt: float) -> None:
            pass
    ''')

    for cls in module_def.classes:
        print(f"Class: {cls.name}")
        for method in cls.methods:
            print(f"  Method: {method.name}")

    # Edge building example:
    from flowforge_backend.ast_parser import EdgeBuilder, GraphNode, NodeType

    nodes = [GraphNode(id="1", type=NodeType.SYSTEM, name="MovementSystem")]
    node_id_map = {"MovementSystem": "1"}
    builder = EdgeBuilder(nodes, node_id_map)
    edges = builder.build_all_edges()
"""

from .types import (
    # General AST types
    ClassDef,
    FieldDef,
    ImportInfo,
    MethodDef,
    MethodKind,
    ModuleDef,
    ParameterDef,
    QueryComponentInfo,
    TrinityDecoratorType,
    TrinityImportInfo,
    # Trinity ECS-specific types
    ComponentDef,
    SystemDef,
    ResourceDef,
    EventDef,
    ImportDef,
    ParseResult,
    DecoratorArgs,
    TrinityDef,
)

from .graph_types import (
    # Graph node/edge types
    NodeType,
    EdgeType,
    SourceLocation,
    NodePosition,
    GraphNode,
    GraphEdge,
    NodeGraph,
    # Node data types
    FieldData,
    ParameterData,
    MethodData,
    ComponentData,
    SystemData,
    ResourceData,
    EventData,
    NodeData,
    # Convenience constructors
    create_component_node,
    create_system_node,
    create_resource_node,
    create_event_node,
    create_edge,
)

from .trinity_nodes import (
    # Trinity-aware graph types
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
    # Trinity base
    _TrinityGraphBase,
    # Graph type converters
    to_trinity_graph,
    # Convenience constructors
    create_trinity_component_node,
    create_trinity_system_node,
    create_trinity_resource_node,
    create_trinity_event_node,
    create_trinity_edge,
    # Registration
    register_all_trinity_graph_types,
)

from .edge_builder import EdgeBuilder

from .graph_builder import (
    GraphBuilder,
    build_graph_from_parse_result,
)

from .visitor import (
    ImportTracker,
    TrinityASTVisitor,
    parse_source,
    parse_file,
)

from .layout import (
    LayoutEngine,
    apply_layout,
    get_layout_info,
)

# Centralized constants
from .constants import (
    # Trinity detection constants
    TRINITY_MODULE,
    TRINITY_DECORATOR_NAMES,
    COMPONENT_BASE_CLASSES,
    SYSTEM_BASE_CLASSES,
    RESOURCE_BASE_CLASSES,
    EVENT_BASE_CLASSES,
    # Fallback strings
    DEFAULT_SOURCE_NAME,
    UNKNOWN_ANNOTATION,
    UNKNOWN_DEFAULT,
    UNKNOWN_DECORATOR,
    # Layout constants
    NODE_WIDTH,
    NODE_HEIGHT,
    HORIZONTAL_SPACING,
    VERTICAL_SPACING,
    COLUMN_SPACING,
    DEFAULT_START_X,
    DEFAULT_START_Y,
    # Graph building constants
    NODE_ID_HASH_LENGTH,
    NODE_ID_PREFIX,
    BUILTIN_TYPES,
    DEFAULT_EXCLUDE_PATTERNS,
)

from .cache import (
    ASTCache,
    get_default_cache,
)

from .graph import (
    build_node_graph,
    incremental_parse_directory,
    parse_and_build_graph,
)

from .incremental import (
    IncrementalParser,
    detect_changed_files,
    reparse_changed,
)

from .project import (
    ProjectParser,
    parse_project,
    parse_files,
)

__all__ = [
    # General AST type definitions
    "ClassDef",
    "FieldDef",
    "ImportInfo",
    "MethodDef",
    "MethodKind",
    "ModuleDef",
    "ParameterDef",
    "QueryComponentInfo",
    "TrinityDecoratorType",
    "TrinityImportInfo",
    # Trinity ECS-specific types
    "ComponentDef",
    "SystemDef",
    "ResourceDef",
    "EventDef",
    "ImportDef",
    "ParseResult",
    "DecoratorArgs",
    "TrinityDef",
    # Graph types
    "NodeType",
    "EdgeType",
    "SourceLocation",
    "NodePosition",
    "GraphNode",
    "GraphEdge",
    "NodeGraph",
    # Node data types
    "FieldData",
    "ParameterData",
    "MethodData",
    "ComponentData",
    "SystemData",
    "ResourceData",
    "EventData",
    "NodeData",
    # Convenience constructors
    "create_component_node",
    "create_system_node",
    "create_resource_node",
    "create_event_node",
    "create_edge",
    # Edge builder
    "EdgeBuilder",
    # Graph builder
    "GraphBuilder",
    "build_graph_from_parse_result",
    # Import tracking
    "ImportTracker",
    # Visitor class
    "TrinityASTVisitor",
    # Convenience functions
    "parse_source",
    "parse_file",
    # Trinity detection constants
    "TRINITY_MODULE",
    "TRINITY_DECORATOR_NAMES",
    "COMPONENT_BASE_CLASSES",
    "SYSTEM_BASE_CLASSES",
    "RESOURCE_BASE_CLASSES",
    "EVENT_BASE_CLASSES",
    # Fallback string constants
    "DEFAULT_SOURCE_NAME",
    "UNKNOWN_ANNOTATION",
    "UNKNOWN_DEFAULT",
    "UNKNOWN_DECORATOR",
    # Layout engine and functions
    "LayoutEngine",
    "apply_layout",
    "get_layout_info",
    # Layout constants
    "NODE_WIDTH",
    "NODE_HEIGHT",
    "HORIZONTAL_SPACING",
    "VERTICAL_SPACING",
    "COLUMN_SPACING",
    "DEFAULT_START_X",
    "DEFAULT_START_Y",
    # Graph building constants
    "NODE_ID_HASH_LENGTH",
    "NODE_ID_PREFIX",
    "BUILTIN_TYPES",
    "DEFAULT_EXCLUDE_PATTERNS",
    # AST cache
    "ASTCache",
    "get_default_cache",
    # High-level graph building API
    "build_node_graph",
    "incremental_parse_directory",
    "parse_and_build_graph",
    # Incremental parsing
    "IncrementalParser",
    "detect_changed_files",
    "reparse_changed",
    # Project parsing API
    "ProjectParser",
    "parse_project",
    "parse_files",
    # Trinity-aware graph types
    "TrinityNodePosition",
    "TrinitySourceLocation",
    "TrinityFieldData",
    "TrinityParameterData",
    "TrinityMethodData",
    "TrinityComponentData",
    "TrinitySystemData",
    "TrinityResourceData",
    "TrinityEventData",
    "TrinityGraphNode",
    "TrinityGraphEdge",
    "TrinityNodeGraph",
    "_TrinityGraphBase",
    # Graph type converters
    "to_trinity_graph",
    # Convenience constructors
    "create_trinity_component_node",
    "create_trinity_system_node",
    "create_trinity_resource_node",
    "create_trinity_event_node",
    "create_trinity_edge",
    # Registration
    "register_all_trinity_graph_types",
]
