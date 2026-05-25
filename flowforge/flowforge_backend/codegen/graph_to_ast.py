"""Graph to AST converter for code generation.

This module converts a node graph (from the visual editor) back to
Python AST that can be unparsed to source code.

It handles:
- Component nodes -> @component decorated classes
- System nodes -> @system decorated classes
- Resource nodes -> @resource decorated classes
- Event nodes -> @event decorated classes
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Optional

from ..ast_parser.graph_types import (
    GraphNode,
    NodeGraph,
    ComponentData,
    SystemData,
    ResourceData,
    EventData,
    FieldData,
    MethodData,
    ParameterData,
)
from ..ast_parser.constants import TRINITY_MODULE
from .imports import generate_imports


def _parse_type_annotation(type_str: str) -> ast.expr:
    """Parse a type annotation string into an AST expression.

    Args:
        type_str: The type annotation as a string (e.g., "int", "list[str]").

    Returns:
        AST expression node representing the type.

    Raises:
        SyntaxError: If the type annotation cannot be parsed.
    """
    if not type_str or type_str == "None":
        return ast.Constant(value=None)

    try:
        # Parse as a standalone expression
        module = ast.parse(type_str, mode="eval")
        return module.body
    except SyntaxError:
        # Fallback to simple Name if parsing fails
        return ast.Name(id=type_str, ctx=ast.Load())


def _parse_default_value(default_str: str) -> ast.expr:
    """Parse a default value string into an AST expression.

    Args:
        default_str: The default value as a string (e.g., "0", "[]", "None").

    Returns:
        AST expression node representing the default value.

    Raises:
        SyntaxError: If the default value cannot be parsed.
    """
    if not default_str:
        return ast.Constant(value=None)

    # Handle common literal values
    if default_str == "None":
        return ast.Constant(value=None)
    elif default_str == "True":
        return ast.Constant(value=True)
    elif default_str == "False":
        return ast.Constant(value=False)

    try:
        # Parse as a standalone expression
        module = ast.parse(default_str, mode="eval")
        return module.body
    except SyntaxError:
        # If parsing fails, treat as string constant
        return ast.Constant(value=default_str)


def _create_decorator(name: str, args: Optional[dict[str, Any]] = None) -> ast.expr:
    """Create a decorator AST node.

    Args:
        name: The decorator name (e.g., "component").
        args: Optional dictionary of keyword arguments.

    Returns:
        AST expression for the decorator.
    """
    if args and (args.get("keyword") or args.get("positional")):
        # Decorator with arguments: @component(...)
        keywords = []
        positional = []

        # Handle keyword arguments
        for key, value in args.get("keyword", {}).items():
            keywords.append(ast.keyword(
                arg=key,
                value=ast.Constant(value=value),
            ))

        # Handle positional arguments
        for value in args.get("positional", []):
            positional.append(ast.Constant(value=value))

        return ast.Call(
            func=ast.Name(id=name, ctx=ast.Load()),
            args=positional,
            keywords=keywords,
        )
    else:
        # Simple decorator: @component
        return ast.Name(id=name, ctx=ast.Load())


def _create_field_assignment(
    field_name: str,
    type_annotation: str,
    default_value: Optional[str] = None,
) -> ast.AnnAssign:
    """Create an annotated assignment for a class field.

    Args:
        field_name: The field name.
        type_annotation: The type annotation string.
        default_value: Optional default value string.

    Returns:
        AST node for annotated assignment (e.g., x: int = 0).
    """
    ann_assign = ast.AnnAssign(
        target=ast.Name(id=field_name, ctx=ast.Store()),
        annotation=_parse_type_annotation(type_annotation),
        simple=1,  # Simple assignment (no parentheses around target)
    )

    if default_value is not None:
        ann_assign.value = _parse_default_value(default_value)

    return ann_assign


def _create_method_def(
    method_name: str,
    parameters: list[dict[str, Any]],
    return_type: Optional[str] = None,
    docstring: Optional[str] = None,
    decorators: Optional[list[str]] = None,
    body: Optional[list[ast.stmt]] = None,
) -> ast.FunctionDef:
    """Create a function definition for a class method.

    Args:
        method_name: The method name.
        parameters: List of parameter dictionaries with name, type_annotation, default_value.
        return_type: Optional return type annotation.
        docstring: Optional method docstring.
        decorators: Optional list of decorator names.
        body: Optional list of body statements.

    Returns:
        AST FunctionDef node.
    """
    # Build arguments
    args_list: list[ast.arg] = []
    defaults: list[ast.expr] = []

    # Always add 'self' as first parameter for instance methods
    args_list.append(ast.arg(arg="self", annotation=None))

    for param in parameters:
        param_name = param.get("name", "")
        type_ann = param.get("type_annotation") or param.get("type")
        default_val = param.get("default_value") or param.get("default")

        # Skip 'self' if it's in the parameters
        if param_name == "self":
            continue

        arg = ast.arg(
            arg=param_name,
            annotation=_parse_type_annotation(type_ann) if type_ann else None,
        )
        args_list.append(arg)

        if default_val is not None:
            defaults.append(_parse_default_value(default_val))

    arguments = ast.arguments(
        posonlyargs=[],
        args=args_list,
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=defaults,
    )

    # Build body
    method_body: list[ast.stmt] = []

    if docstring:
        method_body.append(ast.Expr(value=ast.Constant(value=docstring)))

    if body:
        method_body.extend(body)

    # If body is empty, add pass
    if not method_body:
        method_body.append(ast.Pass())

    # Build decorators
    decorator_list: list[ast.expr] = []
    if decorators:
        for dec_name in decorators:
            decorator_list.append(ast.Name(id=dec_name, ctx=ast.Load()))

    # Build return annotation
    returns = _parse_type_annotation(return_type) if return_type else None

    return ast.FunctionDef(
        name=method_name,
        args=arguments,
        body=method_body,
        decorator_list=decorator_list,
        returns=returns,
        type_comment=None,
    )


@dataclass
class ASTBuilder:
    """Builds Python AST from node graph.

    This class converts GraphNode objects to AST class definitions
    with appropriate decorators, fields, and methods.

    Attributes:
        _statements: List of accumulated AST statements.
        _node_order: Order of nodes by type for consistent output.
    """
    _statements: list[ast.stmt] = field(default_factory=list)
    _node_order: tuple[str, ...] = ("component", "resource", "event", "system")

    def _node_data_to_dict(self, node: GraphNode) -> dict[str, Any]:
        """Convert node data to dictionary format.

        Args:
            node: The graph node.

        Returns:
            Node data as a dictionary.
        """
        if isinstance(node.data, dict):
            return node.data
        return node.data.to_dict()

    def _build_component_class(self, node: GraphNode) -> ast.ClassDef:
        """Build a class definition for a component node.

        Args:
            node: The component graph node.

        Returns:
            AST ClassDef for the component.
        """
        data = self._node_data_to_dict(node)

        # Build class body
        body: list[ast.stmt] = []

        # Add docstring if present
        docstring = data.get("docstring")
        if docstring:
            body.append(ast.Expr(value=ast.Constant(value=docstring)))

        # Add fields
        fields = data.get("fields", [])
        for field_data in fields:
            name = field_data.get("name")
            type_ann = field_data.get("type_annotation") or field_data.get("type", "Any")
            default = field_data.get("default_value") or field_data.get("default")

            body.append(_create_field_assignment(name, type_ann, default))

        # If no body, add pass
        if not body:
            body.append(ast.Pass())

        # Build decorator
        decorator_args = data.get("decorator_args")
        decorator = _create_decorator("component", decorator_args)

        # Build bases
        bases: list[ast.expr] = []
        base_names = data.get("bases", [])
        for base_name in base_names:
            bases.append(ast.Name(id=base_name, ctx=ast.Load()))

        return ast.ClassDef(
            name=node.name,
            bases=bases,
            keywords=[],
            body=body,
            decorator_list=[decorator],
        )

    def _build_system_class(self, node: GraphNode) -> ast.ClassDef:
        """Build a class definition for a system node.

        Args:
            node: The system graph node.

        Returns:
            AST ClassDef for the system.
        """
        data = self._node_data_to_dict(node)

        # Build class body
        body: list[ast.stmt] = []

        # Add docstring if present
        docstring = data.get("docstring")
        if docstring:
            body.append(ast.Expr(value=ast.Constant(value=docstring)))

        # Add fields (if any)
        fields = data.get("fields", [])
        for field_data in fields:
            name = field_data.get("name")
            type_ann = field_data.get("type_annotation") or field_data.get("type", "Any")
            default = field_data.get("default_value") or field_data.get("default")

            body.append(_create_field_assignment(name, type_ann, default))

        # Add methods
        methods = data.get("methods", [])
        for method_data in methods:
            method_name = method_data.get("name")
            params = method_data.get("parameters", [])
            return_type = method_data.get("return_type")
            method_docstring = method_data.get("docstring")
            decorators = method_data.get("decorators", [])

            # Build Query[...] parameter if query_components or query_types present
            query_types = method_data.get("query_components") or method_data.get("query_types", [])
            if query_types:
                # Add Query parameter for entities with these components
                query_param = {
                    "name": "entities",
                    "type_annotation": f"Query[{', '.join(query_types)}]",
                }
                # Insert after self
                params = [query_param] + params

            method_def = _create_method_def(
                method_name=method_name,
                parameters=params,
                return_type=return_type,
                docstring=method_docstring,
                decorators=decorators if decorators else None,
            )
            body.append(method_def)

        # If no body, add pass
        if not body:
            body.append(ast.Pass())

        # Build decorator
        decorator_args = data.get("decorator_args")
        decorator = _create_decorator("system", decorator_args)

        # Build bases
        bases: list[ast.expr] = []
        base_names = data.get("bases", [])
        for base_name in base_names:
            bases.append(ast.Name(id=base_name, ctx=ast.Load()))

        return ast.ClassDef(
            name=node.name,
            bases=bases,
            keywords=[],
            body=body,
            decorator_list=[decorator],
        )

    def _build_resource_class(self, node: GraphNode) -> ast.ClassDef:
        """Build a class definition for a resource node.

        Args:
            node: The resource graph node.

        Returns:
            AST ClassDef for the resource.
        """
        data = self._node_data_to_dict(node)

        # Build class body
        body: list[ast.stmt] = []

        # Add docstring if present
        docstring = data.get("docstring")
        if docstring:
            body.append(ast.Expr(value=ast.Constant(value=docstring)))

        # Add fields
        fields = data.get("fields", [])
        for field_data in fields:
            name = field_data.get("name")
            type_ann = field_data.get("type_annotation") or field_data.get("type", "Any")
            default = field_data.get("default_value") or field_data.get("default")

            body.append(_create_field_assignment(name, type_ann, default))

        # If no body, add pass
        if not body:
            body.append(ast.Pass())

        # Build decorator
        decorator_args = data.get("decorator_args")
        decorator = _create_decorator("resource", decorator_args)

        # Build bases
        bases: list[ast.expr] = []
        base_names = data.get("bases", [])
        for base_name in base_names:
            bases.append(ast.Name(id=base_name, ctx=ast.Load()))

        return ast.ClassDef(
            name=node.name,
            bases=bases,
            keywords=[],
            body=body,
            decorator_list=[decorator],
        )

    def _build_event_class(self, node: GraphNode) -> ast.ClassDef:
        """Build a class definition for an event node.

        Args:
            node: The event graph node.

        Returns:
            AST ClassDef for the event.
        """
        # For events, we need to handle both parsed EventData and raw dict
        # Raw dicts from JSON may have "payload_fields" instead of "fields"
        if isinstance(node.data, dict):
            data = node.data
        else:
            data = node.data.to_dict()

        # Build class body
        body: list[ast.stmt] = []

        # Add docstring if present
        docstring = data.get("docstring")
        if docstring:
            body.append(ast.Expr(value=ast.Constant(value=docstring)))

        # Events may use either "payload_fields" or "fields"
        # Check payload_fields first (frontend sends this), then fall back to fields
        fields = data.get("payload_fields") or data.get("fields", [])
        for field_data in fields:
            name = field_data.get("name")
            type_ann = field_data.get("type_annotation") or field_data.get("type", "Any")
            default = field_data.get("default_value") or field_data.get("default")

            body.append(_create_field_assignment(name, type_ann, default))

        # If no body, add pass
        if not body:
            body.append(ast.Pass())

        # Build decorator
        decorator_args = data.get("decorator_args")
        decorator = _create_decorator("event", decorator_args)

        # Build bases
        bases: list[ast.expr] = []
        base_names = data.get("bases", [])
        for base_name in base_names:
            bases.append(ast.Name(id=base_name, ctx=ast.Load()))

        return ast.ClassDef(
            name=node.name,
            bases=bases,
            keywords=[],
            body=body,
            decorator_list=[decorator],
        )

    def add_node(self, node: GraphNode) -> None:
        """Add a graph node to the AST.

        Args:
            node: The graph node to convert and add.
        """
        if node.type == "component":
            class_def = self._build_component_class(node)
        elif node.type == "system":
            class_def = self._build_system_class(node)
        elif node.type == "resource":
            class_def = self._build_resource_class(node)
        elif node.type == "event":
            class_def = self._build_event_class(node)
        else:
            # Unknown node type, skip
            return

        self._statements.append(class_def)

    def build(self) -> ast.Module:
        """Build the complete AST module.

        Returns:
            AST Module with all accumulated statements.
        """
        return ast.Module(body=self._statements, type_ignores=[])


def graph_to_ast(graph: dict | NodeGraph) -> ast.Module:
    """Convert a node graph to Python AST.

    This is the main entry point for converting a visual node graph
    representation back to Python AST that can be unparsed to source code.

    Args:
        graph: Node graph with nodes and edges. Can be either a
            dictionary (from JSON) or a NodeGraph instance.

    Returns:
        ast.Module ready for unparsing to source code.

    Example:
        >>> graph = {"nodes": [...], "edges": [...]}
        >>> module = graph_to_ast(graph)
        >>> import ast
        >>> source = ast.unparse(module)
    """
    # Convert dict to NodeGraph if needed
    if isinstance(graph, dict):
        graph = NodeGraph.from_dict(graph)

    # Generate import statements
    import_lines = generate_imports(graph)

    # Parse imports into AST
    import_source = "\n".join(import_lines)
    try:
        import_module = ast.parse(import_source)
        import_stmts = import_module.body
    except SyntaxError:
        import_stmts = []

    # Build class definitions
    builder = ASTBuilder()

    # Sort nodes by type for consistent output order
    type_order = {"component": 0, "resource": 1, "event": 2, "system": 3}
    sorted_nodes = sorted(
        graph.nodes,
        key=lambda n: (type_order.get(n.type, 99), n.name)
    )

    for node in sorted_nodes:
        builder.add_node(node)

    # Combine imports and class definitions
    module = builder.build()
    module.body = import_stmts + module.body

    # Fix missing line numbers and column offsets
    ast.fix_missing_locations(module)

    return module


def nodes_to_ast(nodes: list[GraphNode]) -> ast.Module:
    """Convert a list of nodes to Python AST.

    Convenience function for converting nodes without edges.

    Args:
        nodes: List of graph nodes.

    Returns:
        ast.Module ready for unparsing.
    """
    graph = NodeGraph(nodes=nodes, edges=[])
    return graph_to_ast(graph)
