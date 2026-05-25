"""Code Generation module for FlowForge Backend.

This module provides code generation capabilities:
- Converting flow graphs to Python code
- Graph to AST conversion
- Python code emission and formatting
- Code validation
- Diff generation for code preview

Main entry point:
    generate_python(graph) -> str

Example:
    from flowforge_backend.codegen import generate_python

    graph = {
        "nodes": [
            {
                "id": "node_1",
                "type": "component",
                "name": "Position",
                "data": {
                    "fields": [
                        {"name": "x", "type_annotation": "float", "default_value": "0.0"},
                        {"name": "y", "type_annotation": "float", "default_value": "0.0"},
                    ]
                }
            }
        ],
        "edges": []
    }

    source = generate_python(graph)
    print(source)
"""

from __future__ import annotations

import ast
from typing import Any, Union

# Import from existing modules
from .diff import (
    DiffLineType,
    DiffLine,
    DiffHunk,
    DiffStats,
    DiffResult,
    generate_diff,
    generate_side_by_side_diff,
)

from .types import (
    Severity,
    ValidationError,
    ValidationResult,
    ImportInfo,
    GenerationResult,
)

from .validator import (
    validate_python,
    validate_and_format,
    quick_validate,
    get_syntax_error_details,
)

# Import from new modules
from .graph_to_ast import graph_to_ast, nodes_to_ast
from .emitter import (
    emit_python,
    emit_python_minimal,
    emit_class,
    validate_syntax,
    roundtrip_ast,
)
from .imports import (
    generate_imports,
    generate_imports_from_nodes,
    ImportCollector,
)


def generate_python(
    graph: Union[dict[str, Any], "NodeGraph"],
    format_with_black: bool = True,
    line_length: int = 88,
    add_header: bool = True,
) -> str:
    """Convert a node graph to Python source code.

    This is the main high-level function for code generation. It takes
    a visual node graph representation and produces valid, formatted
    Python source code with Trinity ECS decorators.

    Args:
        graph: Node graph with nodes and edges. Can be a dictionary
            (from JSON) or a NodeGraph instance.
        format_with_black: Whether to format with black (default: True).
        line_length: Maximum line length for formatting (default: 88).
        add_header: Whether to add a generated code header (default: True).

    Returns:
        Python source code as a string.

    Example:
        >>> graph = {"nodes": [...], "edges": [...]}
        >>> source = generate_python(graph)
        >>> print(source)
    """
    # Convert graph to AST
    module = graph_to_ast(graph)

    # Convert AST to source code
    source = emit_python(
        module,
        format_with_black=format_with_black,
        line_length=line_length,
        add_header=add_header,
    )

    return source


def generate_python_with_validation(
    graph: Union[dict[str, Any], "NodeGraph"],
    format_with_black: bool = True,
    line_length: int = 88,
    add_header: bool = True,
    check_semantics: bool = False,
) -> GenerationResult:
    """Convert a node graph to Python source code with validation.

    Similar to generate_python but returns a GenerationResult that
    includes validation information and metadata.

    Args:
        graph: Node graph with nodes and edges.
        format_with_black: Whether to format with black (default: True).
        line_length: Maximum line length for formatting (default: 88).
        add_header: Whether to add a generated code header (default: True).
        check_semantics: If True, perform semantic checks (default: False).

    Returns:
        GenerationResult with source, validation, and metadata.
    """
    try:
        # Generate the source
        source = generate_python(
            graph,
            format_with_black=format_with_black,
            line_length=line_length,
            add_header=add_header,
        )

        # Validate the generated source
        validation = validate_python(source, check_semantics=check_semantics)

        # Count nodes
        if isinstance(graph, dict):
            node_count = len(graph.get("nodes", []))
        else:
            node_count = len(graph.nodes)

        return GenerationResult(
            source=source,
            validation=validation,
            node_count=node_count,
            metadata={
                "format_with_black": format_with_black,
                "line_length": line_length,
                "add_header": add_header,
            },
        )

    except Exception as e:
        return GenerationResult.error(str(e))


__all__ = [
    # Main entry points
    "generate_python",
    "generate_python_with_validation",
    # Graph to AST conversion
    "graph_to_ast",
    "nodes_to_ast",
    # AST to source code
    "emit_python",
    "emit_python_minimal",
    "emit_class",
    "validate_syntax",
    "roundtrip_ast",
    # Import generation
    "generate_imports",
    "generate_imports_from_nodes",
    "ImportCollector",
    # Diff types and functions
    "DiffLineType",
    "DiffLine",
    "DiffHunk",
    "DiffStats",
    "DiffResult",
    "generate_diff",
    "generate_side_by_side_diff",
    # Validation types
    "Severity",
    "ValidationError",
    "ValidationResult",
    "ImportInfo",
    "GenerationResult",
    # Validation functions
    "validate_python",
    "validate_and_format",
    "quick_validate",
    "get_syntax_error_details",
]
