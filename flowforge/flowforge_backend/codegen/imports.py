"""Import generation for code generation.

This module handles generating Python import statements based on
the node types and their fields in a graph.

It tracks:
- Trinity imports (from trinity import component, system, etc.)
- Standard library imports
- Third-party imports
- Type annotation imports from typing module
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional, Set

from ..ast_parser.graph_types import GraphNode, NodeGraph
from ..ast_parser.constants import BUILTIN_TYPES, TRINITY_MODULE


# Types from the typing module that need to be imported
TYPING_TYPES: frozenset[str] = frozenset({
    "List", "Dict", "Set", "FrozenSet", "Tuple",
    "Optional", "Union", "Any", "Callable", "Type",
    "Sequence", "Mapping", "MutableMapping", "Iterable",
    "Iterator", "Generator", "Coroutine", "AsyncGenerator",
    "ClassVar", "Final", "Literal", "TypeVar", "Generic",
    "Protocol", "TypedDict", "NamedTuple",
})

# Types from typing_extensions
TYPING_EXTENSIONS_TYPES: frozenset[str] = frozenset({
    "Self", "TypeAlias", "ParamSpec", "Concatenate",
    "Annotated", "TypeGuard", "Required", "NotRequired",
})

# Trinity framework types
TRINITY_TYPES: frozenset[str] = frozenset({
    "Query", "Entity", "Commands", "EventWriter", "EventReader",
    "Res", "ResMut", "Local", "With", "Without", "Changed", "Added",
})


def _extract_type_names(type_annotation: str) -> Set[str]:
    """Extract all type names from a type annotation string.

    Handles generic types like List[int], Optional[Player], etc.

    Args:
        type_annotation: The type annotation string to parse.

    Returns:
        Set of type names found in the annotation.
    """
    if not type_annotation:
        return set()

    # Remove spaces and handle common patterns
    clean = type_annotation.strip()

    # Match word characters (type names)
    # This pattern captures type names in generics: List[Player] -> ["List", "Player"]
    names = set(re.findall(r'\b([A-Z][A-Za-z0-9_]*)\b', clean))

    # Also capture lowercase type names like 'int', 'str', etc.
    lowercase_names = set(re.findall(r'\b([a-z][a-z0-9_]*)\b', clean))

    return names | lowercase_names


@dataclass
class ImportCollector:
    """Collects and categorizes imports from node graph.

    This class tracks which imports are needed based on the types
    used in node fields, method parameters, and return types.

    Attributes:
        trinity_decorators: Set of Trinity decorators needed.
        trinity_types: Set of Trinity types needed (Query, etc.).
        typing_imports: Set of typing module types needed.
        typing_extensions_imports: Set of typing_extensions types needed.
        dataclass_import: Whether dataclasses import is needed.
        other_imports: Set of other imports (third-party, etc.).
        _known_classes: Set of class names defined in the graph.
    """
    trinity_decorators: Set[str] = field(default_factory=set)
    trinity_types: Set[str] = field(default_factory=set)
    typing_imports: Set[str] = field(default_factory=set)
    typing_extensions_imports: Set[str] = field(default_factory=set)
    dataclass_import: bool = False
    other_imports: Set[str] = field(default_factory=set)
    _known_classes: Set[str] = field(default_factory=set)

    def add_node(self, node: GraphNode) -> None:
        """Process a node and collect its required imports.

        Args:
            node: The graph node to process.
        """
        # Track this class as known (defined in graph)
        self._known_classes.add(node.name)

        # Add Trinity decorator based on node type
        if node.type in ("component", "system", "resource", "event"):
            self.trinity_decorators.add(node.type)

        # Process node data
        data = node.data if isinstance(node.data, dict) else node.data.to_dict()

        # Process fields
        fields = data.get("fields", [])
        for field_data in fields:
            type_ann = field_data.get("type_annotation") or field_data.get("type", "")
            self._process_type_annotation(type_ann)

            # Check if default value uses field()
            default = field_data.get("default_value") or field_data.get("default")
            if default and "field(" in str(default):
                self.dataclass_import = True

        # Process payload_fields for events
        payload_fields = data.get("payload_fields", [])
        for field_data in payload_fields:
            type_ann = field_data.get("type_annotation") or field_data.get("type", "")
            self._process_type_annotation(type_ann)

        # Process methods (for systems)
        methods = data.get("methods", [])
        for method in methods:
            # Process parameters
            params = method.get("parameters", [])
            for param in params:
                type_ann = param.get("type_annotation") or param.get("type", "")
                self._process_type_annotation(type_ann)

            # Process return type
            return_type = method.get("return_type")
            if return_type:
                self._process_type_annotation(return_type)

            # Process query types
            query_types = method.get("query_types", [])
            for qt in query_types:
                self._process_type_annotation(qt)

            # If method has Query types, add Query import
            if query_types or method.get("query_components"):
                self.trinity_types.add("Query")

        # Process queries (for systems)
        queries = data.get("queries", [])
        if queries:
            self.trinity_types.add("Query")
            for q in queries:
                self._process_type_annotation(q)

    def _process_type_annotation(self, type_annotation: str) -> None:
        """Process a type annotation and add necessary imports.

        Args:
            type_annotation: The type annotation string.
        """
        if not type_annotation:
            return

        names = _extract_type_names(type_annotation)

        for name in names:
            # Skip Python builtins (lowercase)
            if name in BUILTIN_TYPES or name in ("int", "float", "str", "bool", "bytes", "list", "dict", "set", "tuple", "None"):
                continue

            # Check if it's a typing module type
            if name in TYPING_TYPES:
                self.typing_imports.add(name)
                continue

            # Check if it's a typing_extensions type
            if name in TYPING_EXTENSIONS_TYPES:
                self.typing_extensions_imports.add(name)
                continue

            # Check if it's a Trinity type
            if name in TRINITY_TYPES:
                self.trinity_types.add(name)
                continue

            # Otherwise it might be a custom type - we don't import it
            # since it's probably defined in the same file

    def generate_import_lines(self) -> list[str]:
        """Generate sorted import lines.

        Returns:
            List of import statement strings, properly sorted.
        """
        lines: list[str] = []

        # Future annotations for forward references
        lines.append("from __future__ import annotations")
        lines.append("")

        # Standard library imports
        stdlib_imports: list[str] = []

        if self.dataclass_import:
            stdlib_imports.append("from dataclasses import dataclass, field")

        if self.typing_imports:
            sorted_types = sorted(self.typing_imports)
            stdlib_imports.append(f"from typing import {', '.join(sorted_types)}")

        if self.typing_extensions_imports:
            sorted_types = sorted(self.typing_extensions_imports)
            stdlib_imports.append(f"from typing_extensions import {', '.join(sorted_types)}")

        if stdlib_imports:
            lines.extend(sorted(stdlib_imports))
            lines.append("")

        # Trinity imports
        trinity_imports: list[str] = []

        if self.trinity_decorators:
            sorted_decorators = sorted(self.trinity_decorators)
            trinity_imports.append(f"from {TRINITY_MODULE} import {', '.join(sorted_decorators)}")

        if self.trinity_types:
            sorted_types = sorted(self.trinity_types)
            trinity_imports.append(f"from {TRINITY_MODULE} import {', '.join(sorted_types)}")

        # Combine and deduplicate Trinity imports
        if trinity_imports:
            # If both decorators and types, combine them
            if self.trinity_decorators and self.trinity_types:
                all_names = sorted(self.trinity_decorators | self.trinity_types)
                lines.append(f"from {TRINITY_MODULE} import {', '.join(all_names)}")
            else:
                lines.extend(trinity_imports)
            lines.append("")

        return lines


def generate_imports(graph: NodeGraph) -> list[str]:
    """Generate import statements for a node graph.

    This function analyzes all nodes in the graph and determines
    which imports are needed based on:
    - Node types (component, system, etc.) -> Trinity decorators
    - Field types -> typing module or Trinity types
    - Method parameters and return types
    - Query annotations

    Args:
        graph: The node graph to analyze.

    Returns:
        List of import statement strings.

    Example:
        >>> graph = NodeGraph(nodes=[...])
        >>> imports = generate_imports(graph)
        >>> for line in imports:
        ...     print(line)
        from __future__ import annotations

        from typing import Optional

        from trinity import component, system
    """
    collector = ImportCollector()

    for node in graph.nodes:
        collector.add_node(node)

    return collector.generate_import_lines()


def generate_imports_from_nodes(nodes: list[GraphNode]) -> list[str]:
    """Generate import statements from a list of nodes.

    Convenience function that creates a temporary graph.

    Args:
        nodes: List of graph nodes.

    Returns:
        List of import statement strings.
    """
    graph = NodeGraph(nodes=nodes, edges=[])
    return generate_imports(graph)
