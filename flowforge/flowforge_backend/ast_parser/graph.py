"""Node graph building entry point.

This module provides the high-level API for converting parsed Trinity
definitions into a visual node graph representation.
"""

from __future__ import annotations

import os
from typing import Optional

from .constants import DEFAULT_SOURCE_NAME
from .types import ParseResult
from .graph_types import NodeGraph
from .graph_builder import build_graph_from_parse_result
from .layout import LayoutEngine
from .cache import ASTCache, get_default_cache


def build_node_graph(parse_result: ParseResult, apply_layout: bool = True) -> NodeGraph:
    """Convert parsed Trinity definitions to a node graph.

    Args:
        parse_result: The result from parsing Python source
        apply_layout: Whether to auto-position nodes

    Returns:
        NodeGraph ready for JSON serialization and frontend display
    """
    # Build nodes and edges using the unified builder
    graph = build_graph_from_parse_result(parse_result)

    # Apply layout if requested
    if apply_layout:
        layout = LayoutEngine(graph.nodes, graph.edges)
        layout.apply_hierarchical_layout()

    return graph


def incremental_parse_directory(
    directory: str, previous_graph: Optional[NodeGraph] = None
) -> NodeGraph:
    """Parse a directory incrementally, re-parsing only changed files.

    Uses IncrementalParser to detect which files have changed and only
    re-parses those, preserving node positions for unchanged files.

    Args:
        directory: Path to the project directory.
        previous_graph: Optional previous graph for incremental updates.

    Returns:
        NodeGraph with all nodes and edges.
    """
    from .incremental import IncrementalParser

    parser = IncrementalParser()
    return parser.parse_directory(directory, previous_graph=previous_graph)


def parse_and_build_graph(
    source: str,
    source_file: str = DEFAULT_SOURCE_NAME,
    cache: Optional[ASTCache] = None,
) -> NodeGraph:
    """Parse Python source and build node graph in one step.

    When *source_file* points to a real file, the result is cached and
    returned on subsequent calls as long as the file's mtime is unchanged.

    Args:
        source: Python source code as a string
        source_file: Optional file path for error messages
        cache: Optional ASTCache instance (uses module default if None)

    Returns:
        NodeGraph ready for JSON serialization and frontend display

    Raises:
        SyntaxError: If the source code has syntax errors
    """
    ast_cache = cache if cache is not None else get_default_cache()

    # Try cache for real files
    if source_file != DEFAULT_SOURCE_NAME and os.path.isfile(source_file):
        cached = ast_cache.get(source_file)
        if cached is not None:
            return cached

    from .visitor import parse_source
    result = parse_source(source, source_file)
    graph = build_node_graph(result)

    # Store in cache for real files
    if source_file != DEFAULT_SOURCE_NAME and os.path.isfile(source_file):
        ast_cache.put(source_file, graph)

    return graph
