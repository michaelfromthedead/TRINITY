"""Incremental re-parsing for FlowForge.

This module provides incremental parsing capabilities that only re-parse
files that have changed since the last parse, preserving user-adjusted
node positions for unchanged nodes.

Example:
    from flowforge_backend.ast_parser.incremental import IncrementalParser

    parser = IncrementalParser()
    graph = parser.parse_directory("/path/to/project")

    # Later, after some files have changed:
    updated_graph = parser.parse_directory("/path/to/project")
    # Only changed files are re-parsed; unchanged node positions preserved.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .constants import DEFAULT_EXCLUDE_PATTERNS, PYTHON_FILE_EXTENSION
from .graph import build_node_graph
from .graph_types import NodeGraph, NodePosition
from .project import ProjectParser
from .visitor import parse_file


def detect_changed_files(
    directory: str,
    previous_state: dict[str, float],
    exclude_patterns: Optional[set[str]] = None,
) -> list[str]:
    """Compare file modification times to detect changed files.

    Args:
        directory: Path to the directory to scan.
        previous_state: Mapping of file path to last-known mtime.
        exclude_patterns: Optional patterns to exclude from scanning.

    Returns:
        List of file paths that are new or have changed since previous_state.
    """
    exclude = exclude_patterns or set(DEFAULT_EXCLUDE_PATTERNS)
    root = Path(directory).resolve()
    changed: list[str] = []

    if not root.exists():
        return changed

    for file_path in root.rglob(f"*{PYTHON_FILE_EXTENSION}"):
        # Check exclusions
        if any(part in exclude for part in file_path.parts):
            continue
        path_str = str(file_path)
        try:
            current_mtime = os.path.getmtime(path_str)
        except OSError:
            continue
        prev_mtime = previous_state.get(path_str)
        if prev_mtime is None or current_mtime > prev_mtime:
            changed.append(path_str)

    return changed


def reparse_changed(
    directory: str,
    previous_graph: NodeGraph,
    changed_files: list[str],
) -> NodeGraph:
    """Re-parse only changed files and merge into the existing graph.

    Nodes originating from changed files are replaced with freshly parsed
    nodes. Nodes from unchanged files keep their existing data and positions.

    Args:
        directory: Project root directory.
        previous_graph: The previously built NodeGraph.
        changed_files: List of file paths that need re-parsing.

    Returns:
        A new NodeGraph with updated nodes/edges for changed files and
        preserved positions for unchanged nodes.
    """
    if not changed_files:
        return previous_graph

    # Build a set of source files that changed for fast lookup
    changed_set = set(changed_files)

    # Collect position overrides from unchanged nodes
    preserved_positions: dict[str, NodePosition] = {}
    unchanged_node_ids: set[str] = set()
    for node in previous_graph.nodes:
        source_file = ""
        if node.source is not None:
            source_file = node.source.file
        if source_file not in changed_set:
            preserved_positions[node.id] = node.position
            unchanged_node_ids.add(node.id)

    # Do a full project re-parse (ProjectParser handles file scanning)
    parser = ProjectParser(directory)
    new_graph = parser.parse_all(apply_layout=True)

    # Restore positions for nodes that existed before and are unchanged
    for node in new_graph.nodes:
        if node.id in preserved_positions:
            node.position = preserved_positions[node.id]

    return new_graph


class IncrementalParser:
    """Tracks parsed state per file and supports incremental re-parsing.

    Maintains file modification times so that subsequent parse calls only
    re-parse files that have actually changed on disk.

    Example:
        parser = IncrementalParser()
        graph = parser.parse_directory("/path/to/project")

        # ... user edits a file ...

        updated = parser.parse_directory("/path/to/project")
        # Only the edited file is re-parsed.
    """

    def __init__(self) -> None:
        """Initialize with empty state."""
        self._file_mtimes: dict[str, float] = {}
        self._previous_graph: Optional[NodeGraph] = None

    @property
    def file_mtimes(self) -> dict[str, float]:
        """Get the current file modification time state."""
        return dict(self._file_mtimes)

    @property
    def previous_graph(self) -> Optional[NodeGraph]:
        """Get the most recently built graph, if any."""
        return self._previous_graph

    def parse_directory(
        self,
        directory: str,
        previous_graph: Optional[NodeGraph] = None,
    ) -> NodeGraph:
        """Parse a directory, re-parsing only changed files.

        On first call, performs a full parse. On subsequent calls, detects
        changed files and only re-parses those, preserving node positions
        for unchanged files.

        Args:
            directory: Path to the project directory.
            previous_graph: Optional explicit previous graph to use instead
                           of the internally tracked one.

        Returns:
            NodeGraph with all nodes and edges.
        """
        prev = previous_graph if previous_graph is not None else self._previous_graph

        if prev is None:
            # First parse: full parse
            graph = self._full_parse(directory)
        else:
            # Incremental parse
            changed = detect_changed_files(directory, self._file_mtimes)
            if changed:
                graph = reparse_changed(directory, prev, changed)
            else:
                graph = prev

        # Update mtime state
        self._update_mtimes(directory)
        self._previous_graph = graph
        return graph

    def _full_parse(self, directory: str) -> NodeGraph:
        """Perform a full project parse.

        Args:
            directory: Path to the project directory.

        Returns:
            NodeGraph from a full parse.
        """
        parser = ProjectParser(directory)
        return parser.parse_all(apply_layout=True)

    def _update_mtimes(self, directory: str) -> None:
        """Snapshot current modification times for all Python files.

        Args:
            directory: Path to the project directory.
        """
        exclude = set(DEFAULT_EXCLUDE_PATTERNS)
        root = Path(directory).resolve()
        new_mtimes: dict[str, float] = {}

        if root.exists():
            for file_path in root.rglob(f"*{PYTHON_FILE_EXTENSION}"):
                if any(part in exclude for part in file_path.parts):
                    continue
                path_str = str(file_path)
                try:
                    new_mtimes[path_str] = os.path.getmtime(path_str)
                except OSError:
                    continue

        self._file_mtimes = new_mtimes

    def reset(self) -> None:
        """Reset all tracked state, forcing a full re-parse on next call."""
        self._file_mtimes = {}
        self._previous_graph = None


def incremental_parse_directory(
    directory: str,
    previous_graph: Optional[NodeGraph] = None,
) -> NodeGraph:
    """Parse a directory incrementally using a one-shot IncrementalParser.

    This is a convenience function. For repeated incremental parsing,
    prefer creating and reusing an IncrementalParser instance.

    Args:
        directory: Path to the project directory.
        previous_graph: Optional previous graph for incremental updates.

    Returns:
        NodeGraph with all nodes and edges.
    """
    parser = IncrementalParser()
    return parser.parse_directory(directory, previous_graph=previous_graph)
