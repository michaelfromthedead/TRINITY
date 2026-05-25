"""Multi-file/directory project parsing for FlowForge.

This module provides the ProjectParser class for parsing entire Python projects
and merging the results into a single NodeGraph representation.

Example:
    from flowforge_backend.ast_parser.project import ProjectParser

    parser = ProjectParser("/path/to/project")
    files = parser.scan_files()
    print(f"Found {len(files)} Python files")

    graph = parser.parse_all()
    print(f"Parsed {len(graph.nodes)} nodes")

    # Access file-level statistics
    stats = parser.get_file_stats()
    for file_path, count in stats.items():
        print(f"{file_path}: {count} definitions")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .constants import DEFAULT_EXCLUDE_PATTERNS, PYTHON_FILE_EXTENSION
from .types import (
    ComponentDef,
    EventDef,
    ImportDef,
    ParseResult,
    ResourceDef,
    SystemDef,
)
from .graph import build_node_graph
from .graph_types import NodeGraph
from .visitor import parse_file


class ProjectParser:
    """Parser for multi-file Python projects.

    Scans a directory tree for Python files, parses each file for Trinity
    definitions, and merges the results into a single NodeGraph.

    Attributes:
        root_path: The root directory path being parsed.
        exclude_patterns: Patterns to exclude from scanning (e.g., __pycache__).

    Example:
        parser = ProjectParser("/path/to/game")
        graph = parser.parse_all()

        # Check for parsing errors
        if parser.has_errors:
            for error in parser.all_errors:
                print(f"Error: {error}")

        # Get statistics
        stats = parser.get_file_stats()
        print(f"Parsed {parser.total_files} files")
    """

    def __init__(
        self,
        root_path: str,
        exclude_patterns: Optional[set[str]] = None,
    ) -> None:
        """Initialize the project parser.

        Args:
            root_path: Path to the root directory to parse.
            exclude_patterns: Optional set of directory/file patterns to exclude.
                             Defaults to common Python project exclusions.
        """
        self.root_path = Path(root_path).resolve()
        self.exclude_patterns = exclude_patterns or set(DEFAULT_EXCLUDE_PATTERNS)

        # Internal state
        self._results: list[ParseResult] = []
        self._file_paths: list[Path] = []
        self._parsed: bool = False

    @property
    def has_errors(self) -> bool:
        """Check if any parsing errors occurred."""
        return any(result.has_errors for result in self._results)

    @property
    def all_errors(self) -> list[str]:
        """Get all parsing errors from all files."""
        errors: list[str] = []
        for result in self._results:
            errors.extend(result.errors)
        return errors

    @property
    def total_files(self) -> int:
        """Get the total number of files scanned."""
        return len(self._file_paths)

    @property
    def total_definitions(self) -> int:
        """Get the total number of Trinity definitions found."""
        return sum(len(result.all_definitions) for result in self._results)

    def _should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded from scanning.

        Args:
            path: The path to check.

        Returns:
            True if the path should be excluded.
        """
        for part in path.parts:
            if part in self.exclude_patterns:
                return True
            # Handle wildcard patterns like "*.egg-info"
            for pattern in self.exclude_patterns:
                if pattern.startswith("*") and part.endswith(pattern[1:]):
                    return True
        return False

    def scan_files(self) -> list[Path]:
        """Find all Python files in the project.

        Recursively scans the root directory for .py files, excluding
        directories matching the exclude patterns.

        Returns:
            List of Path objects for all Python files found.
        """
        python_files: list[Path] = []

        if not self.root_path.exists():
            return python_files

        if self.root_path.is_file():
            # If root_path is a file, just return it if it's a Python file
            if self.root_path.suffix == PYTHON_FILE_EXTENSION:
                python_files.append(self.root_path)
        else:
            # Recursively find all Python files
            for file_path in self.root_path.rglob(f"*{PYTHON_FILE_EXTENSION}"):
                if not self._should_exclude(file_path):
                    python_files.append(file_path)

        # Sort by path for consistent ordering
        python_files.sort()
        self._file_paths = python_files
        return python_files

    def parse_file(self, file_path: Path) -> ParseResult:
        """Parse a single Python file.

        Args:
            file_path: Path to the Python file.

        Returns:
            ParseResult containing all Trinity definitions from the file.
        """
        return parse_file(str(file_path))

    def parse_all(self, apply_layout: bool = True) -> NodeGraph:
        """Parse all Python files and merge into a single graph.

        Scans for Python files if not already done, parses each file,
        and merges all definitions into a single NodeGraph.

        Args:
            apply_layout: Whether to auto-position nodes in the graph.

        Returns:
            NodeGraph containing all nodes and edges from the project.
        """
        # Scan files if not already done
        if not self._file_paths:
            self.scan_files()

        # Reset results
        self._results = []

        # Collect all definitions
        all_components: list[ComponentDef] = []
        all_systems: list[SystemDef] = []
        all_resources: list[ResourceDef] = []
        all_events: list[EventDef] = []
        all_imports: list[ImportDef] = []
        all_errors: list[str] = []

        # Track file boundaries for metadata
        file_boundaries: dict[str, dict[str, Any]] = {}

        for file_path in self._file_paths:
            result = self.parse_file(file_path)
            self._results.append(result)

            # Track definitions per file
            file_key = str(file_path)
            relative_path = self._get_relative_path(file_path)
            file_boundaries[file_key] = {
                "relative_path": relative_path,
                "components": len(result.components),
                "systems": len(result.systems),
                "resources": len(result.resources),
                "events": len(result.events),
                "definition_names": [d.name for d in result.all_definitions],
            }

            # Merge definitions
            all_components.extend(result.components)
            all_systems.extend(result.systems)
            all_resources.extend(result.resources)
            all_events.extend(result.events)
            all_imports.extend(result.imports)
            all_errors.extend(result.errors)

        # Create merged ParseResult
        merged = ParseResult(
            source_file=str(self.root_path),
            components=all_components,
            systems=all_systems,
            resources=all_resources,
            events=all_events,
            imports=all_imports,
            errors=all_errors,
        )

        self._parsed = True

        # Build the node graph
        graph = build_node_graph(merged, apply_layout=apply_layout)

        # Add project metadata
        graph.metadata = {
            "project_root": str(self.root_path),
            "total_files": len(self._file_paths),
            "total_definitions": len(merged.all_definitions),
            "file_boundaries": file_boundaries,
            "has_errors": len(all_errors) > 0,
            "error_count": len(all_errors),
        }

        return graph

    def _get_relative_path(self, file_path: Path) -> str:
        """Get the path relative to the project root.

        Args:
            file_path: Absolute path to a file.

        Returns:
            Path string relative to root_path.
        """
        try:
            return str(file_path.relative_to(self.root_path))
        except ValueError:
            # File is not under root_path, return absolute
            return str(file_path)

    def get_file_stats(self) -> dict[str, int]:
        """Get statistics about definitions per file.

        Returns:
            Dictionary mapping file paths to definition counts.
        """
        stats: dict[str, int] = {}
        for result in self._results:
            stats[result.source_file] = len(result.all_definitions)
        return stats

    def get_definitions_by_file(self) -> dict[str, list[str]]:
        """Get definition names grouped by file.

        Returns:
            Dictionary mapping file paths to lists of definition names.
        """
        by_file: dict[str, list[str]] = {}
        for result in self._results:
            by_file[result.source_file] = [d.name for d in result.all_definitions]
        return by_file

    def get_parse_results(self) -> list[ParseResult]:
        """Get all individual parse results.

        Returns:
            List of ParseResult objects, one per parsed file.
        """
        return list(self._results)


def parse_project(
    root_path: str,
    exclude_patterns: Optional[set[str]] = None,
    apply_layout: bool = True,
) -> NodeGraph:
    """Convenience function to parse an entire project directory.

    Args:
        root_path: Path to the project root directory.
        exclude_patterns: Optional patterns to exclude from scanning.
        apply_layout: Whether to auto-position nodes.

    Returns:
        NodeGraph containing all Trinity definitions from the project.

    Example:
        graph = parse_project("/path/to/game")
        for node in graph.nodes:
            print(f"{node.type}: {node.name}")
    """
    parser = ProjectParser(root_path, exclude_patterns)
    return parser.parse_all(apply_layout=apply_layout)


def parse_files(
    file_paths: list[str],
    apply_layout: bool = True,
) -> NodeGraph:
    """Parse a specific list of files and merge into a single graph.

    Unlike parse_project which scans a directory, this function takes
    an explicit list of file paths to parse.

    Args:
        file_paths: List of paths to Python files to parse.
        apply_layout: Whether to auto-position nodes.

    Returns:
        NodeGraph containing all Trinity definitions from the files.

    Example:
        graph = parse_files([
            "/path/to/components.py",
            "/path/to/systems.py",
        ])
    """
    all_components: list[ComponentDef] = []
    all_systems: list[SystemDef] = []
    all_resources: list[ResourceDef] = []
    all_events: list[EventDef] = []
    all_imports: list[ImportDef] = []
    all_errors: list[str] = []

    file_boundaries: dict[str, dict[str, Any]] = {}

    for file_path in file_paths:
        result = parse_file(file_path)

        file_boundaries[file_path] = {
            "components": len(result.components),
            "systems": len(result.systems),
            "resources": len(result.resources),
            "events": len(result.events),
            "definition_names": [d.name for d in result.all_definitions],
        }

        all_components.extend(result.components)
        all_systems.extend(result.systems)
        all_resources.extend(result.resources)
        all_events.extend(result.events)
        all_imports.extend(result.imports)
        all_errors.extend(result.errors)

    # Determine common root if possible
    if file_paths:
        common_root = os.path.commonpath(file_paths) if len(file_paths) > 1 else os.path.dirname(file_paths[0])
    else:
        common_root = ""

    merged = ParseResult(
        source_file=common_root,
        components=all_components,
        systems=all_systems,
        resources=all_resources,
        events=all_events,
        imports=all_imports,
        errors=all_errors,
    )

    graph = build_node_graph(merged, apply_layout=apply_layout)

    graph.metadata = {
        "source_files": file_paths,
        "total_files": len(file_paths),
        "total_definitions": len(merged.all_definitions),
        "file_boundaries": file_boundaries,
        "has_errors": len(all_errors) > 0,
        "error_count": len(all_errors),
    }

    return graph
