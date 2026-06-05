"""WGSL #include directive preprocessor for material shaders.

This module provides include resolution for WGSL shader files, supporting:
- Relative includes: #include "brdf.wgsl" (from current file directory)
- Project includes: #include <pbr/common.wgsl> (from search paths)
- Recursive resolution with cycle detection and max depth enforcement
- Dependency graph tracking for hot-reload invalidation

Example::

    resolver = IncludeResolver(
        search_paths=[Path("shaders/"), Path("engine/shaders/")],
        max_depth=10
    )

    # Resolve all includes in a shader file
    resolved_source = resolver.resolve_file(Path("materials/pbr.wgsl"))

    # Get dependency graph for hot-reload
    deps = resolver.get_dependencies(Path("materials/pbr.wgsl"))
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class IncludeError(Exception):
    """Base exception for include resolution errors."""
    pass


class CyclicIncludeError(IncludeError):
    """Raised when a cyclic include dependency is detected.

    Attributes:
        cycle: The list of paths forming the cycle.
    """

    def __init__(self, cycle: list[Path]):
        self.cycle = cycle
        cycle_str = " -> ".join(str(p) for p in cycle)
        super().__init__(f"Cyclic include detected: {cycle_str}")


class MaxDepthError(IncludeError):
    """Raised when include nesting exceeds the maximum allowed depth.

    Attributes:
        depth: The depth at which the error occurred.
        max_depth: The configured maximum depth.
        include_stack: The include chain that led to this error.
    """

    def __init__(self, depth: int, max_depth: int, include_stack: list[Path]):
        self.depth = depth
        self.max_depth = max_depth
        self.include_stack = include_stack
        stack_str = " -> ".join(str(p) for p in include_stack)
        super().__init__(
            f"Include depth {depth} exceeds maximum {max_depth}. "
            f"Stack: {stack_str}"
        )


class IncludeFileNotFoundError(IncludeError):
    """Raised when an included file cannot be found.

    Attributes:
        include_path: The path specified in the include directive.
        search_paths: The paths that were searched.
        source_file: The file containing the include directive.
    """

    def __init__(
        self,
        include_path: str,
        search_paths: list[Path],
        source_file: Optional[Path] = None
    ):
        self.include_path = include_path
        self.search_paths = search_paths
        self.source_file = source_file

        searched = ", ".join(str(p) for p in search_paths)
        source_info = f" in {source_file}" if source_file else ""
        super().__init__(
            f"Include file not found: '{include_path}'{source_info}. "
            f"Searched: [{searched}]"
        )


@dataclass
class IncludeDirective:
    """Represents a parsed #include directive.

    Attributes:
        path: The path string from the directive.
        is_relative: True for "path", False for <path>.
        line_number: Line number in the source file.
        original_text: The full original directive text.
    """
    path: str
    is_relative: bool
    line_number: int
    original_text: str


@dataclass
class DepGraph:
    """Dependency graph tracking include relationships.

    Tracks bidirectional relationships:
    - includes_to_files: Maps include path -> set of files that include it
    - files_to_includes: Maps file path -> set of files it includes

    This enables efficient invalidation when an include file changes.
    """
    includes_to_files: dict[Path, set[Path]] = field(default_factory=dict)
    files_to_includes: dict[Path, set[Path]] = field(default_factory=dict)

    def add_edge(self, source: Path, included: Path) -> None:
        """Record that source includes the included file.

        Args:
            source: The file containing the include directive.
            included: The file being included.
        """
        # Forward edge: source depends on included
        if source not in self.files_to_includes:
            self.files_to_includes[source] = set()
        self.files_to_includes[source].add(included)

        # Reverse edge: included is used by source
        if included not in self.includes_to_files:
            self.includes_to_files[included] = set()
        self.includes_to_files[included].add(source)

    def get_dependents(self, include_file: Path) -> set[Path]:
        """Get all files that directly include the given file.

        Args:
            include_file: The include file to check.

        Returns:
            Set of file paths that directly include this file.
        """
        return self.includes_to_files.get(include_file, set()).copy()

    def get_dependencies(self, source_file: Path) -> set[Path]:
        """Get all files directly included by the given file.

        Args:
            source_file: The source file to check.

        Returns:
            Set of file paths directly included by this file.
        """
        return self.files_to_includes.get(source_file, set()).copy()

    def get_transitive_dependents(self, include_file: Path) -> set[Path]:
        """Get all files that transitively depend on the given file.

        Uses BFS to find all files that would need recompilation
        if the given include file changes.

        Args:
            include_file: The include file that changed.

        Returns:
            Set of all files transitively depending on this file.
        """
        visited: set[Path] = set()
        queue = list(self.get_dependents(include_file))

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Add files that include the current file
            for dependent in self.get_dependents(current):
                if dependent not in visited:
                    queue.append(dependent)

        return visited

    def get_transitive_dependencies(self, source_file: Path) -> set[Path]:
        """Get all files transitively included by the given file.

        Args:
            source_file: The source file to analyze.

        Returns:
            Set of all files transitively included.
        """
        visited: set[Path] = set()
        queue = list(self.get_dependencies(source_file))

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for dep in self.get_dependencies(current):
                if dep not in visited:
                    queue.append(dep)

        return visited

    def clear(self) -> None:
        """Clear all dependency information."""
        self.includes_to_files.clear()
        self.files_to_includes.clear()

    def remove_file(self, file_path: Path) -> None:
        """Remove a file and all its edges from the graph.

        Args:
            file_path: The file to remove.
        """
        # Remove forward edges
        if file_path in self.files_to_includes:
            for included in self.files_to_includes[file_path]:
                if included in self.includes_to_files:
                    self.includes_to_files[included].discard(file_path)
            del self.files_to_includes[file_path]

        # Remove reverse edges
        if file_path in self.includes_to_files:
            for source in self.includes_to_files[file_path]:
                if source in self.files_to_includes:
                    self.files_to_includes[source].discard(file_path)
            del self.includes_to_files[file_path]


class IncludeResolver:
    """Resolves #include directives in WGSL shader source files.

    Supports two include syntaxes:
    - Relative: #include "path/file.wgsl" - resolved from current file's directory
    - Project: #include <path/file.wgsl> - resolved from search paths

    Features:
    - Recursive include resolution
    - Cycle detection
    - Maximum depth enforcement
    - Dependency graph tracking for hot-reload
    - Include guards via #pragma once

    Attributes:
        search_paths: List of directories to search for project includes.
        max_depth: Maximum allowed include nesting depth.
        dep_graph: Dependency graph tracking include relationships.
    """

    # Regex patterns for include directives
    # Match: #include "path" or #include <path>
    INCLUDE_PATTERN = re.compile(
        r'^[ \t]*#include[ \t]+([<"])([^>"]+)[>"][ \t]*(?://.*)?$',
        re.MULTILINE
    )

    # Match: #pragma once
    PRAGMA_ONCE_PATTERN = re.compile(
        r'^[ \t]*#pragma[ \t]+once[ \t]*(?://.*)?$',
        re.MULTILINE
    )

    def __init__(
        self,
        search_paths: Optional[list[Path]] = None,
        max_depth: int = 10
    ):
        """Initialize the include resolver.

        Args:
            search_paths: Directories to search for project includes (<...>).
                Defaults to empty list.
            max_depth: Maximum allowed include nesting depth. Defaults to 10.
        """
        self.search_paths = [Path(p) for p in (search_paths or [])]
        self.max_depth = max_depth
        self.dep_graph = DepGraph()
        self._included_files: set[Path] = set()  # Track #pragma once files

    def resolve(
        self,
        source: str,
        current_file: Optional[Path] = None,
        _depth: int = 0,
        _include_stack: Optional[list[Path]] = None
    ) -> str:
        """Resolve all #include directives in source code.

        Args:
            source: WGSL source code with include directives.
            current_file: Path to the file containing this source.
                Used for relative include resolution.
            _depth: Internal parameter tracking current recursion depth.
            _include_stack: Internal parameter tracking include chain for cycle detection.

        Returns:
            Source code with all includes resolved and inlined.

        Raises:
            CyclicIncludeError: If a cyclic include dependency is detected.
            MaxDepthError: If include nesting exceeds max_depth.
            IncludeFileNotFoundError: If an included file cannot be found.
        """
        if _include_stack is None:
            _include_stack = []
            self._included_files.clear()  # Reset pragma once tracking

        # Check max depth
        if _depth > self.max_depth:
            raise MaxDepthError(_depth, self.max_depth, _include_stack.copy())

        # Track current file for cycle detection
        if current_file is not None:
            current_file = current_file.resolve()

            # Check for cycles
            if current_file in _include_stack:
                cycle = _include_stack[_include_stack.index(current_file):] + [current_file]
                raise CyclicIncludeError(cycle)

            _include_stack.append(current_file)

        # Check for #pragma once
        if current_file and self.PRAGMA_ONCE_PATTERN.search(source):
            if current_file in self._included_files:
                # Already included, return empty
                if current_file in _include_stack:
                    _include_stack.remove(current_file)
                return ""
            self._included_files.add(current_file)

        # Remove #pragma once directives from output
        source = self.PRAGMA_ONCE_PATTERN.sub("", source)

        # Find and resolve all includes
        def replace_include(match: re.Match) -> str:
            bracket_type = match.group(1)
            include_path = match.group(2)
            is_relative = bracket_type == '"'

            # Resolve the include path
            resolved_path = self._resolve_include_path(
                include_path,
                is_relative,
                current_file
            )

            # Record dependency edge
            if current_file is not None:
                self.dep_graph.add_edge(current_file, resolved_path)

            # Read and recursively resolve the included file
            try:
                included_source = resolved_path.read_text(encoding="utf-8")
            except OSError as e:
                raise IncludeFileNotFoundError(
                    include_path,
                    self._get_search_paths(is_relative, current_file),
                    current_file
                ) from e

            # Recursively resolve includes in the included file
            resolved_content = self.resolve(
                included_source,
                resolved_path,
                _depth + 1,
                _include_stack.copy() if current_file else None
            )

            # Add source markers for debugging
            return (
                f"// >>> BEGIN INCLUDE: {include_path}\n"
                f"{resolved_content}\n"
                f"// <<< END INCLUDE: {include_path}"
            )

        result = self.INCLUDE_PATTERN.sub(replace_include, source)

        # Pop current file from stack
        if current_file is not None and current_file in _include_stack:
            _include_stack.remove(current_file)

        return result

    def resolve_file(self, file_path: Path) -> str:
        """Resolve all includes in a file.

        Args:
            file_path: Path to the WGSL file to process.

        Returns:
            Fully resolved source code.

        Raises:
            CyclicIncludeError: If a cyclic include dependency is detected.
            MaxDepthError: If include nesting exceeds max_depth.
            IncludeFileNotFoundError: If an included file cannot be found.
            FileNotFoundError: If the source file doesn't exist.
        """
        file_path = Path(file_path).resolve()
        source = file_path.read_text(encoding="utf-8")
        return self.resolve(source, file_path)

    def parse_includes(self, source: str) -> list[IncludeDirective]:
        """Parse all include directives from source without resolving them.

        Args:
            source: WGSL source code.

        Returns:
            List of IncludeDirective objects found in the source.
        """
        directives = []
        lines = source.split("\n")

        for line_num, line in enumerate(lines, start=1):
            match = self.INCLUDE_PATTERN.match(line)
            if match:
                bracket_type = match.group(1)
                include_path = match.group(2)
                directives.append(IncludeDirective(
                    path=include_path,
                    is_relative=bracket_type == '"',
                    line_number=line_num,
                    original_text=line.strip()
                ))

        return directives

    def get_dependencies(self, file_path: Path) -> set[Path]:
        """Get all files that the given file depends on (transitively).

        Note: This only returns accurate results after resolve_file()
        has been called on the file, populating the dependency graph.

        Args:
            file_path: The file to analyze.

        Returns:
            Set of all files transitively included by this file.
        """
        return self.dep_graph.get_transitive_dependencies(Path(file_path).resolve())

    def get_dependents(self, file_path: Path) -> set[Path]:
        """Get all files that would need recompilation if this file changes.

        Args:
            file_path: The include file that changed.

        Returns:
            Set of files that transitively depend on this file.
        """
        return self.dep_graph.get_transitive_dependents(Path(file_path).resolve())

    def invalidate(self, changed_file: Path) -> set[Path]:
        """Get files needing recompilation when an include changes.

        This is the primary interface for hot-reload systems.

        Args:
            changed_file: The file that was modified.

        Returns:
            Set of all files that need to be recompiled.
        """
        return self.get_dependents(changed_file)

    def add_search_path(self, path: Path) -> None:
        """Add a directory to the search paths.

        Args:
            path: Directory to add to search paths.
        """
        path = Path(path).resolve()
        if path not in self.search_paths:
            self.search_paths.append(path)

    def clear_cache(self) -> None:
        """Clear the dependency graph and pragma once tracking."""
        self.dep_graph.clear()
        self._included_files.clear()

    def _resolve_include_path(
        self,
        include_path: str,
        is_relative: bool,
        current_file: Optional[Path]
    ) -> Path:
        """Resolve an include path to an absolute file path.

        Args:
            include_path: The path from the include directive.
            is_relative: True if using "..." syntax (relative include).
            current_file: The file containing the include directive.

        Returns:
            Resolved absolute path to the include file.

        Raises:
            IncludeFileNotFoundError: If the file cannot be found.
        """
        search_paths = self._get_search_paths(is_relative, current_file)

        for search_path in search_paths:
            candidate = search_path / include_path
            if candidate.is_file():
                return candidate.resolve()

        raise IncludeFileNotFoundError(include_path, search_paths, current_file)

    def _get_search_paths(
        self,
        is_relative: bool,
        current_file: Optional[Path]
    ) -> list[Path]:
        """Get the list of paths to search for an include.

        Args:
            is_relative: True for relative includes ("...").
            current_file: The file containing the include directive.

        Returns:
            List of directories to search, in order.
        """
        paths = []

        # For relative includes, search from current file's directory first
        if is_relative and current_file is not None:
            paths.append(current_file.parent)

        # Add configured search paths
        paths.extend(self.search_paths)

        return paths


def preprocess_wgsl(
    source: str,
    search_paths: Optional[list[Path]] = None,
    current_file: Optional[Path] = None,
    max_depth: int = 10
) -> str:
    """Convenience function to preprocess WGSL source with includes.

    Args:
        source: WGSL source code with include directives.
        search_paths: Directories to search for project includes.
        current_file: Path to the file containing this source.
        max_depth: Maximum allowed include nesting depth.

    Returns:
        Source code with all includes resolved.
    """
    resolver = IncludeResolver(search_paths, max_depth)
    return resolver.resolve(source, current_file)
