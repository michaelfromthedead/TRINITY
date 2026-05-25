"""
Dependency Tracker - Track module dependencies for cascade reloads.

Analyzes import relationships between modules to determine which
modules need to be reloaded when a dependency changes.
"""
from __future__ import annotations

import ast
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple


@dataclass
class ModuleNode:
    """Node in the dependency graph representing a module."""

    name: str
    file_path: Optional[str] = None
    imports: Set[str] = field(default_factory=set)  # Modules this imports
    imported_by: Set[str] = field(default_factory=set)  # Modules that import this
    is_package: bool = False
    last_analyzed: float = 0.0

    @property
    def dependents(self) -> Set[str]:
        """Alias for imported_by (modules that depend on this one)."""
        return self.imported_by

    @property
    def dependencies(self) -> Set[str]:
        """Alias for imports (modules this depends on)."""
        return self.imports


class DependencyGraph:
    """
    Graph of module dependencies.

    Provides methods for:
    - Adding/removing nodes
    - Computing transitive closures
    - Finding reload order
    - Detecting cycles
    """

    def __init__(self):
        """Initialize the dependency graph."""
        self._nodes: Dict[str, ModuleNode] = {}
        self._lock = threading.RLock()

    @property
    def module_count(self) -> int:
        """Number of modules in the graph."""
        with self._lock:
            return len(self._nodes)

    def add_module(
        self,
        name: str,
        file_path: Optional[str] = None,
        imports: Optional[Set[str]] = None,
        is_package: bool = False,
    ) -> ModuleNode:
        """
        Add a module to the graph.

        Args:
            name: Module name.
            file_path: Path to source file.
            imports: Set of imported module names.
            is_package: Whether this is a package __init__.

        Returns:
            The created or updated ModuleNode.
        """
        with self._lock:
            if name not in self._nodes:
                self._nodes[name] = ModuleNode(
                    name=name,
                    file_path=file_path,
                    is_package=is_package,
                )

            node = self._nodes[name]

            if file_path:
                node.file_path = file_path
            if imports:
                old_imports = node.imports.copy()
                node.imports = imports

                # Update reverse edges
                for imp in old_imports - imports:
                    if imp in self._nodes:
                        self._nodes[imp].imported_by.discard(name)

                for imp in imports:
                    if imp not in self._nodes:
                        self._nodes[imp] = ModuleNode(name=imp)
                    self._nodes[imp].imported_by.add(name)

            return node

    def remove_module(self, name: str) -> bool:
        """
        Remove a module from the graph.

        Args:
            name: Module name to remove.

        Returns:
            True if module was removed.
        """
        with self._lock:
            if name not in self._nodes:
                return False

            node = self._nodes[name]

            # Remove edges
            for imp in node.imports:
                if imp in self._nodes:
                    self._nodes[imp].imported_by.discard(name)

            for dep in node.imported_by:
                if dep in self._nodes:
                    self._nodes[dep].imports.discard(name)

            del self._nodes[name]
            return True

    def get_module(self, name: str) -> Optional[ModuleNode]:
        """Get a module node by name."""
        with self._lock:
            return self._nodes.get(name)

    def get_all_modules(self) -> List[ModuleNode]:
        """Get all module nodes."""
        with self._lock:
            return list(self._nodes.values())

    def get_dependents(self, name: str, transitive: bool = True) -> Set[str]:
        """
        Get modules that depend on the given module.

        Args:
            name: Module name.
            transitive: Include transitive dependents.

        Returns:
            Set of dependent module names.
        """
        with self._lock:
            if name not in self._nodes:
                return set()

            if not transitive:
                return self._nodes[name].imported_by.copy()

            # BFS for transitive closure
            result = set()
            queue = list(self._nodes[name].imported_by)

            while queue:
                current = queue.pop(0)
                if current in result:
                    continue
                result.add(current)
                if current in self._nodes:
                    queue.extend(self._nodes[current].imported_by)

            return result

    def get_dependencies(self, name: str, transitive: bool = True) -> Set[str]:
        """
        Get modules that the given module depends on.

        Args:
            name: Module name.
            transitive: Include transitive dependencies.

        Returns:
            Set of dependency module names.
        """
        with self._lock:
            if name not in self._nodes:
                return set()

            if not transitive:
                return self._nodes[name].imports.copy()

            # BFS for transitive closure
            result = set()
            queue = list(self._nodes[name].imports)

            while queue:
                current = queue.pop(0)
                if current in result:
                    continue
                result.add(current)
                if current in self._nodes:
                    queue.extend(self._nodes[current].imports)

            return result

    def get_reload_order(self, modules: Set[str]) -> List[str]:
        """
        Get the order in which modules should be reloaded.

        Modules are ordered so that dependencies are reloaded before
        dependents (topological sort).

        Args:
            modules: Set of module names to reload.

        Returns:
            List of module names in reload order.
        """
        with self._lock:
            # Build subgraph of modules to reload
            in_degree: Dict[str, int] = {}
            edges: Dict[str, Set[str]] = {}

            for name in modules:
                if name not in self._nodes:
                    continue

                in_degree[name] = 0
                edges[name] = set()

                for dep in self._nodes[name].imports:
                    if dep in modules:
                        edges[name].add(dep)

            # Calculate in-degrees
            for name, deps in edges.items():
                for dep in deps:
                    in_degree[name] = in_degree.get(name, 0) + 1

            # Topological sort (Kahn's algorithm)
            result = []
            queue = [n for n in in_degree if in_degree[n] == 0]

            while queue:
                current = queue.pop(0)
                result.append(current)

                for name, deps in edges.items():
                    if current in deps:
                        in_degree[name] -= 1
                        if in_degree[name] == 0:
                            queue.append(name)

            return result

    def detect_cycles(self) -> List[List[str]]:
        """
        Detect circular dependencies in the graph.

        Returns:
            List of cycles (each cycle is a list of module names).
        """
        with self._lock:
            cycles = []
            visited = set()
            rec_stack = set()

            def dfs(name: str, path: List[str]) -> None:
                visited.add(name)
                rec_stack.add(name)
                path = path + [name]

                if name in self._nodes:
                    for dep in self._nodes[name].imports:
                        if dep not in visited:
                            dfs(dep, path)
                        elif dep in rec_stack:
                            # Found cycle
                            cycle_start = path.index(dep)
                            cycle = path[cycle_start:] + [dep]
                            cycles.append(cycle)

                rec_stack.remove(name)

            for name in self._nodes:
                if name not in visited:
                    dfs(name, [])

            return cycles

    def clear(self) -> None:
        """Clear the entire graph."""
        with self._lock:
            self._nodes.clear()


class DependencyTracker:
    """
    Tracks module dependencies for cascade reloads.

    Features:
    - Parse Python files for imports
    - Build and maintain dependency graph
    - Compute cascade reload sets
    - Watch for changes and update graph
    """

    def __init__(self):
        """Initialize the dependency tracker."""
        self._graph = DependencyGraph()
        self._watched_roots: Set[str] = set()
        self._lock = threading.RLock()

    @property
    def graph(self) -> DependencyGraph:
        """Get the dependency graph."""
        return self._graph

    def add_root(self, path: str) -> None:
        """
        Add a root directory to track.

        Args:
            path: Root directory path.
        """
        abs_path = os.path.abspath(path)
        with self._lock:
            self._watched_roots.add(abs_path)

    def analyze_file(self, file_path: str) -> Optional[ModuleNode]:
        """
        Analyze a Python file and add it to the graph.

        Args:
            file_path: Path to Python file.

        Returns:
            ModuleNode or None if analysis failed.
        """
        abs_path = os.path.abspath(file_path)

        if not os.path.exists(abs_path) or not abs_path.endswith(".py"):
            return None

        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            imports = self._extract_imports(tree)

            # Determine module name
            module_name = self._path_to_module(abs_path)
            if not module_name:
                return None

            is_package = abs_path.endswith("__init__.py")

            return self._graph.add_module(
                name=module_name,
                file_path=abs_path,
                imports=imports,
                is_package=is_package,
            )

        except (SyntaxError, IOError):
            return None

    def analyze_directory(self, path: str, recursive: bool = True) -> int:
        """
        Analyze all Python files in a directory.

        Args:
            path: Directory path.
            recursive: Analyze subdirectories.

        Returns:
            Number of modules analyzed.
        """
        abs_path = os.path.abspath(path)
        count = 0

        for root, dirs, files in os.walk(abs_path):
            # Skip __pycache__ and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith((".", "__pycache__"))]

            for filename in files:
                if filename.endswith(".py"):
                    file_path = os.path.join(root, filename)
                    if self.analyze_file(file_path):
                        count += 1

            if not recursive:
                break

        return count

    def get_cascade_modules(self, module_name: str) -> Set[str]:
        """
        Get all modules that need to be reloaded when a module changes.

        This includes the module itself and all transitive dependents.

        Args:
            module_name: Name of changed module.

        Returns:
            Set of module names to reload.
        """
        result = {module_name}
        result.update(self._graph.get_dependents(module_name, transitive=True))
        return result

    def get_reload_plan(self, module_name: str) -> List[str]:
        """
        Get an ordered list of modules to reload.

        Args:
            module_name: Name of changed module.

        Returns:
            List of modules in reload order (dependencies first).
        """
        cascade = self.get_cascade_modules(module_name)
        return self._graph.get_reload_order(cascade)

    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract import names from an AST."""
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

        return imports

    def _path_to_module(self, file_path: str) -> Optional[str]:
        """Convert a file path to a module name."""
        abs_path = os.path.abspath(file_path)

        # Try to find matching root
        for root in self._watched_roots:
            if abs_path.startswith(root):
                rel_path = os.path.relpath(abs_path, root)

                if rel_path.endswith(".py"):
                    rel_path = rel_path[:-3]

                if rel_path.endswith("__init__"):
                    rel_path = rel_path[:-9]

                module_name = rel_path.replace(os.sep, ".")
                return module_name.strip(".")

        # Try sys.path
        for path in sys.path:
            if not path:
                continue
            path = os.path.abspath(path)
            if abs_path.startswith(path):
                rel_path = os.path.relpath(abs_path, path)

                if rel_path.endswith(".py"):
                    rel_path = rel_path[:-3]

                if rel_path.endswith("__init__"):
                    rel_path = rel_path[:-9]

                module_name = rel_path.replace(os.sep, ".")
                return module_name.strip(".")

        return None

    def on_file_changed(self, file_path: str) -> List[str]:
        """
        Handle a file change event.

        Args:
            file_path: Path to changed file.

        Returns:
            List of modules that should be reloaded.
        """
        node = self.analyze_file(file_path)
        if node:
            return self.get_reload_plan(node.name)
        return []

    def clear(self) -> None:
        """Clear all tracking data."""
        with self._lock:
            self._graph.clear()
            self._watched_roots.clear()


__all__ = [
    "ModuleNode",
    "DependencyGraph",
    "DependencyTracker",
]
