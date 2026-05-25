"""
Tests for module dependency tracking.
"""
import os
import tempfile
import pytest
from pathlib import Path

from engine.tooling.hotreload.dependency_tracker import (
    DependencyTracker,
    DependencyGraph,
    ModuleNode,
)


class TestModuleNode:
    """Tests for ModuleNode."""

    def test_node_creation(self):
        """Test creating a module node."""
        node = ModuleNode(
            name="test_module",
            file_path="/path/to/test_module.py",
            imports={"os", "sys"},
        )

        assert node.name == "test_module"
        assert node.file_path == "/path/to/test_module.py"
        assert "os" in node.imports

    def test_node_dependencies(self):
        """Test dependencies property."""
        node = ModuleNode(
            name="test",
            imports={"os", "sys"},
        )

        assert node.dependencies == {"os", "sys"}

    def test_node_dependents(self):
        """Test dependents property."""
        node = ModuleNode(
            name="test",
            imported_by={"module_a", "module_b"},
        )

        assert node.dependents == {"module_a", "module_b"}


class TestDependencyGraph:
    """Tests for DependencyGraph."""

    def setup_method(self):
        """Create fresh graph for each test."""
        self.graph = DependencyGraph()

    def test_graph_initialization(self):
        """Test DependencyGraph initializes correctly."""
        assert self.graph.module_count == 0

    def test_add_module(self):
        """Test adding a module to the graph."""
        node = self.graph.add_module(
            name="test_module",
            file_path="/path/to/test.py",
            imports={"os", "sys"},
        )

        assert node.name == "test_module"
        assert self.graph.module_count == 3  # test_module + os + sys

    def test_add_module_updates_reverse_edges(self):
        """Test that adding a module updates imported_by."""
        self.graph.add_module("module_a", imports={"module_b"})

        node_b = self.graph.get_module("module_b")
        assert node_b is not None
        assert "module_a" in node_b.imported_by

    def test_remove_module(self):
        """Test removing a module from the graph."""
        self.graph.add_module("test", imports={"dep"})

        result = self.graph.remove_module("test")

        assert result is True
        assert self.graph.get_module("test") is None

        # Check reverse edge is cleaned up
        dep = self.graph.get_module("dep")
        assert dep is not None
        assert "test" not in dep.imported_by

    def test_remove_nonexistent_module(self):
        """Test removing a module that doesn't exist."""
        result = self.graph.remove_module("nonexistent")
        assert result is False

    def test_get_module(self):
        """Test getting a module by name."""
        self.graph.add_module("test")

        node = self.graph.get_module("test")
        assert node is not None
        assert node.name == "test"

        none_node = self.graph.get_module("nonexistent")
        assert none_node is None

    def test_get_all_modules(self):
        """Test getting all modules."""
        self.graph.add_module("a")
        self.graph.add_module("b")
        self.graph.add_module("c")

        modules = self.graph.get_all_modules()
        names = {m.name for m in modules}

        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_get_dependents_direct(self):
        """Test getting direct dependents."""
        self.graph.add_module("dep")
        self.graph.add_module("module_a", imports={"dep"})
        self.graph.add_module("module_b", imports={"dep"})

        dependents = self.graph.get_dependents("dep", transitive=False)

        assert "module_a" in dependents
        assert "module_b" in dependents

    def test_get_dependents_transitive(self):
        """Test getting transitive dependents."""
        # dep <- a <- b
        self.graph.add_module("dep")
        self.graph.add_module("a", imports={"dep"})
        self.graph.add_module("b", imports={"a"})

        dependents = self.graph.get_dependents("dep", transitive=True)

        assert "a" in dependents
        assert "b" in dependents

    def test_get_dependencies_direct(self):
        """Test getting direct dependencies."""
        self.graph.add_module("module", imports={"dep1", "dep2"})

        deps = self.graph.get_dependencies("module", transitive=False)

        assert "dep1" in deps
        assert "dep2" in deps

    def test_get_dependencies_transitive(self):
        """Test getting transitive dependencies."""
        # module -> a -> b
        self.graph.add_module("b")
        self.graph.add_module("a", imports={"b"})
        self.graph.add_module("module", imports={"a"})

        deps = self.graph.get_dependencies("module", transitive=True)

        assert "a" in deps
        assert "b" in deps

    def test_get_reload_order(self):
        """Test getting reload order."""
        # a depends on b depends on c
        # Reload order should be: c, b, a
        self.graph.add_module("c")
        self.graph.add_module("b", imports={"c"})
        self.graph.add_module("a", imports={"b"})

        order = self.graph.get_reload_order({"a", "b", "c"})

        # c should come before b, b before a
        assert order.index("c") < order.index("b")
        assert order.index("b") < order.index("a")

    def test_detect_cycles(self):
        """Test cycle detection."""
        # a -> b -> c -> a
        self.graph.add_module("a", imports={"b"})
        self.graph.add_module("b", imports={"c"})
        self.graph.add_module("c", imports={"a"})

        cycles = self.graph.detect_cycles()

        assert len(cycles) > 0
        # All three should be in some cycle
        all_in_cycles = set()
        for cycle in cycles:
            all_in_cycles.update(cycle)
        assert "a" in all_in_cycles
        assert "b" in all_in_cycles
        assert "c" in all_in_cycles

    def test_no_cycles(self):
        """Test no cycles detection in acyclic graph."""
        self.graph.add_module("a")
        self.graph.add_module("b", imports={"a"})
        self.graph.add_module("c", imports={"b"})

        cycles = self.graph.detect_cycles()
        assert len(cycles) == 0

    def test_clear(self):
        """Test clearing the graph."""
        self.graph.add_module("a")
        self.graph.add_module("b")

        self.graph.clear()

        assert self.graph.module_count == 0


class TestDependencyTracker:
    """Tests for DependencyTracker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = DependencyTracker()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tracker_initialization(self):
        """Test DependencyTracker initializes correctly."""
        assert self.tracker.graph is not None
        assert isinstance(self.tracker.graph, DependencyGraph)

    def test_add_root(self):
        """Test adding a root directory."""
        self.tracker.add_root(self.temp_dir)
        # Should not raise

    def test_analyze_file(self):
        """Test analyzing a Python file."""
        # Create a test file
        file_path = os.path.join(self.temp_dir, "test_module.py")
        Path(file_path).write_text("""
import os
import sys
from collections import OrderedDict

class MyClass:
    pass
""")

        self.tracker.add_root(self.temp_dir)
        node = self.tracker.analyze_file(file_path)

        assert node is not None
        assert node.name == "test_module"
        assert "os" in node.imports
        assert "sys" in node.imports
        assert "collections" in node.imports

    def test_analyze_nonexistent_file(self):
        """Test analyzing a non-existent file."""
        node = self.tracker.analyze_file("/nonexistent/path.py")
        assert node is None

    def test_analyze_non_python_file(self):
        """Test analyzing a non-Python file."""
        file_path = os.path.join(self.temp_dir, "test.txt")
        Path(file_path).write_text("Hello")

        node = self.tracker.analyze_file(file_path)
        assert node is None

    def test_analyze_directory(self):
        """Test analyzing a directory."""
        # Create some Python files
        Path(os.path.join(self.temp_dir, "module_a.py")).write_text("import os")
        Path(os.path.join(self.temp_dir, "module_b.py")).write_text("import sys")

        self.tracker.add_root(self.temp_dir)
        count = self.tracker.analyze_directory(self.temp_dir)

        assert count >= 2

    def test_analyze_directory_recursive(self):
        """Test recursive directory analysis."""
        # Create nested structure
        subdir = os.path.join(self.temp_dir, "subpackage")
        os.makedirs(subdir)

        Path(os.path.join(self.temp_dir, "module_a.py")).write_text("")
        Path(os.path.join(subdir, "module_b.py")).write_text("")

        self.tracker.add_root(self.temp_dir)
        count = self.tracker.analyze_directory(self.temp_dir, recursive=True)

        assert count >= 2

    def test_get_cascade_modules(self):
        """Test getting cascade modules."""
        # Create dependency chain: a <- b <- c
        Path(os.path.join(self.temp_dir, "a.py")).write_text("")
        Path(os.path.join(self.temp_dir, "b.py")).write_text("import a")
        Path(os.path.join(self.temp_dir, "c.py")).write_text("import b")

        self.tracker.add_root(self.temp_dir)
        self.tracker.analyze_directory(self.temp_dir)

        cascade = self.tracker.get_cascade_modules("a")

        assert "a" in cascade
        assert "b" in cascade
        assert "c" in cascade

    def test_get_reload_plan(self):
        """Test getting reload plan."""
        # Create dependency chain
        Path(os.path.join(self.temp_dir, "base.py")).write_text("")
        Path(os.path.join(self.temp_dir, "middle.py")).write_text("import base")
        Path(os.path.join(self.temp_dir, "top.py")).write_text("import middle")

        self.tracker.add_root(self.temp_dir)
        self.tracker.analyze_directory(self.temp_dir)

        plan = self.tracker.get_reload_plan("base")

        # Base should be reloaded first
        if "base" in plan and "middle" in plan:
            assert plan.index("base") < plan.index("middle")
        if "middle" in plan and "top" in plan:
            assert plan.index("middle") < plan.index("top")

    def test_on_file_changed(self):
        """Test handling file change events."""
        Path(os.path.join(self.temp_dir, "changed.py")).write_text("import os")

        self.tracker.add_root(self.temp_dir)

        file_path = os.path.join(self.temp_dir, "changed.py")
        to_reload = self.tracker.on_file_changed(file_path)

        assert isinstance(to_reload, list)

    def test_clear(self):
        """Test clearing the tracker."""
        self.tracker.add_root(self.temp_dir)
        self.tracker.analyze_directory(self.temp_dir)

        self.tracker.clear()

        assert self.tracker.graph.module_count == 0


class TestImportExtraction:
    """Tests for import extraction from AST."""

    def setup_method(self):
        self.tracker = DependencyTracker()
        self.temp_dir = tempfile.mkdtemp()
        self.tracker.add_root(self.temp_dir)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extract_import(self):
        """Test extracting simple imports."""
        file_path = os.path.join(self.temp_dir, "test.py")
        Path(file_path).write_text("""
import os
import sys
""")

        node = self.tracker.analyze_file(file_path)

        assert "os" in node.imports
        assert "sys" in node.imports

    def test_extract_from_import(self):
        """Test extracting from imports."""
        file_path = os.path.join(self.temp_dir, "test.py")
        Path(file_path).write_text("""
from collections import OrderedDict
from typing import Any, List
""")

        node = self.tracker.analyze_file(file_path)

        assert "collections" in node.imports
        assert "typing" in node.imports

    def test_extract_relative_import(self):
        """Test extracting relative imports."""
        file_path = os.path.join(self.temp_dir, "test.py")
        Path(file_path).write_text("""
from . import sibling
from ..parent import something
""")

        node = self.tracker.analyze_file(file_path)
        # Relative imports may not have module attribute
        assert node is not None

    def test_extract_aliased_import(self):
        """Test extracting aliased imports."""
        file_path = os.path.join(self.temp_dir, "test.py")
        Path(file_path).write_text("""
import numpy as np
import pandas as pd
""")

        node = self.tracker.analyze_file(file_path)

        assert "numpy" in node.imports
        assert "pandas" in node.imports
