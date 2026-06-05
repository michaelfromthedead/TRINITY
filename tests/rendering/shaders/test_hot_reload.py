"""Tests for shader hot-reload with dependency cascade (T-CC-3.2).

Tests cover:
- ShaderDependencyGraph: include parsing, dependency tracking, cascade detection
- PSOHotSwap: PSO registration, queuing, atomic swap execution
- ShaderReloader: file watching, compilation, cascade reload, material binding
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Set
from unittest.mock import MagicMock, Mock, patch

import pytest

from engine.core.file_watcher import FileChangeEvent, FileChangeType
from engine.rendering.materials.shader_compiler import (
    CompiledShader,
    CompilationError,
    PermutationKey,
    PSOCache,
    PSODescriptor,
    ShaderStage,
)
from engine.rendering.shaders.hot_reload import (
    CascadeResult,
    DependencyNode,
    IncludeParseError,
    PSOHotSwap,
    ReloadStats,
    ShaderCompileResult,
    ShaderDependencyGraph,
    ShaderHotReloadEvent,
    ShaderReloader,
    ShaderReloadError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_shader_dir():
    """Create a temporary directory with shader files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_wgsl_shader(temp_shader_dir: str) -> str:
    """Create a sample WGSL shader file."""
    shader_path = os.path.join(temp_shader_dir, "test_shader.wgsl")
    content = """
// Test shader
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"""
    Path(shader_path).write_text(content)
    return shader_path


@pytest.fixture
def sample_glsl_shader(temp_shader_dir: str) -> str:
    """Create a sample GLSL shader file."""
    shader_path = os.path.join(temp_shader_dir, "test_shader.glsl")
    content = """
#version 450

layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"""
    Path(shader_path).write_text(content)
    return shader_path


@pytest.fixture
def shader_with_includes(temp_shader_dir: str) -> tuple[str, str, str]:
    """Create shaders with include dependencies."""
    # Header file
    header_path = os.path.join(temp_shader_dir, "common.wgsl")
    header_content = """
// Common utilities
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}
"""
    Path(header_path).write_text(header_content)

    # Main shader including header
    main_path = os.path.join(temp_shader_dir, "main.wgsl")
    main_content = '''
#include "common.wgsl"

@fragment
fn main() -> @location(0) vec4<f32> {
    let value = lerp(0.0, 1.0, 0.5);
    return vec4<f32>(value, 0.0, 0.0, 1.0);
}
'''
    Path(main_path).write_text(main_content)

    # Another shader also including header
    other_path = os.path.join(temp_shader_dir, "other.wgsl")
    other_content = '''
#include "common.wgsl"

@fragment
fn main() -> @location(0) vec4<f32> {
    let v = lerp(1.0, 0.0, 0.5);
    return vec4<f32>(0.0, v, 0.0, 1.0);
}
'''
    Path(other_path).write_text(other_content)

    return header_path, main_path, other_path


@pytest.fixture
def dependency_graph() -> ShaderDependencyGraph:
    """Create a fresh dependency graph."""
    return ShaderDependencyGraph()


@pytest.fixture
def pso_cache() -> PSOCache:
    """Create a fresh PSO cache."""
    return PSOCache()


@pytest.fixture
def pso_hot_swap(pso_cache: PSOCache) -> PSOHotSwap:
    """Create a PSO hot-swap manager."""
    return PSOHotSwap(pso_cache)


@pytest.fixture
def shader_reloader(pso_cache: PSOCache) -> ShaderReloader:
    """Create a shader reloader."""
    return ShaderReloader(pso_cache=pso_cache)


# =============================================================================
# ShaderDependencyGraph Tests
# =============================================================================


class TestDependencyNode:
    """Tests for DependencyNode."""

    def test_create_node(self):
        """Test creating a dependency node."""
        node = DependencyNode(path="/test/shader.wgsl")
        assert node.path == "/test/shader.wgsl"
        assert len(node.includes) == 0
        assert len(node.included_by) == 0
        assert node.content_hash == ""
        assert node.is_header is False

    def test_add_include(self):
        """Test adding include dependencies."""
        node = DependencyNode(path="/test/shader.wgsl")
        node.add_include("/test/common.wgsl")
        assert "/test/common.wgsl" in node.includes

    def test_add_included_by(self):
        """Test tracking reverse dependencies."""
        node = DependencyNode(path="/test/common.wgsl")
        node.add_included_by("/test/main.wgsl")
        assert "/test/main.wgsl" in node.included_by

    def test_node_as_header(self):
        """Test marking node as header."""
        node = DependencyNode(path="/test/utils.wgsl", is_header=True)
        assert node.is_header is True


class TestShaderDependencyGraph:
    """Tests for ShaderDependencyGraph."""

    def test_create_empty_graph(self, dependency_graph: ShaderDependencyGraph):
        """Test creating an empty graph."""
        assert dependency_graph.node_count == 0
        assert dependency_graph.get_include_dirs() == []

    def test_add_include_dir(self, dependency_graph: ShaderDependencyGraph):
        """Test adding include directories."""
        dependency_graph.add_include_dir("/test/includes")
        dirs = dependency_graph.get_include_dirs()
        assert len(dirs) == 1
        assert "/test/includes" in dirs[0]

    def test_remove_include_dir(self, dependency_graph: ShaderDependencyGraph):
        """Test removing include directories."""
        dependency_graph.add_include_dir("/test/includes")
        result = dependency_graph.remove_include_dir("/test/includes")
        assert result is True
        assert len(dependency_graph.get_include_dirs()) == 0

    def test_register_simple_shader(
        self,
        dependency_graph: ShaderDependencyGraph,
        sample_wgsl_shader: str,
    ):
        """Test registering a simple shader."""
        node = dependency_graph.register_shader(sample_wgsl_shader)
        assert node is not None
        assert node.path == sample_wgsl_shader
        assert node.content_hash != ""
        assert node.last_modified > 0
        assert dependency_graph.node_count == 1

    def test_register_nonexistent_shader_raises(
        self,
        dependency_graph: ShaderDependencyGraph,
    ):
        """Test that registering nonexistent file raises."""
        with pytest.raises(FileNotFoundError):
            dependency_graph.register_shader("/nonexistent/shader.wgsl")

    def test_parse_wgsl_includes(
        self,
        dependency_graph: ShaderDependencyGraph,
        shader_with_includes: tuple[str, str, str],
    ):
        """Test parsing #include directives in WGSL."""
        header_path, main_path, _ = shader_with_includes

        # Register header first
        dependency_graph.register_shader(header_path)
        # Register main shader
        node = dependency_graph.register_shader(main_path)

        assert len(node.includes) == 1
        assert header_path in node.includes

    def test_parse_glsl_quoted_includes(
        self,
        dependency_graph: ShaderDependencyGraph,
        temp_shader_dir: str,
    ):
        """Test parsing GLSL quoted includes."""
        header_path = os.path.join(temp_shader_dir, "utils.glsl")
        Path(header_path).write_text("// Utils\n")

        main_path = os.path.join(temp_shader_dir, "main.glsl")
        Path(main_path).write_text('#include "utils.glsl"\nvoid main() {}\n')

        dependency_graph.register_shader(header_path)
        node = dependency_graph.register_shader(main_path)
        assert header_path in node.includes

    def test_parse_glsl_bracket_includes(
        self,
        dependency_graph: ShaderDependencyGraph,
        temp_shader_dir: str,
    ):
        """Test parsing GLSL angle bracket includes."""
        include_dir = os.path.join(temp_shader_dir, "include")
        os.makedirs(include_dir)
        dependency_graph.add_include_dir(include_dir)

        header_path = os.path.join(include_dir, "system.glsl")
        Path(header_path).write_text("// System include\n")

        main_path = os.path.join(temp_shader_dir, "main.glsl")
        Path(main_path).write_text("#include <system.glsl>\nvoid main() {}\n")

        dependency_graph.register_shader(header_path)
        node = dependency_graph.register_shader(main_path)
        assert header_path in node.includes

    def test_get_dependents(
        self,
        dependency_graph: ShaderDependencyGraph,
        shader_with_includes: tuple[str, str, str],
    ):
        """Test getting direct dependents."""
        header_path, main_path, other_path = shader_with_includes

        dependency_graph.register_shader(header_path)
        dependency_graph.register_shader(main_path)
        dependency_graph.register_shader(other_path)

        dependents = dependency_graph.get_dependents(header_path)
        assert main_path in dependents
        assert other_path in dependents

    def test_get_all_dependents_transitive(
        self,
        dependency_graph: ShaderDependencyGraph,
        temp_shader_dir: str,
    ):
        """Test getting transitive dependents."""
        # Create: base -> mid -> top
        base_path = os.path.join(temp_shader_dir, "base.wgsl")
        Path(base_path).write_text("// Base\n")

        mid_path = os.path.join(temp_shader_dir, "mid.wgsl")
        Path(mid_path).write_text('#include "base.wgsl"\n')

        top_path = os.path.join(temp_shader_dir, "top.wgsl")
        Path(top_path).write_text('#include "mid.wgsl"\nfn main() {}\n')

        dependency_graph.register_shader(base_path)
        dependency_graph.register_shader(mid_path)
        dependency_graph.register_shader(top_path)

        all_deps = dependency_graph.get_all_dependents(base_path)
        assert mid_path in all_deps
        assert top_path in all_deps

    def test_get_affected_shaders(
        self,
        dependency_graph: ShaderDependencyGraph,
        shader_with_includes: tuple[str, str, str],
    ):
        """Test getting all affected shaders when a file changes."""
        header_path, main_path, other_path = shader_with_includes

        dependency_graph.register_shader(header_path)
        dependency_graph.register_shader(main_path)
        dependency_graph.register_shader(other_path)

        affected, depth = dependency_graph.get_affected_shaders(header_path)
        assert header_path in affected
        assert main_path in affected
        assert other_path in affected
        assert depth == 1  # One level of dependents

    def test_cascade_depth_calculation(
        self,
        dependency_graph: ShaderDependencyGraph,
        temp_shader_dir: str,
    ):
        """Test cascade depth is calculated correctly."""
        # Create 3-level dependency: level0 -> level1 -> level2 -> level3
        paths = []
        for i in range(4):
            path = os.path.join(temp_shader_dir, f"level{i}.wgsl")
            if i == 0:
                Path(path).write_text("// Base level\n")
            else:
                Path(path).write_text(f'#include "level{i-1}.wgsl"\nfn main() {{}}\n')
            paths.append(path)

        for path in paths:
            dependency_graph.register_shader(path)

        affected, depth = dependency_graph.get_affected_shaders(paths[0])
        assert len(affected) == 4
        assert depth == 3

    def test_unregister_shader(
        self,
        dependency_graph: ShaderDependencyGraph,
        sample_wgsl_shader: str,
    ):
        """Test unregistering a shader."""
        dependency_graph.register_shader(sample_wgsl_shader)
        assert dependency_graph.node_count == 1

        result = dependency_graph.unregister_shader(sample_wgsl_shader)
        assert result is True
        assert dependency_graph.node_count == 0

    def test_unregister_cleans_dependencies(
        self,
        dependency_graph: ShaderDependencyGraph,
        shader_with_includes: tuple[str, str, str],
    ):
        """Test that unregistering cleans up dependency links."""
        header_path, main_path, _ = shader_with_includes

        dependency_graph.register_shader(header_path)
        dependency_graph.register_shader(main_path)

        dependency_graph.unregister_shader(main_path)
        header_node = dependency_graph.get_node(header_path)
        assert main_path not in header_node.included_by

    def test_has_circular_dependency_false(
        self,
        dependency_graph: ShaderDependencyGraph,
        shader_with_includes: tuple[str, str, str],
    ):
        """Test detecting no circular dependency."""
        header_path, main_path, _ = shader_with_includes
        dependency_graph.register_shader(header_path)
        dependency_graph.register_shader(main_path)

        assert dependency_graph.has_circular_dependency(main_path) is False

    def test_rebuild_graph(
        self,
        dependency_graph: ShaderDependencyGraph,
        shader_with_includes: tuple[str, str, str],
    ):
        """Test rebuilding the entire graph."""
        header_path, main_path, other_path = shader_with_includes

        dependency_graph.register_shader(header_path)
        dependency_graph.register_shader(main_path)
        dependency_graph.register_shader(other_path)

        count = dependency_graph.rebuild_graph()
        assert count == 3

    def test_clear_graph(
        self,
        dependency_graph: ShaderDependencyGraph,
        sample_wgsl_shader: str,
    ):
        """Test clearing the graph."""
        dependency_graph.register_shader(sample_wgsl_shader)
        dependency_graph.clear()
        assert dependency_graph.node_count == 0

    def test_header_detection(
        self,
        dependency_graph: ShaderDependencyGraph,
        temp_shader_dir: str,
    ):
        """Test that files without main are detected as headers."""
        header_path = os.path.join(temp_shader_dir, "utils.wgsl")
        Path(header_path).write_text("fn helper() -> f32 { return 1.0; }\n")

        node = dependency_graph.register_shader(header_path)
        assert node.is_header is True

    def test_non_header_detection(
        self,
        dependency_graph: ShaderDependencyGraph,
        sample_wgsl_shader: str,
    ):
        """Test that files with main are not headers."""
        node = dependency_graph.register_shader(sample_wgsl_shader)
        assert node.is_header is False

    def test_get_node(
        self,
        dependency_graph: ShaderDependencyGraph,
        sample_wgsl_shader: str,
    ):
        """Test getting a node by path."""
        dependency_graph.register_shader(sample_wgsl_shader)
        node = dependency_graph.get_node(sample_wgsl_shader)
        assert node is not None
        assert node.path == sample_wgsl_shader

    def test_get_all_nodes(
        self,
        dependency_graph: ShaderDependencyGraph,
        shader_with_includes: tuple[str, str, str],
    ):
        """Test getting all nodes."""
        header_path, main_path, other_path = shader_with_includes

        dependency_graph.register_shader(header_path)
        dependency_graph.register_shader(main_path)
        dependency_graph.register_shader(other_path)

        nodes = dependency_graph.get_all_nodes()
        assert len(nodes) == 3

    def test_reregister_updates_hash(
        self,
        dependency_graph: ShaderDependencyGraph,
        sample_wgsl_shader: str,
    ):
        """Test that re-registering updates content hash."""
        node1 = dependency_graph.register_shader(sample_wgsl_shader)
        old_hash = node1.content_hash

        # Modify file
        Path(sample_wgsl_shader).write_text("// Modified content\nfn main() {}\n")

        node2 = dependency_graph.register_shader(sample_wgsl_shader)
        assert node2.content_hash != old_hash


# =============================================================================
# PSOHotSwap Tests
# =============================================================================


class TestPSOHotSwap:
    """Tests for PSOHotSwap."""

    def test_create_hot_swap(self, pso_hot_swap: PSOHotSwap):
        """Test creating a PSO hot-swap manager."""
        assert pso_hot_swap.pso_cache is not None
        assert pso_hot_swap.has_pending_swaps() is False

    def test_register_pso(self, pso_hot_swap: PSOHotSwap, temp_shader_dir: str):
        """Test registering a PSO with shaders."""
        shader_path = os.path.join(temp_shader_dir, "test.wgsl")
        Path(shader_path).write_text("fn main() {}\n")

        descriptor = PSODescriptor(vertex_shader_hash="abc", fragment_shader_hash="def")
        mock_pso = Mock()

        pso_hot_swap.register_pso(descriptor, mock_pso, [shader_path])
        assert pso_hot_swap.pso_cache.get(descriptor) == mock_pso

    def test_get_affected_psos(self, pso_hot_swap: PSOHotSwap, temp_shader_dir: str):
        """Test getting PSOs affected by shader change."""
        shader_path = os.path.join(temp_shader_dir, "test.wgsl")
        Path(shader_path).write_text("fn main() {}\n")

        descriptor = PSODescriptor(vertex_shader_hash="abc", fragment_shader_hash="def")
        pso_hash = descriptor.get_hash()

        pso_hot_swap.register_pso(descriptor, Mock(), [shader_path])
        affected = pso_hot_swap.get_affected_psos(shader_path)
        assert pso_hash in affected

    def test_queue_swap(self, pso_hot_swap: PSOHotSwap):
        """Test queuing a PSO swap."""
        new_pso = Mock()
        pso_hot_swap.queue_swap("test_hash", new_pso)
        assert pso_hot_swap.has_pending_swaps() is True
        assert pso_hot_swap.get_pending_count() == 1

    def test_execute_swaps(self, pso_hot_swap: PSOHotSwap):
        """Test executing pending swaps."""
        new_pso = Mock()
        pso_hot_swap.queue_swap("test_hash", new_pso)

        swapped = pso_hot_swap.execute_swaps()
        assert "test_hash" in swapped
        assert pso_hot_swap.has_pending_swaps() is False

    def test_swap_callback(self, pso_hot_swap: PSOHotSwap):
        """Test swap completion callback."""
        callback = Mock()
        pso_hot_swap.add_swap_callback(callback)

        pso_hot_swap.queue_swap("test_hash", Mock())
        pso_hot_swap.execute_swaps()

        callback.assert_called_once()
        args = callback.call_args[0][0]
        assert "test_hash" in args

    def test_remove_swap_callback(self, pso_hot_swap: PSOHotSwap):
        """Test removing swap callback."""
        callback = Mock()
        pso_hot_swap.add_swap_callback(callback)
        result = pso_hot_swap.remove_swap_callback(callback)
        assert result is True

        pso_hot_swap.queue_swap("test_hash", Mock())
        pso_hot_swap.execute_swaps()
        callback.assert_not_called()

    def test_clear_pending(self, pso_hot_swap: PSOHotSwap):
        """Test clearing pending swaps."""
        pso_hot_swap.queue_swap("hash1", Mock())
        pso_hot_swap.queue_swap("hash2", Mock())

        count = pso_hot_swap.clear_pending()
        assert count == 2
        assert pso_hot_swap.has_pending_swaps() is False

    def test_unregister_pso(self, pso_hot_swap: PSOHotSwap, temp_shader_dir: str):
        """Test unregistering a PSO."""
        shader_path = os.path.join(temp_shader_dir, "test.wgsl")
        Path(shader_path).write_text("fn main() {}\n")

        descriptor = PSODescriptor(vertex_shader_hash="abc", fragment_shader_hash="def")
        pso_hot_swap.register_pso(descriptor, Mock(), [shader_path])
        pso_hot_swap.unregister_pso(descriptor)

        assert pso_hot_swap.pso_cache.get(descriptor) is None

    def test_clear(self, pso_hot_swap: PSOHotSwap, temp_shader_dir: str):
        """Test clearing all state."""
        shader_path = os.path.join(temp_shader_dir, "test.wgsl")
        Path(shader_path).write_text("fn main() {}\n")

        descriptor = PSODescriptor(vertex_shader_hash="abc", fragment_shader_hash="def")
        pso_hot_swap.register_pso(descriptor, Mock(), [shader_path])
        pso_hot_swap.queue_swap("test_hash", Mock())

        pso_hot_swap.clear()
        assert pso_hot_swap.has_pending_swaps() is False
        assert len(pso_hot_swap.get_affected_psos(shader_path)) == 0


# =============================================================================
# ShaderCompileResult Tests
# =============================================================================


class TestShaderCompileResult:
    """Tests for ShaderCompileResult."""

    def test_successful_result(self):
        """Test creating successful compile result."""
        compiled = CompiledShader(
            source_hash="abc",
            bytecode=b"\x00",
            stage=ShaderStage.FRAGMENT,
            entry_point="main",
            permutation_key=PermutationKey.empty(),
        )
        result = ShaderCompileResult(
            path="/test.wgsl",
            success=True,
            compiled_shader=compiled,
            compile_time_ms=10.5,
        )
        assert result.success is True
        assert result.compiled_shader is not None
        assert result.error is None

    def test_failed_result(self):
        """Test creating failed compile result."""
        result = ShaderCompileResult(
            path="/test.wgsl",
            success=False,
            error="Syntax error at line 5",
            compile_time_ms=2.0,
        )
        assert result.success is False
        assert result.compiled_shader is None
        assert result.error is not None


# =============================================================================
# CascadeResult Tests
# =============================================================================


class TestCascadeResult:
    """Tests for CascadeResult."""

    def test_all_success(self):
        """Test cascade with all successful compilations."""
        result = CascadeResult(root_path="/root.wgsl")
        result.compile_results = [
            ShaderCompileResult(path="/a.wgsl", success=True),
            ShaderCompileResult(path="/b.wgsl", success=True),
        ]
        assert result.success is True
        assert result.failed_count == 0

    def test_partial_failure(self):
        """Test cascade with some failures."""
        result = CascadeResult(root_path="/root.wgsl")
        result.compile_results = [
            ShaderCompileResult(path="/a.wgsl", success=True),
            ShaderCompileResult(path="/b.wgsl", success=False, error="Error"),
        ]
        assert result.success is False
        assert result.failed_count == 1

    def test_empty_cascade(self):
        """Test empty cascade is successful."""
        result = CascadeResult(root_path="/root.wgsl")
        assert result.success is True
        assert result.failed_count == 0


# =============================================================================
# ReloadStats Tests
# =============================================================================


class TestReloadStats:
    """Tests for ReloadStats."""

    def test_initial_stats(self):
        """Test initial statistics values."""
        stats = ReloadStats()
        assert stats.reloads_triggered == 0
        assert stats.shaders_recompiled == 0
        assert stats.cascade_recompiles == 0
        assert stats.pso_swaps == 0
        assert stats.failed_reloads == 0

    def test_record_cascade(self):
        """Test recording cascade statistics."""
        stats = ReloadStats()
        result = CascadeResult(
            root_path="/test.wgsl",
            affected_paths=["/test.wgsl", "/dep.wgsl"],
            cascade_depth=1,
            total_compile_time_ms=50.0,
        )
        result.compile_results = [
            ShaderCompileResult(path="/test.wgsl", success=True),
            ShaderCompileResult(path="/dep.wgsl", success=True),
        ]

        stats.record_cascade(result)
        assert stats.reloads_triggered == 1
        assert stats.shaders_recompiled == 2
        assert stats.cascade_recompiles == 1
        assert stats.total_compile_time_ms == 50.0
        assert stats.failed_reloads == 0

    def test_record_failed_cascade(self):
        """Test recording failed cascade."""
        stats = ReloadStats()
        result = CascadeResult(root_path="/test.wgsl")
        result.compile_results = [
            ShaderCompileResult(path="/test.wgsl", success=False, error="Error"),
        ]

        stats.record_cascade(result)
        assert stats.failed_reloads == 1

    def test_average_cascade_depth(self):
        """Test running average of cascade depth."""
        stats = ReloadStats()

        for depth in [1, 2, 3]:
            result = CascadeResult(root_path="/test.wgsl", cascade_depth=depth)
            result.compile_results = [
                ShaderCompileResult(path="/test.wgsl", success=True)
            ]
            stats.record_cascade(result)

        assert stats.average_cascade_depth == 2.0


# =============================================================================
# ShaderHotReloadEvent Tests
# =============================================================================


class TestShaderHotReloadEvent:
    """Tests for ShaderHotReloadEvent."""

    def test_successful_event(self):
        """Test creating successful reload event."""
        result = CascadeResult(root_path="/test.wgsl")
        result.compile_results = [
            ShaderCompileResult(path="/test.wgsl", success=True)
        ]
        event = ShaderHotReloadEvent(
            source_path="/test.wgsl",
            change_type=FileChangeType.MODIFIED,
            cascade_result=result,
        )
        assert event.success is True
        assert event.timestamp > 0

    def test_event_with_affected_materials(self):
        """Test event with affected materials."""
        result = CascadeResult(root_path="/test.wgsl")
        event = ShaderHotReloadEvent(
            source_path="/test.wgsl",
            change_type=FileChangeType.MODIFIED,
            cascade_result=result,
            affected_materials=["mat_001", "mat_002"],
        )
        assert len(event.affected_materials) == 2


# =============================================================================
# ShaderReloader Tests
# =============================================================================


class TestShaderReloader:
    """Tests for ShaderReloader."""

    def test_create_reloader(self, shader_reloader: ShaderReloader):
        """Test creating a shader reloader."""
        assert shader_reloader.dependency_graph is not None
        assert shader_reloader.pso_swap is not None
        assert shader_reloader.is_running is False

    def test_watch_directory(
        self,
        shader_reloader: ShaderReloader,
        temp_shader_dir: str,
    ):
        """Test watching a directory."""
        result = shader_reloader.watch_directory(temp_shader_dir)
        assert result is True

    def test_watch_nonexistent_directory(self, shader_reloader: ShaderReloader):
        """Test watching nonexistent directory fails."""
        result = shader_reloader.watch_directory("/nonexistent/dir")
        assert result is False

    def test_unwatch_directory(
        self,
        shader_reloader: ShaderReloader,
        temp_shader_dir: str,
    ):
        """Test unwatching a directory."""
        shader_reloader.watch_directory(temp_shader_dir)
        result = shader_reloader.unwatch_directory(temp_shader_dir)
        assert result is True

    def test_register_shader(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test manually registering a shader."""
        node = shader_reloader.register_shader(sample_wgsl_shader)
        assert node is not None
        assert node.path == sample_wgsl_shader

    def test_register_unsupported_extension(
        self,
        shader_reloader: ShaderReloader,
        temp_shader_dir: str,
    ):
        """Test registering unsupported extension returns None."""
        bad_path = os.path.join(temp_shader_dir, "test.txt")
        Path(bad_path).write_text("not a shader\n")

        node = shader_reloader.register_shader(bad_path)
        assert node is None

    def test_compile_shader(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test compiling a shader."""
        result = shader_reloader.compile_shader(sample_wgsl_shader)
        assert result.success is True
        assert result.compiled_shader is not None
        assert result.compile_time_ms > 0

    def test_compile_nonexistent_shader(self, shader_reloader: ShaderReloader):
        """Test compiling nonexistent shader fails."""
        result = shader_reloader.compile_shader("/nonexistent.wgsl")
        assert result.success is False
        assert result.error is not None

    def test_reload_shader(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test reloading a shader."""
        shader_reloader.register_shader(sample_wgsl_shader)
        result = shader_reloader.reload_shader(sample_wgsl_shader)

        assert result.success is True
        assert sample_wgsl_shader in result.affected_paths

    def test_reload_with_cascade(
        self,
        shader_reloader: ShaderReloader,
        shader_with_includes: tuple[str, str, str],
    ):
        """Test reloading triggers cascade to dependents."""
        header_path, main_path, other_path = shader_with_includes

        shader_reloader.register_shader(header_path)
        shader_reloader.register_shader(main_path)
        shader_reloader.register_shader(other_path)

        result = shader_reloader.reload_shader(header_path)

        assert result.success is True
        assert header_path in result.affected_paths
        assert main_path in result.affected_paths
        assert other_path in result.affected_paths
        assert result.cascade_depth == 1

    def test_bind_material(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test binding a material to a shader."""
        shader_reloader.register_shader(sample_wgsl_shader)
        shader_reloader.bind_material(sample_wgsl_shader, "mat_001")

        materials = shader_reloader.get_bound_materials(sample_wgsl_shader)
        assert "mat_001" in materials

    def test_unbind_material(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test unbinding a material."""
        shader_reloader.bind_material(sample_wgsl_shader, "mat_001")
        result = shader_reloader.unbind_material(sample_wgsl_shader, "mat_001")

        assert result is True
        assert "mat_001" not in shader_reloader.get_bound_materials(sample_wgsl_shader)

    def test_get_compiled_shader(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test getting compiled shader."""
        shader_reloader.compile_shader(sample_wgsl_shader)
        compiled = shader_reloader.get_compiled_shader(sample_wgsl_shader)
        assert compiled is not None

    def test_get_compiled_shader_not_found(self, shader_reloader: ShaderReloader):
        """Test getting nonexistent compiled shader."""
        compiled = shader_reloader.get_compiled_shader("/nonexistent.wgsl")
        assert compiled is None

    def test_add_reload_callback(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test adding reload callback."""
        callback = Mock()
        shader_reloader.add_reload_callback(callback)
        shader_reloader.register_shader(sample_wgsl_shader)

        # Trigger reload
        shader_reloader._pending_reloads.append(
            (sample_wgsl_shader, FileChangeType.MODIFIED)
        )
        shader_reloader.process_pending()

        callback.assert_called_once()

    def test_remove_reload_callback(self, shader_reloader: ShaderReloader):
        """Test removing reload callback."""
        callback = Mock()
        shader_reloader.add_reload_callback(callback)
        result = shader_reloader.remove_reload_callback(callback)
        assert result is True

    def test_custom_compile_function(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test setting custom compile function."""
        custom_compiled = CompiledShader(
            source_hash="custom",
            bytecode=b"custom",
            stage=ShaderStage.FRAGMENT,
            entry_point="main",
            permutation_key=PermutationKey.empty(),
        )

        def custom_compile(path, perm):
            return custom_compiled

        shader_reloader.set_compile_function(custom_compile)
        result = shader_reloader.compile_shader(sample_wgsl_shader)

        assert result.compiled_shader.source_hash == "custom"

    def test_compile_with_permutation(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test compiling with permutation key."""
        perm = PermutationKey.from_list(["FEATURE_A", "FEATURE_B"])
        result = shader_reloader.compile_shader(sample_wgsl_shader, perm)

        assert result.success is True
        assert result.permutation_key == perm

    def test_start_stop(self, shader_reloader: ShaderReloader):
        """Test starting and stopping the reloader."""
        shader_reloader.start()
        assert shader_reloader.is_running is True

        shader_reloader.stop()
        assert shader_reloader.is_running is False

    def test_start_idempotent(self, shader_reloader: ShaderReloader):
        """Test that starting twice is safe."""
        shader_reloader.start()
        shader_reloader.start()  # Should not raise
        assert shader_reloader.is_running is True
        shader_reloader.stop()

    def test_execute_pso_swaps(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test executing PSO swaps."""
        shader_reloader.pso_swap.queue_swap("test_hash", Mock())
        count = shader_reloader.execute_pso_swaps()
        assert count == 1
        assert shader_reloader.stats.pso_swaps == 1

    def test_get_stats(self, shader_reloader: ShaderReloader):
        """Test getting statistics."""
        stats = shader_reloader.get_stats()
        assert "reloads_triggered" in stats
        assert "tracked_shaders" in stats
        assert "pending_pso_swaps" in stats

    def test_dispose(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test disposing resources."""
        shader_reloader.start()
        shader_reloader.register_shader(sample_wgsl_shader)
        shader_reloader.dispose()

        assert shader_reloader.is_running is False
        assert shader_reloader.dependency_graph.node_count == 0

    def test_auto_register_on_watch(
        self,
        shader_reloader: ShaderReloader,
        shader_with_includes: tuple[str, str, str],
    ):
        """Test that watching directory auto-registers shaders."""
        header_path, main_path, other_path = shader_with_includes
        shader_dir = os.path.dirname(header_path)

        shader_reloader.watch_directory(shader_dir)

        # All shaders should be registered
        assert shader_reloader.dependency_graph.get_node(header_path) is not None
        assert shader_reloader.dependency_graph.get_node(main_path) is not None
        assert shader_reloader.dependency_graph.get_node(other_path) is not None

    def test_file_change_queues_reload(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test that file changes queue reloads."""
        shader_reloader.register_shader(sample_wgsl_shader)

        # Simulate file change event
        event = FileChangeEvent(
            path=Path(sample_wgsl_shader),
            change_type=FileChangeType.MODIFIED,
        )
        shader_reloader._on_file_change(event)

        # Should have pending reload
        assert len(shader_reloader._pending_reloads) == 1

    def test_deleted_file_unregisters(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test that deleted files are unregistered."""
        shader_reloader.register_shader(sample_wgsl_shader)

        # Queue deletion
        shader_reloader._pending_reloads.append(
            (sample_wgsl_shader, FileChangeType.DELETED)
        )
        shader_reloader.process_pending()

        assert shader_reloader.dependency_graph.get_node(sample_wgsl_shader) is None

    def test_stats_updated_on_reload(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test that statistics are updated on reload."""
        shader_reloader.register_shader(sample_wgsl_shader)
        shader_reloader.reload_shader(sample_wgsl_shader)

        assert shader_reloader.stats.reloads_triggered == 1
        assert shader_reloader.stats.shaders_recompiled >= 1


class TestShaderReloaderIntegration:
    """Integration tests for shader hot-reload."""

    def test_full_reload_cycle(
        self,
        shader_reloader: ShaderReloader,
        shader_with_includes: tuple[str, str, str],
    ):
        """Test complete reload cycle with cascade."""
        header_path, main_path, other_path = shader_with_includes
        shader_dir = os.path.dirname(header_path)

        # Setup
        shader_reloader.watch_directory(shader_dir)
        shader_reloader.bind_material(main_path, "mat_main")
        shader_reloader.bind_material(other_path, "mat_other")

        # Track events
        events: List[ShaderHotReloadEvent] = []
        shader_reloader.add_reload_callback(lambda e: events.append(e))

        # Trigger reload of header (should cascade)
        result = shader_reloader.reload_shader(header_path)

        assert result.success is True
        assert len(result.affected_paths) == 3
        assert result.cascade_depth == 1

    def test_pso_update_after_reload(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test PSO updates are queued after reload."""
        shader_reloader.register_shader(sample_wgsl_shader)

        # Register a PSO using this shader
        descriptor = PSODescriptor(vertex_shader_hash="v", fragment_shader_hash="f")
        shader_reloader.pso_swap.register_pso(
            descriptor, Mock(), [sample_wgsl_shader]
        )

        # Reload
        shader_reloader.reload_shader(sample_wgsl_shader)

        # PSO swap should be queued
        assert shader_reloader.pso_swap.has_pending_swaps() is True

    def test_material_notification(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test materials receive reload notifications."""

        class MockMaterial:
            def __init__(self):
                self.reloaded = False
                self.shader_path = None

            def on_shader_reloaded(self, path: str, shader: CompiledShader):
                self.reloaded = True
                self.shader_path = path

        material = MockMaterial()
        shader_reloader.register_shader(sample_wgsl_shader)
        shader_reloader.bind_material(sample_wgsl_shader, "mat_001", material)

        # Trigger reload through pending queue
        shader_reloader._pending_reloads.append(
            (sample_wgsl_shader, FileChangeType.MODIFIED)
        )
        shader_reloader.process_pending()

        assert material.reloaded is True
        assert material.shader_path == os.path.abspath(sample_wgsl_shader)

    def test_deep_cascade(self, shader_reloader: ShaderReloader, temp_shader_dir: str):
        """Test cascade through multiple dependency levels."""
        # Create: L0 -> L1 -> L2 -> L3
        paths = []
        for i in range(4):
            path = os.path.join(temp_shader_dir, f"level{i}.wgsl")
            if i == 0:
                Path(path).write_text("// Base\n")
            else:
                Path(path).write_text(f'#include "level{i-1}.wgsl"\nfn main() {{}}\n')
            paths.append(path)

        shader_reloader.watch_directory(temp_shader_dir)

        # Reload base
        result = shader_reloader.reload_shader(paths[0])

        assert result.success is True
        assert len(result.affected_paths) == 4
        assert result.cascade_depth == 3

    def test_concurrent_reloads(
        self,
        shader_reloader: ShaderReloader,
        temp_shader_dir: str,
    ):
        """Test handling concurrent reload requests."""
        # Create multiple independent shaders
        shaders = []
        for i in range(5):
            path = os.path.join(temp_shader_dir, f"shader{i}.wgsl")
            Path(path).write_text(f"fn main{i}() {{}}\n")
            shaders.append(path)

        for path in shaders:
            shader_reloader.register_shader(path)

        # Queue all for reload
        for path in shaders:
            shader_reloader._pending_reloads.append(
                (path, FileChangeType.MODIFIED)
            )

        results = shader_reloader.process_pending()
        assert len(results) == 5
        assert all(r.success for r in results)


class TestErrorHandling:
    """Tests for error handling."""

    def test_compile_error_handling(
        self,
        shader_reloader: ShaderReloader,
        temp_shader_dir: str,
    ):
        """Test graceful handling of compilation errors."""

        def failing_compile(path, perm):
            raise CompilationError("Syntax error at line 5")

        shader_reloader.set_compile_function(failing_compile)

        shader_path = os.path.join(temp_shader_dir, "bad.wgsl")
        Path(shader_path).write_text("bad syntax\n")

        result = shader_reloader.compile_shader(shader_path)
        assert result.success is False
        assert "Syntax error" in result.error

    def test_cascade_partial_failure(
        self,
        shader_reloader: ShaderReloader,
        shader_with_includes: tuple[str, str, str],
    ):
        """Test cascade continues even if some compilations fail."""
        header_path, main_path, other_path = shader_with_includes

        compile_count = [0]
        original_compile = shader_reloader._default_compile

        def sometimes_fail(path, perm):
            compile_count[0] += 1
            if "main" in path:
                raise CompilationError("Failed for main")
            return original_compile(path, perm)

        shader_reloader.set_compile_function(sometimes_fail)
        shader_reloader.watch_directory(os.path.dirname(header_path))

        result = shader_reloader.reload_shader(header_path)

        # Should have attempted all 3
        assert compile_count[0] == 3
        # But one failed
        assert result.failed_count == 1

    def test_callback_exception_isolation(
        self,
        shader_reloader: ShaderReloader,
        sample_wgsl_shader: str,
    ):
        """Test that callback exceptions don't break reload."""

        def bad_callback(event):
            raise RuntimeError("Callback error")

        def good_callback(event):
            event.success  # Just access, should work

        shader_reloader.add_reload_callback(bad_callback)
        shader_reloader.add_reload_callback(good_callback)
        shader_reloader.register_shader(sample_wgsl_shader)

        # Should not raise despite bad callback
        shader_reloader._pending_reloads.append(
            (sample_wgsl_shader, FileChangeType.MODIFIED)
        )
        shader_reloader.process_pending()


class TestReloadError:
    """Tests for ShaderReloadError."""

    def test_create_error(self):
        """Test creating a reload error."""
        error = ShaderReloadError("/test.wgsl", "Compilation failed")
        assert error.shader_path == "/test.wgsl"
        assert error.message == "Compilation failed"
        assert "test.wgsl" in str(error)

    def test_error_with_cause(self):
        """Test error with underlying cause."""
        cause = ValueError("Root cause")
        error = ShaderReloadError("/test.wgsl", "Wrapper", cause)
        assert error.cause == cause


class TestIncludeParseError:
    """Tests for IncludeParseError."""

    def test_create_error(self):
        """Test creating include parse error."""
        error = IncludeParseError("/test.wgsl", 10, "Invalid include")
        assert error.path == "/test.wgsl"
        assert error.line == 10
        assert "/test.wgsl:10" in str(error)
