"""Tests for the WGSL #include directive preprocessor (T-MAT-2.5).

Verifies:
- Basic include resolution (relative and project includes)
- Nested/recursive includes
- Cycle detection
- Maximum depth enforcement
- Multiple search paths
- Dependency graph tracking
- #pragma once support
"""

from __future__ import annotations

import pytest
from pathlib import Path
from textwrap import dedent

from trinity.materials.includes import (
    IncludeResolver,
    IncludeError,
    CyclicIncludeError,
    MaxDepthError,
    IncludeFileNotFoundError,
    IncludeDirective,
    DepGraph,
    preprocess_wgsl,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_shader_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with shader files for testing."""
    # Create directory structure
    (tmp_path / "shaders").mkdir()
    (tmp_path / "shaders" / "common").mkdir()
    (tmp_path / "shaders" / "pbr").mkdir()
    (tmp_path / "materials").mkdir()

    # Create common shader files
    (tmp_path / "shaders" / "common" / "math.wgsl").write_text(dedent("""\
        // Common math utilities
        fn saturate(x: f32) -> f32 {
            return clamp(x, 0.0, 1.0);
        }
    """))

    (tmp_path / "shaders" / "common" / "color.wgsl").write_text(dedent("""\
        // Color utilities
        #include "math.wgsl"

        fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
            return pow(c, vec3<f32>(1.0 / 2.2));
        }
    """))

    (tmp_path / "shaders" / "pbr" / "brdf.wgsl").write_text(dedent("""\
        // PBR BRDF functions
        #include <common/math.wgsl>

        fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
            return f0 + (vec3<f32>(1.0) - f0) * pow(saturate(1.0 - cos_theta), 5.0);
        }
    """))

    # Create material file that includes from multiple sources
    (tmp_path / "materials" / "gold.wgsl").write_text(dedent("""\
        // Gold material shader
        #include <pbr/brdf.wgsl>
        #include <common/color.wgsl>

        fn gold_surface() -> vec3<f32> {
            let base_color = vec3<f32>(1.0, 0.766, 0.336);
            return linear_to_srgb(base_color);
        }
    """))

    return tmp_path


@pytest.fixture
def resolver(temp_shader_dir: Path) -> IncludeResolver:
    """Create an IncludeResolver configured with test search paths."""
    return IncludeResolver(
        search_paths=[
            temp_shader_dir / "shaders",
        ],
        max_depth=10
    )


# =============================================================================
# Suite A: Basic Include Resolution
# =============================================================================


class TestBasicIncludeResolution:
    """Tests for basic include directive parsing and resolution."""

    def test_parse_relative_include(self):
        """Parse relative include directive with double quotes."""
        resolver = IncludeResolver()
        source = '#include "brdf.wgsl"'
        directives = resolver.parse_includes(source)

        assert len(directives) == 1
        assert directives[0].path == "brdf.wgsl"
        assert directives[0].is_relative is True
        assert directives[0].line_number == 1

    def test_parse_project_include(self):
        """Parse project include directive with angle brackets."""
        resolver = IncludeResolver()
        source = '#include <pbr/common.wgsl>'
        directives = resolver.parse_includes(source)

        assert len(directives) == 1
        assert directives[0].path == "pbr/common.wgsl"
        assert directives[0].is_relative is False
        assert directives[0].line_number == 1

    def test_parse_multiple_includes(self):
        """Parse multiple include directives from source."""
        resolver = IncludeResolver()
        source = dedent("""\
            // Header comment
            #include "local.wgsl"
            #include <global/shared.wgsl>

            fn main() {}
        """)
        directives = resolver.parse_includes(source)

        assert len(directives) == 2
        assert directives[0].path == "local.wgsl"
        assert directives[0].is_relative is True
        assert directives[0].line_number == 2
        assert directives[1].path == "global/shared.wgsl"
        assert directives[1].is_relative is False
        assert directives[1].line_number == 3

    def test_include_with_trailing_comment(self):
        """Include directives can have trailing comments."""
        resolver = IncludeResolver()
        source = '#include "brdf.wgsl" // BRDF functions'
        directives = resolver.parse_includes(source)

        assert len(directives) == 1
        assert directives[0].path == "brdf.wgsl"

    def test_resolve_project_include(self, resolver: IncludeResolver, temp_shader_dir: Path):
        """Resolve a project include from search paths."""
        source = '#include <common/math.wgsl>'
        result = resolver.resolve(source)

        assert "saturate" in result
        assert "clamp(x, 0.0, 1.0)" in result

    def test_resolve_relative_include(self, temp_shader_dir: Path):
        """Resolve a relative include from current file's directory."""
        resolver = IncludeResolver(
            search_paths=[temp_shader_dir / "shaders"]
        )

        color_wgsl = temp_shader_dir / "shaders" / "common" / "color.wgsl"
        result = resolver.resolve_file(color_wgsl)

        # Should include math.wgsl content
        assert "saturate" in result
        assert "linear_to_srgb" in result


# =============================================================================
# Suite B: Nested Includes
# =============================================================================


class TestNestedIncludes:
    """Tests for nested/recursive include resolution."""

    def test_nested_includes(self, resolver: IncludeResolver, temp_shader_dir: Path):
        """Include files can include other files."""
        # pbr/brdf.wgsl includes common/math.wgsl
        source = '#include <pbr/brdf.wgsl>'
        result = resolver.resolve(source)

        assert "fresnel_schlick" in result
        assert "saturate" in result  # From nested include

    def test_multiple_levels_of_nesting(self, temp_shader_dir: Path):
        """Test deeply nested includes."""
        # Create a chain: a.wgsl -> b.wgsl -> c.wgsl -> d.wgsl
        (temp_shader_dir / "chain").mkdir()
        (temp_shader_dir / "chain" / "d.wgsl").write_text("fn d_func() {}")
        (temp_shader_dir / "chain" / "c.wgsl").write_text('#include "d.wgsl"\nfn c_func() {}')
        (temp_shader_dir / "chain" / "b.wgsl").write_text('#include "c.wgsl"\nfn b_func() {}')
        (temp_shader_dir / "chain" / "a.wgsl").write_text('#include "b.wgsl"\nfn a_func() {}')

        resolver = IncludeResolver(search_paths=[temp_shader_dir / "chain"])
        result = resolver.resolve_file(temp_shader_dir / "chain" / "a.wgsl")

        assert "a_func" in result
        assert "b_func" in result
        assert "c_func" in result
        assert "d_func" in result

    def test_diamond_dependency(self, temp_shader_dir: Path):
        """Test diamond-shaped dependency: A->B, A->C, B->D, C->D."""
        (temp_shader_dir / "diamond").mkdir()
        (temp_shader_dir / "diamond" / "d.wgsl").write_text(
            "#pragma once\nfn d_func() {}"
        )
        (temp_shader_dir / "diamond" / "c.wgsl").write_text(
            '#include "d.wgsl"\nfn c_func() {}'
        )
        (temp_shader_dir / "diamond" / "b.wgsl").write_text(
            '#include "d.wgsl"\nfn b_func() {}'
        )
        (temp_shader_dir / "diamond" / "a.wgsl").write_text(
            '#include "b.wgsl"\n#include "c.wgsl"\nfn a_func() {}'
        )

        resolver = IncludeResolver(search_paths=[temp_shader_dir / "diamond"])
        result = resolver.resolve_file(temp_shader_dir / "diamond" / "a.wgsl")

        # d.wgsl should only appear once due to #pragma once
        assert result.count("d_func") == 1
        assert "a_func" in result
        assert "b_func" in result
        assert "c_func" in result


# =============================================================================
# Suite C: Cycle Detection
# =============================================================================


class TestCycleDetection:
    """Tests for cyclic include detection."""

    def test_self_include_cycle(self, temp_shader_dir: Path):
        """File including itself is detected as a cycle."""
        (temp_shader_dir / "self.wgsl").write_text('#include "self.wgsl"')

        resolver = IncludeResolver(search_paths=[temp_shader_dir])

        with pytest.raises(CyclicIncludeError) as exc_info:
            resolver.resolve_file(temp_shader_dir / "self.wgsl")

        assert len(exc_info.value.cycle) == 2
        assert "self.wgsl" in str(exc_info.value.cycle[-1])

    def test_two_file_cycle(self, temp_shader_dir: Path):
        """Two files including each other is detected as a cycle."""
        (temp_shader_dir / "a.wgsl").write_text('#include "b.wgsl"')
        (temp_shader_dir / "b.wgsl").write_text('#include "a.wgsl"')

        resolver = IncludeResolver(search_paths=[temp_shader_dir])

        with pytest.raises(CyclicIncludeError) as exc_info:
            resolver.resolve_file(temp_shader_dir / "a.wgsl")

        cycle = exc_info.value.cycle
        assert len(cycle) == 3  # a -> b -> a

    def test_three_file_cycle(self, temp_shader_dir: Path):
        """Three-file cycle: a -> b -> c -> a."""
        (temp_shader_dir / "cycle").mkdir()
        (temp_shader_dir / "cycle" / "a.wgsl").write_text('#include "b.wgsl"')
        (temp_shader_dir / "cycle" / "b.wgsl").write_text('#include "c.wgsl"')
        (temp_shader_dir / "cycle" / "c.wgsl").write_text('#include "a.wgsl"')

        resolver = IncludeResolver(search_paths=[temp_shader_dir / "cycle"])

        with pytest.raises(CyclicIncludeError):
            resolver.resolve_file(temp_shader_dir / "cycle" / "a.wgsl")

    def test_cycle_error_message(self, temp_shader_dir: Path):
        """Cycle error message shows the full cycle path."""
        (temp_shader_dir / "loop").mkdir()
        (temp_shader_dir / "loop" / "x.wgsl").write_text('#include "y.wgsl"')
        (temp_shader_dir / "loop" / "y.wgsl").write_text('#include "x.wgsl"')

        resolver = IncludeResolver(search_paths=[temp_shader_dir / "loop"])

        with pytest.raises(CyclicIncludeError) as exc_info:
            resolver.resolve_file(temp_shader_dir / "loop" / "x.wgsl")

        message = str(exc_info.value)
        assert "Cyclic include detected" in message
        assert "x.wgsl" in message
        assert "y.wgsl" in message


# =============================================================================
# Suite D: Max Depth Enforcement
# =============================================================================


class TestMaxDepthEnforcement:
    """Tests for maximum include depth enforcement."""

    def test_max_depth_exceeded(self, temp_shader_dir: Path):
        """Include chain exceeding max depth raises MaxDepthError."""
        # Create a chain longer than max_depth
        (temp_shader_dir / "deep").mkdir()
        for i in range(15):
            if i == 14:
                content = "fn final() {}"
            else:
                content = f'#include "level{i+1}.wgsl"'
            (temp_shader_dir / "deep" / f"level{i}.wgsl").write_text(content)

        resolver = IncludeResolver(
            search_paths=[temp_shader_dir / "deep"],
            max_depth=5
        )

        with pytest.raises(MaxDepthError) as exc_info:
            resolver.resolve_file(temp_shader_dir / "deep" / "level0.wgsl")

        assert exc_info.value.max_depth == 5
        assert exc_info.value.depth > 5

    def test_max_depth_at_limit(self, temp_shader_dir: Path):
        """Include chain exactly at max depth succeeds."""
        (temp_shader_dir / "exact").mkdir()
        # Create chain of exactly 5 levels
        (temp_shader_dir / "exact" / "l4.wgsl").write_text("fn l4() {}")
        (temp_shader_dir / "exact" / "l3.wgsl").write_text('#include "l4.wgsl"')
        (temp_shader_dir / "exact" / "l2.wgsl").write_text('#include "l3.wgsl"')
        (temp_shader_dir / "exact" / "l1.wgsl").write_text('#include "l2.wgsl"')
        (temp_shader_dir / "exact" / "l0.wgsl").write_text('#include "l1.wgsl"')

        resolver = IncludeResolver(
            search_paths=[temp_shader_dir / "exact"],
            max_depth=5
        )

        result = resolver.resolve_file(temp_shader_dir / "exact" / "l0.wgsl")
        assert "fn l4()" in result

    def test_max_depth_error_includes_stack(self, temp_shader_dir: Path):
        """MaxDepthError includes the include stack for debugging."""
        (temp_shader_dir / "stack").mkdir()
        for i in range(10):
            if i == 9:
                content = "fn end() {}"
            else:
                content = f'#include "s{i+1}.wgsl"'
            (temp_shader_dir / "stack" / f"s{i}.wgsl").write_text(content)

        resolver = IncludeResolver(
            search_paths=[temp_shader_dir / "stack"],
            max_depth=3
        )

        with pytest.raises(MaxDepthError) as exc_info:
            resolver.resolve_file(temp_shader_dir / "stack" / "s0.wgsl")

        assert len(exc_info.value.include_stack) > 0
        assert "s0.wgsl" in str(exc_info.value.include_stack[0])


# =============================================================================
# Suite E: Multiple Search Paths
# =============================================================================


class TestMultipleSearchPaths:
    """Tests for include resolution with multiple search paths."""

    def test_search_paths_priority(self, temp_shader_dir: Path):
        """Earlier search paths have priority over later ones."""
        (temp_shader_dir / "first").mkdir()
        (temp_shader_dir / "second").mkdir()

        # Same filename in both directories
        (temp_shader_dir / "first" / "shared.wgsl").write_text("fn first_version() {}")
        (temp_shader_dir / "second" / "shared.wgsl").write_text("fn second_version() {}")

        resolver = IncludeResolver(
            search_paths=[
                temp_shader_dir / "first",
                temp_shader_dir / "second",
            ]
        )

        source = '#include <shared.wgsl>'
        result = resolver.resolve(source)

        assert "first_version" in result
        assert "second_version" not in result

    def test_fallback_to_second_search_path(self, temp_shader_dir: Path):
        """If not found in first search path, search continues."""
        (temp_shader_dir / "primary").mkdir()
        (temp_shader_dir / "fallback").mkdir()

        (temp_shader_dir / "fallback" / "only_here.wgsl").write_text("fn fallback_func() {}")

        resolver = IncludeResolver(
            search_paths=[
                temp_shader_dir / "primary",
                temp_shader_dir / "fallback",
            ]
        )

        source = '#include <only_here.wgsl>'
        result = resolver.resolve(source)

        assert "fallback_func" in result

    def test_relative_include_before_search_paths(self, temp_shader_dir: Path):
        """Relative includes check current directory before search paths."""
        (temp_shader_dir / "local").mkdir()
        (temp_shader_dir / "global").mkdir()

        # Different content in local vs global
        (temp_shader_dir / "local" / "utils.wgsl").write_text("fn local_utils() {}")
        (temp_shader_dir / "global" / "utils.wgsl").write_text("fn global_utils() {}")
        (temp_shader_dir / "local" / "main.wgsl").write_text('#include "utils.wgsl"')

        resolver = IncludeResolver(
            search_paths=[temp_shader_dir / "global"]
        )

        result = resolver.resolve_file(temp_shader_dir / "local" / "main.wgsl")

        # Should use local version
        assert "local_utils" in result
        assert "global_utils" not in result

    def test_add_search_path(self, temp_shader_dir: Path):
        """Search paths can be added dynamically."""
        (temp_shader_dir / "added").mkdir()
        (temp_shader_dir / "added" / "dynamic.wgsl").write_text("fn dynamic_func() {}")

        resolver = IncludeResolver()
        resolver.add_search_path(temp_shader_dir / "added")

        source = '#include <dynamic.wgsl>'
        result = resolver.resolve(source)

        assert "dynamic_func" in result


# =============================================================================
# Suite F: Dependency Graph Tracking
# =============================================================================


class TestDepGraphTracking:
    """Tests for dependency graph construction and queries."""

    def test_dep_graph_records_direct_dependency(self, resolver: IncludeResolver, temp_shader_dir: Path):
        """DepGraph records direct include relationships."""
        source = '#include <common/math.wgsl>'
        resolver.resolve(source, temp_shader_dir / "test.wgsl")

        deps = resolver.dep_graph.get_dependencies(temp_shader_dir / "test.wgsl")
        assert len(deps) == 1
        assert "math.wgsl" in str(list(deps)[0])

    def test_dep_graph_records_nested_dependencies(self, resolver: IncludeResolver, temp_shader_dir: Path):
        """DepGraph records transitive dependencies."""
        # color.wgsl includes math.wgsl
        result = resolver.resolve_file(temp_shader_dir / "shaders" / "common" / "color.wgsl")

        # Direct dependency
        direct_deps = resolver.dep_graph.get_dependencies(
            temp_shader_dir / "shaders" / "common" / "color.wgsl"
        )
        assert len(direct_deps) == 1

        # Transitive dependencies
        all_deps = resolver.get_dependencies(
            temp_shader_dir / "shaders" / "common" / "color.wgsl"
        )
        assert len(all_deps) == 1  # Just math.wgsl

    def test_dep_graph_get_dependents(self, resolver: IncludeResolver, temp_shader_dir: Path):
        """DepGraph can find all files depending on a given file."""
        # Resolve a file that includes common/math.wgsl
        resolver.resolve_file(temp_shader_dir / "shaders" / "pbr" / "brdf.wgsl")

        math_wgsl = (temp_shader_dir / "shaders" / "common" / "math.wgsl").resolve()
        dependents = resolver.dep_graph.get_dependents(math_wgsl)

        assert len(dependents) == 1
        assert "brdf.wgsl" in str(list(dependents)[0])

    def test_invalidate_returns_affected_files(self, resolver: IncludeResolver, temp_shader_dir: Path):
        """invalidate() returns all files needing recompilation."""
        # Build dependency graph
        resolver.resolve_file(temp_shader_dir / "materials" / "gold.wgsl")

        # Invalidate math.wgsl and see what needs recompilation
        math_wgsl = (temp_shader_dir / "shaders" / "common" / "math.wgsl").resolve()
        affected = resolver.invalidate(math_wgsl)

        # gold.wgsl -> pbr/brdf.wgsl -> math.wgsl
        # gold.wgsl -> common/color.wgsl -> math.wgsl
        assert len(affected) >= 1


# =============================================================================
# Suite G: DepGraph Class
# =============================================================================


class TestDepGraph:
    """Tests for the DepGraph class in isolation."""

    def test_add_edge_creates_bidirectional_relationship(self):
        """add_edge creates both forward and reverse relationships."""
        graph = DepGraph()
        graph.add_edge(Path("/a.wgsl"), Path("/b.wgsl"))

        assert Path("/b.wgsl") in graph.get_dependencies(Path("/a.wgsl"))
        assert Path("/a.wgsl") in graph.get_dependents(Path("/b.wgsl"))

    def test_get_transitive_dependents(self):
        """get_transitive_dependents finds all files in dependency chain."""
        graph = DepGraph()
        # a includes b, b includes c
        graph.add_edge(Path("/a.wgsl"), Path("/b.wgsl"))
        graph.add_edge(Path("/b.wgsl"), Path("/c.wgsl"))

        # If c changes, both a and b need recompilation
        dependents = graph.get_transitive_dependents(Path("/c.wgsl"))
        assert Path("/a.wgsl") in dependents
        assert Path("/b.wgsl") in dependents

    def test_get_transitive_dependencies(self):
        """get_transitive_dependencies finds all included files."""
        graph = DepGraph()
        graph.add_edge(Path("/a.wgsl"), Path("/b.wgsl"))
        graph.add_edge(Path("/b.wgsl"), Path("/c.wgsl"))

        deps = graph.get_transitive_dependencies(Path("/a.wgsl"))
        assert Path("/b.wgsl") in deps
        assert Path("/c.wgsl") in deps

    def test_clear_removes_all_edges(self):
        """clear() removes all dependency information."""
        graph = DepGraph()
        graph.add_edge(Path("/a.wgsl"), Path("/b.wgsl"))
        graph.clear()

        assert len(graph.get_dependencies(Path("/a.wgsl"))) == 0
        assert len(graph.get_dependents(Path("/b.wgsl"))) == 0

    def test_remove_file_cleans_up_edges(self):
        """remove_file removes all edges involving the file."""
        graph = DepGraph()
        graph.add_edge(Path("/a.wgsl"), Path("/b.wgsl"))
        graph.add_edge(Path("/b.wgsl"), Path("/c.wgsl"))
        graph.add_edge(Path("/d.wgsl"), Path("/b.wgsl"))

        graph.remove_file(Path("/b.wgsl"))

        # a no longer has b as dependency
        assert Path("/b.wgsl") not in graph.get_dependencies(Path("/a.wgsl"))
        # d no longer has b as dependency
        assert Path("/b.wgsl") not in graph.get_dependencies(Path("/d.wgsl"))


# =============================================================================
# Suite H: Pragma Once
# =============================================================================


class TestPragmaOnce:
    """Tests for #pragma once include guard support."""

    def test_pragma_once_prevents_duplicate_inclusion(self, temp_shader_dir: Path):
        """#pragma once prevents a file from being included twice."""
        (temp_shader_dir / "once").mkdir()
        (temp_shader_dir / "once" / "shared.wgsl").write_text(dedent("""\
            #pragma once
            fn shared_func() {}
        """))
        (temp_shader_dir / "once" / "a.wgsl").write_text('#include "shared.wgsl"')
        (temp_shader_dir / "once" / "b.wgsl").write_text('#include "shared.wgsl"')
        (temp_shader_dir / "once" / "main.wgsl").write_text(dedent("""\
            #include "a.wgsl"
            #include "b.wgsl"
            fn main() {}
        """))

        resolver = IncludeResolver(search_paths=[temp_shader_dir / "once"])
        result = resolver.resolve_file(temp_shader_dir / "once" / "main.wgsl")

        # shared_func should appear only once
        assert result.count("fn shared_func()") == 1

    def test_pragma_once_is_stripped(self, temp_shader_dir: Path):
        """#pragma once directive is removed from output."""
        (temp_shader_dir / "strip.wgsl").write_text(dedent("""\
            #pragma once
            fn test() {}
        """))

        resolver = IncludeResolver(search_paths=[temp_shader_dir])
        result = resolver.resolve_file(temp_shader_dir / "strip.wgsl")

        assert "#pragma once" not in result
        assert "fn test()" in result


# =============================================================================
# Suite I: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in include resolution."""

    def test_include_file_not_found(self, temp_shader_dir: Path):
        """Missing include file raises IncludeFileNotFoundError."""
        source = '#include <nonexistent.wgsl>'
        resolver = IncludeResolver(search_paths=[temp_shader_dir])

        with pytest.raises(IncludeFileNotFoundError) as exc_info:
            resolver.resolve(source)

        assert exc_info.value.include_path == "nonexistent.wgsl"
        assert len(exc_info.value.search_paths) > 0

    def test_include_file_not_found_error_message(self, temp_shader_dir: Path):
        """IncludeFileNotFoundError has helpful error message."""
        source = '#include <missing/file.wgsl>'
        resolver = IncludeResolver(search_paths=[temp_shader_dir])

        with pytest.raises(IncludeFileNotFoundError) as exc_info:
            resolver.resolve(source, temp_shader_dir / "test.wgsl")

        message = str(exc_info.value)
        assert "missing/file.wgsl" in message
        assert "Searched" in message

    def test_relative_include_not_found(self, temp_shader_dir: Path):
        """Missing relative include raises IncludeFileNotFoundError."""
        (temp_shader_dir / "rel.wgsl").write_text('#include "does_not_exist.wgsl"')

        resolver = IncludeResolver(search_paths=[temp_shader_dir])

        with pytest.raises(IncludeFileNotFoundError):
            resolver.resolve_file(temp_shader_dir / "rel.wgsl")


# =============================================================================
# Suite J: Convenience Function
# =============================================================================


class TestPreprocessWGSL:
    """Tests for the preprocess_wgsl convenience function."""

    def test_preprocess_wgsl_basic(self, temp_shader_dir: Path):
        """preprocess_wgsl resolves includes."""
        source = '#include <common/math.wgsl>'
        result = preprocess_wgsl(
            source,
            search_paths=[temp_shader_dir / "shaders"]
        )

        assert "saturate" in result

    def test_preprocess_wgsl_with_current_file(self, temp_shader_dir: Path):
        """preprocess_wgsl handles current_file for relative includes."""
        color_file = temp_shader_dir / "shaders" / "common" / "color.wgsl"
        source = color_file.read_text()

        result = preprocess_wgsl(
            source,
            search_paths=[temp_shader_dir / "shaders"],
            current_file=color_file
        )

        assert "saturate" in result


# =============================================================================
# Suite K: Source Markers
# =============================================================================


class TestSourceMarkers:
    """Tests for include source markers in output."""

    def test_begin_end_markers(self, resolver: IncludeResolver, temp_shader_dir: Path):
        """Resolved includes have BEGIN/END markers."""
        source = '#include <common/math.wgsl>'
        result = resolver.resolve(source)

        assert "// >>> BEGIN INCLUDE: common/math.wgsl" in result
        assert "// <<< END INCLUDE: common/math.wgsl" in result

    def test_nested_markers(self, resolver: IncludeResolver, temp_shader_dir: Path):
        """Nested includes have properly nested markers."""
        source = '#include <common/color.wgsl>'
        result = resolver.resolve(source)

        # color.wgsl includes math.wgsl
        assert "// >>> BEGIN INCLUDE: common/color.wgsl" in result
        assert "// >>> BEGIN INCLUDE: math.wgsl" in result
