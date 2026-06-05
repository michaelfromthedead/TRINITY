"""
Whitebox tests for Build-Time DSL Compilation Pipeline (T-DEMO-5.5).

Tests cover internal implementation details:
  - Script module loading
  - Scene extraction
  - WGSL compilation internals
  - Validation logic
  - Error handling paths

Requirements:
  - 20+ whitebox tests
  - Full coverage of internal functions
  - Edge case handling
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Module Setup
# =============================================================================

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the compile_demo module
compile_demo_path = PROJECT_ROOT / "scripts" / "compile_demo.py"

if compile_demo_path.exists():
    spec = importlib.util.spec_from_file_location("compile_demo", compile_demo_path)
    compile_demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compile_demo)
else:
    pytest.skip("compile_demo.py not found", allow_module_level=True)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def minimal_scene_content():
    """Minimal valid scene file content."""
    return '''"""Minimal test scene."""
from engine.rendering.demoscene.ast_nodes import (
    SphereNode, PositionNode, FloatNode, Vec3Node,
    SceneGraph, MaterialNode, FullSceneNode,
)

p = PositionNode()
sphere = SphereNode(position=p, radius=FloatNode(1.0))
scene_graph = SceneGraph(primitives=(sphere,), name="minimal")
materials = (
    MaterialNode(material_id=0, albedo=Vec3Node(1.0, 0.0, 0.0)),
)

SCENE = FullSceneNode(scene_graph=scene_graph, materials=materials, name="minimal")
'''


@pytest.fixture
def invalid_scene_content():
    """Scene file with invalid SCENE type."""
    return '''"""Invalid scene - SCENE is not FullSceneNode."""
SCENE = "not a FullSceneNode"
'''


@pytest.fixture
def missing_scene_var_content():
    """Scene file without SCENE variable."""
    return '''"""Missing SCENE variable."""
from engine.rendering.demoscene.ast_nodes import SphereNode
x = 42
'''


# =============================================================================
# setup_python_path Tests
# =============================================================================


class TestSetupPythonPath:
    """Tests for setup_python_path function."""

    def test_adds_project_root_to_path(self):
        """setup_python_path adds project root to sys.path."""
        original_path = sys.path.copy()

        # Clear any existing project root entries
        sys.path = [p for p in sys.path if "TRINITY" not in p]

        try:
            compile_demo.setup_python_path()
            # Should have added project root
            assert any("TRINITY" in p for p in sys.path), "Project root not in path"
        finally:
            sys.path = original_path

    def test_idempotent(self):
        """setup_python_path is idempotent (doesn't add duplicates)."""
        original_len = len(sys.path)
        compile_demo.setup_python_path()
        compile_demo.setup_python_path()
        compile_demo.setup_python_path()
        # Should not keep adding the same path
        assert len(sys.path) <= original_len + 1


# =============================================================================
# load_scene_module Tests
# =============================================================================


class TestLoadSceneModule:
    """Tests for load_scene_module function."""

    def test_loads_valid_module(self, temp_dir, minimal_scene_content):
        """load_scene_module loads a valid Python module."""
        scene_file = temp_dir / "test_scene.py"
        scene_file.write_text(minimal_scene_content)

        module = compile_demo.load_scene_module(scene_file)
        assert module is not None
        assert hasattr(module, "SCENE")

    def test_returns_none_for_missing_file(self, temp_dir):
        """load_scene_module returns None for missing file."""
        missing_file = temp_dir / "nonexistent.py"
        module = compile_demo.load_scene_module(missing_file)
        assert module is None

    def test_returns_none_for_non_python_file(self, temp_dir):
        """load_scene_module returns None for non-.py file."""
        txt_file = temp_dir / "scene.txt"
        txt_file.write_text("not python")
        module = compile_demo.load_scene_module(txt_file)
        assert module is None

    def test_handles_import_error(self, temp_dir):
        """load_scene_module handles import errors gracefully."""
        bad_file = temp_dir / "bad_scene.py"
        bad_file.write_text("import nonexistent_module_xyz")
        module = compile_demo.load_scene_module(bad_file)
        assert module is None

    def test_handles_syntax_error(self, temp_dir):
        """load_scene_module handles syntax errors gracefully."""
        bad_file = temp_dir / "syntax_error.py"
        bad_file.write_text("def broken(")
        module = compile_demo.load_scene_module(bad_file)
        assert module is None


# =============================================================================
# extract_scene Tests
# =============================================================================


class TestExtractScene:
    """Tests for extract_scene function."""

    def test_extracts_valid_scene(self, temp_dir, minimal_scene_content):
        """extract_scene extracts FullSceneNode from module."""
        scene_file = temp_dir / "test_scene.py"
        scene_file.write_text(minimal_scene_content)

        module = compile_demo.load_scene_module(scene_file)
        scene = compile_demo.extract_scene(module)

        assert scene is not None
        assert scene.name == "minimal"

    def test_returns_none_for_missing_scene_var(self, temp_dir, missing_scene_var_content):
        """extract_scene returns None if SCENE is missing."""
        scene_file = temp_dir / "no_scene.py"
        scene_file.write_text(missing_scene_var_content)

        module = compile_demo.load_scene_module(scene_file)
        scene = compile_demo.extract_scene(module)
        assert scene is None

    def test_returns_none_for_invalid_scene_type(self, temp_dir, invalid_scene_content):
        """extract_scene returns None if SCENE has wrong type."""
        scene_file = temp_dir / "bad_type.py"
        scene_file.write_text(invalid_scene_content)

        module = compile_demo.load_scene_module(scene_file)
        scene = compile_demo.extract_scene(module)
        assert scene is None


# =============================================================================
# compile_scene_to_wgsl Tests
# =============================================================================


class TestCompileSceneToWgsl:
    """Tests for compile_scene_to_wgsl function."""

    def test_compiles_valid_scene(self, temp_dir, minimal_scene_content):
        """compile_scene_to_wgsl produces WGSL from valid scene."""
        scene_file = temp_dir / "test_scene.py"
        scene_file.write_text(minimal_scene_content)

        module = compile_demo.load_scene_module(scene_file)
        scene = compile_demo.extract_scene(module)
        wgsl = compile_demo.compile_scene_to_wgsl(scene)

        assert wgsl is not None
        assert isinstance(wgsl, str)
        assert len(wgsl) > 0

    def test_wgsl_contains_scene_sdf(self, temp_dir, minimal_scene_content):
        """Compiled WGSL contains scene_sdf function."""
        scene_file = temp_dir / "test_scene.py"
        scene_file.write_text(minimal_scene_content)

        module = compile_demo.load_scene_module(scene_file)
        scene = compile_demo.extract_scene(module)
        wgsl = compile_demo.compile_scene_to_wgsl(scene)

        assert "fn scene_sdf(" in wgsl

    def test_wgsl_contains_compute_entry(self, temp_dir, minimal_scene_content):
        """Compiled WGSL contains @compute entry point."""
        scene_file = temp_dir / "test_scene.py"
        scene_file.write_text(minimal_scene_content)

        module = compile_demo.load_scene_module(scene_file)
        scene = compile_demo.extract_scene(module)
        wgsl = compile_demo.compile_scene_to_wgsl(scene)

        assert "@compute" in wgsl
        assert "fn main(" in wgsl


# =============================================================================
# validate_wgsl Tests
# =============================================================================


class TestValidateWgsl:
    """Tests for validate_wgsl function."""

    def test_valid_wgsl_passes(self):
        """validate_wgsl returns True for valid WGSL."""
        valid = """
            @compute @workgroup_size(8, 8, 1)
            fn main() {}
            fn scene_sdf(p: vec3<f32>) -> vec2<f32> { return vec2(0.0); }
            fn scene_material(id: u32) -> Material { return Material(); }
        """
        assert compile_demo.validate_wgsl(valid) is True

    def test_missing_compute_fails(self):
        """validate_wgsl returns False for missing @compute."""
        invalid = """
            fn main() {}
            fn scene_sdf(p: vec3<f32>) -> vec2<f32> { return vec2(0.0); }
            fn scene_material(id: u32) -> Material { return Material(); }
        """
        assert compile_demo.validate_wgsl(invalid) is False

    def test_missing_main_fails(self):
        """validate_wgsl returns False for missing main()."""
        invalid = """
            @compute @workgroup_size(8, 8, 1)
            fn scene_sdf(p: vec3<f32>) -> vec2<f32> { return vec2(0.0); }
            fn scene_material(id: u32) -> Material { return Material(); }
        """
        assert compile_demo.validate_wgsl(invalid) is False

    def test_missing_scene_sdf_fails(self):
        """validate_wgsl returns False for missing scene_sdf()."""
        invalid = """
            @compute @workgroup_size(8, 8, 1)
            fn main() {}
            fn scene_material(id: u32) -> Material { return Material(); }
        """
        assert compile_demo.validate_wgsl(invalid) is False

    def test_missing_scene_material_fails(self):
        """validate_wgsl returns False for missing scene_material()."""
        invalid = """
            @compute @workgroup_size(8, 8, 1)
            fn main() {}
            fn scene_sdf(p: vec3<f32>) -> vec2<f32> { return vec2(0.0); }
        """
        assert compile_demo.validate_wgsl(invalid) is False

    def test_unbalanced_braces_fails(self):
        """validate_wgsl returns False for unbalanced braces."""
        invalid = """
            @compute @workgroup_size(8, 8, 1)
            fn main() {
            fn scene_sdf(p: vec3<f32>) -> vec2<f32> { return vec2(0.0); }
            fn scene_material(id: u32) -> Material { return Material(); }
        """
        assert compile_demo.validate_wgsl(invalid) is False

    def test_unbalanced_parens_fails(self):
        """validate_wgsl returns False for unbalanced parentheses."""
        invalid = """
            @compute @workgroup_size(8, 8, 1)
            fn main() {}
            fn scene_sdf(p: vec3<f32> -> vec2<f32> { return vec2(0.0); }
            fn scene_material(id: u32) -> Material { return Material(); }
        """
        assert compile_demo.validate_wgsl(invalid) is False


# =============================================================================
# write_output Tests
# =============================================================================


class TestWriteOutput:
    """Tests for write_output function."""

    def test_writes_file(self, temp_dir):
        """write_output writes content to file."""
        output_path = temp_dir / "output.wgsl"
        content = "// Test WGSL content"

        result = compile_demo.write_output(content, output_path)

        assert result is True
        assert output_path.exists()
        assert output_path.read_text() == content

    def test_creates_parent_directories(self, temp_dir):
        """write_output creates parent directories if needed."""
        output_path = temp_dir / "nested" / "dir" / "output.wgsl"
        content = "// Test WGSL content"

        result = compile_demo.write_output(content, output_path)

        assert result is True
        assert output_path.exists()

    def test_overwrites_existing_file(self, temp_dir):
        """write_output overwrites existing file."""
        output_path = temp_dir / "output.wgsl"
        output_path.write_text("old content")

        new_content = "new content"
        result = compile_demo.write_output(new_content, output_path)

        assert result is True
        assert output_path.read_text() == new_content


# =============================================================================
# list_primitives Tests
# =============================================================================


class TestListPrimitives:
    """Tests for list_primitives function."""

    def test_lists_scene_info(self, temp_dir, minimal_scene_content, capsys):
        """list_primitives prints scene information."""
        scene_file = temp_dir / "test_scene.py"
        scene_file.write_text(minimal_scene_content)

        module = compile_demo.load_scene_module(scene_file)
        scene = compile_demo.extract_scene(module)
        compile_demo.list_primitives(scene)

        captured = capsys.readouterr()
        assert "Scene: minimal" in captured.out
        assert "Primitives" in captured.out


# =============================================================================
# get_scene_info Tests
# =============================================================================


class TestGetSceneInfo:
    """Tests for get_scene_info function."""

    def test_returns_scene_dict(self, temp_dir, minimal_scene_content):
        """get_scene_info returns dictionary with scene info."""
        scene_file = temp_dir / "test_scene.py"
        scene_file.write_text(minimal_scene_content)

        module = compile_demo.load_scene_module(scene_file)
        scene = compile_demo.extract_scene(module)
        info = compile_demo.get_scene_info(scene)

        assert isinstance(info, dict)
        assert "name" in info
        assert "primitives" in info
        assert "domain_ops" in info
        assert "materials" in info
        assert "lights" in info
        assert "resolution" in info
        assert "max_steps" in info

    def test_correct_primitive_count(self, temp_dir, minimal_scene_content):
        """get_scene_info returns correct primitive count."""
        scene_file = temp_dir / "test_scene.py"
        scene_file.write_text(minimal_scene_content)

        module = compile_demo.load_scene_module(scene_file)
        scene = compile_demo.extract_scene(module)
        info = compile_demo.get_scene_info(scene)

        assert info["primitives"] == 1  # One sphere


# =============================================================================
# Exit Code Tests
# =============================================================================


class TestExitCodes:
    """Tests for exit code constants."""

    def test_exit_codes_defined(self):
        """All exit codes are defined."""
        assert compile_demo.EXIT_SUCCESS == 0
        assert compile_demo.EXIT_MISSING_SCENE == 1
        assert compile_demo.EXIT_IMPORT_ERROR == 2
        assert compile_demo.EXIT_MISSING_SCENE_VAR == 3
        assert compile_demo.EXIT_INVALID_SCENE_TYPE == 4
        assert compile_demo.EXIT_COMPILATION_ERROR == 5
        assert compile_demo.EXIT_WGSL_VALIDATION_ERROR == 6
        assert compile_demo.EXIT_OUTPUT_WRITE_ERROR == 7
        assert compile_demo.EXIT_INVALID_ARGUMENTS == 8

    def test_exit_codes_unique(self):
        """All exit codes are unique."""
        codes = [
            compile_demo.EXIT_SUCCESS,
            compile_demo.EXIT_MISSING_SCENE,
            compile_demo.EXIT_IMPORT_ERROR,
            compile_demo.EXIT_MISSING_SCENE_VAR,
            compile_demo.EXIT_INVALID_SCENE_TYPE,
            compile_demo.EXIT_COMPILATION_ERROR,
            compile_demo.EXIT_WGSL_VALIDATION_ERROR,
            compile_demo.EXIT_OUTPUT_WRITE_ERROR,
            compile_demo.EXIT_INVALID_ARGUMENTS,
        ]
        assert len(codes) == len(set(codes))
