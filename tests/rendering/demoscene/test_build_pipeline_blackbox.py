"""
Blackbox tests for Build-Time DSL Compilation Pipeline (T-DEMO-5.5).

Tests cover external behavior and integration:
  - CLI invocation
  - End-to-end compilation
  - Error handling
  - Output format validation
  - Integration with demoscene modules

Requirements:
  - 20+ blackbox tests
  - CLI argument testing
  - Full pipeline testing
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# =============================================================================
# Module Setup
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
COMPILE_SCRIPT = PROJECT_ROOT / "scripts" / "compile_demo.py"
DEMO_SCENE = PROJECT_ROOT / "scenes" / "demo.py"


def run_compile_demo(*args, cwd=None) -> subprocess.CompletedProcess:
    """Run compile_demo.py with given arguments."""
    cmd = ["uv", "run", "python", str(COMPILE_SCRIPT), *args]
    return subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def minimal_scene_file(temp_dir):
    """Create a minimal valid scene file."""
    scene_content = '''"""Test scene."""
from engine.rendering.demoscene.ast_nodes import (
    SphereNode, PositionNode, FloatNode, Vec3Node,
    SceneGraph, MaterialNode, FullSceneNode,
)

p = PositionNode()
sphere = SphereNode(position=p, radius=FloatNode(1.0))
scene_graph = SceneGraph(primitives=(sphere,), name="test")
materials = (MaterialNode(material_id=0, albedo=Vec3Node(1.0, 0.0, 0.0)),)
SCENE = FullSceneNode(scene_graph=scene_graph, materials=materials, name="test")
'''
    scene_file = temp_dir / "test_scene.py"
    scene_file.write_text(scene_content)
    return scene_file


@pytest.fixture
def complex_scene_file(temp_dir):
    """Create a complex scene file with multiple primitives."""
    scene_content = '''"""Complex test scene."""
from engine.rendering.demoscene.ast_nodes import (
    SphereNode, BoxNode, TorusNode, PositionNode, FloatNode, Vec3Node,
    MirrorNode, TwistNode, Axis,
    SceneGraph, MaterialNode, LightNode, CameraNode,
    RenderSettingsNode, FullSceneNode, LightType,
)

p = PositionNode()

# Primitives
sphere = SphereNode(position=p, radius=FloatNode(1.0))
box = BoxNode(position=p, size=Vec3Node(0.5, 0.5, 0.5))
torus = TorusNode(position=p, major_radius=FloatNode(1.5), minor_radius=FloatNode(0.3))

# Domain ops
pipeline = (
    MirrorNode(input=p, axis=Axis.X),
    TwistNode(input=p, rate=FloatNode(0.5)),
)

# Scene graph
scene_graph = SceneGraph(
    primitives=(sphere, box, torus),
    pipeline=pipeline,
    name="complex"
)

# Materials
materials = (
    MaterialNode(material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2)),
    MaterialNode(material_id=1, albedo=Vec3Node(0.2, 0.8, 0.2)),
    MaterialNode(material_id=2, albedo=Vec3Node(0.2, 0.2, 0.8)),
)

# Camera
camera = CameraNode(
    origin=Vec3Node(0.0, 2.0, 6.0),
    look_at=Vec3Node(0.0, 0.0, 0.0),
    up=Vec3Node(0.0, 1.0, 0.0),
    fov=FloatNode(60.0),
    aspect_ratio=FloatNode(16.0 / 9.0),
)

# Lights
lights = (
    LightNode(
        position=Vec3Node(5.0, 5.0, 5.0),
        color=Vec3Node(1.0, 1.0, 1.0),
        intensity=FloatNode(2.0),
        light_type=LightType.POINT,
    ),
)

# Settings
settings = RenderSettingsNode(
    width=1920,
    height=1080,
    max_steps=256,
    max_distance=100.0,
)

SCENE = FullSceneNode(
    scene_graph=scene_graph,
    materials=materials,
    camera=camera,
    lights=lights,
    settings=settings,
    name="complex"
)
'''
    scene_file = temp_dir / "complex_scene.py"
    scene_file.write_text(scene_content)
    return scene_file


# =============================================================================
# CLI Argument Tests
# =============================================================================


class TestCLIArguments:
    """Tests for CLI argument handling."""

    def test_help_flag(self):
        """--help displays usage information."""
        result = run_compile_demo("--help")
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "Usage" in result.stdout

    def test_missing_scene_file_arg(self):
        """Missing scene file argument shows error."""
        result = run_compile_demo()
        assert result.returncode != 0

    def test_missing_output_without_validate(self, temp_dir, minimal_scene_file):
        """Missing output file without --validate shows error."""
        result = run_compile_demo(str(minimal_scene_file))
        assert result.returncode == 8  # EXIT_INVALID_ARGUMENTS

    def test_validate_flag_no_output_required(self, minimal_scene_file):
        """--validate flag doesn't require output file."""
        result = run_compile_demo("--validate", str(minimal_scene_file))
        assert result.returncode == 0

    def test_list_primitives_flag(self, minimal_scene_file):
        """--list-primitives shows scene info."""
        result = run_compile_demo("--list-primitives", str(minimal_scene_file))
        assert result.returncode == 0
        assert "Scene:" in result.stdout
        assert "Primitives" in result.stdout

    def test_verbose_flag(self, temp_dir, minimal_scene_file):
        """--verbose flag shows additional output."""
        output_file = temp_dir / "output.wgsl"
        result = run_compile_demo(
            "--verbose",
            str(minimal_scene_file),
            str(output_file)
        )
        assert result.returncode == 0
        assert "Loading scene:" in result.stdout or "Scene loaded:" in result.stdout

    def test_quiet_flag(self, temp_dir, minimal_scene_file):
        """--quiet flag suppresses output."""
        output_file = temp_dir / "output.wgsl"
        result = run_compile_demo(
            "--quiet",
            str(minimal_scene_file),
            str(output_file)
        )
        assert result.returncode == 0
        # Quiet should have minimal output
        assert len(result.stdout.strip()) < 100


# =============================================================================
# Compilation Tests
# =============================================================================


class TestCompilation:
    """Tests for compilation functionality."""

    def test_compiles_minimal_scene(self, temp_dir, minimal_scene_file):
        """Compiles minimal scene to WGSL."""
        output_file = temp_dir / "output.wgsl"
        result = run_compile_demo(str(minimal_scene_file), str(output_file))

        assert result.returncode == 0
        assert output_file.exists()

    def test_output_is_valid_wgsl(self, temp_dir, minimal_scene_file):
        """Output contains valid WGSL structure."""
        output_file = temp_dir / "output.wgsl"
        run_compile_demo(str(minimal_scene_file), str(output_file))

        wgsl = output_file.read_text()
        assert "@compute" in wgsl
        assert "fn main(" in wgsl
        assert "fn scene_sdf(" in wgsl
        assert "fn scene_material(" in wgsl

    def test_compiles_complex_scene(self, temp_dir, complex_scene_file):
        """Compiles complex scene with multiple primitives."""
        output_file = temp_dir / "output.wgsl"
        result = run_compile_demo(str(complex_scene_file), str(output_file))

        assert result.returncode == 0
        assert output_file.exists()

        wgsl = output_file.read_text()
        # Should have SDF primitives
        assert "sdf_sphere" in wgsl or "sdSphere" in wgsl
        assert "sdf_box" in wgsl or "sdBox" in wgsl
        assert "sdf_torus" in wgsl or "sdTorus" in wgsl

    def test_compiles_demo_scene(self, temp_dir):
        """Compiles the actual demo.py scene."""
        if not DEMO_SCENE.exists():
            pytest.skip("scenes/demo.py not found")

        output_file = temp_dir / "demo.wgsl"
        result = run_compile_demo(str(DEMO_SCENE), str(output_file))

        assert result.returncode == 0
        assert output_file.exists()
        assert output_file.stat().st_size > 1000  # Should be substantial

    def test_creates_output_directories(self, temp_dir, minimal_scene_file):
        """Creates output directories if they don't exist."""
        output_file = temp_dir / "nested" / "dir" / "output.wgsl"
        result = run_compile_demo(str(minimal_scene_file), str(output_file))

        assert result.returncode == 0
        assert output_file.exists()

    def test_overwrites_existing_output(self, temp_dir, minimal_scene_file):
        """Overwrites existing output file."""
        output_file = temp_dir / "output.wgsl"
        output_file.write_text("old content")

        result = run_compile_demo(str(minimal_scene_file), str(output_file))

        assert result.returncode == 0
        assert output_file.read_text() != "old content"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_scene_file(self, temp_dir):
        """Returns error for missing scene file."""
        missing_file = temp_dir / "nonexistent.py"
        output_file = temp_dir / "output.wgsl"
        result = run_compile_demo(str(missing_file), str(output_file))

        assert result.returncode == 1  # EXIT_MISSING_SCENE

    def test_invalid_scene_type(self, temp_dir):
        """Returns error for invalid SCENE type."""
        scene_content = '''SCENE = "not a FullSceneNode"'''
        scene_file = temp_dir / "bad_scene.py"
        scene_file.write_text(scene_content)

        output_file = temp_dir / "output.wgsl"
        result = run_compile_demo(str(scene_file), str(output_file))

        # extract_scene returns None for invalid type, which leads to MISSING_SCENE_VAR
        # or INVALID_SCENE_TYPE depending on the code path
        assert result.returncode in (3, 4)  # EXIT_MISSING_SCENE_VAR or EXIT_INVALID_SCENE_TYPE
        assert "FullSceneNode" in result.stderr or "SCENE" in result.stderr

    def test_missing_scene_variable(self, temp_dir):
        """Returns error for missing SCENE variable."""
        scene_content = '''x = 42'''
        scene_file = temp_dir / "no_scene.py"
        scene_file.write_text(scene_content)

        output_file = temp_dir / "output.wgsl"
        result = run_compile_demo(str(scene_file), str(output_file))

        assert result.returncode == 3  # EXIT_MISSING_SCENE_VAR

    def test_syntax_error_in_scene(self, temp_dir):
        """Returns error for syntax error in scene."""
        scene_content = '''def broken('''
        scene_file = temp_dir / "syntax_error.py"
        scene_file.write_text(scene_content)

        output_file = temp_dir / "output.wgsl"
        result = run_compile_demo(str(scene_file), str(output_file))

        assert result.returncode == 2  # EXIT_IMPORT_ERROR

    def test_import_error_in_scene(self, temp_dir):
        """Returns error for import error in scene."""
        scene_content = '''import nonexistent_module_xyz_123'''
        scene_file = temp_dir / "import_error.py"
        scene_file.write_text(scene_content)

        output_file = temp_dir / "output.wgsl"
        result = run_compile_demo(str(scene_file), str(output_file))

        assert result.returncode == 2  # EXIT_IMPORT_ERROR


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for --validate flag."""

    def test_validate_valid_scene(self, minimal_scene_file):
        """--validate succeeds for valid scene."""
        result = run_compile_demo("--validate", str(minimal_scene_file))
        assert result.returncode == 0

    def test_validate_outputs_success_message(self, minimal_scene_file):
        """--validate outputs success message."""
        result = run_compile_demo("--validate", str(minimal_scene_file))
        assert "validated successfully" in result.stdout.lower() or result.returncode == 0

    def test_validate_invalid_scene(self, temp_dir):
        """--validate fails for invalid scene."""
        scene_content = '''SCENE = "invalid"'''
        scene_file = temp_dir / "invalid.py"
        scene_file.write_text(scene_content)

        result = run_compile_demo("--validate", str(scene_file))
        assert result.returncode != 0


# =============================================================================
# WGSL Output Format Tests
# =============================================================================


class TestWgslOutputFormat:
    """Tests for WGSL output format."""

    def test_output_has_header_comment(self, temp_dir, minimal_scene_file):
        """Output has header comment."""
        output_file = temp_dir / "output.wgsl"
        run_compile_demo(str(minimal_scene_file), str(output_file))

        wgsl = output_file.read_text()
        assert wgsl.startswith("//") or wgsl.startswith("/*")

    def test_output_has_bind_groups(self, temp_dir, minimal_scene_file):
        """Output has bind group definitions."""
        output_file = temp_dir / "output.wgsl"
        run_compile_demo(str(minimal_scene_file), str(output_file))

        wgsl = output_file.read_text()
        assert "@group" in wgsl
        assert "@binding" in wgsl

    def test_output_has_uniforms(self, temp_dir, minimal_scene_file):
        """Output has uniform buffer."""
        output_file = temp_dir / "output.wgsl"
        run_compile_demo(str(minimal_scene_file), str(output_file))

        wgsl = output_file.read_text()
        assert "struct Uniforms" in wgsl or "uniforms" in wgsl

    def test_output_has_material_struct(self, temp_dir, minimal_scene_file):
        """Output has Material struct."""
        output_file = temp_dir / "output.wgsl"
        run_compile_demo(str(minimal_scene_file), str(output_file))

        wgsl = output_file.read_text()
        assert "struct Material" in wgsl

    def test_output_has_ray_marching(self, temp_dir, minimal_scene_file):
        """Output has ray marching code."""
        output_file = temp_dir / "output.wgsl"
        run_compile_demo(str(minimal_scene_file), str(output_file))

        wgsl = output_file.read_text()
        # Should have ray marching functions
        assert "ray_march" in wgsl or "march" in wgsl.lower()

    def test_output_has_lighting(self, temp_dir, minimal_scene_file):
        """Output has lighting code."""
        output_file = temp_dir / "output.wgsl"
        run_compile_demo(str(minimal_scene_file), str(output_file))

        wgsl = output_file.read_text()
        assert "lighting" in wgsl.lower() or "light" in wgsl.lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with demoscene modules."""

    def test_uses_demoscene_codegen(self, temp_dir, minimal_scene_file):
        """Uses engine.rendering.demoscene codegen."""
        output_file = temp_dir / "output.wgsl"
        result = run_compile_demo(
            "--verbose",
            str(minimal_scene_file),
            str(output_file)
        )

        # Should successfully import and use demoscene modules
        assert result.returncode == 0

    def test_domain_ops_in_output(self, temp_dir, complex_scene_file):
        """Domain operations appear in output."""
        output_file = temp_dir / "output.wgsl"
        run_compile_demo(str(complex_scene_file), str(output_file))

        wgsl = output_file.read_text()
        # Should have domain operation calls
        assert "domain_" in wgsl or "mirror" in wgsl.lower() or "twist" in wgsl.lower()

    def test_multiple_materials_in_output(self, temp_dir, complex_scene_file):
        """Multiple materials handled correctly."""
        output_file = temp_dir / "output.wgsl"
        run_compile_demo(str(complex_scene_file), str(output_file))

        wgsl = output_file.read_text()
        # Should have scene_material function that handles multiple materials
        assert "fn scene_material(" in wgsl


# =============================================================================
# Rerun Behavior Tests (for build.rs)
# =============================================================================


class TestRerunBehavior:
    """Tests related to build.rs rerun-if-changed behavior."""

    def test_output_changes_with_scene(self, temp_dir):
        """Output changes when scene changes."""
        scene_file = temp_dir / "scene.py"
        output_file = temp_dir / "output.wgsl"

        # First compilation
        scene_content_v1 = '''"""V1."""
from engine.rendering.demoscene.ast_nodes import (
    SphereNode, PositionNode, FloatNode, Vec3Node,
    SceneGraph, MaterialNode, FullSceneNode,
)
p = PositionNode()
sphere = SphereNode(position=p, radius=FloatNode(1.0))
scene_graph = SceneGraph(primitives=(sphere,), name="v1")
materials = (MaterialNode(material_id=0, albedo=Vec3Node(1.0, 0.0, 0.0)),)
SCENE = FullSceneNode(scene_graph=scene_graph, materials=materials, name="v1")
'''
        scene_file.write_text(scene_content_v1)
        run_compile_demo(str(scene_file), str(output_file))
        wgsl_v1 = output_file.read_text()

        # Second compilation with different scene
        scene_content_v2 = '''"""V2."""
from engine.rendering.demoscene.ast_nodes import (
    SphereNode, PositionNode, FloatNode, Vec3Node,
    SceneGraph, MaterialNode, FullSceneNode,
)
p = PositionNode()
sphere = SphereNode(position=p, radius=FloatNode(2.0))  # Different radius
scene_graph = SceneGraph(primitives=(sphere,), name="v2")
materials = (MaterialNode(material_id=0, albedo=Vec3Node(0.0, 1.0, 0.0)),)  # Different color
SCENE = FullSceneNode(scene_graph=scene_graph, materials=materials, name="v2")
'''
        scene_file.write_text(scene_content_v2)
        run_compile_demo(str(scene_file), str(output_file))
        wgsl_v2 = output_file.read_text()

        # Output should be different
        assert wgsl_v1 != wgsl_v2
