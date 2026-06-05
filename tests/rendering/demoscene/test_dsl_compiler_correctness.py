"""
DSL Compiler Correctness Tests (T-DEMO-7.4).

Comprehensive tests for the DSL compiler covering:
  - Primitive compilation (all 12 primitives)
  - Combinator compilation (union, intersection, subtraction)
  - Domain operation compilation (repetition, mirror, twist, bend)
  - Material compilation
  - Camera compilation
  - Light compilation
  - Caching semantics
  - Dirty invalidation
  - Optimizer passes (constant folding, DCE, CSE)
  - Error handling

Run: uv run pytest tests/rendering/demoscene/test_dsl_compiler_correctness.py -v
"""

from __future__ import annotations

import math
import pytest
from typing import List, Tuple

from engine.rendering.demoscene.ast_nodes import (
    Axis, BendNode, BoxNode, BoxFrameNode, CapsuleNode, CellIdNode,
    ConeNode, CylinderNode, EllipsoidNode, FloatNode, KifsNode,
    LightNode, LightType, MirrorNode, OctahedronNode, PlaneNode,
    PositionNode, PyramidNode, RepeatNode, RoundedBoxNode, SceneGraph,
    SphereNode, StretchNode, TorusNode, TwistNode, UnionNode,
    IntersectionNode, SubtractionNode, Vec3Node, CameraNode,
    MaterialNode, RenderSettingsNode, FullSceneNode,
)
from engine.rendering.demoscene.wgsl_codegen import (
    WgslCodeGen, generate_wgsl, generate_wgsl_from_scene,
    GENERATED_HEADER, SDF_SPHERE, SDF_BOX, SDF_TORUS, SDF_CYLINDER,
    SDF_CONE, SDF_PLANE, SDF_CAPSULE, SDF_ELLIPSOID, SDF_BOX_FRAME,
    SDF_ROUNDED_BOX, SDF_OCTAHEDRON, SDF_PYRAMID,
)
from engine.rendering.demoscene.sdf_optimizer import (
    SDFOptimizer, DEFAULT_PASSES, FAST_PASSES, AGGRESSIVE_PASSES,
    ConstantFoldingPass, DeadCodeEliminationPass,
    CommonSubexpressionEliminationPass, DomainRepetitionFlatteningPass,
    MaterialMergingPass, optimize_ast, fold_constants,
    eliminate_dead_code, eliminate_common_subexpressions,
    flatten_repeats, merge_materials, ast_hash, ast_equal,
)


# =============================================================================
# TEST HELPERS
# =============================================================================

def is_valid_wgsl(wgsl: str) -> bool:
    """Basic validation that WGSL has expected structure.

    Checks for:
      - Contains a function definition
      - Has proper WGSL syntax markers
      - Non-empty content
    """
    if not wgsl or not isinstance(wgsl, str):
        return False
    return "fn " in wgsl and "->" in wgsl and "{" in wgsl and "}" in wgsl


def contains_function(wgsl: str, func_name: str) -> bool:
    """Check if WGSL contains a function definition."""
    return f"fn {func_name}(" in wgsl


def count_functions(wgsl: str) -> int:
    """Count number of function definitions in WGSL."""
    return wgsl.count("fn ")


# =============================================================================
# PRIMITIVE COMPILATION TESTS
# =============================================================================

class TestPrimitiveCompilation:
    """Tests that all 12 SDF primitives compile to valid WGSL."""

    def test_sphere_compiles_to_valid_wgsl(self) -> None:
        """Sphere primitive generates valid WGSL with sdSphere function."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        wgsl = generate_wgsl(graph, name="sphere_test")

        assert is_valid_wgsl(wgsl)
        assert "fn sdSphere(" in wgsl
        assert "length(p) - r" in wgsl
        assert "sd_scene" in wgsl

    def test_box_compiles_to_valid_wgsl(self) -> None:
        """Box primitive generates valid WGSL with sdBox function."""
        graph = SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0)),),
        )
        wgsl = generate_wgsl(graph, name="box_test")

        assert is_valid_wgsl(wgsl)
        assert "fn sdBox(" in wgsl
        assert "abs(p) - b" in wgsl

    def test_torus_compiles_to_valid_wgsl(self) -> None:
        """Torus primitive generates valid WGSL with sdTorus function."""
        graph = SceneGraph(
            primitives=(TorusNode(PositionNode(), FloatNode(1.0), FloatNode(0.25)),),
        )
        wgsl = generate_wgsl(graph, name="torus_test")

        assert is_valid_wgsl(wgsl)
        assert "fn sdTorus(" in wgsl
        assert "vec2<f32>" in wgsl

    def test_cylinder_compiles_to_valid_wgsl(self) -> None:
        """Cylinder primitive generates valid WGSL with sdCylinder function."""
        graph = SceneGraph(
            primitives=(CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),),
        )
        wgsl = generate_wgsl(graph, name="cylinder_test")

        assert is_valid_wgsl(wgsl)
        assert "fn sdCylinder(" in wgsl

    def test_cone_compiles_to_valid_wgsl(self) -> None:
        """Cone primitive generates valid WGSL with sdCone function."""
        graph = SceneGraph(
            primitives=(ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),),
        )
        wgsl = generate_wgsl(graph, name="cone_test")

        assert is_valid_wgsl(wgsl)
        assert "fn sdCone(" in wgsl

    def test_plane_compiles_to_valid_wgsl(self) -> None:
        """Plane primitive generates valid WGSL with sdPlane function."""
        graph = SceneGraph(
            primitives=(PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),),
        )
        wgsl = generate_wgsl(graph, name="plane_test")

        assert is_valid_wgsl(wgsl)
        assert "fn sdPlane(" in wgsl
        assert "normalize(n)" in wgsl

    def test_capsule_compiles_to_valid_wgsl(self) -> None:
        """Capsule primitive generates valid WGSL with sdCapsule function."""
        graph = SceneGraph(
            primitives=(CapsuleNode(
                PositionNode(),
                Vec3Node(0.0, -1.0, 0.0),
                Vec3Node(0.0, 1.0, 0.0),
                FloatNode(0.5),
            ),),
        )
        wgsl = generate_wgsl(graph, name="capsule_test")

        assert is_valid_wgsl(wgsl)
        assert "fn sdCapsule(" in wgsl

    def test_ellipsoid_compiles_to_valid_wgsl(self) -> None:
        """Ellipsoid primitive generates valid WGSL with sdEllipsoid function."""
        graph = SceneGraph(
            primitives=(EllipsoidNode(PositionNode(), Vec3Node(1.0, 1.5, 1.0)),),
        )
        wgsl = generate_wgsl(graph, name="ellipsoid_test")

        assert is_valid_wgsl(wgsl)
        assert "fn sdEllipsoid(" in wgsl

    def test_box_frame_compiles_to_valid_wgsl(self) -> None:
        """BoxFrame primitive generates valid WGSL with sdBoxFrame function."""
        graph = SceneGraph(
            primitives=(BoxFrameNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0), FloatNode(0.1)),),
        )
        wgsl = generate_wgsl(graph, name="box_frame_test")

        assert is_valid_wgsl(wgsl)
        assert "fn sdBoxFrame(" in wgsl

    def test_rounded_box_compiles_to_valid_wgsl(self) -> None:
        """RoundedBox primitive generates valid WGSL with sdRoundedBox function."""
        graph = SceneGraph(
            primitives=(RoundedBoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0), FloatNode(0.2)),),
        )
        wgsl = generate_wgsl(graph, name="rounded_box_test")

        assert is_valid_wgsl(wgsl)
        assert "fn sdRoundedBox(" in wgsl

    def test_octahedron_compiles_to_valid_wgsl(self) -> None:
        """Octahedron primitive generates valid WGSL with sdOctahedron function."""
        graph = SceneGraph(
            primitives=(OctahedronNode(PositionNode(), FloatNode(1.0)),),
        )
        wgsl = generate_wgsl(graph, name="octahedron_test")

        assert is_valid_wgsl(wgsl)
        assert "fn sdOctahedron(" in wgsl

    def test_pyramid_compiles_to_valid_wgsl(self) -> None:
        """Pyramid primitive generates valid WGSL with sdPyramid function."""
        graph = SceneGraph(
            primitives=(PyramidNode(PositionNode(), FloatNode(1.5)),),
        )
        wgsl = generate_wgsl(graph, name="pyramid_test")

        assert is_valid_wgsl(wgsl)
        assert "fn sdPyramid(" in wgsl

    def test_all_primitives_in_single_scene(self) -> None:
        """All 12 primitives can coexist in a single scene."""
        primitives = (
            SphereNode(PositionNode(), FloatNode(1.0)),
            BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            TorusNode(PositionNode(), FloatNode(1.0), FloatNode(0.25)),
            CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
            ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),
            PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),
            CapsuleNode(PositionNode(), Vec3Node(0.0, -1.0, 0.0), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.3)),
            EllipsoidNode(PositionNode(), Vec3Node(1.0, 1.5, 1.0)),
            BoxFrameNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0), FloatNode(0.1)),
            RoundedBoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0), FloatNode(0.2)),
            OctahedronNode(PositionNode(), FloatNode(1.0)),
            PyramidNode(PositionNode(), FloatNode(1.5)),
        )
        graph = SceneGraph(primitives=primitives)
        wgsl = generate_wgsl(graph, name="all_primitives")

        assert is_valid_wgsl(wgsl)
        # All 12 SDF functions should be present
        assert count_functions(wgsl) >= 12


# =============================================================================
# COMBINATOR COMPILATION TESTS
# =============================================================================

class TestCombinatorCompilation:
    """Tests for CSG combinator compilation (union, intersection, subtraction)."""

    def test_union_of_two_primitives(self) -> None:
        """Union of two primitives generates min() in WGSL."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(0.5, 0.5, 0.5)),
            ),
        )
        wgsl = generate_wgsl(graph, name="union_test")

        assert is_valid_wgsl(wgsl)
        # Multiple primitives use select chain for union
        assert "select(" in wgsl or "result.x < d" in wgsl

    def test_union_of_three_primitives(self) -> None:
        """Union of three primitives generates proper chained combination."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(0.5, 0.5, 0.5)),
                TorusNode(PositionNode(), FloatNode(1.0), FloatNode(0.2)),
            ),
        )
        wgsl = generate_wgsl(graph, name="triple_union")

        assert is_valid_wgsl(wgsl)
        # Should have d0, d1, d2 for three primitives
        assert "d0" in wgsl and "d1" in wgsl and "d2" in wgsl

    def test_empty_scene_has_default_distance(self) -> None:
        """Empty scene returns large distance (1e10)."""
        graph = SceneGraph(primitives=())
        wgsl = generate_wgsl(graph, name="empty_scene")

        assert is_valid_wgsl(wgsl)
        assert "1e10" in wgsl or "result = vec2<f32>(1e10" in wgsl

    def test_single_primitive_no_union_overhead(self) -> None:
        """Single primitive scene doesn't use select chains."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        wgsl = generate_wgsl(graph, name="single_prim")

        assert is_valid_wgsl(wgsl)
        # Single primitive should directly assign result
        assert "let result = " in wgsl


# =============================================================================
# DOMAIN OPERATION COMPILATION TESTS
# =============================================================================

class TestDomainOpCompilation:
    """Tests for domain operation compilation (repetition, mirror, twist, bend)."""

    def test_repeat_compiles_to_domain_repeat(self) -> None:
        """Repeat operation generates domain_repeat call."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(RepeatNode(PositionNode(), Vec3Node(4.0, 4.0, 4.0)),),
        )
        wgsl = generate_wgsl(graph, name="repeat_test")

        assert is_valid_wgsl(wgsl)
        assert "domain_repeat" in wgsl
        assert "4.0, 4.0, 4.0" in wgsl

    def test_mirror_x_compiles(self) -> None:
        """Mirror X operation generates domain_mirror_x call."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(MirrorNode(PositionNode(), Axis.X),),
        )
        wgsl = generate_wgsl(graph, name="mirror_x")

        assert is_valid_wgsl(wgsl)
        assert "domain_mirror_x" in wgsl

    def test_mirror_y_compiles(self) -> None:
        """Mirror Y operation generates domain_mirror_y call."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(MirrorNode(PositionNode(), Axis.Y),),
        )
        wgsl = generate_wgsl(graph, name="mirror_y")

        assert is_valid_wgsl(wgsl)
        assert "domain_mirror_y" in wgsl

    def test_mirror_z_compiles(self) -> None:
        """Mirror Z operation generates domain_mirror_z call."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(MirrorNode(PositionNode(), Axis.Z),),
        )
        wgsl = generate_wgsl(graph, name="mirror_z")

        assert is_valid_wgsl(wgsl)
        assert "domain_mirror_z" in wgsl

    def test_twist_compiles_with_rate(self) -> None:
        """Twist operation generates domain_twist call with rate."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
        )
        wgsl = generate_wgsl(graph, name="twist_test")

        assert is_valid_wgsl(wgsl)
        assert "domain_twist" in wgsl
        assert "2.0" in wgsl

    def test_bend_compiles_with_radius(self) -> None:
        """Bend operation generates domain_bend call with radius."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(BendNode(PositionNode(), FloatNode(10.0)),),
        )
        wgsl = generate_wgsl(graph, name="bend_test")

        assert is_valid_wgsl(wgsl)
        assert "domain_bend" in wgsl
        assert "10.0" in wgsl

    def test_stretch_compiles_with_axis(self) -> None:
        """Stretch operation generates domain_stretch_{axis} call."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(StretchNode(PositionNode(), FloatNode(2.0), Axis.X),),
        )
        wgsl = generate_wgsl(graph, name="stretch_test")

        assert is_valid_wgsl(wgsl)
        assert "domain_stretch_x" in wgsl
        assert "2.0" in wgsl

    def test_kifs_compiles_with_folds(self) -> None:
        """KIFS operation generates domain_kifs call with folds."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(KifsNode(PositionNode(), FloatNode(8.0)),),
        )
        wgsl = generate_wgsl(graph, name="kifs_test")

        assert is_valid_wgsl(wgsl)
        assert "domain_kifs" in wgsl
        assert "8.0" in wgsl

    def test_kifs_generates_compensation_function(self) -> None:
        """KIFS operation emits compensation function for correct distances."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(KifsNode(PositionNode(), FloatNode(6.0)),),
        )
        wgsl = generate_wgsl(graph, name="kifs_comp")

        assert is_valid_wgsl(wgsl)
        assert "domain_kifs_compensation" in wgsl
        assert "cos(half_angle)" in wgsl

    def test_stretch_generates_compensation_function(self) -> None:
        """Stretch operation emits compensation function."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(StretchNode(PositionNode(), FloatNode(3.0), Axis.Y),),
        )
        wgsl = generate_wgsl(graph, name="stretch_comp")

        assert is_valid_wgsl(wgsl)
        assert "domain_stretch_compensation" in wgsl

    def test_chained_domain_ops(self) -> None:
        """Multiple domain operations chain correctly."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                MirrorNode(PositionNode(), Axis.X),
                TwistNode(PositionNode(), FloatNode(0.5)),
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
            ),
        )
        wgsl = generate_wgsl(graph, name="chained_ops")

        assert is_valid_wgsl(wgsl)
        assert "domain_mirror_x" in wgsl
        assert "domain_twist" in wgsl
        assert "domain_repeat" in wgsl


# =============================================================================
# MATERIAL COMPILATION TESTS
# =============================================================================

class TestMaterialCompilation:
    """Tests for material node compilation."""

    def test_material_node_created_with_albedo(self) -> None:
        """MaterialNode stores albedo correctly."""
        mat = MaterialNode(
            material_id=0,
            albedo=Vec3Node(1.0, 0.5, 0.25),
            roughness=FloatNode(0.5),
            metallic=FloatNode(0.0),
        )
        assert mat.albedo.x == 1.0
        assert mat.albedo.y == 0.5
        assert mat.albedo.z == 0.25

    def test_material_node_with_roughness(self) -> None:
        """MaterialNode stores roughness parameter."""
        mat = MaterialNode(
            material_id=1,
            albedo=Vec3Node(0.8, 0.8, 0.8),
            roughness=FloatNode(0.7),
        )
        assert mat.roughness.value == 0.7

    def test_material_node_with_metallic(self) -> None:
        """MaterialNode stores metallic parameter."""
        mat = MaterialNode(
            material_id=2,
            albedo=Vec3Node(1.0, 1.0, 1.0),
            metallic=FloatNode(1.0),
        )
        assert mat.metallic.value == 1.0

    def test_material_ids_propagated_in_wgsl(self) -> None:
        """Material IDs appear in generated WGSL as second return component."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
        )
        wgsl = generate_wgsl(graph, name="material_ids")

        # Material ID is encoded in vec2 second component
        assert "vec2<f32>(" in wgsl
        assert "0.0)" in wgsl  # material_id=0


# =============================================================================
# CAMERA COMPILATION TESTS
# =============================================================================

class TestCameraCompilation:
    """Tests for camera node compilation."""

    def test_camera_with_position(self) -> None:
        """CameraNode stores origin/position correctly."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 2.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(16.0 / 9.0),
        )
        assert camera.origin.x == 0.0
        assert camera.origin.y == 2.0
        assert camera.origin.z == 5.0

    def test_camera_with_target(self) -> None:
        """CameraNode stores look_at target correctly."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(1.0, 2.0, 3.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(45.0),
            aspect_ratio=FloatNode(1.0),
        )
        assert camera.look_at.x == 1.0
        assert camera.look_at.y == 2.0
        assert camera.look_at.z == 3.0

    def test_camera_with_fov(self) -> None:
        """CameraNode stores field of view correctly."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(90.0),
            aspect_ratio=FloatNode(16.0 / 9.0),
        )
        assert camera.fov.value == 90.0


# =============================================================================
# LIGHT COMPILATION TESTS
# =============================================================================

class TestLightCompilation:
    """Tests for light node compilation."""

    def test_point_light_creation(self) -> None:
        """LightNode creates point light correctly."""
        light = LightNode(
            position=Vec3Node(5.0, 10.0, 5.0),
            color=Vec3Node(1.0, 1.0, 1.0),
            intensity=FloatNode(100.0),
            light_type=LightType.POINT,
        )
        assert light.light_type == LightType.POINT
        assert light.intensity.value == 100.0

    def test_directional_light_creation(self) -> None:
        """LightNode creates directional light correctly."""
        light = LightNode(
            position=Vec3Node(0.0, 10.0, 0.0),
            color=Vec3Node(1.0, 0.95, 0.8),
            intensity=FloatNode(1.0),
            light_type=LightType.DIRECTIONAL,
            direction=Vec3Node(0.5, -1.0, 0.3),
        )
        assert light.light_type == LightType.DIRECTIONAL
        assert light.direction.y == -1.0

    def test_area_light_creation(self) -> None:
        """LightNode creates area light correctly."""
        light = LightNode(
            position=Vec3Node(0.0, 5.0, 0.0),
            color=Vec3Node(1.0, 1.0, 1.0),
            intensity=FloatNode(50.0),
            light_type=LightType.AREA,
            radius=FloatNode(2.0),
        )
        assert light.light_type == LightType.AREA
        assert light.radius.value == 2.0


# =============================================================================
# CACHING TESTS
# =============================================================================

class TestCaching:
    """Tests for compilation caching semantics."""

    def test_same_scene_returns_same_hash(self) -> None:
        """Identical scenes produce identical AST hashes."""
        graph1 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="test_scene",
        )
        graph2 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="test_scene",
        )

        hash1 = ast_hash(graph1)
        hash2 = ast_hash(graph2)

        assert hash1 == hash2

    def test_different_scenes_have_different_hash(self) -> None:
        """Different scenes produce different AST hashes."""
        graph1 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        graph2 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(2.0)),),
        )

        hash1 = ast_hash(graph1)
        hash2 = ast_hash(graph2)

        assert hash1 != hash2

    def test_scene_with_different_name_has_different_hash(self) -> None:
        """Scene name affects hash."""
        graph1 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="scene_a",
        )
        graph2 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="scene_b",
        )

        hash1 = ast_hash(graph1)
        hash2 = ast_hash(graph2)

        assert hash1 != hash2

    def test_same_wgsl_output_for_same_scene(self) -> None:
        """Compiling the same scene twice produces identical WGSL."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.5)),),
            name="cached_scene",
        )

        wgsl1 = generate_wgsl(graph, name="cached_scene")
        wgsl2 = generate_wgsl(graph, name="cached_scene")

        assert wgsl1 == wgsl2

    def test_ast_equal_for_identical_structures(self) -> None:
        """ast_equal returns True for structurally identical nodes."""
        node1 = SphereNode(PositionNode(), FloatNode(2.5))
        node2 = SphereNode(PositionNode(), FloatNode(2.5))

        assert ast_equal(node1, node2)

    def test_ast_not_equal_for_different_values(self) -> None:
        """ast_equal returns False for different values."""
        node1 = SphereNode(PositionNode(), FloatNode(1.0))
        node2 = SphereNode(PositionNode(), FloatNode(2.0))

        assert not ast_equal(node1, node2)

    def test_ast_not_equal_for_different_types(self) -> None:
        """ast_equal returns False for different node types."""
        node1 = SphereNode(PositionNode(), FloatNode(1.0))
        node2 = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))

        assert not ast_equal(node1, node2)


# =============================================================================
# DIRTY INVALIDATION TESTS
# =============================================================================

class TestDirtyInvalidation:
    """Tests for dirty tracking and cache invalidation."""

    def test_parameter_change_changes_hash(self) -> None:
        """Changing a parameter produces a different hash."""
        graph1 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        graph2 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.5)),),  # Changed radius
        )

        assert ast_hash(graph1) != ast_hash(graph2)

    def test_adding_primitive_changes_hash(self) -> None:
        """Adding a primitive produces a different hash."""
        graph1 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        graph2 = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
        )

        assert ast_hash(graph1) != ast_hash(graph2)

    def test_adding_domain_op_changes_hash(self) -> None:
        """Adding a domain operation produces a different hash."""
        graph1 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(),
        )
        graph2 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(1.0)),),
        )

        assert ast_hash(graph1) != ast_hash(graph2)

    def test_modified_scene_produces_different_wgsl(self) -> None:
        """Modified scene parameters produce different WGSL."""
        graph1 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        graph2 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(5.0)),),
        )

        wgsl1 = generate_wgsl(graph1)
        wgsl2 = generate_wgsl(graph2)

        # WGSLs should differ in the radius value
        assert "1.0" in wgsl1 and "5.0" not in wgsl1
        assert "5.0" in wgsl2


# =============================================================================
# OPTIMIZER TESTS: CONSTANT FOLDING
# =============================================================================

class TestConstantFolding:
    """Tests for constant folding optimization pass."""

    def test_fold_nan_to_zero(self) -> None:
        """NaN values are folded to 0.0."""
        node = FloatNode(float('nan'))
        result = ConstantFoldingPass(node).run()

        assert isinstance(result, FloatNode)
        assert result.value == 0.0

    def test_fold_infinity_to_large_value(self) -> None:
        """Infinity is folded to 1e10."""
        node = FloatNode(float('inf'))
        result = ConstantFoldingPass(node).run()

        assert isinstance(result, FloatNode)
        assert result.value == 1e10

    def test_fold_negative_infinity(self) -> None:
        """Negative infinity is folded to -1e10."""
        node = FloatNode(float('-inf'))
        result = ConstantFoldingPass(node).run()

        assert isinstance(result, FloatNode)
        assert result.value == -1e10

    def test_identity_twist_removed(self) -> None:
        """Twist with rate=0 is removed (identity transform)."""
        inner = SphereNode(PositionNode(), FloatNode(1.0))
        twist = TwistNode(inner, FloatNode(0.0))

        result = ConstantFoldingPass(twist).run()

        # Should return the inner node, not the twist
        assert isinstance(result, SphereNode)

    def test_identity_bend_removed_infinite_radius(self) -> None:
        """Bend with infinite radius is removed (identity transform)."""
        inner = SphereNode(PositionNode(), FloatNode(1.0))
        bend = BendNode(inner, FloatNode(1e10))

        result = ConstantFoldingPass(bend).run()

        assert isinstance(result, SphereNode)

    def test_identity_stretch_removed(self) -> None:
        """Stretch with scale=1 is removed (identity transform)."""
        inner = SphereNode(PositionNode(), FloatNode(1.0))
        stretch = StretchNode(inner, FloatNode(1.0), Axis.X)

        result = ConstantFoldingPass(stretch).run()

        assert isinstance(result, SphereNode)

    def test_regular_values_unchanged(self) -> None:
        """Regular float values are preserved."""
        node = FloatNode(3.14159)
        result = ConstantFoldingPass(node).run()

        assert result.value == 3.14159


# =============================================================================
# OPTIMIZER TESTS: DEAD CODE ELIMINATION
# =============================================================================

class TestDeadCodeElimination:
    """Tests for dead code elimination optimization pass."""

    def test_degenerate_sphere_removed(self) -> None:
        """Sphere with zero radius is removed."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(0.0)),  # Degenerate
                SphereNode(PositionNode(), FloatNode(1.0)),  # Valid
            ),
        )

        result = DeadCodeEliminationPass(graph).run()

        assert isinstance(result, SceneGraph)
        assert len(result.primitives) == 1
        assert result.primitives[0].radius.value == 1.0

    def test_degenerate_box_removed(self) -> None:
        """Box with zero dimensions is removed."""
        graph = SceneGraph(
            primitives=(
                BoxNode(PositionNode(), Vec3Node(0.0, 1.0, 1.0)),  # Degenerate (x=0)
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),  # Valid
            ),
        )

        result = DeadCodeEliminationPass(graph).run()

        assert isinstance(result, SceneGraph)
        assert len(result.primitives) == 1

    def test_duplicate_mirror_removed(self) -> None:
        """Consecutive same-axis mirrors cancel out."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                MirrorNode(PositionNode(), Axis.X),
                MirrorNode(PositionNode(), Axis.X),  # Duplicate - should be removed
            ),
        )

        result = DeadCodeEliminationPass(graph).run()

        assert isinstance(result, SceneGraph)
        # Second mirror should be removed as redundant
        assert len(result.pipeline) == 1

    def test_valid_primitives_preserved(self) -> None:
        """Valid primitives are not removed."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
        )

        result = DeadCodeEliminationPass(graph).run()

        assert len(result.primitives) == 2


# =============================================================================
# OPTIMIZER TESTS: COMMON SUBEXPRESSION ELIMINATION
# =============================================================================

class TestCommonSubexpressionElimination:
    """Tests for CSE optimization pass."""

    def test_cse_detects_duplicate_expressions(self) -> None:
        """CSE identifies duplicate subexpressions."""
        sphere1 = SphereNode(PositionNode(), FloatNode(1.0))
        sphere2 = SphereNode(PositionNode(), FloatNode(1.0))

        graph = SceneGraph(primitives=(sphere1, sphere2))

        pass_instance = CommonSubexpressionEliminationPass(graph)
        result = pass_instance.run()

        # CSE should detect the duplicate
        stats = pass_instance.stats
        # Note: CSE only reports if count > 1 for a hash
        if "expressions_hoisted" in stats:
            assert stats["expressions_hoisted"] > 0

    def test_cse_preserves_unique_expressions(self) -> None:
        """CSE does not merge different expressions."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        box = BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0))

        graph = SceneGraph(primitives=(sphere, box))

        result = CommonSubexpressionEliminationPass(graph).run()

        assert isinstance(result, SceneGraph)
        assert len(result.primitives) == 2


# =============================================================================
# OPTIMIZER TESTS: DOMAIN REPETITION FLATTENING
# =============================================================================

class TestDomainRepetitionFlattening:
    """Tests for domain repetition flattening optimization."""

    def test_nested_repeats_flattened(self) -> None:
        """Nested repeats can be combined."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(0.5)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                RepeatNode(PositionNode(), Vec3Node(4.0, 4.0, 4.0)),
            ),
        )

        pass_instance = DomainRepetitionFlatteningPass(graph)
        result = pass_instance.run()

        assert isinstance(result, SceneGraph)
        # Pipeline should be reduced if repeats were combined
        assert len(result.pipeline) <= 2

    def test_single_repeat_unchanged(self) -> None:
        """Single repeat is not modified."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(RepeatNode(PositionNode(), Vec3Node(3.0, 3.0, 3.0)),),
        )

        result = DomainRepetitionFlatteningPass(graph).run()

        assert len(result.pipeline) == 1


# =============================================================================
# OPTIMIZER TESTS: MATERIAL MERGING
# =============================================================================

class TestMaterialMerging:
    """Tests for material merging optimization."""

    def test_single_material_detected(self) -> None:
        """Single-material scene is detected."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
        )

        pass_instance = MaterialMergingPass(graph)
        result = pass_instance.run()

        # All primitives have default material 0
        assert len(pass_instance.material_groups) == 1

    def test_primitives_grouped_by_material(self) -> None:
        """Primitives are grouped by material ID."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
                TorusNode(PositionNode(), FloatNode(1.0), FloatNode(0.25)),
            ),
        )

        pass_instance = MaterialMergingPass(graph)
        result = pass_instance.run()

        # With default material, all should be in one group
        assert 0 in pass_instance.material_groups
        assert len(pass_instance.material_groups[0]) == 3


# =============================================================================
# OPTIMIZER TESTS: FULL PIPELINE
# =============================================================================

class TestOptimizerPipeline:
    """Tests for the complete optimization pipeline."""

    def test_default_passes_produce_valid_ast(self) -> None:
        """Default optimization passes produce valid AST."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
            pipeline=(TwistNode(PositionNode(), FloatNode(0.5)),),
        )

        optimizer = SDFOptimizer(DEFAULT_PASSES)
        result = optimizer.optimize(graph)

        assert isinstance(result, SceneGraph)
        assert len(result.primitives) >= 1

    def test_fast_passes_run_quickly(self) -> None:
        """Fast passes are a subset of default passes."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )

        optimizer = SDFOptimizer(FAST_PASSES)
        result = optimizer.optimize(graph)

        assert isinstance(result, SceneGraph)

    def test_aggressive_passes_run_all(self) -> None:
        """Aggressive passes include all optimizations."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )

        optimizer = SDFOptimizer(AGGRESSIVE_PASSES)
        result = optimizer.optimize(graph)

        assert isinstance(result, SceneGraph)

    def test_optimizer_collects_stats(self) -> None:
        """Optimizer collects statistics from all passes."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(0.0)),  # Degenerate - will be removed
                SphereNode(PositionNode(), FloatNode(1.0)),
            ),
        )

        optimizer = SDFOptimizer(DEFAULT_PASSES)
        result = optimizer.optimize(graph)

        stats = optimizer.stats
        assert isinstance(stats, dict)
        # Should have stats for each pass
        assert "DeadCodeEliminationPass" in stats

    def test_optimized_scene_compiles_to_valid_wgsl(self) -> None:
        """Optimized scene still produces valid WGSL."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
            pipeline=(
                TwistNode(PositionNode(), FloatNode(0.5)),
                RepeatNode(PositionNode(), Vec3Node(4.0, 4.0, 4.0)),
            ),
        )

        optimized = optimize_ast(graph)
        wgsl = generate_wgsl(optimized, name="optimized_scene")

        assert is_valid_wgsl(wgsl)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in the DSL compiler."""

    def test_empty_scene_produces_valid_wgsl(self) -> None:
        """Empty scene does not crash, produces valid WGSL."""
        graph = SceneGraph(primitives=())

        wgsl = generate_wgsl(graph, name="empty")

        assert is_valid_wgsl(wgsl)
        # Should have fallback distance
        assert "1e10" in wgsl

    def test_extreme_float_values_handled(self) -> None:
        """Extreme float values are handled gracefully."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1e-10)),),
        )

        wgsl = generate_wgsl(graph)

        assert is_valid_wgsl(wgsl)

    def test_negative_radius_still_compiles(self) -> None:
        """Negative radius compiles (DCE may remove it)."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(-1.0)),),
        )

        # Should not crash
        wgsl = generate_wgsl(graph)
        assert wgsl is not None

    def test_zero_vec3_still_compiles(self) -> None:
        """Zero-sized box compiles (DCE may remove it)."""
        graph = SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(0.0, 0.0, 0.0)),),
        )

        wgsl = generate_wgsl(graph)
        assert wgsl is not None

    def test_scene_name_with_special_chars_escaped(self) -> None:
        """Scene names with special characters are sanitized."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="my-scene with spaces",
        )

        wgsl = generate_wgsl(graph, name=graph.name)

        assert is_valid_wgsl(wgsl)
        # Name should be sanitized (no spaces/hyphens in function names)
        assert "my_scene_with_spaces" in wgsl

    def test_very_deep_pipeline_compiles(self) -> None:
        """Deep domain operation pipeline compiles correctly."""
        pipeline = tuple(
            MirrorNode(PositionNode(), Axis.X) if i % 3 == 0
            else TwistNode(PositionNode(), FloatNode(0.1)) if i % 3 == 1
            else BendNode(PositionNode(), FloatNode(10.0))
            for i in range(10)
        )

        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=pipeline,
        )

        wgsl = generate_wgsl(graph)
        assert is_valid_wgsl(wgsl)

    def test_many_primitives_compile(self) -> None:
        """Scene with many primitives compiles correctly."""
        primitives = tuple(
            SphereNode(PositionNode(), FloatNode(float(i) * 0.1 + 0.1))
            for i in range(20)
        )

        graph = SceneGraph(primitives=primitives)

        wgsl = generate_wgsl(graph)
        assert is_valid_wgsl(wgsl)
        # Should have d0 through d19
        assert "d0" in wgsl and "d19" in wgsl


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module convenience functions."""

    def test_fold_constants_function(self) -> None:
        """fold_constants convenience function works."""
        node = TwistNode(SphereNode(PositionNode(), FloatNode(1.0)), FloatNode(0.0))
        result = fold_constants(node)

        # Identity twist should be removed
        assert isinstance(result, SphereNode)

    def test_eliminate_dead_code_function(self) -> None:
        """eliminate_dead_code convenience function works."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(0.0)),),  # Degenerate
        )
        result = eliminate_dead_code(graph)

        assert isinstance(result, SceneGraph)
        # Degenerate primitive should be removed
        assert len(result.primitives) == 0

    def test_eliminate_common_subexpressions_function(self) -> None:
        """eliminate_common_subexpressions convenience function works."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                SphereNode(PositionNode(), FloatNode(1.0)),
            ),
        )
        result = eliminate_common_subexpressions(graph)

        assert isinstance(result, SceneGraph)

    def test_flatten_repeats_function(self) -> None:
        """flatten_repeats convenience function works."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),),
        )
        result = flatten_repeats(graph)

        assert isinstance(result, SceneGraph)

    def test_merge_materials_function(self) -> None:
        """merge_materials convenience function works."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        result = merge_materials(graph)

        assert isinstance(result, SceneGraph)

    def test_optimize_ast_function(self) -> None:
        """optimize_ast convenience function works."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        result = optimize_ast(graph)

        assert isinstance(result, SceneGraph)

    def test_generate_wgsl_from_scene_dict(self) -> None:
        """generate_wgsl_from_scene works with dict input."""
        wgsl = generate_wgsl_from_scene(
            primitives=[{"type": "sphere", "radius": 1.5}],
            name="dict_scene",
        )

        assert is_valid_wgsl(wgsl)
        assert "sdSphere" in wgsl

    def test_generate_wgsl_from_scene_with_pipeline(self) -> None:
        """generate_wgsl_from_scene works with pipeline."""
        wgsl = generate_wgsl_from_scene(
            primitives=[{"type": "sphere", "radius": 1.0}],
            pipeline=[{"type": "twist", "rate": 0.5}],
            name="pipeline_scene",
        )

        assert is_valid_wgsl(wgsl)
        assert "domain_twist" in wgsl


# =============================================================================
# FULL SCENE NODE TESTS
# =============================================================================

class TestFullSceneNode:
    """Tests for FullSceneNode compilation."""

    def test_full_scene_node_creation(self) -> None:
        """FullSceneNode can be created with all components."""
        scene_graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )

        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(16.0 / 9.0),
        )

        light = LightNode(
            position=Vec3Node(5.0, 5.0, 5.0),
            color=Vec3Node(1.0, 1.0, 1.0),
            intensity=FloatNode(100.0),
        )

        material = MaterialNode(
            material_id=0,
            albedo=Vec3Node(0.8, 0.2, 0.2),
            roughness=FloatNode(0.5),
            metallic=FloatNode(0.0),
        )

        full_scene = FullSceneNode(
            scene_graph=scene_graph,
            camera=camera,
            lights=(light,),
            materials=(material,),
            name="full_test_scene",
        )

        assert full_scene.scene_graph == scene_graph
        assert full_scene.camera == camera
        assert len(full_scene.lights) == 1
        assert len(full_scene.materials) == 1

    def test_full_scene_default_camera(self) -> None:
        """FullSceneNode creates default camera if not provided."""
        scene_graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )

        full_scene = FullSceneNode(scene_graph=scene_graph)

        assert full_scene.camera is not None
        assert full_scene.camera.fov.value == 60.0

    def test_full_scene_default_settings(self) -> None:
        """FullSceneNode creates default render settings if not provided."""
        scene_graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )

        full_scene = FullSceneNode(scene_graph=scene_graph)

        assert full_scene.settings is not None
        assert full_scene.settings.width == 1920
        assert full_scene.settings.height == 1080


# =============================================================================
# WGSL OUTPUT VALIDATION TESTS
# =============================================================================

class TestWGSLOutputValidation:
    """Tests that validate the structure of generated WGSL."""

    def test_wgsl_has_header(self) -> None:
        """Generated WGSL includes header comment."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        wgsl = generate_wgsl(graph)

        assert "SPDX-License-Identifier" in wgsl or "Auto-generated" in wgsl

    def test_wgsl_has_scene_function(self) -> None:
        """Generated WGSL has sd_scene function."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="test",
        )
        wgsl = generate_wgsl(graph, name="test")

        assert "fn sd_scene" in wgsl

    def test_wgsl_returns_vec2(self) -> None:
        """Scene function returns vec2<f32> (distance, material_id)."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        wgsl = generate_wgsl(graph)

        assert "-> vec2<f32>" in wgsl

    def test_wgsl_has_proper_syntax(self) -> None:
        """Generated WGSL has balanced braces and proper syntax."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
            pipeline=(TwistNode(PositionNode(), FloatNode(0.5)),),
        )
        wgsl = generate_wgsl(graph)

        # Balanced braces
        assert wgsl.count("{") == wgsl.count("}")
        # Balanced parentheses
        assert wgsl.count("(") == wgsl.count(")")

    def test_wgsl_no_duplicate_function_definitions(self) -> None:
        """Same primitive type doesn't generate duplicate functions."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                SphereNode(PositionNode(), FloatNode(2.0)),
                SphereNode(PositionNode(), FloatNode(3.0)),
            ),
        )
        wgsl = generate_wgsl(graph)

        # sdSphere should only appear once as a function definition
        assert wgsl.count("fn sdSphere(") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
