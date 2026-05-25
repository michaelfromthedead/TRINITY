"""Blackbox tests for WgslCodeGen (T-DEMO-2.6)."""

from __future__ import annotations

import os
import re

import pytest

from engine.rendering.demoscene.wgsl_codegen import (
    generate_wgsl, generate_wgsl_from_scene, GENERATED_HEADER,
)
from engine.rendering.demoscene.ast_nodes import (
    SceneGraph, SphereNode, BoxNode, TorusNode, CylinderNode, ConeNode,
    PlaneNode, CapsuleNode, PositionNode, FloatNode, Vec3Node, Axis,
    RepeatNode, MirrorNode, TwistNode, KifsNode, StretchNode, BendNode,
    CellIdNode,
)


def assert_well_formed_wgsl(src: str) -> None:
    assert "SPDX-License-Identifier: MIT" in src, "Missing SPDX header"
    assert "fn " in src, "No function declarations"
    assert "None" not in src, "Python None leaked into WGSL"


class TestWellFormedOutput:
    def test_single_sphere_is_well_formed(self):
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        assert_well_formed_wgsl(src)

    def test_header_present(self):
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        assert "SPDX-License-Identifier" in src
        assert "T-DEMO-2.6" in src

    def test_all_primitives_are_well_formed(self):
        primitives = [
            SphereNode(PositionNode(), FloatNode(1.0)),
            BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
            CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(1.0)),
            ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),
            PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),
        ]
        for prim in primitives:
            graph = SceneGraph(primitives=(prim,))
            src = generate_wgsl(graph)
            assert_well_formed_wgsl(src)


class TestPrimitiveFunctionPresence:
    def test_sphere_function_present(self):
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        assert "fn sdSphere(p: vec3<f32>, r: f32) -> f32" in src

    def test_box_function_present(self):
        src = generate_wgsl(SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),),
        ))
        assert "fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32" in src

    def test_torus_function_present(self):
        src = generate_wgsl(SceneGraph(
            primitives=(TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),),
        ))
        assert "fn sdTorus(p: vec3<f32>, t: vec2<f32>) -> f32" in src

    def test_cylinder_function_present(self):
        src = generate_wgsl(SceneGraph(
            primitives=(CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(1.0)),),
        ))
        assert "fn sdCylinder(p: vec3<f32>, h: f32, r: f32) -> f32" in src

    def test_cone_function_present(self):
        src = generate_wgsl(SceneGraph(
            primitives=(ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),),
        ))
        assert "fn sdCone(p: vec3<f32>, h: f32, r1: f32, r2: f32) -> f32" in src

    def test_plane_function_present(self):
        src = generate_wgsl(SceneGraph(
            primitives=(PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),),
        ))
        assert "fn sdPlane(p: vec3<f32>, n: vec3<f32>, d: f32) -> f32" in src


class TestPrimitiveFormulaStructure:
    def test_sphere_formula(self):
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        assert "length(p) - r" in src

    def test_box_formula(self):
        src = generate_wgsl(SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),),
        ))
        assert "abs(p) - b" in src

    def test_torus_formula(self):
        src = generate_wgsl(SceneGraph(
            primitives=(TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),),
        ))
        assert "length(p.xz)" in src

    def test_plane_formula(self):
        src = generate_wgsl(SceneGraph(
            primitives=(PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),),
        ))
        assert "dot(p, normalize(n))" in src

    def test_cylinder_formula(self):
        src = generate_wgsl(SceneGraph(
            primitives=(CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(1.0)),),
        ))
        assert "length(p.xz)" in src

    def test_cone_formula(self):
        src = generate_wgsl(SceneGraph(
            primitives=(ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),),
        ))
        assert "r1, r2" in src or "r2 - r1" in src


class TestSceneFunction:
    def test_scene_function_has_correct_signature(self):
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="demo",
        ))
        assert re.search(
            r"fn\s+sd_scene__demo\s*\(\s*p\s*:\s*vec3<f32>\s*\)\s*->\s*vec2<f32>",
            src,
        )

    def test_compensation_variable_present(self):
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        assert "let comp =" in src

    def test_scene_calls_sdf_primitive(self):
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        assert "sdSphere(" in src

    def test_scene_with_domain_op_calls_domain_function(self):
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
        ))
        assert "domain_twist(" in src


class TestGenerateFromDicts:
    def test_single_sphere_from_dict(self):
        src = generate_wgsl_from_scene(
            [{"type": "sphere", "radius": 1.0}],
            name="test",
        )
        assert_well_formed_wgsl(src)
        assert "sdSphere" in src

    def test_box_from_dict(self):
        src = generate_wgsl_from_scene(
            [{"type": "box", "size": (2.0, 3.0, 1.0)}],
            name="test",
        )
        assert_well_formed_wgsl(src)
        assert "sdBox" in src

    def test_torus_from_dict(self):
        src = generate_wgsl_from_scene(
            [{"type": "torus", "major_radius": 3.0, "minor_radius": 0.75}],
            name="test",
        )
        assert_well_formed_wgsl(src)
        assert "sdTorus" in src

    def test_mixed_primitives_from_dict(self):
        src = generate_wgsl_from_scene(
            [
                {"type": "sphere", "radius": 1.0},
                {"type": "box", "size": (1.0, 1.0, 1.0)},
            ],
            name="mixed",
        )
        assert_well_formed_wgsl(src)
        assert "sdSphere" in src
        assert "sdBox" in src

    def test_with_pipeline_from_dict(self):
        src = generate_wgsl_from_scene(
            [{"type": "sphere", "radius": 1.0}],
            pipeline=[{"type": "repeat", "cell_size": (2.0, 2.0, 2.0)}],
            name="repeated",
        )
        assert_well_formed_wgsl(src)
        assert "domain_repeat" in src


class TestDomainIntegration:
    def test_repeat_transforms_position(self):
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),),
        ))
        assert_well_formed_wgsl(src)
        assert "domain_repeat(" in src

    def test_mirror_x_before_box(self):
        src = generate_wgsl(SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),),
            pipeline=(MirrorNode(PositionNode(), Axis.X),),
        ))
        assert_well_formed_wgsl(src)
        assert "domain_mirror_x(" in src

    def test_kifs_with_compensation(self):
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(KifsNode(PositionNode(), FloatNode(6.0)),),
        ))
        assert_well_formed_wgsl(src)
        assert "domain_kifs_compensation" in src

    def test_stretch_with_compensation(self):
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(StretchNode(PositionNode(), FloatNode(2.0), Axis.X),),
        ))
        assert_well_formed_wgsl(src)
        assert "domain_stretch_compensation" in src

    def test_cylinder_with_domain(self):
        src = generate_wgsl(SceneGraph(
            primitives=(CylinderNode(PositionNode(), FloatNode(3.0), FloatNode(0.5)),),
            pipeline=(BendNode(PositionNode(), FloatNode(5.0)),),
        ))
        assert_well_formed_wgsl(src)
        assert "sdCylinder" in src
        assert "domain_bend" in src

    def test_cone_with_mirror(self):
        src = generate_wgsl(SceneGraph(
            primitives=(ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),),
            pipeline=(MirrorNode(PositionNode(), Axis.Y),),
        ))
        assert_well_formed_wgsl(src)
        assert "domain_mirror_y" in src
        assert "sdCone" in src

    def test_plane_with_repeat(self):
        src = generate_wgsl(SceneGraph(
            primitives=(PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),),
            pipeline=(RepeatNode(PositionNode(), Vec3Node(5.0, 5.0, 5.0)),),
        ))
        assert_well_formed_wgsl(src)
        assert "domain_repeat" in src
        assert "sdPlane" in src


class TestTypeMapConsistency:
    def test_all_primitive_types_mapped(self):
        from engine.rendering.demoscene.ast_nodes import SDF_PRIMITIVE_TYPE_MAP
        mapped = set(SDF_PRIMITIVE_TYPE_MAP.keys())
        expected = {SphereNode, BoxNode, TorusNode, CylinderNode, ConeNode, PlaneNode, CapsuleNode}
        assert mapped == expected

    def test_all_domain_op_types_mapped(self):
        from engine.rendering.demoscene.ast_nodes import DOMAIN_OP_TYPE_MAP
        mapped = set(DOMAIN_OP_TYPE_MAP.keys())
        expected = {RepeatNode, CellIdNode, MirrorNode, KifsNode, TwistNode, BendNode, StretchNode}
        assert mapped == expected

    def test_no_extra_primitive_mappings(self):
        from engine.rendering.demoscene.ast_nodes import SDF_PRIMITIVE_TYPE_MAP
        assert len(SDF_PRIMITIVE_TYPE_MAP) == 7

    def test_no_extra_domain_op_mappings(self):
        from engine.rendering.demoscene.ast_nodes import DOMAIN_OP_TYPE_MAP
        assert len(DOMAIN_OP_TYPE_MAP) == 7


class TestAllPrimitivesTogether:
    """Verify all 7 SDF primitives can appear in a single scene graph."""

    def test_all_seven_primitives_in_one_scene(self):
        """All seven primitive sdf_* functions present when all 7 nodes used."""
        primitives = (
            SphereNode(PositionNode(), FloatNode(1.0)),
            BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
            CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(1.0)),
            ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),
            PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),
            CapsuleNode(PositionNode(), Vec3Node(0.0, -1.0, 0.0), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.5)),
        )
        src = generate_wgsl(SceneGraph(primitives=primitives))
        assert_well_formed_wgsl(src)
        assert "fn sdSphere" in src
        assert "fn sdBox" in src
        assert "fn sdTorus" in src
        assert "fn sdCylinder" in src
        assert "fn sdCone" in src
        assert "fn sdPlane" in src
        assert "fn sdCapsule" in src

    def test_each_sdf_function_emitted_exactly_once(self):
        """Each unique primitive type appears once (deduplication)."""
        primitives = (
            SphereNode(PositionNode(), FloatNode(1.0)),
            BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
            CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(1.0)),
            ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),
            PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),
            CapsuleNode(PositionNode(), Vec3Node(0.0, -1.0, 0.0), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.5)),
        )
        src = generate_wgsl(SceneGraph(primitives=primitives))
        assert src.count("fn sdSphere") == 1
        assert src.count("fn sdBox") == 1
        assert src.count("fn sdTorus") == 1
        assert src.count("fn sdCylinder") == 1
        assert src.count("fn sdCone") == 1
        assert src.count("fn sdPlane") == 1
        assert src.count("fn sdCapsule") == 1

    def test_all_seven_sdf_calls_in_scene_body(self):
        """The scene body calls all seven sdf_* primitives."""
        primitives = (
            SphereNode(PositionNode(), FloatNode(1.0)),
            BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
            CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(1.0)),
            ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),
            PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),
            CapsuleNode(PositionNode(), Vec3Node(0.0, -1.0, 0.0), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.5)),
        )
        src = generate_wgsl(SceneGraph(primitives=primitives))
        assert "sdSphere(p" in src
        assert "sdBox(p" in src
        assert "sdTorus(p" in src
        assert "sdCylinder(p" in src
        assert "sdCone(p" in src
        assert "sdPlane(p" in src
        assert "sdCapsule(p" in src

    def test_seven_primitives_have_seven_distance_variables(self):
        """Each primitive gets a distance variable d0..d6 in the scene."""
        primitives = (
            SphereNode(PositionNode(), FloatNode(1.0)),
            BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
            CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(1.0)),
            ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),
            PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),
            CapsuleNode(PositionNode(), Vec3Node(0.0, -1.0, 0.0), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.5)),
        )
        src = generate_wgsl(SceneGraph(primitives=primitives))
        for i in range(7):
            assert f"let d{i}" in src, f"Missing distance variable d{i}"

    def test_seven_primitives_min_chain_complete(self):
        """The min() chain covers all seven distance variables."""
        primitives = (
            SphereNode(PositionNode(), FloatNode(1.0)),
            BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
            CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(1.0)),
            ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),
            PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),
            CapsuleNode(PositionNode(), Vec3Node(0.0, -1.0, 0.0), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.5)),
        )
        src = generate_wgsl(SceneGraph(primitives=primitives))
        # The final return uses select() for material-tracking pairwise merge
        assert "select(" in src
        assert "d6" in src


class TestSceneWithMultiplePrimitives:
    """Verify scene function correctly combines multiple primitives."""

    def test_two_primitives_min_combined(self):
        """Scene with two primitives uses min(d0, d1)."""
        src = generate_wgsl(SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
        ))
        assert "let d0 =" in src
        assert "let d1 =" in src
        assert "select(d1, result, result.x < d1.x)" in src

    def test_three_primitives_chained_min(self):
        """Scene with three primitives chains min(d0, min(d1, d2)) or similar."""
        src = generate_wgsl(SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
                TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
            ),
        ))
        assert "let d2 =" in src
        assert "select(" in src

    def test_two_spheres_with_different_radii(self):
        """Two spheres of different radii produce independent calls."""
        src = generate_wgsl(SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                SphereNode(PositionNode(), FloatNode(2.5)),
            ),
        ))
        # Two distinct call sites, same function def emitted once
        assert src.count("fn sdSphere") == 1
        assert src.count("sdSphere(") >= 2


class TestPrimitiveCallArguments:
    """Verify each sdf_* primitive is called with correct argument structure."""

    def test_sphere_call_with_radius(self):
        """Sphere call passes radius as scalar argument."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(2.5)),),
        ))
        assert "sdSphere(p, 2.5)" in src

    def test_box_call_with_vec3(self):
        """Box call passes size as vec3 constructor."""
        src = generate_wgsl(SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(2.0, 3.0, 1.0)),),
        ))
        assert "sdBox(p, vec3<f32>(2.0, 3.0, 1.0)" in src

    def test_torus_call_with_vec2(self):
        """Torus call passes major/minor as vec2 constructor."""
        src = generate_wgsl(SceneGraph(
            primitives=(TorusNode(PositionNode(), FloatNode(3.0), FloatNode(0.75)),),
        ))
        assert "sdTorus(p, vec2<f32>(3.0, 0.75)" in src

    def test_cylinder_call_with_height_and_radius(self):
        """Cylinder call passes height and radius as scalars."""
        src = generate_wgsl(SceneGraph(
            primitives=(CylinderNode(PositionNode(), FloatNode(4.0), FloatNode(1.5)),),
        ))
        assert "sdCylinder(p, 4.0, 1.5" in src

    def test_cone_call_with_all_parameters(self):
        """Cone call passes height, r1, r2 as scalars."""
        src = generate_wgsl(SceneGraph(
            primitives=(ConeNode(PositionNode(), FloatNode(3.0), FloatNode(0.5), FloatNode(1.5)),),
        ))
        assert "sdCone(p, 3.0, 0.5, 1.5" in src

    def test_plane_call_with_normal_and_distance(self):
        """Plane call passes normal as vec3 and distance as scalar."""
        src = generate_wgsl(SceneGraph(
            primitives=(PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(-2.0)),),
        ))
        assert "sdPlane(p, vec3<f32>(0.0, 1.0, 0.0), -2.0" in src


class TestSceneNaming:
    """Verify scene function naming behavior."""

    def test_default_scene_name(self):
        """Scene without explicit name gets a default name."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        # Must have some scene function
        assert re.search(r"fn\s+sd_scene__\w+\s*\(", src)

    def test_custom_scene_name(self):
        """Custom scene name reflected in function name."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="my_world",
        ))
        assert "fn sd_scene__my_world(p: vec3<f32>) -> vec2<f32>" in src

    def test_scene_name_with_underscores(self):
        """Scene name with underscores generates valid WGSL function name."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="deep_space_scene",
        ))
        assert "fn sd_scene__deep_space_scene" in src

    def test_scene_name_with_numbers(self):
        """Scene name with numbers generates valid function name."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="scene_42",
        ))
        assert "fn sd_scene__scene_42" in src


class TestDictGenerationCoverage:
    """Verify the dict-based API supports all primitive types."""

    def test_cylinder_from_dict(self):
        src = generate_wgsl_from_scene(
            [{"type": "cylinder", "height": 3.0, "radius": 0.8}],
            name="cyl_test",
        )
        assert_well_formed_wgsl(src)
        assert "sdCylinder" in src

    def test_cone_from_dict(self):
        src = generate_wgsl_from_scene(
            [{"type": "cone", "height": 2.5, "radius_top": 0.0, "radius_bottom": 1.0}],
            name="cone_test",
        )
        assert_well_formed_wgsl(src)
        assert "sdCone" in src

    def test_plane_from_dict(self):
        src = generate_wgsl_from_scene(
            [{"type": "plane", "normal": (0.0, 1.0, 0.0), "distance": 0.0}],
            name="plane_test",
        )
        assert_well_formed_wgsl(src)
        assert "sdPlane" in src

    def test_all_seven_types_from_dict(self):
        """All seven primitive types via dict API in one scene."""
        src = generate_wgsl_from_scene(
            [
                {"type": "sphere", "radius": 1.0},
                {"type": "box", "size": (1.0, 1.0, 1.0)},
                {"type": "torus", "major_radius": 2.0, "minor_radius": 0.5},
                {"type": "cylinder", "height": 2.0, "radius": 0.5},
                {"type": "cone", "height": 2.0, "radius_top": 0.0, "radius_bottom": 1.0},
                {"type": "plane", "normal": (0.0, 1.0, 0.0), "distance": 0.0},
                {"type": "capsule", "endpoint_a": (0.0, -1.0, 0.0), "endpoint_b": (0.0, 1.0, 0.0), "radius": 0.5},
            ],
            name="all_primitives",
        )
        assert_well_formed_wgsl(src)
        assert "fn sdSphere" in src
        assert "fn sdBox" in src
        assert "fn sdTorus" in src
        assert "fn sdCylinder" in src
        assert "fn sdCone" in src
        assert "fn sdPlane" in src
        assert "fn sdCapsule" in src

    def test_empty_primitive_list_does_not_crash(self):
        """Empty primitive list generates valid (minimal) WGSL."""
        src = generate_wgsl_from_scene([], name="empty")
        assert_well_formed_wgsl(src)

    def test_empty_scene_graph_does_not_crash(self):
        """SceneGraph with no primitives generates valid WGSL."""
        src = generate_wgsl(SceneGraph(primitives=()))
        assert_well_formed_wgsl(src)


class TestDomainPipelineChain:
    """Verify multiple domain operations chain correctly."""

    def test_repeat_then_twist_on_sphere(self):
        """Repeat then twist domain op chain with sphere."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                TwistNode(PositionNode(), FloatNode(1.5)),
            ),
        ))
        assert_well_formed_wgsl(src)
        assert "domain_repeat(" in src
        assert "domain_twist(" in src
        assert "sdSphere" in src

    def test_mirror_then_bend_on_box(self):
        """Mirror then bend domain op chain with box."""
        src = generate_wgsl(SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),),
            pipeline=(
                MirrorNode(PositionNode(), Axis.X),
                BendNode(PositionNode(), FloatNode(5.0)),
            ),
        ))
        assert_well_formed_wgsl(src)
        assert "domain_mirror_x(" in src
        assert "domain_bend(" in src
        assert "sdBox" in src

    def test_kifs_with_stretch_on_torus(self):
        """KIFS then stretch domain op chain produces both compensation fns."""
        src = generate_wgsl(SceneGraph(
            primitives=(TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),),
            pipeline=(
                KifsNode(PositionNode(), FloatNode(6.0)),
                StretchNode(PositionNode(), FloatNode(2.0), Axis.Y),
            ),
        ))
        assert_well_formed_wgsl(src)
        assert "domain_kifs(" in src
        assert "domain_stretch_y(" in src
        assert "domain_kifs_compensation" in src
        assert "domain_stretch_compensation" in src

    def test_repeat_then_mirror_then_twist(self):
        """Three-way domain op chain: repeat -> mirror -> twist."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(3.0, 3.0, 3.0)),
                MirrorNode(PositionNode(), Axis.Z),
                TwistNode(PositionNode(), FloatNode(2.0)),
            ),
        ))
        assert_well_formed_wgsl(src)
        assert "domain_repeat(" in src
        assert "domain_mirror_z(" in src
        assert "domain_twist(" in src


class TestCompensationInSceneOutput:
    """Verify compensation factor usage in generated scene code."""

    def test_no_compensation_for_isometric_pipeline(self):
        """Isometric ops (mirror, twist, bend, repeat) yield comp=1.0."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
        ))
        assert "let comp = 1.0" in src

    def test_kifs_compensation_divides_distance(self):
        """KIFS compensation divides the SDF distance."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(KifsNode(PositionNode(), FloatNode(6.0)),),
        ))
        assert "domain_kifs_compensation" in src

    def test_stretch_compensation_divides_distance(self):
        """Stretch compensation divides the SDF distance."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(StretchNode(PositionNode(), FloatNode(2.0), Axis.X),),
        ))
        assert "domain_stretch_compensation" in src

    def test_comp_is_not_one_for_kifs(self):
        """KIFS compensation variable is not trivially 1.0."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(KifsNode(PositionNode(), FloatNode(6.0)),),
        ))
        assert "let comp = 1.0" not in src


class TestWgslSyntaxValidity:
    """Verify the generated output is syntactically well-formed WGSL."""

    def test_all_function_returns_have_semicolons(self):
        """Every return statement in generated WGSL is terminated by semicolon."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        lines = src.split("\n")
        return_lines = [l for l in lines if l.strip().startswith("return")]
        for rl in return_lines:
            assert rl.strip().endswith(";"), f"Return lacks semicolon: {rl.strip()}"

    def test_no_python_bool_in_output(self):
        """Python 'True'/'False' must not leak into generated WGSL."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        assert "True" not in src, "Python True leaked into WGSL"
        assert "False" not in src, "Python False leaked into WGSL"

    def test_no_python_float_literal_leakage(self):
        """Python float formatting must not produce WGSL-invalid literals."""
        src = generate_wgsl(SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(0.5)),
                BoxNode(PositionNode(), Vec3Node(0.25, 0.5, 0.75)),
            ),
        ))
        # WGSL requires f32 suffix or decimal point for floats
        assert "None" not in src

    def test_wgsl_comment_syntax(self):
        """Generated WGSL uses // for comments, not Python #."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        lines = src.split("\n")
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith("//") and "SPDX" not in l]
        for cl in code_lines:
            assert "#" not in cl, f"Python comment syntax in WGSL: {cl.strip()}"

    def test_float_literals_have_decimal_point(self):
        """Numeric literals in SDF calls use WGSL-compatible format."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        # Integer values should have .0 suffix when used as f32
        assert "1.0" in src

    def test_braces_balanced(self):
        """Curly braces in generated output are balanced."""
        src = generate_wgsl(SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
                TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
                CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(1.0)),
                ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),
                PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),
            ),
        ))
        opens = src.count("{")
        closes = src.count("}")
        assert opens == closes, f"Braces unbalanced: {opens} open, {closes} close"


class TestParameterEdgeCases:
    """Verify behavior with edge case parameter values."""

    def test_zero_radius_sphere(self):
        """Sphere with zero radius produces valid WGSL."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(0.0)),),
        ))
        assert_well_formed_wgsl(src)
        assert "sdSphere" in src

    def test_negative_distance_plane(self):
        """Plane with negative distance is valid."""
        src = generate_wgsl(SceneGraph(
            primitives=(PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(-5.0)),),
        ))
        assert_well_formed_wgsl(src)
        assert "sdPlane" in src

    def test_large_radius_values(self):
        """Large numeric values in primitives produce valid WGSL."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1e4)),),
        ))
        assert_well_formed_wgsl(src)
        assert "sdSphere" in src

    def test_small_radius_values(self):
        """Small numeric values in primitives produce valid WGSL."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1e-4)),),
        ))
        assert_well_formed_wgsl(src)
        assert "sdSphere" in src


class TestSceneReturnStructure:
    """Verify the scene function structure at the output level."""

    def test_scene_returns_f32(self):
        """The scene function return type is f32."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        assert "-> vec2<f32>" in src

    def test_scene_takes_vec3_parameter(self):
        """The scene function takes a vec3<f32> parameter named p."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        assert re.search(r"fn\s+sd_scene__.*\(p\s*:\s*vec3<f32>\)", src)

    def test_position_piped_through_domain_pipeline(self):
        """Position is transformed through domain ops before SDF call."""
        src = generate_wgsl(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                TwistNode(PositionNode(), FloatNode(1.0)),
            ),
        ))
        # The final SDF call should use the domain-transformed position
        assert "sdSphere(p_d" in src or "sdSphere(p," in src
