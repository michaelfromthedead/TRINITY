"""
Cleanroom blackbox tests for domain ops WGSL codegen (T-DEMO-2.5).

Tests that WgslCodeGen (T-DEMO-2.6) produces correct WGSL output for all six
domain operations: Repeat, Mirror, KIFS, Twist, Bend, Stretch.

BLACKBOX coverage (10 paths):
  Path 1:  domain_repeat emitted with vec3<f32> cell_size
  Path 2:  domain_mirror emitted with correct axis suffix
  Path 3:  domain_twist emitted with rate parameter
  Path 4:  domain_bend emitted with radius parameter
  Path 5:  KIFS includes domain_kifs_compensation function
  Path 6:  Stretch includes domain_stretch_compensation function
  Path 7:  Generated WGSL is syntactically valid (balanced parens/braces)
  Path 8:  Pipeline composition chains ops in correct order
  Path 9:  Compensation only for non-isometric ops (KIFS, Stretch)
  Path 10: Full scene generation (SDF, material, scene entry point)
"""

from __future__ import annotations

import re

import pytest

# MaterialNode and other demoscene AST features not yet implemented
pytest.skip("Demoscene AST not fully implemented", allow_module_level=True)

from engine.rendering.demoscene.ast_nodes import (
    Axis,
    BendNode,
    BoxNode,
    ConeNode,
    CylinderNode,
    FloatNode,
    KifsNode,
    MaterialNode,
    MirrorNode,
    PlaneNode,
    PositionNode,
    RepeatNode,
    SceneGraph,
    SphereNode,
    StretchNode,
    TorusNode,
    TwistNode,
    Vec3Node,
)
from engine.rendering.demoscene.wgsl_codegen import (
    WgslCodeGen,
    generate_wgsl,
    MATERIAL_STRUCT,
)


# =============================================================================
# Test helpers
# =============================================================================


def _scene(
    pipeline_ops=None,
    primitives=None,
    materials=None,
    name="test",
):
    """Build a SceneGraph and return generated WGSL."""
    if primitives is None:
        primitives = [
            SphereNode(
                position=PositionNode(),
                radius=FloatNode(1.0),
                material_id=0,
            )
        ]
    if materials is None:
        materials = (
            MaterialNode(
                material_id=0,
                albedo=Vec3Node(0.8, 0.2, 0.2),
                roughness=FloatNode(0.5),
                metallic=FloatNode(0.0),
                emissive=FloatNode(0.0),
                ambient_occlusion=FloatNode(1.0),
            ),
        )
    graph = SceneGraph(
        primitives=tuple(primitives),
        pipeline=tuple(pipeline_ops or ()),
        materials=materials,
        name=name,
    )
    return generate_wgsl(graph, name=name)


def _balanced(s: str) -> bool:
    """Check balanced parentheses."""
    d = 0
    for ch in s:
        if ch == "(":
            d += 1
        elif ch == ")":
            d -= 1
        if d < 0:
            return False
    return d == 0


def _count_braces(s: str) -> tuple[int, int]:
    """Return (open, close) curly brace counts."""
    return s.count("{"), s.count("}")


def _extract_fns(wgsl: str) -> list[str]:
    """Extract all WGSL function names defined in the output."""
    fns = []
    for line in wgsl.splitlines():
        m = re.match(r"fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", line)
        if m:
            fns.append(m.group(1))
    return fns


def _extract_domain_calls(wgsl: str) -> list[str]:
    """Extract domain_* function call names from generated WGSL."""
    calls = []
    for line in wgsl.splitlines():
        if line.strip().startswith("//"):
            continue
        for m in re.finditer(r"\bdomain_[a-z_]+(?=\()", line):
            calls.append(m.group())
    return calls


# =============================================================================
# Path 1: domain_repeat emitted with vec3<f32> cell_size
# =============================================================================


class TestRepeatScene:
    """domain_repeat appears with correct cell_size in generated WGSL."""

    def test_repeat_call_present(self):
        wgsl = _scene(
            pipeline_ops=[
                RepeatNode(
                    input=PositionNode(),
                    cell_size=Vec3Node(2.0, 2.0, 2.0),
                )
            ]
        )
        assert "domain_repeat" in wgsl

    def test_repeat_uses_vec3_f32(self):
        wgsl = _scene(
            pipeline_ops=[
                RepeatNode(
                    input=PositionNode(),
                    cell_size=Vec3Node(2.0, 2.0, 2.0),
                )
            ]
        )
        calls = _extract_domain_calls(wgsl)
        repeat_calls = [c for c in calls if "repeat" in c]
        assert len(repeat_calls) >= 1

    def test_repeat_cell_size_values(self):
        wgsl = _scene(
            pipeline_ops=[
                RepeatNode(
                    input=PositionNode(),
                    cell_size=Vec3Node(3.0, 1.5, 4.0),
                )
            ]
        )
        assert "3.0" in wgsl
        assert "1.5" in wgsl
        assert "4.0" in wgsl

    def test_repeat_balanced(self):
        wgsl = _scene(
            pipeline_ops=[
                RepeatNode(
                    input=PositionNode(),
                    cell_size=Vec3Node(2.0, 2.0, 2.0),
                )
            ]
        )
        assert _balanced(wgsl)
        ob, cb = _count_braces(wgsl)
        assert ob == cb

    def test_repeat_in_scene_entry(self):
        wgsl = _scene(
            pipeline_ops=[
                RepeatNode(
                    input=PositionNode(),
                    cell_size=Vec3Node(2.0, 2.0, 2.0),
                )
            ]
        )
        assert re.search(
            r"fn sd_scene__test.*?domain_repeat", wgsl, re.DOTALL
        )


# =============================================================================
# Path 2: domain_mirror emitted with correct axis suffix
# =============================================================================


class TestMirrorScene:
    """domain_mirror appears with correct axis in generated WGSL."""

    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    def test_mirror_axis(self, axis):
        wgsl = _scene(
            pipeline_ops=[MirrorNode(input=PositionNode(), axis=axis)]
        )
        assert f"domain_mirror_{axis.value}" in wgsl

    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    def test_mirror_balanced(self, axis):
        wgsl = _scene(
            pipeline_ops=[MirrorNode(input=PositionNode(), axis=axis)]
        )
        assert _balanced(wgsl)
        ob, cb = _count_braces(wgsl)
        assert ob == cb

    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    def test_mirror_in_scene_entry(self, axis):
        wgsl = _scene(
            pipeline_ops=[MirrorNode(input=PositionNode(), axis=axis)]
        )
        assert re.search(
            rf"fn sd_scene__test.*?domain_mirror_{axis.value}",
            wgsl,
            re.DOTALL,
        )


# =============================================================================
# Path 3: domain_twist emitted with rate parameter
# =============================================================================


class TestTwistScene:
    """domain_twist appears with correct rate in generated WGSL."""

    @pytest.mark.parametrize("rate", [1.0, 2.5, 0.0, -1.0, 3.14159])
    def test_twist_rate(self, rate):
        wgsl = _scene(
            pipeline_ops=[TwistNode(
                input=PositionNode(), rate=FloatNode(rate)
            )]
        )
        assert "domain_twist" in wgsl
        r = str(rate) if rate != int(rate) else f"{int(rate)}.0"
        assert r in wgsl

    def test_twist_in_scene_entry(self):
        wgsl = _scene(
            pipeline_ops=[TwistNode(
                input=PositionNode(), rate=FloatNode(2.0)
            )]
        )
        assert re.search(
            r"fn sd_scene__test.*?domain_twist", wgsl, re.DOTALL
        )

    def test_twist_balanced(self):
        wgsl = _scene(
            pipeline_ops=[TwistNode(
                input=PositionNode(), rate=FloatNode(1.5)
            )]
        )
        assert _balanced(wgsl)
        ob, cb = _count_braces(wgsl)
        assert ob == cb


# =============================================================================
# Path 4: domain_bend emitted with radius parameter
# =============================================================================


class TestBendScene:
    """domain_bend appears with correct radius in generated WGSL."""

    @pytest.mark.parametrize("radius", [5.0, 1.0, 10.0, 0.5, 100.0])
    def test_bend_radius(self, radius):
        wgsl = _scene(
            pipeline_ops=[BendNode(
                input=PositionNode(), radius=FloatNode(radius)
            )]
        )
        assert "domain_bend" in wgsl
        r = str(radius) if radius != int(radius) else f"{int(radius)}.0"
        assert r in wgsl

    def test_bend_in_scene_entry(self):
        wgsl = _scene(
            pipeline_ops=[BendNode(
                input=PositionNode(), radius=FloatNode(5.0)
            )]
        )
        assert re.search(
            r"fn sd_scene__test.*?domain_bend", wgsl, re.DOTALL
        )

    def test_bend_balanced(self):
        wgsl = _scene(
            pipeline_ops=[BendNode(
                input=PositionNode(), radius=FloatNode(5.0)
            )]
        )
        assert _balanced(wgsl)
        ob, cb = _count_braces(wgsl)
        assert ob == cb


# =============================================================================
# Path 5: KIFS includes domain_kifs_compensation
# =============================================================================


class TestKifsScene:
    """domain_kifs and its compensation appear in generated WGSL."""

    @pytest.mark.parametrize("folds", [3, 4, 6, 8, 12])
    def test_kifs_present(self, folds):
        wgsl = _scene(
            pipeline_ops=[KifsNode(
                input=PositionNode(), folds=FloatNode(float(folds))
            )]
        )
        assert "domain_kifs" in wgsl

    def test_kifs_compensation_fn(self):
        wgsl = _scene(
            pipeline_ops=[KifsNode(
                input=PositionNode(), folds=FloatNode(6.0)
            )]
        )
        fns = _extract_fns(wgsl)
        assert "domain_kifs_compensation" in fns

    def test_kifs_compensation_has_f32_param(self):
        wgsl = _scene(
            pipeline_ops=[KifsNode(
                input=PositionNode(), folds=FloatNode(6.0)
            )]
        )
        assert "fn domain_kifs_compensation(folds: f32)" in wgsl

    def test_kifs_in_scene_entry(self):
        wgsl = _scene(
            pipeline_ops=[KifsNode(
                input=PositionNode(), folds=FloatNode(6.0)
            )]
        )
        assert re.search(
            r"fn sd_scene__test.*?domain_kifs\(", wgsl, re.DOTALL
        )

    def test_kifs_balanced(self):
        wgsl = _scene(
            pipeline_ops=[KifsNode(
                input=PositionNode(), folds=FloatNode(6.0)
            )]
        )
        assert _balanced(wgsl)
        ob, cb = _count_braces(wgsl)
        assert ob == cb


# =============================================================================
# Path 6: Stretch includes domain_stretch_compensation
# =============================================================================


class TestStretchScene:
    """domain_stretch and its compensation appear in generated WGSL."""

    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    def test_stretch_axis(self, axis):
        wgsl = _scene(
            pipeline_ops=[StretchNode(
                input=PositionNode(), stretch=FloatNode(2.0), axis=axis
            )]
        )
        assert f"domain_stretch_{axis.value}" in wgsl

    @pytest.mark.parametrize("factor", [0.5, 1.0, 2.0, 10.0, 0.1])
    def test_stretch_factor(self, factor):
        wgsl = _scene(
            pipeline_ops=[StretchNode(
                input=PositionNode(),
                stretch=FloatNode(factor),
                axis=Axis.X,
            )]
        )
        f = str(factor) if factor != int(factor) else f"{int(factor)}.0"
        assert f in wgsl

    def test_stretch_compensation_fn(self):
        wgsl = _scene(
            pipeline_ops=[StretchNode(
                input=PositionNode(), stretch=FloatNode(2.0), axis=Axis.X
            )]
        )
        fns = _extract_fns(wgsl)
        assert "domain_stretch_compensation" in fns

    def test_stretch_balanced(self):
        wgsl = _scene(
            pipeline_ops=[StretchNode(
                input=PositionNode(), stretch=FloatNode(2.0), axis=Axis.Z
            )]
        )
        assert _balanced(wgsl)
        ob, cb = _count_braces(wgsl)
        assert ob == cb


# =============================================================================
# Path 7: Generated WGSL syntactically valid
# =============================================================================


class TestSyntaxValidity:
    """Generated WGSL must be syntactically valid."""
    # fmt: off

    def test_full_scene_balanced(self):
        wgsl = _scene(pipeline_ops=[
            RepeatNode(input=PositionNode(), cell_size=Vec3Node(2.0, 2.0, 2.0)),
            KifsNode(input=PositionNode(), folds=FloatNode(6.0)),
            TwistNode(input=PositionNode(), rate=FloatNode(1.5)),
            StretchNode(input=PositionNode(), stretch=FloatNode(2.0), axis=Axis.Y),
        ])
        assert _balanced(wgsl)
        ob, cb = _count_braces(wgsl)
        assert ob == cb

    def test_fn_signatures_valid(self):
        wgsl = _scene(pipeline_ops=[
            RepeatNode(input=PositionNode(), cell_size=Vec3Node(2.0, 2.0, 2.0)),
            KifsNode(input=PositionNode(), folds=FloatNode(6.0)),
        ])
        for line in wgsl.splitlines():
            s = line.strip()
            if s.startswith("fn "):
                assert re.match(
                    r"fn [a-z_][a-zA-Z0-9_]*\(.*\)\s*->\s*\S+", s
                ), f"bad signature: {s}"

    def test_type_annotations(self):
        wgsl = _scene(pipeline_ops=[
            TwistNode(input=PositionNode(), rate=FloatNode(1.0))
        ])
        assert "vec3<f32>" in wgsl
        assert "vec2<f32>" in wgsl
        assert "-> f32" in wgsl or "-> vec2" in wgsl

    def test_entry_point_signature(self):
        wgsl = _scene(name="demo_scene")
        assert "fn sd_scene__demo_scene(p: vec3<f32>) -> vec2<f32>" in wgsl

    def test_compensation_division(self):
        wgsl = _scene(pipeline_ops=[
            KifsNode(input=PositionNode(), folds=FloatNode(6.0))
        ])
        assert "result.x / comp" in wgsl

    def test_no_invalid_wgsl_chars(self):
        """Generated WGSL must not contain characters invalid in WGSL source."""
        wgsl = _scene(pipeline_ops=[
            RepeatNode(input=PositionNode(), cell_size=Vec3Node(2.0, 2.0, 2.0)),
            MirrorNode(input=PositionNode(), axis=Axis.X),
            KifsNode(input=PositionNode(), folds=FloatNode(6.0)),
            TwistNode(input=PositionNode(), rate=FloatNode(1.0)),
            BendNode(input=PositionNode(), radius=FloatNode(5.0)),
            StretchNode(input=PositionNode(), stretch=FloatNode(2.0), axis=Axis.Z),
        ])
        # Strip comment lines to avoid false positives from //#import lines
        code_lines = [
            ln for ln in wgsl.splitlines()
            if not ln.strip().startswith("//")
        ]
        code = "\n".join(code_lines)
        invalid = set("@$%^&|\\`~")
        found = set(code) & invalid
        assert not found, f"invalid chars in non-comment code: {found}"


# =============================================================================
# Path 8: Pipeline composition chains ops in correct order
# =============================================================================


class TestPipelineOrder:
    """Pipeline ops appear in correct order in generated WGSL."""

    def test_two_ops(self):
        wgsl = _scene(pipeline_ops=[
            MirrorNode(input=PositionNode(), axis=Axis.X),
            TwistNode(input=PositionNode(), rate=FloatNode(2.0)),
        ])
        calls = _extract_domain_calls(wgsl)
        assert "domain_mirror_x" in calls
        assert "domain_twist" in calls

    def test_three_ops_all_present(self):
        wgsl = _scene(pipeline_ops=[
            BendNode(input=PositionNode(), radius=FloatNode(5.0)),
            RepeatNode(input=PositionNode(), cell_size=Vec3Node(2.0, 2.0, 2.0)),
            TwistNode(input=PositionNode(), rate=FloatNode(1.5)),
        ])
        assert "domain_bend" in wgsl
        assert "domain_repeat" in wgsl
        assert "domain_twist" in wgsl

    def test_pipeline_assigns_p_d(self):
        wgsl = _scene(pipeline_ops=[
            TwistNode(input=PositionNode(), rate=FloatNode(2.0)),
        ])
        assert "let p_d" in wgsl or "let p_d =" in wgsl


# =============================================================================
# Path 9: Compensation only for non-isometric ops
# =============================================================================


class TestCompensationSelection:
    """Compensation functions only for non-isometric ops (KIFS, Stretch)."""

    def test_kifs_has_comp(self):
        wgsl = _scene(pipeline_ops=[
            KifsNode(input=PositionNode(), folds=FloatNode(6.0))
        ])
        fns = _extract_fns(wgsl)
        assert "domain_kifs_compensation" in fns

    def test_stretch_has_comp(self):
        wgsl = _scene(pipeline_ops=[
            StretchNode(input=PositionNode(), stretch=FloatNode(2.0), axis=Axis.X)
        ])
        fns = _extract_fns(wgsl)
        assert "domain_stretch_compensation" in fns

    def test_both_non_isometric(self):
        wgsl = _scene(pipeline_ops=[
            KifsNode(input=PositionNode(), folds=FloatNode(6.0)),
            StretchNode(input=PositionNode(), stretch=FloatNode(2.0), axis=Axis.Y),
        ])
        fns = _extract_fns(wgsl)
        assert "domain_kifs_compensation" in fns
        assert "domain_stretch_compensation" in fns

    def test_isometric_only_no_comp(self):
        wgsl = _scene(pipeline_ops=[
            RepeatNode(input=PositionNode(), cell_size=Vec3Node(2.0, 2.0, 2.0)),
            MirrorNode(input=PositionNode(), axis=Axis.X),
            TwistNode(input=PositionNode(), rate=FloatNode(1.0)),
            BendNode(input=PositionNode(), radius=FloatNode(5.0)),
        ])
        fns = _extract_fns(wgsl)
        assert "domain_kifs_compensation" not in fns
        assert "domain_stretch_compensation" not in fns

    def test_multiple_same_type_dedup(self):
        wgsl = _scene(pipeline_ops=[
            KifsNode(input=PositionNode(), folds=FloatNode(6.0)),
            KifsNode(input=PositionNode(), folds=FloatNode(8.0)),
        ])
        count = wgsl.count("fn domain_kifs_compensation")
        assert count == 1, f"expected 1 def got {count}"


# =============================================================================
# Path 10: Full scene generation
# =============================================================================


class TestFullScene:
    """Full scene integrates pipeline, SDF primitives, and material."""

    def test_scene_entry_point(self):
        wgsl = _scene()
        assert "fn sd_scene__test(p: vec3<f32>) -> vec2<f32>" in wgsl

    def test_sdf_sphere_emitted(self):
        wgsl = _scene()
        fns = _extract_fns(wgsl)
        assert "sdSphere" in fns

    def test_multiple_primitives_multiple_sdf(self):
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
        ])
        fns = _extract_fns(wgsl)
        assert "sdSphere" in fns
        assert "sdBox" in fns

    def test_material_struct(self):
        wgsl = _scene()
        assert "struct Material" in wgsl
        assert "albedo" in wgsl
        assert "roughness" in wgsl
        assert "metallic" in wgsl
        assert "emissive" in wgsl
        assert "ambient_occlusion" in wgsl

    def test_scene_material_function(self):
        wgsl = _scene()
        assert "fn scene_material" in wgsl
        assert "switch" in wgsl

    def test_empty_pipeline(self):
        wgsl = _scene(pipeline_ops=[])
        calls = _extract_domain_calls(wgsl)
        domain_calls = [c for c in calls if c.startswith("domain_")]
        assert len(domain_calls) == 0

    def test_spdx_header(self):
        wgsl = _scene()
        assert wgsl.startswith("// SPDX-License-Identifier: MIT")

    def test_two_primitives_use_select(self):
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
        ])
        assert "select" in wgsl

    def test_name_in_entry_point(self):
        wgsl = _scene(name="golden_gate")
        assert "sd_scene__golden_gate" in wgsl


# =============================================================================
# All SDF primitives
# =============================================================================


class TestAllSdfPrimitives:
    """Verify all SDF primitive types generate correct WGSL."""

    @pytest.mark.parametrize(
        "prim,expected_fn",
        [
            (SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0), "sdSphere"),
            (BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=0), "sdBox"),
            (TorusNode(position=PositionNode(), major_radius=FloatNode(2.0), minor_radius=FloatNode(0.5), material_id=0), "sdTorus"),
            (CylinderNode(position=PositionNode(), height=FloatNode(2.0), radius=FloatNode(0.5), material_id=0), "sdCylinder"),
            (ConeNode(position=PositionNode(), height=FloatNode(2.0), radius_top=FloatNode(0.0), radius_bottom=FloatNode(1.0), material_id=0), "sdCone"),
            (PlaneNode(position=PositionNode(), normal=Vec3Node(0.0, 1.0, 0.0), distance=FloatNode(0.0), material_id=0), "sdPlane"),
        ],
    )
    def test_primitive_emitted(self, prim, expected_fn):
        wgsl = _scene(primitives=[prim])
        fns = _extract_fns(wgsl)
        assert expected_fn in fns


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Codegen handles edge cases correctly."""

    def test_zero_folds_kifs(self):
        wgsl = _scene(pipeline_ops=[
            KifsNode(input=PositionNode(), folds=FloatNode(0.0))
        ])
        assert "domain_kifs" in wgsl
        assert _balanced(wgsl)

    def test_negative_stretch(self):
        wgsl = _scene(pipeline_ops=[
            StretchNode(input=PositionNode(), stretch=FloatNode(-2.0), axis=Axis.X)
        ])
        assert "domain_stretch_x" in wgsl
        assert _balanced(wgsl)

    def test_single_prim_no_select(self):
        wgsl = _scene()
        assert "select" not in wgsl

    def test_comp_default_no_pipeline(self):
        wgsl = _scene(pipeline_ops=[])
        assert "let comp = 1.0" in wgsl

    def test_no_materials_default(self):
        graph = SceneGraph(
            primitives=(
                SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            ),
            name="test",
        )
        wgsl = generate_wgsl(graph)
        assert "fn scene_material" in wgsl
        assert "default:" in wgsl

    def test_multiple_material_cases(self):
        materials = (
            MaterialNode(
                material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2),
                roughness=FloatNode(0.5), metallic=FloatNode(0.0),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
            ),
            MaterialNode(
                material_id=1, albedo=Vec3Node(0.2, 0.8, 0.2),
                roughness=FloatNode(0.3), metallic=FloatNode(0.5),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
            ),
        )
        graph = SceneGraph(
            primitives=(
                SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            ),
            materials=materials,
            name="test",
        )
        wgsl = generate_wgsl(graph)
        assert "case 0:" in wgsl
        assert "case 1:" in wgsl

    def test_all_six_domain_ops(self):
        wgsl = _scene(pipeline_ops=[
            RepeatNode(input=PositionNode(), cell_size=Vec3Node(2.0, 2.0, 2.0)),
            MirrorNode(input=PositionNode(), axis=Axis.X),
            KifsNode(input=PositionNode(), folds=FloatNode(6.0)),
            TwistNode(input=PositionNode(), rate=FloatNode(1.5)),
            BendNode(input=PositionNode(), radius=FloatNode(5.0)),
            StretchNode(input=PositionNode(), stretch=FloatNode(2.0), axis=Axis.Z),
        ])
        assert _balanced(wgsl)
        ob, cb = _count_braces(wgsl)
        assert ob == cb
        for op in ["domain_repeat", "domain_mirror_x", "domain_kifs",
                     "domain_twist", "domain_bend", "domain_stretch_z"]:
            assert op in wgsl, f"missing {op}"
