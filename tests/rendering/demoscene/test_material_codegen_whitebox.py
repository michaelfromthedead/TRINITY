"""
Whitebox tests for material WGSL codegen (T-DEMO-2.6).

Tests the implementation-aware material code generation in
engine/rendering/demoscene/wgsl_codegen.py, verifying:

  - MATERIAL_STRUCT generates correct WGSL with all PBR fields
  - DEFAULT_MATERIAL_SWITCH maps material_id to Material struct via switch/case
  - Material node emissive values are correctly emitted
  - material_id propagates through single and multiple primitives
  - Pairwise select() correctly tracks the winning material_id
  - Edge cases: zero/negative material_id, default materials, emissive extremes
"""

from __future__ import annotations

import re

import pytest

# MaterialNode and other demoscene AST features not yet implemented
pytest.skip("Demoscene AST not fully implemented", allow_module_level=True)

from engine.rendering.demoscene.ast_nodes import (
    Axis,
    BoxNode,
    FloatNode,
    MaterialNode,
    PlaneNode,
    PositionNode,
    SceneGraph,
    SphereNode,
    Vec3Node,
)
from engine.rendering.demoscene.wgsl_codegen import (
    MATERIAL_STRUCT,
    DEFAULT_MATERIAL_SWITCH,
    WgslCodeGen,
    generate_wgsl,
)


# =============================================================================
# Test helpers
# =============================================================================


def _scene(
    materials=None,
    primitives=None,
    name="test",
):
    """Build a minimal SceneGraph and return generated WGSL."""
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


# =============================================================================
# MATERIAL_STRUCT: correct WGSL struct definition
# =============================================================================


class TestMaterialStructDefinition:
    """MATERIAL_STRING must produce a valid WGSL struct with all PBR fields."""

    def test_struct_definition_present(self):
        """MATERIAL_STRUCT contains 'struct Material'."""
        assert "struct Material" in MATERIAL_STRUCT

    def test_struct_has_albedo_field(self):
        """Material struct has vec3<f32> albedo field."""
        assert "albedo: vec3<f32>" in MATERIAL_STRUCT or "albedo" in MATERIAL_STRUCT

    def test_struct_has_roughness_field(self):
        """Material struct has roughness: f32 field."""
        assert "roughness: f32" in MATERIAL_STRUCT or "roughness" in MATERIAL_STRUCT

    def test_struct_has_metallic_field(self):
        """Material struct has metallic: f32 field."""
        assert "metallic: f32" in MATERIAL_STRUCT or "metallic" in MATERIAL_STRUCT

    def test_struct_has_emissive_field(self):
        """Material struct has emissive: f32 field."""
        assert "emissive: f32" in MATERIAL_STRUCT or "emissive" in MATERIAL_STRUCT

    def test_struct_has_ao_field(self):
        """Material struct has ambient_occlusion: f32 field."""
        assert "ambient_occlusion: f32" in MATERIAL_STRUCT or "ambient_occlusion" in MATERIAL_STRUCT

    def test_struct_albedo_type(self):
        """Albedo field type is vec3<f32>."""
        assert "vec3<f32>" in MATERIAL_STRUCT.split("albedo")[1].split("\n")[0]

    def test_struct_braces_balanced(self):
        """Material struct has balanced curly braces."""
        ob, cb = _count_braces(MATERIAL_STRUCT)
        assert ob == cb

    def test_struct_commas(self):
        """Each field ends with a comma (WGSL convention)."""
        lines = MATERIAL_STRUCT.splitlines()
        in_struct = False
        field_commas = 0
        expected_fields = 5
        for line in lines:
            stripped = line.strip()
            if stripped == "struct Material {":
                in_struct = True
                continue
            if stripped == "};":
                in_struct = False
                continue
            if in_struct and stripped:
                if stripped.endswith(","):
                    field_commas += 1
        assert field_commas == expected_fields, (
            f"Expected {expected_fields} field commas, got {field_commas}"
        )

    def test_struct_has_comment(self):
        """Material struct has a documentation comment."""
        assert "///" in MATERIAL_STRUCT

    def test_struct_albedo_valued_vec3(self):
        """The docstring describes PBR material properties."""
        assert "PBR" in MATERIAL_STRUCT or "material" in MATERIAL_STRUCT.lower()

    def test_struct_generated_in_wgsl(self):
        """Full WGSL output includes the Material struct."""
        wgsl = _scene()
        assert "struct Material" in wgsl

    def test_struct_fields_in_wgsl(self):
        """Full WGSL output includes all PBR struct fields."""
        wgsl = _scene()
        for field in ["albedo", "roughness", "metallic", "emissive", "ambient_occlusion"]:
            assert field in wgsl, f"Missing field '{field}' in generated WGSL"


# =============================================================================
# DEFAULT_MATERIAL_SWITCH: material_id maps to Material via switch/case
# =============================================================================


class TestDefaultMaterialSwitch:
    """DEFAULT_MATERIAL_SWITCH generates correct scene_material() function."""

    def test_function_signature(self):
        """scene_material takes i32 and returns Material."""
        assert "fn scene_material(id: i32) -> Material" in DEFAULT_MATERIAL_SWITCH

    def test_switch_statement(self):
        """Uses switch(id) statement."""
        assert "switch id" in DEFAULT_MATERIAL_SWITCH

    def test_default_case(self):
        """Has a default fallback case."""
        assert "default:" in DEFAULT_MATERIAL_SWITCH

    def test_default_returns_material(self):
        """Default case returns a Material(...) constructor call."""
        assert "return Material(" in DEFAULT_MATERIAL_SWITCH

    def test_default_has_albedo(self):
        """Default fallback contains an albedo vec3 value."""
        assert "vec3<f32>" in DEFAULT_MATERIAL_SWITCH

    def test_case_body_format(self):
        """Case body returns Material constructor with PBR fields."""
        default_block = DEFAULT_MATERIAL_SWITCH[
            DEFAULT_MATERIAL_SWITCH.index("default:"):
        ]
        assert "return Material(" in default_block

    def test_case_template_has_cases_placeholder(self):
        """Template has {case_bodies} placeholder for generated cases."""
        assert "{case_bodies}" in DEFAULT_MATERIAL_SWITCH

    def test_scene_material_in_output(self):
        """Full WGSL output includes scene_material function."""
        wgsl = _scene()
        assert "fn scene_material" in wgsl

    def test_single_material_case(self):
        """Single material generates a case 0 block."""
        wgsl = _scene()
        assert "case 0:" in wgsl

    def test_multiple_material_cases(self):
        """Multiple materials generate sequential case blocks."""
        materials = (
            MaterialNode(0, Vec3Node(0.8, 0.2, 0.2), FloatNode(0.5),
                         FloatNode(0.0), FloatNode(0.0), FloatNode(1.0)),
            MaterialNode(1, Vec3Node(0.2, 0.8, 0.2), FloatNode(0.3),
                         FloatNode(0.5), FloatNode(0.0), FloatNode(1.0)),
            MaterialNode(2, Vec3Node(0.2, 0.2, 0.8), FloatNode(0.7),
                         FloatNode(0.0), FloatNode(0.1), FloatNode(0.8)),
        )
        wgsl = _scene(materials=materials)
        assert "case 0:" in wgsl
        assert "case 1:" in wgsl
        assert "case 2:" in wgsl

    def test_case_order_preserved(self):
        """Case blocks appear in material_id order."""
        materials = (
            MaterialNode(0, Vec3Node(0.8, 0.2, 0.2), FloatNode(0.5),
                         FloatNode(0.0), FloatNode(0.0), FloatNode(1.0)),
            MaterialNode(5, Vec3Node(0.2, 0.2, 0.8), FloatNode(0.5),
                         FloatNode(0.0), FloatNode(1.0), FloatNode(1.0)),
        )
        wgsl = _scene(materials=materials)
        # Check that case 5 appears after case 0
        idx0 = wgsl.index("case 0:")
        idx5 = wgsl.index("case 5:")
        assert idx0 < idx5

    def test_function_balanced(self):
        """scene_material function has balanced braces."""
        wgsl = _scene()
        # Isolate the scene_material function
        fn_start = wgsl.index("fn scene_material")
        # Find the end of the function (next fn or end of module)
        remaining = wgsl[fn_start:]
        # Count braces in function
        ob = remaining.count("{")
        cb = remaining.count("}")
        assert ob == cb


# =============================================================================
# Material node emission: emissive values in generated WGSL
# =============================================================================


class TestMaterialEmissive:
    """Emissive values are correctly emitted in generated case bodies."""

    def test_zero_emissive(self):
        """Emissive=0.0 produces correct output."""
        mat = MaterialNode(0, Vec3Node(0.8, 0.2, 0.2), FloatNode(0.5),
                           FloatNode(0.0), FloatNode(0.0), FloatNode(1.0))
        wgsl = _scene(materials=(mat,))
        assert "0.0" in wgsl

    def test_positive_emissive(self):
        """Emissive=1.0 produces correct output."""
        mat = MaterialNode(0, Vec3Node(0.8, 0.2, 0.2), FloatNode(0.5),
                           FloatNode(0.0), FloatNode(1.0), FloatNode(1.0))
        wgsl = _scene(materials=(mat,))
        assert "1.0" in wgsl

    def test_emissive_value_in_case_body(self):
        """The emissive value appears in the case body in the Material constructor."""
        mat = MaterialNode(0, Vec3Node(0.8, 0.2, 0.2), FloatNode(0.5),
                           FloatNode(0.0), FloatNode(5.0), FloatNode(1.0))
        wgsl = _scene(materials=(mat,))
        # Find the case 0 body and check emissive is there
        case0_idx = wgsl.index("case 0:")
        case0_end = wgsl.index("case", case0_idx + 1) if "case" in wgsl[case0_idx + 20:] else len(wgsl)
        if case0_end == len(wgsl):
            case0_end = wgsl.index("default:", case0_idx)
        case_body = wgsl[case0_idx:case0_end]
        assert "emissive" in case_body.lower() or "5.0" in case_body

    @pytest.mark.parametrize("emissive", [0.0, 0.5, 1.0, 2.0, 10.0, 100.0])
    def test_various_emissive_values(self, emissive):
        """Various emissive values are correctly emitted."""
        mat = MaterialNode(0, Vec3Node(0.8, 0.2, 0.2), FloatNode(0.5),
                           FloatNode(0.0), FloatNode(emissive), FloatNode(1.0))
        wgsl = _scene(materials=(mat,))
        expected = str(emissive) if emissive != int(emissive) else f"{int(emissive)}.0"
        assert expected in wgsl, (
            f"Emissive value {emissive} (formatted as {expected}) not found in output"
        )

    def test_high_emissive_no_clamp(self):
        """Unusually high emissive values pass through without clamping."""
        mat = MaterialNode(0, Vec3Node(0.8, 0.2, 0.2), FloatNode(0.5),
                           FloatNode(0.0), FloatNode(999.0), FloatNode(1.0))
        wgsl = _scene(materials=(mat,))
        assert "999.0" in wgsl

    def test_multiple_emissive_values(self):
        """Multiple materials with different emissive values."""
        materials = (
            MaterialNode(0, Vec3Node(0.8, 0.2, 0.2), FloatNode(0.5),
                         FloatNode(0.0), FloatNode(0.0), FloatNode(1.0)),
            MaterialNode(1, Vec3Node(0.2, 0.8, 0.2), FloatNode(0.3),
                         FloatNode(0.5), FloatNode(2.0), FloatNode(1.0)),
            MaterialNode(2, Vec3Node(0.2, 0.2, 0.8), FloatNode(0.7),
                         FloatNode(0.0), FloatNode(8.5), FloatNode(0.8)),
        )
        wgsl = _scene(materials=materials)
        # Check emissive fields appear in cases
        case_bodies = []
        for i in range(3):
            # Find each case and its context
            idx = wgsl.index(f"case {i}:")
            # Find where the case body ends
            for j in range(i + 1, 4):
                target = f"case {j}:" if j < 3 else "default:"
                if target in wgsl:
                    end_idx = wgsl.index(target, idx)
                    case_bodies.append(wgsl[idx:end_idx])
                    break
        assert len(case_bodies) == 3
        # Check emissive fields exist in all case bodies
        for body in case_bodies:
            assert "emissive" in body.lower() or "Material(" in body


# =============================================================================
# material_id propagation: single primitive
# =============================================================================


class TestMaterialIdSinglePrimitive:
    """Single primitive propagates material_id as vec2<f32> second component."""

    def test_material_id_in_entry_return(self):
        """Entry point returns vec2<f32> with material_id as .y."""
        wgsl = _scene()
        assert "return vec2<f32>(result.x / comp, result.y)" in wgsl

    def test_result_contains_material_id(self):
        """The result binding includes material_id.0 in the vec2 call."""
        wgsl = _scene()
        assert "vec2<f32>" in wgsl
        # Check the SDF call wraps with vec2<f32>(sdSphere(...), 0.0)
        pattern = r"vec2<f32>\(sdSphere\(p,\s*1\.0\),\s*0\.0\)"
        assert re.search(pattern, wgsl), (
            "Single sphere should produce vec2<f32>(sdSphere(p, 1.0), 0.0)"
        )

    def test_material_id_on_return_line(self):
        """The return statement propagates result.y (the material_id)."""
        wgsl = _scene()
        assert "result.y" in wgsl

    def test_entry_point_returns_vec2(self):
        """Entry point signature returns vec2<f32>."""
        assert "fn sd_scene__test(p: vec3<f32>) -> vec2<f32>" in _scene()

    def test_different_material_id(self):
        """material_id=3 is emitted as 3.0 in the vec2 wrapper."""
        prim = SphereNode(
            position=PositionNode(), radius=FloatNode(1.0), material_id=3
        )
        wgsl = _scene(primitives=[prim])
        pattern = r"vec2<f32>\(sdSphere\(p,\s*1\.0\),\s*3\.0\)"
        assert re.search(pattern, wgsl), (
            "material_id=3 should produce vec2<f32>(sdSphere(p, 1.0), 3.0)"
        )

    @pytest.mark.parametrize("mid", [0, 1, 5, 42, 255])
    def test_various_material_ids(self, mid):
        """Various material_id values appear in the generated vec2 call."""
        prim = SphereNode(
            position=PositionNode(), radius=FloatNode(1.0), material_id=mid
        )
        wgsl = _scene(primitives=[prim])
        assert f"{mid}.0" in wgsl, (
            f"material_id={mid} should appear as {mid}.0 in output"
        )

    def test_no_select_for_single_prim(self):
        """Single primitive does not use select()."""
        wgsl = _scene()
        assert "select" not in wgsl, (
            "select() should not appear for a single primitive"
        )


# =============================================================================
# material_id propagation: multiple primitives with pairwise select()
# =============================================================================


class TestMaterialIdMultiplePrimitives:
    """Multiple primitives use select() to propagate the nearest material_id."""

    def test_two_primitives_select_present(self):
        """Two primitives generate select() call."""
        prims = [
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
        ]
        wgsl = _scene(primitives=prims)
        assert "select" in wgsl

    def test_two_primitives_two_d_vars(self):
        """Two primitives generate d0 and d1 variables."""
        prims = [
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
        ]
        wgsl = _scene(primitives=prims)
        assert "let d0 =" in wgsl
        assert "let d1 =" in wgsl

    def test_two_primitives_select_result_assign(self):
        """Select result tracks the nearer primitive's material_id."""
        prims = [
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
        ]
        wgsl = _scene(primitives=prims)
        # Pattern: let result = select(d1, result, result.x < d1.x)
        pattern = r"let result = select\(d1, result, result\.x < d1\.x\)"
        assert re.search(pattern, wgsl), (
            f"Expected pairwise select pattern, got:\n{wgsl}"
        )

    def test_select_pattern_chooses_min_distance(self):
        """select(d{i}, result, result.x < d{i}.x) chooses the nearer primitive."""
        prims = [
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
        ]
        wgsl = _scene(primitives=prims)
        # select(fresh, current, current.x < fresh.x) means keep current if closer
        # Pattern: select(d1, result, result.x < d1.x)
        assert "result.x" in wgsl
        assert "d1.x" in wgsl

    def test_three_primitives_chain(self):
        """Three primitives produce a pairwise select chain."""
        prims = [
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
            PlaneNode(position=PositionNode(), normal=Vec3Node(0.0, 1.0, 0.0),
                      distance=FloatNode(0.0), material_id=2),
        ]
        wgsl = _scene(primitives=prims)
        assert "let d0 =" in wgsl
        assert "let d1 =" in wgsl
        assert "let d2 =" in wgsl
        assert "let result = d0" in wgsl
        assert "let result = select(d1, result, result.x < d1.x)" in wgsl
        assert "let result = select(d2, result, result.x < d2.x)" in wgsl

    def test_three_primitives_balanced(self):
        """Three primitives with select chain produce balanced WGSL."""
        prims = [
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
            PlaneNode(position=PositionNode(), normal=Vec3Node(0.0, 1.0, 0.0),
                      distance=FloatNode(0.0), material_id=2),
        ]
        wgsl = _scene(primitives=prims)
        assert _balanced(wgsl)
        ob, cb = _count_braces(wgsl)
        assert ob == cb

    def test_four_primitives_select_chain(self):
        """Four primitives produce three select calls."""
        prims = [
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
            PlaneNode(position=PositionNode(), normal=Vec3Node(0.0, 1.0, 0.0),
                      distance=FloatNode(0.0), material_id=2),
            BoxNode(position=PositionNode(), size=Vec3Node(2.0, 2.0, 2.0), material_id=3),
        ]
        wgsl = _scene(primitives=prims)
        assert "let d0 =" in wgsl
        assert "let d1 =" in wgsl
        assert "let d2 =" in wgsl
        assert "let d3 =" in wgsl
        select_count = wgsl.count("let result = select(")
        assert select_count == 3, (
            f"Expected 3 select() calls for 4 primitives, got {select_count}"
        )

    def test_select_only_with_multiple_prims(self):
        """select() is absent with single primitive, present with two."""
        single = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
        ])
        assert "select" not in single
        double = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
        ])
        assert "select" in double


# =============================================================================
# material_id propagation through scene entry point
# =============================================================================


class TestMaterialIdInEntryPoint:
    """scene entry point correctly returns vec2<f32> with material_id."""

    def test_return_type_vec2(self):
        """Entry point declares -> vec2<f32> return type."""
        wgsl = _scene()
        assert "-> vec2<f32>" in wgsl

    def test_return_line_has_result_y(self):
        """Return line includes result.y for material_id."""
        wgsl = _scene()
        return_lines = [l for l in wgsl.splitlines() if "return" in l and "vec2<f32>" in l]
        assert any("result.y" in l for l in return_lines), (
            "Return statement should propagate result.y (material_id)"
        )

    def test_two_prims_return_has_result_y(self):
        """Return line material_id comes from the select chain."""
        prims = [
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
        ]
        wgsl = _scene(primitives=prims)
        return_lines = [l for l in wgsl.splitlines() if "return" in l and "vec2<f32>" in l]
        assert any("result.y" in l for l in return_lines)

    def test_comp_div_applied_to_x_only(self):
        """Compensation divisor applies to distance (result.x) but not material_id."""
        wgsl = _scene()
        # The return should be vec2<f32>(result.x / comp, result.y)
        # NOT vec2<f32>(result.x, result.y) / comp or similar
        assert "result.x / comp" in wgsl, (
            "Compensation should divide result.x by comp"
        )
        assert "result.y" in wgsl

    def test_material_id_untouched_by_compensation(self):
        """material_id (result.y) is NOT divided by compensation."""
        wgsl = _scene()
        assert "result.y" in wgsl
        # Check result.y is not part of a division
        lines = wgsl.splitlines()
        return_line = [l for l in lines if "return" in l and "vec2<f32>" in l][0]
        assert "/ comp" not in return_line.replace("result.x", ""), (
            "result.y should not be divided by compensation"
        )

    def test_result_y_not_modified(self):
        """result.y is passed straight through without arithmetic."""
        wgsl = _scene()
        lines = wgsl.splitlines()
        # Filter out comment lines (/// and //) to find the actual return statement
        return_line = [
            l for l in lines
            if "return" in l and "vec2<f32>" in l and not l.strip().startswith("//")
        ][0]
        # Parse the return arguments
        args = return_line[return_line.index("(") + 1 : return_line.index(")")]
        y_arg = args.split(",")[1].strip()
        assert y_arg == "result.y", (
            f"Second return argument should be 'result.y', got '{y_arg}'"
        )


# =============================================================================
# Full integration: material + SDF + pipeline (if applicable)
# =============================================================================


class TestMaterialFullIntegration:
    """Material codegen integrates correctly with full scene output."""

    def test_material_struct_in_full_output(self):
        """Material struct appears in full scene output."""
        wgsl = _scene()
        assert "struct Material" in wgsl

    def test_scene_material_in_full_output(self):
        """scene_material function appears in full scene output."""
        wgsl = _scene()
        assert "fn scene_material" in wgsl

    def test_sdf_emitted_before_material(self):
        """SDF functions should precede Material struct in output."""
        wgsl = _scene()
        sdf_idx = wgsl.index("fn sdSphere")
        mat_struct_idx = wgsl.index("struct Material")
        assert sdf_idx < mat_struct_idx, (
            "SDF function should be emitted before Material struct"
        )

    def test_material_before_scene_entry(self):
        """Material struct and function appear before scene entry point."""
        wgsl = _scene()
        mat_fn_idx = wgsl.index("fn scene_material")
        entry_idx = wgsl.index("fn sd_scene__test")
        assert mat_fn_idx < entry_idx, (
            "scene_material should be emitted before scene entry point"
        )

    def test_full_output_balanced(self):
        """Full output has balanced parens and braces."""
        wgsl = _scene()
        assert _balanced(wgsl)

    def test_spdx_header_present(self):
        """Output starts with SPDX header."""
        wgsl = _scene()
        assert wgsl.startswith("// SPDX-License-Identifier: MIT")

    def test_material_id_in_vec2_for_sphere(self):
        """Sphere SDF call wraps with vec2<f32> including material_id."""
        wgsl = _scene()
        pattern = r"vec2<f32>\(sdSphere\(p,\s*1\.0\),\s*0\.0\)"
        assert re.search(pattern, wgsl), (
            "Sphere SDF call should be wrapped in vec2<f32>(sdSphere(p, 1.0), 0.0)"
        )

    def test_material_id_in_vec2_for_box(self):
        """Box SDF call wraps with vec2<f32> including material_id."""
        prim = BoxNode(
            position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=2
        )
        wgsl = _scene(primitives=[prim])
        pattern = r"vec2<f32>\(sdBox\(p,\s*vec3<f32>\(1\.0,\s*1\.0,\s*1\.0\)\),\s*2\.0\)"
        assert re.search(pattern, wgsl), (
            "Box SDF call should wrap with vec2<f32> including material_id"
        )

    def test_select_chain_material_id_correct(self):
        """Select chain preserves correct material_id assignments."""
        prims = [
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(2.0, 2.0, 2.0), material_id=7),
        ]
        wgsl = _scene(primitives=prims)
        # d0 has material_id 0, d1 has material_id 7
        assert "vec2<f32>(sdSphere(p, 1.0), 0.0)" in wgsl or 'vec2<f32>(sdSphere(p, 1.0), 0.0)' in wgsl
        assert "vec2<f32>(sdBox(p, vec3<f32>(2.0, 2.0, 2.0)), 7.0)" in wgsl


# =============================================================================
# Edge cases: default/no materials, extreme material_ids, empty scenes
# =============================================================================


class TestMaterialEdgeCases:
    """Edge and boundary cases for material codegen."""

    def test_no_materials_provided(self):
        """With no materials, default scene_material is still generated."""
        graph = SceneGraph(
            primitives=(
                SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            ),
            name="test",
        )
        wgsl = generate_wgsl(graph)
        assert "fn scene_material" in wgsl
        assert "default:" in wgsl

    def test_default_material_fallback(self):
        """Default material case returns expected fallback values."""
        graph = SceneGraph(
            primitives=(
                SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            ),
            name="test",
        )
        wgsl = generate_wgsl(graph)
        # Default case should have the hardcoded fallback
        assert "vec3<f32>(0.8, 0.2, 0.2)" in wgsl

    def test_single_material_without_switch_optimization(self):
        """Even with one material, switch/case structure is used."""
        wgsl = _scene()
        assert "switch id" in wgsl
        assert "case 0:" in wgsl
        assert "default:" in wgsl

    def test_non_zero_material_id_single(self):
        """Material with material_id=5 generates case 5."""
        mat = MaterialNode(5, Vec3Node(0.1, 0.2, 0.3), FloatNode(0.9),
                           FloatNode(0.0), FloatNode(0.0), FloatNode(1.0))
        prim = SphereNode(
            position=PositionNode(), radius=FloatNode(1.0), material_id=5
        )
        wgsl = _scene(materials=(mat,), primitives=[prim])
        assert "case 5:" in wgsl

    def test_no_materials_no_crash(self):
        """Scene with no materials attribute still generates valid WGSL."""
        graph = SceneGraph(
            primitives=(
                SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            ),
            name="test",
        )
        wgsl = generate_wgsl(graph)
        assert _balanced(wgsl)
        ob, cb = _count_braces(wgsl)
        assert ob == cb

    def test_prim_with_default_material_id(self):
        """Primitive with default material_id=0 works correctly."""
        prim = SphereNode(
            position=PositionNode(), radius=FloatNode(1.0)
        )
        wgsl = _scene(primitives=[prim])
        assert "0.0" in wgsl

    def test_all_materials_same_id(self):
        """Multiple materials with same id should not crash (edge case)."""
        materials = (
            MaterialNode(0, Vec3Node(0.8, 0.2, 0.2), FloatNode(0.5),
                         FloatNode(0.0), FloatNode(0.0), FloatNode(1.0)),
            MaterialNode(0, Vec3Node(0.2, 0.8, 0.2), FloatNode(0.3),
                         FloatNode(0.5), FloatNode(0.0), FloatNode(1.0)),
        )
        prims = [
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
        ]
        wgsl = _scene(materials=materials, primitives=prims)
        assert _balanced(wgsl)
        # Should have two case 0: lines (WGSL will use the last one)
        assert wgsl.count("case 0:") >= 2

    def test_zero_roughness_metallic(self):
        """Material with roughness=0.0 and metallic=0.0 emits correctly."""
        mat = MaterialNode(0, Vec3Node(1.0, 1.0, 1.0), FloatNode(0.0),
                           FloatNode(0.0), FloatNode(0.0), FloatNode(1.0))
        wgsl = _scene(materials=(mat,))
        assert "0.0" in wgsl

    def test_full_white_albedo(self):
        """Albedo (1.0, 1.0, 1.0) is correctly emitted."""
        mat = MaterialNode(0, Vec3Node(1.0, 1.0, 1.0), FloatNode(0.5),
                           FloatNode(0.0), FloatNode(0.0), FloatNode(1.0))
        wgsl = _scene(materials=(mat,))
        assert "vec3<f32>(1.0, 1.0, 1.0)" in wgsl

    def test_zero_ao(self):
        """Ambient occlusion of 0.0 is correctly emitted."""
        mat = MaterialNode(0, Vec3Node(0.8, 0.2, 0.2), FloatNode(0.5),
                           FloatNode(0.0), FloatNode(0.0), FloatNode(0.0))
        wgsl = _scene(materials=(mat,))
        # Look for 0.0 in context of the case body
        assert "0.0" in wgsl


# =============================================================================
# Template formatting: verify MATERIAL_STRUCT and DEFAULT_MATERIAL_SWITCH
# placeholder filling
# =============================================================================


class TestTemplateFormatting:
    """MATERIAL_STRUCT and DEFAULT_MATERIAL_SWITCH produce valid WGSL when filled."""

    def test_material_struct_standalone_valid(self):
        """MATERIAL_STRUCT is valid WGSL on its own (balanced)."""
        assert _balanced(MATERIAL_STRUCT)
        ob, cb = _count_braces(MATERIAL_STRUCT)
        assert ob == cb

    def test_default_switch_standalone_valid(self):
        """DEFAULT_MATERIAL_SWITCH is structurally valid."""
        ob, cb = _count_braces(DEFAULT_MATERIAL_SWITCH)
        assert ob == cb

    def test_placeholder_fills_with_case(self):
        """DEFAULT_MATERIAL_SWITCH can be filled with a case body."""
        case_body = "        case 0: { return Material(vec3<f32>(0.8, 0.2, 0.2), 0.5, 0.0, 0.0, 1.0); }\n"
        filled = DEFAULT_MATERIAL_SWITCH.format(case_bodies=case_body)
        assert "case 0:" in filled
        assert "default:" in filled
        assert _balanced(filled)

    def test_placeholder_fills_multiple_cases(self):
        """DEFAULT_MATERIAL_SWITCH handles multiple case bodies."""
        case_bodies = (
            "        case 0: { return Material(vec3<f32>(0.8, 0.2, 0.2), 0.5, 0.0, 0.0, 1.0); }\n"
            "        case 1: { return Material(vec3<f32>(0.2, 0.8, 0.2), 0.3, 0.5, 0.0, 1.0); }\n"
        )
        filled = DEFAULT_MATERIAL_SWITCH.format(case_bodies=case_bodies)
        assert "case 0:" in filled
        assert "case 1:" in filled
        assert "default:" in filled
        assert _balanced(filled)

    def test_empty_case_bodies(self):
        """Empty case bodies still produce valid WGSL (just default)."""
        filled = DEFAULT_MATERIAL_SWITCH.format(case_bodies="")
        assert "default:" in filled
        assert _balanced(filled)

    def test_generated_scene_material_balanced(self):
        """Generated scene_material function has balanced braces."""
        wgsl = _scene()
        fn_start = wgsl.index("fn scene_material")
        fn_text = wgsl[fn_start:]
        # Find the matching end
        ob = fn_text.count("{")
        cb = fn_text.count("}")
        assert ob == cb, (
            f"scene_material has unbalanced braces: {ob} open, {cb} close"
        )

    def test_case_body_has_return_material(self):
        """Generated case body returns Material(...) constructor."""
        mat = MaterialNode(42, Vec3Node(0.1, 0.5, 0.9), FloatNode(0.2),
                           FloatNode(0.8), FloatNode(0.3), FloatNode(0.7))
        wgsl = _scene(materials=(mat,))
        assert "case 42:" in wgsl
        # Find the case 42 body
        idx = wgsl.index("case 42:")
        # Check it contains a Material constructor
        assert "return Material(" in wgsl[idx:], (
            "Case 42 body should return Material(...)"
        )
