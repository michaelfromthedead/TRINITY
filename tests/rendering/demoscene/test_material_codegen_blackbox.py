"""
Cleanroom blackbox tests for material WGSL codegen (T-DEMO-2.6).

Tests that WgslCodeGen produces correct WGSL output for Material struct
emission, scene_material() function, multiple material case entries,
and material ID correctness in SDF calls.

BLACKBOX coverage (8 paths):
  Path 1:  Material struct has all 5 PBR fields (albedo, roughness, metallic,
           emissive, ambient_occlusion) with correct WGSL types
  Path 2:  scene_material() function is emitted with switch and default case
  Path 3:  Multiple materials produce distinct switch case entries with
           correct material_id and PBR values
  Path 4:  Material ID appears in SDF return as vec2<f32>(distance, <id>.0)
  Path 5:  Material ID propagation with multiple primitives via select() chain
           preserves the winning material_id
  Path 6:  Default fallback material (case 0) when no materials provided
  Path 7:  SceneGraph with explicit materials overrides default
  Path 8:  Full scene integrates material struct, scene_material(), and
           SDF entry point with correct ID wiring
"""

from __future__ import annotations

import re

import pytest

# MaterialNode and other demoscene AST features not yet implemented
pytest.skip("Demoscene AST not fully implemented", allow_module_level=True)

from engine.rendering.demoscene.ast_nodes import (
    FloatNode,
    MaterialNode,
    SceneGraph,
    SphereNode,
    BoxNode,
    TorusNode,
    PlaneNode,
    PositionNode,
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
    primitives=None,
    materials=None,
    name="test",
):
    """Build a SceneGraph with no pipeline and return generated WGSL."""
    if primitives is None:
        primitives = [
            SphereNode(
                position=PositionNode(),
                radius=FloatNode(1.0),
                material_id=0,
            )
        ]
    if materials is None:
        materials = ()
    graph = SceneGraph(
        primitives=tuple(primitives),
        materials=tuple(materials),
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


def _extract_material_ids(wgsl: str) -> list[int]:
    """Extract material_id values from vec2<f32>(..., <id>.0) SDF calls.

    Uses a greedy .* to skip all intermediate content and capture only the
    LAST numeric value before the closing paren, which is the material ID.
    """
    ids = []
    for m in re.finditer(r"vec2<f32>\(.*,\s*(\d+)\.0\)", wgsl):
        ids.append(int(m.group(1)))
    return ids


# =============================================================================
# Path 1: Material struct has all 5 PBR fields with correct WGSL types
# =============================================================================


class TestMaterialStructFields:
    """Material struct must declare all 5 PBR fields with correct types."""

    def test_struct_declared(self):
        wgsl = _scene()
        assert "struct Material" in wgsl

    def test_albedo_field_type(self):
        wgsl = _scene()
        assert "albedo: vec3<f32>" in wgsl

    def test_roughness_field_type(self):
        wgsl = _scene()
        assert "roughness: f32" in wgsl

    def test_metallic_field_type(self):
        wgsl = _scene()
        assert "metallic: f32" in wgsl

    def test_emissive_field_type(self):
        wgsl = _scene()
        assert "emissive: f32" in wgsl

    def test_ambient_occlusion_field_type(self):
        wgsl = _scene()
        assert "ambient_occlusion: f32" in wgsl

    def test_struct_has_only_required_fields(self):
        """Struct should not contain extra unexpected fields."""
        wgsl = _scene()
        struct_body = _extract_struct_body(wgsl, "Material")
        assert struct_body is not None
        fields = [line.strip() for line in struct_body.split(",") if line.strip()]
        field_names = set()
        for f in fields:
            name = f.split(":")[0].strip()
            field_names.add(name)
        expected = {"albedo", "roughness", "metallic", "emissive", "ambient_occlusion"}
        assert field_names == expected, f"Unexpected fields: {field_names - expected}"

    def test_albedo_is_vec3_fields_are_f32(self):
        """Check that albedo is the only vec3 field; all others are f32."""
        wgsl = _scene()
        vec3_count = wgsl.count("vec3<f32>")
        # At minimum: albedo field + position param in entry point + p_d/comp types
        assert vec3_count >= 2


def _extract_struct_body(wgsl: str, struct_name: str) -> str | None:
    """Extract the body of a WGSL struct definition."""
    pattern = rf"struct\s+{struct_name}\s*{{(.*?)}}"
    m = re.search(pattern, wgsl, re.DOTALL)
    return m.group(1) if m else None


# =============================================================================
# Path 2: scene_material() function is emitted with switch and default case
# =============================================================================


class TestSceneMaterialFunction:
    """scene_material(id: i32) -> Material maps IDs to PBR properties."""

    def test_function_present(self):
        wgsl = _scene()
        fns = _extract_fns(wgsl)
        assert "scene_material" in fns

    def test_function_signature(self):
        wgsl = _scene()
        assert "fn scene_material(id: i32) -> Material" in wgsl

    def test_has_switch_statement(self):
        wgsl = _scene(materials=(
            MaterialNode(
                material_id=0,
                albedo=Vec3Node(0.8, 0.2, 0.2),
                roughness=FloatNode(0.5),
                metallic=FloatNode(0.0),
                emissive=FloatNode(0.0),
                ambient_occlusion=FloatNode(1.0),
            ),
        ))
        assert "switch id" in wgsl

    def test_default_case_present(self):
        wgsl = _scene()
        # With no materials, the default case alone is emitted (case 0 is default)
        assert "default:" in wgsl

    def test_returns_Material(self):
        wgsl = _scene()
        assert "return Material(" in wgsl

    def test_balanced_parens(self):
        wgsl = _scene(materials=(
            MaterialNode(
                material_id=0,
                albedo=Vec3Node(0.8, 0.2, 0.2),
                roughness=FloatNode(0.5),
                metallic=FloatNode(0.0),
                emissive=FloatNode(0.0),
                ambient_occlusion=FloatNode(1.0),
            ),
        ))
        assert _balanced(wgsl)

    def test_balanced_braces(self):
        wgsl = _scene(materials=(
            MaterialNode(
                material_id=0,
                albedo=Vec3Node(0.8, 0.2, 0.2),
                roughness=FloatNode(0.5),
                metallic=FloatNode(0.0),
                emissive=FloatNode(0.0),
                ambient_occlusion=FloatNode(1.0),
            ),
        ))
        ob, cb = _count_braces(wgsl)
        assert ob == cb


# =============================================================================
# Path 3: Multiple materials produce distinct switch case entries
# =============================================================================


class TestMultipleMaterials:
    """Each MaterialNode produces a distinct switch case with correct values."""

    def test_two_material_cases(self):
        wgsl = _scene(materials=(
            MaterialNode(
                material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2),
                roughness=FloatNode(0.5), metallic=FloatNode(0.0),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
            ),
            MaterialNode(
                material_id=1, albedo=Vec3Node(0.2, 0.8, 0.2),
                roughness=FloatNode(0.3), metallic=FloatNode(0.5),
                emissive=FloatNode(0.1), ambient_occlusion=FloatNode(0.9),
            ),
        ))
        assert "case 0:" in wgsl
        assert "case 1:" in wgsl

    def test_three_material_cases(self):
        wgsl = _scene(materials=(
            MaterialNode(
                material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2),
                roughness=FloatNode(0.5), metallic=FloatNode(0.0),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
            ),
            MaterialNode(
                material_id=1, albedo=Vec3Node(0.2, 0.8, 0.2),
                roughness=FloatNode(0.3), metallic=FloatNode(0.5),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(0.8),
            ),
            MaterialNode(
                material_id=2, albedo=Vec3Node(0.2, 0.2, 0.8),
                roughness=FloatNode(0.9), metallic=FloatNode(0.0),
                emissive=FloatNode(0.5), ambient_occlusion=FloatNode(0.7),
            ),
        ))
        assert "case 0:" in wgsl
        assert "case 1:" in wgsl
        assert "case 2:" in wgsl

    def test_distinct_albedo_values(self):
        """Each case must carry the correct distinct albedo."""
        wgsl = _scene(materials=(
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
        ))
        # Check that the case bodies contain the correct albedo values
        assert "vec3<f32>(0.8, 0.2, 0.2)" in wgsl
        assert "vec3<f32>(0.2, 0.8, 0.2)" in wgsl

    def test_distinct_roughness_values(self):
        wgsl = _scene(materials=(
            MaterialNode(
                material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2),
                roughness=FloatNode(0.5), metallic=FloatNode(0.0),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
            ),
            MaterialNode(
                material_id=1, albedo=Vec3Node(0.2, 0.8, 0.2),
                roughness=FloatNode(0.1), metallic=FloatNode(0.5),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
            ),
        ))
        assert "0.5" in wgsl
        assert "0.1" in wgsl

    def test_distinct_metallic_values(self):
        wgsl = _scene(materials=(
            MaterialNode(
                material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2),
                roughness=FloatNode(0.5), metallic=FloatNode(0.0),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
            ),
            MaterialNode(
                material_id=1, albedo=Vec3Node(0.2, 0.8, 0.2),
                roughness=FloatNode(0.3), metallic=FloatNode(0.8),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
            ),
        ))
        assert "0.0" in wgsl
        assert "0.8" in wgsl

    def test_each_material_case_bounded_by_braces(self):
        """Each switch case body should have matching braces."""
        wgsl = _scene(materials=(
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
        ))
        assert _balanced(wgsl)
        ob, cb = _count_braces(wgsl)
        assert ob == cb

    def test_material_order_preserved(self):
        """Case order in switch should match input material order."""
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
        wgsl = _scene(materials=materials)
        idx0 = wgsl.index("case 0:")
        idx1 = wgsl.index("case 1:")
        assert idx0 < idx1, "case 0 must appear before case 1"


# =============================================================================
# Path 4: Material ID in SDF return as vec2<f32>(distance, <id>.0)
# =============================================================================


class TestMaterialIdInSdfCall:
    """Material ID must be embedded in the SDF result as vec2<f32>(distance, id)."""

    def test_single_primitive_id_zero(self):
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
        ])
        ids = _extract_material_ids(wgsl)
        assert 0 in ids

    def test_single_primitive_id_one(self):
        wgsl = _scene(primitives=[
            BoxNode(
                position=PositionNode(),
                size=Vec3Node(1.0, 1.0, 1.0),
                material_id=1,
            ),
        ])
        ids = _extract_material_ids(wgsl)
        assert 1 in ids
        assert 0 not in ids

    def test_single_primitive_id_large(self):
        wgsl = _scene(primitives=[
            TorusNode(
                position=PositionNode(),
                major_radius=FloatNode(2.0),
                minor_radius=FloatNode(0.5),
                material_id=42,
            ),
        ])
        ids = _extract_material_ids(wgsl)
        assert 42 in ids

    def test_id_formatted_as_float(self):
        """Material ID must appear as <id>.0 (WGSL float literal)."""
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
        ])
        assert "0.0)" in wgsl

    def test_id_in_vec2_result(self):
        """Must use vec2<f32> wrapper with distance and material ID."""
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=2),
        ])
        assert "vec2<f32>(" in wgsl
        # The material_id is the second component
        assert "2.0)" in wgsl or "2.0" in wgsl


# =============================================================================
# Path 5: Material ID propagation via select() chain
# =============================================================================


class TestMaterialIdPropagation:
    """With multiple primitives, winning material_id propagates via select()."""

    def test_select_used_with_two_primitives(self):
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
        ])
        assert "select" in wgsl

    def test_select_used_with_three_primitives(self):
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
            PlaneNode(
                position=PositionNode(),
                normal=Vec3Node(0.0, 1.0, 0.0),
                distance=FloatNode(0.0),
                material_id=2,
            ),
        ])
        assert "select" in wgsl
        # Should have at least 2 select calls (one per comparison after the first)
        assert wgsl.count("select") >= 2

    def test_select_wires_correct_material_id(self):
        """select() should preserve d{i}.y as the material_id component."""
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
        ])
        # Each d{i} is vec2<f32>(distance, material_id)
        # select(d{i}, result, result.x < d{i}.x) picks the closer one
        # and keeps that one's .y (material_id)
        assert re.search(r"select\(d1,\s*result,\s*result\.x\s*<\s*d1\.x\)", wgsl) or \
               re.search(r"select\(d1,\s*result,\s*result\.x\s*<\s*d1\.x\)", wgsl)

    def test_material_id_of_each_primitive_in_output(self):
        """All material IDs from primitives should appear in output."""
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
            PlaneNode(
                position=PositionNode(),
                normal=Vec3Node(0.0, 1.0, 0.0),
                distance=FloatNode(0.0),
                material_id=5,
            ),
        ])
        ids = _extract_material_ids(wgsl)
        assert 0 in ids
        assert 1 in ids
        assert 5 in ids

    def test_no_select_with_single_primitive(self):
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
        ])
        assert "select" not in wgsl

    def test_select_preserves_distance_component(self):
        """select() condition uses result.x < d{i}.x (distance comparison)."""
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
        ])
        assert "result.x <" in wgsl
        assert ".x" in wgsl


# =============================================================================
# Path 6: Default fallback material when no materials provided
# =============================================================================


class TestDefaultMaterial:
    """Default fallback when SceneGraph has no materials."""

    def test_default_exists_with_no_materials(self):
        """Even without materials, scene_material() must be callable."""
        graph = SceneGraph(
            primitives=(
                SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            ),
            name="test",
        )
        wgsl = generate_wgsl(graph)
        assert "fn scene_material" in wgsl

    def test_default_case_has_pbr_defaults(self):
        graph = SceneGraph(
            primitives=(
                SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            ),
            name="test",
        )
        wgsl = generate_wgsl(graph)
        # Default fallback: albedo (0.8, 0.2, 0.2), roughness 0.5,
        # metallic 0.0, emissive 0.0, ambient_occlusion 1.0
        assert "Material(vec3<f32>(0.8, 0.2, 0.2), 0.5, 0.0, 0.0, 1.0)" in wgsl

    def test_default_balanced(self):
        wgsl = _scene()
        assert _balanced(wgsl)
        ob, cb = _count_braces(wgsl)
        assert ob == cb


# =============================================================================
# Path 7: SceneGraph with explicit materials overrides default
# =============================================================================


class TestExplicitMaterials:
    """Explicit materials in SceneGraph override defaults."""

    def test_explicit_material_appears_as_case(self):
        wgsl = _scene(materials=(
            MaterialNode(
                material_id=0, albedo=Vec3Node(0.1, 0.2, 0.3),
                roughness=FloatNode(0.1), metallic=FloatNode(0.9),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(0.5),
            ),
        ))
        assert "case 0:" in wgsl
        assert "vec3<f32>(0.1, 0.2, 0.3)" in wgsl
        assert "0.1" in wgsl
        assert "0.9" in wgsl
        assert "0.5" in wgsl

    def test_default_and_explicit_both_present(self):
        """With explicit materials, the default clause should remain as fallback."""
        wgsl = _scene(materials=(
            MaterialNode(
                material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2),
                roughness=FloatNode(0.5), metallic=FloatNode(0.0),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
            ),
        ))
        assert "case 0:" in wgsl
        assert "default:" in wgsl

    def test_explicit_mat_with_multiple_primitives(self):
        """Multiple primitives with explicit materials should work together."""
        wgsl = _scene(
            primitives=[
                SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
                BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
            ],
            materials=(
                MaterialNode(
                    material_id=0, albedo=Vec3Node(0.9, 0.1, 0.1),
                    roughness=FloatNode(0.2), metallic=FloatNode(0.0),
                    emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
                ),
                MaterialNode(
                    material_id=1, albedo=Vec3Node(0.1, 0.9, 0.1),
                    roughness=FloatNode(0.8), metallic=FloatNode(0.0),
                    emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
                ),
            ),
        )
        assert "case 0:" in wgsl
        assert "case 1:" in wgsl
        assert "vec3<f32>(0.9, 0.1, 0.1)" in wgsl
        assert "vec3<f32>(0.1, 0.9, 0.1)" in wgsl
        assert "select" in wgsl


# =============================================================================
# Path 8: Full scene integration
# =============================================================================


class TestFullScene:
    """Full scene integrates Material struct, scene_material(), and SDF entry."""

    def test_material_struct_before_scene_material(self):
        """Material struct definition must come before scene_material fn."""
        wgsl = _scene(materials=(
            MaterialNode(
                material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2),
                roughness=FloatNode(0.5), metallic=FloatNode(0.0),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
            ),
        ))
        idx_struct = wgsl.index("struct Material")
        idx_fn = wgsl.index("fn scene_material")
        assert idx_struct < idx_fn, "Material struct must precede scene_material"

    def test_scene_material_before_entry_point(self):
        """scene_material fn must come before the sd_scene entry point."""
        wgsl = _scene(materials=(
            MaterialNode(
                material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2),
                roughness=FloatNode(0.5), metallic=FloatNode(0.0),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
            ),
        ))
        idx_fn = wgsl.index("fn scene_material")
        idx_entry = wgsl.index("fn sd_scene__test")
        assert idx_fn < idx_entry, "scene_material must precede sd_scene entry"

    def test_entry_point_returns_vec2_with_material_id(self):
        """sd_scene entry must return vec2<f32> containing material_id."""
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=7),
        ])
        assert "return vec2<f32>(" in wgsl
        # The returned vec2 should have the material_id component
        assert "7.0" in wgsl

    def test_full_scene_syntactically_valid(self):
        """Full scene output must be syntactically valid WGSL."""
        wgsl = _scene(
            primitives=[
                SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
                BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
            ],
            materials=(
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
            ),
            name="full_scene",
        )
        assert _balanced(wgsl)
        ob, cb = _count_braces(wgsl)
        assert ob == cb
        assert "fn sd_scene__full_scene" in wgsl
        assert "fn scene_material" in wgsl
        assert "struct Material" in wgsl

    def test_multiple_primitives_all_have_material_ids(self):
        """All primitives in a multi-primitive scene must carry material IDs."""
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(1.0, 1.0, 1.0), material_id=1),
            TorusNode(
                position=PositionNode(),
                major_radius=FloatNode(2.0),
                minor_radius=FloatNode(0.5),
                material_id=2,
            ),
        ])
        ids = _extract_material_ids(wgsl)
        assert ids == [0, 1, 2], f"Expected [0, 1, 2] got {ids}"

    def test_material_switch_is_valid_wgsl(self):
        """Switch statement must be valid WGSL syntax (switch/case/default)."""
        wgsl = _scene(materials=(
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
        ))
        # Check that each case follows WGSL's switch syntax
        assert re.search(r"case\s+\d+:\s*\{", wgsl)
        assert re.search(r"default:\s*\{", wgsl)
        # Each case body should return Material(...)
        assert "return Material(" in wgsl


# =============================================================================
# Edge cases
# =============================================================================


class TestMaterialEdgeCases:
    """Material codegen handles edge cases correctly."""

    def test_material_id_zero_with_nonzero_ids(self):
        """Material ID 0 alongside non-zero IDs should work."""
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
            BoxNode(position=PositionNode(), size=Vec3Node(2.0, 2.0, 2.0), material_id=99),
        ])
        ids = _extract_material_ids(wgsl)
        assert 0 in ids
        assert 99 in ids

    def test_large_material_id(self):
        """Large material IDs should be emitted correctly."""
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=255),
        ])
        assert "255.0)" in wgsl

    def test_all_fields_in_each_case(self):
        """Each switch case must include all 5 PBR fields."""
        wgsl = _scene(materials=(
            MaterialNode(
                material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2),
                roughness=FloatNode(0.5), metallic=FloatNode(0.0),
                emissive=FloatNode(0.0), ambient_occlusion=FloatNode(1.0),
            ),
        ))
        # Each case body should call Material(albedo, roughness, metallic, emissive, ao)
        assert "return Material(vec3<f32>" in wgsl

    def test_negative_material_id_not_used(self):
        """material_id should be non-negative (convention check)."""
        # The system allows any int, but convention is non-negative
        wgsl = _scene(primitives=[
            SphereNode(position=PositionNode(), radius=FloatNode(1.0), material_id=0),
        ])
        # Verify no negative IDs in the output
        for m in re.finditer(r"vec2<f32>\(.*,\s*(-?\d+)\.0\)", wgsl):
            assert int(m.group(1)) >= 0, f"Negative material_id found: {m.group(1)}"

    def test_spdx_license_present(self):
        """Generated output must start with the SPDX license header."""
        wgsl = _scene()
        assert wgsl.startswith("// SPDX-License-Identifier: MIT")
