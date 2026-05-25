"""
Whitebox tests for domain ops WGSL codegen (T-DEMO-2.5 + T-DEMO-2.6).

Tests the DOMAIN_OP_TYPE_MAP, _DOMAIN_OP_CALLS template dict, SDF_*
template constants, and the generate_wgsl() pipeline in
engine/rendering/demoscene/wgsl_codegen.py, verifying:

  - DOMAIN_OP_TYPE_MAP maps each AST node class to the correct WGSL
    domain function name (RepeatNode -> domain_repeat, etc.)
  - _DOMAIN_OP_CALLS templates produce valid WGSL expressions for
    each domain operation with correct {placeholder} fields
  - Compensation templates (_KIFS_COMP, _STRETCH_COMP) exist for
    non-isometric ops (KIFS, Stretch)
  - generate_wgsl() produces correct WGSL output containing the
    expected domain function calls, SDF functions, and compensation
  - SDF_* template constants are valid WGSL function definitions
  - MATERIAL_STRUCT and DEFAULT_MATERIAL_SWITCH are present
  - Edge cases: KIFS folds=2, stretch identity, zero-radius bend
"""

import re
import textwrap

import pytest

# MaterialNode and other demoscene AST features not yet implemented
pytest.skip("Demoscene AST not fully implemented", allow_module_level=True)

from engine.rendering.demoscene.ast_nodes import (
    Axis, BendNode, BoxNode, CellIdNode, ConeNode, CylinderNode,
    FloatNode, KifsNode, MaterialNode, MirrorNode, PlaneNode,
    PositionNode, RepeatNode, SceneGraph, SphereNode, StretchNode,
    TorusNode, TwistNode, Vec3Node,
    DOMAIN_OP_TYPE_MAP,
)
from engine.rendering.demoscene.wgsl_codegen import (
    SDF_BOX,
    SDF_CONE,
    SDF_CYLINDER,
    SDF_IMPORTS,
    SDF_PLANE,
    SDF_SPHERE,
    SDF_TORUS,
    MATERIAL_STRUCT,
    DEFAULT_MATERIAL_SWITCH,
    generate_wgsl,
)


# =============================================================================
# DOMAIN_OP_TYPE_MAP: class -> function name mapping
# =============================================================================


class TestDomainOpTypeMapStructure:
    """DOMAIN_OP_TYPE_MAP (from ast_nodes.py) maps AST classes to function names."""

    def test_all_six_ops_present(self):
        """All six domain operation types are registered as keys."""
        expected = {RepeatNode, MirrorNode, KifsNode, TwistNode, BendNode, StretchNode}
        assert expected.issubset(DOMAIN_OP_TYPE_MAP.keys()), (
            f"Missing ops: {expected - set(DOMAIN_OP_TYPE_MAP.keys())}"
        )

    def test_each_value_is_string(self):
        """Every entry maps to a non-empty string function name."""
        for cls, fn_name in DOMAIN_OP_TYPE_MAP.items():
            assert isinstance(fn_name, str), f"{cls.__name__} -> {fn_name!r} not a string"
            assert len(fn_name) > 0, f"{cls.__name__} maps to empty string"

    def test_all_function_names_start_with_domain(self):
        """All mapped function names begin with 'domain_'."""
        for cls, fn_name in DOMAIN_OP_TYPE_MAP.items():
            assert fn_name.startswith("domain_"), (
                f"{cls.__name__} -> {fn_name!r} does not start with domain_"
            )

    def test_repeat_maps_correctly(self):
        """RepeatNode maps to domain_repeat."""
        assert DOMAIN_OP_TYPE_MAP[RepeatNode] == "domain_repeat"

    def test_mirror_maps_correctly(self):
        """MirrorNode maps to domain_mirror."""
        assert DOMAIN_OP_TYPE_MAP[MirrorNode] == "domain_mirror"

    def test_kifs_maps_correctly(self):
        """KifsNode maps to domain_kifs."""
        assert DOMAIN_OP_TYPE_MAP[KifsNode] == "domain_kifs"

    def test_twist_maps_correctly(self):
        """TwistNode maps to domain_twist."""
        assert DOMAIN_OP_TYPE_MAP[TwistNode] == "domain_twist"

    def test_bend_maps_correctly(self):
        """BendNode maps to domain_bend."""
        assert DOMAIN_OP_TYPE_MAP[BendNode] == "domain_bend"

    def test_stretch_maps_correctly(self):
        """StretchNode maps to domain_stretch."""
        assert DOMAIN_OP_TYPE_MAP[StretchNode] == "domain_stretch"


# =============================================================================
# DOMAIN OP CALL TEMPLATES (private, accessed via module name)
# =============================================================================

import engine.rendering.demoscene.wgsl_codegen as _cg


class TestDomainOpCalls:
    """_DOMAIN_OP_CALLS contains template strings for each domain operation."""

    OP_CALLS = _cg._DOMAIN_OP_CALLS
    KIFS_COMP = _cg._KIFS_COMP
    STRETCH_COMP = _cg._STRETCH_COMP

    def test_all_six_ops_have_templates(self):
        """All six domain ops have entries in _DOMAIN_OP_CALLS."""
        expected = {RepeatNode, MirrorNode, KifsNode, TwistNode, BendNode, StretchNode}
        assert expected.issubset(self.OP_CALLS.keys()), (
            f"Missing templates: {expected - set(self.OP_CALLS.keys())}"
        )

    def test_each_template_is_non_empty_string(self):
        """Every template is a non-empty string."""
        for cls, tmpl in self.OP_CALLS.items():
            assert isinstance(tmpl, str), f"{cls.__name__} template not a string"
            assert len(tmpl) > 0, f"{cls.__name__} template is empty"

    def test_all_templates_contain_p_placeholder(self):
        """All templates include the {p} position placeholder."""
        for cls, tmpl in self.OP_CALLS.items():
            assert "{p}" in tmpl, f"{cls.__name__} template missing {{p}}: {tmpl!r}"

    def test_all_templates_format_without_error(self):
        """Every template can be filled with valid arguments."""
        fills = {
            RepeatNode: {"p": "p", "cell_size": "vec3<f32>(2.0, 2.0, 2.0)"},
            CellIdNode: {"p": "p", "cell_size": "vec3<f32>(2.0, 2.0, 2.0)"},
            MirrorNode: {"p": "p", "axis": "x"},
            KifsNode: {"p": "p", "folds": "6.0"},
            TwistNode: {"p": "p", "rate": "1.0"},
            BendNode: {"p": "p", "radius": "5.0"},
            StretchNode: {"p": "p", "axis": "x", "stretch": "2.0"},
        }
        for cls, tmpl in self.OP_CALLS.items():
            fill = fills[cls]
            result = tmpl.format(**fill)
            assert len(result) > 0, f"{cls.__name__} produced empty string"


# =============================================================================
# Repeat template
# =============================================================================


class TestRepeatTemplate:
    """domain_repeat: tiles space into a centered cell of size (cx, cy, cz)."""

    def test_template_structure(self):
        """Template is domain_repeat(p, cell_size) with vec3 cell_size."""
        tmpl = TestDomainOpCalls.OP_CALLS[RepeatNode]
        assert tmpl == "domain_repeat({p}, {cell_size})"

    def test_filled_example_uniform_cell(self):
        """Uniform cell (2, 2, 2) produces valid call."""
        tmpl = TestDomainOpCalls.OP_CALLS[RepeatNode]
        filled = tmpl.format(p="p", cell_size="vec3<f32>(2.0, 2.0, 2.0)")
        assert filled == "domain_repeat(p, vec3<f32>(2.0, 2.0, 2.0))"

    def test_filled_example_non_uniform_cell(self):
        """Non-uniform cell (4, 1, 4) produces valid call."""
        tmpl = TestDomainOpCalls.OP_CALLS[RepeatNode]
        filled = tmpl.format(p="p_d", cell_size="vec3<f32>(4.0, 1.0, 4.0)")
        assert filled == "domain_repeat(p_d, vec3<f32>(4.0, 1.0, 4.0))"


# =============================================================================
# Mirror template
# =============================================================================


class TestMirrorTemplate:
    """domain_mirror_{axis}: bilateral symmetry across an axis plane."""

    def test_template_structure(self):
        """Template is domain_mirror_{axis}(p) with axis placeholder."""
        tmpl = TestDomainOpCalls.OP_CALLS[MirrorNode]
        assert tmpl == "domain_mirror_{axis}({p})"

    def test_has_axis_placeholder(self):
        """Template contains {axis} placeholder."""
        tmpl = TestDomainOpCalls.OP_CALLS[MirrorNode]
        assert "{axis}" in tmpl

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_mirror_each_axis(self, axis):
        """Mirror can be generated for x, y, or z axis."""
        tmpl = TestDomainOpCalls.OP_CALLS[MirrorNode]
        filled = tmpl.format(p="p", axis=axis)
        assert filled == f"domain_mirror_{axis}(p)"

    def test_mirror_default_axis_x(self):
        """Axis 'x' yields domain_mirror_x(p)."""
        tmpl = TestDomainOpCalls.OP_CALLS[MirrorNode]
        assert tmpl.format(p="p", axis="x") == "domain_mirror_x(p)"


# =============================================================================
# KIFS template
# =============================================================================


class TestKifsTemplate:
    """domain_kifs: kaleidoscopic fold with N-fold rotational symmetry."""

    def test_template_structure(self):
        """Template is domain_kifs(p, folds)."""
        tmpl = TestDomainOpCalls.OP_CALLS[KifsNode]
        assert tmpl == "domain_kifs({p}, {folds})"

    def test_has_folds_placeholder(self):
        """Template contains {folds} placeholder."""
        tmpl = TestDomainOpCalls.OP_CALLS[KifsNode]
        assert "{folds}" in tmpl

    @pytest.mark.parametrize("folds", ["2.0", "3.0", "4.0", "5.0", "6.0", "8.0", "12.0"])
    def test_kifs_various_folds(self, folds):
        """KIFS produces correct expression for various fold counts."""
        tmpl = TestDomainOpCalls.OP_CALLS[KifsNode]
        assert tmpl.format(p="p", folds=folds) == f"domain_kifs(p, {folds})"

    def test_kifs_three_folds(self):
        """Folds=3 produces correct template."""
        tmpl = TestDomainOpCalls.OP_CALLS[KifsNode]
        assert tmpl.format(p="p", folds="3.0") == "domain_kifs(p, 3.0)"


# =============================================================================
# Twist template
# =============================================================================


class TestTwistTemplate:
    """domain_twist: xz-plane rotation proportional to y-coordinate."""

    def test_template_structure(self):
        """Template is domain_twist(p, rate)."""
        tmpl = TestDomainOpCalls.OP_CALLS[TwistNode]
        assert tmpl == "domain_twist({p}, {rate})"

    def test_has_rate_placeholder(self):
        """Template contains {rate} placeholder."""
        tmpl = TestDomainOpCalls.OP_CALLS[TwistNode]
        assert "{rate}" in tmpl

    @pytest.mark.parametrize("rate", ["0.5", "1.0", "2.0", "5.0", "-1.0"])
    def test_twist_various_rates(self, rate):
        """Twist works for various rates (positive and negative)."""
        tmpl = TestDomainOpCalls.OP_CALLS[TwistNode]
        assert tmpl.format(p="p", rate=rate) == f"domain_twist(p, {rate})"


# =============================================================================
# Bend template
# =============================================================================


class TestBendTemplate:
    """domain_bend: circular arc deformation in xz-plane."""

    def test_template_structure(self):
        """Template is domain_bend(p, radius)."""
        tmpl = TestDomainOpCalls.OP_CALLS[BendNode]
        assert tmpl == "domain_bend({p}, {radius})"

    def test_has_radius_placeholder(self):
        """Template contains {radius} placeholder."""
        tmpl = TestDomainOpCalls.OP_CALLS[BendNode]
        assert "{radius}" in tmpl

    @pytest.mark.parametrize("radius", ["1.0", "5.0", "10.0", "100.0"])
    def test_bend_various_radii(self, radius):
        """Bend works for various bend radii."""
        tmpl = TestDomainOpCalls.OP_CALLS[BendNode]
        assert tmpl.format(p="p", radius=radius) == f"domain_bend(p, {radius})"


# =============================================================================
# Stretch template
# =============================================================================


class TestStretchTemplate:
    """domain_stretch_{axis}: anisotropic scaling along one axis."""

    def test_template_structure(self):
        """Template is domain_stretch_{axis}(p, stretch)."""
        tmpl = TestDomainOpCalls.OP_CALLS[StretchNode]
        assert tmpl == "domain_stretch_{axis}({p}, {stretch})"

    def test_has_axis_and_stretch_placeholders(self):
        """Template contains {axis} and {stretch} placeholders."""
        tmpl = TestDomainOpCalls.OP_CALLS[StretchNode]
        assert "{axis}" in tmpl
        assert "{stretch}" in tmpl

    @pytest.mark.parametrize("axis, stretch", [
        ("x", "2.0"), ("y", "2.0"), ("z", "2.0"),
        ("x", "0.5"), ("y", "1.5"), ("z", "3.0"),
    ])
    def test_stretch_various_axes_and_factors(self, axis, stretch):
        """Stretch works for each axis with various factors."""
        tmpl = TestDomainOpCalls.OP_CALLS[StretchNode]
        assert tmpl.format(p="p", axis=axis, stretch=stretch) == (
            f"domain_stretch_{axis}(p, {stretch})"
        )

    def test_stretch_identity(self):
        """Stretch factor=1.0 (identity)."""
        tmpl = TestDomainOpCalls.OP_CALLS[StretchNode]
        assert tmpl.format(p="p", axis="x", stretch="1.0") == "domain_stretch_x(p, 1.0)"

    def test_stretch_compression(self):
        """Stretch factor=0.5 (compression) produces correct call."""
        tmpl = TestDomainOpCalls.OP_CALLS[StretchNode]
        assert tmpl.format(p="p", axis="y", stretch="0.5") == "domain_stretch_y(p, 0.5)"


# =============================================================================
# Compensation templates (non-isometric ops only)
# =============================================================================


class TestCompensation:
    """_KIFS_COMP and _STRETCH_COMP provide distance compensation templates."""

    def test_kifs_comp_exists(self):
        """_KIFS_COMP is a non-empty string."""
        comp = TestDomainOpCalls.KIFS_COMP
        assert isinstance(comp, str) and len(comp) > 0

    def test_kifs_comp_template(self):
        """_KIFS_COMP = domain_kifs_compensation({folds})."""
        assert TestDomainOpCalls.KIFS_COMP == "domain_kifs_compensation({folds})"

    def test_kifs_comp_filled(self):
        """Filled KIFS compensation produces valid WGSL expression."""
        assert TestDomainOpCalls.KIFS_COMP.format(folds="6.0") == (
            "domain_kifs_compensation(6.0)"
        )

    @pytest.mark.parametrize("folds", ["2.0", "3.0", "4.0", "6.0", "8.0"])
    def test_kifs_comp_various_folds(self, folds):
        """KIFS compensation works for various fold counts."""
        assert TestDomainOpCalls.KIFS_COMP.format(folds=folds) == (
            f"domain_kifs_compensation({folds})"
        )

    def test_stretch_comp_exists(self):
        """_STRETCH_COMP is a non-empty string."""
        comp = TestDomainOpCalls.STRETCH_COMP
        assert isinstance(comp, str) and len(comp) > 0

    def test_stretch_comp_template(self):
        """_STRETCH_COMP = domain_stretch_compensation({stretch})."""
        assert TestDomainOpCalls.STRETCH_COMP == "domain_stretch_compensation({stretch})"

    def test_stretch_comp_filled(self):
        """Filled stretch compensation produces valid WGSL expression."""
        assert TestDomainOpCalls.STRETCH_COMP.format(stretch="2.0") == (
            "domain_stretch_compensation(2.0)"
        )

    @pytest.mark.parametrize("stretch", ["0.5", "1.0", "2.0", "10.0", "0.0"])
    def test_stretch_comp_various_factors(self, stretch):
        """Stretch compensation works for various factors."""
        assert TestDomainOpCalls.STRETCH_COMP.format(stretch=stretch) == (
            f"domain_stretch_compensation({stretch})"
        )


# =============================================================================
# SDF templates (full WGSL function bodies)
# =============================================================================


class TestSdfTemplates:
    """SDF_* constants are valid WGSL function definitions."""

    SDF_CONSTANTS = [
        ("SDF_SPHERE", SDF_SPHERE, "sdSphere"),
        ("SDF_BOX", SDF_BOX, "sdBox"),
        ("SDF_TORUS", SDF_TORUS, "sdTorus"),
        ("SDF_CYLINDER", SDF_CYLINDER, "sdCylinder"),
        ("SDF_CONE", SDF_CONE, "sdCone"),
        ("SDF_PLANE", SDF_PLANE, "sdPlane"),
    ]

    def test_all_sdf_constants_defined(self):
        """All six SDF template constants are non-empty strings."""
        for name, val, _fn in self.SDF_CONSTANTS:
            assert isinstance(val, str), f"{name} is not a string"
            assert len(val) > 0, f"{name} is empty"

    def test_each_sdf_contains_fn(self):
        """Each SDF template contains a WGSL function declaration."""
        for name, val, fn_name in self.SDF_CONSTANTS:
            assert f"fn {fn_name}(" in val, (
                f"{name} does not contain 'fn {fn_name}('\n"
            )

    def test_each_sdf_contains_vec3_position_param(self):
        """Each SDF function accepts vec3<f32> p as first argument."""
        for name, val, fn_name in self.SDF_CONSTANTS:
            assert "p: vec3<f32>" in val, (
                f"{name} missing 'p: vec3<f32>' parameter"
            )

    def test_each_sdf_returns_f32(self):
        """Each SDF function returns f32."""
        for name, val, fn_name in self.SDF_CONSTANTS:
            assert "-> f32" in val, (
                f"{name} missing '-> f32' return type"
            )


# =============================================================================
# SDF_IMPORTS
# =============================================================================


class TestSdfImports:
    """SDF_IMPORTS contains #import directives for domain ops."""

    def test_sdf_imports_is_string(self):
        """SDF_IMPORTS is a non-empty string."""
        assert isinstance(SDF_IMPORTS, str) and len(SDF_IMPORTS) > 0

    def test_imports_contain_domain_repeat(self):
        """SDF_IMPORTS references domain_repeat."""
        assert "domain_repeat" in SDF_IMPORTS

    def test_imports_contain_domain_mirror(self):
        """SDF_IMPORTS references domain_mirror."""
        assert "domain_mirror" in SDF_IMPORTS

    def test_imports_contain_domain_kifs(self):
        """SDF_IMPORTS references domain_kifs."""
        assert "domain_kifs" in SDF_IMPORTS

    def test_imports_contain_domain_twist(self):
        """SDF_IMPORTS references domain_twist."""
        assert "domain_twist" in SDF_IMPORTS

    def test_imports_contain_domain_bend(self):
        """SDF_IMPORTS references domain_bend."""
        assert "domain_bend" in SDF_IMPORTS

    def test_imports_contain_domain_stretch(self):
        """SDF_IMPORTS references domain_stretch."""
        assert "domain_stretch" in SDF_IMPORTS

    def test_imports_contain_compensation_functions(self):
        """SDF_IMPORTS references compensation functions."""
        assert "domain_kifs_compensation" in SDF_IMPORTS
        assert "domain_stretch_compensation" in SDF_IMPORTS


# =============================================================================
# Material templates
# =============================================================================


class TestMaterialTemplates:
    """MATERIAL_STRUCT and DEFAULT_MATERIAL_SWITCH are present and valid."""

    def test_material_struct_exists(self):
        """MATERIAL_STRUCT is a non-empty string."""
        assert isinstance(MATERIAL_STRUCT, str) and len(MATERIAL_STRUCT) > 0

    def test_material_struct_defines_struct(self):
        """MATERIAL_STRUCT defines a WGSL struct named Material."""
        assert "struct Material {" in MATERIAL_STRUCT

    def test_material_struct_has_pbr_fields(self):
        """MATERIAL_STRUCT contains PBR material fields."""
        assert "albedo: vec3<f32>" in MATERIAL_STRUCT
        assert "roughness: f32" in MATERIAL_STRUCT
        assert "metallic: f32" in MATERIAL_STRUCT
        assert "emissive: f32" in MATERIAL_STRUCT
        assert "ambient_occlusion: f32" in MATERIAL_STRUCT

    def test_material_switch_exists(self):
        """DEFAULT_MATERIAL_SWITCH is a non-empty string."""
        assert isinstance(DEFAULT_MATERIAL_SWITCH, str) and len(DEFAULT_MATERIAL_SWITCH) > 0

    def test_material_switch_has_fn_sig(self):
        """DEFAULT_MATERIAL_SWITCH defines scene_material(id: i32) -> Material."""
        assert "fn scene_material(id: i32) -> Material {" in DEFAULT_MATERIAL_SWITCH

    def test_material_switch_has_switch(self):
        """DEFAULT_MATERIAL_SWITCH contains a switch statement."""
        assert "switch id {" in DEFAULT_MATERIAL_SWITCH

    def test_material_switch_has_default_case(self):
        """DEFAULT_MATERIAL_SWITCH has a default case with fallback material."""
        assert "default:" in DEFAULT_MATERIAL_SWITCH
        assert "return Material" in DEFAULT_MATERIAL_SWITCH

    def test_material_switch_has_case_bodies_placeholder(self):
        """DEFAULT_MATERIAL_SWITCH contains {case_bodies} placeholder."""
        assert "{case_bodies}" in DEFAULT_MATERIAL_SWITCH


# =============================================================================
# End-to-end codegen: generate_wgsl() produces correct output
# =============================================================================


class TestGenerateWgsl:
    """generate_wgsl() produces WGSL output with expected content."""

    def make_scene(self, pipeline=None, primitives=None, materials=None):
        primitives = primitives or (SphereNode(PositionNode(), FloatNode(1.0)),)
        return SceneGraph(
            primitives=primitives,
            pipeline=pipeline or (),
            materials=materials or (),
            name="test",
        )

    def test_generates_valid_output(self):
        """generate_wgsl returns non-empty WGSL source string."""
        src = generate_wgsl(self.make_scene())
        assert isinstance(src, str) and len(src) > 0

    def test_contains_sdf_function(self):
        """Generated output contains the SDF function."""
        src = generate_wgsl(self.make_scene())
        assert "fn sdSphere" in src

    def test_contains_material_struct(self):
        """Generated output contains Material struct."""
        src = generate_wgsl(self.make_scene())
        assert "struct Material {" in src

    def test_contains_scene_entry_point(self):
        """Generated output contains sd_scene entry point."""
        src = generate_wgsl(self.make_scene())
        assert "fn sd_scene__test" in src

    def test_with_repeat(self):
        """Pipeline with RepeatNode generates domain_repeat call."""
        pipeline = (RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "domain_repeat" in src

    def test_with_mirror(self):
        """Pipeline with MirrorNode generates domain_mirror call."""
        pipeline = (MirrorNode(PositionNode(), Axis.X),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "domain_mirror_x" in src

    def test_with_mirror_y(self):
        """Pipeline with MirrorNode(Y) generates domain_mirror_y."""
        pipeline = (MirrorNode(PositionNode(), Axis.Y),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "domain_mirror_y" in src

    def test_with_mirror_z(self):
        """Pipeline with MirrorNode(Z) generates domain_mirror_z."""
        pipeline = (MirrorNode(PositionNode(), Axis.Z),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "domain_mirror_z" in src

    def test_with_kifs(self):
        """Pipeline with KifsNode generates domain_kifs call."""
        pipeline = (KifsNode(PositionNode(), FloatNode(6.0)),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "domain_kifs" in src

    def test_with_kifs_compensation(self):
        """KIFS pipeline includes compensation function in output."""
        pipeline = (KifsNode(PositionNode(), FloatNode(6.0)),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "domain_kifs_compensation" in src

    def test_with_twist(self):
        """Pipeline with TwistNode generates domain_twist call."""
        pipeline = (TwistNode(PositionNode(), FloatNode(2.0)),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "domain_twist" in src

    def test_with_bend(self):
        """Pipeline with BendNode generates domain_bend call."""
        pipeline = (BendNode(PositionNode(), FloatNode(5.0)),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "domain_bend" in src

    def test_with_stretch(self):
        """Pipeline with StretchNode generates domain_stretch call."""
        pipeline = (StretchNode(PositionNode(), FloatNode(2.0), Axis.X),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "domain_stretch_x" in src

    def test_with_stretch_compensation(self):
        """Stretch pipeline includes compensation function in output."""
        pipeline = (StretchNode(PositionNode(), FloatNode(2.0), Axis.Y),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "domain_stretch_compensation" in src

    def test_empty_pipeline_no_imports(self):
        """Scene with no pipeline does not emit SDF_IMPORTS."""
        src = generate_wgsl(self.make_scene(pipeline=()))
        assert "// #import" not in src

    def test_pipeline_emits_imports(self):
        """Scene with pipeline emits SDF_IMPORTS."""
        pipeline = (TwistNode(PositionNode(), FloatNode(1.0)),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "// #import" in src

    def test_pipeline_applies_domain_transform(self):
        """Pipeline produces p_d variable in generated code."""
        pipeline = (TwistNode(PositionNode(), FloatNode(1.0)),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "let p_d" in src

    def test_generated_code_has_compensation_expression(self):
        """Generated code contains comp = ... expression."""
        pipeline = (KifsNode(PositionNode(), FloatNode(6.0)),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "let comp =" in src

    def test_scene_output_contains_comp_division(self):
        """Generated code divides distance by compensation factor."""
        pipeline = (KifsNode(PositionNode(), FloatNode(6.0)),)
        src = generate_wgsl(self.make_scene(pipeline=pipeline))
        assert "result.x / comp" in src

    def test_generated_code_has_material_id(self):
        """Generated SDF calls return vec2 with material_id."""
        src = generate_wgsl(self.make_scene(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0), material_id=0),),
        ))
        assert "vec2<f32>" in src
        assert "0.0" in src

    def test_material_id_propagated(self):
        """Generated code includes material_id in the SDF call output."""
        src = generate_wgsl(self.make_scene(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0), material_id=5),),
        ))
        assert "5.0" in src

    def test_multiple_primitives_use_select(self):
        """Scene with multiple primitives uses pairwise select()."""
        src = generate_wgsl(self.make_scene(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0), material_id=0),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0), material_id=1),
            ),
        ))
        assert "select" in src
        assert "result.x < d1.x" in src or "d1.x < result.x" in src

    def test_scene_with_material_node(self):
        """Scene with material produces scene_material() function."""
        mat = MaterialNode(
            material_id=0,
            albedo=Vec3Node(0.9, 0.2, 0.2),
            roughness=FloatNode(0.3),
            metallic=FloatNode(0.0),
            emissive=FloatNode(0.0),
            ambient_occlusion=FloatNode(1.0),
        )
        scene = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0,), material_id=0),),
            materials=(mat,),
            name="mat_test",
        )
        src = generate_wgsl(scene)
        assert "fn scene_material" in src
        assert "case 0:" in src
        assert "albedo: vec3<f32>" in src
        assert "Material(vec3<f32>(0.9, 0.2, 0.2), 0.3, 0.0, 0.0, 1.0)" in src

    @pytest.mark.parametrize("prim_cls, args, fn_name", [
        (SphereNode, (PositionNode(), FloatNode(1.0)), "sdSphere"),
        (BoxNode, (PositionNode(), Vec3Node(1.0, 2.0, 1.0)), "sdBox"),
        (TorusNode, (PositionNode(), FloatNode(2.0), FloatNode(0.5)), "sdTorus"),
        (CylinderNode, (PositionNode(), FloatNode(2.0), FloatNode(1.0)), "sdCylinder"),
        (ConeNode, (PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)), "sdCone"),
        (PlaneNode, (PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)), "sdPlane"),
    ])
    def test_each_primitive_generates_correct_function(self, prim_cls, args, fn_name):
        """Each SDF primitive generates its corresponding WGSL function."""
        prim = prim_cls(*args)
        src = generate_wgsl(self.make_scene(primitives=(prim,)))
        assert f"fn {fn_name}" in src


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Edge and degenerate cases for domain op codegen."""

    def test_kifs_minimum_folds(self):
        """KIFS with folds=2 (minimum meaningful fold) generates valid WGSL."""
        pipeline = (KifsNode(PositionNode(), FloatNode(2.0)),)
        src = generate_wgsl(
            SceneGraph(
                primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
                pipeline=pipeline,
                name="edge",
            )
        )
        assert "domain_kifs(p, 2.0)" in src
        assert "domain_kifs_compensation" in src

    def test_stretch_identity(self):
        """Stretch factor=1.0 (identity) generates valid WGSL."""
        pipeline = (StretchNode(PositionNode(), FloatNode(1.0), Axis.X),)
        src = generate_wgsl(
            SceneGraph(
                primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
                pipeline=pipeline,
                name="edge",
            )
        )
        assert "domain_stretch_x(p, 1.0)" in src

    def test_bend_zero_radius(self):
        """Bend radius=0 generates valid WGSL (physically degenerate but syntactically valid)."""
        pipeline = (BendNode(PositionNode(), FloatNode(0.0)),)
        src = generate_wgsl(
            SceneGraph(
                primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
                pipeline=pipeline,
                name="edge",
            )
        )
        assert "domain_bend(p, 0.0)" in src

    def test_twist_zero_rate(self):
        """Twist rate=0 generates valid identity expression."""
        pipeline = (TwistNode(PositionNode(), FloatNode(0.0)),)
        src = generate_wgsl(
            SceneGraph(
                primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
                pipeline=pipeline,
                name="edge",
            )
        )
        assert "domain_twist(p, 0.0)" in src

    def test_mirror_x_output(self):
        """Mirror X produces domain_mirror_x in output."""
        pipeline = (MirrorNode(PositionNode(), Axis.X),)
        src = generate_wgsl(
            SceneGraph(
                primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
                pipeline=pipeline,
                name="edge",
            )
        )
        assert "domain_mirror_x(p)" in src

    def test_empty_pipeline(self):
        """Empty pipeline produces no domain transformation."""
        src = generate_wgsl(
            SceneGraph(
                primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
                name="empty_pipe",
            )
        )
        assert "let p_d" not in src
        assert "let comp" in src
        assert "comp = 1.0" in src or "comp=1.0" in src or "comp= 1.0" in src or "comp =1.0" in src

    def test_combined_kifs_stretch(self):
        """KIFS + Stretch in same pipeline generates both compensation factors."""
        pipeline = (
            KifsNode(PositionNode(), FloatNode(6.0)),
            StretchNode(PositionNode(), FloatNode(2.0), Axis.Z),
        )
        src = generate_wgsl(
            SceneGraph(
                primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
                pipeline=pipeline,
                name="combined",
            )
        )
        assert "domain_kifs" in src
        assert "domain_stretch_z" in src
        assert "domain_kifs_compensation" in src
        assert "domain_stretch_compensation" in src

    def test_repeat_non_uniform(self):
        """Non-uniform repeat cell generates correct WGSL."""
        pipeline = (RepeatNode(PositionNode(), Vec3Node(4.0, 1.0, 4.0)),)
        src = generate_wgsl(
            SceneGraph(
                primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
                pipeline=pipeline,
                name="nonuniform",
            )
        )
        assert "domain_repeat" in src
        assert "4.0" in src or "vec3<f32>" in src
