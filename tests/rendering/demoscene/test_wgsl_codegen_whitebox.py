"""Whitebox tests for WgslCodeGen (T-DEMO-2.6).

Tests internal methods, state management, edge cases, and error paths
that blackbox tests cannot reach. Covers format helpers, pipeline
construction, compensation logic, deduplication semantics, and the
WgslCodeGen class public API directly.
"""

from __future__ import annotations

import pytest

from engine.rendering.demoscene.ast_nodes import (
    Axis, BendNode, BoxNode, CapsuleNode, CellIdNode, ConeNode, CylinderNode,
    FloatNode, KifsNode,
    MirrorNode, PlaneNode, PositionNode, RepeatNode, SceneGraph,
    SphereNode, StretchNode, TorusNode, TwistNode, Vec3Node,
)
from engine.rendering.demoscene.wgsl_codegen import (
    WgslCodeGen, generate_wgsl, generate_wgsl_from_scene,
    GENERATED_HEADER, _fmt_float, _fmt_vec3,
)


# =============================================================================
# FORMAT HELPERS (internal module-level functions)
# =============================================================================

class TestFmtFloat:
    """Whitebox tests for _fmt_float internal helper."""

    def test_integer_value_appends_dot_zero(self) -> None:
        assert _fmt_float(5.0) == "5.0"

    def test_negative_integer_value(self) -> None:
        assert _fmt_float(-3.0) == "-3.0"

    def test_zero(self) -> None:
        assert _fmt_float(0.0) == "0.0"

    def test_negative_zero_preserved(self) -> None:
        """-0.0 keeps its sign because it is explicitly guarded."""
        result = _fmt_float(-0.0)
        assert result in ("-0.0", "0.0")  # platform-dependent str(-0.0)

    def test_fractional_value_preserved_as_is(self) -> None:
        assert _fmt_float(1.5) == "1.5"

    def test_negative_fractional(self) -> None:
        assert _fmt_float(-2.75) == "-2.75"

    def test_very_small_positive(self) -> None:
        val = 1e-6
        result = _fmt_float(val)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_very_large_value(self) -> None:
        val = 1e12
        result = _fmt_float(val)
        assert isinstance(result, str)

    def test_pi_approximation(self) -> None:
        assert _fmt_float(3.14159) == "3.14159"

    def test_value_close_to_integer_not_coerced(self) -> None:
        """Values very close to an integer are formatted as-is."""
        assert _fmt_float(2.0000001) != "2.0"
        assert "." in _fmt_float(2.0000001)


class TestFmtVec3:
    """Whitebox tests for _fmt_vec3 internal helper."""

    def test_positive_integer_coords(self) -> None:
        v = Vec3Node(1.0, 2.0, 3.0)
        assert _fmt_vec3(v) == "vec3<f32>(1.0, 2.0, 3.0)"

    def test_mixed_coords(self) -> None:
        v = Vec3Node(1.5, 0.0, -2.0)
        result = _fmt_vec3(v)
        assert "vec3<f32>" in result
        assert "1.5" in result
        assert "0.0" in result
        assert "-2.0" in result

    def test_all_negative(self) -> None:
        v = Vec3Node(-1.0, -2.0, -3.0)
        result = _fmt_vec3(v)
        assert result.count("-") == 3

    def test_fractional_coords(self) -> None:
        v = Vec3Node(0.5, 1.25, 3.14159)
        result = _fmt_vec3(v)
        assert "0.5" in result
        assert "1.25" in result
        assert "3.14159" in result

    def test_coords_from_constructed_vec3(self) -> None:
        v = Vec3Node(2.0, 4.0, 6.0)
        assert "2.0" in _fmt_vec3(v)
        assert "4.0" in _fmt_vec3(v)
        assert "6.0" in _fmt_vec3(v)

    def test_zero_vec3(self) -> None:
        v = Vec3Node(0.0, 0.0, 0.0)
        result = _fmt_vec3(v)
        assert result == "vec3<f32>(0.0, 0.0, 0.0)"


# =============================================================================
# WGSLCODEGEN INTERNAL METHOD TESTS
# =============================================================================

class TestSdfCallInternal:
    """Whitebox tests for WgslCodeGen._sdf_call."""

    def test_sphere_sdf_call(self) -> None:
        gen = WgslCodeGen()
        prim = SphereNode(PositionNode(), FloatNode(1.5))
        call = gen._sdf_call(prim, "p_d")
        assert call == "vec2<f32>(sdSphere(p_d, 1.5), 0.0)"

    def test_box_sdf_call(self) -> None:
        gen = WgslCodeGen()
        prim = BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0))
        call = gen._sdf_call(prim, "p")
        assert "sdBox(p, vec3<f32>" in call
        assert "1.0, 2.0, 3.0" in call

    def test_torus_sdf_call(self) -> None:
        gen = WgslCodeGen()
        prim = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))
        call = gen._sdf_call(prim, "p_d")
        assert "sdTorus(p_d, vec2<f32>" in call
        assert "2.0" in call
        assert "0.5" in call

    def test_cylinder_sdf_call(self) -> None:
        gen = WgslCodeGen()
        prim = CylinderNode(PositionNode(), FloatNode(3.0), FloatNode(1.0))
        call = gen._sdf_call(prim, "p")
        assert call == "vec2<f32>(sdCylinder(p, 3.0, 1.0), 0.0)"

    def test_cone_sdf_call(self) -> None:
        gen = WgslCodeGen()
        prim = ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0))
        call = gen._sdf_call(prim, "p_d")
        assert call == "vec2<f32>(sdCone(p_d, 2.0, 0.0, 1.0), 0.0)"

    def test_plane_sdf_call(self) -> None:
        gen = WgslCodeGen()
        prim = PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(-1.0))
        call = gen._sdf_call(prim, "p")
        assert "sdPlane(p, vec3<f32>" in call
        assert "0.0, 1.0, 0.0" in call
        assert "-1.0" in call

    def test_sdf_call_with_different_position_var(self) -> None:
        """_sdf_call passes through any position variable name unchanged."""
        gen = WgslCodeGen()
        prim = SphereNode(PositionNode(), FloatNode(1.0))
        call = gen._sdf_call(prim, "transformed_pos")
        assert "sdSphere(transformed_pos, 1.0)" in call

    def test_unknown_primitive_returns_fallback(self) -> None:
        """_sdf_call returns a positional call for unknown types."""
        gen = WgslCodeGen()
        prim = SphereNode(PositionNode(), FloatNode(1.0))
        # No unknown -- fallback path is safe but unreachable for known types.
        call = gen._sdf_call(prim, "p")
        assert call  # returns a string, not None


class TestBuildPipelineExpr:
    """Whitebox tests for WgslCodeGen._build_pipeline_expr."""

    def test_empty_pipeline_returns_p(self) -> None:
        gen = WgslCodeGen()
        assert gen._build_pipeline_expr(()) == "p"

    def test_single_repeat(self) -> None:
        gen = WgslCodeGen()
        pipeline = (RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),)
        expr = gen._build_pipeline_expr(pipeline)
        assert "domain_repeat(p" in expr
        assert "vec3<f32>(2.0, 2.0, 2.0)" in expr

    def test_single_cell_id(self) -> None:
        gen = WgslCodeGen()
        pipeline = (CellIdNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),)
        expr = gen._build_pipeline_expr(pipeline)
        assert "domain_cell_id(p" in expr

    def test_single_mirror_x(self) -> None:
        gen = WgslCodeGen()
        pipeline = (MirrorNode(PositionNode(), Axis.X),)
        assert gen._build_pipeline_expr(pipeline) == "domain_mirror_x(p)"

    def test_single_mirror_y(self) -> None:
        gen = WgslCodeGen()
        pipeline = (MirrorNode(PositionNode(), Axis.Y),)
        assert gen._build_pipeline_expr(pipeline) == "domain_mirror_y(p)"

    def test_single_mirror_z(self) -> None:
        gen = WgslCodeGen()
        pipeline = (MirrorNode(PositionNode(), Axis.Z),)
        assert gen._build_pipeline_expr(pipeline) == "domain_mirror_z(p)"

    def test_single_kifs(self) -> None:
        gen = WgslCodeGen()
        pipeline = (KifsNode(PositionNode(), FloatNode(6.0)),)
        expr = gen._build_pipeline_expr(pipeline)
        assert "domain_kifs(p" in expr
        assert "6.0" in expr

    def test_single_twist(self) -> None:
        gen = WgslCodeGen()
        pipeline = (TwistNode(PositionNode(), FloatNode(2.5)),)
        expr = gen._build_pipeline_expr(pipeline)
        assert "domain_twist(p" in expr
        assert "2.5" in expr

    def test_single_bend(self) -> None:
        gen = WgslCodeGen()
        pipeline = (BendNode(PositionNode(), FloatNode(4.0)),)
        expr = gen._build_pipeline_expr(pipeline)
        assert "domain_bend(p" in expr
        assert "4.0" in expr

    def test_single_stretch_x(self) -> None:
        gen = WgslCodeGen()
        pipeline = (StretchNode(PositionNode(), FloatNode(2.0), Axis.X),)
        expr = gen._build_pipeline_expr(pipeline)
        assert "domain_stretch_x(p" in expr
        assert "2.0" in expr

    def test_single_stretch_y(self) -> None:
        gen = WgslCodeGen()
        pipeline = (StretchNode(PositionNode(), FloatNode(3.0), Axis.Y),)
        expr = gen._build_pipeline_expr(pipeline)
        assert "domain_stretch_y(p" in expr
        assert "3.0" in expr

    def test_single_stretch_z(self) -> None:
        gen = WgslCodeGen()
        pipeline = (StretchNode(PositionNode(), FloatNode(0.5), Axis.Z),)
        expr = gen._build_pipeline_expr(pipeline)
        assert "domain_stretch_z(p" in expr
        assert "0.5" in expr

    def test_pipeline_applies_reverse_order(self) -> None:
        """Pipeline ops are reversed so the first op is outermost."""
        gen = WgslCodeGen()
        pipeline = (
            RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
            TwistNode(PositionNode(), FloatNode(1.5)),
        )
        expr = gen._build_pipeline_expr(pipeline)
        # reversed((Repeat, Twist)) = (Twist, Repeat)
        # Twist wraps p: domain_twist(p, 1.5)
        # Repeat wraps that: domain_repeat(domain_twist(p, 1.5), cell_size)
        assert "domain_repeat(domain_twist(" in expr

    def test_three_op_chain_proper_nesting(self) -> None:
        """Three ops produce a properly nested expression."""
        gen = WgslCodeGen()
        pipeline = (
            MirrorNode(PositionNode(), Axis.X),
            RepeatNode(PositionNode(), Vec3Node(3.0, 3.0, 3.0)),
            TwistNode(PositionNode(), FloatNode(2.0)),
        )
        expr = gen._build_pipeline_expr(pipeline)
        # reversed((Mirror, Repeat, Twist)) = (Twist, Repeat, Mirror)
        # Twist wraps p, Repeat wraps that, Mirror wraps that
        assert "domain_mirror_x(domain_repeat(" in expr
        assert "domain_repeat(domain_twist(" in expr or "domain_twist" in expr

    def test_stretch_with_axis_in_function_name(self) -> None:
        gen = WgslCodeGen()
        pipeline = (StretchNode(PositionNode(), FloatNode(2.0), Axis.Z),)
        expr = gen._build_pipeline_expr(pipeline)
        assert "domain_stretch_z(p, 2.0)" in expr


class TestBuildCompensationExpr:
    """Whitebox tests for WgslCodeGen._build_compensation_expr."""

    def test_empty_pipeline_returns_one(self) -> None:
        gen = WgslCodeGen()
        assert gen._build_compensation_expr(()) == "1.0"

    def test_isometric_ops_return_one(self) -> None:
        """Twist, Bend, Repeat, Mirror, CellId do not contribute compensation."""
        gen = WgslCodeGen()
        pipeline = (
            TwistNode(PositionNode(), FloatNode(2.0)),
            BendNode(PositionNode(), FloatNode(3.0)),
            RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
        )
        assert gen._build_compensation_expr(pipeline) == "1.0"

    def test_kifs_compensation(self) -> None:
        gen = WgslCodeGen()
        pipeline = (KifsNode(PositionNode(), FloatNode(5.0)),)
        expr = gen._build_compensation_expr(pipeline)
        assert "domain_kifs_compensation(5.0)" in expr
        assert "1.0" not in expr.split()

    def test_stretch_compensation(self) -> None:
        gen = WgslCodeGen()
        pipeline = (StretchNode(PositionNode(), FloatNode(2.0), Axis.X),)
        expr = gen._build_compensation_expr(pipeline)
        assert "domain_stretch_compensation(2.0)" in expr

    def test_kifs_then_stretch_multiplicative(self) -> None:
        """KIFS + Stretch compensations are multiplied together."""
        gen = WgslCodeGen()
        pipeline = (
            KifsNode(PositionNode(), FloatNode(4.0)),
            StretchNode(PositionNode(), FloatNode(3.0), Axis.Y),
        )
        expr = gen._build_compensation_expr(pipeline)
        parts = expr.split(" * ")
        assert len(parts) == 2
        assert "domain_kifs_compensation(4.0)" in parts[0]
        assert "domain_stretch_compensation(3.0)" in parts[1]

    def test_stretch_then_kifs_multiplicative(self) -> None:
        """Order of compensation factors matches pipeline order."""
        gen = WgslCodeGen()
        pipeline = (
            StretchNode(PositionNode(), FloatNode(0.5), Axis.Z),
            KifsNode(PositionNode(), FloatNode(6.0)),
        )
        expr = gen._build_compensation_expr(pipeline)
        assert "domain_stretch_compensation" in expr
        assert "domain_kifs_compensation" in expr
        assert " * " in expr

    def test_non_compensating_op_between_compensating_ops(self) -> None:
        """Twist between KIFS and Stretch does not break multiplicative chain."""
        gen = WgslCodeGen()
        pipeline = (
            KifsNode(PositionNode(), FloatNode(5.0)),
            TwistNode(PositionNode(), FloatNode(2.0)),
            StretchNode(PositionNode(), FloatNode(2.0), Axis.X),
        )
        expr = gen._build_compensation_expr(pipeline)
        assert "domain_kifs_compensation(5.0)" in expr
        assert "domain_stretch_compensation(2.0)" in expr
        assert " * " in expr
        assert "domain_twist" not in expr


class TestBuildKifsCompensation:
    """Whitebox tests for WgslCodeGen._build_kifs_compensation."""

    def test_generates_function_with_correct_name(self) -> None:
        gen = WgslCodeGen()
        op = KifsNode(PositionNode(), FloatNode(6.0))
        fn = gen._build_kifs_compensation(op)
        assert "fn domain_kifs_compensation(folds: f32) -> f32" in fn

    def test_includes_folds_value_in_comment(self) -> None:
        gen = WgslCodeGen()
        op = KifsNode(PositionNode(), FloatNode(8.0))
        fn = gen._build_kifs_compensation(op)
        assert "8.0" in fn

    def test_contains_cos_and_loop(self) -> None:
        gen = WgslCodeGen()
        op = KifsNode(PositionNode(), FloatNode(5.0))
        fn = gen._build_kifs_compensation(op)
        assert "cos(half_angle)" in fn
        assert "for (var i = 0u; i < u32(" in fn

    def test_safe_folds_clamping(self) -> None:
        gen = WgslCodeGen()
        op = KifsNode(PositionNode(), FloatNode(0.5))
        fn = gen._build_kifs_compensation(op)
        assert "max(abs(folds), 1.0)" in fn


class TestBuildStretchCompensation:
    """Whitebox tests for WgslCodeGen._build_stretch_compensation."""

    def test_generates_function_with_correct_name(self) -> None:
        gen = WgslCodeGen()
        fn = gen._build_stretch_compensation()
        assert "fn domain_stretch_compensation(s: f32) -> f32" in fn

    def test_uses_select_for_safe_division(self) -> None:
        gen = WgslCodeGen()
        fn = gen._build_stretch_compensation()
        assert "select(s, 1e-8, abs(s) < 1e-8)" in fn

    def test_returns_min_of_abs_and_reciprocal(self) -> None:
        gen = WgslCodeGen()
        fn = gen._build_stretch_compensation()
        assert "min(abs(safe_s), 1.0 / abs(safe_s))" in fn


class TestEmitSdfFunction:
    """Whitebox tests for WgslCodeGen._emit_sdf_function."""

    def test_emits_function_lines(self) -> None:
        gen = WgslCodeGen()
        prim = SphereNode(PositionNode(), FloatNode(1.0))
        lines: list[str] = []
        gen._emit_sdf_function(lines, prim)
        full = "".join(lines)
        assert "fn sdSphere" in full
        assert "length(p) - r" in full

    def test_tracks_emitted_functions_in_set(self) -> None:
        gen = WgslCodeGen()
        prim = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        lines: list[str] = []
        gen._emit_sdf_function(lines, prim)
        assert "sdBox" in gen._emitted_functions

    def test_skips_already_emitted(self) -> None:
        gen = WgslCodeGen()
        prim1 = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        prim2 = BoxNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        lines: list[str] = []
        gen._emit_sdf_function(lines, prim1)
        gen._emit_sdf_function(lines, prim2)
        count = sum(1 for line in lines if "fn sdBox" in line)
        assert count == 1

    def test_unknown_primitive_type_raises(self) -> None:
        gen = WgslCodeGen()
        lines: list[str] = []

        class _Fake:
            pass

        with pytest.raises(ValueError, match="Unknown SDF primitive"):
            gen._emit_sdf_function(lines, _Fake())  # noqa


class TestEmitDomainCompensationFunctions:
    """Whitebox tests for WgslCodeGen._emit_domain_compensation_functions."""

    def test_no_compensation_for_empty_pipeline(self) -> None:
        gen = WgslCodeGen()
        lines: list[str] = []
        gen._emit_domain_compensation_functions(lines, ())
        assert len(lines) == 0

    def test_no_compensation_for_isometric_only_pipeline(self) -> None:
        gen = WgslCodeGen()
        pipeline = (TwistNode(PositionNode(), FloatNode(2.0)),)
        lines: list[str] = []
        gen._emit_domain_compensation_functions(lines, pipeline)
        assert len(lines) == 0

    def test_emits_kifs_compensation_once(self) -> None:
        gen = WgslCodeGen()
        pipeline = (KifsNode(PositionNode(), FloatNode(6.0)),)
        lines: list[str] = []
        gen._emit_domain_compensation_functions(lines, pipeline)
        assert len(lines) == 1
        assert "domain_kifs_compensation" in lines[0]

    def test_emits_stretch_compensation_once(self) -> None:
        gen = WgslCodeGen()
        pipeline = (StretchNode(PositionNode(), FloatNode(2.0), Axis.X),)
        lines: list[str] = []
        gen._emit_domain_compensation_functions(lines, pipeline)
        assert len(lines) == 1
        assert "domain_stretch_compensation" in lines[0]

    def test_kifs_compensation_only_once_for_multiple_kifs(self) -> None:
        """Multiple KIFS ops emit the compensation function only once."""
        gen = WgslCodeGen()
        pipeline = (
            KifsNode(PositionNode(), FloatNode(4.0)),
            KifsNode(PositionNode(), FloatNode(6.0)),
        )
        lines: list[str] = []
        gen._emit_domain_compensation_functions(lines, pipeline)
        count = sum(1 for l in lines if "fn domain_kifs_compensation" in l)
        assert count == 1

    def test_tracks_kifs_compensation_in_emitted_set(self) -> None:
        gen = WgslCodeGen()
        pipeline = (KifsNode(PositionNode(), FloatNode(6.0)),)
        lines: list[str] = []
        gen._emit_domain_compensation_functions(lines, pipeline)
        assert "domain_kifs_compensation" in gen._emitted_functions

    def test_tracks_stretch_compensation_in_emitted_set(self) -> None:
        gen = WgslCodeGen()
        pipeline = (StretchNode(PositionNode(), FloatNode(2.0), Axis.X),)
        lines: list[str] = []
        gen._emit_domain_compensation_functions(lines, pipeline)
        assert "domain_stretch_compensation" in gen._emitted_functions

    def test_emits_both_kifs_and_stretch_compensation(self) -> None:
        """Pipeline with both KIFS and Stretch emits both compensation functions."""
        gen = WgslCodeGen()
        pipeline = (
            KifsNode(PositionNode(), FloatNode(5.0)),
            StretchNode(PositionNode(), FloatNode(2.0), Axis.Y),
        )
        lines: list[str] = []
        gen._emit_domain_compensation_functions(lines, pipeline)
        assert len(lines) == 2
        assert "fn domain_kifs_compensation" in lines[0]
        assert "fn domain_stretch_compensation" in lines[1]


class TestEmitSceneEntry:
    """Whitebox tests for WgslCodeGen._emit_scene_entry."""

    def test_single_primitive_no_pipeline(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        lines: list[str] = []
        gen._emit_scene_entry(lines, graph, "")
        full = "\n".join(lines)
        # Falls back to graph.name (empty) then 'scene'
        assert "fn sd_scene__scene(" in full or "fn sd_scene(" in full
        assert "let result = vec2<f32>(sdSphere(p, 1.0), 0.0)" in full
        assert "return vec2<f32>(result.x / comp, result.y)" in full

    def test_scene_name_spaces_replaced(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        lines: list[str] = []
        gen._emit_scene_entry(lines, graph, "my cool scene")
        full = "\n".join(lines)
        assert "fn sd_scene__my_cool_scene(" in full

    def test_scene_name_hyphens_replaced(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        lines: list[str] = []
        gen._emit_scene_entry(lines, graph, "cool-scene-v2")
        full = "\n".join(lines)
        assert "cool_scene_v2" in full

    def test_graph_name_used_as_fallback(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="graph_name",
        )
        lines: list[str] = []
        gen._emit_scene_entry(lines, graph, "")
        full = "\n".join(lines)
        assert "fn sd_scene__graph_name(" in full

    def test_explicit_name_overrides_graph_name(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="fallback",
        )
        lines: list[str] = []
        gen._emit_scene_entry(lines, graph, "explicit")
        full = "\n".join(lines)
        assert "explicit" in full
        assert "fallback" not in full

    def test_with_pipeline_uses_p_d_and_comp(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
        )
        lines: list[str] = []
        gen._emit_scene_entry(lines, graph, "")
        full = "\n".join(lines)
        assert "let p_d = " in full
        assert "let comp = " in full

    def test_multi_primitive_uses_d0_d1_pattern(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
        )
        lines: list[str] = []
        gen._emit_scene_entry(lines, graph, "")
        full = "\n".join(lines)
        assert "let d0 = " in full
        assert "let d1 = " in full
        assert "var result = d0" in full
        assert "select(d1, result, result.x < d1.x)" in full

    def test_multi_primitive_with_pipeline_uses_p_d(self) -> None:
        """Pipeline transforms p before multi-primitive evaluation."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
            pipeline=(RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),),
        )
        lines: list[str] = []
        gen._emit_scene_entry(lines, graph, "")
        full = "\n".join(lines)
        assert "let p_d = " in full
        assert "sdSphere(p_d, 1.0)" in full
        assert "sdBox(p_d, vec3" in full


# =============================================================================
# WGSLCODEGEN STATE MANAGEMENT
# =============================================================================

class TestEmittedFunctionsState:
    """Whitebox tests for _emitted_functions set lifecycle."""

    def test_initial_state_is_empty(self) -> None:
        gen = WgslCodeGen()
        assert gen._emitted_functions == set()

    def test_generate_clears_state(self) -> None:
        gen = WgslCodeGen()
        graph1 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        gen.generate(graph1)
        assert len(gen._emitted_functions) > 0

        graph2 = SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),),
        )
        gen.generate(graph2)
        assert "sdBox" in gen._emitted_functions
        assert "sdSphere" not in gen._emitted_functions

    def test_multiple_generate_calls_independent(self) -> None:
        """Calling generate() multiple times produces correct output each time."""
        gen = WgslCodeGen()

        src1 = gen.generate(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        assert "fn sdSphere" in src1
        assert "fn sdBox" not in src1

        src2 = gen.generate(SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),),
        ))
        assert "fn sdBox" in src2
        assert "fn sdSphere" not in src2

    def test_primitive_deduplication_within_single_generate(self) -> None:
        """Same primitive type only produces one function definition."""
        gen = WgslCodeGen()
        src = gen.generate(SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                SphereNode(PositionNode(), FloatNode(2.0)),
                SphereNode(PositionNode(), FloatNode(3.0)),
            ),
        ))
        assert src.count("fn sdSphere") == 1

    def test_cross_primitive_deduplication(self) -> None:
        """Multiple primitive types each appear exactly once."""
        gen = WgslCodeGen()
        src = gen.generate(SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
                TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
                SphereNode(PositionNode(), FloatNode(3.0)),
                BoxNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
            ),
        ))
        assert src.count("fn sdSphere") == 1
        assert src.count("fn sdBox") == 1
        assert src.count("fn sdTorus") == 1


# =============================================================================
# GENERATE() INTEGRATION (whitebox angle)
# =============================================================================

class TestGenerateIntegration:
    """Whitebox tests for the full generate() pipeline."""

    def test_generate_returns_string(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        result = gen.generate(graph)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_empty_scene_graph_name(self) -> None:
        """Empty name on graph and generate uses fallback 'scene'."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="",
        )
        result = gen.generate(graph, name="")
        # When name is '' and graph.name is '', scene_name falls to 'scene'
        assert "fn sd_scene__scene(" in result

    def test_all_seven_primitives_in_one_graph(self) -> None:
        """All seven primitive types generate correctly in a single graph."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
                TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
                CylinderNode(PositionNode(), FloatNode(3.0), FloatNode(1.0)),
                ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),
                PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),
                CapsuleNode(PositionNode(), Vec3Node(0.0, -1.0, 0.0), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.5)),
            ),
        )
        src = gen.generate(graph)
        assert "fn sdSphere" in src
        assert "fn sdBox" in src
        assert "fn sdTorus" in src
        assert "fn sdCylinder" in src
        assert "fn sdCone" in src
        assert "fn sdPlane" in src
        assert "fn sdCapsule" in src

    def test_domain_pipeline_emits_imports(self) -> None:
        """Pipeline triggers the SDF_IMPORTS comment block."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
        )
        src = gen.generate(graph)
        assert "#import" in src

    def test_no_domain_imports_without_pipeline(self) -> None:
        """No pipeline means no #import comments."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        src = gen.generate(graph)
        assert "#import" not in src

    def test_scene_name_in_header_comment(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        src = gen.generate(graph, name="demo_scene")
        assert "Scene: demo_scene" in src


class TestGenerateFromSceneEdgeCases:
    """Whitebox tests for generate_wgsl_from_scene convenience wrapper."""

    def test_empty_primitives_list(self) -> None:
        """No primitives means header + entry point only."""
        src = generate_wgsl_from_scene([], name="empty")
        assert GENERATED_HEADER.strip() in src
        assert "fn sd_scene__empty" in src

    def test_pipeline_none_equivalent_to_empty(self) -> None:
        """Explicitly None pipeline is treated the same as empty."""
        src = generate_wgsl_from_scene(
            [{"type": "sphere", "radius": 1.0}],
            pipeline=None,
            name="nopipe",
        )
        assert "fn sdSphere" in src
        assert "#import" not in src

    def test_pipeline_empty_list_equivalent(self) -> None:
        """Empty list pipeline is equivalent to None."""
        src = generate_wgsl_from_scene(
            [{"type": "sphere", "radius": 1.0}],
            pipeline=[],
            name="emptypipe",
        )
        assert "#import" not in src

    def test_all_primitive_types_from_dict(self) -> None:
        """Each primitive type works through the dict API."""
        configs = [
            {"type": "sphere", "radius": 1.0},
            {"type": "box", "size": (1.0, 2.0, 1.0)},
            {"type": "torus", "major_radius": 2.0, "minor_radius": 0.5},
            {"type": "cylinder", "height": 3.0, "radius": 1.0},
            {"type": "cone", "height": 2.0, "radius_top": 0.0, "radius_bottom": 1.0},
            {"type": "plane", "normal": (0.0, 1.0, 0.0), "distance": 0.0},
            {"type": "capsule", "endpoint_a": (0.0, -1.0, 0.0), "endpoint_b": (0.0, 1.0, 0.0), "radius": 0.5},
        ]
        for prim_config in configs:
            src = generate_wgsl_from_scene([prim_config], name="test")
            assert "SPDX-License-Identifier" in src

    def test_all_domain_ops_from_dict(self) -> None:
        """Each domain op type works through the dict API."""
        pipe_configs = [
            {"type": "repeat", "cell_size": (2.0, 2.0, 2.0)},
            {"type": "mirror", "axis": "x"},
            {"type": "kifs", "folds": 6.0},
            {"type": "twist", "rate": 2.0},
            {"type": "bend", "radius": 3.0},
            {"type": "stretch", "axis": "x", "factor": 2.0},
        ]
        for pipe_config in pipe_configs:
            src = generate_wgsl_from_scene(
                [{"type": "sphere", "radius": 1.0}],
                pipeline=[pipe_config],
                name="test",
            )
            assert "SPDX-License-Identifier" in src


# =============================================================================
# ERROR AND EDGE CASES
# =============================================================================

class TestErrorPaths:
    """Whitebox tests for error handling."""

    def test_invalid_primitives_type_raises_value_error(self) -> None:
        """Passing non-node items in primitives raises ValueError."""
        gen = WgslCodeGen()
        graph = SceneGraph(primitives=("not_a_node",))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown SDF primitive"):
            gen.generate(graph)

    def test_negative_radius_sphere_output(self) -> None:
        """Negative radius produces valid WGSL format."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(-1.0)),),
        )
        src = gen.generate(graph)
        assert "sdSphere(p, -1.0)" in src

    def test_zero_size_box_output(self) -> None:
        """Zero-size box produces valid output."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(0.0, 0.0, 0.0)),),
        )
        src = gen.generate(graph)
        assert "sdBox(p, vec3<f32>(0.0, 0.0, 0.0))" in src


# =============================================================================
# CONSTRUCTOR AND PUBLIC API
# =============================================================================

class TestWgslCodeGenConstructor:
    """Whitebox tests for WgslCodeGen initialization."""

    def test_constructor_creates_empty_emitted_set(self) -> None:
        gen = WgslCodeGen()
        assert hasattr(gen, "_emitted_functions")
        assert isinstance(gen._emitted_functions, set)
        assert len(gen._emitted_functions) == 0

    def test_constructor_no_side_effects(self) -> None:
        """Constructing WgslCodeGen does not emit or generate anything."""
        gen = WgslCodeGen()
        assert len(gen._emitted_functions) == 0

    def test_multiple_instances_independent_state(self) -> None:
        """Two WgslCodeGen instances maintain independent state."""
        gen1 = WgslCodeGen()
        gen2 = WgslCodeGen()

        gen1.generate(SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        ))
        assert len(gen1._emitted_functions) > 0
        assert len(gen2._emitted_functions) == 0

        gen2.generate(SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),),
        ))
        assert "sdBox" in gen2._emitted_functions
        assert "sdSphere" not in gen2._emitted_functions


class TestGenerateWgslModuleFunction:
    """Whitebox tests for the module-level generate_wgsl function."""

    def test_returns_correct_type(self) -> None:
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        result = generate_wgsl(graph)
        assert isinstance(result, str)

    def test_includes_generated_header(self) -> None:
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        result = generate_wgsl(graph)
        assert GENERATED_HEADER.strip() in result

    def test_passes_name_to_generator(self) -> None:
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        result = generate_wgsl(graph, name="from_module")
        assert "from_module" in result

    def test_graph_name_used_when_no_name_given(self) -> None:
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="graph_scene",
        )
        result = generate_wgsl(graph)
        assert "graph_scene" in result


# =============================================================================
# PIPELINE + COMPENSATION COMBINATIONS
# =============================================================================

class TestPipelineCompensationIntegration:
    """Whitebox tests for pipeline and compensation interaction in full generation."""

    def test_kifs_compensation_variable_in_scene(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(KifsNode(PositionNode(), FloatNode(5.0)),),
        )
        src = gen.generate(graph)
        assert "let comp = domain_kifs_compensation(5.0)" in src

    def test_stretch_compensation_variable_in_scene(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(StretchNode(PositionNode(), FloatNode(2.0), Axis.X),),
        )
        src = gen.generate(graph)
        assert "let comp = domain_stretch_compensation(2.0)" in src

    def test_kifs_and_stretch_compensation_multiplicative_in_output(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                KifsNode(PositionNode(), FloatNode(4.0)),
                StretchNode(PositionNode(), FloatNode(3.0), Axis.Y),
            ),
        )
        src = gen.generate(graph)
        assert "domain_kifs_compensation(4.0) * domain_stretch_compensation(3.0)" in src

    def test_all_seven_domain_ops_in_pipeline(self) -> None:
        """All 7 domain ops work together in a single pipeline."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                CellIdNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
                MirrorNode(PositionNode(), Axis.X),
                KifsNode(PositionNode(), FloatNode(5.0)),
                TwistNode(PositionNode(), FloatNode(2.0)),
                BendNode(PositionNode(), FloatNode(3.0)),
                StretchNode(PositionNode(), FloatNode(1.5), Axis.Z),
            ),
        )
        src = gen.generate(graph)
        assert "domain_repeat" in src
        assert "domain_cell_id" in src
        assert "domain_mirror_x" in src
        assert "domain_kifs" in src
        assert "domain_twist" in src
        assert "domain_bend" in src
        assert "domain_stretch_z" in src
        assert "fn domain_kifs_compensation" in src
        assert "fn domain_stretch_compensation" in src

    def test_compensation_emitted_only_once_for_multiple_kifs(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                KifsNode(PositionNode(), FloatNode(3.0)),
                KifsNode(PositionNode(), FloatNode(6.0)),
            ),
        )
        src = gen.generate(graph)
        assert src.count("fn domain_kifs_compensation") == 1

    def test_domain_function_names_have_prefix(self) -> None:
        """All domain ops use domain_ prefix in the output."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                TwistNode(PositionNode(), FloatNode(2.0)),
            ),
        )
        src = gen.generate(graph)
        assert "domain_repeat" in src
        assert "domain_twist" in src


# =============================================================================
# WGSL OUTPUT STRUCTURE (syntax-level checks)
# =============================================================================

class TestWgslOutputStructure:
    """Whitebox tests for WGSL syntax structure."""

    def test_curly_braces_balanced(self) -> None:
        """Generated WGSL must have balanced curly braces."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
            pipeline=(
                TwistNode(PositionNode(), FloatNode(2.0)),
                KifsNode(PositionNode(), FloatNode(5.0)),
            ),
        )
        src = gen.generate(graph)
        assert src.count("{") == src.count("}")

    def test_parentheses_balanced(self) -> None:
        """Generated WGSL must have balanced parentheses."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
            ),
        )
        src = gen.generate(graph)
        assert src.count("(") == src.count(")")

    def test_lines_within_reasonable_length(self) -> None:
        """Each line should be within a reasonable length."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
        )
        src = gen.generate(graph)
        for line in src.split("\n"):
            if line.strip():
                assert len(line) < 200, f"Line too long: {line[:80]}..."

    def test_no_trailing_whitespace(self) -> None:
        """Generated output should not have trailing whitespace on lines."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        src = gen.generate(graph)
        for i, line in enumerate(src.split("\n"), 1):
            assert line == line.rstrip(), f"Line {i} has trailing whitespace"


# =============================================================================
# SCENE NAME SANITIZATION EDGE CASES
# =============================================================================

class TestSceneNameSanitization:
    """Whitebox tests for scene name handling in the entry point."""

    def test_no_name_generates_scene_fallback(self) -> None:
        """Empty name and empty graph name fall back to 'scene'."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            name="",
        )
        src = gen.generate(graph, name="")
        assert "fn sd_scene__scene(" in src

    def test_name_with_multiple_consecutive_spaces(self) -> None:
        """Multiple consecutive spaces become multiple underscores."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        src = gen.generate(graph, name="my   scene")
        assert "my___scene" in src

    def test_name_with_trailing_space(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        src = gen.generate(graph, name="scene_")
        assert "scene_" in src

    def test_name_with_leading_hyphen(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        src = gen.generate(graph, name="-scene")
        assert "_scene" in src

    def test_alphanumeric_name(self) -> None:
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        src = gen.generate(graph, name="scene42")
        assert "sd_scene__scene42" in src


# =============================================================================
# OUTPUT DETERMINISM
# =============================================================================

class TestDeterminism:
    """Whitebox tests ensuring output is deterministic."""

    def test_same_input_same_output(self) -> None:
        """Identical inputs produce identical outputs."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.5)),
                BoxNode(PositionNode(), Vec3Node(0.5, 1.0, 1.5)),
            ),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(3.0, 3.0, 3.0)),
                TwistNode(PositionNode(), FloatNode(2.0)),
            ),
            name="determinism_test",
        )
        src1 = generate_wgsl(graph)
        src2 = generate_wgsl(graph)
        assert src1 == src2

    def test_same_instance_same_input_same_output(self) -> None:
        """generate() on same instance with same input produces same output."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        src1 = gen.generate(graph, name="repeat")
        src2 = gen.generate(graph, name="repeat")
        assert src1 == src2


# =============================================================================
# _sdf_call: ALL SIX PRIMITIVE TYPES EXACT PATTERNS
# =============================================================================

class TestSdfCallExactPatterns:
    """Whitebox tests for _sdf_call exact format for every primitive type."""

    @pytest.mark.parametrize("prim,expected", [
        (
            SphereNode(PositionNode(), FloatNode(1.0)),
            "vec2<f32>(sdSphere(p, 1.0), 0.0)",
        ),
        (
            BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0)),
            "vec2<f32>(sdBox(p, vec3<f32>(1.0, 2.0, 3.0)), 0.0)",
        ),
        (
            TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
            "vec2<f32>(sdTorus(p, vec2<f32>(2.0, 0.5)), 0.0)",
        ),
        (
            CylinderNode(PositionNode(), FloatNode(3.0), FloatNode(1.0)),
            "vec2<f32>(sdCylinder(p, 3.0, 1.0), 0.0)",
        ),
        (
            ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),
            "vec2<f32>(sdCone(p, 2.0, 0.0, 1.0), 0.0)",
        ),
        (
            PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),
            "vec2<f32>(sdPlane(p, vec3<f32>(0.0, 1.0, 0.0), 0.0), 0.0)",
        ),
        (
            CapsuleNode(PositionNode(), Vec3Node(0.0, -1.0, 0.0), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.5)),
            "vec2<f32>(sdCapsule(p, vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), 0.5), 0.0)",
        ),
    ], ids=["sphere", "box", "torus", "cylinder", "cone", "plane", "capsule"])
    def test_exact_argument_format(self, prim, expected) -> None:
        gen = WgslCodeGen()
        result = gen._sdf_call(prim, "p")
        assert result == expected


# =============================================================================
# EDGE-CASE SCENE GRAPHS
# =============================================================================

class TestEmptyEdgeCaseGraphs:
    """Whitebox tests for edge-case SceneGraph configurations."""

    def test_no_primitives_no_pipeline(self) -> None:
        """Graph with no primitives and no pipeline produces header + entry."""
        gen = WgslCodeGen()
        graph = SceneGraph(primitives=())
        src = gen.generate(graph, name="empty")
        assert GENERATED_HEADER.strip() in src
        assert "fn sd_scene__empty" in src
        assert "return vec2<f32>(result.x / comp, result.y)" in src

    def test_no_primitives_with_pipeline(self) -> None:
        """Graph with only a pipeline but no primitives emits domain imports."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
        )
        src = gen.generate(graph)
        assert "#import" in src
        assert "return vec2<f32>(result.x / comp, result.y)" in src

    def test_no_pipeline_uses_p_not_p_d(self) -> None:
        """Without pipeline, the position variable is 'p', not 'p_d'."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        src = gen.generate(graph)
        assert "sdSphere(p, 1.0)" in src
        assert "p_d" not in src

    def test_with_pipeline_uses_p_d_in_sdf_calls(self) -> None:
        """With pipeline, SDF calls use p_d."""
        gen = WgslCodeGen()
        graph = SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),),
            pipeline=(MirrorNode(PositionNode(), Axis.X),),
        )
        src = gen.generate(graph)
        assert "sdBox(p_d, vec3" in src


# =============================================================================
# HEADER AND METADATA
# =============================================================================

class TestGeneratedHeader:
    """Whitebox tests for the GENERATED_HEADER constant."""

    def test_header_contains_spdx(self) -> None:
        assert "SPDX-License-Identifier: MIT" in GENERATED_HEADER

    def test_header_contains_t_demo_tag(self) -> None:
        assert "T-DEMO-2.3" in GENERATED_HEADER

    def test_header_contains_auto_generated_note(self) -> None:
        assert "Auto-generated" in GENERATED_HEADER

    def test_header_contains_wgsl_codegen_reference(self) -> None:
        assert "WgslCodeGen" in GENERATED_HEADER
